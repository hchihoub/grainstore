//! Write-ahead log with group commit.
//!
//! The WAL is the durability boundary and source of truth. A single committer
//! thread owns the file; client threads submit [`WriteReq`]s and block on an
//! ack. The committer drains the current queue into one batch, assigns commit
//! timestamps, writes all frames, performs **one** `fsync`, then applies the
//! batch to the [`OrderedKv`] and acks. Batching amortizes the fsync across
//! concurrent transactions.
//!
//! Idempotency is enforced here, in the single committer, so it is race-free by
//! construction: a repeated `idem_key` is answered with the original commit
//! timestamp and never re-appended.
//!
//! On-disk frame: `[len: u32 LE][crc32: u32 LE][payload]`.
//! Recovery decodes frames until one fails validation (a torn tail from a crash
//! mid-write), and stops there.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::thread::JoinHandle;

use crate::cdc::{CdcPublisher, CdcRecord};
use crate::error::{Error, Result};
use crate::hlc::HlcOracle;
use crate::keys::grain_key;
use crate::kv::OrderedKv;
use crate::model::{Hlc, PredId, Sid};

const META_APPLIED_SEQ: &[u8] = b"applied_seq";
const OP_PUT: u8 = 0;
const OP_DELETE: u8 = 1;

/// Fixed-size prefix of an encoded record (everything before the value bytes).
const REC_FIXED: usize = 61; // 8+16+4+1+4+8+16+4

/// A request to durably commit one grain version.
#[derive(Clone, Debug)]
pub struct WriteReq {
    pub sid: u128,
    pub pred: u32,
    pub op: u8,
    pub val: Vec<u8>,
    pub c: f32,
    pub t_valid: u64,
    pub idem_key: u128,
}

impl WriteReq {
    pub fn put(sid: Sid, pred: PredId, val: Vec<u8>, c: f32, t_valid: u64, idem: u128) -> Self {
        Self {
            sid: sid.0,
            pred: pred.0,
            op: OP_PUT,
            val,
            c,
            t_valid,
            idem_key: idem,
        }
    }
    pub fn delete(sid: Sid, pred: PredId, c: f32, t_valid: u64, idem: u128) -> Self {
        Self {
            sid: sid.0,
            pred: pred.0,
            op: OP_DELETE,
            val: Vec::new(),
            c,
            t_valid,
            idem_key: idem,
        }
    }
    pub fn is_delete(&self) -> bool {
        self.op == OP_DELETE
    }
}

/// A fully decoded WAL record (carries the assigned commit `seq`).
#[derive(Clone, Debug)]
pub struct WalRecord {
    pub seq: u64,
    pub sid: u128,
    pub pred: u32,
    pub op: u8,
    pub c: f32,
    pub t_valid: u64,
    pub idem_key: u128,
    pub val: Vec<u8>,
}

impl WalRecord {
    pub fn is_delete(&self) -> bool {
        self.op == OP_DELETE
    }
}

// ---------------------------------------------------------------------------
// CRC32 (IEEE 802.3), table-free to keep the crate dependency-free.
// ---------------------------------------------------------------------------

fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

// ---------------------------------------------------------------------------
// Value encoding stored in the KV: [op:1][c:4 BE][t_valid:8 BE][val...]
// ---------------------------------------------------------------------------

/// Encode the KV value for a grain version.
pub(crate) fn encode_value(op: u8, val: &[u8], c: f32, t_valid: u64) -> Vec<u8> {
    let mut v = Vec::with_capacity(13 + val.len());
    v.push(op);
    v.extend_from_slice(&c.to_be_bytes());
    v.extend_from_slice(&t_valid.to_be_bytes());
    v.extend_from_slice(val);
    v
}

/// Decode a KV value into `(op, c, t_valid, val)`.
pub(crate) fn decode_value(v: &[u8]) -> Result<(u8, f32, u64, Vec<u8>)> {
    if v.len() < 13 {
        return Err(Error::Corrupt(format!(
            "value too short: {} bytes",
            v.len()
        )));
    }
    let op = v[0];
    let c = f32::from_be_bytes(v[1..5].try_into().expect("checked len"));
    let t_valid = u64::from_be_bytes(v[5..13].try_into().expect("checked len"));
    let val = v[13..].to_vec();
    Ok((op, c, t_valid, val))
}

// ---------------------------------------------------------------------------
// Record + frame encoding
// ---------------------------------------------------------------------------

fn encode_payload(r: &WalRecord) -> Vec<u8> {
    let mut p = Vec::with_capacity(REC_FIXED + r.val.len());
    p.extend_from_slice(&r.seq.to_be_bytes());
    p.extend_from_slice(&r.sid.to_be_bytes());
    p.extend_from_slice(&r.pred.to_be_bytes());
    p.push(r.op);
    p.extend_from_slice(&r.c.to_be_bytes());
    p.extend_from_slice(&r.t_valid.to_be_bytes());
    p.extend_from_slice(&r.idem_key.to_be_bytes());
    p.extend_from_slice(&(r.val.len() as u32).to_be_bytes());
    p.extend_from_slice(&r.val);
    p
}

fn decode_payload(b: &[u8]) -> Option<WalRecord> {
    if b.len() < REC_FIXED {
        return None;
    }
    let seq = u64::from_be_bytes(b[0..8].try_into().ok()?);
    let sid = u128::from_be_bytes(b[8..24].try_into().ok()?);
    let pred = u32::from_be_bytes(b[24..28].try_into().ok()?);
    let op = b[28];
    let c = f32::from_be_bytes(b[29..33].try_into().ok()?);
    let t_valid = u64::from_be_bytes(b[33..41].try_into().ok()?);
    let idem_key = u128::from_be_bytes(b[41..57].try_into().ok()?);
    let vlen = u32::from_be_bytes(b[57..61].try_into().ok()?) as usize;
    if b.len() < REC_FIXED + vlen {
        return None;
    }
    let val = b[REC_FIXED..REC_FIXED + vlen].to_vec();
    Some(WalRecord {
        seq,
        sid,
        pred,
        op,
        c,
        t_valid,
        idem_key,
        val,
    })
}

/// Append one framed record to `buf`.
pub fn encode_frame(r: &WalRecord, buf: &mut Vec<u8>) {
    let payload = encode_payload(r);
    let crc = crc32(&payload);
    buf.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    buf.extend_from_slice(&crc.to_le_bytes());
    buf.extend_from_slice(&payload);
}

/// Decode the first frame in `input`, returning the record and bytes consumed.
///
/// Returns `None` on an incomplete or CRC-invalid frame — the signal that the
/// log ends here (a torn tail). This is how recovery finds the durable prefix.
pub fn decode_frame(input: &[u8]) -> Option<(WalRecord, usize)> {
    if input.len() < 8 {
        return None;
    }
    let len = u32::from_le_bytes(input[0..4].try_into().ok()?) as usize;
    let crc = u32::from_le_bytes(input[4..8].try_into().ok()?);
    let end = 8usize.checked_add(len)?;
    if input.len() < end {
        return None; // torn tail: frame not fully written
    }
    let payload = &input[8..end];
    if crc32(payload) != crc {
        return None; // torn/corrupt tail
    }
    let rec = decode_payload(payload)?;
    Some((rec, end))
}

// ---------------------------------------------------------------------------
// Group-commit WAL
// ---------------------------------------------------------------------------

struct Pending {
    req: WriteReq,
    ack: Sender<Result<u64>>,
}

/// Handle to the WAL. Cloneable senders are not exposed; commit goes through the
/// owning [`TruthStore`](crate::truth::TruthStore).
pub struct Wal {
    tx: Option<Sender<Pending>>,
    handle: Option<JoinHandle<()>>,
    /// Set by the committer if a fatal I/O error stops it.
    healthy: Arc<AtomicU64>, // 0 = healthy, 1 = stopped
}

impl Wal {
    /// Open (creating if needed) the WAL at `path` and spawn the committer.
    ///
    /// `idem_init` seeds the committer's dedup map from recovery so idempotency
    /// survives restarts.
    pub fn open(
        path: &Path,
        kv: Arc<dyn OrderedKv>,
        hlc: Arc<HlcOracle>,
        idem_init: HashMap<u128, u64>,
        cdc: CdcPublisher,
    ) -> Result<Self> {
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        let (tx, rx) = channel::<Pending>();
        let healthy = Arc::new(AtomicU64::new(0));
        let healthy_c = healthy.clone();
        let handle = std::thread::Builder::new()
            .name("grainstore-committer".into())
            .spawn(move || committer_loop(file, rx, kv, hlc, idem_init, healthy_c, cdc))
            .map_err(Error::Io)?;
        Ok(Self {
            tx: Some(tx),
            handle: Some(handle),
            healthy,
        })
    }

    /// Durably commit one write; returns the assigned commit timestamp.
    pub fn commit(&self, req: WriteReq) -> Result<u64> {
        let (atx, arx) = channel();
        let tx = self.tx.as_ref().ok_or(Error::Closed)?;
        tx.send(Pending { req, ack: atx })
            .map_err(|_| Error::Closed)?;
        arx.recv().map_err(|_| Error::Closed)?
    }

    /// Whether the committer is still running.
    pub fn is_healthy(&self) -> bool {
        self.healthy.load(Ordering::Acquire) == 0
    }
}

impl Drop for Wal {
    fn drop(&mut self) {
        // Dropping the sender ends the committer's recv loop; join guarantees the
        // file is flushed and closed before any subsequent reopen/recovery.
        self.tx.take();
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn committer_loop(
    mut file: File,
    rx: Receiver<Pending>,
    kv: Arc<dyn OrderedKv>,
    hlc: Arc<HlcOracle>,
    mut idem: HashMap<u128, u64>,
    healthy: Arc<AtomicU64>,
    cdc: CdcPublisher,
) {
    let mut buf: Vec<u8> = Vec::with_capacity(1 << 16);

    while let Ok(first) = rx.recv() {
        let mut batch = vec![first];
        while let Ok(more) = rx.try_recv() {
            batch.push(more);
        }

        buf.clear();
        let mut puts: Vec<(Vec<u8>, Vec<u8>)> = Vec::with_capacity(batch.len());
        let mut acks: Vec<(Sender<Result<u64>>, u64)> = Vec::with_capacity(batch.len());
        let mut cdc_recs: Vec<CdcRecord> = Vec::with_capacity(batch.len());
        let mut max_seq = hlc.now();

        for pw in batch {
            // Race-free dedup: only this thread touches `idem`.
            if let Some(&orig) = idem.get(&pw.req.idem_key) {
                let _ = pw.ack.send(Ok(orig));
                continue;
            }
            let seq = hlc.next();
            idem.insert(pw.req.idem_key, seq);

            let value = encode_value(pw.req.op, &pw.req.val, pw.req.c, pw.req.t_valid);
            cdc_recs.push(CdcRecord {
                sid: pw.req.sid,
                pred: pw.req.pred,
                seq,
                op: pw.req.op,
                val: pw.req.val.clone(),
            });
            let rec = WalRecord {
                seq,
                sid: pw.req.sid,
                pred: pw.req.pred,
                op: pw.req.op,
                c: pw.req.c,
                t_valid: pw.req.t_valid,
                idem_key: pw.req.idem_key,
                val: pw.req.val,
            };
            encode_frame(&rec, &mut buf);
            puts.push((grain_key(Sid(rec.sid), PredId(rec.pred), Hlc(seq)), value));
            max_seq = max_seq.max(seq);
            acks.push((pw.ack, seq));
        }

        // Durability ordering: WAL fsync → KV apply → ack. A crash before KV
        // apply is fine: recovery replays the (already durable) frames.
        if !buf.is_empty() {
            if let Err(e) = file.write_all(&buf).and_then(|_| file.sync_all()) {
                // Fatal: cannot guarantee durability. Fail every pending ack and stop.
                healthy.store(1, Ordering::Release);
                for (ack, _) in acks {
                    let _ = ack.send(Err(Error::Io(clone_io(&e))));
                }
                return;
            }
        }

        for (k, v) in puts {
            kv.put_grain(k, v);
        }
        kv.put_meta(META_APPLIED_SEQ, max_seq.to_be_bytes().to_vec());

        // Publish to derived materializations only after the change is durable
        // and applied to the truth KV, in commit order.
        for rec in &cdc_recs {
            cdc.publish(rec);
        }

        for (ack, seq) in acks {
            let _ = ack.send(Ok(seq));
        }
    }
}

/// `std::io::Error` is not `Clone`; reconstruct an equivalent for fan-out acks.
fn clone_io(e: &std::io::Error) -> std::io::Error {
    std::io::Error::new(e.kind(), e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_roundtrips() {
        let rec = WalRecord {
            seq: 42,
            sid: 7,
            pred: 3,
            op: OP_PUT,
            c: 0.5,
            t_valid: 99,
            idem_key: 123,
            val: vec![1, 2, 3],
        };
        let mut buf = Vec::new();
        encode_frame(&rec, &mut buf);
        let (back, n) = decode_frame(&buf).expect("decode");
        assert_eq!(n, buf.len());
        assert_eq!(back.seq, 42);
        assert_eq!(back.val, vec![1, 2, 3]);
        assert_eq!(back.idem_key, 123);
    }

    #[test]
    fn torn_tail_is_rejected() {
        let rec = WalRecord {
            seq: 1,
            sid: 1,
            pred: 1,
            op: OP_PUT,
            c: 1.0,
            t_valid: 0,
            idem_key: 1,
            val: vec![9; 10],
        };
        let mut buf = Vec::new();
        encode_frame(&rec, &mut buf);
        buf.truncate(buf.len() - 3); // lop the tail
        assert!(decode_frame(&buf).is_none());
    }

    #[test]
    fn crc_detects_corruption() {
        let rec = WalRecord {
            seq: 1,
            sid: 1,
            pred: 1,
            op: OP_PUT,
            c: 1.0,
            t_valid: 0,
            idem_key: 1,
            val: vec![5],
        };
        let mut buf = Vec::new();
        encode_frame(&rec, &mut buf);
        let last = buf.len() - 1;
        buf[last] ^= 0xFF; // flip a payload bit
        assert!(decode_frame(&buf).is_none());
    }
}
