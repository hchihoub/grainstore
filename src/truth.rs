//! The truth store: durable writes + MVCC snapshot reads.
//!
//! `TruthStore` ties the WAL, the ordered KV, and the clock together. On open it
//! recovers the KV from the WAL; thereafter every write is group-committed and
//! every read resolves the correct version at a snapshot with a single seek.

use std::path::Path;
use std::sync::mpsc::Receiver;
use std::sync::Arc;

use crate::cdc::{CdcPublisher, CdcRecord};
use crate::error::Result;
use crate::hlc::HlcOracle;
use crate::keys::{grain_prefix, seek_target, t_tx_from_key, KEY_LEN, PREFIX_LEN};
use crate::kv::{MemKv, OrderedKv};
use crate::model::{Confidence, Grain, Hlc, PredId, Sid, Val};
use crate::recovery::recover;
use crate::wal::{decode_value, Wal, WriteReq};

/// Read consistency selected by the caller's intent.
#[derive(Clone, Copy, Debug)]
pub enum ReadMode {
    /// Read the latest committed state.
    Strong,
    /// Read as of a specific snapshot timestamp.
    Snapshot(Hlc),
}

/// The transactional core. Cloneable handles share one underlying store via the
/// owning `Arc` in the caller (see tests); the type itself is not `Clone`
/// because it owns the committer.
pub struct TruthStore {
    kv: Arc<dyn OrderedKv>,
    wal: Wal,
    hlc: Arc<HlcOracle>,
    cdc: CdcPublisher,
}

impl TruthStore {
    /// Open the store at `wal_path`, recovering from any existing WAL.
    pub fn open(wal_path: &Path) -> Result<Self> {
        let kv: Arc<dyn OrderedKv> = Arc::new(MemKv::new());
        let recovered = recover(kv.as_ref(), wal_path)?;
        let hlc = Arc::new(HlcOracle::resume_from(recovered.high_watermark));
        let cdc = CdcPublisher::new();
        let wal = Wal::open(
            wal_path,
            kv.clone(),
            hlc.clone(),
            recovered.idem,
            cdc.clone(),
        )?;
        Ok(Self { kv, wal, hlc, cdc })
    }

    /// Subscribe to the change stream that feeds derived materializations.
    /// Subscribe before issuing writes to observe them; the vector index is
    /// rebuildable from a truth scan otherwise.
    pub fn subscribe(&self) -> Receiver<CdcRecord> {
        self.cdc.subscribe()
    }

    /// Commit a live value for `(sid, pred)`. Returns the commit timestamp.
    pub fn put(
        &self,
        sid: Sid,
        pred: PredId,
        val: Vec<u8>,
        c: f32,
        t_valid: u64,
        idem_key: u128,
    ) -> Result<Hlc> {
        let seq = self
            .wal
            .commit(WriteReq::put(sid, pred, val, c, t_valid, idem_key))?;
        Ok(Hlc(seq))
    }

    /// Commit a tombstone (logical delete) for `(sid, pred)`.
    pub fn delete(
        &self,
        sid: Sid,
        pred: PredId,
        c: f32,
        t_valid: u64,
        idem_key: u128,
    ) -> Result<Hlc> {
        let seq = self
            .wal
            .commit(WriteReq::delete(sid, pred, c, t_valid, idem_key))?;
        Ok(Hlc(seq))
    }

    /// Read `(sid, pred)` under a read mode. Returns the live grain, or `None`
    /// if the visible version is a tombstone or no version exists.
    pub fn get(&self, sid: Sid, pred: PredId, mode: ReadMode) -> Result<Option<Grain>> {
        let snap = match mode {
            ReadMode::Strong => Hlc(self.hlc.now()),
            ReadMode::Snapshot(ts) => ts,
        };
        self.get_at(sid, pred, snap)
    }

    /// Read the version of `(sid, pred)` visible at `snapshot`: the largest
    /// `t_tx <= snapshot`, or `None` if that version is a tombstone / absent.
    pub fn get_at(&self, sid: Sid, pred: PredId, snapshot: Hlc) -> Result<Option<Grain>> {
        let target = seek_target(sid, pred, snapshot);
        let prefix = grain_prefix(sid, pred);
        match self.kv.seek_ge(&target) {
            Some((key, value)) if key.starts_with(&prefix) => {
                let (op, c, t_valid, bytes) = decode_value(&value)?;
                if op == 1 {
                    return Ok(None); // tombstone
                }
                let t_tx = t_tx_from_key(&key).unwrap_or(Hlc(0));
                Ok(Some(Grain {
                    sid,
                    pred,
                    val: Val::Bytes(bytes),
                    c: Confidence(c),
                    t_valid: Hlc(t_valid),
                    t_tx,
                }))
            }
            _ => Ok(None),
        }
    }

    /// Every currently-live grain (newest version per `(sid, pred)`, tombstones
    /// excluded). Used to rebuild derived materializations from the truth.
    pub fn scan_live(&self) -> Result<Vec<Grain>> {
        let mut out = Vec::new();
        let mut last_prefix: Option<Vec<u8>> = None;
        for (k, v) in self.kv.scan_all() {
            if k.len() < KEY_LEN {
                continue;
            }
            let prefix = k[..PREFIX_LEN].to_vec();
            // Keys are sorted; the first key of each prefix is the newest version.
            if last_prefix.as_deref() == Some(prefix.as_slice()) {
                continue;
            }
            last_prefix = Some(prefix);
            let (op, c, t_valid, bytes) = decode_value(&v)?;
            if op == 1 {
                continue; // tombstone → no live grain
            }
            let sid = Sid(u128::from_be_bytes(k[0..16].try_into().expect("16 bytes")));
            let pred = PredId(u32::from_be_bytes(k[16..20].try_into().expect("4 bytes")));
            let t_tx = t_tx_from_key(&k).unwrap_or(Hlc(0));
            out.push(Grain {
                sid,
                pred,
                val: Val::Bytes(bytes),
                c: Confidence(c),
                t_valid: Hlc(t_valid),
                t_tx,
            });
        }
        Ok(out)
    }

    /// Current clock high-watermark (a valid "read latest" snapshot).
    pub fn now(&self) -> Hlc {
        Hlc(self.hlc.now())
    }

    /// Total number of stored versions across all keys (introspection / tests).
    pub fn total_versions(&self) -> usize {
        self.kv.grain_count()
    }

    /// Number of versions for one `(sid, pred)` (introspection / tests).
    pub fn version_count(&self, sid: Sid, pred: PredId) -> usize {
        self.kv.prefix_count(&grain_prefix(sid, pred))
    }

    /// Whether the durability committer is still running.
    pub fn is_healthy(&self) -> bool {
        self.wal.is_healthy()
    }
}
