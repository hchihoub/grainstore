//! Crash recovery: rebuild the (volatile) KV from the durable WAL.
//!
//! The KV is discarded on crash, so recovery replays every intact frame in the
//! WAL into a fresh KV and reconstructs the committer's idempotency map. Replay
//! is idempotent: a record's key is `(sid, pred, seq)` with a fixed `seq`, so
//! re-applying an already-applied record overwrites identically.

use std::collections::HashMap;
use std::path::Path;

use crate::error::{Error, Result};
use crate::keys::grain_key;
use crate::kv::OrderedKv;
use crate::model::{Hlc, PredId, Sid};
use crate::wal::{decode_frame, encode_value};

/// Result of replaying the WAL.
pub struct Recovered {
    /// Highest commit timestamp seen (the clock high-watermark).
    pub high_watermark: u64,
    /// `idem_key -> commit seq`, to seed the committer's dedup map.
    pub idem: HashMap<u128, u64>,
    /// Number of intact frames applied.
    pub frames: usize,
}

/// Replay `wal_path` into `kv`. A missing file is treated as an empty log.
pub fn recover(kv: &dyn OrderedKv, wal_path: &Path) -> Result<Recovered> {
    let bytes = match std::fs::read(wal_path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Vec::new(),
        Err(e) => return Err(Error::Io(e)),
    };

    let mut off = 0usize;
    let mut high_watermark = 0u64;
    let mut idem: HashMap<u128, u64> = HashMap::new();
    let mut frames = 0usize;

    while let Some((rec, n)) = decode_frame(&bytes[off..]) {
        off += n;
        frames += 1;

        let key = grain_key(Sid(rec.sid), PredId(rec.pred), Hlc(rec.seq));
        let value = encode_value(rec.op, &rec.val, rec.c, rec.t_valid);
        kv.put_grain(key, value);

        // First occurrence wins; the committer never writes a duplicate idem_key,
        // so in practice each key appears at most once.
        idem.entry(rec.idem_key).or_insert(rec.seq);
        high_watermark = high_watermark.max(rec.seq);
    }

    Ok(Recovered {
        high_watermark,
        idem,
        frames,
    })
}
