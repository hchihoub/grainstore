//! Shared test support: a reference MVCC model, a deterministic PRNG, and
//! WAL-file helpers. The model is a deliberately simple, obviously-correct
//! reimplementation of snapshot semantics that the real store is checked against.

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// A version payload: `Some(bytes)` for a live value, `None` for a tombstone.
type VersionPayload = Option<Vec<u8>>;
/// The ordered version chain for one `(sid, pred)`.
type VersionChain = Vec<(u64, VersionPayload)>;
/// All version chains, keyed by `(sid, pred)`.
type VersionMap = BTreeMap<(u128, u32), VersionChain>;

/// Oracle model of MVCC visibility. Records every committed version and mirrors
/// the store's idempotency (a repeated `idem_key` creates no new version).
pub struct Model {
    versions: VersionMap,
    idem: BTreeMap<u128, u64>,
}

impl Model {
    pub fn new() -> Self {
        Self {
            versions: BTreeMap::new(),
            idem: BTreeMap::new(),
        }
    }

    /// Mirror a committed write. `payload = None` is a tombstone.
    pub fn apply(&mut self, sid: u128, pred: u32, seq: u64, payload: Option<Vec<u8>>, idem: u128) {
        if self.idem.contains_key(&idem) {
            return; // deduped — no new version, exactly like the committer
        }
        self.idem.insert(idem, seq);
        let chain = self.versions.entry((sid, pred)).or_default();
        chain.push((seq, payload));
        chain.sort_by_key(|(t, _)| *t);
    }

    pub fn idem_seq(&self, idem: u128) -> Option<u64> {
        self.idem.get(&idem).copied()
    }

    /// The spec: the live bytes of the newest version with `t_tx <= snap`, or
    /// `None` if that version is a tombstone or no version exists.
    pub fn get_at(&self, sid: u128, pred: u32, snap: u64) -> Option<Vec<u8>> {
        let chain = self.versions.get(&(sid, pred))?;
        let (_, payload) = chain.iter().rev().find(|(t, _)| *t <= snap)?;
        payload.clone()
    }

    /// A copy retaining only commits with `seq <= cutoff` — the expected state
    /// after recovery from a WAL truncated at the last intact frame.
    pub fn truncated(&self, cutoff: u64) -> Model {
        let mut m = Model::new();
        for (k, chain) in &self.versions {
            let kept: Vec<_> = chain
                .iter()
                .filter(|(t, _)| *t <= cutoff)
                .cloned()
                .collect();
            if !kept.is_empty() {
                m.versions.insert(*k, kept);
            }
        }
        for (idem, seq) in &self.idem {
            if *seq <= cutoff {
                m.idem.insert(*idem, *seq);
            }
        }
        m
    }
}

impl Default for Model {
    fn default() -> Self {
        Self::new()
    }
}

/// Deterministic, seedable PRNG (a 64-bit LCG). No external deps; reproducible.
pub struct Lcg(u64);

impl Lcg {
    pub fn new(seed: u64) -> Self {
        Lcg(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    /// A value in `0..n` (returns 0 if `n == 0`).
    pub fn range(&mut self, n: u64) -> u64 {
        if n == 0 {
            0
        } else {
            self.next_u64() % n
        }
    }
}

/// A unique temp directory for one test case; removed on drop.
pub struct TempCase {
    pub dir: PathBuf,
}

impl TempCase {
    pub fn new(tag: &str, seed: u64) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "grainstore_{}_{}_{}",
            tag,
            std::process::id(),
            seed
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create temp dir");
        Self { dir }
    }
    pub fn wal(&self) -> PathBuf {
        self.dir.join("wal.log")
    }
}

impl Drop for TempCase {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// Decode the WAL and return the `seq` of the last intact frame (0 if none).
pub fn last_intact_seq(path: &Path) -> u64 {
    let bytes = std::fs::read(path).unwrap_or_default();
    let mut off = 0usize;
    let mut last = 0u64;
    while let Some((rec, n)) = grainstore::wal::decode_frame(&bytes[off..]) {
        off += n;
        last = rec.seq;
    }
    last
}

/// Truncate `n` bytes off the WAL tail, simulating a crash mid-write.
pub fn truncate_tail(path: &Path, n: u64) -> u64 {
    use std::fs::OpenOptions;
    let f = OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .expect("open wal");
    let len = f.metadata().expect("metadata").len();
    let new_len = len.saturating_sub(n);
    f.set_len(new_len).expect("truncate");
    new_len
}
