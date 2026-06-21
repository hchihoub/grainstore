//! The ordered key-value seam.
//!
//! [`OrderedKv`] is the abstraction the truth store reads and writes through.
//! P0 ships [`MemKv`], an in-memory `BTreeMap` backing that is intentionally
//! **volatile**: it models a RocksDB instance running with its internal WAL
//! disabled and nothing flushed, so a "crash" discards it and recovery rebuilds
//! it from the durable WAL. A production build swaps in a RocksDB implementation
//! behind this same trait with no change to [`crate::truth`].

use std::collections::BTreeMap;
use std::ops::Bound;
use std::sync::Mutex;

/// An ordered, prefix-iterable key-value store with a separate metadata space.
pub trait OrderedKv: Send + Sync {
    /// Insert or overwrite a grain key. Keys are globally ordered lexicographically.
    fn put_grain(&self, key: Vec<u8>, value: Vec<u8>);

    /// Return the smallest `(key, value)` whose key is `>= target`, if any.
    fn seek_ge(&self, target: &[u8]) -> Option<(Vec<u8>, Vec<u8>)>;

    /// Total number of distinct grain keys (versions). Introspection / tests.
    fn grain_count(&self) -> usize;

    /// Number of grain keys sharing `prefix`. Introspection / tests.
    fn prefix_count(&self, prefix: &[u8]) -> usize;

    /// All `(key, value)` pairs in ascending key order. Used to rebuild derived
    /// materializations (e.g. the vector index) from the truth on startup.
    fn scan_all(&self) -> Vec<(Vec<u8>, Vec<u8>)>;

    /// Write a metadata key (e.g. the applied-sequence watermark).
    fn put_meta(&self, key: &[u8], value: Vec<u8>);

    /// Read a metadata key.
    fn get_meta(&self, key: &[u8]) -> Option<Vec<u8>>;
}

/// In-memory `OrderedKv` for the test build. Thread-safe via a coarse mutex —
/// adequate because the only writer is the single committer thread.
#[derive(Default)]
pub struct MemKv {
    grains: Mutex<BTreeMap<Vec<u8>, Vec<u8>>>,
    meta: Mutex<BTreeMap<Vec<u8>, Vec<u8>>>,
}

impl MemKv {
    pub fn new() -> Self {
        Self::default()
    }
}

impl OrderedKv for MemKv {
    fn put_grain(&self, key: Vec<u8>, value: Vec<u8>) {
        self.grains
            .lock()
            .expect("grains lock poisoned")
            .insert(key, value);
    }

    fn seek_ge(&self, target: &[u8]) -> Option<(Vec<u8>, Vec<u8>)> {
        let g = self.grains.lock().expect("grains lock poisoned");
        g.range::<[u8], _>((Bound::Included(target), Bound::Unbounded))
            .next()
            .map(|(k, v)| (k.clone(), v.clone()))
    }

    fn grain_count(&self) -> usize {
        self.grains.lock().expect("grains lock poisoned").len()
    }

    fn prefix_count(&self, prefix: &[u8]) -> usize {
        let g = self.grains.lock().expect("grains lock poisoned");
        g.range::<[u8], _>((Bound::Included(prefix), Bound::Unbounded))
            .take_while(|(k, _)| k.starts_with(prefix))
            .count()
    }

    fn scan_all(&self) -> Vec<(Vec<u8>, Vec<u8>)> {
        let g = self.grains.lock().expect("grains lock poisoned");
        g.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
    }

    fn put_meta(&self, key: &[u8], value: Vec<u8>) {
        self.meta
            .lock()
            .expect("meta lock poisoned")
            .insert(key.to_vec(), value);
    }

    fn get_meta(&self, key: &[u8]) -> Option<Vec<u8>> {
        self.meta
            .lock()
            .expect("meta lock poisoned")
            .get(key)
            .cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seek_ge_finds_first_at_or_after() {
        let kv = MemKv::new();
        kv.put_grain(vec![1, 0], vec![b'a']);
        kv.put_grain(vec![1, 5], vec![b'b']);
        kv.put_grain(vec![2, 0], vec![b'c']);
        assert_eq!(kv.seek_ge(&[1, 3]).unwrap().0, vec![1, 5]);
        assert_eq!(kv.seek_ge(&[1, 5]).unwrap().0, vec![1, 5]);
        assert_eq!(kv.seek_ge(&[3, 0]), None);
    }
}
