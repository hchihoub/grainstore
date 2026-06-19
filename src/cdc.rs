//! Change-data-capture: the committer publishes every applied write to
//! subscribers, in commit order. This is the seam that feeds derived
//! materializations (the vector index, columnar projections, …) from the single
//! source of truth.
//!
//! Ordering is total and gap-free: the committer is single-threaded, so records
//! are published strictly in increasing `seq`. Subscribers attached before a
//! write observe it; a subscriber attached later starts from that point and is
//! expected to seed itself from a truth scan (the index is rebuildable).

use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};

/// One committed change. `val` is the raw user value (the vector payload for the
/// embedder); a delete carries `op = 1` and an empty `val`.
#[derive(Clone, Debug)]
pub struct CdcRecord {
    pub sid: u128,
    pub pred: u32,
    pub seq: u64,
    pub op: u8,
    pub val: Vec<u8>,
}

impl CdcRecord {
    pub fn is_delete(&self) -> bool {
        self.op == 1
    }
}

/// Fan-out publisher held by the truth store and shared with the committer.
#[derive(Clone, Default)]
pub struct CdcPublisher {
    subs: Arc<Mutex<Vec<Sender<CdcRecord>>>>,
}

impl CdcPublisher {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a subscriber and receive its stream of changes.
    pub fn subscribe(&self) -> Receiver<CdcRecord> {
        let (tx, rx) = channel();
        self.subs.lock().expect("cdc lock poisoned").push(tx);
        rx
    }

    /// Publish a record to all live subscribers; drops any that have hung up.
    pub fn publish(&self, rec: &CdcRecord) {
        let mut subs = self.subs.lock().expect("cdc lock poisoned");
        subs.retain(|s| s.send(rec.clone()).is_ok());
    }

    /// Number of live subscribers (introspection / tests).
    pub fn subscriber_count(&self) -> usize {
        self.subs.lock().expect("cdc lock poisoned").len()
    }
}
