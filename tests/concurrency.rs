//! Concurrency: many threads commit through the single group-committer. Verifies
//! that timestamps are unique, idempotency holds under contention (the bug the
//! committer-serialized dedup fixes), and the clock never collides.

mod common;

use common::TempCase;
use grainstore::hlc::HlcOracle;
use grainstore::model::{PredId, Sid};
use grainstore::TruthStore;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

#[test]
fn concurrent_writers_preserve_invariants() {
    const WRITERS: usize = 12;
    const PER: usize = 300;

    let case = TempCase::new("concurrency", 0);
    let store = Arc::new(TruthStore::open(&case.wal()).expect("open"));

    let mut handles = Vec::new();
    for w in 0..WRITERS {
        let store = store.clone();
        handles.push(std::thread::spawn(move || {
            let mut local: Vec<(u128, u64)> = Vec::with_capacity(PER);
            for i in 0..PER {
                // One quarter of writes reuse GLOBAL idem keys (0..50) to force
                // dedup contention; the rest are unique per (writer, i).
                let idem: u128 = if i % 4 == 0 {
                    (i as u128) % 50
                } else {
                    ((w as u128) << 40) | (i as u128)
                };
                let seq = store
                    .put(
                        Sid((i % 64) as u128),
                        PredId(0),
                        vec![w as u8, i as u8],
                        1.0,
                        0,
                        idem,
                    )
                    .expect("put")
                    .0;
                local.push((idem, seq));
            }
            local
        }));
    }

    let mut all: Vec<(u128, u64)> = Vec::new();
    for h in handles {
        all.extend(h.join().expect("thread"));
    }

    // Distinct idem_keys actually used.
    let distinct_idem: HashSet<u128> = all.iter().map(|(i, _)| *i).collect();
    let distinct = distinct_idem.len();

    // INVARIANT 1: exactly one stored version per distinct idem_key.
    assert_eq!(
        store.total_versions(),
        distinct,
        "stored versions must equal distinct idem_keys"
    );

    // INVARIANT 2: distinct commit timestamps equal distinct idem_keys
    //              (retries of a shared key collapse to one seq).
    let distinct_seqs: HashSet<u64> = all.iter().map(|(_, s)| *s).collect();
    assert_eq!(
        distinct_seqs.len(),
        distinct,
        "distinct seqs must equal distinct idem_keys"
    );

    // INVARIANT 3: a commit timestamp is never shared by two different idem_keys.
    let mut seq_to_idem: HashMap<u64, u128> = HashMap::new();
    for (idem, seq) in &all {
        if let Some(prev) = seq_to_idem.insert(*seq, *idem) {
            assert_eq!(prev, *idem, "seq {seq} shared by idem {prev} and {idem}");
        }
    }
}

#[test]
fn hlc_is_unique_under_threads() {
    let oracle = Arc::new(HlcOracle::new());
    let mut handles = Vec::new();
    for _ in 0..8 {
        let o = oracle.clone();
        handles.push(std::thread::spawn(move || {
            let mut v = Vec::with_capacity(2000);
            for _ in 0..2000 {
                v.push(o.next());
            }
            v
        }));
    }
    let mut all = Vec::new();
    for h in handles {
        all.extend(h.join().expect("thread"));
    }
    let unique: HashSet<u64> = all.iter().copied().collect();
    assert_eq!(unique.len(), all.len(), "HLC issued a duplicate timestamp");
    assert!(!all.contains(&0), "HLC must never issue 0");
}
