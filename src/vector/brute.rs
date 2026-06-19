//! Exact brute-force kNN. O(n·d) per query — used as the recall oracle for the
//! HNSW index and as a correctness baseline in the mixed-query path.

use std::collections::HashMap;
use std::sync::Mutex;

use super::{l2_sq, Candidate, VectorIndex};
use crate::model::Sid;

#[derive(Default)]
pub struct BruteForceIndex {
    inner: Mutex<HashMap<Sid, Vec<f32>>>,
}

impl BruteForceIndex {
    pub fn new() -> Self {
        Self::default()
    }
}

impl VectorIndex for BruteForceIndex {
    fn insert(&self, sid: Sid, vector: &[f32]) {
        self.inner
            .lock()
            .expect("brute lock poisoned")
            .insert(sid, vector.to_vec());
    }

    fn remove(&self, sid: Sid) {
        self.inner.lock().expect("brute lock poisoned").remove(&sid);
    }

    fn search(&self, query: &[f32], k: usize, _ef: usize) -> Vec<Candidate> {
        let map = self.inner.lock().expect("brute lock poisoned");
        let mut all: Vec<Candidate> = map
            .iter()
            .map(|(&sid, v)| Candidate {
                sid,
                dist: l2_sq(query, v),
            })
            .collect();
        all.sort_by(|a, b| {
            a.dist
                .partial_cmp(&b.dist)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all.truncate(k);
        all
    }

    fn len(&self) -> usize {
        self.inner.lock().expect("brute lock poisoned").len()
    }
}
