//! The vector plane: an approximate-nearest-neighbour index over grain `ψ`
//! coordinates, fed by CDC. Two implementations behind one trait:
//!
//! - [`BruteForceIndex`] — exact kNN; the recall oracle and a correctness baseline.
//! - [`Hnsw`] — an incremental Hierarchical Navigable Small World graph; the
//!   scalable approximate index, validated against the brute-force oracle.
//!
//! Distance is squared L2. Search returns candidates ordered nearest-first.

mod brute;
mod hnsw;
mod sharded;

pub use brute::BruteForceIndex;
pub use hnsw::{Hnsw, HnswConfig};
pub use sharded::ShardedHnsw;

use crate::model::Sid;

/// A search result: a grain id and its squared-L2 distance to the query.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Candidate {
    pub sid: Sid,
    pub dist: f32,
}

/// An index over `(sid, ψ)` pairs. All methods take `&self`; implementations use
/// interior mutability so the materializer can write while readers query.
pub trait VectorIndex: Send + Sync {
    /// Insert or replace the vector for `sid`.
    fn insert(&self, sid: Sid, vector: &[f32]);
    /// Remove `sid` from the index (tombstone). A no-op if absent.
    fn remove(&self, sid: Sid);
    /// Return up to `k` nearest candidates to `query`. `ef` is the search beam
    /// width (ignored by exact backends); larger `ef` trades latency for recall.
    fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<Candidate>;
    /// Number of indexed vectors.
    fn len(&self) -> usize;
    /// Whether the index is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Squared Euclidean distance. Panics on dimension mismatch (a programming bug).
#[inline]
pub(crate) fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "vector dimension mismatch");
    let mut s = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}
