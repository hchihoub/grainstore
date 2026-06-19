//! Sharded HNSW: a parallel-build vector index.
//!
//! Holds `S` independent [`Hnsw`] sub-graphs, routing each vector to a shard by a
//! hash of its `sid`. Because every shard has its own lock, concurrent inserts to
//! *different* shards proceed in parallel — and each shard's graph is `~n/S`, so
//! per-insert cost (≈ log of graph size) is also lower. Build throughput scales
//! with cores; the trade is a slightly heavier query (search all shards, merge).
//!
//! Same [`VectorIndex`] trait as [`Hnsw`], so it swaps in via
//! [`GrainEngine::open_with`](crate::GrainEngine::open_with).

use super::{Candidate, Hnsw, HnswConfig, VectorIndex};
use crate::model::Sid;

pub struct ShardedHnsw {
    shards: Vec<Hnsw>,
}

impl ShardedHnsw {
    /// Create `shards` sub-indexes of dimension `dim`. Each shard gets a distinct
    /// level-assignment seed so they are not correlated.
    pub fn new(shards: usize, dim: usize, cfg: HnswConfig) -> Self {
        let shards = shards.max(1);
        let subs = (0..shards)
            .map(|i| {
                let mut c = cfg;
                c.seed = cfg.seed
                    ^ (i as u64)
                        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        .wrapping_add(1);
                Hnsw::new(dim, c)
            })
            .collect();
        Self { shards: subs }
    }

    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    #[inline]
    fn shard_of(&self, sid: Sid) -> usize {
        // Mix both halves of the 128-bit id, then map to a shard.
        let mut h = (sid.0 as u64) ^ ((sid.0 >> 64) as u64);
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
        h ^= h >> 33;
        (h % self.shards.len() as u64) as usize
    }
}

impl VectorIndex for ShardedHnsw {
    fn insert(&self, sid: Sid, vector: &[f32]) {
        self.shards[self.shard_of(sid)].insert(sid, vector);
    }

    fn remove(&self, sid: Sid) {
        self.shards[self.shard_of(sid)].remove(sid);
    }

    fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<Candidate> {
        // Each shard contributes its local top-k; merge to the global top-k.
        // The global top-k members each fall in some shard and are top-k there,
        // so requesting k per shard recovers them. Shards are searched in
        // parallel (each has its own lock — no contention across shards).
        let mut all: Vec<Candidate> = if self.shards.len() == 1 {
            self.shards[0].search(query, k, ef)
        } else {
            std::thread::scope(|scope| {
                let handles: Vec<_> = self
                    .shards
                    .iter()
                    .map(|s| scope.spawn(move || s.search(query, k, ef)))
                    .collect();
                let mut out = Vec::with_capacity(k * self.shards.len());
                for h in handles {
                    out.extend(h.join().expect("shard search panicked"));
                }
                out
            })
        };
        all.sort_by(|a, b| a.dist.total_cmp(&b.dist));
        all.truncate(k);
        all
    }

    fn len(&self) -> usize {
        self.shards.iter().map(|s| s.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sharded_routes_and_counts() {
        let idx = ShardedHnsw::new(8, 4, HnswConfig::default());
        for i in 0..500u128 {
            idx.insert(Sid(i), &[i as f32, 0.0, 0.0, 0.0]);
        }
        assert_eq!(idx.len(), 500);
        // a near query around a known point returns it
        let res = idx.search(&[42.0, 0.0, 0.0, 0.0], 1, 64);
        assert_eq!(res[0].sid, Sid(42));
    }
}
