//! Self-describing, meaning-clustered storage blocks (P3).
//!
//! Grains are grouped into blocks by ψ-proximity (a light k-means), and each
//! block carries a **descriptor**: centroid + radius, a sid Bloom filter, the set
//! of predicates present, a confidence histogram, a `t_tx` range, and a category
//! sketch. A query reads the descriptor first and skips the whole block without
//! decoding its body when:
//!
//! - point lookup: the Bloom filter says the sid is absent;
//! - kNN: `dist(q, centroid) − radius` exceeds the current k-th best distance
//!   (a triangle-inequality bound — so the pruning is **exact**, not approximate);
//! - predicate / category scan: the sketch shows zero matches.
//!
//! This is the on-disk format that compaction would flush to (replacing RocksDB
//! SSTables); here it is a self-contained module with a pruning benchmark.

/// Actual (non-squared) Euclidean distance — needed for the triangle bound.
fn l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

// ---------------------------------------------------------------------------
// Bloom filter (dependency-free)
// ---------------------------------------------------------------------------

/// A small Bloom filter over `u128` sids.
#[derive(Clone)]
pub struct Bloom {
    bits: Vec<u64>,
    m_bits: u64,
    k: u32,
}

impl Bloom {
    pub fn new(expected: usize, fp: f64) -> Self {
        let n = expected.max(1) as f64;
        let m = (-(n * fp.ln()) / (std::f64::consts::LN_2 * std::f64::consts::LN_2)).ceil();
        let m_bits = (m as u64).max(64);
        let k = ((m_bits as f64 / n) * std::f64::consts::LN_2)
            .round()
            .max(1.0) as u32;
        Self {
            bits: vec![0u64; m_bits.div_ceil(64) as usize],
            m_bits,
            k,
        }
    }

    #[inline]
    fn splitmix(mut z: u64) -> u64 {
        z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    #[inline]
    fn hashes(x: u128) -> (u64, u64) {
        let lo = x as u64;
        let hi = (x >> 64) as u64;
        // Both hashes must vary with `lo` even when `hi == 0` (common for small
        // sids), or double-hashing degenerates and the false-positive rate blows up.
        let h1 = Self::splitmix(lo ^ hi.wrapping_mul(0xD6E8_FEB8_6659_FD93));
        let h2 = Self::splitmix(hi ^ lo.wrapping_mul(0x9E37_79B9_7F4A_7C15)) | 1;
        (h1, h2)
    }

    pub fn insert(&mut self, x: u128) {
        let (h1, h2) = Self::hashes(x);
        for i in 0..self.k as u64 {
            let bit = h1.wrapping_add(i.wrapping_mul(h2)) % self.m_bits;
            self.bits[(bit / 64) as usize] |= 1u64 << (bit % 64);
        }
    }

    pub fn maybe_contains(&self, x: u128) -> bool {
        let (h1, h2) = Self::hashes(x);
        for i in 0..self.k as u64 {
            let bit = h1.wrapping_add(i.wrapping_mul(h2)) % self.m_bits;
            if self.bits[(bit / 64) as usize] & (1u64 << (bit % 64)) == 0 {
                return false;
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Records, descriptors, blocks
// ---------------------------------------------------------------------------

/// One grain as fed to the block builder.
#[derive(Clone)]
pub struct GrainRec {
    pub sid: u128,
    pub pred: u32,
    pub cat: u8, // header byte 0
    pub vector: Vec<f32>,
    pub t_tx: u64,
    pub conf: f32,
}

/// The self-describing header read before any block body.
pub struct BlockDescriptor {
    pub centroid: Vec<f32>,
    pub radius: f32,
    pub preds: Vec<u32>,
    pub conf_hist: [u32; 8],
    pub t_min: u64,
    pub t_max: u64,
    pub sid_bloom: Bloom,
    pub count: u32,
    pub cat_counts: Box<[u32; 256]>,
}

impl BlockDescriptor {
    pub fn has_pred(&self, pred: u32) -> bool {
        self.preds.binary_search(&pred).is_ok()
    }
    pub fn category_count(&self, cat: u8) -> u32 {
        self.cat_counts[cat as usize]
    }
    /// Lower bound on the distance from `q` to any point in the block.
    pub fn min_possible_dist(&self, q: &[f32]) -> f32 {
        (l2(q, &self.centroid) - self.radius).max(0.0)
    }
}

/// A block: its descriptor plus the columnar body.
pub struct Block {
    pub descriptor: BlockDescriptor,
    sids: Vec<u128>,
    cats: Vec<u8>,
    vectors: Vec<Vec<f32>>,
}

impl Block {
    fn build(recs: &[GrainRec]) -> Self {
        let dim = recs[0].vector.len();
        let mut centroid = vec![0.0f32; dim];
        for r in recs {
            for (c, x) in centroid.iter_mut().zip(&r.vector) {
                *c += *x;
            }
        }
        for c in &mut centroid {
            *c /= recs.len() as f32;
        }
        let radius = recs
            .iter()
            .map(|r| l2(&r.vector, &centroid))
            .fold(0.0f32, f32::max);

        let mut preds: Vec<u32> = recs.iter().map(|r| r.pred).collect();
        preds.sort_unstable();
        preds.dedup();

        let mut conf_hist = [0u32; 8];
        let mut cat_counts = Box::new([0u32; 256]);
        let (mut t_min, mut t_max) = (u64::MAX, 0u64);
        let mut bloom = Bloom::new(recs.len(), 0.01);
        let (mut sids, mut cats, mut vectors) = (Vec::new(), Vec::new(), Vec::new());
        for r in recs {
            let b = ((r.conf * 8.0) as usize).min(7);
            conf_hist[b] += 1;
            cat_counts[r.cat as usize] += 1;
            t_min = t_min.min(r.t_tx);
            t_max = t_max.max(r.t_tx);
            bloom.insert(r.sid);
            sids.push(r.sid);
            cats.push(r.cat);
            vectors.push(r.vector.clone());
        }

        Block {
            descriptor: BlockDescriptor {
                centroid,
                radius,
                preds,
                conf_hist,
                t_min,
                t_max,
                sid_bloom: bloom,
                count: recs.len() as u32,
                cat_counts,
            },
            sids,
            cats,
            vectors,
        }
    }

    fn find(&self, sid: u128) -> Option<(u8, usize)> {
        self.sids
            .iter()
            .position(|&s| s == sid)
            .map(|i| (self.cats[i], i))
    }
}

/// A nearest-neighbour candidate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Candidate {
    pub sid: u128,
    pub dist: f32,
}

/// Statistics on how much the descriptor pruning skipped.
#[derive(Clone, Copy, Debug)]
pub struct QueryStats {
    pub blocks_touched: usize,
    pub blocks_total: usize,
}

/// A collection of meaning-clustered blocks with descriptor-pruned queries.
pub struct BlockStore {
    blocks: Vec<Block>,
}

impl BlockStore {
    /// Build blocks from `recs`, clustering by ψ into ~`n_blocks` groups.
    pub fn build_clustered(recs: &[GrainRec], n_blocks: usize, iters: usize) -> Self {
        let assign = kmeans_assign(recs, n_blocks, iters);
        let mut groups: Vec<Vec<GrainRec>> = vec![Vec::new(); n_blocks];
        for (r, &a) in recs.iter().zip(&assign) {
            groups[a].push(r.clone());
        }
        let blocks = groups
            .into_iter()
            .filter(|g| !g.is_empty())
            .map(|g| Block::build(&g))
            .collect();
        Self { blocks }
    }

    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Point lookup, pruned by the per-block Bloom filter.
    pub fn get(&self, sid: u128) -> (Option<u8>, QueryStats) {
        let mut touched = 0;
        for b in &self.blocks {
            if b.descriptor.sid_bloom.maybe_contains(sid) {
                touched += 1;
                if let Some((cat, _)) = b.find(sid) {
                    return (
                        Some(cat),
                        QueryStats {
                            blocks_touched: touched,
                            blocks_total: self.blocks.len(),
                        },
                    );
                }
            }
        }
        (
            None,
            QueryStats {
                blocks_touched: touched,
                blocks_total: self.blocks.len(),
            },
        )
    }

    /// Exact kNN with cluster pruning: blocks are visited nearest-centroid first,
    /// and a block is skipped when its lower-bound distance exceeds the current
    /// k-th best. The triangle inequality makes this exact.
    pub fn near(&self, q: &[f32], k: usize) -> (Vec<Candidate>, QueryStats) {
        let mut order: Vec<(f32, usize)> = self
            .blocks
            .iter()
            .enumerate()
            .map(|(i, b)| (l2(q, &b.descriptor.centroid), i))
            .collect();
        order.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut best: Vec<Candidate> = Vec::with_capacity(k + 1);
        let mut touched = 0;
        for (d_centroid, bi) in order {
            let kth = if best.len() >= k {
                best[k - 1].dist
            } else {
                f32::INFINITY
            };
            if d_centroid - self.blocks[bi].descriptor.radius > kth {
                continue; // no point here can beat the current k-th best
            }
            touched += 1;
            let b = &self.blocks[bi];
            for (i, v) in b.vectors.iter().enumerate() {
                best.push(Candidate {
                    sid: b.sids[i],
                    dist: l2(q, v),
                });
            }
            best.sort_by(|a, c| {
                a.dist
                    .partial_cmp(&c.dist)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            best.truncate(k);
        }
        (
            best,
            QueryStats {
                blocks_touched: touched,
                blocks_total: self.blocks.len(),
            },
        )
    }

    /// Count grains in a category, pruning blocks whose sketch shows none.
    pub fn count_category(&self, cat: u8) -> (usize, QueryStats) {
        let mut touched = 0;
        let mut total = 0usize;
        for b in &self.blocks {
            if b.descriptor.category_count(cat) == 0 {
                continue;
            }
            touched += 1;
            total += b.cats.iter().filter(|&&c| c == cat).count();
        }
        (
            total,
            QueryStats {
                blocks_touched: touched,
                blocks_total: self.blocks.len(),
            },
        )
    }
}

/// Light k-means: even seeds, then `iters` assign/recompute passes.
fn kmeans_assign(recs: &[GrainRec], n_blocks: usize, iters: usize) -> Vec<usize> {
    let n = recs.len();
    let blocks = n_blocks.max(1).min(n.max(1));
    let dim = recs[0].vector.len();
    let mut centroids: Vec<Vec<f32>> = (0..blocks)
        .map(|i| recs[i * n / blocks].vector.clone())
        .collect();
    let mut assign = vec![0usize; n];

    for _ in 0..iters.max(1) {
        for (i, r) in recs.iter().enumerate() {
            let mut best = (f32::INFINITY, 0usize);
            for (c, cen) in centroids.iter().enumerate() {
                let d = l2(&r.vector, cen);
                if d < best.0 {
                    best = (d, c);
                }
            }
            assign[i] = best.1;
        }
        let mut sums = vec![vec![0.0f32; dim]; blocks];
        let mut counts = vec![0u32; blocks];
        for (i, r) in recs.iter().enumerate() {
            let a = assign[i];
            counts[a] += 1;
            for (s, x) in sums[a].iter_mut().zip(&r.vector) {
                *s += *x;
            }
        }
        for c in 0..blocks {
            if counts[c] > 0 {
                for (cen, s) in centroids[c].iter_mut().zip(&sums[c]) {
                    *cen = *s / counts[c] as f32;
                }
            }
        }
    }
    assign
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(sid: u128, cat: u8, v: Vec<f32>) -> GrainRec {
        GrainRec {
            sid,
            pred: 0,
            cat,
            vector: v,
            t_tx: sid as u64,
            conf: 1.0,
        }
    }

    #[test]
    fn bloom_has_no_false_negatives() {
        let mut b = Bloom::new(1000, 0.01);
        for i in 0..1000u128 {
            b.insert(i);
        }
        for i in 0..1000u128 {
            assert!(b.maybe_contains(i), "false negative for {i}");
        }
    }

    #[test]
    fn near_is_exact_vs_brute_force() {
        // 2-D points; cluster-pruned near must equal brute force.
        let mut recs = Vec::new();
        let mut x = 0u128;
        for gx in 0..20 {
            for gy in 0..20 {
                recs.push(rec(x, (x % 3) as u8, vec![gx as f32, gy as f32]));
                x += 1;
            }
        }
        let store = BlockStore::build_clustered(&recs, 16, 3);
        let q = vec![5.3, 9.7];
        let (got, stats) = store.near(&q, 5);

        // Exactness under ties is checked on the distance multiset (which tied
        // sid sits at the boundary is implementation-defined; the distances are not).
        let mut brute: Vec<f32> = recs.iter().map(|r| l2(&q, &r.vector)).collect();
        brute.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let got_dists: Vec<f32> = got.iter().map(|c| c.dist).collect();
        for (i, d) in got_dists.iter().enumerate() {
            assert!((d - brute[i]).abs() < 1e-5, "rank {i}: {d} vs {}", brute[i]);
        }
        assert!(stats.blocks_touched <= stats.blocks_total);
    }

    #[test]
    fn get_and_category_prune() {
        let recs: Vec<GrainRec> = (0..400u128)
            .map(|i| rec(i, (i % 4) as u8, vec![i as f32, (i * 2) as f32]))
            .collect();
        let store = BlockStore::build_clustered(&recs, 16, 3);
        let (hit, _) = store.get(123);
        assert_eq!(hit, Some((123 % 4) as u8));
        let (miss, _) = store.get(99_999);
        assert_eq!(miss, None);
        let (cnt, _) = store.count_category(0);
        assert_eq!(cnt, 100); // 400/4
    }
}
