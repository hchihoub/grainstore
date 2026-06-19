//! Incremental Hierarchical Navigable Small World (HNSW) index.
//!
//! A faithful, compact implementation of Malkov & Yashunin (2016): a multi-layer
//! proximity graph with greedy descent through upper layers and a beam search
//! (`ef`) at the base layer, plus neighbour pruning to bound degree. Level
//! assignment uses a seeded LCG so builds are deterministic and tests are
//! reproducible. Deletes are soft (the node stays for navigation, is filtered
//! from results); a compaction/rebuild reclaims them in a later phase.
//!
//! Recall is validated against [`super::BruteForceIndex`] in the test suite.

use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::Mutex;

use super::{l2_sq, Candidate, VectorIndex};
use crate::model::Sid;

/// Tunables. Defaults follow common practice (M=16, efConstruction=200).
#[derive(Clone, Copy, Debug)]
pub struct HnswConfig {
    /// Max neighbours per node on layers > 0.
    pub m: usize,
    /// Max neighbours per node on layer 0 (typically 2·M).
    pub m0: usize,
    /// Beam width during construction.
    pub ef_construction: usize,
    /// PRNG seed for level assignment (determinism).
    pub seed: u64,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 16,
            m0: 32,
            ef_construction: 200,
            seed: 0x1234_5678_9abc_def0,
        }
    }
}

struct Node {
    sid: Sid,
    vector: Vec<f32>,
    /// `links[level]` = neighbour node indices at that level.
    links: Vec<Vec<usize>>,
}

#[derive(Clone, Copy)]
struct DistIdx {
    dist: f32,
    idx: usize,
}

impl PartialEq for DistIdx {
    fn eq(&self, o: &Self) -> bool {
        self.idx == o.idx && self.dist.total_cmp(&o.dist) == Ordering::Equal
    }
}
impl Eq for DistIdx {}
impl Ord for DistIdx {
    fn cmp(&self, o: &Self) -> Ordering {
        self.dist.total_cmp(&o.dist).then(self.idx.cmp(&o.idx))
    }
}
impl PartialOrd for DistIdx {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

struct State {
    nodes: Vec<Node>,
    deleted: Vec<bool>,
    sid_to_idx: HashMap<Sid, usize>,
    entry: Option<usize>,
    max_level: usize,
    live: usize,
    rng: u64,
    cfg: HnswConfig,
    ml: f64,
}

impl State {
    fn distance(&self, q: &[f32], idx: usize) -> f32 {
        l2_sq(q, &self.nodes[idx].vector)
    }

    fn rand_level(&mut self) -> usize {
        self.rng = self
            .rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let unit = ((self.rng >> 11) as f64 / (1u64 << 53) as f64).max(1e-12);
        (-unit.ln() * self.ml).floor() as usize
    }

    /// Beam search of one layer; returns candidates ascending by distance.
    fn search_layer(
        &self,
        q: &[f32],
        entry_points: &[usize],
        ef: usize,
        level: usize,
    ) -> Vec<DistIdx> {
        let mut visited: HashSet<usize> = entry_points.iter().copied().collect();
        let mut candidates: BinaryHeap<Reverse<DistIdx>> = BinaryHeap::new();
        let mut result: BinaryHeap<DistIdx> = BinaryHeap::new();

        for &ep in entry_points {
            let d = self.distance(q, ep);
            candidates.push(Reverse(DistIdx { dist: d, idx: ep }));
            result.push(DistIdx { dist: d, idx: ep });
        }

        while let Some(Reverse(c)) = candidates.pop() {
            let farthest = result.peek().map(|x| x.dist).unwrap_or(f32::INFINITY);
            if c.dist > farthest && result.len() >= ef {
                break;
            }
            if let Some(neighbors) = self.nodes[c.idx].links.get(level) {
                for &n in neighbors {
                    if visited.insert(n) {
                        let d = self.distance(q, n);
                        let farthest = result.peek().map(|x| x.dist).unwrap_or(f32::INFINITY);
                        if d < farthest || result.len() < ef {
                            candidates.push(Reverse(DistIdx { dist: d, idx: n }));
                            result.push(DistIdx { dist: d, idx: n });
                            if result.len() > ef {
                                result.pop(); // drop the farthest
                            }
                        }
                    }
                }
            }
        }
        result.into_sorted_vec()
    }

    fn link(&mut self, a: usize, b: usize, level: usize) {
        if let Some(l) = self.nodes[a].links.get_mut(level) {
            if !l.contains(&b) {
                l.push(b);
            }
        }
    }

    /// Bound a node's degree at `level` to the configured maximum, keeping the
    /// nearest neighbours.
    fn prune(&mut self, node: usize, level: usize) {
        let max_m = if level == 0 { self.cfg.m0 } else { self.cfg.m };
        let links = match self.nodes[node].links.get(level) {
            Some(l) if l.len() > max_m => l.clone(),
            _ => return,
        };
        let nv = self.nodes[node].vector.clone();
        let mut scored: Vec<DistIdx> = links
            .iter()
            .map(|&n| DistIdx {
                dist: l2_sq(&nv, &self.nodes[n].vector),
                idx: n,
            })
            .collect();
        scored.sort_unstable();
        scored.truncate(max_m);
        self.nodes[node].links[level] = scored.into_iter().map(|d| d.idx).collect();
    }

    fn insert(&mut self, sid: Sid, vector: &[f32]) {
        // Replace: soft-delete the prior version of this sid.
        if let Some(&old) = self.sid_to_idx.get(&sid) {
            if !self.deleted[old] {
                self.deleted[old] = true;
                self.live -= 1;
            }
        }

        let level = self.rand_level();
        let idx = self.nodes.len();
        self.nodes.push(Node {
            sid,
            vector: vector.to_vec(),
            links: vec![Vec::new(); level + 1],
        });
        self.deleted.push(false);
        self.sid_to_idx.insert(sid, idx);
        self.live += 1;

        let entry = match self.entry {
            None => {
                self.entry = Some(idx);
                self.max_level = level;
                return;
            }
            Some(e) => e,
        };

        let l_max = self.max_level;
        let mut ep = entry;

        // Greedy descent through layers above the new node's top layer.
        if level < l_max {
            for lc in ((level + 1)..=l_max).rev() {
                let w = self.search_layer(vector, &[ep], 1, lc);
                if let Some(best) = w.first() {
                    ep = best.idx;
                }
            }
        }

        // Connect on each layer from min(level, l_max) down to 0.
        let top = level.min(l_max);
        for lc in (0..=top).rev() {
            let w = self.search_layer(vector, &[ep], self.cfg.ef_construction, lc);
            let m = if lc == 0 { self.cfg.m0 } else { self.cfg.m };
            let neighbors: Vec<usize> = w.iter().take(m).map(|d| d.idx).collect();
            for &n in &neighbors {
                self.link(idx, n, lc);
                self.link(n, idx, lc);
                self.prune(n, lc);
            }
            self.prune(idx, lc);
            if let Some(best) = w.first() {
                ep = best.idx;
            }
        }

        if level > l_max {
            self.entry = Some(idx);
            self.max_level = level;
        }
    }

    fn search(&self, q: &[f32], k: usize, ef: usize) -> Vec<Candidate> {
        let entry = match self.entry {
            None => return Vec::new(),
            Some(e) => e,
        };
        let mut ep = entry;
        for lc in (1..=self.max_level).rev() {
            let w = self.search_layer(q, &[ep], 1, lc);
            if let Some(best) = w.first() {
                ep = best.idx;
            }
        }
        let w = self.search_layer(q, &[ep], ef.max(k), 0);
        w.into_iter()
            .filter(|d| !self.deleted[d.idx])
            .take(k)
            .map(|d| Candidate {
                sid: self.nodes[d.idx].sid,
                dist: d.dist,
            })
            .collect()
    }
}

/// Thread-safe HNSW index. A coarse mutex serializes writers and readers; the
/// materializer is single-threaded, so this is adequate for P1.
pub struct Hnsw {
    dim: usize,
    state: Mutex<State>,
}

impl Hnsw {
    pub fn new(dim: usize, cfg: HnswConfig) -> Self {
        let ml = 1.0 / (cfg.m as f64).ln();
        Self {
            dim,
            state: Mutex::new(State {
                nodes: Vec::new(),
                deleted: Vec::new(),
                sid_to_idx: HashMap::new(),
                entry: None,
                max_level: 0,
                live: 0,
                rng: cfg.seed,
                cfg,
                ml,
            }),
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
}

impl VectorIndex for Hnsw {
    fn insert(&self, sid: Sid, vector: &[f32]) {
        debug_assert_eq!(vector.len(), self.dim, "vector dim mismatch");
        self.state
            .lock()
            .expect("hnsw lock poisoned")
            .insert(sid, vector);
    }

    fn remove(&self, sid: Sid) {
        let mut st = self.state.lock().expect("hnsw lock poisoned");
        if let Some(&idx) = st.sid_to_idx.get(&sid) {
            if !st.deleted[idx] {
                st.deleted[idx] = true;
                st.live -= 1;
            }
            st.sid_to_idx.remove(&sid);
        }
    }

    fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<Candidate> {
        self.state
            .lock()
            .expect("hnsw lock poisoned")
            .search(query, k, ef)
    }

    fn len(&self) -> usize {
        self.state.lock().expect("hnsw lock poisoned").live
    }
}
