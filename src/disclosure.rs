//! Staged disclosure with continuations (the tokens-returned half of the planner).
//!
//! A mixed query runs the expensive part (ANN search + join + filter) ONCE,
//! materializes the ranked result, and returns **Stage 0**: a compact summary
//! (a few representatives + match count + a sharpness/confidence signal), plus a
//! continuation handle. `drill(handle)` returns **Stage 1** — the full grains —
//! from the pinned state, with no re-search. The caller pays tokens proportional
//! to the certainty it actually needs.
//!
//! Token cost is estimated from serialized payload size; for large (text)
//! payloads the Stage-0 vs Stage-1 ratio is large. A natural Stage 2 (exact
//! re-rank / full provenance) extends the same continuation.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use crate::model::{Grain, Val};
use crate::query::Ranked;

/// Rough token cost of serializing one grain (≈ 4 bytes/token + framing).
pub fn grain_tokens(g: &Grain) -> usize {
    let bytes = match &g.val {
        Val::Bytes(b) => b.len(),
        Val::Tombstone => 0,
    };
    bytes / 4 + 8
}

/// Stage 0 — a compact summary of the materialized result.
#[derive(Clone, Debug)]
pub struct Summary {
    /// Number of filtered, ranked matches.
    pub total_matches: usize,
    /// A handful of representatives (nearest of each distance band).
    pub representatives: Vec<Ranked>,
    pub dist_min: f32,
    pub dist_median: f32,
    pub dist_max: f32,
    /// Normalized gap between the top two results in `[0,1]`. High = a clear
    /// winner (Stage 0 likely sufficient); low = ambiguous (consider drilling).
    pub sharpness: f32,
    /// Tokens to materialize the full result (Stage 1).
    pub est_full_tokens: usize,
}

/// Opaque handle to a pinned, materialized query result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContinuationHandle(pub u64);

/// The payload of a disclosure stage.
#[derive(Clone, Debug)]
pub enum Stage {
    Summary(Summary),
    Full(Vec<Ranked>),
}

/// One stage of a staged result.
#[derive(Clone, Debug)]
pub struct Staged {
    pub stage_index: u8,
    pub stage: Stage,
    /// Estimated tokens for *this* stage's payload.
    pub est_tokens: usize,
    /// Handle to drill to the next stage, if any.
    pub continuation: Option<ContinuationHandle>,
}

pub(crate) struct Continuation {
    pub ranked: Vec<Ranked>,
    pub full_tokens: usize,
}

/// Bounded TTL-ish cache of pinned results; evicts the oldest beyond capacity.
pub(crate) struct ContinuationCache {
    map: Mutex<BTreeMap<u64, Continuation>>,
    next: AtomicU64,
    cap: usize,
}

impl ContinuationCache {
    pub fn new(cap: usize) -> Self {
        Self {
            map: Mutex::new(BTreeMap::new()),
            next: AtomicU64::new(1),
            cap: cap.max(1),
        }
    }

    pub fn store(&self, ranked: Vec<Ranked>, full_tokens: usize) -> ContinuationHandle {
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        let mut m = self.map.lock().expect("continuation lock poisoned");
        m.insert(
            id,
            Continuation {
                ranked,
                full_tokens,
            },
        );
        while m.len() > self.cap {
            let oldest = *m.keys().next().expect("non-empty");
            m.remove(&oldest);
        }
        ContinuationHandle(id)
    }

    /// Returns the pinned full result, or `None` if expired/evicted.
    pub fn get_full(&self, h: ContinuationHandle) -> Option<(Vec<Ranked>, usize)> {
        let m = self.map.lock().expect("continuation lock poisoned");
        m.get(&h.0).map(|c| (c.ranked.clone(), c.full_tokens))
    }
}

/// Build a Stage-0 summary from a dist-ascending ranked result.
pub fn build_summary(ranked: &[Ranked], reps: usize) -> Summary {
    let total = ranked.len();
    let mut representatives = Vec::new();
    if total > 0 {
        let bands = reps.max(1).min(total);
        let band_size = total.div_ceil(bands);
        let mut i = 0;
        while i < total {
            representatives.push(ranked[i].clone()); // nearest of this band
            i += band_size;
        }
    }
    let (dist_min, dist_median, dist_max) = if total == 0 {
        (0.0, 0.0, 0.0)
    } else {
        let mut d: Vec<f32> = ranked.iter().map(|r| r.dist).collect();
        d.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        (d[0], d[d.len() / 2], d[d.len() - 1])
    };
    let sharpness = if total >= 2 {
        let spread = (dist_max - dist_min).max(1e-6);
        ((ranked[1].dist - ranked[0].dist) / spread).clamp(0.0, 1.0)
    } else {
        1.0
    };
    let est_full_tokens = ranked.iter().map(|r| grain_tokens(&r.grain)).sum();
    Summary {
        total_matches: total,
        representatives,
        dist_min,
        dist_median,
        dist_max,
        sharpness,
        est_full_tokens,
    }
}

/// Estimated tokens to serialize a Stage-0 summary.
pub fn summary_tokens(s: &Summary) -> usize {
    s.representatives
        .iter()
        .map(|r| grain_tokens(&r.grain))
        .sum::<usize>()
        + 16
}
