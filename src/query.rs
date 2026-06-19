//! The `Near ⋈ Select` executor — the single mixed-query shape P1 must run end
//! to end: generate candidates from the vector plane, join each back to the
//! truth store by `sid`, apply an exact symbolic predicate, and return the top-k
//! ranked nearest.
//!
//! P1 implements the **post-filter** strategy (over-fetch `k·over_fetch`
//! candidates, then filter), which is the realistic path when the predicate is
//! not pre-indexed. The over-fetch factor is the planner's lever in P2; here the
//! caller supplies it from a selectivity estimate. Block-guided pre-filtering is
//! a later phase.

use crate::error::Result;
use crate::model::{Grain, PredId};
use crate::truth::{ReadMode, TruthStore};
use crate::vector::VectorIndex;

/// Parameters of a mixed `near ⋈ select` query.
pub struct MixedQuery<'a> {
    /// Predicate id under which grains are stored / looked up.
    pub pred: PredId,
    /// The query vector.
    pub query: &'a [f32],
    /// Number of results to return.
    pub k: usize,
    /// ANN beam width.
    pub ef: usize,
    /// Candidate multiplier; `k·over_fetch` are pulled before filtering so a
    /// selective predicate still yields `k` survivors (see the selectivity /
    /// over-fetch analysis in the research note).
    pub over_fetch: usize,
}

/// A ranked mixed-query result: the joined grain and its distance to the query.
#[derive(Clone, Debug)]
pub struct Ranked {
    pub grain: Grain,
    pub dist: f32,
}

/// Execute `near(query, k) ⋈ get(pred) WHERE predicate`, post-filtered.
///
/// `predicate` is the exact symbolic filter, evaluated on the joined grain.
pub fn near_join_select<I, P>(
    truth: &TruthStore,
    index: &I,
    q: &MixedQuery<'_>,
    predicate: P,
) -> Result<Vec<Ranked>>
where
    I: VectorIndex + ?Sized,
    P: Fn(&Grain) -> bool,
{
    let k_prime = q.k.saturating_mul(q.over_fetch.max(1)).max(q.k);
    let candidates = index.search(q.query, k_prime, q.ef);

    let mut out: Vec<Ranked> = Vec::with_capacity(candidates.len());
    for c in candidates {
        // Join to the source of truth; apply the predicate on the live grain.
        if let Some(grain) = truth.get(c.sid, q.pred, ReadMode::Strong)? {
            if predicate(&grain) {
                out.push(Ranked {
                    grain,
                    dist: c.dist,
                });
            }
        }
    }

    out.sort_by(|a, b| {
        a.dist
            .partial_cmp(&b.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out.truncate(q.k);
    Ok(out)
}
