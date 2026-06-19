//! Recall-targeting query planner (P2).
//!
//! Instead of the caller hand-tuning `over_fetch`/`ef`, the planner estimates a
//! filter's selectivity and picks parameters to hit a target recall at minimum
//! candidate-fetch cost. The core relationship (from the filtered-ANN analysis):
//! a post-filter over `k·over_fetch` ANN candidates survives a selectivity-`sel`
//! filter only if `over_fetch·sel` clears a margin; so `over_fetch ≈ margin/sel`,
//! and the margin grows with the recall target.
//!
//! The margin/ef constants are calibrated from the benchmark recall curves; in
//! production they would be refined online from observed recall (the research's
//! learned-policy direction).

use crate::model::{Grain, Val};

/// A structured predicate the planner can both evaluate and estimate.
#[derive(Clone, Debug)]
pub enum Filter {
    /// Matches every grain (selectivity 1.0).
    Any,
    /// The header byte at `offset` equals `value` (e.g. a category tag at 0).
    HeaderByteEq { offset: usize, value: u8 },
}

impl Filter {
    /// Exact evaluation on a joined grain.
    pub fn eval(&self, g: &Grain) -> bool {
        match self {
            Filter::Any => true,
            Filter::HeaderByteEq { offset, value } => match &g.val {
                Val::Bytes(b) => b.get(*offset) == Some(value),
                Val::Tombstone => false,
            },
        }
    }
}

/// The chosen physical parameters plus the estimates behind them.
#[derive(Clone, Copy, Debug)]
pub struct Plan {
    pub over_fetch: usize,
    pub ef: usize,
    pub est_selectivity: f64,
    /// `k · over_fetch` — the candidate budget, a proxy for query cost.
    pub est_candidates: usize,
}

/// Stateless cost-model planner.
#[derive(Clone, Copy, Debug, Default)]
pub struct Planner;

impl Planner {
    pub fn new() -> Self {
        Planner
    }

    /// Plan parameters for a query of `k` results over a filter of estimated
    /// `selectivity`, to reach `target_recall`.
    pub fn plan(&self, selectivity: f64, k: usize, target_recall: f64) -> Plan {
        let sel = selectivity.clamp(1e-4, 1.0);
        let margin = Self::margin_for(target_recall);
        let over_fetch = ((margin / sel).ceil() as usize).clamp(1, 4096);
        let ef = Self::ef_for(target_recall);
        Plan {
            over_fetch,
            ef,
            est_selectivity: sel,
            est_candidates: k.saturating_mul(over_fetch),
        }
    }

    /// Survival margin (expected filter-passing candidates per requested result)
    /// needed for a recall target. Calibrated from the benchmark curves.
    fn margin_for(t: f64) -> f64 {
        if t >= 0.99 {
            12.0
        } else if t >= 0.95 {
            7.0
        } else if t >= 0.90 {
            4.5
        } else if t >= 0.80 {
            2.5
        } else {
            1.5
        }
    }

    /// Base ANN beam width for a recall target.
    fn ef_for(t: f64) -> usize {
        if t >= 0.99 {
            256
        } else if t >= 0.95 {
            192
        } else if t >= 0.90 {
            128
        } else {
            64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn over_fetch_scales_inversely_with_selectivity() {
        let p = Planner::new();
        let common = p.plan(0.5, 10, 0.90); // selective filter is cheap
        let rare = p.plan(0.02, 10, 0.90); // rare filter needs more candidates
        assert!(rare.over_fetch > common.over_fetch);
        assert_eq!(common.over_fetch, (4.5f64 / 0.5).ceil() as usize); // = 9
        assert_eq!(rare.over_fetch, (4.5f64 / 0.02).ceil() as usize); // = 225
    }

    #[test]
    fn higher_target_costs_more() {
        let p = Planner::new();
        assert!(p.plan(0.1, 10, 0.99).over_fetch > p.plan(0.1, 10, 0.80).over_fetch);
        assert!(p.plan(0.1, 10, 0.99).ef >= p.plan(0.1, 10, 0.80).ef);
    }
}
