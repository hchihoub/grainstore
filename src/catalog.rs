//! Selectivity catalog: approximate counts of header-byte-0 (category) values,
//! used by the [`Planner`](crate::planner::Planner) to estimate predicate
//! selectivity.
//!
//! Maintained inline on the write path — no extra thread, always current with
//! committed writes. Counts are approximate by design (selectivity is a ratio
//! used only for planning); deletes are not subtracted, a deliberate
//! simplification adequate for estimation.

use std::sync::Mutex;

struct CatState {
    total: u64,
    by_value: Box<[u64; 256]>,
}

/// Thread-safe selectivity statistics keyed by a single header byte.
pub struct SelectivityCatalog {
    state: Mutex<CatState>,
}

impl SelectivityCatalog {
    pub fn new() -> Self {
        Self {
            state: Mutex::new(CatState {
                total: 0,
                by_value: Box::new([0u64; 256]),
            }),
        }
    }

    /// Record one observed header-byte-0 `value`.
    pub fn observe(&self, value: u8) {
        let mut s = self.state.lock().expect("catalog lock poisoned");
        s.total += 1;
        s.by_value[value as usize] += 1;
    }

    /// Estimated fraction of grains whose header byte 0 equals `value`, or `None`
    /// if nothing has been observed yet.
    pub fn selectivity(&self, value: u8) -> Option<f64> {
        let s = self.state.lock().expect("catalog lock poisoned");
        if s.total == 0 {
            None
        } else {
            Some(s.by_value[value as usize] as f64 / s.total as f64)
        }
    }

    /// Total observations recorded.
    pub fn total(&self) -> u64 {
        self.state.lock().expect("catalog lock poisoned").total
    }
}

impl Default for SelectivityCatalog {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ratios_reflect_observations() {
        let c = SelectivityCatalog::new();
        for _ in 0..50 {
            c.observe(0);
        }
        for _ in 0..50 {
            c.observe(1);
        }
        assert_eq!(c.total(), 100);
        assert!((c.selectivity(0).unwrap() - 0.5).abs() < 1e-9);
        assert!((c.selectivity(1).unwrap() - 0.5).abs() < 1e-9);
        assert_eq!(c.selectivity(2), Some(0.0));
    }
}
