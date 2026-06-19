//! Hybrid logical clock oracle.
//!
//! P0 runs single-node, so the oracle is a monotonic counter that never issues a
//! duplicate even under concurrent `next()` calls. The CAS loop is the unit that
//! a `loom` model checks for uniqueness/monotonicity (see SETUP.md). The same
//! interface generalizes to a multi-node HLC by folding in physical time.

use std::sync::atomic::{AtomicU64, Ordering};

/// Issues strictly increasing, unique timestamps.
#[derive(Debug)]
pub struct HlcOracle {
    last: AtomicU64,
}

impl HlcOracle {
    /// A fresh oracle; the first `next()` returns 1 (0 is reserved for
    /// "before any write").
    pub fn new() -> Self {
        Self {
            last: AtomicU64::new(0),
        }
    }

    /// Resume after recovery so the clock never moves backwards.
    pub fn resume_from(high_watermark: u64) -> Self {
        Self {
            last: AtomicU64::new(high_watermark),
        }
    }

    /// Allocate the next timestamp. Lock-free and safe under contention.
    pub fn next(&self) -> u64 {
        let mut prev = self.last.load(Ordering::Relaxed);
        loop {
            let next = prev + 1;
            match self
                .last
                .compare_exchange_weak(prev, next, Ordering::AcqRel, Ordering::Relaxed)
            {
                Ok(_) => return next,
                Err(observed) => prev = observed,
            }
        }
    }

    /// The most recently issued timestamp (for strong "read latest" snapshots).
    pub fn now(&self) -> u64 {
        self.last.load(Ordering::Acquire)
    }
}

impl Default for HlcOracle {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn monotonic_single_threaded() {
        let o = HlcOracle::new();
        assert_eq!(o.next(), 1);
        assert_eq!(o.next(), 2);
        assert_eq!(o.now(), 2);
    }

    #[test]
    fn resume_does_not_regress() {
        let o = HlcOracle::resume_from(100);
        assert_eq!(o.next(), 101);
    }
}
