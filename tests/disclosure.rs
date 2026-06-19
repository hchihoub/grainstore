//! Staged disclosure: Stage 0 is a cheap summary with a continuation; drilling
//! returns the full result from pinned state (no re-search), and the full result
//! matches a direct query.

use std::sync::Arc;
use std::time::Duration;

use grainstore::embed::RawVectorEmbedder;
use grainstore::model::{PredId, Sid};
use grainstore::vector::{HnswConfig, ShardedHnsw, VectorIndex};
use grainstore::{ContinuationHandle, EngineConfig, Filter, GrainEngine, Stage, WriteMeta};

struct R(u64);
impl R {
    fn new(s: u64) -> Self {
        R(s ^ 0x9E37_79B9_7F4A_7C15)
    }
    fn nx(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn f32(&mut self) -> f32 {
        (self.nx() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn vec(&mut self, d: usize) -> Vec<f32> {
        (0..d).map(|_| self.f32()).collect()
    }
}

fn wal(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("grainstore_disc_{}_{}", std::process::id(), tag));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("mkdir");
    dir.join("wal.log")
}

const DIM: usize = 16;
const N: usize = 1500;
const PRED: PredId = PredId(0);

#[test]
fn staged_disclosure_summary_then_drill() {
    let cfg = EngineConfig::new(DIM).with_header(1).with_workers(4);
    let index: Arc<dyn VectorIndex> = Arc::new(ShardedHnsw::new(4, DIM, HnswConfig::default()));
    let embed = Arc::new(RawVectorEmbedder::new(DIM, 1));
    let engine = GrainEngine::open_with(&wal("disc"), cfg, index, embed).expect("open");

    let mut rng = R::new(3);
    let mut last = grainstore::model::Hlc(0);
    for i in 0..N {
        let cat = (i % 2) as u8; // two categories, sel ~0.5
        last = engine
            .put_vector(
                Sid(i as u128),
                PRED,
                &[cat],
                &rng.vec(DIM),
                WriteMeta::new(i as u128),
            )
            .expect("put");
    }
    assert!(
        engine.sync(last, Duration::from_secs(30)),
        "materializer lagged"
    );

    let q = rng.vec(DIM);
    let filter = Filter::HeaderByteEq {
        offset: 0,
        value: 0,
    };
    let k = 40;
    let reps = 5;

    // Stage 0
    let s0 = engine
        .query_staged(PRED, &q, &filter, k, 0.90, reps)
        .expect("staged");
    assert_eq!(s0.stage_index, 0);
    let handle = s0.continuation.expect("stage 0 must carry a continuation");
    let summary = match &s0.stage {
        Stage::Summary(s) => s.clone(),
        _ => panic!("stage 0 must be a summary"),
    };
    assert!(summary.total_matches > 0);
    assert!(!summary.representatives.is_empty() && summary.representatives.len() <= reps);
    // Stage 0 is cheaper than fully materializing.
    assert!(
        s0.est_tokens < summary.est_full_tokens,
        "stage 0 ({}) should cost fewer tokens than full ({})",
        s0.est_tokens,
        summary.est_full_tokens
    );

    // Drill → Stage 1 (full), from pinned state.
    let s1 = engine.drill(handle).expect("drill").expect("handle live");
    assert_eq!(s1.stage_index, 1);
    let full = match &s1.stage {
        Stage::Full(v) => v.clone(),
        _ => panic!("stage 1 must be full"),
    };
    assert_eq!(full.len(), summary.total_matches);
    assert_eq!(s1.est_tokens, summary.est_full_tokens);
    // Every returned grain satisfies the predicate.
    assert!(full.iter().all(|r| filter.eval(&r.grain)));

    // The drilled full result equals a direct planned query (same sids).
    let (direct, _) = engine
        .query_planned(PRED, &q, &filter, k, 0.90)
        .expect("direct");
    let drilled_sids: std::collections::HashSet<u128> =
        full.iter().map(|r| r.grain.sid.0).collect();
    let direct_sids: std::collections::HashSet<u128> =
        direct.iter().map(|r| r.grain.sid.0).collect();
    assert_eq!(drilled_sids, direct_sids, "drill must match a direct query");

    // An unknown handle yields None, not an error.
    assert!(engine
        .drill(ContinuationHandle(9_999_999))
        .expect("drill")
        .is_none());
}
