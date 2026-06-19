//! The recall-targeting planner: across filters of very different selectivity,
//! `query_planned` hits the target recall, adapting `over_fetch` to selectivity
//! (rare filters get more candidates, common filters stay cheap).

use std::time::Duration;

use grainstore::model::{PredId, Sid};
use grainstore::{EngineConfig, Filter, GrainEngine, WriteMeta};

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

fn l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Skewed category distribution: ~50/30/15/4/1 %.
fn category(i: usize) -> u8 {
    match i % 100 {
        0..=49 => 0,
        50..=79 => 1,
        80..=94 => 2,
        95..=98 => 3,
        _ => 4,
    }
}

fn wal(tag: &str) -> std::path::PathBuf {
    let dir =
        std::env::temp_dir().join(format!("grainstore_planner_{}_{}", std::process::id(), tag));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("mkdir");
    dir.join("wal.log")
}

const DIM: usize = 16;
const N: usize = 2000;
const K: usize = 8;
const PRED: PredId = PredId(0);

#[test]
fn planner_hits_target_recall_across_selectivities() {
    let cfg = EngineConfig::new(DIM).with_header(1).with_workers(4);
    let index: std::sync::Arc<dyn grainstore::VectorIndex> = std::sync::Arc::new(
        grainstore::ShardedHnsw::new(4, DIM, grainstore::HnswConfig::default()),
    );
    let embed = std::sync::Arc::new(grainstore::embed::RawVectorEmbedder::new(DIM, 1));
    let engine = GrainEngine::open_with(&wal("skew"), cfg, index, embed).expect("open");

    let mut rng = R::new(7);
    let vectors: Vec<Vec<f32>> = (0..N).map(|_| rng.vec(DIM)).collect();
    let cats: Vec<u8> = (0..N).map(category).collect();
    let mut last = grainstore::model::Hlc(0);
    for i in 0..N {
        last = engine
            .put_vector(
                Sid(i as u128),
                PRED,
                &[cats[i]],
                &vectors[i],
                WriteMeta::new(i as u128),
            )
            .expect("put");
    }
    assert!(
        engine.sync(last, Duration::from_secs(30)),
        "materializer lagged"
    );

    let target = 0.90;
    // Compare a common (cat 0, ~0.5) and a rare (cat 4, ~0.01) filter.
    let mut plans = Vec::new();
    for &tgt_cat in &[0u8, 4u8] {
        let mut qrng = R::new(1234 + tgt_cat as u64);
        let mut total_recall = 0.0f64;
        let mut last_plan = None;
        let trials = 25;
        for _ in 0..trials {
            let q = qrng.vec(DIM);
            // exact filtered oracle
            let mut exact: Vec<(usize, f32)> = (0..N)
                .filter(|&i| cats[i] == tgt_cat)
                .map(|i| (i, l2(&q, &vectors[i])))
                .collect();
            exact.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let want: std::collections::HashSet<u128> =
                exact.iter().take(K).map(|(i, _)| *i as u128).collect();
            let want_n = want.len().max(1);

            let filter = Filter::HeaderByteEq {
                offset: 0,
                value: tgt_cat,
            };
            let (res, plan) = engine
                .query_planned(PRED, &q, &filter, K, target)
                .expect("query");
            last_plan = Some(plan);
            assert!(
                res.iter().all(|r| filter.eval(&r.grain)),
                "predicate violated"
            );
            let got: std::collections::HashSet<u128> = res.iter().map(|r| r.grain.sid.0).collect();
            total_recall += want.intersection(&got).count() as f64 / want_n as f64;
        }
        let recall = total_recall / 25.0;
        let plan = last_plan.unwrap();
        println!(
            "cat {tgt_cat}: est_sel={:.3} over_fetch={} recall={recall:.3}",
            plan.est_selectivity, plan.over_fetch
        );
        assert!(
            recall >= 0.85,
            "cat {tgt_cat}: planner missed target, recall={recall:.3}"
        );
        plans.push(plan);
    }

    // The planner adapted: the rare filter (cat 4) got a larger candidate budget.
    assert!(
        plans[1].over_fetch > plans[0].over_fetch,
        "planner should over-fetch more for the rarer filter"
    );
    // And the common filter's selectivity estimate is far higher than the rare one's.
    assert!(plans[0].est_selectivity > plans[1].est_selectivity * 5.0);
}
