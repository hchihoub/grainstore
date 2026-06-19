//! P2 planner benchmark: adaptive recall-targeting vs a fixed over-fetch.
//!
//! One index, a skewed category distribution (selectivities from ~0.5 down to
//! ~0.01). For each category the planner estimates selectivity and picks
//! `over_fetch`/`ef` for a target recall; we compare its recall + candidate cost
//! to a fixed `over_fetch` baseline. The planner hits the target everywhere at
//! cost ∝ 1/selectivity; the fixed baseline either overspends (common filters)
//! or misses recall (rare filters).
//!
//! Run: cargo run --release --example bench_planner

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use grainstore::embed::{Embedder, RawVectorEmbedder};
use grainstore::model::{Grain, PredId, Sid, Val};
use grainstore::vector::{HnswConfig, ShardedHnsw, VectorIndex};
use grainstore::{EngineConfig, Filter, GrainEngine, MixedQuery, WriteMeta};

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

fn cat_of(g: &Grain) -> Option<u8> {
    match &g.val {
        Val::Bytes(b) => b.first().copied(),
        Val::Tombstone => None,
    }
}

fn category(i: usize) -> u8 {
    match i % 100 {
        0..=49 => 0,  // ~0.50
        50..=79 => 1, // ~0.30
        80..=94 => 2, // ~0.15
        95..=98 => 3, // ~0.04
        _ => 4,       // ~0.01
    }
}

fn tmp(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "grainstore_benchplan_{}_{}",
        std::process::id(),
        tag
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("mkdir");
    dir.join("wal.log")
}

fn main() {
    let dim = 32;
    let n = 50_000usize;
    let k = 10;
    let target = 0.90;
    let trials = 100;
    let fixed_over_fetch = 20usize;
    let pred = PredId(0);

    println!("P2 planner benchmark — adaptive recall-targeting vs fixed over_fetch");
    println!("  n={n} dim={dim} k={k} target_recall={target} fixed_baseline_over_fetch={fixed_over_fetch}\n");

    let mut rng = R::new(2024);
    let vectors: Arc<Vec<Vec<f32>>> = Arc::new((0..n).map(|_| rng.vec(dim)).collect());
    let cats: Arc<Vec<u8>> = Arc::new((0..n).map(category).collect());

    let cfg = EngineConfig::new(dim).with_header(1).with_workers(14);
    let index: Arc<dyn VectorIndex> = Arc::new(ShardedHnsw::new(16, dim, HnswConfig::default()));
    let embed: Arc<dyn Embedder> = Arc::new(RawVectorEmbedder::new(dim, 1));
    let engine = Arc::new(GrainEngine::open_with(&tmp("plan"), cfg, index, embed).expect("open"));

    let last_seq = Arc::new(AtomicU64::new(0));
    let mut handles = Vec::new();
    for t in 0..64usize {
        let (engine, vectors, cats, last_seq) = (
            engine.clone(),
            vectors.clone(),
            cats.clone(),
            last_seq.clone(),
        );
        handles.push(std::thread::spawn(move || {
            let mut i = t;
            while i < n {
                let h = engine
                    .put_vector(
                        Sid(i as u128),
                        pred,
                        &[cats[i]],
                        &vectors[i],
                        WriteMeta::new(i as u128),
                    )
                    .expect("put")
                    .0;
                last_seq.fetch_max(h, Ordering::AcqRel);
                i += 64;
            }
        }));
    }
    for h in handles {
        h.join().expect("join");
    }
    assert!(engine.sync(
        grainstore::Hlc(last_seq.load(Ordering::Acquire)),
        Duration::from_secs(120)
    ));
    println!("  built {} vectors\n", engine.index_len());

    // Per-category: oracle, planner, fixed baseline.
    println!(
        "  {:>3}  {:>8}  | {:>10} {:>8} {:>8}  | {:>8} {:>8} {:>8}",
        "cat", "sel", "plan_ofch", "plan_rec", "plan_cand", "fix_ofch", "fix_rec", "fix_cand"
    );
    for tgt_cat in 0u8..5 {
        let sel = engine.selectivity(tgt_cat).unwrap_or(0.0);
        let filter = Filter::HeaderByteEq {
            offset: 0,
            value: tgt_cat,
        };

        let mut qrng = R::new(900 + tgt_cat as u64);
        let (mut plan_recall, mut fix_recall) = (0.0f64, 0.0f64);
        let (mut plan_ofch, mut plan_cand) = (0usize, 0usize);
        for _ in 0..trials {
            let q = qrng.vec(dim);
            let mut exact: Vec<(usize, f32)> = (0..n)
                .filter(|&i| cats[i] == tgt_cat)
                .map(|i| (i, l2(&q, &vectors[i])))
                .collect();
            exact.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let want: std::collections::HashSet<u128> =
                exact.iter().take(k).map(|(i, _)| *i as u128).collect();
            let want_n = want.len().max(1);

            // planner
            let (res, plan) = engine
                .query_planned(pred, &q, &filter, k, target)
                .expect("q");
            plan_ofch = plan.over_fetch;
            plan_cand = plan.est_candidates;
            let got: std::collections::HashSet<u128> = res.iter().map(|r| r.grain.sid.0).collect();
            plan_recall += want.intersection(&got).count() as f64 / want_n as f64;

            // fixed baseline
            let mq = MixedQuery {
                pred,
                query: &q,
                k,
                ef: 128,
                over_fetch: fixed_over_fetch,
            };
            let res2 = engine
                .query(&mq, |g| cat_of(g) == Some(tgt_cat))
                .expect("q2");
            let got2: std::collections::HashSet<u128> =
                res2.iter().map(|r| r.grain.sid.0).collect();
            fix_recall += want.intersection(&got2).count() as f64 / want_n as f64;
        }
        println!(
            "  {:>3}  {:>8.3}  | {:>10} {:>8.3} {:>8}  | {:>8} {:>8.3} {:>8}",
            tgt_cat,
            sel,
            plan_ofch,
            plan_recall / trials as f64,
            plan_cand,
            fixed_over_fetch,
            fix_recall / trials as f64,
            k * fixed_over_fetch
        );
    }
    println!("\nplan_* = planner (adaptive)   fix_* = fixed over_fetch={fixed_over_fetch}");
    println!("done.");
}
