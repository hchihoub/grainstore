//! The `GrainEngine` facade: the default embedded setup works out of the box,
//! and a custom index swaps in through `open_with` with no other change.

use std::sync::Arc;
use std::time::Duration;

use grainstore::embed::RawVectorEmbedder;
use grainstore::model::{Grain, PredId, Sid, Val};
use grainstore::vector::{BruteForceIndex, ShardedHnsw, VectorIndex};
use grainstore::{EngineConfig, GrainEngine, HnswConfig, MixedQuery, WriteMeta};

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

fn cat(g: &Grain) -> Option<u8> {
    match &g.val {
        Val::Bytes(b) => b.first().copied(),
        Val::Tombstone => None,
    }
}

fn wal(tag: &str) -> std::path::PathBuf {
    let dir =
        std::env::temp_dir().join(format!("grainstore_engine_{}_{}", std::process::id(), tag));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("mkdir");
    dir.join("wal.log")
}

const DIM: usize = 12;
const N: usize = 400;
const CATS: u8 = 4;
const K: usize = 8;
const PRED: PredId = PredId(0);

/// Load N grains into an engine, return (vectors, cats) for the oracle.
fn load(engine: &GrainEngine) -> (Vec<Vec<f32>>, Vec<u8>) {
    let mut rng = R::new(11);
    let vectors: Vec<Vec<f32>> = (0..N).map(|_| rng.vec(DIM)).collect();
    let cats: Vec<u8> = (0..N).map(|_| (rng.nx() % CATS as u64) as u8).collect();
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
        engine.sync(last, Duration::from_secs(20)),
        "materializer lagged"
    );
    assert_eq!(engine.index_len(), N);
    (vectors, cats)
}

fn run_queries(engine: &GrainEngine, vectors: &[Vec<f32>], cats: &[u8]) -> f64 {
    let mut qrng = R::new(999);
    let mut total = 0.0f64;
    let trials = 30;
    for _ in 0..trials {
        let q = qrng.vec(DIM);
        let tgt = (qrng.nx() % CATS as u64) as u8;

        let mut exact: Vec<(usize, f32)> = (0..N)
            .filter(|&i| cats[i] == tgt)
            .map(|i| (i, l2(&q, &vectors[i])))
            .collect();
        exact.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let want: std::collections::HashSet<u128> =
            exact.iter().take(K).map(|(i, _)| *i as u128).collect();

        let mq = MixedQuery {
            pred: PRED,
            query: &q,
            k: K,
            ef: 64,
            over_fetch: 16,
        };
        let got_v = engine.query(&mq, |g| cat(g) == Some(tgt)).expect("query");
        assert!(
            got_v.iter().all(|r| cat(&r.grain) == Some(tgt)),
            "predicate violated"
        );
        let got: std::collections::HashSet<u128> = got_v.iter().map(|r| r.grain.sid.0).collect();
        total += want.intersection(&got).count() as f64 / K as f64;
    }
    total / trials as f64
}

#[test]
fn default_engine_works_out_of_the_box() {
    let cfg = EngineConfig::new(DIM).with_header(1);
    let engine = GrainEngine::open(&wal("default"), cfg).expect("open");
    let (vectors, cats) = load(&engine);
    let recall = run_queries(&engine, &vectors, &cats);
    println!("default (HNSW) engine recall: {recall:.3}");
    assert!(recall >= 0.90, "default engine recall too low: {recall:.3}");
}

#[test]
fn custom_index_swaps_in_via_open_with() {
    // Same engine API, but back it with the exact brute-force index instead.
    let cfg = EngineConfig::new(DIM).with_header(1);
    let index: Arc<dyn VectorIndex> = Arc::new(BruteForceIndex::new());
    let embed = Arc::new(RawVectorEmbedder::new(DIM, 1));
    let engine = GrainEngine::open_with(&wal("custom"), cfg, index, embed).expect("open");

    let (vectors, cats) = load(&engine);
    let recall = run_queries(&engine, &vectors, &cats);
    println!("custom (brute-force) engine recall: {recall:.3}");
    // Exact index → the only misses are over_fetch starvation, which 16× avoids here.
    assert!(
        recall >= 0.99,
        "exact-index engine should be ~perfect: {recall:.3}"
    );
}

#[test]
fn parallel_pipeline_sharded_pool() {
    // The live pipeline with a sharded index + worker-pool materializer: inserts
    // parallelize across shards, the contiguous watermark stays correct, and the
    // mixed query returns correct results.
    let cfg = EngineConfig::new(DIM).with_header(1).with_workers(6);
    let index: Arc<dyn VectorIndex> = Arc::new(ShardedHnsw::new(8, DIM, HnswConfig::default()));
    let embed = Arc::new(RawVectorEmbedder::new(DIM, 1));
    let engine = GrainEngine::open_with(&wal("pool"), cfg, index, embed).expect("open");

    let (vectors, cats) = load(&engine); // load() asserts index_len == N via the watermark
    let recall = run_queries(&engine, &vectors, &cats);
    println!("parallel sharded-pool engine recall: {recall:.3}");
    assert!(
        recall >= 0.95,
        "parallel pipeline recall too low: {recall:.3}"
    );
}
