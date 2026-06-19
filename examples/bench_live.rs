//! Live-pipeline benchmark: the FULL path with parallel build wired in.
//!
//! truth store (durable WAL) → CDC → worker-pool materializer → sharded HNSW,
//! via `GrainEngine`. Unlike `bench_build` (which drives index inserts directly),
//! this includes the WAL fsync write cost and the CDC hand-off — the honest
//! end-to-end build time — then measures mixed-query latency and recall.
//!
//! Run:   cargo run --release --example bench_live
//! Scale: GS_N=1000000 GS_DIM=32 GS_SHARDS=16 GS_WORKERS=14 cargo run --release --example bench_live

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use grainstore::embed::{Embedder, RawVectorEmbedder};
use grainstore::model::{Grain, Hlc, PredId, Sid, Val};
use grainstore::vector::{HnswConfig, ShardedHnsw, VectorIndex};
use grainstore::{EngineConfig, GrainEngine, MixedQuery, WriteMeta};

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

fn category_of(g: &Grain) -> Option<u8> {
    match &g.val {
        Val::Bytes(b) => b.first().copied(),
        Val::Tombstone => None,
    }
}

fn pct(s: &[u64], p: f64) -> Duration {
    if s.is_empty() {
        return Duration::ZERO;
    }
    let r = ((p / 100.0) * (s.len() as f64 - 1.0)).round() as usize;
    Duration::from_nanos(s[r.min(s.len() - 1)])
}

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

fn tmp(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("grainstore_live_{}_{}", std::process::id(), tag));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("mkdir");
    dir.join("wal.log")
}

fn main() {
    let dim = env_usize("GS_DIM", 32);
    let n = env_usize("GS_N", 1_000_000);
    let shards = env_usize("GS_SHARDS", 16);
    let workers = env_usize("GS_WORKERS", 14);
    let categories = env_usize("GS_CATS", 10) as u8;
    let k = env_usize("GS_K", 10);
    let ef = env_usize("GS_EF", 128);
    let queries = env_usize("GS_QUERIES", 200);
    let pred = PredId(0);

    println!("GrainStore LIVE pipeline benchmark (truth → CDC → pool → sharded HNSW)");
    println!("  n={n} dim={dim} shards={shards} workers={workers} ef={ef}\n");

    let mut rng = R::new(2024);
    let vectors: Arc<Vec<Vec<f32>>> = Arc::new((0..n).map(|_| rng.vec(dim)).collect());
    let cats: Arc<Vec<u8>> = Arc::new(
        (0..n)
            .map(|_| (rng.nx() % categories as u64) as u8)
            .collect(),
    );

    // Engine: sharded index + worker-pool materializer, wired via the facade.
    let cfg = EngineConfig::new(dim).with_header(1).with_workers(workers);
    let index: Arc<dyn VectorIndex> =
        Arc::new(ShardedHnsw::new(shards, dim, HnswConfig::default()));
    let embed: Arc<dyn Embedder> = Arc::new(RawVectorEmbedder::new(dim, 1));
    let engine = Arc::new(GrainEngine::open_with(&tmp("live"), cfg, index, embed).expect("open"));

    // Concurrent writes through the durable pipeline.
    let last_seq = Arc::new(AtomicU64::new(0));
    let build_start = Instant::now();
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
    let writes_done = build_start.elapsed();
    let target = last_seq.load(Ordering::Acquire);
    // Wait for the parallel materializer to reflect every commit in the index.
    assert!(
        engine.sync(Hlc(target), Duration::from_secs(600)),
        "materializer lagged"
    );
    let build = build_start.elapsed();
    println!(
        "  writes durable: {writes_done:.2?}   end-to-end build (incl. index): {build:.2?}  ({} vectors)\n",
        engine.index_len()
    );

    // Oracle + queries.
    let mut qrng = R::new(777);
    let qset: Vec<(Vec<f32>, u8)> = (0..queries)
        .map(|_| (qrng.vec(dim), (qrng.nx() % categories as u64) as u8))
        .collect();
    let oracle: Vec<std::collections::HashSet<u128>> = qset
        .iter()
        .map(|(q, tgt)| {
            let mut e: Vec<(usize, f32)> = (0..n)
                .filter(|&i| cats[i] == *tgt)
                .map(|i| (i, l2(q, &vectors[i])))
                .collect();
            e.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            e.iter().take(k).map(|(i, _)| *i as u128).collect()
        })
        .collect();

    println!("== Mixed query (sharded parallel search): latency & recall ==");
    println!(
        "  {:>10}  {:>10}  {:>10}  {:>8}",
        "over_fetch", "p50", "p99", "recall"
    );
    for &over_fetch in &[20usize, 40, 80] {
        let mut samples = Vec::with_capacity(queries);
        let mut total_recall = 0.0f64;
        for (qi, (q, tgt)) in qset.iter().enumerate() {
            let tgt = *tgt;
            let t0 = Instant::now();
            let mq = MixedQuery {
                pred,
                query: q,
                k,
                ef,
                over_fetch,
            };
            let ranked = engine
                .query(&mq, |g| category_of(g) == Some(tgt))
                .expect("query");
            samples.push(t0.elapsed().as_nanos() as u64);
            let got: std::collections::HashSet<u128> =
                ranked.iter().map(|r| r.grain.sid.0).collect();
            total_recall += oracle[qi].intersection(&got).count() as f64 / k as f64;
        }
        samples.sort_unstable();
        println!(
            "  {:>10}  {:>10.2?}  {:>10.2?}  {:>8.3}",
            over_fetch,
            pct(&samples, 50.0),
            pct(&samples, 99.0),
            total_recall / queries as f64
        );
    }
    println!("\ndone.");
}
