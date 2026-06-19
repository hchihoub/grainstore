//! Parallel build benchmark (addresses the single-threaded HNSW build bottleneck).
//!
//! Builds N vectors into a [`ShardedHnsw`] using T threads issuing concurrent
//! inserts — because shards are independently locked, inserts to different shards
//! run in parallel. Reports build throughput and query recall/latency vs an exact
//! filtered-kNN oracle, so the build speedup can be weighed against any query cost
//! from searching multiple shards.
//!
//! The inserts here are driven directly (the production integration is a
//! worker-pool materializer feeding the same `VectorIndex`); this isolates the
//! index's parallel-build capability.
//!
//! Run:   cargo run --release --example bench_build
//! Scale: GS_N=1000000 GS_DIM=32 GS_SHARDS=16 GS_THREADS=14 cargo run --release --example bench_build

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use grainstore::vector::{HnswConfig, ShardedHnsw, VectorIndex};

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

fn main() {
    let dim = env_usize("GS_DIM", 32);
    let n = env_usize("GS_N", 1_000_000);
    let shards = env_usize("GS_SHARDS", 16);
    let threads = env_usize(
        "GS_THREADS",
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8),
    );
    let categories = env_usize("GS_CATS", 10) as u8;
    let k = env_usize("GS_K", 10);
    let ef = env_usize("GS_EF", 128);
    let queries = env_usize("GS_QUERIES", 200);

    println!("Parallel HNSW build benchmark (ShardedHnsw)");
    println!("  n={n} dim={dim} shards={shards} threads={threads} ef={ef}\n");

    // Data + category tags (cats just to size the recall oracle the same way).
    let mut rng = R::new(2024);
    let vectors: Arc<Vec<Vec<f32>>> = Arc::new((0..n).map(|_| rng.vec(dim)).collect());
    let cats: Vec<u8> = (0..n)
        .map(|_| (rng.nx() % categories as u64) as u8)
        .collect();

    let index = Arc::new(ShardedHnsw::new(shards, dim, HnswConfig::default()));

    // Parallel build: T threads pull a shared cursor and insert concurrently.
    let cursor = Arc::new(AtomicUsize::new(0));
    let build_start = Instant::now();
    let mut handles = Vec::new();
    for _ in 0..threads {
        let (index, vectors, cursor) = (index.clone(), vectors.clone(), cursor.clone());
        handles.push(std::thread::spawn(move || {
            const CHUNK: usize = 256;
            loop {
                let start = cursor.fetch_add(CHUNK, Ordering::Relaxed);
                if start >= n {
                    break;
                }
                let end = (start + CHUNK).min(n);
                for i in start..end {
                    index.insert(grainstore::model::Sid(i as u128), &vectors[i]);
                }
            }
        }));
    }
    for h in handles {
        h.join().expect("join");
    }
    let build = build_start.elapsed();
    let rate = n as f64 / build.as_secs_f64();
    println!(
        "  build: {build:.2?}  ({} vectors, {rate:.0}/s)",
        index.len()
    );
    println!(
        "  vs single-thread HNSW baseline (prior 1M run): 2216/s → speedup ≈ {:.1}×\n",
        rate / 2216.0
    );

    // Query recall/latency on the sharded index (exact filtered-kNN oracle).
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

    // Post-filter over the sharded index: over-fetch per shard then filter by category.
    println!("== Query (sharded index): recall vs exact filtered-kNN ==");
    println!(
        "  {:>10}  {:>10}  {:>10}  {:>8}",
        "fetch_k", "p50", "p99", "recall"
    );
    for &fetch_mult in &[20usize, 40, 80] {
        let fetch_k = k * fetch_mult;
        let mut samples = Vec::with_capacity(queries);
        let mut total_recall = 0.0f64;
        for (qi, (q, tgt)) in qset.iter().enumerate() {
            let t0 = Instant::now();
            let cands = index.search(q, fetch_k, ef);
            let got: std::collections::HashSet<u128> = cands
                .iter()
                .filter(|c| cats[c.sid.0 as usize] == *tgt)
                .take(k)
                .map(|c| c.sid.0)
                .collect();
            samples.push(t0.elapsed().as_nanos() as u64);
            total_recall += oracle[qi].intersection(&got).count() as f64 / k as f64;
        }
        samples.sort_unstable();
        println!(
            "  {:>10}  {:>10.2?}  {:>10.2?}  {:>8.3}",
            fetch_k,
            pct(&samples, 50.0),
            pct(&samples, 99.0),
            total_recall / queries as f64
        );
    }
    println!("\ndone.");
}
