//! P1 benchmark: the mixed `near ⋈ filter` query end to end, scalable.
//!
//! Builds N grains through the full pipeline (truth store → CDC → materializer →
//! HNSW), printing build progress, then measures mixed-query latency percentiles
//! and recall versus an exact filtered-kNN oracle, swept across over-fetch.
//!
//! Run:   cargo run --release --example bench_p1
//! Scale: GS_N=1000000 GS_DIM=32 GS_QUERIES=200 cargo run --release --example bench_p1
//!
//! NOTE: the materializer indexes single-threaded (the HNSW uses a coarse lock),
//! so "build rate" measures that path honestly — at scale a parallel-build or
//! FAISS/DiskANN backend (same `VectorIndex` trait) is the upgrade.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use grainstore::embed::{encode_value_with_header, RawVectorEmbedder};
use grainstore::model::{Grain, PredId, Sid, Val};
use grainstore::{
    near_join_select, Hnsw, HnswConfig, MixedQuery, TruthStore, VectorIndex, VectorMaterializer,
};

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
    let dir =
        std::env::temp_dir().join(format!("grainstore_benchp1_{}_{}", std::process::id(), tag));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("mkdir");
    dir.join("wal.log")
}

fn main() {
    let dim = env_usize("GS_DIM", 32);
    let n = env_usize("GS_N", 20_000);
    let categories = env_usize("GS_CATS", 10) as u8;
    let k = env_usize("GS_K", 10);
    let ef = env_usize("GS_EF", 128);
    let queries = env_usize("GS_QUERIES", 500);
    let pred = PredId(0);

    println!("GrainStore P1 benchmark — mixed near ⋈ filter");
    println!(
        "  n={n} dim={dim} categories={categories} (sel≈{:.3}) k={k} ef={ef} queries={queries}",
        1.0 / categories as f64
    );

    // Ground truth kept for the exact oracle.
    let mut rng = R::new(2024);
    let vectors: Arc<Vec<Vec<f32>>> = Arc::new((0..n).map(|_| rng.vec(dim)).collect());
    let cats: Arc<Vec<u8>> = Arc::new(
        (0..n)
            .map(|_| (rng.nx() % categories as u64) as u8)
            .collect(),
    );

    // Build through the full pipeline, with progress.
    let truth = Arc::new(TruthStore::open(&tmp("p1")).expect("open"));
    let rx = truth.subscribe();
    let index = Arc::new(Hnsw::new(dim, HnswConfig::default()));
    let embed = Arc::new(RawVectorEmbedder::new(dim, 1));
    // Kept alive for the whole run: owns the materializer thread.
    let _mat = VectorMaterializer::spawn(rx, index.clone(), embed.clone());

    let last_seq = Arc::new(AtomicU64::new(0));
    let build_start = Instant::now();
    let threads = 64usize;
    let mut handles = Vec::new();
    for t in 0..threads {
        let (truth, vectors, cats, last_seq) = (
            truth.clone(),
            vectors.clone(),
            cats.clone(),
            last_seq.clone(),
        );
        handles.push(std::thread::spawn(move || {
            let mut i = t;
            while i < n {
                let value = encode_value_with_header(&[cats[i]], &vectors[i]);
                let seq = truth
                    .put(Sid(i as u128), pred, value, 1.0, 0, i as u128)
                    .expect("put")
                    .0;
                last_seq.fetch_max(seq, Ordering::AcqRel);
                i += threads;
            }
        }));
    }

    // Poll the index watermark for progress until it reflects all N.
    let mut last_print = Instant::now();
    let mut last_count = 0usize;
    loop {
        let c = index.len();
        if c >= n {
            break;
        }
        if last_print.elapsed() >= Duration::from_secs(2) {
            let rate = (c - last_count) as f64 / last_print.elapsed().as_secs_f64();
            println!(
                "  indexing… {c}/{n}  ({rate:.0}/s, {:.1?} elapsed)",
                build_start.elapsed()
            );
            last_print = Instant::now();
            last_count = c;
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    for h in handles {
        h.join().expect("join");
    }
    let build = build_start.elapsed();
    println!(
        "  build complete: {build:.2?}  ({} vectors, {:.0}/s)\n",
        index.len(),
        n as f64 / build.as_secs_f64()
    );

    // Precompute the exact filtered-kNN oracle for the query set (once).
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

    println!("== Mixed query: latency & recall vs exact filtered-kNN oracle ==");
    println!(
        "  {:>10}  {:>10}  {:>10}  {:>8}",
        "over_fetch", "p50", "p99", "recall"
    );
    for &over_fetch in &[2usize, 5, 10, 20, 40] {
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
            let ranked = near_join_select(truth.as_ref(), index.as_ref(), &mq, |g| {
                category_of(g) == Some(tgt)
            })
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
