//! Staged disclosure benchmark: two wins.
//!   1. Token reduction — Stage 0 (a few representatives + stats) costs far fewer
//!      tokens than Stage 1 (the full k grains). If the summary suffices, the
//!      agent pays only Stage 0.
//!   2. Continuation reuse — drilling to Stage 1 returns from pinned state with
//!      NO re-search, so drill latency ≪ the original query latency.
//!
//! Grains carry a 256-byte payload (simulating a document) so token counts are
//! representative of real workloads.
//!
//! Run: cargo run --release --example bench_disclosure

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use grainstore::embed::{Embedder, RawVectorEmbedder};
use grainstore::model::{PredId, Sid};
use grainstore::vector::{HnswConfig, ShardedHnsw, VectorIndex};
use grainstore::{EngineConfig, Filter, GrainEngine, Stage, WriteMeta};

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

fn pct(s: &[u64], p: f64) -> Duration {
    if s.is_empty() {
        return Duration::ZERO;
    }
    let r = ((p / 100.0) * (s.len() as f64 - 1.0)).round() as usize;
    Duration::from_nanos(s[r.min(s.len() - 1)])
}

fn tmp(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "grainstore_benchdisc_{}_{}",
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
    let header_len = 256; // byte 0 = category, rest = "document" payload
    let k = 100;
    let reps = 5;
    let trials = 300;
    let pred = PredId(0);

    println!("Staged disclosure benchmark — token reduction + continuation reuse");
    println!("  n={n} dim={dim} payload={header_len}B k={k} reps={reps}\n");

    let mut rng = R::new(2024);
    let vectors: Arc<Vec<Vec<f32>>> = Arc::new((0..n).map(|_| rng.vec(dim)).collect());

    let cfg = EngineConfig::new(dim)
        .with_header(header_len)
        .with_workers(14);
    let index: Arc<dyn VectorIndex> = Arc::new(ShardedHnsw::new(16, dim, HnswConfig::default()));
    let embed: Arc<dyn Embedder> = Arc::new(RawVectorEmbedder::new(dim, header_len));
    let engine = Arc::new(GrainEngine::open_with(&tmp("disc"), cfg, index, embed).expect("open"));

    let last_seq = Arc::new(AtomicU64::new(0));
    let mut handles = Vec::new();
    for t in 0..64usize {
        let (engine, vectors, last_seq) = (engine.clone(), vectors.clone(), last_seq.clone());
        handles.push(std::thread::spawn(move || {
            let mut i = t;
            while i < n {
                let mut header = vec![0u8; header_len];
                header[0] = (i % 2) as u8; // two categories
                for (j, b) in header.iter_mut().enumerate().skip(1) {
                    *b = (i.wrapping_add(j) & 0xff) as u8; // filler payload
                }
                let h = engine
                    .put_vector(
                        Sid(i as u128),
                        pred,
                        &header,
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

    let filter = Filter::HeaderByteEq {
        offset: 0,
        value: 0,
    };
    let mut qrng = R::new(777);
    let (mut stage0_lat, mut drill_lat) = (Vec::new(), Vec::new());
    let (mut s0_tokens, mut s1_tokens) = (0usize, 0usize);

    for _ in 0..trials {
        let q = qrng.vec(dim);

        let t0 = Instant::now();
        let s0 = engine
            .query_staged(pred, &q, &filter, k, 0.90, reps)
            .expect("staged");
        stage0_lat.push(t0.elapsed().as_nanos() as u64);

        let handle = s0.continuation.unwrap();
        if let Stage::Summary(s) = &s0.stage {
            s0_tokens = s0.est_tokens;
            s1_tokens = s.est_full_tokens;
        }

        let t1 = Instant::now();
        let _s1 = engine.drill(handle).expect("drill").expect("live");
        drill_lat.push(t1.elapsed().as_nanos() as u64);
    }
    stage0_lat.sort_unstable();
    drill_lat.sort_unstable();

    println!("== Token cost ==");
    println!("  Stage 0 (summary, {reps} reps + stats): {s0_tokens} tokens");
    println!("  Stage 1 (full {k} grains):              {s1_tokens} tokens");
    println!(
        "  reduction if summary suffices:          {:.1}×\n",
        s1_tokens as f64 / s0_tokens.max(1) as f64
    );

    println!("== Latency (continuation reuse) ==");
    let q_p50 = pct(&stage0_lat, 50.0);
    let d_p50 = pct(&drill_lat, 50.0);
    println!("  query_staged p50 (runs the search): {q_p50:.2?}");
    println!("  drill        p50 (pinned, no search): {d_p50:.2?}");
    println!(
        "  drill speedup (search amortized):    {:.0}×",
        q_p50.as_secs_f64() / d_p50.as_secs_f64().max(1e-9)
    );
    println!("\ndone.");
}
