//! GrainStore P0 benchmark harness.
//!
//! Measures the transactional core that exists today:
//!   1. Write throughput, single- vs multi-threaded — demonstrates group-commit
//!      fsync amortization (aggregate throughput rises as concurrency grows).
//!   2. Point-read latency percentiles (the MVCC seek path).
//!   3. Recovery time as a function of WAL size.
//!
//! Methodology: warmup, fixed op counts, per-op latency capture, percentiles
//! from sorted samples, wall-clock throughput. Dependency-free.
//!
//! Run:  cargo run --release --example bench
//! Tune: GS_BENCH_WRITES, GS_BENCH_READS, GS_BENCH_KEYS env vars.
//!
//! NOTE: this benchmarks P0 only. The P1 gate — mixed `near ⋈ filter` vs a
//! Postgres+pgvector baseline — slots in once the vector plane lands; its
//! harness reuses the percentile/report scaffolding below.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use grainstore::model::{PredId, Sid};
use grainstore::truth::ReadMode;
use grainstore::TruthStore;

// ---------------------------------------------------------------------------
// Tiny stats + PRNG (no deps)
// ---------------------------------------------------------------------------

struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
}

/// Percentile (nearest-rank) from an unsorted slice of nanosecond latencies.
fn pct(sorted_ns: &[u64], p: f64) -> Duration {
    if sorted_ns.is_empty() {
        return Duration::ZERO;
    }
    let rank = ((p / 100.0) * (sorted_ns.len() as f64 - 1.0)).round() as usize;
    Duration::from_nanos(sorted_ns[rank.min(sorted_ns.len() - 1)])
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn tmp(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("grainstore_bench_{}_{}", std::process::id(), tag));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("mkdir");
    dir.join("wal.log")
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

/// Aggregate write throughput at a given thread count. Each thread issues
/// `per_thread` durable (fsync'd) writes; more threads => larger group-commit
/// batches => higher aggregate throughput until fsync saturates.
fn bench_write_throughput(threads: usize, per_thread: usize) -> (f64, Duration) {
    let wal = tmp(&format!("wt{threads}"));
    let store = Arc::new(TruthStore::open(&wal).expect("open"));

    let start = Instant::now();
    let mut handles = Vec::new();
    for t in 0..threads {
        let store = store.clone();
        handles.push(std::thread::spawn(move || {
            let base = (t as u128) << 64;
            for i in 0..per_thread {
                let idem = base | (i as u128);
                store
                    .put(Sid(idem), PredId(0), vec![0u8; 32], 1.0, 0, idem)
                    .expect("put");
            }
        }));
    }
    for h in handles {
        h.join().expect("join");
    }
    let elapsed = start.elapsed();
    let total = (threads * per_thread) as f64;
    let throughput = total / elapsed.as_secs_f64();
    (throughput, elapsed)
}

/// Bulk-load `n` records (sids `0..n`) concurrently so group-commit batches the
/// setup fsyncs — otherwise a serial fsync'd load dominates every benchmark.
fn populate_concurrent(store: &Arc<TruthStore>, n: usize, threads: usize) {
    let threads = threads.max(1);
    let mut handles = Vec::new();
    for t in 0..threads {
        let store = store.clone();
        handles.push(std::thread::spawn(move || {
            let mut i = t;
            while i < n {
                store
                    .put(Sid(i as u128), PredId(0), vec![7u8; 32], 1.0, 0, i as u128)
                    .expect("put");
                i += threads;
            }
        }));
    }
    for h in handles {
        h.join().expect("join");
    }
}

/// Point-read latency percentiles over a warm, pre-populated store.
fn bench_read_latency(keys: usize, reads: usize) -> (Vec<u64>, f64) {
    let wal = tmp("rd");
    let store = Arc::new(TruthStore::open(&wal).expect("open"));
    populate_concurrent(&store, keys, 64);

    // warmup
    let mut rng = Lcg::new(1);
    for _ in 0..(reads / 10).max(1) {
        let k = (rng.next_u64() as usize) % keys;
        let _ = store.get(Sid(k as u128), PredId(0), ReadMode::Strong);
    }

    let mut samples = Vec::with_capacity(reads);
    let start = Instant::now();
    for _ in 0..reads {
        let k = (rng.next_u64() as usize) % keys;
        let t0 = Instant::now();
        let got = store
            .get(Sid(k as u128), PredId(0), ReadMode::Strong)
            .expect("get");
        samples.push(t0.elapsed().as_nanos() as u64);
        debug_assert!(got.is_some());
    }
    let total_s = start.elapsed().as_secs_f64();
    let throughput = reads as f64 / total_s;
    samples.sort_unstable();
    (samples, throughput)
}

/// Recovery time: write `n` records, drop, reopen, measure replay. The replay
/// itself does no fsync (it rebuilds the volatile KV), so this isolates decode +
/// apply cost; the setup load is parallelized so it does not dominate.
fn bench_recovery(n: usize) -> (Duration, usize) {
    let wal = tmp("rec");
    {
        let store = Arc::new(TruthStore::open(&wal).expect("open"));
        populate_concurrent(&store, n, 64);
        // drop joins committer; WAL durable
        drop(Arc::try_unwrap(store).map_err(|_| ()).expect("sole owner"));
    }

    let start = Instant::now();
    let store = TruthStore::open(&wal).expect("reopen");
    let elapsed = start.elapsed();
    (elapsed, store.total_versions())
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

fn main() {
    let writes = env_usize("GS_BENCH_WRITES", 2400);
    let reads = env_usize("GS_BENCH_READS", 200_000);
    let keys = env_usize("GS_BENCH_KEYS", 50_000);

    println!("GrainStore P0 benchmark");
    println!(
        "  host threads available: {}",
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(0)
    );
    println!();

    // 1. Write throughput / group-commit amortization
    println!("== Write throughput (durable, fsync per group) ==");
    println!("  {:>8}  {:>14}  {:>12}", "threads", "writes/sec", "wall");
    for &threads in &[1usize, 2, 4, 8] {
        let per = (writes / threads).max(1);
        let (tput, wall) = bench_write_throughput(threads, per);
        println!("  {threads:>8}  {tput:>14.0}  {wall:>12.2?}");
    }
    println!();

    // 2. Read latency
    println!("== Point-read latency (MVCC seek, warm) ==");
    let (samples, rtput) = bench_read_latency(keys, reads);
    println!("  keys={keys} reads={reads}");
    println!("  throughput : {rtput:>12.0} reads/sec");
    println!("  p50        : {:>12.2?}", pct(&samples, 50.0));
    println!("  p99        : {:>12.2?}", pct(&samples, 99.0));
    println!("  p99.9      : {:>12.2?}", pct(&samples, 99.9));
    println!(
        "  max        : {:>12.2?}",
        Duration::from_nanos(*samples.last().unwrap_or(&0))
    );
    println!();

    // 3. Recovery
    println!("== Recovery (WAL replay) ==");
    println!(
        "  {:>10}  {:>12}  {:>14}",
        "records", "recover", "records/sec"
    );
    for &n in &[1_000usize, 10_000, 50_000] {
        let (dur, recovered) = bench_recovery(n);
        let rate = recovered as f64 / dur.as_secs_f64();
        println!("  {recovered:>10}  {dur:>12.2?}  {rate:>14.0}");
    }
    println!();
    println!("done.");
}
