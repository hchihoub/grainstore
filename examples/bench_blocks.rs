//! P3 block-pruning benchmark.
//!
//! Builds N grains (from latent clusters, like real embeddings) into
//! meaning-clustered self-describing blocks, then measures how many blocks each
//! query actually TOUCHES after descriptor pruning — vs the total. Point lookups
//! prune via the per-block Bloom filter; kNN prunes via the centroid+radius bound
//! (exact); category counts prune via the sketch.
//!
//! Run: cargo run --release --example bench_blocks

use std::time::Instant;

use grainstore::block::{BlockStore, GrainRec};

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

fn main() {
    let dim = 32;
    let n = 100_000usize;
    let latent = 100usize; // latent clusters (data has structure, like embeddings)
    let n_blocks = 256usize;
    let cats = 4u8;

    println!("P3 block pruning benchmark");
    println!("  n={n} dim={dim} latent_clusters={latent} blocks={n_blocks}\n");

    // Latent cluster centers, then points = center + small noise.
    let mut rng = R::new(2024);
    let centers: Vec<Vec<f32>> = (0..latent).map(|_| rng.vec(dim)).collect();
    let recs: Vec<GrainRec> = (0..n)
        .map(|i| {
            let c = &centers[(rng.nx() as usize) % latent];
            let v: Vec<f32> = c.iter().map(|&x| x + (rng.f32() - 0.5) * 0.15).collect();
            GrainRec {
                sid: i as u128,
                pred: 0,
                cat: (i % cats as usize) as u8,
                vector: v,
                t_tx: i as u64,
                conf: 1.0,
            }
        })
        .collect();

    let t0 = Instant::now();
    let store = BlockStore::build_clustered(&recs, n_blocks, 5);
    println!(
        "  built {} blocks in {:.2?}\n",
        store.block_count(),
        t0.elapsed()
    );
    let total = store.block_count();

    // Point lookups — Bloom pruning.
    let mut prng = R::new(11);
    let mut sum_get = 0usize;
    let probes = 2000;
    for _ in 0..probes {
        let sid = (prng.nx() as usize % n) as u128;
        let (_hit, s) = store.get(sid);
        sum_get += s.blocks_touched;
    }
    let avg_get = sum_get as f64 / probes as f64;

    // kNN — centroid+radius pruning (exact).
    let mut qrng = R::new(77);
    let mut sum_near = 0usize;
    let queries = 500;
    for _ in 0..queries {
        // query near a latent cluster (realistic)
        let c = &centers[(qrng.nx() as usize) % latent];
        let q: Vec<f32> = c.iter().map(|&x| x + (qrng.f32() - 0.5) * 0.15).collect();
        let (_res, s) = store.near(&q, 10);
        sum_near += s.blocks_touched;
    }
    let avg_near = sum_near as f64 / queries as f64;

    // Category count — sketch pruning.
    let (cnt, cat_stats) = store.count_category(0);

    println!("== Blocks touched (of {total}) after descriptor pruning ==");
    println!(
        "  point lookup (Bloom)        : {avg_get:.2}  ({:.1}% pruned)",
        100.0 * (1.0 - avg_get / total as f64)
    );
    println!(
        "  kNN k=10 (centroid+radius)  : {avg_near:.1}  ({:.1}% pruned)",
        100.0 * (1.0 - avg_near / total as f64)
    );
    println!(
        "  category=0 count (sketch)   : {} blocks  (found {cnt} grains)",
        cat_stats.blocks_touched
    );
    println!("\nkNN pruning is EXACT (triangle-inequality bound) — same results as a full scan.");
    println!("done.");
}
