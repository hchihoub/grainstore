//! HNSW correctness: recall of the approximate index measured against the exact
//! brute-force oracle on the same data. This is the gate that the scalable index
//! does not silently lose neighbours.

use grainstore::model::Sid;
use grainstore::vector::{BruteForceIndex, Hnsw, HnswConfig, VectorIndex};

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

#[test]
fn hnsw_recall_matches_brute_force() {
    let dim = 32;
    let n = 2000;
    let k = 10;
    let ef = 64;
    let queries = 100;

    let mut rng = R::new(42);
    let data: Vec<Vec<f32>> = (0..n).map(|_| rng.vec(dim)).collect();

    let hnsw = Hnsw::new(dim, HnswConfig::default());
    let brute = BruteForceIndex::new();
    for (i, v) in data.iter().enumerate() {
        hnsw.insert(Sid(i as u128), v);
        brute.insert(Sid(i as u128), v);
    }
    assert_eq!(hnsw.len(), n);

    let mut total_recall = 0.0f64;
    for _ in 0..queries {
        let q = rng.vec(dim);
        let exact: std::collections::HashSet<u128> = brute
            .search(&q, k, 0)
            .into_iter()
            .map(|c| c.sid.0)
            .collect();
        let approx: std::collections::HashSet<u128> = hnsw
            .search(&q, k, ef)
            .into_iter()
            .map(|c| c.sid.0)
            .collect();
        let hit = exact.intersection(&approx).count();
        total_recall += hit as f64 / k as f64;
    }
    let recall = total_recall / queries as f64;
    println!("HNSW recall@{k} (ef={ef}, n={n}, d={dim}): {recall:.3}");
    assert!(recall >= 0.90, "HNSW recall too low: {recall:.3}");
}

#[test]
fn brute_force_is_exact() {
    // The oracle must return true nearest neighbours in distance order.
    let dim = 8;
    let mut rng = R::new(7);
    let data: Vec<Vec<f32>> = (0..200).map(|_| rng.vec(dim)).collect();
    let brute = BruteForceIndex::new();
    for (i, v) in data.iter().enumerate() {
        brute.insert(Sid(i as u128), v);
    }
    let q = rng.vec(dim);
    let got = brute.search(&q, 5, 0);

    // Independently compute the true top-5.
    let mut all: Vec<(usize, f32)> = data
        .iter()
        .enumerate()
        .map(|(i, v)| (i, l2(&q, v)))
        .collect();
    all.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    for (rank, c) in got.iter().enumerate() {
        assert_eq!(c.sid.0 as usize, all[rank].0, "rank {rank} mismatch");
    }
}

#[test]
fn hnsw_handles_deletes() {
    let dim = 16;
    let mut rng = R::new(99);
    let hnsw = Hnsw::new(dim, HnswConfig::default());
    for i in 0..500u128 {
        hnsw.insert(Sid(i), &rng.vec(dim));
    }
    assert_eq!(hnsw.len(), 500);
    for i in 0..100u128 {
        hnsw.remove(Sid(i));
    }
    assert_eq!(hnsw.len(), 400);
    // Deleted sids never appear in results.
    let q = rng.vec(dim);
    let res = hnsw.search(&q, 50, 128);
    assert!(
        res.iter().all(|c| c.sid.0 >= 100),
        "a deleted sid was returned"
    );
}
