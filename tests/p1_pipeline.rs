//! End-to-end P1: write grains to the truth store → CDC → materializer embeds
//! and indexes → `Near ⋈ Select` mixed query. Validated against an exact
//! filtered-kNN oracle. The predicate (category) is INDEPENDENT of vector
//! geometry — the hard filtered-ANN case that exercises post-filter over-fetch.

use std::sync::Arc;
use std::time::Duration;

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

fn tmp(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("grainstore_p1_{}_{}", std::process::id(), tag));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("mkdir");
    dir.join("wal.log")
}

#[test]
fn mixed_filtered_query_matches_exact_oracle() {
    let dim = 24;
    let n = 3000usize;
    let categories = 5u8; // selectivity ≈ 1/5
    let k = 10;
    let ef = 128;
    let over_fetch = 12; // ≳ #categories so post-filter survives selectivity
    let pred = PredId(0);

    // Ground-truth data we keep to compute the exact oracle.
    let mut rng = R::new(2024);
    let vectors: Vec<Vec<f32>> = (0..n).map(|_| rng.vec(dim)).collect();
    let cats: Vec<u8> = (0..n)
        .map(|_| (rng.nx() % categories as u64) as u8)
        .collect();

    // Stand up the truth store + vector materializer.
    let truth = TruthStore::open(&tmp("mix")).expect("open");
    let rx = truth.subscribe();
    let index = Arc::new(Hnsw::new(dim, HnswConfig::default()));
    let embed = Arc::new(RawVectorEmbedder::new(dim, 1)); // 1-byte category header
    let mat = VectorMaterializer::spawn(rx, index.clone(), embed.clone());

    // Write grains: value = [category][vector].
    let mut last_seq = 0u64;
    for i in 0..n {
        let value = encode_value_with_header(&[cats[i]], &vectors[i]);
        last_seq = truth
            .put(Sid(i as u128), pred, value, 1.0, 0, i as u128)
            .expect("put")
            .0;
    }
    // Wait for the index to reflect every commit (bounded staleness → 0 here).
    assert!(
        mat.wait_for(last_seq, Duration::from_secs(30)),
        "materializer lagged"
    );
    assert_eq!(index.len(), n, "index missing vectors");

    // Run several filtered mixed queries and compare to the exact filtered kNN.
    let mut total_recall = 0.0f64;
    let trials = 50;
    for _ in 0..trials {
        let q = rng.vec(dim);
        let target = (rng.nx() % categories as u64) as u8;

        // Exact filtered oracle: nearest k among category == target.
        let mut exact: Vec<(usize, f32)> = (0..n)
            .filter(|&i| cats[i] == target)
            .map(|i| (i, l2(&q, &vectors[i])))
            .collect();
        exact.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let exact_top: std::collections::HashSet<u128> =
            exact.iter().take(k).map(|(i, _)| *i as u128).collect();

        // GrainStore mixed query.
        let mq = MixedQuery {
            pred,
            query: &q,
            k,
            ef,
            over_fetch,
        };
        let ranked = near_join_select(&truth, index.as_ref(), &mq, |g| {
            category_of(g) == Some(target)
        })
        .expect("query");

        // Every returned grain must satisfy the predicate (join+filter correctness).
        assert!(
            ranked.iter().all(|r| category_of(&r.grain) == Some(target)),
            "predicate violated in result"
        );

        let got: std::collections::HashSet<u128> = ranked.iter().map(|r| r.grain.sid.0).collect();
        let hit = exact_top.intersection(&got).count();
        total_recall += hit as f64 / k as f64;
    }
    let recall = total_recall / trials as f64;
    println!("P1 filtered mixed-query recall@{k}: {recall:.3} (cats={categories}, over_fetch={over_fetch})");
    assert!(
        recall >= 0.85,
        "filtered mixed-query recall too low: {recall:.3}"
    );
}

#[test]
fn tombstone_removes_from_index() {
    let dim = 8;
    let pred = PredId(0);
    let mut rng = R::new(5);

    let truth = TruthStore::open(&tmp("tomb")).expect("open");
    let rx = truth.subscribe();
    let index = Arc::new(Hnsw::new(dim, HnswConfig::default()));
    let embed = Arc::new(RawVectorEmbedder::new(dim, 1));
    let mat = VectorMaterializer::spawn(rx, index.clone(), embed.clone());

    let mut last = 0u64;
    for i in 0..200u128 {
        let value = encode_value_with_header(&[0u8], &rng.vec(dim));
        last = truth.put(Sid(i), pred, value, 1.0, 0, i).expect("put").0;
    }
    assert!(mat.wait_for(last, Duration::from_secs(10)));
    assert_eq!(index.len(), 200);

    // Delete half (idempotency keys distinct from the puts).
    for i in 0..100u128 {
        last = truth
            .delete(Sid(i), pred, 1.0, 0, 1_000_000 + i)
            .expect("del")
            .0;
    }
    assert!(mat.wait_for(last, Duration::from_secs(10)));
    assert_eq!(index.len(), 100, "tombstones not reflected in index");
}
