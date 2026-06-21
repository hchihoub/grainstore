//! Shared core for the GrainStore deployment binaries.
//!
//! A `Store` wraps the engine with the value layout `[category: u8][amount: i64
//! BE][text…]`, a deterministic text embedder, and load/query helpers. The server
//! daemon and the MCP server both build on this.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use grainstore::embed::Embedder;
use grainstore::model::{Grain, PredId, Sid, Val};
use grainstore::vector::{HnswConfig, ShardedHnsw, VectorIndex};
use grainstore::{EngineConfig, GrainEngine, Hlc, MixedQuery, WriteMeta};

pub const PRED: PredId = PredId(0);
/// Bytes before the text in a stored value: `[category: 1][amount: 8]`.
pub const TEXT_OFFSET: usize = 9;

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Map text to a unit vector via signed feature hashing of its 1–3 byte n-grams.
pub fn embed_text(text: &[u8], dim: usize) -> Vec<f32> {
    let mut v = vec![0f32; dim];
    let n = text.len();
    for w in 1..=3usize {
        if n < w {
            continue;
        }
        for i in 0..=n - w {
            let h = fnv1a(&text[i..i + w]);
            let idx = (h % dim as u64) as usize;
            let sign = if (h >> 63) & 1 == 0 { 1.0 } else { -1.0 };
            v[idx] += sign;
        }
    }
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
    for x in &mut v {
        *x /= norm;
    }
    v
}

struct TextHashEmbedder {
    dim: usize,
}
impl Embedder for TextHashEmbedder {
    fn embed(&self, value: &[u8]) -> Option<Vec<f32>> {
        if value.len() <= TEXT_OFFSET {
            return None;
        }
        Some(embed_text(&value[TEXT_OFFSET..], self.dim))
    }
    fn dim(&self) -> usize {
        self.dim
    }
}

fn encode_val(cat: u8, amount: i64, text: &str) -> Vec<u8> {
    let mut v = Vec::with_capacity(TEXT_OFFSET + text.len());
    v.push(cat);
    v.extend_from_slice(&amount.to_be_bytes());
    v.extend_from_slice(text.as_bytes());
    v
}

fn decode_val(v: &Val) -> (u8, i64, String) {
    match v {
        Val::Bytes(b) if b.len() >= TEXT_OFFSET => (
            b[0],
            i64::from_be_bytes(b[1..9].try_into().expect("8 bytes")),
            String::from_utf8_lossy(&b[TEXT_OFFSET..]).into_owned(),
        ),
        _ => (0, 0, String::new()),
    }
}

/// A grain as supplied by a client.
pub struct GrainRec {
    pub sid: u64,
    pub category: u8,
    pub amount: i64,
    pub text: String,
}

/// A query result row.
pub struct QueryHit {
    pub sid: u64,
    pub category: u8,
    pub amount: i64,
    pub text: String,
    pub distance: f32,
}

/// Configuration for opening a store.
pub struct Config {
    pub data_dir: String,
    pub dim: usize,
    pub shards: usize,
    pub workers: usize,
}

/// The deployable store: engine + value codec + load/query.
pub struct Store {
    engine: GrainEngine,
    last_seq: AtomicU64,
    pub dim: usize,
}

impl Store {
    /// Open (or recover) a store and rebuild the index from the durable truth.
    /// Returns the store and how many grains were restored.
    pub fn open(cfg: &Config) -> Result<(Self, usize), String> {
        std::fs::create_dir_all(&cfg.data_dir).map_err(|e| e.to_string())?;
        let wal = std::path::Path::new(&cfg.data_dir).join("wal.log");
        let ecfg = EngineConfig::new(cfg.dim).with_header(TEXT_OFFSET).with_workers(cfg.workers);
        let index: Arc<dyn VectorIndex> =
            Arc::new(ShardedHnsw::new(cfg.shards, cfg.dim, HnswConfig::default()));
        let embed: Arc<dyn Embedder> = Arc::new(TextHashEmbedder { dim: cfg.dim });
        let engine = GrainEngine::open_with(&wal, ecfg, index, embed).map_err(|e| e.to_string())?;
        let restored = engine.rebuild_index_from_truth().map_err(|e| e.to_string())?;
        Ok((Self { engine, last_seq: AtomicU64::new(0), dim: cfg.dim }, restored))
    }

    pub fn indexed(&self) -> usize {
        self.engine.index_len()
    }

    /// Durably store and index a batch; blocks until queryable. Returns the count.
    pub fn load(&self, recs: &[GrainRec]) -> Result<usize, String> {
        let mut max_seq = self.last_seq.load(Ordering::Acquire);
        for r in recs {
            let value = encode_val(r.category, r.amount, &r.text);
            let h = self
                .engine
                .put_raw(Sid(r.sid as u128), PRED, value, WriteMeta::new(r.sid as u128))
                .map_err(|e| e.to_string())?;
            max_seq = max_seq.max(h.0);
        }
        self.last_seq.fetch_max(max_seq, Ordering::AcqRel);
        self.engine.sync(Hlc(max_seq), Duration::from_secs(300));
        Ok(recs.len())
    }

    /// `near ⋈ select`: semantic text + optional category + optional amount range.
    pub fn query(
        &self,
        text: &str,
        category: Option<u8>,
        min_amount: Option<i64>,
        max_amount: Option<i64>,
        k: usize,
    ) -> Result<Vec<QueryHit>, String> {
        let qvec = embed_text(text.as_bytes(), self.dim);
        let predicate = |g: &Grain| {
            let (c, amount, _) = decode_val(&g.val);
            category.map(|w| c == w).unwrap_or(true)
                && min_amount.map(|lo| amount >= lo).unwrap_or(true)
                && max_amount.map(|hi| amount <= hi).unwrap_or(true)
        };
        let mq = MixedQuery { pred: PRED, query: &qvec, k, ef: 128, over_fetch: 64 };
        let ranked = self.engine.query(&mq, predicate).map_err(|e| e.to_string())?;
        Ok(ranked
            .iter()
            .map(|r| {
                let (c, amount, t) = decode_val(&r.grain.val);
                QueryHit { sid: r.grain.sid.0 as u64, category: c, amount, text: t, distance: r.dist }
            })
            .collect())
    }
}
