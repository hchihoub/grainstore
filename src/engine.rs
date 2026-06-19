//! `GrainEngine` — the batteries-included facade.
//!
//! Bundles the truth store, a vector index, an embedder, and the materializer
//! into one object with the wiring done for you. The index and embedder are held
//! as **trait objects** (`Arc<dyn VectorIndex>` / `Arc<dyn Embedder>`), so the
//! choice is a runtime value:
//!
//! - [`GrainEngine::open`] — the default embedded setup: an in-process [`Hnsw`]
//!   index and a [`RawVectorEmbedder`]. Zero wiring.
//! - [`GrainEngine::open_with`] — bring your own index and/or embedder (a FAISS
//!   backend, a remote model-server embedder, …) with no other code changes.
//!
//! Because the index is a derived, rebuildable materialization (it is fed from
//! the truth CDC stream), swapping it later is sound: construct a new engine
//! over the same WAL with a different backend and it repopulates from truth.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use crate::embed::{encode_value_with_header, Embedder, RawVectorEmbedder};
use crate::error::Result;
use crate::materializer::VectorMaterializer;
use crate::model::{Grain, Hlc, PredId, Sid};
use crate::query::{near_join_select, MixedQuery, Ranked};
use crate::truth::TruthStore;
use crate::vector::{Hnsw, HnswConfig, VectorIndex};

/// Configuration for the default engine. Build with [`EngineConfig::new`] and
/// the chainable setters.
#[derive(Clone, Copy, Debug)]
pub struct EngineConfig {
    /// Embedding dimensionality.
    pub dim: usize,
    /// Bytes of opaque header prepended to the vector in stored values (used by
    /// query predicates, e.g. a category tag). 0 for none.
    pub header_len: usize,
    /// HNSW tuning for the default index.
    pub hnsw: HnswConfig,
    /// Materializer worker threads. >1 parallelizes index build via a worker
    /// pool — effective with a sharded index. Default 1 (single-threaded).
    pub materializer_workers: usize,
}

impl EngineConfig {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            header_len: 0,
            hnsw: HnswConfig::default(),
            materializer_workers: 1,
        }
    }
    pub fn with_header(mut self, header_len: usize) -> Self {
        self.header_len = header_len;
        self
    }
    pub fn with_hnsw(mut self, hnsw: HnswConfig) -> Self {
        self.hnsw = hnsw;
        self
    }
    pub fn with_workers(mut self, workers: usize) -> Self {
        self.materializer_workers = workers.max(1);
        self
    }
}

/// Per-write metadata. Build with [`WriteMeta::new`] (confidence 1.0, t_valid 0)
/// and override as needed.
#[derive(Clone, Copy, Debug)]
pub struct WriteMeta {
    pub c: f32,
    pub t_valid: u64,
    pub idem_key: u128,
}

impl WriteMeta {
    pub fn new(idem_key: u128) -> Self {
        Self {
            c: 1.0,
            t_valid: 0,
            idem_key,
        }
    }
    pub fn with_confidence(mut self, c: f32) -> Self {
        self.c = c;
        self
    }
    pub fn with_t_valid(mut self, t_valid: u64) -> Self {
        self.t_valid = t_valid;
        self
    }
}

/// The unified engine: truth + vector plane + materializer, wired together.
pub struct GrainEngine {
    truth: Arc<TruthStore>,
    index: Arc<dyn VectorIndex>,
    #[allow(dead_code)]
    embed: Arc<dyn Embedder>,
    mat: VectorMaterializer,
    cfg: EngineConfig,
}

impl GrainEngine {
    /// Default embedded engine: in-process HNSW + raw-vector embedder.
    pub fn open(path: &Path, cfg: EngineConfig) -> Result<Self> {
        let index: Arc<dyn VectorIndex> = Arc::new(Hnsw::new(cfg.dim, cfg.hnsw));
        let embed: Arc<dyn Embedder> = Arc::new(RawVectorEmbedder::new(cfg.dim, cfg.header_len));
        Self::open_with(path, cfg, index, embed)
    }

    /// Bring your own index and/or embedder. The rest of the pipeline is
    /// unchanged — this is the swap point.
    pub fn open_with(
        path: &Path,
        cfg: EngineConfig,
        index: Arc<dyn VectorIndex>,
        embed: Arc<dyn Embedder>,
    ) -> Result<Self> {
        let truth = Arc::new(TruthStore::open(path)?);
        let rx = truth.subscribe();
        let mat = if cfg.materializer_workers > 1 {
            VectorMaterializer::spawn_pooled(
                rx,
                index.clone(),
                embed.clone(),
                cfg.materializer_workers,
            )
        } else {
            VectorMaterializer::spawn(rx, index.clone(), embed.clone())
        };
        Ok(Self {
            truth,
            index,
            embed,
            mat,
            cfg,
        })
    }

    /// Write a `[header][vector]` grain — the default raw-vector layout.
    pub fn put_vector(
        &self,
        sid: Sid,
        pred: PredId,
        header: &[u8],
        vector: &[f32],
        meta: WriteMeta,
    ) -> Result<Hlc> {
        debug_assert_eq!(header.len(), self.cfg.header_len, "header length mismatch");
        debug_assert_eq!(vector.len(), self.cfg.dim, "vector dim mismatch");
        let value = encode_value_with_header(header, vector);
        self.truth
            .put(sid, pred, value, meta.c, meta.t_valid, meta.idem_key)
    }

    /// Write an arbitrary value; the configured embedder turns it into `ψ`.
    /// Use this with a non-raw embedder (e.g. a text/model embedder).
    pub fn put_raw(&self, sid: Sid, pred: PredId, value: Vec<u8>, meta: WriteMeta) -> Result<Hlc> {
        self.truth
            .put(sid, pred, value, meta.c, meta.t_valid, meta.idem_key)
    }

    /// Tombstone a grain (removed from the index on materialization).
    pub fn delete(&self, sid: Sid, pred: PredId, meta: WriteMeta) -> Result<Hlc> {
        self.truth
            .delete(sid, pred, meta.c, meta.t_valid, meta.idem_key)
    }

    /// Run a mixed `near ⋈ select` query.
    pub fn query<P>(&self, q: &MixedQuery<'_>, predicate: P) -> Result<Vec<Ranked>>
    where
        P: Fn(&Grain) -> bool,
    {
        near_join_select(self.truth.as_ref(), self.index.as_ref(), q, predicate)
    }

    /// Block until the vector index reflects commit `seq` (a bounded-staleness
    /// barrier; returns `false` on timeout).
    pub fn sync(&self, seq: Hlc, timeout: Duration) -> bool {
        self.mat.wait_for(seq.0, timeout)
    }

    /// Direct access to the underlying truth store (strong/snapshot reads, etc.).
    pub fn truth(&self) -> &TruthStore {
        &self.truth
    }

    /// Number of vectors currently in the index.
    pub fn index_len(&self) -> usize {
        self.index.len()
    }
}
