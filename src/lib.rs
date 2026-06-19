//! GrainStore P0 — the durable truth store.
//!
//! This crate implements the transactional core of the GrainStore architecture:
//! a group-committed write-ahead log (the source of truth) and an MVCC point
//! store layered over an [`OrderedKv`](kv::OrderedKv) abstraction.
//!
//! # Design invariants
//!
//! 1. **The WAL is the only durable artifact.** The key-value store is treated as
//!    a *rebuildable materialization*: on crash it is discarded and replayed from
//!    the WAL. This makes "truth is sacred, indexes rebuild" a literal property.
//! 2. **MVCC is encoded in the key.** Versions of a `(sid, pred)` are ordered
//!    newest-first by storing `!t_tx` as the key suffix, so a snapshot read is a
//!    single ordered seek (see [`keys`]).
//! 3. **Idempotency is serialized in the committer.** A client-supplied
//!    `idem_key` is deduplicated by the single committer thread, so concurrent
//!    retries can never produce duplicate versions.
//!
//! The [`OrderedKv`](kv::OrderedKv) trait is the seam where a production engine
//! (RocksDB with its internal WAL disabled) replaces the in-memory test backing.

pub mod block;
pub mod catalog;
pub mod cdc;
pub mod disclosure;
pub mod embed;
pub mod engine;
pub mod error;
pub mod governance;
pub mod hlc;
pub mod keys;
pub mod kv;
pub mod materializer;
pub mod model;
pub mod planner;
pub mod query;
pub mod recovery;
pub mod truth;
pub mod vector;
pub mod wal;

pub use disclosure::{ContinuationHandle, Stage, Staged, Summary};
pub use engine::{EngineConfig, GrainEngine, WriteMeta};
pub use error::{Error, Result};
pub use governance::{
    Action, AgentId, AuditRecord, Decision, GovernedEngine, Match, Policy, RuleSet,
};
pub use materializer::VectorMaterializer;
pub use model::{Confidence, Grain, Hlc, PredId, Sid, Val};
pub use planner::{Filter, Plan, Planner};
pub use query::{near_join_select, MixedQuery, Ranked};
pub use truth::TruthStore;
pub use vector::{BruteForceIndex, Hnsw, HnswConfig, ShardedHnsw, VectorIndex};
