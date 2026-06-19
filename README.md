# GrainStore

A data platform designed for AI agents, not humans — a neuro-symbolic store whose
unit of value is *a correct decision* and whose unit of cost is *tokens spent
reaching it*. The atom is a **Grain**: a fact that lives simultaneously as an exact
symbol (for joins, constraints, transactions) and as a point in continuous
meaning-space (for similarity and analogy).

This repository is the working engine. It is built phase by phase, each gated on
passing tests and benchmarks.

## What's here

- **P0 — Truth store.** A group-committed write-ahead log (the only durable
  artifact) with MVCC snapshot reads encoded directly in the key. The key-value
  store is a *rebuildable materialization*: on crash it is discarded and replayed
  from the WAL. Idempotency is deduplicated inside the single committer, so
  concurrent retries can never double-write.
- **P1 — Vector plane.** Committed writes flow over a CDC stream to a
  materializer that embeds each value and maintains an ANN index. The mixed
  `near ⋈ select` query generates candidates from the vector plane, joins them
  back to the truth store by `sid`, applies an exact predicate, and ranks — all in
  one path.
- **P2 — Token-cost planner.** Given a target recall, the planner estimates a
  filter's selectivity (from a catalog kept on the write path) and picks
  `over_fetch`/`ef` to hit it at minimum cost — rare filters get more candidates,
  common ones stay cheap. **Staged disclosure** then returns a compact summary
  first (Stage 0) with a continuation handle; `drill` returns the full result from
  pinned state with no re-search.

Everything is behind traits, so implementations swap with no pipeline changes:

| Seam | Default | Production swap |
|------|---------|-----------------|
| `OrderedKv` | in-memory map | RocksDB (internal WAL off; our WAL is truth) |
| `VectorIndex` | `Hnsw` / `ShardedHnsw` | FAISS / DiskANN |
| `Embedder` | `RawVectorEmbedder` | ONNX / remote model server |

The `GrainEngine` facade wires it together: `GrainEngine::open(path, cfg)` for the
batteries-included setup, `open_with(...)` to bring your own index/embedder.

## Status

- Durable MVCC truth store with crash recovery (model-checked + crash-injection tests)
- CDC + worker-pool materializer (parallel index build, contiguous-prefix watermark)
- Three index backends behind one trait: exact brute-force, HNSW, sharded HNSW
- Recall-targeting planner + selectivity catalog; staged disclosure with continuations
- 27 tests green · clippy + rustfmt clean · **zero runtime dependencies** in the core

## Benchmarks (1M vectors, dim 32, selectivity 0.10, k=10)

Head-to-head vs PostgreSQL 18 + pgvector on identical data and queries:

| system | build | query p50 | recall |
|---|---|---|---|
| PostgreSQL 18 + pgvector | 76s | 4.89ms | 0.786 |
| **GrainStore (live, parallel pipeline)** | **82s** | **1.36ms** | **0.991** |

GrainStore matches the build (with *stronger* per-record durability) while winning
query latency ~3.6× at higher recall. With the parallel materializer, the index
build is fully hidden behind the WAL write cost — it is no longer on the critical
path. Point reads are ~208ns p50 (MVCC is a single ordered seek).

See [`SETUP.md`](SETUP.md) for prerequisites, the test matrix, and how to run each
benchmark.

## Build & test

```sh
cargo test                                     # 27 tests, no system deps
cargo clippy --all-targets
cargo run --release --example bench            # P0: write/read/recovery
cargo run --release --example bench_p1         # P1: mixed near⋈filter vs over_fetch
cargo run --release --example bench_build      # parallel sharded build
cargo run --release --example bench_live       # full live pipeline
cargo run --release --example bench_planner     # P2: planner vs fixed over_fetch
cargo run --release --example bench_disclosure  # P2: staged disclosure (tokens + reuse)
```

## Honest scope

This is a research engine, built to validate an architecture rather than to win
production benchmarks today. The HNSW is a from-scratch implementation (correct,
validated to ≥0.99 recall vs an exact oracle) — at very high dimension or scale a
FAISS/DiskANN backend behind the same trait is the right call. The in-memory
`OrderedKv` is the P0 backing; RocksDB is the production swap. Recall numbers are
measured against an exact in-process oracle; the pgvector comparison includes its
client/server round trip (intrinsic to a database server) versus GrainStore's
embedded path.

## License

Apache-2.0
