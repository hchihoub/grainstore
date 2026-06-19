# GrainStore — Setup & Build (P0)

The durable truth store: a group-committed write-ahead log (source of truth) and
an MVCC point store over an ordered-KV abstraction. This P0 crate is **dependency
-free** so it builds and tests offline with no RocksDB/FAISS C++ toolchain.

## Prerequisites

| Tool | Version used | Install (macOS) |
|------|--------------|-----------------|
| Rust + Cargo | 1.81+ (tested on 1.95) | `brew install rust` or `rustup` |
| A C compiler | clang (Xcode CLT) | `xcode-select --install` |

Verify:

```sh
rustc --version
cargo --version
```

No other system libraries are required for P0.

## Build & test

```sh
cd ~/grainstore
cargo build                 # compile the library
cargo test                  # unit + integration property tests
cargo test -- --nocapture   # with test output
cargo clippy --all-targets  # lints (if clippy installed)
cargo fmt --check           # formatting (if rustfmt installed)
```

### What the tests cover

| Suite | Property |
|-------|----------|
| `tests/mvcc_props.rs` | Every snapshot read returns the newest version with `t_tx <= snapshot`, never a tombstoned/future one (model-checked over 150 random programs). |
| `tests/recovery_props.rs` | After a crash — including a WAL tail truncated mid-frame — recovery yields exactly the model truncated at the last intact frame; idempotency survives restart; double recovery is a fixpoint. |
| `tests/concurrency.rs` | Under 12 concurrent writers: one stored version per distinct `idem_key`, distinct commit timestamps, no timestamp shared by two keys; HLC uniqueness under threads. |
| `tests/vector_recall.rs` | **(P1)** HNSW recall ≥ 0.90 vs the exact brute-force oracle; brute force is exact; HNSW soft-deletes are filtered from results. |
| `tests/p1_pipeline.rs` | **(P1)** End-to-end truth → CDC → materialize → `near ⋈ select`: filtered mixed-query recall ≥ 0.85 vs an exact filtered-kNN oracle (predicate independent of geometry); tombstones remove from the index. |
| in-crate unit tests | key ordering, `seek_ge`, frame round-trip, torn-tail and CRC rejection, HLC monotonicity, embed round-trip. |

### Benchmarks

```sh
cargo run --release --example bench       # P0: write throughput, read latency, recovery
cargo run --release --example bench_p1    # P1: mixed near⋈filter latency + recall vs over_fetch
```

## Architecture of this slice

```
TruthStore (truth.rs)
  ├─ Wal (wal.rs) ── group-commit committer thread ── WAL file (durable, fsync)
  │     • idempotency deduped here (single thread → race-free)
  ├─ OrderedKv (kv.rs) ── MemKv (volatile; rebuilt from WAL on crash)
  │     • MVCC encoded in the key: [sid][pred][!t_tx]  (keys.rs)
  ├─ HlcOracle (hlc.rs) ── monotonic, unique commit timestamps
  └─ recover() (recovery.rs) ── replay WAL → KV + idem map on open
```

**Durability ordering** (per group): append frames → **one** `fsync` → apply to
KV → ack → **publish CDC**. A crash before the KV apply is safe: recovery
replays the already-durable frames (replay is idempotent — keys are
`(sid, pred, seq)`).

### P1 — the vector plane (dual address)

```
TruthStore ──CDC──▶ VectorMaterializer ──▶ VectorIndex (HNSW)
   (truth)   (cdc.rs)   (materializer.rs)      (vector/)
                          • Embedder turns value → ψ (embed.rs)
                          • watermark tracks bounded staleness
near_join_select (query.rs): ANN candidates → join truth by sid → predicate
   → rank by distance → top-k.  Post-filter over-fetch = k·over_fetch.
```

- **Truth is sid-keyed and strongly consistent; the vector index is a derived,
  CDC-fed, bounded-staleness materialization** — rebuildable from truth, never
  authoritative. Losing the index is recoverable; losing the WAL is not.
- The mixed query is **one path**: `near` (vector) ⋈ `get` (truth) + predicate.
- `VectorIndex` has two backends: `BruteForceIndex` (exact oracle) and `Hnsw`
  (incremental, validated to ≥0.99 recall in practice). A FAISS/DiskANN backend
  slots in behind the same trait for scale.
- `Embedder` is the seam for a production model server (ONNX / remote gRPC);
  P1 ships `RawVectorEmbedder` (value bytes carry the vector).

## Production swap-ins (later phases)

- **Storage backend**: implement `kv::OrderedKv` over RocksDB (internal WAL
  *disabled*; our WAL stays the source of truth). No change to `truth.rs`.
- **Self-describing SSTable blocks** (P3) replace RocksDB; a custom lock-free
  skiplist memtable enters behind the same trait.
- **Vector plane** (P1): a CDC consumer tails commits and feeds an ANN index.

## Optional: stronger testing tools

The property tests use a self-contained seeded PRNG so they run with zero deps.
For the production test matrix, add (in `[dev-dependencies]`):

```toml
proptest = "1"     # shrinking failing cases to minimal reproductions
loom     = "0.7"   # exhaustive interleaving check of the HLC CAS + committer
```

Run loom models under its cfg:

```sh
RUSTFLAGS="--cfg loom" cargo test --test loom_hlc
```

Keep loom models tiny (2 threads, 2 ops) — the interleaving space is factorial.

## Troubleshooting

- **Linker / `cc` errors**: run `xcode-select --install`.
- **Stale temp dirs**: tests write under `$TMPDIR/grainstore_*` and clean up on
  drop; safe to delete manually if a run is killed.
