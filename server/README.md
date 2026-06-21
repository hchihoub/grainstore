# GrainStore — standalone deployment

Run GrainStore as a standalone server you deploy and connect clients to, like
Postgres or a modern vector database (Qdrant/Weaviate). Three binaries:

| Binary | Role | Analogy |
|--------|------|---------|
| `grainstored` | the **server daemon** — HTTP/JSON on a TCP port, concurrent clients, durable WAL | `postgres` |
| `grainstore` | the **CLI client** — connects to the daemon, one-shot or REPL | `psql` |
| `grainstore-mcp` | **MCP server** for AI agents (stdio) | — |

Data is durable (group-committed WAL under the data dir); the vector index is a
derived materialization rebuilt from the truth on startup, so it survives
restarts.

## Quick start

```sh
cargo build --release

# start the server (foreground)
./target/release/grainstored --port 7700 --data ~/.grainstore --dim 64

# in another shell — use the CLI client
./target/release/grainstore stats
./target/release/grainstore load sales.ndjson
./target/release/grainstore query "enterprise SaaS renewal" --min 100000 -k 5
./target/release/grainstore                 # interactive REPL
```

## Install as a service (auto-start, restart-on-crash)

```sh
./deploy/install.sh
```

Installs the three binaries to `~/.local/bin`, creates `~/.grainstore`, and
offers to register a background service:

- **macOS** — a launchd agent (`~/Library/LaunchAgents/com.grainstore.grainstored.plist`):
  ```sh
  launchctl load   ~/Library/LaunchAgents/com.grainstore.grainstored.plist   # start
  launchctl unload ~/Library/LaunchAgents/com.grainstore.grainstored.plist   # stop
  ```
- **Linux** — systemd (`deploy/grainstore.service`):
  ```sh
  sudo cp deploy/grainstore.service /etc/systemd/system/
  sudo systemctl enable --now grainstore
  ```

Then it runs in the background and clients connect over the network.

## HTTP API

| Method · path | Body | Returns |
|---|---|---|
| `GET /health` | — | `{"status":"ok","indexed":N}` |
| `GET /stats` | — | `{"indexed":N,"dim":D}` |
| `POST /grains` | JSON array, `{grains:[…]}`, or NDJSON of `{sid,category?,amount?,text}` | `{"loaded":N,"indexed":M}` |
| `POST /query` | `{text, category?, min_amount?, max_amount?, k?}` | `{"matches":[{sid,category,amount,text,distance}]}` |

```sh
curl localhost:7700/query -H 'content-type: application/json' \
  -d '{"text":"churned to a cheaper competitor","k":2}'
```

A grain combines **semantic text + a category + a numeric amount**; queries filter
by category and amount range and rank by meaning — one `near ⋈ select`.

## CLI reference

```
grainstore [--server URL] stats
grainstore [--server URL] load <file.ndjson|file.json>
grainstore [--server URL] query "<text>" [--category C] [--min N] [--max N] [-k K]
grainstore [--server URL]                       # REPL
```
Default server `http://localhost:7700` (override with `--server` or `GS_SERVER`).

## Agents (MCP)

For AI agents, register `grainstore-mcp` (talks to its own local store):

```json
{ "mcpServers": { "grainstore": {
    "command": "/Users/you/.local/bin/grainstore-mcp",
    "env": { "GS_DATA": "/Users/you/.grainstore", "GS_DIM": "64" }
} } }
```

## Config

| Flag | Env | Default | Meaning |
|------|-----|---------|---------|
| `--port` | `GS_PORT` | 7700 | listen port |
| `--data` | `GS_DATA` | `~/.grainstore` | data directory (durable WAL) |
| `--dim` | `GS_DIM` | 64 | embedding dimension |
| `--shards` | `GS_SHARDS` | 8 | sharded-HNSW shards |
| `--workers` | `GS_WORKERS` | #cores | materializer workers |

## Production notes

- The text embedder is a deterministic hashing embedder (no model needed). Swap a
  real model-server `Embedder` for production semantics — same trait seam.
- The server is single-store and unauthenticated; front it with the
  `GovernedEngine` (P4: identity, ACL, masking, token budgets, audit) and TLS for
  multi-tenant / internet-facing deployment.
- Every write is fsync-durable before the response returns, so an abrupt stop
  (Ctrl-C, crash, power loss) loses nothing committed.
