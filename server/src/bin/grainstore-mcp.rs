//! grainstore-mcp — MCP server for agents (stdio JSON-RPC).
//!
//! The agent-facing transport for the same store the `grainstored` daemon serves.
//! Tools: grain_load, grain_query, grain_stats. See the deployment README.

use std::io::{self, BufRead, Write};

use gsserver::{Config, GrainRec, Store};
use serde_json::{json, Value};

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn ok(id: Option<Value>, result: Value) -> Value {
    json!({"jsonrpc":"2.0","id": id.unwrap_or(Value::Null), "result": result})
}
fn rpc_err(id: Option<Value>, code: i64, msg: &str) -> Value {
    json!({"jsonrpc":"2.0","id": id.unwrap_or(Value::Null), "error": {"code": code, "message": msg}})
}
fn tool_text(s: String) -> Value {
    json!({"content": [{"type":"text","text": s}]})
}
fn tool_err(s: &str) -> Value {
    json!({"content": [{"type":"text","text": s}], "isError": true})
}

fn tool_defs() -> Value {
    json!([
        {"name":"grain_load","description":"Durably store grains (facts) and index them for semantic search. Each grain has sid, optional category (filter), optional amount (numeric, range-filterable), and text.",
         "inputSchema":{"type":"object","properties":{"grains":{"type":"array","items":{"type":"object","properties":{"sid":{"type":"integer"},"category":{"type":"integer"},"amount":{"type":"integer"},"text":{"type":"string"}},"required":["sid","text"]}}},"required":["grains"]}},
        {"name":"grain_query","description":"Find grains most similar to a query text, optionally filtered by category and an amount range. Returns sid, category, amount, text, distance.",
         "inputSchema":{"type":"object","properties":{"text":{"type":"string"},"category":{"type":"integer"},"min_amount":{"type":"integer"},"max_amount":{"type":"integer"},"k":{"type":"integer"}},"required":["text"]}},
        {"name":"grain_stats","description":"Report how many grains are indexed.","inputSchema":{"type":"object","properties":{}}}
    ])
}

fn tool_load(store: &Store, args: &Value) -> Value {
    let arr = match args.get("grains").and_then(|g| g.as_array()) {
        Some(a) => a,
        None => return tool_err("missing 'grains' array"),
    };
    let mut recs = Vec::with_capacity(arr.len());
    for g in arr {
        let sid = g.get("sid").and_then(|s| s.as_u64());
        let text = g.get("text").and_then(|t| t.as_str());
        let (sid, text) = match (sid, text) {
            (Some(s), Some(t)) => (s, t.to_string()),
            _ => return tool_err("each grain needs 'sid' and 'text'"),
        };
        recs.push(GrainRec {
            sid,
            category: g.get("category").and_then(|c| c.as_u64()).unwrap_or(0) as u8,
            amount: g.get("amount").and_then(|a| a.as_i64()).unwrap_or(0),
            text,
        });
    }
    match store.load(&recs) {
        Ok(n) => tool_text(format!("Loaded {n} grains. Index now holds {} vectors.", store.indexed())),
        Err(e) => tool_err(&format!("load failed: {e}")),
    }
}

fn tool_query(store: &Store, args: &Value) -> Value {
    let text = match args.get("text").and_then(|t| t.as_str()) {
        Some(t) => t,
        None => return tool_err("missing 'text'"),
    };
    let category = args.get("category").and_then(|c| c.as_u64()).map(|c| c as u8);
    let min = args.get("min_amount").and_then(|x| x.as_i64());
    let max = args.get("max_amount").and_then(|x| x.as_i64());
    let k = args.get("k").and_then(|x| x.as_u64()).unwrap_or(5) as usize;
    match store.query(text, category, min, max, k) {
        Ok(hits) => {
            let arr: Vec<Value> = hits
                .iter()
                .map(|h| json!({"sid":h.sid,"category":h.category,"amount":h.amount,"text":h.text,"distance":h.distance}))
                .collect();
            tool_text(serde_json::to_string_pretty(&json!({"matches":arr})).unwrap_or_default())
        }
        Err(e) => tool_err(&format!("query failed: {e}")),
    }
}

fn main() {
    let dim = env_usize("GS_DIM", 64);
    let shards = env_usize("GS_SHARDS", 8);
    let workers = env_usize("GS_WORKERS", std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4));
    let data_dir = std::env::var("GS_DATA")
        .unwrap_or_else(|_| format!("{}/.grainstore", std::env::var("HOME").unwrap_or_else(|_| "/tmp".into())));

    let (store, restored) = match Store::open(&Config { data_dir: data_dir.clone(), dim, shards, workers }) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("grainstore-mcp: open failed: {e}");
            std::process::exit(1);
        }
    };
    eprintln!("grainstore-mcp: data={data_dir} dim={dim}; restored {restored} grains");

    let stdin = io::stdin();
    let mut stdout = io::stdout();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        if line.trim().is_empty() {
            continue;
        }
        let req: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("parse error: {e}");
                continue;
            }
        };
        let id = req.get("id").cloned();
        let method = req.get("method").and_then(|m| m.as_str()).unwrap_or("");
        let response = match method {
            "initialize" => Some(ok(id, json!({"protocolVersion":"2024-11-05","capabilities":{"tools":{}},"serverInfo":{"name":"grainstore","version":"0.1.0"}}))),
            "tools/list" => Some(ok(id, json!({"tools": tool_defs()}))),
            "tools/call" => {
                let p = req.get("params").cloned().unwrap_or(json!({}));
                let name = p.get("name").and_then(|n| n.as_str()).unwrap_or("");
                let args = p.get("arguments").cloned().unwrap_or(json!({}));
                let r = match name {
                    "grain_load" => tool_load(&store, &args),
                    "grain_query" => tool_query(&store, &args),
                    "grain_stats" => tool_text(json!({"indexed":store.indexed(),"dim":store.dim}).to_string()),
                    other => tool_err(&format!("unknown tool: {other}")),
                };
                Some(ok(id, r))
            }
            "ping" => Some(ok(id, json!({}))),
            "notifications/initialized" | "notifications/cancelled" => None,
            _ => id.map(|i| rpc_err(Some(i), -32601, "method not found")),
        };
        if let Some(resp) = response {
            let _ = writeln!(stdout, "{}", serde_json::to_string(&resp).unwrap_or_default());
            let _ = stdout.flush();
        }
    }
}
