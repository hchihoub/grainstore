//! grainstored — the standalone GrainStore server daemon.
//!
//! Listens on a TCP port and serves a JSON/HTTP API (like Qdrant/Weaviate/
//! Elasticsearch). Concurrent clients; durable WAL; the vector index is rebuilt
//! from the truth on startup.
//!
//! Endpoints:
//!   GET  /health                 → {"status":"ok","indexed":N}
//!   GET  /stats                  → {"indexed":N,"dim":D}
//!   POST /grains  (JSON array or NDJSON of {sid,category?,amount?,text})
//!   POST /query   ({text,category?,min_amount?,max_amount?,k?})
//!
//! Usage: grainstored [--port 7700] [--data DIR] [--dim 64] [--shards 8] [--workers N]
//! Env fallbacks: GS_PORT, GS_DATA, GS_DIM, GS_SHARDS, GS_WORKERS.

use std::sync::Arc;

use gsserver::{Config, GrainRec, Store};
use serde_json::{json, Value};
use tiny_http::{Header, Method, Request, Response, Server};

fn arg_or_env(args: &[String], flag: &str, env: &str, default: &str) -> String {
    if let Some(i) = args.iter().position(|a| a == flag) {
        if let Some(v) = args.get(i + 1) {
            return v.clone();
        }
    }
    std::env::var(env).unwrap_or_else(|_| default.to_string())
}

fn json_resp(code: u16, body: String) -> Response<std::io::Cursor<Vec<u8>>> {
    let h = Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap();
    Response::from_string(body).with_status_code(code).with_header(h)
}

fn grain_from_value(v: &Value) -> Option<GrainRec> {
    Some(GrainRec {
        sid: v.get("sid")?.as_u64()?,
        category: v.get("category").and_then(|c| c.as_u64()).unwrap_or(0) as u8,
        amount: v.get("amount").and_then(|a| a.as_i64()).unwrap_or(0),
        text: v.get("text")?.as_str()?.to_string(),
    })
}

fn parse_grains(body: &str) -> Result<Vec<GrainRec>, String> {
    // First try the whole body as one JSON value: an array, a {grains:[...]}
    // wrapper, or a single grain object. Multi-line NDJSON fails this parse and
    // falls through to line-by-line below.
    if let Ok(v) = serde_json::from_str::<Value>(body) {
        let mut out = Vec::new();
        match &v {
            Value::Array(a) => {
                for item in a {
                    out.push(grain_from_value(item).ok_or("each grain needs sid and text")?);
                }
            }
            Value::Object(o) => {
                if let Some(arr) = o.get("grains").and_then(|g| g.as_array()) {
                    for item in arr {
                        out.push(grain_from_value(item).ok_or("each grain needs sid and text")?);
                    }
                } else {
                    out.push(grain_from_value(&v).ok_or("each grain needs sid and text")?);
                }
            }
            _ => return Err("expected a JSON object/array or NDJSON".into()),
        }
        return Ok(out);
    }
    // NDJSON: one grain per line.
    let mut out = Vec::new();
    for (i, line) in body.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(line).map_err(|e| format!("line {i}: {e}"))?;
        out.push(grain_from_value(&v).ok_or("each grain needs sid and text")?);
    }
    if out.is_empty() {
        return Err("no grains parsed".into());
    }
    Ok(out)
}

fn handle(rq: &mut Request, store: &Store) -> Response<std::io::Cursor<Vec<u8>>> {
    let path = rq.url().split('?').next().unwrap_or("/").to_string();
    let method = rq.method().clone();
    match (&method, path.as_str()) {
        (Method::Get, "/health") => {
            json_resp(200, json!({"status":"ok","indexed":store.indexed()}).to_string())
        }
        (Method::Get, "/stats") => {
            json_resp(200, json!({"indexed":store.indexed(),"dim":store.dim}).to_string())
        }
        (Method::Post, "/grains") => {
            let mut body = String::new();
            if rq.as_reader().read_to_string(&mut body).is_err() {
                return json_resp(400, json!({"error":"bad body"}).to_string());
            }
            match parse_grains(&body) {
                Ok(recs) => match store.load(&recs) {
                    Ok(n) => json_resp(
                        200,
                        json!({"loaded":n,"indexed":store.indexed()}).to_string(),
                    ),
                    Err(e) => json_resp(500, json!({"error":e}).to_string()),
                },
                Err(e) => json_resp(400, json!({"error":e}).to_string()),
            }
        }
        (Method::Post, "/query") => {
            let mut body = String::new();
            if rq.as_reader().read_to_string(&mut body).is_err() {
                return json_resp(400, json!({"error":"bad body"}).to_string());
            }
            let v: Value = match serde_json::from_str(&body) {
                Ok(v) => v,
                Err(e) => return json_resp(400, json!({"error":e.to_string()}).to_string()),
            };
            let text = match v.get("text").and_then(|t| t.as_str()) {
                Some(t) => t,
                None => return json_resp(400, json!({"error":"missing 'text'"}).to_string()),
            };
            let category = v.get("category").and_then(|c| c.as_u64()).map(|c| c as u8);
            let min_amount = v.get("min_amount").and_then(|x| x.as_i64());
            let max_amount = v.get("max_amount").and_then(|x| x.as_i64());
            let k = v.get("k").and_then(|x| x.as_u64()).unwrap_or(5) as usize;
            match store.query(text, category, min_amount, max_amount, k) {
                Ok(hits) => {
                    let arr: Vec<Value> = hits
                        .iter()
                        .map(|h| json!({"sid":h.sid,"category":h.category,"amount":h.amount,"text":h.text,"distance":h.distance}))
                        .collect();
                    json_resp(200, json!({"matches":arr}).to_string())
                }
                Err(e) => json_resp(500, json!({"error":e}).to_string()),
            }
        }
        _ => json_resp(404, json!({"error":"not found"}).to_string()),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let port: u16 = arg_or_env(&args, "--port", "GS_PORT", "7700").parse().unwrap_or(7700);
    let dim: usize = arg_or_env(&args, "--dim", "GS_DIM", "64").parse().unwrap_or(64);
    let shards: usize = arg_or_env(&args, "--shards", "GS_SHARDS", "8").parse().unwrap_or(8);
    let default_workers = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).to_string();
    let workers: usize =
        arg_or_env(&args, "--workers", "GS_WORKERS", &default_workers).parse().unwrap_or(4);
    let default_data = format!("{}/.grainstore", std::env::var("HOME").unwrap_or_else(|_| "/tmp".into()));
    let data_dir = arg_or_env(&args, "--data", "GS_DATA", &default_data);

    let cfg = Config { data_dir: data_dir.clone(), dim, shards, workers };
    let (store, restored) = match Store::open(&cfg) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("grainstored: failed to open store: {e}");
            std::process::exit(1);
        }
    };
    let store = Arc::new(store);

    let addr = format!("0.0.0.0:{port}");
    let server = match Server::http(&addr) {
        Ok(s) => Arc::new(s),
        Err(e) => {
            eprintln!("grainstored: failed to bind {addr}: {e}");
            std::process::exit(1);
        }
    };

    println!("grainstored 0.1.0  —  GrainStore standalone server");
    println!("  listening : http://{addr}");
    println!("  data dir  : {data_dir}");
    println!("  config    : dim={dim} shards={shards} workers={workers}");
    println!("  restored  : {restored} grains from truth");
    println!("  ready. (Ctrl-C to stop; every write is already durable.)");

    let n_threads = workers.clamp(2, 16);
    let mut handles = Vec::new();
    for _ in 0..n_threads {
        let (server, store) = (server.clone(), store.clone());
        handles.push(std::thread::spawn(move || {
            while let Ok(mut rq) = server.recv() {
                let resp = handle(&mut rq, &store);
                let _ = rq.respond(resp);
            }
        }));
    }
    for h in handles {
        let _ = h.join();
    }
}
