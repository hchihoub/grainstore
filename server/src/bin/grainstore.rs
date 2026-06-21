//! grainstore — CLI client for a grainstored server (the `psql` equivalent).
//!
//! Usage:
//!   grainstore [--server URL] stats
//!   grainstore [--server URL] load <file.ndjson|file.json>
//!   grainstore [--server URL] query "<text>" [--category C] [--min N] [--max N] [-k K]
//!   grainstore [--server URL]                 # interactive REPL
//!
//! Default server: http://localhost:7700  (override with --server or GS_SERVER).

use std::io::{BufRead, Write};

use serde_json::{json, Value};

fn server_url(args: &mut Vec<String>) -> String {
    if let Some(i) = args.iter().position(|a| a == "--server") {
        if i + 1 < args.len() {
            let url = args[i + 1].clone();
            args.drain(i..=i + 1);
            return url;
        }
    }
    std::env::var("GS_SERVER").unwrap_or_else(|_| "http://localhost:7700".into())
}

fn get(url: &str) -> Result<Value, String> {
    let resp = ureq::get(url).call().map_err(|e| e.to_string())?;
    let s = resp.into_string().map_err(|e| e.to_string())?;
    serde_json::from_str(&s).map_err(|e| e.to_string())
}

fn post(url: &str, body: &str) -> Result<Value, String> {
    let resp = ureq::post(url)
        .set("Content-Type", "application/json")
        .send_string(body)
        .map_err(|e| e.to_string())?;
    let s = resp.into_string().map_err(|e| e.to_string())?;
    serde_json::from_str(&s).map_err(|e| e.to_string())
}

fn print_matches(v: &Value) {
    match v.get("matches").and_then(|m| m.as_array()) {
        Some(arr) if !arr.is_empty() => {
            println!("  {:>6}  {:>4}  {:>12}  {:>6}   text", "sid", "cat", "amount", "dist");
            for m in arr {
                println!(
                    "  {:>6}  {:>4}  {:>12}  {:>6.2}   {}",
                    m["sid"].as_u64().unwrap_or(0),
                    m["category"].as_u64().unwrap_or(0),
                    m["amount"].as_i64().unwrap_or(0),
                    m["distance"].as_f64().unwrap_or(0.0),
                    m["text"].as_str().unwrap_or("")
                );
            }
        }
        _ => {
            if let Some(e) = v.get("error") {
                println!("  error: {e}");
            } else {
                println!("  (no matches)");
            }
        }
    }
}

fn do_query(server: &str, text: &str, flags: &[String]) {
    let mut body = json!({"text": text, "k": 5});
    let mut i = 0;
    while i < flags.len() {
        match flags[i].as_str() {
            "--category" | "-c" => {
                if let Some(v) = flags.get(i + 1).and_then(|s| s.parse::<u64>().ok()) {
                    body["category"] = json!(v);
                }
                i += 2;
            }
            "--min" => {
                if let Some(v) = flags.get(i + 1).and_then(|s| s.parse::<i64>().ok()) {
                    body["min_amount"] = json!(v);
                }
                i += 2;
            }
            "--max" => {
                if let Some(v) = flags.get(i + 1).and_then(|s| s.parse::<i64>().ok()) {
                    body["max_amount"] = json!(v);
                }
                i += 2;
            }
            "-k" => {
                if let Some(v) = flags.get(i + 1).and_then(|s| s.parse::<u64>().ok()) {
                    body["k"] = json!(v);
                }
                i += 2;
            }
            _ => i += 1,
        }
    }
    match post(&format!("{server}/query"), &body.to_string()) {
        Ok(v) => print_matches(&v),
        Err(e) => println!("  request failed: {e}"),
    }
}

fn do_load(server: &str, file: &str) {
    let body = match std::fs::read_to_string(file) {
        Ok(b) => b,
        Err(e) => {
            println!("  cannot read {file}: {e}");
            return;
        }
    };
    match post(&format!("{server}/grains"), &body) {
        Ok(v) => println!("  {v}"),
        Err(e) => println!("  request failed: {e}"),
    }
}

fn repl(server: &str) {
    println!("grainstore CLI — connected to {server}");
    println!("commands:  query <text> [--category C --min N --max N -k K] | load <file> | stats | \\q");
    let stdin = std::io::stdin();
    loop {
        print!("grainstore> ");
        let _ = std::io::stdout().flush();
        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap_or(0) == 0 {
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let toks: Vec<String> = line.split_whitespace().map(|s| s.to_string()).collect();
        match toks[0].as_str() {
            "\\q" | "\\quit" | "exit" | "quit" => break,
            "stats" => match get(&format!("{server}/stats")) {
                Ok(v) => println!("  {v}"),
                Err(e) => println!("  request failed: {e}"),
            },
            "load" => {
                if let Some(f) = toks.get(1) {
                    do_load(server, f);
                } else {
                    println!("  usage: load <file>");
                }
            }
            "query" => {
                // text = tokens until the first flag (starts with '-')
                let rest = &toks[1..];
                let split = rest.iter().position(|t| t.starts_with('-')).unwrap_or(rest.len());
                let text = rest[..split].join(" ");
                if text.is_empty() {
                    println!("  usage: query <text> [flags]");
                } else {
                    do_query(server, &text, &rest[split..]);
                }
            }
            _ => println!("  unknown command (try: query, load, stats, \\q)"),
        }
    }
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let server = server_url(&mut args);

    if args.is_empty() {
        repl(&server);
        return;
    }
    match args[0].as_str() {
        "stats" => match get(&format!("{server}/stats")) {
            Ok(v) => println!("{v}"),
            Err(e) => eprintln!("request failed: {e}"),
        },
        "health" => match get(&format!("{server}/health")) {
            Ok(v) => println!("{v}"),
            Err(e) => eprintln!("request failed: {e}"),
        },
        "load" => {
            if let Some(f) = args.get(1) {
                do_load(&server, f);
            } else {
                eprintln!("usage: grainstore load <file>");
            }
        }
        "query" => {
            let rest = &args[1..];
            let split = rest.iter().position(|t| t.starts_with('-')).unwrap_or(rest.len());
            let text = rest[..split].join(" ");
            if text.is_empty() {
                eprintln!("usage: grainstore query \"<text>\" [--category C --min N --max N -k K]");
            } else {
                do_query(&server, &text, &rest[split..]);
            }
        }
        "repl" => repl(&server),
        other => eprintln!("unknown command: {other} (try: stats, load, query, repl)"),
    }
}
