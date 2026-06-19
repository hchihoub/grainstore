//! Vector materializer: tails the CDC stream, embeds each committed value, and
//! maintains the vector index. Runs on its own thread. Exposes a watermark (the
//! highest commit `seq` reflected in the index) so readers and tests can reason
//! about bounded staleness.

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use crate::cdc::CdcRecord;
use crate::embed::Embedder;
use crate::model::Sid;
use crate::vector::VectorIndex;

/// Tracks the highest *contiguous* commit `seq` that has been applied, so the
/// watermark stays correct even when workers complete out of order. Published
/// CDC seqs are contiguous (the committer ticks the clock once per published
/// record), starting at 1 for a fresh index.
struct Completion {
    next: u64,
    pending: BTreeSet<u64>,
}

impl Completion {
    fn new() -> Self {
        Self {
            next: 1,
            pending: BTreeSet::new(),
        }
    }
    /// Record that `seq` is applied; advance the watermark over the contiguous
    /// prefix. Conservative: the watermark never claims a seq is applied before
    /// every earlier seq is.
    fn complete(&mut self, seq: u64, watermark: &AtomicU64) {
        self.pending.insert(seq);
        while self.pending.remove(&self.next) {
            self.next += 1;
        }
        watermark.store(self.next - 1, Ordering::Release);
    }
}

/// Owns the background thread. `Drop` signals shutdown and joins; it never blocks
/// indefinitely, so it is safe regardless of whether the truth store (which owns
/// the CDC sender) is dropped before or after the materializer.
pub struct VectorMaterializer {
    watermark: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl VectorMaterializer {
    /// Spawn a materializer consuming `rx`, feeding `index` via `embed`.
    pub fn spawn<I, E>(rx: Receiver<CdcRecord>, index: Arc<I>, embed: Arc<E>) -> Self
    where
        I: VectorIndex + ?Sized + 'static,
        E: Embedder + ?Sized + 'static,
    {
        let watermark = Arc::new(AtomicU64::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let wm = watermark.clone();
        let stop_c = stop.clone();
        let handle = std::thread::Builder::new()
            .name("grainstore-materializer".into())
            .spawn(move || loop {
                match rx.recv_timeout(Duration::from_millis(50)) {
                    Ok(rec) => {
                        apply(&rec, index.as_ref(), embed.as_ref());
                        // Advance only after the index reflects this commit.
                        wm.store(rec.seq, Ordering::Release);
                    }
                    Err(RecvTimeoutError::Timeout) => {
                        if stop_c.load(Ordering::Acquire) {
                            break; // explicit shutdown while idle
                        }
                    }
                    Err(RecvTimeoutError::Disconnected) => break, // truth dropped
                }
            })
            .expect("spawn materializer");
        Self {
            watermark,
            stop,
            handle: Some(handle),
        }
    }

    /// Spawn a **worker-pool** materializer: a dispatcher reads the CDC stream and
    /// round-robins records to `workers` threads that embed + insert concurrently.
    /// With a sharded index (independently-locked shards) this parallelizes the
    /// build; with a single-locked index it stays correct but serial. The
    /// watermark advances over the contiguous applied prefix, so bounded-staleness
    /// reads remain correct under out-of-order completion. Assumes a fresh index
    /// (CDC seqs starting at 1).
    pub fn spawn_pooled<I, E>(
        rx: Receiver<CdcRecord>,
        index: Arc<I>,
        embed: Arc<E>,
        workers: usize,
    ) -> Self
    where
        I: VectorIndex + ?Sized + 'static,
        E: Embedder + ?Sized + 'static,
    {
        let workers = workers.max(1);
        let watermark = Arc::new(AtomicU64::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let completion = Arc::new(Mutex::new(Completion::new()));

        // One channel per worker; the dispatcher round-robins to them.
        let mut worker_txs = Vec::with_capacity(workers);
        let mut worker_handles = Vec::with_capacity(workers);
        for w in 0..workers {
            let (jtx, jrx) = channel::<CdcRecord>();
            worker_txs.push(jtx);
            let (index, embed, wm, comp) = (
                index.clone(),
                embed.clone(),
                watermark.clone(),
                completion.clone(),
            );
            let h = std::thread::Builder::new()
                .name(format!("grainstore-materializer-w{w}"))
                .spawn(move || {
                    while let Ok(rec) = jrx.recv() {
                        let seq = rec.seq;
                        apply(&rec, index.as_ref(), embed.as_ref());
                        comp.lock().expect("completion lock").complete(seq, &wm);
                    }
                })
                .expect("spawn worker");
            worker_handles.push(h);
        }

        let stop_c = stop.clone();
        let dispatcher = std::thread::Builder::new()
            .name("grainstore-materializer-dispatch".into())
            .spawn(move || {
                let mut i = 0usize;
                loop {
                    match rx.recv_timeout(Duration::from_millis(50)) {
                        Ok(rec) => {
                            // Drop on a dead worker channel is harmless (shutdown).
                            let _ = worker_txs[i % workers].send(rec);
                            i += 1;
                        }
                        Err(RecvTimeoutError::Timeout) => {
                            if stop_c.load(Ordering::Acquire) {
                                break;
                            }
                        }
                        Err(RecvTimeoutError::Disconnected) => break,
                    }
                }
                // Close worker queues, then drain + join them.
                drop(worker_txs);
                for h in worker_handles {
                    let _ = h.join();
                }
            })
            .expect("spawn dispatcher");

        Self {
            watermark,
            stop,
            handle: Some(dispatcher),
        }
    }

    /// Highest commit `seq` reflected in the index.
    pub fn watermark(&self) -> u64 {
        self.watermark.load(Ordering::Acquire)
    }

    /// Block (spin with short sleeps) until the index reflects `seq`. Returns
    /// `false` if the deadline passes first.
    pub fn wait_for(&self, seq: u64, timeout: std::time::Duration) -> bool {
        let start = std::time::Instant::now();
        while self.watermark() < seq {
            if start.elapsed() > timeout {
                return false;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        true
    }
}

impl Drop for VectorMaterializer {
    fn drop(&mut self) {
        // Signal shutdown; the loop checks this on its next idle tick (≤ 50ms),
        // so the join cannot deadlock even if the CDC sender is still open.
        self.stop.store(true, Ordering::Release);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn apply<I: VectorIndex + ?Sized, E: Embedder + ?Sized>(rec: &CdcRecord, index: &I, embed: &E) {
    let sid = Sid(rec.sid);
    if rec.is_delete() {
        index.remove(sid);
        return;
    }
    match embed.embed(&rec.val) {
        Some(v) => index.insert(sid, &v),
        None => index.remove(sid), // unembeddable value → ensure not stale
    }
}
