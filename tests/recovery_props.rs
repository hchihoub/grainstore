//! Property: after a crash (optionally truncating the WAL tail), recovery yields
//! exactly the model truncated at the last intact frame — no lost durable
//! commits, nothing past the first torn byte, and idempotency survives restart.

mod common;

use common::{last_intact_seq, truncate_tail, Lcg, Model, TempCase};
use grainstore::model::{Hlc, PredId, Sid, Val};
use grainstore::TruthStore;

#[test]
fn recover_is_consistent_after_crash() {
    for seed in 0..120u64 {
        let case = TempCase::new("recovery", seed);
        let wal = case.wal();
        let mut model = Model::new();
        let nops = 10 + seed % 50;

        // 1. Run a program, then drop the store WITHOUT clean shutdown of the KV
        //    (Drop joins the committer, so the WAL is durable and closed).
        {
            let store = TruthStore::open(&wal).expect("open");
            let mut rng = Lcg::new(seed);
            for op_i in 0..nops {
                let id = op_i as u128 + 1;
                let sid = rng.range(8);
                let pred = rng.range(3);
                let is_del = rng.range(4) == 0;
                let v = vec![rng.range(256) as u8, sid as u8, pred as u8];
                let seq = if is_del {
                    store
                        .delete(Sid(sid as u128), PredId(pred as u32), 1.0, 0, id)
                        .expect("delete")
                        .0
                } else {
                    store
                        .put(Sid(sid as u128), PredId(pred as u32), v.clone(), 1.0, 0, id)
                        .expect("put")
                        .0
                };
                model.apply(
                    sid as u128,
                    pred as u32,
                    seq,
                    if is_del { None } else { Some(v) },
                    id,
                );
            }
        }

        // 2. Injected crash: maybe lop bytes off the WAL tail.
        let tail = seed % 24;
        if tail > 0 {
            truncate_tail(&wal, tail);
        }
        let cutoff = last_intact_seq(&wal);

        // 3. Reopen → recover.
        let store = TruthStore::open(&wal).expect("reopen");
        let expected = store_truncated_expectation(&model, cutoff);

        // 4. Reads must equal the model truncated at the last intact frame.
        let mut rng = Lcg::new(seed ^ 0xABCD);
        for _ in 0..400 {
            let sid = rng.range(8) as u128;
            let pred = rng.range(3) as u32;
            let snap = rng.range(nops + 2);
            let got = store
                .get_at(Sid(sid), PredId(pred), Hlc(snap))
                .expect("get_at")
                .map(|g| match g.val {
                    Val::Bytes(b) => b,
                    Val::Tombstone => Vec::new(),
                });
            let exp = expected.get_at(sid, pred, snap);
            assert_eq!(
                got, exp,
                "seed={seed} tail={tail} cutoff={cutoff} sid={sid} pred={pred} snap={snap}"
            );
        }

        // 5. Idempotency survives recovery: re-issuing a recovered idem_key
        //    returns the original seq and creates no new version.
        if let Some(orig) = expected.idem_seq(1) {
            let before = store.total_versions();
            let s = store
                .put(Sid(0), PredId(0), vec![9, 9, 9], 1.0, 0, 1)
                .expect("idem put")
                .0;
            assert_eq!(s, orig, "recovered idem_key not deduped (seed={seed})");
            assert_eq!(
                store.total_versions(),
                before,
                "idem dup created a version (seed={seed})"
            );
        }

        // 6. Double recovery is a fixpoint.
        drop(store);
        let store3 = TruthStore::open(&wal).expect("reopen 3");
        let mut rng = Lcg::new(seed ^ 0x1234);
        for _ in 0..128 {
            let sid = rng.range(8) as u128;
            let pred = rng.range(3) as u32;
            let snap = rng.range(nops + 2);
            let got = store3
                .get_at(Sid(sid), PredId(pred), Hlc(snap))
                .expect("get_at")
                .map(|g| match g.val {
                    Val::Bytes(b) => b,
                    Val::Tombstone => Vec::new(),
                });
            // Note: step 5 may have added a version under idem_key 1 at seq
            // cutoff? No — it deduped, so the expectation is unchanged.
            let exp = expected.get_at(sid, pred, snap);
            assert_eq!(got, exp, "fixpoint mismatch seed={seed}");
        }
    }
}

fn store_truncated_expectation(model: &Model, cutoff: u64) -> Model {
    model.truncated(cutoff)
}
