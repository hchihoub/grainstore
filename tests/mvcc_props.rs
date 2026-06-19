//! Property: every snapshot read returns exactly the newest version with
//! `t_tx <= snapshot`, never a tombstoned or future version. Checked against the
//! reference model over many seeded random programs.

mod common;

use common::{Lcg, Model, TempCase};
use grainstore::model::{Hlc, PredId, Sid, Val};
use grainstore::TruthStore;

#[test]
fn mvcc_reads_match_model() {
    for seed in 0..150u64 {
        let case = TempCase::new("mvcc", seed);
        let store = TruthStore::open(&case.wal()).expect("open");
        let mut model = Model::new();
        let mut rng = Lcg::new(seed);

        let nops = 20 + seed % 60;

        for op_i in 0..nops {
            // Unique idem per op here; idempotency has its own dedicated test.
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

        // Probe arbitrary (sid, pred, snapshot) — including snapshots between and
        // beyond commit timestamps.
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
            let expect = model.get_at(sid, pred, snap);
            assert_eq!(got, expect, "seed={seed} sid={sid} pred={pred} snap={snap}");
        }
    }
}
