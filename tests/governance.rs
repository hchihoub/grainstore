//! Governance enforcement, end to end: an unauthorized write never reaches the
//! store; reads are denied or masked per policy; the per-agent token budget is a
//! hard ceiling; and every operation is audited.

use std::sync::Arc;
use std::time::Duration;

use grainstore::model::{Hlc, PredId, Sid, Val};
use grainstore::{
    Action, AgentId, Decision, EngineConfig, Filter, GovernedEngine, GrainEngine, Match, RuleSet,
    WriteMeta,
};

struct R(u64);
impl R {
    fn new(s: u64) -> Self {
        R(s ^ 0x9E37_79B9_7F4A_7C15)
    }
    fn nx(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn f32(&mut self) -> f32 {
        (self.nx() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn vec(&mut self, d: usize) -> Vec<f32> {
        (0..d).map(|_| self.f32()).collect()
    }
}

fn wal(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("grainstore_gov_{}_{}", std::process::id(), tag));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("mkdir");
    dir.join("wal.log")
}

const DIM: usize = 8;
const PRED: PredId = PredId(0);
const ADMIN: AgentId = AgentId(0);
const WRITER: AgentId = AgentId(1);
const READER: AgentId = AgentId(2);
const MASKED: AgentId = AgentId(3);
const DENIED: AgentId = AgentId(4);

#[test]
fn governance_is_enforced_inline() {
    let cfg = EngineConfig::new(DIM).with_header(1);
    let inner = GrainEngine::open(&wal("gov"), cfg).expect("open");

    let policy = Arc::new(
        RuleSet::new()
            .allow(ADMIN, Action::Write, Match::Any)
            .allow(ADMIN, Action::Read, Match::Any)
            .allow(WRITER, Action::Write, Match::Any)
            .allow(READER, Action::Read, Match::Category(0))
            .mask(MASKED, Match::Category(0)),
        // DENIED has no rules → default deny everywhere.
    );
    let eng = GovernedEngine::new(inner, policy, /* budget tokens */ 200);

    // Writer writes 20 grains, category = sid % 2.
    let mut rng = R::new(1);
    let mut last = Hlc(0);
    for i in 0..20u128 {
        last = eng
            .put_vector(
                WRITER,
                Sid(i),
                PRED,
                &[(i % 2) as u8],
                &rng.vec(DIM),
                WriteMeta::new(i),
            )
            .expect("writer put");
    }

    // A denied agent's write is refused AND never persists.
    let denied_write = eng.put_vector(
        DENIED,
        Sid(999),
        PRED,
        &[0u8],
        &rng.vec(DIM),
        WriteMeta::new(999),
    );
    assert!(
        matches!(denied_write, Err(grainstore::Error::Denied(_))),
        "denied write must error"
    );
    assert!(
        eng.sync(last, Duration::from_secs(10)),
        "materializer lagged"
    );
    assert!(
        eng.get(ADMIN, Sid(999), PRED).expect("admin get").is_none(),
        "denied write must not persist"
    );

    // Reads: allow / category-scoped / mask / deny.
    let full = eng
        .get(READER, Sid(0), PRED)
        .expect("reader get")
        .expect("present"); // cat 0
    assert!(
        matches!(&full.val, Val::Bytes(b) if b.first() == Some(&0)),
        "reader sees full value"
    );
    assert!(
        eng.get(READER, Sid(1), PRED).expect("reader get").is_none(),
        "reader cannot read cat 1"
    );

    let masked = eng
        .get(MASKED, Sid(0), PRED)
        .expect("masked get")
        .expect("present");
    assert_eq!(
        masked.val,
        Val::Bytes(b"[REDACTED]".to_vec()),
        "masked agent gets redacted value"
    );

    assert!(
        eng.get(DENIED, Sid(0), PRED).expect("denied get").is_none(),
        "denied agent reads nothing"
    );

    // Token budget is a hard ceiling: query until it is exhausted.
    let filter = Filter::HeaderByteEq {
        offset: 0,
        value: 0,
    };
    let mut ok = 0;
    let mut hit_limit = false;
    for _ in 0..10 {
        let q = rng.vec(DIM);
        match eng.query_planned(READER, PRED, &q, &filter, 10, 0.90) {
            Ok((res, _)) => {
                assert!(res
                    .iter()
                    .all(|r| matches!(&r.grain.val, Val::Bytes(b) if b.first() == Some(&0))));
                ok += 1;
            }
            Err(grainstore::Error::Denied(m)) => {
                assert!(m.contains("budget"), "expected a budget denial, got: {m}");
                hit_limit = true;
                break;
            }
            Err(e) => panic!("unexpected error: {e}"),
        }
    }
    assert!(ok >= 1, "at least one query should succeed within budget");
    assert!(hit_limit, "budget ceiling should eventually deny");
    assert!(eng.remaining_budget(READER) < 200);

    // Masked reads also worked but cost fewer tokens than full reads.
    // Audit log recorded every operation, including denials.
    let audit = eng.audit_log();
    assert!(
        audit.len() >= 25,
        "every op is audited (got {})",
        audit.len()
    );
    assert!(
        audit.iter().any(|r| r.agent == DENIED
            && r.action == Action::Write
            && r.decision == Decision::Deny),
        "the denied write must appear in the audit log"
    );
    assert!(
        audit
            .iter()
            .any(|r| r.agent == READER && r.decision == Decision::Deny && r.tokens > 0),
        "the budget denial must be audited with its cost"
    );
}
