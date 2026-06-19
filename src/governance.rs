//! Governance (P4): identity, grain-level access control, value masking,
//! per-agent token budgets, and an audit trail — enforced INLINE.
//!
//! An agent holds only a [`GovernedEngine`]; the underlying [`GrainEngine`] is
//! private, so there is no path to the data that skips `authorize`. Policy is
//! executable (a [`Policy`] trait), default-deny, and every operation — including
//! denials — is recorded in the audit log. The per-agent token budget is debited
//! by the tokens actually returned, so an agent cannot exceed it; masked values
//! cost fewer tokens, so masking also conserves budget.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::disclosure::grain_tokens;
use crate::engine::{GrainEngine, WriteMeta};
use crate::error::{Error, Result};
use crate::model::{Grain, Hlc, PredId, Sid, Val};
use crate::planner::{Filter, Plan};
use crate::query::Ranked;
use crate::truth::ReadMode;

/// An authenticated caller. In production this is established by mTLS / a signed
/// token at the gateway; here it is an opaque id.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct AgentId(pub u64);

/// The kind of access requested.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Action {
    Read,
    Write,
    Delete,
}

/// A policy decision.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Decision {
    Allow,
    /// Read permitted, but the value is redacted.
    Mask,
    Deny,
}

/// The governance-relevant view of a grain that a policy decides over.
#[derive(Clone, Copy, Debug)]
pub struct GrainRef {
    pub sid: Sid,
    pub pred: PredId,
    pub category: Option<u8>,
}

/// What a rule matches on.
#[derive(Clone, Copy, Debug)]
pub enum Match {
    Any,
    Category(u8),
    Pred(PredId),
}

impl Match {
    fn matches(&self, t: &GrainRef) -> bool {
        match self {
            Match::Any => true,
            Match::Category(c) => t.category == Some(*c),
            Match::Pred(p) => t.pred == *p,
        }
    }
}

/// Pluggable authorization. Implement this for a custom policy source (OPA, a
/// directory, attribute-based rules, …).
pub trait Policy: Send + Sync {
    fn authorize(&self, agent: AgentId, action: Action, target: &GrainRef) -> Decision;
}

#[derive(Clone, Copy)]
enum Effect {
    Allow,
    Deny,
    Mask,
}

struct Rule {
    agent: AgentId,
    action: Option<Action>, // None = any action
    effect: Effect,
    matcher: Match,
}

/// A rule-based, default-deny policy. Precedence: any matching Deny wins, then
/// Mask (for reads), then Allow; no matching rule means Deny.
#[derive(Default)]
pub struct RuleSet {
    rules: Vec<Rule>,
}

impl RuleSet {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn allow(mut self, agent: AgentId, action: Action, on: Match) -> Self {
        self.rules.push(Rule {
            agent,
            action: Some(action),
            effect: Effect::Allow,
            matcher: on,
        });
        self
    }
    pub fn deny(mut self, agent: AgentId, action: Action, on: Match) -> Self {
        self.rules.push(Rule {
            agent,
            action: Some(action),
            effect: Effect::Deny,
            matcher: on,
        });
        self
    }
    /// Reads of matching grains are permitted but redacted.
    pub fn mask(mut self, agent: AgentId, on: Match) -> Self {
        self.rules.push(Rule {
            agent,
            action: Some(Action::Read),
            effect: Effect::Mask,
            matcher: on,
        });
        self
    }
}

impl Policy for RuleSet {
    fn authorize(&self, agent: AgentId, action: Action, target: &GrainRef) -> Decision {
        let mut allow = false;
        let mut mask = false;
        for r in &self.rules {
            if r.agent != agent {
                continue;
            }
            if let Some(a) = r.action {
                if a != action {
                    continue;
                }
            }
            if !r.matcher.matches(target) {
                continue;
            }
            match r.effect {
                Effect::Deny => return Decision::Deny, // deny wins immediately
                Effect::Mask => mask = true,
                Effect::Allow => allow = true,
            }
        }
        if mask && action == Action::Read {
            Decision::Mask
        } else if allow {
            Decision::Allow
        } else {
            Decision::Deny // default deny
        }
    }
}

/// One audit entry. The `seq` gives a total order (the audit log is itself
/// append-only, mirroring the WAL).
#[derive(Clone, Debug)]
pub struct AuditRecord {
    pub seq: u64,
    pub agent: AgentId,
    pub action: Action,
    pub sid: Sid,
    pub pred: PredId,
    pub decision: Decision,
    pub tokens: u64,
}

fn category_of(g: &Grain) -> Option<u8> {
    match &g.val {
        Val::Bytes(b) => b.first().copied(),
        Val::Tombstone => None,
    }
}

fn redacted() -> Val {
    Val::Bytes(b"[REDACTED]".to_vec())
}

/// The engine wrapper that enforces governance inline. An agent is given only
/// this; the inner [`GrainEngine`] is unreachable.
pub struct GovernedEngine {
    inner: GrainEngine,
    policy: Arc<dyn Policy>,
    default_budget: u64,
    budgets: Mutex<HashMap<AgentId, u64>>,
    audit: Mutex<Vec<AuditRecord>>,
    audit_seq: AtomicU64,
}

impl GovernedEngine {
    pub fn new(inner: GrainEngine, policy: Arc<dyn Policy>, default_budget_tokens: u64) -> Self {
        Self {
            inner,
            policy,
            default_budget: default_budget_tokens,
            budgets: Mutex::new(HashMap::new()),
            audit: Mutex::new(Vec::new()),
            audit_seq: AtomicU64::new(0),
        }
    }

    fn record(
        &self,
        agent: AgentId,
        action: Action,
        sid: Sid,
        pred: PredId,
        decision: Decision,
        tokens: u64,
    ) {
        let seq = self.audit_seq.fetch_add(1, Ordering::Relaxed);
        self.audit
            .lock()
            .expect("audit lock poisoned")
            .push(AuditRecord {
                seq,
                agent,
                action,
                sid,
                pred,
                decision,
                tokens,
            });
    }

    /// Write a grain — denied if policy forbids it (the write never reaches the WAL).
    pub fn put_vector(
        &self,
        agent: AgentId,
        sid: Sid,
        pred: PredId,
        header: &[u8],
        vector: &[f32],
        meta: WriteMeta,
    ) -> Result<Hlc> {
        let target = GrainRef {
            sid,
            pred,
            category: header.first().copied(),
        };
        let d = self.policy.authorize(agent, Action::Write, &target);
        self.record(agent, Action::Write, sid, pred, d, 0);
        if d == Decision::Deny {
            return Err(Error::Denied(format!(
                "agent {} cannot write {:?}",
                agent.0, target
            )));
        }
        self.inner.put_vector(sid, pred, header, vector, meta)
    }

    /// Read a grain. `Deny` → `None`; `Mask` → value redacted.
    pub fn get(&self, agent: AgentId, sid: Sid, pred: PredId) -> Result<Option<Grain>> {
        let grain = match self.inner.truth().get(sid, pred, ReadMode::Strong)? {
            Some(g) => g,
            None => return Ok(None),
        };
        let target = GrainRef {
            sid,
            pred,
            category: category_of(&grain),
        };
        let d = self.policy.authorize(agent, Action::Read, &target);
        match d {
            Decision::Deny => {
                self.record(agent, Action::Read, sid, pred, d, 0);
                Ok(None)
            }
            Decision::Mask => {
                let mut g = grain;
                g.val = redacted();
                self.record(agent, Action::Read, sid, pred, d, grain_tokens(&g) as u64);
                Ok(Some(g))
            }
            Decision::Allow => {
                self.record(
                    agent,
                    Action::Read,
                    sid,
                    pred,
                    d,
                    grain_tokens(&grain) as u64,
                );
                Ok(Some(grain))
            }
        }
    }

    /// Run a planned mixed query under governance: results the agent cannot read
    /// are dropped, masked ones redacted, and the agent's token budget is debited
    /// by what is actually returned (over-budget → denied).
    pub fn query_planned(
        &self,
        agent: AgentId,
        pred: PredId,
        query: &[f32],
        filter: &Filter,
        k: usize,
        target_recall: f64,
    ) -> Result<(Vec<Ranked>, Plan)> {
        let (ranked, plan) = self
            .inner
            .query_planned(pred, query, filter, k, target_recall)?;
        let mut out = Vec::with_capacity(ranked.len());
        for mut r in ranked {
            let target = GrainRef {
                sid: r.grain.sid,
                pred,
                category: category_of(&r.grain),
            };
            match self.policy.authorize(agent, Action::Read, &target) {
                Decision::Deny => continue,
                Decision::Mask => {
                    r.grain.val = redacted();
                    out.push(r);
                }
                Decision::Allow => out.push(r),
            }
        }
        let cost: u64 = out.iter().map(|r| grain_tokens(&r.grain) as u64).sum();
        if !self.try_spend(agent, cost) {
            self.record(agent, Action::Read, Sid(0), pred, Decision::Deny, cost);
            return Err(Error::Denied(format!(
                "token budget exceeded: need {cost}, remaining {}",
                self.remaining_budget(agent)
            )));
        }
        self.record(agent, Action::Read, Sid(0), pred, Decision::Allow, cost);
        Ok((out, plan))
    }

    fn try_spend(&self, agent: AgentId, cost: u64) -> bool {
        let mut b = self.budgets.lock().expect("budget lock poisoned");
        let rem = b.entry(agent).or_insert(self.default_budget);
        if *rem >= cost {
            *rem -= cost;
            true
        } else {
            false
        }
    }

    /// Remaining token budget for an agent.
    pub fn remaining_budget(&self, agent: AgentId) -> u64 {
        *self
            .budgets
            .lock()
            .expect("budget lock poisoned")
            .get(&agent)
            .unwrap_or(&self.default_budget)
    }

    /// A snapshot of the audit log.
    pub fn audit_log(&self) -> Vec<AuditRecord> {
        self.audit.lock().expect("audit lock poisoned").clone()
    }

    /// Block until the index reflects commit `seq` (delegates to the inner engine).
    pub fn sync(&self, seq: Hlc, timeout: Duration) -> bool {
        self.inner.sync(seq, timeout)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_deny_and_precedence() {
        let a = AgentId(1);
        let p = RuleSet::new().allow(a, Action::Read, Match::Any).deny(
            a,
            Action::Read,
            Match::Category(9),
        );
        let t_ok = GrainRef {
            sid: Sid(1),
            pred: PredId(0),
            category: Some(0),
        };
        let t_deny = GrainRef {
            sid: Sid(2),
            pred: PredId(0),
            category: Some(9),
        };
        assert_eq!(p.authorize(a, Action::Read, &t_ok), Decision::Allow);
        assert_eq!(p.authorize(a, Action::Read, &t_deny), Decision::Deny); // deny wins
        assert_eq!(
            p.authorize(AgentId(99), Action::Read, &t_ok),
            Decision::Deny
        ); // default deny
    }
}
