//! MVCC key encoding.
//!
//! The "version chain" is not a linked list — it is a key sort order. For a
//! fixed `(sid, pred)` prefix, versions are stored with suffix `!t_tx` (bitwise
//! complement, big-endian). Because larger `t_tx` maps to a smaller suffix, the
//! ordered store iterates versions **newest-first**, and a snapshot read becomes
//! a single `seek_ge` (see [`seek_target`]).
//!
//! Layout: `[sid: 16 BE][pred: 4 BE][!t_tx: 8 BE]` = 28 bytes.

use crate::model::{Hlc, PredId, Sid};

/// Length of a full grain key, in bytes.
pub const KEY_LEN: usize = 28;
/// Length of the `(sid, pred)` prefix, in bytes.
pub const PREFIX_LEN: usize = 20;

/// Encode the full versioned key for `(sid, pred)` at transaction time `t_tx`.
pub fn grain_key(sid: Sid, pred: PredId, t_tx: Hlc) -> Vec<u8> {
    let mut k = Vec::with_capacity(KEY_LEN);
    k.extend_from_slice(&sid.0.to_be_bytes()); // 16
    k.extend_from_slice(&pred.0.to_be_bytes()); // 4
    k.extend_from_slice(&(!t_tx.0).to_be_bytes()); // 8, complemented → newest-first
    k
}

/// Encode the `(sid, pred)` prefix used to bound iteration to one logical key.
pub fn grain_prefix(sid: Sid, pred: PredId) -> Vec<u8> {
    let mut p = Vec::with_capacity(PREFIX_LEN);
    p.extend_from_slice(&sid.0.to_be_bytes());
    p.extend_from_slice(&pred.0.to_be_bytes());
    p
}

/// Seek target for a snapshot read at `snapshot`.
///
/// `seek_ge(seek_target(..))` returns the smallest key `>=` the target, which —
/// because the suffix is `!t_tx` — is the version with the **largest `t_tx`
/// that is `<= snapshot`** within the prefix (or a key from a different prefix,
/// which the caller rejects via the prefix check).
pub fn seek_target(sid: Sid, pred: PredId, snapshot: Hlc) -> Vec<u8> {
    grain_key(sid, pred, snapshot)
}

/// Recover `t_tx` from a stored key suffix.
pub fn t_tx_from_key(key: &[u8]) -> Option<Hlc> {
    if key.len() < KEY_LEN {
        return None;
    }
    let raw: [u8; 8] = key[PREFIX_LEN..KEY_LEN].try_into().ok()?;
    Some(Hlc(!u64::from_be_bytes(raw)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn newer_versions_sort_before_older() {
        let s = Sid(7);
        let p = PredId(3);
        let older = grain_key(s, p, Hlc(10));
        let newer = grain_key(s, p, Hlc(20));
        assert!(newer < older, "newer t_tx must sort first within a prefix");
    }

    #[test]
    fn key_roundtrips_t_tx() {
        let k = grain_key(Sid(1), PredId(2), Hlc(12345));
        assert_eq!(t_tx_from_key(&k), Some(Hlc(12345)));
    }
}
