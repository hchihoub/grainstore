//! Core domain types. P0 carries the minimal Grain needed to exercise the WAL
//! and MVCC paths; the embedding coordinate `psi`, provenance, and bitemporal
//! `t_valid` semantics are present in the architecture and added in later phases.

/// Exact symbolic identity of a grain.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct Sid(pub u128);

/// Dictionary-encoded predicate (attribute / relation) identifier.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct PredId(pub u32);

/// Confidence annotation `c ∈ [0, 1]` — the only quantity that composes through
/// the algebra (Viterbi semiring). Stored per version.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Confidence(pub f32);

/// Hybrid logical clock timestamp. Monotonic and unique across the process.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Hlc(pub u64);

/// The payload of a grain version. A `Tombstone` marks a logical delete; it is a
/// real version in the MVCC chain so snapshot reads before it still see the
/// prior live value.
#[derive(Clone, Debug, PartialEq)]
pub enum Val {
    Bytes(Vec<u8>),
    Tombstone,
}

/// A single contextualized assertion as materialized for a reader at a snapshot.
#[derive(Clone, Debug, PartialEq)]
pub struct Grain {
    pub sid: Sid,
    pub pred: PredId,
    pub val: Val,
    pub c: Confidence,
    /// When the fact is asserted true in the world.
    pub t_valid: Hlc,
    /// When the fact was recorded (the commit timestamp; also the MVCC version).
    pub t_tx: Hlc,
}
