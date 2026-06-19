//! Embedding: turn a grain's raw value into its semantic coordinate `ψ ∈ ℝ^d`.
//!
//! The trait is the seam where a production model server (ONNX / a remote gRPC
//! embedder) plugs in. P1 ships [`RawVectorEmbedder`], which interprets the
//! value bytes as a length-prefixed little-endian `f32` vector after a fixed
//! header offset — so callers that already hold vectors (and a small category
//! header for filtering) flow straight through.

/// Produces a vector for a stored value, or `None` if the value carries no
/// embeddable payload (e.g. a tombstone, signalled upstream by an empty value).
pub trait Embedder: Send + Sync {
    /// Embed a raw value. Returns `None` when there is nothing to index.
    fn embed(&self, value: &[u8]) -> Option<Vec<f32>>;
    /// Dimensionality of produced vectors.
    fn dim(&self) -> usize;
}

/// Value layout `[header: N bytes][f32 * dim, little-endian]`. The header is
/// opaque to the embedder (used by query predicates, e.g. a category tag).
pub struct RawVectorEmbedder {
    dim: usize,
    header: usize,
}

impl RawVectorEmbedder {
    pub fn new(dim: usize, header: usize) -> Self {
        Self { dim, header }
    }
}

impl Embedder for RawVectorEmbedder {
    fn embed(&self, value: &[u8]) -> Option<Vec<f32>> {
        let want = self.header + self.dim * 4;
        if value.len() != want {
            return None; // tombstone or malformed → not indexable
        }
        let mut v = Vec::with_capacity(self.dim);
        for chunk in value[self.header..].chunks_exact(4) {
            v.push(f32::from_le_bytes(
                chunk.try_into().expect("chunk is 4 bytes"),
            ));
        }
        Some(v)
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

/// Encode a value as `[header bytes][vector]` for the `RawVectorEmbedder`.
pub fn encode_value_with_header(header: &[u8], vec: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(header.len() + vec.len() * 4);
    out.extend_from_slice(header);
    for &x in vec {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_with_header() {
        let e = RawVectorEmbedder::new(3, 1);
        let bytes = encode_value_with_header(&[5u8], &[1.0, 2.0, 3.0]);
        assert_eq!(e.embed(&bytes), Some(vec![1.0, 2.0, 3.0]));
        assert_eq!(bytes[0], 5); // header preserved for predicates
        assert_eq!(e.embed(&[]), None); // tombstone-shaped
    }
}
