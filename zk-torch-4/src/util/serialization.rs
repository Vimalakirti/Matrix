use almost_goldilocks_cuda::field::AlmostGoldilocksField;

/// Serialize an `AlmostGoldilocksField` to 8 little-endian bytes. The stored
/// representation is the canonical form (in `[0, q)`).
pub fn field_to_bytes(f: &AlmostGoldilocksField) -> [u8; 8] {
    f.reduce().0.to_le_bytes()
}

/// Deserialize an `AlmostGoldilocksField` from up to 8 little-endian bytes.
/// Missing high bytes are treated as zero; the result is not auto-canonicalized
/// (mirrors zk-torch-3 behaviour).
pub fn bytes_to_field(bytes: &[u8]) -> AlmostGoldilocksField {
    let mut buf = [0u8; 8];
    let len = bytes.len().min(8);
    buf[..len].copy_from_slice(&bytes[..len]);
    AlmostGoldilocksField(u64::from_le_bytes(buf))
}

#[cfg(test)]
mod tests {
    use super::*;
    use almost_goldilocks_cuda::field::ALMOST_GOLDILOCKS_PRIME;

    #[test]
    fn roundtrip_random_values() {
        for v in [0u64, 1, 42, 0xDEAD_BEEF, ALMOST_GOLDILOCKS_PRIME - 1] {
            let f = AlmostGoldilocksField(v);
            let bytes = field_to_bytes(&f);
            let back = bytes_to_field(&bytes);
            assert_eq!(back, f.reduce());
        }
    }

    #[test]
    fn bytes_to_field_pads_short_input() {
        assert_eq!(bytes_to_field(&[]), AlmostGoldilocksField(0));
        assert_eq!(bytes_to_field(&[0xAB]), AlmostGoldilocksField(0xAB));
        assert_eq!(bytes_to_field(&[0xAB, 0xCD]), AlmostGoldilocksField(0xCDAB));
    }

    #[test]
    fn field_to_bytes_canonicalizes_first() {
        // A non-canonical representative serializes to its canonical form.
        let raw = ALMOST_GOLDILOCKS_PRIME + 5;
        let f = AlmostGoldilocksField(raw);
        assert_eq!(field_to_bytes(&f), 5u64.to_le_bytes());
    }
}
