use goldilocks_cuda::GoldilocksField;

/// Serialize a GoldilocksField to little-endian bytes.
pub fn field_to_bytes(f: &GoldilocksField) -> [u8; 8] {
    f.0.to_le_bytes()
}

/// Deserialize a GoldilocksField from little-endian bytes.
pub fn bytes_to_field(bytes: &[u8]) -> GoldilocksField {
    let mut buf = [0u8; 8];
    let len = bytes.len().min(8);
    buf[..len].copy_from_slice(&bytes[..len]);
    GoldilocksField(u64::from_le_bytes(buf))
}
