//! MLPerf Inference v6.0 accuracy-harness support (M0 shared infrastructure).
//!
//! This module holds the pieces every per-model accuracy binary shares:
//! fixed-point quantize/dequantize, the Python-bridge tensor format, output
//! decoding, and a range/overflow health check. See `MLPERF_ACCURACY.md`.
//!
//! Design (ported from `zk-torch-3/src/bin/resnet_mlperf_acc.rs`):
//!   Python `scripts/export_<model>.py` writes weight + input tensors and a
//!   `metadata.txt`; the Rust `<model>_mlperf_acc` bin reads them, runs
//!   `dag.run()` (the fixed-point forward pass), and decodes predictions. The
//!   zk proof only *certifies* those witness values, so accuracy == a pure
//!   function of `dag.run()` — proving is a separate per-sample spot check.

use std::fs;
use std::io::Read;
use std::path::Path;

use almost_goldilocks_cuda::field::AlmostGoldilocksField;

use crate::dag::{DataType, Role, Witness};
use crate::poly::MLPoly;
use crate::util::arith::{f_to_int, int_to_f};

// ============================================================================
// Fixed-point <-> float (must match `scripts/export_*.py::float_to_field`)
// ============================================================================

/// Quantize a real value to a signed fixed-point field element:
/// `round(x * 2^sf_log)` lifted into the field (negatives via two's-complement,
/// matching [`int_to_f`]). Inverse of [`dequantize`].
#[inline]
pub fn quantize(x: f64, sf_log: usize) -> AlmostGoldilocksField {
    let sf = (1u64 << sf_log) as f64;
    int_to_f((x * sf).round() as i128)
}

/// Dequantize a field element back to a real value: `f_to_int(f) / 2^sf_log`.
#[inline]
pub fn dequantize(f: AlmostGoldilocksField, sf_log: usize) -> f64 {
    f_to_int(f) as f64 / (1u64 << sf_log) as f64
}

// ============================================================================
// Python-bridge tensor format
// ============================================================================
// Binary layout: [ndim: u32] [shape: ndim x u32] [data: n x u64 little-endian].
// Values are already field-encoded (a signed fixed-point i64/i128 lifted into
// [0, q) by the exporter), so we read them straight into field elements.

/// Read a `[ndim][shape...][u64 data...]` tensor file.
/// Returns `(shape, field_data)`.
pub fn read_tensor(path: &Path) -> (Vec<usize>, Vec<AlmostGoldilocksField>) {
    let mut buf = Vec::new();
    fs::File::open(path)
        .unwrap_or_else(|e| panic!("mlperf::read_tensor: cannot open {}: {}", path.display(), e))
        .read_to_end(&mut buf)
        .unwrap();

    let mut off = 0usize;
    let rd_u32 = |b: &[u8], o: &mut usize| -> u32 {
        let v = u32::from_le_bytes(b[*o..*o + 4].try_into().unwrap());
        *o += 4;
        v
    };

    let ndim = rd_u32(&buf, &mut off) as usize;
    let shape: Vec<usize> = (0..ndim).map(|_| rd_u32(&buf, &mut off) as usize).collect();

    let n = (buf.len() - off) / 8;
    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        let v = u64::from_le_bytes(buf[off + i * 8..off + (i + 1) * 8].try_into().unwrap());
        data.push(AlmostGoldilocksField(v));
    }
    (shape, data)
}

/// Write a tensor in the same format (for Rust-side dumps / round-trip tests).
pub fn write_tensor(path: &Path, shape: &[usize], data: &[AlmostGoldilocksField]) {
    let mut out = Vec::with_capacity(4 + 4 * shape.len() + 8 * data.len());
    out.extend_from_slice(&(shape.len() as u32).to_le_bytes());
    for &s in shape {
        out.extend_from_slice(&(s as u32).to_le_bytes());
    }
    for f in data {
        out.extend_from_slice(&f.0.to_le_bytes());
    }
    fs::write(path, out)
        .unwrap_or_else(|e| panic!("mlperf::write_tensor: cannot write {}: {}", path.display(), e));
}

/// Read a tensor file straight into a [`Witness`] with the given scale/role.
pub fn load_witness(path: &Path, sf_log: usize, data_type: DataType, role: Role) -> Witness {
    let (shape, data) = read_tensor(path);
    Witness::new(shape, data, data_type, sf_log, role)
}

// ============================================================================
// metadata.txt
// ============================================================================

/// Parsed `metadata.txt` (written by every `export_*.py`).
#[derive(Debug, Clone)]
pub struct Metadata {
    pub sf_log: usize,
    pub num_conv: usize,
    pub num_classes: usize,
    /// `(c_in, c_out, kh, kw)` per conv, in graph order.
    pub conv_configs: Vec<(usize, usize, usize, usize)>,
}

/// Parse `metadata.txt`: `sf_log=`, `num_conv=`, `num_classes=`, `conv_i=ci,co,kh,kw`.
pub fn read_metadata(dir: &Path) -> Metadata {
    let content = fs::read_to_string(dir.join("metadata.txt"))
        .expect("mlperf::read_metadata: cannot read metadata.txt");
    let mut m = Metadata { sf_log: 10, num_conv: 0, num_classes: 1000, conv_configs: Vec::new() };
    for line in content.lines() {
        if let Some(v) = line.strip_prefix("sf_log=") {
            m.sf_log = v.trim().parse().unwrap();
        } else if let Some(v) = line.strip_prefix("num_conv=") {
            m.num_conv = v.trim().parse().unwrap();
        } else if let Some(v) = line.strip_prefix("num_classes=") {
            m.num_classes = v.trim().parse().unwrap();
        } else if line.starts_with("conv_") {
            if let Some((_, rhs)) = line.split_once('=') {
                let nums: Vec<usize> = rhs.split(',').filter_map(|s| s.trim().parse().ok()).collect();
                if nums.len() == 4 {
                    m.conv_configs.push((nums[0], nums[1], nums[2], nums[3]));
                }
            }
        }
    }
    m
}

// ============================================================================
// Output decoding
// ============================================================================

/// Dequantized values of a witness's first `count` entries (flat order).
pub fn decode_logits(w: &Witness, count: usize, sf_log: usize) -> Vec<f64> {
    let evals = w.data.as_ref().expect("decode_logits: witness has no data").evaluations_ref();
    (0..count).map(|i| dequantize(evals[i], sf_log)).collect()
}

/// `(argmax_index, top1_value)` over the first `count` logits.
pub fn decode_argmax(w: &Witness, count: usize, sf_log: usize) -> (usize, f64) {
    let logits = decode_logits(w, count, sf_log);
    let mut best = (0usize, f64::NEG_INFINITY);
    for (i, &v) in logits.iter().enumerate() {
        if v > best.1 {
            best = (i, v);
        }
    }
    best
}

/// Top-`k` `(class, value)` pairs, highest first (for debug prints).
pub fn topk(w: &Witness, count: usize, k: usize, sf_log: usize) -> Vec<(usize, f64)> {
    let mut v: Vec<(usize, f64)> = decode_logits(w, count, sf_log)
        .into_iter()
        .enumerate()
        .collect();
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    v.truncate(k);
    v
}

// ============================================================================
// Range / overflow health check
// ============================================================================
// The authoritative overflow signal is the `[range] WARNING` printed by
// `basicblock/range.rs` during `dag.run()` when a NonNegative check sees a
// value outside `[0, 2^table_size_log)` — such a sample is INVALID (the proof
// will not verify), not merely "slightly wrong". This scan is a cheap host-side
// health check the harness can act on programmatically: if the largest
// activation magnitude reaches the table bound, at least one range/relu chunk
// is at risk. Treat a non-clean report as a red flag for that sample.

/// Summary of activation magnitudes after a `dag.run()`.
#[derive(Debug, Clone)]
pub struct RangeReport {
    /// Largest |signed value| seen across all witness activations.
    pub max_abs: i128,
    /// Edge id carrying that maximum.
    pub max_abs_edge: usize,
    /// Count of individual values with |value| >= 2^table_size_log.
    pub over_table: usize,
    pub table_size_log: usize,
}

impl RangeReport {
    /// True iff no activation reached the range-table bound.
    pub fn is_clean(&self) -> bool {
        self.over_table == 0
    }
}

/// Scan every witness's evaluations for magnitude vs `2^table_size_log`.
pub fn range_health_check(witnesses: &[Vec<Witness>], table_size_log: usize) -> RangeReport {
    let bound = 1i128 << table_size_log;
    let mut rep = RangeReport { max_abs: 0, max_abs_edge: 0, over_table: 0, table_size_log };
    for (eid, ws) in witnesses.iter().enumerate() {
        for w in ws {
            let Some(d) = w.data.as_ref() else { continue };
            for &f in d.evaluations_ref() {
                let a = f_to_int(f).abs();
                if a > rep.max_abs {
                    rep.max_abs = a;
                    rep.max_abs_edge = eid;
                }
                if a >= bound {
                    rep.over_table += 1;
                }
            }
        }
    }
    rep
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_roundtrips() {
        for &sf in &[10usize, 16] {
            for &x in &[0.0f64, 1.0, -1.0, 3.14159, -2.71828, 0.5, -0.25] {
                let q = quantize(x, sf);
                let back = dequantize(q, sf);
                // within one quantization step
                assert!((back - x).abs() <= 1.0 / (1u64 << sf) as f64 + 1e-9, "x={x} back={back}");
            }
        }
    }

    #[test]
    fn tensor_roundtrips() {
        let dir = std::env::temp_dir();
        let path = dir.join("zk4_mlperf_tensor_roundtrip.bin");
        let shape = vec![2usize, 3];
        let data: Vec<AlmostGoldilocksField> =
            (0..6).map(|i| int_to_f(i as i128 - 3)).collect();
        write_tensor(&path, &shape, &data);
        let (s2, d2) = read_tensor(&path);
        assert_eq!(shape, s2);
        assert_eq!(data, d2);
        let _ = fs::remove_file(&path);
    }
}
