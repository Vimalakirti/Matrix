use goldilocks_cuda::{GoldilocksField, GoldilocksExt2};

use crate::basicblock::BasicBlock;
use crate::dag::{Claim, DataType, Role, Witness};
use crate::sumcheck::SumcheckProof;
use crate::transcript::Transcript;
use crate::util::arith::{ext2_add, ext2_mul, ext2_sub, get_n, log2_ceil};

/// Concat: equal-size channel concatenation.
/// For inputs A[C, ...spatial...] and B[C, ...spatial...],
/// output Y[2C, ...spatial...].
///
/// In little-endian MLE layout, channels are the highest bits.
/// The extra bit (MSB of channel dim) selects A (0) or B (1):
///   Y(r_spatial, r_c_low, r_c_top) = (1 - r_c_top) * A(r_spatial, r_c_low) + r_c_top * B(r_spatial, r_c_low)
///
/// This is a zero-sumcheck claim transform — no sumcheck needed.
#[derive(Clone, Debug)]
pub struct Concat {
    pub channels_a: usize,  // channels of input A (= channels of input B)
    pub spatial_dims: Vec<usize>,  // e.g., [D, H, W] or [H, W]
}

impl BasicBlock for Concat {
    fn run(&self, inputs: &[&Witness]) -> Vec<Witness> {
        assert_eq!(inputs.len(), 2);
        let a = inputs[0];
        let b = inputs[1];

        let c_a = self.channels_a;
        let c_out = 2 * c_a;

        // Compute padded sizes
        let spatial_pads: Vec<usize> = self.spatial_dims.iter().map(|&s| s.next_power_of_two()).collect();
        let c_a_pad = c_a.next_power_of_two();
        let c_out_pad = c_out.next_power_of_two();

        // Spatial stride product
        let spatial_size: usize = spatial_pads.iter().product();
        let out_size = c_out_pad * spatial_size;
        let in_size_a = c_a_pad * spatial_size;

        let mut out_data = vec![GoldilocksField(0); out_size];

        // Copy A into first half of channels, B into second half
        // Little-endian: spatial bits (lowest) | channel bits (highest)
        // A occupies channels 0..c_a, B occupies channels c_a..2*c_a
        for c in 0..c_a {
            for s in 0..spatial_size {
                let a_idx = s + c * spatial_size;
                let out_idx = s + c * spatial_size;
                if a_idx < in_size_a {
                    out_data[out_idx] = a.data.as_ref().unwrap().index(a_idx);
                }
            }
        }
        for c in 0..c_a {
            for s in 0..spatial_size {
                let b_idx = s + c * spatial_size;
                let out_idx = s + (c + c_a) * spatial_size;
                if b_idx < in_size_a {
                    out_data[out_idx] = b.data.as_ref().unwrap().index(b_idx);
                }
            }
        }

        let mut out_shape = vec![c_out];
        out_shape.extend_from_slice(&self.spatial_dims);
        let n = get_n(&out_shape);
        if out_data.len() < (1 << n) {
            out_data.resize(1 << n, GoldilocksField(0));
        }
        vec![Witness::new(out_shape, out_data, DataType::Uint, 0, Role::Output)]
    }

    fn prove(
        &self,
        witnesses: &[&Witness],
        edge_ids: &[usize],
        out_claims: &[&Claim],
        _transcript: &mut Transcript,
    ) -> (Vec<SumcheckProof>, Vec<Claim>) {
        let out_claim = out_claims[0];
        let a_edge = edge_ids[0];
        let b_edge = edge_ids[1];

        let c_a = self.channels_a;
        let c_out = 2 * c_a;
        let l_c_out = log2_ceil(c_out.max(1));
        let l_c_a = log2_ceil(c_a.max(1));

        // Number of spatial variables
        let l_spatial: usize = self.spatial_dims.iter().map(|&s| log2_ceil(s.max(1))).sum();

        // r_out = [spatial_bits | c_bits]
        // c_bits has l_c_out bits. The MSB (r_c_top) selects A vs B.
        // The lower l_c_a bits are r_c_low.
        let r_c_top = out_claim.point[l_spatial + l_c_out - 1];

        // point_ab = spatial bits ++ lower channel bits
        let mut point_ab = Vec::with_capacity(l_spatial + l_c_a);
        point_ab.extend_from_slice(&out_claim.point[..l_spatial]);  // spatial
        point_ab.extend_from_slice(&out_claim.point[l_spatial..l_spatial + l_c_a]);  // lower channel bits

        // Evaluate A and B at point_ab
        let a_data = witnesses[0];
        let b_data = witnesses[1];
        let eval_a = a_data.data.as_ref().unwrap().evaluate_at_point_ext2(&point_ab);
        let eval_b = b_data.data.as_ref().unwrap().evaluate_at_point_ext2(&point_ab);

        let one = GoldilocksExt2::from_base(GoldilocksField(1));
        let one_minus_top = ext2_sub(one, r_c_top);

        // Verify: v_out = (1 - r_c_top) * eval_a + r_c_top * eval_b
        let _expected = ext2_add(ext2_mul(one_minus_top, eval_a), ext2_mul(r_c_top, eval_b));

        let a_claim = Claim {
            edge_id: a_edge,
            sparse_id: 0,
            point: point_ab.clone(),
            eval: eval_a,
        };
        let b_claim = Claim {
            edge_id: b_edge,
            sparse_id: 0,
            point: point_ab,
            eval: eval_b,
        };

        (vec![], vec![a_claim, b_claim])
    }

    fn verify(
        &self,
        _witnesses: &[&Witness],
        claims: &[&Claim],
        _sumcheck_proofs: &[&SumcheckProof],
        _transcript: &mut Transcript,
    ) -> bool {
        // claims: [a_claim, b_claim, out_claim]
        let out_claim = claims.last().unwrap();
        let a_claim = &claims[0];
        let b_claim = &claims[1];

        let c_out = 2 * self.channels_a;
        let l_c_out = log2_ceil(c_out.max(1));
        let l_c_a = log2_ceil(self.channels_a.max(1));
        let l_spatial: usize = self.spatial_dims.iter().map(|&s| log2_ceil(s.max(1))).sum();

        let r_c_top = out_claim.point[l_spatial + l_c_out - 1];

        let one = GoldilocksExt2::from_base(GoldilocksField(1));
        let one_minus_top = ext2_sub(one, r_c_top);

        // Check: v_out = (1 - r_c_top) * eval_a + r_c_top * eval_b
        let expected = ext2_add(ext2_mul(one_minus_top, a_claim.eval), ext2_mul(r_c_top, b_claim.eval));
        if expected != out_claim.eval {
            println!("Concat: eval mismatch: expected {:?}, got {:?}", expected, out_claim.eval);
            return false;
        }

        // Check points match
        let expected_point_len = l_spatial + l_c_a;
        if a_claim.point.len() != expected_point_len || b_claim.point.len() != expected_point_len {
            println!("Concat: point length mismatch");
            return false;
        }

        // Check spatial + lower channel bits match
        if a_claim.point[..l_spatial] != out_claim.point[..l_spatial] {
            println!("Concat: spatial point mismatch");
            return false;
        }
        if a_claim.point[l_spatial..] != out_claim.point[l_spatial..l_spatial + l_c_a] {
            println!("Concat: channel point mismatch");
            return false;
        }
        if a_claim.point != b_claim.point {
            println!("Concat: A and B point mismatch");
            return false;
        }

        true
    }
}

/// ChannelSlice: extracts a contiguous slice of channels from a tensor.
/// Input X[C_in, ...spatial...], Output Y[C_out, ...spatial...] starting at channel `channel_start`.
///
/// Zero-sumcheck claim transform: no sumcheck needed.
/// The claim on Y at point (r_spatial, r_c_out) maps to a claim on X at
/// (r_spatial, r_c_out_bits ++ selector_bits) where selector_bits encode (channel_start / C_out)
/// in the upper channel bit positions.
#[derive(Clone, Debug)]
pub struct ChannelSlice {
    pub channels_in: usize,
    pub channels_out: usize,
    pub channel_start: usize,
    pub spatial_dims: Vec<usize>,
}

impl BasicBlock for ChannelSlice {
    fn run(&self, inputs: &[&Witness]) -> Vec<Witness> {
        assert_eq!(inputs.len(), 1);
        let x = inputs[0];

        let c_out_pad = self.channels_out.next_power_of_two();

        let spatial_pads: Vec<usize> = self.spatial_dims.iter().map(|&s| s.next_power_of_two()).collect();
        let spatial_size: usize = spatial_pads.iter().product();

        let out_size = c_out_pad * spatial_size;
        let mut out_data = vec![GoldilocksField(0); out_size];

        for c in 0..self.channels_out {
            let c_in = c + self.channel_start;
            for s in 0..spatial_size {
                let x_idx = s + c_in * spatial_size;
                let o_idx = s + c * spatial_size;
                if c_in < self.channels_in {
                    out_data[o_idx] = x.data.as_ref().unwrap().index(x_idx);
                }
            }
        }

        let mut out_shape = vec![self.channels_out];
        out_shape.extend_from_slice(&self.spatial_dims);
        let n = get_n(&out_shape);
        if out_data.len() < (1 << n) {
            out_data.resize(1 << n, GoldilocksField(0));
        }
        vec![Witness::new(out_shape, out_data, DataType::Uint, 0, Role::Output)]
    }

    fn prove(
        &self,
        witnesses: &[&Witness],
        edge_ids: &[usize],
        out_claims: &[&Claim],
        _transcript: &mut Transcript,
    ) -> (Vec<SumcheckProof>, Vec<Claim>) {
        let out_claim = out_claims[0];
        let x_edge = edge_ids[0];

        let l_c_out = log2_ceil(self.channels_out.max(1));
        let l_c_in = log2_ceil(self.channels_in.max(1));
        let l_spatial: usize = self.spatial_dims.iter().map(|&s| log2_ceil(s.max(1))).sum();

        // Build X evaluation point: spatial bits ++ c_out bits ++ selector bits
        let mut x_point = Vec::with_capacity(l_spatial + l_c_in);

        // Spatial bits: same as output claim
        x_point.extend_from_slice(&out_claim.point[..l_spatial]);

        // Lower channel bits: same as output claim's channel bits
        x_point.extend_from_slice(&out_claim.point[l_spatial..l_spatial + l_c_out]);

        // Upper channel bits: encode channel_start / channels_out
        let selector = self.channel_start / self.channels_out;
        let num_selector_bits = l_c_in - l_c_out;
        for bit in 0..num_selector_bits {
            let bit_val = (selector >> bit) & 1;
            x_point.push(GoldilocksExt2::from_base(GoldilocksField(bit_val as u64)));
        }

        let x_data = witnesses[0];
        let x_eval = x_data.data.as_ref().unwrap().evaluate_at_point_ext2(&x_point);

        let x_claim = Claim {
            edge_id: x_edge,
            sparse_id: 0,
            point: x_point,
            eval: x_eval,
        };

        (vec![], vec![x_claim])
    }

    fn verify(
        &self,
        _witnesses: &[&Witness],
        claims: &[&Claim],
        _sumcheck_proofs: &[&SumcheckProof],
        _transcript: &mut Transcript,
    ) -> bool {
        // claims: [x_claim, out_claim]
        let out_claim = claims.last().unwrap();
        let x_claim = &claims[0];

        let l_c_out = log2_ceil(self.channels_out.max(1));
        let l_c_in = log2_ceil(self.channels_in.max(1));
        let l_spatial: usize = self.spatial_dims.iter().map(|&s| log2_ceil(s.max(1))).sum();

        // Check point length
        if x_claim.point.len() != l_spatial + l_c_in {
            println!("ChannelSlice: x_claim point length mismatch: {} vs expected {}", x_claim.point.len(), l_spatial + l_c_in);
            return false;
        }

        // Check spatial bits match
        if x_claim.point[..l_spatial] != out_claim.point[..l_spatial] {
            println!("ChannelSlice: spatial point mismatch");
            return false;
        }

        // Check lower channel bits match
        if x_claim.point[l_spatial..l_spatial + l_c_out] != out_claim.point[l_spatial..l_spatial + l_c_out] {
            println!("ChannelSlice: lower channel bits mismatch");
            return false;
        }

        // Check upper channel bits are correct selector bits
        let selector = self.channel_start / self.channels_out;
        let num_selector_bits = l_c_in - l_c_out;
        for bit in 0..num_selector_bits {
            let bit_val = (selector >> bit) & 1;
            let expected = GoldilocksExt2::from_base(GoldilocksField(bit_val as u64));
            if x_claim.point[l_spatial + l_c_out + bit] != expected {
                println!("ChannelSlice: selector bit {} mismatch", bit);
                return false;
            }
        }

        // The eval should be passed through directly: X(extended_point) = Y(point)
        if x_claim.eval != out_claim.eval {
            println!("ChannelSlice: eval mismatch: x={:?}, out={:?}", x_claim.eval, out_claim.eval);
            return false;
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::{Witness, DataType, Role};
    use goldilocks_cuda::GoldilocksField;

    fn make_witness(shape: Vec<usize>, data: Vec<u64>) -> Witness {
        let data: Vec<GoldilocksField> = data.into_iter().map(GoldilocksField).collect();
        Witness::new(shape, data, DataType::Uint, 0, Role::Input)
    }

    #[test]
    fn test_concat_run() {
        // A[2, 4], B[2, 4] → Y[4, 4]
        let cat = Concat { channels_a: 2, spatial_dims: vec![4] };
        let a = make_witness(vec![2, 4], vec![1, 2, 3, 4, 5, 6, 7, 8]);
        let b = make_witness(vec![2, 4], vec![10, 20, 30, 40, 50, 60, 70, 80]);
        let result = cat.run(&[&a, &b]);
        let y = &result[0];

        assert_eq!(y.shape, vec![4, 4]);
        // Channel 0 from A: [1,2,3,4]
        assert_eq!(y.data.as_ref().unwrap().index(0), GoldilocksField(1));
        assert_eq!(y.data.as_ref().unwrap().index(1), GoldilocksField(2));
        // Channel 1 from A: [5,6,7,8]
        assert_eq!(y.data.as_ref().unwrap().index(4), GoldilocksField(5));
        // Channel 2 from B: [10,20,30,40]
        assert_eq!(y.data.as_ref().unwrap().index(8), GoldilocksField(10));
        assert_eq!(y.data.as_ref().unwrap().index(9), GoldilocksField(20));
        // Channel 3 from B: [50,60,70,80]
        assert_eq!(y.data.as_ref().unwrap().index(12), GoldilocksField(50));
    }

    #[test]
    fn test_concat_prove_verify() {
        use crate::transcript::Transcript;

        let cat = Concat { channels_a: 2, spatial_dims: vec![4] };
        let a = make_witness(vec![2, 4], vec![1, 2, 3, 4, 5, 6, 7, 8]);
        let b = make_witness(vec![2, 4], vec![10, 20, 30, 40, 50, 60, 70, 80]);
        let result = cat.run(&[&a, &b]);
        let y = &result[0];

        // Create output claim
        let n_y = y.data.as_ref().unwrap().n();
        let mut transcript = Transcript::new(b"test_cat");
        let point: Vec<GoldilocksExt2> = (0..n_y)
            .map(|_| transcript.challenge_ext2(b"ch"))
            .collect();
        let eval = y.data.as_ref().unwrap().evaluate_at_point_ext2(&point);
        let out_claim = Claim {
            edge_id: 2,
            sparse_id: 0,
            point,
            eval,
        };

        // Prove
        let mut prove_transcript = Transcript::new(b"test_cat_prove");
        let (proofs, claims) = cat.prove(
            &[&a, &b, y],
            &[0, 1, 2],
            &[&out_claim],
            &mut prove_transcript,
        );
        assert_eq!(proofs.len(), 0);
        assert_eq!(claims.len(), 2);

        // Check evals match
        let a_eval = a.data.as_ref().unwrap().evaluate_at_point_ext2(&claims[0].point);
        assert_eq!(claims[0].eval, a_eval, "A claim eval should match");
        let b_eval = b.data.as_ref().unwrap().evaluate_at_point_ext2(&claims[1].point);
        assert_eq!(claims[1].eval, b_eval, "B claim eval should match");

        // Verify
        let mut verify_transcript = Transcript::new(b"test_cat_prove");
        let mut all_claims: Vec<&Claim> = claims.iter().collect();
        all_claims.push(&out_claim);
        let proofs_ref: Vec<&SumcheckProof> = proofs.iter().collect();
        let verified = cat.verify(
            &[&a, &b, y],
            &all_claims,
            &proofs_ref,
            &mut verify_transcript,
        );
        assert!(verified, "Concat prove/verify should pass");
    }

    #[test]
    fn test_channel_slice_run() {
        // X[4, 4]: 4 channels, 4-element spatial
        let cs = ChannelSlice {
            channels_in: 4,
            channels_out: 2,
            channel_start: 2,
            spatial_dims: vec![4],
        };
        let x = make_witness(vec![4, 4], vec![
            1,2,3,4,  // c=0
            5,6,7,8,  // c=1
            10,20,30,40,  // c=2
            50,60,70,80,  // c=3
        ]);
        let result = cs.run(&[&x]);
        let y = &result[0];
        assert_eq!(y.shape, vec![2, 4]);
        // Y[0] = X[2] = [10,20,30,40]
        assert_eq!(y.data.as_ref().unwrap().index(0), GoldilocksField(10));
        assert_eq!(y.data.as_ref().unwrap().index(1), GoldilocksField(20));
        // Y[1] = X[3] = [50,60,70,80]
        assert_eq!(y.data.as_ref().unwrap().index(4), GoldilocksField(50));
        assert_eq!(y.data.as_ref().unwrap().index(5), GoldilocksField(60));
    }

    #[test]
    fn test_channel_slice_prove_verify() {
        let cs = ChannelSlice {
            channels_in: 4,
            channels_out: 2,
            channel_start: 0,
            spatial_dims: vec![4],
        };
        let x = make_witness(vec![4, 4], vec![
            1,2,3,4,
            5,6,7,8,
            10,20,30,40,
            50,60,70,80,
        ]);
        let result = cs.run(&[&x]);
        let y = &result[0];

        let n_y = y.data.as_ref().unwrap().n();
        let mut transcript = Transcript::new(b"test_cs");
        let point: Vec<GoldilocksExt2> = (0..n_y)
            .map(|_| transcript.challenge_ext2(b"ch"))
            .collect();
        let eval = y.data.as_ref().unwrap().evaluate_at_point_ext2(&point);
        let out_claim = Claim {
            edge_id: 1,
            sparse_id: 0,
            point,
            eval,
        };

        let mut prove_transcript = Transcript::new(b"test_cs_prove");
        let (proofs, claims) = cs.prove(
            &[&x, y],
            &[0, 1],
            &[&out_claim],
            &mut prove_transcript,
        );
        assert_eq!(proofs.len(), 0);
        assert_eq!(claims.len(), 1);

        // Verify
        let mut verify_transcript = Transcript::new(b"test_cs_prove");
        let mut all_claims: Vec<&Claim> = claims.iter().collect();
        all_claims.push(&out_claim);
        let proofs_ref: Vec<&SumcheckProof> = proofs.iter().collect();
        let verified = cs.verify(
            &[&x, y],
            &all_claims,
            &proofs_ref,
            &mut verify_transcript,
        );
        assert!(verified, "ChannelSlice prove/verify should pass");
    }

    #[test]
    fn test_channel_slice_upper_half() {
        // Test slicing the upper half of channels
        let cs = ChannelSlice {
            channels_in: 4,
            channels_out: 2,
            channel_start: 2,
            spatial_dims: vec![4],
        };
        let x = make_witness(vec![4, 4], vec![
            1,2,3,4,
            5,6,7,8,
            10,20,30,40,
            50,60,70,80,
        ]);
        let result = cs.run(&[&x]);
        let y = &result[0];

        let n_y = y.data.as_ref().unwrap().n();
        let mut transcript = Transcript::new(b"test_cs2");
        let point: Vec<GoldilocksExt2> = (0..n_y)
            .map(|_| transcript.challenge_ext2(b"ch"))
            .collect();
        let eval = y.data.as_ref().unwrap().evaluate_at_point_ext2(&point);
        let out_claim = Claim {
            edge_id: 1,
            sparse_id: 0,
            point,
            eval,
        };

        let mut prove_transcript = Transcript::new(b"test_cs2_prove");
        let (proofs, claims) = cs.prove(
            &[&x, y],
            &[0, 1],
            &[&out_claim],
            &mut prove_transcript,
        );

        let mut verify_transcript = Transcript::new(b"test_cs2_prove");
        let mut all_claims: Vec<&Claim> = claims.iter().collect();
        all_claims.push(&out_claim);
        let proofs_ref: Vec<&SumcheckProof> = proofs.iter().collect();
        let verified = cs.verify(
            &[&x, y],
            &all_claims,
            &proofs_ref,
            &mut verify_transcript,
        );
        assert!(verified, "ChannelSlice upper half prove/verify should pass");
    }
}
