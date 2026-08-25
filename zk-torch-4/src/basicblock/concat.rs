//! [`Concat`] (equal-size channel concat) and [`ChannelSlice`] (contiguous
//! channel slice). Both are zero-sumcheck claim transforms — verifier sees
//! `Y(r) = (1−r_top)·A(r_low) + r_top·B(r_low)` for Concat, and a direct
//! point-extension for ChannelSlice.

use almost_goldilocks_cuda::extension::AlmostGoldilocksExt2;
use almost_goldilocks_cuda::field::AlmostGoldilocksField;

use crate::basicblock::BasicBlock;
use crate::dag::{Claim, DataType, Role, Witness};
use crate::sumcheck::SumcheckProof;
use crate::transcript::Transcript;
use crate::util::arith::{ext2_add, ext2_field_eq, ext2_mul, ext2_sub, get_n, log2_ceil};

// ============================================================================
// Concat
// ============================================================================

#[derive(Clone, Debug)]
pub struct Concat {
    pub channels_a: usize,
    pub spatial_dims: Vec<usize>,
}

impl BasicBlock for Concat {
    fn run(&self, inputs: &[&Witness]) -> Vec<Witness> {
        assert_eq!(inputs.len(), 2, "Concat expects 2 inputs");
        let a = inputs[0];
        let b = inputs[1];

        let c_a = self.channels_a;
        let c_out = 2 * c_a;
        let spatial_pads: Vec<usize> =
            self.spatial_dims.iter().map(|&s| s.next_power_of_two()).collect();
        let c_a_pad = c_a.next_power_of_two();
        let c_out_pad = c_out.next_power_of_two();
        let spatial_size: usize = spatial_pads.iter().product();
        let out_size = c_out_pad * spatial_size;
        let in_size_a = c_a_pad * spatial_size;

        let mut out_data = vec![AlmostGoldilocksField(0); out_size];
        let a_data = a.data.as_ref().unwrap();
        let b_data = b.data.as_ref().unwrap();
        // Channels 0..c_a from A, c_a..2c_a from B (little-endian: spatial bits
        // are low, channel bits are high).
        for c in 0..c_a {
            for s in 0..spatial_size {
                let idx = s + c * spatial_size;
                if idx < in_size_a {
                    out_data[idx] = a_data.index(idx);
                }
            }
        }
        for c in 0..c_a {
            for s in 0..spatial_size {
                let b_idx = s + c * spatial_size;
                let out_idx = s + (c + c_a) * spatial_size;
                if b_idx < in_size_a {
                    out_data[out_idx] = b_data.index(b_idx);
                }
            }
        }

        let mut out_shape = vec![c_out];
        out_shape.extend_from_slice(&self.spatial_dims);
        let n = get_n(&out_shape);
        if out_data.len() < (1 << n) {
            out_data.resize(1 << n, AlmostGoldilocksField(0));
        }
        vec![Witness::new(out_shape, out_data, DataType::Uint, inputs[0].sf, Role::Output)]
    }

    /// Pure index-rearrangement work — bandwidth-bound, no algebraic
    /// shortcut. CPU is the documented default.
    fn run_gpu(&self, inputs: &[&Witness]) -> Vec<Witness> {
        self.run(inputs)
    }

    fn prove(
        &self,
        witnesses: &[&Witness],
        edge_ids: &[usize],
        out_claims: &[&Claim],
        _transcript: &mut Transcript,
    ) -> (Vec<SumcheckProof>, Vec<Claim>) {
        let out_claim = out_claims[0];
        let c_out = 2 * self.channels_a;
        let l_c_out = log2_ceil(c_out.max(1));
        let l_c_a = log2_ceil(self.channels_a.max(1));
        let l_spatial: usize =
            self.spatial_dims.iter().map(|&s| log2_ceil(s.max(1))).sum();

        // Output claim point layout (little-endian): spatial_bits ++ c_bits.
        // c_bits[l_c_out-1] (the MSB) selects A vs B.
        let r_c_top = out_claim.point[l_spatial + l_c_out - 1];

        // Input point: spatial_bits ++ lower c_a bits.
        let mut point_ab = Vec::with_capacity(l_spatial + l_c_a);
        point_ab.extend_from_slice(&out_claim.point[..l_spatial]);
        point_ab.extend_from_slice(&out_claim.point[l_spatial..l_spatial + l_c_a]);

        let a_eval = witnesses[0]
            .data
            .as_ref()
            .unwrap()
            .evaluate_at_point_ext2(&point_ab);
        let b_eval = witnesses[1]
            .data
            .as_ref()
            .unwrap()
            .evaluate_at_point_ext2(&point_ab);

        // Sanity for the prover: (1 - r_top)·a + r_top·b should equal the
        // claimed output eval. Not enforced as a panic — the verifier checks
        // it the same way.
        let _expected_y =
            ext2_add(ext2_mul(ext2_sub(AlmostGoldilocksExt2::one(), r_c_top), a_eval),
                     ext2_mul(r_c_top, b_eval));

        let a_claim = Claim {
            edge_id: edge_ids[0],
            sparse_id: 0,
            point: point_ab.clone(),
            eval: a_eval,
        };
        let b_claim = Claim {
            edge_id: edge_ids[1],
            sparse_id: 0,
            point: point_ab,
            eval: b_eval,
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
        // claims = [a, b, out]
        let out_claim = claims.last().unwrap();
        let a_claim = &claims[0];
        let b_claim = &claims[1];

        let c_out = 2 * self.channels_a;
        let l_c_out = log2_ceil(c_out.max(1));
        let l_c_a = log2_ceil(self.channels_a.max(1));
        let l_spatial: usize =
            self.spatial_dims.iter().map(|&s| log2_ceil(s.max(1))).sum();

        let r_c_top = out_claim.point[l_spatial + l_c_out - 1];
        let one_minus_top = ext2_sub(AlmostGoldilocksExt2::one(), r_c_top);
        let expected_out_eval =
            ext2_add(ext2_mul(one_minus_top, a_claim.eval), ext2_mul(r_c_top, b_claim.eval));
        if !ext2_field_eq(expected_out_eval, out_claim.eval) {
            return false;
        }

        let expected_len = l_spatial + l_c_a;
        if a_claim.point.len() != expected_len || b_claim.point.len() != expected_len {
            return false;
        }
        if a_claim.point[..l_spatial] != out_claim.point[..l_spatial] {
            return false;
        }
        if a_claim.point[l_spatial..]
            != out_claim.point[l_spatial..l_spatial + l_c_a]
        {
            return false;
        }
        if a_claim.point != b_claim.point {
            return false;
        }
        true
    }
}

// ============================================================================
// ChannelSlice
// ============================================================================

#[derive(Clone, Debug)]
pub struct ChannelSlice {
    pub channels_in: usize,
    pub channels_out: usize,
    pub channel_start: usize,
    pub spatial_dims: Vec<usize>,
}

impl BasicBlock for ChannelSlice {
    fn run(&self, inputs: &[&Witness]) -> Vec<Witness> {
        assert_eq!(inputs.len(), 1, "ChannelSlice expects 1 input");
        let x = inputs[0];
        let spatial_pads: Vec<usize> =
            self.spatial_dims.iter().map(|&s| s.next_power_of_two()).collect();
        let spatial_size: usize = spatial_pads.iter().product();
        let c_out_pad = self.channels_out.next_power_of_two();
        let out_size = c_out_pad * spatial_size;
        let mut out_data = vec![AlmostGoldilocksField(0); out_size];
        let x_data = x.data.as_ref().unwrap();
        for c in 0..self.channels_out {
            let c_in = c + self.channel_start;
            for s in 0..spatial_size {
                let x_idx = s + c_in * spatial_size;
                let o_idx = s + c * spatial_size;
                if c_in < self.channels_in {
                    out_data[o_idx] = x_data.index(x_idx);
                }
            }
        }
        let mut out_shape = vec![self.channels_out];
        out_shape.extend_from_slice(&self.spatial_dims);
        let n = get_n(&out_shape);
        if out_data.len() < (1 << n) {
            out_data.resize(1 << n, AlmostGoldilocksField(0));
        }
        vec![Witness::new(out_shape, out_data, DataType::Uint, inputs[0].sf, Role::Output)]
    }

    fn run_gpu(&self, inputs: &[&Witness]) -> Vec<Witness> {
        self.run(inputs)
    }

    fn prove(
        &self,
        witnesses: &[&Witness],
        edge_ids: &[usize],
        out_claims: &[&Claim],
        _transcript: &mut Transcript,
    ) -> (Vec<SumcheckProof>, Vec<Claim>) {
        let out_claim = out_claims[0];
        let l_c_out = log2_ceil(self.channels_out.max(1));
        let l_c_in = log2_ceil(self.channels_in.max(1));
        let l_spatial: usize =
            self.spatial_dims.iter().map(|&s| log2_ceil(s.max(1))).sum();

        // X point = spatial_bits ++ output channel bits ++ selector bits.
        let mut x_point = Vec::with_capacity(l_spatial + l_c_in);
        x_point.extend_from_slice(&out_claim.point[..l_spatial]);
        x_point.extend_from_slice(&out_claim.point[l_spatial..l_spatial + l_c_out]);
        let selector = self.channel_start / self.channels_out.max(1);
        let num_selector_bits = l_c_in.saturating_sub(l_c_out);
        for bit in 0..num_selector_bits {
            let bit_val = (selector >> bit) & 1;
            x_point.push(AlmostGoldilocksExt2::from_base(AlmostGoldilocksField(bit_val as u64)));
        }
        let x_eval = witnesses[0]
            .data
            .as_ref()
            .unwrap()
            .evaluate_at_point_ext2(&x_point);
        let x_claim = Claim {
            edge_id: edge_ids[0],
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
        // claims = [x, out]
        let out_claim = claims.last().unwrap();
        let x_claim = &claims[0];
        let l_c_out = log2_ceil(self.channels_out.max(1));
        let l_c_in = log2_ceil(self.channels_in.max(1));
        let l_spatial: usize =
            self.spatial_dims.iter().map(|&s| log2_ceil(s.max(1))).sum();
        if x_claim.point.len() != l_spatial + l_c_in {
            return false;
        }
        if x_claim.point[..l_spatial] != out_claim.point[..l_spatial] {
            return false;
        }
        if x_claim.point[l_spatial..l_spatial + l_c_out]
            != out_claim.point[l_spatial..l_spatial + l_c_out]
        {
            return false;
        }
        let selector = self.channel_start / self.channels_out.max(1);
        let num_selector_bits = l_c_in.saturating_sub(l_c_out);
        for bit in 0..num_selector_bits {
            let bit_val = (selector >> bit) & 1;
            let expected = AlmostGoldilocksExt2::from_base(AlmostGoldilocksField(bit_val as u64));
            if x_claim.point[l_spatial + l_c_out + bit] != expected {
                return false;
            }
        }
        ext2_field_eq(x_claim.eval, out_claim.eval)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn agl(v: u64) -> AlmostGoldilocksField {
        AlmostGoldilocksField(v)
    }

    fn make_witness(shape: Vec<usize>, vals: Vec<u64>) -> Witness {
        Witness::new(
            shape,
            vals.into_iter().map(agl).collect(),
            DataType::Uint,
            0,
            Role::Input,
        )
    }

    #[test]
    fn concat_run_stitches_channels() {
        // A[2, 4], B[2, 4] → Y[4, 4].
        let cat = Concat { channels_a: 2, spatial_dims: vec![4] };
        let a = make_witness(vec![2, 4], vec![1, 2, 3, 4, 5, 6, 7, 8]);
        let b = make_witness(
            vec![2, 4],
            vec![10, 20, 30, 40, 50, 60, 70, 80],
        );
        let y = cat.run(&[&a, &b]);
        let evals = y[0].data.as_ref().unwrap().evaluations_ref();
        // Channel 0: A row 0 → [1,2,3,4]
        assert_eq!(evals[0..4], [agl(1), agl(2), agl(3), agl(4)]);
        // Channel 1: A row 1 → [5,6,7,8]
        assert_eq!(evals[4..8], [agl(5), agl(6), agl(7), agl(8)]);
        // Channel 2: B row 0
        assert_eq!(evals[8..12], [agl(10), agl(20), agl(30), agl(40)]);
        // Channel 3: B row 1
        assert_eq!(evals[12..16], [agl(50), agl(60), agl(70), agl(80)]);
    }

    #[test]
    fn concat_prove_verify_roundtrip() {
        let cat = Concat { channels_a: 2, spatial_dims: vec![4] };
        let a = make_witness(vec![2, 4], vec![1, 2, 3, 4, 5, 6, 7, 8]);
        let b = make_witness(
            vec![2, 4],
            vec![10, 20, 30, 40, 50, 60, 70, 80],
        );
        let outs = cat.run(&[&a, &b]);
        let y = &outs[0];
        let n_y = y.data.as_ref().unwrap().n();

        let mut t_in = Transcript::new(b"cat");
        let point: Vec<_> = (0..n_y).map(|_| t_in.challenge_ext2(b"r")).collect();
        let eval = y.data.as_ref().unwrap().evaluate_at_point_ext2(&point);
        let out_claim = Claim { edge_id: 2, sparse_id: 0, point, eval };

        let mut t_prove = Transcript::new(b"cat-prove");
        let (proofs, claims) =
            cat.prove(&[&a, &b, y], &[0, 1, 2], &[&out_claim], &mut t_prove);
        assert!(proofs.is_empty());
        assert_eq!(claims.len(), 2);

        // A's claim and B's claim are at the same input point but with their
        // own evals.
        let a_direct = a.data.as_ref().unwrap().evaluate_at_point_ext2(&claims[0].point);
        let b_direct = b.data.as_ref().unwrap().evaluate_at_point_ext2(&claims[1].point);
        assert!(ext2_field_eq(claims[0].eval, a_direct));
        assert!(ext2_field_eq(claims[1].eval, b_direct));

        let mut t_verify = Transcript::new(b"cat-prove");
        let all = [&claims[0], &claims[1], &out_claim];
        assert!(cat.verify(&[&a, &b, y], &all, &[], &mut t_verify));
    }

    #[test]
    fn channel_slice_run_extracts_correct_range() {
        let cs = ChannelSlice {
            channels_in: 4,
            channels_out: 2,
            channel_start: 2,
            spatial_dims: vec![4],
        };
        let x = make_witness(
            vec![4, 4],
            vec![1, 2, 3, 4, 5, 6, 7, 8, 10, 20, 30, 40, 50, 60, 70, 80],
        );
        let y = cs.run(&[&x]);
        let evals = y[0].data.as_ref().unwrap().evaluations_ref();
        // Channels 2..4 → [10,20,30,40, 50,60,70,80].
        assert_eq!(evals[0..4], [agl(10), agl(20), agl(30), agl(40)]);
        assert_eq!(evals[4..8], [agl(50), agl(60), agl(70), agl(80)]);
    }

    #[test]
    fn channel_slice_prove_verify_lower_half() {
        let cs = ChannelSlice {
            channels_in: 4,
            channels_out: 2,
            channel_start: 0,
            spatial_dims: vec![4],
        };
        let x = make_witness(
            vec![4, 4],
            vec![1, 2, 3, 4, 5, 6, 7, 8, 10, 20, 30, 40, 50, 60, 70, 80],
        );
        let outs = cs.run(&[&x]);
        let y = &outs[0];
        let n_y = y.data.as_ref().unwrap().n();
        let mut t_in = Transcript::new(b"cs");
        let point: Vec<_> = (0..n_y).map(|_| t_in.challenge_ext2(b"r")).collect();
        let eval = y.data.as_ref().unwrap().evaluate_at_point_ext2(&point);
        let out_claim = Claim { edge_id: 1, sparse_id: 0, point, eval };

        let mut t_prove = Transcript::new(b"cs-prove");
        let (_, claims) = cs.prove(&[&x, y], &[0, 1], &[&out_claim], &mut t_prove);
        let mut t_verify = Transcript::new(b"cs-prove");
        let all = [&claims[0], &out_claim];
        assert!(cs.verify(&[&x, y], &all, &[], &mut t_verify));
    }

    #[test]
    fn channel_slice_prove_verify_upper_half() {
        let cs = ChannelSlice {
            channels_in: 4,
            channels_out: 2,
            channel_start: 2,
            spatial_dims: vec![4],
        };
        let x = make_witness(
            vec![4, 4],
            vec![1, 2, 3, 4, 5, 6, 7, 8, 10, 20, 30, 40, 50, 60, 70, 80],
        );
        let outs = cs.run(&[&x]);
        let y = &outs[0];
        let n_y = y.data.as_ref().unwrap().n();
        let mut t_in = Transcript::new(b"cs2");
        let point: Vec<_> = (0..n_y).map(|_| t_in.challenge_ext2(b"r")).collect();
        let eval = y.data.as_ref().unwrap().evaluate_at_point_ext2(&point);
        let out_claim = Claim { edge_id: 1, sparse_id: 0, point, eval };
        let mut t_prove = Transcript::new(b"cs2-prove");
        let (_, claims) = cs.prove(&[&x, y], &[0, 1], &[&out_claim], &mut t_prove);
        let mut t_verify = Transcript::new(b"cs2-prove");
        let all = [&claims[0], &out_claim];
        assert!(cs.verify(&[&x, y], &all, &[], &mut t_verify));
    }
}
