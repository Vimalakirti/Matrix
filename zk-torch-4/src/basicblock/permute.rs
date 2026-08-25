//! [`Permute`] — variable reorder. Produces both the permuted output and an
//! auxiliary witness holding the original (unpermuted) data, so downstream
//! blocks can claim against either ordering.

use std::sync::Arc;

use almost_goldilocks_cuda::field::{AlmostGoldilocksBatch, AlmostGoldilocksField};
use almost_goldilocks_cuda::memory::DeviceBuffer;

use crate::basicblock::BasicBlock;
use crate::dag::{Claim, Role, Witness};
use crate::sumcheck::SumcheckProof;
use crate::transcript::Transcript;

#[derive(Clone, Debug)]
pub struct Permute {
    pub ranges: Vec<(usize, usize)>,
}

impl Permute {
    pub fn new(ranges: Vec<(usize, usize)>) -> Self {
        Self { ranges }
    }

    /// Build `perm_map[old_var] = new_var_pos` from the [`Self::ranges`]
    /// description.
    fn build_perm_map(&self, n: usize) -> Vec<i32> {
        let mut new_var_order = Vec::with_capacity(n);
        for &(start, end) in &self.ranges {
            for v in start..end {
                new_var_order.push(v);
            }
        }
        assert_eq!(new_var_order.len(), n, "Permute ranges don't cover all {} vars", n);
        let mut perm_map = vec![0i32; n];
        for (new_pos, &old_var) in new_var_order.iter().enumerate() {
            perm_map[old_var] = new_pos as i32;
        }
        perm_map
    }
}

fn permute_evals(
    evals: &[AlmostGoldilocksField],
    n: usize,
    perm_map: &[i32],
) -> Vec<AlmostGoldilocksField> {
    let size = 1usize << n;
    assert_eq!(evals.len(), size);
    let mut out = vec![AlmostGoldilocksField(0); size];
    for idx_old in 0..size {
        let mut idx_new = 0usize;
        for old_var in 0..n {
            if idx_old & (1 << old_var) != 0 {
                idx_new |= 1 << perm_map[old_var];
            }
        }
        out[idx_new] = evals[idx_old];
    }
    out
}

impl BasicBlock for Permute {
    fn run(&self, inputs: &[&Witness]) -> Vec<Witness> {
        assert_eq!(inputs.len(), 1, "Permute expects 1 input");
        let input = inputs[0];
        let data = input.data.as_ref().expect("Permute: input has no data");
        let n = data.n();
        let evals = data.evaluations_ref();
        let perm_map = self.build_perm_map(n);
        let permuted = permute_evals(evals, n, &perm_map);
        vec![
            Witness::new(input.shape.clone(), permuted, input.data_type, input.sf, Role::Output),
            Witness::new(input.shape.clone(), evals.to_vec(), input.data_type, input.sf, Role::Auxiliary),
        ]
    }

    /// GPU path uses `AlmostGoldilocksBatch::bit_permute`. Auxiliary
    /// witness shares the input device buffer via `Arc` — no extra HBM
    /// traffic. Identity permutation is a fast-path alias.
    fn run_gpu(&self, inputs: &[&Witness]) -> Vec<Witness> {
        assert_eq!(inputs.len(), 1, "Permute expects 1 input");
        let input = inputs[0];
        let n = input.data.as_ref().unwrap().n();
        let size = 1usize << n;
        let perm_map = self.build_perm_map(n);
        let is_identity = perm_map.iter().enumerate().all(|(i, &v)| v == i as i32);
        let d_in = input.as_device_buf();
        let d_out_arc = if is_identity {
            Arc::clone(&d_in)
        } else {
            let mut d_out =
                DeviceBuffer::<u64>::new(size).expect("Permute: alloc out failed");
            AlmostGoldilocksBatch::bit_permute(&d_in, &mut d_out, &perm_map)
                .expect("Permute: GPU bit_permute failed");
            Arc::new(d_out)
        };
        vec![
            Witness::new_device(input.shape.clone(), d_out_arc, input.data_type, input.sf, Role::Output),
            Witness::new_device(input.shape.clone(), d_in, input.data_type, input.sf, Role::Auxiliary),
        ]
    }

    /// Claim transform: the output's evaluation at point `r` equals the
    /// input's evaluation at `perm(r)`. Since the verifier is permuting the
    /// challenge bits the same way, we just pass the claim through.
    fn prove(
        &self,
        _witnesses: &[&Witness],
        edge_ids: &[usize],
        out_claims: &[&Claim],
        _transcript: &mut Transcript,
    ) -> (Vec<SumcheckProof>, Vec<Claim>) {
        let claim = out_claims[0];
        let new_claim = Claim {
            edge_id: edge_ids[0],
            sparse_id: 0,
            point: claim.point.clone(),
            eval: claim.eval,
        };
        (vec![], vec![new_claim])
    }

    fn verify(
        &self,
        _witnesses: &[&Witness],
        _claims: &[&Claim],
        _sumcheck_proofs: &[&SumcheckProof],
        _transcript: &mut Transcript,
    ) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::DataType;

    fn agl(v: u64) -> AlmostGoldilocksField {
        AlmostGoldilocksField(v)
    }

    fn build_input(shape: Vec<usize>, vals: Vec<u64>) -> Witness {
        Witness::new(
            shape,
            vals.into_iter().map(agl).collect(),
            DataType::Int,
            0,
            Role::Input,
        )
    }

    #[test]
    fn permute_identity_passes_through() {
        // ranges = [(0, 3)] = identity for n=3.
        let p = Permute::new(vec![(0, 3)]);
        let x = build_input(vec![8], (0..8u64).collect());
        let out = p.run(&[&x]);
        assert_eq!(out.len(), 2);
        let permuted = out[0].data.as_ref().unwrap().evaluations_ref();
        let aux = out[1].data.as_ref().unwrap().evaluations_ref();
        for i in 0..8 {
            assert_eq!(permuted[i].0, i as u64);
            assert_eq!(aux[i].0, i as u64);
        }
    }

    /// ranges = [(2, 3), (0, 2)] cycles var2 → pos0, var0 → pos1, var1 → pos2.
    /// Old index `i = b0 + 2·b1 + 4·b2` → new index `b2 + 2·b0 + 4·b1`.
    #[test]
    fn permute_simple_3var_cycle() {
        let p = Permute::new(vec![(2, 3), (0, 2)]);
        // perm_map[old_var] = new_pos:
        //   build_perm_map collects new_var_order = [2, 0, 1] → perm_map[2]=0,
        //   perm_map[0]=1, perm_map[1]=2.
        let perm_map = p.build_perm_map(3);
        assert_eq!(perm_map, vec![1, 2, 0]);

        // For each old index, compute expected new index manually.
        let x = build_input(vec![8], (0..8u64).collect());
        let out = p.run(&[&x]);
        let permuted = out[0].data.as_ref().unwrap().evaluations_ref();
        for idx_old in 0..8u64 {
            let b0 = (idx_old >> 0) & 1;
            let b1 = (idx_old >> 1) & 1;
            let b2 = (idx_old >> 2) & 1;
            // new bits: b0 → pos1, b1 → pos2, b2 → pos0.
            let idx_new = (b2 << 0) | (b0 << 1) | (b1 << 2);
            assert_eq!(permuted[idx_new as usize].0, idx_old);
        }
    }

    #[test]
    fn permute_aux_holds_unpermuted_input() {
        let p = Permute::new(vec![(2, 3), (0, 2)]);
        let x = build_input(vec![8], (10..18u64).collect());
        let out = p.run(&[&x]);
        let aux = out[1].data.as_ref().unwrap().evaluations_ref();
        let inp = x.data.as_ref().unwrap().evaluations_ref();
        assert_eq!(aux, inp);
    }

    #[test]
    fn permute_prove_passes_claim_through() {
        let p = Permute::new(vec![(2, 3), (0, 2)]);
        let pt = vec![
            almost_goldilocks_cuda::extension::AlmostGoldilocksExt2::from_base(agl(3)),
            almost_goldilocks_cuda::extension::AlmostGoldilocksExt2::from_base(agl(5)),
            almost_goldilocks_cuda::extension::AlmostGoldilocksExt2::from_base(agl(7)),
        ];
        let claim = Claim {
            edge_id: 1,
            sparse_id: 0,
            point: pt.clone(),
            eval: almost_goldilocks_cuda::extension::AlmostGoldilocksExt2::from_base(agl(42)),
        };
        let mut t = Transcript::new(b"perm");
        let (proofs, claims) = p.prove(&[], &[0], &[&claim], &mut t);
        assert!(proofs.is_empty());
        assert_eq!(claims[0].point, pt);
        assert_eq!(claims[0].eval, claim.eval);
    }

    // ---------- GPU ----------

    fn cuda_ready() -> bool {
        almost_goldilocks_cuda::init().is_ok()
    }

    #[test]
    fn permute_run_gpu_matches_cpu() {
        if !cuda_ready() {
            eprintln!("skipping GPU test: CUDA not available");
            return;
        }
        let p = Permute::new(vec![(2, 3), (0, 2)]);
        let x = build_input(vec![8], (1..9u64).collect());
        let cpu = p.run(&[&x]);
        let gpu = p.run_gpu(&[&x]);
        let cpu_perm = cpu[0].data.as_ref().unwrap().evaluations();
        let gpu_perm = gpu[0].data.as_ref().unwrap().evaluations();
        for i in 0..cpu_perm.len() {
            assert_eq!(cpu_perm[i].reduce(), gpu_perm[i].reduce(), "i = {}", i);
        }
    }

    #[test]
    fn permute_run_gpu_identity_aliases_buffer() {
        if !cuda_ready() {
            eprintln!("skipping GPU test: CUDA not available");
            return;
        }
        // n = 3 identity permutation.
        let p = Permute::new(vec![(0, 3)]);
        let raw: Vec<u64> = (0..8u64).collect();
        let buf = Arc::new(
            DeviceBuffer::<u64>::from_slice(&raw).expect("upload"),
        );
        let w = Witness::new_device(vec![8], Arc::clone(&buf), DataType::Int, 0, Role::Input);
        let out = p.run_gpu(&[&w]);
        // Output buffer pointer equals input buffer pointer.
        assert!(Arc::ptr_eq(&buf, &out[0].device_buf().unwrap()));
    }
}
