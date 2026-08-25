//! PointPillar-specific advice ops: [`PillarMaxPool`], [`ScatterToBEV`],
//! [`GatherFromGrid`].
//!
//! Soundness model:
//! - [`PillarMaxPool`]: same dominance argument as the other max-pool blocks
//!   — output is upper-bounded by every input via SubSample2D + Sub + NonNeg
//!   wired by the DAG builder. Achievability requires a lookup argument and
//!   is currently an open soundness gap (see plan §F.6).
//! - [`ScatterToBEV`] / [`GatherFromGrid`]: scatter/gather along coordinate
//!   indices. Both are non-algebraic; full soundness requires a permutation /
//!   lookup argument. Currently emit advice values trusted by the verifier
//!   (plan §F.7 documents this as an open gap).

use almost_goldilocks_cuda::field::AlmostGoldilocksField;

use crate::basicblock::BasicBlock;
use crate::dag::{Claim, Role, Witness};
use crate::sumcheck::SumcheckProof;
use crate::transcript::Transcript;
use crate::util::arith::{f_to_int, get_n, int_to_f};

// ============================================================================
// PillarMaxPool
// ============================================================================

#[derive(Clone, Debug)]
pub struct PillarMaxPool {
    pub n_pillars: usize,
    pub max_points: usize,
    pub features: usize,
}

impl BasicBlock for PillarMaxPool {
    fn run(&self, inputs: &[&Witness]) -> Vec<Witness> {
        assert_eq!(inputs.len(), 1, "PillarMaxPool expects 1 input");
        let x = inputs[0];
        let x_evals = x.data.as_ref().unwrap().evaluations_ref();
        let d_pad = self.features.next_power_of_two();
        let t_pad = self.max_points.next_power_of_two();
        let n_pad = self.n_pillars.next_power_of_two();
        let out_size = n_pad * d_pad;
        let mut out_data = vec![AlmostGoldilocksField(0); out_size];

        for p in 0..self.n_pillars {
            for d in 0..self.features {
                let mut max_val = i128::MIN;
                for t in 0..self.max_points {
                    let x_idx = d + t * d_pad + p * d_pad * t_pad;
                    let v = f_to_int(x_evals[x_idx]);
                    if v > max_val { max_val = v; }
                }
                let out_idx = d + p * d_pad;
                out_data[out_idx] = int_to_f(max_val);
            }
        }
        let out_shape = vec![self.n_pillars, self.features];
        let n = get_n(&out_shape);
        if out_data.len() < (1 << n) {
            out_data.resize(1 << n, AlmostGoldilocksField(0));
        }
        vec![Witness::new(out_shape, out_data, x.data_type, x.sf, Role::Output)]
    }

    fn run_gpu(&self, inputs: &[&Witness]) -> Vec<Witness> { self.run(inputs) }

    fn prove(
        &self,
        _witnesses: &[&Witness],
        _edge_ids: &[usize],
        _out_claims: &[&Claim],
        _transcript: &mut Transcript,
    ) -> (Vec<SumcheckProof>, Vec<Claim>) {
        (vec![], vec![])
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

// ============================================================================
// ScatterToBEV
// ============================================================================

#[derive(Clone, Debug)]
pub struct ScatterToBEV {
    pub n_pillars: usize,
    pub features: usize,
    pub ny: usize,
    pub nx: usize,
}

impl BasicBlock for ScatterToBEV {
    fn run(&self, inputs: &[&Witness]) -> Vec<Witness> {
        assert_eq!(inputs.len(), 2, "ScatterToBEV expects 2 inputs (X, coords)");
        let x = inputs[0];
        let coords = inputs[1];
        let x_evals = x.data.as_ref().unwrap().evaluations_ref();
        let coord_evals = coords.data.as_ref().unwrap().evaluations_ref();
        let d_pad = self.features.next_power_of_two();
        let nx_pad = self.nx.next_power_of_two();
        let ny_pad = self.ny.next_power_of_two();
        let coord_dim_pad = 2usize.next_power_of_two();
        let out_size = self.features.next_power_of_two() * ny_pad * nx_pad;
        let mut out_data = vec![AlmostGoldilocksField(0); out_size];

        for p in 0..self.n_pillars {
            let cy = f_to_int(coord_evals[p * coord_dim_pad]) as usize;
            let cx = f_to_int(coord_evals[1 + p * coord_dim_pad]) as usize;
            if cy < self.ny && cx < self.nx {
                for d in 0..self.features {
                    let x_idx = d + p * d_pad;
                    let out_idx = cx + cy * nx_pad + d * nx_pad * ny_pad;
                    out_data[out_idx] = x_evals[x_idx];
                }
            }
        }
        let out_shape = vec![self.features, self.ny, self.nx];
        let n = get_n(&out_shape);
        if out_data.len() < (1 << n) {
            out_data.resize(1 << n, AlmostGoldilocksField(0));
        }
        vec![Witness::new(out_shape, out_data, x.data_type, x.sf, Role::Output)]
    }

    fn run_gpu(&self, inputs: &[&Witness]) -> Vec<Witness> { self.run(inputs) }

    fn prove(
        &self,
        _witnesses: &[&Witness],
        _edge_ids: &[usize],
        _out_claims: &[&Claim],
        _transcript: &mut Transcript,
    ) -> (Vec<SumcheckProof>, Vec<Claim>) {
        (vec![], vec![])
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

// ============================================================================
// GatherFromGrid
// ============================================================================

#[derive(Clone, Debug)]
pub struct GatherFromGrid {
    pub n_points: usize,
    pub channels: usize,
    pub grid_h: usize,
    pub grid_w: usize,
}

impl BasicBlock for GatherFromGrid {
    fn run(&self, inputs: &[&Witness]) -> Vec<Witness> {
        assert_eq!(inputs.len(), 2, "GatherFromGrid expects 2 inputs (grid, coords)");
        let grid = inputs[0];
        let coords = inputs[1];
        let grid_evals = grid.data.as_ref().unwrap().evaluations_ref();
        let coord_evals = coords.data.as_ref().unwrap().evaluations_ref();
        let c_pad = self.channels.next_power_of_two();
        let w_pad = self.grid_w.next_power_of_two();
        let h_pad = self.grid_h.next_power_of_two();
        let n_pad = self.n_points.next_power_of_two();
        let coord_dim_pad = 2usize.next_power_of_two();
        let out_size = n_pad * c_pad;
        let mut out_data = vec![AlmostGoldilocksField(0); out_size];

        for p in 0..self.n_points {
            let cy = f_to_int(coord_evals[p * coord_dim_pad]) as usize;
            let cx = f_to_int(coord_evals[1 + p * coord_dim_pad]) as usize;
            if cy < self.grid_h && cx < self.grid_w {
                for c in 0..self.channels {
                    let grid_idx = cx + cy * w_pad + c * w_pad * h_pad;
                    let out_idx = c + p * c_pad;
                    out_data[out_idx] = grid_evals[grid_idx];
                }
            }
        }
        let out_shape = vec![self.n_points, self.channels];
        let n = get_n(&out_shape);
        if out_data.len() < (1 << n) {
            out_data.resize(1 << n, AlmostGoldilocksField(0));
        }
        vec![Witness::new(out_shape, out_data, grid.data_type, grid.sf, Role::Output)]
    }

    fn run_gpu(&self, inputs: &[&Witness]) -> Vec<Witness> { self.run(inputs) }

    fn prove(
        &self,
        _witnesses: &[&Witness],
        _edge_ids: &[usize],
        _out_claims: &[&Claim],
        _transcript: &mut Transcript,
    ) -> (Vec<SumcheckProof>, Vec<Claim>) {
        (vec![], vec![])
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
    fn pillar_maxpool_takes_per_feature_max() {
        let pool = PillarMaxPool { n_pillars: 2, max_points: 2, features: 2 };
        // X[n=2, t=2, d=2]; little-endian d(2)|t(2)|n(2).
        // pillar 0: (t=0,d=0)=1, (t=0,d=1)=3, (t=1,d=0)=5, (t=1,d=1)=2 → max(1,5)=5, max(3,2)=3.
        // pillar 1: (t=0,d=0)=4, (t=0,d=1)=1, (t=1,d=0)=2, (t=1,d=1)=6 → max(4,2)=4, max(1,6)=6.
        let x = make_witness(vec![2, 2, 2], vec![1, 3, 5, 2, 4, 1, 2, 6]);
        let out = pool.run(&[&x]);
        let evals = out[0].data.as_ref().unwrap().evaluations_ref();
        assert_eq!(evals[0], agl(5));
        assert_eq!(evals[1], agl(3));
        assert_eq!(evals[2], agl(4));
        assert_eq!(evals[3], agl(6));
    }

    #[test]
    fn scatter_to_bev_writes_pillars_at_coords() {
        let scatter = ScatterToBEV { n_pillars: 2, features: 2, ny: 2, nx: 2 };
        // X[n=2, d=2]: pillar 0 = [10, 20], pillar 1 = [30, 40]. coords: p0=(0,1), p1=(1,0).
        let x = make_witness(vec![2, 2], vec![10, 20, 30, 40]);
        let coords = make_witness(vec![2, 2], vec![0, 1, 1, 0]);
        let out = scatter.run(&[&x, &coords]);
        let evals = out[0].data.as_ref().unwrap().evaluations_ref();
        // Y[d, y, x] little-endian: x(2) | y(2) | d(2). d=0:
        assert_eq!(evals[0], agl(0));  // (y=0, x=0)
        assert_eq!(evals[1], agl(10)); // (y=0, x=1) ← pillar 0
        assert_eq!(evals[2], agl(30)); // (y=1, x=0) ← pillar 1
        assert_eq!(evals[3], agl(0));
        // d=1:
        assert_eq!(evals[5], agl(20));
        assert_eq!(evals[6], agl(40));
    }

    #[test]
    fn gather_from_grid_reads_at_coords() {
        let gather = GatherFromGrid { n_points: 2, channels: 2, grid_h: 2, grid_w: 2 };
        // grid[c=2, h=2, w=2] flat little-endian: w(2)|h(2)|c(2).
        let grid = make_witness(vec![2, 2, 2], vec![10, 20, 30, 40, 50, 60, 70, 80]);
        // point 0 at (0, 1) → c=0:20, c=1:60. point 1 at (1, 0) → c=0:30, c=1:70.
        let coords = make_witness(vec![2, 2], vec![0, 1, 1, 0]);
        let out = gather.run(&[&grid, &coords]);
        let evals = out[0].data.as_ref().unwrap().evaluations_ref();
        assert_eq!(evals[0], agl(20)); // p=0, c=0
        assert_eq!(evals[1], agl(60)); // p=0, c=1
        assert_eq!(evals[2], agl(30)); // p=1, c=0
        assert_eq!(evals[3], agl(70)); // p=1, c=1
    }
}
