use goldilocks_cuda::GoldilocksField;

use crate::basicblock::BasicBlock;
use crate::dag::{Claim, Role, Witness};
use crate::sumcheck::SumcheckProof;
use crate::transcript::Transcript;
use crate::util::arith::{f_to_int, get_n, int_to_f};

/// PillarMaxPool: max-pool over points dimension per feature per pillar.
/// Input:  X[N_pillars, max_points, D]
/// Output: Y[N_pillars, D]
///
/// For each pillar p and feature d:
///   Y[p, d] = max_{t=0..max_points} X[p, t, d]
///
/// This is an advice op — the prover asserts the output without proving
/// the max operation. Similar to InstanceNorm3D.
#[derive(Clone, Debug)]
pub struct PillarMaxPool {
    pub n_pillars: usize,
    pub max_points: usize,
    pub features: usize,
}

impl BasicBlock for PillarMaxPool {
    fn run(&self, inputs: &[&Witness]) -> Vec<Witness> {
        assert_eq!(inputs.len(), 1);
        let x = inputs[0];
        let x_evals = x.data.as_ref().unwrap().evaluations_ref();

        let d_pad = self.features.next_power_of_two();
        let t_pad = self.max_points.next_power_of_two();
        let n_pad = self.n_pillars.next_power_of_two();

        let out_d_pad = d_pad;
        let out_n_pad = n_pad;
        let out_size = out_n_pad * out_d_pad;
        let mut out_data = vec![GoldilocksField(0); out_size];

        for p in 0..self.n_pillars {
            for d in 0..self.features {
                let mut max_val = i128::MIN;
                for t in 0..self.max_points {
                    // X little-endian: d bits (lowest) | t bits | n bits
                    let x_idx = d + t * d_pad + p * d_pad * t_pad;
                    let val = f_to_int(x_evals[x_idx]);
                    if val > max_val {
                        max_val = val;
                    }
                }
                // Y little-endian: d bits (lowest) | n bits
                let out_idx = d + p * out_d_pad;
                out_data[out_idx] = int_to_f(max_val);
            }
        }

        let out_shape = vec![self.n_pillars, self.features];
        let n = get_n(&out_shape);
        if out_data.len() < (1 << n) {
            out_data.resize(1 << n, GoldilocksField(0));
        }
        vec![Witness::new(out_shape, out_data, x.data_type, x.sf, Role::Output)]
    }

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

/// ScatterToBEV: scatter pillar features to BEV grid.
/// Inputs: X[N_pillars, D], coords[N_pillars, 2]
/// Output: Y[D, ny, nx]
///
/// For each pillar p with coords (y, x):
///   Y[d, y, x] = X[p, d]
/// Empty cells are zero.
///
/// SOUNDNESS NOTE: This is an advice op with no proof constraints.
/// A full proof would require a lookup/permutation argument to verify
/// that each Y[d,y,x] either equals some X[p,d] where coords[p]=(y,x),
/// or is zero (empty cell). This is a non-algebraic operation that
/// requires future work to implement soundly.
/// TODO: Implement lookup-based scatter verification.
#[derive(Clone, Debug)]
pub struct ScatterToBEV {
    pub n_pillars: usize,
    pub features: usize,
    pub ny: usize,
    pub nx: usize,
}

impl BasicBlock for ScatterToBEV {
    fn run(&self, inputs: &[&Witness]) -> Vec<Witness> {
        assert_eq!(inputs.len(), 2);
        let x = inputs[0];      // [N_pillars, D]
        let coords = inputs[1]; // [N_pillars, 2]
        let x_evals = x.data.as_ref().unwrap().evaluations_ref();
        let coord_evals = coords.data.as_ref().unwrap().evaluations_ref();

        let d_pad = self.features.next_power_of_two();
        let _n_pad = self.n_pillars.next_power_of_two();
        let nx_pad = self.nx.next_power_of_two();
        let ny_pad = self.ny.next_power_of_two();
        let coord_dim_pad = 2usize.next_power_of_two();

        let out_size = self.features.next_power_of_two() * ny_pad * nx_pad;
        let mut out_data = vec![GoldilocksField(0); out_size];

        for p in 0..self.n_pillars {
            // coords little-endian: coord_dim bits (lowest) | n bits
            let cy = f_to_int(coord_evals[0 + p * coord_dim_pad]) as usize;
            let cx = f_to_int(coord_evals[1 + p * coord_dim_pad]) as usize;
            if cy < self.ny && cx < self.nx {
                for d in 0..self.features {
                    // X little-endian: d bits (lowest) | n bits
                    let x_idx = d + p * d_pad;
                    // Y little-endian: x bits (lowest) | y bits | d bits
                    let out_idx = cx + cy * nx_pad + d * nx_pad * ny_pad;
                    out_data[out_idx] = x_evals[x_idx];
                }
            }
        }

        let out_shape = vec![self.features, self.ny, self.nx];
        let n = get_n(&out_shape);
        if out_data.len() < (1 << n) {
            out_data.resize(1 << n, GoldilocksField(0));
        }
        vec![Witness::new(out_shape, out_data, x.data_type, x.sf, Role::Output)]
    }

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

/// GatherFromGrid: gather values from a spatial grid at given coordinates.
/// Inputs: grid[C, H, W], coords[N_points, 2] (y, x pixel coordinates)
/// Output: Y[N_points, C]
///
/// For each point p with coords (y, x):
///   Y[p, c] = grid[c, y, x]
/// Out-of-bounds coords yield zero.
///
/// This is the inverse of ScatterToBEV: scatter writes points→grid,
/// gather reads grid→points. Used in PointPainting fusion to sample
/// segmentation scores at projected LiDAR point locations.
///
/// SOUNDNESS NOTE: This is an advice op with no proof constraints,
/// following the same pattern as ScatterToBEV and PillarMaxPool.
/// A full proof would require a lookup/permutation argument.
/// TODO: Implement lookup-based gather verification.
#[derive(Clone, Debug)]
pub struct GatherFromGrid {
    pub n_points: usize,
    pub channels: usize,
    pub grid_h: usize,
    pub grid_w: usize,
}

impl BasicBlock for GatherFromGrid {
    fn run(&self, inputs: &[&Witness]) -> Vec<Witness> {
        assert_eq!(inputs.len(), 2);
        let grid = inputs[0];    // [C, H, W]
        let coords = inputs[1];  // [N_points, 2]
        let grid_evals = grid.data.as_ref().unwrap().evaluations_ref();
        let coord_evals = coords.data.as_ref().unwrap().evaluations_ref();

        let c_pad = self.channels.next_power_of_two();
        let w_pad = self.grid_w.next_power_of_two();
        let h_pad = self.grid_h.next_power_of_two();
        let n_pad = self.n_points.next_power_of_two();
        let coord_dim_pad = 2usize.next_power_of_two();

        let out_size = n_pad * c_pad;
        let mut out_data = vec![GoldilocksField(0); out_size];

        for p in 0..self.n_points {
            // coords little-endian: coord_dim bits (lowest) | n bits
            let cy = f_to_int(coord_evals[0 + p * coord_dim_pad]) as usize;
            let cx = f_to_int(coord_evals[1 + p * coord_dim_pad]) as usize;
            if cy < self.grid_h && cx < self.grid_w {
                for c in 0..self.channels {
                    // grid little-endian: w bits (lowest) | h bits | c bits
                    let grid_idx = cx + cy * w_pad + c * w_pad * h_pad;
                    // Y little-endian: c bits (lowest) | n bits
                    let out_idx = c + p * c_pad;
                    out_data[out_idx] = grid_evals[grid_idx];
                }
            }
        }

        let out_shape = vec![self.n_points, self.channels];
        let n = get_n(&out_shape);
        if out_data.len() < (1 << n) {
            out_data.resize(1 << n, GoldilocksField(0));
        }
        vec![Witness::new(out_shape, out_data, grid.data_type, grid.sf, Role::Output)]
    }

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
    use crate::dag::{Witness, DataType, Role};
    use goldilocks_cuda::GoldilocksField;

    fn make_witness(shape: Vec<usize>, data: Vec<u64>) -> Witness {
        let data: Vec<GoldilocksField> = data.into_iter().map(GoldilocksField).collect();
        Witness::new(shape, data, DataType::Uint, 0, Role::Input)
    }

    #[test]
    fn test_pillar_maxpool_run() {
        let pool = PillarMaxPool { n_pillars: 2, max_points: 2, features: 2 };
        // X[2, 2, 2] — little-endian: d(2) | t(2) | n(2)
        // pillar 0: t=0 d0=1 d1=3, t=1 d0=5 d1=2
        // pillar 1: t=0 d0=4 d1=1, t=1 d0=2 d1=6
        let x = make_witness(vec![2, 2, 2], vec![
            1, 3, 5, 2,  // pillar 0: (t=0,d=0)=1, (t=0,d=1)=3, (t=1,d=0)=5, (t=1,d=1)=2
            4, 1, 2, 6,  // pillar 1: (t=0,d=0)=4, (t=0,d=1)=1, (t=1,d=0)=2, (t=1,d=1)=6
        ]);
        let result = pool.run(&[&x]);
        let y = &result[0];
        assert_eq!(y.shape, vec![2, 2]);
        // pillar 0: max(1,5)=5, max(3,2)=3
        assert_eq!(y.data.as_ref().unwrap().index(0), GoldilocksField(5));
        assert_eq!(y.data.as_ref().unwrap().index(1), GoldilocksField(3));
        // pillar 1: max(4,2)=4, max(1,6)=6
        assert_eq!(y.data.as_ref().unwrap().index(2), GoldilocksField(4));
        assert_eq!(y.data.as_ref().unwrap().index(3), GoldilocksField(6));
    }

    #[test]
    fn test_scatter_to_bev_run() {
        let scatter = ScatterToBEV { n_pillars: 2, features: 2, ny: 2, nx: 2 };
        // X[2, 2] — pillar 0: d0=10 d1=20, pillar 1: d0=30 d1=40
        let x = make_witness(vec![2, 2], vec![10, 20, 30, 40]);
        // coords[2, 2] — pillar 0: (y=0, x=1), pillar 1: (y=1, x=0)
        let coords = make_witness(vec![2, 2], vec![0, 1, 1, 0]);
        let result = scatter.run(&[&x, &coords]);
        let y = &result[0];
        assert_eq!(y.shape, vec![2, 2, 2]);
        // Y[d, y, x] little-endian: x(2) | y(2) | d(2)
        // d=0: (0,0)=0, (0,1)=10, (1,0)=30, (1,1)=0
        // d=1: (0,0)=0, (0,1)=20, (1,0)=40, (1,1)=0
        assert_eq!(y.data.as_ref().unwrap().index(0), GoldilocksField(0));  // d=0,y=0,x=0
        assert_eq!(y.data.as_ref().unwrap().index(1), GoldilocksField(10)); // d=0,y=0,x=1
        assert_eq!(y.data.as_ref().unwrap().index(2), GoldilocksField(30)); // d=0,y=1,x=0
        assert_eq!(y.data.as_ref().unwrap().index(3), GoldilocksField(0));  // d=0,y=1,x=1
        assert_eq!(y.data.as_ref().unwrap().index(4), GoldilocksField(0));  // d=1,y=0,x=0
        assert_eq!(y.data.as_ref().unwrap().index(5), GoldilocksField(20)); // d=1,y=0,x=1
        assert_eq!(y.data.as_ref().unwrap().index(6), GoldilocksField(40)); // d=1,y=1,x=0
        assert_eq!(y.data.as_ref().unwrap().index(7), GoldilocksField(0));  // d=1,y=1,x=1
    }

    #[test]
    fn test_gather_from_grid_run() {
        let gather = GatherFromGrid { n_points: 2, channels: 2, grid_h: 2, grid_w: 2 };
        // grid[2, 2, 2] — little-endian: w(2) | h(2) | c(2)
        // c=0: (h=0,w=0)=10, (h=0,w=1)=20, (h=1,w=0)=30, (h=1,w=1)=40
        // c=1: (h=0,w=0)=50, (h=0,w=1)=60, (h=1,w=0)=70, (h=1,w=1)=80
        let grid = make_witness(vec![2, 2, 2], vec![
            10, 20, 30, 40,  // c=0
            50, 60, 70, 80,  // c=1
        ]);
        // coords[2, 2] — point 0: (y=0, x=1), point 1: (y=1, x=0)
        let coords = make_witness(vec![2, 2], vec![0, 1, 1, 0]);
        let result = gather.run(&[&grid, &coords]);
        let y = &result[0];
        assert_eq!(y.shape, vec![2, 2]);
        // Y[p, c] little-endian: c bits (lowest) | p bits
        // point 0 at (0,1): c=0→20, c=1→60
        // point 1 at (1,0): c=0→30, c=1→70
        assert_eq!(y.data.as_ref().unwrap().index(0), GoldilocksField(20)); // p=0,c=0
        assert_eq!(y.data.as_ref().unwrap().index(1), GoldilocksField(60)); // p=0,c=1
        assert_eq!(y.data.as_ref().unwrap().index(2), GoldilocksField(30)); // p=1,c=0
        assert_eq!(y.data.as_ref().unwrap().index(3), GoldilocksField(70)); // p=1,c=1
    }
}
