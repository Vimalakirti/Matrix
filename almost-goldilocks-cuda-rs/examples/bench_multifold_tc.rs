//! Three-way A/B/C benchmark: scalar mixed multifold vs the two
//! tensor-core variants.
//!
//!   scalar   — int8 challenge lookups + i32 add (no tensor cores)
//!   tc_v1    — WMMA INT8 matmul on a materialized [N_ring, M·64] z_mat
//!   tc_fused — WMMA INT8 matmul, z unpacked on-the-fly into shared mem
//!
//! cargo run --release --example bench_multifold_tc

use almost_goldilocks_cuda::ajtai::{
    multifold_mixed_witness, multifold_mixed_witness_tc,
    multifold_mixed_witness_tc_fused, multifold_witness, split_witness_device,
    RingChallenge, TernaryChunksDevice, SPLIT_K_CHUNKS,
};
use almost_goldilocks_cuda::memory::{synchronize, DeviceBuffer};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn rand_z(rng: &mut StdRng, n: usize) -> Vec<u64> {
    (0..n).map(|_| rng.gen::<u64>()).collect()
}

fn rand_r(rng: &mut StdRng) -> RingChallenge {
    let mut c = [0i8; 64];
    for v in c.iter_mut() {
        *v = match rng.gen_range(0..4) { 0 => -1, 1 => 0, 2 => 1, _ => 2 };
    }
    RingChallenge::new(c).unwrap()
}

fn time_ms<F: FnMut()>(iters: usize, mut f: F) -> f64 {
    f(); synchronize().ok();
    f(); synchronize().ok();
    let mut best = f64::INFINITY;
    for _ in 0..iters {
        let t = std::time::Instant::now();
        f();
        synchronize().ok();
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        if ms < best { best = ms; }
    }
    best
}

fn main() {
    almost_goldilocks_cuda::init().unwrap();
    let mut rng = StdRng::seed_from_u64(101);

    let k_bin  = 50usize;
    let k_tern = SPLIT_K_CHUNKS;          // = 13
    let m      = k_bin + k_tern;

    println!("# Mixed multifold @ K={}, T={}  (min ms over 5 iters)", k_bin, k_tern);
    println!("log_n,N_ring,scalar_ms,tc_v1_ms,tc_fused_ms,fused_vs_scalar,fused_vs_v1");

    for log_n in [20, 22, 24, 26, 28].iter() {
        let n_ring = 1usize << (log_n - 6);

        let bins: Vec<Vec<u64>> = (0..k_bin).map(|_| rand_z(&mut rng, n_ring)).collect();
        let bin_refs: Vec<&[u64]> = bins.iter().map(|v| v.as_slice()).collect();

        // Realistic 13-chunk ternary input: splitb of a prior multifold output.
        let inner_w: Vec<Vec<u64>> =
            (0..(k_bin + k_tern)).map(|_| rand_z(&mut rng, n_ring)).collect();
        let inner_refs: Vec<&[u64]> = inner_w.iter().map(|v| v.as_slice()).collect();
        let inner_chal: Vec<RingChallenge> = (0..(m - 1)).map(|_| rand_r(&mut rng)).collect();
        let z_wide = multifold_witness(&inner_refs, &inner_chal).unwrap();
        let d_z_wide = DeviceBuffer::<i16>::from_slice(&z_wide).unwrap();
        let chunks: TernaryChunksDevice = split_witness_device(&d_z_wide).unwrap();

        let challenges: Vec<RingChallenge> = (0..(m - 1)).map(|_| rand_r(&mut rng)).collect();

        // Use fewer iters at very large sizes to keep the bench tractable.
        let iters = if *log_n >= 26 { 2 } else { 5 };
        let scalar_ms = time_ms(iters, || {
            let _ = multifold_mixed_witness(&bin_refs, &chunks, &challenges).unwrap();
        });
        let tc1_ms = time_ms(iters, || {
            let _ = multifold_mixed_witness_tc(&bin_refs, &chunks, &challenges).unwrap();
        });
        let tcf_ms = time_ms(iters, || {
            let _ = multifold_mixed_witness_tc_fused(&bin_refs, &chunks, &challenges).unwrap();
        });

        println!("{},{},{:.2},{:.2},{:.2},{:.2}x,{:.2}x",
                 log_n, n_ring, scalar_ms, tc1_ms, tcf_ms,
                 scalar_ms / tcf_ms, tc1_ms / tcf_ms);
    }
}
