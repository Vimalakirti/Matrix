//! Split perf: wide i16 folded witness → 13 ternary chunks (pos, neg bitmasks).
//!
//! Reports both:
//!   * `host_us`    — full host API (H2D copy + kernel + D2H copy). What you
//!                    pay if the wide witness lives in host memory.
//!   * `device_us`  — device-resident split (kernel + alloc only). What the
//!                    real prover pays, since the wide witness comes straight
//!                    out of `multifold_witness` on-GPU.
//!
//! cargo run --release --example bench_split

use almost_goldilocks_cuda::ajtai::{
    multifold_witness, split_witness, split_witness_device,
    RingChallenge, SPLIT_K_CHUNKS,
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

fn time_us<F: FnMut()>(iters: usize, mut f: F) -> f64 {
    // Two warmup runs — first launch in a CUDA context pays for module load.
    f(); synchronize().ok();
    f(); synchronize().ok();
    let mut best = f64::INFINITY;
    for _ in 0..iters {
        let t = std::time::Instant::now();
        f();
        synchronize().ok();
        let us = t.elapsed().as_secs_f64() * 1_000_000.0;
        if us < best { best = us; }
    }
    best
}

fn main() {
    almost_goldilocks_cuda::init().unwrap();
    let mut rng = StdRng::seed_from_u64(11);
    let m = 63;

    // Top-level warmup so the first log_n size doesn't pay module-load cost.
    {
        let dummy = vec![0i16; 64];
        let _ = split_witness(&dummy).unwrap();
        synchronize().ok();
    }

    println!("# Split @ K+k = {}  (min over 8 iters, after 2 warmups)", m);
    println!("log_n,N_ring,host_us,device_us,coefs/us(dev),output_MB");

    for log_n in [14, 16, 18, 20, 22].iter() {
        let n_ring = 1usize << (log_n - 6);

        // Realistic wide-witness input from multifold.
        let witnesses: Vec<Vec<u64>> = (0..m).map(|_| rand_z(&mut rng, n_ring)).collect();
        let challenges: Vec<RingChallenge> = (0..(m - 1)).map(|_| rand_r(&mut rng)).collect();
        let w_refs: Vec<&[u64]> = witnesses.iter().map(|w| w.as_slice()).collect();
        let z_wide_host = multifold_witness(&w_refs, &challenges).unwrap();
        let d_z_wide = DeviceBuffer::<i16>::from_slice(&z_wide_host).unwrap();

        let host_us = time_us(8, || {
            let _ = split_witness(&z_wide_host).unwrap();
        });

        let device_us = time_us(8, || {
            let _ = split_witness_device(&d_z_wide).unwrap();
        });

        let coefs = (n_ring * 64) as f64;
        let out_mb = (SPLIT_K_CHUNKS * n_ring * 2 * 8) as f64 / (1024.0 * 1024.0);
        println!("{},{},{:.1},{:.1},{:.0},{:.2}",
                 log_n, n_ring, host_us, device_us, coefs / device_us, out_mb);
    }
}
