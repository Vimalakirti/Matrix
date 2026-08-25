//! Quick perf measurement of fold_witness and fold_commitment.
//!
//! cargo run --release --example bench_fold

use almost_goldilocks_cuda::ajtai::{
    commit, fold_commitment, fold_witness_device, ChunkSize, RingChallenge, Seed,
    RING_DIM,
};
use almost_goldilocks_cuda::memory::synchronize;
use almost_goldilocks_cuda::DeviceBuffer;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn rand_z(rng: &mut StdRng, n: usize) -> Vec<u64> {
    (0..n).map(|_| rng.gen::<u64>()).collect()
}

fn rand_r(rng: &mut StdRng) -> RingChallenge {
    let mut coeffs = [0i8; 64];
    for c in coeffs.iter_mut() {
        *c = match rng.gen_range(0..4) {
            0 => -1, 1 => 0, 2 => 1, _ => 2,
        };
    }
    RingChallenge::new(coeffs).unwrap()
}

fn time<F: FnMut()>(iters: usize, mut f: F) -> f64 {
    f(); // warmup
    synchronize().ok();
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = std::time::Instant::now();
        f();
        synchronize().ok();
        samples.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    samples.iter().fold(f64::INFINITY, |a, b| a.min(*b))
}

fn main() {
    almost_goldilocks_cuda::init().unwrap();
    let mut rng = StdRng::seed_from_u64(123);
    let seed = Seed([1, 2, 3, 4, 5, 6, 7, 8]);

    println!("# Fold benchmarks (A100, min ms over 5 iters)");
    println!("log_n,N_ring,fold_witness_ms,fold_commitment_us,output_size_MB");

    for log_n in [14, 16, 18, 20, 22, 24].iter() {
        let n_ring = 1usize << (log_n - 6);
        let z1 = rand_z(&mut rng, n_ring);
        let z2 = rand_z(&mut rng, n_ring);
        let r = rand_r(&mut rng);

        let d_z1 = DeviceBuffer::<u64>::from_slice(&z1).unwrap();
        let d_z2 = DeviceBuffer::<u64>::from_slice(&z2).unwrap();
        let mut d_out = DeviceBuffer::<u64>::new(n_ring * RING_DIM).unwrap();

        let fw_ms = time(5, || {
            fold_witness_device(&d_z1, &r, &d_z2, &mut d_out).unwrap();
        });

        // fold_commitment perf
        let c1 = commit(seed, &z1, Some(ChunkSize::C256)).unwrap();
        let c2 = commit(seed, &z2, Some(ChunkSize::C256)).unwrap();
        let fc_ms = time(20, || {
            let _ = fold_commitment(&c1, &r, &c2).unwrap();
        });

        let output_mb = (n_ring * RING_DIM * 8) as f64 / (1024.0 * 1024.0);
        println!("{},{},{:.2},{:.1},{:.1}",
            log_n, n_ring, fw_ms, fc_ms * 1000.0, output_mb);
    }
}
