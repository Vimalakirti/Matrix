//! Multifold perf (K=50 fresh + k=13 accumulator = M=63 binary instances).
//!
//! cargo run --release --example bench_multifold

use almost_goldilocks_cuda::ajtai::{
    commit, multifold_commitment, multifold_witness, ChunkSize, RingChallenge,
    RingCommitment, Seed,
};
use almost_goldilocks_cuda::memory::synchronize;
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
    f();
    synchronize().ok();
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = std::time::Instant::now();
        f();
        synchronize().ok();
        samples.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    samples.iter().fold(f64::INFINITY, |a, b| a.min(*b))
}

fn main() {
    almost_goldilocks_cuda::init().unwrap();
    let mut rng = StdRng::seed_from_u64(7);
    let seed = Seed([1, 2, 3, 4, 5, 6, 7, 8]);
    let m = 63;

    println!("# Multifold @ K+k = {} binary instances (A100, min ms over 3 iters)", m);
    println!("log_n,N_ring,mf_witness_ms,mf_commit_us,output_size_MB");

    for log_n in [14, 16, 18, 20, 22].iter() {
        let n_ring = 1usize << (log_n - 6);
        let witnesses: Vec<Vec<u64>> = (0..m).map(|_| rand_z(&mut rng, n_ring)).collect();
        // M - 1 challenges; witnesses[0] has implicit weight 1.
        let challenges: Vec<RingChallenge> = (0..(m - 1)).map(|_| rand_r(&mut rng)).collect();
        let w_refs: Vec<&[u64]> = witnesses.iter().map(|w| w.as_slice()).collect();

        let fw_ms = time_ms(3, || {
            let _ = multifold_witness(&w_refs, &challenges).unwrap();
        });

        let commits: Vec<RingCommitment> = witnesses.iter()
            .map(|z| commit(seed, z, Some(ChunkSize::C256)).unwrap())
            .collect();
        let c_refs: Vec<&RingCommitment> = commits.iter().collect();

        let fc_us = time_ms(20, || {
            let _ = multifold_commitment(&c_refs, &challenges).unwrap();
        }) * 1000.0;

        let out_mb = (n_ring * 64 * 2) as f64 / (1024.0 * 1024.0);
        println!("{},{},{:.2},{:.1},{:.1}", log_n, n_ring, fw_ms, fc_us, out_mb);
    }
}
