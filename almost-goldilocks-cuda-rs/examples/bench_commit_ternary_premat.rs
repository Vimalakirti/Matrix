//! A/B benchmark: ternary commit with on-the-fly ChaCha8 PRG vs
//! pre-materialized M. Inputs are pre-uploaded to device so we time
//! the kernel work only.
//!
//! cargo run --release --example bench_commit_ternary_premat

use almost_goldilocks_cuda::ajtai::{
    commit_ternary, commit_ternary_premat, multifold_witness, split_witness_device,
    MaterializedM, RingChallenge, Seed, SPLIT_K_CHUNKS, TernaryChunksDevice,
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
    let mut rng = StdRng::seed_from_u64(909);

    // log_n encodes the witness size: N_ring = 2^(log_n - 6).
    // Pre-mat M memory at this N_ring = 7680 * N_ring bytes:
    //   log_n=14   →   1.9 MB
    //   log_n=18   →   30 MB
    //   log_n=20   →   120 MB
    //   log_n=22   →   480 MB
    //   log_n=24   →   1.9 GB
    //   log_n=26   →   7.5 GB
    //   log_n=27   →   15 GB     ← still fits on A100 80 GB
    //   log_n=28   →   30 GB
    let log_ns = [14, 18, 20, 22, 24, 26];

    println!("# Ternary commit (13 chunks) — on-the-fly PRG vs pre-materialized M");
    println!("# Kernel only (inputs pre-uploaded). Min ms over 5 iters; mat is single shot.");
    println!("log_n,N_ring,on_the_fly_ms,premat_commit_ms,mat_one_shot_ms,speedup,M_size_MB");

    for log_n in log_ns.iter() {
        let n_ring = 1usize << (log_n - 6);
        let m_bytes = (n_ring as u64) * 7680;
        if m_bytes > 8 * 1024 * 1024 * 1024 {     // skip if M > 8 GB (still allocatable but slow)
            // Continue anyway for the requested sizes — A100 80 GB is fine.
        }

        let seed = Seed([rng.gen(), rng.gen(), rng.gen(), rng.gen(),
                         rng.gen(), rng.gen(), rng.gen(), rng.gen()]);

        // Build realistic ternary chunks via real multifold + split.
        let m_inst = 50 + SPLIT_K_CHUNKS;
        let inner_w: Vec<Vec<u64>> =
            (0..m_inst).map(|_| rand_z(&mut rng, n_ring)).collect();
        let inner_refs: Vec<&[u64]> = inner_w.iter().map(|v| v.as_slice()).collect();
        let inner_chal: Vec<RingChallenge> = (0..(m_inst - 1)).map(|_| rand_r(&mut rng)).collect();
        let z_wide = multifold_witness(&inner_refs, &inner_chal).unwrap();
        drop(inner_w);
        let d_z_wide = DeviceBuffer::<i16>::from_slice(&z_wide).unwrap();
        drop(z_wide);
        let chunks: TernaryChunksDevice = split_witness_device(&d_z_wide).unwrap();
        drop(d_z_wide);

        // On-the-fly: re-runs PRG inside the kernel every call.
        let on_the_fly_ms = time_ms(5, || {
            let _ = commit_ternary(seed, &chunks, None).unwrap();
        });

        // Materialize once (count this separately).
        let mat_one_shot_ms = time_ms(3, || {
            // Each call allocates a fresh MaterializedM. Time it under the
            // same warmup/iter regime as a normal kernel.
            let _ = MaterializedM::new(seed, n_ring).unwrap();
        });

        let m = MaterializedM::new(seed, n_ring).unwrap();
        let premat_ms = time_ms(5, || {
            let _ = commit_ternary_premat(&m, &chunks, None).unwrap();
        });

        println!("{},{},{:.2},{:.2},{:.2},{:.2}x,{:.1}",
                 log_n, n_ring, on_the_fly_ms, premat_ms, mat_one_shot_ms,
                 on_the_fly_ms / premat_ms,
                 m_bytes as f64 / (1024.0 * 1024.0));
    }
}
