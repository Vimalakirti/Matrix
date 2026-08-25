//! Kernel-only A/B/C benchmark — inputs pre-uploaded so PCIe transfer
//! is excluded from the timing. Calls the raw FFI directly against
//! pre-allocated device buffers.
//!
//! cargo run --release --example bench_multifold_tc_kernel_only

use almost_goldilocks_cuda::ajtai::{
    multifold_witness, split_witness_device, RingChallenge,
    SPLIT_K_CHUNKS, TernaryChunksDevice,
};
use almost_goldilocks_cuda::ffi;
use almost_goldilocks_cuda::memory::{synchronize, DeviceBuffer};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::os::raw::c_int;

const RING_DIM: usize = 64;

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

fn constant_one_packed() -> [i8; 64] {
    let mut c = [0i8; 64];
    c[0] = 1;
    c
}

fn main() {
    almost_goldilocks_cuda::init().unwrap();
    let mut rng = StdRng::seed_from_u64(303);

    let k_bin  = 50usize;
    let k_tern = SPLIT_K_CHUNKS;
    let m      = k_bin + k_tern;

    println!("# Mixed multifold @ K={}, T={}  (kernel-only, min ms)", k_bin, k_tern);
    println!("# (inputs pre-uploaded; FFI call + cuda sync only)");
    println!("log_n,N_ring,scalar_ms,tc_v1_ms,tc_fused_ms,fused/scalar,fused/v1");

    for log_n in [20, 22, 24, 26, 28].iter() {
        let n_ring = 1usize << (log_n - 6);

        // Build inputs on host then upload ONCE.
        let mut z_packed = Vec::<u64>::with_capacity(k_bin * n_ring);
        for _ in 0..k_bin {
            for _ in 0..n_ring {
                z_packed.push(rng.gen::<u64>());
            }
        }
        let d_z_bin = DeviceBuffer::<u64>::from_slice(&z_packed).unwrap();
        drop(z_packed);  // free host memory

        // Build a realistic ternary chunks via real split.
        let inner_w: Vec<Vec<u64>> = (0..(m)).map(|_| rand_z(&mut rng, n_ring)).collect();
        let inner_refs: Vec<&[u64]> = inner_w.iter().map(|v| v.as_slice()).collect();
        let inner_chal: Vec<RingChallenge> = (0..(m - 1)).map(|_| rand_r(&mut rng)).collect();
        let z_wide = multifold_witness(&inner_refs, &inner_chal).unwrap();
        drop(inner_w);
        let d_z_wide = DeviceBuffer::<i16>::from_slice(&z_wide).unwrap();
        drop(z_wide);
        let chunks: TernaryChunksDevice = split_witness_device(&d_z_wide).unwrap();
        drop(d_z_wide);

        // Pack challenges (with constant-one prefix for binary[0]).
        let mut r_packed = Vec::<i8>::with_capacity(m * 64);
        r_packed.extend_from_slice(&constant_one_packed());
        for _ in 0..(m - 1) {
            r_packed.extend_from_slice(&rand_r(&mut rng).coeffs);
        }
        let d_r = DeviceBuffer::<i8>::from_slice(&r_packed).unwrap();

        let mut d_out = DeviceBuffer::<i16>::new(n_ring * RING_DIM).unwrap();

        let iters = if *log_n >= 26 { 2 } else { 5 };

        // Scalar mixed multifold (chunk_size matches Rust default heuristic).
        let chunk_size: u64 = if n_ring <= 64 { 1 }
                              else if n_ring <= 4096 { 4 }
                              else { 16 };
        let scalar_ms = time_ms(iters, || {
            let ret = unsafe {
                ffi::ajtai_multifold_mixed_witness_ffi(
                    d_z_bin.as_ptr(),
                    chunks.pos.as_ptr(),
                    chunks.neg.as_ptr(),
                    d_r.as_ptr(),
                    d_out.as_mut_ptr(),
                    k_bin as c_int,
                    k_tern as c_int,
                    n_ring as u64,
                    chunk_size,
                )
            };
            assert_eq!(ret, 0);
        });

        let tc1_ms = time_ms(iters, || {
            let ret = unsafe {
                ffi::ajtai_multifold_mixed_witness_tc_ffi(
                    d_z_bin.as_ptr(),
                    chunks.pos.as_ptr(),
                    chunks.neg.as_ptr(),
                    d_r.as_ptr(),
                    d_out.as_mut_ptr(),
                    k_bin as c_int,
                    k_tern as c_int,
                    n_ring as u64,
                )
            };
            assert_eq!(ret, 0);
        });

        let tcf_ms = time_ms(iters, || {
            let ret = unsafe {
                ffi::ajtai_multifold_mixed_witness_tc_fused_ffi(
                    d_z_bin.as_ptr(),
                    chunks.pos.as_ptr(),
                    chunks.neg.as_ptr(),
                    d_r.as_ptr(),
                    d_out.as_mut_ptr(),
                    k_bin as c_int,
                    k_tern as c_int,
                    n_ring as u64,
                )
            };
            assert_eq!(ret, 0);
        });

        println!("{},{},{:.2},{:.2},{:.2},{:.2}x,{:.2}x",
                 log_n, n_ring, scalar_ms, tc1_ms, tcf_ms,
                 scalar_ms / tcf_ms, tc1_ms / tcf_ms);
    }
}
