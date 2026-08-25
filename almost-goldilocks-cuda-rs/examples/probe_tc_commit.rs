//! MVP go/no-go probe — measure pure WMMA throughput at commit's shape
//! (B=16 padded, K = N_ring·64, OUT = 960) using random INT8 buffers.
//! Output is meaningless; only the kernel time matters.
//!
//! If pure compute is fast at log_n=22 (say < 5 ms), the full 8-limb
//! commit kernel is worth building. Otherwise stop.

use almost_goldilocks_cuda::ffi;
use almost_goldilocks_cuda::memory::{synchronize, DeviceBuffer};

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

    let log_ns = [18usize, 20, 22, 24];
    let chunk_k_tiles_per_block = 64;

    println!("# TC commit MVP probe — pure WMMA matmul at commit shape");
    println!("# z[16, K] * M_int8[K, 960] via mma.sync m16n16k16 s8.s8.s32");
    println!("# Output meaningless; pure compute time only.");
    println!("log_n,N_ring,K,M_int8_GB,kernel_ms,projected_8_limb_ms");

    for &log_n in &log_ns {
        let n_ring = 1usize << (log_n - 6);
        let k_total: i64 = (n_ring as i64) * 64;
        let m_bytes = k_total * 960;
        if m_bytes > 8 * 1024 * 1024 * 1024 {
            println!("log_n={} skipped (M_int8 = {:.1} GB)", log_n, m_bytes as f64 / 1e9);
            continue;
        }

        // Allocate z, M, partial. Random content is fine for compute timing.
        // (zero-initialize for determinism — content doesn't affect mma throughput).
        let z_host = vec![0i8; (16 * k_total) as usize];
        let m_host = vec![0i8; m_bytes as usize];
        let d_z   = DeviceBuffer::<i8>::from_slice(&z_host).unwrap();
        let d_m   = DeviceBuffer::<i8>::from_slice(&m_host).unwrap();
        drop(z_host); drop(m_host);

        let num_k_chunks = ((k_total / 16) as usize) / chunk_k_tiles_per_block;
        let partial_size = num_k_chunks * 60 * 256;
        let mut d_partial = DeviceBuffer::<i32>::new(partial_size).unwrap();

        let kernel_ms = time_ms(5, || {
            let ret = unsafe {
                ffi::ajtai_tc_commit_probe_ffi(
                    d_z.as_ptr(),
                    d_m.as_ptr(),
                    d_partial.as_mut_ptr(),
                    k_total as i32,
                    num_k_chunks as i32,
                )
            };
            assert_eq!(ret, 0);
        });

        let m_gb = m_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        println!("{},{},{},{:.2},{:.2},{:.2}",
                 log_n, n_ring, k_total, m_gb, kernel_ms, 8.0 * kernel_ms);
    }
}
