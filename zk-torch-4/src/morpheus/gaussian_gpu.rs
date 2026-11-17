//! GPU driver of the certified Gaussian sampler
//! (`cuda_almost_goldilocks/morpheus_gauss.cuh`).
//!
//! The kernel runs the P128 first stage per coordinate on its addressed
//! coins and finalizes every coordinate whose first raw trials certify and
//! pass the 64-bit correction gate. Flagged coordinates are replayed on the
//! CPU by `gaussian_fast::sample` on the same addresses, which yields the
//! identical value; the output is therefore exactly the CPU sampler's.

use super::coins::Coins;
use super::gaussian::{Abort, CoordinateStreams, Stats};
use super::gaussian_fast::{self, FastSigma};
use almost_goldilocks_cuda::{ffi, DeviceBuffer};
use rayon::prelude::*;

fn words(fs: &FastSigma) -> Vec<u64> {
    let mut v = Vec::new();
    for c in fs.coeff_lo.iter() { v.extend_from_slice(&c.0[..3]); }
    for c in fs.coeff_hi.iter() { v.extend_from_slice(&c.0[..3]); }
    assert!(fs.coeff_lo.iter().chain(&fs.coeff_hi).all(|c| c.0[3..].iter().all(|&w| w == 0)));
    v.extend_from_slice(&fs.ln2_lo.0[..2]);
    v.extend_from_slice(&fs.ln2_hi.0[..2]);
    v.push(fs.tail.low_u64());
    v.extend_from_slice(&fs.s8.0[..2]);
    v.extend_from_slice(&fs.d4.0[..3]);
    v.push(fs.sigma);
    v.push(fs.bound.low_u64());
    v.push(fs.gate_small);
    assert_eq!(v.len() as i32, unsafe { ffi::agl_morpheus_gauss_words() });
    v
}

/// Timing and fallback counts of one batch.
#[derive(Debug, Default, Clone)]
pub struct GpuGaussReport {
    pub n: usize,
    pub replayed: usize,
    pub gpu_ms: f64,
    pub replay_ms: f64,
}

/// Samples coordinates [coord0, coord0+n) of (opening, attempt) at width σ
/// into a device buffer (the responses are formed on the GPU).
pub fn sample_range_device(
    coins: &Coins,
    opening: u64,
    attempt: u16,
    fs: &FastSigma,
    coord0: u32,
    n: u32,
) -> Result<(DeviceBuffer<i64>, GpuGaussReport), Abort> {
    if super::coins::no_zk() {
        let mut out = DeviceBuffer::<i64>::new(n as usize).expect("device alloc");
        out.zero().expect("zero masks");
        return Ok((out, GpuGaussReport { n: n as usize, replayed: 0, gpu_ms: 0.0, replay_ms: 0.0 }));
    }
    let consts = words(fs);
    let base = coins.gaussian_base_lanes(opening, attempt);
    let mut out = DeviceBuffer::<i64>::new(n as usize).expect("device alloc");
    let mut idx: Vec<u32> = Vec::with_capacity(n as usize);
    let mut nflag = 0u32;
    let t = std::time::Instant::now();
    let rc = unsafe { ffi::agl_morpheus_gauss_dev(consts.as_ptr(), base.as_ptr(), coord0, n, out.as_mut_ptr(), idx.as_mut_ptr(), &mut nflag) };
    assert_eq!(rc, 0, "gaussian kernel failed");
    unsafe { idx.set_len(nflag as usize) };
    let gpu_ms = t.elapsed().as_secs_f64() * 1e3;
    let t = std::time::Instant::now();
    let vals: Vec<Result<i64, Abort>> = idx
        .par_iter()
        .map(|&i| {
            let mut s = CoordinateStreams { coins, opening, attempt, coordinate: coord0 + i };
            gaussian_fast::sample(fs, &mut s, &mut Stats::default())
        })
        .collect();
    let vals: Vec<i64> = vals.into_iter().collect::<Result<_, _>>()?;
    let rc = unsafe { ffi::agl_morpheus_gauss_scatter(out.as_mut_ptr(), idx.as_ptr(), vals.as_ptr(), nflag) };
    assert_eq!(rc, 0);
    let rep = GpuGaussReport { n: n as usize, replayed: nflag as usize, gpu_ms, replay_ms: t.elapsed().as_secs_f64() * 1e3 };
    Ok((out, rep))
}

/// Samples into an existing device range `d_out[0..n)`.
pub fn sample_into(coins: &Coins, opening: u64, attempt: u16, fs: &FastSigma, coord0: u32, n: u32, d_out: *mut i64) -> Result<GpuGaussReport, Abort> {
    if super::coins::no_zk() {
        let rc = unsafe { ffi::cuda_memset(d_out as *mut std::ffi::c_void, 0, n as usize * 8) };
        assert_eq!(rc, 0, "zero masks");
        return Ok(GpuGaussReport { n: n as usize, replayed: 0, gpu_ms: 0.0, replay_ms: 0.0 });
    }
    let consts = words(fs);
    let base = coins.gaussian_base_lanes(opening, attempt);
    let mut idx: Vec<u32> = Vec::with_capacity(n as usize);
    let mut nflag = 0u32;
    let t = std::time::Instant::now();
    let rc = unsafe { ffi::agl_morpheus_gauss_dev(consts.as_ptr(), base.as_ptr(), coord0, n, d_out, idx.as_mut_ptr(), &mut nflag) };
    assert_eq!(rc, 0, "gaussian kernel failed");
    unsafe { idx.set_len(nflag as usize) };
    let gpu_ms = t.elapsed().as_secs_f64() * 1e3;
    let t = std::time::Instant::now();
    let vals: Vec<i64> = idx.par_iter().map(|&i| {
        let mut s = CoordinateStreams { coins, opening, attempt, coordinate: coord0 + i };
        gaussian_fast::sample(fs, &mut s, &mut Stats::default())
    }).collect::<Result<_, _>>()?;
    let rc = unsafe { ffi::agl_morpheus_gauss_scatter(d_out, idx.as_ptr(), vals.as_ptr(), nflag) };
    assert_eq!(rc, 0);
    Ok(GpuGaussReport { n: n as usize, replayed: nflag as usize, gpu_ms, replay_ms: t.elapsed().as_secs_f64() * 1e3 })
}

/// Host-side convenience wrapper (tests).
pub fn sample_range(coins: &Coins, opening: u64, attempt: u16, fs: &FastSigma, coord0: u32, n: u32) -> Result<(Vec<i64>, GpuGaussReport), Abort> {
    let (d, rep) = sample_range_device(coins, opening, attempt, fs, coord0, n)?;
    Ok((d.to_vec().expect("dtoh"), rep))
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::coins::CoinProfile;

    #[test]
    fn gpu_sampler_equals_cpu_sampler() {
        let coins = Coins::new([11u8; 64], CoinProfile::Ternary);
        for &sigma in &[3u64, 1344, 3_076_346_150] {
            let fs = FastSigma::new(sigma);
            let n = 20_000u32;
            let (gpu, rep) = sample_range(&coins, 5, 3, &fs, 1000, n).unwrap();
            let cpu: Vec<i64> = (0..n).into_par_iter().map(|i| {
                let mut s = CoordinateStreams { coins: &coins, opening: 5, attempt: 3, coordinate: 1000 + i };
                gaussian_fast::sample(&fs, &mut s, &mut Stats::default()).unwrap()
            }).collect();
            assert_eq!(gpu, cpu, "sigma {sigma} {rep:?}");
            eprintln!("sigma {sigma}: {rep:?}");
        }
    }

    #[test]
    #[ignore]
    fn gpu_throughput() {
        let coins = Coins::new([12u8; 64], CoinProfile::Ternary);
        let fs = FastSigma::new(super::super::params::Profile::ternary().sigma);
        let _ = sample_range_device(&coins, 0, 0, &fs, 0, 1 << 20).unwrap();
        for &n in &[1u32 << 24, 143_425_536, 143_425_536] {
            let (_, rep) = sample_range_device(&coins, 0, 0, &fs, 0, n).unwrap();
            println!("n {n}: gpu {:.1} ms ({:.1} M/s), replayed {} in {:.1} ms", rep.gpu_ms, n as f64 / rep.gpu_ms / 1e3, rep.replayed, rep.replay_ms);
        }
    }
}
