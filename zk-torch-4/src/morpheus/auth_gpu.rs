//! GPU prover of the masked authentication (same output as `auth::prove`).
//!
//! Everything proportional to the response dimension runs on the device:
//! Gaussian masks (written straight into per-block row buffers), the
//! commitments of both rows at once (V = 2 shares every key read), Ev_raw,
//! the digit decodes behind H and E_O, the β combination of the witnesses
//! and the norms. The host keeps only O(64)-sized algebra (the final A
//! products, K combinations), the transcript and the rejection coin.

use super::auth::{AuthError, AuthProof, AuxTuple, Point, Responses, Tuple};
use super::coins::{Address, BitStream, Coins, Purpose};
use super::commit_gpu::{finish, partial_raw, KeyCache};
use super::field8::{eq_table, K8};
use super::gaussian_fast::FastSigma;
use super::gaussian_gpu::sample_into;
use super::keys::{KeyKind, KeySeed};
use super::params::{Profile, A_ENC_COLS};
use super::pcs_maps::{aux_index, mul_a};
use super::rejection::{self, Coin, Ratio};
use super::ring::{g_of_weights, AElem, DS, N};
use super::transcript::{Item, MTranscript, Stage};
use almost_goldilocks_cuda::{ffi, DeviceBuffer};
use almost_goldilocks_cuda::field::ALMOST_GOLDILOCKS_PRIME as Q;
use num_bigint::{BigInt, BigUint};
use std::os::raw::c_void;
use std::time::Instant;

/// Key caches of the three block kinds (5 primes: Gaussian inputs).
pub struct Keys { pub w: KeyCache, pub l: KeyCache, pub o: KeyCache }

impl Keys {
    pub fn build(p: &Profile, seed: &KeySeed) -> Self {
        Keys {
            w: KeyCache::build(seed, KeyKind::Source, p.rank, 0, p.source_payload_cols + p.hiding, 5),
            l: KeyCache::build(seed, KeyKind::Link, p.rank, 0, A_ENC_COLS + p.hiding, 5),
            o: KeyCache::build(seed, KeyKind::Aux, p.rank, 0, A_ENC_COLS * p.aux_coeffs + p.hiding, 5),
        }
    }
}

/// Point-dependent device state: w[l][j] = e_l e_j and the host factors.
pub struct DevPoint { pub(crate) w: DeviceBuffer<u64>, cols: usize, gk: AElem, rho_f: K8, x: Vec<K8>, degs: Vec<usize> }

impl DevPoint {
    pub fn new(p: &Profile, pt: &Point) -> Self {
        let cols = p.source_cols();
        let el = eq_table(&pt.xi[12..]);
        let ej = eq_table(&pt.xi[6..12]);
        let ek = eq_table(&pt.xi[..6]);
        assert_eq!(el.len(), cols);
        let flat = |v: &[K8]| -> Vec<u64> { v.iter().flat_map(|k| k.0).collect() };
        let mut w = DeviceBuffer::<u64>::new(cols * 64 * 8).unwrap();
        let rc = unsafe { ffi::agl_mauth_wtable(flat(&el).as_ptr(), cols as i32, flat(&ej).as_ptr(), w.as_mut_ptr()) };
        assert_eq!(rc, 0);
        let wk: [K8; DS] = std::array::from_fn(|k| ek[k]);
        DevPoint { w, cols, gk: g_of_weights(&wk), rho_f: pt.rho.mul(&K8::ONE.sub(&pt.rho)), x: pt.aux_vars(), degs: p.degs() }
    }
    /// Ev_raw of `cols` columns at ptr (int8 or int64).
    pub fn ev(&self, ptr: *const c_void, bytes: i32) -> AElem {
        let mut out = vec![0u64; 64 * 8];
        let rc = unsafe { ffi::agl_mauth_ev(ptr, bytes, self.cols as i32, self.w.as_ptr(), out.as_mut_ptr()) };
        assert_eq!(rc, 0);
        let pm = AElem((0..DS).map(|m| K8(std::array::from_fn(|a| out[m * 8 + a]))).collect());
        mul_a(&self.gk, &pm)
    }
}

fn decode(ptr: *const i64, ncols: usize) -> Vec<u64> {
    let mut out = vec![0u64; ncols * 64];
    let rc = unsafe { ffi::agl_mauth_decode(ptr, ncols as i32, out.as_mut_ptr()) };
    assert_eq!(rc, 0);
    out
}
/// H from decoded values of 8 digit columns starting at column `c0`.
fn h_of(dec: &[u64], c0: usize) -> AElem {
    AElem((0..DS).map(|m| K8(std::array::from_fn(|a| dec[(c0 + a) * 64 + m]))).collect())
}

fn ring_rows(v: Vec<u64>, rows: usize) -> Vec<Vec<Vec<u64>>> {
    // [V][rows][N] -> per vector, per row
    v.chunks(rows * N).map(|blk| blk.chunks(N).map(|r| r.to_vec()).collect()).collect()
}

/// Device witnesses of one source (int8): W (source columns) and r (link columns).
pub struct DevSource { pub w: DeviceBuffer<i8>, pub r: DeviceBuffer<i8> }

#[derive(Debug, Default, Clone)]
pub struct AuthTiming { pub attempts: u32, pub sample_ms: f64, pub first_msg_ms: f64, pub respond_ms: f64, pub decide_ms: f64, pub extract_ms: f64 }

/// Public response support of the source block as column runs [start, end),
/// sorted and disjoint (`auth_compact::Support`). The masks are zero outside
/// it and only its columns are sent, so a proof equals `auth_compact::prove`.
pub type Runs = Vec<(usize, usize)>;
pub fn runs_of(s: &super::auth_compact::Support) -> Runs {
    let mut r: Runs = Vec::new();
    for &c in s.columns() {
        match r.last_mut() { Some(l) if l.1 == c => l.1 = c + 1, _ => r.push((c, c + 1)) }
    }
    r
}
fn run_cols(r: &[(usize, usize)]) -> usize { r.iter().map(|&(a, b)| b - a).sum() }

/// Support columns of τ rows of a dense [row][cols][N] block, packed in order.
fn gather(z: &DeviceBuffer<i64>, cols: usize, runs: &[(usize, usize)]) -> DeviceBuffer<i64> {
    let rows = z.len() / (cols * N);
    let k = run_cols(runs);
    let mut out = DeviceBuffer::<i64>::new(rows * k * N).unwrap();
    for r in 0..rows {
        let mut o = r * k * N;
        for &(a, b) in runs {
            unsafe { almost_goldilocks_cuda::memory::memcpy_dtod(out.as_mut_ptr().add(o) as *mut c_void, z.as_ptr().add((r * cols + a) * N) as *const c_void, (b - a) * N * 8).unwrap(); }
            o += (b - a) * N;
        }
    }
    out
}
/// Inverse of `gather`: zero extension to the full block.
fn scatter(c: &DeviceBuffer<i64>, cols: usize, runs: &[(usize, usize)]) -> DeviceBuffer<i64> {
    let k = run_cols(runs);
    let rows = c.len() / (k * N);
    let mut out = DeviceBuffer::<i64>::new(rows * cols * N).unwrap();
    out.zero().unwrap();
    for r in 0..rows {
        let mut o = r * k * N;
        for &(a, b) in runs {
            unsafe { almost_goldilocks_cuda::memory::memcpy_dtod(out.as_mut_ptr().add((r * cols + a) * N) as *mut c_void, c.as_ptr().add(o) as *const c_void, (b - a) * N * 8).unwrap(); }
            o += (b - a) * N;
        }
    }
    out
}

fn norm(ptr: *const i64, n: usize) -> (u128, u64) {
    let mut o = [0u64; 3];
    let rc = unsafe { ffi::agl_mauth_norm(ptr, n as u64, o.as_mut_ptr()) };
    assert_eq!(rc, 0);
    (((o[1] as u128) << 64) | o[0] as u128, o[2])
}

/// The device inputs of one authentication prover (the primary device or a
/// replica): key caches, source witnesses and the auxiliary witness.
pub struct AuthDev<'a> { pub dev: i32, pub keys: &'a Keys, pub sources: &'a [DevSource], pub s_o: &'a DeviceBuffer<i8>, pub runs: Option<&'a [(usize, usize)]> }

/// Attempt masks sampled ahead of the authentication (device 0): they
/// depend only on (opening, attempt), so they are drawn while the GPU is idle
/// in the PIOP. Keyed by (opening, attempt).
type Masks = (DeviceBuffer<i64>, DeviceBuffer<i64>, DeviceBuffer<i64>);
/// Presampled masks: kept on the device when it has room, else parked in
/// host memory (re-uploaded on use, far cheaper than resampling).
enum Pre { Dev(Masks), Host(Vec<i64>, Vec<i64>, Vec<i64>) }
static PRESAMPLED: std::sync::Mutex<Vec<((u64, u16), Pre)>> = std::sync::Mutex::new(Vec::new());
fn to_host(b: &DeviceBuffer<i64>) -> Vec<i64> {
    let mut v: Vec<i64> = Vec::with_capacity(b.len());
    unsafe { v.set_len(b.len()); }
    { use rayon::prelude::*; v.par_chunks_mut(1 << 17).for_each(|c| { let mut i = 0; while i < c.len() { c[i] = 0; i += 512; } }); }
    assert_eq!(unsafe { ffi::agl_d2h_staged(v.as_mut_ptr() as *mut std::ffi::c_void, b.as_ptr() as *const std::ffi::c_void, b.len() * 8) }, 0);
    v
}

fn sample_masks(p: &Profile, coins: &Coins, opening: u64, attempt: u16, runs: Option<&[(usize, usize)]>) -> Result<Masks, AuthError> {
    let fs = FastSigma::new(p.sigma);
    let (sw, sl, so) = (p.source_cols(), p.link_cols(), p.aux_cols());
    let row = ((sw + sl + so) * N) as u64;
    let mut uw = DeviceBuffer::<i64>::new(2 * sw * N).unwrap();
    let mut ul = DeviceBuffer::<i64>::new(2 * sl * N).unwrap();
    let mut uo = DeviceBuffer::<i64>::new(2 * so * N).unwrap();
    // Source masks only on the public support (same coordinates as the
    // dense sampler); zero elsewhere.
    let full = [(0, sw)];
    let runs = runs.unwrap_or(&full);
    if run_cols(runs) < sw { uw.zero().unwrap(); }
    for r in 0..p.tau {
        let b = r as u64 * row;
        unsafe {
            for &(c0, c1) in runs {
                sample_into(coins, opening, attempt, &fs, (b + (c0 * N) as u64) as u32, ((c1 - c0) * N) as u32, uw.as_mut_ptr().add((r * sw + c0) * N)).map_err(AuthError::Sampler)?;
            }
            sample_into(coins, opening, attempt, &fs, (b + (sw * N) as u64) as u32, (sl * N) as u32, ul.as_mut_ptr().add(r * sl * N)).map_err(AuthError::Sampler)?;
            sample_into(coins, opening, attempt, &fs, (b + ((sw + sl) * N) as u64) as u32, (so * N) as u32, uo.as_mut_ptr().add(r * so * N)).map_err(AuthError::Sampler)?;
        }
    }
    Ok((uw, ul, uo))
}

/// Draw the masks of attempts 0..count on device 0 ahead of time, while
/// device memory allows (each attempt's masks are kept until used).
pub fn presample(p: &Profile, coins: &Coins, opening: u64, count: u16, runs: Option<&[(usize, usize)]>) {
    // Its own small thread pool: the PIOP running meanwhile is host-latency
    // bound and uses the global pool.
    match rayon::ThreadPoolBuilder::new().num_threads(4).build() {
        Ok(pool) => pool.install(|| presample_in(p, coins, opening, count, runs)),
        Err(_) => presample_in(p, coins, opening, count, runs),
    }
}
fn presample_in(p: &Profile, coins: &Coins, opening: u64, count: u16, runs: Option<&[(usize, usize)]>) {
    almost_goldilocks_cuda::set_device(0).expect("set device");
    // This thread's sampling kernels yield to the PIOP's.
    unsafe { ffi::agl_gauss_low_priority(1); }
    let bytes = 2 * (p.source_cols() + p.link_cols() + p.aux_cols()) * N * 8;
    let t = Instant::now();
    let mut done = 0;
    for attempt in 0..count {
        // Driver-free plus the caching allocator's free lists (released on demand).
        // Keep the masks on the device if 16 GB stay free for the link; else
        // sample there (one attempt's buffers at a time) and park them on the host.
        let free = almost_goldilocks_cuda::mem_get_info().unwrap_or((0, 0)).0 + unsafe { almost_goldilocks_cuda::ffi::agl_cached_bytes() };
        // Host parking needs large device-to-host copies, which delay the
        // PIOP's small synchronous transfers (MORPHEUS_PRESAMPLE_HOST=1 enables it).
        let reserve: usize = std::env::var("MORPHEUS_PRESAMPLE_RESERVE_GB").ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(4) << 30;
        let on_dev = free >= bytes + reserve;
        let host_ok = std::env::var("MORPHEUS_PRESAMPLE_HOST").is_ok();
        if !on_dev && (!host_ok || free < bytes + (2usize << 30)) { if std::env::var("MORPHEUS_TIMING").is_ok() { eprintln!("[auth] presample stops: {:.1} GB free", free as f64 / 1e9); } break; }
        match sample_masks(p, coins, opening, attempt, runs) {
            Ok(m) => {
                let pre = if on_dev { Pre::Dev(m) } else { Pre::Host(to_host(&m.0), to_host(&m.1), to_host(&m.2)) };
                PRESAMPLED.lock().unwrap().push(((opening, attempt), pre)); done += 1;
            }
            Err(_) => break,
        }
    }
    unsafe { ffi::agl_gauss_low_priority(0); }
    if std::env::var("MORPHEUS_TIMING").is_ok() { eprintln!("[auth] presampled {done} attempts for opening {opening} in {:.2} s", t.elapsed().as_secs_f64()); }
}
fn take_presampled(opening: u64, attempt: u16) -> Option<Masks> {
    let pre = {
        let mut g = PRESAMPLED.lock().unwrap();
        let i = g.iter().position(|(k, _)| *k == (opening, attempt))?;
        g.swap_remove(i).1
    };
    Some(match pre {
        Pre::Dev(m) => m,
        Pre::Host(w, l, o) => (DeviceBuffer::from_slice(&w).unwrap(), DeviceBuffer::from_slice(&l).unwrap(), DeviceBuffer::from_slice(&o).unwrap()),
    })
}
/// Drop any unused presampled masks.
pub fn clear_presampled() { PRESAMPLED.lock().unwrap().clear(); }

/// One attempt's first messages, verdict and (device) responses.
struct AttemptOut { verdict: Coin, d: Vec<Tuple>, d_o: Vec<AuxTuple>, z: (DeviceBuffer<i64>, DeviceBuffer<i64>, DeviceBuffer<i64>) }

/// Attempt `attempt` on the current device. Its masks, challenges and coin
/// depend on the attempt index only, so attempts are independent.
#[allow(clippy::too_many_arguments)]
fn run_attempt(
    p: &Profile, ad: &AuthDev, coins: &Coins, opening: u64, tr: &MTranscript, pt: &Point, attempt: u16, timing: &mut AuthTiming,
) -> Result<AttemptOut, AuthError> {
    let (keys, sources, s_o) = (ad.keys, ad.sources, ad.s_o);
    let fs = FastSigma::new(p.sigma);
    let dp = DevPoint::new(p, pt);
    let (sw, sl, so) = (p.source_cols(), p.link_cols(), p.aux_cols());
    let row = ((sw + sl + so) * N) as u64;
    let n = sources.len();
    let degs = p.degs();
    let half = K8::ONE.scale((Q + 1) / 2);
    {
        timing.attempts += 1;
        let t0 = Instant::now();
        // U blocks laid out [row][cols][N] so both rows commit in one pass.
        let (uw, ul, uo) = match take_presampled(opening, attempt) { Some(m) if ad.dev == 0 => m, _ => sample_masks(p, coins, opening, attempt, ad.runs)? };
        let _ = (&fs, row);
        timing.sample_ms += t0.elapsed().as_secs_f64() * 1e3;
        let t0 = Instant::now();
        let (d, d_o) = first_message(p, keys, None, &dp, &uw, &ul, &uo, ad.runs);
        let _ = &dp.degs;
        timing.first_msg_ms += t0.elapsed().as_secs_f64() * 1e3;
        let t0 = Instant::now();
        let beta = super::auth::challenges_pub(tr, &d, &d_o, attempt, n);
        let mut zw = uw.clone_on_device().unwrap();
        let mut zl = ul.clone_on_device().unwrap();
        let mut zo = uo.clone_on_device().unwrap();
        assert_eq!(p.tau, 2);
        let b2 = |e: usize| -> [i8; 128] { std::array::from_fn(|i| beta[i / 64][e][i % 64]) };
        // All sources in one pass per column block.
        let bb: Vec<i8> = (0..sources.len()).flat_map(b2).collect();
        let vw: Vec<*const i8> = sources.iter().map(|s| s.w.as_ptr()).collect();
        let vl: Vec<*const i8> = sources.iter().map(|s| s.r.as_ptr()).collect();
        unsafe {
            let rc = ffi::agl_mauth_combine_multi(vw.as_ptr(), vw.len() as i32, sw as i32, bb.as_ptr(), zw.as_mut_ptr(), zw.as_mut_ptr().add(sw * N));
            assert_eq!(rc, 0);
            let rc = ffi::agl_mauth_combine_multi(vl.as_ptr(), vl.len() as i32, sl as i32, bb.as_ptr(), zl.as_mut_ptr(), zl.as_mut_ptr().add(sl * N));
            assert_eq!(rc, 0);
        }
        let bb = b2(n);
        unsafe { assert_eq!(ffi::agl_mauth_combine(s_o.as_ptr(), so as i32, bb.as_ptr(), zo.as_mut_ptr(), zo.as_mut_ptr().add(so * N)), 0); }
        timing.respond_ms += t0.elapsed().as_secs_f64() * 1e3;
        let t0 = Instant::now();
        // Caps per row and block; joint norms.
        let mut caps = true;
        let (mut nu, mut nz) = (0u128, 0u128);
        for (ub, zb, cols) in [(&uw, &zw, sw), (&ul, &zl, sl), (&uo, &zo, so)] {
            for r in 0..p.tau {
                let (zn, zmax) = norm(unsafe { zb.as_ptr().add(r * cols * N) }, cols * N);
                let (un, _) = norm(unsafe { ub.as_ptr().add(r * cols * N) }, cols * N);
                nu += un;
                nz += zn;
                if zmax >= p.b_z || zn > p.r2(cols * N) { caps = false; }
            }
        }
        let mut verdict = Coin::Reject;
        if caps {
            let x = Ratio { num: BigInt::from(nu) - BigInt::from(nz), den: BigInt::from(2u128 * p.sigma as u128 * p.sigma as u128) };
            let mut a = Address::new(Purpose::ProtocolCoin, opening);
            a.attempt = attempt;
            let mut st = coins.stream(a);
            let mut ub = BigUint::from(0u8);
            for _ in 0..rejection::B { ub = (ub << 1u32) + BigUint::from(st.next_bit().map_err(|_| AuthError::CoinAbort)?); }
            verdict = rejection::decide(&x, &ub);
        }
        timing.decide_ms += t0.elapsed().as_secs_f64() * 1e3;
        Ok(AttemptOut { verdict, d, d_o, z: (zw, zl, zo) })
    }
}

/// A device's sources for the sharded authentication: physical columns of
/// sources `srcs` at `w` (stride `len` bytes, in that order) and their link
/// columns `r` (same order).
/// `w`: each source's `len` bytes of int8 physical columns, or of packed
/// physical columns (1024 bytes per column) when `packed`.
pub struct ShardSrc { pub dev: i32, pub srcs: Vec<usize>, pub w: *const i8, pub len: usize, pub packed: bool, pub r: DeviceBuffer<i8> }
unsafe impl Send for ShardSrc {}
unsafe impl Sync for ShardSrc {}

/// Attempt `attempt` with sources spread over devices: masks, first message
/// and decision on device 0 (`keys`, `s_o`); each device adds its sources'
/// share of the responses, summed on the host. Same output as `run_attempt`.
#[allow(clippy::too_many_arguments)]
fn run_attempt_sharded(
    p: &Profile, keys: &Keys, seed: Option<&KeySeed>, shards: &[ShardSrc], s_o: &DeviceBuffer<i8>, n: usize, coins: &Coins, opening: u64, tr: &MTranscript, pt: &Point, attempt: u16, runs: Option<&[(usize, usize)]>, timing: &mut AuthTiming,
) -> Result<AttemptOut, AuthError> {
    almost_goldilocks_cuda::set_device(0).expect("set device");
    let fs = FastSigma::new(p.sigma);
    let dp = DevPoint::new(p, pt);
    let (sw, sl, so) = (p.source_cols(), p.link_cols(), p.aux_cols());
    let row = ((sw + sl + so) * N) as u64;
    timing.attempts += 1;
    let t0 = Instant::now();
    let (uw, ul, uo) = match take_presampled(opening, attempt) { Some(m) => m, None => sample_masks(p, coins, opening, attempt, runs)? };
    let _ = (&fs, row);
    timing.sample_ms += t0.elapsed().as_secs_f64() * 1e3;
    let t0 = Instant::now();
    let (d, d_o) = first_message(p, keys, seed, &dp, &uw, &ul, &uo, runs);
    timing.first_msg_ms += t0.elapsed().as_secs_f64() * 1e3;
    let t0 = Instant::now();
    let beta = super::auth::challenges_pub(tr, &d, &d_o, attempt, n);
    assert_eq!(p.tau, 2);
    let b2 = |e: usize| -> [i8; 128] { std::array::from_fn(|i| beta[i / 64][e][i % 64]) };
    // Responses: device 0 adds the one global mask and its sources; the other
    // devices add only their sources' shifts from zero and ship them as i32.
    // Bound: every source/link digit has |v| ≤ B_x − 1 = 1 and a C_0
    // challenge has 64 coefficients of size ≤ 2, so one source adds at most
    // 128 per coordinate and a device partial is at most 128·#sources. The
    // guard below keeps that below 2^31; the host sum is in i64.
    for sh in shards.iter().filter(|sh| sh.dev != 0) {
        assert!((sh.srcs.len() as u64) * 128 < (1u64 << 31), "i32 response partial could overflow: {} sources on device {}", sh.srcs.len(), sh.dev);
    }
    let mut zw = uw.clone_on_device().unwrap();
    let mut zl = ul.clone_on_device().unwrap();
    // All of a shard's sources in one pass per block (sources are contiguous
    // at strides sh.len and sl·N).
    let combine = |sh: &ShardSrc, zw: &mut DeviceBuffer<i64>, zl: &mut DeviceBuffer<i64>| {
        if sh.srcs.is_empty() { return; }
        let bb: Vec<i8> = sh.srcs.iter().flat_map(|&e| b2(e)).collect();
        let ns = sh.srcs.len();
        let vw: Vec<*const i8> = (0..ns).map(|j| unsafe { sh.w.add(j * sh.len) }).collect();
        let vl: Vec<*const i8> = (0..ns).map(|j| unsafe { sh.r.as_ptr().add(j * sl * N) }).collect();
        unsafe {
            let rc = if sh.packed { ffi::agl_mauth_combine_packed(vw.as_ptr() as *const *const u8, ns as i32, sw as i32, bb.as_ptr(), zw.as_mut_ptr(), zw.as_mut_ptr().add(sw * N)) }
                else { ffi::agl_mauth_combine_multi(vw.as_ptr(), ns as i32, sw as i32, bb.as_ptr(), zw.as_mut_ptr(), zw.as_mut_ptr().add(sw * N)) };
            assert_eq!(rc, 0);
            let rc = ffi::agl_mauth_combine_multi(vl.as_ptr(), ns as i32, sl as i32, bb.as_ptr(), zl.as_mut_ptr(), zl.as_mut_ptr().add(sl * N));
            assert_eq!(rc, 0);
        }
    };
    let parts: Vec<(Vec<i32>, Vec<i32>)> = std::thread::scope(|sc| {
        let hs: Vec<_> = shards.iter().filter(|sh| sh.dev != 0).map(|sh| {
            let combine = &combine;
            sc.spawn(move || {
                almost_goldilocks_cuda::set_device(sh.dev).expect("set device");
                let mut pw = DeviceBuffer::<i64>::new(2 * sw * N).unwrap();
                let mut pl = DeviceBuffer::<i64>::new(2 * sl * N).unwrap();
                pw.zero().unwrap(); pl.zero().unwrap();
                combine(sh, &mut pw, &mut pl);
                let narrow = |b: &DeviceBuffer<i64>| -> Vec<i32> {
                    let mut d = DeviceBuffer::<i32>::new(b.len()).unwrap();
                    assert_eq!(unsafe { ffi::agl_i64_to_i32(b.as_ptr(), d.as_mut_ptr(), b.len() as u64) }, 0);
                    // Pinned staged readback into pre-faulted pages.
                    let mut v: Vec<i32> = Vec::with_capacity(b.len());
                    unsafe { v.set_len(b.len()); }
                    { use rayon::prelude::*; v.par_chunks_mut(1 << 18).for_each(|c| { let mut i = 0; while i < c.len() { c[i] = 0; i += 1024; } }); }
                    assert_eq!(unsafe { ffi::agl_d2h_staged(v.as_mut_ptr() as *mut c_void, d.as_ptr() as *const c_void, v.len() * 4) }, 0);
                    v
                };
                (narrow(&pw), narrow(&pl))
            })
        }).collect();
        almost_goldilocks_cuda::set_device(0).expect("set device");
        for sh in shards.iter().filter(|sh| sh.dev == 0) { combine(sh, &mut zw, &mut zl); }
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    almost_goldilocks_cuda::set_device(0).expect("set device");
    for (pw, pl) in &parts {
        for (dst, src) in [(&mut zw, pw), (&mut zl, pl)] {
            let mut d = DeviceBuffer::<i32>::new(src.len()).unwrap();
            assert_eq!(unsafe { ffi::agl_h2d_staged(d.as_mut_ptr() as *mut c_void, src.as_ptr() as *const c_void, src.len() * 4) }, 0);
            assert_eq!(unsafe { ffi::agl_add_i32_i64(dst.as_mut_ptr(), d.as_ptr(), src.len() as u64) }, 0);
        }
    }
    let mut zo = uo.clone_on_device().unwrap();
    let bb = b2(n);
    unsafe { assert_eq!(ffi::agl_mauth_combine(s_o.as_ptr(), so as i32, bb.as_ptr(), zo.as_mut_ptr(), zo.as_mut_ptr().add(so * N)), 0); }
    timing.respond_ms += t0.elapsed().as_secs_f64() * 1e3;
    let t0 = Instant::now();
    let mut caps = true;
    let (mut nu, mut nz) = (0u128, 0u128);
    for (ub, zb, cols) in [(&uw, &zw, sw), (&ul, &zl, sl), (&uo, &zo, so)] {
        for r in 0..p.tau {
            let (zn, zmax) = norm(unsafe { zb.as_ptr().add(r * cols * N) }, cols * N);
            let (un, _) = norm(unsafe { ub.as_ptr().add(r * cols * N) }, cols * N);
            nu += un;
            nz += zn;
            if zmax >= p.b_z || zn > p.r2(cols * N) { caps = false; }
        }
    }
    let mut verdict = Coin::Reject;
    if caps {
        let x = Ratio { num: BigInt::from(nu) - BigInt::from(nz), den: BigInt::from(2u128 * p.sigma as u128 * p.sigma as u128) };
        let mut a = Address::new(Purpose::ProtocolCoin, opening);
        a.attempt = attempt;
        let mut st = coins.stream(a);
        let mut ub = BigUint::from(0u8);
        for _ in 0..rejection::B { ub = (ub << 1u32) + BigUint::from(st.next_bit().map_err(|_| AuthError::CoinAbort)?); }
        verdict = rejection::decide(&x, &ub);
    }
    timing.decide_ms += t0.elapsed().as_secs_f64() * 1e3;
    Ok(AttemptOut { verdict, d, d_o, z: (zw, zl, zo) })
}

/// Sharded prover (attempts in order, each over every device); equals
/// `prove` on the same inputs.
#[allow(clippy::too_many_arguments)]
pub fn prove_sharded(
    p: &Profile, keys: &Keys, seed: Option<&KeySeed>, shards: &[ShardSrc], s_o: &DeviceBuffer<i8>, n: usize, coins: &Coins, opening: u64, tr: &MTranscript, pt: &Point, keep_z: bool, runs: Option<&[(usize, usize)]>, timing: &mut AuthTiming,
) -> Result<AuthOut, AuthError> {
    for attempt in 0..p.attempts as u16 {
        let o = run_attempt_sharded(p, keys, seed, shards, s_o, n, coins, opening, tr, pt, attempt, runs, timing)?;
        match o.verdict {
            Coin::Accept => { let t = Instant::now(); let out = accept(p, attempt, o, keep_z, runs); timing.extract_ms += t.elapsed().as_secs_f64() * 1e3; return Ok(out); }
            Coin::Reject => {}
            Coin::Abort => return Err(AuthError::CoinAbort),
        }
    }
    Err(AuthError::Exhausted)
}

/// (f_x, f_ox) of τ rows of device blocks laid out [row][cols][N] (i64):
/// the prover's first messages on the masks, the verifier's left sides on z.
/// A·u over the payload and hiding columns of both source rows. Without a
/// resident source key cache (it was freed for the link), the key is
/// generated from `seed` in column chunks and the NTT-domain partial
/// products are summed; the result is the same.
fn source_product(p: &Profile, keys: &Keys, seed: Option<&KeySeed>, uw: &DeviceBuffer<i64>, runs: Option<&[(usize, usize)]>) -> DeviceBuffer<u64> {
    let (sw, c) = (p.source_cols(), p.source_cols() - p.rank);
    if let (Some(r), true) = (runs, keys.w.buf.len() > 1) {
        // Only the support columns are nonzero: gather them (the identity
        // error columns, always in the support, come last) and index the key.
        let k = run_cols(r);
        if k < sw {
            let g = gather(uw, sw, r);
            let cols: Vec<u32> = r.iter().flat_map(|&(a, b)| (a..b).map(|x| x as u32)).filter(|&x| (x as usize) < c).collect();
            assert_eq!(cols.len(), k - p.rank);
            let idx = DeviceBuffer::from_slice(&cols).unwrap();
            return finish(partial_raw(&keys.w, 5, g.as_ptr() as *const c_void, 8, 2, cols.len(), k * N, Some(&idx)), 2, 5, p.rank, None);
        }
    }
    if keys.w.buf.len() > 1 {
        return finish(partial_raw(&keys.w, 5, uw.as_ptr() as *const c_void, 8, 2, c, sw * N, None), 2, 5, p.rank, None);
    }
    let seed = seed.expect("source key cache dropped and no key seed for chunked generation");
    let chunk: usize = std::env::var("MORPHEUS_W_CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(4096).max(1);
    let mut acc: Option<DeviceBuffer<u32>> = None;
    for c0 in (0..c).step_by(chunk) {
        let nc = chunk.min(c - c0);
        let key = KeyCache::build(seed, KeyKind::Source, p.rank, c0 as u64, nc, 5);
        let part = partial_raw(&key, 5, unsafe { uw.as_ptr().add(c0 * N) } as *const c_void, 8, 2, nc, sw * N, None);
        match acc.as_mut() { Some(a) => super::commit_gpu::add(a, &part, 2, 5, p.rank), None => acc = Some(part) }
    }
    finish(acc.expect("no source columns"), 2, 5, p.rank, None)
}

fn first_message(p: &Profile, keys: &Keys, seed: Option<&KeySeed>, dp: &DevPoint, uw: &DeviceBuffer<i64>, ul: &DeviceBuffer<i64>, uo: &DeviceBuffer<i64>, runs: Option<&[(usize, usize)]>) -> (Vec<Tuple>, Vec<AuxTuple>) {
    let (sw, sl, so) = (p.source_cols(), p.link_cols(), p.aux_cols());
    let degs = p.degs();
    let half = K8::ONE.scale((Q + 1) / 2);
    let cw = ring_rows(source_product(p, keys, seed, uw, runs).to_vec().unwrap(), p.rank);
    let cl = ring_rows(finish(partial_raw(&keys.l, 5, ul.as_ptr() as *const c_void, 8, 2, sl - p.rank, sl * N, None), 2, 5, p.rank, None).to_vec().unwrap(), p.rank);
    let co = ring_rows(finish(partial_raw(&keys.o, 5, uo.as_ptr() as *const c_void, 8, 2, so - p.rank, so * N, None), 2, 5, p.rank, None).to_vec().unwrap(), p.rank);
    let (mut d, mut d_o) = (Vec::new(), Vec::new());
    for r in 0..p.tau {
        // The identity error columns are the last `rank` columns of each block.
        let add_err = |c: &Vec<Vec<u64>>, base: *const i64, cols: usize| -> Vec<Vec<u64>> {
            let mut e = vec![0i64; p.rank * N];
            let src = unsafe { base.add((r * cols + cols - p.rank) * N) };
            let rc = unsafe { ffi::cuda_memcpy_dtoh(e.as_mut_ptr() as *mut c_void, src as *const c_void, p.rank * N * 8) };
            assert_eq!(rc, 0);
            c.iter().enumerate().map(|(i, row)| row.iter().enumerate().map(|(j, &v)| {
                let x = e[i * N + j];
                let xe = if x >= 0 { x as u64 } else { Q - x.unsigned_abs() };
                ((v as u128 + xe as u128) % Q as u128) as u64
            }).collect()).collect()
        };
        let c = add_err(&cw[r], uw.as_ptr(), sw);
        let g = add_err(&cl[r], ul.as_ptr(), sl);
        let hl = h_of(&decode(unsafe { ul.as_ptr().add(r * sl * N) }, A_ENC_COLS), 0);
        let a = dp.ev(unsafe { uw.as_ptr().add(r * sw * N) } as *const c_void, 8).add(&hl.scale_k(&dp.rho_f));
        d.push(Tuple { c, g, a });
        let ncoef = p.aux_coeffs;
        let dec = decode(unsafe { uo.as_ptr().add(r * so * N) }, A_ENC_COLS * ncoef);
        let mut t = AElem::zero();
        for (i, &dg) in degs.iter().enumerate() {
            let mut xp = K8::ONE;
            for k in 1..=dg {
                xp = xp.mul(&dp.x[i]);
                t = t.add(&h_of(&dec, A_ENC_COLS * aux_index(&degs, i, k)).scale_k(&xp.sub(&half)));
            }
        }
        d_o.push(AuxTuple { c: add_err(&co[r], uo.as_ptr(), so), t });
    }
    (d, d_o)
}

type AuthOut = (AuthProof, Option<(DeviceBuffer<i64>, DeviceBuffer<i64>, DeviceBuffer<i64>)>);

fn accept(p: &Profile, attempt: u16, o: AttemptOut, keep_z: bool, runs: Option<&[(usize, usize)]>) -> AuthOut {
    let (zw, zl, zo) = o.z;
    // Serialize on the device (37 bits per coefficient), one D2H copy per
    // block; the source block carries only its support columns.
    let w = |zw: &DeviceBuffer<i64>| match runs { Some(r) if run_cols(r) < p.source_cols() => pack_dev(p, &gather(zw, p.source_cols(), r)), _ => pack_dev(p, zw) };
    let z = if keep_z { Responses { w: w(&zw), l: pack_dev(p, &zl), o: pack_dev(p, &zo) } } else { Responses::default() };
    (AuthProof { attempt, d: o.d, d_o: o.d_o, z }, Some((zw, zl, zo)))
}

fn pack_dev(p: &Profile, z: &DeviceBuffer<i64>) -> Vec<u8> {
    let len = super::wire::packed_len(z.len());
    let mut b = DeviceBuffer::<u8>::new(len).unwrap();
    assert_eq!(unsafe { ffi::agl_mauth_pack37(z.as_ptr(), z.len() as u64, super::wire::z_off(p), b.as_mut_ptr()) }, 0);
    // Fault the destination's pages in parallel (a fresh GB-scale vector
    // otherwise takes its page faults serially inside the copy), then a
    // staged pinned copy.
    let mut v: Vec<u8> = Vec::with_capacity(len);
    unsafe { v.set_len(len); }
    { use rayon::prelude::*; v.par_chunks_mut(1 << 20).for_each(|c| { let mut i = 0; while i < c.len() { c[i] = 0; i += 4096; } }); }
    assert_eq!(unsafe { ffi::agl_d2h_staged(v.as_mut_ptr() as *mut std::ffi::c_void, b.as_ptr() as *const std::ffi::c_void, len) }, 0);
    v
}

/// Device responses of one block from the wire; None if a code is outside
/// the support or the length is wrong.
fn unpack_dev(p: &Profile, b: &[u8], cols: usize) -> Option<DeviceBuffer<i64>> {
    let n = p.tau * cols * N;
    if b.len() != super::wire::packed_len(n) { return None; }
    let d = DeviceBuffer::from_slice(b).unwrap();
    let mut z = DeviceBuffer::<i64>::new(n).unwrap();
    let rc = unsafe { ffi::agl_mauth_unpack37(d.as_ptr(), n as u64, super::wire::z_off(p), super::wire::z_umax(p), z.as_mut_ptr()) };
    assert!(rc >= 0, "unpack37 failed");
    (rc == 0).then_some(z)
}

/// GPU prover; equals `auth::prove` on the same inputs.
#[allow(clippy::too_many_arguments)]
pub fn prove(
    p: &Profile, keys: &Keys, coins: &Coins, opening: u64, tr: &MTranscript, pt: &Point,
    sources: &[DevSource], s_o: &DeviceBuffer<i8>, keep_z: bool, timing: &mut AuthTiming,
) -> Result<AuthOut, AuthError> {
    let ad = AuthDev { dev: -1, keys, sources, s_o, runs: None };
    for attempt in 0..p.attempts as u16 {
        let o = run_attempt(p, &ad, coins, opening, tr, pt, attempt, timing)?;
        match o.verdict {
            Coin::Accept => { let t = Instant::now(); let out = accept(p, attempt, o, keep_z, ad.runs); timing.extract_ms += t.elapsed().as_secs_f64() * 1e3; return Ok(out); }
            Coin::Reject => {}
            Coin::Abort => return Err(AuthError::CoinAbort),
        }
    }
    Err(AuthError::Exhausted)
}

/// Same output as `prove`, with attempts run in batches of `devs.len()`, one
/// per device; the first non-rejected attempt in index order decides.
#[allow(clippy::too_many_arguments)]
pub fn prove_multi(
    p: &Profile, devs: &[AuthDev], coins: &Coins, opening: u64, tr: &MTranscript, pt: &Point, keep_z: bool, timing: &mut AuthTiming,
) -> Result<AuthOut, AuthError> {
    let g = devs.len().max(1);
    let total = p.attempts as u16;
    let mut base = 0u16;
    while base < total {
        let outs: Vec<(u16, Result<AttemptOut, AuthError>, AuthTiming)> = std::thread::scope(|sc| {
            let hs: Vec<_> = devs.iter().enumerate().filter(|(j, _)| base as usize + j < total as usize).map(|(j, ad)| {
                let attempt = base + j as u16;
                sc.spawn(move || {
                    if ad.dev >= 0 { almost_goldilocks_cuda::set_device(ad.dev).expect("set device"); }
                    let mut tm = AuthTiming::default();
                    let r = run_attempt(p, ad, coins, opening, tr, pt, attempt, &mut tm);
                    (attempt, r, tm)
                })
            }).collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let mut done: Option<Result<AuthOut, AuthError>> = None;
        for (attempt, r, tm) in outs {
            if done.is_some() { continue; }
            timing.attempts += tm.attempts;
            timing.sample_ms += tm.sample_ms; timing.first_msg_ms += tm.first_msg_ms;
            timing.respond_ms += tm.respond_ms; timing.decide_ms += tm.decide_ms;
            match r {
                Err(e) => done = Some(Err(e)),
                Ok(o) => match o.verdict {
                    Coin::Accept => {
                        // The accepted responses live on that attempt's device.
                        let dev = devs[(attempt - base) as usize].dev;
                        if dev >= 0 { almost_goldilocks_cuda::set_device(dev).expect("set device"); }
                        let t = Instant::now();
                        done = Some(Ok(accept(p, attempt, o, keep_z, devs[0].runs)));
                        timing.extract_ms += t.elapsed().as_secs_f64() * 1e3;
                        if devs[0].dev >= 0 { almost_goldilocks_cuda::set_device(devs[0].dev).expect("set device"); }
                    }
                    Coin::Reject => {}
                    Coin::Abort => done = Some(Err(AuthError::CoinAbort)),
                },
            }
        }
        if let Some(d) = done { return d; }
        base += g as u16;
    }
    Err(AuthError::Exhausted)
}

/// GPU verifier of the masked authentication (same verdict as `auth::verify`).
#[allow(clippy::too_many_arguments)]
pub fn verify(p: &Profile, keys: &Keys, tr: &MTranscript, pt: &Point, targets: &[Tuple], target_o: &AuxTuple, pf: &AuthProof) -> bool {
    verify_runs(p, keys, tr, pt, targets, target_o, pf, None)
}
/// `verify` with source responses on the public support `runs` only
/// (virtual zero extension; same verdict as `auth_compact::verify`).
#[allow(clippy::too_many_arguments)]
pub fn verify_runs(p: &Profile, keys: &Keys, tr: &MTranscript, pt: &Point, targets: &[Tuple], target_o: &AuxTuple, pf: &AuthProof, runs: Option<&[(usize, usize)]>) -> bool {
    use super::auth::{act_tuple, add_ring, add_tuple};
    use super::pcs_maps::{act_c0_a, act_c0_ring_q};
    use rayon::prelude::*;
    let wcols = runs.map_or(p.source_cols(), run_cols);
    if !super::wire::auth_shape_ok_cols(p, pf, wcols) || targets.len() > p.n_max
        || pt.xi.len() != p.ell() || !pt.xi.iter().all(super::wire::canonical_k)
        || !super::wire::canonical_k(&pt.rho) || pt.rho == K8::ZERO || pt.rho == K8::ONE
        || !targets.iter().all(|t| super::wire::commitment_ok(p, &t.c)
            && super::wire::commitment_ok(p, &t.g) && super::wire::endpoint_ok(&t.a))
        || !super::wire::commitment_ok(p, &target_o.c) || !super::wire::endpoint_ok(&target_o.t) { return false; }
    let (sw, sl, so) = (p.source_cols(), p.link_cols(), p.aux_cols());
    if (pf.attempt as usize) >= p.attempts || pf.d.len() != p.tau || pf.d_o.len() != p.tau {
        return false;
    }
    let n = targets.len();
    let beta = super::auth::challenges_pub(tr, &pf.d, &pf.d_o, pf.attempt, n);
    let (zw, zl, zo) = match (unpack_dev(p, &pf.z.w, wcols), unpack_dev(p, &pf.z.l, sl), unpack_dev(p, &pf.z.o, so)) {
        (Some(a), Some(b), Some(c)) => (a, b, c),
        _ => return false,
    };
    let zw = match runs { Some(r) if wcols < sw => scatter(&zw, sw, r), _ => zw };
    // Caps per row and block.
    for (zb, cols) in [(&zw, sw), (&zl, sl), (&zo, so)] {
        for r in 0..p.tau {
            let (zn, zmax) = norm(unsafe { zb.as_ptr().add(r * cols * N) }, cols * N);
            if zmax >= p.b_z || zn > p.r2(cols * N) { return false; }
        }
    }
    let dp = DevPoint::new(p, pt);
    let (lhs, lhs_o) = first_message(p, keys, None, &dp, &zw, &zl, &zo, runs);
    for r in 0..p.tau {
        let acted: Vec<Tuple> = targets.par_iter().enumerate().map(|(e, t)| act_tuple(&beta[r][e], t)).collect();
        let rhs = acted.iter().fold(pf.d[r].clone(), |acc, t| add_tuple(&acc, t));
        if lhs[r] != rhs { return false; }
        let bt = &beta[r][n];
        let ro = AuxTuple {
            c: add_ring(&pf.d_o[r].c, &target_o.c.par_iter().map(|x| act_c0_ring_q(bt, x)).collect::<Vec<_>>()),
            t: pf.d_o[r].t.add(&act_c0_a(bt, &target_o.t)),
        };
        if lhs_o[r] != ro { return false; }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::auth::{self, f_ox, f_x, SourceVec};
    use super::super::coins::CoinProfile;
    use super::super::pcs_maps::enc;
    use super::super::sampler::ternary_vec;
    use rand::{Rng, SeedableRng};

    fn rk<R: rand::Rng>(rng: &mut R) -> K8 { K8(std::array::from_fn(|_| rng.gen::<u64>() % Q)) }
    fn to_dev(cols: &[Vec<i64>]) -> DeviceBuffer<i8> {
        DeviceBuffer::from_slice(&cols.iter().flatten().map(|&v| v as i8).collect::<Vec<_>>()).unwrap()
    }

    #[test]
    fn gpu_authentication_equals_cpu_authentication() {
        let p = Profile::toy();
        let seed = KeySeed([3; 8]);
        let coins = Coins::new([9u8; 64], CoinProfile::Ternary);
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(2);
        let tern = |rng: &mut rand_chacha::ChaCha20Rng, cols: usize| -> Vec<Vec<i64>> { (0..cols).map(|_| ternary_vec(rng, N).unwrap()).collect() };
        let sources: Vec<SourceVec> = (0..2).map(|_| {
            let w = tern(&mut rng, p.source_cols());
            let u = AElem((0..64).map(|_| rk(&mut rng)).collect());
            SourceVec { w, r: [enc(&u), tern(&mut rng, p.hiding + p.rank)].concat() }
        }).collect();
        let aux_payload: Vec<Vec<i64>> = (0..p.aux_coeffs).flat_map(|_| enc(&AElem((0..64).map(|_| rk(&mut rng)).collect()))).collect();
        let s_o = [aux_payload, tern(&mut rng, p.hiding + p.rank)].concat();
        let pt = Point { xi: (0..p.ell()).map(|_| rk(&mut rng)).collect(), rho: rk(&mut rng) };
        let targets: Vec<Tuple> = sources.iter().map(|s| f_x(&p, &seed, &pt, &s.w, &s.r)).collect();
        let target_o = f_ox(&p, &seed, &pt, &s_o);
        let mut tr = MTranscript::new(p.name, &coins.salt(0));
        for t in &targets { for c in &t.c { tr.absorb(Item::SourceCommitment, c); } }
        let cpu = auth::prove(&p, &seed, &coins, 0, &tr, &pt, &sources, &s_o).unwrap();

        let keys = Keys::build(&p, &seed);
        let dsrc: Vec<DevSource> = sources.iter().map(|s| DevSource { w: to_dev(&s.w), r: to_dev(&s.r) }).collect();
        let mut tm = AuthTiming::default();
        let (gpu, _) = prove(&p, &keys, &coins, 0, &tr, &pt, &dsrc, &to_dev(&s_o), true, &mut tm).unwrap();
        assert_eq!(gpu.attempt, cpu.attempt);
        for r in 0..2 { eprintln!("row {r}: c {} g {} a {} | o.c {} o.t {}", gpu.d[r].c == cpu.d[r].c, gpu.d[r].g == cpu.d[r].g, gpu.d[r].a == cpu.d[r].a, gpu.d_o[r].c == cpu.d_o[r].c, gpu.d_o[r].t == cpu.d_o[r].t); }
        assert_eq!(gpu.d, cpu.d);
        assert_eq!(gpu.d_o, cpu.d_o);
        assert_eq!(gpu.z, cpu.z);
        assert!(auth::verify(&p, &seed, &tr, &pt, &targets, &target_o, &gpu));
        let _ = rng.gen::<u8>();
    }

    #[test]
    fn gpu_compact_authentication_equals_cpu_compact() {
        use super::super::auth_compact::{self, Support};
        let p = Profile::toy();
        let seed = KeySeed([5; 8]);
        let coins = Coins::new([11u8; 64], CoinProfile::Ternary);
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(7);
        let support = Support::from_public_columns(&p, vec![0, 2, 3, 4, 5, 6, 7]).unwrap();
        let tern = |rng: &mut rand_chacha::ChaCha20Rng, cols: usize| -> Vec<Vec<i64>> { (0..cols).map(|_| ternary_vec(rng, N).unwrap()).collect() };
        let mut raw = Vec::new();
        let mut sources = Vec::new();
        for _ in 0..2 {
            let mut w = tern(&mut rng, p.source_cols());
            w[1] = vec![0; N];
            let mut r8 = vec![0i8; p.source_cols() * N];
            for c in 0..p.source_cols() { for i in 0..N { r8[c * N + i] = w[c][super::super::ring::phys_of_logical(i)] as i8; } }
            assert!((0..p.source_cols()).all(|c| auth_compact::raw_column(&r8, c) == w[c]));
            let u = AElem((0..64).map(|_| rk(&mut rng)).collect());
            sources.push(SourceVec { w, r: [enc(&u), tern(&mut rng, p.hiding + p.rank)].concat() });
            raw.push(r8);
        }
        let aux_payload: Vec<Vec<i64>> = (0..p.aux_coeffs).flat_map(|_| enc(&AElem((0..64).map(|_| rk(&mut rng)).collect()))).collect();
        let s_o = [aux_payload, tern(&mut rng, p.hiding + p.rank)].concat();
        let pt = Point { xi: (0..p.ell()).map(|_| rk(&mut rng)).collect(), rho: rk(&mut rng) };
        let targets: Vec<Tuple> = sources.iter().map(|s| f_x(&p, &seed, &pt, &s.w, &s.r)).collect();
        let target_o = f_ox(&p, &seed, &pt, &s_o);
        let mut tr = MTranscript::new(p.name, &coins.salt(0));
        for t in &targets { for c in &t.c { tr.absorb(Item::SourceCommitment, c); } }
        let links: Vec<Vec<Vec<i64>>> = sources.iter().map(|s| s.r.clone()).collect();
        let cpu = auth_compact::prove(&p, &seed, &coins, 0, &tr, &pt, &support, &raw, &links, &s_o).unwrap();

        let keys = Keys::build(&p, &seed);
        let runs = runs_of(&support);
        assert_eq!(runs, vec![(0, 1), (2, 8)]);
        let dsrc: Vec<DevSource> = sources.iter().map(|s| DevSource { w: to_dev(&s.w), r: to_dev(&s.r) }).collect();
        let so = to_dev(&s_o);
        let devs = [AuthDev { dev: -1, keys: &keys, sources: &dsrc, s_o: &so, runs: Some(&runs) }];
        let mut tm = AuthTiming::default();
        let (gpu, _) = prove_multi(&p, &devs, &coins, 0, &tr, &pt, true, &mut tm).unwrap();
        assert_eq!(gpu, cpu);
        assert!(verify_runs(&p, &keys, &tr, &pt, &targets, &target_o, &gpu, Some(&runs)));
        assert!(auth_compact::verify(&p, &seed, &tr, &pt, &support, &targets, &target_o, &gpu));
        assert!(!verify(&p, &keys, &tr, &pt, &targets, &target_o, &gpu));
        let mut bad = gpu.clone();
        bad.z.w[3] ^= 1;
        assert!(!verify_runs(&p, &keys, &tr, &pt, &targets, &target_o, &bad, Some(&runs)));
    }

    #[test]
    fn chunked_source_product_equals_cached_product() {
        let p = Profile::toy();
        let seed = KeySeed([5; 8]);
        let mut keys = Keys::build(&p, &seed);
        let sw = p.source_cols();
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(7);
        let u: Vec<i64> = (0..2 * sw * N).map(|_| rng.gen_range(-1_000_000_000i64..1_000_000_000)).collect();
        let uw = DeviceBuffer::from_slice(&u).unwrap();
        let full = source_product(&p, &keys, None, &uw, None).to_vec().unwrap();
        keys.w = KeyCache { buf: DeviceBuffer::<u32>::new(1).unwrap(), ..keys.w.clone_meta() };
        for chunk in ["1", "2", "100"] {
            std::env::set_var("MORPHEUS_W_CHUNK", chunk);
            assert_eq!(source_product(&p, &keys, Some(&seed), &uw, None).to_vec().unwrap(), full, "chunk {chunk}");
        }
        std::env::remove_var("MORPHEUS_W_CHUNK");
    }

    #[test]
    #[ignore]
    fn gpu_authentication_full_profile() {
        // Ternary profile, synthetic sources: time per attempt on one GPU.
        let p = Profile::ternary();
        let seed = KeySeed([3; 8]);
        let coins = Coins::new([9u8; 64], CoinProfile::Ternary);
        let n: usize = std::env::var("AUTH_SOURCES").ok().and_then(|s| s.parse().ok()).unwrap_or(2);
        let t = Instant::now();
        let keys = Keys::build(&p, &seed);
        println!("key caches: {:.1} s, {:.1} GB", t.elapsed().as_secs_f64(), (keys.w.bytes() + keys.l.bytes() + keys.o.bytes()) as f64 / 1e9);
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(1);
        let mut rnd8 = |len: usize| -> DeviceBuffer<i8> { DeviceBuffer::from_slice(&(0..len).map(|_| rng.gen_range(-1i8..=1)).collect::<Vec<_>>()).unwrap() };
        let dsrc: Vec<DevSource> = (0..n).map(|_| DevSource { w: rnd8(p.source_cols() * N), r: rnd8(p.link_cols() * N) }).collect();
        let s_o = rnd8(p.aux_cols() * N);
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(2);
        let pt = Point { xi: (0..p.ell()).map(|_| rk(&mut rng)).collect(), rho: rk(&mut rng) };
        let tr = MTranscript::new(p.name, &coins.salt(0));
        for op in 0..3u64 {
            let mut tm = AuthTiming::default();
            let t = Instant::now();
            let (pf, _) = prove(&p, &keys, &coins, op, &tr, &pt, &dsrc, &s_o, false, &mut tm).unwrap();
            println!("opening {op}: {n} sources, accepted at attempt {} in {:.2} s  {tm:?}", pf.attempt, t.elapsed().as_secs_f64());
        }
    }

    #[test]
    fn ev_and_decode_kernels_match_cpu() {
        let p = Profile::toy();
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(5);
        let pt = Point { xi: (0..p.ell()).map(|_| rk(&mut rng)).collect(), rho: rk(&mut rng) };
        let dp = DevPoint::new(&p, &pt);
        for big in [false, true] {
            let cols: Vec<Vec<i64>> = (0..p.source_cols()).map(|_| (0..N).map(|_| if big { rng.gen_range(-(1i64 << 40)..(1i64 << 40)) } else { rng.gen_range(-1i64..=1) }).collect()).collect();
            let flat: Vec<i64> = cols.iter().flatten().copied().collect();
            let d = DeviceBuffer::from_slice(&flat).unwrap();
            assert_eq!(dp.ev(d.as_ptr() as *const c_void, 8), super::super::pcs_maps::ev_raw(&cols, &pt.xi), "big={big}");
            let dec = decode(d.as_ptr(), 8);
            assert_eq!(h_of(&dec, 0), super::super::pcs_maps::decode_h(&cols[..8]), "decode big={big}");
        }
    }
}

// ---------------------------------------------------------------- column shards

/// The authentication's source block split by columns over devices: device
/// `dev` holds source columns c0..c1 of every source, packed two bits per
/// digit in physical order ([source][column][1024 bytes]). Masks, A·u,
/// Ev and responses of those columns stay on the device; only the partial
/// A·u products, Ev sums and norms (a few MB) cross devices.
pub struct ColShard { pub dev: i32, pub c0: usize, pub c1: usize, pub nsrc: usize, pub raw: DeviceBuffer<u8> }
unsafe impl Send for ColShard {}
unsafe impl Sync for ColShard {}

/// The source key over columns c0..c1 (payload and hiding columns only).
pub struct ColKey { pub dev: i32, pub c0: usize, pub c1: usize, pub key: Option<KeyCache> }
unsafe impl Send for ColKey {}
unsafe impl Sync for ColKey {}

/// Contiguous column ranges of the source block, one per device.
pub fn col_ranges(p: &Profile, g: usize) -> Vec<(usize, usize)> {
    let sw = p.source_cols();
    (0..g).map(|d| (sw * d / g, sw * (d + 1) / g)).collect()
}

/// Source keys of the column ranges (built once per process and device).
pub fn col_keys(p: &Profile, seed: &KeySeed, devs: &[i32]) -> Vec<ColKey> {
    let c = p.source_cols() - p.rank;
    std::thread::scope(|sc| {
        let hs: Vec<_> = devs.iter().zip(col_ranges(p, devs.len())).map(|(&dev, (c0, c1))| sc.spawn(move || {
            almost_goldilocks_cuda::set_device(dev).expect("set device");
            let key = (c0 < c.min(c1)).then(|| KeyCache::build(seed, KeyKind::Source, p.rank, c0 as u64, c.min(c1) - c0, 5));
            ColKey { dev, c0, c1, key }
        })).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    })
}

/// Column shards of `raw` (logical packed host sources) on `devs`.
pub fn col_shards(p: &Profile, raw: &dyn super::packed::Sources, devs: &[i32]) -> Vec<ColShard> {
    let n = raw.count();
    let out = std::thread::scope(|sc| {
        let hs: Vec<_> = devs.iter().zip(col_ranges(p, devs.len())).map(|(&dev, (c0, c1))| sc.spawn(move || {
            almost_goldilocks_cuda::set_device(dev).expect("set device");
            let nc = c1 - c0;
            let buf = DeviceBuffer::<u8>::new((n * nc * 1024).max(1)).expect("column shard");
            // Several copy threads per device (pageable copies of one thread
            // run at a few GB/s), each with its own staging buffer.
            let nt = std::env::var("MORPHEUS_COPY_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(4usize).max(1);
            let dst = buf.as_ptr() as usize;
            std::thread::scope(|s2| {
                for t in 0..nt {
                    s2.spawn(move || {
                        almost_goldilocks_cuda::set_device(dev).expect("set device");
                        let mut stage = DeviceBuffer::<u8>::new(nc * 1024).expect("column shard staging");
                        for s in (t..n).step_by(nt) {
                            let own;
                            let bytes: &[u8] = match raw.packed(s) { Some(x) => &x.b, None => { own = super::packed::Src::from_i8(&raw.to_i8(s)); &own.b } };
                            stage.copy_from_slice_at(0, &bytes[c0 * 1024..c1 * 1024]).expect("column upload");
                            let rc = unsafe { ffi::agl_morpheus_permute2(stage.as_ptr(), (dst as *mut u8).add(s * nc * 1024), nc as u64) };
                            assert_eq!(rc, 0, "permute2");
                        }
                    });
                }
            });
            ColShard { dev, c0, c1, nsrc: n, raw: buf }
        })).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    almost_goldilocks_cuda::set_device(0).unwrap();
    out
}

impl DevPoint {
    /// The 64×8 raw Ev sums (before g_k) of `ncols` columns at ptr starting
    /// at source column c0 (bytes 0: packed, 1: int8, 8: int64).
    pub fn ev_part(&self, ptr: *const c_void, bytes: i32, c0: usize, ncols: usize) -> Vec<u64> {
        let mut out = vec![0u64; 64 * 8];
        if ncols == 0 { return out; }
        let rc = unsafe { ffi::agl_mauth_ev(ptr, bytes, ncols as i32, self.w.as_ptr().add(c0 * 64 * 8), out.as_mut_ptr()) };
        assert_eq!(rc, 0);
        out
    }
    /// Ev from the summed raw sums.
    pub fn ev_finish(&self, out: &[u64]) -> AElem {
        let pm = AElem((0..DS).map(|m| K8(std::array::from_fn(|a| out[m * 8 + a]))).collect());
        mul_a(&self.gk, &pm)
    }
}

fn add_mod(a: &mut [u64], b: &[u64]) { for (x, y) in a.iter_mut().zip(b) { *x = ((*x as u128 + *y as u128) % Q as u128) as u64; } }

/// Ev_raw(W_e; ξ) of every source from the column shards.
pub fn ev_cols(p: &Profile, shards: &[ColShard], xi: &[K8]) -> Vec<AElem> {
    let pt = Point { xi: xi.to_vec(), rho: K8::ZERO };
    let n = shards.first().map_or(0, |s| s.nsrc);
    let parts: Vec<Vec<Vec<u64>>> = std::thread::scope(|sc| {
        let hs: Vec<_> = shards.iter().map(|sh| { let pt = &pt; sc.spawn(move || {
            almost_goldilocks_cuda::set_device(sh.dev).expect("set device");
            let t = Instant::now();
            let dp = DevPoint::new(p, pt);
            let t_w = t.elapsed().as_secs_f64() * 1e3;
            let nc = sh.c1 - sh.c0;
            // Every source in one launch.
            let mut out = vec![0u64; sh.nsrc * 512];
            let rc = unsafe { ffi::agl_mauth_ev_packed_batch(sh.raw.as_ptr(), (nc * 1024) as u64, sh.nsrc as i32, nc as i32, dp.w.as_ptr().add(sh.c0 * 64 * 8), out.as_mut_ptr()) };
            assert_eq!(rc, 0, "batched Ev");
            if std::env::var("MORPHEUS_TIMING").is_ok() { eprintln!("[auth] endpoint device {}: weights {t_w:.0} ms, sums {:.0} ms", sh.dev, t.elapsed().as_secs_f64() * 1e3 - t_w); }
            out.chunks(512).map(|c| c.to_vec()).collect::<Vec<_>>()
        }) }).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    almost_goldilocks_cuda::set_device(0).unwrap();
    let t = Instant::now();
    let dp = DevPoint::new(p, &pt);
    use rayon::prelude::*;
    let out = (0..n).into_par_iter().map(|s| { let mut acc = vec![0u64; 512]; for d in &parts { add_mod(&mut acc, &d[s]); } dp.ev_finish(&acc) }).collect();
    if std::env::var("MORPHEUS_TIMING").is_ok() { eprintln!("[auth] endpoint combine {:.2} s", t.elapsed().as_secs_f64()); }
    out
}

/// One device's share of an attempt.
struct ColPart { uw: DeviceBuffer<i64>, part: Vec<u32>, pm: [Vec<u64>; 2], err: Option<[Vec<i64>; 2]> }

/// Column-sharded prover (attempts in order, every device working on its
/// columns); equals `prove` / `prove_sharded` on the same inputs. `r_all`
/// holds every source's link witness (int8, sl·N each) on device 0. Full
/// response support only.
#[allow(clippy::too_many_arguments)]
pub fn prove_cols(
    p: &Profile, keys: &Keys, ckeys: &[ColKey], shards: &[ColShard], r_all: &DeviceBuffer<i8>, s_o: &DeviceBuffer<i8>, n: usize,
    coins: &Coins, opening: u64, tr: &MTranscript, pt: &Point, keep_z: bool, timing: &mut AuthTiming,
) -> Result<AuthOut, AuthError> {
    let (sw, sl, so) = (p.source_cols(), p.link_cols(), p.aux_cols());
    let c = sw - p.rank;
    let row = ((sw + sl + so) * N) as u64;
    let fs = FastSigma::new(p.sigma);
    let degs = p.degs();
    let half = K8::ONE.scale((Q + 1) / 2);
    assert_eq!(shards.len(), ckeys.len());
    // Per-device evaluation tables (the point is fixed for the opening).
    let dps: Vec<DevPoint> = shards.iter().map(|sh| { almost_goldilocks_cuda::set_device(sh.dev).unwrap(); DevPoint::new(p, pt) }).collect();
    almost_goldilocks_cuda::set_device(0).unwrap();
    let dp0 = DevPoint::new(p, pt);
    let b2 = |beta: &Vec<Vec<[i8; 64]>>, e: usize| -> [i8; 128] { std::array::from_fn(|i| beta[i / 64][e][i % 64]) };
    for attempt in 0..p.attempts as u16 {
        timing.attempts += 1;
        // Masks of the device's columns, its A·u partial product, raw Ev
        // sums and (last device) the identity error columns.
        let t0 = Instant::now();
        let parts: Vec<Result<ColPart, AuthError>> = std::thread::scope(|sc| {
            let hs: Vec<_> = shards.iter().zip(ckeys).zip(&dps).map(|((sh, ck), dp)| { let fs = &fs; sc.spawn(move || -> Result<ColPart, AuthError> {
                almost_goldilocks_cuda::set_device(sh.dev).expect("set device");
                let nc = sh.c1 - sh.c0;
                let mut uw = DeviceBuffer::<i64>::new(2 * nc * N).unwrap();
                for r in 0..p.tau {
                    unsafe { sample_into(coins, opening, attempt, fs, (r as u64 * row + (sh.c0 * N) as u64) as u32, (nc * N) as u32, uw.as_mut_ptr().add(r * nc * N)).map_err(AuthError::Sampler)?; }
                }
                let part = match &ck.key { Some(k) => partial_raw(k, 5, uw.as_ptr() as *const c_void, 8, 2, c.min(sh.c1) - sh.c0, nc * N, None).to_vec().unwrap(), None => Vec::new() };
                let pm = [0, 1].map(|r| dp.ev_part(unsafe { uw.as_ptr().add(r * nc * N) } as *const c_void, 8, sh.c0, nc));
                let err = (sh.c1 > c).then(|| [0, 1].map(|r| {
                    let mut e = vec![0i64; p.rank * N];
                    let src = unsafe { uw.as_ptr().add((r * nc + (c - sh.c0)) * N) };
                    assert_eq!(unsafe { ffi::cuda_memcpy_dtoh(e.as_mut_ptr() as *mut c_void, src as *const c_void, p.rank * N * 8) }, 0);
                    e
                }));
                Ok(ColPart { uw, part, pm, err })
            }) }).collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let mut parts: Vec<ColPart> = parts.into_iter().collect::<Result<_, _>>()?;
        almost_goldilocks_cuda::set_device(0).unwrap();
        // Link and auxiliary masks (device 0).
        let mut ul = DeviceBuffer::<i64>::new(2 * sl * N).unwrap();
        let mut uo = DeviceBuffer::<i64>::new(2 * so * N).unwrap();
        for r in 0..p.tau {
            let b = r as u64 * row;
            unsafe {
                sample_into(coins, opening, attempt, &fs, (b + (sw * N) as u64) as u32, (sl * N) as u32, ul.as_mut_ptr().add(r * sl * N)).map_err(AuthError::Sampler)?;
                sample_into(coins, opening, attempt, &fs, (b + ((sw + sl) * N) as u64) as u32, (so * N) as u32, uo.as_mut_ptr().add(r * so * N)).map_err(AuthError::Sampler)?;
            }
        }
        timing.sample_ms += t0.elapsed().as_secs_f64() * 1e3;
        // First message.
        let t0 = Instant::now();
        let mut acc: Option<DeviceBuffer<u32>> = None;
        for pc in parts.iter().filter(|x| !x.part.is_empty()) {
            let d = DeviceBuffer::from_slice(&pc.part).unwrap();
            match acc.as_mut() { Some(a) => super::commit_gpu::add(a, &d, 2, 5, p.rank), None => acc = Some(d) }
        }
        let cw = ring_rows(finish(acc.expect("source key columns"), 2, 5, p.rank, None).to_vec().unwrap(), p.rank);
        let cl = ring_rows(finish(partial_raw(&keys.l, 5, ul.as_ptr() as *const c_void, 8, 2, sl - p.rank, sl * N, None), 2, 5, p.rank, None).to_vec().unwrap(), p.rank);
        let co = ring_rows(finish(partial_raw(&keys.o, 5, uo.as_ptr() as *const c_void, 8, 2, so - p.rank, so * N, None), 2, 5, p.rank, None).to_vec().unwrap(), p.rank);
        let add_e = |cm: &Vec<Vec<u64>>, e: &[i64]| -> Vec<Vec<u64>> {
            cm.iter().enumerate().map(|(i, rw)| rw.iter().enumerate().map(|(j, &v)| {
                let x = e[i * N + j];
                let xe = if x >= 0 { x as u64 } else { Q - x.unsigned_abs() };
                ((v as u128 + xe as u128) % Q as u128) as u64
            }).collect()).collect()
        };
        let dev_err = |base: *const i64, cols: usize, r: usize| -> Vec<i64> {
            let mut e = vec![0i64; p.rank * N];
            let src = unsafe { base.add((r * cols + cols - p.rank) * N) };
            assert_eq!(unsafe { ffi::cuda_memcpy_dtoh(e.as_mut_ptr() as *mut c_void, src as *const c_void, p.rank * N * 8) }, 0);
            e
        };
        let werr = parts.iter().find_map(|x| x.err.clone()).expect("identity error columns");
        let (mut d, mut d_o) = (Vec::new(), Vec::new());
        for r in 0..p.tau {
            let mut pm = vec![0u64; 512];
            for x in &parts { add_mod(&mut pm, &x.pm[r]); }
            let hl = h_of(&decode(unsafe { ul.as_ptr().add(r * sl * N) }, A_ENC_COLS), 0);
            let a = dp0.ev_finish(&pm).add(&hl.scale_k(&dp0.rho_f));
            d.push(Tuple { c: add_e(&cw[r], &werr[r]), g: add_e(&cl[r], &dev_err(ul.as_ptr(), sl, r)), a });
            let dec = decode(unsafe { uo.as_ptr().add(r * so * N) }, A_ENC_COLS * p.aux_coeffs);
            let mut t = AElem::zero();
            for (i, &dg) in degs.iter().enumerate() {
                let mut xp = K8::ONE;
                for k in 1..=dg {
                    xp = xp.mul(&dp0.x[i]);
                    t = t.add(&h_of(&dec, A_ENC_COLS * aux_index(&degs, i, k)).scale_k(&xp.sub(&half)));
                }
            }
            d_o.push(AuxTuple { c: add_e(&co[r], &dev_err(uo.as_ptr(), so, r)), t });
        }
        timing.first_msg_ms += t0.elapsed().as_secs_f64() * 1e3;
        // Responses.
        let t0 = Instant::now();
        let beta = super::auth::challenges_pub(tr, &d, &d_o, attempt, n);
        let t_beta = t0.elapsed().as_secs_f64() * 1e3;
        let bb: Vec<i8> = (0..n).flat_map(|e| b2(&beta, e)).collect();
        let dp4a = std::env::var_os("MORPHEUS_COMBINE_DP4A").is_some();
        let zws: Vec<DeviceBuffer<i64>> = std::thread::scope(|sc| {
            let hs: Vec<_> = shards.iter().zip(parts.iter_mut()).map(|(sh, pc)| { let bb = &bb; sc.spawn(move || {
                almost_goldilocks_cuda::set_device(sh.dev).expect("set device");
                let nc = sh.c1 - sh.c0;
                let mut zw = pc.uw.clone_on_device().unwrap();
                let vp: Vec<*const u8> = (0..sh.nsrc).map(|s| unsafe { sh.raw.as_ptr().add(s * nc * 1024) }).collect();
                // Tensor-core GEMM (MORPHEUS_COMBINE_DP4A=1: the dp4a kernel).
                let rc = unsafe { if dp4a { ffi::agl_mauth_combine_packed(vp.as_ptr(), sh.nsrc as i32, nc as i32, bb.as_ptr(), zw.as_mut_ptr(), zw.as_mut_ptr().add(nc * N)) }
                                  else { ffi::agl_mauth_combine_tc(vp.as_ptr(), 1, sh.nsrc as i32, nc as i32, bb.as_ptr(), zw.as_mut_ptr(), zw.as_mut_ptr().add(nc * N)) } };
                assert_eq!(rc, 0, "packed combine");
                zw
            }) }).collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        almost_goldilocks_cuda::set_device(0).unwrap();
        let t_cols = t0.elapsed().as_secs_f64() * 1e3;
        let mut zl = ul.clone_on_device().unwrap();
        let vl: Vec<*const i8> = (0..n).map(|e| unsafe { r_all.as_ptr().add(e * sl * N) }).collect();
        unsafe { assert_eq!(if dp4a { ffi::agl_mauth_combine_multi(vl.as_ptr(), n as i32, sl as i32, bb.as_ptr(), zl.as_mut_ptr(), zl.as_mut_ptr().add(sl * N)) }
                            else { ffi::agl_mauth_combine_tc(vl.as_ptr() as *const *const u8, 0, n as i32, sl as i32, bb.as_ptr(), zl.as_mut_ptr(), zl.as_mut_ptr().add(sl * N)) }, 0); }
        let mut zo = uo.clone_on_device().unwrap();
        let bo = b2(&beta, n);
        unsafe { assert_eq!(ffi::agl_mauth_combine(s_o.as_ptr(), so as i32, bo.as_ptr(), zo.as_mut_ptr(), zo.as_mut_ptr().add(so * N)), 0); }
        if std::env::var_os("MORPHEUS_AUTH_TRACE").is_some() { eprintln!("[auth] attempt {attempt}: challenges {t_beta:.1} ms, column responses {:.1} ms, link/aux responses {:.1} ms", t_cols - t_beta, t0.elapsed().as_secs_f64() * 1e3 - t_cols); }
        timing.respond_ms += t0.elapsed().as_secs_f64() * 1e3;
        // Decision: exact norms summed over devices.
        let t0 = Instant::now();
        let mut caps = true;
        let (mut nu, mut nz) = (0u128, 0u128);
        let norms: Vec<[(u128, u64, u128); 2]> = std::thread::scope(|sc| {
            let hs: Vec<_> = shards.iter().zip(&parts).zip(&zws).map(|((sh, pc), zw)| sc.spawn(move || {
                almost_goldilocks_cuda::set_device(sh.dev).expect("set device");
                let len = (sh.c1 - sh.c0) * N;
                [0, 1].map(|r| { let (zn, zm) = norm(unsafe { zw.as_ptr().add(r * len) }, len); let (un, _) = norm(unsafe { pc.uw.as_ptr().add(r * len) }, len); (zn, zm, un) })
            })).collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        almost_goldilocks_cuda::set_device(0).unwrap();
        for r in 0..p.tau {
            let (zn, zmax, un) = norms.iter().fold((0u128, 0u64, 0u128), |a, x| (a.0 + x[r].0, a.1.max(x[r].1), a.2 + x[r].2));
            nu += un; nz += zn;
            if zmax >= p.b_z || zn > p.r2(sw * N) { caps = false; }
        }
        for (ub, zb, cols) in [(&ul, &zl, sl), (&uo, &zo, so)] {
            for r in 0..p.tau {
                let (zn, zmax) = norm(unsafe { zb.as_ptr().add(r * cols * N) }, cols * N);
                let (un, _) = norm(unsafe { ub.as_ptr().add(r * cols * N) }, cols * N);
                nu += un; nz += zn;
                if zmax >= p.b_z || zn > p.r2(cols * N) { caps = false; }
            }
        }
        let mut verdict = Coin::Reject;
        if caps {
            let x = Ratio { num: BigInt::from(nu) - BigInt::from(nz), den: BigInt::from(2u128 * p.sigma as u128 * p.sigma as u128) };
            let mut a = Address::new(Purpose::ProtocolCoin, opening);
            a.attempt = attempt;
            let mut st = coins.stream(a);
            let mut ub = BigUint::from(0u8);
            for _ in 0..rejection::B { ub = (ub << 1u32) + BigUint::from(st.next_bit().map_err(|_| AuthError::CoinAbort)?); }
            verdict = rejection::decide(&x, &ub);
        }
        timing.decide_ms += t0.elapsed().as_secs_f64() * 1e3;
        match verdict {
            Coin::Accept => {
                let t = Instant::now();
                // Row-major responses: row r is every device's columns in order.
                let w = if keep_z {
                    let pieces: Vec<[Vec<u8>; 2]> = std::thread::scope(|sc| {
                        let hs: Vec<_> = shards.iter().zip(&zws).map(|(sh, zw)| sc.spawn(move || {
                            almost_goldilocks_cuda::set_device(sh.dev).expect("set device");
                            let len = (sh.c1 - sh.c0) * N;
                            [0, 1].map(|r| {
                                let mut s = DeviceBuffer::<i64>::new(len).unwrap(); s.copy_range_from_device(0, zw, r * len, len).unwrap();
                                pack_dev(p, &s)
                            })
                        })).collect();
                        hs.into_iter().map(|h| h.join().unwrap()).collect()
                    });
                    almost_goldilocks_cuda::set_device(0).unwrap();
                    use rayon::prelude::*;
                    let t_p = t.elapsed().as_secs_f64() * 1e3;
                    // Row r, device k at its offset; copied in parallel into
                    // a buffer whose pages are faulted by many threads.
                    let parts: Vec<&[u8]> = (0..p.tau).flat_map(|r| pieces.iter().map(move |x| &x[r][..])).collect();
                    let total: usize = parts.iter().map(|x| x.len()).sum();
                    let mut w: Vec<u8> = Vec::with_capacity(total);
                    unsafe { w.set_len(total); }
                    let mut rest = &mut w[..];
                    let mut dst: Vec<&mut [u8]> = Vec::with_capacity(parts.len());
                    for x in &parts { let (a, b) = rest.split_at_mut(x.len()); dst.push(a); rest = b; }
                    dst.into_par_iter().zip(parts.par_iter()).for_each(|(d, x)| d.par_chunks_mut(1 << 20).zip(x.par_chunks(1 << 20)).for_each(|(a, b)| a.copy_from_slice(b)));
                    if std::env::var("MORPHEUS_TIMING").is_ok() { eprintln!("[auth] extract: device packs {t_p:.0} ms, join {:.0} ms", t.elapsed().as_secs_f64() * 1e3 - t_p); }
                    w
                } else { Vec::new() };
                let z = if keep_z { Responses { w, l: pack_dev(p, &zl), o: pack_dev(p, &zo) } } else { Responses::default() };
                timing.extract_ms += t.elapsed().as_secs_f64() * 1e3;
                return Ok((AuthProof { attempt, d, d_o, z }, None));
            }
            Coin::Reject => {}
            Coin::Abort => return Err(AuthError::CoinAbort),
        }
    }
    Err(AuthError::Exhausted)
}
