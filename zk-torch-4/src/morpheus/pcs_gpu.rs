//! GPU orchestration of the PCS opening: sources resident on the device as
//! physical int8 columns, commitments against NTT-domain key caches, private
//! coins sampled on the GPU, raw evaluation endpoints on the GPU.

use super::auth::Point;
use super::auth_gpu::{DevPoint, DevSource, Keys};
use super::coins::{Address, BitStream, Coins, Purpose};
use super::commit_gpu::{finish, partial, Cols};
use super::field8::K8;
use super::keys::KeySeed;
use super::link::{raw_to_cols, uniform_q_coin};
use super::link_fast::{LinkServices, Masks};
use super::params::{Profile, A_ENC_COLS};
use super::pcs_maps::enc;
use super::ring::{AElem, DS, N};
use almost_goldilocks_cuda::{ffi, DeviceBuffer};
use rayon::prelude::*;
use std::os::raw::c_void;

pub struct PcsGpu {
    pub p: Profile,
    pub seed: KeySeed,
    pub keys: Keys,
    /// Physical columns (payload, hiding, error) of every source, int8.
    pub src: Vec<DeviceBuffer<i8>>,
    /// Identity-error columns of every source as i64 (for the commitments).
    err: Vec<DeviceBuffer<i64>>,
    /// All sources, logical raw order, concatenated.
    pub flat: Option<DeviceBuffer<i8>>,
    pub coin_ms: f64,
    pub commit_ms: f64,
    /// Streamed mode: sources are not resident; the link keeps per-device
    /// shards here for the endpoints and the authentication.
    pub streamed: bool,
    pub kept: std::sync::Mutex<Vec<super::link_gpu::Kept>>,
    /// Streamed request sources uploaded ahead of the link (device 0).
    /// Packed sources uploaded to device 0 during the PIOP (one device).
    pub pre_raw: std::sync::Mutex<Option<super::link_gpu::PreShards>>,
    /// Streamed, several authentication devices: the source block is split
    /// by columns (auth_gpu::prove_cols). `col_devs` are the devices; the
    /// column keys persist, the column shards are per opening.
    pub col_devs: Vec<i32>,
    pub col_keys: Vec<super::auth_gpu::ColKey>,
    pub cols: Vec<super::auth_gpu::ColShard>,
}

fn ternary_cpu(coins: &Coins, purpose: Purpose, owner: u64, coord: u32) -> i8 {
    if super::coins::no_zk() { return 0; }
    let mut a = Address::new(purpose, owner);
    a.coordinate = coord;
    let mut s = coins.stream(a);
    loop {
        let v = (s.next_bit().unwrap() << 1) | s.next_bit().unwrap();
        if v < 3 { return v as i8 - 1; }
    }
}

/// `n` ternary coins at coordinates [0, n) of (purpose, owner).
pub fn ternary_gpu(coins: &Coins, purpose: Purpose, owner: u64, n: usize) -> Vec<i8> {
    if super::coins::no_zk() { return vec![0; n]; }
    let base = coins.gaussian_base_lanes(owner, 0);
    let mut d = DeviceBuffer::<i8>::new(n).unwrap();
    let mut f = DeviceBuffer::<u8>::new(n).unwrap();
    let rc = unsafe { ffi::agl_morpheus_coins(base.as_ptr(), purpose as i32, 0, n as u32, 0, d.as_mut_ptr() as *mut c_void, f.as_mut_ptr()) };
    assert_eq!(rc, 0, "ternary coin kernel");
    let mut v = d.to_vec().unwrap();
    let fl = f.to_vec().unwrap();
    for (i, &x) in fl.iter().enumerate() { if x != 0 { v[i] = ternary_cpu(coins, purpose, owner, i as u32); } }
    v
}

/// `n` uniform-q coins at coordinates [c0, c0 + n) of (purpose, owner).
pub fn uniform_q_gpu(coins: &Coins, purpose: Purpose, owner: u64, c0: u32, n: usize) -> Vec<u64> {
    if super::coins::no_zk() { return vec![0; n]; }
    let base = coins.gaussian_base_lanes(owner, 0);
    let mut d = DeviceBuffer::<u64>::new(n).unwrap();
    let mut f = DeviceBuffer::<u8>::new(n).unwrap();
    let rc = unsafe { ffi::agl_morpheus_coins(base.as_ptr(), purpose as i32, c0, n as u32, 1, d.as_mut_ptr() as *mut c_void, f.as_mut_ptr()) };
    assert_eq!(rc, 0, "uniform-q coin kernel");
    let mut v = d.to_vec().unwrap();
    let fl = f.to_vec().unwrap();
    for (i, &x) in fl.iter().enumerate() { if x != 0 { v[i] = uniform_q_coin(coins, purpose, owner, c0 + i as u32); } }
    v
}

fn dev0() { almost_goldilocks_cuda::set_device(0).expect("set device"); }

fn a_from(v: &[u64]) -> AElem { AElem((0..DS).map(|m| K8(std::array::from_fn(|a| v[m * 8 + a]))).collect()) }

fn i8_cols(cols: &[Vec<i64>]) -> Vec<i8> { cols.iter().flat_map(|c| c.iter().map(|&x| x as i8)).collect() }

impl PcsGpu {
    pub fn new(p: Profile, seed: KeySeed) -> Self {
        let keys = Keys::build(&p, &seed);
        PcsGpu { p, seed, keys, src: Vec::new(), err: Vec::new(), flat: None, coin_ms: 0.0, commit_ms: 0.0, streamed: false, kept: std::sync::Mutex::new(Vec::new()), pre_raw: std::sync::Mutex::new(None), col_devs: Vec::new(), col_keys: Vec::new(), cols: Vec::new() }
    }

    /// Upload the sources (raw logical order) once; physical columns are
    /// produced on the device. The flat logical buffer stays resident for the
    /// link.
    pub fn load_sources(&mut self, raw: &[Vec<i8>]) {
        let rank = self.p.rank;
        let len = raw[0].len();
        let mut flat = DeviceBuffer::<i8>::new(len * raw.len()).unwrap();
        for (s, r) in raw.iter().enumerate() { flat.copy_from_slice_at(s * len, r).unwrap(); }
        self.src.clear();
        self.err.clear();
        for (s, _) in raw.iter().enumerate() {
            let mut phys = DeviceBuffer::<i8>::new(len).unwrap();
            let rc = unsafe { ffi::agl_morpheus_permute(flat.as_ptr().add(s * len), phys.as_mut_ptr(), len as u64) };
            assert_eq!(rc, 0);
            let ncols = len / N;
            let e8 = phys.read_slice((ncols - rank) * N, rank * N).unwrap();
            self.err.push(DeviceBuffer::from_slice(&e8.iter().map(|&v| v as i64).collect::<Vec<_>>()).unwrap());
            self.src.push(phys);
        }
        self.flat = Some(flat);
    }

    /// Append one source (logical raw order) after `load_sources`; returns
    /// its commitment.
    pub fn push_source(&mut self, r: &[i8]) -> Vec<Vec<u64>> {
        let len = r.len();
        let n = self.src.len();
        let old = self.flat.take().expect("load_sources first");
        let mut flat = DeviceBuffer::<i8>::new(len * (n + 1)).unwrap();
        flat.copy_range_from_device(0, &old, 0, len * n).unwrap();
        drop(old);
        flat.copy_from_slice_at(n * len, r).unwrap();
        let mut phys = DeviceBuffer::<i8>::new(len).unwrap();
        let rc = unsafe { ffi::agl_morpheus_permute(flat.as_ptr().add(n * len), phys.as_mut_ptr(), len as u64) };
        assert_eq!(rc, 0);
        let rank = self.p.rank;
        let e8 = phys.read_slice((len / N - rank) * N, rank * N).unwrap();
        let err = DeviceBuffer::from_slice(&e8.iter().map(|&v| v as i64).collect::<Vec<_>>()).unwrap();
        let t = std::time::Instant::now();
        let ncols = self.p.source_payload_cols + self.p.hiding;
        let acc = super::commit_gpu::partial_raw(&self.keys.w, 3, phys.as_ptr() as *const c_void, 1, 1, ncols, 0, None);
        let c = ring_rows(finish(acc, 1, 3, self.p.rank, Some(&err)).to_vec().unwrap());
        self.commit_ms += t.elapsed().as_secs_f64() * 1e3;
        self.src.push(phys);
        self.err.push(err);
        self.flat = Some(flat);
        c
    }

    /// Free the source key cache (streamed mode, during the PIOP and the
    /// link); `ensure_w` rebuilds it.
    pub fn drop_w(&mut self) {
        if self.keys.w.buf.len() > 1 { self.keys.w = super::commit_gpu::KeyCache { buf: DeviceBuffer::<u32>::new(1).unwrap(), ..self.keys.w.clone_meta() }; }
    }
    pub fn ensure_w(&mut self) {
        if self.keys.w.buf.len() <= 1 {
            almost_goldilocks_cuda::set_device(0).unwrap();
            let p = &self.p;
            self.keys.w = super::commit_gpu::KeyCache::build(&self.seed, super::keys::KeyKind::Source, p.rank, 0, p.source_payload_cols + p.hiding, 5);
        }
    }

    /// Drop the resident sources (keeps the key caches).
    pub fn unload(&mut self) { self.src.clear(); self.err.clear(); self.flat = None; self.kept.lock().unwrap().clear(); *self.pre_raw.lock().unwrap() = None; self.cols.clear(); self.col_devs.clear(); self.col_keys.clear(); }
    /// Column-sharded authentication over `devs` for this opening.
    /// The column keys are built after the link (ev) and released with the
    /// opening, so the link has the devices' memory.
    pub fn use_cols(&mut self, devs: Vec<i32>) { self.col_devs = devs; }

    /// Kept shards in physical column order (in place, per source).
    pub fn kept_to_phys(&self) {
        let mut k = self.kept.lock().unwrap();
        std::thread::scope(|sc| {
            for sh in k.iter_mut() {
                if sh.phys { continue; }
                sc.spawn(move || {
                    almost_goldilocks_cuda::set_device(sh.dev).expect("set device");
                    let len = sh.buf.len() / sh.srcs.len();
                    let mut tmp = DeviceBuffer::<i8>::new(len).unwrap();
                    for j in 0..sh.srcs.len() {
                        let rc = unsafe { ffi::agl_morpheus_permute(sh.buf.as_ptr().add(j * len), tmp.as_mut_ptr(), len as u64) };
                        assert_eq!(rc, 0);
                        sh.buf.copy_range_from_device(j * len, &tmp, 0, len).unwrap();
                    }
                    sh.phys = true;
                });
            }
        });
        almost_goldilocks_cuda::set_device(0).unwrap();
    }

    /// Source commitments C_e = A_W W_e.
    pub fn commit_sources(&mut self) -> Vec<Vec<Vec<u64>>> {
        let t = std::time::Instant::now();
        let ncols = self.p.source_payload_cols + self.p.hiding;
        let rank = self.p.rank;
        // Batches of sources copied contiguously, one pass over the key cache each.
        let n = self.src.len();
        let bv = if n > 0 { commit_batch(ncols, self.src[0].len()) } else { 1 };
        let mut out = Vec::with_capacity(n);
        if n > 0 {
            let len = self.src[0].len();
            let el = self.err[0].len();
            let mut tmp = DeviceBuffer::<i8>::new(len * bv.min(n)).unwrap();
            let mut ecat = DeviceBuffer::<i64>::new(el * bv.min(n)).unwrap();
            for s0 in (0..n).step_by(bv) {
                let nb = bv.min(n - s0);
                for q in 0..nb {
                    tmp.copy_range_from_device(q * len, &self.src[s0 + q], 0, len).unwrap();
                    ecat.copy_range_from_device(q * el, &self.err[s0 + q], 0, el).unwrap();
                }
                let acc = super::commit_gpu::partial_raw(&self.keys.w, 3, tmp.as_ptr() as *const c_void, 1, nb, ncols, len, None);
                let mut e = DeviceBuffer::<i64>::new(el * nb).unwrap();
                e.copy_range_from_device(0, &ecat, 0, el * nb).unwrap();
                let rr = ring_rows(finish(acc, nb, 3, rank, Some(&e)).to_vec().unwrap());
                out.extend(rr.chunks(rank).map(|c| c.to_vec()));
            }
        }
        self.commit_ms += t.elapsed().as_secs_f64() * 1e3;
        out
    }

    /// Device sources for the authentication; takes the resident physical
    /// columns (the opening is the last use of them).
    pub fn dev_sources(&mut self, r8: &[Vec<i8>]) -> Vec<DevSource> {
        std::mem::take(&mut self.src).into_iter().zip(r8).map(|(w, r)| DevSource { w, r: DeviceBuffer::from_slice(r).unwrap() }).collect()
    }

    /// commit_cols from int8 digits (columns of N, logical order).
    fn commit_cols8(&self, cache: &super::commit_gpu::KeyCache, v: &[i8]) -> Vec<Vec<u64>> {
        let rank = self.p.rank;
        let ncols = v.len() / N;
        let body = DeviceBuffer::from_slice(&v[..(ncols - rank) * N]).unwrap();
        let err = DeviceBuffer::from_slice(&v[(ncols - rank) * N..].iter().map(|&x| x as i64).collect::<Vec<i64>>()).unwrap();
        let acc = partial(cache, 3, Cols::I8(&body), 1, ncols - rank, None);
        ring_rows(finish(acc, 1, 3, rank, Some(&err)).to_vec().unwrap())
    }
    fn commit_cols(&self, cache: &super::commit_gpu::KeyCache, cols: &[Vec<i64>]) -> Vec<Vec<u64>> {
        let rank = self.p.rank;
        let body = DeviceBuffer::from_slice(&i8_cols(&cols[..cols.len() - rank])).unwrap();
        let err = DeviceBuffer::from_slice(&cols[cols.len() - rank..].iter().flatten().copied().collect::<Vec<i64>>()).unwrap();
        let acc = partial(cache, 3, Cols::I8(&body), 1, cols.len() - rank, None);
        ring_rows(finish(acc, 1, 3, rank, Some(&err)).to_vec().unwrap())
    }
}

fn ring_rows(v: Vec<u64>) -> Vec<Vec<u64>> { v.chunks(N).map(|c| c.to_vec()).collect() }

/// Sources per commit pass (≤ MORPHEUS_COMMIT_V, default 8) that fit in the
/// current device's free memory (driver free + allocator cache, 4 GB kept):
/// each needs its NTT residues (3 primes) and a staged copy of `len` bytes.
fn commit_batch(ncols: usize, len: usize) -> usize {
    let want: usize = std::env::var("MORPHEUS_COMMIT_V").ok().and_then(|s| s.parse().ok()).unwrap_or(8).clamp(1, 8);
    let free = almost_goldilocks_cuda::mem_get_info().map(|x| x.0).unwrap_or(0) + unsafe { ffi::agl_cached_bytes() };
    let per = 3 * ncols * N * 4 + len + (1 << 20);
    let b = want.min(free.saturating_sub(4usize << 30) / per).max(1);
    if std::env::var("MORPHEUS_TIMING").is_ok() { eprintln!("[commit] {b} sources per key pass ({:.1} GB free)", free as f64 / 1e9); }
    b
}

impl LinkServices for PcsGpu {
    fn masks(&mut self, p: &Profile, coins: &Coins, opening: u64, n: usize) -> Masks {
        let t = std::time::Instant::now();
        let owner = |e: usize| opening * p.n_max as u64 + e as u64;
        let tern_cols = |purpose, owner, cols: usize| -> Vec<Vec<i64>> {
            ternary_gpu(coins, purpose, owner, cols * N).chunks(N).map(|c| c.iter().map(|&x| x as i64).collect()).collect()
        };
        // Per source in parallel (device 0: the PCS device of this process).
        // Coin kernels spread over this process's devices (MORPHEUS_GPUS);
        // the coins depend only on (purpose, owner, coordinate).
        let ndev: usize = std::env::var("MORPHEUS_GPUS").ok().and_then(|s| s.parse().ok()).unwrap_or(1usize).max(1);
        let on = |e: usize| almost_goldilocks_cuda::set_device((e % ndev) as i32).expect("set device");
        // Each task returns its thread to device 0 (the pool is shared).
        let u: Vec<AElem> = (0..n).into_par_iter().map(|e| { on(e); let x = a_from(&uniform_q_gpu(coins, Purpose::LinkPayload, owner(e), 0, 512)); dev0(); x }).collect();
        let t_u = t.elapsed().as_secs_f64();
        // Link witnesses as int8 digits (encoded u, then ternary hiding and
        // error columns); only the encoded-u columns are kept as i64.
        let (r_cols, r8): (Vec<Vec<Vec<i64>>>, Vec<Vec<i8>>) = (0..n).into_par_iter().map(|e| {
            on(e);
            let ec = enc(&u[e]);
            let mut v8: Vec<i8> = Vec::with_capacity(p.link_cols() * N);
            for c in &ec { v8.extend(c.iter().map(|&x| x as i8)); }
            v8.extend(ternary_gpu(coins, Purpose::LinkSecret, owner(e), p.hiding * N));
            v8.extend(ternary_gpu(coins, Purpose::LinkError, owner(e), p.rank * N));
            dev0();
            (ec, v8)
        }).unzip();
        let t_r = t.elapsed().as_secs_f64();
        if std::env::var("MORPHEUS_TIMING").is_ok() { eprintln!("[link] masks: u {:.2} s, r columns {:.2} s", t_u, t_r - t_u); }
        dev0();
        let aq = uniform_q_gpu(coins, Purpose::AuxPayload, opening, 0, p.aux_coeffs * 512);
        let a_aux: Vec<AElem> = aq.chunks(512).map(a_from).collect();
        let mut s_o: Vec<Vec<i64>> = a_aux.iter().flat_map(enc).collect();
        s_o.extend(tern_cols(Purpose::AuxSecret, opening, p.hiding));
        s_o.extend(tern_cols(Purpose::AuxError, opening, p.rank));
        self.coin_ms += t.elapsed().as_secs_f64() * 1e3;
        Masks { u, r_cols, a_aux, s_o, r8: Some(r8) }
    }
    fn commit_link(&mut self, _p: &Profile, _seed: &KeySeed, r_cols: &[Vec<Vec<i64>>]) -> Vec<Vec<Vec<u64>>> {
        let t = std::time::Instant::now();
        let out = r_cols.par_iter().map(|r| { dev0(); self.commit_cols(&self.keys.l, r) }).collect();
        self.commit_ms += t.elapsed().as_secs_f64() * 1e3;
        out
    }
    fn commit_link8(&mut self, _p: &Profile, _seed: &KeySeed, r8: &[Vec<i8>]) -> Vec<Vec<Vec<u64>>> {
        let t = std::time::Instant::now();
        let out = r8.par_iter().map(|r| { dev0(); self.commit_cols8(&self.keys.l, r) }).collect();
        self.commit_ms += t.elapsed().as_secs_f64() * 1e3;
        out
    }
    fn commit_aux(&mut self, _p: &Profile, _seed: &KeySeed, s_o: &[Vec<i64>]) -> Vec<Vec<u64>> {
        let t = std::time::Instant::now();
        let out = self.commit_cols(&self.keys.o, s_o);
        self.commit_ms += t.elapsed().as_secs_f64() * 1e3;
        out
    }
    fn dev_raw(&self) -> Option<&DeviceBuffer<i8>> { self.flat.as_ref() }
    fn kept(&self) -> Option<&std::sync::Mutex<Vec<super::link_gpu::Kept>>> { if self.streamed && self.col_devs.is_empty() { Some(&self.kept) } else { None } }
    fn take_pre_raw(&self) -> Option<super::link_gpu::PreShards> { self.pre_raw.lock().unwrap().take() }
    fn ev(&mut self, p: &Profile, raw: &dyn super::packed::Sources, xi: &[K8]) -> Vec<AElem> {
        if self.streamed && !self.col_devs.is_empty() {
            // The link's shards are released; the column shards serve the
            // endpoints and then the authentication.
            let t = std::time::Instant::now();
            if self.cols.is_empty() { self.cols = super::auth_gpu::col_shards(p, raw, &self.col_devs); }
            if self.col_keys.is_empty() { self.col_keys = super::auth_gpu::col_keys(&self.p, &self.seed, &self.col_devs); almost_goldilocks_cuda::set_device(0).unwrap(); }
            if std::env::var("MORPHEUS_TIMING").is_ok() { eprintln!("[auth] column shards {:.2} s", t.elapsed().as_secs_f64()); }
            return super::auth_gpu::ev_cols(p, &self.cols, xi);
        }
        if self.streamed {
            self.kept_to_phys();
            let k = self.kept.lock().unwrap();
            let pt = Point { xi: xi.to_vec(), rho: K8::ZERO };
            let out: Vec<Vec<(usize, AElem)>> = std::thread::scope(|sc| {
                let hs: Vec<_> = k.iter().map(|sh| {
                    let pt = &pt;
                    sc.spawn(move || {
                        almost_goldilocks_cuda::set_device(sh.dev).expect("set device");
                        let dp = DevPoint::new(p, pt);
                        if let Some(pk) = &sh.pk {
                            // Packed physical columns: every source in one launch.
                            let stride = pk.len() / sh.srcs.len();
                            let mut out = vec![0u64; sh.srcs.len() * 512];
                            let rc = unsafe { ffi::agl_mauth_ev_packed_batch(pk.as_ptr(), stride as u64, sh.srcs.len() as i32, (stride / 1024) as i32, dp.w.as_ptr(), out.as_mut_ptr()) };
                            assert_eq!(rc, 0, "packed Ev");
                            return sh.srcs.iter().zip(out.chunks(512)).map(|(&s, c)| (s, dp.ev_finish(c))).collect::<Vec<_>>();
                        }
                        let len = sh.buf.len() / sh.srcs.len();
                        sh.srcs.iter().enumerate().map(|(j, &s)| (s, dp.ev(unsafe { sh.buf.as_ptr().add(j * len) } as *const c_void, 1))).collect::<Vec<_>>()
                    })
                }).collect();
                hs.into_iter().map(|h| h.join().unwrap()).collect()
            });
            almost_goldilocks_cuda::set_device(0).unwrap();
            let mut v: Vec<(usize, AElem)> = out.into_iter().flatten().collect();
            v.sort_by_key(|x| x.0);
            return v.into_iter().map(|x| x.1).collect();
        }
        let dp = DevPoint::new(p, &Point { xi: xi.to_vec(), rho: K8::ZERO });
        self.src.iter().map(|s| dp.ev(s.as_ptr() as *const c_void, 1)).collect()
    }
}

/// Commitments of host sources streamed through the devices `devs`
/// (device id, its W key cache), contiguous chunks per device; nothing stays
/// resident.
pub fn commit_streamed(p: &Profile, devs: &[(i32, &Keys)], raw: &dyn super::packed::Sources) -> Vec<Vec<Vec<u64>>> {
    commit_streamed_on(p, None, devs, raw, None)
}

/// Same; devices in `extra` (id) build a temporary 3-prime source key cache
/// from `seed` for the duration of the commitment.
/// Commitments of every source, on `devs` (with key caches) and the extra
/// devices. Batches go to devices round-robin, so sources finish roughly in
/// order; `done(s, c)` is called as each source's commitment is ready.
pub fn commit_streamed_on(p: &Profile, extra: Option<(&KeySeed, &[i32])>, devs: &[(i32, &Keys)], raw: &dyn super::packed::Sources,
    done: Option<&(dyn Fn(usize, &[Vec<u64>]) + Sync)>) -> Vec<Vec<Vec<u64>>> {
    let n = raw.count();
    let mut all: Vec<(i32, Option<&Keys>)> = devs.iter().map(|&(d, k)| (d, Some(k))).collect();
    if let Some((_, ex)) = extra { for &d in ex { all.push((d, None)); } }
    let g = all.len().max(1).min(n.max(1));
    let len = raw.src_len();
    let ncols = p.source_payload_cols + p.hiding;
    let bv = commit_batch(ncols, len);
    let mut out: Vec<(usize, Vec<Vec<u64>>)> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..g).map(|k| {
            let (dev, keys) = all[k];
            sc.spawn(move || {
                almost_goldilocks_cuda::set_device(dev).expect("set device");
                let tmp;
                let wk: &super::commit_gpu::KeyCache = match keys {
                    Some(k) => &k.w,
                    None => { let (seed, _) = extra.unwrap(); tmp = super::commit_gpu::KeyCache::build(seed, super::keys::KeyKind::Source, p.rank, 0, p.source_payload_cols + p.hiding, 3); &tmp }
                };
                let rank = p.rank;
                // Up to MORPHEUS_COMMIT_V sources per pass over the key cache
                // (multi-vector MAC: each key element read once per pass).
                let mut scratch: Option<DeviceBuffer<u8>> = None;
                let mut phys = DeviceBuffer::<i8>::new(len * bv).unwrap();
                let mut v = Vec::new();
                for s0 in (k * bv..n).step_by(g * bv) {
                    let nb = bv.min(n - s0);
                    let mut err8: Vec<i8> = Vec::with_capacity(nb * rank * N);
                    for q in 0..nb {
                        super::packed::upload_i8(raw, s0 + q, &mut phys, q * len, true, &mut scratch);
                        err8.extend(phys.read_slice(q * len + (len / N - rank) * N, rank * N).unwrap());
                    }
                    let err = DeviceBuffer::from_slice(&err8.iter().map(|&x| x as i64).collect::<Vec<_>>()).unwrap();
                    let acc = super::commit_gpu::partial_raw(wk, 3, phys.as_ptr() as *const c_void, 1, nb, ncols, len, None);
                    let rr = ring_rows(finish(acc, nb, 3, rank, Some(&err)).to_vec().unwrap());
                    for (q, c) in rr.chunks(rank).enumerate() {
                        if let Some(f) = done { f(s0 + q, c); }
                        v.push((s0 + q, c.to_vec()));
                    }
                }
                v
            })
        }).collect();
        hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    almost_goldilocks_cuda::set_device(0).unwrap();
    out.sort_by_key(|x| x.0);
    assert_eq!(out.len(), n);
    out.into_iter().map(|x| x.1).collect()
}

/// Authentication replica on another device: key caches and the physical
/// source columns, built from the host sources (no peer copies).
pub struct AuthReplica { pub dev: i32, pub keys: Keys, pub w: Vec<DeviceBuffer<i8>> }

impl AuthReplica {
    pub fn build(dev: i32, p: &Profile, seed: &KeySeed, raw: &[Vec<i8>]) -> Self {
        let mut r = Self::new(dev, p, seed);
        r.load(raw);
        r
    }
    /// Key caches only; sources come with `load` / `push`.
    pub fn new(dev: i32, p: &Profile, seed: &KeySeed) -> Self {
        almost_goldilocks_cuda::set_device(dev).expect("set device");
        AuthReplica { dev, keys: Keys::build(p, seed), w: Vec::new() }
    }
    pub fn load(&mut self, raw: &[Vec<i8>]) {
        self.w.clear();
        for r in raw { self.push(r); }
    }
    pub fn push(&mut self, r: &[i8]) {
        almost_goldilocks_cuda::set_device(self.dev).expect("set device");
        let len = r.len();
        let flat = DeviceBuffer::<i8>::from_slice(r).unwrap();
        let mut phys = DeviceBuffer::<i8>::new(len).unwrap();
        let rc = unsafe { ffi::agl_morpheus_permute(flat.as_ptr(), phys.as_mut_ptr(), len as u64) };
        assert_eq!(rc, 0);
        self.w.push(phys);
    }
    /// Device sources with the link columns attached (on this replica's device).
    pub fn sources(&mut self, r8: &[Vec<i8>]) -> Vec<DevSource> {
        almost_goldilocks_cuda::set_device(self.dev).expect("set device");
        std::mem::take(&mut self.w).into_iter().zip(r8).map(|(w, r)| DevSource { w, r: DeviceBuffer::from_slice(r).unwrap() }).collect()
    }
}