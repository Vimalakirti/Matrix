//! GPU backend of the structured Π_link prover
//! (`cuda_almost_goldilocks/morpheus_link.cuh`).

use super::field8::K8;
use super::link_fast::{Phase1, Phase1Input, RoundTables, TermInfo};
use almost_goldilocks_cuda::{ffi, DeviceBuffer};
use std::os::raw::c_void;

#[repr(C)]
#[derive(Clone, Copy)]
struct BlockDesc { src: u32, id: u32, toff: u32, tcnt: u32, qoff: u32, pad: [u32; 3], cr: [u64; 8], cb: [u64; 8] }
#[repr(C)]
#[derive(Clone, Copy)]
struct BlockTerm { term: u32, pad: u32, c: [u64; 8] }
#[repr(C)]
#[derive(Clone, Copy)]
struct TermTab { aoff: u32, boff: u32, pa: u32, pad: u32 }

const NACC: usize = 10;

fn as_bytes<T>(v: &[T]) -> &[u8] { unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) } }
fn upload<T>(v: &[T]) -> DeviceBuffer<u8> {
    let b = as_bytes(v);
    let mut d = DeviceBuffer::<u8>::new(b.len().max(1)).expect("device alloc");
    if !b.is_empty() { d.copy_from_slice(b).expect("htod"); }
    d
}
fn kwords(v: &[K8]) -> Vec<u64> { v.iter().flat_map(|k| k.words()).collect() }

pub struct GpuPhase1 {
    raw: DeviceBuffer<i8>,
    src_len: u64,
    b0: usize,
    nblocks: u32,
    blocks: DeviceBuffer<u8>,
    bterms: DeviceBuffer<u8>,
    qd: DeviceBuffer<u8>,
    pub kernel_ms: f64,
}

impl GpuPhase1 {
    pub fn new(inp: Phase1Input) -> Self {
        let src_len = inp.raw[0].len() as u64;
        let flat: Vec<i8> = inp.raw.iter().flat_map(|r| r.iter().copied()).collect();
        let raw = DeviceBuffer::from_slice(&flat).expect("raw upload");
        let mut descs = Vec::with_capacity(inp.blocks.len());
        let mut bts = Vec::new();
        let mut qd: Vec<u64> = Vec::new();
        for blk in inp.blocks {
            let cr = blk.c_alpha.mul(&inp.g_range[blk.source]);
            let cb = if blk.chi { blk.c_alpha.mul(&inp.g_bit[blk.source]) } else { K8::ZERO };
            let qoff = match &blk.qd { Some(t) => { let o = (qd.len() / 8) as u32; qd.extend(kwords(t)); o } None => u32::MAX };
            descs.push(BlockDesc { src: blk.source as u32, id: blk.id as u32, toff: bts.len() as u32, tcnt: blk.terms.len() as u32, qoff, pad: [0; 3], cr: cr.words(), cb: cb.words() });
            for bt in &blk.terms { bts.push(BlockTerm { term: bt.term as u32, pad: 0, c: bt.c.words() }); }
        }
        if std::env::var("MORPHEUS_TIMING").is_ok() {
            let dense = descs.iter().filter(|d| d.qoff != u32::MAX).count();
            eprintln!("[link]   {} blocks ({} dense-q), {} block terms, {} sources", descs.len(), dense, bts.len(), inp.raw.len());
        }
        GpuPhase1 { raw, src_len, b0: inp.b0, nblocks: descs.len() as u32, blocks: upload(&descs), bterms: upload(&bts), qd: upload(&qd), kernel_ms: 0.0 }
    }
}

impl Phase1 for GpuPhase1 {
    fn round(&mut self, rt: &RoundTables, terms: &[TermInfo], d: usize) -> Vec<K8> {
        let tr0 = std::time::Instant::now();
        let (mut a, mut b, mut tt) = (Vec::new(), Vec::new(), Vec::new());
        for k in 0..terms.len() {
            tt.push(TermTab { aoff: (a.len() / 8) as u32, boff: (b.len() / 8) as u32, pa: rt.pa[k] as u32, pad: 0 });
            a.extend(kwords(&rt.a[k]));
            b.extend(kwords(&rt.b[k]));
        }
        let (da, db, dt) = (upload(&a), upload(&b), upload(&tt));
        let (de, dal) = (upload(&kwords(&rt.e_pre)), upload(&kwords(&rt.e_alpha)));
        let mut out = vec![0u64; 8 * NACC * 8192];
        let t = std::time::Instant::now();
        let grid = unsafe {
            ffi::agl_mlink_phase1(self.raw.as_ptr(), self.src_len, self.b0 as i32, rt.i as i32, d as i32,
                self.blocks.as_ptr() as *const c_void, self.nblocks, self.bterms.as_ptr() as *const c_void, dt.as_ptr() as *const c_void,
                da.as_ptr() as *const u64, db.as_ptr() as *const u64, de.as_ptr() as *const u64, dal.as_ptr() as *const u64,
                self.qd.as_ptr() as *const u64, out.as_mut_ptr())
        };
        self.kernel_ms += t.elapsed().as_secs_f64() * 1e3;
        if std::env::var("MORPHEUS_TIMING").is_ok() { eprintln!("[link]   round {} kernel {:.1} ms, upload {:.1} ms, {} terms", rt.i, t.elapsed().as_secs_f64() * 1e3, (t - tr0).as_secs_f64() * 1e3, terms.len()); }
        assert!(grid > 0 && (grid as usize) <= 8192, "phase1 kernel failed ({grid})");
        let mut sum = vec![K8::ZERO; 2 * (d + 1)];
        for g in 0..grid as usize {
            for (k, sk) in sum.iter_mut().enumerate() {
                let o = (g * NACC + k) * 8;
                *sk = sk.add(&K8(out[o..o + 8].try_into().unwrap()));
            }
        }
        sum
    }
    fn fold(&mut self, e_full: &[K8]) -> Vec<K8> {
        let de = upload(&kwords(e_full));
        let mut dout = DeviceBuffer::<u64>::new(8 * self.nblocks as usize).expect("alloc");
        let rc = unsafe { ffi::agl_mlink_fold(self.raw.as_ptr(), self.src_len, self.b0 as i32, self.blocks.as_ptr() as *const c_void, self.nblocks, de.as_ptr() as *const u64, dout.as_mut_ptr()) };
        assert_eq!(rc, 0, "fold kernel failed");
        dout.to_vec().unwrap().chunks(8).map(|c| K8(c.try_into().unwrap())).collect()
    }
}

pub fn gpu_backend<'a>(inp: Phase1Input<'a>) -> Box<dyn Phase1 + 'a> { Box::new(GpuPhase1::new(inp)) }

// ======================================================================
// v2: region-organised GPU engine (morpheus_link2.cuh)
// ======================================================================

use super::link::{interpolate, Statement};
use std::collections::HashMap;
use super::link_fast::{eqb, eqt_k, finish_round, m_term, term_tables, DENSE_P};
use super::field8::eq_table;
use super::params::Profile;
use super::transcript::{Item, MTranscript, Stage};
use rayon::prelude::*;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct RegionC { src: u32, base_blk: u32, h: u32, p: u32, flags: u32, toff: u32, tcnt: u32, qd_off: u32 }
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct TermC { pa: u32, aoff: u32, boff: u32, hl_off: u32, hh_off: u32, h1: u32, pad0: u32, pad1: u32, cs: [u64; 8] }
#[repr(C)]
#[derive(Clone, Copy)]
struct Blk { src: u32, id: u32, region: u32, k: u32 }
#[derive(Clone, Copy)]
struct RawBlk(*mut Blk);
unsafe impl Send for RawBlk {}
unsafe impl Sync for RawBlk {}
#[repr(C)]
#[derive(Clone, Copy)]
struct ConstsC {
    ca_lo: u64, ca_hi: u64, ca_lo_bits: i32, pad: i32,
    g_range: u64, g_bit: u64,
    regions: u64, terms: u64, hl: u64, hh: u64,
    a: u64, b: u64, qd: u64, qidx: u64,
    bco: u64, oh_base: u64,
}

fn dptr<T>(b: &DeviceBuffer<T>) -> u64 { b.as_ptr() as u64 }
fn up_k(v: &[K8]) -> DeviceBuffer<u64> { DeviceBuffer::from_slice(&kwords(v)).expect("upload") }
fn kvec(words: &[u64]) -> Vec<K8> { words.chunks(8).map(|c| K8(c.try_into().unwrap())).collect() }

/// Phases 1 and 2 of the structured link on the GPU (v2). Returns the round
/// messages, the point ξ, the final (w, q, χ) per source, Π eq(α_j, r_j) and
/// the M_O prefix — exactly what the CPU phases produce.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
/// Link source shards uploaded ahead of the link (during the PIOP): the
/// placement they follow and each device's buffer (None: upload later).
pub struct PreShards { pub owner: Vec<usize>, pub bufs: Vec<Option<DeviceBuffer<u8>>> }

/// Devices the link shards its sources over (MORPHEUS_GPUS).
/// MORPHEUS_DENSE_LINK=1: the same link without structure. Every block of
/// every source is visited, and the one-hot, binned, factored, reduced-round,
/// index, fast-query and many-term paths are off (a same-protocol ablation).
pub fn dense_link() -> bool { std::env::var_os("MORPHEUS_DENSE_LINK").is_some() }
/// MORPHEUS_NO_SKIP=1: visit all-zero chunks too, with every link kernel
/// still on (a same-protocol ablation of zero-chunk skipping alone).
pub fn no_skip() -> bool { dense_link() || std::env::var_os("MORPHEUS_NO_SKIP").is_some() }

pub fn shard_gpus(n: usize) -> usize {
    std::env::var("MORPHEUS_GPUS").ok().and_then(|s| s.parse().ok()).unwrap_or(1usize).max(1).min(n.max(1))
}

/// Sources go to devices greedily by weighted load, from the raw digits and
/// the declared bit blocks alone. Streamed mode with kept shards scales device 0
/// down (MORPHEUS_SHARD0_WEIGHT, default 0.5): it also hosts the authentication.
pub fn shard_owner(raw: &dyn crate::morpheus::packed::Sources, bits: &[Vec<(usize, usize)>], b0: usize, ngpu: usize, keep: bool) -> Vec<usize> {
    let n = raw.count();
    let wgen: usize = std::env::var("MORPHEUS_GEN_WEIGHT").ok().and_then(|s| s.parse().ok()).unwrap_or(5);
    let nblk = raw.src_len() >> b0;
    let dense = no_skip();
    let per_src: Vec<usize> = (0..n).into_par_iter().map(|s| {
        if dense { return nblk; }
        let mut bit = vec![false; nblk];
        for &(base, lg) in &bits[s] { let (a, c) = (base >> b0, 1usize << lg.saturating_sub(b0)); for x in bit.iter_mut().skip(a).take(c) { *x = true; } }
        (0..nblk).into_par_iter().map(|k| if !raw.is_zero(s, k << b0, 1usize << b0) { if bit[k] { 1 } else { wgen } } else { 0 }).sum::<usize>()
    }).collect();
    let w0: f64 = if keep { std::env::var("MORPHEUS_SHARD0_WEIGHT").ok().and_then(|s| s.parse().ok()).unwrap_or(0.5) } else { 1.0 };
    let cap: Vec<f64> = (0..ngpu).map(|g| if g == 0 { w0 } else { 1.0 }).collect();
    let mut by_load: Vec<usize> = (0..n).collect();
    by_load.sort_by_key(|&s| std::cmp::Reverse(per_src[s]));
    let mut load = vec![0usize; ngpu];
    let mut owner = vec![0usize; n];
    for s in by_load {
        let g = (0..ngpu).min_by(|&a, &b| ((load[a] + per_src[s] + 1) as f64 / cap[a]).partial_cmp(&((load[b] + per_src[s] + 1) as f64 / cap[b])).unwrap()).unwrap();
        owner[s] = g;
        load[g] += per_src[s] + 1;
    }
    owner
}

/// Device g's packed sources (those it owns, in order) on device g.
pub fn shard_upload(raw: &dyn crate::morpheus::packed::Sources, owner: &[usize], g: usize) -> DeviceBuffer<u8> {
    almost_goldilocks_cuda::set_device(g as i32).expect("set device");
    let src_len = raw.src_len();
    let srcs: Vec<usize> = (0..raw.count()).filter(|&s| owner[s] == g).collect();
    let d = DeviceBuffer::<u8>::new((srcs.len() * src_len / 4).max(1)).expect("raw alloc");
    // Several copy threads (pageable copies of one thread run at a few GB/s).
    let nt = std::env::var("MORPHEUS_COPY_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(4usize).max(1);
    let (base, len) = (d.as_ptr() as usize, d.len());
    std::thread::scope(|sc| {
        for t in 0..nt {
            let srcs = &srcs;
            sc.spawn(move || {
                almost_goldilocks_cuda::set_device(g as i32).expect("set device");
                for j in (t..srcs.len()).step_by(nt) {
                    let at = j * src_len / 4;
                    let own;
                    let bytes: &[u8] = match raw.packed(srcs[j]) { Some(x) => &x.b, None => { own = crate::morpheus::packed::Src::from_i8(&raw.to_i8(srcs[j])); &own.b } };
                    assert!(at + bytes.len() <= len);
                    let rc = unsafe { almost_goldilocks_cuda::ffi::cuda_memcpy_htod((base + at) as *mut std::ffi::c_void, bytes.as_ptr() as *const std::ffi::c_void, bytes.len()) };
                    assert_eq!(rc, 0, "packed upload");
                }
            });
        }
    });
    d
}

pub fn phases_v2(
    p: &Profile, st: &Statement, raw: &dyn crate::morpheus::packed::Sources, b0: usize, alpha: &[K8], eta: &K8, gp: &[K8],
    degs: &[usize], a_coef: &[Vec<K8>], tr: &mut MTranscript, dev_raw: Option<&DeviceBuffer<i8>>,
    keep: Option<&std::sync::Mutex<Vec<Kept>>>, pre_raw: Option<PreShards>,
) -> (Vec<Vec<K8>>, Vec<K8>, Vec<K8>, Vec<K8>, Vec<K8>, K8, K8) {
    // The device sources are packed (two bits per digit, packed.rs); the
    // resident int8 copy of small requests is not used here.
    let dev_raw: Option<&DeviceBuffer<i8>> = { let _ = dev_raw; None };
    // Device shards (MORPHEUS_GPUS): sources go to devices greedily by
    // weighted load, from the raw digits and the declared bit blocks alone,
    // so the uploads run while the host prepares terms and tables. Plane
    // rounds 0..3 run the generic kernel on non-one-hot blocks only (bit
    // blocks stand in for one-hot ones here); streamed mode scales device 0
    // down (MORPHEUS_SHARD0_WEIGHT, default 0.5), it also hosts the
    // authentication.
    let n = raw.count();
    let ngpu = shard_gpus(n);
    // A placement made during the PIOP (with its uploaded shards) is reused;
    // any placement gives the same proof.
    let (owner, mut pre) = match pre_raw {
        Some(pr) if pr.owner.len() == n && pr.bufs.len() == ngpu => (pr.owner, pr.bufs),
        _ => (shard_owner(raw, &st.bits, b0, ngpu, keep.is_some()), Vec::new()),
    };
    pre.resize_with(ngpu, || None);
    std::thread::scope(|sc| {
        let ups: Vec<Option<std::thread::ScopedJoinHandle<'_, DeviceBuffer<u8>>>> = (0..ngpu).map(|g| {
            if g == 0 && dev_raw.is_some() { return None; }
            if let Some(b) = pre[g].take() { return Some(sc.spawn(move || b)); }
            let owner = &owner;
            Some(sc.spawn(move || shard_upload(raw, owner, g)))
        }).collect();
        phases_v2_body(p, st, raw, b0, alpha, eta, gp, degs, a_coef, tr, dev_raw, keep, owner.clone(), ups)
    })
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn phases_v2_body<'u>(
    p: &Profile, st: &Statement, raw: &dyn crate::morpheus::packed::Sources, b0: usize, alpha: &[K8], eta: &K8, gp: &[K8],
    degs: &[usize], a_coef: &[Vec<K8>], tr: &mut MTranscript, dev_raw: Option<&DeviceBuffer<i8>>,
    keep: Option<&std::sync::Mutex<Vec<Kept>>>, owner: Vec<usize>, mut ups: Vec<Option<std::thread::ScopedJoinHandle<'u, DeviceBuffer<u8>>>>,
) -> (Vec<Vec<K8>>, Vec<K8>, Vec<K8>, Vec<K8>, Vec<K8>, K8, K8) {
    let tm = std::env::var("MORPHEUS_TIMING").is_ok();
    let mut t0 = std::time::Instant::now();
    let mut lap = |what: &str| { if tm { eprintln!("[link2] {what:28} {:8.1} ms", t0.elapsed().as_secs_f64() * 1e3); } t0 = std::time::Instant::now(); };
    let ell = p.ell();
    let n = st.n;
    let nvars = ell + 1;
    // ---- Regions and terms (grouped by (source, base)).
    let mut reg_index: HashMap<(usize, usize), usize> = HashMap::new();
    struct Reg { src: usize, base: usize, m: usize, p: usize, terms: Vec<usize> }
    let mut regs: Vec<Reg> = Vec::new();
    let mut terms: Vec<TermInfo> = Vec::new();
    let mut tcs: Vec<K8> = Vec::new();
    let mut treg: Vec<usize> = Vec::new();
    // Term data per row in parallel; regions assigned in row order.
    let per_row: Vec<Vec<(TermInfo, K8)>> = st.rows.par_iter().enumerate().map(|(j, row)| {
        let coef = eta.mul(&gp[2 * n + j]);
        row.terms.iter().map(|t| (TermInfo { p: t.region.p, pw: t.pw.clone(), r: t.r.clone() }, coef.mul(&t.scale))).collect()
    }).collect();
    for (row, infos) in st.rows.iter().zip(per_row) {
        for (t, (info, c)) in row.terms.iter().zip(infos) {
            let key = (t.region.source, t.region.base);
            let ri = *reg_index.entry(key).or_insert_with(|| { regs.push(Reg { src: key.0, base: key.1, m: t.region.m, p: t.region.p, terms: Vec::new() }); regs.len() - 1 });
            let ti = terms.len();
            terms.push(info);
            tcs.push(c);
            treg.push(ri);
            regs[ri].terms.push(ti);
        }
    }
    lap("terms");
    let bits: std::collections::HashSet<(usize, usize)> = st.bits.iter().enumerate().flat_map(|(s, bl)| bl.iter().map(move |&(b, _)| (s, b))).collect();
    // Terms in region order (each region's terms contiguous).
    let order: Vec<usize> = regs.iter().flat_map(|r| r.terms.iter().copied()).collect();
    let pos_of: Vec<usize> = { let mut v = vec![0; order.len()]; for (k, &t) in order.iter().enumerate() { v[t] = k; } v };
    let mut terms_c: Vec<TermC> = vec![TermC::default(); order.len()];
    // Split high-index eq tables of every term, written in parallel into
    // flat buffers at precomputed offsets.
    // Dense regions (many planes or many terms) fold their terms into dense
    // query tables; their terms need no index tables.
    let is_dense = |r: &Reg| r.p > DENSE_P || r.terms.len() > 32;
    let hsz: Vec<(usize, usize, usize)> = order.iter().map(|&t| {
        let r = &regs[treg[t]];
        if is_dense(r) { return (0, 0, 0); }
        let h = r.m + r.p - b0;
        (h / 2, 1usize << (h / 2), 1usize << (h - h / 2))
    }).collect();
    let (mut lo_off, mut hi_off) = (Vec::with_capacity(order.len()), Vec::with_capacity(order.len()));
    let (mut nlo, mut nhi) = (0usize, 0usize);
    for &(_, a, b) in &hsz { lo_off.push(nlo); hi_off.push(nhi); nlo += a; nhi += b; }
    let mut hl_all = vec![K8::ZERO; nlo];
    let mut hh_all = vec![K8::ZERO; nhi];
    {
        let (pl, ph) = (SyncPtr(hl_all.as_mut_ptr()), SyncPtr(hh_all.as_mut_ptr()));
        order.par_iter().enumerate().for_each(|(k, &t)| {
            let r = &regs[treg[t]];
            if hsz[k].1 == 0 { return; }
            let idx_bits = b0 - r.p;
            let h1 = hsz[k].0;
            let lo = eq_table(&terms[t].r[idx_bits..idx_bits + h1]);
            let hi = eq_table(&terms[t].r[idx_bits + h1..]);
            // Disjoint ranges [off, off + len) per term.
            unsafe {
                std::ptr::copy_nonoverlapping(lo.as_ptr(), pl.get().add(lo_off[k]), lo.len());
                std::ptr::copy_nonoverlapping(hi.as_ptr(), ph.get().add(hi_off[k]), hi.len());
            }
        });
    }
    for (k, &t) in order.iter().enumerate() {
        terms_c[k] = TermC { pa: 0, aoff: 0, boff: 0, hl_off: lo_off[k] as u32, hh_off: hi_off[k] as u32, h1: hsz[k].0 as u32, pad0: 0, pad1: 0, cs: tcs[t].words() };
    }
    lap("hi tables");
    // One-hot detection (bit-typed regions whose indices each hold one 1).
    let dense = dense_link();
    let no_bin = dense || std::env::var("MORPHEUS_NO_BIN").is_ok();
    let onehot: Vec<bool> = regs.par_iter().map(|r| {
        if dense || r.p == 0 || r.p > DENSE_P || !bits.contains(&(r.src, r.base)) { return false; }
        let np = 1usize << r.p;
        (0..1usize << r.m).into_par_iter().with_min_len(1 << 12).all(|k| raw.nonzeros(r.src, r.base + k * np, np) == (1, true))
    }).collect();
    lap("one-hot detection");
    // Dense query tables (many-plane regions). When every dense region's terms
    // share one plane-weight vector pw (and one plane count p), a block's table
    // is a_k ⊗ pw and only a_k (2^{b0−p} entries) is kept ("factored"); each
    // round's folded tables are expanded on the device (`dense_round_tables`).
    let fac_p: Option<usize> = {
        let dr: Vec<&Reg> = regs.iter().filter(|r| r.p > DENSE_P || r.terms.len() > 32).collect();
        let ok = !dr.is_empty() && !dense && std::env::var("MORPHEUS_NO_FACTOR").is_err()
            && dr.iter().all(|r| r.p == dr[0].p && r.terms.iter().all(|&t| std::sync::Arc::ptr_eq(&terms[t].pw, &terms[r.terms[0]].pw)));
        ok.then_some(dr[0].p)
    };
    let tsz = match fac_p { Some(pp) => 1usize << (b0 - pp), None => 1usize << b0 };
    let mut dense_pw: Vec<std::sync::Arc<Vec<K8>>> = Vec::new();
    let mut tab_pw: Vec<u32> = Vec::new();
    let mut qidx: Vec<u32> = Vec::new();
    let mut regions_c: Vec<RegionC> = Vec::new();
    let mut toff = 0u32;
    // Per region (in parallel): flags and, for dense regions, the block
    // tables (block k, table); then assembled in region order.
    let per: Vec<(u32, Vec<(usize, Vec<K8>)>)> = regs.par_iter().enumerate().map(|(ri, r)| {
        let h = r.m + r.p - b0;
        let dense = r.p > DENSE_P || r.terms.len() > 32;
        let mut flags = 0u32;
        if bits.contains(&(r.src, r.base)) { flags |= 1; }
        if onehot[ri] && r.terms.len() <= 4 && r.p <= 5 { flags |= 2; }
        // Binned one-hot (see `onehot_bin_kernel`): round i = p from class bins.
        if flags & 2 != 0 && r.p == 4 && r.terms.len() <= 2 && b0 == 11 && !(r.p > DENSE_P || r.terms.len() > 32) && !no_bin { flags |= 8; }
        if !dense { return (flags, Vec::new()); }
        flags |= 4;
        let idx_bits = b0 - r.p;
        // (block k, term) pairs: a term whose high point bits are boolean
        // addresses one block, others every block. Each block's table is
        // accumulated in place (no per-term contribution vectors).
        let kb = |ti: &TermInfo| -> Option<usize> {
            if ti.r[idx_bits..].iter().all(|x| *x == K8::ZERO || *x == K8::ONE) { Some(ti.r[idx_bits..].iter().enumerate().fold(0usize, |a, (jj, x)| a | (((*x == K8::ONE) as usize) << jj))) } else { None }
        };
        let mut pairs: Vec<(usize, usize)> = r.terms.par_iter().flat_map_iter(|&t| match kb(&terms[t]) {
            Some(k) => vec![(k, t)],
            None => (0..1usize << h).map(|k| (k, t)).collect(),
        }).collect();
        pairs.par_sort_unstable();
        let mut groups: Vec<(usize, usize, usize)> = Vec::new();
        for (q, &(k, _)) in pairs.iter().enumerate() {
            match groups.last_mut() { Some(g) if g.0 == k => g.2 = q + 1, _ => groups.push((k, q, q + 1)) }
        }
        let rp = r.p;
        let fac = fac_p.is_some();
        let tabs: Vec<(usize, Vec<K8>)> = groups.par_iter().map(|&(k, a, b)| {
            let mut tab = vec![K8::ZERO; tsz];
            for &(_, t) in &pairs[a..b] {
                if fac {
                    let ti = &terms[t];
                    let hi = ti.r[idx_bits..].iter().enumerate().fold(K8::ONE, |acc, (jj, x)| acc.mul(&eqb(x, (k >> jj) & 1)));
                    let c = tcs[t].mul(&hi);
                    for (il, e) in eq_table(&ti.r[..idx_bits]).iter().enumerate() { tab[il] = tab[il].add(&c.mul(e)); }
                    continue;
                }
                let ti = &terms[t];
                let hi = ti.r[idx_bits..].iter().enumerate().fold(K8::ONE, |acc, (jj, x)| acc.mul(&eqb(x, (k >> jj) & 1)));
                let c = tcs[t].mul(&hi);
                let el = eq_table(&ti.r[..idx_bits]);
                for (il, e) in el.iter().enumerate() {
                    if e.is_zero() { continue; }
                    let ce = c.mul(e);
                    let row = &mut tab[il << rp..(il + 1) << rp];
                    for (bb, w) in ti.pw.iter().enumerate().take(1usize << rp) { if !w.is_zero() { row[bb] = row[bb].add(&ce.mul(w)); } }
                }
            }
            (k, tab)
        }).collect();
        if std::env::var("MORPHEUS_LINK_STATS").is_ok() {
            let mut ptrs: std::collections::HashSet<usize> = Default::default();
            let mut vals: std::collections::HashSet<Vec<[u64; 8]>> = Default::default();
            for &t in &r.terms { ptrs.insert(std::sync::Arc::as_ptr(&terms[t].pw) as usize); if vals.len() < 64 { vals.insert(terms[t].pw.iter().map(|x| x.0).collect()); } }
            eprintln!("[densestats] region p={} m={} h={h} terms {} pairs {} tables {}; distinct pw arcs {} (values among first: {})", r.p, r.m, r.terms.len(), pairs.len(), tabs.len(), ptrs.len(), vals.len());
        }
        (flags, tabs)
    }).collect();
    let ntab: usize = per.iter().map(|x| x.1.len()).sum();
    let mut dst: Vec<(usize, &[K8])> = Vec::with_capacity(ntab);
    for (ri, r) in regs.iter().enumerate() {
        let h = r.m + r.p - b0;
        let (flags, tabs) = &per[ri];
        let dense = flags & 4 != 0;
        let qd_off = qidx.len() as u32;
        if dense {
            if fac_p.is_some() { dense_pw.push(terms[r.terms[0]].pw.clone()); }
            let mut ix = vec![u32::MAX; 1usize << h];
            for (k, tab) in tabs { ix[*k] = dst.len() as u32; dst.push((dst.len(), &tab[..])); if fac_p.is_some() { tab_pw.push(dense_pw.len() as u32 - 1); } }
            qidx.extend(ix);
        }
        let tcnt = if dense { 0 } else { r.terms.len() as u32 };
        assert!(tcnt as usize <= 32, "region with more than 32 query terms");
        regions_c.push(RegionC { src: r.src as u32, base_blk: (r.base >> b0) as u32, h: h as u32, p: r.p as u32, flags: *flags, toff, tcnt, qd_off });
        toff += r.terms.len() as u32;
    }
    // Every table at its index, copied in parallel.
    let mut qtabs: Vec<K8> = Vec::with_capacity(ntab * tsz);
    unsafe { qtabs.set_len(ntab * tsz); }
    qtabs.par_chunks_mut(tsz.max(1)).zip(dst.par_iter()).for_each(|(o, (_, t))| o.copy_from_slice(t));
    drop(dst);
    drop(per);
    // One-hot regions (never dense-q) keep, in qd_off, the base of their
    // terms in the compact per-term bin sums of the one-hot pass.
    let mut n_ohw = 0u32;
    for r in regions_c.iter_mut() { if r.flags & 2 != 0 { r.qd_off = n_ohw; n_ohw += r.tcnt; } }
    lap("dense tables + regions");
    if std::env::var("MORPHEUS_LINK_STATS").is_ok() {
        // Term work: region blocks x query terms, by plane bits and term count.
        let mut agg: std::collections::BTreeMap<(usize, usize, bool), (usize, usize)> = Default::default();
        for (ri, r) in regs.iter().enumerate() {
            let rc = &regions_c[ri];
            let nb = 1usize << rc.h;
            let tb = match r.terms.len() { 0..=2 => 2, 3..=4 => 4, 5..=8 => 8, 9..=16 => 16, 17..=32 => 32, _ => 999 };
            let e = agg.entry((r.p, tb, rc.flags & 2 != 0)).or_default();
            e.0 += nb; e.1 += nb * rc.tcnt as usize;
        }
        let tot: usize = agg.values().map(|v| v.1).sum();
        eprintln!("[linkstats] total block-terms {tot}");
        let mut supp = 0usize; let mut supp_oh = 0usize; let mut tot_oh = 0usize;
        for (ri, r) in regs.iter().enumerate() {
            let rc = &regions_c[ri];
            if rc.flags & 4 != 0 { continue; }
            let idx_bits = b0 - r.p;
            for &t in &r.terms {
                let fr = terms[t].r[idx_bits..].iter().filter(|x| **x != K8::ZERO && **x != K8::ONE).count();
                supp += 1usize << fr;
                if rc.flags & 2 != 0 { supp_oh += 1usize << fr; tot_oh += 1usize << rc.h; }
            }
        }
        eprintln!("[linkstats] supported block-terms {supp}; onehot supported {supp_oh} of {tot_oh}");
        let mut v: Vec<_> = agg.into_iter().collect(); v.sort_by_key(|x| std::cmp::Reverse(x.1 .1));
        for ((p, tb, oh), (nb, w)) in v.iter().take(20) { eprintln!("[linkstats] p={p:2} terms<={tb:3} onehot={oh:5} blocks {nb:9} block-terms {w:10} ({:4.1}%)", 100.0 * *w as f64 / tot.max(1) as f64); }
    }
    // Nonzero blocks and their regions.
    let mut reg_sorted: Vec<(usize, usize, usize, usize)> = regs.iter().enumerate().map(|(i, r)| (r.src, r.base >> b0, 1usize << (r.m + r.p - b0), i)).collect();
    reg_sorted.sort();
    let nblk_src = 1usize << (ell - b0);
    let find = |s: usize, id: usize| -> Option<(usize, usize)> {
        let i = reg_sorted.partition_point(|x| (x.0, x.1) <= (s, id));
        if i == 0 { return None; }
        let (rs, rb, rn, ri) = reg_sorted[i - 1];
        if rs == s && id < rb + rn { Some((ri, id - rb)) } else { None }
    };
    let skip_off = no_skip();
    let blocks: Vec<Blk> = (0..n).into_par_iter().flat_map_iter(|s| {
        (0..nblk_src).filter(move |&id| skip_off || !raw.is_zero(s, id << b0, 1usize << b0)).map(move |id| (s, id)).collect::<Vec<_>>()
    }).map(|(s, id)| match find(s, id) { Some((ri, k)) => Blk { src: s as u32, id: id as u32, region: ri as u32, k: k as u32 }, None => Blk { src: s as u32, id: id as u32, region: u32::MAX, k: 0 } }).collect();
    lap("nonzero block scan");
    if std::env::var("MORPHEUS_LINK_STATS").is_ok() {
        // Generic (non one-hot) blocks: digit values, nonzero pairs at rounds 1..3,
        // by region kind (p, has terms).
        let mut agg: std::collections::BTreeMap<(i64, bool), [u64; 9]> = Default::default();
        for b in &blocks {
            let (p, ht) = if b.region == u32::MAX { (-1i64, false) } else { let r = &regions_c[b.region as usize]; if r.flags & 2 != 0 { continue; } (r.p as i64, r.tcnt > 0 || r.flags & 4 != 0) };
            let mut d = vec![0i8; 1usize << b0];
            raw.read(b.src as usize, (b.id as usize) << b0, &mut d);
            let e = agg.entry((p, ht)).or_default();
            e[0] += 1;
            for &x in &d { match x { 0 => e[1] += 1, 1 => e[2] += 1, -1 => e[3] += 1, _ => e[4] += 1 } }
            for (k, i) in [(5usize, 1usize), (6, 2), (7, 3)] {
                let w = 1usize << (i + 1);
                e[k] += d.chunks(w).filter(|c| c.iter().any(|&x| x != 0)).count() as u64;
            }
            let mut cls = std::collections::HashSet::new();
            for c in d.chunks(4) { cls.insert((c[0], c[1], c[2], c[3])); }
            e[8] += cls.len() as u64;
        }
        for ((p, ht), e) in &agg {
            let n = e[0].max(1) as f64;
            eprintln!("[genstats] p={p:2} terms={ht:5} blocks {:8} digits: 0 {:.3} +1 {:.3} -1 {:.3} other {:.3}; nonzero pairs/block r1 {:.1} r2 {:.1} r3 {:.1}; distinct r1 classes/block {:.1}",
                e[0], e[1] as f64 / (n * 2048.0), e[2] as f64 / (n * 2048.0), e[3] as f64 / (n * 2048.0), e[4] as f64 / (n * 2048.0), e[5] as f64 / n, e[6] as f64 / n, e[7] as f64 / n, e[8] as f64 / n);
        }
    }
    let (rid, kk): (Vec<u32>, Vec<u32>) = regions_c.iter().enumerate().flat_map(|(i, r)| (0..1u32 << r.h).map(move |k| (i as u32, k))).unzip();
    lap("region block list");

    // ---- Device shards (MORPHEUS_GPUS, default 1): contiguous source ranges
    // balanced by nonzero blocks; each device holds only its range (shard 0
    // may use the resident `dev_raw` holding every source). The coarse phase
    // also runs per shard.
    let src_len = raw.src_len() as u64;
    let ca_lo_bits = (ell - b0) / 2;
    let calo_h = eq_table(&alpha[b0..b0 + ca_lo_bits]);
    let cahi_h = eq_table(&alpha[b0 + ca_lo_bits..]);
    let mut claims: Option<(K8, K8)> = None;
    let no_claim = dense || std::env::var("MORPHEUS_NO_CLAIM").is_ok();
    // Blocks grouped by region plane bits, so each round's one-hot list is a prefix.
    let is_oh = |b: &Blk| b.region != u32::MAX && regions_c[b.region as usize].flags & 2 != 0;
    lap("eq tables");
    // One-hot blocks in a stable parallel bucket pass on the small key
    // (binned, p), descending.
    let oh: Vec<Blk> = {
        let key = |b: &Blk| -> usize { if !is_oh(b) { return 256; } let r = &regions_c[b.region as usize]; 255 - (((r.flags & 8 != 0) as usize) * 128 + r.p as usize).min(255) };
        let ch = blocks.len().div_ceil(256).max(1 << 14);
        let hist: Vec<[usize; 256]> = blocks.par_chunks(ch).map(|c| { let mut h = [0usize; 256]; for b in c { let k = key(b); if k < 256 { h[k] += 1; } } h }).collect();
        let mut off = vec![[0usize; 256]; hist.len()];
        let mut tot = 0usize;
        for k in 0..256 { for (c, h) in hist.iter().enumerate() { off[c][k] = tot; tot += h[k]; } }
        let mut out: Vec<Blk> = Vec::with_capacity(tot);
        let ptr = RawBlk(out.as_mut_ptr());
        blocks.par_chunks(ch).zip(off.into_par_iter()).for_each(|(c, mut o)| {
            let ptr = ptr;
            for b in c { let k = key(b); if k < 256 { unsafe { ptr.0.add(o[k]).write(*b); } o[k] += 1; } }
        });
        unsafe { out.set_len(tot); }
        out
    };
    lap("one-hot order");
    let maxp_oh = oh.iter().map(|b| regions_c[b.region as usize].p as usize).max().unwrap_or(0);
    let n_gen_base = blocks.len() - oh.len();
    if tm { eprintln!("[link2] {} nonzero blocks ({} one-hot), {} regions, {} terms, {} region blocks, {} dense tables, {} GPUs", blocks.len(), oh.len(), regions_c.len(), order.len(), rid.len(), qtabs.len() / tsz, ups.len()); }
    let ngpu = ups.len();
    let mut shards: Vec<Shard> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..ngpu).map(|g| {
            let up = ups[g].take();
            let (owner, blocks, oh, regions_c, terms_c, qtabs, qidx, hl_all, hh_all, tab_pw) = (&owner, &blocks, &oh, &regions_c, &terms_c, &qtabs, &qidx, &hl_all, &hh_all, &tab_pw);
            let (calo_h, cahi_h) = (&calo_h, &cahi_h);
            sc.spawn(move || {
                almost_goldilocks_cuda::set_device(g as i32).expect("set device");
                // Local numbering: owned sources in order (shard 0 over the
                // resident `dev_raw` keeps the global numbering).
                let global = g == 0 && dev_raw.is_some();
                let srcs: Vec<usize> = (0..n).filter(|&s| owner[s] == g).collect();
                let mut loc = vec![u32::MAX; n];
                if global { for s in 0..n { loc[s] = s as u32; } } else { for (j, &s) in srcs.iter().enumerate() { loc[s] = j as u32; } }
                let t_sh = std::time::Instant::now();
                let raw_own = if global { None } else { Some(up.expect("shard upload").join().expect("raw upload")) };
                let t_raw = t_sh.elapsed().as_secs_f64() * 1e3;
                let mine = |b: &Blk| owner[b.src as usize] == g;
                let rl = |b: &Blk| Blk { src: loc[b.src as usize], ..*b };
                let gen_base: Vec<Blk> = blocks.par_iter().filter(|b| mine(b) && !(b.region != u32::MAX && regions_c[b.region as usize].flags & 2 != 0)).map(rl).collect();
                let oh_g: Vec<Blk> = oh.par_iter().filter(|b| mine(b)).map(rl).collect();
                let t_f = t_sh.elapsed().as_secs_f64() * 1e3;
                let mut gen_all = gen_base.clone();
                gen_all.extend(oh_g.iter().rev().copied());
                // Per-block constant slots: cr, cb and one per region term.
                let mut bco: Vec<u64> = Vec::with_capacity(gen_all.len());
                let mut bc_len = 0u64;
                for b in &gen_all {
                    bco.push(bc_len);
                    let tc = if b.region == u32::MAX { 0 } else { regions_c[b.region as usize].tcnt as u64 };
                    bc_len += 2 + tc;
                }
                let bc_len = bc_len as usize;
                let mut regions_l: Vec<RegionC> = regions_c.iter().map(|r| RegionC { src: loc[r.src as usize], ..*r }).collect();
                // Dense query tables of this device's dense regions only.
                let (mut qt_g, mut qi_g, mut tp_g): (Vec<K8>, Vec<u32>, Vec<u32>) = (Vec::new(), Vec::new(), Vec::new());
                let tl = tsz;
                for (ri, r) in regions_c.iter().enumerate() {
                    if r.flags & 4 == 0 || owner[r.src as usize] != g { continue; }
                    let off = qi_g.len() as u32;
                    for k in 0..1usize << r.h {
                        let q = qidx[r.qd_off as usize + k];
                        if q == u32::MAX { qi_g.push(u32::MAX); continue; }
                        qi_g.push((qt_g.len() / tl) as u32);
                        qt_g.extend_from_slice(&qtabs[q as usize * tl..(q as usize + 1) * tl]);
                        if fac_p.is_some() { tp_g.push(tab_pw[q as usize]); }
                    }
                    regions_l[ri].qd_off = off;
                }
                let t_d = t_sh.elapsed().as_secs_f64() * 1e3;
                let order_l: Vec<usize> = if global { (0..n).collect() } else { srcs.clone() };
                let gr: Vec<K8> = order_l.iter().map(|&s| gp[s]).collect();
                let gb: Vec<K8> = order_l.iter().map(|&s| gp[n + s]).collect();
                let mut sh = Shard {
                    dev: g as i32, raw: raw_own, srcs: order_l, owned: srcs.clone(),
                    calo: up_k(calo_h), cahi: up_k(cahi_h), grange: up_k(if gr.is_empty() { &[K8::ZERO][..] } else { &gr }), gbit: up_k(if gb.is_empty() { &[K8::ZERO][..] } else { &gb }),
                    regions: upload(&regions_l), hl: up_k(hl_all), hh: up_k(hh_all),
                    qd: up_k(if qt_g.is_empty() || fac_p.is_some() { &[K8::ZERO] } else { &qt_g }), qdl: None, qdl_len: if fac_p.is_some() { 0 } else { qt_g.len() },
                    fac: fac_p.map(|_| (qt_g.clone(), tp_g.clone())), bdev: None, b_round: None,
                    qidx: DeviceBuffer::from_slice(if qi_g.is_empty() { &[0u32][..] } else { &qi_g }).unwrap(),
                    gen_base: gen_base.len(), oh: oh_g, gen_all: upload(&gen_all), gen_all_h: gen_all.clone(),
                    bc: DeviceBuffer::<u64>::new(bc_len.max(1) * 8).unwrap(), bco: DeviceBuffer::from_slice(if bco.is_empty() { &[0u64][..] } else { &bco }).unwrap(), ca_lo_bits,
                };
                let d_t0 = upload(terms_c);
                let c0 = sh.consts(&d_t0, 0, 0, dptr(&sh.qd));
                let rc = unsafe { ffi::agl_mlink2_consts(sh.gen_all.as_ptr() as *const c_void, gen_all.len() as u32, &c0 as *const ConstsC as *const c_void, sh.bc.as_mut_ptr()) };
                assert_eq!(rc, 0, "block constants");
                if tm { eprintln!("[link2]   shard {g}: raw wait {t_raw:.0} ms, tables {:.0} ms (filters {:.0}, block constants + dense {:.0}, uploads {:.0}; {} blocks, {} dense entries)", t_sh.elapsed().as_secs_f64() * 1e3 - t_raw, t_f - t_raw, t_d - t_f, t_sh.elapsed().as_secs_f64() * 1e3 - t_d, gen_all.len(), qt_g.len()); }
                sh
            })
        }).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    lap("upload");
    // One-hot plane rounds: bin sums from one pass (round 0), then host work.
    let mut oh_sums: Option<OhSums> = None;
    let mut oh_terms: Vec<Vec<(usize, usize)>> = vec![Vec::new(); 6];
    {
        let mut seen = std::collections::HashSet::new();
        for b in &oh {
            if seen.insert(b.region) {
                let r = &regions_c[b.region as usize];
                for k in 0..r.tcnt as usize { oh_terms[r.p as usize].push((r.toff as usize + k, r.qd_off as usize + k)); }
            }
        }
    }
    let mut bin_terms: Vec<(usize, usize)> = Vec::new();
    for r in regions_c.iter().filter(|r| r.flags & 8 != 0) { for k in 0..r.tcnt as usize { bin_terms.push((r.toff as usize + k, r.qd_off as usize + k)); } }
    let bin_terms_all: Vec<usize> = regions_c.iter().enumerate().filter(|(_, r)| r.flags & 8 != 0).map(|(i, _)| i).collect();
    // Round-5 B tables of the binned terms (eq over index bits 2.., 32 entries), by compact slot.
    let mut b5 = vec![K8::ZERO; (n_ohw as usize).max(1) * 32];
    for &(k, kw) in &bin_terms { b5[kw * 32..(kw + 1) * 32].copy_from_slice(&eq_table(&terms[order[k]].r[2..7])); }
    // Fixed device offsets of the per-term B tables (round-0 sizes), and
    // their round-0 contents as eq tables over the index coordinates (built
    // on the device: `b_build_kernel`). Round 0 binds a plane bit (p > 0) or
    // index bit 0 (p = 0).
    let mut boff0 = vec![0u32; order.len()];
    let mut b_total = 0usize;
    let mut bseg = BSeg::default();
    for (k, &t) in order.iter().enumerate() {
        boff0[k] = b_total as u32;
        let r = &regs[treg[t]];
        if !(r.p > DENSE_P || r.terms.len() > 32) {
            let idx_bits = b0 - r.p;
            let s0 = if r.p > 0 { 0 } else { 1 };
            bseg.co.push((bseg.coords.len() / 8) as u32);
            bseg.nv.push((idx_bits - s0) as u8);
            bseg.off.push(boff0[k]);
            bseg.pre.push(bseg.total);
            bseg.total += 1u64 << (idx_bits - s0);
            bseg.coords.extend(terms[t].r[s0..idx_bits].iter().flat_map(|x| x.0));
            b_total += 1usize << idx_bits;
        }
    }
    // Round-1 range sums of the generic blocks from digit classes (`range_bin1_kernel`).
    let rb1 = b0 >= 3 && !dense && std::env::var("MORPHEUS_NO_RB1").is_err();
    let mut rb1_bins: Vec<K8> = Vec::new();
    let mut rounds = Vec::new();
    let mut xi: Vec<K8> = Vec::new();
    let mut mo_prefix = K8::ZERO;
    let mut pre_alpha = K8::ONE;
    let mut kms = 0.0;
    for i in 0..b0 {
        let d = degs[i];
        let t_round = std::time::Instant::now();
        // Rows t = 0, 1 only (weights are linear in t); dense regions' terms
        // live in the dense query tables.
        // A tables on the host; B tables (eq over the unbound index bits) live
        // on the device at fixed offsets: built once at round 0, then halved by
        // pairwise sums when a term's index bit is bound (`b_halve_kernel`).
        let tabs: Vec<(Vec<K8>, Vec<K8>, usize)> = order.par_iter().map(|&t| {
            if regs[treg[t]].p > DENSE_P || regs[treg[t]].terms.len() > 32 { (Vec::new(), Vec::new(), 0) }
            else { let (a, pa) = term_a(&terms[t], &xi, i, 1, b0); (a, Vec::new(), pa) }
        }).collect();
        let mut a_all = Vec::new();
        let mut tc = terms_c.clone();
        for (k, (a, _, pa)) in tabs.iter().enumerate() {
            tc[k].pa = *pa as u32; tc[k].aoff = (a_all.len()) as u32; tc[k].boff = boff0[k];
            a_all.extend_from_slice(a);
        }
        let b_all: Vec<K8> = Vec::new();
        // Terms whose B halves entering round i (index bit bound in round i − 1).
        let halve: Vec<(u32, u64)> = if i == 0 { Vec::new() } else {
            order.iter().enumerate().filter_map(|(k, &t)| {
                let r = &regs[treg[t]];
                if r.p > DENSE_P || r.terms.len() > 32 || i < r.p { return None; }
                Some((boff0[k], 1u64 << (b0 - r.p - 1 - (i - r.p))))
            }).collect()
        };
        let t_tab = t_round.elapsed().as_secs_f64() * 1e3;
        let e1 = eq_table(&xi);
        let e2: Vec<K8> = e1.iter().map(|x| x.mul(x)).collect();
        let e3: Vec<K8> = e1.iter().zip(&e2).map(|(x, y)| x.mul(y)).collect();
        let ea = eq_table(&alpha[i + 1..b0]);
        let n_oh = if i < maxp_oh { 1 } else { 0 };
        if tm { eprintln!("[link2]   round {i} host: tables {t_tab:.1} ms, prep {:.1} ms (A {} K, B {} K)", t_round.elapsed().as_secs_f64() * 1e3, a_all.len(), b_all.len()); }
        let t = std::time::Instant::now();
        // Reduced rounds: range at t = 0, 2, 3 and query at t = 0, 2 on the
        // device; t = 1 from the previous round's claims
        //   R_{i-1}(r) = (1 − α_i) R_i(0) + α_i R_i(1),  Q_{i-1}(r) = Q_i(0) + Q_i(1).
        let reduced = d == 4 && claims.is_some() && !alpha[i].is_zero() && !no_claim;
        let use_index = reduced && (1usize << (b0 - i)) <= 128 && i >= 4 && b0 >= 9 && !dense && std::env::var("MORPHEUS_NO_INDEX").is_err();
        let rin = RoundIn { i, d, b0, src_len, reduced, use_index, halve: &halve, b_total, bseg: &bseg, r_prev: xi.last().copied(), full: false, b5: &b5, xi: &xi, fac_p, dense_pw: &dense_pw, skip_range: rb1 && i == 1, rb1, a_all: &a_all, b_all: &b_all, tc: &tc, e1: &e1, e2: &e2, e3: &e3, ea: &ea };
        let dev_ms = std::sync::Mutex::new(vec![0f64; shards.len()]);
        let outs: Vec<(Vec<K8>, Option<OhSums>, Option<Vec<K8>>)> = std::thread::scope(|sc| {
            let hs: Vec<_> = shards.iter_mut().enumerate().map(|(k, sh)| {
                let (rin, regions_c, nterms, dev_ms) = (&rin, &regions_c, n_ohw as usize, &dev_ms);
                sc.spawn(move || { let t = std::time::Instant::now(); let r = shard_round(sh, dev_raw, rin, regions_c, alpha, nterms); dev_ms.lock().unwrap()[k] = t.elapsed().as_secs_f64() * 1e3; r })
            }).collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        if tm && shards.len() > 1 { eprintln!("[link2]   round {i} per device: {:?} ms", dev_ms.lock().unwrap().iter().map(|x| x.round()).collect::<Vec<_>>()); }
        let mut sum = vec![K8::ZERO; 2 * (d + 1)];
        let mut ohs: Option<OhSums> = None;
        for (s, o, rb) in outs {
            for (a, b) in sum.iter_mut().zip(&s) { *a = a.add(b); }
            if let Some(rb) = rb { if rb1_bins.is_empty() { rb1_bins = rb; } else { for (a, b) in rb1_bins.iter_mut().zip(&rb) { *a = a.add(b); } } }
            if let Some(o) = o { ohs = Some(match ohs { None => o, Some(o0) => o0.add(&o) }); }
        }
        let t_gen = t.elapsed().as_secs_f64() * 1e3;
        let sum_dev = sum.clone();
        if i == 0 && !oh.is_empty() {
            oh_sums = ohs;
            // Binned blocks' plane-round bins: G (p = 4) from GV, W from U.
            if let Some(os) = oh_sums.as_mut() {
                if !bin_terms.is_empty() || regions_c.iter().any(|r| r.flags & 8 != 0) {
                    let a4 = alpha[4];
                    for s in 0..16 {
                        for c in 0..2 {
                            let v = K8::ONE.sub(&a4).mul(&os.gv[(c * 2) * 32 + s]).add(&a4.mul(&os.gv[(c * 2 + 1) * 32 + s]));
                            os.g[(4 * 32 + s) * 2 + c] = os.g[(4 * 32 + s) * 2 + c].add(&v);
                        }
                    }
                    os.u4 = vec![K8::ZERO; os.u.len() / 2];
                    // Per term (distinct slots kw): its U4 rows and W bins.
                    let per: Vec<(usize, Vec<K8>, Vec<K8>)> = bin_terms.par_iter().map(|&(k, kw)| {
                        let (r0, r1) = (terms[order[k]].r[0], terms[order[k]].r[1]);
                        let mut u4 = vec![K8::ZERO; 64];
                        for par in 0..2 {
                            for s in 0..16 {
                                u4[par * 32 + s] = K8::ONE.sub(&r1).mul(&os.u[(kw * 4 + par) * 32 + s]).add(&r1.mul(&os.u[(kw * 4 + 2 + par) * 32 + s]));
                            }
                        }
                        let w: Vec<K8> = (0..16).map(|s| K8::ONE.sub(&r0).mul(&u4[s]).add(&r0.mul(&u4[32 + s]))).collect();
                        (kw, u4, w)
                    }).collect();
                    for (kw, u4, w) in per {
                        os.u4[kw * 64..kw * 64 + 64].copy_from_slice(&u4);
                        for s in 0..16 { os.w[kw * 32 + s] = os.w[kw * 32 + s].add(&w[s]); }
                    }
                }
            }
        }
        let binned = (i == 4 || i == 5) && !bin_terms_all.is_empty();
        if n_oh > 0 || binned {
            let os = oh_sums.as_ref().unwrap();
            let mut c = if n_oh > 0 { onehot_round(&os.g, &os.w, &oh_terms, i, b0, alpha, &e1, &e2, &e3, &a_all, &tc) } else { [K8::ZERO; 7] };
            if binned {
                let cb = if i == 4 { binned_round(os, &bin_terms, &e1, &a_all, &tc) } else { binned_round5(os, &bin_terms, &e1, &a_all, &tc) };
                if std::env::var("MORPHEUS_BIN_CHECK").is_ok() {
                    let rin2 = RoundIn { full: true, r_prev: None, ..rin };
                    let full: Vec<K8> = shards.iter_mut().map(|sh| shard_round(sh, dev_raw, &rin2, &regions_c, alpha, n_ohw as usize).0).fold(vec![K8::ZERO; 2 * (d + 1)], |a, v| a.iter().zip(&v).map(|(x, y)| x.add(y)).collect());
                    for t in [0usize, 2, 3] {
                        let tk = K8::from_base(t as u64);
                        let ev = cb[0].add(&cb[1].mul(&tk)).add(&cb[2].mul(&tk.mul(&tk))).add(&cb[3].mul(&tk.mul(&tk).mul(&tk)));
                        eprintln!("[bincheck] range t={t}: direct−rest {:?} bins {:?}", full[t].sub(&sum[t]).0[0], ev.0[0]);
                        if t < 3 {
                            let eq = cb[4].add(&cb[5].mul(&tk)).add(&cb[6].mul(&tk.mul(&tk)));
                            eprintln!("[bincheck] query t={t}: direct−rest {:?} bins {:?}", full[d + 1 + t].sub(&sum[d + 1 + t]).0[0], eq.0[0]);
                        }
                    }
                }
                for (a, b) in c.iter_mut().zip(&cb) { *a = a.add(b); }
            }
            for t in 0..=d {
                let tk = K8::from_base(t as u64);
                let (t2, t3) = (tk.mul(&tk), tk.mul(&tk).mul(&tk));
                sum[t] = sum[t].add(&c[0].add(&c[1].mul(&tk)).add(&c[2].mul(&t2)).add(&c[3].mul(&t3)));
                sum[d + 1 + t] = sum[d + 1 + t].add(&c[4].add(&c[5].mul(&tk)).add(&c[6].mul(&t2)));
            }
        }
        if i == 1 && rb1 {
            let c = range_bin1_round(&rb1_bins, &xi[0]);
            if std::env::var("MORPHEUS_BIN_CHECK").is_ok() {
                let rin2 = RoundIn { skip_range: false, r_prev: None, ..rin };
                let full: Vec<K8> = shards.iter_mut().map(|sh| shard_round(sh, dev_raw, &rin2, &regions_c, alpha, n_ohw as usize).0).fold(vec![K8::ZERO; 2 * (d + 1)], |a, v| a.iter().zip(&v).map(|(x, y)| x.add(y)).collect());
                for t in [0usize, 2, 3] {
                    let tk = K8::from_base(t as u64);
                    let ev = c[0].add(&c[1].mul(&tk)).add(&c[2].mul(&tk.mul(&tk))).add(&c[3].mul(&tk.mul(&tk).mul(&tk)));
                    eprintln!("[rb1check] range t={t}: direct−rest {:?} bins {:?}", full[t].sub(&sum_dev[t]).0[0], ev.0[0]);
                }
            }
            for t in 0..=d {
                let tk = K8::from_base(t as u64);
                sum[t] = sum[t].add(&c[0].add(&c[1].mul(&tk)).add(&c[2].mul(&tk.mul(&tk))).add(&c[3].mul(&tk.mul(&tk).mul(&tk))));
            }
        }
        if reduced {
            let (cr, cq) = claims.unwrap();
            let a = alpha[i];
            sum[1] = cr.sub(&K8::ONE.sub(&a).mul(&sum[0])).mul(&a.inv());
            sum[d + 2] = cq.sub(&sum[d + 1]);
            // Extrapolate: range (cubic) from t = 0..3, query (quadratic) from t = 0..2.
            let s = |x: &K8, c: i64| -> K8 { if c >= 0 { x.scale(c as u64) } else { x.scale((-c) as u64).neg() } };
            sum[4] = s(&sum[0], -1).add(&s(&sum[1], 4)).add(&s(&sum[2], -6)).add(&s(&sum[3], 4));
            let (q0, q1, q2) = (sum[d + 1], sum[d + 2], sum[d + 3]);
            sum[d + 4] = s(&q0, 1).add(&s(&q1, -3)).add(&s(&q2, 3));
            sum[d + 5] = s(&q0, 3).add(&s(&q1, -8)).add(&s(&q2, 6));
        }
        kms += t.elapsed().as_secs_f64() * 1e3;
        if tm { eprintln!("[link2]   round {i}: {:.1} ms (devices {:.1} ms; {} generic blocks)", t.elapsed().as_secs_f64() * 1e3, t_gen, n_gen_base + oh.len() - oh.partition_point(|b| !in_gen(&regions_c[b.region as usize], i))); }
        if std::env::var("MORPHEUS_BIN_CHECK").is_ok() { eprintln!("[bincheck] round {i} sums {:?}", sum.iter().map(|x| x.0[0]).collect::<Vec<_>>()); }
        let g = finish_round(&sum, &pre_alpha, &alpha[i], d, nvars, i, &mo_prefix, &a_coef[i]);
        tr.absorb_k(Item::RoundPoly, &g);
        let r = tr.challenge_k(Stage::LinkRound, 0, i as u64);
        rounds.push(g);
        claims = Some((interpolate(&sum[..d + 1], &r), interpolate(&sum[d + 1..2 * (d + 1)], &r)));
        mo_prefix = mo_prefix.add(&m_term(&a_coef[i], &r));
        pre_alpha = pre_alpha.mul(&eqt_k(&alpha[i], &r));
        xi.push(r);
    }
    lap(&format!("phase 1 ({kms:.0} ms kernels)"));
    // ---- Coarse phase on the devices, each over its own sources. The plane
    // rounds' block constants, block lists and B / dense tables are released
    // first (at full model scale they hold several GB per device).
    for sh in shards.iter_mut() {
        almost_goldilocks_cuda::set_device(sh.dev).expect("set device");
        sh.bdev = None; sh.qdl = None;
        sh.bc = DeviceBuffer::<u64>::new(1).unwrap();
        sh.gen_all = DeviceBuffer::<u8>::new(1).unwrap();
        sh.gen_all_h = Vec::new(); sh.oh = Vec::new();
    }
    almost_goldilocks_cuda::set_device(0).unwrap();
    let coarse = 1u64 << (ell - b0);
    let tfold: Vec<K8> = order.par_iter().map(|&t| {
        let rg = &regs[treg[t]];
        if rg.p > DENSE_P || rg.terms.len() > 32 { return K8::ZERO; }
        let tt = &terms[t];
        let ep = eq_table(&xi[..tt.p]);
        let pwv = tt.pw.iter().zip(&ep).fold(K8::ZERO, |acc, (a, b)| acc.add(&a.mul(b)));
        let idx_bits = b0 - tt.p;
        let er = tt.r[..idx_bits].iter().zip(&xi[tt.p..b0]).fold(K8::ONE, |acc, (a, b)| acc.mul(&eqt_k(a, b)));
        pwv.mul(&er)
    }).collect();
    let efull = eq_table(&xi);
    struct Coarse { w: DeviceBuffer<u64>, q: DeviceBuffer<u64>, ch: DeviceBuffer<u64>, w2: DeviceBuffer<u64>, q2: DeviceBuffer<u64>, c2: DeviceBuffer<u64> }
    let mut cs: Vec<Coarse> = std::thread::scope(|sc| {
        let hs: Vec<_> = shards.iter().map(|sh| {
            let (blocks, rid, kk, terms_c, regions_c, tfold, efull, owner, xi, dense_pw) = (&blocks, &rid, &kk, &terms_c, &regions_c, &tfold, &efull, &owner, &xi, &dense_pw);
            sc.spawn(move || {
                almost_goldilocks_cuda::set_device(sh.dev).expect("set device");
                let ns = sh.srcs.len();
                let tsz = (ns.max(1) as u64 * coarse * 8) as usize;
                let (mut w, mut q, mut ch) = (DeviceBuffer::<u64>::new(tsz).unwrap(), DeviceBuffer::<u64>::new(tsz).unwrap(), DeviceBuffer::<u64>::new(tsz).unwrap());
                let mut loc = vec![u32::MAX; n];
                for (j, &s) in sh.srcs.iter().enumerate() { loc[s] = j as u32; }
                let own = |s: usize| owner[s] as i32 == sh.dev;
                let mine: Vec<Blk> = blocks.iter().filter(|b| own(b.src as usize)).map(|b| Blk { src: loc[b.src as usize], ..*b }).collect();
                let (rid_g, kk_g): (Vec<u32>, Vec<u32>) = rid.iter().zip(kk.iter()).filter(|(r, _)| own(regions_c[**r as usize].src as usize)).map(|(a, b)| (*a, *b)).unzip();
                let d_blocks = upload(&mine);
                let (d_rid, d_kk) = (DeviceBuffer::from_slice(if rid_g.is_empty() { &[0u32][..] } else { &rid_g }).unwrap(), DeviceBuffer::from_slice(if kk_g.is_empty() { &[0u32][..] } else { &kk_g }).unwrap());
                let d_t0 = upload(terms_c);
                let (d_tfold, d_efull) = (up_k(tfold), up_k(efull));
                // Factored dense tables reduce against e_full on the host:
                // ⟨a_t ⊗ pw, e_full⟩ = ⟨a_t, eq(r_[p..b0))⟩·⟨pw, eq(r_[0..p))⟩.
                let (qbits, d_qs) = match (fac_p, sh.fac.as_ref()) {
                    (Some(pp), Some((at, tp))) => {
                        let (el, eh) = (eq_table(&xi[..pp]), eq_table(&xi[pp..b0]));
                        let pwv: Vec<K8> = dense_pw.iter().map(|w| w.iter().zip(&el).fold(K8::ZERO, |a, (x, y)| a.add(&x.mul(y)))).collect();
                        let qs: Vec<K8> = at.chunks(1usize << (b0 - pp)).zip(tp).map(|(a, &r)| a.iter().zip(&eh).fold(K8::ZERO, |acc, (x, y)| acc.add(&x.mul(y))).mul(&pwv[r as usize])).collect();
                        (0, Some(up_k(if qs.is_empty() { &[K8::ZERO][..] } else { &qs })))
                    }
                    _ => (b0 as i32, None),
                };
                let consts = sh.consts(&d_t0, 0, 0, d_qs.as_ref().map(dptr).unwrap_or(dptr(&sh.qd)));
                let rc = unsafe {
                    ffi::agl_mlink2_coarse_init2(sh.raw_ptr(dev_raw, src_len), src_len, b0 as i32, &consts as *const ConstsC as *const c_void,
                        d_blocks.as_ptr() as *const c_void, mine.len() as u32, d_rid.as_ptr(), d_kk.as_ptr(), rid_g.len() as u64,
                        dptr(&d_tfold) as *const u64, dptr(&d_efull) as *const u64, qbits, coarse, 0, ns as u32, w.as_mut_ptr(), q.as_mut_ptr(), ch.as_mut_ptr())
                };
                assert_eq!(rc, 0, "coarse init failed");
                let h = (tsz / 2).max(8);
                Coarse { w, q, ch, w2: DeviceBuffer::<u64>::new(h).unwrap(), q2: DeviceBuffer::<u64>::new(h).unwrap(), c2: DeviceBuffer::<u64>::new(h).unwrap() }
            })
        }).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    lap("coarse init");
    let mut len = coarse;
    for i in b0..ell {
        let d = degs[i];
        let ea = eq_table(&alpha[i + 1..]);
        let sums: Vec<Vec<K8>> = std::thread::scope(|sc| {
            let hs: Vec<_> = shards.iter().zip(cs.iter()).map(|(sh, c)| {
                let ea = &ea;
                sc.spawn(move || {
                    almost_goldilocks_cuda::set_device(sh.dev).expect("set device");
                    let ns = sh.srcs.len();
                    if ns == 0 { return vec![K8::ZERO; 2 * (d + 1)]; }
                    let d_ea = up_k(ea);
                    let mut out = vec![0u64; 8 * 10];
                    let rc = unsafe { ffi::agl_mlink2_coarse_round(c.w.as_ptr(), c.q.as_ptr(), c.ch.as_ptr(), ns as u32, len, dptr(&sh.grange) as *const u64, dptr(&sh.gbit) as *const u64, dptr(&d_ea) as *const u64, d as i32, out.as_mut_ptr()) };
                    assert_eq!(rc, 0, "coarse round {i}");
                    kvec(&out)[..2 * (d + 1)].to_vec()
                })
            }).collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let mut sum = vec![K8::ZERO; 2 * (d + 1)];
        for v in &sums { for (a, b) in sum.iter_mut().zip(v) { *a = a.add(b); } }
        let g = finish_round(&sum, &pre_alpha, &alpha[i], d, nvars, i, &mo_prefix, &a_coef[i]);
        tr.absorb_k(Item::RoundPoly, &g);
        let r = tr.challenge_k(Stage::LinkRound, 0, i as u64);
        rounds.push(g);
        mo_prefix = mo_prefix.add(&m_term(&a_coef[i], &r));
        pre_alpha = pre_alpha.mul(&eqt_k(&alpha[i], &r));
        let rw = r.words();
        std::thread::scope(|sc| {
            for (sh, c) in shards.iter().zip(cs.iter_mut()) {
                let rw = &rw;
                sc.spawn(move || {
                    almost_goldilocks_cuda::set_device(sh.dev).expect("set device");
                    let ns = sh.srcs.len() as u32;
                    if ns == 0 { return; }
                    let Coarse { w, q, ch, w2, q2, c2 } = c;
                    for (src, dst) in [(&*w, &mut *w2), (&*q, &mut *q2), (&*ch, &mut *c2)] {
                        let rc = unsafe { ffi::agl_mlink2_coarse_fold(src.as_ptr(), dst.as_mut_ptr(), ns, len, rw.as_ptr()) };
                        assert_eq!(rc, 0, "coarse fold");
                    }
                    std::mem::swap(w, w2); std::mem::swap(q, q2); std::mem::swap(ch, c2);
                });
            }
        });
        len /= 2;
        xi.push(r);
    }
    lap("coarse rounds");
    let fin = |f: &dyn Fn(&Coarse) -> &DeviceBuffer<u64>| -> Vec<K8> {
        let mut v = vec![K8::ZERO; n];
        for (sh, c) in shards.iter().zip(&cs) {
            almost_goldilocks_cuda::set_device(sh.dev).unwrap();
            let ns = sh.srcs.len();
            if ns == 0 { continue; }
            let x = kvec(&f(c).read_slice(0, ns * 8).unwrap());
            for (j, &s) in sh.srcs.iter().enumerate() { if owner[s] as i32 == sh.dev { v[s] = x[j]; } }
        }
        v
    };
    let (fw, fq, fc) = (fin(&|c| &c.w), fin(&|c| &c.q), fin(&|c| &c.ch));
    drop(cs);
    if let Some(k) = keep {
        let mut k = k.lock().unwrap();
        for sh in shards.into_iter() {
            if let Some(pk) = sh.raw { if !sh.owned.is_empty() {
                // The authentication reads int8 digits in physical column order.
                almost_goldilocks_cuda::set_device(sh.dev).expect("set device");
                let len = src_len as usize;
                // int8 when it fits (4x the packed bytes); else the packed
                // shard itself, permuted in place to physical columns.
                let _ = almost_goldilocks_cuda::memory::pool_trim(0);
                let need = sh.owned.len() * len;
                let fits = std::env::var("MORPHEUS_KEEP_PACKED").is_err()
                    && almost_goldilocks_cuda::mem_get_info().map(|(free, _)| free > need + (4usize << 30)).unwrap_or(false);
                if fits {
                    let mut buf = DeviceBuffer::<i8>::new(need).expect("kept sources");
                    for j in 0..sh.owned.len() {
                        let rc = unsafe { ffi::agl_morpheus_unpack2(pk.as_ptr().add(j * len / 4), buf.as_mut_ptr().add(j * len), len as u64, 1) };
                        assert_eq!(rc, 0, "unpack kept sources");
                    }
                    drop(pk);
                    k.push(Kept { dev: sh.dev, srcs: sh.owned, buf, pk: None, phys: true });
                } else {
                    let mut pk = pk;
                    let mut tmp = DeviceBuffer::<u8>::new(len / 4).expect("permute scratch");
                    for j in 0..sh.owned.len() {
                        let rc = unsafe { ffi::agl_morpheus_permute2(pk.as_ptr().add(j * len / 4), tmp.as_mut_ptr(), (len / 4096) as u64) };
                        assert_eq!(rc, 0, "permute kept sources");
                        pk.copy_range_from_device(j * len / 4, &tmp, 0, len / 4).unwrap();
                    }
                    let buf = DeviceBuffer::<i8>::new(1).unwrap();
                    k.push(Kept { dev: sh.dev, srcs: sh.owned, buf, pk: Some(pk), phys: true });
                }
            } }
        }
    }
    almost_goldilocks_cuda::set_device(0).unwrap();
    (rounds, xi, fw, fq, fc, pre_alpha, mo_prefix)
}

/// G (per plane count p and bin t: [G3, G2]) and W (per term and bin) of the
/// one-hot regions; see `onehot_pre_kernel`.
#[allow(clippy::too_many_arguments)]
fn onehot_sums(
    raw_ptr: *const u8, src_len: u64, b0: usize, consts: &ConstsC, oh: &[Blk], d_bc: &DeviceBuffer<u64>,
    n_gen_base: usize, regions_c: &[RegionC], alpha: &[K8], nterms: usize, b5: &[K8],
) -> OhSums {
    let t0 = std::time::Instant::now();
    let sync = || { almost_goldilocks_cuda::synchronize().ok(); t0.elapsed().as_secs_f64() * 1e3 };
    let maxp = regions_c.iter().map(|r| r.p as usize).max().unwrap_or(0);
    let (mut eai, mut eai_off) = (Vec::new(), Vec::new());
    for pp in 0..=maxp {
        eai_off.push(eai.len() as u32);
        if pp <= b0 { eai.extend(eq_table(&alpha[pp..b0])); }
    }
    let d_eai = up_k(&eai);
    let d_eai_off = DeviceBuffer::from_slice(&eai_off).unwrap();
    let d_oh = upload(oh);
    let ng = 6 * 32 * 2;
    let d_g = { let mut x = DeviceBuffer::<u64>::new(ng * 16).unwrap(); x.zero().unwrap(); x };
    let d_w = { let mut x = DeviceBuffer::<u64>::new(nterms.max(1) * 32 * 16).unwrap(); x.zero().unwrap(); x };
    let nu = nterms.max(1) * 4 * 32;
    let nj = 2 * 256;
    let d_u = { let mut x = DeviceBuffer::<u64>::new(nu * 16).unwrap(); x.zero().unwrap(); x };
    let d_j = { let mut x = DeviceBuffer::<u64>::new(nj * 16).unwrap(); x.zero().unwrap(); x };
    let ngv = 2 * 2 * 32;
    let d_gv = { let mut x = DeviceBuffer::<u64>::new(ngv * 16).unwrap(); x.zero().unwrap(); x };
    let any_bin = oh.iter().any(|b| regions_c[b.region as usize].flags & 8 != 0);
    let (nj5, j5_rep) = (2 * 65536, 8);
    let mut d_j5 = DeviceBuffer::<u64>::new(if any_bin { j5_rep * nj5 * 16 } else { 16 }).unwrap();
    d_j5.zero().unwrap();
    let bc_oh = d_bc.as_ptr();
    let mut c2 = *consts;
    c2.oh_base = n_gen_base as u64;
    let consts = &c2;
    // Binned blocks are a prefix of `oh` (one_hot_bin_kernel), the rest a
    // suffix (onehot_pre_kernel); each kernel gets only its range. Block k of
    // a range starting at s0 (length n) has constants at
    // bco[oh_base + N − 1 − (s0 + k)] = bco[oh_base' + n − 1 − k], oh_base' = oh_base + N − s0 − n.
    let t_set = sync();
    let nbin = oh.partition_point(|b| regions_c[b.region as usize].flags & 8 != 0);
    let n_all = oh.len();
    let blk_sz = std::mem::size_of::<Blk>();
    if nbin < n_all {
        let mut c3 = *consts;
        c3.oh_base = n_gen_base as u64;   // suffix: s0 + n = N
        let rc = unsafe {
            ffi::agl_mlink2_onehot_pre(raw_ptr, src_len, b0 as i32, &c3 as *const ConstsC as *const c_void,
                (d_oh.as_ptr() as *const u8).add(nbin * blk_sz) as *const c_void, (n_all - nbin) as u32, bc_oh, dptr(&d_eai) as *const u64, d_eai_off.as_ptr(),
                d_g.as_ptr() as *mut u64, d_w.as_ptr() as *mut u64)
        };
        assert_eq!(rc, 0, "one-hot sums");
    }
    let t_pre = sync();
    if any_bin {
        let d_ea1 = up_k(&eq_table(&alpha[5..b0]));
        let d_ea5 = up_k(&eq_table(&alpha[6..b0]));
        let d_b5 = up_k(b5);
        let mut c4 = *consts;
        c4.oh_base = (n_gen_base + n_all - nbin) as u64;   // prefix: s0 = 0, n = nbin
        let rc = unsafe {
            ffi::agl_mlink2_onehot_bin(raw_ptr, src_len, b0 as i32, &c4 as *const ConstsC as *const c_void,
                d_oh.as_ptr() as *const c_void, nbin as u32, bc_oh, dptr(&d_ea1) as *const u64, dptr(&d_ea5) as *const u64, dptr(&d_b5) as *const u64,
                d_gv.as_ptr() as *mut u64, d_u.as_ptr() as *mut u64, d_j.as_ptr() as *mut u64, d_j5.as_ptr() as *mut u64)
        };
        assert_eq!(rc, 0, "one-hot bins");
    }
    let t_bin = sync();
    let halves = |v: Vec<u64>| -> Vec<K8> {
        const Q: u128 = 18446744069414584289;
        v.par_chunks(16).map(|c| K8(std::array::from_fn(|w| ((c[2 * w] as u128 + ((c[2 * w + 1] as u128 % Q) << 32)) % Q) as u64))).collect()
    };
    // Large accumulators are reduced to field elements on the device: half the readback.
    let reduced = |d: &DeviceBuffer<u64>, n: usize| -> Vec<K8> {
        let d_s = DeviceBuffer::<u64>::new(n * 8).unwrap();
        assert_eq!(unsafe { ffi::agl_mlink2_rep_reduce(d.as_ptr(), 1, n as u64, d_s.as_ptr() as *mut u64) }, 0);
        // Pages faulted in parallel, then one copy straight into the K8s.
        let mut v = vec![K8::ZERO; 0];
        v.reserve_exact(n);
        unsafe { v.set_len(n); }
        v.par_chunks_mut(1 << 14).for_each(|c| c.fill(K8::ZERO));
        let w = unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u64, n * 8) };
        d_s.copy_to_slice(w).unwrap();
        v
    };
    let out = OhSums {
        g: halves(d_g.read_slice(0, ng * 16).unwrap()), w: reduced(&d_w, nterms.max(1) * 32),
        u: if any_bin { reduced(&d_u, nu) } else { vec![K8::ZERO; nu] }, j: halves(d_j.read_slice(0, nj * 16).unwrap()),
        gv: halves(d_gv.read_slice(0, ngv * 16).unwrap()),
        j5: if any_bin {
            // Replicas summed on the device: 1/8 of the readback.
            let d_s = DeviceBuffer::<u64>::new(nj5 * 8).unwrap();
            assert_eq!(unsafe { ffi::agl_mlink2_rep_reduce(d_j5.as_ptr(), j5_rep as i32, nj5 as u64, d_s.as_ptr() as *mut u64) }, 0);
            kvec(&d_s.to_vec().unwrap())
        } else { vec![K8::ZERO; nj5] },
        u4: Vec::new(),
    };
    if std::env::var("MORPHEUS_TIMING").is_ok() { eprintln!("[link2]   one-hot sums: {nterms} terms, {:.0} MB readback; setup {t_set:.0} pre {:.0} bin {:.0} read {:.0} ms", (ng + nterms.max(1) * 32 + nu + nj + ngv) as f64 * 128.0 / 1e6, t_pre - t_set, t_bin - t_pre, sync() - t_bin); }
    out
}

/// Round-0 B table segments: per term its index coordinates (flattened K8
/// words from `coords[co..]`, `nv` of them) and device offset.
#[derive(Default)]
struct BSeg { coords: Vec<u64>, co: Vec<u32>, nv: Vec<u8>, off: Vec<u32>, pre: Vec<u64>, total: u64 }

/// The A table and pa of `term_tables` (its B table is kept on the device).
fn term_a(t: &TermInfo, r_pre: &[K8], i: usize, d: usize, b0: usize) -> (Vec<K8>, usize) {
    let p = t.p;
    if i < p {
        let pa = p - i - 1;
        let e = eq_table(r_pre);
        let mut a = vec![K8::ZERO; (d + 1) << pa];
        for pr in 0..1usize << pa {
            let f = |bit: usize| -> K8 { (0..1usize << i).fold(K8::ZERO, |acc, b| acc.add(&e[b].mul(&t.pw[b + (bit << i) + (pr << (i + 1))]))) };
            let (f0, f1) = (f(0), f(1));
            for tt in 0..=d { a[(tt << pa) + pr] = f0.add(&K8::from_base(tt as u64).mul(&f1.sub(&f0))); }
        }
        (a, pa)
    } else {
        let _ = b0;
        let ep = eq_table(&r_pre[..p]);
        let pwv = t.pw.iter().zip(&ep).fold(K8::ZERO, |acc, (w, e)| acc.add(&w.mul(e)));
        let j = i - p;
        let pre = t.r[..j].iter().zip(&r_pre[p..i]).fold(K8::ONE, |acc, (a, b)| acc.mul(&a.mul(b).add(&K8::ONE.sub(a).mul(&K8::ONE.sub(b)))));
        let s = pwv.mul(&pre);
        ((0..=d).map(|tt| s.mul(&eqt_k(&t.r[j], &K8::from_base(tt as u64)))).collect(), 0)
    }
}

/// Halve `v` once per challenge (low variable first).
fn fold_k(mut v: Vec<K8>, rs: &[K8]) -> Vec<K8> {
    for r in rs { v = v.chunks(2).map(|c| c[0].add(&r.mul(&c[1].sub(&c[0])))).collect(); }
    v
}

/// Round-i dense query tables of factored dense regions: table t at level i
/// is fold_{i}(a_t ⊗ pw) = fold_{i−p}(a_t) ⊗ fold_{min(i,p)}(pw) (plane bits
/// low); expanded on the device to 2^{b0−i} entries per table.
#[allow(clippy::too_many_arguments)]
fn dense_round_tables(at: &[K8], tp_h: &[u32], pws: &[std::sync::Arc<Vec<K8>>], p: usize, i: usize, b0: usize, xi: &[K8]) -> Option<DeviceBuffer<u64>> {
    let an = 1usize << (b0 - p);
    let ntab = at.len() / an;
    if ntab == 0 { return None; }
    let (ip, ia) = (i.min(p), i.saturating_sub(p));
    let pf_h: Vec<Vec<K8>> = pws.iter().map(|w| fold_k(w.to_vec(), &xi[..ip])).collect();
    let pn = 1usize << (p - ip);
    let pf: Vec<K8> = pf_h.iter().flatten().copied().collect();
    let pf_off: Vec<u32> = tp_h.iter().map(|&r| (r as usize * pn) as u32).collect();
    let af: Vec<K8> = if ia == 0 { at.to_vec() } else { at.par_chunks(an).flat_map_iter(|c| fold_k(c.to_vec(), &xi[p..i])).collect() };
    let afn = an >> ia;
    let ybits = b0 - i;
    let out = DeviceBuffer::<u64>::new((ntab << ybits) * 8).unwrap();
    let (d_af, d_pf) = (up_k(&af), up_k(&pf));
    let d_off = DeviceBuffer::from_slice(&pf_off).unwrap();
    let rc = unsafe { ffi::agl_mlink2_dense_expand(dptr(&d_af) as *const u64, afn as i32, dptr(&d_pf) as *const u64, d_off.as_ptr(), (p - ip) as i32, ybits as i32, ntab as u64, out.as_ptr() as *mut u64) };
    assert_eq!(rc, 0, "dense expand");
    Some(out)
}

/// Whether one-hot block region `r` is in round i's generic list: unbinned
/// from its first index round (i ≥ p); binned (p = 4) from i = 6.
fn in_gen(r: &RegionC, i: usize) -> bool { if r.flags & 8 != 0 { i >= 6 } else { r.p as usize <= i } }

/// Round 5 of the binned blocks ([range c0..c3, query c0..c2]). Pair r covers
/// indices 4r..4r+3 with classes (s0, s1, s2, s3); with e = eq(r_[0..5)),
/// w_0 = e[s0] + e[16 + s1], w_1 = e[s2] + e[16 + s3]; the query uses
/// u_b = Σ_{k,s} e[16k + s]·U[T][b][k][s].
fn binned_round5(os: &OhSums, bin_terms: &[(usize, usize)], e: &[K8], a_all: &[K8], tc: &[TermC]) -> [K8; 7] {
    let (two, three) = (K8::from_base(2), K8::from_base(3));
    let mut c = (0..65536usize).into_par_iter().fold(|| [K8::ZERO; 7], |mut c, cl| {
        let (j3, j2) = (os.j5[cl], os.j5[65536 + cl]);
        if j3.is_zero() && j2.is_zero() { return c; }
        let (s0, s1, s2, s3) = (cl >> 12, (cl >> 8) & 15, (cl >> 4) & 15, cl & 15);
        let a = e[s0].add(&e[16 + s1]);
        let dl = e[s2].add(&e[16 + s3]).sub(&a);
        let (a2, dl2) = (a.mul(&a), dl.mul(&dl));
        c[0] = c[0].add(&j3.mul(&a2.mul(&a).sub(&a))).add(&j2.mul(&a2.sub(&a)));
        c[1] = c[1].add(&j3.mul(&three.mul(&a2).mul(&dl).sub(&dl))).add(&j2.mul(&two.mul(&a).mul(&dl).sub(&dl)));
        c[2] = c[2].add(&j3.mul(&three.mul(&a).mul(&dl2))).add(&j2.mul(&dl2));
        c[3] = c[3].add(&j3.mul(&dl2.mul(&dl)));
        c
    }).reduce(|| [K8::ZERO; 7], |mut x, y| { for (a, b) in x.iter_mut().zip(&y) { *a = a.add(b); } x });
    let q = bin_terms.par_iter().map(|&(k, kw)| {
        let u = |b: usize| (0..2).fold(K8::ZERO, |acc, kk| (0..16).fold(acc, |acc, s| acc.add(&e[16 * kk + s].mul(&os.u[(kw * 4 + b * 2 + kk) * 32 + s]))));
        let (u0, u1) = (u(0), u(1));
        let aoff = tc[k].aoff as usize;
        debug_assert_eq!(tc[k].pa, 0);
        let (a0, a1) = (a_all[aoff], a_all[aoff + 1]);
        let (p0, x, p1) = (a0.mul(&u0), a0.mul(&u1).add(&a1.mul(&u0)), a1.mul(&u1));
        [p0, x.sub(&two.mul(&p0)), p0.sub(&x).add(&p1)]
    }).reduce(|| [K8::ZERO; 3], |a, b| [a[0].add(&b[0]), a[1].add(&b[1]), a[2].add(&b[2])]);
    for j in 0..3 { c[4 + j] = c[4 + j].add(&q[j]); }
    c
}

/// One-hot sums of the round-0 pass: plane bins G, W (`onehot_pre_kernel`)
/// and the binned blocks' round-p bins U, J (`onehot_bin_kernel`).
struct OhSums { g: Vec<K8>, w: Vec<K8>, u: Vec<K8>, j: Vec<K8>, gv: Vec<K8>, j5: Vec<K8>, u4: Vec<K8> }
impl OhSums {
    fn add(&self, o: &OhSums) -> OhSums {
        let f = |a: &[K8], b: &[K8]| a.par_iter().zip(b).map(|(x, y)| x.add(y)).collect();
        OhSums { g: f(&self.g, &o.g), w: f(&self.w, &o.w), u: f(&self.u, &o.u), j: f(&self.j, &o.j), gv: f(&self.gv, &o.gv), j5: f(&self.j5, &o.j5), u4: Vec::new() }
    }
}

/// Round i = p = 4 of the binned one-hot blocks as polynomials in t
/// ([range c0..c3, query c0..c2]). Pair r has w_t = (1 − t)E[s0] + tE[s1]:
///   range: Σ_{s0,s1} J3·(w³ − w) + J2·(w² − w);
///   query: Σ_T (A0(1 − t) + A1 t)((1 − t)u0 + t u1), u_par = Σ_s E[s]·U_T[par][s].
fn binned_round(os: &OhSums, bin_terms: &[(usize, usize)], e: &[K8], a_all: &[K8], tc: &[TermC]) -> [K8; 7] {
    let mut c = [K8::ZERO; 7];
    let (two, three) = (K8::from_base(2), K8::from_base(3));
    for s0 in 0..16 {
        for s1 in 0..16 {
            let (j3, j2) = (os.j[s0 * 16 + s1], os.j[256 + s0 * 16 + s1]);
            if j3.is_zero() && j2.is_zero() { continue; }
            let a = e[s0];
            let dl = e[s1].sub(&a);
            let (a2, dl2) = (a.mul(&a), dl.mul(&dl));
            // w = a + tδ: w² = a² + 2aδt + δ²t², w³ = a³ + 3a²δt + 3aδ²t² + δ³t³.
            c[0] = c[0].add(&j3.mul(&a2.mul(&a).sub(&a))).add(&j2.mul(&a2.sub(&a)));
            c[1] = c[1].add(&j3.mul(&three.mul(&a2).mul(&dl).sub(&dl))).add(&j2.mul(&two.mul(&a).mul(&dl).sub(&dl)));
            c[2] = c[2].add(&j3.mul(&three.mul(&a).mul(&dl2))).add(&j2.mul(&dl2));
            c[3] = c[3].add(&j3.mul(&dl2.mul(&dl)));
        }
    }
    let q = bin_terms.par_iter().map(|&(k, kw)| {
        let u = |par: usize| (0..16).fold(K8::ZERO, |acc, s| acc.add(&e[s].mul(&os.u4[(kw * 2 + par) * 32 + s])));
        let (u0, u1) = (u(0), u(1));
        let aoff = tc[k].aoff as usize;
        debug_assert_eq!(tc[k].pa, 0);
        let (a0, a1) = (a_all[aoff], a_all[aoff + 1]);
        // A0u0(1 − t)² + (A0u1 + A1u0)t(1 − t) + A1u1 t².
        let (p0, x, p1) = (a0.mul(&u0), a0.mul(&u1).add(&a1.mul(&u0)), a1.mul(&u1));
        [p0, x.sub(&two.mul(&p0)), p0.sub(&x).add(&p1)]
    }).reduce(|| [K8::ZERO; 3], |a, b| [a[0].add(&b[0]), a[1].add(&b[1]), a[2].add(&b[2])]);
    for j in 0..3 { c[4 + j] = c[4 + j].add(&q[j]); }
    c
}

/// One-hot contribution of plane round i as polynomials in t:
/// [range c0..c3, query c0..c2] (same algebra as `onehot3_kernel`).
#[allow(clippy::too_many_arguments)]
fn onehot_round(
    g: &[K8], w: &[K8], oh_terms: &[Vec<(usize, usize)>], i: usize, b0: usize, alpha: &[K8],
    e1: &[K8], e2: &[K8], e3: &[K8], a_all: &[K8], tc: &[TermC],
) -> [K8; 7] {
    let mut c = [K8::ZERO; 7];
    let three = K8::from_base(3);
    let two = K8::from_base(2);
    for p in (i + 1)..oh_terms.len() {
        if p > b0 { break; }
        let np = 1usize << p;
        let eapr = eq_table(&alpha[i + 1..p]);
        // Query: Σ_T W_T[t]·A_T[0][pr], Σ_T W_T[t]·dA_T[pr] per bin.
        let zero = || vec![(K8::ZERO, K8::ZERO); np];
        let ys = oh_terms[p].par_iter().map(|&(k, kw)| {
            let (aoff, pa) = (tc[k].aoff as usize, tc[k].pa as usize);
            let mut v = zero();
            for t in 0..np {
                let wt = &w[kw * 32 + t];
                if wt.is_zero() { continue; }
                let pr = t >> (i + 1);
                let a0 = a_all[aoff + pr];
                let a1 = a_all[aoff + (1 << pa) + pr];
                v[t] = (wt.mul(&a0), wt.mul(&a1.sub(&a0)));
            }
            v
        }).reduce(zero, |mut x, y| { for (a, b) in x.iter_mut().zip(y) { a.0 = a.0.add(&b.0); a.1 = a.1.add(&b.1); } x });
        for t in 0..np {
            let (g3, g2) = (g[(p * 32 + t) * 2], g[(p * 32 + t) * 2 + 1]);
            let bit = (t >> i) & 1;
            let low = t & ((1 << i) - 1);
            let pr = t >> (i + 1);
            let (e, ep) = (e1[low], eapr[pr]);
            let x3 = g3.mul(&ep).mul(&e3[low]);
            let x2 = g2.mul(&ep).mul(&e2[low]);
            let x1 = g3.add(&g2).mul(&ep).mul(&e);
            let y0 = ys[t].0.mul(&e);
            let y1 = ys[t].1.mul(&e);
            if bit == 1 {
                c[3] = c[3].add(&x3); c[2] = c[2].add(&x2); c[1] = c[1].sub(&x1);
                c[5] = c[5].add(&y0); c[6] = c[6].add(&y1);
            } else {
                c[0] = c[0].add(&x3.add(&x2).sub(&x1));
                c[1] = c[1].add(&x1.sub(&three.mul(&x3)).sub(&two.mul(&x2)));
                c[2] = c[2].add(&three.mul(&x3).add(&x2));
                c[3] = c[3].sub(&x3);
                c[4] = c[4].add(&y0); c[5] = c[5].add(&y1.sub(&y0)); c[6] = c[6].sub(&y1);
            }
        }
    }
    c
}

#[derive(Clone, Copy)]
struct SyncPtr(*mut K8);
unsafe impl Send for SyncPtr {}
unsafe impl Sync for SyncPtr {}
impl SyncPtr { fn get(&self) -> *mut K8 { self.0 } }

/// A device's range of sources [lo, hi), kept after the link (streamed
/// mode) for the endpoints and the authentication; logical raw order until
/// `phys` (then the commitment's physical column order).
/// A device's kept sources: int8 digits in `buf`, or (when that does not
/// fit) packed physical columns in `pk`, 1024 bytes per column.
pub struct Kept { pub dev: i32, pub srcs: Vec<usize>, pub buf: DeviceBuffer<i8>, pub pk: Option<DeviceBuffer<u8>>, pub phys: bool }

/// One device's share of the plane rounds.
struct Shard {
    dev: i32,
    raw: Option<DeviceBuffer<u8>>,
    /// Local numbering: local index j is global source srcs[j] (`raw`,
    /// γ powers, blocks and regions use it); `owned` are this device's sources.
    srcs: Vec<usize>, owned: Vec<usize>,
    calo: DeviceBuffer<u64>, cahi: DeviceBuffer<u64>, grange: DeviceBuffer<u64>, gbit: DeviceBuffer<u64>,
    regions: DeviceBuffer<u8>, hl: DeviceBuffer<u64>, hh: DeviceBuffer<u64>,
    qd: DeviceBuffer<u64>, qdl: Option<DeviceBuffer<u64>>, qdl_len: usize, qidx: DeviceBuffer<u32>,
    /// Factored dense tables: host a_k tables and, per table, its region's pw index.
    fac: Option<(Vec<K8>, Vec<u32>)>,
    /// Device B tables (current, scratch) and the round they hold.
    bdev: Option<(DeviceBuffer<u64>, DeviceBuffer<u64>)>, b_round: Option<usize>,
    gen_base: usize, oh: Vec<Blk>, gen_all: DeviceBuffer<u8>, gen_all_h: Vec<Blk>, bc: DeviceBuffer<u64>, bco: DeviceBuffer<u64>, ca_lo_bits: usize,
}

impl Shard {
    fn raw_ptr(&self, _dev_raw: Option<&DeviceBuffer<i8>>, _src_len: u64) -> *const u8 {
        self.raw.as_ref().expect("packed shard sources").as_ptr()
    }
    fn consts(&self, d_terms: &DeviceBuffer<u8>, a: u64, b: u64, qd: u64) -> ConstsC {
        ConstsC {
            ca_lo: dptr(&self.calo), ca_hi: dptr(&self.cahi), ca_lo_bits: self.ca_lo_bits as i32, pad: 0,
            g_range: dptr(&self.grange), g_bit: dptr(&self.gbit), regions: dptr(&self.regions), terms: dptr(d_terms),
            hl: dptr(&self.hl), hh: dptr(&self.hh), a, b, qd, qidx: dptr(&self.qidx),
            bco: dptr(&self.bco), oh_base: 0,
        }
    }
}

/// Host tables of one plane round, shared by the shards.
struct RoundIn<'a> {
    i: usize, d: usize, b0: usize, src_len: u64, reduced: bool, use_index: bool, r_prev: Option<K8>, full: bool, b5: &'a [K8],
    xi: &'a [K8], fac_p: Option<usize>, dense_pw: &'a [std::sync::Arc<Vec<K8>>], skip_range: bool, rb1: bool,
    halve: &'a [(u32, u64)], b_total: usize, bseg: &'a BSeg,
    a_all: &'a [K8], b_all: &'a [K8], tc: &'a [TermC], e1: &'a [K8], e2: &'a [K8], e3: &'a [K8], ea: &'a [K8],
}

/// One shard's plane round: fold its dense query tables with the previous
/// challenge, run its blocks; in round 0 also its one-hot bin sums.
#[allow(clippy::type_complexity)]
fn shard_round(sh: &mut Shard, dev_raw: Option<&DeviceBuffer<i8>>, rin: &RoundIn, regions_c: &[RegionC], alpha: &[K8], nterms: usize)
    -> (Vec<K8>, Option<OhSums>, Option<Vec<K8>>) {
    almost_goldilocks_cuda::set_device(sh.dev).expect("set device");
    let (i, d, b0) = (rin.i, rin.d, rin.b0);
    if let (Some(pp), Some((at, tp))) = (rin.fac_p, sh.fac.as_ref()) {
        if !rin.full { sh.qdl = dense_round_tables(at, tp, rin.dense_pw, pp, i, b0, rin.xi); }
    } else if let Some(r) = rin.r_prev {
        if sh.qdl_len > 1 {
            let dst = DeviceBuffer::<u64>::new(8 * (sh.qdl_len / 2)).unwrap();
            let src = sh.qdl.as_ref().unwrap_or(&sh.qd);
            let rc = unsafe { ffi::agl_mpiop_sc_fold(src.as_ptr(), dst.as_ptr() as *mut u64, sh.qdl_len as u64, r.words().as_ptr()) };
            assert_eq!(rc, 0, "dense query fold");
            sh.qdl = Some(dst);
            sh.qdl_len /= 2;
        }
    }
    let t0 = std::time::Instant::now();
    let tm0 = i == 0 && std::env::var("MORPHEUS_TIMING").is_ok();
    let raw_ptr = sh.raw_ptr(dev_raw, rin.src_len);
    let (d_a, d_t) = (up_k(rin.a_all), upload(rin.tc));
    if i == 0 && sh.b_round.is_none() {
        let mut b = DeviceBuffer::<u64>::new(rin.b_total.max(1) * 8).unwrap();
        let bs = rin.bseg;
        let rc = unsafe { ffi::agl_mlink2_b_build(bs.coords.as_ptr(), (bs.coords.len() / 8) as u64, bs.co.as_ptr(), bs.nv.as_ptr(), bs.off.as_ptr(), bs.pre.as_ptr(), bs.co.len() as u32, bs.total, b.as_mut_ptr(), rin.b_total.max(1) as u64) };
        assert_eq!(rc, 0, "B tables");
        sh.bdev = Some((b, DeviceBuffer::<u64>::new(rin.b_total.max(1) * 8).unwrap()));
        sh.b_round = Some(0);
    } else if sh.b_round.is_some_and(|r| r < i) {
        let (src, dst) = sh.bdev.as_mut().unwrap();
        let mut pre = Vec::with_capacity(rin.halve.len());
        let mut tot = 0u64;
        for &(_, n) in rin.halve { pre.push(tot); tot += n; }
        let off: Vec<u32> = rin.halve.iter().map(|x| x.0).collect();
        let (d_off, d_pre) = (DeviceBuffer::from_slice(if off.is_empty() { &[0u32][..] } else { &off }).unwrap(), DeviceBuffer::from_slice(if pre.is_empty() { &[0u64][..] } else { &pre }).unwrap());
        let rc = unsafe { ffi::agl_mlink2_b_halve(src.as_ptr(), dst.as_ptr() as *mut u64, rin.b_total.max(1) as u64, d_off.as_ptr(), d_pre.as_ptr(), off.len() as u32, tot) };
        assert_eq!(rc, 0, "B halving");
        std::mem::swap(src, dst);
        sh.b_round = Some(i);
    }
    let t_b = t0.elapsed().as_secs_f64() * 1e3;
    let d_b = &sh.bdev.as_ref().unwrap().0;
    let (d_e1, d_e2, d_e3, d_ea) = (up_k(rin.e1), up_k(rin.e2), up_k(rin.e3), up_k(rin.ea));
    // Generic tail: one-hot blocks with p < i, and p = i unless binned.
    let n_ohb = if rin.full { sh.oh.len() - sh.oh.iter().filter(|b| regions_c[b.region as usize].p as usize <= i).count() } else { sh.oh.partition_point(|b| !in_gen(&regions_c[b.region as usize], i)) };
    let n_gen = sh.gen_base + (sh.oh.len() - n_ohb);
    let consts = sh.consts(&d_t, dptr(&d_a), dptr(&d_b), sh.qdl.as_ref().map(dptr).unwrap_or(dptr(&sh.qd)));
    let sum = if n_gen == 0 { vec![K8::ZERO; 2 * (d + 1)] } else if rin.use_index {
        let mut o7 = vec![0u64; 7 * 8];
        let rc = unsafe {
            ffi::agl_mlink2_index(raw_ptr, rin.src_len, b0 as i32, i as i32, &consts as *const ConstsC as *const c_void,
                sh.gen_all.as_ptr() as *const c_void, n_gen as u32, sh.bc.as_ptr(), dptr(&d_e1) as *const u64, dptr(&d_ea) as *const u64, o7.as_mut_ptr())
        };
        assert_eq!(rc, 0, "link v2 index round {i} failed");
        let v = kvec(&o7);
        vec![v[0], K8::ZERO, v[2], v[3], K8::ZERO, v[4], K8::ZERO, v[6], K8::ZERO, K8::ZERO]
    } else {
        // Generic blocks by kind (specialised kernels): 0 no query, 1 fast
        // query (≤ 2 terms, 2^pa ≤ 8, no dense table), 2 general, 3 many
        // terms (2^{i+1+pa} ≤ 32: sums by additions). Blocks without a
        // query have nothing to do when the range is binned.
        let dense = dense_link();
        let many = !dense && std::env::var_os("MORPHEUS_NO_MANYQ").is_none();
        let mut lists: [Vec<u32>; 4] = Default::default();
        for (bi, b) in sh.gen_all_h[..n_gen].iter().enumerate() {
            let mode = if b.region == u32::MAX { 0 } else {
                let r = &regions_c[b.region as usize];
                let pa = if i < r.p as usize { r.p as usize - i - 1 } else { 0 };
                if r.flags & 4 != 0 { 2 } else if r.tcnt == 0 { 0 } else if !dense && r.tcnt <= 2 && pa <= 3 { 1 } else if many && i + 1 + pa <= 5 && (r.tcnt as usize) << pa <= 128 { 3 } else { 2 }
            };
            if mode == 0 && rin.skip_range { continue; }
            lists[mode].push(bi as u32);
        }
        let d_lists: Vec<DeviceBuffer<u32>> = lists.iter().map(|l| DeviceBuffer::from_slice(if l.is_empty() { &[0u32][..] } else { l }).unwrap()).collect();
        let lptr: Vec<*const u32> = d_lists.iter().map(|b| b.as_ptr() as *const u32).collect();
        let lcnt: Vec<u32> = lists.iter().map(|l| l.len() as u32).collect();
        let mut out = vec![0u64; 8 * 10];
        let rc = unsafe {
            ffi::agl_mlink2_round(raw_ptr, rin.src_len, b0 as i32, i as i32, d as i32, rin.reduced as i32, &consts as *const ConstsC as *const c_void,
                std::ptr::null(), 0, sh.gen_all.as_ptr() as *const c_void, lptr.as_ptr(), lcnt.as_ptr(), sh.bc.as_ptr(),
                dptr(&d_e1) as *const u64, dptr(&d_e2) as *const u64, dptr(&d_e3) as *const u64, dptr(&d_ea) as *const u64, rin.skip_range as i32, out.as_mut_ptr())
        };
        assert_eq!(rc, 0, "link v2 round {i} failed");
        kvec(&out)[..2 * (d + 1)].to_vec()
    };
    let t_s = t0.elapsed().as_secs_f64() * 1e3;
    let ohs = if i == 0 && !sh.oh.is_empty() {
        Some(onehot_sums(raw_ptr, rin.src_len, b0, &consts, &sh.oh, &sh.bc, sh.gen_base, regions_c, alpha, nterms, rin.b5))
    } else { None };
    let t_o = t0.elapsed().as_secs_f64() * 1e3;
    // Round-1 range bins over round 1's generic list.
    let rb = if i == 0 && rin.rb1 {
        let n1 = sh.gen_base + (sh.oh.len() - sh.oh.partition_point(|b| !in_gen(&regions_c[b.region as usize], 1)));
        let nr = 2 * 81 * 16 + 1;
        let d_r = DeviceBuffer::from_slice(&vec![0u64; nr]).unwrap();
        let d_ea1 = up_k(&eq_table(&alpha[2..b0]));
        let rc = unsafe { ffi::agl_mlink2_range_bin1(raw_ptr, rin.src_len, b0 as i32, &consts as *const ConstsC as *const c_void, sh.gen_all.as_ptr() as *const c_void, n1 as u32, sh.bc.as_ptr(), dptr(&d_ea1) as *const u64, d_r.as_ptr() as *mut u64) };
        assert_eq!(rc, 0, "range bins");
        let h = d_r.read_slice(0, nr).unwrap();
        assert_eq!(h[nr - 1], 0, "non-ternary digits in generic blocks");
        const Q: u128 = 18446744069414584289;
        Some(h[..nr - 1].chunks(16).map(|c| K8(std::array::from_fn(|w| ((c[2 * w] as u128 + ((c[2 * w + 1] as u128 % Q) << 32)) % Q) as u64))).collect())
    } else { None };
    if tm0 { eprintln!("[link2]   round 0 device {}: B {t_b:.0} ms, sum {:.0} ms, one-hot {:.0} ms, bins {:.0} ms", sh.dev, t_s - t_b, t_o - t_s, t0.elapsed().as_secs_f64() * 1e3 - t_o); }
    (sum, ohs, rb)
}

/// Round-1 range sum of the generic blocks as a cubic in t from the class bins
/// (class c: digits d_j = ⌊c/3^j⌋ mod 3 − 1; w_0 = (1−r0)d0 + r0 d1, w_1 likewise).
fn range_bin1_round(bins: &[K8], r0: &K8) -> [K8; 4] {
    let (two, three) = (K8::from_base(2), K8::from_base(3));
    let mut c = [K8::ZERO; 4];
    let dig = |x: i64| -> K8 { if x >= 0 { K8::from_base(x as u64) } else { K8::from_base((-x) as u64).neg() } };
    for cl in 0..81usize {
        let (j3, j2) = (bins[cl], bins[81 + cl]);
        if j3.is_zero() && j2.is_zero() { continue; }
        let d: Vec<K8> = (0..4).map(|j| dig(((cl / 3usize.pow(j)) % 3) as i64 - 1)).collect();
        let lin = |a: &K8, b: &K8| K8::ONE.sub(r0).mul(a).add(&r0.mul(b));
        let a = lin(&d[0], &d[1]);
        let dl = lin(&d[2], &d[3]).sub(&a);
        let (a2, dl2) = (a.mul(&a), dl.mul(&dl));
        c[0] = c[0].add(&j3.mul(&a2.mul(&a).sub(&a))).add(&j2.mul(&a2.sub(&a)));
        c[1] = c[1].add(&j3.mul(&three.mul(&a2).mul(&dl).sub(&dl))).add(&j2.mul(&two.mul(&a).mul(&dl).sub(&dl)));
        c[2] = c[2].add(&j3.mul(&three.mul(&a).mul(&dl2))).add(&j2.mul(&dl2));
        c[3] = c[3].add(&j3.mul(&dl2.mul(&dl)));
    }
    c
}
