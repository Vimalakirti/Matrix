//! Device side of the masked graph PIOP (`cuda_almost_goldilocks/
//! morpheus_piop.cuh`): edge values cached on the GPU, partial evaluations,
//! MLEs, eq tables, variable gathers and product-sumcheck tables over K.
//! Everything here equals the host functions in `tensor.rs` / the CPU tables.

use crate::morpheus::field8::K8;
use almost_goldilocks_cuda::{ffi, DeviceBuffer};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Use the device for tables of at least this many entries.
pub fn threshold() -> usize {
    static T: once_cell::sync::Lazy<usize> = once_cell::sync::Lazy::new(|| std::env::var("MORPHEUS_GPU_MIN").ok().and_then(|s| s.parse().ok()).unwrap_or(1 << 12));
    *T
}
/// Smallest table evaluated by `mle` on the device (one kernel, one readback).
/// Device sumcheck tables at most this long finish on the host
/// (MORPHEUS_SC_HOST_LEN, default 256; 0 keeps them on the device).
pub fn sc_host_len() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("MORPHEUS_SC_HOST_LEN").ok().and_then(|s| s.parse().ok()).unwrap_or(256))
}
pub fn mle_threshold() -> usize {
    static T: once_cell::sync::Lazy<usize> = once_cell::sync::Lazy::new(|| std::env::var("MORPHEUS_MLE_GPU_MIN").ok().and_then(|s| s.parse().ok()).unwrap_or(1 << 9));
    *T
}
pub fn enabled() -> bool {
    if !cfg!(feature = "cuda") {
        assert!(std::env::var_os("MORPHEUS_PIOP_GPU").map(|v| v == "0").unwrap_or(true), "GPU requested in a CUDA-free build");
        return false;
    }
    static E: once_cell::sync::Lazy<bool> = once_cell::sync::Lazy::new(|| std::env::var("MORPHEUS_PIOP_GPU").ok().as_deref() != Some("0"));
    *E
}

fn kw(v: &[K8]) -> Vec<u64> { v.iter().flat_map(|k| k.words()).collect() }
pub fn kvec(w: &[u64]) -> Vec<K8> { w.chunks(8).map(|c| K8(c.try_into().unwrap())).collect() }

/// A K table on the device (8 words per entry).
pub struct KTab { pub buf: DeviceBuffer<u64>, pub len: usize }

impl KTab {
    pub fn new(len: usize) -> Self { KTab { buf: DeviceBuffer::<u64>::new(8 * len.max(1)).expect("device alloc"), len } }
    pub fn zeroed(len: usize) -> Self { let mut t = KTab::new(len); t.buf.zero().expect("memset"); t }
    pub fn from_host(v: &[K8]) -> Self { KTab { buf: DeviceBuffer::from_slice(&kw(v)).unwrap(), len: v.len() } }
    pub fn to_host(&self) -> Vec<K8> { kvec(&self.buf.read_slice(0, 8 * self.len).unwrap()) }
    pub fn first(&self) -> K8 { kvec(&self.buf.read_slice(0, 8).unwrap())[0] }
}

/// (uploads, bytes, seconds) of edge-value uploads.
pub static UPLOAD: Mutex<(usize, usize, f64)> = Mutex::new((0, 0, 0.0));

/// Edge values on the device, uploaded once per proof up to a byte budget
/// (MORPHEUS_EDGE_CACHE_GB, default 24); beyond it the oldest entries are
/// dropped and re-uploaded on demand.
#[derive(Default)]
pub struct EdgeCache { m: Mutex<(HashMap<usize, Arc<DeviceBuffer<u64>>>, std::collections::VecDeque<usize>, usize)> }

pub fn edge_budget() -> usize {
    static B: once_cell::sync::Lazy<usize> = once_cell::sync::Lazy::new(|| (std::env::var("MORPHEUS_EDGE_CACHE_GB").ok().and_then(|s| s.parse::<f64>().ok()).unwrap_or(24.0) * 1e9) as usize);
    *B
}

impl EdgeCache {
    pub fn get(&self, e: usize, vals: &Arc<Vec<u64>>) -> Arc<DeviceBuffer<u64>> {
        let mut g = self.m.lock().unwrap();
        if let Some(b) = g.0.get(&e) { return b.clone(); }
        let bytes = vals.len() * 8;
        while g.2 + bytes > edge_budget() {
            let Some(old) = g.1.pop_front() else { break };
            if let Some(b) = g.0.remove(&old) { g.2 -= b.len() * 8; }
        }
        let t = std::time::Instant::now();
        // Under memory pressure (other resident buffers), drop the whole
        // cache once and retry.
        let b = match DeviceBuffer::from_slice(vals) {
            Ok(b) => Arc::new(b),
            Err(_) => { g.0.clear(); g.1.clear(); g.2 = 0; Arc::new(DeviceBuffer::from_slice(vals).expect("edge upload")) }
        };
        let mut st = UPLOAD.lock().unwrap();
        st.0 += 1; st.1 += bytes; st.2 += t.elapsed().as_secs_f64();
        g.0.insert(e, b.clone());
        g.1.push_back(e);
        g.2 += bytes;
        b
    }
    /// Bytes resident.
    pub fn bytes(&self) -> usize { self.m.lock().unwrap().2 }
    /// Free every cached edge.
    pub fn clear(&self) { let mut g = self.m.lock().unwrap(); g.0.clear(); g.1.clear(); g.2 = 0; }
}

/// Partial evaluation of an n-variable base table at `fixed`, result over
/// the remaining variables in order (= tensor::partial_eval).
pub fn partial_eval(vals: &DeviceBuffer<u64>, n: usize, fixed: &[(usize, K8)]) -> KTab {
    let mut is_fixed = vec![false; n];
    for &(v, _) in fixed { is_fixed[v] = true; }
    let rest: Vec<i32> = (0..n).filter(|&v| !is_fixed[v]).map(|v| v as i32).collect();
    let fix: Vec<i32> = fixed.iter().map(|x| x.0 as i32).collect();
    let e = eq(&fixed.iter().map(|x| x.1).collect::<Vec<_>>());
    let out = KTab::new(1usize << rest.len());
    let rc = unsafe { ffi::agl_mpiop_pe(vals.as_ptr(), rest.as_ptr(), rest.len() as i32, fix.as_ptr(), fix.len() as i32, e.buf.as_ptr(), out.buf.as_ptr() as *mut u64) };
    assert_eq!(rc, 0, "partial evaluation kernel");
    out
}

/// MLE of a base table at a K point.
/// (calls, host eq ns, device call ns) of `mle`.
pub static MLE_STATS: (std::sync::atomic::AtomicU64, std::sync::atomic::AtomicU64, std::sync::atomic::AtomicU64) = (std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0));
pub fn mle(vals: &DeviceBuffer<u64>, point: &[K8]) -> K8 {
    use std::sync::atomic::Ordering::Relaxed;
    let t0 = std::time::Instant::now();
    // One kernel over the split eq table (no 2^n table, one small readback).
    // Point split low/mid/high on the device (eq tables of ≤ 2^8 entries per block).
    let n = point.len();
    if n > 24 { return partial_eval(vals, n, &point.iter().copied().enumerate().collect::<Vec<_>>()).first(); }
    let pw: Vec<u64> = point.iter().flat_map(|k| k.0).collect();
    let mut out = [0u64; 8];
    let t1 = std::time::Instant::now();
    let rc = unsafe { ffi::agl_mpiop_mle(vals.as_ptr(), pw.as_ptr(), n as i32, out.as_mut_ptr()) };
    assert_eq!(rc, 0, "mle kernel");
    MLE_STATS.0.fetch_add(1, Relaxed); MLE_STATS.1.fetch_add((t1 - t0).as_nanos() as u64, Relaxed); MLE_STATS.2.fetch_add(t1.elapsed().as_nanos() as u64, Relaxed);
    K8(out)
}

/// eq table of a point (little-endian, = field8::eq_table).
/// Small tables come from the host; larger ones are the outer product of the
/// two halves' tables (one launch instead of one per variable).
pub fn eq(point: &[K8]) -> KTab {
    let n = point.len();
    if n <= 10 { return KTab::from_host(&crate::morpheus::field8::eq_table(point)); }
    let nlo = n / 2;
    let lo = KTab::from_host(&crate::morpheus::field8::eq_table(&point[..nlo]));
    let hi = KTab::from_host(&crate::morpheus::field8::eq_table(&point[nlo..]));
    let out = KTab::new(1usize << n);
    let rc = unsafe { ffi::agl_mpiop_eq_outer(lo.buf.as_ptr(), nlo as i32, hi.buf.as_ptr(), (n - nlo) as i32, out.buf.as_ptr() as *mut u64) };
    assert_eq!(rc, 0, "eq kernel");
    out
}

/// out[x] = in[Σ_j bit(x, pos_j) << j] over 2^n entries.
pub fn gather(t: &KTab, pos: &[usize], n: usize) -> KTab {
    let p: Vec<i32> = pos.iter().map(|&x| x as i32).collect();
    let out = KTab::new(1usize << n);
    let rc = unsafe { ffi::agl_mpiop_gather(t.buf.as_ptr(), p.as_ptr(), p.len() as i32, 1u64 << n, out.buf.as_ptr() as *mut u64) };
    assert_eq!(rc, 0, "gather kernel");
    out
}

/// out[i] = table[(sel[i] >> shift) & (2^w − 1)].
pub fn sel_lookup(sel: &DeviceBuffer<u64>, shift: usize, w: usize, table: &[K8]) -> KTab {
    assert_eq!(table.len(), 1usize << w);
    let t = KTab::from_host(table);
    let out = KTab::new(sel.len());
    let rc = unsafe { ffi::agl_mpiop_sel_lookup(sel.as_ptr(), sel.len() as u64, shift as i32, (1u64 << w) - 1, t.buf.as_ptr(), out.buf.as_ptr() as *mut u64) };
    assert_eq!(rc, 0, "selector lookup kernel");
    out
}

pub fn axpy(out: &mut KTab, t: &KTab, c: &K8) {
    let rc = unsafe { ffi::agl_mpiop_axpy(out.buf.as_mut_ptr(), t.buf.as_ptr(), out.len as u64, c.words().as_ptr()) };
    assert_eq!(rc, 0);
}

pub fn lift(vals: &DeviceBuffer<u64>, n: usize) -> KTab {
    let out = KTab::new(n);
    let rc = unsafe { ffi::agl_mpiop_lift(vals.as_ptr(), n as u64, out.buf.as_ptr() as *mut u64) };
    assert_eq!(rc, 0);
    out
}

/// Σ_k Π_j lerp(T_j, k, t) for t = 0..d.
pub fn sc_round(tabs: &[KTab], d: usize) -> Vec<K8> {
    let ptrs: Vec<*const u64> = tabs.iter().map(|t| t.buf.as_ptr()).collect();
    let mut out = vec![0u64; 8 * (d + 1)];
    let rc = unsafe { ffi::agl_mpiop_sc_round(ptrs.as_ptr(), ptrs.len() as i32, tabs[0].len as u64, d as i32, out.as_mut_ptr()) };
    assert_eq!(rc, 0, "sumcheck round kernel");
    kvec(&out)
}

/// Fold every table with r and return the next round's d+1 sums over the
/// folded tables (= sc_fold on each table, then sc_round), in one launch.
pub fn sc_fold_round(tabs: &[KTab], r: &K8, d: usize) -> (Vec<KTab>, Vec<K8>) {
    let len = tabs[0].len;
    let outs: Vec<KTab> = tabs.iter().map(|_| KTab::new(len / 2)).collect();
    let ins: Vec<*const u64> = tabs.iter().map(|t| t.buf.as_ptr()).collect();
    let ops: Vec<*mut u64> = outs.iter().map(|t| t.buf.as_ptr() as *mut u64).collect();
    let mut out = vec![0u64; 8 * (d + 1)];
    let rc = unsafe { ffi::agl_mpiop_sc_fold_round(ins.as_ptr(), ops.as_ptr(), tabs.len() as i32, len as u64, r.words().as_ptr(), d as i32, out.as_mut_ptr()) };
    assert_eq!(rc, 0, "fused fold/round kernel");
    (outs, kvec(&out))
}

/// `sc_fold_round` into caller-owned tables whose buffers hold at least
/// len/2 entries (ping-pong buffers; no allocation per round).
pub fn sc_fold_round_into(tabs: &[KTab], outs: &mut [KTab], r: &K8, d: usize) -> Vec<K8> {
    let len = tabs[0].len;
    for o in outs.iter_mut() { assert!(o.buf.len() >= 8 * (len / 2), "ping-pong buffer too small"); o.len = len / 2; }
    let ins: Vec<*const u64> = tabs.iter().map(|t| t.buf.as_ptr()).collect();
    let ops: Vec<*mut u64> = outs.iter().map(|t| t.buf.as_ptr() as *mut u64).collect();
    let mut out = vec![0u64; 8 * (d + 1)];
    let rc = unsafe { ffi::agl_mpiop_sc_fold_round(ins.as_ptr(), ops.as_ptr(), tabs.len() as i32, len as u64, r.words().as_ptr(), d as i32, out.as_mut_ptr()) };
    assert_eq!(rc, 0, "fused fold/round kernel");
    kvec(&out)
}

pub fn sc_fold(t: &KTab, r: &K8) -> KTab {
    let out = KTab::new(t.len / 2);
    let rc = unsafe { ffi::agl_mpiop_sc_fold(t.buf.as_ptr(), out.buf.as_ptr() as *mut u64, t.len as u64, r.words().as_ptr()) };
    assert_eq!(rc, 0);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::morpheus::field8::eq_table;
    use crate::morpheus::piop::tensor;
    use almost_goldilocks_cuda::field::ALMOST_GOLDILOCKS_PRIME as Q;
    use rand::{Rng, SeedableRng};

    #[test]
    fn device_tables_match_host() {
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(9);
        let rk = |rng: &mut rand_chacha::ChaCha20Rng| K8(std::array::from_fn(|_| rng.gen::<u64>() % Q));
        for n in [5usize, 13] {
            let vals: Vec<u64> = (0..1usize << n).map(|i| if i % 3 == 0 { 0 } else if i % 3 == 1 { 1 } else { rng.gen::<u64>() % Q }).collect();
            let dv = DeviceBuffer::from_slice(&vals).unwrap();
            let pt: Vec<K8> = (0..n).map(|_| rk(&mut rng)).collect();
            assert_eq!(mle(&dv, &pt), tensor::mle(&vals, &pt), "mle n={n}");
            for fixed in [vec![(0usize, pt[0]), (3, pt[3])], vec![(1, pt[1])], (0..n - 1).map(|j| (j, pt[j])).collect::<Vec<_>>()] {
                assert_eq!(partial_eval(&dv, n, &fixed).to_host(), tensor::partial_eval(&vals, n, &fixed), "pe n={n}");
            }
            assert_eq!(eq(&pt).to_host(), eq_table(&pt));
            // A product round and a fold.
            let a = KTab::from_host(&eq_table(&pt));
            let b = lift(&dv, 1 << n);
            let s = sc_round(&[a, b], 2);
            let at = eq_table(&pt);
            let bt: Vec<K8> = vals.iter().map(|&v| K8::from_base(v)).collect();
            for t in 0..=2u64 {
                let tk = K8::from_base(t);
                let want = (0..1usize << (n - 1)).fold(K8::ZERO, |acc, k| {
                    let x = at[2 * k].add(&tk.mul(&at[2 * k + 1].sub(&at[2 * k])));
                    let y = bt[2 * k].add(&tk.mul(&bt[2 * k + 1].sub(&bt[2 * k])));
                    acc.add(&x.mul(&y))
                });
                assert_eq!(s[t as usize], want);
            }
            // Fused fold + next round equals the separate kernels.
            let r = rk(&mut rng);
            let tabs = [KTab::from_host(&eq_table(&pt)), lift(&dv, 1 << n)];
            let (fused, s2) = sc_fold_round(&tabs, &r, 3);
            let sep: Vec<KTab> = tabs.iter().map(|t| sc_fold(t, &r)).collect();
            assert_eq!(s2, sc_round(&sep, 3), "fused round sums n={n}");
            for (a, b) in fused.iter().zip(&sep) { assert_eq!(a.to_host(), b.to_host(), "fused fold n={n}"); }
        }
    }
}
