//! GPU randomized Ajtai commitment (`cuda_almost_goldilocks/morpheus_commit.cuh`).
//!
//! The PRG key of one kind is expanded on the GPU once and cached in the NTT
//! domain for the primes in use (`KeyCache`). A commitment is then
//!     partial (per column range, per GPU) -> add -> finish,
//! where partials are NTT-domain accumulators [V][k][rank][4096] mod p_i, so
//! column shards on different GPUs combine by addition before one inverse
//! NTT and CRT. Results are exactly `commit::commit` / `ring::matvec`.

use super::keys::{KeyKind, KeySeed};
use super::ring::N;
use almost_goldilocks_cuda::{ffi, DeviceBuffer};
use std::os::raw::c_void;

/// NTT-domain key cache for columns [col0, col0+ncols) of one key kind.
pub struct KeyCache {
    pub kind: KeyKind,
    pub rows: usize,
    pub col0: u64,
    pub ncols: usize,
    /// Primes cached (3 for ternary/binary inputs, 5 for Gaussian ones).
    pub k: usize,
    pub buf: DeviceBuffer<u32>,
}

impl KeyCache {
    pub fn build(seed: &KeySeed, kind: KeyKind, rows: usize, col0: u64, ncols: usize, k: usize) -> Self {
        let mut buf = DeviceBuffer::<u32>::new(k * ncols * rows * N).expect("key cache alloc");
        let rc = unsafe { ffi::agl_mcommit_keygen(seed.0.as_ptr(), kind as u32, rows as i32, col0, ncols as i32, k as i32, buf.as_mut_ptr()) };
        assert_eq!(rc, 0, "keygen failed");
        KeyCache { kind, rows, col0, ncols, k, buf }
    }
    pub fn bytes(&self) -> usize { self.buf.len() * 4 }
    /// The same descriptor with an empty buffer (to be replaced).
    pub fn clone_meta(&self) -> Self { KeyCache { kind: self.kind, rows: self.rows, col0: self.col0, ncols: self.ncols, k: self.k, buf: DeviceBuffer::<u32>::new(1).unwrap() } }
}

/// Input columns on the device: V vectors x C columns x 4096 coefficients.
pub enum Cols<'a> {
    I8(&'a DeviceBuffer<i8>),
    I64(&'a DeviceBuffer<i64>),
}

/// NTT-domain partial commitment with `k` (<= cache.k) primes. `colidx[c]`
/// is the cache-relative key column of input column c (None = identity).
pub fn partial(cache: &KeyCache, k: usize, x: Cols, v: usize, c: usize, colidx: Option<&DeviceBuffer<u32>>) -> DeviceBuffer<u32> {
    assert!(k <= cache.k && (1..=2).contains(&v));
    let mut acc = DeviceBuffer::<u32>::new(v * k * cache.rows * N).expect("acc alloc");
    let (ptr, bytes, len) = match x {
        Cols::I8(b) => (b.as_ptr() as *const c_void, 1, b.len()),
        Cols::I64(b) => (b.as_ptr() as *const c_void, 8, b.len()),
    };
    assert_eq!(len, v * c * N);
    // The cache stores primes [0, cache.k); using its first k primes needs the
    // per-prime stride of the cache, so the kernel indexes with kcols = ncols
    // and prime-major layout: only k < cache.k is a prefix of the buffer.
    let idx = colidx.map(|b| b.as_ptr()).unwrap_or(std::ptr::null());
    let rc = unsafe {
        ffi::agl_mcommit_partial(cache.buf.as_ptr(), cache.ncols as i32, cache.rows as i32, k as i32, ptr, bytes, v as i32, c as i32, 0, idx, acc.as_mut_ptr())
    };
    assert_eq!(rc, 0, "partial commit failed");
    acc
}

/// `partial` on a raw device range of `v·c·N` elements of `bytes` (1 or 8).
pub fn partial_raw(cache: &KeyCache, k: usize, ptr: *const c_void, bytes: i32, v: usize, c: usize, vstride: usize, colidx: Option<&DeviceBuffer<u32>>) -> DeviceBuffer<u32> {
    assert!(k <= cache.k && (1..=8).contains(&v));
    let mut acc = DeviceBuffer::<u32>::new(v * k * cache.rows * N).expect("acc alloc");
    let idx = colidx.map(|b| b.as_ptr()).unwrap_or(std::ptr::null());
    let rc = unsafe { ffi::agl_mcommit_partial(cache.buf.as_ptr(), cache.ncols as i32, cache.rows as i32, k as i32, ptr, bytes, v as i32, c as i32, vstride as u64, idx, acc.as_mut_ptr()) };
    assert_eq!(rc, 0, "partial commit failed");
    acc
}

/// dst += src for NTT-domain accumulators on the same device.
pub fn add(dst: &mut DeviceBuffer<u32>, src: &DeviceBuffer<u32>, v: usize, k: usize, rows: usize) {
    let rc = unsafe { ffi::agl_mcommit_add(dst.as_mut_ptr(), src.as_ptr(), v as i32, k as i32, rows as i32) };
    assert_eq!(rc, 0);
}

/// Inverse NTT, CRT and identity error: [V][rows][4096] canonical mod q.
pub fn finish(mut acc: DeviceBuffer<u32>, v: usize, k: usize, rows: usize, err: Option<&DeviceBuffer<i64>>) -> DeviceBuffer<u64> {
    let mut out = DeviceBuffer::<u64>::new(v * rows * N).expect("out alloc");
    let e = err.map(|b| b.as_ptr()).unwrap_or(std::ptr::null());
    let rc = unsafe { ffi::agl_mcommit_finish(acc.as_mut_ptr(), v as i32, k as i32, rows as i32, e, out.as_mut_ptr()) };
    assert_eq!(rc, 0, "finish failed");
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::commit::{commit, BlockWitness};
    use super::super::params::Profile;
    use super::super::ring::{center, matvec, primes_for_bound};
    use super::super::keys::key_element;
    use super::super::sampler::ternary_vec;
    use rand::{Rng, SeedableRng};

    #[test]
    fn gpu_commit_matches_cpu_commit() {
        for profile in [Profile::toy(), Profile { rank: 16, hiding: 5, source_payload_cols: 6, ..Profile::ternary() }] {
            let seed = KeySeed([4, 4, 2, 2, 1, 1, 0, 9]);
            let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(8);
            let pc = profile.source_payload_cols;
            let payload: Vec<Option<Vec<i64>>> = (0..pc).map(|c| if c == 1 { None } else { Some(ternary_vec(&mut rng, N).unwrap()) }).collect();
            let w = BlockWitness::with_fresh_hiding(&profile, payload, &mut rng).unwrap();
            let want = commit(&profile, &seed, KeyKind::Source, pc, &w);

            let total = pc + profile.hiding;
            let cache = KeyCache::build(&seed, KeyKind::Source, profile.rank, 0, total, 3);
            // Columns that contribute, as in commit(): nonzero payload, then hiding.
            let mut idx: Vec<u32> = Vec::new();
            let mut xs: Vec<i8> = Vec::new();
            for (c, col) in w.payload.iter().enumerate() {
                if let Some(v) = col { idx.push(c as u32); xs.extend(v.iter().map(|&t| t as i8)); }
            }
            for (h, col) in w.hiding.iter().enumerate() { idx.push((pc + h) as u32); xs.extend(col.iter().map(|&t| t as i8)); }
            let k = primes_for_bound((idx.len() as f64).log2() + 12.0 + 63.0);
            assert_eq!(k, 3);
            let dx = DeviceBuffer::from_slice(&xs).unwrap();
            let di = DeviceBuffer::from_slice(&idx).unwrap();
            let acc = partial(&cache, k, Cols::I8(&dx), 1, idx.len(), Some(&di));
            let err: Vec<i64> = w.error.iter().flatten().copied().collect();
            let de = DeviceBuffer::from_slice(&err).unwrap();
            let got = finish(acc, 1, k, profile.rank, Some(&de)).to_vec().unwrap();
            for r in 0..profile.rank { assert_eq!(&got[r * N..(r + 1) * N], &want[r][..], "rank {} row {r}", profile.rank); }
        }
    }

    #[test]
    fn gpu_gaussian_matvec_two_vectors_and_shards() {
        // Two Gaussian-sized vectors (|x| < 2^36), 5 primes, columns split in
        // two shards whose NTT-domain partials are added.
        let seed = KeySeed([7; 8]);
        let (rows, cols) = (16usize, 5usize);
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(9);
        let x: Vec<Vec<Vec<i64>>> = (0..2).map(|_| (0..cols).map(|_| (0..N).map(|_| rng.gen_range(-(1i64 << 36)..(1i64 << 36))).collect()).collect()).collect();
        let log2b = (cols as f64).log2() + 12.0 + 63.0 + 36.0;
        let k = primes_for_bound(log2b);
        assert!(k >= 4);
        let cache = KeyCache::build(&seed, KeyKind::Link, rows, 0, cols, 5);
        let split = 2;
        let pack = |r: std::ops::Range<usize>| -> Vec<i64> { let mut o = Vec::new(); for v in 0..2 { for c in r.clone() { o.extend_from_slice(&x[v][c]); } } o };
        let a = DeviceBuffer::from_slice(&pack(0..split)).unwrap();
        let b = DeviceBuffer::from_slice(&pack(split..cols)).unwrap();
        let ia = DeviceBuffer::from_slice(&(0..split as u32).collect::<Vec<_>>()).unwrap();
        let ib = DeviceBuffer::from_slice(&(split as u32..cols as u32).collect::<Vec<_>>()).unwrap();
        let mut acc = partial(&cache, k, Cols::I64(&a), 2, split, Some(&ia));
        let acc2 = partial(&cache, k, Cols::I64(&b), 2, cols - split, Some(&ib));
        add(&mut acc, &acc2, 2, k, rows);
        let got = finish(acc, 2, k, rows, None).to_vec().unwrap();
        for v in 0..2 {
            let want = matvec(rows, cols, &|i, c| key_element(&seed, KeyKind::Link, i as u32, c as u64).iter().map(|&t| center(t)).collect(), &x[v], log2b);
            for r in 0..rows { assert_eq!(&got[(v * rows + r) * N..(v * rows + r + 1) * N], &want[r][..], "v {v} row {r}"); }
        }
    }

    #[test]
    #[ignore]
    fn gpu_commit_throughput() {
        // One full ternary-profile source commitment (3 primes) and one dense
        // two-row Gaussian mask commitment (5 primes) against a key cache.
        let p = Profile::ternary();
        let seed = KeySeed([1; 8]);
        let cols: usize = std::env::var("MC_COLS").ok().and_then(|s| s.parse().ok()).unwrap_or(8192);
        let t = std::time::Instant::now();
        let cache = KeyCache::build(&seed, KeyKind::Source, p.rank, 0, cols, 5);
        println!("keygen {cols} cols x 16 rows x 5 primes: {:.1} ms ({:.2} GB)", t.elapsed().as_secs_f64() * 1e3, cache.bytes() as f64 / 1e9);
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(1);
        let x8: Vec<i8> = (0..cols * N).map(|_| rng.gen_range(-1i8..=1)).collect();
        let dx8 = DeviceBuffer::from_slice(&x8).unwrap();
        let x64: Vec<i64> = (0..2 * cols * N).map(|_| rng.gen_range(-(1i64 << 35)..(1i64 << 35))).collect();
        let dx64 = DeviceBuffer::from_slice(&x64).unwrap();
        for _ in 0..2 {
            let t = std::time::Instant::now();
            let acc = partial(&cache, 3, Cols::I8(&dx8), 1, cols, None);
            let _ = finish(acc, 1, 3, p.rank, None);
            let t1 = t.elapsed().as_secs_f64() * 1e3;
            let t = std::time::Instant::now();
            let acc = partial(&cache, 5, Cols::I64(&dx64), 2, cols, None);
            let _ = finish(acc, 2, 5, p.rank, None);
            let t2 = t.elapsed().as_secs_f64() * 1e3;
            println!("{cols} cols: ternary 1 vec 3 primes {t1:.1} ms; gaussian 2 vec 5 primes {t2:.1} ms  (full source = {} cols)", p.source_cols() - p.rank);
        }
    }
}
