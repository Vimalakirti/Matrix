//! Raw source digits stored two bits each.
//!
//! Every raw source digit lies in {-1, 0, 1} (signed bit planes, one-hot
//! selector bits, ternary hiding suffixes), so a source of 2^ell digits needs
//! 2^(ell-2) bytes instead of 2^ell. Digit i is bits 2(i mod 4)..2(i mod 4)+2
//! of byte i/4, two's complement: 00 = 0, 01 = 1, 11 = -1. The proofs are
//! unchanged; only the prover's storage (host and device) shrinks four times.

use rayon::prelude::*;
use std::sync::atomic::{AtomicU8, Ordering};

#[inline]
pub fn code(d: i8) -> u8 {
    debug_assert!((-1..=1).contains(&d), "raw digit {d} outside {{-1, 0, 1}}");
    (d as u8) & 3
}
#[inline]
pub fn digit(c: u8) -> i8 { (((c & 3) << 6) as i8) >> 6 }

/// One packed source.
#[derive(Clone, Default)]
pub struct Src {
    /// Packed bytes (len = digits / 4).
    pub b: Vec<u8>,
}

impl Src {
    pub fn zeros(digits: usize) -> Self {
        assert_eq!(digits % 4, 0);
        // Zeroed allocation: pages are mapped on first write (by the
        // parallel fill), and never-written ones stay shared zero pages.
        Src { b: vec![0u8; digits / 4] }
    }
    pub fn from_i8(d: &[i8]) -> Self {
        let mut s = Src::zeros(d.len());
        s.b.par_chunks_mut(1 << 18).zip(d.par_chunks(1 << 20)).for_each(|(o, i)| pack_into(i, o));
        s
    }
    #[inline] pub fn len(&self) -> usize { self.b.len() * 4 }
    #[inline] pub fn is_empty(&self) -> bool { self.b.is_empty() }
    #[inline] pub fn get(&self, i: usize) -> i8 { digit(self.b[i >> 2] >> (2 * (i & 3))) }
    /// Digits [start, start + out.len()) unpacked.
    pub fn unpack_into(&self, start: usize, out: &mut [i8]) {
        for (k, o) in out.iter_mut().enumerate() { *o = self.get(start + k); }
    }
    pub fn unpack(&self) -> Vec<i8> {
        let mut v = vec![0i8; self.len()];
        v.par_chunks_mut(1 << 20).zip(self.b.par_chunks(1 << 18)).for_each(|(o, i)| unpack_bytes(i, o));
        v
    }
    /// Overwrite digits [start, start + d.len()); start and the length are
    /// multiples of 4.
    pub fn put(&mut self, start: usize, d: &[i8]) {
        assert!(start % 4 == 0 && d.len() % 4 == 0);
        pack_into(d, &mut self.b[start / 4..(start + d.len()) / 4]);
    }
    /// Whether digits [start, start + n) are all zero.
    pub fn is_zero(&self, start: usize, n: usize) -> bool {
        if start % 4 == 0 && n % 4 == 0 { return self.b[start / 4..(start + n) / 4].iter().all(|&x| x == 0); }
        (start..start + n).all(|i| self.get(i) == 0)
    }
    /// Number of nonzero digits in [start, start + n), and whether every
    /// nonzero digit is 1.
    pub fn nonzeros(&self, start: usize, n: usize) -> (usize, bool) {
        let (mut cnt, mut ones) = (0usize, true);
        if start % 4 == 0 && n % 4 == 0 {
            for &x in &self.b[start / 4..(start + n) / 4] {
                if x == 0 { continue; }
                // Nonzero 2-bit codes; a -1 (11) has its high bit set.
                let nz = (x | (x >> 1)) & 0x55;
                cnt += nz.count_ones() as usize;
                if x & 0xaa != 0 { ones = false; }
            }
        } else {
            for i in start..start + n { let d = self.get(i); if d != 0 { cnt += 1; if d != 1 { ones = false; } } }
        }
        (cnt, ones)
    }
}

/// Pack d (len multiple of 4) into out (len d.len() / 4).
pub fn pack_into(d: &[i8], out: &mut [u8]) {
    for (o, q) in out.iter_mut().zip(d.chunks_exact(4)) {
        *o = code(q[0]) | code(q[1]) << 2 | code(q[2]) << 4 | code(q[3]) << 6;
    }
}
pub fn unpack_bytes(b: &[u8], out: &mut [i8]) {
    for (q, &x) in out.chunks_exact_mut(4).zip(b) {
        q[0] = digit(x); q[1] = digit(x >> 2); q[2] = digit(x >> 4); q[3] = digit(x >> 6);
    }
}

/// Raw pointer to a packed source for concurrent filling: rows that start
/// and end on byte boundaries are stored directly, other digits with an
/// atomic OR (the bytes start zero and each digit is written once).
#[derive(Clone, Copy)]
pub struct PackedPtr(pub *mut u8, pub usize);
unsafe impl Send for PackedPtr {}
unsafe impl Sync for PackedPtr {}
impl PackedPtr {
    pub fn of(s: &mut Src) -> Self { PackedPtr(s.b.as_mut_ptr(), s.b.len()) }
    /// Write digits d at digit offset `start`.
    pub fn write(&self, start: usize, d: &[i8]) {
        let mut i = 0;
        // Unaligned head and tail: atomic OR per digit.
        let or = |pos: usize, v: i8| {
            if v == 0 { return; }
            assert!(pos / 4 < self.1);
            let a = unsafe { &*(self.0.add(pos / 4) as *const AtomicU8) };
            a.fetch_or(code(v) << (2 * (pos & 3)), Ordering::Relaxed);
        };
        while i < d.len() && (start + i) % 4 != 0 { or(start + i, d[i]); i += 1; }
        let whole = (d.len() - i) / 4 * 4;
        if whole > 0 {
            assert!((start + i + whole) / 4 <= self.1);
            let out = unsafe { std::slice::from_raw_parts_mut(self.0.add((start + i) / 4), whole / 4) };
            pack_into(&d[i..i + whole], out);
            i += whole;
        }
        while i < d.len() { or(start + i, d[i]); i += 1; }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pack_roundtrip_and_scans() {
        let d: Vec<i8> = (0..4096).map(|i| [0i8, 1, -1, 0, 1][(i * 7 + i / 3) % 5]).collect();
        let s = Src::from_i8(&d);
        assert_eq!(s.unpack(), d);
        for i in 0..d.len() { assert_eq!(s.get(i), d[i]); }
        for (a, n) in [(0, 64), (3, 61), (100, 7), (4000, 96)] {
            let want = (d[a..a + n].iter().filter(|&&x| x != 0).count(), d[a..a + n].iter().all(|&x| x >= 0));
            assert_eq!(s.nonzeros(a, n), want);
            assert_eq!(s.is_zero(a, n), want.0 == 0);
        }
        let mut t = Src::zeros(4096);
        let p = PackedPtr::of(&mut t);
        p.write(0, &d[0..5]); p.write(5, &d[5..2049]); p.write(2049, &d[2049..]);
        assert_eq!(t.unpack(), d);
    }
}

/// A set of raw sources, as int8 digits (CPU reference paths) or packed.
pub trait Sources: Sync {
    fn count(&self) -> usize;
    /// Digits per source.
    fn src_len(&self) -> usize;
    fn digit(&self, s: usize, i: usize) -> i8;
    /// (nonzero digits, all nonzero digits are 1) of digits [start, start + n).
    fn nonzeros(&self, s: usize, start: usize, n: usize) -> (usize, bool);
    fn is_zero(&self, s: usize, start: usize, n: usize) -> bool { self.nonzeros(s, start, n).0 == 0 }
    /// Digits [start, start + out.len()) of source s.
    fn read(&self, s: usize, start: usize, out: &mut [i8]);
    /// The int8 sources, when stored that way.
    fn as_i8(&self) -> Option<&[Vec<i8>]> { None }
    /// Packed source s, when stored that way.
    fn packed(&self, _s: usize) -> Option<&Src> { None }
    /// Source s as int8 digits (a copy when packed).
    fn to_i8(&self, s: usize) -> std::borrow::Cow<'_, [i8]>;
}

/// int8 sources as a `Sources`.
#[derive(Clone, Copy)]
pub struct I8s<'a>(pub &'a [Vec<i8>]);
/// Packed sources as a `Sources`.
#[derive(Clone, Copy)]
pub struct Packs<'a>(pub &'a [Src]);

impl Sources for I8s<'_> {
    fn count(&self) -> usize { self.0.len() }
    fn src_len(&self) -> usize { self.0.first().map_or(0, |r| r.len()) }
    fn digit(&self, s: usize, i: usize) -> i8 { self.0[s][i] }
    fn nonzeros(&self, s: usize, start: usize, n: usize) -> (usize, bool) {
        let d = &self.0[s][start..start + n];
        (d.iter().filter(|&&x| x != 0).count(), d.iter().all(|&x| x == 0 || x == 1))
    }
    fn is_zero(&self, s: usize, start: usize, n: usize) -> bool { self.0[s][start..start + n].iter().all(|&x| x == 0) }
    fn read(&self, s: usize, start: usize, out: &mut [i8]) { out.copy_from_slice(&self.0[s][start..start + out.len()]); }
    fn as_i8(&self) -> Option<&[Vec<i8>]> { Some(self.0) }
    fn to_i8(&self, s: usize) -> std::borrow::Cow<'_, [i8]> { std::borrow::Cow::Borrowed(&self.0[s]) }
}

impl Sources for Packs<'_> {
    fn count(&self) -> usize { self.0.len() }
    fn src_len(&self) -> usize { self.0.first().map_or(0, |r| r.len()) }
    fn digit(&self, s: usize, i: usize) -> i8 { self.0[s].get(i) }
    fn nonzeros(&self, s: usize, start: usize, n: usize) -> (usize, bool) { self.0[s].nonzeros(start, n) }
    fn is_zero(&self, s: usize, start: usize, n: usize) -> bool { self.0[s].is_zero(start, n) }
    fn read(&self, s: usize, start: usize, out: &mut [i8]) {
        if start % 4 == 0 && out.len() % 4 == 0 { unpack_bytes(&self.0[s].b[start / 4..(start + out.len()) / 4], out); } else { self.0[s].unpack_into(start, out); }
    }
    fn packed(&self, s: usize) -> Option<&Src> { Some(&self.0[s]) }
    fn to_i8(&self, s: usize) -> std::borrow::Cow<'_, [i8]> { std::borrow::Cow::Owned(self.0[s].unpack()) }
}

/// Packed sources held by reference (registered model and input sources
/// followed by the request's own).
#[derive(Clone, Copy)]
pub struct Refs<'a>(pub &'a [&'a Src]);

impl Sources for Refs<'_> {
    fn count(&self) -> usize { self.0.len() }
    fn src_len(&self) -> usize { self.0.first().map_or(0, |r| r.len()) }
    fn digit(&self, s: usize, i: usize) -> i8 { self.0[s].get(i) }
    fn nonzeros(&self, s: usize, start: usize, n: usize) -> (usize, bool) { self.0[s].nonzeros(start, n) }
    fn is_zero(&self, s: usize, start: usize, n: usize) -> bool { self.0[s].is_zero(start, n) }
    fn read(&self, s: usize, start: usize, out: &mut [i8]) { Packs(std::slice::from_ref(self.0[s])).read(0, start, out) }
    fn packed(&self, s: usize) -> Option<&Src> { Some(self.0[s]) }
    fn to_i8(&self, s: usize) -> std::borrow::Cow<'_, [i8]> { std::borrow::Cow::Owned(self.0[s].unpack()) }
}

/// Upload source s of `raw` as int8 digits into `dst` at digit offset `at`
/// (logical order; `phys` permutes each 4096-digit column to physical
/// order). Packed sources travel packed and are unpacked on the device.
#[cfg(feature = "cuda")]
pub fn upload_i8(raw: &dyn Sources, s: usize, dst: &mut almost_goldilocks_cuda::DeviceBuffer<i8>, at: usize, phys: bool, scratch: &mut Option<almost_goldilocks_cuda::DeviceBuffer<u8>>) {
    use almost_goldilocks_cuda::{ffi, DeviceBuffer};
    let len = raw.src_len();
    assert!(at + len <= dst.len());
    if let Some(ps) = raw.packed(s) {
        if scratch.as_ref().map_or(true, |b| b.len() < len / 4) { *scratch = Some(DeviceBuffer::<u8>::new(len / 4).expect("packed upload scratch")); }
        let sb = scratch.as_mut().unwrap();
        sb.copy_from_slice_at(0, &ps.b).expect("packed upload");
        let rc = unsafe { ffi::agl_morpheus_unpack2(sb.as_ptr(), dst.as_mut_ptr().add(at), len as u64, phys as i32) };
        assert_eq!(rc, 0, "unpack2");
    } else if !phys {
        dst.copy_from_slice_at(at, &raw.as_i8().unwrap()[s]).expect("raw upload");
    } else {
        let tmp = DeviceBuffer::from_slice(&raw.as_i8().unwrap()[s]).expect("raw upload");
        let rc = unsafe { ffi::agl_morpheus_permute(tmp.as_ptr(), dst.as_mut_ptr().add(at), len as u64) };
        assert_eq!(rc, 0, "permute");
    }
}

/// Upload source s of `raw` packed into `dst` at byte offset `at`.
#[cfg(feature = "cuda")]
pub fn upload_packed(raw: &dyn Sources, s: usize, dst: &mut almost_goldilocks_cuda::DeviceBuffer<u8>, at: usize) {
    match raw.packed(s) {
        Some(p) => dst.copy_from_slice_at(at, &p.b).expect("packed upload"),
        None => dst.copy_from_slice_at(at, &Src::from_i8(&raw.to_i8(s)).b).expect("packed upload"),
    }
}
