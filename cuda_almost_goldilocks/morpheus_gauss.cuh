// Morpheus S&P'27: GPU first pass of the certified discrete Gaussian sampler.
//
// One thread per Gaussian coordinate. Each thread derives its private coins
// from the addressed backend (SHAKE-256 of the fixed 106-byte message, one
// Keccak-f[1600] permutation per 512-bit block) and runs the recommended P128
// first-stage certificate (80 coordinate bits, P = 128, 43 atanh terms) in
// 64-bit limbs. Outcomes:
//   flag 0: certified k accepted by the 64-bit correction gate (final);
//   flag 1: anything else (refine, gate miss, hint miss, trial cap) -- the
//           host replays the coordinate with the exact sampler on the same
//           addresses, which returns the identical value when it exists.
// Mirrors zk-torch-4/src/morpheus/gaussian_fast.rs line by line.
#pragma once
#include <cstdint>
#include "morpheus_shake.cuh"

namespace mgauss {

typedef unsigned long long u64;

using mshake::keccak_f;

// The 106-byte coin message with padding, as 17 rate lanes. The host fills
// every fixed field; the kernel patches purpose (byte 82), coordinate
// (93..96, big-endian), outer (97..98) and trial (99..100).
struct CoinBase { u64 lane[17]; };

__device__ __forceinline__ void put_byte(u64 st[25], int b, u64 v) { st[b >> 3] ^= (v & 0xff) << (8 * (b & 7)); }

/// First 128 output bits (bytes 0..15, big-endian) of block 0 at an address.
__device__ __noinline__ void coin_block0(const CoinBase& base, int purpose, uint32_t coord, uint32_t outer, uint32_t trial,
                                            u64& hi, u64& lo) {
    u64 st[25];
#pragma unroll
    for (int i = 0; i < 17; i++) st[i] = base.lane[i];
#pragma unroll
    for (int i = 17; i < 25; i++) st[i] = 0;
    put_byte(st, 82, purpose);
    put_byte(st, 93, coord >> 24); put_byte(st, 94, coord >> 16); put_byte(st, 95, coord >> 8); put_byte(st, 96, coord);
    put_byte(st, 97, outer >> 8); put_byte(st, 98, outer);
    put_byte(st, 99, trial >> 8); put_byte(st, 100, trial);
    keccak_f(st);
    hi = __byte_perm((uint32_t)(st[0] >> 32), 0, 0x0123) | ((u64)__byte_perm((uint32_t)st[0], 0, 0x0123) << 32);
    lo = __byte_perm((uint32_t)(st[1] >> 32), 0, 0x0123) | ((u64)__byte_perm((uint32_t)st[1], 0, 0x0123) << 32);
}

// ---------------------------------------------------------------- limbs
template <int N> struct U { u64 w[N]; };

template <int N> __device__ __forceinline__ U<N> uzero() { U<N> r; for (int i = 0; i < N; i++) r.w[i] = 0; return r; }
template <int M, int N> __device__ __forceinline__ U<M> resize(const U<N>& a) {
    U<M> r;
#pragma unroll
    for (int i = 0; i < M; i++) r.w[i] = i < N ? a.w[i] : 0;
    return r;
}
template <int A, int B> __device__ __forceinline__ U<A + B> mul(const U<A>& a, const U<B>& b) {
    U<A + B> r = uzero<A + B>();
#pragma unroll
    for (int i = 0; i < A; i++) {
        u64 carry = 0;
#pragma unroll
        for (int j = 0; j < B; j++) {
            u64 lo = a.w[i] * b.w[j], hi = __umul64hi(a.w[i], b.w[j]);
            u64 s = r.w[i + j] + lo; u64 c1 = s < lo;
            u64 s2 = s + carry; u64 c2 = s2 < carry;
            r.w[i + j] = s2; carry = hi + c1 + c2;
        }
        r.w[i + B] = carry;
    }
    return r;
}
template <int N> __device__ __forceinline__ U<N> add(const U<N>& a, const U<N>& b) {
    U<N> r; u64 c = 0;
#pragma unroll
    for (int i = 0; i < N; i++) { u64 s = a.w[i] + b.w[i]; u64 c1 = s < a.w[i]; u64 s2 = s + c; u64 c2 = s2 < c; r.w[i] = s2; c = c1 | c2; }
    return r;
}
template <int N> __device__ __forceinline__ U<N> sub(const U<N>& a, const U<N>& b) {
    U<N> r; u64 br = 0;
#pragma unroll
    for (int i = 0; i < N; i++) { u64 d = a.w[i] - b.w[i]; u64 b1 = a.w[i] < b.w[i]; u64 d2 = d - br; u64 b2 = d < br; r.w[i] = d2; br = b1 | b2; }
    return r;
}
template <int N> __device__ __forceinline__ int cmp(const U<N>& a, const U<N>& b) {
#pragma unroll
    for (int i = N - 1; i >= 0; i--) { if (a.w[i] != b.w[i]) return a.w[i] < b.w[i] ? -1 : 1; }
    return 0;
}
template <int N> __device__ __forceinline__ bool is_zero(const U<N>& a) {
    u64 o = 0;
#pragma unroll
    for (int i = 0; i < N; i++) o |= a.w[i];
    return o == 0;
}
/// floor(a / 2^128) into M limbs.
template <int M, int N> __device__ __forceinline__ U<M> shr128(const U<N>& a) {
    U<M> r;
#pragma unroll
    for (int i = 0; i < M; i++) r.w[i] = i + 2 < N ? a.w[i + 2] : 0;
    return r;
}
/// ceil(a / 2^128) into M limbs.
template <int M, int N> __device__ __forceinline__ U<M> cshr128(const U<N>& a) {
    U<M> r = shr128<M>(a);
    if (a.w[0] | a.w[1]) { U<M> one = uzero<M>(); one.w[0] = 1; r = add(r, one); }
    return r;
}
template <int N> __device__ __forceinline__ U<N> shl(const U<N>& a, int s) {
    U<N> r = uzero<N>();
    int q = s >> 6, b = s & 63;
#pragma unroll
    for (int i = N - 1; i >= 0; i--) {
        int j = i - q;
        if (j >= 0) {
            r.w[i] = a.w[j] << b;
            if (b && j >= 1) r.w[i] |= a.w[j - 1] >> (64 - b);
        }
    }
    return r;
}
template <int N> __device__ __forceinline__ int bits(const U<N>& a) {
#pragma unroll
    for (int i = N - 1; i >= 0; i--) if (a.w[i]) return 64 * i + 64 - __clzll(a.w[i]);
    return 0;
}
template <int N> __device__ __forceinline__ U<N> small(u64 v) { U<N> r = uzero<N>(); r.w[0] = v; return r; }

// ---------------------------------------------------------------- constants
struct Consts {
    u64 clo[43][3], chi[43][3];
    u64 ln2lo[2], ln2hi[2];
    u64 tail;
    u64 s8[2];      // 8 sigma^2
    u64 d4[3];      // 8 sigma^4 (gate denominator)
    u64 sigma, bound, gate_small;
};

__device__ __forceinline__ U<3> one3() { U<3> r = uzero<3>(); r.w[2] = 1; return r; }

template <int N> __device__ __forceinline__ double to_double(const U<N>& a) {
    double r = 0;
#pragma unroll
    for (int i = N - 1; i >= 0; i--) r = r * 18446744073709551616.0 + (double)a.w[i];
    return r;
}

// Exact bounded fallback: long division of d * 2^128 by den.
// Precondition: d < 2^128 < den. No floating-point estimate is trusted.
__device__ __forceinline__ U<2> div_z_exact(const U<2>& d, const U<3>& den, bool& inexact) {
    U<3> rem = resize<3>(d);
    U<2> q = uzero<2>();
    for (int bit = 127; bit >= 0; --bit) {
        rem = shl(rem, 1);
        if (cmp(rem, den) >= 0) {
            rem = sub(rem, den);
            q.w[bit >> 6] |= 1ULL << (bit & 63);
        }
    }
    inexact = !is_zero(rem);
    return q;
}

/// floor and remainder-nonzero of (d * 2^128) / den, d < 2^128 < den < 2^130.
/// Quotient digits are estimated in double precision, biased low by 2^-46 so
/// an exact product guard validates each estimate before subtraction. Bounded
/// correction falls back to restoring division; correctness does not rely on
/// the floating-point underestimate claim. The result is the exact floor.
__device__ __forceinline__ U<2> div_z(const U<2>& d, const U<3>& den, bool& inexact) {
    U<4> rem = uzero<4>();
    rem.w[2] = d.w[0]; rem.w[3] = d.w[1];
    U<2> q = uzero<2>();
    const double dd = to_double(den);
#pragma unroll
    for (int it = 0; it < 3; it++) {
        double qd = to_double(rem) / dd * (1.0 - 1.4210854715202004e-14);
        if (!isfinite(qd) || qd >= ldexp(1.0, 128)) return div_z_exact(d, den, inexact);
        if (!(qd >= 1.0)) break;
        U<2> qe;
        if (qd < 18446744073709551616.0) { qe.w[0] = (u64)qd; qe.w[1] = 0; }
        else { double h = floor(qd * 5.421010862427522e-20); qe.w[1] = (u64)h; qe.w[0] = (u64)(qd - h * 18446744073709551616.0); }
        U<5> product = mul(qe, den);
        U<5> remainder = resize<5>(rem);
        if (cmp(product, remainder) > 0) return div_z_exact(d, den, inexact);
        rem = resize<4>(sub(remainder, product));
        q = add(q, qe);
    }
    U<4> den4 = resize<4>(den);
    for (int it = 0; it < 2 && cmp(rem, den4) >= 0; ++it) {
        rem = sub(rem, den4); q = add(q, small<2>(1));
    }
    if (cmp(rem, den4) >= 0) return div_z_exact(d, den, inexact);
    inexact = !is_zero(rem);
    return q;
}

__device__ U<3> ans_lo(const Consts& c, const U<3>& m) {
    U<2> d = resize<2>(sub(m, one3()));
    U<3> den = add(m, one3());
    bool inexact;
    U<2> z = div_z(d, den, inexact);
    U<2> zz = shr128<2>(mul(z, z));
    U<3> acc; for (int i = 0; i < 3; i++) acc.w[i] = c.clo[42][i];
#pragma unroll 1
    for (int j = 41; j >= 0; j--) {
        U<3> cj; for (int i = 0; i < 3; i++) cj.w[i] = c.clo[j][i];
        acc = add(shr128<3>(mul(acc, zz)), cj);
    }
    U<3> r = shr128<3>(mul(z, acc));
    return add(r, r);
}

__device__ U<3> ans_hi(const Consts& c, const U<3>& m) {
    U<2> d = resize<2>(sub(m, one3()));
    U<3> den = add(m, one3());
    bool inexact;
    U<2> z = div_z(d, den, inexact);
    if (inexact) z = add(z, small<2>(1));
    U<2> zz = cshr128<2>(mul(z, z));
    U<3> acc; for (int i = 0; i < 3; i++) acc.w[i] = c.chi[42][i];
#pragma unroll 1
    for (int j = 41; j >= 0; j--) {
        U<3> cj; for (int i = 0; i < 3; i++) cj.w[i] = c.chi[j][i];
        acc = add(cshr128<3>(mul(acc, zz)), cj);
    }
    U<3> r = cshr128<3>(mul(z, acc));
    return add(add(r, r), small<3>(c.tail));
}

/// Square interval of X = 2V-1 for an 80-bit prefix (hi16:lo64); sign +1/-1/0.
__device__ __forceinline__ void square_iv(u64 hi, u64 lo, U<3>& xl, U<3>& xh, int& sign) {
    U<3> u = uzero<3>(); u.w[0] = lo; u.w[1] = hi;
    U<3> a = shl(u, 49);
    U<3> b = shl(add(u, small<3>(1)), 49);
    U<3> one = one3();
    bool apos = cmp(a, one) >= 0, bpos = cmp(b, one) >= 0;
    U<3> ml = apos ? sub(a, one) : sub(one, a);
    U<3> mh = bpos ? sub(b, one) : sub(one, b);
    U<6> l = mul(ml, ml), h = mul(mh, mh);
    bool straddle = (!apos && bpos) || is_zero(ml) || is_zero(mh);
    U<6> lo6 = cmp(l, h) <= 0 ? l : h;
    U<6> hi6 = cmp(l, h) <= 0 ? h : l;
    xl = straddle ? uzero<3>() : shr128<3>(lo6);
    xh = cshr128<3>(hi6);
    sign = (apos || is_zero(ml)) ? 1 : ((!bpos || is_zero(mh)) ? -1 : 0);
}

__device__ __forceinline__ bool certify(const U<6>& tlo, const U<6>& thi, const U<3>& slo, const U<3>& shi, u64 m) {
    u64 up = 2 * m + 1;
    U<2> up2; up2.w[0] = up * up; up2.w[1] = __umul64hi(up, up);
    if (!(cmp(thi, resize<6>(mul(up2, slo))) < 0)) return false;
    if (m == 0) return true;
    u64 dn = 2 * m - 1;
    U<2> dn2; dn2.w[0] = dn * dn; dn2.w[1] = __umul64hi(dn, dn);
    return cmp(resize<6>(mul(dn2, shi)), tlo) < 0;
}

/// 0 = disk rejected, 1 = rounded (k set), 2 = refine.
__device__ __noinline__ int trial(const Consts& c, u64 uh, u64 ul, u64 vh, u64 vl, long long& k) {
    U<3> xl, xh, yl, yh; int xs, ys;
    square_iv(uh, ul, xl, xh, xs);
    square_iv(vh, vl, yl, yh, ys);
    U<3> slo = add(xl, yl), shi = add(xh, yh);
    U<3> one = one3();
    if (cmp(slo, one) >= 0) return 0;
    const u64 top_hi = 0xffff, top_lo = ~0ULL;
    if ((uh == 0 && ul == 0) || (vh == 0 && vl == 0) || (uh == top_hi && ul == top_lo) || (vh == top_hi && vl == top_lo)) return 2;
    if (is_zero(slo) || cmp(shi, sub(one, small<3>(1))) > 0) return 2;
    int elo = 129 - bits(slo), ehi = 129 - bits(shi);
    U<2> l2lo; l2lo.w[0] = c.ln2lo[0]; l2lo.w[1] = c.ln2lo[1];
    U<2> l2hi; l2hi.w[0] = c.ln2hi[0]; l2hi.w[1] = c.ln2hi[1];
    U<3> nl_hi = sub(resize<3>(mul(l2hi, small<1>((u64)elo))), ans_lo(c, shl(slo, elo)));
    U<3> a = resize<3>(mul(l2lo, small<1>((u64)ehi)));
    U<3> b = ans_hi(c, shl(shi, ehi));
    U<3> fl = sub(one, shi);
    U<3> nl_lo = fl;
    if (cmp(a, b) > 0) { U<3> dlt = sub(a, b); if (cmp(dlt, fl) > 0) nl_lo = dlt; }
    U<2> s8; s8.w[0] = c.s8[0]; s8.w[1] = c.s8[1];
    U<6> tlo = resize<6>(mul(shr128<4>(mul(xl, nl_lo)), s8));
    U<6> thi = resize<6>(mul(cshr128<4>(mul(xh, nl_hi)), s8));
    // Untrusted float hint.
    double x = ((double)uh * 18446744073709551616.0 + (double)ul + 0.5) * ldexp(1.0, -79) - 1.0;
    double y = ((double)vh * 18446744073709551616.0 + (double)vl + 0.5) * ldexp(1.0, -79) - 1.0;
    double s = x * x + y * y;
    double r = fabs((double)c.sigma * x * sqrt(-2.0 * log(s) / s));
    if (!isfinite(r)) return 2;
    long long h = (long long)floor(r + 0.5);
    long long m = -1;
    for (int d = 0; d < 3; d++) {
        long long cand = d == 0 ? h : (d == 1 ? h - 1 : h + 1);
        if (cand >= 0 && (u64)cand < c.bound && certify(tlo, thi, slo, shi, (u64)cand)) { m = cand; break; }
    }
    if (m < 0) return 2;
    if (m == 0) { k = 0; return 1; }
    if (xs == 0) return 2;
    k = xs > 0 ? m : -m;
    return 1;
}

/// Gate threshold floor(L_k 2^64); returns false if the host must decide.
__device__ __forceinline__ bool gate(const Consts& c, long long k, u64& thr) {
    u64 ak = (u64)(k < 0 ? -k : k);
    if (ak <= c.sigma) { thr = c.gate_small; return true; }
    // c = ceil(k^2 2^64 / (8 sigma^4)); thr = 2^64 - c.
    U<3> D; D.w[0] = c.d4[0]; D.w[1] = c.d4[1]; D.w[2] = c.d4[2];
    U<2> n; n.w[0] = ak * ak; n.w[1] = __umul64hi(ak, ak);
    U<3> N = shl(resize<3>(n), 64);
    double Dd = (double)D.w[0] + ldexp((double)D.w[1], 64) + ldexp((double)D.w[2], 128);
    double est = ceil(ldexp((double)ak * (double)ak, 64) / Dd);
    if (!(est >= 1.0 && est < 4611686018427387904.0)) return false;
    u64 cc = (u64)est;
    for (int it = 0; it < 4; it++) {
        U<4> hi4 = mul(D, small<1>(cc));
        U<4> lo4 = mul(D, small<1>(cc - 1));
        U<4> N4 = resize<4>(N);
        bool up_ok = cmp(N4, hi4) <= 0, lo_ok = cmp(lo4, N4) < 0;
        if (up_ok && lo_ok) { thr = (u64)0 - cc; return true; }
        if (!up_ok) cc++; else cc--;
        if (cc == 0) return false;
    }
    return false;
}

/// One raw trial `tr` for every queued coordinate. Coordinates whose polar
/// pair falls outside the disk are appended to `out_idx` for trial tr+1;
/// all others are finalized (flag 0) or sent to the host (flag 1). Running
/// one trial per launch keeps warps convergent: a warp no longer waits for
/// its unluckiest lane's disk rejections.
__global__ void trial_kernel(const Consts* __restrict__ cp, CoinBase base, uint32_t coord0, uint32_t tr,
                             const uint32_t* __restrict__ in_idx, uint32_t in_n,
                             uint32_t* __restrict__ out_idx, uint32_t* __restrict__ out_n,
                             long long* __restrict__ out, uint32_t* __restrict__ fl_idx, uint32_t* __restrict__ fl_n) {
    __shared__ Consts c;
    for (int i = threadIdx.x; i < (int)(sizeof(Consts) / 8); i += blockDim.x) ((u64*)&c)[i] = ((const u64*)cp)[i];
    __syncthreads();
    for (uint32_t base_t = blockIdx.x * blockDim.x; base_t < in_n; base_t += gridDim.x * blockDim.x) {
        uint32_t t = base_t + threadIdx.x;
        bool active = t < in_n;
        bool again = false;
        if (active) {
            uint32_t i = in_idx ? in_idx[t] : t;
            uint32_t coord = coord0 + i;
            u64 xh, xl, yh, yl;
            coin_block0(base, 10, coord, 0, tr, xh, xl);
            coin_block0(base, 11, coord, 0, tr, yh, yl);
            u64 uh = xh >> 48, ul = (xh << 16) | (xl >> 48);
            u64 vh = yh >> 48, vl = (yh << 16) | (yl >> 48);
            long long k = 0;
            int r = trial(c, uh, ul, vh, vl, k);
            if (r == 0) {
                again = true;
            } else {
                uint8_t f = 1;
                if (r == 1) {
                    u64 thr;
                    if (gate(c, k, thr)) {
                        u64 gh, gl;
                        coin_block0(base, 12, coord, 0, 0, gh, gl);
                        if (gh < thr) f = 0;
                    }
                }
                out[i] = f ? 0 : k;
                if (f) fl_idx[atomicAdd(fl_n, 1u)] = i;
            }
            // Warp-aggregated append of the disk-rejected coordinates.
            unsigned mask = __activemask();
            unsigned vote = __ballot_sync(mask, again);
            int leader = __ffs(mask) - 1;
            uint32_t pos = 0;
            if ((int)(threadIdx.x & 31) == leader && vote) pos = atomicAdd(out_n, __popc(vote));
            pos = __shfl_sync(mask, pos, leader);
            if (again) out_idx[pos + __popc(vote & ((1u << (threadIdx.x & 31)) - 1))] = i;
        }
    }
}

__global__ void flag_rest_kernel(const uint32_t* __restrict__ idx, uint32_t n, long long* out, uint32_t* fl_idx, uint32_t* fl_n) {
    for (uint32_t t = blockIdx.x * blockDim.x + threadIdx.x; t < n; t += gridDim.x * blockDim.x) { out[idx[t]] = 0; fl_idx[atomicAdd(fl_n, 1u)] = idx[t]; }
}

__global__ void scatter_kernel(long long* out, const uint32_t* idx, const long long* val, uint32_t m) {
    for (uint32_t t = blockIdx.x * blockDim.x + threadIdx.x; t < m; t += gridDim.x * blockDim.x) out[idx[t]] = val[t];
}

/// Ternary coins: out[c] ∈ {−1,0,1} from the first accepted two-bit proposal
/// (v < 3 → v − 1) of block 0 at coordinate coord0 + c; flag[c] = 1 if all
/// 256 proposals of block 0 were rejected (the host continues the stream).
__global__ void ternary_kernel(CoinBase base, int purpose, uint32_t coord0, uint32_t n, int8_t* __restrict__ out,
                               uint8_t* __restrict__ flag) {
    for (uint32_t t = blockIdx.x * blockDim.x + threadIdx.x; t < n; t += gridDim.x * blockDim.x) {
        u64 st[25];
#pragma unroll
        for (int i = 0; i < 17; i++) st[i] = base.lane[i];
#pragma unroll
        for (int i = 17; i < 25; i++) st[i] = 0;
        uint32_t coord = coord0 + t;
        put_byte(st, 82, purpose);
        put_byte(st, 93, coord >> 24); put_byte(st, 94, coord >> 16); put_byte(st, 95, coord >> 8); put_byte(st, 96, coord);
        keccak_f(st);
        int8_t v = 0; uint8_t f = 1;
        for (int w = 0; w < 8 && f; w++) {
            // Bytes of lane w in stream order are little-endian bytes of the lane.
            u64 lane = st[w];
            for (int b = 0; b < 8 && f; b++) {
                uint32_t byte = (uint32_t)(lane >> (8 * b)) & 0xff;
                for (int k = 3; k >= 0; k--) {
                    uint32_t pr = (byte >> (2 * k)) & 3;
                    if (pr < 3) { v = (int8_t)pr - 1; f = 0; break; }
                }
            }
        }
        out[t] = v; flag[t] = f;
    }
}

/// Uniform-q coins: trial 0's first 64 bits (big-endian) if < q, else flag.
__global__ void uniform_q_kernel(CoinBase base, int purpose, uint32_t coord0, uint32_t n, u64* __restrict__ out,
                                 uint8_t* __restrict__ flag) {
    for (uint32_t t = blockIdx.x * blockDim.x + threadIdx.x; t < n; t += gridDim.x * blockDim.x) {
        u64 hi, lo;
        coin_block0(base, purpose, coord0 + t, 0, 0, hi, lo);
        const u64 Qv = 18446744069414584289ULL;
        out[t] = hi < Qv ? hi : 0;
        flag[t] = hi < Qv ? 0 : 1;
    }
}

}  // namespace mgauss
