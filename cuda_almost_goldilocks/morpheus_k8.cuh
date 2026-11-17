// K = F_q[Θ]/(Θ^8 − 3) multiplication with lazy reduction.
// All 64 cross products are accumulated in 192-bit sums per output
// coefficient (low part and the Θ^8-wrapped part separately), then reduced
// once: c_k = s1_k + 3·s3_k mod q. Inputs may be non-canonical (< 2^64);
// the output is canonical.
#pragma once
#include <cstdint>
#include "almost_goldilocks.cuh"

namespace mk8 {
typedef unsigned long long u64;
constexpr u64 Q = 18446744069414584289ULL;

struct U192 { u64 w0, w1, w2; };

__device__ __forceinline__ void acc_mul(U192& s, u64 a, u64 b) {
    u64 lo = a * b, hi = __umul64hi(a, b);
    u64 n0 = s.w0 + lo; u64 c0 = n0 < lo;
    u64 t1 = s.w1 + hi; u64 c1 = t1 < hi;
    u64 n1 = t1 + c0; c1 += n1 < c0;
    s.w0 = n0; s.w1 = n1; s.w2 += c1;
}

/// x0 + x1·2^64 + x2·2^128 mod q, x2 < 2^16.
__device__ __forceinline__ u64 reduce192(const U192& x) {
    agl_uint128_t lohi; lohi.lo = x.w0; lohi.hi = x.w1;
    u64 r = agl_canonicalize(agl_reduce128(lohi));
    // 2^128 ≡ (2^32 + 31)^2 ≡ 63·2^32 + 992 (mod q).
    u64 t = x.w2 * (63ULL * (1ULL << 32) + 992ULL);  // < 2^54
    u64 s = r + t;
    if (s < r || s >= Q) s -= Q;
    return s;
}

/// out = a·b in K.
__device__ __forceinline__ void mul(const u64* a, const u64* b, u64* out) {
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        U192 s1 = {0, 0, 0}, s3 = {0, 0, 0};
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int j = k - i;
            if (j >= 0) acc_mul(s1, a[i], b[j]);
            else acc_mul(s3, a[i], b[j + 8]);
        }
        // s1 + 3·s3 (≤ 4·8·2^128 < 2^133): fold 3·s3 into s1.
        U192 t = s3;
        U192 t2 = {t.w0 << 1, (t.w1 << 1) | (t.w0 >> 63), (t.w2 << 1) | (t.w1 >> 63)};
        // s1 += t + t2
        u64 n0 = s1.w0 + t.w0; u64 c = n0 < t.w0;
        u64 n1 = s1.w1 + t.w1; u64 c1 = n1 < t.w1; n1 += c; c1 += n1 < c;
        u64 n2 = s1.w2 + t.w2 + c1;
        u64 m0 = n0 + t2.w0; c = m0 < t2.w0;
        u64 m1 = n1 + t2.w1; c1 = m1 < t2.w1; m1 += c; c1 += m1 < c;
        u64 m2 = n2 + t2.w2 + c1;
        U192 r = {m0, m1, m2};
        out[k] = reduce192(r);
    }
}
}  // namespace mk8
