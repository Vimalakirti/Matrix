//! Integration tests for the almost-Goldilocks CUDA crate.
//!
//! Each test is a self-contained correctness check that compares the GPU
//! output against a CPU reference computed with `u128` arithmetic.

use almost_goldilocks_cuda::{
    eq_lagrange, init, partial_eval, sumcheck_prover::GpuSumcheckState,
    AlmostExt2Ops, AlmostGoldilocksExt2, AlmostGoldilocksField, AlmostGoldilocksOps,
    ALMOST_GOLDILOCKS_PRIME, ALMOST_REDUCE_C, ALMOST_HALF_P_PLUS_ONE,
    AEXT2_W, AEXT2_DTH_ROOT,
};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

const P: u128 = ALMOST_GOLDILOCKS_PRIME as u128;

// CPU reference helpers (canonical u64 → canonical u64)
fn r_mul(a: u64, b: u64) -> u64 {
    (((a as u128) * (b as u128)) % P) as u64
}
fn r_add(a: u64, b: u64) -> u64 {
    (((a as u128) + (b as u128)) % P) as u64
}
fn r_sub(a: u64, b: u64) -> u64 {
    if a >= b { a - b } else { ALMOST_GOLDILOCKS_PRIME - (b - a) }
}
fn r_inv(a: u64) -> u64 {
    let mut res = 1u64;
    let mut base = a;
    let mut exp = ALMOST_GOLDILOCKS_PRIME - 2;
    while exp > 0 {
        if exp & 1 == 1 { res = r_mul(res, base); }
        base = r_mul(base, base);
        exp >>= 1;
    }
    res
}

fn canon(v: u64) -> u64 {
    if v >= ALMOST_GOLDILOCKS_PRIME { v - ALMOST_GOLDILOCKS_PRIME } else { v }
}

fn rand_field_vec(rng: &mut StdRng, n: usize) -> Vec<AlmostGoldilocksField> {
    (0..n).map(|_| AlmostGoldilocksField(rng.gen::<u64>() % ALMOST_GOLDILOCKS_PRIME)).collect()
}

fn rand_ext2_vec(rng: &mut StdRng, n: usize) -> Vec<AlmostGoldilocksExt2> {
    (0..n).map(|_| AlmostGoldilocksExt2::new(
        AlmostGoldilocksField(rng.gen::<u64>() % ALMOST_GOLDILOCKS_PRIME),
        AlmostGoldilocksField(rng.gen::<u64>() % ALMOST_GOLDILOCKS_PRIME),
    )).collect()
}

// ============================================================================
// Constants
// ============================================================================

#[test]
fn test_constants() {
    assert_eq!(ALMOST_GOLDILOCKS_PRIME, 0xFFFFFFFEFFFFFFE1);
    assert_eq!(ALMOST_REDUCE_C, 0x10000001F);
    assert_eq!(ALMOST_HALF_P_PLUS_ONE, 0x7FFFFFFF7FFFFFF1);
    assert_eq!(AEXT2_W, 3);
    assert_eq!(AEXT2_DTH_ROOT, ALMOST_GOLDILOCKS_PRIME - 1);
    // sanity: 2 * inv2 ≡ 1
    assert_eq!((2u128 * ALMOST_HALF_P_PLUS_ONE as u128) % P, 1);
    // sanity: P + c = 2^64
    assert_eq!(P + ALMOST_REDUCE_C as u128, 1u128 << 64);
}

// ============================================================================
// Host arithmetic
// ============================================================================

#[test]
fn test_host_arithmetic() {
    let a = AlmostGoldilocksField(12345);
    let b = AlmostGoldilocksField(67890);
    assert_eq!((a + b).reduce().0, 12345 + 67890);
    assert_eq!((b - a).reduce().0, 67890 - 12345);
    assert_eq!((a * b).reduce().0, 12345u64 * 67890);

    // wrap on add
    let near = AlmostGoldilocksField(ALMOST_GOLDILOCKS_PRIME - 1);
    let one = AlmostGoldilocksField(1);
    assert_eq!((near + one).reduce().0, 0);

    // (-1)^2 = 1
    let m1 = AlmostGoldilocksField(ALMOST_GOLDILOCKS_PRIME - 1);
    assert_eq!((m1 * m1).reduce().0, 1);

    // 2^32 * 2^32 = 2^64 ≡ c
    let p32 = AlmostGoldilocksField(1u64 << 32);
    assert_eq!((p32 * p32).reduce().0, ALMOST_REDUCE_C);
}

// ============================================================================
// Field GPU batch ops
// ============================================================================

#[test]
fn test_gpu_field_add_mul() {
    init().expect("CUDA init failed");
    let mut rng = StdRng::seed_from_u64(42);
    let n = 10000;
    let a = rand_field_vec(&mut rng, n);
    let b = rand_field_vec(&mut rng, n);

    let sum = AlmostGoldilocksOps::add(&a, &b).expect("gpu add failed");
    let prod = AlmostGoldilocksOps::mul(&a, &b).expect("gpu mul failed");

    for i in 0..n {
        assert_eq!(canon(sum[i].0), r_add(a[i].0, b[i].0), "add mismatch at {}", i);
        assert_eq!(canon(prod[i].0), r_mul(a[i].0, b[i].0), "mul mismatch at {}", i);
    }
}

#[test]
fn test_gpu_field_inverse() {
    init().expect("CUDA init failed");
    let mut rng = StdRng::seed_from_u64(43);
    let n = 500;
    let mut a = rand_field_vec(&mut rng, n);
    for x in a.iter_mut() { if x.0 == 0 { x.0 = 1; } }

    let inv = AlmostGoldilocksOps::inverse(&a).expect("gpu inverse failed");
    let prod = AlmostGoldilocksOps::mul(&a, &inv).expect("gpu mul failed");
    for i in 0..n {
        assert_eq!(canon(prod[i].0), 1, "a * a^-1 != 1 at {}", i);
    }
}

// ============================================================================
// Ext2 batch ops
// ============================================================================

#[test]
fn test_gpu_ext2_mul() {
    init().expect("CUDA init failed");

    // (1 + 2X)(3 + 4X) with W=3: (3 + 24) + 10X = 27 + 10X
    let a = vec![AlmostGoldilocksExt2::new(AlmostGoldilocksField(1), AlmostGoldilocksField(2))];
    let b = vec![AlmostGoldilocksExt2::new(AlmostGoldilocksField(3), AlmostGoldilocksField(4))];
    let r = AlmostExt2Ops::mul(&a, &b).expect("gpu ext2 mul failed");
    assert_eq!(canon(r[0].c0.0), 27);
    assert_eq!(canon(r[0].c1.0), 10);
}

#[test]
fn test_gpu_ext2_inverse() {
    init().expect("CUDA init failed");
    let mut rng = StdRng::seed_from_u64(44);
    let n = 200;
    let a = rand_ext2_vec(&mut rng, n);

    let inv = AlmostExt2Ops::inverse(&a).expect("gpu ext2 inverse failed");
    let prod = AlmostExt2Ops::mul(&a, &inv).expect("gpu ext2 mul failed");
    for i in 0..n {
        assert_eq!(canon(prod[i].c0.0), 1, "ext2 inverse c0 at {}", i);
        assert_eq!(canon(prod[i].c1.0), 0, "ext2 inverse c1 at {}", i);
    }
}

#[test]
fn test_gpu_ext2_from_base_to_base() {
    init().expect("CUDA init failed");
    let mut rng = StdRng::seed_from_u64(45);
    let base = rand_field_vec(&mut rng, 100);
    let ext = AlmostExt2Ops::from_base(&base).expect("from_base failed");
    for (b, e) in base.iter().zip(ext.iter()) {
        assert_eq!(canon(e.c0.0), b.0);
        assert_eq!(canon(e.c1.0), 0);
    }
    let back = AlmostExt2Ops::to_base(&ext).expect("to_base failed");
    for (b, x) in base.iter().zip(back.iter()) {
        assert_eq!(canon(x.0), b.0);
    }
}

// ============================================================================
// eq_lagrange
// ============================================================================

fn cpu_eq(r: &[AlmostGoldilocksField]) -> Vec<AlmostGoldilocksField> {
    let log_n = r.len();
    let n = 1usize << log_n;
    let mut out = Vec::with_capacity(n);
    for x in 0..n {
        let mut acc = 1u64;
        for i in 0..log_n {
            let bit = (x >> i) & 1;
            if bit == 1 {
                acc = r_mul(acc, r[i].0);
            } else {
                acc = r_mul(acc, r_sub(1, r[i].0));
            }
        }
        out.push(AlmostGoldilocksField(acc));
    }
    out
}

#[test]
fn test_eq_dp_all() {
    init().expect("CUDA init failed");
    let mut rng = StdRng::seed_from_u64(46);
    for log_n in [1, 4, 8, 12] {
        let r = rand_field_vec(&mut rng, log_n);
        let gpu = eq_lagrange::eq_dp_all(&r).expect("gpu eq failed");
        let cpu = cpu_eq(&r);
        assert_eq!(gpu.len(), cpu.len());
        for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            assert_eq!(canon(g.0), canon(c.0), "mismatch at {} for log_n={}", i, log_n);
        }
    }
}

// ============================================================================
// partial_eval
// ============================================================================

fn cpu_partial_eval(evals: &[u64], r: &[u64]) -> Vec<u64> {
    let mut data = evals.to_vec();
    let mut size = data.len();
    for &ri in r {
        let half = size / 2;
        for j in 0..half {
            let a = data[2*j];
            let b = data[2*j + 1];
            let diff = r_sub(b, a);
            data[j] = r_add(a, r_mul(ri, diff));
        }
        size = half;
    }
    data[..size].to_vec()
}

#[test]
fn test_partial_eval_base() {
    init().expect("CUDA init failed");
    let mut rng = StdRng::seed_from_u64(47);
    let log_n = 10;
    let m = 4;
    let evals = rand_field_vec(&mut rng, 1 << log_n);
    let r = rand_field_vec(&mut rng, m);

    let gpu = partial_eval::partial_eval(&evals, &r).expect("gpu partial_eval failed");
    let cpu = cpu_partial_eval(
        &evals.iter().map(|x| canon(x.0)).collect::<Vec<_>>(),
        &r.iter().map(|x| x.0).collect::<Vec<_>>(),
    );
    assert_eq!(gpu.len(), cpu.len());
    for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        assert_eq!(canon(g.0), *c, "mismatch at {}", i);
    }
}

// ============================================================================
// sumcheck
// ============================================================================

#[test]
fn test_sumcheck_base_round_zero() {
    init().expect("CUDA init failed");
    let mut rng = StdRng::seed_from_u64(48);
    let log_n = 8;
    let n = 1usize << log_n;
    let d = 3;

    let polys: Vec<Vec<AlmostGoldilocksField>> = (0..d).map(|_| rand_field_vec(&mut rng, n)).collect();
    let refs: Vec<&[AlmostGoldilocksField]> = polys.iter().map(|p| p.as_slice()).collect();

    let mut state = GpuSumcheckState::new(&refs).expect("state new failed");
    let gpu_msg = state.compute_round_message().expect("round msg failed");
    assert_eq!(gpu_msg.len(), d + 1);

    // CPU reference: for each c ∈ {0..d}, sum_y Π_i (poly_i[2y] + c*(poly_i[2y+1] - poly_i[2y]))
    let half = n / 2;
    let dp1 = d + 1;
    let mut cpu_msg = vec![0u64; dp1];
    for y in 0..half {
        let mut even = vec![0u64; d];
        let mut diff = vec![0u64; d];
        for i in 0..d {
            even[i] = polys[i][2*y].0;
            let odd = polys[i][2*y + 1].0;
            diff[i] = r_sub(odd, even[i]);
        }
        for c in 0..dp1 {
            let cv = c as u64;
            let mut product = 1u64;
            for i in 0..d {
                let val = r_add(even[i], r_mul(cv, diff[i]));
                product = r_mul(product, val);
            }
            cpu_msg[c] = r_add(cpu_msg[c], product);
        }
    }

    for c in 0..dp1 {
        assert_eq!(canon(gpu_msg[c].0), cpu_msg[c], "round msg mismatch at c={}", c);
    }
}

#[test]
fn test_sumcheck_base_full_protocol() {
    init().expect("CUDA init failed");
    let mut rng = StdRng::seed_from_u64(49);
    let log_n = 6;
    let n = 1usize << log_n;
    let d = 2;

    let polys: Vec<Vec<AlmostGoldilocksField>> = (0..d).map(|_| rand_field_vec(&mut rng, n)).collect();
    let refs: Vec<&[AlmostGoldilocksField]> = polys.iter().map(|p| p.as_slice()).collect();

    let mut state = GpuSumcheckState::new(&refs).expect("state new failed");

    // Run all log_n rounds; verify each round message after folding matches CPU.
    let mut cpu_polys: Vec<Vec<u64>> = polys.iter()
        .map(|p| p.iter().map(|x| x.0).collect()).collect();

    for round in 0..log_n {
        let gpu_msg = state.compute_round_message().expect("round msg failed");
        assert_eq!(gpu_msg.len(), d + 1);

        // CPU reference for this round
        let cur_size = n >> round;
        let half = cur_size / 2;
        let dp1 = d + 1;
        let mut cpu_msg = vec![0u64; dp1];
        for y in 0..half {
            let mut even = vec![0u64; d];
            let mut diff = vec![0u64; d];
            for i in 0..d {
                even[i] = cpu_polys[i][2*y];
                let odd = cpu_polys[i][2*y + 1];
                diff[i] = r_sub(odd, even[i]);
            }
            for c in 0..dp1 {
                let cv = c as u64;
                let mut product = 1u64;
                for i in 0..d {
                    let val = r_add(even[i], r_mul(cv, diff[i]));
                    product = r_mul(product, val);
                }
                cpu_msg[c] = r_add(cpu_msg[c], product);
            }
        }
        for c in 0..dp1 {
            assert_eq!(canon(gpu_msg[c].0), cpu_msg[c],
                       "round {} msg mismatch at c={}", round, c);
        }

        // Pick a challenge; fold both GPU and CPU
        let ch = rng.gen::<u64>() % ALMOST_GOLDILOCKS_PRIME;
        state.fold(AlmostGoldilocksField(ch)).expect("fold failed");
        for i in 0..d {
            for y in 0..half {
                let a = cpu_polys[i][2*y];
                let b = cpu_polys[i][2*y + 1];
                cpu_polys[i][y] = r_add(a, r_mul(ch, r_sub(b, a)));
            }
        }
    }

    let gpu_final = state.final_evaluations().expect("final evals failed");
    for i in 0..d {
        assert_eq!(canon(gpu_final[i].0), cpu_polys[i][0],
                   "final eval mismatch at i={}", i);
    }
    let _ = r_inv; // silence dead-code warning when this test is alone
}
