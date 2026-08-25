//! Microbenchmark for the packed-PCS design decisions.
//!
//! Measures the two constants every sizing decision in the new PCS depends on:
//!
//!   1. binary Ajtai commit throughput, and how much the batched path
//!      amortizes the ChaCha8 matrix PRG (`commit_batched` b = 1..16).
//!      The masked-RLC mask commitment `D_l = L(U_l)` is a wide (~35-bit)
//!      vector, committed as ~35 bit-planes, so its cost is
//!      `ceil(35 / 16)` batched calls.
//!
//!   2. the link baseline: one `same_point` sumcheck pass over the same
//!      coefficients. `w` in the sizing model is
//!      (mask ns/coeff) / (link ns/coeff).
//!
//! Run: `./target/release/pcs_microbench [arity_lo] [arity_hi]`

use std::time::Instant;

use almost_goldilocks_cuda::ajtai::{self, RingCommitment, Seed};
use almost_goldilocks_cuda::extension::AlmostGoldilocksExt2;
use almost_goldilocks_cuda::field::AlmostGoldilocksField;

use zk_torch_4::fold::{FoldData, FoldInstance,
    same_point_sumcheck::prove_same_point_gpu_batched};
use zk_torch_4::transcript::Transcript;

fn agl(x: u64) -> AlmostGoldilocksField { AlmostGoldilocksField(x) }
fn ext2(a: u64, b: u64) -> AlmostGoldilocksExt2 { AlmostGoldilocksExt2::new(agl(a), agl(b)) }

/// Deterministic xorshift so runs are reproducible.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        self.0 = x;
        x
    }
}

fn packed_binary(arity: usize, rng: &mut Rng) -> Vec<u64> {
    (0..(1usize << (arity - 6))).map(|_| rng.next()).collect()
}


// ===================================================================
// Phase 1 verification + timing for the wide-commit kernel
// ===================================================================

fn ring_eq(a: &RingCommitment, b: &RingCommitment) -> bool {
    for i in 0..ajtai::KAPPA {
        for r in 0..ajtai::RING_DIM {
            if a.rows[i][r] != b.rows[i][r] { return false; }
        }
    }
    true
}

fn ring_add(a: &RingCommitment, b: &RingCommitment) -> RingCommitment {
    let q: u128 = ((1u128 << 64) - (1u128 << 32) + 1) - 32;
    let mut out = RingCommitment::zero();
    for i in 0..ajtai::KAPPA {
        for r in 0..ajtai::RING_DIM {
            let s = (a.rows[i][r] as u128 + b.rows[i][r] as u128) % q;
            out.rows[i][r] = s as u64;
        }
    }
    out
}

/// Canonicalize each row so two congruent-but-non-canonical commitments compare
/// equal. The kernels return values reduced mod q by `agl_canonicalize`, but the
/// host-side ring_add above works mod q too, so both sides are canonical.
fn canon(c: &RingCommitment) -> RingCommitment {
    let q: u128 = ((1u128 << 64) - (1u128 << 32) + 1) - 32;
    let mut out = RingCommitment::zero();
    for i in 0..ajtai::KAPPA {
        for r in 0..ajtai::RING_DIM {
            out.rows[i][r] = ((c.rows[i][r] as u128) % q) as u64;
        }
    }
    out
}

fn phase1_verify(seed: Seed, rng: &mut Rng) -> bool {
    let mut ok = true;

    // (a) parity with the binary kernel: a wide commit of a 0/1 witness must
    //     be bit-identical to commit_batched on the packed bits.
    for arity in [12usize, 16, 20] {
        let packed = packed_binary(arity, rng);
        let mut wide = vec![0u64; 1usize << arity];
        for (j, word) in packed.iter().enumerate() {
            for l in 0..64 {
                wide[j * 64 + l] = (word >> l) & 1;
            }
        }
        let c_bin = ajtai::commit_batched(seed, &[packed.as_slice()], None).expect("bin")[0].clone();
        let c_wide = ajtai::commit_wide(seed, &wide, 0, None).expect("wide");
        let good = ring_eq(&canon(&c_bin), &canon(&c_wide));
        println!("  [{}] binary parity @ arity {:>2}", if good { "ok" } else { "FAIL" }, arity);
        ok &= good;
    }

    // (b) column-offset linearity: committing a 2-block witness in one call must
    //     equal the ring-sum of each block committed at its own column window.
    //     This is exactly what packing relies on.
    {
        let arity = 16usize;
        let half = 1usize << (arity - 1);
        let full: Vec<u64> = (0..(1usize << arity)).map(|_| rng.next() >> 20).collect();
        let c_full = ajtai::commit_wide(seed, &full, 0, None).expect("full");
        let c_lo = ajtai::commit_wide(seed, &full[..half], 0, None).expect("lo");
        let c_hi = ajtai::commit_wide(seed, &full[half..], (half / 64) as u64, None).expect("hi");
        let good = ring_eq(&canon(&c_full), &canon(&ring_add(&c_lo, &c_hi)));
        println!("  [{}] column-offset linearity (packing primitive)",
                 if good { "ok" } else { "FAIL" });
        ok &= good;
    }

    // (b2) the same identity on the BINARY path: a packed commitment must
    //      equal the ring-sum of its blocks committed at their own windows.
    //      This is the packing primitive the layout module depends on.
    {
        let arity = 16usize;
        let half_words = 1usize << (arity - 1 - 6);
        let full = packed_binary(arity, rng);
        let c_full = ajtai::commit_batched(seed, &[full.as_slice()], None).expect("full")[0].clone();
        let c_lo = ajtai::commit_batched_at(seed, &[&full[..half_words]], 0, None).expect("lo")[0].clone();
        let c_hi = ajtai::commit_batched_at(
            seed, &[&full[half_words..]], half_words as u64, None).expect("hi")[0].clone();
        let good = ring_eq(&canon(&c_full), &canon(&ring_add(&c_lo, &c_hi)));
        println!("  [{}] column-offset linearity, BINARY path",
                 if good { "ok" } else { "FAIL" });
        ok &= good;
    }

    // (b3) sparse path: shifting every position by col_offset*RING_DIM moves the
    //      column window, so sparse leaves pack with no kernel change at all.
    {
        let base_arity = 14usize;
        let shift_rings = 1u64 << (base_arity - 6);
        let positions: Vec<u64> = (0..64u64).map(|i| (i * 137) % (1 << base_arity)).collect();
        let shifted: Vec<u64> = positions.iter()
            .map(|p| p + shift_rings * (ajtai::RING_DIM as u64)).collect();
        // dense equivalent of the shifted sparse witness, committed at offset
        let mut dense = vec![0u64; 1usize << base_arity];
        for p in &positions { dense[*p as usize] = 1; }
        let c_sparse = ajtai::commit_sparse(seed, &shifted, None).expect("sparse");
        let c_dense = ajtai::commit_wide(seed, &dense, shift_rings, None).expect("dense@off");
        let good = ring_eq(&canon(&c_sparse), &canon(&c_dense));
        println!("  [{}] sparse position-shift == column window",
                 if good { "ok" } else { "FAIL" });
        ok &= good;
    }

    // (c) additive homomorphism over wide values: L(x) + L(y) == L(x+y).
    {
        let arity = 14usize;
        let n = 1usize << arity;
        let q: u128 = ((1u128 << 64) - (1u128 << 32) + 1) - 32;
        let x: Vec<u64> = (0..n).map(|_| rng.next() >> 20).collect();
        let y: Vec<u64> = (0..n).map(|_| rng.next() >> 20).collect();
        let sum: Vec<u64> = x.iter().zip(&y)
            .map(|(a, b)| (((*a as u128) + (*b as u128)) % q) as u64).collect();
        let cx = ajtai::commit_wide(seed, &x, 0, None).expect("x");
        let cy = ajtai::commit_wide(seed, &y, 0, None).expect("y");
        let cs = ajtai::commit_wide(seed, &sum, 0, None).expect("sum");
        let good = ring_eq(&canon(&cs), &canon(&ring_add(&cx, &cy)));
        println!("  [{}] additive homomorphism on wide coefficients",
                 if good { "ok" } else { "FAIL" });
        ok &= good;
    }

    ok
}


/// Cost of the link's degree-4 round polynomial against a degree-2 same-point
/// round, measured on identical CPU tables. The GPU same-point kernel is the
/// projection's baseline, so this ratio is what converts it into a link estimate.
fn link_degree_cost(rng: &mut Rng) {
    use zk_torch_4::pcs::link::{prove_link, LinkQuery, LinkWitness};
    use zk_torch_4::transcript::Transcript;
    use almost_goldilocks_cuda::extension::AlmostGoldilocksExt2 as Ext2;

    println!("--- link prover cost (CPU reference) ---");
    println!("{:>6} {:>5} {:>12} {:>14}", "arity", "|I|", "total ms", "ns/coeff");
    for (arity, ncommit) in [(12usize, 16usize), (14, 16), (16, 8)] {
        let ws: Vec<LinkWitness> = (0..ncommit)
            .map(|_| LinkWitness::dense((0..(1usize << arity)).map(|_| rng.next() & 1).collect()))
            .collect();
        // one claim per commitment, at a full-domain point
        let qs: Vec<LinkQuery> = ws.iter().enumerate().map(|(e, w)| {
            let point: Vec<Ext2> = (0..arity).map(|_| ext2(rng.next() >> 4, rng.next() >> 4)).collect();
            let value = zk_torch_4::pcs::link::mle_eval_for_bench(&w.coeffs, &point);
            LinkQuery { commitment: e, point, value, prefix_len: 0 }
        }).collect();

        let t = Instant::now();
        let mut tr = Transcript::new(b"bench");
        let _ = prove_link(&ws, &qs, arity, 1, &mut tr);
        let secs = t.elapsed().as_secs_f64();
        let coeffs = (ncommit as f64) * ((1u64 << arity) as f64);
        println!("{:>6} {:>5} {:>12.3} {:>14.2}", arity, ncommit, secs * 1e3, secs * 1e9 / coeffs);
    }
    println!("--- GPU link prover (full sumcheck, device-resident) ---");
    println!("{:>6} {:>6} {:>10} {:>14} {:>12}", "arity", "|I|", "spacing", "ns/coeff", "density");
    for (arity, ncommit) in [(16usize, 16usize), (18, 16), (20, 8)] {
        for spacing in [1usize, 16, 256] {
            match zk_torch_4::pcs::link::bench_gpu_round(ncommit, arity, spacing) {
                Some(ns) => println!("{:>6} {:>6} {:>10} {:>14.4} {:>12.4}",
                                     arity, ncommit, spacing, ns, 1.0 / spacing as f64),
                None => println!("{:>6} {:>6} {:>10}   (gpu path unavailable)", arity, ncommit, spacing),
            }
        }
    }
    println!();

    println!("--- support sensitivity: round cost vs witness density ---");
    println!("{:>8} {:>14} {:>10}", "density", "ns/coeff", "vs dense");
    let dense_ref = zk_torch_4::pcs::link::bench_round_density(8, 18, 1.0);
    for d in [1.0f64, 0.5, 0.25, 0.1, 0.01, 0.001] {
        let ns = zk_torch_4::pcs::link::bench_round_density(8, 18, d);
        println!("{:>8.3} {:>14.3} {:>9.2}x", d, ns, dense_ref / ns);
    }
    println!();

    println!("--- degree penalty: link round (deg 4) vs same-point round (deg 2) ---");
    println!("{:>6} {:>5} {:>12} {:>12} {:>8}", "arity", "|I|", "deg4 ns", "deg2 ns", "ratio");
    for (arity, ncommit) in [(14usize, 16usize), (16, 16), (18, 8)] {
        let (full, evalonly) = zk_torch_4::pcs::link::bench_round_halves(ncommit, arity);
        println!("{:>6} {:>5} {:>12.3} {:>12.3} {:>7.2}x", arity, ncommit, full, evalonly, full / evalonly);
    }
    println!();
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let lo: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(20);
    let hi: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(24);
    let reps: usize = std::env::var("REPS").ok().and_then(|s| s.parse().ok()).unwrap_or(3);

    println!("kappa = {}, ring_dim = {}", ajtai::KAPPA, ajtai::RING_DIM);
    println!("reps  = {} (median reported)\n", reps);

    let seed = Seed([1, 2, 3, 4, 5, 6, 7, 8]);
    let mut rng = Rng(0x243F6A8885A308D3);


    if std::env::var("LINK_ONLY").is_ok() { link_degree_cost(&mut rng); return; }

    // --- Phase 1: wide-commit kernel ---
    println!("--- wide-commit kernel: correctness ---");
    let all_ok = phase1_verify(seed, &mut rng);
    println!("  => {}\n", if all_ok { "ALL PASS" } else { "FAILURES PRESENT" });

    println!("--- wide-commit kernel: throughput (one pass, {} planes replaced) ---", 36);
    println!("{:>6} {:>12} {:>14} {:>16}", "arity", "total ms", "ns/coeff", "vs 36-plane");
    let mut widek_ns: Vec<(usize, f64)> = Vec::new();
    for arity in lo..=hi {
        let z: Vec<u64> = (0..(1usize << arity)).map(|_| rng.next() >> 20).collect();
        let _ = ajtai::commit_wide(seed, &z, 0, None).expect("warm");
        let mut ts = Vec::new();
        for _ in 0..reps {
            let t = Instant::now();
            let _c = ajtai::commit_wide(seed, &z, 0, None).expect("wide");
            ts.push(t.elapsed().as_secs_f64());
        }
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let secs = ts[ts.len() / 2];
        let ns = secs * 1e9 / ((1u64 << arity) as f64);
        widek_ns.push((arity, ns));
        println!("{:>6} {:>12.3} {:>14.4} {:>16}", arity, secs * 1e3, ns, "-");
    }
    println!();
    // ---------------------------------------------------------------
    // 1. binary commit: batch scaling (PRG amortization)
    // ---------------------------------------------------------------
    println!("--- binary Ajtai commit (commit_batched) ---");
    println!("{:>6} {:>4} {:>12} {:>14} {:>12}",
             "arity", "b", "total ms", "ns/coeff", "amort vs b=1");
    let mut ns_per_coeff_b16: Vec<(usize, f64)> = Vec::new();
    for arity in lo..=hi {
        let mut base = f64::NAN;
        for b in [1usize, 4, 16] {
            let wits: Vec<Vec<u64>> = (0..b).map(|_| packed_binary(arity, &mut rng)).collect();
            let refs: Vec<&[u64]> = wits.iter().map(|w| w.as_slice()).collect();
            // warm up
            let _ = ajtai::commit_batched(seed, &refs, None).expect("commit");
            let mut times = Vec::new();
            for _ in 0..reps {
                let t = Instant::now();
                let _c = ajtai::commit_batched(seed, &refs, None).expect("commit");
                times.push(t.elapsed().as_secs_f64());
            }
            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let secs = times[times.len() / 2];
            let coeffs = (b as f64) * ((1u64 << arity) as f64);
            let ns = secs * 1e9 / coeffs;
            if b == 1 { base = ns; }
            if b == 16 { ns_per_coeff_b16.push((arity, ns)); }
            println!("{:>6} {:>4} {:>12.3} {:>14.4} {:>11.2}x",
                     arity, b, secs * 1e3, ns, base / ns);
        }
    }

    // ---------------------------------------------------------------
    // 2a. wide commit, on-the-fly PRG: 36 single-witness commits
    // ---------------------------------------------------------------
    // B_Z ~ 3.4e10 at gamma=32 => 36 bits incl. sign.
    const WIDE_BITS: usize = 36;
    println!("\n--- wide commit D_l = L(U_l), {} planes, on-the-fly PRG (b=1) ---", WIDE_BITS);
    println!("{:>6} {:>12} {:>14}", "arity", "total ms", "ns/coeff");
    let mut wide_ns: Vec<(usize, f64)> = Vec::new();
    for arity in lo..=hi {
        let w = packed_binary(arity, &mut rng);
        let refs: Vec<&[u64]> = vec![w.as_slice()];
        let _ = ajtai::commit_batched(seed, &refs, None).expect("commit");
        let t = Instant::now();
        for _ in 0..WIDE_BITS {
            let _c = ajtai::commit_batched(seed, &refs, None).expect("commit");
        }
        let secs = t.elapsed().as_secs_f64();
        let ns = secs * 1e9 / ((1u64 << arity) as f64);
        wide_ns.push((arity, ns));
        println!("{:>6} {:>12.3} {:>14.4}", arity, secs * 1e3, ns);
    }

    // ---------------------------------------------------------------
    // 2b. wide commit, pre-materialized M (PRG paid once): 13 planes/call
    // ---------------------------------------------------------------
    println!("\n--- wide commit with pre-materialized M (13 planes/call) ---");
    println!("{:>6} {:>10} {:>12} {:>12} {:>14}",
             "arity", "M GiB", "matz ms", "36pl ms", "ns/coeff");
    let mut wide_premat_ns: Vec<(usize, f64)> = Vec::new();
    for arity in lo..=hi {
        let n_ring = 1usize << (arity - 6);
        let gib = (n_ring * ajtai::KAPPA * ajtai::RING_DIM * 8) as f64 / (1024.0 * 1024.0 * 1024.0);
        if gib > 40.0 { println!("{:>6} {:>10.2} {:>12} skipped (M too large)", arity, gib, "-"); continue; }

        let t = Instant::now();
        let mat = match ajtai::MaterializedM::new(seed, n_ring) {
            Ok(m) => m,
            Err(e) => { println!("{:>6} {:>10.2}  materialize failed: {:?}", arity, gib, e); continue; }
        };
        let matz_ms = t.elapsed().as_secs_f64() * 1e3;

        let wide: Vec<i16> = (0..(1usize << arity)).map(|_| (rng.next() % 8191) as i16).collect();
        let d_wide = match almost_goldilocks_cuda::memory::DeviceBuffer::<i16>::from_slice(&wide) {
            Ok(b) => b, Err(e) => { println!("  upload failed {:?}", e); continue; }
        };
        let chunks = match ajtai::split_witness_device(&d_wide) {
            Ok(c) => c, Err(e) => { println!("  split failed {:?}", e); continue; }
        };

        let _ = ajtai::commit_ternary_premat(&mat, &chunks, None);
        let calls = (WIDE_BITS + 12) / 13;   // 13 planes per call
        let t = Instant::now();
        for _ in 0..calls {
            let _c = ajtai::commit_ternary_premat(&mat, &chunks, None).expect("premat commit");
        }
        let secs = t.elapsed().as_secs_f64();
        let ns = secs * 1e9 / ((1u64 << arity) as f64);
        wide_premat_ns.push((arity, ns));
        println!("{:>6} {:>10.2} {:>12.1} {:>12.1} {:>14.4}",
                 arity, gib, matz_ms, secs * 1e3, ns);
    }

    // ---------------------------------------------------------------
    // 3. link baseline: one same_point sumcheck pass
    // ---------------------------------------------------------------
    println!("\n--- link baseline (same_point sumcheck GPU, M instances @ arity) ---");
    println!("{:>6} {:>5} {:>12} {:>14}", "arity", "M", "total ms", "ns/coeff");
    let mut link_ns: Vec<(usize, f64)> = Vec::new();
    for arity in lo..=hi {
        let m: usize = if arity >= 23 { 8 } else { 32 };
        let insts: Vec<FoldInstance> = (0..m).map(|_| {
            let data = FoldData::Binary(packed_binary(arity, &mut rng));
            let pt: Vec<AlmostGoldilocksExt2> =
                (0..arity).map(|_| ext2(rng.next() >> 3, rng.next() >> 3)).collect();
            let val = data.evaluate_at_ext2(&pt);
            FoldInstance {
                commitment: RingCommitment::zero(),
                data,
                arity,
                claim_pt: pt,
                claim_val: val,
            }
        }).collect();

        let mut t0 = Transcript::new(b"micro-warm");
        let _ = prove_same_point_gpu_batched(&insts, arity, &mut t0);

        let t = Instant::now();
        let mut tr = Transcript::new(b"micro");
        let _ = prove_same_point_gpu_batched(&insts, arity, &mut tr);
        let secs = t.elapsed().as_secs_f64();

        let coeffs = (m as f64) * ((1u64 << arity) as f64);
        let ns = secs * 1e9 / coeffs;
        link_ns.push((arity, ns));
        println!("{:>6} {:>5} {:>12.3} {:>14.4}", arity, m, secs * 1e3, ns);
    }

    // ---------------------------------------------------------------
    // 4. w = mask cost / link cost
    // ---------------------------------------------------------------
    println!("\n--- w = (wide commit ns/coeff) / (link ns/coeff) ---");
    println!("{:>6} {:>12} {:>12} {:>12} {:>9} {:>9}",
             "arity", "wide(prg)", "wide(prem)", "link", "w prg", "w prem");
    for (a, wns) in &wide_ns {
        let lns = match link_ns.iter().find(|(x, _)| x == a) { Some((_, l)) => *l, None => continue };
        let pns = wide_premat_ns.iter().find(|(x, _)| x == a).map(|(_, p)| *p);
        match pns {
            Some(p) => println!("{:>6} {:>12.3} {:>12.3} {:>12.3} {:>9.2} {:>9.2}",
                                a, wns, p, lns, wns / lns, p / lns),
            None    => println!("{:>6} {:>12.3} {:>12} {:>12.3} {:>9.2} {:>9}",
                                a, wns, "-", lns, wns / lns, "-"),
        }
    }
}
