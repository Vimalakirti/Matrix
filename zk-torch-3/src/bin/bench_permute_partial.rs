//! Microbenchmark for the permute + partial_eval pipeline that dominates
//! Einsum prove time. For each LLaMA-like shape we measure in isolation:
//!   - CPU permute_evals_by_ranges
//!   - GPU partial_eval_ext2 (upload + kernel + download)
//!   - GPU fused_permute_partial_eval (current production path)
//!   - GPU partial_eval_ext2_device_u64 split: upload, kernel, download
//!   - CPU partial_eval_ext2_cpu (reference)
//!
//! The point is to confirm which sub-step inside Einsum's 93-98% "overhead"
//! is actually expensive and therefore worth accelerating.
//!
//! Usage:
//!     cargo run --release --bin bench_permute_partial

use std::time::Instant;

use goldilocks_cuda::{GoldilocksField, GoldilocksExt2, DeviceBuffer};
use goldilocks_cuda::partial_eval::{
    partial_eval_ext2, partial_eval_ext2_device_u64, fused_permute_partial_eval,
};
use rand::Rng;

use zk_torch_3::basicblock::einsum::{permute_evals_by_ranges, partial_eval_ext2_cpu};

fn random_field_vec(size: usize) -> Vec<GoldilocksField> {
    let mut rng = rand::thread_rng();
    (0..size)
        .map(|_| GoldilocksField(rng.gen::<u64>() % (1u64 << 32)))
        .collect()
}

fn random_ext2_vec(size: usize) -> Vec<GoldilocksExt2> {
    let mut rng = rand::thread_rng();
    (0..size)
        .map(|_| GoldilocksExt2::new(
            GoldilocksField(rng.gen::<u64>() % (1u64 << 63)),
            GoldilocksField(rng.gen::<u64>() % (1u64 << 63)),
        ))
        .collect()
}

fn log2c(x: usize) -> usize {
    if x <= 1 { 0 } else { (x as f64).log2().ceil() as usize }
}

fn ms(t: Instant) -> f64 { t.elapsed().as_secs_f64() * 1000.0 }

/// Time the GPU partial_eval_ext2 by broken-out steps.
/// Two timings: production-style (Vec<u64>::collect + upload) vs direct API.
fn bench_gpu_partial_eval_split(
    permuted: &[GoldilocksField],
    r: &[GoldilocksExt2],
    n: usize,
) -> (f64, f64, f64, f64) {
    // (A) Production fallback path: Vec<u64> collect, then upload, kernel, download.
    let t_alloc = Instant::now();
    let permuted_u64: Vec<u64> = permuted.iter().map(|v| v.0).collect();
    let alloc_ms = ms(t_alloc);

    let t_up = Instant::now();
    let d_input = DeviceBuffer::<u64>::from_slice(&permuted_u64).unwrap();
    let d_r = DeviceBuffer::<GoldilocksExt2>::from_slice(r).unwrap();
    let output_half = 1usize << (n - 1);
    let mut d_output = DeviceBuffer::<GoldilocksExt2>::new(output_half).unwrap();
    goldilocks_cuda::memory::synchronize().unwrap();
    let upload_ms = ms(t_up);

    let t_ker = Instant::now();
    partial_eval_ext2_device_u64(&d_input, &mut d_output, &d_r, n, r.len()).unwrap();
    goldilocks_cuda::memory::synchronize().unwrap();
    let kernel_ms = ms(t_ker);

    let t_dn = Instant::now();
    let result_len = 1usize << (n - r.len());
    let _: Vec<GoldilocksExt2> = d_output.read_slice(0, result_len).unwrap();
    let download_ms = ms(t_dn);

    (alloc_ms, upload_ms, kernel_ms, download_ms)
}

struct Config {
    label: String,
    dim_i: usize,
    dim_j: usize,
    dim_k: usize,
    /// Which input are we exercising: "A" (shape ij, permute for i free, j summ)
    /// or "B" (shape jk, permute for k free, j summ).
    input: &'static str,
}

fn bench_config(cfg: &Config, trials: usize) {
    let i_bits = log2c(cfg.dim_i);
    let j_bits = log2c(cfg.dim_j);
    let k_bits = log2c(cfg.dim_k);

    let (witness, n, m, permute_vec, label) = match cfg.input {
        "A" => {
            // Input A shape (i, j): variables laid out j_bits low, i_bits high (shape==[i,j], n-1..0 ordering).
            // For "ij,jk->ik": free_once = [i,k], summation=[j]; all_indices=[i,k,j].
            // permute_vec for A (spec "ij") keeps indices that appear in spec from all_indices order:
            //   - i is at bit range [j_bits, j_bits+i_bits)
            //   - j is at bit range [0, j_bits)
            // So permute_vec = [(j_bits, j_bits+i_bits), (0, j_bits)].
            let n_a = i_bits + j_bits;
            let w = random_field_vec(1 << n_a);
            (w, n_a, i_bits,
             vec![(j_bits, j_bits + i_bits), (0, j_bits)],
             format!("{} A({}x{})", cfg.label, cfg.dim_i, cfg.dim_j))
        }
        "B" => {
            // Input B shape (j, k): variables layout k_bits low, j_bits high.
            // permute_vec for B (spec "jk"): keep [k, j] from [i,k,j]:
            //   - k is at bit range [0, k_bits)
            //   - j is at bit range [k_bits, k_bits+j_bits)
            // So permute_vec = [(0, k_bits), (k_bits, k_bits + j_bits)]? No — all_indices puts k before j,
            // and we push range of whichever index is found. Each pushed range is the CURRENT bit location.
            // For B the natural bit layout (before permute) is j high, k low. So:
            //   Looking up index "k" → range (0, k_bits)
            //   Looking up index "j" → range (k_bits, k_bits + j_bits)
            // permute_vec = [(0, k_bits), (k_bits, k_bits + j_bits)].
            // This is identity! That means no permute is needed for B. Hmm.
            //
            // But wait — Einsum indexing convention. Looking at einsum.rs: `char_to_range(spec, shape)`
            // builds ranges. Need to check its ordering. Typically for shape [j, k] with spec "jk",
            // the last dim (k) is at low bits (row-major with low bit = last dim).
            // So k -> (0, k_bits), j -> (k_bits, k_bits+j_bits). Yes.
            //
            // Then all_indices = [i, k, j]. Pushing for those in B: k then j → [(0, k_bits), (k_bits, k_bits+j_bits)].
            // That's identity. Hmm. Then there'd be no permute cost. Let me re-examine.
            //
            // Actually the Einsum puts summation (j) LAST in all_indices, so after permute, j occupies
            // the HIGH bits and the free/output indices occupy the LOW bits. Partial_eval on Ext2 fixes
            // the LOW m variables (reading pairs at stride 1, fold). free_once gives m = k_bits.
            // So for B we want k at low bits. Layout is k at low bits already → identity for B!
            //
            // For A (spec "ij", shape [i, j]): j at low bits, i at high bits. We want k (not present)
            // and j last. all_indices= [i, k, j]. Pushing for A: i, then j. Result: [(i's range), (j's range)].
            // But A has i at high (j_bits..j_bits+i_bits), j at low (0..j_bits). So we push
            // (j_bits, j_bits+i_bits), (0, j_bits) → NON-identity, swap halves!
            //
            // Partial eval fixes the low m = i_bits variables = index i → free for A → matches.
            let n_b = j_bits + k_bits;
            let w = random_field_vec(1 << n_b);
            (w, n_b, k_bits,
             vec![(0, k_bits), (k_bits, k_bits + j_bits)],
             format!("{} B({}x{})", cfg.label, cfg.dim_j, cfg.dim_k))
        }
        _ => unreachable!(),
    };

    if m == 0 {
        println!("\n  {} [n={}, m={}]   -- skip (no partial_eval vars)", label, n, m);
        return;
    }
    let challenges = random_ext2_vec(m);

    // --- 1. CPU permute alone (median of trials) ---
    let mut t_perm = Vec::with_capacity(trials);
    for _ in 0..trials {
        let t = Instant::now();
        let _ = permute_evals_by_ranges(&witness, n, &permute_vec);
        t_perm.push(ms(t));
    }
    t_perm.sort_by(|a,b| a.partial_cmp(b).unwrap());
    let cpu_permute_ms = t_perm[trials/2];

    let permuted = permute_evals_by_ranges(&witness, n, &permute_vec);

    // --- 2. GPU partial_eval_ext2 (convenience API: upload+kernel+download) ---
    let mut t_pe = Vec::with_capacity(trials);
    for _ in 0..trials {
        let t = Instant::now();
        let _ = partial_eval_ext2(&permuted, &challenges).unwrap();
        t_pe.push(ms(t));
    }
    t_pe.sort_by(|a,b| a.partial_cmp(b).unwrap());
    let gpu_pe_ms = t_pe[trials/2];

    // --- 3. GPU partial_eval split into alloc/upload/kernel/download ---
    let mut t_alloc = Vec::new();
    let mut t_up = Vec::new();
    let mut t_ker = Vec::new();
    let mut t_dn = Vec::new();
    for _ in 0..trials {
        let (a, u, k, d) = bench_gpu_partial_eval_split(&permuted, &challenges, n);
        t_alloc.push(a); t_up.push(u); t_ker.push(k); t_dn.push(d);
    }
    t_alloc.sort_by(|a,b| a.partial_cmp(b).unwrap());
    t_up.sort_by(|a,b| a.partial_cmp(b).unwrap());
    t_ker.sort_by(|a,b| a.partial_cmp(b).unwrap());
    t_dn.sort_by(|a,b| a.partial_cmp(b).unwrap());

    // --- 4. GPU fused_permute_partial_eval ---
    let mut t_fused = Vec::with_capacity(trials);
    if n <= 28 {
        for _ in 0..trials {
            let t = Instant::now();
            let _ = fused_permute_partial_eval(&witness, &challenges, &permute_vec, n).unwrap();
            t_fused.push(ms(t));
        }
        t_fused.sort_by(|a,b| a.partial_cmp(b).unwrap());
    }

    // --- 5. CPU partial_eval for baseline (skip if huge) ---
    let cpu_pe_ms = if n <= 22 {
        let t = Instant::now();
        let _ = partial_eval_ext2_cpu(&permuted, &challenges);
        ms(t)
    } else {
        f64::NAN
    };

    println!("\n  {} [n={}, m={}]", label, n, m);
    println!("    CPU permute:              {:>8.3} ms", cpu_permute_ms);
    println!("    GPU partial_eval (fast API): {:>8.3} ms   (DeviceBuffer::from_slice<GL> direct)", gpu_pe_ms);
    println!("    GPU PE prod-path split:      alloc_u64 {:.3}  upload {:.3}  kernel {:.3}  download {:.3}  ms",
             t_alloc[trials/2], t_up[trials/2], t_ker[trials/2], t_dn[trials/2]);
    let prod_path_ms = cpu_permute_ms + t_alloc[trials/2] + t_up[trials/2] + t_ker[trials/2] + t_dn[trials/2];
    println!("    CPU permute + GPU PE total:  {:>8.3} ms   <-- einsum fallback path (n>28)",
             prod_path_ms);
    if !t_fused.is_empty() {
        println!("    GPU fused permute+PE:     {:>8.3} ms   <-- production path (n>16, n<=28)",
                 t_fused[trials/2]);
        let speedup = prod_path_ms / t_fused[trials/2];
        println!("                               (fused vs prod-split: {:.2}x speedup)", speedup);
    }
    println!("    CPU partial_eval:         {:>8.3} ms   (reference)", cpu_pe_ms);
}

fn main() {
    goldilocks_cuda::init().expect("CUDA init failed");

    // Warmup
    {
        let w = random_field_vec(1 << 20);
        let r = random_ext2_vec(10);
        let _ = partial_eval_ext2(&w, &r).unwrap();
    }

    let trials = 5;

    println!("=== Permute + partial_eval microbenchmark (median of {} trials) ===", trials);
    println!("All shapes below correspond to Einsum 'ij,jk->ik' matmul inputs from LLaMA.");

    // LLaMA-like configs: 1x4096x4096 (QKV proj), 1x4096x11008 (FFN gate/up),
    // 1x11008x4096 (FFN down), plus 32x1x128 attn shapes.
    let configs = vec![
        Config { label: "QKV     ".into(), dim_i: 1,  dim_j: 4096,  dim_k: 4096,  input: "B" },
        Config { label: "QKV     ".into(), dim_i: 1,  dim_j: 4096,  dim_k: 4096,  input: "A" },
        Config { label: "FFN up  ".into(), dim_i: 1,  dim_j: 4096,  dim_k: 11008, input: "B" },
        Config { label: "FFN up  ".into(), dim_i: 1,  dim_j: 4096,  dim_k: 11008, input: "A" },
        Config { label: "FFN down".into(), dim_i: 1,  dim_j: 11008, dim_k: 4096,  input: "B" },
        Config { label: "FFN down".into(), dim_i: 1,  dim_j: 11008, dim_k: 4096,  input: "A" },
        Config { label: "AttnQK  ".into(), dim_i: 32, dim_j: 1,     dim_k: 128,   input: "B" },
        Config { label: "AttnSV  ".into(), dim_i: 32, dim_j: 128,   dim_k: 1,     input: "A" },
    ];

    for c in &configs {
        bench_config(c, trials);
    }

    // === Scaling: how cost breaks down as n grows (symmetric NxNxN input B) ===
    println!("\n\n=== Scaling: input B size from small to large (symmetric NxNxN) ===");
    for n_dim in [64usize, 128, 256, 512, 1024, 2048, 4096] {
        let c = Config {
            label: format!("  {}x{}x{}", n_dim, n_dim, n_dim),
            dim_i: n_dim, dim_j: n_dim, dim_k: n_dim, input: "B",
        };
        bench_config(&c, 3);
    }
}
