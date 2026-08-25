//! Streaming LeNet-5, the exact model EZKL ships in `examples/onnx/lenet_5`,
//! composed with the cross-proof streaming accumulator. Each streamed proof is
//! one image; WEIGHTS (conv, bias, fc, and the constant average-pooling kernel)
//! are Role::Constant (deferred -> amortized into one finalize opening) and the
//! per-image INPUT is Role::Input (committed and opened per proof).
//!
//! The graph follows that ONNX operator for operator. Two properties are worth
//! stating because they set this model apart from the other CNNs here: the
//! activation is the quadratic `x^2 + x` rather than ReLU, so it needs no range
//! check and no lookup; and pooling is AVERAGE, which is linear. LeNet is
//! therefore almost entirely linear algebra, and its lookup share should be far
//! below every other model in the table.
//!
//! Run with `cv_config.yaml` as args[1]. Env: N_PROOFS(3) MAX_NUM_VARS(22)
//! NUM_PARTITIONS(1) ZK4_B(21) ZK4_BASE(2).

use std::time::{Duration, Instant};

use almost_goldilocks_cuda::ajtai::Seed;
use almost_goldilocks_cuda::field::AlmostGoldilocksField;

use zk_torch_4::commit::{AjtaiKey, GpuAjtaiStore};
use zk_torch_4::dag::lenet::lenet5;
use zk_torch_4::dag::streaming_accumulator::{AccumulatorState, VerifierAccumulator};
use zk_torch_4::dag::{DagBuilder, DataType, EdgeId, Role, Witness};
use zk_torch_4::transcript::Transcript;
use zk_torch_4::ser_len;
use zk_torch_4::SF_LOG;

fn ms(d: Duration) -> f64 { d.as_secs_f64() * 1e3 }

/// EZKL's LeNet shapes, read from network.onnx.
const CONV: [(usize, usize, usize); 2] = [(6, 1, 5), (16, 6, 5)];   // (c_out, c_in, k)
const FC: [(usize, usize); 3] = [(400, 120), (120, 84), (84, 10)];
const SPATIAL: [(usize, usize); 2] = [(28, 28), (10, 10)];          // conv output dims

/// Conv weights are ZERO and biases are small. `x^2 + x` squares its input, so
/// a full 5x5xc_in accumulator would be squared and leave the commit range at
/// b=21 almost immediately. Zeroing the weights makes each activation input the
/// bias alone, which stays in range and exercises exactly the same graph: the
/// prover's cost depends on the shapes, not the values.
fn conv_weight(c_out: usize, c_in: usize, k: usize) -> Witness {
    let size = c_out.next_power_of_two() * c_in.next_power_of_two()
        * k.next_power_of_two() * k.next_power_of_two();
    Witness::new(vec![c_out, c_in, k, k], zk_torch_4::zero_witness_vec(size),
                 DataType::Uint, *SF_LOG, Role::Constant)
}

fn conv_bias(c_out: usize, h: usize, w: usize) -> Witness {
    let (cp, hp, wp) = (c_out.next_power_of_two(), h.next_power_of_two(), w.next_power_of_two());
    let mut data = zk_torch_4::zero_witness_vec(cp * hp * wp);
    for c in 0..c_out {
        let v = AlmostGoldilocksField((c % 3) as u64);
        for y in 0..h { for x in 0..w { data[c * wp * hp + y * wp + x] = v; } }
    }
    Witness::new(vec![c_out, h, w], data, DataType::Uint, *SF_LOG, Role::Constant)
}

/// ONNX Gemm has transB=1, so the stored weight is [out, in].
fn fc_weight(in_dim: usize, out_dim: usize) -> Witness {
    let size = in_dim.next_power_of_two() * out_dim.next_power_of_two();
    Witness::new(vec![out_dim, in_dim], zk_torch_4::zero_witness_vec(size),
                 DataType::Uint, *SF_LOG, Role::Constant)
}

fn fc_bias(out_dim: usize) -> Witness {
    let mut data = zk_torch_4::zero_witness_vec(out_dim.next_power_of_two());
    for i in 0..out_dim { data[i] = AlmostGoldilocksField((i % 2) as u64); }
    Witness::new(vec![out_dim], data, DataType::Uint, *SF_LOG, Role::Constant)
}

fn demo_seed() -> Seed {
    Seed([0x0123_4567, 0x89AB_CDEF, 0xFEED_FACE, 0xDEAD_BEEF,
          0xCAFE_BABE, 0x1357_9BDF, 0x2468_ACE0, 0x0BAD_C0DE])
}

fn main() {
    env_logger::init();
    almost_goldilocks_cuda::init().expect("CUDA init");

    let env = |k: &str, d: usize| std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d);
    let n_proofs = env("N_PROOFS", 3);
    let max_num_vars = env("MAX_NUM_VARS", 22);
    let num_partitions = env("NUM_PARTITIONS", 1);
    let b = env("ZK4_B", 21);
    let base = env("ZK4_BASE", 2);

    println!("=== Streaming LeNet-5 (EZKL examples/onnx/lenet_5) ===");
    println!("N_PROOFS={} max_num_vars={} partitions={}", n_proofs, max_num_vars, num_partitions);

    let conv_w: Vec<Witness> = CONV.iter().map(|&(co, ci, k)| conv_weight(co, ci, k)).collect();
    let conv_b: Vec<Witness> = CONV.iter().zip(SPATIAL.iter())
        .map(|(&(co, _, _), &(h, w))| conv_bias(co, h, w)).collect();
    let fc_w: Vec<Witness> = FC.iter().map(|&(i, o)| fc_weight(i, o)).collect();
    let fc_b: Vec<Witness> = FC.iter().map(|&(_, o)| fc_bias(o)).collect();

    let mut g = DagBuilder::new();
    let x = g.input(vec![1, 32, 32], DataType::Uint);
    let _out = g.pipe(&[x], lenet5(conv_w, conv_b, fc_w, fc_b));
    let (mut dag, witnesses_template) = g.compile();
    println!("Compile: {} nodes, {} edges", dag.nodes.len(), dag.num_edges());
    assert_eq!(witnesses_template[x][0].role, Role::Input,
               "LeNet input edge must be Role::Input (per-proof), not deferred");

    if num_partitions > 1 {
        dag.set_partition_boundaries(num_partitions);
        println!("Partitions: {} (boundaries: {})",
                 dag.boundary_edges.len() + 1, dag.boundary_edges.len());
    }

    let key = AjtaiKey::new_with_base(demo_seed(), max_num_vars, b, base);
    let mut store = GpuAjtaiStore::new(dag.num_edges(), key);
    let t_off = Instant::now();
    dag.commit_constants(&witnesses_template, &mut store);
    println!("Offline commit (weights, amortized): {:.2}ms", ms(t_off.elapsed()));

    let label = b"zkml-lenet-streaming";
    let mut prover_acc = AccumulatorState::new(label);
    let mut verifier_acc = VerifierAccumulator::new(label);
    let (mut t_run, mut t_commit, mut t_prove, mut t_verify, mut t_acc, mut t_accv) =
        (Duration::ZERO, Duration::ZERO, Duration::ZERO, Duration::ZERO, Duration::ZERO, Duration::ZERO);
    let mut proof_bytes = 0usize;
    let mut breakdown: Option<String> = None;
    let mut checked_role = false;

    println!("Streaming {} images:", n_proofs);
    for it in 0..n_proofs {
        let mut witnesses = witnesses_template.clone();
        // 1 channel pads to 1; 32 is already a power of two.
        let d = zk_torch_4::rand_witness_vec(1 * 32 * 32, 128);
        let batch_inputs = vec![(x, Witness::new(vec![1, 32, 32], d, DataType::Uint, *SF_LOG, Role::Input))];

        let s0 = Instant::now(); dag.run(&mut witnesses, &batch_inputs);
        let d_run = s0.elapsed(); t_run += d_run;

        store.clear_non_constants(&witnesses);
        let s1 = Instant::now(); dag.commit_remaining(&witnesses, &mut store);
        let d_commit = s1.elapsed(); t_commit += d_commit;

        let mut tp = Transcript::new(b"per-img");
        let s2 = Instant::now();
        let (dp, fp) = dag.prove_with_fold_tree_modes(&witnesses, &store, &mut tp, true);
        let d_prove = s2.elapsed(); t_prove += d_prove;

        let mut tv = Transcript::new(b"per-img");
        let s3 = Instant::now();
        let r = dag.verify_with_fold_tree_deferred(&witnesses, &store, &dp, &fp, &mut tv);
        let d_verify = s3.elapsed(); t_verify += d_verify;
        if !r.ok { eprintln!("per-image verify failed at {}", it); return; }

        if !checked_role {
            for dc in &r.claims {
                assert!(dc.edge_id != x, "input edge was deferred as a shared weight -- unsound");
            }
            checked_role = true;
        }

        let s4 = Instant::now();
        let chunk = prover_acc.add_proof(&r, &witnesses);
        let d_acc = s4.elapsed(); t_acc += d_acc;
        proof_bytes += ser_len(&dp) + ser_len(&fp) + ser_len(&chunk);
        breakdown = Some(zk_torch_4::proof_size_report(
            &dp.node_proofs, &dp.edge_proofs, &dp.range_proof,
            &dp.two_pow_proof, &dp.output_claims, &dp, &fp));
        let s5 = Instant::now();
        let ok = verifier_acc.verify_add_proof(&r, &witnesses, &chunk);
        let d_accv = s5.elapsed(); t_accv += d_accv;
        if !ok { eprintln!("streaming verifier rejected at {}", it); return; }

        // Device-wide used bytes, the same probe bench_streaming_llama2 prints.
        // It is whole-device rather than per-process, so it is only meaningful
        // on a GPU this run has to itself; nvidia-smi reports 0 for these
        // binaries, so an in-process probe is the only thing that works.
        let gpu_mem = almost_goldilocks_cuda::mem_get_info()
            .map(|(free, total)| (total - free) / (1024 * 1024))
            .unwrap_or(0);
        println!("  [{:>2}/{}] run {:>7.1}ms commit {:>6.1}ms prove {:>7.1}ms verify {:>6.1}ms acc {:>7.1}ms acc-v {:>6.1}ms gpu-mem {:>6} MiB",
            it + 1, n_proofs, ms(d_run), ms(d_commit), ms(d_prove), ms(d_verify), ms(d_acc), ms(d_accv), gpu_mem);
    }

    let n_steps = prover_acc.num_steps();
    let n_const = prover_acc.num_edges();
    let s_fp = Instant::now();
    let final_proof = prover_acc.finalize(&witnesses_template, &store);
    let t_finalize = s_fp.elapsed();
    let s_fv = Instant::now();
    let ok = verifier_acc.verify_finalize(&store, &final_proof);
    let t_fv = s_fv.elapsed();
    if !ok { eprintln!("verify_finalize REJECTED -- soundness chain broken"); return; }

    let n = n_proofs as f64;
    println!("\n=== Results ({} weight edges deferred, {} reducer steps) ===", n_const, n_steps);
    println!("  prove(defer)  per-img : {:>8.2}ms", ms(t_prove) / n);
    println!("  acc-update    per-img : {:>8.2}ms", ms(t_acc) / n);
    println!("  finalize / N          : {:>8.2}ms", ms(t_finalize) / n);
    println!("  = streaming per-img   : {:>8.2}ms",
             ms(t_prove) / n + ms(t_acc) / n + ms(t_finalize) / n);
    println!("  finalize (one-time)   : {:>8.2}ms  (+verify {:.2}ms)", ms(t_finalize), ms(t_fv));
    println!("  proof         per-unit: {:>8} bytes", proof_bytes / n_proofs);
    if let Some(bk) = &breakdown { println!("{}", bk); }
    println!("  proof     finalize    : {:>8} bytes", ser_len(&final_proof));
    let _ = (t_run, t_commit, t_verify, t_accv);
    println!("\nVerified: true (LeNet-5, weights amortized across {} images)", n_proofs);
}
