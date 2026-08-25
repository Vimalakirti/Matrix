//! `precommit_model` — offline-phase demo binary (plan §4.1).
//!
//! Builds a tiny demo DAG with a Role::Constant weight edge, runs the
//! offline weight-commit pass, writes the result to `model.ajtai_commits`,
//! reloads it, and asserts the loaded store is bit-identical to what was
//! produced in memory.
//!
//! In production this binary will be replaced by a model-specific variant
//! that loads a real model from ONNX (or similar) and runs the same
//! offline commit pass. For step 3 it is the smallest viable
//! demonstration of the disk persistence path.

use std::path::PathBuf;

use almost_goldilocks_cuda::ajtai::Seed;
use zk_torch_4::commit::{AjtaiKey, GpuAjtaiStore};
use zk_torch_4::dag::{DagBuilder, DataType, Role, Witness};
use zk_torch_4::util::arith::int_to_f;

fn main() {
    // CLI: optional output path (default ./model.ajtai_commits).
    let args: Vec<String> = std::env::args().collect();
    let out_path: PathBuf = if args.len() >= 2 {
        PathBuf::from(&args[1])
    } else {
        PathBuf::from("model.ajtai_commits")
    };

    // CUDA must be available for the Ajtai kernels.
    almost_goldilocks_cuda::init().expect(
        "precommit_model: CUDA init failed — Ajtai commit kernels require a CUDA-capable GPU",
    );

    // ----- Build a tiny demo DAG: y = x + w, with w as the constant. -----
    let mut g = DagBuilder::new();
    let x = g.input(vec![64], DataType::Int);
    let w_witness = make_const(vec![64], (0..64i128).map(|i| i % 11 - 5).collect());
    let w = g.param(w_witness);
    let _y = g.add(x, w)[0];
    let (dag, witnesses) = g.compile();

    println!(
        "Demo DAG: {} edges, {} nodes (constants: 1)",
        dag.num_edges(),
        dag.nodes.len(),
    );

    // ----- Offline commit. -----
    let key = AjtaiKey::new(demo_seed(), /*max_num_vars=*/ 6, /*b=*/ 8);
    let mut store = GpuAjtaiStore::new(dag.num_edges(), key);
    dag.commit_constants(&witnesses, &mut store);

    let committed: usize = (0..store.num_edges())
        .filter(|&e| store.get(e).is_some())
        .count();
    println!("Committed {} constant edge(s) at AjtaiKey {{ b = {}, max_num_vars = {} }}",
             committed, key.b, key.max_num_vars);

    // ----- Persist. -----
    store.save(&out_path).expect("save");
    let bytes = std::fs::metadata(&out_path).map(|m| m.len()).unwrap_or(0);
    println!("Saved {} bytes to {}", bytes, out_path.display());

    // ----- Reload and verify. -----
    let loaded = GpuAjtaiStore::load(&out_path).expect("load");
    assert_eq!(loaded.num_edges(), store.num_edges());
    assert_eq!(loaded.key.seed.0, store.key.seed.0);
    assert_eq!(loaded.key.max_num_vars, store.key.max_num_vars);
    assert_eq!(loaded.key.b, store.key.b);
    let mut checked = 0usize;
    for edge_id in 0..store.num_edges() {
        match (store.get(edge_id), loaded.get(edge_id)) {
            (None, None) => {}
            (Some(a), Some(b)) => {
                assert_eq!(a.is_sparse, b.is_sparse, "edge {} is_sparse", edge_id);
                assert_eq!(a.planes.len(), b.planes.len(), "edge {} planes", edge_id);
                for p in 0..a.planes.len() {
                    for row in 0..15 {
                        for coef in 0..64 {
                            assert_eq!(
                                a.planes[p].rows[row][coef],
                                b.planes[p].rows[row][coef],
                                "edge {} plane {} row {} coef {}", edge_id, p, row, coef,
                            );
                        }
                    }
                }
                checked += 1;
            }
            (a, b) => panic!(
                "edge {} presence mismatch: saved={} loaded={}",
                edge_id, a.is_some(), b.is_some(),
            ),
        }
    }
    println!("Reloaded and bit-exact verified {} edge(s).", checked);
    println!("OK.");
}

fn make_const(shape: Vec<usize>, raw: Vec<i128>) -> Witness {
    let evals = raw.iter().map(|&v| int_to_f(v)).collect();
    Witness::new(shape, evals, DataType::Int, 0, Role::Constant)
}

/// Demo seed — fixed so the binary's output is reproducible across runs.
fn demo_seed() -> Seed {
    Seed([
        0x01234567, 0x89ABCDEF, 0xFEEDFACE, 0xDEADBEEF,
        0xCAFEBABE, 0x13579BDF, 0x2468ACE0, 0x0BAD_C0DE,
    ])
}
