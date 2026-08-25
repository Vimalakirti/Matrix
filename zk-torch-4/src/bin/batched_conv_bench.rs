//! Batched conv vs replicated subgraphs, at equal total work.
//!
//! Both sides convolve `B` images with the same weights. The replicated side
//! builds `B` conv nodes (what `for &inp in x { ... }` in the model builders
//! does today); the batched side builds ONE node over a `[B, C, H, W]` tensor
//! and lets the verifier bind the batch index. Same arithmetic, same images --
//! the difference is how many commitment leaves the opening has to carry.
//!
//!   BATCH=4 LAYERS=4 SIZE=16 cargo run --release --bin batched_conv_bench -- bench_config.yaml

use std::time::Instant;
use zk_torch_4::commit::{AjtaiKey, GpuAjtaiStore};
use zk_torch_4::dag::{DagBuilder, DataType, Role, Witness};
use zk_torch_4::transcript::Transcript;

fn env(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d)
}

fn kernel(c_out: usize, c_in: usize) -> Witness {
    let n = c_out * c_in * 4;
    Witness::new(
        vec![c_out, c_in, 2, 2],
        (0..n)
            .map(|i| zk_torch_4::AlmostGoldilocksField(((i % 3) + 1) as u64))
            .collect(),
        DataType::Uint,
        0,
        Role::Constant,
    )
}

fn main() {
    almost_goldilocks_cuda::init().expect("CUDA init");
    let batch = env("BATCH", 4);
    let layers = env("LAYERS", 4);
    let size = env("SIZE", 16);
    let ch = env("CH", 4);

    for replicated in [true, false] {
        let mut g = DagBuilder::new();
        let mut inputs = Vec::new();
        let n_graphs = if replicated { batch } else { 1 };
        for _ in 0..n_graphs {
            let shape = if replicated {
                vec![ch, size, size]
            } else {
                vec![batch, ch, size, size]
            };
            let x = g.input(shape, DataType::Int);
            inputs.push(x);
            let mut h = x;
            let mut s = size;
            for _ in 0..layers {
                let w = g.param(kernel(ch, ch));
                h = g.conv2d(h, w, (2, 2))[0];
                s -= 1;
            }
            let _ = s;
        }
        let (dag, mut witnesses) = g.compile();

        let per = ch * size * size;
        let feed: Vec<_> = inputs
            .iter()
            .enumerate()
            .map(|(i, x_edge)| {
                let n = if replicated { per } else { per * batch };
                (
                    *x_edge,
                    Witness::new(
                        if replicated {
                            vec![ch, size, size]
                        } else {
                            vec![batch, ch, size, size]
                        },
                        (0..n)
                            .map(|v| {
                                zk_torch_4::AlmostGoldilocksField(((v + i) % 5 + 1) as u64)
                            })
                            .collect(),
                        DataType::Int,
                        0,
                        Role::Input,
                    ),
                )
            })
            .collect();
        dag.run(&mut witnesses, &feed);

        let seed = almost_goldilocks_cuda::ajtai::Seed([
            0x0123_4567, 0x89AB_CDEF, 0xFEED_FACE, 0xDEAD_BEEF,
            0xCAFE_BABE, 0x1357_9BDF, 0x2468_ACE0, 0x0BAD_C0DE,
        ]);
        let key = AjtaiKey::new(seed, 24, 21);
        let mut store = GpuAjtaiStore::new(dag.num_edges(), key);
        dag.commit(&witnesses, &mut store);

        let mut tp = Transcript::new(b"batched-conv-bench");
        let t = Instant::now();
        let (dp, fp) = dag.prove_with_fold_tree(&witnesses, &store, &mut tp);
        let prove = t.elapsed().as_secs_f64();

        let mut tv = Transcript::new(b"batched-conv-bench");
        let ok = dag.verify_with_fold_tree(&witnesses, &store, &dp, &fp, &mut tv);

        println!(
            "{:<11} nodes {:>4}  edges {:>4}  prove {:>7.3}s  verified {}",
            if replicated { "replicated" } else { "batched" },
            dag.nodes.len(),
            dag.num_edges(),
            prove,
            ok
        );
    }
}
