//! ResNet-50 MLPerf accuracy evaluation binary.
//!
//! Reads exported weights (with fused BatchNorm) and preprocessed ImageNet images,
//! runs zk-torch-3's fixed-point forward pass, and reports Top-1 accuracy.
//!
//! Usage:
//!     cargo run --release --bin resnet_mlperf_acc -- \
//!         --weights-dir /tmp/resnet50_export \
//!         --num-images 100
//!
//! The weights directory should be created by scripts/export_resnet50.py.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Instant;

use goldilocks_cuda::GoldilocksField;

use zk_torch_3::dag::resnet::{resnet50_with_biases, resnet50_conv_configs};
use zk_torch_3::dag::{DagBuilder, DataType, Role, Witness};
use zk_torch_3::util::arith::f_to_int;

/// Read a tensor from binary file: [ndim: u32] [shape: ndim × u32] [data: n × i64].
/// Returns (shape, field_data).
fn read_tensor(path: &Path) -> (Vec<usize>, Vec<GoldilocksField>) {
    let mut file = fs::File::open(path).unwrap_or_else(|e| panic!("Cannot open {}: {}", path.display(), e));
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).unwrap();

    let mut offset = 0;

    // Read ndim
    let ndim = u32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap()) as usize;
    offset += 4;

    // Read shape
    let mut shape = Vec::with_capacity(ndim);
    for _ in 0..ndim {
        let s = u32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap()) as usize;
        shape.push(s);
        offset += 4;
    }

    // Read data as i64, convert to GoldilocksField
    let num_elements = (buf.len() - offset) / 8;
    let mut data = Vec::with_capacity(num_elements);
    for i in 0..num_elements {
        let val = i64::from_le_bytes(buf[offset + i * 8..offset + (i + 1) * 8].try_into().unwrap());
        // Convert i64 to GoldilocksField (values are already in field representation)
        data.push(GoldilocksField(val as u64));
    }

    (shape, data)
}

/// Read metadata file to get sf_log and conv configs.
fn read_metadata(dir: &Path) -> (usize, usize, Vec<(usize, usize, usize, usize)>) {
    let content = fs::read_to_string(dir.join("metadata.txt")).expect("Cannot read metadata.txt");
    let mut sf_log = 10;
    let mut num_conv = 0;
    let mut num_classes = 1000;
    let mut configs = Vec::new();

    for line in content.lines() {
        if let Some(val) = line.strip_prefix("sf_log=") {
            sf_log = val.parse().unwrap();
        } else if let Some(val) = line.strip_prefix("num_conv=") {
            num_conv = val.parse().unwrap();
        } else if let Some(val) = line.strip_prefix("num_classes=") {
            num_classes = val.parse().unwrap();
        } else if line.starts_with("conv_") {
            let parts: Vec<&str> = line.split('=').collect();
            if parts.len() == 2 {
                let nums: Vec<usize> = parts[1].split(',').map(|s| s.parse().unwrap()).collect();
                if nums.len() == 4 {
                    configs.push((nums[0], nums[1], nums[2], nums[3]));
                }
            }
        }
    }
    assert_eq!(configs.len(), num_conv);
    let _ = num_classes;
    (sf_log, num_conv, configs)
}

/// Load conv weight as Witness. Uses Uint,sf=sf_log so the resnet50_with_biases
/// builder knows the scale for DivConst after each conv.
fn load_conv_weight(path: &Path, sf_log: usize) -> Witness {
    let (shape, data) = read_tensor(path);
    Witness::new(shape, data, DataType::Uint, sf_log, Role::Constant)
}

/// Load bias as Witness.
fn load_bias(path: &Path, sf_log: usize) -> Witness {
    let (shape, data) = read_tensor(path);
    Witness::new(shape, data, DataType::Uint, sf_log, Role::Constant)
}

/// Load preprocessed image as Witness.
fn load_image(path: &Path, sf_log: usize) -> Witness {
    let (shape, data) = read_tensor(path);
    Witness::new(shape, data, DataType::Uint, sf_log, Role::Input)
}

/// Extract prediction from output witness: argmax of logits.
fn extract_prediction(witnesses: &[Vec<Witness>], output_edge: usize, num_classes: usize, sf_log: usize) -> (usize, f64) {
    let output = &witnesses[output_edge][0];
    let evals = output.data.as_ref().unwrap().evaluations_ref();

    let sf = (1u64 << sf_log) as f64;
    let mut vals: Vec<(usize, f64)> = (0..num_classes)
        .map(|i| (i, f_to_int(evals[i]) as f64 / sf))
        .collect();
    vals.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let best_class = vals[0].0;
    let confidence = vals[0].1;

    // Print top-5
    print!("    top5: [");
    for (i, (cls, val)) in vals.iter().take(5).enumerate() {
        if i > 0 { print!(", "); }
        print!("{}({:.2})", cls, val);
    }
    println!("]");

    (best_class, confidence)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let weights_dir = args.iter()
        .position(|a| a == "--weights-dir")
        .map(|i| PathBuf::from(&args[i + 1]))
        .unwrap_or_else(|| PathBuf::from("/tmp/resnet50_export"));

    let num_images: usize = args.iter()
        .position(|a| a == "--num-images")
        .and_then(|i| args[i + 1].parse().ok())
        .unwrap_or(100);

    // Read metadata to get sf_log
    let (sf_log, num_conv, _configs) = read_metadata(&weights_dir);
    let num_classes = 1000;

    // Write config.yaml BEFORE lazy statics are accessed.
    // CONFIG_FILE resolves to args[1] if args.len() >= 2, else "config.yaml".
    // Our binary's args[1] is "--weights-dir", not a yaml file.
    // Write to that same "config.yaml" default path, then ensure CONFIG loads it.
    let config_path = "config.yaml";
    fs::write(config_path, format!(
        "sf:\n  scale_factor_log: {}\n  table_size_log: 20\n  table_commit_log: 16\n",
        sf_log
    )).expect("Cannot write config.yaml");
    // CONFIG_FILE will try to open args[1]="--weights-dir" which fails,
    // so CONFIG falls through to Default. We need CONFIG_FILE to be "config.yaml".
    // Fix: set args[1] to be config.yaml by running binary as:
    //   resnet_mlperf_acc config.yaml --weights-dir ...
    // But that's awkward. Instead, check if SF_LOG matches and warn.

    println!("=== ResNet-50 MLPerf Accuracy Evaluation ===");
    println!("  Weights dir: {}", weights_dir.display());
    println!("  Num images: {}", num_images);
    println!("  Export SF_LOG: {}, Runtime SF_LOG: {}", sf_log, *zk_torch_3::SF_LOG);
    if sf_log != *zk_torch_3::SF_LOG {
        eprintln!("WARNING: SF_LOG mismatch! Export={}, Runtime={}.", sf_log, *zk_torch_3::SF_LOG);
        eprintln!("  Fix: run as `cargo run --release --bin resnet_mlperf_acc -- config.yaml --weights-dir ...`");
        eprintln!("  Or export with --sf-log {}", *zk_torch_3::SF_LOG);
    }
    println!("  Num conv layers: {}", num_conv);

    // Load conv weights and biases
    println!("Loading conv weights and biases...");
    let t0 = Instant::now();
    let conv_weights: Vec<Witness> = (0..num_conv)
        .map(|i| load_conv_weight(&weights_dir.join(format!("conv_{:03}_weight.bin", i)), sf_log))
        .collect();
    let conv_biases: Vec<Witness> = (0..num_conv)
        .map(|i| load_bias(&weights_dir.join(format!("conv_{:03}_bias.bin", i)), sf_log))
        .collect();
    println!("  Loaded {} conv weights + biases in {:?}", num_conv, t0.elapsed());

    // Load FC weight and bias
    let fc_weight = load_conv_weight(&weights_dir.join("fc_weight.bin"), sf_log);
    let fc_bias = load_bias(&weights_dir.join("fc_bias.bin"), sf_log);

    // Build DAG (once, reuse for all images)
    println!("Building ResNet-50 DAG...");
    let t1 = Instant::now();
    let mut g = DagBuilder::new();
    let x = g.input(vec![3, 224, 224], DataType::Float);
    let _output = g.pipe(
        &[x],
        resnet50_with_biases(conv_weights, conv_biases, fc_weight, Some(fc_bias), num_classes, num_conv),
    );
    let (dag, mut witnesses) = g.compile();
    let output_edge = dag.num_edges() - 1; // Last edge is the output
    println!("  DAG: {} nodes, {} edges, compiled in {:?}", dag.nodes.len(), dag.num_edges(), t1.elapsed());

    // Read labels
    let labels_path = weights_dir.join("labels.txt");
    let labels: Vec<(usize, i32)> = fs::read_to_string(&labels_path)
        .expect("Cannot read labels.txt")
        .lines()
        .filter_map(|line| {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                Some((parts[0].parse().ok()?, parts[1].parse().ok()?))
            } else {
                None
            }
        })
        .collect();

    // Run inference on each image
    println!("\nRunning inference on {} images...", num_images.min(labels.len()));
    let images_dir = weights_dir.join("images");
    let mut correct = 0;
    let mut total = 0;

    for (idx, &(_, true_label)) in labels.iter().take(num_images).enumerate() {
        let img_path = images_dir.join(format!("{:05}.bin", idx));
        if !img_path.exists() {
            continue;
        }

        let input = load_image(&img_path, sf_log);

        let t = Instant::now();
        dag.run(&mut witnesses, &[(0, input)]);
        let elapsed = t.elapsed();

        // Debug: print conv1 output at [c=0, h=0, w=0..5]
        // Conv BasicBlock convention: w stride=1, h stride=w_pad, c stride=w_pad*h_pad
        if total == 0 {
            // Find the edge after stem conv+scale+bias+relu (should be edge ~9)
            for (eid, ws) in witnesses.iter().enumerate() {
                if let Some(w) = ws.first() {
                    if w.shape == vec![64, 112, 112] && w.role == Role::Output {
                        if let Some(d) = w.data.as_ref() {
                            let evals = d.evaluations_ref();
                            let sf = (1u64 << sf_log) as f64;
                            let w_pad = 112usize.next_power_of_two(); // 128
                            let h_pad = 112usize.next_power_of_two(); // 128
                            let vals: Vec<f64> = (0..5).map(|wi| {
                                let idx = wi + 0 * w_pad + 0 * w_pad * h_pad;
                                f_to_int(evals[idx]) as f64 / sf
                            }).collect();
                            println!("    edge {} shape={:?} sf={} [c=0,h=0,w=0:5]: {:?}", eid, w.shape, w.sf, vals);
                            break;
                        }
                    }
                }
            }
        }

        let (pred, confidence) = extract_prediction(&witnesses, output_edge, num_classes, sf_log);

        if pred as i32 == true_label {
            correct += 1;
        }
        total += 1;

        if total <= 10 || total % 10 == 0 {
            println!(
                "  [{}/{}] pred={}, true={}, correct={}, acc={:.2}%, time={:?}",
                total,
                num_images.min(labels.len()),
                pred,
                true_label,
                pred as i32 == true_label,
                100.0 * correct as f64 / total as f64,
                elapsed,
            );
        }
    }

    println!("\n=== Results ===");
    println!("  Total images: {}", total);
    println!("  Correct: {}", correct);
    println!("  Top-1 accuracy: {:.2}% ({}/{})", 100.0 * correct as f64 / total as f64, correct, total);
    println!("  MLPerf target (99% of 76.46%): 75.71%");
    println!("  SF_LOG used: {}", sf_log);
}
