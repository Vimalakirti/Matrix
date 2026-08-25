use std::env;
use std::path::PathBuf;
use std::process::Command;

/// Find the CUDA installation path.
/// Prefers the system CUDA 11.5 toolkit (/usr/lib/nvidia-cuda-toolkit) over
/// newer versions in /usr/local/cuda, as CUDA 11.5 is more compatible with
/// various driver versions.
fn find_cuda_paths() -> (String, String) {
    // Preferred: Ubuntu's system CUDA toolkit (typically 11.5)
    // This has better driver compatibility
    let system_nvcc = "/usr/lib/nvidia-cuda-toolkit/bin/nvcc";
    let system_lib = "/usr/lib/x86_64-linux-gnu";

    if std::path::Path::new(system_nvcc).exists()
        && std::path::Path::new(&format!("{}/libcudart_static.a", system_lib)).exists()
    {
        eprintln!("[build.rs] Using system CUDA toolkit (better driver compatibility)");
        eprintln!("[build.rs] nvcc: {}", system_nvcc);
        eprintln!("[build.rs] lib: {}", system_lib);
        return (system_nvcc.to_string(), system_lib.to_string());
    }

    // Fallback: Try to find nvcc in PATH
    let which_output = Command::new("which").arg("nvcc").output().ok();

    if let Some(output) = which_output {
        if output.status.success() {
            let nvcc_path_str = String::from_utf8_lossy(&output.stdout).trim().to_string();
            let nvcc_path = std::path::Path::new(&nvcc_path_str);

            // Follow symlinks to get the real path
            let real_nvcc =
                std::fs::canonicalize(nvcc_path).unwrap_or_else(|_| nvcc_path.to_path_buf());
            eprintln!(
                "[build.rs] Found nvcc at: {} -> {}",
                nvcc_path_str,
                real_nvcc.display()
            );

            // nvcc is typically at /path/to/cuda/bin/nvcc
            // So CUDA root is two directories up
            if let Some(bin_dir) = real_nvcc.parent() {
                if let Some(cuda_root) = bin_dir.parent() {
                    let lib64 = cuda_root.join("lib64");
                    let lib = cuda_root.join("lib");
                    let lib_path = if lib64.exists() {
                        lib64.to_string_lossy().to_string()
                    } else if lib.exists() {
                        lib.to_string_lossy().to_string()
                    } else {
                        eprintln!(
                            "[build.rs] {} doesn't look like CUDA root (no lib64 or lib)",
                            cuda_root.display()
                        );
                        "/usr/lib/x86_64-linux-gnu".to_string()
                    };
                    return (real_nvcc.to_string_lossy().to_string(), lib_path);
                }
            }
        }
    }

    // Last resort fallback
    let cuda_path = env::var("CUDA_PATH")
        .or_else(|_| env::var("CUDA_HOME"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());

    let cuda_path = std::fs::canonicalize(&cuda_path)
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or(cuda_path);

    let nvcc = format!("{}/bin/nvcc", cuda_path);
    let lib_path = format!("{}/lib64", cuda_path);
    eprintln!("[build.rs] Using CUDA path from env: {}", cuda_path);
    eprintln!("[build.rs] Using nvcc: {}", nvcc);

    (nvcc, lib_path)
}

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    let cuda_wrapper = manifest_dir.join("cuda/wrapper.cu");
    let cuda_dir = manifest_dir.join("../cuda");

    // Find CUDA toolkit
    let (nvcc, cuda_lib_path) = find_cuda_paths();

    // Check if nvcc exists
    if !std::path::Path::new(&nvcc).exists() {
        panic!(
            "CUDA toolkit not found at {}. Please install CUDA and ensure nvcc is in PATH.",
            nvcc
        );
    }

    let output_lib = out_dir.join("libgoldilocks_cuda_wrapper.a");

    // Compile CUDA code to object file
    let obj_file = out_dir.join("wrapper.o");

    // Get compute capability from environment, default to compute_80 (A100)
    // We use PTX-only compilation (code=compute_XX) for better driver compatibility
    // The driver will JIT compile the PTX at runtime
    let compute_cap = env::var("CUDA_COMPUTE").unwrap_or_else(|_| "80".to_string());

    eprintln!("[build.rs] Compiling with nvcc: {}", nvcc);
    eprintln!("[build.rs] Compute capability: {}", compute_cap);
    eprintln!("[build.rs] CUDA include path: {}", cuda_dir.display());

    // Use PTX-only compilation for better driver compatibility
    // This avoids issues with mismatched CUDA toolkit and driver versions
    let gencode = format!(
        "-gencode=arch=compute_{},code=compute_{}",
        compute_cap, compute_cap
    );

    let status = Command::new(&nvcc)
        .args([
            "-O3",
            &gencode,
            "-std=c++17",
            "--default-stream", "per-thread",
            "-Xcompiler",
            "-fPIC",
            "-c",
            cuda_wrapper.to_str().unwrap(),
            "-o",
            obj_file.to_str().unwrap(),
            &format!("-I{}", cuda_dir.to_str().unwrap()),
        ])
        .status()
        .expect("Failed to execute nvcc");

    if !status.success() {
        panic!("Failed to compile CUDA code");
    }

    // Create static library
    let status = Command::new("ar")
        .args([
            "rcs",
            output_lib.to_str().unwrap(),
            obj_file.to_str().unwrap(),
        ])
        .status()
        .expect("Failed to execute ar");

    if !status.success() {
        panic!("Failed to create static library");
    }

    // Link instructions
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=goldilocks_cuda_wrapper");

    // Link CUDA runtime from the appropriate location
    eprintln!("[build.rs] Linking CUDA libraries from: {}", cuda_lib_path);
    println!("cargo:rustc-link-search=native={}", cuda_lib_path);

    // Use static CUDA runtime to avoid driver version mismatch issues
    println!("cargo:rustc-link-lib=static=cudart_static");
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rustc-link-lib=dl");
    println!("cargo:rustc-link-lib=rt");
    println!("cargo:rustc-link-lib=pthread");

    // Rebuild if CUDA files change or architecture changes
    println!("cargo:rerun-if-env-changed=CUDA_COMPUTE");
    println!("cargo:rerun-if-changed=cuda/wrapper.cu");
    println!(
        "cargo:rerun-if-changed={}",
        cuda_dir.join("goldilocks.cuh").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        cuda_dir.join("extension.cuh").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        cuda_dir.join("poseidon2.cuh").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        cuda_dir.join("challenger.cuh").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        cuda_dir.join("basefold.cuh").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        cuda_dir.join("eq_lagrange.cuh").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        cuda_dir.join("partial_eval.cuh").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        cuda_dir.join("sumcheck_prover.cuh").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        cuda_dir.join("fused_permute_peval.cuh").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        cuda_dir.join("monolith.cuh").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        cuda_dir.join("monolith_kernels.cu").display()
    );
}
