use std::env;
use std::path::PathBuf;
use std::process::Command;

/// Find the CUDA installation (mirrors goldilocks-cuda-rs/build.rs).
fn find_cuda_paths() -> (String, String) {
    let system_nvcc = "/usr/lib/nvidia-cuda-toolkit/bin/nvcc";
    let system_lib = "/usr/lib/x86_64-linux-gnu";
    if std::path::Path::new(system_nvcc).exists()
        && std::path::Path::new(&format!("{}/libcudart_static.a", system_lib)).exists()
    {
        eprintln!("[build.rs] Using system CUDA toolkit");
        return (system_nvcc.to_string(), system_lib.to_string());
    }

    let which_output = Command::new("which").arg("nvcc").output().ok();
    if let Some(output) = which_output {
        if output.status.success() {
            let nvcc_path_str = String::from_utf8_lossy(&output.stdout).trim().to_string();
            let nvcc_path = std::path::Path::new(&nvcc_path_str);
            let real_nvcc =
                std::fs::canonicalize(nvcc_path).unwrap_or_else(|_| nvcc_path.to_path_buf());
            if let Some(bin_dir) = real_nvcc.parent() {
                if let Some(cuda_root) = bin_dir.parent() {
                    let lib64 = cuda_root.join("lib64");
                    let lib = cuda_root.join("lib");
                    let lib_path = if lib64.exists() {
                        lib64.to_string_lossy().to_string()
                    } else if lib.exists() {
                        lib.to_string_lossy().to_string()
                    } else {
                        "/usr/lib/x86_64-linux-gnu".to_string()
                    };
                    return (real_nvcc.to_string_lossy().to_string(), lib_path);
                }
            }
        }
    }

    let cuda_path = env::var("CUDA_PATH")
        .or_else(|_| env::var("CUDA_HOME"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());
    let cuda_path = std::fs::canonicalize(&cuda_path)
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or(cuda_path);
    (
        format!("{}/bin/nvcc", cuda_path),
        format!("{}/lib64", cuda_path),
    )
}

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    let cuda_wrapper = manifest_dir.join("cuda/wrapper.cu");
    // Headers live in the sibling cuda_almost_goldilocks/ directory.
    let cuda_dir = manifest_dir.join("../cuda_almost_goldilocks");

    let (nvcc, cuda_lib_path) = find_cuda_paths();
    if !std::path::Path::new(&nvcc).exists() {
        panic!("CUDA toolkit not found at {}", nvcc);
    }

    let output_lib = out_dir.join("libalmost_goldilocks_cuda_wrapper.a");
    let obj_file = out_dir.join("wrapper.o");
    let compute_cap = env::var("CUDA_COMPUTE").unwrap_or_else(|_| "80".to_string());
    let gencode = format!(
        "-gencode=arch=compute_{},code=compute_{}",
        compute_cap, compute_cap
    );

    eprintln!("[build.rs] nvcc: {}", nvcc);
    eprintln!("[build.rs] compute capability: {}", compute_cap);
    eprintln!("[build.rs] header include path: {}", cuda_dir.display());

    let status = Command::new(&nvcc)
        .args([
            "-O3",
            &gencode,
            "-std=c++17",
            "--default-stream",
            "per-thread",
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
        panic!("Failed to compile CUDA wrapper");
    }

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

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=almost_goldilocks_cuda_wrapper");
    println!("cargo:rustc-link-search=native={}", cuda_lib_path);
    println!("cargo:rustc-link-lib=static=cudart_static");
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rustc-link-lib=dl");
    println!("cargo:rustc-link-lib=rt");
    println!("cargo:rustc-link-lib=pthread");

    println!("cargo:rerun-if-env-changed=CUDA_COMPUTE");
    println!("cargo:rerun-if-changed=cuda/wrapper.cu");
    for header in [
        "almost_goldilocks.cuh",
        "almost_extension.cuh",
        "almost_eq_lagrange.cuh",
        "almost_partial_eval.cuh",
        "almost_fused_permute_peval.cuh",
        "almost_sumcheck_prover.cuh",
        "ajtai.cuh",
            "link_sumcheck.cuh",
        "ajtai_chacha8.cuh",
    ] {
        println!(
            "cargo:rerun-if-changed={}",
            cuda_dir.join(header).display()
        );
    }
}
