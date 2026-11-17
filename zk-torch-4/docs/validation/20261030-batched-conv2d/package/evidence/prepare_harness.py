#!/usr/bin/env python3
"""Fresh, disclosed build-only CPU harness. No FFI stubs or body substitutions."""
import argparse, difflib, hashlib, json, shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('destination',type=Path);args=p.parse_args()
src=args.source.resolve(strict=True);dst=args.destination.resolve()
if not src.is_dir() or dst.exists() or src==dst or src in dst.parents or dst in src.parents:
    raise SystemExit('require an existing source and distinct fresh non-overlapping destination')
if not (src/'zk-torch-4/src/morpheus/piop/cv_batch_tests.rs').is_file():
    raise SystemExit('not a Conv2D extension source tree')
# Do not follow a source-tree symlink into unrelated files.
if any(p.is_symlink() for p in src.rglob('*') if '.git' not in p.parts):
    raise SystemExit('source-tree symlinks are not supported')
shutil.copytree(src,dst,ignore=shutil.ignore_patterns('.git','target','__pycache__'))
changes=[];patch=[]
def change(rel,new):
    path=dst/rel;old=path.read_text() if path.exists() else ''
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(new)
    changes.append({'path':rel,'original_sha256':hashlib.sha256(old.encode()).hexdigest() if old else None,'adapted_sha256':hashlib.sha256(new.encode()).hexdigest()})
    patch.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='original/'+rel,tofile='adapted/'+rel))
rel='almost-goldilocks-cuda-rs/build.rs';s=(dst/rel).read_text()
a=s.index('    let (nvcc, cuda_lib_path) = find_cuda_paths();');b=s.index('    println!("cargo:rerun-if-env-changed=CUDA_COMPUTE");',a)
change(rel,s[:a]+'    // VALIDATION ONLY: omit CUDA discovery, compilation and linker directives.\n'+s[b:])
rel='almost-goldilocks-cuda-rs/src/ffi.rs';s=(dst/rel).read_text();needle='#[link(name = "almost_goldilocks_cuda_wrapper", kind = "static")]\n'
assert s.count(needle)==1
change(rel,s.replace(needle,''))
# All extern declarations remain byte-identical after removing only #[link].
assert (dst/rel).read_text()==(src/rel).read_text().replace(needle,'')
rel='zk-torch-4/src/morpheus/piop/mod.rs';s=(dst/rel).read_text();needle='#[cfg(test)]\nmod cv_batch_tests;';assert s.count(needle)==1
change(rel,s.replace(needle,'pub mod cv_batch_tests;'))
rel='zk-torch-4/src/morpheus/piop/cv_batch_tests.rs';s=(dst/rel).read_text();body=s.replace('#[test]\n','');change(rel,body)
assert (dst/rel).read_text()==(src/rel).read_text().replace('#[test]\n','')
change('zk-torch-4/src/bin/cpu_conv_forward.rs','''fn main() {
use zk_torch_4::morpheus::piop::cv_batch_tests::*;
batched_conv_cpu_semantics_and_explicit_shape();
println!("PASS actual CPU Conv2D/FlattenKernel forward semantics: 32 cases");
batched_conv_actual_field_tables_match_forward();
println!("PASS actual normal/PP reduction-table helpers: 8 cases and alpha=0,1,nonbase");
conv_kernel_source_cpu_and_collision_guards();
println!("PASS actual kernel source fixtures, collision/overflow rejection; not PIOP/PCS/CUDA");
}
''')
change('zk-torch-4/src/bin/cpu_conv_replay.rs','''fn main() {
use zk_torch_4::morpheus::piop::cv_batch_tests::*;
println!("RUN authored masked PIOP replay (not PCS/CUDA)");
batched_conv_piop_masks_padding_and_rejects_false_outputs();
batched_conv_piop_rejects_padded_output_and_full_advice();
batched_conv_piop_stride_dilation_and_shared_kernel();
conv_kernel_source_piop_preserves_nonidentity_producers();
println!("PASS authored masked PIOP verifier/link rows (not PCS/CUDA)");
}
''')
(dst/'harness-adaptations.json').write_text(json.dumps(changes,indent=2)+'\n')
(dst/'harness-adaptations.patch').write_text(''.join(patch))
(dst/'test-body-identity.json').write_text(json.dumps({'normalization':'remove only exact #[test] attribute lines; every function body preserved','production_test_file_sha256':hashlib.sha256(s.encode()).hexdigest(),'exposed_unchanged_bodies_sha256':hashlib.sha256(body.encode()).hexdigest(),'all_ffi_declarations_unchanged':True},indent=2)+'\n')
print(json.dumps({'destination':str(dst),'changes':len(changes),'test_body_sha256':hashlib.sha256(body.encode()).hexdigest()}))
