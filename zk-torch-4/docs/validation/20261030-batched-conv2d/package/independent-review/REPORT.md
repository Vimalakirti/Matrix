# Independent targeted review: Conv2D batch extension

## Scope and evidence boundary

This is a targeted algebra, index-map, padding, and source-identity review of the
new Conv2D normal and polynomial-product reductions. It is not one of five full
independent project acceptances, a native CUDA build, a Rust PIOP execution, or an
end-to-end PCS/security validation. The tests below independently implement the
field/index equations over q=2^64−2^32−31; they do not execute the Rust reductions.

Sources inspected: `morpheus-batched-convolution/zk-torch-4/src/morpheus/piop/cv.rs`
and the three Conv2D builders in `src/dag/builder.rs`, against
`morpheus-selected-normalization-joint`. The author's source is evolving; the final reviewed hashes and repair status
are recorded below.

## Reproduction

```
python check_relations.py
python check_fallback.py
```

First suite: 80 cases, 147,192 assertions; B∈{1,2,3,5}, C,D∈{1,3}, non-square
spatial sizes, strides 1/2/3, dilation 1/2, α∈{0,1,random non-Boolean field value}.
All input/weight padding carries unrelated nonzero poison. Checked:

* low-to-high W,H,C,B indexing and sequential width/height contraction;
* logical-batch fold with a prefix mask, normal channel-mask sum;
* normal input and shared-weight child-claim identities at non-Boolean points;
* full-convolution coefficients, exact public crop MLE, and no-carry spatial
  flattening on every valid output/tap;
* PP coefficient identity and both input/flat-weight child-claim identities;
* masked-axis contraction at arbitrary offsets, including inner output-channel
  position used by transposed shared helper paths.

Second suite: 96 cases independently verify virtual-kernel construction and its
Rel map back to the actual flat weight source, including padded input/output
channels and non-Boolean points. Additional concrete source-identity witnesses
are stored in `results.json` and `fallback-results.json`.

## Algebraic conclusions

For canonical positive valid geometry, the reviewed normal batching formula is
consistent: batch is folded only from X, not W; C is a public prefix factor;
non-power-of-two D is folded with its prefix mask and then reduced with the same
mask. The input link carries exactly the original batch identity/mask. Thus no
cross-batch weight product is introduced.

The PP formula is also consistent: full coefficients are sampled at unmasked
random (b,d) so their padded batch/channel slots are constrained, while RHS X/W
folds are masked by logical B/D and their product sums only c<C. Sparse tap powers
exclude gap padding. Computing α^(S−1−p) directly handles α=0 without inversion.
Exact crop relation avoids the old power-of-two stride bit-affinity restriction.

For valid outputs and taps, `wo*sw+kw*dw < W <= Wp`; hence
`p=(ho*sh+kh*dh)*Wp+(wo*sw+kw*dw)` has no row carry. Its coefficient index obeys
`S−1−p+j=S−1−ho*sh*Wp−wo*sw`, and 0≤S−1−p+j<S+J−1. This is an integer exponent
identity, not reduction modulo a cyclic convolution length.

The convolution values themselves are field arithmetic. Integer-valued CNN
semantics additionally need an independently enforced bound such as
`C*KH*KW*M_X*M_W < q/2` for signed lifts; this review does not establish those
model-wide range bounds. PP polynomial identity soundness includes degree at
most `(Lp−1 + log2(Bp) + log2(Dp)) / |K|` before the separate sumcheck/link terms;
malicious coefficients can occupy every padded exponent slot.

## Findings communicated to author

1. **Wrong source through mismatched FlattenKernel (confirmed).** Old normal
   `unflatten` ignored FK geometry. X=[1,...,8], raw W=[2,3], FK dw=2, Conv dw=1
   yields actual [2,4,6,8,10,12,14] but bypass [8,13,18,23,28,33,38]. A same-padded-
   extent variant (Conv dw=2, FK dw=3, both flat extent 4) yields actual
   [2,4,6,8,10,12] versus bypass [11,16,21,26,31,36]. Author added matching-geometry
   bypass and an actual-flat-edge tap-map fallback. Its map equations passed
   the independent 96-case suite. Final enforcement status is recorded below after reinspection.
2. **Non-identity ChangeShape bypass (confirmed in first repair).** A matching
   FK DW2 yields [2,0,3,0]; truncate to 2 and pad back to 4 yields [2,0,0,0].
   Blindly walking both views restores the lost coefficient and proves wrong
   source semantics. Only equal-arity ChangeShape edges may be skipped; every
   encountered width-changing view must force binding the original flat edge.
3. **Noninjective FlattenKernel scatter (confirmed).** FK s_w=1, KH=KW=2, dh=dw=1
   maps (0,1) and (1,0) to j1. CPU scatter overwrites (last-write), whereas a Rel
   with both pairs sums the two weights. Canonical builder geometry avoids this,
   but alternate-producer fallback requires fail-closed duplicate-j validation
   (preferable to giving GPU colliding writes a meaning).

## Cost caveat

Batch support is not batch-independent prover work. Normal contraction reads
O(Bp*Cp*Hp*Wp) input cells, then contracts spatial axes; non-power-of-two B adds
batch variables to the input map. PP full advice is Bp*Dp*Lp, its input reduction
has spatial+batch variables, its weight reduction has flat-tap+output-channel
variables, and its crop map now uses the full exponent domain. No old benchmark
numbers were validated for this changed protocol.


## Final targeted disposition (2026-10-30 07:35 UTC)

All three concrete source-identity findings above have been repaired in the
reviewed source snapshot, and the repairs were reinspected:

* `conv2d_kernel` checks actual flat extent before either path, bypasses only
  matching public FlattenKernel geometry, and otherwise links the actual flat
  edge through the tested sparse virtual-kernel map.
* Only equal-arity ChangeShape edges may be bypassed. Truncation/padding chains
  force the actual-edge fallback even when final extent is restored.
* `flatten_kernel_rel` uses checked arithmetic, requires positive geometry,
  rejects out-of-bounds and duplicate tap indices, and is used by the actual
  FlattenKernel reduction. The Conv2D geometry validator separately requires
  consistent spatial/flat strides, effective kernel fit, and output sizes.

No remaining blocker was found in this targeted Conv2D algebra/index/source-map
scope. The exact reviewed sources are identified by
`reviewed-source-hashes.json`; subsequent changes require rechecking. The
independent Python suites passed; author-reported Rust PIOP replay is still
linker-blocked by retained CUDA FFI, so zero replay bodies executed there. This
review does not turn that blocker into a pass and does not certify full-model,
GPU, PCS, or deferred-proof batching security.

`check_mask_mutants.py` additionally rejects separate dropped-B, dropped-C,
dropped-D, and dropped-all masks on deterministic poisoned padding, demonstrates
that equal/opposite batch output errors do not cancel at non-Boolean batch
points, and confirms padded full advice has nonzero polynomial-test influence.
These are arithmetic mutation checks, not acceptance/rejection of production
Rust proofs.

## Narrow CNN-wrapper integration supplement

At the other worker's request, inspected
`morpheus-cnn-layout-q20/zk-torch-4/src/dag/builder.rs` hash
`c38fbd735363f1a15d05fe22fb2403bcac6f07aac59d9bf0bda509f9dfa7fc5e`.
`fold_per_channel2d_input` and `restore_per_channel2d_output` preserve W,H,C,B
low-to-high bits through equal-arity registered views. Their reverse-shape
column-major public mask zeros exposed invalid B/C/spatial coordinates; it does
not claim the underlying maxpool advice itself is zero. All three imported
Conv2D builders read logical B directly from an explicit shape and restore that
shape, closing the B=3,C=3 lead=9 versus Cp=4 inference failure.

`check_wrapper_integration.py` independently passed 36 cases and 36,552
index/source assertions; exact results/hash are in
`wrapper-integration-results.json`. This narrow supplement does not repair or
certify maxpool achievability, general classifier fidelity, or PIOP execution.

## Exact combined-overlay recheck (2026-10-30 07:42 UTC)

Rechecked immutable `morpheus-batched-convolution-joint` against frozen
`morpheus-cnn-layout-joint`, whose 454-source manifest SHA256 is
`da3f4f1bae5927b06547d61bd061725a46634db1cb944df469da532084770113`.
The original 454 base source hashes all matched. The combined canonical source
universe contains exactly those paths plus `cv_batch_tests.rs` (455 files),
recorded in `joint-reviewed-source-manifest.json` with SHA256
`a57be8f20529556584fef5a97d8bf24c54d77809835890bd4e2678848c227579`.

Only three source paths differ: `cv.rs`, `cv_batch_tests.rs`, and PIOP `mod.rs`.
The first two are byte-identical to the reviewed snapshot. `mod.rs` is exactly
its frozen base plus the cfg(test) Conv batch module and PP full-advice
registration for `outputs.len() >= 2` (previously `== 2`); existing `exp_tests`
and `cnn_layout_tests` modules remain. This condition agrees with PP reduction
dispatch on `outputs.len() > 1`, including the additional grand-product advice
case without treating that separate advice as the full convolution.

The entire builder is byte-identical to the frozen joint base. Its six relevant
mask, explicit-batch wrapper, and Conv2D builder function bodies are also
byte-identical to the previously reviewed CNN supplement. Unrelated earlier
normalization changes in the frozen joint builder are not re-reviewed here.

Five inherited `__pycache__/*.pyc` files exist physically in both trees with
identical hashes. They are explicitly recorded in `joint-overlay-results.json`
and excluded from the canonical source manifest rather than silently omitted
from a claim about all physical files.

`check_joint_overlay.py` reproduces these checks. Scoped source verdict: **no
new Conv2D/wrapper integration blocker found**. This is an overlay/source verdict,
not whole-project acceptance. Author-reported CPU runner results are not
independently executed by this recheck, and the native PIOP linker blocker
remains unchanged.

## Portable-input verification (2026-10-30 07:48 UTC)

Only filesystem inputs were parameterized; assertions and findings were not
changed. `check_wrapper_integration.py` accepts `--root`.
`check_joint_overlay.py` accepts `--base`, `--root`, `--manifest`,
`--supplement`, and `--reviewed-hashes`. Original paths remain optional defaults.
Script hashes before/after are recorded in `portability-script-update.json`.

Both scripts passed again on the exact current inputs. They then passed from a
fresh copied directory under `/tmp`, using relative paths only:

```
python review/check_wrapper_integration.py --root base
python review/check_joint_overlay.py --base base --root combined \
  --supplement base --manifest base-manifest.json \
  --reviewed-hashes review/reviewed-source-hashes.json
```

The frozen base is a valid supplement input because the six relevant reviewed
function bodies were already checked byte-identical. The relocated overlay
result is byte-identical to the original result and reproduces the same
`a57be8f20529556584fef5a97d8bf24c54d77809835890bd4e2678848c227579` source-manifest
hash. The wrapper index suite again passed 36 cases / 36,552 assertions. See
`portability-relocation-results.json` and the two `portable-*.log` files.
