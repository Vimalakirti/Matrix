# Independent targeted CNN layout / proof-map review

Review snapshot: 2026-10-30 UTC. Candidate `Q20-based source`, compared with `frozen Q20 base`; reviewed-source hashes are in `reviewed-source-sha256.json`. Read-only audit: candidate sources were not edited by this reviewer. This is **not a full paper review/acceptance**, not independent full-model inference, and not cryptographic runtime validation.

## Result

No remaining algebraic/index-map blocker found in the reviewed staged flatten, row-major channel-bias helper, LeNet FC masks, or corrected ResNet GAP-to-FC logical-shape views. The row-major SiLU adapter is the right metadata interpretation for the generic column-major finite sigmoid profile; its joint runtime validation is owned by the implementation worker, not claimed here.

One shape-validation gap found during review was repaired by the implementation worker: the ResNet head originally checked only padded arity. I re-read the corrected code: explicit `[B,C,H,W]` now checks exact B and C; an unbatched `[C,H,W]` checks exact C; folded input explicitly requires the declared B/C padded-stride conventions. The remaining ambiguity of a padded folded batch is inherent in that legacy representation and is not presented as recovering a lost logical B. New streaming ResNet inputs explicitly retain `[B,3,H,W]`.

## Exact mapping audit

### Flatten

Let reversed logical input dimensions be `d0,...,d(r-1)`, padded dimensions `Dj`, logical flattened length `N=prod(dj)`, and padded output length `Np`. Public factor j is

`Gj[z,a] = [z<N] [a = floor(z/prod_{k<j}(dk)) mod dj]`,

stored at `z + Np*a`, with shape `[N,dj]`. It is 0 on both output padding and invalid axis coordinates. Metadata reversal of the original row-major tensor preserves the raw polynomial exactly; `ChangeShape` neither truncates nor grows here.

The emitted stages are `abc,za->zbc`, then `zbc,zb->zc`, then `zc,zc->z` in the rank-3 case. The first stage has only free-once output axes: its claim fixes the output `z` coordinates in the public factor and fixes the remaining input-axis coordinates in the data operand, then sums axis a. Later stages sum their common z together with the eliminated axis and include the output equality factor over z. These are precisely the maps constructed by canonical `Walk::einsum` in `morpheus/piop/mod.rs`. No secret-dependent shape/index choice occurs.

This staged structure also avoids the legacy `BasicBlock::Einsum` missing-axis broadcast pitfall: after free-once partial evaluation, both operands contain every remaining sumcheck axis in the same order. The earlier all-factors-at-once representation would not have had that useful legacy property; it is not the reviewed final lowering.

For `[16,5,5] -> [400]`, public factors contain exactly `512*(8+8+16)=16,384` slots, with 1,200 one entries. Derived stage sizes are 65,536, 8,192, and 512 slots. The three sumcheck spaces have respectively 3, 12, and 13 variables, not one joint 19-variable product. These derived intermediates are not committed advice.

### Channel bias / masks

`[B,C,(D),H,W]` is reinterpreted as column-major `[W,H,(D),C,B]`, with equal padded arity. A channel-vector bias becomes `[1,1,(1),C,1]`; full row-major single-image biases become the matching spatial view with final B=1. Canonical Add projects its claim to the matching coordinates, so it reads the channel bits rather than low spatial bits. The explicit logical mask is a binary Einsum, which includes the required equality factor for coordinatewise multiplication. Restoring original shape again preserves raw bits.

### FC and logical batch

LeNet FC uses documented column-major `[out,in]` matrices. Input masks exclude padded summation coordinates and output masks prevent padded bias/weight columns from entering the next layer. ResNet head uses `[features,B]` so C is low and B high, normalizes vector or `[classes,1]` bias to the appropriate shape, and zeros padded classes and padded B after bias broadcast. Exact logical shape assertions now precede the padded-arity check.

## Important cost correction (reported to implementer)

Public does **not** mean uncommitted in the current implementation. `Graph::new` marks `Role::Input` literals committed; `tensor_specs` packs 0/1 public literals as one-bit sources; `Walk` adds a public-value binding row for each. Thus the 16,384 flatten-map slots cost 16,384 one-bit source coordinates, as well as dense host/program-identity storage. They are nonsecret and non-model, but not zero-cost. Each new channel-bias mask likewise adds its padded tensor-size bit source; e.g. a full 224px ResNet stem output mask is `64*128*128=1,048,576` bit coordinates. Corrected cost/performance must be recounted, not inherited from old runs.

The statement codec freezes literal membership **and values** into program identity and rejects replacement. No modified private source was substituted for a public map in the reviewed lowering.

## Independently executed checks

`check_maps.py` implements a separate plain-coordinate contraction and direct multilinear-evaluation reference over the actual base prime `2^64-2^32-31`:

- 158 logical flatten shapes, including all ranks 1..3 with extents 1..5, `[16,5,5]`, and nontrivial rank-4 examples;
- 428 non-Boolean-point checks of the emitted staged Einsum identities, including first-stage free-once transport and subsequent equality-weighted shared-z sums;
- 36 channel-bias shapes across B=1/3/5, C=1/3/5, and non-square padded spatial axes;
- poisoned padding and exact zero output padding in every flatten/bias case.

All passed; see `check_maps.json` and `check_maps.log`. These are independent algebra/index checks, **not Rust PIOP execution, proof acceptance, CUDA execution, or PCS validation**.

## Remaining gates and limits

1. Canonical Conv2D PIOP still explicitly asserts `batch==1` in both direct and polynomial-product branches. Tensor batching is not validated merely by fixing head layout. It must not be conflated with batching independent proofs or deferred weight openings.
2. Actual new PIOP mutation tests are authored but not executed here; the known native CUDA linkage blocker remains. No full-proof/GPU result is inferred.
3. No full ResNet, LeNet, YOLO, PointPillar, VGG, or UNet inference/accuracy result is established. The small helper tests do not establish complete model import/lowering fidelity.
4. Integer no-wrap/ScaleDown soundness beyond the existing finite profiles was not expanded or certified by this layout audit. Layout gather itself is exact over the field (one logical entry selected), not an approximate numerical operation.
5. I flagged stale LeNet weight-dimension documentation; implementation uses column-major `[120,400]`, `[84,120]`, `[10,84]`, not raw ONNX row-major weight bytes. The exact source documentation should state that convention before publication.

## Supplemental review: explicit-batch pad/pool wrappers (2026-10-30 07:22 UTC)

Requested scope: `fold_per_channel2d_input`, `restore_per_channel2d_output`, and their uses by `pad`, `maxpool2d`, and `maxpool_general`. Supplemental snapshot hashes are separately retained in `supplemental-reviewed-source-sha256.json` so the original review snapshot is not overwritten.

### Layout result

The wrappers preserve explicit `[B,C,H,W]` correctly:

1. Raw storage is viewed as column-major `[W,H,C,B]` without changing arity.
2. A public logical mask zeros invalid spatial, C, and B coordinates **before** advice/range consumers can inspect input padding.
3. A registered `ChangeShape` folds to `[Bp*Cp,H,W]`; underlying spatial blocks and their runtime metadata remain consistently 3D.
4. A registered equal-arity `ChangeShape` restores `[B,C,Hout,Wout]`. The original B/C are public shape metadata, not inferred from padded storage or payloads.

The canonical pad/index-map relation treats the folded high bits as a channel coordinate. Restoring the split does not permute those bits, so it commutes with the wrapper views. Pool subsample and replication comparisons receive identical folded input/output layouts, avoiding a mixed 4D/3D Add/Sub broadcast. No wrapper geometry mismatch was found.

Independent `check_batch_wrappers.py` passed **144** honest-forward integer/layout cases: B and C in {1,3,5}; two non-square input sizes per spatial axis; poison ±2^40; padding followed by 2×2/stride2 pooling or 3×3/stride1 general pooling. It checks all padded and logical output coordinates against a separate logical dictionary reference. This is neither Rust proof execution nor proof soundness validation.

### Newly surfaced pre-existing proof blocker: maxpool achievability

`basicblock/maxpool.rs` explicitly documents that only dominance is enforced and true achievability was omitted from the initial port. Both maxpool helpers are committed advice (`advice_kind` in PIOP); their primitive prove/verify methods are trivial. The reviewed builder emits only `Y−X_k≥0` roots, not a condition that Y equals a window entry.

A concrete counterexample is an all-zero logical window with claimed `Y=1`: every difference is 1 and satisfies the current range predicates, though the true maximum is 0. The same freedom can produce nonzero invalid B/C output padding; merely restoring explicit shape does not authenticate zero padding. Thus **the wrappers repair layout but do not prove the maxpool function**. This blocker was reported immediately to the implementer. Honest CPU forwarding that computes true maxima cannot close it.

A possible separate repair is an achievability equation `prod_k(Y−X_k)=0` using binary products and a zero-product root, in addition to dominance and suitable comparison/lift bounds. A separate output mask can bind invalid output padding to zero. Neither change is claimed to be in this supplemental reviewed snapshot.

### Separate integration dependency

The current isolated wrapper branch restores `[B,3,H,W]` after initial pad, while its not-yet-integrated Conv2D builder still infers batch from `B*3 / 4`. For B=3 this gives lead=9 and fails its assertion. A separate worker is extending Conv2D parsing/reduction; that work is outside this supplemental review. The wrapper branch alone is not a working/certified tensor-batched ResNet body until that integration is validated.

Public-mask source costs remain as described above: these masks are dense public arrays and committed one-bit sources, not free metadata-only operations.
