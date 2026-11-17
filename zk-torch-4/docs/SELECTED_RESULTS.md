# Current reproducible selected-parameter CPU result

This checkpoint addresses numerical correspondence, actual accepted-proof execution and current-profile measurements. It does not add another architecture redesign or treat component audits as whole-paper acceptance.

## Results

The public digits Q10 classifier is 64 → 32 ReLU → 10. Both float and integer variants classify 435/450 holdout rows correctly, with no prediction disagreements. All 450 actual Rust DAG logit vectors agree with the independent integer reference. The small linked-source reproduction package regenerates the dataset, split, model and outputs locally and verifies seven exact hashes; no dataset/model archive is vendored.

Three trials were declared before execution, all using fixed held-out row 1329 (class 1), with fresh OS coins. Every prover and separate public-only verifier completed successfully, including the retained-proof negative controls. These are three repeated proofs of one input, not 450 proofs or a full-model benchmark suite.

- Full serialized proof: 130,885,407 bytes in each trial; Gaussian response payload: 124,007,424 bytes. The JSON provides exact additive wire accounting and each proof/registry hash.
- Prover API: 341.632, 477.654, 214.384 seconds; median 341.632 seconds. This includes request source packing/commitments, PIOP/link/authentication and the embedded link self-check, but excludes model/input registration, forward witness generation and serialization.
- Single accepted-proof verifier API: 62.118, 59.418, 60.736 seconds; median 60.736 seconds. This excludes registry reconstruction, proof I/O/decode/canonical re-encoding and negative controls.
- Zero-based accepted attempts: 1, 2, 0 (respectively 2, 3, 1 candidates). No run was replaced or chosen as fastest. Peak RSS is for whole processes, not individual stages; every process was capped at 3 GiB virtual address space.

The CPU identifies as AMD EPYC 9V74; nine logical CPUs are visible, with four Rayon threads used. This is a shared cloud host, not dedicated cores. Paper build/check activity overlapped run-1 verification; paper compile/render activity overlapped run-2 proving. Run 3 has no known coordinated project overlap, but systemwide quiescence was not verified. No elapsed time is adjusted or subtracted. The public trial record preserves all three values and the caveat.

## Scope and verification

The profile retains rank 16, ring dimension 4096, source dimension 2^27 and the original q/sigma/norm bounds, with 648 hiding and 16 identity columns. Source Gaussian support is derived from the public layout: 1,025 physical columns, not observed private zero entries. This is the reviewed public-support-restricted Gaussian construction with compact transport; it is **not lossless compression of the old full-support Gaussian distribution**. The old 1,326,686,208-byte full response is a formula/reference count, not a completed dense selected run.

Public registries were independently pinned before proof files existed. The standalone verifier used only that public registry and saved proof, with no original private model/input/openings/coins. It runs the same frozen verifier implementation independently, not a second implementation. Native auth/transport/resource checks and the two-file timing delta were separately reviewed; a deterministic mini proof was byte-identical before and after the timing-only change. The final source inventory identifies the exact tested files.

The numerical domain bounds inputs in [0,1024] and parameters in ±16383, with bounded quotients, exact rounding and lowest-index argmax ties. The universal ReLU zero-product no-wrap bound before constraints is (2^21−1)^2 = 4,398,042,316,801 < q; it must not be replaced with an already-constrained honest-activation bound. Original missing training convergence diagnostics remain unavailable; the exact successful training rerun does not retroactively fill them.

These results do not empirically establish SHAKE ideality, OS entropy quality, module hardness, QROM security, side-channel protection or an unconditional 100-bit guarantee. No GPU execution or five whole-paper acceptances is claimed. CUDA compilation is a separate check, not a GPU performance measurement.

## Reproduce

See [CPU build/prove/verify instructions](REPRODUCE_SELECTED.md), [small data-generation package](../repro/digits/README.md), [exact source inventory](SELECTED_SOURCE.json), and [machine-readable results](SELECTED_RESULTS.json). Source links, dependency versions and CC BY 4.0 attribution are provided; reviewers obtain public inputs through the original distribution. No historical log archive, old source snapshot, compiled cache or binary proof is added by this checkpoint.
