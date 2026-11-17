# Trained CNN: selected CPU results

Current work focuses on correctness and reproducibility. Performance evaluation is deferred to a separate host; the existing local timing values below are retained diagnostics from the original direct-rustc executable, not new benchmark claims. No timing instrumentation or new prover trials were added for this update.

This is separate from the retained MLP results. Frozen CNN v3 source manifest SHA256: `6cd8ebd1a82e4822f2a8761a0cbe8f28c21ea4d5a6c85e80ea6fad1d5c4d3f1f`. Full executable, profile-source, fixture and per-case proof/registry hashes and all registration stages are in `CNN_RESULTS.json`. Profile-source hash identifies source, not a serialized profile digest.

| Case | Proof bytes | Public registry bytes | Prover API s | Valid verifier API s | Accepted index / total attempts |
|---|---:|---:|---:|---:|---:|
| B1/default |165443423|1903907|909.022606305|106.352757249|4 / 5|
| B4/default |186656735|2703419|531.237030601|138.012240951|1 / 2|
| B16/default |273681503|5901467|1406.853794226|234.414718342|3 / 4|
| B4/pp |190304927|2703663|727.472880335|136.228523973|2 / 3|

All four rows passed owner public-only saved-wire verification and isolated negative suites (12 groups for B1, 14 for each other case). B1 and B4/PP also passed separately invoked public-only replays against the independently pinned registries, with all controls. B4/default and B16/default have independent evidence/hash checks but no second heavyweight replay. These replays use the same verifier implementation, not a second cryptographic implementation.

Two earlier independent PP verifier-only attempts lost execution-session access without terminal receipts. Both remain locally preserved and are classified as incomplete, not as verification failures or successes. The final recovery replay used the unchanged binary, wire, pinned registry and controls. No prover trial was repeated or replaced; all recovery timings are diagnostic and excluded from the benchmark rows.

At batch4, default mode saves 3,648,192 wire bytes (1.92%) against PP. Of this, 3,637,248 bytes are the 96-column response difference; registry sizes are reported separately. This is a fixed-workload size comparison, not a total-time speedup.

Each row is one predeclared fresh-process case, not repeated trials or a broad scalability estimate. Authentication retries use fresh unrecorded OS randomness; raw wall times are not normalized for different attempt counts. Registration is fresh per case, separately timed, and NOT amortized. Proof bytes exclude the required independently pinned public registry. Prover API includes source packing/commitment and embedded link self-check; valid verifier API excludes wire reading/decoding and negative controls. Existing-DAG witness replay excludes compilation/import. Shared-host timings do not establish an unqualified default/PP speedup.

Only four source files differ from the published MLP checkpoint: three CNN harness files and the audited exact-zero/support-count/move-not-clone link evaluator repair. Earlier MLP source identity and measurements remain historical, not silently rerun on this repair. Initial mini OOM is retained locally and is a resource failure, not a verifier rejection.

The public numerical domain uses input [0,1024] and model coefficients ±16383; training stopped at the declared iteration limit, not convergence. See source-linked reproduction for the full model/accuracy caveats. Raw model storage padding is not blanket zero-enforced: logical operations mask unused model entries; PP binds full convolution advice and its zero tails through its polynomial identity. Static correspondence is not unconditional concrete-SHAKE security, whole-compiler refinement, a private-entropy audit, or GPU execution evidence.
