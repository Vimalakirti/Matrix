# Fixed learned CNN fixture (public digits)

This is a numerical workload fixture, not evidence that a cryptographic proof has been generated. Native DAG/proof conformance is a separate test.

## Model and results

NCHW 1×8×8 → Conv(8,3×3 valid) → Q10/ReLU → MaxPool(2×2) → Conv(16,2×2 valid) → Q10/ReLU → MaxPool(2×2) → flatten16 → Dense16→10 → Q10/argmax. Both convolution layers and the classifier are learned. Total parameters: **778**. Inference performs **4,800 multiply-accumulates per image** (2,592+2,048+160), excluding bias/activation/pooling.

One float L-BFGS training run with the predeclared seed1729, splitseed20261030 and max400 iterations was performed. It stopped at the iteration limit, **not optimizer convergence**; diagnostics are retained. No model reselection, clipping, retraining or test-based tuning occurred. Max absolute quantized parameter is2613, admitted under the predeclared16383 cap before evaluation.

The frozen float and Q10 models both classify **434/450 (96.4444%)** held-out examples correctly, with **zero prediction disagreements**. All16 wrong cases remain in `report.json`. This is our stratified split of sklearn's1,797-row subset, not the original UCI writer-disjoint benchmark. The first four fixed test rows are1329,1379,1319,909; no example is replaced based on correctness.

## Reproduction without vendored data/models

Use a fresh Python environment with `requirements.txt`, then:

```
python reproduce.py /absolute/path/to/a-new-output-directory
```

The official pinned scikit-learn distribution supplies the public dataset. No separate dataset upload or model download is required. The wrapper requires a nonexistent output directory, pinned versions and single-thread BLAS. It runs the gradient checks, trains once, freezes/admits the model before evaluating the test set, and checks all seven exact expected artifact hashes. A mismatch is retained and reported, never fixed by searching seeds. Floating optimization can differ across platforms/BLAS builds; the pins identify this recorded experiment, not a proof of cross-platform floating determinism.

For development, `train_export.py` alone refuses an existing frozen model but does not enforce the wrapper's entire fresh-directory/version policy. `evaluate_frozen.py` refuses an existing report. Use the wrapper for reproducibility. Training duration and optimizer logs are diagnostic, not proving performance.

Only scripts, requirements, plan, hash pins, README and provenance need publication. Generated datasets, weights,450 traces, logs and model outputs remain local outputs. See `PROVENANCE.md` for source/version/license.

## Layout and numerical semantics

- `conv1_weight`: nested OIHW[8][1][3][3], `conv1_bias`[8].
- `conv2_weight`: nested OIHW[16][8][2][2], `conv2_bias`[16].
- `fc_weight`: nested OI[10][16], `fc_bias`[10]. A Rust input-major dense tensor must transpose explicitly; do not reinterpret flatten order.
- Convolution is cross-correlation, valid padding, stride1; pooling windows use row-major first-index tie selection. Final argmax chooses the lowest class index on ties.
- Parameters use floor(float_parameter×1024+0.5). After each integer dot: floor((dot+512)/1024), then add the Q10 bias. ReLU follows convolution bias. The dense output has no ReLU.
- `integer_reference.py` independently implements scalar Python loops, not the training einsum/gradient code. All450 test outputs include intermediate dots, quotients, biases, activation/pool results and final logits.
- Before training, finite differences checked100 scalar coordinates plus10 directional derivatives on a positive-hidden non-tie synthetic batch, then40 mixed-sign coordinates. ReLU derivative is0 at0; pool gradients use the first maximizing index. These are gradient checks, not a proof of all training code.

The complete admitted-domain field/no-wrap certificate and native operator mapping are supplied separately by the numerical audit; observed small parameter values do not replace the public±16383 bound.
