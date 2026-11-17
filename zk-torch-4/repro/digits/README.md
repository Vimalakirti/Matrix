# Reproduce the public digits workload without vendored data

This directory supplies only scripts, plans, versions, hashes and source attribution. It does not contain the dataset, trained weights, test inputs, proof files, build artifacts or historical logs.

## Source and license

Use Python 3.12.14 and the versions pinned in requirements.txt. Installing scikit-learn 1.8.0 from its official PyPI distribution supplies the bundled `load_digits(n_class=10)` dataset; no separate dataset archive is needed. See https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html and https://pypi.org/project/scikit-learn/1.8.0/ . Original source: Alpaydin and Kaynak (1998), UCI Optical Recognition of Handwritten Digits, https://doi.org/10.24432/C50P49 , licensed CC BY 4.0 at https://creativecommons.org/licenses/by/4.0/ . PROVENANCE.md gives attribution and transformations.

This uses the 1,797-example sklearn copy, not all 5,620 UCI examples. A new stratified 1,347/450 split is not the original UCI train/test split or writer-disjoint evaluation.

## Regenerate and verify

From this directory, in a fresh Python 3.12.14 environment:

```
python -m pip install -r requirements.txt
python reproduce.py --output /tmp/morpheus-trained-digits
```

Use reproduce.py as the entry point; train_export.py is its internal helper and does not provide the same existing-directory protection when invoked directly. The output directory must not already exist. For an existing separately generated fixture, use `python reproduce.py --verify-only --output PATH`. The script does not install packages itself. It runs exactly one declared fit, using single-threaded BLAS, and fails if any of seven generated artifacts differs from the frozen SHA256 pins. It never searches training seeds or substitutes a different model.

The Rust proof utility currently reads the three local generated files model.json, test_inputs.json and integer_test_outputs.json from `/tmp/morpheus-trained-digits`. To avoid changing its pinned source, regenerate at that path in a fresh isolated environment. Do not overwrite unrelated existing files.

Declared split seed 20261030; training seed 1729;64 → 32 ReLU → 10, trained with LBFGS; Q10 nearest rounding with half ties toward positive infinity. The original 8191 parameter bound rejected the learned maximum 9116 before held-out evaluation; 16383 was adopted without retraining/clipping or changing examples. The original plan and profile_decision.json retain that sequence.

The public Q10 model hash is 7620a762dc49e1bd1ea08b9ff0c0193e0826fd590a337829083bc4064db352d9. A single same-environment rerun reproduced all seven files byte-for-byte: dataset, split, float weights, quantized weights, test inputs, all 450 integer traces and fixed examples. Both models classify 435/450 correctly with zero disagreements. Proof rows remain 1329,1379,1319,909; no mistaken example was discarded.

Reproduction was checked on Linux x86_64 / glibc 2.41 with OpenBLAS 0.3.30 and the pinned Python packages. A different numerical backend/platform may produce a different LBFGS fit. Preserve such failure; do not tune until the expected model appears. Hash agreement, not a similar accuracy, identifies this benchmark.

The original first export stopped on the rejected parameter bound before iteration/loss/warning diagnostics were saved. Those original convergence diagnostics remain unavailable. The successful reproduction's new diagnostics do not retroactively fill that gap.

## Proof execution scope

Use the repository's accompanying current-source build/prove/verify instructions after generating this fixture. Training seeds are public reproducibility parameters, not cryptographic private coins. Actual selected proof generation uses fresh OS entropy; its wire bytes need not match a previous run. Public registry identity must be independently pinned before accepting a proof. Execution success does not establish the conditional hardness/setup/random-oracle or side-channel assumptions.
