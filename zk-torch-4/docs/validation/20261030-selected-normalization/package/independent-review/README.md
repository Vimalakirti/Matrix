# Independent selected-normalization review package

## Run the portable arithmetic check

Requires Python3 with only the standard library; no Rust, CUDA, network access,
or third-party Python packages are needed.

```sh
python check.py /path/to/morpheus-repository
# Also accepted:
python check.py /path/to/morpheus-repository/zk-torch-4
# Or, from either source directory:
python /path/to/this-package/check.py
```

The optional repository argument is used only to report source SHA256 hashes.
The arithmetic itself is independently implemented in this script and imports no
implementation/reference helpers. It checks modular division/lifting, the actual
integer cubic bounds, and the SF30/context9 counterexample. A successful run
prints1,351,857 checks, then hashes the two reviewed implementation files and the
selected-normalization profile document.

`results.txt` is the successful rerun against the final joint source. The
explicit-root and default-current-directory invocations produced byte-identical
output (`optional-path-results.txt`). Only source-root resolution changed when
making the script portable; the entire arithmetic prefix was checked byte-for-
byte against `historical/check-original.py`.

## Files and evidence scope

- `REVIEW.md`: independent selected-normalization source/math review; includes
  findings, their corrections, and precise evidence limitations.
- `JOINT_ADDENDUM.md`: joint InstanceNorm integration and Whisper routing/test
  review, including pinned implementation hashes.
- `check.py`, `results.txt`: portable arithmetic check and final rerun receipt.
- `historical/`: the original fixed-local-path script and first result snapshot.

Absolute paths in historical review prose identify the inspected local snapshots;
they are not runtime dependencies. The script is portable as shown above.

This package is not one of the five whole-paper acceptance reviews, does not
assert historical model accuracy or whole-model numerical equivalence, and does
not supply GPU/PIOP/PCS proof execution. The separate author's exact-source CPU,
typecheck, production-build and replay-link receipts retain their own scopes.
