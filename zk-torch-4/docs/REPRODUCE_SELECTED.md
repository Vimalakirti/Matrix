# Build and reproduce the current selected CPU instance

These instructions use the repository source, not a bundled build cache or binary. Dataset/model files are generated locally by the small linked-source digits reproduction package. No GPU or CUDA installation is needed for this CPU route. The tested Rust toolchain is rustc 1.99.1 / cargo 1.99.1.

## Build

From the repository root, after installing an appropriate Rust toolchain:

```
export CARGO_INCREMENTAL=0
export CARGO_PROFILE_DEV_DEBUG=0
export CARGO_TARGET_DIR="$PWD/target/reviewer-cpu"
cargo build --locked --manifest-path zk-torch-4/Cargo.toml \
  --no-default-features --features cpu-replay \
  --bin morpheus_trained_digits --message-format=json \
  --config 'profile.dev.package.zk-torch-4.opt-level=3' \
  --config 'profile.dev.package.almost-goldilocks-cuda.opt-level=3' \
  --config 'profile.dev.package.keccak.opt-level=3' \
  --config 'profile.dev.package.sha3.opt-level=3' \
  --config 'profile.dev.package.num-bigint.opt-level=3' \
  --config 'profile.dev.package.num-integer.opt-level=3' \
  --config 'profile.dev.overflow-checks=true' > build.jsonl
```

Keep the Cargo compiler-artifact receipt. The executable is the `morpheus_trained_digits` target in that exact receipt, normally `$CARGO_TARGET_DIR/debug/morpheus_trained_digits`. Do not select a shared library or executable by newest modification time. Dependency optimization is part of the measured configuration; overflow checks stay enabled.

## Regenerate the public input fixture

Follow the adjacent digits reproduction README to create `/tmp/morpheus-trained-digits` from the pinned scikit-learn distribution. Seven SHA256 checks identify the exact dataset/split/model/output fixture. Do not substitute a different fit when a hash check fails.

## Preflight and prove

Use a new output directory on a Linux machine with at least 3.5 GiB available RAM and 512 MiB free destination storage, without competing heavy jobs. The demonstration uses the existing selected parameter profile and a fixed public layout; it does not lower security parameters to fit memory. The `RUN_ADMITTED` flag acknowledges the resource check, not a security certificate.

```
BIN="$CARGO_TARGET_DIR/debug/morpheus_trained_digits"
DIGITS_SELECTED_COMPACT=1 DIGITS_PREFLIGHT_ONLY=1 "$BIN"
mkdir selected-proof || exit 1
(
  ulimit -v 3145728
  env -u MORPHEUS_COIN_SEED -u DIGITS_CENSUS \
      -u DIGITS_VERIFY -u DIGITS_REGISTRY_ONLY -u DIGITS_PREFLIGHT_ONLY \
      RAYON_NUM_THREADS=4 DIGITS_SELECTED_COMPACT=1 \
      DIGITS_SELECTED_RUN_ADMITTED=1 DIGITS_MEASURE=1 \
      DIGITS_PROVE="$PWD/selected-proof" "$BIN" \
      > selected-prove.log 2> selected-prove.stderr
)
```

Do not enable the private diagnostic census when testing confidentiality. Cryptographic coins are fresh 64-byte OS entropy and are not exported. Public training seeds are unrelated. A new proof normally has a different wire hash. Keep failures and all originally configured rejection attempts; do not report a prover abort as verifier rejection or repeatedly select the fastest successful trial.

## Independently pin and verify

A real verifier must obtain model/input registration from its trusted registration procedure, not trust a replacement registry merely because the prover supplied it. For the recorded functional audit, an independent Linux `IN_CLOSE_WRITE` watcher pinned the completed public registry before the proof file existed, then separately validated canonical registry metadata. The proof file and that previously pinned registry were copied into an otherwise empty directory. This is an execution test, not an identity-authentication service.

After independently obtaining the trusted registry and completed wire, put only `public-registry.bin` and `request-0.bin` in a fresh verification directory:

```
(
  ulimit -v 3145728
  env -u MORPHEUS_COIN_SEED -u DIGITS_PROVE -u DIGITS_CENSUS \
      -u DIGITS_REGISTRY_ONLY -u DIGITS_PREFLIGHT_ONLY \
      RAYON_NUM_THREADS=4 DIGITS_SELECTED_COMPACT=1 DIGITS_MEASURE=1 \
      DIGITS_VERIFY="$PWD/public-verification" "$BIN" \
      > selected-verify.log 2> selected-verify.stderr
)
```

The hardened saved-wire mode returns before reading training fixture files. It reconstructs canonical public semantics, erases dummy values, and uses no original private model, input, openings or coins. It also runs real retained-proof mutation checks; whole-process verification time therefore is not single-proof acceptance latency. `MEASURE stage=single_valid_verify_api` measures the accepted proof API call, excluding proof read/decode/canonical re-encoding and all negative controls. The `valid_*` components partition parts of that call. Proving API time explicitly includes request source packing/commitments and the embedded link self-check; model/input registration, encoding and file writes are reported separately. The single-case witness stage replays an already constructed DAG, so it is not model import/compilation. The listed stages do not exhaustively partition total process time. The independent audit used the same implementation in a separate process, not a separately implemented verifier.

## What the result establishes

This runs the public-support-restricted Gaussian construction and compact transport on the selected parameters. It is not lossless compression of the former full-support Gaussian distribution. Conditional module-hardness, setup, random-oracle, OS-entropy and side-channel assumptions are not proven by runtime success. No GPU execution, legacy-model-wide result or whole-paper acceptance is inferred. Concise recorded results and exact source identities accompany these instructions; raw historical archives are deliberately not vendored.
