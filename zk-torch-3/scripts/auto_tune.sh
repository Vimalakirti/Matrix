#!/usr/bin/env bash
# auto_tune.sh — Build and run the Rust microbenchmark for GPU threshold tuning.
#
# Usage:
#   bash scripts/auto_tune.sh
#
# This replaces the old shell-script-based threshold sweep with a single
# Rust binary that directly measures CPU vs GPU crossover points (~2 min).
set -euo pipefail

cd "$(dirname "$0")/.."
cargo build --release --bin bench_thresholds 2>&1 | tail -1
./target/release/bench_thresholds "$@"
