# Rebuttal experiment scripts

The code tree is the fixed build used for the rebuttal reruns. It contains the range-check soundness fix (`certify_rescales`) and the per-channel real weights. It is zk-torch-3 commit c7b92bb plus uncommitted changes. That diff is saved as `zk-torch-4/scripts/bench/morpheus/worktree_final.diff`.

- `zk-torch-4/scripts/bench/morpheus/run_ablations.sh`: Table 3. It checks that all four variants give the same proof hash, then times each variant. `TIMED` picks which variants are timed.
- `zk-torch-4/scripts/bench/morpheus/run_paper.sh one`: Table 2 (DeepProve/EZKL, Morpheus side).
- `zk-torch-4/scripts/bench/morpheus/run_models.sh g1` and `convpp`: Table 5 (1 GPU) and the convolution ablation.
- `census/run_bounds.sh`: the F5 full-depth range census (`morpheus_bounds`, CPU only). `census/aggregate.py` summarizes the outputs.
- `runlogs/resume_f1b.sh`: the F1(a) driver that runs Table 3, Table 2 and Table 5 in order. `runlogs/PARTIAL_REPORT.md` holds the results so far.

The scripts use absolute paths on the A100 server (`/scratch/bjchen4_icgpu/goldilocks/...`).
