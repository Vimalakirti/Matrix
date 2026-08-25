"""The evaluation model zoo: paper row name -> binary, env, config file.

Every binary takes the global config YAML as its first positional argument
(`zk_torch_4::CONFIG_FILE` reads `args[1]`). With no argument the built-in
default applies (scale_factor_log 10, table_size_log 20).

CONFIG CHOICE IS REPORTABLE. `bench_config.yaml` sets `table_size_log: 10`
and says so itself: "Benchmark-only; production uses 20". A 2^10 range table
does not cover realistic activation ranges, and a run whose values leave the
table prints a "WILL fail to verify" warning (the harness flags this as
`range_table_overflow`). Section 9.1 must state the config used per model.
Prefer the largest table the configuration can afford.

Knob names are not uniform across bins: yolo sizes by NUM_STAGES, unet3d by
INPUT_D/H/W, pointpillar by NX/NY/N_PILLARS, whisper by N_ENC/N_DEC plus
context lengths. Several bins also default to full-scale inputs
(yolo INPUT_SIZE=640, whisper N_AUDIO_CTX=1500, bench_streaming_resnet
INPUT_SIZE=224), so an omitted knob is not a small run.

Three profiles:
  smoke  - seconds per model. Validates the pipeline, NOT paper numbers.
  paper  - reduced but non-trivial; what fits a 4-GPU node today.
  full   - the true published model sizes (see _*_FULL below). This is the
           target for the paper; on a small node many rows will not fit, and
           the harness records that rather than hiding it.
"""

from __future__ import annotations

from typing import Dict, List

from harness import Case

BENCH = "bench_config.yaml"        # table 2^10: fast, benchmark-only
CV = "cv_config.yaml"              # sf 16, table 2^18
LLAMA = "llama2_config.yaml"       # table 2^12, for hidden >= 2048

# --------------------------------------------------------------------------
# transformer / language models
# --------------------------------------------------------------------------

_TRANSFORMERS_PAPER = [
    # name,               binary,   env,                                              config
    ("GPT-2 (12L, seq 64)",  "gpt2",   {"NUM_LAYERS": "12", "SEQ_LEN": "64",
                                        "MAX_NUM_VARS": "22"},                          BENCH),
    ("GPT-2 (12L, seq 256)", "gpt2",   {"NUM_LAYERS": "12", "SEQ_LEN": "256",
                                        "MAX_NUM_VARS": "24"},                          BENCH),
    ("BERT (12L, seq 64)",   "bert",   {"NUM_LAYERS": "12", "SEQ_LEN": "64",
                                        "MAX_NUM_VARS": "22"},                          BENCH),
    ("Llama-2 (8L, seq 64)", "llama2", {"NUM_LAYERS": "8", "SEQ_LEN": "64",
                                        "MAX_NUM_VARS": "27"},                          LLAMA),
    # llama3's own MAX_NUM_VARS default is 29; do not lower it, the full
    # vocab (128256) and ffn (14336) leaves need the headroom
    ("Llama-3 (8L, seq 64)", "llama3", {"NUM_LAYERS": "8", "SEQ_LEN": "64"},            LLAMA),
    ("GPT-J (8L, seq 64)",   "gptj",   {"NUM_LAYERS": "8", "SEQ_LEN": "64",
                                        "MAX_NUM_VARS": "27"},                          LLAMA),
]

# --------------------------------------------------------------------------
# convolutional / vision / point-cloud models
# --------------------------------------------------------------------------

_VISION_PAPER = [
    ("ResNet-50 (53 conv)",  "resnet",      {"NUM_LAYERS": "53", "INPUT_SIZE": "32"},   CV),
    ("VGG-16 (13 conv)",     "vgg",         {"VGG_VARIANT": "16", "NUM_LAYERS": "13",
                                             "INPUT_SIZE": "64"},                       CV),
    # yolo sizes by NUM_STAGES, not NUM_LAYERS; its INPUT_SIZE default is 640
    ("YOLO",                 "yolo",        {"NUM_STAGES": "4", "INPUT_SIZE": "64"},    CV),
    ("3D-UNet",              "unet3d",      {"INPUT_D": "16", "INPUT_H": "16",
                                             "INPUT_W": "16"},                          CV),
    ("PointPillars",         "pointpillar", {},                                          CV),
]

# --------------------------------------------------------------------------
# heterogeneous: convolutional front end + transformer encoder-decoder
# --------------------------------------------------------------------------

_HETERO_PAPER = [
    # BENCH, not LLAMA: at the default n_state=384 the LayerNorm mean tolerance
    # is 192, well inside BENCH's 2^10 table, and LLAMA's table_commit_log=12
    # would put the sparse aux for the encoder MLP (32x2048 = 2^16 entries) at
    # arity 16+12 = 28. The fold tree sends one final witness per arity bucket
    # in the clear, so that single bucket costs 872 MB and the whole proof
    # 1.75 GB instead of 115 MB. Whisper large-v3 keeps LLAMA below: n_state
    # =1280 needs a table wider than 2^10.
    ("Whisper (1+1L)", "whisper", {"NUM_ENC_LAYERS": "1", "NUM_DEC_LAYERS": "1",
                                   "N_MELS": "16", "N_AUDIO_CTX": "64",
                                   "N_TEXT_CTX": "16"},                                 BENCH),
    ("Whisper (2+2L)", "whisper", {"NUM_ENC_LAYERS": "2", "NUM_DEC_LAYERS": "2",
                                   "N_MELS": "16", "N_AUDIO_CTX": "64",
                                   "N_TEXT_CTX": "16"},                                 BENCH),
]

# --------------------------------------------------------------------------
# smoke profile: same binaries, tiny configurations
# --------------------------------------------------------------------------

_SMOKE = [
    ("GPT-2 (1L, seq 1)",   "gpt2",        {"NUM_LAYERS": "1", "SEQ_LEN": "1"},        BENCH),
    ("BERT (1L, seq 1)",    "bert",        {"NUM_LAYERS": "1", "SEQ_LEN": "1"},        BENCH),
    ("Llama-2 (1L, seq 1)", "llama2",      {"NUM_LAYERS": "1", "SEQ_LEN": "1"},        LLAMA),
    ("ResNet-50 (3 conv)",  "resnet",      {"NUM_LAYERS": "3", "INPUT_SIZE": "32"},    CV),
    ("VGG-16 (3 conv)",     "vgg",         {"VGG_VARIANT": "16", "NUM_LAYERS": "3",
                                            "INPUT_SIZE": "32"},                        CV),
    ("YOLO (2 stages)",     "yolo",        {"NUM_STAGES": "2", "INPUT_SIZE": "32"},    CV),
    ("3D-UNet (16^3)",      "unet3d",      {},                                          CV),
    ("PointPillars",        "pointpillar", {},                                          CV),
    ("Whisper (1+1L)",      "whisper",     {"NUM_ENC_LAYERS": "1", "NUM_DEC_LAYERS": "1",
                                            "N_MELS": "16", "N_AUDIO_CTX": "32",
                                            "N_TEXT_CTX": "8"},                         BENCH),
]


# --------------------------------------------------------------------------
# TRUE model sizes (the configurations the paper claims to cover).
#
# These are the published architectures, not reduced stand-ins. Sequence and
# input sizes follow the MLPerf Inference v6.0 task where one is defined
# (BERT-Large/SQuAD seq 384, 3D-UNet/KiTS19 128^3, ResNet-50 and VGG-16
# 224^2, YOLO 640^2), otherwise the model's own context length.
#
# Dimensions the binary hardcodes are noted; everything else is set here.
# GPT-2 is Small (hidden 768 = 12x64, ffn 3072) and BERT is Large
# (hidden 1024 = 16x64, ffn 4096, 2 classes) inside the binaries themselves.
#
# Expect these to exceed a 4x80GB node. MAX_NUM_VARS and the *_SHARDS knobs
# are the two things to tune per model when a run does not fit: shards split
# an oversized leaf, MAX_NUM_VARS must cover input_n + table_size_log for the
# largest auxiliary. Failures are recorded rather than hidden, so start here
# and walk knobs until every row verifies.
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Per-model lookup-table geometry (ZK4_TABLE_SIZE_LOG / ZK4_TABLE_COMMIT_LOG).
#
# table_commit_log has a per-MODEL optimum, not a per-family one. Each sparse
# lookup aux is split into ceil(tsl/tcl) chunks at arity input_n + tcl, and the
# fold tree runs GPU same-point only at arity <= 24. So the target is
#
#     tcl = 24 - max_input_n
#
# and max_input_n is a shape property, printed per model by
# `ZK4_SHAPE_REPORT=1 ./target/release/<bin> <config>`. Measured across this
# profile it spans 16 (VGG-16) to 26 (Llama-3-8B), which is why one value per
# config file cannot serve every row sharing that file.
#
# Do NOT try to fix the above-cap models by raising ZK4_GPU_SP_MAX_ARITY. The
# GPU kernel is dense over 2^arity while the CPU path above the cap is sparse;
# measured on llama2 8L/seq64, raising the cap cost 24% at arity 26 and 39% at
# 28. Lower the arity instead. Even when the cap is unreachable, lower arity
# still wins on CPU (fold tree 21.8s at arity 26 vs 39.3s at 28), so those rows
# take the smallest practical tcl, traded off against chunk count.
#
# table_size_log is the other half: it sets range COVERAGE and only changes the
# chunk count. Measured free on llama2 (12 -> 16 was inside run-to-run noise),
# so it is chosen for correctness margin: 16 for LLMs, 12 under evaluation for
# CV (which runs scale_factor_log 16, so 12 still needs a coverage check).
#
# VALIDATED = measured optimum. DERIVED = from the rule, not yet measured at
# full size; the reduced-size llama2 sweep is the only end-to-end confirmation
# (max_input_n 18 -> tcl 6 predicted and measured best, 2.6x over tcl 12).
# --------------------------------------------------------------------------

# tcl=4, not 6. Fold-tree memory AND proof size scale as 2^arity =
# 2^(max_input_n + tcl), which dominates the chunk-count term a larger tcl
# saves on -- the opposite of the "fewer chunks is safer" intuition. Measured:
# GPT-J at tcl 6 (arity 31) was host-OOM-killed (SIGKILL) during prove on a
# 905GB box; GPT-2 at tcl 6 (arity 30) verified but peaked 191.8GB with a
# 6.86GB proof; PointPillars at tcl 2 (arity 27) beat tcl 8 (arity 33) on BOTH
# time (568.6s vs 724.5s) and memory (50.7 vs 52.9GB). Lower wins on every axis
# until the rising chunk count starts costing range-lookup time.
_TRANSFORMERS_FULL = [
    ("GPT-2 Small (12L, seq 1024)", "gpt2",
     {"NUM_LAYERS": "12", "SEQ_LEN": "1024", "MAX_NUM_VARS": "26",
     # max_input_n=24; MEASURED tcl sweep at tsl=24, 4xA100: tcl 6 (arity 30) =
     # 972.5s prove / 6.86GB proof / 33.8s verify; tcl 4 (arity 28) = 872.5s /
     # 1.19GB / 9.2s <- BEST; tcl 2 (arity 26) = host OOM in leaf build (12
     # chunks at tsl 24). Narrow window: arity cost above, chunk-count memory
     # below. Proof size and verify time are far more sensitive to tcl than
     # prove time is (5.8x and 3.7x vs 1.11x).
     "ZK4_TABLE_SIZE_LOG": "24", "ZK4_TABLE_COMMIT_LOG": "4",},              BENCH),
    ("BERT-Large (24L, seq 384)", "bert",
     {"NUM_LAYERS": "24", "SEQ_LEN": "384", "MAX_NUM_VARS": "26",
     # max_input_n=22. Its earlier range_table_overflow was NOT a coverage problem:
     # it was the llama_rms_norm outer-product bug (fixed), whose off-diagonal
     # garbage reached 1049088 and read as 'needs tsl>=21'. After the fix,
     # MEASURED at tcl=4: tsl 10 = 452.2s/44.0GB, tsl 16 = 465.0s/122.9GB,
     # tsl 24 = 470.1s/98.5GB. The shipped tsl=10 is best, so keep it. tsl is
     # ~free in TIME but NOT in memory, since chunks = ceil(tsl/tcl).
     "ZK4_TABLE_SIZE_LOG": "10", "ZK4_TABLE_COMMIT_LOG": "4",},               BENCH),
    ("Llama-2-7B (32L, seq 512)", "llama2",
     {"NUM_LAYERS": "32", "SEQ_LEN": "512", "NUM_HEADS": "32", "HEAD_DIM": "128",
      "FFN_DIM": "11008", "VOCAB": "32000", "LOGITS_SHARDS": "32",
      "FFN_SHARDS": "16", "MAX_NUM_VARS": "27",
     # max_input_n=23; tsl 24, tcl 6 = 4 chunks (arity 29)
     "ZK4_TABLE_SIZE_LOG": "24", "ZK4_TABLE_COMMIT_LOG": "4",},                                 LLAMA),
    ("Llama-3-8B (32L, seq 512)", "llama3",
     {"NUM_LAYERS": "32", "SEQ_LEN": "512", "HIDDEN_DIM": "4096",
      "NUM_HEADS": "32", "NUM_KV_HEADS": "8", "HEAD_DIM": "128",
      "FFN_DIM": "14336", "VOCAB_SIZE": "128256",
     # max_input_n=26; tsl 24, tcl 6 = 4 chunks (arity 32). Largest row here; expect the memory ceiling first
     "ZK4_TABLE_SIZE_LOG": "24", "ZK4_TABLE_COMMIT_LOG": "4",},                               LLAMA),
    ("GPT-J-6B (28L, seq 512)", "gptj",
     {"NUM_LAYERS": "28", "SEQ_LEN": "512", "NUM_HEADS": "16", "HEAD_DIM": "256",
      "FFN_DIM": "16384", "VOCAB": "50400", "MAX_NUM_VARS": "27",
     # max_input_n=25; tsl 24, tcl 6 = 4 chunks (arity 31)
     "ZK4_TABLE_SIZE_LOG": "24", "ZK4_TABLE_COMMIT_LOG": "4",},               LLAMA),
]

_VISION_FULL = [
    ("ResNet-50 (53 conv, 224^2)", "resnet",
     {"NUM_LAYERS": "53", "INPUT_SIZE": "224", "MAX_NUM_VARS": "28",
     # max_input_n=20; MEASURED 28.8s vs 37.2s at the shipped tsl18/tcl8 = 1.29x. tcl 4 puts the top bucket on the arity-24 GPU cap
     "ZK4_TABLE_SIZE_LOG": "12", "ZK4_TABLE_COMMIT_LOG": "4",},            CV),
    ("VGG-16 (13 conv, 224^2)", "vgg",
     {"VGG_VARIANT": "16", "NUM_LAYERS": "13", "INPUT_SIZE": "224",
      "MAX_NUM_VARS": "28",
     # max_input_n=16; MEASURED 4.50s, identical to shipped. tcl already optimal there, so this row only varies tsl (18->12) and is the control showing table_size_log is free
     "ZK4_TABLE_SIZE_LOG": "12", "ZK4_TABLE_COMMIT_LOG": "8",},                                                     CV),
    ("YOLO (640^2)", "yolo",
     {"NUM_STAGES": "4", "INPUT_SIZE": "640", "MAX_NUM_VARS": "28",
     # max_input_n=22; MEASURED 101.7s vs 100.8s shipped = NO GAIN, and peak GPU 48.7GB vs 21.7GB. The arity rule predicts a win here and does not deliver one; fold-tree arity is evidently not YOLO's bottleneck. Revisit before trusting this value
     "ZK4_TABLE_SIZE_LOG": "12", "ZK4_TABLE_COMMIT_LOG": "2",},             CV),
    ("3D-UNet (KiTS19, 128^3)", "unet3d",
     {"INPUT_D": "128", "INPUT_H": "128", "INPUT_W": "128",
      "MAX_NUM_VARS": "28",
     # max_input_n=26; BROKEN at 128^3 on 4xA100 regardless of table geometry - panics in poly/dense.rs 'device->host copy failed: MemcpyFailed' at BOTH tsl12/tcl2 and the shipped tsl18/tcl8. Not a config problem
     "ZK4_TABLE_SIZE_LOG": "12", "ZK4_TABLE_COMMIT_LOG": "2",},                                                     CV),
    ("PointPillars (KITTI)", "pointpillar",
     {"NX": "432", "NY": "496", "N_PILLARS": "12000", "MAX_POINTS": "32",
      "NUM_ANCHORS": "2", "NUM_CLASSES": "3", "MAX_NUM_VARS": "28",
     # max_input_n=25; MEASURED 568.6s vs 724.5s shipped = 1.27x. Above the GPU cap at every tcl, so this is pure arity reduction on the CPU sparse path (arity 27 vs 33)
     "ZK4_TABLE_SIZE_LOG": "12", "ZK4_TABLE_COMMIT_LOG": "2",},             CV),
]

_HETERO_FULL = [
    ("Whisper large-v3 (32+32L)", "whisper",
     {"NUM_ENC_LAYERS": "32", "NUM_DEC_LAYERS": "32", "N_MELS": "128",
      "N_AUDIO_CTX": "1500", "N_TEXT_CTX": "448", "N_STATE": "1280",
      "N_HEAD": "20", "MAX_NUM_VARS": "28",
     # max_input_n unmeasured (dag build >900s); tsl 24, tcl 6 = 4 chunks
     "ZK4_TABLE_SIZE_LOG": "24", "ZK4_TABLE_COMMIT_LOG": "4",},                                     LLAMA),
]

GROUPS = {
    "full": {"Transformer": _TRANSFORMERS_FULL,
             "Convolutional": _VISION_FULL,
             "Heterogeneous": _HETERO_FULL},
    "paper": {"Transformer": _TRANSFORMERS_PAPER,
              "Convolutional": _VISION_PAPER,
              "Heterogeneous": _HETERO_PAPER},
    "smoke": {"Smoke": _SMOKE},
}


def zoo(profile: str = "paper", timeout_s: int = 3600) -> List[Case]:
    if profile not in GROUPS:
        raise SystemExit(f"unknown profile {profile!r}; try {list(GROUPS)}")
    cases: List[Case] = []
    for group, entries in GROUPS[profile].items():
        for name, binary, env, config in entries:
            cases.append(Case(name=name, binary=binary, env=dict(env),
                              config=config, group=group, timeout_s=timeout_s))
    return cases


def binaries(profile: str = "paper") -> List[str]:
    return [c.binary for c in zoo(profile)]


# Streaming counterparts, for the deferred-weight-opening ablation. These
# binaries prove N inferences of one model and print the per-proof amortized
# cost plus the one-time finalize.
STREAMING: Dict[str, tuple] = {
    "GPT-2":        ("bench_streaming_gpt2",   {"NUM_LAYERS": "12", "SEQ_LEN": "64"},  BENCH),
    "Llama-2":      ("bench_streaming_llama2", {"NUM_LAYERS": "8", "SEQ_LEN": "64",
                                                "MAX_NUM_VARS": "27"},                  LLAMA),
    "BERT":         ("bench_streaming_bert",   {"NUM_LAYERS": "12", "SEQ_LEN": "64"},  BENCH),
    # bench_streaming_resnet defaults INPUT_SIZE to 224 and caps magnitudes
    # with ZK4_WMAG / ZK4_XMAG, which the monolithic bin does not have
    "ResNet-50":    ("bench_streaming_resnet", {"NUM_LAYERS": "53", "INPUT_SIZE": "32"}, CV),
    # bench_streaming_vgg hardcodes VGG-16 "paper" style: no VGG_VARIANT knob
    "VGG-16":       ("bench_streaming_vgg",    {"NUM_LAYERS": "13", "INPUT_SIZE": "64"}, CV),
    "Whisper":      ("bench_streaming_whisper", {"NUM_ENC_LAYERS": "1", "NUM_DEC_LAYERS": "1",
                                                 "N_MELS": "16", "N_AUDIO_CTX": "32",
                                                 "N_TEXT_CTX": "8"},                     LLAMA),
}


# --------------------------------------------------------------------------
# Technique bundle for the end-to-end table.
#
# Section 9.2 should report the system as the paper describes it, so the
# prover-side techniques are all enabled: sparse opening paths, the
# device-resident fold tree, device-resident witnesses feeding the commitment
# kernels, terminal-opening sharding over the device pool, and backward-pass
# graph partitioning.
#
# Deferred weight opening is deliberately NOT here. It defers work to a
# finalize that only exists across a stream, and a single proof with deferral
# prints "Verified (modulo N deferred constant claims)", which is not a
# standalone pass. Its amortization is what run_deferred.py measures.
# --------------------------------------------------------------------------

def techniques(gpus: int, forward: str = "gpu") -> Dict[str, str]:
    env = {
        "ZK4_SPARSE_SP": "1",            # sparse same-point opening
        "ZK4_SPARSE_BOOL": "1",          # sparse boolean-consistency prover
        "ZK4_SHARED_EQ": "1",            # shared eq across leaves
        "ZK4_DEVICE_RESIDENT_FOLD": "1",  # device-resident fold tree
        # Phase timings on stderr. Costs nothing to produce and is the only
        # way the CSV can say how much of prover time is sumcheck (lookup vs
        # not) versus PCS -- the same split the proof-size breakdown reports,
        # so the two are directly comparable.
        "ZK4_TIMING": "1",
    }
    if forward == "gpu":
        # Keeps witness tables on the device so the commitment kernels consume
        # them without a host round trip. Not a claimed technique in the paper
        # (which reports prover performance, not the forward pass), but it is
        # how the prover is meant to run. ZK4_GPU_FWD_DBG=1 cross-checks every
        # GPU op against its CPU counterpart if a shape ever desynchronizes
        # again.
        env["ZKT_RUN_BACKEND"] = "gpu"
    if gpus >= 1:
        env["ZK4_GPU_DEVICES"] = ",".join(str(i) for i in range(gpus))
    if gpus > 1:
        env["NUM_PARTITIONS"] = str(gpus)
    return env
