#!/usr/bin/env python3
"""Lay out DeepProve's model cache from the local Hugging Face cache.

DeepProve's `--hf <repo>` loader reads exactly three files from
`./model_cache/<repo>/` relative to the working directory:
model.safetensors, tokenizer.json, config.json. When any is missing it
downloads `model.safetensors` from HF, which does not exist for sharded
checkpoints such as Llama-2-7B (two shards plus an index). So for sharded
repos we merge the shards into one file here. Tensors are copied verbatim
(same names, same dtype), so DeepProve sees the identical weights.

Usage: prepare_models.py <run_dir> [repo ...]
"""
import json, os, sys
from pathlib import Path

HF_HUB = Path(os.environ.get("HF_HUB_CACHE",
                             Path.home() / ".cache/huggingface/hub"))
DEFAULT_REPOS = ["openai-community/gpt2", "meta-llama/Llama-2-7b-hf"]


def snapshot(repo: str) -> Path:
    snaps = HF_HUB / f"models--{repo.replace('/', '--')}" / "snapshots"
    cands = [p for p in snaps.iterdir() if (p / "config.json").exists()]
    if not cands:
        sys.exit(f"no snapshot with config.json for {repo} under {snaps}")
    return max(cands, key=lambda p: p.stat().st_mtime)


def prepare(run_dir: Path, repo: str) -> None:
    src = snapshot(repo)
    dst = run_dir / "model_cache" / repo
    dst.mkdir(parents=True, exist_ok=True)
    if not (dst / "tokenizer.json").exists():
        os.symlink((src / "tokenizer.json").resolve(), dst / "tokenizer.json")
    # DeepProve's Llama loader requires `rope_theta`. Older Llama-2 snapshots
    # omit it and rely on the HF default (10000.0, which later revisions of
    # the same repo spell out), so add exactly that default when absent.
    cfg = json.loads((src / "config.json").read_text())
    if cfg.get("model_type") == "llama":
        cfg.setdefault("rope_theta", 10000.0)
    if (dst / "config.json").is_symlink():
        (dst / "config.json").unlink()
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))
    out = dst / "model.safetensors"
    if out.exists():
        print(f"{repo}: present")
        return
    if (src / "model.safetensors").exists():
        os.symlink((src / "model.safetensors").resolve(), out)
        print(f"{repo}: linked")
        return
    from safetensors import safe_open
    from safetensors.torch import save_file
    index = json.loads((src / "model.safetensors.index.json").read_text())
    tensors = {}
    for shard in sorted(set(index["weight_map"].values())):
        with safe_open(src / shard, framework="pt") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
    tmp = out.with_suffix(".tmp")
    save_file(tensors, tmp, metadata={"format": "pt"})
    tmp.rename(out)
    print(f"{repo}: merged {len(tensors)} tensors -> {out}")


def prepare_truncated(run_dir: Path, repo: str, layers: int) -> str:
    """`<repo>` cut to its first `layers` decoder blocks, at full width.

    DeepProve builds exactly `num_hidden_layers` blocks and ignores the other
    tensors in the file, so the weights are a symlink to the full merged file
    and only config.json changes. Returns the repo id to pass to --hf.
    """
    src = run_dir / "model_cache" / repo
    name = f"trunc/{repo.split('/')[-1]}-L{layers}"
    dst = run_dir / "model_cache" / name
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("model.safetensors", "tokenizer.json"):
        if not (dst / f).exists():
            os.symlink((src / f).resolve(), dst / f)
    cfg = json.loads((src / "config.json").read_text())
    cfg["num_hidden_layers"] = layers
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))
    return name


if __name__ == "__main__":
    # prepare_models.py <run_dir> [repo ...] [--truncate repo:L ...]
    run_dir = Path(sys.argv[1])
    args = sys.argv[2:]
    trunc = [a for a in args if ":" in a]
    repos = [a for a in args if ":" not in a and a != "--truncate"]
    for repo in repos or DEFAULT_REPOS:
        prepare(run_dir, repo)
    for t in trunc:
        repo, layers = t.rsplit(":", 1)
        print(prepare_truncated(run_dir, repo, int(layers)))
