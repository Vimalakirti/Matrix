#!/usr/bin/env python3
"""Export a pretrained HF transformer checkpoint for Morpheus (WEIGHTS=real).

Writes fixed-point weights round(w * 2^SF) as signed int32, already in the
graph's slot layout (src/morpheus/models.rs, src/dag/{gpt2,llama,gptj}.rs):

  * every matmul weight is [in, out] (the graph computes "bsi,ij->bsj");
    HF nn.Linear weights ([out, in]) are transposed, GPT-2 Conv1D weights
    are already [in, out] and c_attn [h, 3h] is split into q / k / v;
  * the graph's RoPE (rope_with_vecs) rotates interleaved pairs (2i, 2i+1)
    by the NEGATIVE angle. Llama q/k rows (HF rotate_half layout: pair
    (i, i + d/2)) and GPT-J q/k rows (interleaved, first rotary_dim dims) are
    permuted so that graph slot 2i holds the pair's second component and 2i+1
    its first. Rotating the swapped pair by -angle equals rotating the HF pair
    by +angle; q and k get the same permutation, so attention scores are
    unchanged;
  * norm gains, biases, embeddings and LM heads at the same scale 2^SF.

File format (one file per tensor, <name>.bin):
  b"ZKRW" | ndim:u32 | shape:u32*ndim | data:i32*prod(shape)  (little endian,
  row-major logical order; models.rs pads to powers of two and lays the
  values out column-major).
Plus meta.txt (key=value) and prompts.bin ([N, 16] token ids, int32) with N
wikitext-2 (test split) prompts tokenized with the model's tokenizer
(Llama prompts start with BOS, as the HF tokenizer adds it).

--weights hf writes only meta.txt (source=hf, snapshot=<dir>) and the
prompts: models.rs then reads the safetensors snapshot directly and applies
the same transforms in Rust (Real::raw). Used for the 6B-8B models, whose
int exports (13-32 GB each) do not fit the available disks. The snapshot
must hold *.safetensors (GPT-J's HF float16 branch ships a pytorch .bin:
convert it once with save_file, see --snapshot).

Usage:
  python scripts/export_transformer.py --model gpt2 --out DIR [--sf 10]
        [--layers L] [--prompts 64] [--prompt-len 16]
        [--weights zkrw|hf] [--snapshot DIR]
"""
import argparse
import glob
import json
import os
import struct

import numpy as np
import torch

HUB = os.path.expanduser("~/.cache/huggingface/hub")
REPOS = {
    "gpt2": "models--openai-community--gpt2",
    "llama2": "models--meta-llama--Llama-2-7b-hf",
    "llama3": "models--meta-llama--Llama-3.1-8B",
    "gptj": "models--EleutherAI--gpt-j-6b",
}


def snapshot(model):
    snaps = sorted(glob.glob(os.path.join(HUB, REPOS[model], "snapshots", "*")))
    snaps = [s for s in snaps if glob.glob(os.path.join(s, "*.safetensors")) or glob.glob(os.path.join(s, "*.bin"))]
    if not snaps:
        raise SystemExit(f"no checkpoint for {model} under {HUB}/{REPOS[model]}")
    return snaps[-1]


class Ckpt:
    """Lazy tensor access over safetensors shards or a pytorch .bin."""

    def __init__(self, path):
        self.path = path
        st = sorted(glob.glob(os.path.join(path, "*.safetensors")))
        self.files = {}
        if st:
            from safetensors import safe_open
            self.handles = [safe_open(f, framework="pt") for f in st]
            for h in self.handles:
                for k in h.keys():
                    self.files[k] = h
            self.sd = None
        else:
            bins = sorted(glob.glob(os.path.join(path, "*.bin")))
            self.sd = {}
            for b in bins:
                self.sd.update(torch.load(b, map_location="cpu", weights_only=True))

    def __call__(self, name):
        if self.sd is not None:
            t = self.sd[name]
        else:
            t = self.files[name].get_tensor(name)
        return t.to(torch.float64)


RESID = ("embed", "wte", "wpe", "o_w", "o_b", "down_w", "fc2_w", "fc2_b", "fc_out_w", "fc_out_b")
RESID_SCALE = 1.0


def write(out, name, t, sf, stats):
    t = t.to(torch.float64)
    if name.split(".")[-1] in RESID:  # see models.rs REAL_RESID_SCALE
        t = t * RESID_SCALE
    q = torch.round(t * (1 << sf))
    m = float(q.abs().max())
    assert m < 2**31, f"{name}: |w|*2^{sf} = {m} does not fit int32"
    a = q.to(torch.int32).contiguous().numpy()
    with open(os.path.join(out, name + ".bin"), "wb") as f:
        f.write(b"ZKRW")
        f.write(struct.pack("<I", a.ndim))
        f.write(struct.pack(f"<{a.ndim}I", *a.shape))
        f.write(a.astype("<i4").tobytes())
    kind = name.split(".")[-1]
    s = stats.setdefault(kind, [0, 0.0, 0])
    s[0] = max(s[0], int(m))
    s[1] = max(s[1], float(t.abs().max()))
    s[2] += a.size


def pair_swap_rows(w, heads, hd, rot, half_split):
    """w: [heads*hd, in] HF q/k weight (rows = output features). Returns rows
    reordered so graph slot (h, 2i) = second component, (h, 2i+1) = first."""
    w = w.reshape(heads, hd, -1)
    idx = list(range(hd))
    for i in range(rot // 2):
        if half_split:  # HF rotate_half: pair (i, i + rot/2)
            first, second = i, i + rot // 2
        else:  # interleaved: pair (2i, 2i + 1)
            first, second = 2 * i, 2 * i + 1
        idx[2 * i] = second
        idx[2 * i + 1] = first
    return w[:, idx, :].reshape(heads * hd, -1)


def export_gpt2(ck, cfg, out, sf, layers, stats):
    h = cfg["n_embd"]
    for l in range(layers):
        p = f"h.{l}."
        g = lambda n: ck(p + n)
        write(out, f"l{l}.ln1_w", g("ln_1.weight"), sf, stats)
        write(out, f"l{l}.ln1_b", g("ln_1.bias"), sf, stats)
        w = g("attn.c_attn.weight")  # [h, 3h], Conv1D = [in, out]
        b = g("attn.c_attn.bias")
        for i, n in enumerate("qkv"):
            write(out, f"l{l}.{n}_w", w[:, i * h:(i + 1) * h], sf, stats)
            write(out, f"l{l}.{n}_b", b[i * h:(i + 1) * h], sf, stats)
        write(out, f"l{l}.o_w", g("attn.c_proj.weight"), sf, stats)
        write(out, f"l{l}.o_b", g("attn.c_proj.bias"), sf, stats)
        write(out, f"l{l}.ln2_w", g("ln_2.weight"), sf, stats)
        write(out, f"l{l}.ln2_b", g("ln_2.bias"), sf, stats)
        write(out, f"l{l}.fc1_w", g("mlp.c_fc.weight"), sf, stats)
        write(out, f"l{l}.fc1_b", g("mlp.c_fc.bias"), sf, stats)
        write(out, f"l{l}.fc2_w", g("mlp.c_proj.weight"), sf, stats)
        write(out, f"l{l}.fc2_b", g("mlp.c_proj.bias"), sf, stats)
    write(out, "lnf_w", ck("ln_f.weight"), sf, stats)
    write(out, "lnf_b", ck("ln_f.bias"), sf, stats)
    write(out, "wte", ck("wte.weight"), sf, stats)
    write(out, "wpe", ck("wpe.weight"), sf, stats)


def export_llama(ck, cfg, out, sf, layers, stats):
    nh, nkv = cfg["num_attention_heads"], cfg.get("num_key_value_heads", cfg["num_attention_heads"])
    h = cfg["hidden_size"]
    hd = h // nh
    for l in range(layers):
        p = f"model.layers.{l}."
        g = lambda n: ck(p + n)
        write(out, f"l{l}.attn_norm", g("input_layernorm.weight"), sf, stats)
        write(out, f"l{l}.q_w", pair_swap_rows(g("self_attn.q_proj.weight"), nh, hd, hd, True).T, sf, stats)
        write(out, f"l{l}.k_w", pair_swap_rows(g("self_attn.k_proj.weight"), nkv, hd, hd, True).T, sf, stats)
        write(out, f"l{l}.v_w", g("self_attn.v_proj.weight").T, sf, stats)
        write(out, f"l{l}.o_w", g("self_attn.o_proj.weight").T, sf, stats)
        write(out, f"l{l}.mlp_norm", g("post_attention_layernorm.weight"), sf, stats)
        write(out, f"l{l}.gate_w", g("mlp.gate_proj.weight").T, sf, stats)
        write(out, f"l{l}.up_w", g("mlp.up_proj.weight").T, sf, stats)
        write(out, f"l{l}.down_w", g("mlp.down_proj.weight").T, sf, stats)
    write(out, "final_norm", ck("model.norm.weight"), sf, stats)
    write(out, "embed", ck("model.embed_tokens.weight"), sf, stats)
    write(out, "lm_head", ck("lm_head.weight"), sf, stats)


def export_gptj(ck, cfg, out, sf, layers, stats):
    nh, h = cfg["n_head"], cfg["n_embd"]
    hd, rot = h // nh, cfg["rotary_dim"]
    for l in range(layers):
        p = f"transformer.h.{l}."
        g = lambda n: ck(p + n)
        write(out, f"l{l}.ln1_w", g("ln_1.weight"), sf, stats)
        write(out, f"l{l}.ln1_b", g("ln_1.bias"), sf, stats)
        write(out, f"l{l}.q_w", pair_swap_rows(g("attn.q_proj.weight"), nh, hd, rot, False).T, sf, stats)
        write(out, f"l{l}.k_w", pair_swap_rows(g("attn.k_proj.weight"), nh, hd, rot, False).T, sf, stats)
        write(out, f"l{l}.v_w", g("attn.v_proj.weight").T, sf, stats)
        write(out, f"l{l}.o_w", g("attn.out_proj.weight").T, sf, stats)
        write(out, f"l{l}.fc_in_w", g("mlp.fc_in.weight").T, sf, stats)
        write(out, f"l{l}.fc_in_b", g("mlp.fc_in.bias"), sf, stats)
        write(out, f"l{l}.fc_out_w", g("mlp.fc_out.weight").T, sf, stats)
        write(out, f"l{l}.fc_out_b", g("mlp.fc_out.bias"), sf, stats)
    write(out, "lnf_w", ck("transformer.ln_f.weight"), sf, stats)
    write(out, "lnf_b", ck("transformer.ln_f.bias"), sf, stats)
    write(out, "wte", ck("transformer.wte.weight"), sf, stats)
    write(out, "lm_head", ck("lm_head.weight"), sf, stats)
    write(out, "lm_head_b", ck("lm_head.bias"), sf, stats)


def prompts(path, n, plen):
    """First `plen` tokens of the first `n` wikitext-2 test paragraphs (not
    headings) with at least 2*plen tokens."""
    from transformers import AutoTokenizer
    import pandas as pd
    tok = AutoTokenizer.from_pretrained(path)
    pq = glob.glob(os.path.join(HUB, "datasets--wikitext", "snapshots", "*", "wikitext-2-raw-v1", "test-*.parquet"))[0]
    lines = pd.read_parquet(pq)["text"].tolist()
    out, texts = [], []
    for t in lines:
        s = t.strip()
        if not s or s.startswith("="):
            continue
        ids = tok(s)["input_ids"]
        if len(ids) < 2 * plen:
            continue
        out.append(ids[:plen])
        texts.append(tok.decode(ids[:plen]))
        if len(out) == n:
            break
    return np.array(out, dtype=np.int32), texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(REPOS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--sf", type=int, default=10)
    ap.add_argument("--layers", type=int, default=None, help="export only the first L layers")
    ap.add_argument("--prompts", type=int, default=64)
    ap.add_argument("--prompt-len", type=int, default=16)
    ap.add_argument("--weights", choices=["zkrw", "hf"], default="zkrw")
    ap.add_argument("--resid-scale", type=float, default=1.0,
                    help="multiply residual-stream writes (embeddings, output projections) by c; "
                         "the direct path does this with REAL_RESID_SCALE")
    ap.add_argument("--snapshot", default=None, help="checkpoint dir (default: the HF cache snapshot)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    path = a.snapshot or snapshot(a.model)
    cfg = json.load(open(os.path.join(path, "config.json")))
    nl = cfg.get("n_layer", cfg.get("num_hidden_layers"))
    layers = nl if a.layers is None else min(a.layers, nl)
    stats = {}
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    if a.weights == "hf":
        p, texts = prompts(path, a.prompts, a.prompt_len)
        with open(os.path.join(a.out, "prompts.bin"), "wb") as f:
            f.write(b"ZKRW" + struct.pack("<I", 2) + struct.pack("<2I", *p.shape) + p.astype("<i4").tobytes())
        with open(os.path.join(a.out, "prompts.txt"), "w") as f:
            for i, t in enumerate(texts):
                f.write(f"{i}\t{t!r}\n")
        with open(os.path.join(a.out, "meta.txt"), "w") as f:
            f.write(f"model={a.model}\nsf={a.sf}\nsource=hf\nsnapshot={os.path.abspath(path)}\nprompts={len(p)}\nprompt_len={a.prompt_len}\n")
        print(f"wrote meta (direct source {path}) + {len(p)} prompts to {a.out}")
        return
    global RESID_SCALE
    RESID_SCALE = a.resid_scale
    ck = Ckpt(path)
    {"gpt2": export_gpt2, "llama2": export_llama, "llama3": export_llama, "gptj": export_gptj}[a.model](ck, cfg, a.out, a.sf, layers, stats)
    p, texts = prompts(path, a.prompts, a.prompt_len)
    with open(os.path.join(a.out, "prompts.bin"), "wb") as f:
        f.write(b"ZKRW" + struct.pack("<I", 2) + struct.pack("<2I", *p.shape) + p.astype("<i4").tobytes())
    with open(os.path.join(a.out, "prompts.txt"), "w") as f:
        for i, t in enumerate(texts):
            f.write(f"{i}\t{t!r}\n")
    with open(os.path.join(a.out, "meta.txt"), "w") as f:
        f.write(f"model={a.model}\nsf={a.sf}\nlayers={layers}\nsnapshot={path}\nprompts={len(p)}\nprompt_len={a.prompt_len}\nresid_scale={a.resid_scale}\n")
        for k, (mi, mf, n) in sorted(stats.items()):
            bits = int(mi).bit_length() + 1  # signed
            f.write(f"stat.{k}=max_int {mi} ({bits} signed bits) max_abs {mf:.4f} elems {n}\n")
    for k, (mi, mf, n) in sorted(stats.items()):
        print(f"{k:12s} max|q| {mi:>10d} ({int(mi).bit_length() + 1:2d} signed bits)  max|w| {mf:9.4f}  elems {n}")
    print(f"exported {layers} layers + {len(p)} prompts to {a.out}")


if __name__ == "__main__":
    main()
