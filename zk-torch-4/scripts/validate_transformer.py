#!/usr/bin/env python3
"""Compare a Morpheus fixed-point generation (morpheus_real_check CHECK_DUMP)
with the float HF model on CPU.

Reports, for the dumped request:
  * greedy generation: fraction of the generated tokens equal to the float
    model's greedy tokens from the same prompt, and the first divergence;
  * prompt logits (positions 0..P-1): top-1 agreement and max |error|;
  * teacher forcing on the fixed-point sequence: top-1 agreement of the
    float model at every position (the next-token decision the fixed-point
    model made), and max |logit error|.

Usage: python scripts/validate_transformer.py --model gpt2 --dump FILE
           [--layers L] [--prompt-len 16]
"""
import argparse
import os
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_transformer import snapshot  # noqa: E402


def load_dump(p):
    b = open(p, "rb").read()
    assert b[:4] == b"ZKLG"
    seq, vocab, sf = struct.unpack("<3I", b[4:16])
    toks = np.frombuffer(b, dtype="<u4", count=seq, offset=16).astype(np.int64)
    lg = np.frombuffer(b, dtype="<f4", count=seq * vocab, offset=16 + 4 * seq).reshape(seq, vocab)
    return seq, vocab, sf, toks, lg


def truncate(model, name, layers):
    if layers is None:
        return
    if name == "gpt2" or name == "gptj":
        model.transformer.h = model.transformer.h[:layers]
        model.config.n_layer = layers
    else:
        model.model.layers = model.model.layers[:layers]
        model.config.num_hidden_layers = layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--prompt-len", type=int, default=16)
    ap.add_argument("--snapshot", default=None, help="checkpoint dir (default: the HF cache snapshot)")
    ap.add_argument("--logit-scale", type=float, default=1.0,
                    help="divide the fixed-point logits by this (GPT-2 with REAL_RESID_SCALE=c: tied head, logits x c)")
    ap.add_argument("--graph-equivalent", "--no-eps", dest="graph_eq", action="store_true",
                    help="float reference of the graph's function: norm epsilon 0 (the graph has none) and, for "
                         "GPT-2 / GPT-J, GELU as h*sigmoid(1.702h) (the graph's gadget); isolates fixed-point error")
    a = ap.parse_args()
    snap = a.snapshot or snapshot(a.model)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    from transformers import AutoModelForCausalLM
    seq, vocab, sf, toks, lg = load_dump(a.dump)
    lg = lg / a.logit_scale
    P = a.prompt_len
    model = AutoModelForCausalLM.from_pretrained(snap, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    truncate(model, a.model, a.layers)
    if a.graph_eq:
        class SigmoidGelu(torch.nn.Module):
            def forward(self, x):
                return x * torch.sigmoid(1.702 * x)
        if a.model in ("gpt2", "gptj"):
            for b in model.transformer.h:
                b.mlp.act = SigmoidGelu()
        for mod in model.modules():
            if hasattr(mod, "variance_epsilon"):
                mod.variance_epsilon = 0.0
            elif isinstance(mod, torch.nn.LayerNorm):
                mod.eps = 0.0
    model.eval()
    ids = torch.tensor(toks[None, :])
    with torch.no_grad():
        # Teacher forcing on the fixed-point sequence.
        ref = model(ids).logits[0, :, :vocab].float().numpy()
        # Float greedy generation from the same prompt.
        out = model(ids[:, :P], use_cache=True)
        past, nxt = out.past_key_values, out.logits[0, -1, :vocab].argmax().item()
        gen = [nxt]
        for _ in range(seq - P - 1):
            out = model(torch.tensor([[nxt]]), past_key_values=past, use_cache=True)
            past, nxt = out.past_key_values, out.logits[0, -1, :vocab].argmax().item()
            gen.append(nxt)
    gen = np.array(gen)
    fx = toks[P:]
    match = (gen == fx)
    first = int(np.argmin(match)) if not match.all() else None
    tf_fx = lg.argmax(-1)  # fixed-point argmax at every position
    tf_ref = ref.argmax(-1)
    agree = (tf_fx[:seq - 1] == tf_ref[:seq - 1])
    err = np.abs(lg - ref)
    # Per-position common offset removed (softmax / argmax invariant).
    cerr = np.abs((lg - lg.mean(-1, keepdims=True)) - (ref - ref.mean(-1, keepdims=True)))
    print(f"model {a.model} layers {a.layers if a.layers is not None else 'all'} seq {seq} vocab {vocab} scale 2^{sf}"
          + (" (graph-equivalent float reference: no norm epsilon"
             + (", sigmoid GELU)" if a.model in ("gpt2", "gptj") else ")") if a.graph_eq else " (exact HF float reference)"))
    print(f"greedy generation: {match.sum()}/{len(fx)} generated tokens match the float model "
          f"({100 * match.mean():.1f}%), first divergence at generated position {first}"
          + ("" if first is None else f" (sequence position {P + first})"))
    print(f"  prefix-exact match length {first if first is not None else len(fx)}")
    print(f"prompt logits (pos 0..{P - 1}): top-1 agreement {int(agree[:P].sum())}/{P}, "
          f"max |err| {err[:P].max():.4f}, mean |err| {err[:P].mean():.4f}, "
          f"logit range float [{ref[:P].min():.1f}, {ref[:P].max():.1f}]")
    print(f"teacher-forced on the fixed-point sequence (pos 0..{seq - 2}): top-1 agreement "
          f"{int(agree.sum())}/{seq - 1} ({100 * agree.mean():.1f}%), max |err| {err[:seq - 1].max():.4f}, "
          f"mean |err| {err[:seq - 1].mean():.4f}; offset-free max |err| {cerr[:seq - 1].max():.4f} "
          f"mean {cerr[:seq - 1].mean():.4f} (prompt: max {cerr[:P].max():.4f} mean {cerr[:P].mean():.4f})")
    # Margin view: where the two disagree, how close were the float top-2?
    if not agree.all():
        idx = np.where(~agree)[0]
        srt = np.sort(ref[idx], axis=-1)
        gaps = srt[:, -1] - srt[:, -2]
        print(f"  disagreements at {idx.tolist()[:20]}; float top-1/top-2 gap there: median {np.median(gaps):.3f}, max {gaps.max():.3f}")
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(snap)
        print("prompt   :", repr(tok.decode(toks[:P])))
        print("fixed-pt :", repr(tok.decode(fx)))
        print("float    :", repr(tok.decode(gen)))
    except Exception as e:  # tokenizer is optional for the numbers
        print("(no tokenizer:", e, ")")


if __name__ == "__main__":
    main()
