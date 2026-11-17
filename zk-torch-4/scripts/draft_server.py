#!/usr/bin/env python3
"""Float draft model for the fixed-point Jacobi generation (models.rs ar_run,
MORPHEUS_DRAFT_CMD). Reads lines "SEQ t0 t1 ... tk" on stdin and answers one
line with SEQ tokens: the given prefix followed by the float HF model's
greedy continuation. The drafts only choose the Jacobi starting point: the
fixed-point model still confirms every token, so the generated suffix is the
fixed-point greedy suffix whatever the drafts are; good drafts cut the number
of fixed-point forward passes from ~one per token to ~one per divergence.

Usage (spawned by Rust): python scripts/draft_server.py --snapshot DIR [--layers L]
"""
import argparse
import os
import sys

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--vocab", type=int, default=None)
    ap.add_argument("--dtype", default="float32")
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(a.snapshot, torch_dtype=getattr(torch, a.dtype), low_cpu_mem_usage=True)
    if a.layers is not None:
        if hasattr(m, "transformer"):
            m.transformer.h = m.transformer.h[:a.layers]
            m.config.n_layer = a.layers
        else:
            m.model.layers = m.model.layers[:a.layers]
            m.config.num_hidden_layers = a.layers
    m.eval()
    print("ready", flush=True)
    for line in sys.stdin:
        xs = [int(x) for x in line.split()]
        if not xs:
            continue
        seq, toks = xs[0], xs[1:]
        with torch.no_grad():
            out = m(torch.tensor([toks]), use_cache=True)
            past = out.past_key_values
            lg = out.logits[0, -1]
            while len(toks) < seq:
                if a.vocab:
                    lg = lg[:a.vocab]
                nxt = int(lg.argmax())
                toks.append(nxt)
                if len(toks) == seq:
                    break
                out = m(torch.tensor([[nxt]]), past_key_values=past, use_cache=True)
                past, lg = out.past_key_values, out.logits[0, -1]
        print(" ".join(map(str, toks)), flush=True)


if __name__ == "__main__":
    main()
