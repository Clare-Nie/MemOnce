#!/usr/bin/env python
"""Zero-shot effect evaluation for a MemOnce checkpoint (lm-evaluation-harness).

  python eval_zeroshot.py --ckpt /path/to/step21748.pt --tokenizer /path/to/llama2_tokenizer

Runs the paper's eight tasks zero-shot and prints per-task accuracy + AVG8:
lambada_openai piqa hellaswag winogrande arc_easy arc_challenge sciq boolq
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pre-training"))
from memonce import MemOnce

DEFAULT_TASKS = ("lambada_openai,piqa,hellaswag,winogrande,"
                 "arc_easy,arc_challenge,sciq,boolq")
NORM_TASKS = {"hellaswag", "arc_easy", "arc_challenge"}


def load_model(ckpt_path, size=None, device="cuda"):
    sd = torch.load(ckpt_path, map_location="cpu")
    weights = sd["model"] if (isinstance(sd, dict) and "model" in sd) else sd
    cfg = sd.get("args", {}) if isinstance(sd, dict) else {}
    if "overrides" in cfg and isinstance(cfg["overrides"], str):
        cfg = {**json.loads(cfg["overrides"] or "{}"), **cfg}
    dim2size = {640: "158m", 1024: "397m", 1536: "603m", 2048: "945m"}
    size = size or cfg.get("size") or dim2size.get(weights["backbone.embed.weight"].shape[1], "397m")
    model = MemOnce.from_size(
        size, span=cfg.get("span", 128), mem_tokens=cfg.get("mem_tokens", 16),
        block_size=cfg.get("block_size", 1024)).to(device).eval()
    model.load_state_dict(weights, strict=True)
    return model, size


class MemOnceEval:
    def __init__(self, model, tokenizer, device="cuda", max_length=4096):
        self.model = model
        self.tok = tokenizer
        self.device = device
        self.max_length = max_length

    def _encode(self, s):
        return self.tok.encode(s)

    def _encode_pair(self, ctx, cont):
        if not ctx:
            return [self.tok.bos_token_id], self._encode(cont)
        n_spaces = len(ctx) - len(ctx.rstrip())
        if n_spaces:
            cont = ctx[-n_spaces:] + cont
            ctx = ctx[:-n_spaces]
        whole = self._encode(ctx + cont)
        c_ids = self._encode(ctx)
        return c_ids, whole[len(c_ids):]

    @torch.no_grad()
    def loglikelihood(self, pairs, batch_size=8):
        """pairs: list of (context, continuation) -> list of (ll, is_greedy)."""
        encoded = []
        for ctx, cont in pairs:
            c_ids, t_ids = self._encode_pair(ctx, cont)
            keep = self.max_length - 1
            if len(c_ids) + len(t_ids) > keep:
                overflow = len(c_ids) + len(t_ids) - keep
                c_ids = c_ids[:1] + c_ids[1 + overflow:]
            encoded.append((c_ids, t_ids))
        order = sorted(range(len(encoded)), key=lambda i: sum(map(len, encoded[i])))
        results = [None] * len(encoded)
        for bs in range(0, len(order), batch_size):
            idxs = order[bs:bs + batch_size]
            seqs = [torch.tensor(encoded[i][0] + encoded[i][1]) for i in idxs]
            L = max(len(s) for s in seqs)
            x = torch.zeros(len(seqs), L, dtype=torch.long)
            for r, s in enumerate(seqs):
                x[r, :len(s)] = s
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = self.model(x.to(self.device))["logits"].float()
            for r, i in enumerate(idxs):
                n_cont = len(encoded[i][1])
                n_in = len(seqs[r]) - 1
                lp = torch.log_softmax(logits[r, n_in - n_cont:n_in], dim=-1)
                tgt = torch.tensor(encoded[i][1], device=lp.device)
                token_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                greedy = (lp.argmax(-1) == tgt).all().item()
                results[i] = (token_lp.sum().item(), greedy)
        return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tokenizer", required=True,
                   help="LLaMA-2 tokenizer (local dir or HF hub name)")
    p.add_argument("--size", default=None, choices=sorted(MemOnce.SIZES))
    p.add_argument("--tasks", default=DEFAULT_TASKS)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--out", default="")
    args = p.parse_args()

    from lm_eval import evaluator
    from lm_eval.api.model import LM
    from lm_eval.api.registry import register_model
    from transformers import AutoTokenizer

    model, size = load_model(args.ckpt, args.size)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    wrapper = MemOnceEval(model, tok)
    print(f"MemOnce-{size} loaded from {args.ckpt}")

    @register_model("memonce")
    class _MemOnceLM(LM):
        def __init__(self, **kw):
            super().__init__()
            self._w = wrapper

        def loglikelihood(self, requests):
            return self._w.loglikelihood([r.args for r in requests],
                                         batch_size=args.batch_size)

        def loglikelihood_rolling(self, requests):
            return [self._w.loglikelihood([(r.args[0], "")])[0][0]
                    for r in requests]

        def generate_until(self, requests):
            raise NotImplementedError

    results = evaluator.simple_evaluate(
        model="memonce", model_args="",
        tasks=args.tasks.split(","),
        batch_size=args.batch_size,
        log_samples=False)

    rows, accs = [], []
    print(f"{'task':<18}{'metric':<10}{'value':>8}")
    for t in args.tasks.split(","):
        r = results["results"][t]
        key = "acc_norm,none" if t in NORM_TASKS and "acc_norm,none" in r else "acc,none"
        rows.append((t, key, r[key]))
        accs.append(r[key])
        print(f"{t:<18}{key:<10}{r[key]:>8.4f}")
    print(f"{'AVG8':<18}{'':<10}{sum(accs)/len(accs):>8.4f}")
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "results.json"), "w") as f:
            json.dump(results, f, indent=2, default=str)
        print("saved ->", args.out)


if __name__ == "__main__":
    main()
