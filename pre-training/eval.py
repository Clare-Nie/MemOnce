#!/usr/bin/env python
"""python eval.py --ckpt ckpt/memonce_158/step21748.pt --size 158m --data val.bin"""
import argparse
import numpy as np
import torch

from memonce import MemOnce


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--size", default=None, choices=sorted(MemOnce.SIZES))
    p.add_argument("--data", required=True)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--max-batches", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    sd = torch.load(args.ckpt, map_location="cpu")
    weights = sd["model"] if (isinstance(sd, dict) and "model" in sd) else sd
    cfg = sd.get("args", {}) if isinstance(sd, dict) else {}
    if "overrides" in cfg and isinstance(cfg["overrides"], str):
        import json
        cfg = {**json.loads(cfg["overrides"] or "{}"), **cfg}
    dim2size = {640: "158m", 1024: "397m", 1536: "603m", 2048: "945m"}
    dim = weights["backbone.embed.weight"].shape[1]
    size = args.size or cfg.get("size") or dim2size.get(dim, "397m")
    model = MemOnce.from_size(
        size, span=cfg.get("span", 128), mem_tokens=cfg.get("mem_tokens", 16),
        block_size=cfg.get("block_size", 1024)).cuda().eval()
    model.load_state_dict(weights, strict=True)
    print(f"loaded {args.ckpt} (MemOnce-{size}, step {sd.get('step', '?')})")

    data = np.memmap(args.data, dtype=np.uint16, mode="r")
    n = (len(data) - 1) // args.seq_len
    if args.max_batches:
        n = min(n, args.max_batches * args.batch)
    print(f"evaluating {n} windows of {args.seq_len} tokens from {args.data}")
    losses, seen = [], 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for start in range(0, n - n % args.batch, args.batch):
            batch = [torch.from_numpy(
                data[i * args.seq_len:(i + 1) * args.seq_len + 1].astype(np.int64))
                for i in range(start, start + args.batch)]
            seq = torch.stack(batch).cuda()
            out = model(seq[:, :-1], targets=seq[:, 1:])
            losses.append(out["loss"].item())
            seen += args.batch
            if seen % (args.batch * 50) == 0:
                print(f"  {seen}/{n} windows, running ppl {np.exp(np.mean(losses)):.3f}",
                      flush=True)
    loss = float(np.mean(losses))
    print(f"loss {loss:.4f}  ppl {np.exp(loss):.3f}  ({seen} windows)")


if __name__ == "__main__":
    main()
