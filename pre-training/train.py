#!/usr/bin/env python
"""python train.py --size 158m --data train.bin --steps 30 --out ckpt/smoke
   torchrun --nproc_per_node=8 train.py --size 945m --data train.bin --val-data val.bin --batch-seqs 1 --grad-accum 16 --out ckpt/memonce_945"""
import argparse, contextlib, math, os, sys, time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from memonce import MemOnce


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--size", default="397m", choices=sorted(MemOnce.SIZES))
    p.add_argument("--span", type=int, default=128)
    p.add_argument("--mem-tokens", type=int, default=16)
    p.add_argument("--block-size", type=int, default=1024)
    p.add_argument("--mem-budget", type=int, default=0)
    p.add_argument("--data", required=True)
    p.add_argument("--val-data", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=21748)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--batch-seqs", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--ckpt-every", type=int, default=5000)
    p.add_argument("--val-every", type=int, default=100)
    p.add_argument("--val-batches", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--init-from", default="")
    p.add_argument("--start-step", type=int, default=0)
    p.add_argument("--rewarmup", type=int, default=0)
    return p.parse_args()


def lr_at(step, args):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    t = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr * (args.min_lr_ratio + (1 - args.min_lr_ratio)
                      * 0.5 * (1 + math.cos(math.pi * t)))


def get_batch(data, seq_len, batch, device):
    idx = torch.randint(0, len(data) - seq_len - 1, (batch,))
    x = torch.stack([torch.from_numpy(data[i:i + seq_len].astype(np.int64)) for i in idx])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + seq_len].astype(np.int64)) for i in idx])
    return x.to(device), y.to(device)


def main():
    args = parse_args()
    ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(rank)
        local = f"cuda:{rank}"
    else:
        rank, world, local = 0, 1, "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed + rank)

    data = np.memmap(args.data, dtype=np.uint16, mode="r")
    val = np.memmap(args.val_data, dtype=np.uint16, mode="r") if args.val_data else None

    model = MemOnce.from_size(args.size, span=args.span, mem_tokens=args.mem_tokens,
                              block_size=args.block_size,
                              mem_budget=args.mem_budget or None).to(local)
    if args.init_from:
        sd = torch.load(args.init_from, map_location="cpu")
        model.load_state_dict(sd["model"] if "model" in sd else sd)
        if rank == 0:
            print(f"resumed weights from {args.init_from} (step {sd.get('step', '?')})")
    if ddp:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    raw = model.module if ddp else model
    if rank == 0:
        print(f"MemOnce-{args.size} non-emb {raw.num_params()/1e6:.0f}M | "
              f"span={raw.span} mem={raw.mem_tokens} (ratio {raw.span/raw.mem_tokens:g}x) "
              f"block={raw.block_size} budget={raw.mem_budget or 'full'}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.1, fused=(local != "cpu"))
    os.makedirs(args.out, exist_ok=True)
    t0, running = time.time(), []
    for step in range(args.start_step, args.steps):
        lr = lr_at(step, args)
        if args.rewarmup > 0 and step < args.start_step + args.rewarmup:
            lr *= (step - args.start_step + 1) / args.rewarmup
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        losses = []
        for micro in range(args.grad_accum):
            x, y = get_batch(data, args.seq_len, args.batch_seqs, local)
            sync = micro == args.grad_accum - 1 or args.grad_accum == 1
            ctx = model.no_sync() if (ddp and not sync) else contextlib.nullcontext()
            with ctx, torch.autocast("cuda" if local != "cpu" else "cpu", dtype=torch.bfloat16):
                out = model(x, targets=y)
            loss = out["loss"] / args.grad_accum
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}: {loss.item()}")
            loss.backward()
            losses.append(out["loss"].item())
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if torch.isfinite(grad_norm):
            opt.step()
        else:
            print(f"step {step}: non-finite grad norm; skipping update", flush=True)
        running.append(sum(losses) / len(losses))
        if rank == 0 and (step % args.log_every == 0 or step == args.steps - 1):
            speed = ((step - args.start_step + 1) * args.batch_seqs * args.seq_len
                     * world * args.grad_accum / (time.time() - t0))
            print(f"step {step:6d} loss {sum(running)/len(running):.4f} "
                  f"lr {lr:.2e} tok/s {speed:,.0f}", flush=True)
            running = []
        if args.val_every and val is not None and (step + 1) % args.val_every == 0:
            model.eval()
            vls = []
            with torch.no_grad(), torch.autocast("cuda" if local != "cpu" else "cpu", dtype=torch.bfloat16):
                for _ in range(args.val_batches):
                    vx, vy = get_batch(val, args.seq_len, args.batch_seqs, local)
                    vls.append(model(vx, targets=vy)["loss"].item())
            print(f"VAL step {step+1} loss {sum(vls)/len(vls):.4f}", flush=True)
            model.train()
        if rank == 0 and ((step + 1) % args.ckpt_every == 0 or step == args.steps - 1):
            path = os.path.join(args.out, f"step{step+1}.pt")
            torch.save({"model": raw.state_dict(), "model_name": f"memonce_{args.size}",
                        "step": step + 1, "args": vars(args)}, path)
            print(f"saved {path}", flush=True)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
