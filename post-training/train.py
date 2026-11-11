#!/usr/bin/env python
# MemOnce post-training on Qwen3.5-4B (the released recipe).
#
#   torchrun --standalone --nproc_per_node=8 train.py \
#     --model-name-or-path /path/to/Qwen3.5-4B \
#     --dataset-path /path/to/stage1_training_dataset \
#     --output-dir output/memonce4b
#
# Data: a directory of *.json files; each sample is {"messages": [...]}
# ending with an assistant turn. Recipe: adamw lr 5e-5 (linear), batch
# 1 x accum 4 x world ranks, bf16, grad-checkpointing, max_length 32768,
# save every 500 steps (max_steps 15625; the released checkpoint is step 10000).
import argparse, json, os, sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments, TrainerCallback

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import MemOnceConfig, MemOnceModel

# Checkpoints are trusted self-written files: no-op the torch.load safety gate
# so Trainer resume works on older torch versions.
import transformers
import transformers.trainer as _tf_trainer
import transformers.utils as _tf_utils

_tf_trainer.check_torch_load_is_safe = lambda: None
_tf_utils.check_torch_load_is_safe = lambda: None


def rank0(*x):
    if int(os.environ.get("RANK", "0")) == 0:
        print(*x, flush=True)


class JsonDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def load_rows(path: Path, limit: int = 0):
    files = sorted(path.glob("*.json"))
    rows = []
    for f in files:
        with f.open() as h:
            obj = json.load(h)
        if isinstance(obj, list):
            rows.extend(obj)
        elif isinstance(obj, dict):
            rows.append(obj)
        if limit and len(rows) >= limit:
            break
    return rows[:limit] if limit else rows


def parse_template(
    ids: List[int], tok
) -> tuple[List[List[int]], List[tuple[int, int]]]:
    start_id = tok.convert_tokens_to_ids("<|im_start|>")
    end_id = tok.convert_tokens_to_ids("<|im_end|>")
    user = tok("user", add_special_tokens=False)["input_ids"]
    assistant = tok("assistant", add_special_tokens=False)["input_ids"]
    roles = []
    i = 0
    while i < len(ids):
        if ids[i] != start_id:
            i += 1
            continue
        role = None
        if ids[i + 1 : i + 1 + len(user)] == user:
            role = "user"
        elif ids[i + 1 : i + 1 + len(assistant)] == assistant:
            role = "assistant"
        if role is None:
            i += 1
            continue
        p = i + 1 + len(user if role == "user" else assistant)
        # Header is followed by a newline token (198); content starts after it.
        while p < len(ids) and ids[p] != 198:
            p += 1
        content_start = min(p + 1, len(ids))
        q = content_start
        while q < len(ids) and ids[q] != end_id:
            q += 1
        roles.append((role, i, content_start, min(q + 1, len(ids))))
        i = q + 1
    spans, assistant_ranges = [], []
    # Every user starts a dialogue turn; a directly following assistant is
    # part of that turn, otherwise the user message is a standalone turn.
    index = 0
    while index < len(roles):
        role, start, content_start, end = roles[index]
        if role != "user":
            index += 1
            continue
        if index + 1 < len(roles) and roles[index + 1][0] == "assistant":
            assistant = roles[index + 1]
            spans.append([start, assistant[1], assistant[3]])
            index += 2
        else:
            spans.append([start, end, end])
            index += 1
    for r in roles:
        if r[0] == "assistant":
            assistant_ranges.append((r[2], r[3]))
    return spans, assistant_ranges


class Collator:
    def __init__(self, tok, max_length):
        self.tok, self.max_length = tok, max_length

    def __call__(self, features):
        if len(features) != 1:
            raise ValueError("MemOnce training requires batch size 1")
        row = features[0]
        messages = row["messages"]
        if not messages or messages[-1].get("role") != "assistant":
            raise ValueError("training sample must end with an assistant response")
        ids = self.tok.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        )
        if not isinstance(ids, list):
            ids = list(ids["input_ids"])
        if len(ids) > self.max_length:
            raise ValueError(
                f"tokenized sample length {len(ids)} exceeds max_length={self.max_length}"
            )
        spans, aranges = parse_template(ids, self.tok)
        if len(spans) < 2:
            raise ValueError("sample needs at least one history and one query turn")
        x = torch.tensor([ids], dtype=torch.long)
        labels = torch.full_like(x, -100)
        s, e = aranges[-1]
        labels[0, s:e] = x[0, s:e]
        meta = dict(row.get("meta") or {})
        meta["turn_token_spans"] = spans
        return {
            "input_ids": x,
            "attention_mask": torch.ones_like(x),
            "labels": labels,
            "meta": [meta],
        }


class FiniteGradCallback(TrainerCallback):
    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        opt = kwargs.get("optimizer")
        bad = (
            any(
                p.grad is not None and not torch.isfinite(p.grad).all()
                for p in model.parameters()
            )
            if model
            else False
        )
        if bad:
            rank0("[WARN] non-finite gradient; skipping update")
            (opt.zero_grad if opt else model.zero_grad)(set_to_none=True)
        return control


class LengthAwareTrainer(Trainer):
    """With --group-by-length: sort samples by char-length proxy, cut into
    per-step windows of world*batch*accum, shuffle window order (seeded).
    Off by default."""

    def _get_train_sampler(self, dataset=None, train_dataset=None):
        grouped = bool(getattr(self.args, "group_by_length", False)) or (
            getattr(self.args, "train_sampling_strategy", "") == "group_by_length"
        )
        ds = train_dataset if train_dataset is not None else dataset
        if ds is None:
            ds = self.train_dataset
        if not grouped:
            return super()._get_train_sampler(ds)
        from transformers.trainer_pt_utils import LengthGroupedSampler

        lengths = [int(r.get("length") or 0) for r in ds.rows]
        world = 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world = torch.distributed.get_world_size()
        per_rank = max(1, int(getattr(self.args, "per_device_train_batch_size", 1))) * max(
            1, int(self.args.gradient_accumulation_steps)
        )
        window = world * per_rank
        if world <= 1 or window < 2 or len(lengths) % window:
            return LengthGroupedSampler(
                per_rank, dataset=ds, lengths=lengths
            )

        import random as _random

        rng = _random.Random(int(getattr(self.args, "seed", 42)))
        order = sorted(range(len(lengths)), key=lambda i: lengths[i])
        n_windows = len(order) // window
        windows = [order[s * window : (s + 1) * window] for s in range(n_windows)]
        rng.shuffle(windows)
        out: List[int] = []
        for w in windows:
            out.extend(w)
        out.extend(order[n_windows * window :])
        return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name-or-path", required=True,
                   help="local Qwen3.5-4B model dir")
    p.add_argument("--dataset-path", type=Path, required=True,
                   help="dir of *.json training samples (messages format)")
    p.add_argument("--output-dir", type=Path, default=Path("output/memonce4b"))
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--max-length", type=int, default=32768)
    p.add_argument("--init-from", type=Path, default=None,
                   help="checkpoint dir whose model.safetensors initializes the trained params")
    p.add_argument("--resume", type=Path, default=None,
                   help="Trainer checkpoint dir to resume from")
    p.add_argument("--group-by-length", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--max-steps", type=int, default=15625)
    p.add_argument("--num-train-epochs", type=float, default=1.0)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--optim", type=str, default="adamw_torch")
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--logging-steps", type=int, default=1)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True
    )
    a = p.parse_args()
    rows = load_rows(a.dataset_path, a.max_samples)
    rank0(f"[DATA] {len(rows)} samples")
    if a.group_by_length:
        # char-length proxy per row: grouping only needs the ordering
        for row in rows:
            row["length"] = sum(len(m.get("content", "")) for m in row["messages"])
    tok = AutoTokenizer.from_pretrained(
        a.model_name_or_path, local_files_only=True, trust_remote_code=True
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    config = MemOnceConfig(base_model_name_or_path=a.model_name_or_path)
    model = MemOnceModel(config, tokenizer=tok)
    if a.init_from:
        from safetensors.torch import load_file

        weights = a.init_from / "model.safetensors"
        missing, unexpected = model.load_state_dict(load_file(str(weights)), strict=False)
        bad = [
            k
            for k in list(missing) + list(unexpected)
            if ("lora" in k or "latent_to_memory" in k or "mem_to_emb" in k
                or "memory_queries" in k)
        ]
        if bad:
            raise RuntimeError(f"init-from params not loaded: {bad[:8]}")
        rank0(f"[INIT] loaded trained params from {a.init_from}")
    if a.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    rank0(
        f"[MODEL] trainable={sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )
    args = TrainingArguments(
        output_dir=str(a.output_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=a.gradient_accumulation_steps,
        learning_rate=a.learning_rate,
        optim=a.optim,
        num_train_epochs=a.num_train_epochs,
        max_steps=a.max_steps,
        logging_steps=a.logging_steps,
        save_steps=a.save_steps,
        save_total_limit=20,
        report_to="none",
        remove_unused_columns=False,
        # Retrieval-only parameters may get no gradient when evidence is empty.
        ddp_find_unused_parameters=True,
        bf16=a.bf16,
        gradient_checkpointing=a.gradient_checkpointing,
        **(
            {"group_by_length": a.group_by_length}
            if int(transformers.__version__.split(".")[0]) < 5
            else (
                {"train_sampling_strategy": "group_by_length"}
                if a.group_by_length
                else {}
            )
        ),
    )
    trainer = LengthAwareTrainer(
        model=model,
        args=args,
        train_dataset=JsonDataset(rows),
        data_collator=Collator(tok, a.max_length),
        callbacks=[FiniteGradCallback()],
    )
    trainer.train(resume_from_checkpoint=str(a.resume) if a.resume else None)
    trainer.save_model(str(a.output_dir))
    rank0(f"[DONE] checkpoint={a.output_dir}")
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
