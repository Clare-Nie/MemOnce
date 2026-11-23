#!/usr/bin/env python
"""MemOnce (post-training) inference.

Loads a released post-training checkpoint on top of Qwen3.5-4B and runs
the write-once / read-many memory loop:

  write(history)   one 64-token memory block per <=4096-token segment,
                   FIFO short state (16 blocks), cosine retrieval keys
  answer(question) retrieve top-64 blocks + short state as an input
                   prefix, then greedy-decode with the LoRA reader

Usage:
  Self-test (weight loading, memory writes, generation):
      python inference.py --base-model M --checkpoint C --self-test

  TEXT (write four documents, answer two multi-hop questions):
      python inference.py --base-model M --checkpoint C --TEXT

  Batch evaluation (one {"qid", "messages": [...]} object per line; the
  final user turn is the question, preceding turns are written to memory):
      python inference.py --base-model M --checkpoint C \
          --input data.jsonl --output generations.jsonl

Checkpoint format: a directory containing model.safetensors with the
trained parameters (LoRA write path, readout projections, mem_to_emb),
as produced by post-training/train.py.
"""
import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "post-training"))
from model import MemOnceConfig, MemOnceModel  # noqa: E402
from train import parse_template  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


def load_memonce(base_model: str, checkpoint: str = None,
                 attn_implementation: str = "sdpa", device: str = "cuda"):
    tok = AutoTokenizer.from_pretrained(base_model, local_files_only=True,
                                        trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    config = MemOnceConfig(base_model_name_or_path=base_model,
                           attn_implementation=attn_implementation)
    model = MemOnceModel(config, tokenizer=tok)
    if checkpoint:
        from safetensors.torch import load_file
        sd = load_file(str(Path(checkpoint) / "model.safetensors"))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        trained = [k for k in list(missing) + list(unexpected)
                   if ("lora" in k or "readout" in k or "mem_to_emb" in k)]
        if trained:
            raise RuntimeError(f"checkpoint params not loaded: {trained[:8]}")
        print(f"[load] trained params loaded from {checkpoint}", flush=True)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, tok


class MemOnceChat:
    """Write-once/read-many memory session over one MemOnceModel."""

    def __init__(self, model, tok, device: str = "cuda"):
        self.model, self.tok, self.device = model, tok, device
        self.state = None          # (1, <=1024, hidden) FIFO short state
        self.pool = []             # list of (1, 64, hidden) memory blocks
        self.keys = []             # list of (1, dim) retrieval keys
        self.n_written = 0

    @torch.no_grad()
    def write(self, messages):
        """Ingest history turns into memory. `messages` is a list of
        {"role", "content"} dicts; documents should be individual user
        turns."""
        m, tok = self.model, self.tok
        ids = tok.apply_chat_template(messages, tokenize=True,
                                      add_generation_prompt=False,
                                      enable_thinking=False)
        if not isinstance(ids, list):
            ids = list(ids["input_ids"])
        ids = torch.tensor([ids], device=self.device)
        spans, _ = parse_template(list(ids[0].cpu().numpy()), tok)
        segs = [ids[:, s:e] for s, _, e in spans]
        # split over-long turns into <= target_len segments: one block each
        t_len = int(m.config.target_len)
        split = []
        for s in segs:
            while s.shape[1] > t_len:
                split.append(s[:, :t_len])
                s = s[:, t_len:]
            if s.shape[1]:
                split.append(s)
        segs = split
        max_seg = max(1, m.config.turns_per_state)
        off = 0
        while off < len(segs):
            c_len, c_end = 0, off
            while c_end < len(segs) and c_end - off < max_seg:
                d = segs[c_end].shape[1]
                if c_end > off and c_len + d > t_len:
                    break
                c_len += d
                c_end += 1
            blocks = m._generate_memory_blocks(segs[off:c_end], self.state)
            for r in range(blocks.shape[0]):
                b = blocks[r:r + 1]
                self.pool.append(b)
                self.keys.append(m.mem_to_emb(b[:, -1, :]))
                self.state = m._update_state(self.state, b)
            off = c_end
        self.n_written += len(segs)
        return len(segs)

    @torch.no_grad()
    def answer(self, question: str, max_new_tokens: int = 256,
               retrieve_blocks: int = None):
        """Retrieve relevant memory blocks and generate the reply."""
        m, tok = self.model, self.tok
        # enable_thinking=False: the Qwen3.5 template then inserts the empty
        # thinking block after the prompt, so generation starts at the answer
        # — matching the training-time label layout.
        q_ids = tok.apply_chat_template([{"role": "user", "content": question}],
                                        tokenize=True, add_generation_prompt=True,
                                        enable_thinking=False)
        if not isinstance(q_ids, list):
            q_ids = list(q_ids["input_ids"])
        q_ids = torch.tensor([q_ids], device=self.device)

        im_end = tok.convert_tokens_to_ids("<|im_end|>")
        ends = (q_ids[0] == im_end).nonzero()
        key_end = int(ends[0]) if ends.numel() else q_ids.shape[1]
        q_key_ids = q_ids[:, :max(1, key_end)]
        q_block = m._generate_memory_blocks([q_key_ids], self.state)
        # recent turns already live in the short state: exclude them from pool
        keep = max(0, len(self.pool) - m.config.turns_per_state)
        retrieved, indices = None, torch.empty(0, dtype=torch.long, device=self.device)
        if keep:
            if retrieve_blocks is not None:
                m.config.retrieve_blocks = int(retrieve_blocks)
            key_matrix = torch.cat(self.keys[:keep], dim=0).unsqueeze(0)
            retrieved, indices = m._retrieve(m.mem_to_emb(q_block[:, -1, :]),
                                             key_matrix, self.pool[:keep])
        mem = torch.cat([p for p in (retrieved, self.state) if p is not None], dim=1) \
            if (retrieved is not None or self.state is not None) else None
        mem_cache = m._build_memory_cache(mem)
        mem_len = 0 if mem is None else mem.shape[1]

        q_emb = m.get_input_embeddings()(q_ids)
        da = m.model.disable_adapter if hasattr(m.model, "disable_adapter") \
            else contextlib.nullcontext
        gen_ids = []
        with da():
            cpos = torch.arange(mem_len, mem_len + q_ids.shape[1], device=self.device)
            amask = torch.ones(1, mem_len + q_ids.shape[1], device=self.device,
                               dtype=torch.long)
            out = m.model(inputs_embeds=q_emb, attention_mask=amask,
                          past_key_values=mem_cache, cache_position=cpos,
                          position_ids=cpos.unsqueeze(0), use_cache=True,
                          return_dict=True)
            cache, logits = out.past_key_values, out.logits[:, -1]
            nxt = logits.argmax(-1)
            gen_ids.append(nxt.item())
            for _ in range(max_new_tokens - 1):
                if nxt.item() == tok.eos_token_id:
                    break
                npos = mem_len + q_ids.shape[1] + len(gen_ids)
                nemb = m.get_input_embeddings()(nxt.unsqueeze(0))
                ncp = torch.tensor([npos], device=self.device)
                nm = torch.ones(1, mem_len + q_ids.shape[1] + len(gen_ids),
                                device=self.device, dtype=torch.long)
                out = m.model(inputs_embeds=nemb, attention_mask=nm,
                              past_key_values=cache, cache_position=ncp,
                              position_ids=ncp.unsqueeze(0), use_cache=True,
                              return_dict=True)
                cache, logits = out.past_key_values, out.logits[:, -1]
                nxt = logits.argmax(-1)
                gen_ids.append(nxt.item())
        info = {"blocks": len(self.pool), "state_tokens": 0 if self.state is None
                else int(self.state.shape[1]), "retrieved": indices.tolist()}
        return tok.decode(gen_ids, skip_special_tokens=True), info


TEXT_DOCS = [
    {"role": "user", "content":
     "[doc1] The city of Luminara.\nLuminara is a coastal city founded in 1623. "
     "Its harbor lighthouse, the Amber Spire, was designed by architect Elena Voss "
     "and completed in 1701. The city is famous for its glass-blowing workshops."},
    {"role": "user", "content":
     "[doc2] Elena Voss.\nElena Voss (1655-1724) was a Luminaran architect and "
     "engineer. Besides the Amber Spire, she designed the tidal gates of the old "
     "harbor. She studied engineering in Port Kellen under master Bram Holt."},
    {"role": "user", "content":
     "[doc3] Glass-blowing in Luminara.\nThe glass workshops of Luminara export "
     "colored glassware across the southern sea. The signature 'Voss blue' tint was "
     "named after the architect's family, who funded the first public furnace."},
    {"role": "user", "content":
     "[doc4] Port Kellen.\nPort Kellen is a university town 40 km inland. Its "
     "engineering academy, founded in 1602, trained many of the region's harbor "
     "architects, including Elena Voss."},
]
TEXT_QS = [
    "Who designed the Amber Spire lighthouse, and where did that person study engineering?",
    "Which Luminara export is named after the architect's family?",
]

TEXT_FILLERS = [
    {"role": "user", "content":
     f"[note {i:02d}] Miscellaneous harbor records.\nTide tables, customs "
     f"ledgers and shipping schedules for the southern coast, volume {i}. "
     "These routine entries contain no information about architects or "
     "lighthouses."}
    for i in range(1, 17)
]


def run_TEXT(chat):
    n = chat.write(TEXT_DOCS + TEXT_FILLERS)
    keep = max(0, len(chat.pool) - chat.model.config.turns_per_state)
    print(f"[TEXT] wrote {n} segments -> {len(chat.pool)} memory blocks "
          f"({keep} retrievable, rest in short state)", flush=True)
    for q in TEXT_QS:
        t0 = time.time()
        ans, info = chat.answer(q)
        print(f"\nQ: {q}\nA: {ans}\n"
              f"[retrieved block indices: {info['retrieved']}, "
              f"{time.time() - t0:.1f}s]", flush=True)


def run_self_test(chat, checkpoint):
    assert checkpoint, "--self-test requires --checkpoint"
    n = chat.write(TEXT_DOCS)
    assert len(chat.pool) >= n and chat.state is not None, "memory not written"
    ans, info = chat.answer(TEXT_QS[0], max_new_tokens=64)
    assert ans.strip(), "empty generation"
    assert info["state_tokens"] > 0, "empty memory state"
    print(f"[self-test] OK: {n} segments, {len(chat.pool)} blocks, "
          f"state={info['state_tokens']} tokens, answer='{ans[:60]}...'", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", required=True, help="local Qwen3.5-4B dir")
    p.add_argument("--checkpoint", default=None,
                   help="released checkpoint dir (model.safetensors)")
    p.add_argument("--attn-implementation", default="sdpa")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--retrieve-blocks", type=int, default=None)
    p.add_argument("--TEXT", action="store_true")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--input", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    a = p.parse_args()

    model, tok = load_memonce(a.base_model, a.checkpoint, a.attn_implementation)
    if a.TEXT:
        run_TEXT(MemOnceChat(model, tok))
        return
    if a.self_test:
        run_self_test(MemOnceChat(model, tok), a.checkpoint)
        return
    if a.input:
        rows = [json.loads(l) for l in open(a.input)]
        out = open(a.output, "w") if a.output else sys.stdout
        for i, r in enumerate(rows):
            chat = MemOnceChat(model, tok)
            msgs = r["messages"]
            if len(msgs) > 1:
                chat.write(msgs[:-1])
            ans, info = chat.answer(msgs[-1]["content"], a.max_new_tokens,
                                    a.retrieve_blocks)
            rec = dict(r)
            rec["generated"] = ans
            rec["mem_info"] = info
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            print(f"[{i + 1}/{len(rows)}] blocks={info['blocks']} "
                  f"retrieved={len(info['retrieved'])}", flush=True)
        return
    p.error("one of --TEXT / --self-test / --input is required")


if __name__ == "__main__":
    main()
