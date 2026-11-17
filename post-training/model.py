"""MemOnce: native-memory model for Qwen3.5-4B (post-training definition).

Frozen hybrid backbone + LoRA writer. Conversations are compressed into
64-token memory blocks (tail hidden states); keys come from mem_to_emb;
retrieval is cosine/temperature over the block pool, top-scoring blocks
capped at retrieve_blocks. Loss = LM + per-positive NCE over [pos, negs],
scaled by log(num_negatives + 1).
"""

import contextlib
import copy
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM, PreTrainedModel, PretrainedConfig
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast


class MemOnceConfig(PretrainedConfig):
    model_type = "memonce"

    def __init__(
        self,
        base_model_name_or_path: str = None,
        num_memory_tokens: int = 64,
        short_state_tokens: int = 1024,
        retrieve_blocks: int = 64,
        retrieval_embedding_dim: int = 1024,
        lora_r: int = 128,
        lora_alpha: int = 256,
        lora_dropout: float = 0.05,
        local_files_only: bool = True,
        attn_implementation: Optional[str] = "sdpa",
        target_len: int = 4096,
        retrieval_loss_weight: float = 0.1,
        retrieval_temperature: float = 0.1,
        retrieval_score_threshold: float = 5.0,
        retrieval_margin: float = 2.0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if short_state_tokens % num_memory_tokens:
            raise ValueError("short_state_tokens must be divisible by num_memory_tokens")
        self.base_model_name_or_path = base_model_name_or_path
        self._name_or_path = base_model_name_or_path
        self.num_memory_tokens = num_memory_tokens
        self.short_state_tokens = short_state_tokens
        self.retrieve_blocks = retrieve_blocks
        self.retrieval_embedding_dim = retrieval_embedding_dim
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.local_files_only = local_files_only
        self.attn_implementation = attn_implementation
        self.target_len = target_len
        self.retrieval_loss_weight = retrieval_loss_weight
        self.retrieval_temperature = retrieval_temperature
        self.retrieval_score_threshold = retrieval_score_threshold
        self.retrieval_margin = retrieval_margin

    @property
    def turns_per_state(self) -> int:
        return self.short_state_tokens // self.num_memory_tokens


class MemOnceModel(PreTrainedModel):
    config_class = MemOnceConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: MemOnceConfig, tokenizer: Any = None):
        super().__init__(config)
        if not config.base_model_name_or_path:
            raise ValueError("base_model_name_or_path is required (Qwen3.5-4B)")
        self.tokenizer = tokenizer
        self.memory_gradient_checkpointing = False

        load_kwargs: Dict[str, Any] = {
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
            "local_files_only": config.local_files_only,
        }
        if config.attn_implementation:
            load_kwargs["attn_implementation"] = config.attn_implementation
        base = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path, **load_kwargs
        )
        for parameter in base.parameters():
            parameter.requires_grad = False
        lora = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        self.model = get_peft_model(base, lora, adapter_name="memory")
        self.model.set_adapter("memory")
        self.hidden_size = int(base.config.hidden_size)
        self.num_layers = int(base.config.num_hidden_layers)

        self.mem_to_emb = nn.Sequential(
            nn.Linear(self.hidden_size, config.retrieval_embedding_dim),
            nn.SiLU(),
            nn.Linear(config.retrieval_embedding_dim, config.retrieval_embedding_dim),
        ).to(dtype=torch.bfloat16)

        # Learnable cache generation: fp32 copies of the frozen originals,
        # one per full-attention layer.
        transformer = self._transformer()
        self.readout_ln = nn.ModuleDict()
        self.readout_kproj = nn.ModuleDict()
        self.readout_knorm = nn.ModuleDict()
        self.readout_vproj = nn.ModuleDict()
        with torch.no_grad():
            for l, layer in enumerate(transformer.layers[: self.num_layers]):
                if transformer.config.layer_types[l] != "full_attention":
                    continue
                attn = layer.self_attn
                kproj = nn.Linear(
                    self.hidden_size, attn.k_proj.out_features,
                    bias=attn.k_proj.bias is not None, dtype=torch.float32,
                )
                vproj = nn.Linear(
                    self.hidden_size, attn.v_proj.out_features,
                    bias=attn.v_proj.bias is not None, dtype=torch.float32,
                )
                kproj.weight.copy_(attn.k_proj.weight)
                vproj.weight.copy_(attn.v_proj.weight)
                if kproj.bias is not None and attn.k_proj.bias is not None:
                    kproj.bias.copy_(attn.k_proj.bias)
                if vproj.bias is not None and attn.v_proj.bias is not None:
                    vproj.bias.copy_(attn.v_proj.bias)
                self.readout_ln[str(l)] = copy.deepcopy(layer.input_layernorm).float()
                self.readout_kproj[str(l)] = kproj
                self.readout_knorm[str(l)] = copy.deepcopy(attn.k_norm).float()
                self.readout_vproj[str(l)] = vproj

        for name, parameter in self.model.named_parameters():
            parameter.requires_grad = "lora_" in name and ".memory." in name
        for module in (self.mem_to_emb, self.readout_ln, self.readout_kproj,
                       self.readout_knorm, self.readout_vproj):
            for parameter in module.parameters():
                parameter.requires_grad = True

    def _transformer(self) -> nn.Module:
        backbone = (
            self.model.get_base_model()
            if hasattr(self.model, "get_base_model")
            else self.model
        )
        return backbone.model if hasattr(backbone.model, "layers") else backbone

    def state_dict(self, *args: Any, **kwargs: Any) -> Dict[str, torch.Tensor]:
        full = super().state_dict(*args, **kwargs)
        trainable = {
            name for name, parameter in self.named_parameters() if parameter.requires_grad
        }
        return {name: tensor for name, tensor in full.items() if name in trainable}

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def gradient_checkpointing_enable(
        self, gradient_checkpointing_kwargs: Optional[Dict[str, Any]] = None
    ) -> None:
        self.memory_gradient_checkpointing = True
        self._transformer().config.use_cache = False

    def gradient_checkpointing_disable(self) -> None:
        self.memory_gradient_checkpointing = False

    def _turn_spans(
        self, input_ids: torch.Tensor, meta: Dict[str, Any]
    ) -> List[Tuple[int, int, int]]:
        raw = meta.get("turn_token_spans") or meta.get("spans")
        if raw:
            spans = [tuple(map(int, item)) for item in raw]
            if all(len(item) == 3 for item in spans) and all(
                0 <= start <= user_end <= end <= input_ids.shape[1]
                for start, user_end, end in spans
            ):
                return spans
        raise ValueError(
            "meta must contain valid turn_token_spans=[[start,user_end,end], ...]"
        )

    def _build_memory_cache(
        self, memory: Optional[torch.Tensor]
    ) -> Optional[DynamicCache]:
        if memory is None or memory.shape[1] == 0:
            return None
        transformer = self._transformer()
        ref_dtype = memory.dtype
        cache = DynamicCache(config=transformer.config)
        for l in range(self.num_layers):
            if transformer.config.layer_types[l] != "full_attention":
                continue
            key = str(l)
            head_dim = transformer.layers[l].self_attn.head_dim
            normed = self.readout_ln[key](memory.to(torch.float32))
            shape = (*normed.shape[:-1], -1, head_dim)
            k = self.readout_knorm[key](
                self.readout_kproj[key](normed).view(shape)
            ).transpose(1, 2)
            v = self.readout_vproj[key](normed).view(shape).transpose(1, 2)
            cache.layers[l].update(k.to(ref_dtype).contiguous(), v.to(ref_dtype).contiguous())
        return cache

    def _hidden_states(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        state: Optional[torch.Tensor],
    ) -> torch.Tensor:
        def run(embeds, mask, state_in):
            state_len = 0 if state_in is None else state_in.shape[1]
            past = None if state_in is None else self._build_memory_cache(state_in)
            cache_position = torch.arange(
                state_len, state_len + embeds.shape[1], device=embeds.device
            )
            output = self.model(
                inputs_embeds=embeds,
                attention_mask=mask,
                past_key_values=past,
                cache_position=cache_position,
                position_ids=cache_position.unsqueeze(0),
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            return output.hidden_states[self.num_layers]

        if self.training and self.memory_gradient_checkpointing:
            return checkpoint(run, inputs_embeds, attention_mask, state, use_reentrant=False)
        return run(inputs_embeds, attention_mask, state)

    def _generate_memory_blocks(
        self, turn_ids: Sequence[torch.Tensor], state: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Last-layer hidden over each turn's last 64 tokens, left-zero-padded.
        Returns (num_turns, 64, hidden)."""
        if not turn_ids:
            raise ValueError("turn_ids cannot be empty")
        dtype = self.get_input_embeddings().weight.dtype
        nq = int(self.config.num_memory_tokens)
        parts: List[torch.Tensor] = []
        spans: List[Tuple[int, int]] = []
        cursor = 0
        for ids in turn_ids:
            text = self.get_input_embeddings()(ids).to(dtype=dtype)
            parts.append(text)
            spans.append((cursor, cursor + text.shape[1]))
            cursor += text.shape[1]
        embeds = torch.cat(parts, dim=1)
        mask = torch.ones(embeds.shape[:2], device=embeds.device, dtype=torch.long)
        hidden = self._hidden_states(embeds, mask, state)

        blocks = []
        for ts, te in spans:
            length = te - ts
            if length >= nq:
                seg = hidden[:, te - nq : te, :]
            else:
                pad = hidden.new_zeros((1, nq - length, hidden.shape[-1]))
                seg = torch.cat([pad, hidden[:, ts:te, :]], dim=1)
            blocks.append(seg)
        return torch.cat(blocks, dim=0)

    def _update_state(
        self, state: Optional[torch.Tensor], block: torch.Tensor
    ) -> torch.Tensor:
        if state is None:
            return block
        block_size = self.config.num_memory_tokens
        if state.shape[1] < self.config.short_state_tokens:
            return torch.cat([state, block], dim=1)
        return torch.cat([state[:, block_size:], block], dim=1)

    def _retrieve(
        self,
        query_key: torch.Tensor,
        cached_keys: torch.Tensor,
        blocks: Sequence[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        if not blocks:
            return None, torch.empty(0, dtype=torch.long, device=query_key.device)
        scores = F.cosine_similarity(
            query_key[:, None, :].float(), cached_keys.float(), dim=-1
        ) / float(self.config.retrieval_temperature)
        # No threshold: top-scoring blocks, capped at retrieve_blocks.
        max_blocks = int(self.config.retrieve_blocks)
        if max_blocks == 0:
            # retrieve_blocks=0 -> retrieval disabled entirely.
            return None, torch.empty(0, dtype=torch.long, device=blocks[0].device)
        count = len(blocks) if max_blocks < 0 else min(max_blocks, len(blocks))
        top_indices = torch.topk(scores[0], k=count, dim=0).indices
        indices = torch.sort(top_indices).values
        return torch.cat([blocks[int(index)] for index in indices], dim=1), indices

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        meta: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ):
        if input_ids.shape[0] != 1:
            raise ValueError("native memory training requires batch_size=1")
        meta_item = meta[0] if isinstance(meta, list) and meta else {}
        spans = self._turn_spans(input_ids, meta_item)
        if len(spans) < 2:
            raise ValueError("At least one history turn and one query turn are required")

        state: Optional[torch.Tensor] = None
        pool: List[torch.Tensor] = []
        cached_keys: List[torch.Tensor] = []

        history_segments = [input_ids[:, start:end] for start, _, end in spans[:-1]]
        target_len = int(self.config.target_len)
        max_segments = max(1, self.config.turns_per_state)
        offset = 0
        while offset < len(history_segments):
            # Budget the chunk TEXT only (<= target_len); the fixed 1024-token
            # state prefix rides on top and is not counted.
            chunk_len, chunk_end = 0, offset
            while chunk_end < len(history_segments) and chunk_end - offset < max_segments:
                segment_len = history_segments[chunk_end].shape[1]
                if chunk_end > offset and chunk_len + segment_len > target_len:
                    break
                if chunk_end == offset and chunk_len + segment_len > target_len:
                    chunk_end += 1
                    break
                chunk_len += segment_len
                chunk_end += 1
            blocks = self._generate_memory_blocks(
                history_segments[offset:chunk_end], state
            )
            for t in range(blocks.shape[0]):
                block = blocks[t : t + 1]
                pool.append(block)
                cached_keys.append(self.mem_to_emb(block[:, -1, :]))
                state = self._update_state(state, block)
            offset = chunk_end

        query_start, query_user_end, query_end = spans[-1]
        query_ids = input_ids[:, query_start:query_end]
        prompt_end = query_user_end if query_user_end > query_start else query_end
        query_blocks = self._generate_memory_blocks(
            [input_ids[:, query_start:prompt_end]], state
        )
        query_key = self.mem_to_emb(query_blocks[:, -1, :])

        _nms = os.environ.get("NATIVE_MEMORY_STATS_FILE")
        if _nms:
            import json as _json
            _pool_b = sum(b.numel() * b.element_size() for b in pool)
            _keys_b = sum(k.numel() * k.element_size() for k in cached_keys)
            _qs = max(0, len(pool) - self.config.turns_per_state)
            _fifo_b = sum(b.numel() * b.element_size() for b in pool[_qs:])
            with open(_nms, "a") as _f:
                _f.write(_json.dumps({
                    "n_blocks": len(pool),
                    "cache_mb": round((_pool_b + _keys_b) / 2 ** 20, 1),
                    "fifo_mb": round(_fifo_b / 2 ** 20, 1)}) + "\n")
        # Retrieval over pool blocks outside the FIFO window.
        queue_start = max(0, len(pool) - self.config.turns_per_state)
        retrieval_pool = pool[:queue_start]
        retrieved: Optional[torch.Tensor] = None
        retrieved_indices = torch.empty(0, dtype=torch.long, device=input_ids.device)
        if retrieval_pool:
            key_matrix = torch.cat(cached_keys[:queue_start], dim=0).unsqueeze(0)
            retrieved, retrieved_indices = self._retrieve(query_key, key_matrix, retrieval_pool)

        # Per-positive NCE over the retrievable pool.
        retrieval_loss = None
        evidence_indices = meta_item.get("evidence_turns") or []
        pos_scores_debug: List[float] = []
        neg_scores_debug: List[float] = []
        pos_score_max = neg_score_max = 0.0
        pool_keys = cached_keys[:queue_start]
        if not pool_keys:
            # All blocks inside the FIFO window: nothing retrievable, but
            # the evidence labels still supervise the query key over the pool.
            pool_keys = cached_keys
        if evidence_indices and pool_keys:
            key_matrix = torch.cat(pool_keys, dim=0).unsqueeze(0).float()
            query_norm = F.normalize(query_key.float(), p=2, dim=-1)
            history_norm = F.normalize(key_matrix, p=2, dim=-1)
            valid_pos = sorted(
                {
                    int(index)
                    for index in evidence_indices
                    if 0 <= int(index) < history_norm.shape[1]
                }
            )
            if valid_pos:
                all_scores = torch.bmm(
                    query_norm[:, None, :], history_norm.transpose(1, 2)
                ).squeeze(1) / float(self.config.retrieval_temperature)
                pos_scores = all_scores[:, valid_pos]
                pos_scores_debug = [
                    round(float(value), 4)
                    for value in pos_scores.detach().float().flatten().cpu().tolist()
                ]
                pos_score_max = float(pos_scores.detach().max().item())
                neg = [
                    index
                    for index in range(history_norm.shape[1])
                    if index not in valid_pos
                ]
                if neg:
                    neg_scores = all_scores[:, neg]
                    # NCE per positive: logaddexp(s_pos, logsumexp(s_neg)) - s_pos,
                    # scaled by log(num_negatives + 1), averaged over positives.
                    nce_per_pos = (
                        torch.logaddexp(
                            pos_scores,
                            torch.logsumexp(neg_scores, dim=-1, keepdim=True),
                        )
                        - pos_scores
                    )
                    retrieval_loss = nce_per_pos.mean() * float(
                        torch.log(
                            torch.tensor(
                                float(neg_scores.shape[1]) + 1.0,
                                device=neg_scores.device,
                            )
                        )
                    )
                    neg_scores_debug = [
                        round(float(value), 4)
                        for value in neg_scores.detach().float().flatten().cpu().tolist()
                    ]
                    neg_score_max = float(neg_scores.detach().max().item())
                else:
                    retrieval_loss = all_scores.new_zeros(())

        # Reader: query over [retrieved + FIFO] prefix, LoRA off.
        memory_parts = [p for p in (retrieved, state) if p is not None]
        memory = torch.cat(memory_parts, dim=1) if memory_parts else None
        memory_len = memory.shape[1] if memory is not None else 0
        query_labels = labels[:, query_start:query_end] if labels is not None else None
        if query_labels is None or not (query_labels != -100).any():
            raise ValueError("labels must contain supervised query tokens")

        memory_cache = self._build_memory_cache(memory)
        query_embeds = self.get_input_embeddings()(query_ids)
        mask_dtype = attention_mask.dtype if attention_mask is not None else torch.long
        query_attention_mask = torch.ones(
            (query_ids.shape[0], memory_len + query_ids.shape[1]),
            device=input_ids.device,
            dtype=mask_dtype,
        )
        query_cache_position = torch.arange(
            memory_len, memory_len + query_ids.shape[1], device=input_ids.device
        )
        disable_adapter = (
            self.model.disable_adapter
            if hasattr(self.model, "disable_adapter")
            else contextlib.nullcontext()
        )
        with disable_adapter():
            output = self.model(
                inputs_embeds=query_embeds,
                attention_mask=query_attention_mask,
                labels=query_labels,
                past_key_values=memory_cache,
                cache_position=query_cache_position,
                position_ids=query_cache_position.unsqueeze(0),
                use_cache=True,
                return_dict=True,
            )

        lm_loss = output.loss
        token_correct = token_total = 0
        logits = getattr(output, "logits", None)
        if logits is not None:
            with torch.no_grad():
                valid = query_labels[:, 1:] != -100
                if valid.any() and logits.shape[1] > 1:
                    pred = logits[:, :-1, :].argmax(dim=-1)
                    token_correct = int(((pred == query_labels[:, 1:]) & valid).sum())
                    token_total = int(valid.sum())
        token_acc = token_correct / max(1, token_total)
        total_loss = lm_loss
        if retrieval_loss is not None:
            total_loss = total_loss + self.config.retrieval_loss_weight * retrieval_loss
        evidence_turns = sorted(
            {int(i) for i in evidence_indices if 0 <= int(i) < len(pool)}
        )
        print(
            f"[NATIVEMEMORY] rank={os.environ.get('RANK', '0')} "
            f"history_turns={len(spans) - 1} memory={memory_len} "
            f"retrievable_turns={len(retrieval_pool)} "
            f"retrieved_k=[{retrieved_indices.tolist()}] "
            f"evidence_turns={evidence_turns} "
            f"pos_scores={pos_scores_debug} neg_scores={neg_scores_debug} "
            f"pos_score_max={pos_score_max:.4f} neg_score_max={neg_score_max:.4f} "
            f"lm_loss={lm_loss.detach().float().item():.4f} "
            f"retrieval_loss={retrieval_loss.detach().float().item() if retrieval_loss is not None else 0.0:.4f} "
            f"total_loss={total_loss.detach().float().item():.4f} "
            f"last_answer_token_acc={token_acc:.4f} "
            f"last_answer_token={token_correct}/{token_total}",
            flush=True,
        )
        return CausalLMOutputWithPast(
            loss=total_loss,
            logits=logits,
            past_key_values=getattr(output, "past_key_values", None),
        )


__all__ = ["MemOnceConfig", "MemOnceModel"]
