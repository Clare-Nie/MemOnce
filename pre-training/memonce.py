"""MemOnce: 3:1 GatedDeltaNet:attention backbone + write-once compressed memory.

    from memonce import MemOnce
    model = MemOnce.from_size("158m")                  # 128->16 spans (8x), full memory
    model = MemOnce.from_size("945m", mem_budget=1024) # + top-k memory retrieval

Requires: torch >= 2.4, fla (chunk_gated_delta_rule).
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from fla.ops.gated_delta_rule import chunk_gated_delta_rule as _fla_chunk
from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule as _fla_recurrent


@dataclass
class MemOnceConfig:
    vocab_size: int = 32000
    dim: int = 1024
    n_layers: int = 24
    n_heads: int = 6
    head_dim: int = 256
    n_kv_heads: int = 2
    attn_gated: bool = True
    mlp_mult: float = 8 / 3
    rope_base: float = 10000000.0
    tie_embeddings: bool = False
    max_seq_len: int = 4096


class MemOnce(nn.Module):
    """Write-once compressed memory over a 3:1 linear:attention backbone.

    The sequence is processed in serial blocks of block_size tokens. Every
    span of span tokens appends its last mem_tokens hiddens to a growing
    memory state (span/mem_tokens = compression ratio, 128/16 = 8x). Each
    full-attention layer projects the memory to its own K/V and attends to
    [memory | own block]; the GDN recurrent state persists across blocks.
    mem_budget=None attends to the full memory (at 4096 context: everything).
    """

    persist_state = True

    SIZES = {
        "158m": dict(dim=640,  mlp_mult=1.6, head_dim=160),
        "397m": dict(),
        "603m": dict(dim=1536, mlp_mult=1.6, head_dim=256, n_heads=9,  n_kv_heads=3),
        "945m": dict(dim=2048, mlp_mult=1.6, head_dim=256, n_heads=12, n_kv_heads=3),
    }

    @classmethod
    def from_size(cls, size="397m", **kw):
        return cls(MemOnceConfig(**cls.SIZES[size]), **kw)

    def __init__(self, cfg: MemOnceConfig, span=128, mem_tokens=16,
                 block_size=1024, mem_budget=None):
        super().__init__()
        if span <= 0 or mem_tokens <= 0 or block_size <= 0:
            raise ValueError("span, mem_tokens and block_size must be positive")
        if span % mem_tokens or block_size % span:
            raise ValueError("mem_tokens must divide span; span must divide block_size")
        if mem_budget is not None and (mem_budget <= 0 or mem_budget % mem_tokens):
            raise ValueError("mem_budget must be a positive multiple of mem_tokens")
        self.backbone = Backbone(cfg)
        self.cfg = cfg
        self.span = span
        self.mem_tokens = mem_tokens
        self.block_size = block_size
        self.mem_budget = mem_budget

        self.readout_ln = nn.ModuleDict()
        self.readout_kproj = nn.ModuleDict()
        self.readout_vproj = nn.ModuleDict()
        self.readout_knorm = nn.ModuleDict()
        for l, blk in enumerate(self.backbone.blocks):
            mix = blk.mixer
            if not isinstance(mix, AttentionMixer):
                continue
            key = str(l)
            self.readout_ln[key] = RMSNorm(cfg.dim)
            self.readout_kproj[key] = nn.Linear(cfg.dim, mix.wk.out_features, bias=False)
            self.readout_vproj[key] = nn.Linear(cfg.dim, mix.wv.out_features, bias=False)
            self.readout_knorm[key] = RMSNorm(mix.head_dim)

    def _retrieve(self, state, block_embed):
        if (self.mem_budget is None or state is None
                or state.shape[1] <= self.mem_budget):
            return state
        B, M, D = state.shape
        mt = self.mem_tokens
        keep = self.mem_budget // mt
        with torch.no_grad():
            span_mean = state.detach().view(B, M // mt, mt, D).mean(dim=2).float()
            query = block_embed.detach().mean(dim=1, keepdim=True).float()
            scores = F.cosine_similarity(span_mean, query, dim=-1)
            idx = scores.topk(keep, dim=1).indices.sort(dim=1).values
        gathered = state.view(B, M // mt, mt, D).gather(
            1, idx.unsqueeze(-1).unsqueeze(-1).expand(B, keep, mt, D))
        return gathered.reshape(B, keep * mt, D)

    def _build_memory_cache(self, state):
        if state is None or state.shape[1] == 0:
            return None
        caches = {}
        state_f = state.float()
        for l, blk in enumerate(self.backbone.blocks):
            mix = blk.mixer
            if not isinstance(mix, AttentionMixer):
                continue
            key = str(l)
            hd = mix.head_dim
            normed = self.readout_ln[key](state_f).to(self.readout_kproj[key].weight.dtype)
            k = self.readout_knorm[key](
                self.readout_kproj[key](normed).view(*normed.shape[:-1], -1, hd)
            ).transpose(1, 2)
            v = self.readout_vproj[key](normed).view(
                *normed.shape[:-1], -1, hd).transpose(1, 2)
            caches[l] = (k.to(state.dtype), v.to(state.dtype))
        return caches

    def _hidden_states(self, input_ids, mem_state, linear_state=None):
        x = self.backbone.embed(input_ids)
        mem_state = self._retrieve(mem_state, x)
        M = 0 if mem_state is None else mem_state.shape[1]
        caches = {} if mem_state is None else self._build_memory_cache(mem_state)
        if linear_state is None:
            linear_state = _LayerStateCache()
        for i, blk in enumerate(self.backbone.blocks):
            kw = {}
            if i in caches:
                kw.update(past_kv=caches[i], position_offset=M)
            if blk.mixer_code == "q":
                kw.update(past_kv=linear_state, use_cache=True)
            x, new_state = blk(x, **kw)
            if blk.mixer_code == "q":
                linear_state = new_state
        return x, linear_state

    def forward(self, input_ids, targets=None):
        _, L = input_ids.shape
        bs = self.block_size
        if L == 0:
            raise ValueError("input sequence must be non-empty")
        n_blocks = (L + bs - 1) // bs

        state = None
        linear_state = None
        logits_blocks = []
        for b in range(n_blocks - 1):
            start, end = b * bs, min((b + 1) * bs, L)
            seg_ids = input_ids[:, start:end]
            hidden, linear_state = self._hidden_states(seg_ids, state, linear_state)
            logits_blocks.append(self.backbone.lm_head(self.backbone.norm_f(hidden)))
            for s in range(seg_ids.shape[1] // self.span):
                tail = s * self.span + self.span
                block_mem = hidden[:, tail - self.mem_tokens: tail]
                state = block_mem if state is None else torch.cat([state, block_mem], dim=1)

        h, _ = self._hidden_states(input_ids[:, (n_blocks - 1) * bs:], state, linear_state)
        logits_blocks.append(self.backbone.lm_head(self.backbone.norm_f(h)))
        logits = torch.cat(logits_blocks, dim=1)

        out = {"logits": logits,
               "mem_len": 0 if state is None else state.shape[1]}
        if targets is None:
            return out
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)).float(), targets.reshape(-1),
            ignore_index=-100)
        out["loss"] = loss
        out["supervised_tokens"] = int((targets != -100).sum().item())
        return out

    def num_params(self, non_embedding=True):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.backbone.embed.weight.numel()
            if not self.cfg.tie_embeddings:
                n -= self.backbone.lm_head.weight.numel()
        return n


def RMSNorm(dim):
    return nn.RMSNorm(dim, eps=1e-6)


def _build_rope(head_dim, base):
    return 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))


def _apply_rope(x, inv_freq, offset=0):
    T = x.shape[-2]
    hd = x.shape[-1]
    pos = torch.arange(offset, offset + T, device=x.device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq.to(x.device))
    cos = freqs.cos().view(1, 1, T, -1).to(x.dtype)
    sin = freqs.sin().view(1, 1, T, -1).to(x.dtype)
    x1, x2 = x[..., : hd // 2], x[..., hd // 2 :]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class GatedMLP(nn.Module):
    def __init__(self, dim, hidden, multiple_of=256):
        hidden = multiple_of * ((hidden + multiple_of - 1) // multiple_of)
        super().__init__()
        self.w1 = nn.Linear(dim, 2 * hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        g, u = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(g) * u)


class AttentionMixer(nn.Module):
    def __init__(self, dim, n_heads, head_dim, n_kv_heads, rope_base, gated, layer_idx):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.head_dim = head_dim
        self.gated = gated
        hidden_q = self.n_heads * self.head_dim
        hidden_kv = self.n_kv_heads * self.head_dim
        self.wq = nn.Linear(dim, hidden_q * (2 if gated else 1), bias=False)
        self.wk = nn.Linear(dim, hidden_kv, bias=False)
        self.wv = nn.Linear(dim, hidden_kv, bias=False)
        self.wo = nn.Linear(hidden_q, dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.register_buffer("rope_freqs", _build_rope(self.head_dim, rope_base),
                             persistent=False)
        self.layer_idx = layer_idx

    def forward(self, x, past_kv=None, use_cache=False, position_offset=0):
        B, T, _ = x.shape
        q = self.wq(x)
        if self.gated:
            q, gate = q.view(B, T, -1, self.head_dim * 2).chunk(2, dim=-1)
            q = q.reshape(B, T, self.n_heads, self.head_dim)
            gate = gate.reshape(B, T, self.n_heads * self.head_dim)
        else:
            q = q.view(B, T, self.n_heads, self.head_dim)
            gate = None
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)
        q, k = self.q_norm(q), self.k_norm(k)
        q = _apply_rope(q.transpose(1, 2), self.rope_freqs, position_offset)
        k = _apply_rope(k.transpose(1, 2), self.rope_freqs, position_offset)
        v = v.transpose(1, 2)
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        new_kv = (k, v) if use_cache else None
        if k.shape[1] != q.shape[1]:
            reps = q.shape[1] // k.shape[1]
            k = k.repeat_interleave(reps, dim=1)
            v = v.repeat_interleave(reps, dim=1)
        if past_kv is None:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            i = (torch.arange(T, device=q.device) + past_kv[0].shape[2]).view(-1, 1)
            j = torch.arange(k.shape[-2], device=q.device).view(1, -1)
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=(j <= i))
        o = o.transpose(1, 2).reshape(B, T, -1)
        if gate is not None:
            o = o * torch.sigmoid(gate)
        return self.wo(o), new_kv


class _RMSNormGated(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x, gate):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = self.weight * x.to(dtype)
        return (x * F.silu(gate.float())).to(dtype)


class Qwen35GDNMixer(nn.Module):
    def __init__(self, dim, layer_idx=0, n_k_heads=8, head_k_dim=128,
                 n_v_heads=16, head_v_dim=128, conv_kernel=4, **_):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_k_heads, self.head_k_dim = n_k_heads, head_k_dim
        self.n_v_heads, self.head_v_dim = n_v_heads, head_v_dim
        key_dim = n_k_heads * head_k_dim
        value_dim = n_v_heads * head_v_dim
        self.key_dim, self.value_dim = key_dim, value_dim
        self.conv_dim = key_dim * 2 + value_dim
        self.conv_kernel = conv_kernel
        self.in_proj_qkv = nn.Linear(dim, key_dim * 2 + value_dim, bias=False)
        self.in_proj_z = nn.Linear(dim, value_dim, bias=False)
        self.in_proj_b = nn.Linear(dim, n_v_heads, bias=False)
        self.in_proj_a = nn.Linear(dim, n_v_heads, bias=False)
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, conv_kernel,
                                groups=self.conv_dim, padding=conv_kernel - 1,
                                bias=False)
        self.dt_bias = nn.Parameter(torch.ones(n_v_heads))
        A = torch.empty(n_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.norm = _RMSNormGated(head_v_dim)
        self.out_proj = nn.Linear(value_dim, dim, bias=False)

    def forward(self, x, past_kv=None, use_cache=False, position_offset=0, **_):
        B, T, _ = x.shape
        state = past_kv.states[self.layer_idx] if (
            past_kv is not None and len(past_kv.states) > self.layer_idx) else None
        conv_state = state["conv_state"] if state else None
        recurrent_state = state["recurrent_state"] if state else None

        mixed_qkv = self.in_proj_qkv(x).transpose(1, 2)
        z = self.in_proj_z(x)
        b = self.in_proj_b(x)
        a = self.in_proj_a(x)

        if conv_state is not None:
            mixed_qkv = torch.cat([conv_state, mixed_qkv], dim=-1)
        if use_cache:
            new_conv_state = F.pad(
                mixed_qkv, (self.conv_kernel - mixed_qkv.shape[-1], 0))
        mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :mixed_qkv.shape[-1]])
        if conv_state is not None:
            mixed_qkv = mixed_qkv[:, :, -T:]

        q, k, v = torch.split(mixed_qkv.transpose(1, 2),
                              [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.view(B, T, self.n_k_heads, self.head_k_dim)
        k = k.view(B, T, self.n_k_heads, self.head_k_dim)
        v = v.view(B, T, self.n_v_heads, self.head_v_dim)
        reps = self.n_v_heads // self.n_k_heads
        if reps > 1:
            q = q.repeat_interleave(reps, dim=2)
            k = k.repeat_interleave(reps, dim=2)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        if T == 1:
            core, new_recurrent = _fla_recurrent(
                q, k, v, g=g, beta=beta, initial_state=recurrent_state,
                output_final_state=use_cache, use_qk_l2norm_in_kernel=True)
        else:
            core, new_recurrent = _fla_chunk(
                q, k, v, g=g, beta=beta, initial_state=recurrent_state,
                output_final_state=use_cache, use_qk_l2norm_in_kernel=True)

        core = self.norm(core.reshape(-1, self.head_v_dim),
                         z.reshape(-1, self.head_v_dim))
        out = self.out_proj(core.reshape(B, T, self.value_dim))

        if use_cache and past_kv is not None:
            past_kv.update(conv_state=new_conv_state,
                           recurrent_state=new_recurrent,
                           layer_idx=self.layer_idx, offset=T)
        return out, (past_kv if use_cache else None)


class _LayerStateCache:
    def __init__(self, seen_tokens=0):
        self.states = []
        self._seen_tokens = int(seen_tokens)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, layer_idx):
        return self.states[layer_idx]

    def update(self, recurrent_state=None, attn_state=None, conv_state=None,
               ffn_state=None, layer_idx=0, offset=1, cache_kwargs=None, **_):
        if len(self.states) <= layer_idx:
            if layer_idx == 0:
                self._seen_tokens += offset
            while len(self.states) <= layer_idx:
                self.states.append({
                    "recurrent_state": None, "attn_state": None,
                    "conv_state": None, "ffn_state": None})
            self.states[layer_idx].update({
                "recurrent_state": recurrent_state, "attn_state": attn_state,
                "conv_state": conv_state, "ffn_state": ffn_state})
        else:
            if layer_idx == len(self.states) - 1:
                self._seen_tokens += offset
            state = self.states[layer_idx]
            if recurrent_state is not None:
                state["recurrent_state"] = recurrent_state
            if attn_state is not None:
                state["attn_state"] = attn_state
            if conv_state is not None:
                state["conv_state"] = conv_state
            if ffn_state is not None:
                state["ffn_state"] = ffn_state
        return self.states[layer_idx]

    def get_seq_length(self, layer_idx=0):
        return self._seen_tokens if len(self.states) > layer_idx else 0

    def reset(self):
        self.states.clear()
        self._seen_tokens = 0


class Block(nn.Module):
    def __init__(self, cfg, mixer_code, layer_idx):
        super().__init__()
        self.mixer_code = mixer_code
        self.norm1 = RMSNorm(cfg.dim)
        self.norm2 = RMSNorm(cfg.dim)
        if mixer_code == "a":
            self.mixer = AttentionMixer(cfg.dim, cfg.n_heads, cfg.head_dim,
                                        cfg.n_kv_heads, cfg.rope_base,
                                        cfg.attn_gated, layer_idx)
        elif mixer_code == "q":
            self.mixer = Qwen35GDNMixer(dim=cfg.dim, layer_idx=layer_idx)
        else:
            raise ValueError(mixer_code)
        self.mlp = GatedMLP(cfg.dim, int(cfg.dim * cfg.mlp_mult))

    def forward(self, x, **kw):
        h, cache = self.mixer(self.norm1(x), **kw)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x, cache


class Backbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        unit = ["q", "q", "q", "a"]
        self.blocks = nn.ModuleList(Block(cfg, unit[i % 4], i)
                                    for i in range(cfg.n_layers))
        self.norm_f = RMSNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.lm_head.weight, std=0.02)
        for block in self.blocks:
            block.mlp.apply(self._init)
            if block.mixer_code == "a":
                block.mixer.apply(self._init)

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)


__all__ = ["MemOnce", "MemOnceConfig"]
