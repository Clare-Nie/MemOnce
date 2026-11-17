<div align="center">

<img src="docs/assets/icon.png" width="260" height="120"  alt="MemOnce"/>

## A Scalable Memory Architecture for Large Language Models

<p>
  <a href="https://clare-nie.github.io/MemOnce/"><img src="https://img.shields.io/badge/Website-1B42E0?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Website"></a>
  <a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="https://img.shields.io/badge/Paper-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white" alt="Paper"></a>
  <a href="https://huggingface.co/ClareNie/MemOnce"><img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="HuggingFace"></a>
</p>

</div>

---

## 🧬 Overview

**MemOnce** is a foundation architecture designed for long-horizon continual language modeling.

Existing architectures force a painful trade-off: standard Transformers accumulate massive, multi-layer KV caches that easily blow up GPU memory, while linear/recurrent models compress context into fixed-size states that suffer from severe information loss.

MemOnce breaks this bottleneck with a simple yet effective principle: **cache a single compact global memory, and retrieve it only once.**

<p align="center">
  <img src="docs/assets/overall_performance.png" width="92%" alt="MemOnce Teaser"/>
  <br>
  <sub><b>Figure 1.</b> <b>(Left)</b> Memory state cache vs. context length. <b>(Right)</b> Overall memory performance vs. prefill latency and decoding throughput.</sub>
</p>

## ✨ Highlights

- 🧠 **Scalable Architecture:** Caches and retrieves memory **only once**, simultaneously achieving **accurate recall**, **linear-time complexity**, and a **constant active GPU memory budget**.
- 🧱 **Foundation Performance:** Consistently outperforms Transformer, linear GDN, and YOCO across model sizes, benefiting from **depth-time cache recurrence and compression**.
- 🎯 **Native Memory:** The first native memory model to surpass full-context baselines, beating prior methods like M+ and Metis without catastrophic forgetting.
- ⚡ **Inference Scalability:** At 1M context, slashes active GPU cache by **1,313×**, accelerates prefill by **6.9×**, and boosts decoding throughput by **3.5×**.

---

## 📖 Architecture

<p align="center">
  <img src="docs/assets/arch.png" width="60%" alt="MemOnce Architecture"/>
  <br>
  <sub><b>Figure 1.</b> Overview of the MemOnce hybrid backbone, memory encoder-decoder, and memory sparse attention.</sub>
</p>

MemOnce unifies two coupled designs:

- **Memory Encoder-Decoder:** Caches only a trailing `1/r` subset of final-layer hidden states as compact memory and decodes them into layer-specific KV caches on demand, achieving **`L × r` compression** across depth and time.
- **Memory Sparse Attention:** Retrieves top-`K` task-relevant memory states **only once** per input and shares the routing indices across all layers and decoding steps, keeping the active GPU state constant while the persistent bank scales on CPU/SSD.

<div align="center">

| Architecture | State Cache | Prefill Time |
|---|:---:|:---:|
| Transformer | `O(LND)` | `O(LN²D)` |
| YOCO | `O(ND)` | `O(LN²D)` |
| GDN / SWA / M+ | `O(CLD)` | `O(CLND)` |
| **MemOnce (Ours)** | **`O(CD)`** | **`O(CLND)`** |

<sub><code>L</code>: depth · <code>N</code>: context length · <code>D</code>: hidden dim · <code>C</code>: constant active budget (<code>C ≪ N</code>)</sub>

</div>

---

## 🏆 Main Results

### 1. Foundation Capability

<p align="center">
  <img src="docs/assets/training_curves.png" width="85%" alt="Training Curves"/>
  <br>
  <sub><b>Figure 2.</b> Validation perplexity across compression ratios (<code>r = 1, 2, 4, 8</code>) and model scales (0.15B–1.0B).</sub>
</p>

<p align="center">
  <img src="docs/assets/foundation_performance.png" width="85%" alt="Foundation Performance"/>
  <br>
  <sub><b>Figure 3.</b> Language modeling perplexity and 8-task zero-shot reasoning accuracy of foundation architectures.</sub>
</p>

- **Near-Lossless 64× Depth–Time Compression:** Combines layer-depth elimination with temporal pooling (`L × r = 8 × 8 = 64×`) while discarding cross-segment recurrent states, incurring virtually zero downstream accuracy loss (< 0.3%, 52.63% → 52.35%).
- **Superior Foundation Scaling:** Consistently achieves the lowest validation perplexity and highest zero-shot accuracy across 150M to 1B scales. It outperforms Transformer, GDN, SWA, MLA, and YOCO, surpasses a **1.4× larger** Transformer on downstream reasoning, and avoids the catastrophic long-range perplexity spikes seen in pure linear RNNs (PPL 19.28 vs 26.03+).

### 2. Long-Context Memory (4K to 1M Tokens)

<p align="center">
  <img src="docs/assets/memory_performance.png" width="88%" alt="Memory Benchmarks"/>
  <br>
  <sub><b>Figure 4.</b> Evaluation across five long-context memory benchmarks spanning 4K to 1M tokens.</sub>
</p>

- **Surpassing Full Context:** Outperforms full-context Qwen3.5 baselines at both 2B (**+1.84%**) and 4B (**+5.77%**), while surpassing M+ by **+18.68%** and Metis by **+32.27%**.
- **Zero Catastrophic Forgetting:** Query-conditioned sparse retrieval mitigates attention dilution over ultra-long sequences, while frozen-backbone generation with decoded KV caches fully preserves general reasoning capabilities on MMLU, GSM8K, and GPQA.

### 3. Inference Efficiency and Scalability

<p align="center">
  <img src="docs/assets/inference_scalability.png" width="99%" alt="Inference Scalability"/>
  <br>
  <sub><b>Figure 5.</b> Prefill latency, active GPU memory cache, and decoding throughput from 32K to 1M tokens.</sub>
</p>

<div align="center">

| Method (@ 1M Context) | Prefill ↓ (s) | Decode ↑ (tok/s) | Active GPU Cache ↓ | Offloaded Pool |
|---|:---:|:---:|:---:|:---:|
| Qwen3.5-4B (Full Context) | 286.86 | 8.4 | 32.05 GB | — |
| M+ (8B) | 1479.50 | 21.5 | 2.50 GB | 35.90 GB |
| δ-Mem | 1065.88 | 18.0 | 4.5 KB | — |
| EvoEmbedding | 160.33 | 29.1 | 2.6 MB | — |
| Metis | 132.26 | 28.0 | 80.1 MB | — |
| **MemOnce (Ours)** | **41.36** | **29.3** | **25.0 MB** | 640.0 MB |

</div>

- **Linear Prefill & Constant Cache:** Prefill scales linearly from **1.38 s** (32K) to **41.36 s** (1M), while active GPU memory remains bounded at **25.0 MB** with near-invariant decoding throughput (**29.3 tok/s**).
- **Fast Retrieval & On-Demand Decoding:** End-to-end memory retrieval and layer-wise KV decoding take only **64.24 ms** at 128K context — **12.0× faster** than layer-wise retrieval in M+ (`772.50 ms`) and **17.0× faster** than RAG-Rewrite (`1094.30 ms`).

### 4. Zero-Shot Transfer to Representation Tasks

The learned 1024-d memory index descriptors transfer zero-shot to downstream retrieval and context reranking without task-specific fine-tuning:

<div align="center">

| Model | Size | Dim | R@10 ↑ (8 Benchmarks Avg.) | N@10 ↑ (8 Benchmarks Avg.) |
|---|:---:|:---:|:---:|:---:|
| BGE-M3 | 1.2B | 1024 | 65.60 | 50.00 |
| Qwen3-Embedding-4B | 4B | 2560 | 62.30 | 48.10 |
| **MemOnce (Ours)** | **4B** | **1024** | **69.21** | **53.06** |

</div>

---

## 🚀 Quick Start

### 1. Installation

Note that Qwen3.5 post-training requires `transformers >= 5.0`.

```bash
# For post-training adaptation & long-context inference
pip install -r requirements/posttraining.txt

# For foundation pre-training from scratch
pip install -r requirements/pretraining.txt
```

### 2. Download Checkpoint

The released checkpoint (`264 MB`) contains only the ~71M trained parameters (LoRA write adapter, KV readout projections, `mem_to_emb`). Download it alongside the base `Qwen3.5-4B` model:

```bash
pip install -U huggingface_hub
hf download ClareNie/MemOnce --local-dir model
```

### 3. Run Inference

```bash
cd eval/post-training
export BASE=/path/to/Qwen3.5-4B
export CKPT=../../model

# 1. Self-test (verifies all trained tensor keys are strictly consumed)
python inference.py --base-model "$BASE" --checkpoint "$CKPT" --self-test

# 2. Built-in multi-hop demo (20 documents + 2 questions)
python inference.py --base-model "$BASE" --checkpoint "$CKPT" --TEXT

# 3. Batch inference on custom JSONL
python inference.py --base-model "$BASE" --checkpoint "$CKPT" \
    --input my_data.jsonl --output generations.jsonl
```

<details>
<summary><b>📝 Custom JSONL Format & Inference Flags (Click to expand)</b></summary>
<br>

- **Input format:** One JSON object per line containing a `messages` list. All preceding turns are written into memory (pass documents as individual `user` turns); the final `user` turn serves as the query.
- **Segmentation:** Turns exceeding 4,096 tokens are automatically split into 4,096-token segments (producing one 64-token memory block per segment).
- **Useful flags:** `--max-new-tokens`, `--retrieve-blocks`, `--attn-implementation [sdpa|flash_attention_2|eager]`.
- **Checksum verification:**
  ```bash
  cd model && sha256sum model.safetensors config.json
  # f5588a7527bc748f686e26f1a76854ca88bd0061f9733505b8cd23703ebfc927  model.safetensors
  # b2b56eb6680c2af7f4c1a5fbeef79d99a2ed9ea396ce46a6a8b0f745c047c1bf  config.json
  ```
</details>

---

## 🛠️ Training & Evaluation

<details>
<summary><b>📂 Repository Layout</b></summary>

```text
├── docs/                            # Project page & figure assets
├── pre-training/                    # Train MemOnce from scratch
│   ├── memonce.py                   #   Self-contained model (torch + fla)
│   ├── train.py                     #   DDP pre-training entry
│   └── eval.py                      #   Validation perplexity
├── post-training/                   # Adapt a frozen Qwen3.5 backbone
│   ├── model.py                     #   MemOnceConfig + MemOnceModel
│   └── train.py                     #   HF Trainer / DDP entry
├── eval/
│   ├── pre-training/eval_zeroshot.py # 8-task lm-eval harness
│   └── post-training/
│       ├── inference.py             #   Load ckpt → write memory → answer
│       ├── bench.py                 #   Unified benchmark runner
│       └── results/                 #   Released generations + judge logs
└── requirements/                    # Dependency lists
```
</details>

<details>
<summary><b>🔥 Pre-training from Scratch (158M – 945M)</b></summary>
<br>

Reads headerless little-endian `uint16` `.bin` token streams (`vocab_size = 32000`). Supported sizes: `158m`, `397m`, `603m`, `945m`.

```bash
cd pre-training

# Quick smoke test (158M model, 30 steps)
python train.py --size 158m --data train.bin --steps 30 --out ckpt/smoke

# 8-GPU DDP pre-training (945M model, 524K tokens/step)
torchrun --nproc_per_node=8 train.py --size 945m \
    --data train.bin --val-data val.bin \
    --batch-seqs 1 --grad-accum 16 --out ckpt/memonce_945

# Validation perplexity & 8-task zero-shot evaluation
python eval.py --ckpt ckpt/memonce_945/step21748.pt --data val.bin
python ../eval/pre-training/eval_zeroshot.py \
    --ckpt ckpt/memonce_945/step21748.pt --tokenizer /path/to/llama2_tokenizer
```
- **Defaults:** `seq-len = 4096`, `span = 128`, `mem-tokens = 16` (`r = 8`), `block-size = 1024`. Use `--mem-tokens 128` for uncompressed (`r = 1`) and `--mem-budget` to enable top-`K` retrieval. Resume with `--init-from <ckpt> --start-step N`.
</details>

<details>
<summary><b>⚡ Post-training Adaptation (Qwen3.5 Backbone)</b></summary>
<br>

Adapts a frozen Qwen3.5-2B/4B backbone using a directory of `*.json` files (`{"messages": [...]}` ending with an assistant turn), optimizing next-token prediction + InfoNCE contrastive retrieval loss:

```bash
cd post-training
torchrun --standalone --nproc_per_node=8 train.py \
    --model-name-or-path /path/to/Qwen3.5-4B \
    --dataset-path /path/to/stage1_training_dataset \
    --output-dir output/memonce4b
```
</details>

<details>
<summary><b>📊 Long-Context Benchmarks</b></summary>
<br>

Supported benchmarks via `--bench`: `membench`, `locomo`, `longmemeval`, `personamem128`, `personamem1m`, and `HotpotQA`.

```bash
cd eval/post-training
python bench.py --help  # View all benchmark arguments

# Option-letter scoring (no LLM judge needed)
python bench.py --bench membench --data-dir DIR \
    --base-model "$BASE" --checkpoint "$CKPT" --out membench.json

# LLM-as-a-Judge evaluation (required for locomo, longmemeval, HotpotQA)
python bench.py --bench longmemeval --data-dir DIR \
    --base-model "$BASE" --checkpoint "$CKPT" \
    --judge-url http://JUDGE/v1 --judge-model NAME \
    --out longmemeval.json --gens-out longmemeval.jsonl
```
</details>

---

## 📄 Citation

```bibtex
@article{nie2026memonce,
  title   = {MemOnce: A Scalable Memory Architecture for Large Language Models},
  author  = {Nie, Chang and Fu, Chaoyou and Liu, Songlin and Hu, Yiqing and Shan, Caifeng},
  journal = {arXiv preprint},
  year    = {2026}
}
```