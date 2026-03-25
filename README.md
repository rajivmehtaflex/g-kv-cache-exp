# KV Cache & Prefix Caching — Conference Demo

## What Is a KV Cache? (Layman's Explanation)

### The Legal Assistant Analogy

Imagine you hire a legal assistant. Their first task: answer client questions about a 500-page contract.

**Without a KV Cache:**
Every time a client calls, the assistant reads all 500 pages from scratch, then answers. The next client calls — 500 pages again. And again. It is slow, repetitive, and burns through resources for every single request.

**With a KV Cache:**
The assistant reads the document once and writes down key summaries on a notepad — the **cache**. When the next client calls, they only need to read the new question and consult their notes. Instant answer. The 500 pages never get re-read.

---

## What Are "Keys" and "Values"?

LLMs use a mechanism called **Attention** to understand how every word relates to every other word. For each token (word/subword), the model computes two vectors:

| Vector | Question it answers | Analogy |
|--------|--------------------|--------------------|
| **Key (K)** | "What kind of information am I?" | The label on a filing folder |
| **Value (V)** | "What is my actual content?" | The documents inside the folder |

Every time the model generates a new token, it needs to look back at **all previous tokens** via their K and V vectors.

```
WITHOUT KV Cache (grows quadratically):
─────────────────────────────────────────────────────
Token 1  → compute K1, V1
Token 2  → compute K1,V1 + K2,V2          (recomputing K1,V1!)
Token 3  → compute K1,V1 + K2,V2 + K3,V3 (recomputing K1,V1, K2,V2!)
  ...    → O(n²) work — gets slower with every token

WITH KV Cache (grows linearly):
─────────────────────────────────────────────────────
Token 1  → compute K1,V1 → store in GPU cache
Token 2  → load K1,V1 from cache, compute K2,V2 → store K2,V2
Token 3  → load K1,V1,K2,V2 from cache, compute K3,V3 → store K3,V3
  ...    → O(n) work — cache hits are essentially free
```

The KV Cache stores those computed vectors in VRAM so they never need to be recomputed.

---

## What Is Prefix Caching (APC)?

**Prefix Caching** (Automatic Prefix Caching in vLLM) extends the idea one step further:

> If the **beginning** (prefix) of two prompts is identical, reuse the entire KV cache for that prefix across both requests.

**Real-world scenario — a SaaS chatbot:**

```
Every user request looks like this:
┌──────────────────────────────────────────────┐
│  [SYSTEM PROMPT — 2,000 tokens]              │  ← identical for every user
│  You are a helpful financial assistant...    │
│  Company policy: ...                         │
│                                              │
│  User ID: USR-8f3a2b1c         ← changes    │
│  Question: How do I submit...  ← changes    │
└──────────────────────────────────────────────┘
```

- **Without APC:** compute 2,000-token attention for every single user request — even though the system prompt never changes.
- **With APC:** compute the 2,000-token prefix once, cache it in VRAM, and serve the cache to every user. Only the small dynamic tail (User ID + Question) needs fresh computation.

---

## The Three Demo Runs

The live demo (`main.py`) sends a **6,000-token financial policy** to `Qwen2.5-1.5B-Instruct` via vLLM three times. The only thing that changes is **where the User ID is placed**.

| Run | Prompt Structure | Cache Result | Why |
|-----|-----------------|--------------|-----|
| **1 — Cold Cache** | `[System] + [6k context] + [User ID] + [Question]` | **MISS** (~0.4s) | First time — GPU computes all 6k tokens from scratch |
| **2 — Hot Cache** | `[System] + [same 6k context] + [different User ID] + [different Question]` | **HIT** (~0.05s) | Prefix identical to Run 1 — KV tensors loaded directly from VRAM |
| **3 — Cache Destroyed** | `[User ID] + [System] + [same 6k context] + [Question]` | **MISS** (~0.4s) | User ID moved to token #1 — breaks the prefix at byte 1, invalidates the entire cache |

**Key insight from Run 3:** This is the "Enterprise Mistake." Engineering teams routinely embed dynamic identifiers (user IDs, timestamps, request IDs) at the top of their prompts. This single structural decision silently destroys prefix caching and forces full recomputation on every request — even when the document never changes.

---

## Business Impact

| Metric | Without APC | With APC |
|--------|------------|----------|
| TTFT (Time to First Token) | ~0.4s | ~0.05s |
| Speedup | 1× | **~8×** |
| GPU compute for shared prefix | Full recompute per request | **Computed once, reused forever** |
| Cost implication | Pay for 6,000 tokens × N requests | Pay for 6,000 tokens once |

### The Golden Rule of Prompt Engineering for Production

```
GOOD (cache-friendly):          BAD (cache-busting):
┌────────────────────┐          ┌────────────────────┐
│ [System Prompt]    │  STATIC  │ [User ID]          │ ← dynamic at TOP
│ [Document / RAG]   │  ──────  │ [System Prompt]    │    breaks cache!
│ [Few-shot examples]│  FIRST   │ [Document / RAG]   │
├────────────────────┤          ├────────────────────┤
│ [User ID]          │ DYNAMIC  │ [Question]         │
│ [User Question]    │  ──────  └────────────────────┘
└────────────────────┘   LAST
         ↑                               ↑
   Cache survives              Cache destroyed
```

**Rule:** Static content at the TOP, dynamic variables at the BOTTOM.

---

## Running the Demo

```bash
# Prerequisites: L4 GPU (sm_89), Python 3.12, uv
uv sync
python main.py
```

Expected output:
```
 Run  Scenario                                   TTFT (s)    Cache Status
──────────────────────────────────────────────────────────────────────────
  1   Cold Cache (variable at BOTTOM)             0.412s    MISS (Computed)
  2   Hot Cache (same prefix, variable BOTTOM)    0.049s    HIT (Reused)
  3   Cache Destroyed (variable at TOP)           0.401s    MISS (Computed)

Cache speedup: 8.4x faster  (Run 1 vs Run 2)
Cache bust:    Run 3 = same latency as Run 1 despite identical document
```

---

## Technical Stack

| Component | Choice | Reason |
|-----------|--------|--------|
| Inference Engine | vLLM 0.9.2 | Automatic Prefix Caching built-in |
| Model | Qwen2.5-1.5B-Instruct | GQA architecture — efficient KV storage |
| GPU | NVIDIA L4 (24 GB) | sm_89, bfloat16, FlashAttention-2 |
| UI | Python `rich` | Live terminal table |

---

## KV Cache: Universal vs vLLM-Specific

KV Cache is a **fundamental transformer mechanism** — every inference server implements it. vLLM just adds smarter management on top. Here's the full landscape:

### Layer 1 — Basic KV Cache (Everyone Has This)

Every serious inference engine implements the baseline KV cache (storing K,V tensors per token during generation):

| Engine | Basic KV Cache | Notes |
|--------|---------------|-------|
| **vLLM** | Yes | + advanced management on top |
| **TensorRT-LLM** (NVIDIA) | Yes | Highly optimized for NVIDIA hardware |
| **TGI** (HuggingFace) | Yes | Also called "past_key_values" in HF transformers |
| **Ollama** | Yes | Built on llama.cpp underneath |
| **llama.cpp** | Yes | CPU + GPU, "kv_cache" config |
| **DeepSpeed-FastGen** (Microsoft) | Yes | |
| **MLC-LLM** | Yes | |
| **ExLlamaV2** | Yes | |
| Raw HuggingFace `transformers` | Yes | `use_cache=True` (default) |

**Bottom line:** if a system runs a transformer model token-by-token, it has a KV cache. No exceptions.

---

### Layer 2 — Prefix Caching (What Our Demo Shows)

This is where engines **diverge**. Reusing KV cache *across separate requests* requires active cache management:

| Engine | Prefix Caching | Details |
|--------|---------------|---------|
| **vLLM** | Yes (APC) | Automatic Prefix Caching, hash-based block matching |
| **TGI** | Partial | Experimental, less mature |
| **TensorRT-LLM** | Yes | KV Cache Reuse feature |
| **SGLang** | Yes | RadixAttention — most advanced (tree-based, handles branching) |
| **llama.cpp** | Partial | Prompt caching (`--prompt-cache`) — file-based, manual |
| **Ollama** | Yes | Wraps llama.cpp prompt cache automatically |
| **DeepSpeed-FastGen** | No | Uses SplitFuse instead |

---

### Layer 3 — PagedAttention (vLLM's Core Innovation)

vLLM's biggest unique contribution is **PagedAttention** — it manages KV cache memory like an OS manages RAM pages:

```
Traditional engines:           vLLM PagedAttention:
┌──────────────────────┐       ┌──────────────────────┐
│ Request A: [████████]│       │ Request A: [pg1][pg3] │  ← non-contiguous
│ Request B: [██░░░░░░]│       │ Request B: [pg2][pg5] │     pages in VRAM
│ (░ = wasted VRAM)    │       │ (zero internal waste) │
└──────────────────────┘       └──────────────────────┘
```

- Traditional: pre-allocates a contiguous VRAM block per request → massive waste (up to 60-80% fragmentation)
- PagedAttention: allocates fixed-size blocks (pages) on demand → near-zero waste → fits more requests in VRAM

---

### Layer 4 — SGLang's RadixAttention (Most Advanced)

[SGLang](https://github.com/sgl-project/sglang) from the Berkeley Sky Computing Lab goes further than vLLM's flat prefix matching:

```
vLLM APC (linear prefix only):    SGLang RadixAttention (tree):
  [System][DocA][Q1] → cache        [System] ─┬─ [DocA] ─┬─ [Q1]
  [System][DocA][Q2] → cache                  │           └─ [Q2]
  [System][DocB][Q1] → no sharing             └─ [DocB] ─── [Q1]
                                          ↑ shares [System] node across ALL branches
```

RadixAttention can share cache at any branching point in a tree of prompts — useful for multi-turn chat, tree-of-thought, and batch prompt variations.

---

### Summary for Your Conference

```
Basic KV Cache          ← every engine, non-negotiable
    │
    ├── Prefix Caching  ← vLLM, TensorRT-LLM, SGLang, Ollama
    │       └── RadixAttention (SGLang) ← most advanced, tree-based sharing
    │
    └── PagedAttention  ← vLLM's core memory mgmt innovation
                           (TensorRT-LLM has a similar paging system)
```

**One-liner for your audience:**
> "KV Cache is in every LLM runtime. vLLM's contribution is *how* it manages that cache — PagedAttention eliminates memory waste, and Automatic Prefix Caching eliminates redundant compute across requests. Both effects are what our demo measures."
