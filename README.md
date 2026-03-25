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

---

## How KV Cache Stores Itself — Physical Storage

### Where Does It Live?

KV Cache lives in **GPU VRAM** (not CPU RAM, not disk):

```
GPU Memory Layout:
┌─────────────────────────────────────────┐
│  Model Weights (loaded once)            │  ← 1.5B model ≈ 3 GB
├─────────────────────────────────────────┤
│  Activations (computed during forward)  │  ← temporary, freed after
├─────────────────────────────────────────┤
│  KV Cache (grows with sequence length)  │  ← THIS IS WHAT WE STORE
├─────────────────────────────────────────┤
│  Free VRAM                              │
└─────────────────────────────────────────┘
```

**Why GPU?** Because the model needs to access K and V vectors **in nanoseconds** during attention computation. Moving them to CPU RAM would be 100-1000× slower (PCIe bottleneck).

---

### Data Format: Tensors (Multi-Dimensional Arrays)

KV Cache is stored as **PyTorch tensors** (or similar: TensorFlow, JAX). Here's the shape:

```
K tensor: [batch_size, num_heads, seq_length, head_dim]
V tensor: [batch_size, num_heads, seq_length, head_dim]

Example for Qwen2.5-1.5B (our demo):
- batch_size = 1 (one request)
- num_heads = 12 (model has 12 attention heads)
- seq_length = 6000 (our demo's context)
- head_dim = 128 (hidden_size=1536 ÷ 12 heads)

K shape: [1, 12, 6000, 128]
V shape: [1, 12, 6000, 128]

Memory per tensor:
  1 × 12 × 6000 × 128 × 2 bytes (float16) = 18.4 MB per tensor
  Total K + V = 36.8 MB for one 6000-token request
```

For a SaaS system with 1,000 concurrent requests:
```
36.8 MB × 1,000 = 36.8 GB of VRAM just for KV cache
```
This is why L4's 24 GB VRAM becomes limiting at scale — and why PagedAttention (vLLM) is so important.

---

### Data Type: Which Precision?

KV Cache can be stored in different precisions:

| Dtype | Bytes per Value | Memory for 6k tokens (1 batch) | Notes |
|-------|-----------------|-------------------------------|-------|
| float32 | 4 bytes | 73.7 MB | Full precision, rarely used (wastes memory) |
| float16 | 2 bytes | 36.8 MB | **Most common** (GPU native, fast) |
| bfloat16 | 2 bytes | 36.8 MB | Better numerical stability than float16 |
| int8 | 1 byte | 18.4 MB | Quantized (lossy), experimental |

**Our demo (L4 GPU):** automatically uses `bfloat16` because L4 has hardware support for it. Older T4 uses `float16`.

---

### How Tokens Are Added to Cache (Step-by-Step)

**Token 1 (first word):**
```
Input: "Financial"
  ↓
GPU computes:
  K_1 = attention_key_projection(embedding_1)      [shape: 12×128]
  V_1 = attention_value_projection(embedding_1)    [shape: 12×128]
  ↓
Store in cache:
  K_cache[0, :, 0, :] = K_1    ← slot 0, all heads
  V_cache[0, :, 0, :] = V_1

Cache state: [1, 12, 1, 128] — only 1 token stored
```

**Token 2 (second word):**
```
Input: "Policy"
  ↓
Load from cache:
  K_prev = K_cache[0, :, 0:1, :]    ← retrieve token 1 (INSTANT)
  V_prev = V_cache[0, :, 0:1, :]
  ↓
Compute new:
  K_2 = attention_key_projection(embedding_2)
  V_2 = attention_value_projection(embedding_2)
  ↓
Append to cache:
  K_cache[0, :, 1, :] = K_2
  V_cache[0, :, 1, :] = V_2

Cache state: [1, 12, 2, 128] — now 2 tokens stored
```

**Token 6000 (last token):**
```
Attention computation looks at ALL previous 5999 tokens:
  scores = Q_6000 @ K_cache[0, :, 0:6000, :].T     ← direct VRAM access
  weights = softmax(scores)
  output = weights @ V_cache[0, :, 0:6000, :]

Notice: K and V from tokens 1-5999 are **already in VRAM**. No recomputation.
```

---

### Physical Layout in GPU Memory: PagedAttention

**Standard approach (wastes memory):**
```
Request 1: [K,V for tokens 1-100] allocated as one huge block
Request 2: [K,V for tokens 1-50]  allocated as one huge block
Request 3: [K,V for tokens 1-150] allocated as one huge block

Fragmented memory:
┌─────────────────────────────────┐
│ Req1 [████████████] 100 tokens  │  ← fully used
├─────────────────────────────────┤
│ Req2 [██████] 50 tokens         │
│     [░░░░░░░░░░░░░░░░░] empty   │  ← wasted (fragmentation)
├─────────────────────────────────┤
│ Req3 [██████████████████] 150   │  ← requires new allocation
├─────────────────────────────────┤
│ Free space (too scattered)      │
└─────────────────────────────────┘
```

**vLLM's PagedAttention (smarter approach):**
```
Fixed 16-token "pages" allocated on-demand:
┌──────┐
│Page1 │ Req1: tokens 1-16
├──────┤
│Page2 │ Req1: tokens 17-32
├──────┤
│Page3 │ Req2: tokens 1-16
├──────┤
│Page4 │ Req3: tokens 1-16
├──────┤
│Page5 │ Req1: tokens 33-48
├──────┤
│Page6 │ Req3: tokens 17-32
├──────┤
│Page7 │ FREE (ready to allocate)
└──────┘

Requests can use non-contiguous pages
→ near-zero fragmentation
→ fits 2-3× more concurrent requests in same VRAM
```

---

### Prefix Caching Storage: Deduplication

When two requests share a prefix, vLLM doesn't duplicate the K,V tensors:

**Without Prefix Caching:**
```
Request A: [System Prompt] + [Doc] + [Q1]
  Stores: K,V for all 6000 tokens
  Memory: 36.8 MB

Request B: [System Prompt] + [Doc] + [Q2]   (same prefix as A)
  Stores: K,V for all 6000 tokens AGAIN
  Memory: 36.8 MB

Total: 73.6 MB (wasteful!)
```

**With Prefix Caching (vLLM's APC):**
```
Request A: [System Prompt] + [Doc] + [Q1]
  Computes & Stores: K,V for all 6000 tokens
  Memory: 36.8 MB

Request B: [same prefix] + [Q2]
  Checks hash of prefix:
    hash("SystemPrompt+Doc") matches Request A's prefix hash
  ↓
  Reuses: K,V cache from Request A (pages 1-375)
  Computes only: K,V for new tokens (Q2)
  Memory: 36.8 MB × 1 + overhead for Request B's tail

Total: ~40 MB (shared prefix not duplicated!)
```

The hash is computed over the token sequence. If even one token differs, the hash changes and cache is not shared.

---

### Memory Lifecycle: When Does Cache Die?

```
Request arrives with 1000 tokens
  ↓
[vLLM allocates pages for K,V cache]
  ↓
Model generates token 1 → add to cache
Model generates token 2 → add to cache
  ... (cache keeps growing)
Model generates token 100 (STOP token)
  ↓
[Request complete]
  ↓
Pages are freed and returned to pool
  ↓
Next request can reuse those pages (no allocation overhead)
```

If the same request makes 5 different API calls with a shared prompt:
```
Call 1: [Shared prompt] + Q1  → cache built, 36.8 MB stored
Call 2: [Shared prompt] + Q2  → reuse cache from Call 1 (instant)
Call 3: [Shared prompt] + Q3  → reuse cache from Call 1 (instant)
Call 4: [Shared prompt] + Q4  → reuse cache from Call 1 (instant)
Call 5: [Shared prompt] + Q5  → reuse cache from Call 1 (instant)

Only computed once. Reused 4 times.
Memory: 36.8 MB + small overhead per call
Compute saved: equivalent to 4 × 1000 token forward passes
```

---

### Practical Example: Our Demo

```
Qwen2.5-1.5B model:
├─ Model weights: ~3 GB
├─ Per-request KV cache (6000 tokens):
│   K: [1, 12, 6000, 128] @ float16 = 18.4 MB
│   V: [1, 12, 6000, 128] @ float16 = 18.4 MB
│   Total: 36.8 MB
│
└─ GPU VRAM utilization:
    Model: 3 GB
    KV for Request 1: 36.8 MB
    KV for Requests 2-3 (prefix reused): ~1 MB each (only tail)
    Activations + temp: ~500 MB
    ─────────────────
    Total: ~4.1 GB / 24 GB (L4) = 17% utilization
```

---

### Key Takeaways for Your Conference

1. **KV Cache = GPU VRAM tensors** — fast, local storage, multi-dimensional arrays
2. **Size grows with sequence length** — longer context = more memory required
3. **Precision matters** — float16/bfloat16 are standard, int8 experimental
4. **PagedAttention fragments smartly** — paging eliminates 60-80% memory waste
5. **Prefix caching deduplicates** — hash-based matching prevents redundant storage
6. **Once computed, reused immediately** — cache access is nanosecond-level, nearly free

The demo shows this: Run 1 computes & stores, Run 2 reuses (8.4× faster), Run 3 breaks the cache by changing token #1 (recompute required).
