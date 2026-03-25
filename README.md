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
