# Section 1 — Imports & Constants
import math
import time
import uuid

import tiktoken
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text
from vllm import LLM, SamplingParams

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
# NOTE: PRD specifies Qwen3.5-1.5B-Instruct but that model ID does not exist on
# HuggingFace. Qwen2.5-1.5B-Instruct is the correct 1.5B GQA model.

TARGET_CONTEXT_TOKENS = 6000
MAX_MODEL_LEN = 8192
GPU_UTIL = 0.85
MAX_NEW_TOKENS = 1          # max_tokens=1 gives cleanest TTFT measurement
TTFT_HIT_THRESHOLD = 0.5   # seconds: below = HIT, above = MISS


# Section 2 — generate_context()
FINANCIAL_PARAGRAPH = (
    "This Financial Policy Document governs all expenditure, reimbursement, and "
    "procurement activities across all business units of the organization. "
    "Employees must submit all expense reports within thirty calendar days of incurring costs. "
    "All purchases exceeding five hundred dollars require prior written approval from a department head. "
    "Travel expenses must be pre-approved by the finance committee at least two weeks in advance. "
    "Receipts are mandatory for all reimbursements exceeding twenty-five dollars. "
)

def generate_context(target_tokens: int) -> str:
    enc = tiktoken.get_encoding("cl100k_base")
    para_tokens = len(enc.encode(FINANCIAL_PARAGRAPH))
    reps = math.ceil(target_tokens / para_tokens)
    context = FINANCIAL_PARAGRAPH * reps
    return context


# Section 3 — make_table()
def make_table(rows_data: list) -> Table:
    table = Table(
        title="KV Cache Prefix Caching Demo",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Run", width=5, justify="center")
    table.add_column("Scenario", width=44, justify="left")
    table.add_column("TTFT (s)", width=12, justify="right")
    table.add_column("Cache Status", width=20, justify="center")

    for row in rows_data:
        run_num, scenario, ttft_display, cache_display = row
        table.add_row(str(run_num), scenario, ttft_display, cache_display)

    return table


# Section 4 — measure_ttft()
def measure_ttft(llm: LLM, prompt: str) -> float:
    sampling = SamplingParams(max_tokens=MAX_NEW_TOKENS, temperature=0.0)
    t0 = time.perf_counter()
    llm.generate([prompt], sampling_params=sampling)
    return time.perf_counter() - t0


# Section 5 — run_experiment()
def run_experiment(llm: LLM, context: str, live: Live, rows_data: list) -> None:
    runs = [
        {
            "run_num": 1,
            "scenario": "Cold Cache (variable at BOTTOM)",
            "get_prompt": lambda ctx: (
                f"[SYSTEM: Financial Policy Assistant]\n"
                f"{ctx}\n\n"
                f"User ID: USR-{uuid.uuid4().hex[:8]}\n"
                f"Question: What is the maximum expense allowed without prior approval?"
            ),
        },
        {
            "run_num": 2,
            "scenario": "Hot Cache (same prefix, variable BOTTOM)",
            "get_prompt": lambda ctx: (
                f"[SYSTEM: Financial Policy Assistant]\n"
                f"{ctx}\n\n"
                f"User ID: USR-{uuid.uuid4().hex[:8]}\n"
                f"Question: Can I submit receipts after 45 days?"
            ),
        },
        {
            "run_num": 3,
            "scenario": "Cache Destroyed (variable at TOP)",
            "get_prompt": lambda ctx: (
                f"User ID: USR-{uuid.uuid4().hex[:8]}\n\n"
                f"[SYSTEM: Financial Policy Assistant]\n"
                f"{ctx}\n\n"
                f"Question: How many days do I have to submit an expense report?"
            ),
        },
    ]

    for run in runs:
        # Add "Running..." placeholder row
        rows_data.append((
            str(run["run_num"]),
            run["scenario"],
            Text("Running...", style="dim"),
            Text("...", style="dim"),
        ))
        live.update(make_table(rows_data))

        prompt = run["get_prompt"](context)
        ttft = measure_ttft(llm, prompt)

        # Build colored result
        is_hit = ttft < TTFT_HIT_THRESHOLD
        ttft_text = Text(f"{ttft:.3f}s", style="bold green" if is_hit else "bold red")
        cache_text = Text(
            "HIT (Reused)" if is_hit else "MISS (Computed)",
            style="green" if is_hit else "red",
        )

        # Replace placeholder with result
        rows_data[-1] = (str(run["run_num"]), run["scenario"], ttft_text, cache_text)
        live.update(make_table(rows_data))


# Section 6 — main()
def main() -> None:
    console = Console()
    console.print("[bold cyan]Initializing vLLM engine...[/]")
    console.print(f"[dim]Model: {MODEL_ID}[/]")

    llm = LLM(
        model=MODEL_ID,
        enable_prefix_caching=True,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_UTIL,
        dtype="float16",          # T4 (sm_75) has no bfloat16 hardware support
        trust_remote_code=True,
        tensor_parallel_size=1,
        enforce_eager=True,       # Disables CUDA graphs — required on T4 (sm_75)
        enable_chunked_prefill=True,  # Avoids context_attention_fwd Triton kernel (broken on T4+Triton3.x)
    )

    console.print("[bold cyan]Building 6000-token context...[/]")
    context = generate_context(TARGET_CONTEXT_TOKENS)

    enc = tiktoken.get_encoding("cl100k_base")
    actual_tokens = len(enc.encode(context))
    console.print(f"[dim]Context size: {actual_tokens} tokens[/]")

    rows_data = []
    with Live(make_table(rows_data), console=console, refresh_per_second=4) as live:
        run_experiment(llm, context, live, rows_data)

    console.print("\n[bold green]Demo complete.[/]")


if __name__ == "__main__":
    main()
