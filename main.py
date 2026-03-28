# Section 1 — Imports & Constants
import math
import os
import subprocess
import sys
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
MIN_FREE_GPU_GIB = 4.0
MAX_NEW_TOKENS = 1          # max_tokens=1 gives cleanest TTFT measurement
TTFT_HIT_THRESHOLD = 0.2   # seconds: below = HIT, above = MISS (L4 cold ~0.4s, hot ~0.06s)


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
        run_num, scenario, ttft_display, cache_display = row[0], row[1], row[2], row[3]
        table.add_row(str(run_num), scenario, ttft_display, cache_display)

    return table


# Section 4 — measure_ttft()
def measure_ttft(llm: LLM, prompt: str) -> float:
    sampling = SamplingParams(max_tokens=MAX_NEW_TOKENS, temperature=0.0)
    t0 = time.perf_counter()
    llm.generate([prompt], sampling_params=sampling)
    return time.perf_counter() - t0


def _run_text_command(args: list[str]) -> str:
    result = subprocess.run(args, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def find_conflicting_gpu_processes() -> list[tuple[int, str]]:
    try:
        raw_pids = _run_text_command([
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ])
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []

    conflicts = []
    for line in raw_pids.splitlines():
        line = line.strip()
        if not line:
            continue

        pid = int(line)
        if pid == os.getpid():
            continue

        try:
            cmdline = open(f"/proc/{pid}/cmdline", "r", encoding="utf-8").read()
        except OSError:
            continue

        rendered = cmdline.replace("\x00", " ").strip()
        if os.path.basename(sys.argv[0]) in rendered:
            conflicts.append((pid, rendered))

    return conflicts


def get_effective_gpu_utilization(console: Console) -> float:
    try:
        raw = _run_text_command([
            "nvidia-smi",
            "--query-gpu=memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ])
    except (FileNotFoundError, subprocess.CalledProcessError):
        return GPU_UTIL

    total_mib, free_mib = (int(part.strip()) for part in raw.splitlines()[0].split(","))
    free_gib = free_mib / 1024
    total_gib = total_mib / 1024

    if free_gib < MIN_FREE_GPU_GIB:
        raise RuntimeError(
            f"Only {free_gib:.2f} GiB of {total_gib:.2f} GiB GPU memory is free. "
            f"Free at least {MIN_FREE_GPU_GIB:.1f} GiB and retry."
        )

    free_fraction = free_mib / total_mib
    effective_util = min(GPU_UTIL, max(0.10, free_fraction - 0.02))
    if effective_util < GPU_UTIL:
        console.print(
            f"[yellow]Reducing gpu_memory_utilization from {GPU_UTIL:.2f} to "
            f"{effective_util:.2f} based on current free GPU memory.[/]"
        )

    return effective_util


# Section 5 — run_experiment()
def run_experiment(llm: LLM, context: str, live: Live, rows_data: list) -> None:
    runs = [
        {
            "run_num": 1,
            "scenario": "Cold Cache (variable at BOTTOM)",
            "narration": (
                "Run 1: Sending the 6,000-token document for the FIRST time.\n"
                "       The GPU has no cached state — it must compute attention for every token."
            ),
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
            "narration": (
                "Run 2: Same 6,000-token prefix — only the User ID and Question changed (at BOTTOM).\n"
                "       vLLM detects the matching prefix and loads KV tensors directly from VRAM."
            ),
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
            "narration": (
                "Run 3: User ID moved to the TOP of the prompt — before the 6,000-token context.\n"
                "       The prefix now differs at token #1. The entire cached KV state is invalid."
            ),
            "get_prompt": lambda ctx: (
                f"User ID: USR-{uuid.uuid4().hex[:8]}\n\n"
                f"[SYSTEM: Financial Policy Assistant]\n"
                f"{ctx}\n\n"
                f"Question: How many days do I have to submit an expense report?"
            ),
        },
    ]

    console = live.console
    for run in runs:
        # Print narration above the live table
        console.print(f"\n[dim]{run['narration']}[/]")

        # Add "Running..." placeholder row
        rows_data.append((
            str(run["run_num"]),
            run["scenario"],
            Text("Running...", style="dim"),
            Text("...", style="dim"),
            None,  # raw ttft float (not yet known)
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

        # Replace placeholder with result (5-tuple: last field = raw float for speedup calc)
        rows_data[-1] = (str(run["run_num"]), run["scenario"], ttft_text, cache_text, ttft)
        live.update(make_table(rows_data))


# Section 6 — main()
def main() -> None:
    console = Console()
    console.print("[bold cyan]Initializing vLLM engine...[/]")
    console.print(f"[dim]Model: {MODEL_ID}[/]")

    conflicts = find_conflicting_gpu_processes()
    if conflicts:
        details = ", ".join(f"PID {pid} ({cmd})" for pid, cmd in conflicts)
        raise RuntimeError(
            "Another GPU process from this demo is already running. "
            f"Stop it before launching a new copy: {details}"
        )

    gpu_util = get_effective_gpu_utilization(console)

    llm = LLM(
        model=MODEL_ID,
        enable_prefix_caching=True,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=gpu_util,
        dtype="auto",             # bfloat16 works natively on L4 (sm_89)
        trust_remote_code=True,
        tensor_parallel_size=1,
    )

    console.print("[bold cyan]Building 6000-token context...[/]")
    context = generate_context(TARGET_CONTEXT_TOKENS)

    enc = tiktoken.get_encoding("cl100k_base")
    actual_tokens = len(enc.encode(context))
    console.print(f"[dim]Context size: {actual_tokens} tokens[/]")

    console.rule("[bold cyan]KV Cache & Prefix Caching Live Demo[/]")
    console.print(
        "\n[bold]What you are about to see:[/]\n"
        "  A [bold]6,000-token[/] financial policy is sent to the model [bold]3 times[/].\n"
        "  The [bold]only[/] difference between runs is [underline]where[/] the User ID is placed.\n\n"
        "  [green]Run 1[/]  Cold cache   — first request, GPU computes everything  (slow)\n"
        "  [green]Run 2[/]  Hot cache    — identical prefix, KV tensors reused      (fast)\n"
        "  [red]Run 3[/]  Cache bust   — User ID moved to TOP, cache invalidated  (slow again)\n"
    )
    console.rule()

    rows_data = []
    with Live(make_table(rows_data), console=console, refresh_per_second=4) as live:
        run_experiment(llm, context, live, rows_data)

    # Final speedup summary
    baseline_ttft = rows_data[0][4]   # Run 1 raw float
    hit_ttft      = rows_data[1][4]   # Run 2 raw float
    bust_ttft     = rows_data[2][4]   # Run 3 raw float

    console.rule()
    speedup = baseline_ttft / hit_ttft
    console.print(
        f"\n[bold green]Cache speedup:[/]  [bold]{speedup:.1f}x faster[/]"
        f"  (Run 2 vs Run 1: {hit_ttft:.3f}s vs {baseline_ttft:.3f}s)"
    )
    console.print(
        f"[bold red]Cache bust:[/]     Run 3 ({bust_ttft:.3f}s) = same latency as Run 1"
        f" despite the document being identical."
    )
    console.print(
        "\n[bold]Golden rule:[/] put [green]static content[/] (system prompt, documents)"
        " at the [green]TOP[/]; [red]dynamic variables[/] (user ID, query) at the [red]BOTTOM[/].\n"
    )


if __name__ == "__main__":
    main()
