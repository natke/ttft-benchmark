#!/usr/bin/env python3
"""
Unified TTFT Benchmark
=======================
Measures Time to First Token (TTFT) across prompt lengths for both
llama.cpp and Foundry Local backends.

Works on macOS, Windows, and Linux — auto-detects platform-specific
behaviour (process flags, GPU info, timeouts).

Replaces the previous platform-specific scripts:
  - generate_ttft_constit.py  (Mac M4)
  - run_windows_ttft.py       (Windows Intel Iris Xe)

Usage:
    python run_ttft_benchmark.py                         # defaults
    python run_ttft_benchmark.py --output-dir results/my_run
    python run_ttft_benchmark.py --skip-llamacpp         # Foundry only
    python run_ttft_benchmark.py --skip-foundry           # llama.cpp only
    python run_ttft_benchmark.py --prompt-lengths 1000 5000 10000
    python run_ttft_benchmark.py --iterations 5
"""

import argparse
import concurrent.futures
import datetime
import json
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
import traceback
import urllib.request

from openai import OpenAI

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPT_FILE = os.path.join(BASE_DIR, "prompts", "constit.txt")
MODELS_DIR = os.path.join(BASE_DIR, "models")

DEFAULT_PROMPT_LENGTHS = [1000, 2000, 5000, 10000, 15000, 20000, 25000, 30000]
DEFAULT_ITERATIONS = 3
DEFAULT_MAX_TOKENS = 200
CHARS_PER_TOKEN = 4
ITERATION_TIMEOUT = 300  # 5 minutes max per iteration

LLAMACPP_PORT = 8080
LLAMACPP_URL = f"http://localhost:{LLAMACPP_PORT}/v1"

LLAMACPP_MODELS = {
    "Phi-3.5-mini Q4 (llama.cpp)": os.path.join(
        MODELS_DIR, "Phi-3.5-mini-instruct-Q4_K_M.gguf"
    ),
    "Qwen2.5-1.5B Q4 (llama.cpp)": os.path.join(
        MODELS_DIR, "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
    ),
}

FOUNDRY_MODELS = {
    "Phi-3.5-mini (Foundry Local)": "phi-3.5-mini",
    "Qwen2.5-1.5B (Foundry Local)": "qwen2.5-1.5b",
}

SYSTEM_PROMPT = (
    "You are a helpful assistant. The user will provide text from a document. "
    "Summarize the key points concisely."
)

IS_WINDOWS = sys.platform == "win32"

# ---------------------------------------------------------------------------
# Process isolation helpers
# ---------------------------------------------------------------------------

def kill_llamacpp_processes():
    """Kill any running llama-server processes (system-wide)."""
    log("  Killing any remaining llama-server processes...")
    if IS_WINDOWS:
        subprocess.run(
            ["taskkill", "/F", "/T", "/IM", "llama-server.exe"],
            capture_output=True, timeout=30,
        )
    else:
        subprocess.run(["pkill", "-9", "-f", "llama-server"], capture_output=True, timeout=10)
    time.sleep(2)


def _kill_by_port(port):
    """Kill whatever process is listening on *port* (Windows only)."""
    if not IS_WINDOWS:
        return
    try:
        r = subprocess.run(
            ["powershell", "-Command",
             f"(Get-NetTCPConnection -LocalPort {port} -ErrorAction SilentlyContinue).OwningProcess"],
            capture_output=True, text=True, timeout=10,
        )
        for line in r.stdout.strip().splitlines():
            pid = line.strip()
            if pid and pid.isdigit() and int(pid) > 0:
                log(f"  Killing PID {pid} on port {port}")
                subprocess.run(["taskkill", "/F", "/T", "/PID", pid],
                               capture_output=True, timeout=30)
    except Exception:
        pass


def kill_foundry_processes():
    """Kill any running Foundry Local / neutron-server processes."""
    log("  Killing any remaining Foundry Local processes...")
    names = (
        ("foundry.exe", "foundry-local.exe", "neutron-server.exe",
         "Inference.Service.Agent.exe")
        if IS_WINDOWS else
        ("foundry", "foundry-local", "neutron-server",
         "Inference.Service.Agent")
    )
    for name in names:
        if IS_WINDOWS:
            subprocess.run(
                ["taskkill", "/F", "/T", "/IM", name],
                capture_output=True, timeout=30,
            )
        else:
            subprocess.run(["pkill", "-9", "-f", name], capture_output=True, timeout=10)
    # Fallback: kill whatever is on port 5272
    _kill_by_port(5272)
    time.sleep(3)


def ensure_port_free(port, label="service"):
    """Wait up to 15 s for a TCP port to become free."""
    for _ in range(15):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return
        log(f"  Port {port} still in use by {label}, waiting...")
        time.sleep(1)
    log(f"  WARNING: Port {port} still occupied after 15 s")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_log_fh = None


def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    if _log_fh:
        _log_fh.write(line + "\n")
        _log_fh.flush()


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def load_document():
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        text = f.read()
    return text.replace("\\n", "\n")


def build_prompt(document, target_tokens):
    instruction = "Please summarize the following document:\n\n"
    available = target_tokens * CHARS_PER_TOKEN - len(instruction)
    if available <= 0:
        return instruction + document[:100]
    text = document
    while len(text) < available:
        text += "\n\n--- (document continues) ---\n\n" + document
    return instruction + text[:available]


# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------

def make_client(base_url, api_key="dummy"):
    """Create an OpenAI client with a generous timeout for slow hardware."""
    return OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=7200.0,
    )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def measure_with_response(client, model_id, prompt, max_tokens, temperature=0.7):
    start = time.time()
    first_token_time = None
    tokens = 0
    response_text = []

    stream = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
    )

    for chunk in stream:
        if first_token_time is None:
            first_token_time = time.time()
        delta = chunk.choices[0].delta.content
        if delta:
            tokens += 1
            response_text.append(delta)

    end = time.time()
    ttft = (first_token_time - start) if first_token_time else 0
    total = end - start
    tps = tokens / total if total > 0 else 0

    return {
        "time_to_first_token": ttft,
        "total_time": total,
        "tokens_generated": tokens,
        "tokens_per_second": tps,
        "response_text": "".join(response_text),
    }


# ---------------------------------------------------------------------------
# llama.cpp helpers (cross-platform)
# ---------------------------------------------------------------------------

_llamacpp_proc = None


def start_llamacpp(model_path, ctx_size=32768):
    global _llamacpp_proc
    stop_llamacpp()
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"GGUF not found: {model_path}")

    cmd = [
        "llama-server",
        "--host", "0.0.0.0",
        "--port", str(LLAMACPP_PORT),
        "--model", model_path,
        "-ngl", "99",
        "-c", str(ctx_size),
        "--no-cache-prompt",
    ]
    log(f"CMD: {' '.join(cmd)}")

    popen_kwargs = dict(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if IS_WINDOWS:
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    _llamacpp_proc = subprocess.Popen(cmd, **popen_kwargs)

    for _ in range(120):
        time.sleep(1)
        try:
            urllib.request.urlopen(f"http://localhost:{LLAMACPP_PORT}/health")
            log("llama-server ready.")
            return
        except Exception:
            pass
    raise RuntimeError("llama-server did not start within 120 s")


def stop_llamacpp():
    global _llamacpp_proc
    if _llamacpp_proc is not None:
        log("Stopping llama-server ...")
        pid = _llamacpp_proc.pid
        if IS_WINDOWS:
            # taskkill /F /T is the only reliable way on Windows
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True, timeout=30,
            )
        else:
            _llamacpp_proc.kill()
        try:
            _llamacpp_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        _llamacpp_proc = None


# ---------------------------------------------------------------------------
# Run tests for one model
# ---------------------------------------------------------------------------

def _run_measure_with_timeout(client, model_id, prompt, max_tokens, timeout_s):
    """Run measure_with_response in a thread; return result or None on timeout."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(measure_with_response, client, model_id, prompt, max_tokens)
        try:
            return future.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError:
            return None


def run_model_tests(client, model_id, label, document, prompt_lengths, num_iterations, max_tokens):
    series = {}

    for pl in prompt_lengths:
        prompt = build_prompt(document, pl)
        log(f"  Prompt length target: ~{pl} tokens ({len(prompt)} chars)")

        ttft_values = []
        timed_out = False
        for i in range(num_iterations):
            log(f"    Iteration {i + 1}/{num_iterations}...")
            try:
                result = _run_measure_with_timeout(
                    client, model_id, prompt, max_tokens, ITERATION_TIMEOUT
                )
                if result is None:
                    log(f"      TIMED OUT after {ITERATION_TIMEOUT}s -- skipping remaining iterations")
                    timed_out = True
                    break

                ttft = result["time_to_first_token"]
                tps = result["tokens_per_second"]
                ttft_values.append(ttft)

                log(
                    f"      TTFT: {ttft:.3f}s | Throughput: {tps:.1f} t/s | "
                    f"Tokens: {result['tokens_generated']}"
                )
                if i == 0:
                    log(f"      --- Response ({label}, ~{pl} tokens) ---")
                    for resp_line in result["response_text"].splitlines():
                        log(f"      | {resp_line}")
                    log(f"      --- End response ---")
            except Exception as e:
                log(f"      ERROR: {e}")
                continue

        if timed_out:
            log(f"    TIMEOUT: prompt length {pl} exceeded {ITERATION_TIMEOUT}s -- skipping longer prompts too")
            series[str(pl)] = ["TIMEOUT"]
            # Skip all remaining (longer) prompt lengths for this model
            for remaining_pl in prompt_lengths[prompt_lengths.index(pl) + 1:]:
                series[str(remaining_pl)] = ["TIMEOUT"]
                log(f"  Skipping ~{remaining_pl} tokens (would be even slower)")
            break

        if ttft_values:
            avg = statistics.mean(ttft_values)
            log(
                f"    Summary: TTFT avg={avg:.3f}s, "
                f"min={min(ttft_values):.3f}s, max={max(ttft_values):.3f}s"
            )
        series[str(pl)] = ttft_values

    return series


# ---------------------------------------------------------------------------
# System info
# ---------------------------------------------------------------------------

def detect_gpu():
    """Best-effort GPU detection."""
    if IS_WINDOWS:
        try:
            r = subprocess.run(
                ["wmic", "path", "Win32_VideoController", "get", "Name", "/format:list"],
                capture_output=True, text=True, timeout=10,
            )
            for line in r.stdout.splitlines():
                if line.startswith("Name="):
                    return line.split("=", 1)[1].strip()
        except Exception:
            pass
    else:
        # macOS — check for Apple Silicon
        try:
            r = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5,
            )
            cpu = r.stdout.strip()
            if "Apple" in cpu:
                return f"{cpu} (Metal)"
        except Exception:
            pass
        # Linux — try nvidia-smi
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode == 0:
                return r.stdout.strip().split("\n")[0]
        except Exception:
            pass
    return "Unknown"


def get_ram():
    """Best-effort RAM detection."""
    if IS_WINDOWS:
        try:
            r = subprocess.run(
                ["wmic", "memorychip", "get", "Capacity", "/format:list"],
                capture_output=True, text=True, timeout=10,
            )
            total = 0
            for line in r.stdout.splitlines():
                if line.startswith("Capacity="):
                    total += int(line.split("=")[1])
            if total > 0:
                return f"{total // (1024 ** 3)} GB"
        except Exception:
            pass
    else:
        try:
            r = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0:
                return f"{int(r.stdout.strip()) // (1024 ** 3)} GB"
        except Exception:
            pass
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return f"{kb // (1024 ** 2)} GB"
        except Exception:
            pass
    return "N/A"


def collect_system_info(prompt_lengths, num_iterations, max_tokens):
    gpu = detect_gpu()
    ram = get_ram()
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "hostname": socket.gethostname(),
        "python": sys.version,
        "gpu": gpu,
        "ram": ram,
        "date": datetime.datetime.now().isoformat(),
        "prompt_lengths": prompt_lengths,
        "num_iterations": num_iterations,
        "max_tokens": max_tokens,
    }


# ---------------------------------------------------------------------------
# Chart generation
# ---------------------------------------------------------------------------

def generate_chart(all_series, chart_path, title_suffix=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    markers = ["o", "s", "^", "D"]

    for idx, (label, series) in enumerate(all_series.items()):
        lengths = sorted(int(k) for k in series.keys())
        valid = []
        for l in lengths:
            numeric = [v for v in series[str(l)] if isinstance(v, (int, float))]
            if numeric:
                valid.append((l, np.mean(numeric)))
        if valid:
            ax.plot(
                [v[0] for v in valid],
                [v[1] for v in valid],
                marker=markers[idx % 4],
                color=colors[idx % 4],
                linewidth=2,
                markersize=6,
                label=label,
            )

    ax.set_xlabel("Prompt Length (tokens)", fontsize=12)
    ax.set_ylabel("Time to First Token (s)", fontsize=12)
    title = "TTFT vs Prompt Length"
    if title_suffix:
        title += f" — {title_suffix}"
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    log(f"Chart saved to {chart_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Unified TTFT benchmark for llama.cpp and Foundry Local"
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Results directory (default: auto-detected from platform)",
    )
    p.add_argument(
        "--prompt-lengths",
        type=int,
        nargs="+",
        default=DEFAULT_PROMPT_LENGTHS,
        help=f"Prompt lengths to test (default: {DEFAULT_PROMPT_LENGTHS})",
    )
    p.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help=f"Iterations per prompt length (default: {DEFAULT_ITERATIONS})",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Max output tokens (default: {DEFAULT_MAX_TOKENS})",
    )
    p.add_argument(
        "--skip-llamacpp",
        action="store_true",
        help="Skip llama.cpp tests",
    )
    p.add_argument(
        "--skip-foundry",
        action="store_true",
        help="Skip Foundry Local tests",
    )
    p.add_argument(
        "--ctx-size",
        type=int,
        default=32768,
        help="llama.cpp context size (default: 32768)",
    )
    p.add_argument(
        "--merge-data",
        default=None,
        help="Path to existing ttft_data.json to merge with new results",
    )
    p.add_argument(
        "--skip-model",
        action="append",
        default=[],
        help="Skip a specific model label (can be repeated)",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=ITERATION_TIMEOUT,
        help=f"Per-iteration timeout in seconds (default: {ITERATION_TIMEOUT})",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global _log_fh
    args = parse_args()

    # Auto-detect output directory
    if args.output_dir:
        results_dir = args.output_dir
        if not os.path.isabs(results_dir):
            results_dir = os.path.join(BASE_DIR, results_dir)
    else:
        suffix = "windows" if IS_WINDOWS else platform.system().lower()
        results_dir = os.path.join(BASE_DIR, "results", f"ttft_constit_{suffix}")

    os.makedirs(results_dir, exist_ok=True)
    log_file = os.path.join(results_dir, "test_run.log")
    _log_fh = open(log_file, "a", encoding="utf-8")  # append mode

    # Apply timeout from CLI
    global ITERATION_TIMEOUT
    ITERATION_TIMEOUT = args.timeout

    log("=" * 70)
    log("TTFT Constitution Document Benchmark")
    log("=" * 70)
    log(f"Platform : {platform.platform()}")
    log(f"Machine  : {platform.machine()}")
    log(f"Processor: {platform.processor()}")
    log(f"Hostname : {socket.gethostname()}")
    log(f"Python   : {sys.version}")
    log(f"Output   : {results_dir}")
    log(f"Prompt lengths: {args.prompt_lengths}")
    log(f"Iterations    : {args.iterations}")
    log(f"Max tokens    : {args.max_tokens}")
    log(f"Timeout/iter  : {ITERATION_TIMEOUT}s")
    log(f"Skip llama.cpp: {args.skip_llamacpp}")
    log(f"Skip Foundry  : {args.skip_foundry}")
    log(f"Skip models   : {args.skip_model}")
    log(f"Merge data    : {args.merge_data}")
    log("")

    document = load_document()
    log(f"Document loaded: {len(document)} chars (~{len(document) // CHARS_PER_TOKEN} tokens)")
    log("")

    # Load existing data to merge with
    all_series = {}
    if args.merge_data:
        try:
            with open(args.merge_data, "r") as f:
                all_series = json.load(f)
            log(f"Loaded existing data: {list(all_series.keys())}")
        except Exception as e:
            log(f"WARNING: Could not load merge data: {e}")

    # --- llama.cpp tests ---
    if not args.skip_llamacpp:
        log("Ensuring Foundry Local is stopped before llama.cpp tests...")
        kill_foundry_processes()
        ensure_port_free(5272, "Foundry Local")
        ensure_port_free(LLAMACPP_PORT, "llama-server")
        log("")

        for label, model_path in LLAMACPP_MODELS.items():
            if label in args.skip_model:
                log(f"Skipping {label} (--skip-model)")
                continue
            log("")
            log("=" * 70)
            log(f"  {label}")
            log("=" * 70)
            try:
                start_llamacpp(model_path, ctx_size=args.ctx_size)
                client = make_client(base_url=LLAMACPP_URL, api_key="none")
                all_series[label] = run_model_tests(
                    client, label, label, document,
                    args.prompt_lengths, args.iterations, args.max_tokens,
                )
            except Exception as e:
                log(f"  ERROR running {label}: {e}")
                log(traceback.format_exc())
            finally:
                stop_llamacpp()

    # --- Foundry Local tests ---
    if not args.skip_foundry:
        log("Ensuring llama-server is stopped before Foundry Local tests...")
        stop_llamacpp()
        kill_llamacpp_processes()
        ensure_port_free(LLAMACPP_PORT, "llama-server")
        log("")

        try:
            from foundry_local import FoundryLocalManager

            log("")
            log("=" * 70)
            log("  Starting Foundry Local service")
            log("=" * 70)
            manager = FoundryLocalManager()
            manager.start_service()
            log(f"Foundry Local endpoint: {manager.endpoint}")

            for label, alias in FOUNDRY_MODELS.items():
                if label in args.skip_model:
                    log(f"Skipping {label} (--skip-model)")
                    continue
                log("")
                log("=" * 70)
                log(f"  {label}")
                log("=" * 70)
                try:
                    log(f"Downloading model '{alias}'...")
                    manager.download_model(alias)
                    log(f"Loading model '{alias}'...")
                    info = manager.load_model(alias)
                    log(f"Model loaded as: {info.id}")
                    client = make_client(
                        base_url=manager.endpoint, api_key=manager.api_key
                    )
                    all_series[label] = run_model_tests(
                        client, info.id, label, document,
                        args.prompt_lengths, args.iterations, args.max_tokens,
                    )
                except Exception as e:
                    log(f"  ERROR running {label}: {e}")
                    log(traceback.format_exc())
        except ImportError:
            log("WARNING: foundry-local package not installed -- skipping Foundry Local tests")
        except Exception as e:
            log(f"WARNING: Skipping Foundry Local tests: {e}")
            log(traceback.format_exc())

    # --- Save raw data ---
    raw_path = os.path.join(results_dir, "ttft_data.json")
    with open(raw_path, "w") as f:
        json.dump(all_series, f, indent=2)
    log(f"Raw data saved to {raw_path}")

    # --- Save system metadata ---
    sysinfo = collect_system_info(args.prompt_lengths, args.iterations, args.max_tokens)
    meta_path = os.path.join(results_dir, "system_info.json")
    with open(meta_path, "w") as f:
        json.dump(sysinfo, f, indent=2)
    log(f"System metadata saved to {meta_path}")

    log("")
    log("Test run complete.")
    log("To generate a chart, run: python generate_ttft_chart.py")
    _log_fh.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Interrupted.")
    finally:
        stop_llamacpp()
