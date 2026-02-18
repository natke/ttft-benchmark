#!/usr/bin/env python3
"""
Generate a TTFT chart from existing benchmark results.
======================================================
Reads ttft_data.json (and optionally system_info.json) from
any results directory and produces a styled comparison chart.

Works with both macOS and Windows results.

Usage:
    python generate_ttft_chart.py                          # auto-detect platform
    python generate_ttft_chart.py --results-dir results/ttft_constit_mac
    python generate_ttft_chart.py --results-dir results/ttft_constit_windows
"""

import argparse
import json
import os
import platform
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STYLE = {
    "Phi-3.5-mini (Foundry Local)":  {"color": "#1f77b4", "marker": "^", "linestyle": "-"},
    "Phi-3.5-mini Q4 (llama.cpp)":  {"color": "#1f77b4", "marker": "o", "linestyle": ":"},
    "Qwen2.5-1.5B (Foundry Local)":  {"color": "#ff7f0e", "marker": "D", "linestyle": "-"},
    "Qwen2.5-1.5B Q4 (llama.cpp)":  {"color": "#ff7f0e", "marker": "s", "linestyle": ":"},
}

# Desired display order
MODEL_ORDER = [
    "Phi-3.5-mini (Foundry Local)",
    "Qwen2.5-1.5B (Foundry Local)",
    "Phi-3.5-mini Q4 (llama.cpp)",
    "Qwen2.5-1.5B Q4 (llama.cpp)",
]


def detect_platform_from_dir(results_dir):
    """Infer platform from the results directory name (e.g. ttft_constit_mac)."""
    dirname = os.path.basename(os.path.normpath(results_dir)).lower()
    if "mac" in dirname or "macos" in dirname or "darwin" in dirname:
        return "macOS"
    elif "windows" in dirname or "win" in dirname:
        return "Windows"
    elif "linux" in dirname:
        return "Linux"
    return None


def detect_platform_label():
    """Return a short platform label for auto-detecting the results dir."""
    if sys.platform == "darwin":
        return "mac"
    elif sys.platform == "win32":
        return "windows"
    else:
        return "linux"


def load_system_info(results_dir):
    """Load system_info.json if it exists, otherwise return None."""
    path = os.path.join(results_dir, "system_info.json")
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return None


def build_specs_text(sysinfo):
    """Build the machine specs text box content from system_info."""
    ded = sysinfo.get("gpu_dedicated_mb")
    shared = sysinfo.get("gpu_shared_mb")

    specs_lines = [
        f"OS:    {sysinfo.get('platform', 'N/A')}",
        f"CPU:   {sysinfo.get('processor', 'N/A')}",
        f"RAM:   {sysinfo.get('ram', 'N/A')}",
        f"GPU:   {sysinfo.get('gpu', 'N/A')}",
    ]
    if ded:
        specs_lines.append(f"VRAM:  {ded} MB ded + {shared} MB shared")
    specs_lines += [
        f"Host:  {sysinfo.get('hostname', 'N/A')}",
        f"Date:  {sysinfo.get('date', 'N/A')[:10]}",
    ]
    return "\n".join(specs_lines)


def generate_chart(results_dir):
    data_file = os.path.join(results_dir, "ttft_data.json")
    if not os.path.isfile(data_file):
        print(f"ERROR: {data_file} not found.")
        sys.exit(1)

    with open(data_file) as f:
        data = json.load(f)

    sysinfo = load_system_info(results_dir)

    # Derive platform name for chart title (priority: system_info > dir name > current OS)
    dir_platform = detect_platform_from_dir(results_dir)
    if sysinfo:
        plat = sysinfo.get("platform", "")
        if "darwin" in plat.lower() or "macos" in plat.lower():
            platform_name = "macOS"
        elif "windows" in plat.lower():
            platform_name = "Windows"
        else:
            platform_name = plat.split("-")[0]
    elif dir_platform:
        platform_name = dir_platform
    else:
        platform_name = detect_platform_label().capitalize()

    fig, ax = plt.subplots(figsize=(10, 6.5))

    # Collect all prompt lengths across all series for x-axis ticks
    all_lengths = set()

    for label in MODEL_ORDER:
        series = data.get(label, {})
        pts = []
        for k, v in series.items():
            all_lengths.add(int(k))
            # skip empty lists and lists containing only "TIMEOUT" strings
            numeric = [x for x in v if isinstance(x, (int, float))]
            if numeric:
                pts.append((int(k), np.mean(numeric)))
        pts.sort()
        if not pts:
            continue

        xs, ys = zip(*pts)
        s = STYLE.get(label, {"color": "#333333", "marker": "x", "linestyle": "-"})
        ax.plot(
            xs, ys,
            marker=s["marker"],
            color=s["color"],
            linestyle=s["linestyle"],
            linewidth=2,
            markersize=7,
            label=label,
        )

        # Annotate last point with the TTFT value
        last_x, last_y = pts[-1]
        ax.annotate(
            f"{last_y:.1f}s",
            (last_x, last_y),
            textcoords="offset points",
            xytext=(10, 0),
            fontsize=8,
            color=s["color"],
            fontweight="bold",
        )

    ax.set_xlabel("Prompt Length (tokens)", fontsize=12)
    ax.set_ylabel("Time to First Token (seconds)", fontsize=12)
    ax.set_title(
        f"TTFT vs Prompt Length -- {platform_name}\n"
        "Available results only; empty data points omitted",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    # Machine specs box (only if system_info exists)
    if sysinfo:
        specs_text = build_specs_text(sysinfo)
        ax.text(
            0.58, 0.62, specs_text,
            transform=ax.transAxes,
            fontsize=7.5,
            fontfamily="monospace",
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0",
                      edgecolor="#cccccc", alpha=0.9),
        )

    # Format x-axis with K labels
    ticks = sorted(all_lengths) if all_lengths else [1000, 5000, 10000, 15000, 20000, 25000, 30000]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t // 1000}K" for t in ticks])

    # Add a note about missing data (timeouts / empty)
    missing_info = []
    for label in MODEL_ORDER:
        series = data.get(label, {})
        empties = sorted(
            int(k) for k, v in series.items()
            if not v or all(isinstance(x, str) for x in v)
        )
        if empties:
            short = label.split(" (")[0]
            missing_info.append(f"{short}: {', '.join(f'{e // 1000}K' for e in empties)}")
    if missing_info:
        note = "Missing (timed out / crashed): " + "  |  ".join(missing_info)
        ax.text(
            0.5, -0.12, note,
            transform=ax.transAxes,
            fontsize=7.5,
            ha="center",
            color="#666666",
            style="italic",
        )

    fig.tight_layout()
    out_name = f"ttft_{platform_name.lower()}_results.png"
    out_path = os.path.join(results_dir, out_name)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved to {out_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate a TTFT chart from existing benchmark results"
    )
    p.add_argument(
        "--results-dir",
        default=None,
        help=(
            "Path to the results directory containing ttft_data.json. "
            "Default: results/ttft_constit_{platform}/ (auto-detected)"
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.results_dir:
        results_dir = args.results_dir
        if not os.path.isabs(results_dir):
            results_dir = os.path.join(BASE_DIR, results_dir)
    else:
        plat = detect_platform_label()
        results_dir = os.path.join(BASE_DIR, "results", f"ttft_constit_{plat}")

    if not os.path.isdir(results_dir):
        print(f"ERROR: Results directory not found: {results_dir}")
        sys.exit(1)

    generate_chart(results_dir)


if __name__ == "__main__":
    main()
