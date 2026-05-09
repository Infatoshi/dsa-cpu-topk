"""Larger DSA hybrid scaling sweep with a matplotlib speedup chart.

The chart uses score-matrix elements (B*T*T) as the x-axis because that is the
payload the hybrid path has to compute, transfer to CPU, and threshold.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch

from bench_hybrid import _make
from dsa_attention import dsa_attention_triton
from dsa_hybrid import HybridContext, dsa_hybrid_cpu_topk


OUT_DIR = "/home/infatoshi/work/elliotarledge.com/public/blog/dsa-cpu-topk"
CSV_PATH = os.path.join(OUT_DIR, "scaling_sweep.csv")
PLOT_PATH = os.path.join(OUT_DIR, "scaling_law_speedup.png")

BG = "#0a0a0a"
FG = "#ededed"
MUTED = "#a1a1aa"
GRID = "#27272a"
PARITY = "#71717a"


@dataclass(frozen=True)
class Result:
    B: int
    T: int
    k: int
    gpu_ms: float
    hybrid_ms: float
    max_diff: float

    @property
    def speedup(self) -> float:
        return self.gpu_ms / self.hybrid_ms

    @property
    def score_elements(self) -> int:
        return self.B * self.T * self.T


def _configs() -> list[tuple[int, int, int]]:
    batches = (1, 2, 4, 8, 16, 32, 64)
    lengths = (64, 128, 256, 512)
    k_values = (8, 16, 32, 64, 128)
    return [
        (B, T, k)
        for B in batches
        for T in lengths
        for k in k_values
        if k <= T
    ]


def _style(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=MUTED, which="both")
    ax.title.set_color(FG)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.grid(True, which="major", color=GRID, linewidth=0.6, alpha=0.8)
    ax.grid(True, which="minor", color=GRID, linewidth=0.35, alpha=0.45)
    for spine in ax.spines.values():
        spine.set_color(GRID)


def _write_csv(results: list[Result], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["B", "T", "k", "score_elements", "gpu_ms", "hybrid_ms", "speedup", "max_diff"])
        for r in results:
            w.writerow([
                r.B, r.T, r.k, r.score_elements,
                f"{r.gpu_ms:.6f}", f"{r.hybrid_ms:.6f}",
                f"{r.speedup:.6f}", f"{r.max_diff:.6e}",
            ])


def _read_csv(path: str) -> list[Result]:
    out = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out.append(Result(
                B=int(row["B"]),
                T=int(row["T"]),
                k=int(row["k"]),
                gpu_ms=float(row["gpu_ms"]),
                hybrid_ms=float(row["hybrid_ms"]),
                max_diff=float(row["max_diff"]),
            ))
    return out


def _time_wall(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        fn()
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000


def _bench(B: int, T: int, k: int, warmup: int, iters: int) -> Result:
    qi, ki, wi, Q, K, V = _make(B, T)
    ctx = HybridContext(B, T)

    out_gpu = dsa_attention_triton(qi, ki, wi, Q, K, V, top_k=k, BLOCK_S=32)
    out_hyb = dsa_hybrid_cpu_topk(qi, ki, wi, Q, K, V, top_k=k, ctx=ctx)
    torch.cuda.synchronize()
    max_diff = (out_gpu - out_hyb).abs().max().item()

    gpu_ms = _time_wall(
        lambda: dsa_attention_triton(qi, ki, wi, Q, K, V, top_k=k, BLOCK_S=32),
        warmup,
        iters,
    )
    hybrid_ms = _time_wall(
        lambda: dsa_hybrid_cpu_topk(qi, ki, wi, Q, K, V, top_k=k, ctx=ctx),
        warmup,
        iters,
    )
    return Result(B, T, k, gpu_ms, hybrid_ms, max_diff)


def _plot(results: list[Result], path: str):
    fig, ax = plt.subplots(figsize=(11.5, 6.5), facecolor=BG)
    colors = plt.get_cmap("viridis")(np.linspace(0.12, 0.88, 5))
    k_values = sorted({r.k for r in results})
    markers = {64: "o", 128: "s", 256: "^", 512: "D"}

    for color, k in zip(colors, k_values):
        xs = []
        ys = []
        for r in sorted((x for x in results if x.k == k), key=lambda x: x.score_elements):
            xs.append(r.score_elements)
            ys.append(r.speedup)
            ax.scatter(
                r.score_elements,
                r.speedup,
                color=color,
                marker=markers.get(r.T, "o"),
                s=38 + 6 * math.log2(r.B),
                alpha=0.86,
                edgecolor=BG,
                linewidth=0.4,
            )
        if len(xs) >= 2:
            by_x: dict[int, list[float]] = {}
            for x, y in zip(xs, ys):
                by_x.setdefault(x, []).append(y)
            line_x = np.array(sorted(by_x))
            line_y = np.array([np.median(by_x[x]) for x in line_x])
            ax.plot(line_x, line_y, color=color, linewidth=1.5, alpha=0.75, label=f"k={k}")

    ax.axhline(1.0, color=PARITY, linewidth=1.0, linestyle="--")
    ax.text(max(r.score_elements for r in results), 1.02, "parity",
            color=MUTED, fontsize=9, ha="right", va="bottom")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Score-matrix elements per forward pass (B*T*T, log2)")
    ax.set_ylabel("Speedup: all-GPU fused / CPU-topk hybrid")
    ax.set_title("DSA CPU-topk offload scaling sweep", fontsize=14, pad=14)
    _style(ax)

    topk_handles = [
        plt.Line2D([0], [0], color=color, linewidth=2, label=f"k={k}")
        for color, k in zip(colors, k_values)
    ]
    legend = ax.legend(handles=topk_handles, title="top-k", facecolor=BG,
                       edgecolor=GRID, labelcolor=FG, framealpha=0.92)
    legend.get_title().set_color(FG)

    handles = [
        plt.Line2D([0], [0], marker=m, linestyle="", color=MUTED,
                   markerfacecolor=MUTED, markersize=7, label=f"T={T}")
        for T, m in markers.items()
    ]
    leg2 = ax.legend(handles=handles, title="sequence", loc="lower right",
                     facecolor=BG, edgecolor=GRID, labelcolor=FG, framealpha=0.92)
    leg2.get_title().set_color(FG)
    ax.add_artist(legend)

    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=180, facecolor=BG, bbox_inches="tight")


def run_sweep(limit: int | None, warmup: int, iters: int) -> list[Result]:
    results = []
    configs = _configs()
    if limit is not None:
        configs = configs[:limit]

    print(f"{'B':>3} {'T':>5} {'k':>4} | {'GPU fused':>10} {'CPU hybrid':>11} "
          f"{'speedup':>8} {'max diff':>10}", flush=True)
    print("-" * 70, flush=True)
    for B, T, k in configs:
        try:
            result = _bench(B, T, k, warmup, iters)
        except (RuntimeError, AssertionError) as e:
            print(f"{B:>3} {T:>5} {k:>4} | skipped: {type(e).__name__}: {str(e)[:60]}",
                  flush=True)
            continue
        results.append(result)
        marker = "  <-- hybrid wins" if result.speedup > 1 else ""
        print(f"{B:>3} {T:>5} {k:>4} | {result.gpu_ms:>10.3f} {result.hybrid_ms:>11.3f} "
              f"{result.speedup:>7.2f}x {result.max_diff:>10.2e}{marker}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=CSV_PATH)
    parser.add_argument("--plot", default=PLOT_PATH)
    parser.add_argument("--from-csv", action="store_true",
                        help="Plot an existing CSV instead of running benchmarks.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only run the first N configs, useful for smoke tests.")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    if args.from_csv:
        results = _read_csv(args.csv)
    else:
        results = run_sweep(args.limit, args.warmup, args.iters)
        _write_csv(results, args.csv)
        print(f"saved {args.csv}")

    _plot(results, args.plot)
    print(f"saved {args.plot}")


if __name__ == "__main__":
    main()
