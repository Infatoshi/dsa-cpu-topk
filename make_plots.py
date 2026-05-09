"""Generate plots for the blog post from fresh benchmark runs."""

from __future__ import annotations

import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.cpp_extension import load

from dsa_attention import dsa_attention_triton
from dsa_hybrid import dsa_hybrid_cpu_topk, HybridContext


OUT_DIR = "/home/infatoshi/work/elliotarledge.com/public/blog/dsa-cpu-topk"
os.makedirs(OUT_DIR, exist_ok=True)

# Reuse the same extension as dsa_hybrid
ext = load(
    name="topk_avx512_ext",
    sources=[os.path.join(os.path.dirname(__file__), "topk_avx512.cpp")],
    extra_cflags=[
        "-O3", "-fopenmp", "-march=native", "-mavx512f", "-mavx512bw",
        "-mavx512dq", "-mavx512vl", "-mavx512vbmi", "-mavx512vbmi2",
        "-funroll-loops", "-ffast-math",
    ],
    extra_ldflags=["-fopenmp"],
    verbose=False,
)


# Site theme: dark background, accent color matching site
BG = "#0a0a0a"
FG = "#ededed"
MUTED = "#a1a1aa"
ACCENT = "#fb923c"      # warm orange
GPU_COLOR = "#60a5fa"   # blue
CPU_COLOR = "#fb923c"   # orange
GRID = "#27272a"


def style_axes(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=MUTED, which="both")
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.title.set_color(FG)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.grid(True, axis="y", color=GRID, linewidth=0.5, alpha=0.7)


def time_cpu(fn, warmup=3, iters=20):
    for _ in range(warmup):
        fn()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t) / iters * 1000


def time_gpu(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


# --------------------------------------------------------------------------
# Plot 1: standalone CPU top-K (torch vs AVX-512)
# --------------------------------------------------------------------------
def plot_standalone_topk():
    configs = [
        (1, 128, 8), (1, 256, 16), (1, 512, 32), (1, 1024, 64),
        (4, 128, 8), (4, 256, 16), (4, 512, 32), (4, 1024, 64),
        (16, 128, 8), (16, 256, 16), (16, 512, 32), (16, 1024, 64),
        (64, 128, 8), (64, 256, 16), (64, 512, 32), (64, 1024, 64),
    ]
    labels, torch_times, avx_times = [], [], []
    for B, T, k in configs:
        x = torch.randn(B, T, T, dtype=torch.float32)
        t_t = time_cpu(lambda: x.topk(k, dim=-1))
        t_a = time_cpu(lambda: ext.topk_threshold(x, k))
        labels.append(f"B={B}\nT={T}\nk={k}")
        torch_times.append(t_t)
        avx_times.append(t_a)
        print(f"({B},{T},{k}): torch={t_t:.3f} avx={t_a:.3f} -> {t_t/t_a:.2f}x")

    speedups = np.array(torch_times) / np.array(avx_times)
    x_pos = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(12, 5), facecolor=BG)
    bars = ax.bar(x_pos, speedups, color=CPU_COLOR, edgecolor=FG, linewidth=0.5)
    for i, (b, s) in enumerate(zip(bars, speedups)):
        ax.text(b.get_x() + b.get_width() / 2, s + 0.3, f"{s:.1f}x",
                ha="center", color=FG, fontsize=9)
    ax.axhline(1.0, color=MUTED, linewidth=0.8, linestyle="--")
    ax.text(len(labels) - 0.5, 1.15, "torch.topk parity", color=MUTED,
            fontsize=8, ha="right")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Speedup vs torch.topk (CPU)")
    ax.set_title("Standalone top-K: hand-written AVX-512 vs torch.topk on CPU",
                 fontsize=13, pad=15)
    style_axes(ax)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "standalone_topk.png")
    fig.savefig(out, dpi=160, facecolor=BG, bbox_inches="tight")
    print(f"saved {out}")


# --------------------------------------------------------------------------
# Plot 2: end-to-end DSA — all-GPU fused vs CPU-offload hybrid
# --------------------------------------------------------------------------
def _make(B, T):
    HI, DI, HQ, DQ, DV = 4, 32, 4, 32, 32

    def r(s, sd):
        g = torch.Generator(device="cuda").manual_seed(sd)
        return torch.randn(s, generator=g, device="cuda", dtype=torch.float32) * 0.3

    return (r((B, T, HI, DI), 0), r((B, T, DI), 1), r((B, T, HI), 2),
            r((B, T, HQ, DQ), 3), r((B, T, HQ, DQ), 4),
            torch.randn((B, T, HQ, DV), device="cuda"))


def plot_hybrid_endtoend():
    configs = [
        (16, 128, 16), (4, 256, 32), (16, 256, 32), (4, 512, 64),
        (16, 512, 32), (16, 512, 64), (16, 512, 128),
        (32, 512, 64), (64, 512, 64),
    ]
    labels, gpu_t, hyb_t = [], [], []
    for B, T, k in configs:
        qi, ki, wi, Q, K, V = _make(B, T)
        ctx = HybridContext(B, T)
        t_g = time_gpu(lambda: dsa_attention_triton(qi, ki, wi, Q, K, V, top_k=k, BLOCK_S=32))
        t_h = time_gpu(lambda: dsa_hybrid_cpu_topk(qi, ki, wi, Q, K, V, top_k=k, ctx=ctx))
        labels.append(f"B={B}\nT={T}\nk={k}")
        gpu_t.append(t_g)
        hyb_t.append(t_h)
        print(f"({B},{T},{k}): gpu={t_g:.3f} hyb={t_h:.3f} -> {t_g/t_h:.2f}x")

    x_pos = np.arange(len(labels))
    w = 0.4
    fig, ax = plt.subplots(figsize=(12, 5.5), facecolor=BG)
    ax.bar(x_pos - w / 2, gpu_t, w, color=GPU_COLOR, label="All-GPU fused (Triton)",
           edgecolor=FG, linewidth=0.5)
    ax.bar(x_pos + w / 2, hyb_t, w, color=CPU_COLOR, label="CPU AVX-512 top-K hybrid",
           edgecolor=FG, linewidth=0.5)
    for i in range(len(labels)):
        s = gpu_t[i] / hyb_t[i]
        ymax = max(gpu_t[i], hyb_t[i])
        color = ACCENT if s > 1 else MUTED
        ax.text(x_pos[i], ymax * 1.05, f"{s:.2f}x",
                ha="center", color=color, fontsize=9, weight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("End-to-end DSA forward (ms)")
    ax.set_title("DSA forward pass: all-GPU fused vs CPU-offload hybrid",
                 fontsize=13, pad=15)
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, framealpha=0.9)
    style_axes(ax)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "hybrid_endtoend.png")
    fig.savefig(out, dpi=160, facecolor=BG, bbox_inches="tight")
    print(f"saved {out}")


# --------------------------------------------------------------------------
# Plot 3: speedup heatmap across (T, k) for fixed B=16 — where the win lives
# --------------------------------------------------------------------------
def plot_crossover_heatmap():
    Ts = [128, 256, 512]
    ks = [8, 16, 32, 64, 128]
    B = 16
    grid = np.full((len(ks), len(Ts)), np.nan)
    for j, T in enumerate(Ts):
        qi, ki, wi, Q, K, V = _make(B, T)
        ctx = HybridContext(B, T)
        for i, k in enumerate(ks):
            if k > T:
                continue
            t_g = time_gpu(lambda: dsa_attention_triton(qi, ki, wi, Q, K, V, top_k=k, BLOCK_S=32))
            t_h = time_gpu(lambda: dsa_hybrid_cpu_topk(qi, ki, wi, Q, K, V, top_k=k, ctx=ctx))
            grid[i, j] = t_g / t_h
            print(f"  T={T} k={k}: gpu={t_g:.3f} hyb={t_h:.3f} ratio={t_g/t_h:.2f}")

    fig, ax = plt.subplots(figsize=(7, 5), facecolor=BG)
    cmap = plt.get_cmap("RdYlGn")
    vmax = max(1.5, np.nanmax(grid))
    vmin = min(0.5, np.nanmin(grid))
    im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect="auto", origin="lower")
    ax.set_xticks(np.arange(len(Ts)))
    ax.set_xticklabels(Ts)
    ax.set_yticks(np.arange(len(ks)))
    ax.set_yticklabels(ks)
    ax.set_xlabel("Sequence length T")
    ax.set_ylabel("top-k")
    ax.set_title(f"Hybrid speedup over all-GPU fused (B={B})\n"
                 f"Green = hybrid wins, red = GPU wins", fontsize=12, pad=12)
    style_axes(ax)
    for i in range(len(ks)):
        for j in range(len(Ts)):
            v = grid[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.2f}x", ha="center", va="center",
                    color="black" if 0.7 < v < 1.3 else "white",
                    fontsize=10, weight="bold")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("speedup", color=FG)
    cbar.ax.yaxis.set_tick_params(color=MUTED)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=MUTED)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "crossover_heatmap.png")
    fig.savefig(out, dpi=160, facecolor=BG, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    print("=== standalone top-K ===")
    plot_standalone_topk()
    print("\n=== hybrid end-to-end ===")
    plot_hybrid_endtoend()
    print("\n=== crossover heatmap ===")
    plot_crossover_heatmap()
