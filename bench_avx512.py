"""Benchmark custom AVX-512 top-K threshold vs torch.topk on CPU."""

from __future__ import annotations
import os
import time
import torch
from torch.utils.cpp_extension import load

os.environ.setdefault("MAX_JOBS", "16")

# Build the extension with AVX-512 + OpenMP
ext = load(
    name="topk_avx512_ext",
    sources=["topk_avx512.cpp"],
    extra_cflags=[
        "-O3", "-fopenmp", "-march=native", "-mavx512f", "-mavx512bw",
        "-mavx512dq", "-mavx512vl", "-mavx512vbmi", "-mavx512vbmi2",
        "-funroll-loops", "-ffast-math",
    ],
    extra_ldflags=["-fopenmp"],
    verbose=False,
)


def time_fn(fn, warmup=5, iters=30):
    for _ in range(warmup):
        fn()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t) / iters * 1000  # ms


def bench(B, T, k):
    x = torch.randn(B, T, T, dtype=torch.float32)

    def torch_path():
        topv, _ = x.topk(k, dim=-1)
        return topv[..., -1:]

    def avx_path():
        return ext.topk_threshold(x, k)

    # Sanity: thresholds must match exactly (k-th largest is unique up to ties)
    t_ref = torch_path()
    t_avx = avx_path()
    diff = (t_ref - t_avx).abs().max().item()
    assert diff < 1e-5, f"mismatch {diff} at B={B} T={T} k={k}"

    t_t = time_fn(torch_path)
    t_a = time_fn(avx_path)
    return t_t, t_a


def main():
    print(f"OMP threads: {os.environ.get('OMP_NUM_THREADS', 'default')}, "
          f"torch threads: {torch.get_num_threads()}")
    print()
    print(f"{'B':>3} {'T':>5} {'k':>5} | {'torch (ms)':>11} {'avx512 (ms)':>12} "
          f"{'speedup':>8}")
    print("-" * 60)
    rows = []
    for B in (1, 4, 16, 64):
        for T in (128, 256, 512, 1024, 2048):
            k = max(8, T // 16)
            t_t, t_a = bench(B, T, k)
            rows.append((B, T, k, t_t, t_a))
            print(f"{B:>3} {T:>5} {k:>5} | {t_t:>11.3f} {t_a:>12.3f} "
                  f"{t_t / t_a:>7.2f}x")


if __name__ == "__main__":
    main()
