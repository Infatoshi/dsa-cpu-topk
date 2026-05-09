"""End-to-end DSA timing: all-GPU fused vs hybrid (CPU AVX-512 top-K offload)."""

from __future__ import annotations

import argparse
import torch

from dsa_attention import dsa_attention_triton
from dsa_hybrid import dsa_hybrid_cpu_topk, HybridContext


HI, DI, HQ, DQ, DV = 4, 32, 4, 32, 32


def _rand(shape, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(shape, generator=g, device="cuda", dtype=torch.float32) * 0.3


def _make(B, T):
    return (_rand((B, T, HI, DI), 0), _rand((B, T, DI), 1), _rand((B, T, HI), 2),
            _rand((B, T, HQ, DQ), 3), _rand((B, T, HQ, DQ), 4),
            torch.randn((B, T, HQ, DV), device="cuda"))


def time_event(fn, warmup=5, iters=20):
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


def bench(B, T, top_k):
    qi, ki, wi, Q, K, V = _make(B, T)
    ctx = HybridContext(B, T)

    # Verify correctness first
    out_gpu = dsa_attention_triton(qi, ki, wi, Q, K, V, top_k=top_k, BLOCK_S=32)
    out_hyb = dsa_hybrid_cpu_topk(qi, ki, wi, Q, K, V, top_k=top_k, ctx=ctx)
    diff = (out_gpu - out_hyb).abs().max().item()

    t_gpu = time_event(
        lambda: dsa_attention_triton(qi, ki, wi, Q, K, V, top_k=top_k, BLOCK_S=32))
    t_hyb = time_event(
        lambda: dsa_hybrid_cpu_topk(qi, ki, wi, Q, K, V, top_k=top_k, ctx=ctx))
    return t_gpu, t_hyb, diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", nargs="*", default=None,
                    help="Override shapes: 'B,T,k' tuples")
    args = ap.parse_args()

    if args.shapes:
        configs = [tuple(int(x) for x in s.split(",")) for s in args.shapes]
    else:
        configs = []
        for B in (1, 4, 16):
            for T in (64, 128, 256, 512):
                configs.append((B, T, max(8, T // 8)))

    print(f"{'B':>3} {'T':>5} {'k':>4} | {'GPU fused':>10} {'CPU hybrid':>11} "
          f"{'speedup':>8} {'max diff':>10}")
    print("-" * 70)
    for B, T, k in configs:
        try:
            t_gpu, t_hyb, diff = bench(B, T, k)
        except (RuntimeError, AssertionError) as e:
            print(f"{B:>3} {T:>5} {k:>4} | skipped: {type(e).__name__}: {str(e)[:60]}")
            continue
        speedup = t_gpu / t_hyb
        marker = "  <-- hybrid wins" if speedup > 1 else ""
        print(f"{B:>3} {T:>5} {k:>4} | {t_gpu:>10.3f} {t_hyb:>11.3f} "
              f"{speedup:>7.2f}x {diff:>10.2e}{marker}")


if __name__ == "__main__":
    main()
