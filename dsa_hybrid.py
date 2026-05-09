"""DSA with CPU-offloaded top-K, mid-pipeline.

Hybrid pipeline:
  1. GPU: compute indexer scores I[B,T,T]   (uses indexer_scores_triton)
  2. d2h (pinned): scores -> CPU
  3. CPU: AVX-512 k-th-largest -> threshold[B,T,1]
  4. h2d (pinned): threshold -> GPU
  5. GPU: sparse attention masked by (scores >= threshold) & causal
     (uses _sparse_attn_kernel — same body as dsa_attention but threshold is
     an input, no in-kernel sort)
"""

from __future__ import annotations

import os
import torch
from torch.utils.cpp_extension import load

from dsa_attention import sparse_attention_with_threshold_triton
from lightning_indexer import indexer_scores_triton

# Build the AVX-512 top-K extension once at import.
_ext = load(
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


class HybridContext:
    """Holds reusable pinned buffers so we don't re-allocate per call."""

    def __init__(self, B, T):
        self.B, self.T = B, T
        self.scores_cpu = torch.empty((B, T, T), dtype=torch.float32, pin_memory=True)
        self.thr_cpu = torch.empty((B, T, 1), dtype=torch.float32, pin_memory=True)
        self.thr_gpu = torch.empty((B, T, 1), dtype=torch.float32, device="cuda")


def dsa_hybrid_cpu_topk(q_idx, k_idx, w_idx, Q, K, V, top_k: int,
                        ctx: HybridContext | None = None) -> torch.Tensor:
    B, T = Q.shape[0], Q.shape[1]
    if ctx is None:
        ctx = HybridContext(B, T)
    assert ctx.B == B and ctx.T == T

    # 1. GPU indexer
    scores = indexer_scores_triton(q_idx, k_idx, w_idx, BLOCK_T=32, BLOCK_S=32)

    # 2. d2h (pinned, blocking — CPU needs the data)
    ctx.scores_cpu.copy_(scores, non_blocking=True)
    torch.cuda.current_stream().synchronize()

    # 3. CPU AVX-512 top-K (writes threshold into ctx.thr_cpu)
    thr = _ext.topk_threshold(ctx.scores_cpu, top_k)         # [B,T,1] cpu
    # 4. h2d
    ctx.thr_gpu.copy_(thr, non_blocking=True)

    # 5. GPU sparse attention using precomputed threshold + scores
    return sparse_attention_with_threshold_triton(scores, ctx.thr_gpu, Q, K, V)
