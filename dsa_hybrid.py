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

import math
import os
import torch
import triton
import triton.language as tl
from torch.utils.cpp_extension import load

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


@triton.jit
def _sparse_attn_kernel(
    SCORES_ptr, THR_ptr,
    Q_ptr, K_ptr, V_ptr, O_ptr,
    sScb, sSct, sScs,
    sTb, sTt,
    sQb, sQt, sQh, sQd,
    sKb, sKt, sKh, sKd,
    sVb, sVt, sVh, sVd,
    sOb, sOt, sOh, sOd,
    SCALE,
    T_const: tl.constexpr,
    HQ: tl.constexpr, DQ: tl.constexpr, DV: tl.constexpr,
):
    b = tl.program_id(0)
    t = tl.program_id(1)

    offs_dq = tl.arange(0, DQ)
    offs_dv = tl.arange(0, DV)
    offs_hq = tl.arange(0, HQ)
    offs_T = tl.arange(0, T_const)

    # Load this query's scores row + threshold scalar
    scores = tl.load(SCORES_ptr + b * sScb + t * sSct + offs_T * sScs).to(tl.float32)
    thr = tl.load(THR_ptr + b * sTb + t * sTt).to(tl.float32)

    keep_full = (scores >= thr) & (offs_T <= t)

    q = tl.load(Q_ptr + b * sQb + t * sQt
                + offs_hq[:, None] * sQh + offs_dq[None, :] * sQd).to(tl.float32)

    K_full = tl.load(
        K_ptr + b * sKb
        + offs_T[:, None, None] * sKt
        + offs_hq[None, :, None] * sKh
        + offs_dq[None, None, :] * sKd,
    ).to(tl.float32)
    V_full = tl.load(
        V_ptr + b * sVb
        + offs_T[:, None, None] * sVt
        + offs_hq[None, :, None] * sVh
        + offs_dv[None, None, :] * sVd,
    ).to(tl.float32)

    logits_sh = tl.sum(q[None, :, :] * K_full, axis=2) * SCALE
    logits = tl.trans(logits_sh)
    NEG = -50.0
    logits = tl.where(keep_full[None, :], logits, NEG)

    m = tl.max(logits, axis=1)
    p = tl.exp(logits - m[:, None])
    p = tl.where(keep_full[None, :], p, 0.0)
    denom = tl.sum(p, axis=1)

    p_T = tl.trans(p)
    weighted_V = p_T[:, :, None] * V_full
    out = tl.sum(weighted_V, axis=0) / denom[:, None]
    tl.store(
        O_ptr + b * sOb + t * sOt
        + offs_hq[:, None] * sOh + offs_dv[None, :] * sOd,
        out,
    )


def _sparse_attention_with_threshold(scores, threshold, Q, K, V):
    B, T, _ = scores.shape
    HQ, DQ = Q.shape[-2], Q.shape[-1]
    DV = V.shape[-1]
    scale = 1.0 / math.sqrt(DQ)
    O = torch.empty(B, T, HQ, DV, device=Q.device, dtype=torch.float32)

    grid = (B, T)
    _sparse_attn_kernel[grid](
        scores, threshold, Q, K, V, O,
        scores.stride(0), scores.stride(1), scores.stride(2),
        threshold.stride(0), threshold.stride(1),
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        scale,
        T_const=T, HQ=HQ, DQ=DQ, DV=DV,
    )
    return O


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
    return _sparse_attention_with_threshold(scores, ctx.thr_gpu, Q, K, V)
