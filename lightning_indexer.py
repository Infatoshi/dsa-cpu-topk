"""DeepSeek V3.2 Lightning Indexer: Torch reference + Triton kernel.

Score formula (per DeepSeek-V3.2 paper):
    I[b, t, s] = sum_{j=1..H} w[b, t, j] * ReLU( <q[b, t, j, :], k[b, s, :]> )
for causal positions s <= t; masked to -inf otherwise.

Shapes:
    h:   [B, T, D]   hidden states
    W_Q: [D, H*d]    per-head query projection
    W_K: [D, d]      shared key projection (single head)
    W_W: [D, H]      per-head scalar weights
    -> scores [B, T, T]
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Torch reference (operates on already-projected q, k, w)
# ---------------------------------------------------------------------------

def indexer_scores_torch(q: torch.Tensor, k: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """q: [B,T,H,d]   k: [B,T,d]   w: [B,T,H]   -> scores [B,T,T]."""
    B, T, H, d = q.shape
    dots = torch.einsum("bthd,bsd->bths", q, k)            # [B,T,H,T]
    relu = torch.relu(dots)
    scores = (w.unsqueeze(-1) * relu).sum(dim=2)           # [B,T,T]
    causal = torch.ones(T, T, device=q.device, dtype=torch.bool).tril()
    return scores.masked_fill(~causal, float("-inf"))


def project(h: torch.Tensor, W_Q: torch.Tensor, W_K: torch.Tensor,
            W_W: torch.Tensor, num_heads: int, head_dim: int):
    B, T, _ = h.shape
    q = (h @ W_Q).view(B, T, num_heads, head_dim)
    k = h @ W_K
    w = h @ W_W
    return q, k, w


def indexer_torch(h, W_Q, W_K, W_W, num_heads, head_dim):
    q, k, w = project(h, W_Q, W_K, W_W, num_heads, head_dim)
    return indexer_scores_torch(q, k, w)


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

@triton.jit
def _indexer_score_kernel(
    Q_ptr, K_ptr, W_ptr, S_ptr,
    T, H,
    sQb, sQt, sQh, sQd,
    sKb, sKt, sKd,
    sWb, sWt, sWh,
    sSb, sSt, sSs,
    BLOCK_T: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_s = tl.program_id(2)

    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_d = tl.arange(0, BLOCK_D)

    mask_t = offs_t < T
    mask_s = offs_s < T

    # K block [BS, BD]
    k_ptrs = K_ptr + pid_b * sKb + offs_s[:, None] * sKt + offs_d[None, :] * sKd
    k = tl.load(k_ptrs, mask=mask_s[:, None], other=0.0).to(tl.float32)

    acc = tl.zeros((BLOCK_T, BLOCK_S), dtype=tl.float32)

    for j in range(0, H):
        q_ptrs = Q_ptr + pid_b * sQb + offs_t[:, None] * sQt + j * sQh + offs_d[None, :] * sQd
        q = tl.load(q_ptrs, mask=mask_t[:, None], other=0.0).to(tl.float32)
        dot = tl.dot(q, tl.trans(k), input_precision="ieee")       # [BT, BS]
        relu = tl.maximum(dot, 0.0)
        w_ptrs = W_ptr + pid_b * sWb + offs_t * sWt + j * sWh
        w = tl.load(w_ptrs, mask=mask_t, other=0.0).to(tl.float32)
        acc += relu * w[:, None]

    causal = offs_s[None, :] <= offs_t[:, None]
    valid = causal & mask_t[:, None] & mask_s[None, :]
    acc = tl.where(valid, acc, float("-inf"))

    s_ptrs = S_ptr + pid_b * sSb + offs_t[:, None] * sSt + offs_s[None, :] * sSs
    tl.store(s_ptrs, acc, mask=mask_t[:, None] & mask_s[None, :])


def indexer_scores_triton(q: torch.Tensor, k: torch.Tensor, w: torch.Tensor,
                          BLOCK_T: int = 64, BLOCK_S: int = 64) -> torch.Tensor:
    assert q.is_cuda and k.is_cuda and w.is_cuda
    B, T, H, d = q.shape
    assert k.shape == (B, T, d)
    assert w.shape == (B, T, H)
    # head_dim must be a power of two for tl.dot; pad if needed.
    BLOCK_D = triton.next_power_of_2(d)
    assert d == BLOCK_D, f"head_dim {d} must be power of 2 (got BLOCK_D={BLOCK_D})"

    q = q.contiguous()
    k = k.contiguous()
    w = w.contiguous()
    scores = torch.empty((B, T, T), device=q.device, dtype=torch.float32)

    grid = (B, triton.cdiv(T, BLOCK_T), triton.cdiv(T, BLOCK_S))
    _indexer_score_kernel[grid](
        q, k, w, scores,
        T, H,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2),
        w.stride(0), w.stride(1), w.stride(2),
        scores.stride(0), scores.stride(1), scores.stride(2),
        BLOCK_T=BLOCK_T, BLOCK_S=BLOCK_S, BLOCK_D=BLOCK_D,
    )
    return scores


def indexer_triton(h, W_Q, W_K, W_W, num_heads, head_dim, **kw):
    q, k, w = project(h, W_Q, W_K, W_W, num_heads, head_dim)
    return indexer_scores_triton(q, k, w, **kw)


# ---------------------------------------------------------------------------
# Top-k selection helper (the actual sparse attention mask producer)
# ---------------------------------------------------------------------------

def topk_mask(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Boolean mask of top-k key positions per query (over causal entries)."""
    B, T, _ = scores.shape
    k_eff = min(k, T)
    _, idx = scores.topk(k_eff, dim=-1)
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(-1, idx, True)
    return mask
