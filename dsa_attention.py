"""Full DeepSeek Sparse Attention (DSA) forward in three flavors.

Pipeline:
  1. Lightning indexer scores  I[t,s] = sum_j w[t,j] * ReLU(<q_idx[t,j], k_idx[s]>)
  2. Causally mask, take top-k key positions per query (selection)
  3. Standard scaled-dot-product attention restricted to selected positions

Inputs (all on CUDA):
  q_idx : [B, T, HI, DI]   indexer per-head queries
  k_idx : [B, T, DI]       indexer shared key
  w_idx : [B, T, HI]       indexer per-head scalar weights
  Q     : [B, T, HQ, DQ]   main attention queries
  K     : [B, T, HQ, DQ]   main attention keys
  V     : [B, T, HQ, DV]   main attention values
  top_k : int

Output:
  O     : [B, T, HQ, DV]
"""

from __future__ import annotations

import math
import torch
import triton
import triton.language as tl

from lightning_indexer import indexer_scores_torch, indexer_scores_triton


# ---------------------------------------------------------------------------
# 1. PyTorch (vectorized)
# ---------------------------------------------------------------------------

def dsa_attention_torch(q_idx, k_idx, w_idx, Q, K, V, top_k: int) -> torch.Tensor:
    B, T, HQ, DQ = Q.shape
    scale = 1.0 / math.sqrt(DQ)

    scores = indexer_scores_torch(q_idx, k_idx, w_idx)  # [B,T,T] (-inf off-causal)

    k_eff = min(top_k, T)
    # Threshold-based selection (keeps all ties) to match Triton kernel semantics.
    topv, _ = scores.topk(k_eff, dim=-1)                 # [B,T,k_eff]
    threshold = topv[..., -1:]                           # [B,T,1]
    causal = torch.ones(T, T, device=Q.device, dtype=torch.bool).tril()
    sel = (scores >= threshold) & causal                 # [B,T,T]

    logits = torch.einsum("bthd,bshd->bths", Q, K) * scale  # [B,T,HQ,T]
    logits = logits.masked_fill(~sel.unsqueeze(2), float("-inf"))
    attn = torch.softmax(logits, dim=-1)                    # [B,T,HQ,T]
    out = torch.einsum("bths,bshd->bthd", attn, V)          # [B,T,HQ,DV]
    return out


# ---------------------------------------------------------------------------
# 2. Naive (explicit per-(batch, query) Python loop)
# ---------------------------------------------------------------------------

def dsa_attention_naive(q_idx, k_idx, w_idx, Q, K, V, top_k: int) -> torch.Tensor:
    B, T, HQ, DQ = Q.shape
    DV = V.shape[-1]
    scale = 1.0 / math.sqrt(DQ)
    out = torch.zeros(B, T, HQ, DV, device=Q.device, dtype=torch.float32)

    for b in range(B):
        for t in range(T):
            # ---- indexer scores for this query ----
            scores = torch.full((T,), float("-inf"), device=Q.device)
            qi_t = q_idx[b, t]               # [HI, DI]
            wi_t = w_idx[b, t]               # [HI]
            for s in range(t + 1):
                ks = k_idx[b, s]             # [DI]
                dots = (qi_t * ks.unsqueeze(0)).sum(-1)   # [HI]
                scores[s] = (wi_t * torch.relu(dots)).sum()

            # ---- top-k threshold selection (keeps ties) ----
            k_eff = min(top_k, T)
            topv, _ = scores.topk(k_eff)
            thresh = topv[-1]
            sel_idx = torch.nonzero((scores >= thresh) & (torch.arange(T, device=Q.device) <= t),
                                    as_tuple=False).squeeze(-1)
            Ks = K[b, sel_idx]                            # [n_sel, HQ, DQ]
            Vs = V[b, sel_idx]                            # [n_sel, HQ, DV]
            qt = Q[b, t]                                  # [HQ, DQ]
            attn_logits = torch.einsum("hd,khd->kh", qt, Ks) * scale  # [k_eff, HQ]
            attn = torch.softmax(attn_logits, dim=0)
            out[b, t] = torch.einsum("kh,khd->hd", attn, Vs)

    return out


# ---------------------------------------------------------------------------
# 3. Triton fused kernel (indexer + top-k + sparse attention in one launch)
# ---------------------------------------------------------------------------

@triton.jit
def _dsa_fused_kernel(
    QI_ptr, KI_ptr, WI_ptr,
    Q_ptr, K_ptr, V_ptr, O_ptr,
    sQIb, sQIt, sQIh, sQId,
    sKIb, sKIt, sKId,
    sWIb, sWIt, sWIh,
    sQb, sQt, sQh, sQd,
    sKb, sKt, sKh, sKd,
    sVb, sVt, sVh, sVd,
    sOb, sOt, sOh, sOd,
    SCALE,
    T_const: tl.constexpr,
    HI: tl.constexpr, DI: tl.constexpr,
    HQ: tl.constexpr, DQ: tl.constexpr, DV: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    b = tl.program_id(0)
    t = tl.program_id(1)

    offs_di = tl.arange(0, DI)
    offs_dq = tl.arange(0, DQ)
    offs_dv = tl.arange(0, DV)
    offs_hi = tl.arange(0, HI)
    offs_hq = tl.arange(0, HQ)
    offs_T = tl.arange(0, T_const)

    # ---- Load query-side projections ----
    qi = tl.load(QI_ptr + b * sQIb + t * sQIt
                 + offs_hi[:, None] * sQIh + offs_di[None, :] * sQId).to(tl.float32)
    wi = tl.load(WI_ptr + b * sWIb + t * sWIt + offs_hi * sWIh).to(tl.float32)
    q = tl.load(Q_ptr + b * sQb + t * sQt
                + offs_hq[:, None] * sQh + offs_dq[None, :] * sQd).to(tl.float32)

    # ---- Pass 1: indexer scores into [T_const] register tensor ----
    scores = tl.full([T_const], -float("inf"), tl.float32)
    for s_start in tl.static_range(0, T_const, BLOCK_S):
        offs_s = s_start + tl.arange(0, BLOCK_S)
        ki = tl.load(
            KI_ptr + b * sKIb + offs_s[:, None] * sKIt + offs_di[None, :] * sKId,
        ).to(tl.float32)                                # [BS, DI]
        dots = tl.sum(qi[None, :, :] * ki[:, None, :], axis=2)  # [BS, HI]
        relu = tl.maximum(dots, 0.0)
        weighted = tl.sum(relu * wi[None, :], axis=1)            # [BS]
        weighted = tl.where(offs_s <= t, weighted, -float("inf"))
        # scatter weighted (length BS) into scores (length T_const) at offs_s.
        # Use tl.where (not multiply) to avoid 0.0 * -inf = NaN.
        scatter_mask = offs_T[:, None] == offs_s[None, :]                 # [T_const, BS]
        val_at_i = tl.sum(
            tl.where(scatter_mask, weighted[None, :], 0.0), axis=1
        )                                                                 # [T_const]
        is_in_tile = (offs_T >= s_start) & (offs_T < s_start + BLOCK_S)
        scores = tl.where(is_in_tile, val_at_i, scores)

    # ---- Top-k threshold via sort (descending) ----
    sorted_desc = tl.sort(scores, descending=True)
    # Pick element at index TOP_K-1; use tl.where to avoid 0 * -inf = NaN.
    picked = tl.where(offs_T == (TOP_K - 1), sorted_desc, 0.0)
    threshold = tl.sum(picked)

    # ---- Pass 2: load full K/V, compute logits, masked softmax ----
    keep_full = (scores >= threshold) & (offs_T <= t)                 # [T_const]

    # K_full: [T_const, HQ, DQ]
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
    ).to(tl.float32)                                                  # [T_const, HQ, DV]

    # logits[s, h] = sum_d q[h,d] * K_full[s,h,d]
    logits_sh = tl.sum(q[None, :, :] * K_full, axis=2) * SCALE        # [T_const, HQ]
    logits = tl.trans(logits_sh)                                      # [HQ, T_const]
    NEG = -50.0
    logits = tl.where(keep_full[None, :], logits, NEG)

    m = tl.max(logits, axis=1)                                        # [HQ]
    p = tl.exp(logits - m[:, None])                                   # [HQ, T_const]
    p = tl.where(keep_full[None, :], p, 0.0)
    denom = tl.sum(p, axis=1)                                         # [HQ]

    # out[h, d] = sum_s p[h,s] * V_full[s,h,d] / denom[h]
    p_T = tl.trans(p)                                                 # [T_const, HQ]
    weighted_V = p_T[:, :, None] * V_full                             # [T_const, HQ, DV]
    out = tl.sum(weighted_V, axis=0) / denom[:, None]                 # [HQ, DV]
    tl.store(
        O_ptr + b * sOb + t * sOt
        + offs_hq[:, None] * sOh + offs_dv[None, :] * sOd,
        out,
    )


def dsa_attention_triton(q_idx, k_idx, w_idx, Q, K, V, top_k: int,
                         BLOCK_S: int = 32) -> torch.Tensor:
    B, T, HQ, DQ = Q.shape
    DV = V.shape[-1]
    HI, DI = q_idx.shape[2], q_idx.shape[3]

    for n, v in [("T", T), ("HI", HI), ("DI", DI), ("HQ", HQ), ("DQ", DQ),
                 ("DV", DV), ("BLOCK_S", BLOCK_S), ("top_k", top_k)]:
        assert v & (v - 1) == 0 and v > 0, f"{n}={v} must be a power of 2"
    assert T % BLOCK_S == 0, "T must be divisible by BLOCK_S"
    assert top_k <= T

    q_idx, k_idx, w_idx = q_idx.contiguous(), k_idx.contiguous(), w_idx.contiguous()
    Q, K, V = Q.contiguous(), K.contiguous(), V.contiguous()
    out = torch.empty(B, T, HQ, DV, device=Q.device, dtype=torch.float32)
    scale = 1.0 / math.sqrt(DQ)

    grid = (B, T)
    _dsa_fused_kernel[grid](
        q_idx, k_idx, w_idx, Q, K, V, out,
        q_idx.stride(0), q_idx.stride(1), q_idx.stride(2), q_idx.stride(3),
        k_idx.stride(0), k_idx.stride(1), k_idx.stride(2),
        w_idx.stride(0), w_idx.stride(1), w_idx.stride(2),
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        scale,
        T_const=T, HI=HI, DI=DI, HQ=HQ, DQ=DQ, DV=DV,
        TOP_K=top_k, BLOCK_S=BLOCK_S,
    )
    return out


# ---------------------------------------------------------------------------
# 4. Split Triton path (indexer kernel + GPU top-k + sparse attention kernel)
# ---------------------------------------------------------------------------

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

    scores = tl.load(SCORES_ptr + b * sScb + t * sSct + offs_T * sScs).to(tl.float32)
    thr = tl.load(THR_ptr + b * sTb + t * sTt).to(tl.float32)
    keep_full = (scores >= thr) & (offs_T <= t)

    q = tl.load(
        Q_ptr + b * sQb + t * sQt
        + offs_hq[:, None] * sQh + offs_dq[None, :] * sQd,
    ).to(tl.float32)

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


def sparse_attention_with_threshold_triton(scores, threshold, Q, K, V) -> torch.Tensor:
    B, T, _ = scores.shape
    HQ, DQ = Q.shape[-2], Q.shape[-1]
    DV = V.shape[-1]

    for n, v in [("T", T), ("HQ", HQ), ("DQ", DQ), ("DV", DV)]:
        assert v & (v - 1) == 0 and v > 0, f"{n}={v} must be a power of 2"

    scores = scores.contiguous()
    threshold = threshold.contiguous()
    Q, K, V = Q.contiguous(), K.contiguous(), V.contiguous()
    out = torch.empty(B, T, HQ, DV, device=Q.device, dtype=torch.float32)
    scale = 1.0 / math.sqrt(DQ)

    grid = (B, T)
    _sparse_attn_kernel[grid](
        scores, threshold, Q, K, V, out,
        scores.stride(0), scores.stride(1), scores.stride(2),
        threshold.stride(0), threshold.stride(1),
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        scale,
        T_const=T, HQ=HQ, DQ=DQ, DV=DV,
    )
    return out


def dsa_attention_split_triton(q_idx, k_idx, w_idx, Q, K, V, top_k: int,
                               BLOCK_T: int = 32,
                               BLOCK_S: int = 32) -> torch.Tensor:
    B, T, HQ, DQ = Q.shape
    DV = V.shape[-1]
    HI, DI = q_idx.shape[2], q_idx.shape[3]

    for n, v in [("T", T), ("HI", HI), ("DI", DI), ("HQ", HQ), ("DQ", DQ),
                 ("DV", DV), ("BLOCK_T", BLOCK_T), ("BLOCK_S", BLOCK_S),
                 ("top_k", top_k)]:
        assert v & (v - 1) == 0 and v > 0, f"{n}={v} must be a power of 2"
    assert T % BLOCK_T == 0, "T must be divisible by BLOCK_T"
    assert T % BLOCK_S == 0, "T must be divisible by BLOCK_S"
    assert top_k <= T

    scores = indexer_scores_triton(q_idx, k_idx, w_idx, BLOCK_T=BLOCK_T, BLOCK_S=BLOCK_S)
    topv, _ = scores.topk(top_k, dim=-1)
    threshold = topv[..., -1:]
    return sparse_attention_with_threshold_triton(scores, threshold, Q, K, V)
