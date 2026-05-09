import torch
import pytest

from lightning_indexer import (
    indexer_scores_torch,
    indexer_scores_triton,
    indexer_torch,
    indexer_triton,
    topk_mask,
)


def _rand(shape, dtype=torch.float32, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(shape, generator=g, device="cuda", dtype=dtype)


@pytest.mark.parametrize("B,T,H,d", [(1, 64, 4, 64), (2, 128, 8, 64), (1, 200, 4, 32)])
def test_scores_match(B, T, H, d):
    q = _rand((B, T, H, d))
    k = _rand((B, T, d), seed=1)
    w = _rand((B, T, H), seed=2)

    ref = indexer_scores_torch(q, k, w)
    out = indexer_scores_triton(q, k, w, BLOCK_T=32, BLOCK_S=32)

    # Compare only causal entries (off-causal are -inf in both)
    causal = torch.ones(T, T, device="cuda", dtype=torch.bool).tril()
    assert torch.allclose(out[:, causal], ref[:, causal], atol=1e-4, rtol=1e-4), \
        f"max diff {(out[:, causal] - ref[:, causal]).abs().max().item()}"
    # Off-causal both -inf
    assert torch.isinf(out[:, ~causal]).all() and (out[:, ~causal] < 0).all()
    assert torch.isinf(ref[:, ~causal]).all()


def test_full_pipeline_match():
    B, T, D, H, d = 2, 96, 128, 4, 64
    h = _rand((B, T, D))
    W_Q = _rand((D, H * d), seed=10) * 0.1
    W_K = _rand((D, d), seed=11) * 0.1
    W_W = _rand((D, H), seed=12) * 0.1

    ref = indexer_torch(h, W_Q, W_K, W_W, H, d)
    out = indexer_triton(h, W_Q, W_K, W_W, H, d, BLOCK_T=32, BLOCK_S=32)

    causal = torch.ones(T, T, device="cuda", dtype=torch.bool).tril()
    diff = (out[:, causal] - ref[:, causal]).abs().max().item()
    assert diff < 1e-3, f"max diff {diff}"


def test_topk_mask_consistency():
    B, T, H, d = 1, 64, 4, 64
    q = _rand((B, T, H, d))
    k = _rand((B, T, d), seed=1)
    w = _rand((B, T, H), seed=2)
    ref = indexer_scores_torch(q, k, w)
    out = indexer_scores_triton(q, k, w, BLOCK_T=32, BLOCK_S=32)
    m_ref = topk_mask(ref, k=8)
    m_out = topk_mask(out, k=8)
    # Allow ties to differ in a small number of slots
    agreement = (m_ref == m_out).float().mean().item()
    assert agreement > 0.99, f"top-k agreement only {agreement}"
