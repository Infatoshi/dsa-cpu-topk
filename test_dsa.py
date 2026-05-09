import math
import torch
import pytest

from dsa_attention import (
    dsa_attention_torch,
    dsa_attention_naive,
    dsa_attention_triton,
)


def _rand(shape, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(shape, generator=g, device="cuda", dtype=torch.float32)


def _make(B=1, T=32, HI=4, DI=32, HQ=4, DQ=32, DV=32, scale=0.3):
    q_idx = _rand((B, T, HI, DI), 0) * scale
    k_idx = _rand((B, T, DI), 1) * scale
    w_idx = _rand((B, T, HI), 2) * scale
    Q = _rand((B, T, HQ, DQ), 3) * scale
    K = _rand((B, T, HQ, DQ), 4) * scale
    V = _rand((B, T, HQ, DV), 5)
    return q_idx, k_idx, w_idx, Q, K, V


def test_naive_matches_torch():
    q_idx, k_idx, w_idx, Q, K, V = _make(B=1, T=16, HI=2, DI=16, HQ=2, DQ=16, DV=16)
    a = dsa_attention_naive(q_idx, k_idx, w_idx, Q, K, V, top_k=4)
    b = dsa_attention_torch(q_idx, k_idx, w_idx, Q, K, V, top_k=4)
    diff = (a - b).abs().max().item()
    assert diff < 1e-4, f"naive vs torch max diff {diff}"


@pytest.mark.parametrize("T,top_k,BLOCK_S", [(32, 8, 16), (64, 16, 32), (128, 32, 32)])
def test_triton_matches_torch(T, top_k, BLOCK_S):
    q_idx, k_idx, w_idx, Q, K, V = _make(T=T)
    ref = dsa_attention_torch(q_idx, k_idx, w_idx, Q, K, V, top_k=top_k)
    out = dsa_attention_triton(q_idx, k_idx, w_idx, Q, K, V, top_k=top_k, BLOCK_S=BLOCK_S)
    diff = (out - ref).abs().max().item()
    assert diff < 5e-4, f"triton vs torch max diff {diff}"


def test_all_three_agree_small():
    q_idx, k_idx, w_idx, Q, K, V = _make(B=2, T=32, HI=2, DI=16, HQ=2, DQ=16, DV=16)
    a = dsa_attention_naive(q_idx, k_idx, w_idx, Q, K, V, top_k=8)
    b = dsa_attention_torch(q_idx, k_idx, w_idx, Q, K, V, top_k=8)
    c = dsa_attention_triton(q_idx, k_idx, w_idx, Q, K, V, top_k=8, BLOCK_S=16)
    d_ab = (a - b).abs().max().item()
    d_bc = (b - c).abs().max().item()
    assert d_ab < 1e-4, f"naive/torch diff {d_ab}"
    assert d_bc < 5e-4, f"torch/triton diff {d_bc}"
