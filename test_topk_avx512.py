import os

import torch
from torch.utils.cpp_extension import load


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


def test_avx512_topk_threshold_matches_torch_specialized_and_generic_k():
    torch.manual_seed(0)
    x = torch.randn(3, 128, 128, dtype=torch.float32)
    for k in (1, 8, 16, 32, 64, 96, 128):
        ref = x.topk(k, dim=-1).values[..., -1:]
        out = ext.topk_threshold(x.contiguous(), k)
        assert torch.equal(out, ref), f"k={k}"


def test_avx512_batch_merge_variant_matches_torch():
    torch.manual_seed(1)
    x = torch.randn(3, 128, 128, dtype=torch.float32)
    for k in (64, 128):
        ref = x.topk(k, dim=-1).values[..., -1:]
        out = ext.topk_threshold_batch_merge(x.contiguous(), k)
        assert torch.equal(out, ref), f"k={k}"
