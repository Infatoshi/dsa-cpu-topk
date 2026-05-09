# DSA + AVX-512 CPU top-K offload

A from-scratch implementation of [DeepSeek Sparse Attention](https://arxiv.org/abs/2512.02556)
(Lightning Indexer + top-K + sparse attention) with three forward variants
(naive PyTorch loop, vectorized PyTorch, fused Triton) and a single-machine
experiment: can we beat the all-GPU fused kernel by mid-pipeline offloading
the top-K stage to a hand-written AVX-512 kernel on the CPU?

Short answer: **yes, in a narrow regime.** Best result on a Zen 5 + RTX PRO
6000 Blackwell box: **1.20× end-to-end speedup at B=16, T=512, k=32**
(5.62 ms → 4.68 ms).

Full write-up: <https://elliotarledge.com/blog/dsa-cpu-topk>

## What's in here

| File | Purpose |
|---|---|
| `lightning_indexer.py` | Torch reference + Triton kernel for the indexer score matrix `I[t,s] = Σⱼ wⱼ · ReLU(⟨qⱼ, kₛ⟩)` |
| `dsa_attention.py` | Three end-to-end DSA forwards: `naive` (per-query Python loop), `torch` (vectorized), `triton` (fused single-kernel: indexer → `tl.sort` top-K → sparse attention) |
| `topk_avx512.cpp` | C++ extension: per-row k-th-largest using AVX-512 vector compare + `_mm512_mask_compressstoreu_ps` + scalar buffer insertion, OpenMP-parallel over rows |
| `dsa_hybrid.py` | Hybrid pipeline: GPU indexer → pinned d2h → AVX-512 CPU top-K → pinned h2d → GPU sparse-attention kernel that takes a precomputed threshold (no in-kernel sort) |
| `bench_avx512.py` | Standalone top-K-only bench: AVX-512 vs `torch.topk` on CPU |
| `bench_hybrid.py` | End-to-end bench: all-GPU fused vs CPU-offload hybrid (the headline experiment) |
| `test_indexer.py`, `test_dsa.py` | Numerical equivalence tests across all three implementations |

## Quick start

```bash
uv sync
uv run pytest -q                  # 10 tests, all 3 implementations agree
uv run python bench_avx512.py     # CPU AVX-512 vs torch.topk
uv run python bench_hybrid.py     # all-GPU fused vs CPU-offload hybrid
```

The C++ extension JIT-compiles on first run via `torch.utils.cpp_extension.load`
with `-march=native -mavx512f -mavx512vl -mavx512vbmi2 -fopenmp`.

## Headline results

End-to-end DSA forward pass, RTX PRO 6000 Blackwell + AMD Ryzen 9 9950X3D
(Zen 5, 16c/32t, full AVX-512), `HI=DI=HQ=DQ=DV=4·32`:

```
 B     T    k | GPU fused (ms)  CPU hybrid (ms)  speedup
─────────────────────────────────────────────────────────
16   512   32 |          5.62             4.68    1.20x   ← best
16   512   64 |          5.66             5.47    1.03x
32   512   64 |         11.98            11.72    1.02x
64   512   64 |         25.96            24.97    1.04x
16   512  128 |          5.67             8.39    0.68x   k too big
 4   512   64 |          1.34             1.56    0.86x   B too small
16   128   16 |          0.10             0.20    0.50x   T too small
```

Standalone CPU top-K:

```
 B     T     k | torch (ms)  avx512 (ms)  speedup
─────────────────────────────────────────────────
 1   128     8 |     0.026       0.005     5.5x
 4   128     8 |     0.251       0.007    34.2x
 1   256    16 |     0.189       0.008    22.5x
16  1024    64 |     8.395       6.771     1.2x
64  2048   128 |   150.5       143.9       1.0x
```

PyTorch's CPU `topk` uses scalar `std::__introselect` over `pair<float,long>`
even though the binary advertises AVX-512 capability — that op simply isn't
vectorized in the standard build. A hand-written AVX-512 kernel beats it
1.3–34× across most shapes. See the blog post for the full profile.

## Why the hybrid wins (when it does)

At T ≥ 512 the all-GPU kernel's hot path is the in-kernel `tl.sort` over a
512-element register tensor (bitonic, O(N log² N) shared work). For small k,
the AVX-512 CPU top-K finishes in well under that sort time, and the
PCIe 5.0 d2h of the score matrix amortizes over the full pipeline.
Outside that regime — small T, large k, or small B — PCIe and CPU compute
are pure overhead.

## Caveats

This is an experimental result, not a production play. The honest fix for
the GPU-fused kernel is to replace `tl.sort` with a radix-select or
FAISS-style block-select; that would close the 1.2× gap without any CPU
involvement. The point of the experiment is just: with the right primitive,
mid-pipeline CPU offload can beat a naively-fused GPU kernel on a single
workstation today.

## Hardware

Anvil-LAN: AMD Ryzen 9 9950X3D (Zen 5, 16c/32t, full AVX-512 incl.
vbmi2/vp2intersect/bf16/vnni), 92GB DDR5, NVIDIA RTX PRO 6000 Blackwell
Workstation Edition (sm_120, 96 GB), PCIe 5.0 x16, Ubuntu 24.04, CUDA 13.0,
PyTorch 2.11, Triton 3.6.
