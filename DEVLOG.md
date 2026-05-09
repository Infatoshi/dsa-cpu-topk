# DEVLOG

Handoff notes for the next agent (Codex). Self-contained — read this first,
then `README.md` for the polished writeup.

## Context

Single workstation: AMD Ryzen 9 9950X3D (Zen 5, 16c/32t, full AVX-512
incl. vbmi2/vp2intersect/bf16/vnni), 92 GB DDR5, NVIDIA RTX PRO 6000
Blackwell Workstation Edition (sm_120, 96 GB), PCIe 5.0 x16, Ubuntu 24.04,
CUDA 13.0, PyTorch 2.11+cu130, Triton 3.6.

Use `uv run` for everything. Repo root: `/home/infatoshi/cuda/DSA`.
GitHub: <https://github.com/Infatoshi/dsa-cpu-topk>.
Blog post: `/home/infatoshi/work/elliotarledge.com/app/blog/dsa-cpu-topk/page.tsx`
(public at <https://elliotarledge.com/blog/dsa-cpu-topk> after Vercel
rebuild).

## What got built (chronological)

1. **Lightning Indexer** (`lightning_indexer.py`). Per the DeepSeek-V3.2
   paper:
   `I[b,t,s] = sum_j w[b,t,j] * ReLU(<q[b,t,j,:], k[b,s,:]>)` with
   causal mask. Two impls — fp32 einsum reference and a Triton kernel
   (3D grid B × T_q × T_k, loop over heads, IEEE fp32 dot).

2. **Full DSA forward** (`dsa_attention.py`). Three implementations,
   identical numerics:
   - `dsa_attention_naive`: per-(batch, query) Python loop
   - `dsa_attention_torch`: fully vectorized
   - `dsa_attention_triton`: single fused kernel doing indexer →
     `tl.sort` top-K → masked softmax + weighted-V. One program per
     query (grid `(B, T)`), `T_const` worth of state in registers, so
     practical T <= ~512 before PTXAS register pressure kills it.

   Selection policy is **threshold-keeps-ties** (`scores >= topk[k-1]`),
   not `topk` indices, because that's what the Triton kernel naturally
   produces. All three impls were aligned on this policy after a
   tie-handling debug fight.

3. **CPU offload investigation**. Sub-agent confirmed Zen 5 has full
   AVX-512. Initial `torch.topk` on CPU vs GPU: GPU wins 20–130× across
   every shape. `perf record` on the hot loop showed **78% in
   `std::__introselect<pair<float,long>*, ..., AVX2::topk_impl_loop>`**
   — scalar introselect, not vectorized despite the namespace name. 13%
   in `gomp_team_barrier_wait_end` (OMP barrier overhead, granularity
   too small). PyTorch CPU `topk` is not using AVX-512.

4. **Hand-written AVX-512 top-K** (`topk_avx512.cpp`). Per row maintain
   a descending-sorted buffer of size k. Scan 16 floats at a time:
   `_mm512_cmp_ps_mask` against `buf[k-1]`, skip chunk if mask zero,
   else `_mm512_mask_compressstoreu_ps` and scalar-insert into buffer.
   OpenMP-parallel over rows. JIT-built via
   `torch.utils.cpp_extension.load` with
   `-march=native -mavx512f -mavx512vbmi2 -fopenmp`. Beats `torch.topk`
   CPU 1.3× to 34× across 17 of 20 shapes; ties at large k where O(k)
   insertion catches up.

5. **Hybrid pipeline** (`dsa_hybrid.py`). GPU indexer →
   pinned d2h scores → CPU AVX-512 k-th-largest → pinned h2d threshold
   → GPU sparse-attention kernel that takes the precomputed threshold
   (a stripped variant of `_dsa_fused_kernel` with no in-kernel sort).
   `HybridContext` holds reusable pinned buffers.

6. **End-to-end bench** (`bench_hybrid.py`). Headline: **1.24× at
   B=16, T=512, k=32** (5.68 ms → 4.59 ms). Crossover regime is
   **large T (~512), small k (≤32), large B (≥16)**. Outside this
   window the all-GPU fused kernel wins. Bit-identical outputs at
   every shape except one tied-cluster case (7.6e-2 abs diff).

7. **Plots** (`make_plots.py`). Three matplotlib figures saved to
   `/home/infatoshi/work/elliotarledge.com/public/blog/dsa-cpu-topk/`:
   `standalone_topk.png`, `hybrid_endtoend.png`, `crossover_heatmap.png`.
   Embedded in the blog post.

## File map

| File | Purpose |
|---|---|
| `lightning_indexer.py` | Indexer scores: torch ref + Triton kernel |
| `dsa_attention.py` | Full DSA forward: naive / torch / fused-Triton |
| `topk_avx512.cpp` | AVX-512 k-th-largest C++ extension |
| `dsa_hybrid.py` | Hybrid pipeline + threshold-only sparse-attn kernel |
| `bench_avx512.py` | Standalone CPU top-K bench |
| `bench_hybrid.py` | End-to-end DSA bench (the headline) |
| `make_plots.py` | Reproduces blog figures |
| `test_indexer.py`, `test_dsa.py` | Numerical equivalence tests |
| `pyproject.toml`, `uv.lock` | uv project |
| `README.md` | Polished readme pointing at the blog post |

## How to reproduce

```bash
cd /home/infatoshi/cuda/DSA
uv sync
uv run pytest -q                    # 10 tests pass
uv run python bench_avx512.py       # standalone CPU top-K table
uv run python bench_hybrid.py       # headline end-to-end table
uv run python make_plots.py         # regenerate blog figures
```

The C++ extension JIT-compiles on first run; cached in
`~/.cache/torch_extensions/.../topk_avx512_ext/`.

## Known issues / sharp edges

- **Triton kernel register pressure**. `_dsa_fused_kernel` and
  `_sparse_attn_kernel` both materialize K/V tiles of shape
  `[T_const, HQ, DQ]` in registers. PTXAS exits with code -9 (OOM)
  beyond moderate T. Practical ceiling is T ≈ 512 with HQ=4, DQ=32.
- **NaN traps in Triton softmax**. Hit twice during build:
  - `0.0 * -inf = NaN` in scatter-via-multiply tricks → use `tl.where`
  - `0.0 * -inf = NaN` in one-hot threshold extraction from sorted
    array → use `tl.where`
  - `tl.exp(-inf)` may produce NaN with fast intrinsics in some
    paths; current code uses a finite sentinel (`NEG = -50.0`) instead
    of `-inf` for the masking value. See comments in
    `dsa_attention.py::_dsa_fused_kernel`.
- **Tie-handling**. All three DSA impls use threshold-based selection
  (keeps ties), not `topk` indices. Don't switch one without switching
  all three or numerical tests will fail.
- **Other process contention**. There's a long-running
  `python3 benchmark.py` (KernelBench-Hard sonic-moe MoE work) burning
  CPU on this box. Pin top-K benches to specific cores with `taskset
  -c 0-15` for clean numbers, or expect 5-15% noise.
- **`bench_avx512.py` and `dsa_hybrid.py`** both compile the same
  extension via `cpp_extension.load` with `name="topk_avx512_ext"`.
  Torch caches by name+source-hash, so this is fine, but if you
  change compile flags in only one file you'll get stale cached
  binaries. Either bump the name or clear
  `~/.cache/torch_extensions/`.

## Numerical notes

- Indexer scores match torch ref to atol=rtol=1e-4 with
  `tl.dot(..., input_precision="ieee")`. Without IEEE, TF32 gives
  ~5e-2 max abs diff at the values we hit.
- Hybrid vs all-GPU output max abs diff is 0 across most shapes; one
  tied-cluster case (B=64, T=256, k=32) showed 7.6e-2 because GPU's
  internal `tl.sort` extracts a slightly different threshold than the
  CPU's k-th-largest when many scores tie at the boundary. Both
  selections are valid top-K.

## Suggested next steps

1. **Replace `tl.sort` with radix-select on the GPU.** This is the
   actually-correct fix and would close the 1.2× hybrid win. Reference:
   FAISS `BlockSelect` / `WarpSelect`. Would also raise the T ceiling
   for the fused kernel (sort is a big chunk of the register/scratch
   pressure budget).

2. **Rewrite the AVX-512 buffer-insertion path.** The O(k) scalar
   insertion is what kills the CPU at large k. Two options:
   - Vectorized min-heap in a single ZMM (k=16 fits trivially)
   - Bitonic-block-merge: collect candidates into a 16-wide buffer,
     then merge with the existing top-k using AVX-512 bitonic
     primitives. Could close the break-even at k=128 and push the
     hybrid win to higher k regimes.

3. **Split the Triton kernel into indexer + attention.** Right now
   they're glued; splitting lets you reuse the indexer for the hybrid
   path without recompilation, and makes the register-pressure
   ceiling per-kernel rather than for both at once.

4. **Try FP8 indexer per the paper.** Indexer has tiny heads (HI=4,
   DI=32) so tensor cores are underutilized in fp32. The paper does
   FP8 here. Would need to handle scaling/dequant inside the kernel.

5. **Overlap d2h with the next batch's GPU work.** Current hybrid
   path is serial (GPU → CPU → GPU). With CUDA streams you could
   pipeline so the CPU top-K of batch i runs while GPU does indexer
   of batch i+1. Only matters in batch-streaming inference.

6. **Profile the all-GPU kernel's sort cost.** Add Nsight Compute
   to confirm `tl.sort` is the actual hot path at T=512. The argument
   that "GPU sort dominates at T=512" is inferred from end-to-end
   timing, not directly measured.

7. **Rerun on Grace+Rubin** when accessible. NVLink-C2C makes d2h
   nearly free for moderate sizes; this is the only realistic
   platform where CPU offload could become a real production tool.

## Current git state

Both repos clean and pushed:

```
dsa-cpu-topk:        main @ ff9ee39  Add matplotlib plot script for blog visuals
elliotarledge.com:   main @ eed7b8e  Add matplotlib visuals to DSA CPU top-K blog post
```

Last bench run numbers (the ones in the blog plots):
- Standalone top-K wins: 24.5× (1,256,16) down to 1.4× (64,1024,64)
- Hybrid end-to-end wins: 1.24× (16,512,32), 1.07× (64,512,64)

If those numbers look very different on a rerun, check (a) whether
the contender benchmark.py is still hammering the CPU, (b) whether
torch_extensions cache is warm, (c) GPU clock state via
`nvidia-smi --query-gpu=clocks.gr,clocks.mem,power.draw --format=csv`.
