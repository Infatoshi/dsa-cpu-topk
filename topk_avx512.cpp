// AVX-512 top-K threshold for fp32 rows.
// Computes per-row k-th largest value (descending) — what DSA needs as the
// selection threshold. No indices, no full sort. Parallel over rows via OpenMP.
//
// Strategy: maintain a small descending-sorted buffer of size k per row.
// Scan input 16 floats at a time with AVX-512:
//   1. Vector compare against current threshold (= buf[k-1])
//   2. If no element exceeds, skip (cheap, common case once buffer fills)
//   3. Else compress-store the candidates and insert each into the sorted buf
//
// The vector scan is the dominant cost; insertion is amortized cheap.

#include <torch/extension.h>
#include <immintrin.h>
#include <algorithm>
#include <limits>
#include <vector>

namespace {

inline void insert_into_sorted_desc(float* buf, int k, float v) {
    // buf is sorted descending; buf[k-1] is the smallest. Insert v if v > buf[k-1].
    if (v <= buf[k - 1]) return;
    int idx = k - 1;
    while (idx > 0 && buf[idx - 1] < v) {
        buf[idx] = buf[idx - 1];
        --idx;
    }
    buf[idx] = v;
}

float row_kth_largest(const float* row, int64_t N, int k) {
    std::vector<float> buf(k, -std::numeric_limits<float>::infinity());

    int64_t i = 0;
    alignas(64) float spill[16];

    for (; i + 16 <= N; i += 16) {
        __m512 x = _mm512_loadu_ps(row + i);
        __m512 thr = _mm512_set1_ps(buf[k - 1]);
        __mmask16 m = _mm512_cmp_ps_mask(x, thr, _CMP_GT_OS);
        if (m == 0) continue;
        // Compress-store the elements that beat the current threshold
        _mm512_mask_compressstoreu_ps(spill, m, x);
        int n = __builtin_popcount((unsigned)m);
        for (int j = 0; j < n; ++j) insert_into_sorted_desc(buf.data(), k, spill[j]);
    }
    for (; i < N; ++i) insert_into_sorted_desc(buf.data(), k, row[i]);
    return buf[k - 1];
}

}  // namespace

torch::Tensor topk_threshold_avx512(torch::Tensor scores, int64_t k) {
    TORCH_CHECK(scores.is_contiguous(), "scores must be contiguous");
    TORCH_CHECK(scores.scalar_type() == torch::kFloat32, "fp32 only");
    TORCH_CHECK(scores.device().is_cpu(), "cpu only");
    TORCH_CHECK(k > 0, "k must be > 0");

    auto sizes = scores.sizes().vec();
    int64_t N = sizes.back();
    TORCH_CHECK(k <= N, "k must be <= N");
    int64_t rows = scores.numel() / N;
    auto out_sizes = sizes;
    out_sizes.back() = 1;
    auto out = torch::empty(out_sizes, scores.options());

    const float* in = scores.data_ptr<float>();
    float* op = out.data_ptr<float>();

    int64_t r;
#pragma omp parallel for schedule(static)
    for (r = 0; r < rows; ++r) {
        op[r] = row_kth_largest(in + r * N, N, (int)k);
    }
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("topk_threshold", &topk_threshold_avx512,
          "AVX-512 top-K threshold (k-th largest per row, fp32)");
}
