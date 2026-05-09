// AVX-512 top-K threshold for fp32 rows.
// Computes per-row k-th largest value (descending) — what DSA needs as the
// selection threshold. No indices, no full sort. Parallel over rows via OpenMP.
//
// Strategy: maintain a top-k buffer per row.
// For k <= 32, keep it descending-sorted; insertion cost is small.
// For larger k, keep a min-heap; accepted candidates cost O(log k), not O(k).
// Scan input 16 floats at a time with AVX-512:
//   1. Vector compare against current threshold (smallest value in top-k)
//   2. If no element exceeds, skip (cheap, common case once buffer fills)
//   3. Else compress-store the candidates and update the top-k buffer
//
// The vector scan is the dominant cost; insertion is amortized cheap.

#include <torch/extension.h>
#include <immintrin.h>
#include <algorithm>
#include <functional>
#include <limits>
#include <vector>

namespace {

template <int K>
inline void insert_into_sorted_desc_fixed(float* buf, float v) {
    if (v <= buf[K - 1]) return;
    int idx = K - 1;
    while (idx > 0 && buf[idx - 1] < v) {
        buf[idx] = buf[idx - 1];
        --idx;
    }
    buf[idx] = v;
}

template <int K>
float row_kth_largest_fixed_buf(const float* row, int64_t N, float* buf) {
    std::copy_n(row, K, buf);
    std::sort(buf, buf + K, std::greater<float>());

    int64_t i = K;
    alignas(64) float spill[16];

    for (; i + 16 <= N; i += 16) {
        __m512 x = _mm512_loadu_ps(row + i);
        __m512 thr = _mm512_set1_ps(buf[K - 1]);
        __mmask16 m = _mm512_cmp_ps_mask(x, thr, _CMP_GT_OS);
        if (m == 0) continue;
        _mm512_mask_compressstoreu_ps(spill, m, x);
        int n = __builtin_popcount((unsigned)m);
        for (int j = 0; j < n; ++j) insert_into_sorted_desc_fixed<K>(buf, spill[j]);
    }
    for (; i < N; ++i) insert_into_sorted_desc_fixed<K>(buf, row[i]);
    return buf[K - 1];
}

inline void min_heap_replace_root(float* heap, int k, float v) {
    int idx = 0;
    while (true) {
        int child = 2 * idx + 1;
        if (child >= k) break;
        int right = child + 1;
        if (right < k && heap[right] < heap[child]) child = right;
        if (v <= heap[child]) break;
        heap[idx] = heap[child];
        idx = child;
    }
    heap[idx] = v;
}

float row_kth_largest_heap_buf(const float* row, int64_t N, int k, float* heap) {
    std::copy_n(row, k, heap);
    std::make_heap(heap, heap + k, std::greater<float>());

    int64_t i = k;
    alignas(64) float spill[16];

    for (; i + 16 <= N; i += 16) {
        __m512 x = _mm512_loadu_ps(row + i);
        __m512 thr = _mm512_set1_ps(heap[0]);
        __mmask16 m = _mm512_cmp_ps_mask(x, thr, _CMP_GT_OS);
        if (m == 0) continue;
        _mm512_mask_compressstoreu_ps(spill, m, x);
        int n = __builtin_popcount((unsigned)m);
        for (int j = 0; j < n; ++j) {
            if (spill[j] > heap[0]) min_heap_replace_root(heap, k, spill[j]);
        }
    }
    for (; i < N; ++i) {
        if (row[i] > heap[0]) min_heap_replace_root(heap, k, row[i]);
    }
    return heap[0];
}

template <int K>
void topk_threshold_fixed_k(const float* in, float* out, int64_t rows, int64_t N) {
#pragma omp parallel
    {
        alignas(64) float buf[K];
        int64_t r;
#pragma omp for schedule(static)
        for (r = 0; r < rows; ++r) {
            out[r] = row_kth_largest_fixed_buf<K>(in + r * N, N, buf);
        }
    }
}

template <int K, int C>
inline void flush_batch_merge(float* top, float* cand, int& cand_count) {
    if (cand_count == 0) return;

    std::sort(cand, cand + cand_count, std::greater<float>());

    alignas(64) float merged[K];
    int i = 0;
    int j = 0;
    for (int out = 0; out < K; ++out) {
        if (j >= cand_count || (i < K && top[i] >= cand[j])) {
            merged[out] = top[i++];
        } else {
            merged[out] = cand[j++];
        }
    }
    std::copy_n(merged, K, top);
    cand_count = 0;
}

template <int K, int C>
float row_kth_largest_batch_merge_buf(const float* row, int64_t N, float* top, float* cand) {
    std::copy_n(row, K, top);
    std::sort(top, top + K, std::greater<float>());

    int cand_count = 0;
    int64_t i = K;
    alignas(64) float spill[16];

    for (; i + 16 <= N; i += 16) {
        __m512 x = _mm512_loadu_ps(row + i);
        __m512 thr = _mm512_set1_ps(top[K - 1]);
        __mmask16 m = _mm512_cmp_ps_mask(x, thr, _CMP_GT_OS);
        if (m == 0) continue;
        _mm512_mask_compressstoreu_ps(spill, m, x);
        int n = __builtin_popcount((unsigned)m);
        for (int j = 0; j < n; ++j) {
            if (spill[j] <= top[K - 1]) continue;
            cand[cand_count++] = spill[j];
            if (cand_count == C) flush_batch_merge<K, C>(top, cand, cand_count);
        }
    }
    for (; i < N; ++i) {
        if (row[i] <= top[K - 1]) continue;
        cand[cand_count++] = row[i];
        if (cand_count == C) flush_batch_merge<K, C>(top, cand, cand_count);
    }
    flush_batch_merge<K, C>(top, cand, cand_count);
    return top[K - 1];
}

template <int K, int C>
void topk_threshold_batch_merge_k(const float* in, float* out, int64_t rows, int64_t N) {
#pragma omp parallel
    {
        alignas(64) float top[K];
        alignas(64) float cand[C];
        int64_t r;
#pragma omp for schedule(static)
        for (r = 0; r < rows; ++r) {
            out[r] = row_kth_largest_batch_merge_buf<K, C>(in + r * N, N, top, cand);
        }
    }
}

void topk_threshold_generic_k(const float* in, float* out, int64_t rows, int64_t N, int k) {
#pragma omp parallel
    {
        std::vector<float> buf(k);
        int64_t r;
#pragma omp for schedule(static)
        for (r = 0; r < rows; ++r) {
            out[r] = row_kth_largest_heap_buf(in + r * N, N, k, buf.data());
        }
    }
}

void topk_threshold_dispatch_heap(const float* in, float* op, int64_t rows, int64_t N, int k) {
    switch (k) {
        case 8:
            topk_threshold_fixed_k<8>(in, op, rows, N);
            break;
        case 16:
            topk_threshold_fixed_k<16>(in, op, rows, N);
            break;
        case 32:
            topk_threshold_fixed_k<32>(in, op, rows, N);
            break;
        default:
            topk_threshold_generic_k(in, op, rows, N, k);
            break;
    }
}

void topk_threshold_dispatch_batch_merge(const float* in, float* op, int64_t rows, int64_t N, int k) {
    switch (k) {
        case 8:
            topk_threshold_fixed_k<8>(in, op, rows, N);
            break;
        case 16:
            topk_threshold_fixed_k<16>(in, op, rows, N);
            break;
        case 32:
            topk_threshold_fixed_k<32>(in, op, rows, N);
            break;
        case 64:
            topk_threshold_batch_merge_k<64, 64>(in, op, rows, N);
            break;
        case 128:
            topk_threshold_batch_merge_k<128, 64>(in, op, rows, N);
            break;
        default:
            topk_threshold_generic_k(in, op, rows, N, k);
            break;
    }
}

}  // namespace

torch::Tensor topk_threshold_impl(torch::Tensor scores, int64_t k, bool batch_merge) {
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

    if (batch_merge) {
        topk_threshold_dispatch_batch_merge(in, op, rows, N, (int)k);
    } else {
        topk_threshold_dispatch_heap(in, op, rows, N, (int)k);
    }
    return out;
}

torch::Tensor topk_threshold_avx512(torch::Tensor scores, int64_t k) {
    return topk_threshold_impl(scores, k, false);
}

torch::Tensor topk_threshold_batch_merge_avx512(torch::Tensor scores, int64_t k) {
    return topk_threshold_impl(scores, k, true);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("topk_threshold", &topk_threshold_avx512,
          "AVX-512 top-K threshold (k-th largest per row, fp32)");
    m.def("topk_threshold_batch_merge", &topk_threshold_batch_merge_avx512,
          "AVX-512 top-K threshold with batch-merge path for k=64/128");
}
