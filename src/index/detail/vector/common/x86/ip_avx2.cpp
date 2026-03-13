// Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "../vector_base.h"

#include <immintrin.h>

namespace vectordb {

float inner_product_avx2_kernel(const void* v1, const void* v2,
                                const void* params) {
  const float* pv1 = static_cast<const float*>(v1);
  const float* pv2 = static_cast<const float*>(v2);
  size_t dim = *static_cast<const size_t*>(params);

  __m256 sum = _mm256_setzero_ps();
  size_t i = 0;

  for (; i + 8 <= dim; i += 8) {
    __m256 a = _mm256_loadu_ps(pv1 + i);
    __m256 b = _mm256_loadu_ps(pv2 + i);
    sum = _mm256_add_ps(sum, _mm256_mul_ps(a, b));
  }

  __m128 sum_low = _mm256_extractf128_ps(sum, 0);
  __m128 sum_high = _mm256_extractf128_ps(sum, 1);
  __m128 sum128 = _mm_add_ps(sum_low, sum_high);
  sum128 = _mm_hadd_ps(sum128, sum128);
  sum128 = _mm_hadd_ps(sum128, sum128);

  float res = _mm_cvtss_f32(sum128);
  for (; i < dim; ++i) {
    res += pv1[i] * pv2[i];
  }

  return res;
}

}  // namespace vectordb
