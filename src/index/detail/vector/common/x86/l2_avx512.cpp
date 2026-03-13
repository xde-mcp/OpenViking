// Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "../vector_base.h"

#include <immintrin.h>

namespace vectordb {

float l2_sqr_avx512_kernel(const void* v1, const void* v2, const void* params) {
  const float* pv1 = static_cast<const float*>(v1);
  const float* pv2 = static_cast<const float*>(v2);
  size_t dim = *static_cast<const size_t*>(params);

  __m512 sum = _mm512_setzero_ps();
  size_t i = 0;

  for (; i + 16 <= dim; i += 16) {
    __m512 a = _mm512_loadu_ps(pv1 + i);
    __m512 b = _mm512_loadu_ps(pv2 + i);
    __m512 diff = _mm512_sub_ps(a, b);
    sum = _mm512_add_ps(sum, _mm512_mul_ps(diff, diff));
  }

  float res = _mm512_reduce_add_ps(sum);
  for (; i < dim; ++i) {
    float diff = pv1[i] - pv2[i];
    res += diff * diff;
  }

  return res;
}

}  // namespace vectordb
