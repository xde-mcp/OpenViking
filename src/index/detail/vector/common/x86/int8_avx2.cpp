// Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "../vector_base.h"

#include <algorithm>
#include <cstdint>
#include <immintrin.h>

namespace vectordb {

namespace {

int32_t inner_product_int8_avx2_kernel(const void* v1, const void* v2,
                                       const void* params) {
  const int8_t* pv1 = static_cast<const int8_t*>(v1);
  const int8_t* pv2 = static_cast<const int8_t*>(v2);
  size_t dim = *static_cast<const size_t*>(params);

  size_t dim32 = (dim / 32) * 32;
  __m256i sum_vec = _mm256_setzero_si256();

  for (size_t i = 0; i < dim32; i += 32) {
    __m256i vec1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pv1 + i));
    __m256i vec2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pv2 + i));

    __m128i v1_lo = _mm256_castsi256_si128(vec1);
    __m128i v1_hi = _mm256_extracti128_si256(vec1, 1);
    __m128i v2_lo = _mm256_castsi256_si128(vec2);
    __m128i v2_hi = _mm256_extracti128_si256(vec2, 1);

    __m256i v1_lo_16 = _mm256_cvtepi8_epi16(v1_lo);
    __m256i v2_lo_16 = _mm256_cvtepi8_epi16(v2_lo);
    __m256i v1_hi_16 = _mm256_cvtepi8_epi16(v1_hi);
    __m256i v2_hi_16 = _mm256_cvtepi8_epi16(v2_hi);

    __m256i prod_lo = _mm256_mullo_epi16(v1_lo_16, v2_lo_16);
    __m256i prod_hi = _mm256_mullo_epi16(v1_hi_16, v2_hi_16);

    sum_vec = _mm256_add_epi32(
        sum_vec, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(prod_lo)));
    sum_vec = _mm256_add_epi32(
        sum_vec, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(prod_lo, 1)));
    sum_vec = _mm256_add_epi32(
        sum_vec, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(prod_hi)));
    sum_vec = _mm256_add_epi32(
        sum_vec, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(prod_hi, 1)));
  }

  __m128i sum_hi = _mm256_extracti128_si256(sum_vec, 1);
  __m128i sum_lo = _mm256_castsi256_si128(sum_vec);
  __m128i sum128 = _mm_add_epi32(sum_lo, sum_hi);

  int32_t OV_ALIGN_32 temp[4];
  _mm_store_si128(reinterpret_cast<__m128i*>(temp), sum128);
  int32_t sum = temp[0] + temp[1] + temp[2] + temp[3];

  for (size_t i = dim32; i < dim; ++i) {
    sum += static_cast<int32_t>(pv1[i]) * static_cast<int32_t>(pv2[i]);
  }

  return sum;
}

}  // namespace

float inner_product_distance_int8_avx2_kernel(const void* v1, const void* v2,
                                              const void* params) {
  size_t dim = *static_cast<const size_t*>(params);
  const float* scale1_ptr =
      reinterpret_cast<const float*>(static_cast<const int8_t*>(v1) + dim);
  const float* scale2_ptr =
      reinterpret_cast<const float*>(static_cast<const int8_t*>(v2) + dim);

  float scale1 = *scale1_ptr;
  float scale2 = *scale2_ptr;
  int32_t ip = inner_product_int8_avx2_kernel(v1, v2, params);
  return static_cast<float>(ip) * scale1 * scale2;
}

float l2_distance_int8_avx2_kernel(const void* v1, const void* v2,
                                   const void* params) {
  size_t dim = *static_cast<const size_t*>(params);
  const float* meta1 =
      reinterpret_cast<const float*>(static_cast<const int8_t*>(v1) + dim);
  const float* meta2 =
      reinterpret_cast<const float*>(static_cast<const int8_t*>(v2) + dim);

  float scale1 = meta1[0];
  float norm_sq1 = meta1[1];
  float scale2 = meta2[0];
  float norm_sq2 = meta2[1];

  int32_t ip = inner_product_int8_avx2_kernel(v1, v2, params);
  float real_ip = static_cast<float>(ip) * scale1 * scale2;
  return std::max(0.0f, norm_sq1 + norm_sq2 - 2.0f * real_ip);
}

}  // namespace vectordb
