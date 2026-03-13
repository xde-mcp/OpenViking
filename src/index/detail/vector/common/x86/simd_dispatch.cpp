// Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "simd_dispatch.h"
#include "../vector_base.h"

#include <mutex>

namespace vectordb {

namespace {

bool PortableAvx512Compiled() {
#if defined(OV_X86_AVX512_DISPATCH_COMPILED)
  return true;
#else
  return false;
#endif
}

bool PortableAvx2Compiled() {
#if defined(OV_X86_AVX2_DISPATCH_COMPILED)
  return true;
#else
  return false;
#endif
}

}  // namespace

X86SimdSupport DetectX86SimdSupport() {
  X86SimdSupport support{};

#if defined(OV_PLATFORM_X86) && (defined(__GNUC__) || defined(__clang__))
  __builtin_cpu_init();
  support.sse3 = __builtin_cpu_supports("sse3");
  support.avx2 = __builtin_cpu_supports("avx2");
  support.avx512f = __builtin_cpu_supports("avx512f");
  support.avx512bw = __builtin_cpu_supports("avx512bw");
  support.avx512dq = __builtin_cpu_supports("avx512dq");
  support.avx512vl = __builtin_cpu_supports("avx512vl");
  support.ymm_state = support.avx2;
  support.zmm_state = support.avx512f && support.avx512bw &&
                      support.avx512dq && support.avx512vl;
#endif

  return support;
}

X86SimdBackend ResolvePortableX86SimdBackend(const X86SimdSupport& support,
                                             bool avx2_compiled,
                                             bool avx512_compiled) {
  if (avx512_compiled && support.avx512f && support.avx512bw &&
      support.avx512dq && support.avx512vl && support.zmm_state) {
    return X86SimdBackend::kAvx512;
  }

  if (avx2_compiled && support.avx2 && support.ymm_state) {
    return X86SimdBackend::kAvx2;
  }

  if (support.sse3) {
    return X86SimdBackend::kSse3;
  }

  return X86SimdBackend::kScalar;
}

X86SimdBackend GetPortableX86SimdBackend() {
  static std::once_flag once;
  static X86SimdBackend backend = X86SimdBackend::kScalar;

  std::call_once(once, []() {
    backend = ResolvePortableX86SimdBackend(
        DetectX86SimdSupport(), PortableAvx2Compiled(), PortableAvx512Compiled());
  });

  return backend;
}

const char* X86SimdBackendName(X86SimdBackend backend) {
  switch (backend) {
    case X86SimdBackend::kAvx512:
      return "avx512";
    case X86SimdBackend::kAvx2:
      return "avx2";
    case X86SimdBackend::kSse3:
      return "sse3";
    case X86SimdBackend::kScalar:
    default:
      return "scalar";
  }
}

const char* GetActiveSimdBackendName() {
#if defined(OV_X86_RUNTIME_DISPATCH)
  return X86SimdBackendName(GetPortableX86SimdBackend());
#elif defined(OV_SIMD_NEON)
  return "neon";
#elif defined(OV_SIMD_AVX512)
  return "avx512";
#elif defined(__AVX2__)
  return "avx2";
#elif defined(OV_SIMD_AVX)
  return "avx";
#elif defined(OV_SIMD_SSE)
  return "sse3";
#else
  return "scalar";
#endif
}

}  // namespace vectordb
