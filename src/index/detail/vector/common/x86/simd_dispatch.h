// Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace vectordb {

enum class X86SimdBackend {
  kScalar = 0,
  kSse3 = 1,
  kAvx2 = 2,
  kAvx512 = 3,
};

struct X86SimdSupport {
  bool sse3 = false;
  bool avx2 = false;
  bool avx512f = false;
  bool avx512bw = false;
  bool avx512dq = false;
  bool avx512vl = false;
  bool ymm_state = false;
  bool zmm_state = false;
};

X86SimdSupport DetectX86SimdSupport();

X86SimdBackend ResolvePortableX86SimdBackend(const X86SimdSupport& support,
                                             bool avx2_compiled,
                                             bool avx512_compiled);

X86SimdBackend GetPortableX86SimdBackend();

const char* X86SimdBackendName(X86SimdBackend backend);
const char* GetActiveSimdBackendName();

}  // namespace vectordb
