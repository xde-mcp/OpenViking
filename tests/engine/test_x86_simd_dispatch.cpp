// Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "index/detail/vector/common/x86/simd_dispatch.h"

#include <cassert>

using vectordb::ResolvePortableX86SimdBackend;
using vectordb::X86SimdBackend;
using vectordb::X86SimdSupport;

namespace {

X86SimdSupport MakeSupport(bool sse3, bool avx2, bool avx512) {
  X86SimdSupport support{};
  support.sse3 = sse3;
  support.avx2 = avx2;
  support.avx512f = avx512;
  support.avx512bw = avx512;
  support.avx512dq = avx512;
  support.avx512vl = avx512;
  support.ymm_state = avx2 || avx512;
  support.zmm_state = avx512;
  return support;
}

}  // namespace

int main() {
  assert(ResolvePortableX86SimdBackend(MakeSupport(false, false, false), true,
                                       true) == X86SimdBackend::kScalar);
  assert(ResolvePortableX86SimdBackend(MakeSupport(true, false, false), true,
                                       true) == X86SimdBackend::kSse3);
  assert(ResolvePortableX86SimdBackend(MakeSupport(true, true, false), true,
                                       true) == X86SimdBackend::kAvx2);
  assert(ResolvePortableX86SimdBackend(MakeSupport(true, true, true), true,
                                       true) == X86SimdBackend::kAvx512);
  assert(ResolvePortableX86SimdBackend(MakeSupport(true, true, true), true,
                                       false) == X86SimdBackend::kAvx2);

  X86SimdSupport missing_os_state = MakeSupport(true, true, true);
  missing_os_state.zmm_state = false;
  assert(ResolvePortableX86SimdBackend(missing_os_state, true, true) ==
         X86SimdBackend::kAvx2);

  return 0;
}
