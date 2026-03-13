from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

_SUPPORTED_PROFILES = {"portable", "native", "fixed"}
_SUPPORTED_LEVELS = {"SSE3", "AVX2", "AVX512", "NATIVE"}


@dataclass(frozen=True)
class X86BuildConfig:
    profile: str
    simd_level: str
    baseline: str
    dispatch_enabled: bool


def _normalize_profile(profile: str) -> str:
    normalized = profile.strip().lower()
    if normalized not in _SUPPORTED_PROFILES:
        raise ValueError(
            f"Unsupported OV_X86_BUILD_PROFILE={profile!r}; "
            "expected one of: fixed, native, portable"
        )
    return normalized


def _normalize_level(level: str) -> str:
    normalized = level.strip().upper()
    if normalized not in _SUPPORTED_LEVELS:
        raise ValueError(
            f"Unsupported OV_X86_SIMD_LEVEL={level!r}; expected one of: SSE3, AVX2, AVX512, NATIVE"
        )
    return normalized


def _is_wheel_build(argv: Sequence[str]) -> bool:
    wheel_commands = {"bdist_wheel", "editable_wheel"}
    return any(arg in wheel_commands for arg in argv)


def resolve_x86_build_config(
    env: Mapping[str, str] | None = None,
    argv: Sequence[str] | None = None,
) -> X86BuildConfig:
    env = env or {}
    argv = argv or []

    baseline = _normalize_level(env.get("OV_X86_PORTABLE_BASELINE", "SSE3"))
    if baseline == "NATIVE":
        raise ValueError("OV_X86_PORTABLE_BASELINE cannot be NATIVE")

    explicit_profile = env.get("OV_X86_BUILD_PROFILE")
    explicit_level = env.get("OV_X86_SIMD_LEVEL")

    if explicit_profile:
        profile = _normalize_profile(explicit_profile)
    elif explicit_level:
        normalized_level = _normalize_level(explicit_level)
        profile = "native" if normalized_level == "NATIVE" else "fixed"
    else:
        profile = "portable" if _is_wheel_build(argv) else "native"

    if profile == "portable":
        return X86BuildConfig(
            profile="portable",
            simd_level=baseline,
            baseline=baseline,
            dispatch_enabled=True,
        )

    if profile == "native":
        return X86BuildConfig(
            profile="native",
            simd_level="NATIVE",
            baseline=baseline,
            dispatch_enabled=False,
        )

    simd_level = _normalize_level(explicit_level or "AVX2")
    if simd_level == "NATIVE":
        raise ValueError("OV_X86_SIMD_LEVEL=NATIVE requires OV_X86_BUILD_PROFILE=native")

    return X86BuildConfig(
        profile="fixed",
        simd_level=simd_level,
        baseline=baseline,
        dispatch_enabled=False,
    )
