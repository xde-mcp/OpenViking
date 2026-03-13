from build_support.x86_profiles import resolve_x86_build_config


def test_wheel_build_defaults_to_portable_profile():
    config = resolve_x86_build_config(env={}, argv=["setup.py", "bdist_wheel"])

    assert config.profile == "portable"
    assert config.baseline == "SSE3"
    assert config.dispatch_enabled is True


def test_source_build_defaults_to_native_profile():
    config = resolve_x86_build_config(env={}, argv=["setup.py", "build_ext", "--inplace"])

    assert config.profile == "native"
    assert config.simd_level == "NATIVE"
    assert config.dispatch_enabled is False


def test_explicit_fixed_profile_uses_requested_simd_level():
    config = resolve_x86_build_config(
        env={"OV_X86_BUILD_PROFILE": "fixed", "OV_X86_SIMD_LEVEL": "AVX2"},
        argv=["setup.py", "build_ext", "--inplace"],
    )

    assert config.profile == "fixed"
    assert config.simd_level == "AVX2"
    assert config.dispatch_enabled is False


def test_legacy_native_simd_level_maps_to_native_profile():
    config = resolve_x86_build_config(
        env={"OV_X86_SIMD_LEVEL": "native"},
        argv=["setup.py", "build_ext", "--inplace"],
    )

    assert config.profile == "native"
    assert config.simd_level == "NATIVE"
