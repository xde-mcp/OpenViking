from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DISALLOWED_PREFIXES = (
    "get_current_telemetry(",
    "telemetry.event(",
    "telemetry.count(",
    "telemetry.set(",
    "telemetry.set_error(",
)

ALLOWED_FILES = {
    "openviking/session/memory_deduplicator.py",
    "openviking/telemetry/resource_hooks.py",
    "openviking/telemetry/session_hooks.py",
    "openviking/telemetry/retriever_hooks.py",
    "openviking/telemetry/search_hooks.py",
}

CHECK_DIRS = (
    "openviking/service",
    "openviking/session",
    "openviking/retrieve",
)


def test_core_layers_do_not_directly_call_telemetry_collectors():
    offenders: list[str] = []
    for check_dir in CHECK_DIRS:
        for path in (ROOT / check_dir).rglob("*.py"):
            rel = path.relative_to(ROOT).as_posix()
            if rel in ALLOWED_FILES:
                continue
            text = path.read_text()
            for needle in DISALLOWED_PREFIXES:
                if needle in text:
                    offenders.append(f"{rel}: {needle}")

    assert offenders == []
