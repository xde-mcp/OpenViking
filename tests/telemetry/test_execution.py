from types import SimpleNamespace

import pytest

from openviking_cli.exceptions import InvalidArgumentError
from openviking_cli.retrieve.types import FindResult


@pytest.mark.asyncio
async def test_run_with_telemetry_returns_usage_and_payload():
    from openviking.telemetry.execution import run_with_telemetry

    async def _run():
        return {"status": "ok"}

    execution = await run_with_telemetry(
        operation="search.find",
        telemetry=True,
        fn=_run,
    )

    assert execution.result == {"status": "ok"}
    assert execution.usage is not None
    assert execution.telemetry is not None
    assert execution.telemetry["summary"]["operation"] == "search.find"


@pytest.mark.asyncio
async def test_run_with_telemetry_raises_invalid_argument_for_bad_request():
    from openviking.telemetry.execution import run_with_telemetry

    async def _run():
        return {"status": "ok"}

    with pytest.raises(InvalidArgumentError, match="Unsupported telemetry options: invalid"):
        await run_with_telemetry(
            operation="search.find",
            telemetry={"invalid": True},
            fn=_run,
        )


def test_attach_telemetry_payload_adds_usage_and_telemetry_to_dict_result():
    from openviking.telemetry.execution import attach_telemetry_payload

    result = attach_telemetry_payload(
        {"root_uri": "viking://resources/demo"},
        {"duration_ms": 1.2, "token_total": 3},
        {"id": "tm_123", "summary": {"operation": "resources.add_resource"}},
    )

    assert result["usage"]["token_total"] == 3
    assert result["telemetry"]["summary"]["operation"] == "resources.add_resource"


def test_attach_telemetry_payload_does_not_mutate_object_result():
    from openviking.telemetry.execution import attach_telemetry_payload

    result = SimpleNamespace(total=1)

    attached = attach_telemetry_payload(
        result,
        {"duration_ms": 1.2, "token_total": 3},
        {"id": "tm_123", "summary": {"operation": "search.find"}},
    )

    assert attached is result
    assert not hasattr(result, "usage")
    assert not hasattr(result, "telemetry")


def test_find_result_ignores_usage_and_telemetry_payload_fields():
    result = FindResult.from_dict(
        {
            "memories": [],
            "resources": [],
            "skills": [],
            "usage": {"duration_ms": 1.2, "token_total": 3},
            "telemetry": {"id": "tm_123", "summary": {"operation": "search.find"}},
        }
    )

    assert not hasattr(result, "usage")
    assert not hasattr(result, "telemetry")
    assert result.to_dict() == {
        "memories": [],
        "resources": [],
        "skills": [],
        "total": 0,
    }
