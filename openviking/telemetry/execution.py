# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for telemetry-wrapped operation execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Generic, Optional, TypeVar

from openviking_cli.exceptions import InvalidArgumentError

from .context import bind_telemetry
from .operation import OperationTelemetry
from .request import TelemetryRequest, TelemetrySelection, normalize_telemetry_request

T = TypeVar("T")


@dataclass
class TelemetryExecutionResult(Generic[T]):
    """Executed operation result plus telemetry payloads."""

    result: T
    usage: Optional[dict[str, Any]]
    telemetry: Optional[dict[str, Any]]
    selection: TelemetrySelection


def parse_telemetry_selection(telemetry: TelemetryRequest) -> TelemetrySelection:
    """Validate and normalize a telemetry request for public API usage."""
    try:
        return normalize_telemetry_request(telemetry)
    except ValueError as exc:
        raise InvalidArgumentError(str(exc)) from exc


def build_telemetry_payload(
    collector: OperationTelemetry,
    selection: TelemetrySelection,
    *,
    status: str = "ok",
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Build usage and telemetry payloads from a finished collector."""
    snapshot = collector.finish(status=status)
    if snapshot is None:
        return None, None

    usage = snapshot.to_usage_dict()
    telemetry_payload = None
    if selection.include_payload:
        telemetry_payload = snapshot.to_dict(
            include_summary=selection.include_summary,
            include_events=selection.include_events,
        )
    return usage, telemetry_payload


def attach_telemetry_payload(
    result: Any,
    usage: Optional[dict[str, Any]],
    telemetry_payload: Optional[dict[str, Any]],
) -> Any:
    """Attach telemetry payloads to a dict or result object."""
    if usage is None and telemetry_payload is None:
        return result

    if result is None:
        payload: dict[str, Any] = {}
        if usage is not None:
            payload["usage"] = usage
        if telemetry_payload is not None:
            payload["telemetry"] = telemetry_payload
        return payload

    if isinstance(result, dict):
        if usage is not None:
            result["usage"] = usage
        if telemetry_payload is not None:
            result["telemetry"] = telemetry_payload
        return result

    return result


async def run_with_telemetry(
    *,
    operation: str,
    telemetry: TelemetryRequest,
    fn: Callable[[], Awaitable[T]],
    error_status: str = "error",
) -> TelemetryExecutionResult[T]:
    """Execute an async operation with a bound operation-scoped collector."""
    selection = parse_telemetry_selection(telemetry)
    collector = OperationTelemetry(
        operation=operation,
        enabled=True,
        capture_events=selection.include_events,
    )

    try:
        with bind_telemetry(collector):
            result = await fn()
    except Exception as exc:
        collector.set_error(operation, type(exc).__name__, str(exc))
        collector.finish(status=error_status)
        raise

    usage, telemetry_payload = build_telemetry_payload(
        collector,
        selection,
        status="ok",
    )
    return TelemetryExecutionResult(
        result=result,
        usage=usage,
        telemetry=telemetry_payload,
        selection=selection,
    )


__all__ = [
    "TelemetryExecutionResult",
    "attach_telemetry_payload",
    "build_telemetry_payload",
    "parse_telemetry_selection",
    "run_with_telemetry",
]
