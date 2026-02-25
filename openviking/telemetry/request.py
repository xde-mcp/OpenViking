# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Telemetry request parsing helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias

TelemetryRequest: TypeAlias = bool | dict[str, bool]

_ALLOWED_TELEMETRY_KEYS = frozenset({"summary", "events"})


@dataclass(frozen=True)
class TelemetrySelection:
    """Normalized telemetry payload selection."""

    include_summary: bool
    include_events: bool

    @property
    def include_payload(self) -> bool:
        return self.include_summary or self.include_events


def normalize_telemetry_request(
    request: TelemetryRequest | Mapping[str, Any] | None,
) -> TelemetrySelection:
    """Normalize a telemetry request into explicit response selection flags."""
    if request is None or request is False:
        return TelemetrySelection(include_summary=False, include_events=False)
    if request is True:
        return TelemetrySelection(include_summary=True, include_events=True)
    if not isinstance(request, Mapping):
        raise ValueError("telemetry must be a boolean or an object")

    unknown_keys = set(request) - _ALLOWED_TELEMETRY_KEYS
    if unknown_keys:
        joined = ", ".join(sorted(unknown_keys))
        raise ValueError(f"Unsupported telemetry options: {joined}")

    include_summary = request.get("summary", True)
    include_events = request.get("events", False)

    if not isinstance(include_summary, bool):
        raise ValueError("telemetry.summary must be a boolean")
    if not isinstance(include_events, bool):
        raise ValueError("telemetry.events must be a boolean")
    return TelemetrySelection(
        include_summary=include_summary,
        include_events=include_events,
    )


__all__ = ["TelemetryRequest", "TelemetrySelection", "normalize_telemetry_request"]
