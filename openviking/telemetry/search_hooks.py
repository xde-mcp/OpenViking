# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Search service telemetry hooks."""

from __future__ import annotations

from .context import get_current_telemetry


def on_search_start(*, target_uri_present: bool, has_session: bool) -> None:
    get_current_telemetry().event(
        "search_service.search",
        "start",
        {"target_uri_present": target_uri_present, "has_session": has_session},
    )


def on_search_done(*, total: int) -> None:
    get_current_telemetry().event(
        "search_service.search",
        "done",
        {"total": total},
    )


def on_find_start(*, target_uri_present: bool) -> None:
    get_current_telemetry().event(
        "search_service.find",
        "start",
        {"target_uri_present": target_uri_present},
    )


def on_find_done(*, total: int) -> None:
    get_current_telemetry().event(
        "search_service.find",
        "done",
        {"total": total},
    )
