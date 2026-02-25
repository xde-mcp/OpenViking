# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Resource operation telemetry hooks."""

from __future__ import annotations

from typing import Dict

from .context import get_current_telemetry


def on_add_resource_start(*, wait: bool, has_target: bool) -> None:
    get_current_telemetry().event(
        "resource_service.add_resource",
        "start",
        {"wait": wait, "has_target": has_target},
    )


def on_resource_processed(*, status: str) -> None:
    get_current_telemetry().event(
        "resource_service.add_resource",
        "resource_processed",
        {"status": status},
    )


def on_queue_wait_start(*, timeout: float | None) -> None:
    get_current_telemetry().event(
        "resource_service.add_resource",
        "queue_wait_start",
        {"timeout": timeout},
    )


def on_queue_wait_complete(*, duration_ms: float) -> None:
    get_current_telemetry().event(
        "resource_service.add_resource",
        "queue_wait_complete",
        {"duration_ms": duration_ms},
    )


def on_queue_wait_timeout(exc: Exception) -> None:
    get_current_telemetry().set_error(
        "resource_service.wait_complete",
        "DEADLINE_EXCEEDED",
        str(exc),
    )


def on_queue_status_collected(queue_summary: Dict[str, Dict[str, int]]) -> None:
    get_current_telemetry().event(
        "resource_service.add_resource",
        "queue_status_collected",
        {
            "semantic_processed": queue_summary["semantic"]["processed"],
            "semantic_error_count": queue_summary["semantic"]["error_count"],
            "embedding_processed": queue_summary["embedding"]["processed"],
            "embedding_error_count": queue_summary["embedding"]["error_count"],
        },
    )


def on_add_resource_done(
    *,
    root_uri: str,
    wait: bool,
    result_status: str,
    total_duration_ms: float,
) -> None:
    get_current_telemetry().event(
        "resource_service.add_resource",
        "done",
        {
            "root_uri": root_uri,
            "wait": wait,
            "result_status": result_status,
            "total_duration_ms": total_duration_ms,
        },
    )


def on_add_resource_error(*, wait: bool, exc: Exception, total_duration_ms: float) -> None:
    get_current_telemetry().event(
        "resource_service.add_resource",
        "done",
        {
            "wait": wait,
            "error": type(exc).__name__,
            "message": str(exc),
            "total_duration_ms": total_duration_ms,
        },
        status="error",
    )


def on_add_skill_start(*, wait: bool) -> None:
    get_current_telemetry().event(
        "resource_service.add_skill",
        "start",
        {"wait": wait},
    )


def on_skill_processed(*, status: str) -> None:
    get_current_telemetry().event(
        "resource_service.add_skill",
        "skill_processed",
        {"status": status},
    )


def on_add_skill_queue_wait_complete(*, duration_ms: float) -> None:
    get_current_telemetry().event(
        "resource_service.add_skill",
        "queue_wait_complete",
        {"duration_ms": duration_ms},
    )


def on_add_skill_queue_wait_timeout(exc: Exception) -> None:
    get_current_telemetry().set_error(
        "resource_service.wait_complete",
        "DEADLINE_EXCEEDED",
        str(exc),
    )
