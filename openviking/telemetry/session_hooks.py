# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Session operation telemetry hooks."""

from __future__ import annotations

from .context import get_current_telemetry


def on_service_commit_start(session_id: str) -> None:
    get_current_telemetry().event(
        "session_service.commit",
        "start",
        {"session_id": session_id},
    )


def on_service_commit_done(memories_extracted: int) -> None:
    get_current_telemetry().event(
        "session_service.commit",
        "done",
        {"memories_extracted": memories_extracted},
    )


def on_commit_start(session_id: str) -> None:
    get_current_telemetry().event(
        "session.commit",
        "commit_start",
        {"session_id": session_id},
    )


def on_memory_extracted(memories_extracted: int) -> None:
    telemetry = get_current_telemetry()
    telemetry.set("memory.extracted", memories_extracted)
    telemetry.event(
        "session.commit",
        "memory_extraction_done",
        {"memories_extracted": memories_extracted},
    )


def on_commit_done(memories_extracted: int) -> None:
    get_current_telemetry().event(
        "session.commit",
        "commit_done",
        {"memories_extracted": memories_extracted},
    )


def on_commit_done_with_memory(memories_extracted: int) -> None:
    telemetry = get_current_telemetry()
    telemetry.set("memory.extracted", memories_extracted)
    telemetry.event(
        "session.commit",
        "commit_done",
        {"memories_extracted": memories_extracted},
    )
