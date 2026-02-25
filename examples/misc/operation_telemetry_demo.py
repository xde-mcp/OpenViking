#!/usr/bin/env python3
# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
PANEL_WIDTH = 120
AsyncOpenViking = None
OpenVikingConfigSingleton = None


@dataclass
class TelemetryOperationResult:
    label: str
    operation: str
    telemetry: dict[str, Any]


@dataclass
class TelemetryDemoResult:
    sample_path: str
    root_uri: str
    skill_uri: str
    session_id: str
    search_total: int
    find_total: int
    operations: list[TelemetryOperationResult]


def build_sample_markdown(base_dir: Path) -> Path:
    sample_path = base_dir / "operation-telemetry-demo.md"
    sample_path.write_text(
        "\n".join(
            [
                "# Operation Telemetry Demo",
                "",
                "This sample document is used to validate operation-level telemetry metrics.",
                "It contains the phrases sample document and telemetry metrics for retrieval.",
                "The latest commit adds operation telemetry summaries and event streams.",
            ]
        ),
        encoding="utf-8",
    )
    return sample_path


def build_sample_skill_dir(base_dir: Path) -> Path:
    skill_dir = base_dir / "telemetry-demo-skill"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: telemetry-demo-skill
description: A sample skill used to validate telemetry-enabled skill ingestion.
tags:
  - telemetry
  - demo
---

# Telemetry Demo Skill

## Instructions
Use this skill when validating telemetry-enabled skill ingestion in the operation demo.
""",
        encoding="utf-8",
    )
    return skill_dir


def render_telemetry_summary(title: str, telemetry: dict[str, Any]) -> Panel:
    summary = telemetry.get("summary", {})
    tokens = summary.get("tokens", {})
    llm_usage = tokens.get("llm", {})
    embedding_usage = tokens.get("embedding", {})
    queue = summary.get("queue", {})
    vector = summary.get("vector", {})
    semantic = summary.get("semantic_nodes", {})
    memory = summary.get("memory", {})
    errors = summary.get("errors", {})

    table = Table(box=box.SIMPLE_HEAVY, show_header=False, expand=True)
    table.add_column("Metric", style="cyan", ratio=1)
    table.add_column("Value", style="white", ratio=2)
    table.add_row("id", str(telemetry.get("id", "")))
    table.add_row("operation", str(summary.get("operation", "")))
    table.add_row("status", str(summary.get("status", "")))
    table.add_row("duration_ms", str(summary.get("duration_ms", "")))
    table.add_row(
        "tokens",
        f"total={tokens.get('total', 0)}",
    )
    table.add_row(
        "tokens.llm",
        (
            f"in={llm_usage.get('input', 0)}  "
            f"out={llm_usage.get('output', 0)}  "
            f"total={llm_usage.get('total', 0)}"
        ),
    )
    table.add_row(
        "tokens.embedding",
        f"total={embedding_usage.get('total', 0)}",
    )

    if "queue" in summary:
        table.add_row(
            "queue",
            (
                f"semantic=processed:{queue.get('semantic', {}).get('processed', 0)} "
                f"errors:{queue.get('semantic', {}).get('error_count', 0)}  "
                f"embedding=processed:{queue.get('embedding', {}).get('processed', 0)} "
                f"errors:{queue.get('embedding', {}).get('error_count', 0)}"
            ),
        )

    if "vector" in summary:
        table.add_row(
            "vector",
            (
                f"searches={vector.get('searches', 0)}  "
                f"scored={vector.get('scored', 0)}  "
                f"passed={vector.get('passed', 0)}  "
                f"returned={vector.get('returned', 0)}"
            ),
        )

    if "semantic_nodes" in summary:
        table.add_row(
            "semantic_nodes",
            (
                f"total={semantic.get('total')}  "
                f"done={semantic.get('done')}  "
                f"pending={semantic.get('pending')}  "
                f"running={semantic.get('running')}"
            ),
        )

    if "memory" in summary:
        table.add_row("memory.extracted", str(memory.get("extracted")))

    if "errors" in summary:
        table.add_row(
            "errors",
            (
                f"stage={errors.get('stage', '')}  "
                f"code={errors.get('error_code', '')}  "
                f"message={errors.get('message', '')}"
            ).strip(),
        )

    table.add_row("dropped", str(telemetry.get("dropped", 0)))

    return Panel(table, title=title, border_style="bright_blue", width=PANEL_WIDTH)


def render_telemetry_events(telemetry: dict[str, Any], max_events: int = 8) -> Panel:
    events = telemetry.get("events", [])
    table = Table(box=box.MINIMAL_DOUBLE_HEAD, expand=True)
    table.add_column("t(ms)", style="cyan", justify="right", width=10, no_wrap=True)
    table.add_column("stage / event", style="magenta", width=32, overflow="fold")
    table.add_column("details", style="white", overflow="fold")

    for event in events[:max_events]:
        attrs = json.dumps(
            event.get("attrs", {}),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        table.add_row(
            str(event.get("ts_ms", "")),
            f"{event.get('stage', '')}\n{event.get('name', '')}",
            f"status={event.get('status', '')}\nattrs={attrs}",
        )

    if not events:
        table.add_row("-", "no events", "-")

    return Panel(table, title="Telemetry Events", border_style="bright_black", width=PANEL_WIDTH)


def _load_runtime_deps() -> tuple[Any, Any]:
    import sys

    global AsyncOpenViking
    global OpenVikingConfigSingleton

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    if AsyncOpenViking is None:
        from openviking import AsyncOpenViking as runtime_client

        AsyncOpenViking = runtime_client

    if OpenVikingConfigSingleton is None:
        from openviking_cli.utils.config.open_viking_config import (
            OpenVikingConfigSingleton as runtime_config_singleton,
        )

        OpenVikingConfigSingleton = runtime_config_singleton

    return AsyncOpenViking, OpenVikingConfigSingleton


async def run_demo_workflow(base_dir: Path) -> TelemetryDemoResult:
    runtime_client, runtime_config_singleton = _load_runtime_deps()
    work_dir = Path(base_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    sample_path = build_sample_markdown(work_dir)
    skill_dir = build_sample_skill_dir(work_dir)

    await runtime_client.reset()
    client = runtime_client(path=str(work_dir / "data"))

    try:
        await client.initialize()

        resource_result = await client.add_resource(
            path=str(sample_path),
            reason="operation telemetry demo",
            wait=True,
            telemetry=True,
        )
        skill_result = await client.add_skill(
            data=skill_dir,
            wait=True,
            telemetry=True,
        )
        session_result = await client.create_session()
        session_id = session_result["session_id"]
        await client.add_message(
            session_id,
            "user",
            "Find the telemetry demo resource and explain the skill coverage.",
        )
        search_result = await client.search(
            query="telemetry demo skill coverage",
            target_uri=resource_result["root_uri"],
            session_id=session_id,
            limit=3,
            telemetry=True,
        )
        find_result = await client.find(
            query="sample document telemetry metrics",
            target_uri=resource_result["root_uri"],
            limit=3,
            telemetry=True,
        )
        commit_result = await client.commit_session(session_id, telemetry=True)
        return TelemetryDemoResult(
            sample_path=str(sample_path),
            root_uri=resource_result["root_uri"],
            skill_uri=skill_result["uri"],
            session_id=session_id,
            search_total=getattr(search_result, "total", 0),
            find_total=getattr(find_result, "total", 0),
            operations=[
                TelemetryOperationResult(
                    label="Add Resource",
                    operation="resources.add_resource",
                    telemetry=resource_result["telemetry"],
                ),
                TelemetryOperationResult(
                    label="Add Skill",
                    operation="resources.add_skill",
                    telemetry=skill_result["telemetry"],
                ),
                TelemetryOperationResult(
                    label="Search",
                    operation="search.search",
                    telemetry=search_result.telemetry or {},
                ),
                TelemetryOperationResult(
                    label="Find",
                    operation="search.find",
                    telemetry=find_result.telemetry or {},
                ),
                TelemetryOperationResult(
                    label="Commit Session",
                    operation="sessions.commit",
                    telemetry=commit_result["telemetry"],
                ),
            ],
        )
    finally:
        await client.close()
        await runtime_client.reset()
        runtime_config_singleton.reset_instance()


def render_demo_report(output_console: Console, result: TelemetryDemoResult) -> None:
    overview = Table.grid(expand=True)
    overview.add_column(style="cyan", ratio=1)
    overview.add_column(style="white", ratio=3)
    overview.add_row("sample_path", result.sample_path)
    overview.add_row("root_uri", result.root_uri)
    overview.add_row("skill_uri", result.skill_uri)
    overview.add_row("session_id", result.session_id)
    overview.add_row("search_total", str(result.search_total))
    overview.add_row("find_total", str(result.find_total))
    overview.add_row("operation_count", str(len(result.operations)))

    output_console.print(
        Panel(
            overview,
            title="Operation Telemetry Demo",
            subtitle="Latest commit: operation telemetry metrics and API support",
            border_style="bold green",
            width=PANEL_WIDTH,
        )
    )
    for operation in result.operations:
        output_console.print(
            render_telemetry_summary(f"{operation.label} Telemetry", operation.telemetry)
        )
        output_console.print(
            render_telemetry_events(
                operation.telemetry,
                max_events=None if operation.operation == "resources.add_resource" else 8,
            )
        )


async def async_main(workdir: str | None = None) -> int:
    if workdir:
        base_dir = Path(workdir)
    else:
        base_dir = Path(tempfile.mkdtemp(prefix="ov-telemetry-demo-"))

    result = await run_demo_workflow(base_dir)
    render_demo_report(console, result)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a terminal demo for operation telemetry output."
    )
    parser.add_argument(
        "--workdir",
        default=None,
        help="Optional directory used for temporary demo data.",
    )
    args = parser.parse_args()
    return asyncio.run(async_main(args.workdir))


if __name__ == "__main__":
    raise SystemExit(main())
