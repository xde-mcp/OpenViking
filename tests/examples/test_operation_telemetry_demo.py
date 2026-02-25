# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from rich.console import Console

from openviking import AsyncOpenViking


def load_demo_module():
    module_path = Path(__file__).resolve().parents[2] / "examples" / "operation_telemetry_demo.py"
    if not module_path.exists():
        pytest.fail(f"Demo module is missing: {module_path}")

    spec = importlib.util.spec_from_file_location("operation_telemetry_demo", module_path)
    if spec is None or spec.loader is None:
        pytest.fail(f"Unable to load demo module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sample_telemetry() -> dict:
    return {
        "id": "demo_telemetry",
        "truncated": False,
        "dropped": 0,
        "summary": {
            "operation": "search.find",
            "status": "ok",
            "duration_ms": 12.3,
            "tokens": {
                "total": 8,
                "llm": {
                    "input": 6,
                    "output": 0,
                    "total": 6,
                },
                "embedding": {
                    "total": 2,
                },
            },
            "queue": {
                "semantic": {
                    "processed": 1,
                    "error_count": 0,
                },
                "embedding": {
                    "processed": 2,
                    "error_count": 0,
                },
            },
            "vector": {
                "searches": 2,
                "scored": 6,
                "passed": 3,
                "returned": 2,
                "scanned": 6,
                "scan_reason": "",
            },
            "semantic_nodes": {
                "total": None,
                "done": None,
                "pending": None,
                "running": None,
            },
            "memory": {
                "extracted": None,
            },
            "errors": {
                "stage": "",
                "error_code": "",
                "message": "",
            },
        },
        "events": [
            {
                "stage": "retriever.global_search",
                "name": "global_search_done",
                "ts_ms": 8.512,
                "status": "ok",
                "attrs": {"hits": 3},
            }
        ],
    }


def sample_operation_telemetry(operation: str) -> dict:
    return sample_telemetry() | {
        "summary": sample_telemetry()["summary"] | {"operation": operation}
    }


def test_render_telemetry_panels_include_operation_and_event_name():
    demo = load_demo_module()

    console = Console(record=True, width=120)
    console.print(demo.render_telemetry_summary("Search Telemetry", sample_telemetry()))
    console.print(demo.render_telemetry_events(sample_telemetry(), max_events=5))
    rendered = console.export_text()

    assert "Search Telemetry" in rendered
    assert "search.find" in rendered
    assert "tokens.llm" in rendered
    assert "tokens.embedding" in rendered
    assert "total=2" in rendered
    assert "queue" in rendered
    assert "embedding=processed:2" in rendered
    assert "global_search_done" in rendered


def test_render_telemetry_summary_hides_missing_optional_groups():
    demo = load_demo_module()
    telemetry = sample_telemetry() | {
        "summary": {
            "operation": "session.commit",
            "status": "ok",
            "duration_ms": 9.8,
            "tokens": {
                "total": 8,
                "llm": {"input": 8, "output": 0, "total": 8},
                "embedding": {"total": 0},
            },
            "memory": {"extracted": 2},
        }
    }

    console = Console(record=True, width=120)
    console.print(demo.render_telemetry_summary("Commit Telemetry", telemetry))
    rendered = console.export_text()

    assert "Commit Telemetry" in rendered
    assert "memory.extracted" in rendered
    assert "semantic_nodes" not in rendered
    assert "\nqueue" not in rendered
    assert "\nvector" not in rendered
    assert "\nerrors" not in rendered


def test_render_telemetry_events_shows_full_attrs_without_ellipsis():
    demo = load_demo_module()
    telemetry = sample_telemetry() | {
        "events": [
            {
                "stage": "resource_service.add_resource",
                "name": "queue_status_collected",
                "ts_ms": 18.2,
                "status": "ok",
                "attrs": {
                    "semantic_processed": 12,
                    "embedding_processed": 34,
                    "root_uri": "viking://resources/demo/really/long/path/operation-telemetry-demo",
                },
            }
        ]
    }

    console = Console(record=True, width=120)
    console.print(demo.render_telemetry_events(telemetry, max_events=5))
    rendered = console.export_text()
    assert "queue_status_collected" in rendered
    assert "semantic_processed" in rendered
    assert "embedding_processed" in rendered
    assert "viking://resources/demo/really/long/path/operation-telemetry-de" in rendered
    assert 'mo",' in rendered
    assert "…" not in rendered


def test_render_telemetry_events_without_limit_shows_late_done_event():
    demo = load_demo_module()
    telemetry = sample_telemetry() | {
        "events": [
            {
                "stage": "resource_service.add_resource",
                "name": f"step_{idx}",
                "ts_ms": float(idx),
                "status": "ok",
                "attrs": {},
            }
            for idx in range(9)
        ]
        + [
            {
                "stage": "resource_service.add_resource",
                "name": "done",
                "ts_ms": 99.0,
                "status": "ok",
                "attrs": {"total_duration_ms": 99.0},
            }
        ]
    }

    console = Console(record=True, width=120)
    console.print(demo.render_telemetry_events(telemetry, max_events=None))
    rendered = console.export_text()

    assert "step_8" in rendered
    assert "done" in rendered
    assert "total_duration_ms" in rendered


@pytest.mark.asyncio
async def test_run_demo_workflow_collects_operation_telemetry(tmp_path: Path):
    demo = load_demo_module()
    await AsyncOpenViking.reset()

    def fail_offline_runtime():
        raise AssertionError("run_demo_workflow should not force offline demo runtime")

    class FakeFindResult:
        def __init__(self):
            self.total = 1
            self.telemetry = sample_operation_telemetry("search.find")

    class FakeSearchResult:
        def __init__(self):
            self.total = 2
            self.telemetry = sample_operation_telemetry("search.search")

    class FakeClient:
        def __init__(self, path: str):
            self.path = path

        async def initialize(self):
            return None

        async def add_resource(self, **kwargs):
            return {
                "root_uri": "viking://resources/operation-telemetry-demo",
                "telemetry": sample_operation_telemetry("resources.add_resource"),
            }

        async def add_skill(self, **kwargs):
            data = kwargs["data"]
            assert isinstance(data, Path)
            assert data.is_dir()
            assert (data / "SKILL.md").exists()
            return {
                "uri": "viking://agent/skills/demo-telemetry-skill",
                "telemetry": sample_operation_telemetry("resources.add_skill"),
            }

        async def create_session(self):
            return {"session_id": "sess_demo_telemetry"}

        async def add_message(self, *args, **kwargs):
            return {"message_count": 1}

        async def search(self, **kwargs):
            return FakeSearchResult()

        async def find(self, **kwargs):
            return FakeFindResult()

        async def commit_session(self, session_id: str, telemetry: bool = False):
            return {
                "session_id": session_id,
                "telemetry": sample_operation_telemetry("sessions.commit"),
            }

        async def close(self):
            return None

    async def fake_reset():
        return None

    demo.offline_demo_runtime = fail_offline_runtime
    demo.AsyncOpenViking = FakeClient
    demo.AsyncOpenViking.reset = staticmethod(fake_reset)

    result = await demo.run_demo_workflow(tmp_path)

    assert result.root_uri.startswith("viking://")
    assert result.skill_uri.startswith("viking://")
    assert result.session_id == "sess_demo_telemetry"
    assert result.root_uri.startswith("viking://")
    assert [item.label for item in result.operations] == [
        "Add Resource",
        "Add Skill",
        "Search",
        "Find",
        "Commit Session",
    ]
    assert [item.operation for item in result.operations] == [
        "resources.add_resource",
        "resources.add_skill",
        "search.search",
        "search.find",
        "sessions.commit",
    ]
    assert result.find_total == 1
    assert result.search_total == 2
    assert all(item.telemetry["events"] for item in result.operations)


def test_render_demo_report_contains_sections():
    demo = load_demo_module()

    result = demo.TelemetryDemoResult(
        sample_path="sample.md",
        root_uri="viking://resources/demo/sample.md",
        skill_uri="viking://agent/skills/demo/sample-skill",
        session_id="sess_demo_sample",
        search_total=3,
        find_total=2,
        operations=[
            demo.TelemetryOperationResult(
                label="Add Resource",
                operation="resources.add_resource",
                telemetry=sample_operation_telemetry("resources.add_resource"),
            ),
            demo.TelemetryOperationResult(
                label="Add Skill",
                operation="resources.add_skill",
                telemetry=sample_operation_telemetry("resources.add_skill"),
            ),
            demo.TelemetryOperationResult(
                label="Search",
                operation="search.search",
                telemetry=sample_operation_telemetry("search.search"),
            ),
            demo.TelemetryOperationResult(
                label="Find",
                operation="search.find",
                telemetry=sample_operation_telemetry("search.find"),
            ),
            demo.TelemetryOperationResult(
                label="Commit Session",
                operation="sessions.commit",
                telemetry=sample_operation_telemetry("sessions.commit"),
            ),
        ],
    )

    console = Console(record=True, width=120)
    demo.render_demo_report(console, result)
    rendered = console.export_text()

    assert "Operation Telemetry Demo" in rendered
    assert "Add Resource Telemetry" in rendered
    assert "Add Skill Telemetry" in rendered
    assert "Search Telemetry" in rendered
    assert "Find Telemetry" in rendered
    assert "Commit Session Telemetry" in rendered
    assert "viking://resources/demo/sample.md" in rendered
    assert "viking://agent/skills/demo/sample-skill" in rendered
    assert "sess_demo_sample" in rendered
