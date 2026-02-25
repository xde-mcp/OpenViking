# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Hierarchical retriever telemetry hooks."""

from __future__ import annotations

from .context import get_current_telemetry


def on_retrieve_start(*, context_type: str, limit: int) -> None:
    get_current_telemetry().event(
        "retriever.retrieve",
        "retrieve_start",
        {"context_type": context_type, "limit": limit},
    )


def on_collection_missing(*, collection: str) -> None:
    get_current_telemetry().event(
        "retriever.retrieve",
        "collection_missing",
        {"collection": collection},
    )


def on_global_search_done(*, hits: int) -> None:
    get_current_telemetry().event(
        "retriever.global_search",
        "global_search_done",
        {"hits": hits},
    )


def on_starting_points_merged(*, count: int) -> None:
    get_current_telemetry().event(
        "retriever.starting_points",
        "starting_points_merged",
        {"count": count},
    )


def on_recursive_search_done(*, candidates: int) -> None:
    get_current_telemetry().event(
        "retriever.recursive",
        "recursive_search_done",
        {"candidates": candidates},
    )


def on_retrieve_done(*, matched_contexts: int) -> None:
    get_current_telemetry().event(
        "retriever.retrieve",
        "retrieve_done",
        {"matched_contexts": matched_contexts},
    )


def count_vector_search(*, scored: int, scanned: int | None = None) -> None:
    telemetry = get_current_telemetry()
    telemetry.count("vector.searches", 1)
    telemetry.count("vector.scored", scored)
    telemetry.count("vector.scanned", scored if scanned is None else scanned)


def on_directory_entered(*, uri: str, queue_size: int) -> None:
    get_current_telemetry().event(
        "retriever.directory",
        "directory_entered",
        {"uri": uri, "queue_size": queue_size},
    )


def on_directory_results(*, uri: str, hits: int) -> None:
    get_current_telemetry().event(
        "retriever.directory",
        "directory_results",
        {"uri": uri, "hits": hits},
    )


def count_vector_passed(count: int = 1) -> None:
    get_current_telemetry().count("vector.passed", count)
