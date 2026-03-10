# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""
Context Processor for OpenViking.

Handles coordinated writes and self-iteration processes
as described in the OpenViking design document.
"""

import time
import traceback
import urllib.request
import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from openviking.parse.tree_builder import TreeBuilder
from openviking.server.identity import RequestContext
from openviking.storage import VikingDBManager
from openviking.storage.viking_fs import get_viking_fs
from openviking.utils.embedding_utils import index_resource
from openviking.utils.summarizer import Summarizer
from openviking_cli.utils import get_logger
from openviking_cli.utils.storage import StoragePath
from openviking.resource.update_context import UpdateContext
from openviking.resource.resource_lock import (
    LockInfo,
    ResourceLockConflictError,
    ResourceLockManager,
)

if TYPE_CHECKING:
    from openviking.parse.vlm import VLMProcessor

logger = get_logger(__name__)

#region debug-point
def _debug_log(event: str, data: Dict[str, Any]) -> None:
    """Report debug event to Debug Server."""
    try:
        payload = json.dumps({"event": event, "data": data}).encode('utf-8')
        req = urllib.request.Request(
            "http://127.0.0.1:9527/event",
            data=payload,
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        urllib.request.urlopen(req, timeout=1)
    except Exception:
        pass
#endregion


class ResourceProcessor:
    """
    Handles coordinated write operations.

    When new data is added, automatically:
    1. Download if URL (prefer PDF format)
    2. Parse and structure the content (Parser writes to temp directory)
    3. Extract images/tables for mixed content
    4. Use VLM to understand non-text content
    5. TreeBuilder finalizes from temp (move to AGFS)
    6. SemanticQueue generates L0/L1 and vectorizes asynchronously
    """

    def __init__(
        self,
        vikingdb: VikingDBManager,
        media_storage: Optional["StoragePath"] = None,
        max_context_size: int = 2000,
        max_split_depth: int = 3,
        lock_manager: Optional[ResourceLockManager] = None,
    ):
        """Initialize coordinated writer."""
        self.vikingdb = vikingdb
        self.embedder = vikingdb.get_embedder()
        self.media_storage = media_storage
        self.tree_builder = TreeBuilder()
        self._vlm_processor = None
        self._media_processor = None
        self._summarizer = None
        self._lock_manager = lock_manager

    def _get_summarizer(self) -> "Summarizer":
        """Lazy initialization of Summarizer."""
        if self._summarizer is None:
            self._summarizer = Summarizer(self._get_vlm_processor())
        return self._summarizer

    def _get_vlm_processor(self) -> "VLMProcessor":
        """Lazy initialization of VLM processor."""
        if self._vlm_processor is None:
            from openviking.parse.vlm import VLMProcessor

            self._vlm_processor = VLMProcessor()
        return self._vlm_processor

    def _get_media_processor(self):
        """Lazy initialization of unified media processor."""
        if self._media_processor is None:
            from openviking.utils.media_processor import UnifiedResourceProcessor

            self._media_processor = UnifiedResourceProcessor(
                vlm_processor=self._get_vlm_processor(),
                storage=self.media_storage,
            )
        return self._media_processor

    async def build_index(self, resource_uris: List[str], ctx: RequestContext, **kwargs) -> Dict[str, Any]:
        """Expose index building as a standalone method."""
        for uri in resource_uris:
            await index_resource(uri, ctx)
        return {"status": "success", "message": f"Indexed {len(resource_uris)} resources"}

    async def summarize(self, resource_uris: List[str], ctx: RequestContext, **kwargs) -> Dict[str, Any]:
        """Expose summarization as a standalone method."""
        return await self._get_summarizer().summarize(resource_uris, ctx, **kwargs)
    async def process_resource(
        self,
        path: str,
        ctx: RequestContext,
        reason: str = "",
        instruction: str = "",
        scope: str = "resources",
        user: Optional[str] = None,
        target: Optional[str] = None,
        build_index: bool = True,
        summarize: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Process and store a new resource.

        Workflow:
        1. Parse source (writes to temp directory)
        2. TreeBuilder moves to AGFS
        3. (Optional) Build vector index
        4. (Optional) Summarize
        """
        #region debug-point
        start_time = time.time()
        _debug_log("process_resource_start", {
            "path": path,
            "target": target,
            "build_index": build_index,
            "summarize": summarize,
        })
        #endregion
        
        result = {
            "status": "success",
            "errors": [],
            "source_path": None,
        }

        if target:
            if not target.startswith("viking://"):
                target = f"viking://resources/{target}"
            logger.info(f"Using target location: {target}")
        update_ctx = UpdateContext(source_url=path, target_uri=target, request_context=ctx)
        update_ctx.scope = scope

         # 先获得锁资源,判断是否可以进行写操作,以及是否是增量更新模式
        try:
            #region debug-point
            lock_start = time.time()
            #endregion
            await self._acquire_lock(update_ctx)
            #region debug-point
            lock_duration = time.time() - lock_start
            _debug_log("acquire_lock_complete", {
                "duration_ms": round(lock_duration * 1000, 2),
                "target": target,
            })
            #endregion
        except ResourceLockConflictError as e:
            self._release_lock(update_ctx)
            logger.warning(f"Resource lock conflict: {e}")
            result["status"] = "error"
            result["errors"].append(f"Resource lock conflict: {e}")
            return result
        viking_fs = get_viking_fs()
        
        with viking_fs.bind_request_context(ctx):
            update_ctx.is_incremental = await viking_fs.exists(target, ctx=ctx)
            logger.info(
                f"Resource exists: {update_ctx.is_incremental}, target: {target}"
            )

        # ============ Phase 1: Parse source and writes to temp viking fs ============
        try:
            #region debug-point
            parse_start = time.time()
            _debug_log("parse_start", {"path": path})
            #endregion
            media_processor = self._get_media_processor()
            # Use reason as instruction fallback so it influences L0/L1
            # generation and improves search relevance as documented.
            effective_instruction = instruction or reason
            with viking_fs.bind_request_context(ctx):
                parse_result = await media_processor.process(
                    source=path,
                    instruction=effective_instruction,
                    **kwargs,
                )
            #region debug-point
            parse_duration = time.time() - parse_start
            _debug_log("parse_complete", {
                "duration_ms": round(parse_duration * 1000, 2),
                "temp_dir": parse_result.temp_dir_path,
                "source_format": parse_result.source_format,
            })
            #endregion
            result["source_path"] = parse_result.source_path or path
            result["meta"] = parse_result.meta

            # Only abort when no temp content was produced at all.
            # For directory imports partial success (some files failed) is
            # normal – finalization should still proceed.
            if not parse_result.temp_dir_path:
                result["status"] = "error"
                result["errors"].extend(
                    parse_result.warnings or ["Parse failed: no content generated"],
                )
                return result

            if parse_result.warnings:
                result["errors"].extend(parse_result.warnings)
            update_ctx.temp_vikingfs_path = parse_result.temp_dir_path
            update_ctx.source_format = parse_result.source_format
            
            # Pass document name from parser to TreeBuilder (avoid duplicate logic)
            if "repo_name" in parse_result.meta:
                update_ctx.document_name = parse_result.meta["repo_name"]

        except Exception as e:
            self._release_lock(update_ctx)
            logger.error(f"[ResourceProcessor] Parse error: {e}")
            result["status"] = "error"
            result["errors"].append(f"Parse error: {e}")
            traceback.print_exc()
            return result

        # parse_result contains:
        # - root: ResourceNode tree (with L0/L1 in meta)
        # - temp_vikingfs_path: Temporary directory path (Parser wrote all files)
        # - source_format: File format (e.g., "pdf", "markdown")

        # ============ Phase 2: TreeBuilder finalizes from temp (scan + move to AGFS) ============
        try:
            #region debug-point
            finalize_start = time.time()
            _debug_log("finalize_start", {"temp_dir": parse_result.temp_dir_path})
            #endregion
            with get_viking_fs().bind_request_context(ctx):
                context_tree = await self.tree_builder.finalize_from_temp(
                    update_ctx=update_ctx,
                )
                #region debug-point
                finalize_duration = time.time() - finalize_start
                _debug_log("finalize_complete", {
                    "duration_ms": round(finalize_duration * 1000, 2),
                    "root_uri": context_tree.root.uri if context_tree and context_tree.root else None,
                })
                #endregion
                if context_tree and context_tree.root:
                    result["root_uri"] = context_tree.root.uri
        except Exception as e:
            result["status"] = "error"
            result["errors"].append(f"Finalize from temp error: {e}")

            # Cleanup temporary directory on error (via VikingFS)
            try:
                if parse_result.temp_dir_path:
                    await get_viking_fs().delete_temp(parse_result.temp_dir_path, ctx=ctx)
            except Exception:
                pass

            return result

        # ============ Phase 4: Optional Steps ============
        lock_resource_uri = update_ctx.target_uri
        lock_id = update_ctx.lock_info.lock_id if update_ctx.lock_info else ""
        
        if summarize:
             # Explicit summarization request.
             # If build_index is ALSO True, we want vectorization.
             # If build_index is False, we skip vectorization.
             skip_vec = not build_index
             try:
                #region debug-point
                summarize_start = time.time()
                _debug_log("summarize_start", {"root_uri": result.get("root_uri")})
                #endregion
                await self._get_summarizer().summarize(
                    resource_uris=[result["root_uri"]],
                    ctx=ctx,
                    skip_vectorization=skip_vec,
                    lock_resource_uri=lock_resource_uri,
                    lock_id=lock_id,
                    **kwargs
                )
                #region debug-point
                summarize_duration = time.time() - summarize_start
                _debug_log("summarize_complete", {
                    "duration_ms": round(summarize_duration * 1000, 2),
                })
                #endregion
             except Exception as e:
                logger.error(f"Summarization failed: {e}")
                result["warnings"] = result.get("warnings", []) + [f"Summarization failed: {e}"]

        elif build_index:
             # Standard compatibility mode: "Just Index it" usually implies ingestion flow.
             # We assume this means "Ingest and Index", which requires summarization.
             try:
                #region debug-point
                auto_index_start = time.time()
                _debug_log("auto_index_start", {"root_uri": result.get("root_uri")})
                #endregion
                await self._get_summarizer().summarize(
                    resource_uris=[result["root_uri"]],
                    ctx=ctx,
                    skip_vectorization=False,
                    lock_resource_uri=lock_resource_uri,
                    lock_id=lock_id,
                    **kwargs
                )
                #region debug-point
                auto_index_duration = time.time() - auto_index_start
                _debug_log("auto_index_complete", {
                    "duration_ms": round(auto_index_duration * 1000, 2),
                })
                #endregion
             except Exception as e:
                logger.error(f"Auto-index failed: {e}")
                result["warnings"] = result.get("warnings", []) + [f"Auto-index failed: {e}"]

        #region debug-point
        total_duration = time.time() - start_time
        _debug_log("process_resource_complete", {
            "total_duration_ms": round(total_duration * 1000, 2),
            "status": result["status"],
            "root_uri": result.get("root_uri"),
        })
        #endregion
        
        return result


    def _log_step(self, step: str, ctx: UpdateContext, **kwargs) -> None:
            """Log a step with structured context."""
            logger.info(
                f"[ResourceProcessor] Step: {step}, "
                f"source_url={ctx.source_url}, "
                + ", ".join(f"{k}={v}" for k, v in kwargs.items())
            )

    async def _acquire_lock(
        self,
        ctx: UpdateContext,
        ttl: int = 3600,
    ) -> LockInfo:
        """Acquire resource lock."""
        self._log_step("acquire_lock", ctx)
        
        lock_info = self._lock_manager.acquire_lock(
            resource_uri=ctx.target_uri,
            operation="incremental_update" if ctx.is_incremental else "full_update",
            ttl=ttl,
        )
        
        ctx.lock_info = lock_info
        self._log_step("lock_acquired", ctx, lock_id=lock_info.lock_id)
        
        return lock_info

    def _release_lock(self, ctx: UpdateContext) -> bool:
        """Release resource lock."""
        if not ctx.lock_info:
            return True
        
        self._log_step("release_lock", ctx, lock_id=ctx.lock_info.lock_id)
        
        return self._lock_manager.release_lock(
            resource_uri=ctx.resource_uri,
            lock_id=ctx.lock_info.lock_id,
        )