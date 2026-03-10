# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""SemanticProcessor: Processes messages from SemanticQueue, generates .abstract.md and .overview.md."""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple

from openviking.parse.parsers.constants import (
    CODE_EXTENSIONS,
    DOCUMENTATION_EXTENSIONS,
    FILE_TYPE_CODE,
    FILE_TYPE_DOCUMENTATION,
    FILE_TYPE_OTHER,
)
from openviking.parse.parsers.media.utils import (
    generate_audio_summary,
    generate_image_summary,
    generate_video_summary,
    get_media_type,
)
from openviking.prompts import render_prompt
from openviking.server.identity import RequestContext, Role
from openviking.storage.queuefs.named_queue import DequeueHandlerBase
from openviking.storage.queuefs.semantic_dag import DagStats, SemanticDagExecutor
from openviking.storage.queuefs.semantic_msg import SemanticMsg
from openviking.storage.viking_fs import get_viking_fs
from openviking_cli.session.user_id import UserIdentifier
from openviking_cli.utils import VikingURI
from openviking_cli.utils.config import get_openviking_config
from openviking_cli.utils.logger import get_logger


from .embedding_tracker import EmbeddingTaskTracker

logger = get_logger(__name__)


@dataclass
class DiffResult:
    """Directory diff result for sync operations."""
    added_files: List[str] = field(default_factory=list)
    deleted_files: List[str] = field(default_factory=list)
    updated_files: List[str] = field(default_factory=list)
    added_dirs: List[str] = field(default_factory=list)
    deleted_dirs: List[str] = field(default_factory=list)


class SemanticProcessor(DequeueHandlerBase):
    """
    Semantic processor, generates .abstract.md and .overview.md bottom-up.

    Processing flow:
    1. Concurrently generate summaries for files in directory
    2. Collect .abstract.md from subdirectories
    3. Generate .abstract.md and .overview.md for this directory
    4. Enqueue to EmbeddingQueue for vectorization
    """

    def __init__(self, max_concurrent_llm: int = 100):
        """
        Initialize SemanticProcessor.

        Args:
            max_concurrent_llm: Maximum concurrent LLM calls
        """
        self.max_concurrent_llm = max_concurrent_llm
        self._dag_executor: Optional[SemanticDagExecutor] = None
        self._current_ctx = RequestContext(user=UserIdentifier.the_default_user(), role=Role.ROOT)
        self._current_msg: Optional[SemanticMsg] = None

    @staticmethod
    def _owner_space_for_uri(uri: str, ctx: RequestContext) -> str:
        """Derive owner_space from a URI.

        Resources (viking://resources/...) always get owner_space="" so they
        are globally visible.  User / agent / session URIs inherit the
        caller's space name.
        """
        if uri.startswith("viking://agent/"):
            return ctx.user.agent_space_name()
        if uri.startswith("viking://user/") or uri.startswith("viking://session/"):
            return ctx.user.user_space_name()
        # resources and anything else → shared (empty owner_space)
        return ""

    @staticmethod
    def _ctx_from_semantic_msg(msg: SemanticMsg) -> RequestContext:
        role = Role(msg.role) if msg.role in {r.value for r in Role} else Role.ROOT
        return RequestContext(
            user=UserIdentifier(msg.account_id, msg.user_id, msg.agent_id),
            role=role,
        )

    def _detect_file_type(self, file_name: str) -> str:
        """
        Detect file type based on extension using constants from code parser.

        Args:
            file_name: File name with extension

        Returns:
            FILE_TYPE_CODE, FILE_TYPE_DOCUMENTATION, or FILE_TYPE_OTHER
        """
        file_name_lower = file_name.lower()

        # Check if file is a code file
        for ext in CODE_EXTENSIONS:
            if file_name_lower.endswith(ext):
                return FILE_TYPE_CODE

        # Check if file is a documentation file
        for ext in DOCUMENTATION_EXTENSIONS:
            if file_name_lower.endswith(ext):
                return FILE_TYPE_DOCUMENTATION

        # Default to other
        return FILE_TYPE_OTHER

    async def _enqueue_semantic_msg(self, msg: SemanticMsg) -> None:
        """Enqueue a SemanticMsg to the semantic queue for processing."""
        from openviking.storage.queuefs import get_queue_manager

        queue_manager = get_queue_manager()
        semantic_queue = queue_manager.get_queue(queue_manager.SEMANTIC)
        # The queue manager returns SemanticQueue but method signature says NamedQueue
        # We need to ignore the type error for the enqueue call
        await semantic_queue.enqueue(msg)  # type: ignore
        logger.debug(f"Enqueued semantic message for processing: {msg.uri}")

    async def _collect_directory_info(
        self,
        uri: str,
        result: List[Tuple[str, List[str], List[str]]],
    ) -> None:
        """Recursively collect directory info, post-order traversal ensures bottom-up order."""
        viking_fs = get_viking_fs()

        try:
            entries = await viking_fs.ls(uri, ctx=self._current_ctx)
        except Exception as e:
            logger.warning(f"Failed to list directory {uri}: {e}")
            return

        children_uris = []
        file_paths = []

        for entry in entries:
            name = entry.get("name", "")
            if not name or name.startswith(".") or name in [".", ".."]:
                continue

            item_uri = VikingURI(uri).join(name).uri

            if entry.get("isDir", False):
                # Child directory
                children_uris.append(item_uri)
                # Recursively collect children
                await self._collect_directory_info(item_uri, result)
            else:
                # File (not starting with .)
                file_paths.append(item_uri)

        # Add current directory info
        result.append((uri, children_uris, file_paths))

    async def on_dequeue(self, data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Process dequeued SemanticMsg, recursively process all subdirectories."""
        try:
            import json

            if not data:
                return None

            if "data" in data and isinstance(data["data"], str):
                data = json.loads(data["data"])

            # data is guaranteed to be not None at this point
            assert data is not None
            msg = SemanticMsg.from_dict(data)
            self._current_msg = msg
            self._current_ctx = self._ctx_from_semantic_msg(msg)
            logger.info(
                f"Processing semantic generation for: {msg})"
            )

            if msg.recursive:
                executor = SemanticDagExecutor(
                    processor=self,
                    context_type=msg.context_type,
                    max_concurrent_llm=self.max_concurrent_llm,
                    ctx=self._current_ctx,
                    incremental_update=msg.is_incremental_update,
                    target_uri=msg.target_uri if msg.target_uri else None,
                    semantic_msg_id=msg.id,
                    lock_resource_uri=msg.lock_resource_uri,
                    lock_id=msg.lock_id,
                )
                self._dag_executor = executor
                await executor.run(msg.uri)
                logger.info(f"Completed semantic generation for: {msg.uri}")
                self.report_success()
                return None
            else:
                children_uris = []
                file_paths = []

                viking_fs = get_viking_fs()
                try:
                    entries = await viking_fs.ls(msg.uri, ctx=self._current_ctx)
                    for entry in entries:
                        name = entry.get("name", "")
                        if not name or name.startswith(".") or name in [".", ".."]:
                            continue

                        item_uri = VikingURI(msg.uri).join(name).uri

                        if entry.get("isDir", False):
                            children_uris.append(item_uri)
                        else:
                            file_paths.append(item_uri)
                except Exception as e:
                    logger.warning(f"Failed to list directory {msg.uri}: {e}")

                incremental_update = msg.is_incremental_update
                target_uri = msg.target_uri if msg.target_uri else None

                if incremental_update and target_uri:
                    children_changed = await self._check_dir_children_changed(
                        msg.uri, file_paths, children_uris, target_uri
                    )
                    if not children_changed:
                        overview, abstract = await self._read_existing_overview_abstract(
                            msg.uri, target_uri
                        )
                        if overview and abstract:
                            await viking_fs.write_file(f"{msg.uri}/.overview.md", overview, ctx=self._current_ctx)
                            await viking_fs.write_file(f"{msg.uri}/.abstract.md", abstract, ctx=self._current_ctx)
                            logger.debug(f"Reused overview and abstract for {msg.uri}")
                            try:
                                embedding_count = 2
                                await self._register_embedding_tracker(
                                    msg.id, embedding_count, msg.uri,
                                    lock_resource_uri=msg.lock_resource_uri,
                                    lock_id=msg.lock_id
                                )
                                await self._vectorize_directory_simple(
                                    msg.uri, msg.context_type, abstract, overview,
                                    semantic_msg_id=msg.id
                                )
                            except Exception as e:
                                logger.error(f"Failed to vectorize directory {msg.uri}: {e}", exc_info=True)
                            logger.info(f"Completed semantic generation for: {msg.uri}")
                            self.report_success()
                            return None

                await self._process_single_directory(
                    uri=msg.uri,
                    context_type=msg.context_type,
                    children_uris=children_uris,
                    file_paths=file_paths,
                    incremental_update=incremental_update,
                    target_uri=target_uri,
                    semantic_msg_id=msg.id,
                    lock_resource_uri=msg.lock_resource_uri,
                    lock_id=msg.lock_id,
                )

                logger.info(f"Completed semantic generation for: {msg.uri}")
                self.report_success()
                return None

        except Exception as e:
            logger.error(f"Failed to process semantic message: {e}", exc_info=True)
            self.report_error(str(e), data)
            return None
        finally:
            self._current_msg = None

    def get_dag_stats(self) -> Optional["DagStats"]:
        if not self._dag_executor:
            return None
        return self._dag_executor.get_stats()

    async def _process_single_directory(
        self,
        uri: str,
        context_type: str,
        children_uris: List[str],
        file_paths: List[str],
        incremental_update: bool = False,
        target_uri: Optional[str] = None,
        semantic_msg_id: Optional[str] = None,
        lock_resource_uri: str = "",
        lock_id: str = "",
    ) -> None:
        """Process single directory, generate .abstract.md and .overview.md."""
        viking_fs = get_viking_fs()

        children_abstracts = await self._collect_children_abstracts(children_uris)

        file_summaries = await self._generate_file_summaries(
            file_paths, 
            context_type=context_type, 
            parent_uri=uri, 
            enqueue_files=True,
            incremental_update=incremental_update,
            target_uri=target_uri,
            semantic_msg_id=semantic_msg_id,
        )

        overview = await self._generate_overview(uri, file_summaries, children_abstracts)

        abstract = self._extract_abstract_from_overview(overview)

        await viking_fs.write_file(f"{uri}/.overview.md", overview, ctx=self._current_ctx)
        await viking_fs.write_file(f"{uri}/.abstract.md", abstract, ctx=self._current_ctx)

        logger.debug(f"Generated overview and abstract for {uri}")

        try:
            embedding_count = len(file_paths) + 2
            await self._register_embedding_tracker(
                semantic_msg_id, embedding_count, uri,
                lock_resource_uri=lock_resource_uri,
                lock_id=lock_id
            )
            await self._vectorize_directory_simple(uri, context_type, abstract, overview, semantic_msg_id=semantic_msg_id)
        except Exception as e:
            logger.error(f"Failed to vectorize directory {uri}: {e}", exc_info=True)
    
    async def _register_embedding_tracker(
        self,
        semantic_msg_id: Optional[str],
        total_count: int,
        uri: str,
        lock_resource_uri: str = "",
        lock_id: str = "",
    ) -> None:
        """Register embedding task tracker for a SemanticMsg.
        
        Args:
            semantic_msg_id: The ID of the SemanticMsg
            total_count: Total number of embedding tasks
            uri: The URI being processed
            lock_resource_uri: Resource URI for lock release on completion
            lock_id: Lock ID for release on completion
        """
        if not semantic_msg_id or total_count <= 0:
            return
        
        from .embedding_tracker import EmbeddingTaskTracker
        tracker = EmbeddingTaskTracker.get_instance()
        
        on_complete = None
        if lock_resource_uri and lock_id:
            on_complete = self._create_lock_release_callback(lock_resource_uri, lock_id)
        
        await tracker.register(
            semantic_msg_id=semantic_msg_id,
            total_count=total_count,
            on_complete=on_complete,
            metadata={"uri": uri, "lock_resource_uri": lock_resource_uri, "lock_id": lock_id}
        )
    
    def _create_lock_release_callback(self, lock_resource_uri: str, lock_id: str):
        """Create a callback function to release the resource lock.
        
        Args:
            lock_resource_uri: Resource URI to release lock for
            lock_id: Lock ID to release
            
        Returns:
            Async callback function
        """
        async def release_lock_callback():
            try:
                from openviking.resource.resource_lock import ResourceLockManager
                viking_fs = get_viking_fs()
                if hasattr(viking_fs, 'agfs') and viking_fs.agfs:
                    lock_manager = ResourceLockManager(viking_fs.agfs)
                    success = lock_manager.release_lock(
                        resource_uri=lock_resource_uri,
                        lock_id=lock_id,
                    )
                    if success:
                        logger.info(f"Released lock for resource {lock_resource_uri}, lock_id={lock_id}")
                    else:
                        logger.warning(f"Failed to release lock for resource {lock_resource_uri}, lock_id={lock_id}")
                else:
                    logger.warning("Cannot release lock: agfs not available")
            except Exception as e:
                logger.error(f"Error releasing lock for {lock_resource_uri}: {e}", exc_info=True)
        
        return release_lock_callback

    def _create_sync_diff_callback(
        self,
        root_uri: str,
        target_uri: str,
        lock_id: str,
    ) -> Callable[[], Awaitable[None]]:
        """
        Create a callback function to sync directory differences.

        This callback compares root_uri (new content) with target_uri (old content),
        handles added/updated/deleted files, then cleans up root_uri and releases lock.

        Args:
            root_uri: Source directory URI (new content)
            target_uri: Target directory URI (old content)
            lock_id: Lock ID to release after completion

        Returns:
            Async callback function
        """
        
        async def sync_diff_callback() -> None:
            logger.info(
                f"[SyncDiff] Starting sync diff callback: "
                f"root_uri={root_uri}, target_uri={target_uri}, lock_id={lock_id}"
            )
            
            if root_uri == target_uri:
                logger.warning(
                    f"[SyncDiff] root_uri and target_uri are the same ({root_uri}), "
                    f"skipping diff comparison and sync operations, releasing lock directly"
                )
                try:
                    viking_fs = get_viking_fs()
                    from openviking.resource.resource_lock import ResourceLockManager
                    if hasattr(viking_fs, 'agfs') and viking_fs.agfs:
                        lock_manager = ResourceLockManager(viking_fs.agfs)
                        success = lock_manager.release_lock(
                            resource_uri=target_uri,
                            lock_id=lock_id,
                        )
                        if success:
                            logger.info(
                                f"[SyncDiff] Successfully released lock for {target_uri}, lock_id={lock_id}"
                            )
                        else:
                            logger.warning(
                                f"[SyncDiff] Failed to release lock for {target_uri}, lock_id={lock_id}"
                            )
                    else:
                        logger.warning("[SyncDiff] Cannot release lock: agfs not available")
                except Exception as e:
                    logger.error(
                        f"[SyncDiff] Error releasing lock for {target_uri}: {e}",
                        exc_info=True
                    )
                return
            
            try:
                viking_fs = get_viking_fs()
                
                logger.info(f"[SyncDiff] Step 1: Collecting tree info for root_uri={root_uri}")
                root_tree = await self._collect_tree_info(root_uri)
                root_dir_count = len(root_tree)
                root_file_count = sum(len(files) for _, files in root_tree.values())
                logger.info(
                    f"[SyncDiff] Root tree collected: {root_dir_count} dirs, {root_file_count} files"
                )
                
                logger.info(f"[SyncDiff] Step 2: Collecting tree info for target_uri={target_uri}")
                target_tree = await self._collect_tree_info(target_uri)
                target_dir_count = len(target_tree)
                target_file_count = sum(len(files) for _, files in target_tree.values())
                logger.info(
                    f"[SyncDiff] Target tree collected: {target_dir_count} dirs, {target_file_count} files"
                )
                
                logger.info("[SyncDiff] Step 3: Computing diff between root and target")
                diff = await self._compute_diff(root_tree, target_tree, root_uri, target_uri)
                logger.info(
                    f"[SyncDiff] Diff computed: "
                    f"added_files={len(diff.added_files)}, "
                    f"deleted_files={len(diff.deleted_files)}, "
                    f"updated_files={len(diff.updated_files)}, "
                    f"added_dirs={len(diff.added_dirs)}, "
                    f"deleted_dirs={len(diff.deleted_dirs)}"
                )
                if diff.added_files:
                    logger.debug(f"[SyncDiff] Added files: {diff.added_files}")
                if diff.deleted_files:
                    logger.debug(f"[SyncDiff] Deleted files: {diff.deleted_files}")
                if diff.updated_files:
                    logger.debug(f"[SyncDiff] Updated files: {diff.updated_files}")
                if diff.added_dirs:
                    logger.debug(f"[SyncDiff] Added dirs: {diff.added_dirs}")
                if diff.deleted_dirs:
                    logger.debug(f"[SyncDiff] Deleted dirs: {diff.deleted_dirs}")
                
                logger.info("[SyncDiff] Step 4: Executing sync operations")
                await self._execute_sync_operations(diff, root_uri, target_uri)
                logger.info("[SyncDiff] Sync operations completed")
                
                logger.info(f"[SyncDiff] Step 5: Deleting root directory {root_uri}")
                try:
                    await viking_fs.rm(root_uri, recursive=True, ctx=self._current_ctx)
                    logger.info(f"[SyncDiff] Successfully deleted root directory: {root_uri}")
                except Exception as e:
                    logger.warning(f"[SyncDiff] Failed to delete root directory {root_uri}: {e}")
                
                logger.info(f"[SyncDiff] Step 6: Releasing lock for {target_uri}")
                from openviking.resource.resource_lock import ResourceLockManager
                if hasattr(viking_fs, 'agfs') and viking_fs.agfs:
                    lock_manager = ResourceLockManager(viking_fs.agfs)
                    success = lock_manager.release_lock(
                        resource_uri=target_uri,
                        lock_id=lock_id,
                    )
                    if success:
                        logger.info(
                            f"[SyncDiff] Successfully released lock for {target_uri}, lock_id={lock_id}"
                        )
                    else:
                        logger.warning(
                            f"[SyncDiff] Failed to release lock for {target_uri}, lock_id={lock_id}"
                        )
                else:
                    logger.warning("[SyncDiff] Cannot release lock: agfs not available")
                
                logger.info(
                    f"[SyncDiff] Sync diff callback completed successfully: "
                    f"root_uri={root_uri}, target_uri={target_uri}"
                )
                    
            except Exception as e:
                logger.error(
                    f"[SyncDiff] Error in sync_diff_callback: "
                    f"root_uri={root_uri}, target_uri={target_uri}, lock_id={lock_id}, "
                    f"error={e}",
                    exc_info=True
                )
        
        return sync_diff_callback

    async def _collect_tree_info(
        self,
        uri: str,
    ) -> Dict[str, Tuple[List[str], List[str]]]:
        """
        Recursively collect directory tree information.

        Args:
            uri: Directory URI

        Returns:
            Dictionary: {dir_uri: ([subdir_uris], [file_uris])}
        """
        viking_fs = get_viking_fs()
        result: Dict[str, Tuple[List[str], List[str]]] = {}
        total_dirs = 0
        total_files = 0
        
        async def collect_recursive(current_uri: str, depth: int = 0) -> None:
            nonlocal total_dirs, total_files
            indent = "  " * depth
            try:
                logger.debug(f"[SyncDiff]{indent} Listing directory: {current_uri}")
                entries = await viking_fs.ls(current_uri, ctx=self._current_ctx)
            except Exception as e:
                logger.warning(f"[SyncDiff]{indent} Failed to list {current_uri}: {e}")
                return
            
            sub_dirs: List[str] = []
            files: List[str] = []
            
            for entry in entries:
                name = entry.get("name", "")
                if not name or name.startswith(".") or name in [".", ".."]:
                    continue
                
                item_uri = VikingURI(current_uri).join(name).uri
                
                if entry.get("isDir", False):
                    sub_dirs.append(item_uri)
                    total_dirs += 1
                    await collect_recursive(item_uri, depth + 1)
                else:
                    files.append(item_uri)
                    total_files += 1
            
            result[current_uri] = (sub_dirs, files)
            logger.debug(
                f"[SyncDiff]{indent} Collected {current_uri}: "
                f"{len(sub_dirs)} subdirs, {len(files)} files"
            )
        
        await collect_recursive(uri)
        logger.info(
            f"[SyncDiff] Tree info collection completed for {uri}: "
            f"total_dirs={total_dirs}, total_files={total_files}"
        )
        return result

    async def _compute_diff(
        self,
        root_tree: Dict[str, Tuple[List[str], List[str]]],
        target_tree: Dict[str, Tuple[List[str], List[str]]],
        root_uri: str,
        target_uri: str,
    ) -> DiffResult:
        """
        Compute differences between two directory trees.

        Args:
            root_tree: Directory tree from root_uri
            target_tree: Directory tree from target_uri
            root_uri: Source directory URI
            target_uri: Target directory URI

        Returns:
            DiffResult with added/deleted/updated files and directories
        """
        logger.debug(f"[SyncDiff] Computing diff: root_uri={root_uri}, target_uri={target_uri}")
        
        def get_relative_path(uri: str, base_uri: str) -> str:
            if uri.startswith(base_uri):
                rel = uri[len(base_uri):]
                return rel.lstrip("/")
            return uri
        
        root_files: Set[str] = set()
        root_dirs: Set[str] = set()
        target_files: Set[str] = set()
        target_dirs: Set[str] = set()
        
        for dir_uri, (sub_dirs, files) in root_tree.items():
            rel_dir = get_relative_path(dir_uri, root_uri)
            if rel_dir:
                root_dirs.add(rel_dir)
            for f in files:
                root_files.add(get_relative_path(f, root_uri))
            for d in sub_dirs:
                root_dirs.add(get_relative_path(d, root_uri))
        
        for dir_uri, (sub_dirs, files) in target_tree.items():
            rel_dir = get_relative_path(dir_uri, target_uri)
            if rel_dir:
                target_dirs.add(rel_dir)
            for f in files:
                target_files.add(get_relative_path(f, target_uri))
            for d in sub_dirs:
                target_dirs.add(get_relative_path(d, target_uri))
        
        logger.debug(
            f"[SyncDiff] Root stats: {len(root_files)} files, {len(root_dirs)} dirs"
        )
        logger.debug(
            f"[SyncDiff] Target stats: {len(target_files)} files, {len(target_dirs)} dirs"
        )
        
        added_files_rel = root_files - target_files
        deleted_files_rel = target_files - root_files
        common_files = root_files & target_files
        
        added_dirs_rel = root_dirs - target_dirs
        deleted_dirs_rel = target_dirs - root_dirs
        
        logger.debug(
            f"[SyncDiff] File diff: added={len(added_files_rel)}, "
            f"deleted={len(deleted_files_rel)}, common={len(common_files)}"
        )
        logger.debug(
            f"[SyncDiff] Dir diff: added={len(added_dirs_rel)}, deleted={len(deleted_dirs_rel)}"
        )
        
        updated_files: List[str] = []
        logger.debug(f"[SyncDiff] Checking content changes for {len(common_files)} common files")
        for rel_file in common_files:
            root_file = f"{root_uri}/{rel_file}"
            target_file = f"{target_uri}/{rel_file}"
            try:
                if await self._check_file_content_changed(root_file, target_file):
                    updated_files.append(root_file)
                    logger.debug(f"[SyncDiff] File content changed: {rel_file}")
            except Exception as e:
                logger.warning(
                    f"[SyncDiff] Failed to compare file content for {rel_file}: {e}, "
                    f"treating as unchanged"
                )
        
        added_files = [f"{root_uri}/{f}" for f in added_files_rel]
        deleted_files = [f"{target_uri}/{f}" for f in deleted_files_rel]
        added_dirs = [f"{root_uri}/{d}" for d in added_dirs_rel]
        deleted_dirs = [f"{target_uri}/{d}" for d in deleted_dirs_rel]
        
        result = DiffResult(
            added_files=added_files,
            deleted_files=deleted_files,
            updated_files=updated_files,
            added_dirs=added_dirs,
            deleted_dirs=deleted_dirs,
        )
        
        logger.info(
            f"[SyncDiff] Diff computation completed: "
            f"added_files={len(added_files)}, deleted_files={len(deleted_files)}, "
            f"updated_files={len(updated_files)}, added_dirs={len(added_dirs)}, "
            f"deleted_dirs={len(deleted_dirs)}"
        )
        
        return result

    async def _execute_sync_operations(
        self,
        diff: DiffResult,
        root_uri: str,
        target_uri: str,
    ) -> None:
        """
        Execute sync operations based on diff result.

        Processing order:
        1. Delete files in target that don't exist in root
        2. Move added/updated files from root to target
        3. Delete directories in target that don't exist in root

        Args:
            diff: DiffResult containing operations to perform
            root_uri: Source directory URI
            target_uri: Target directory URI
        """
        viking_fs = get_viking_fs()
        
        def map_to_target(root_item_uri: str) -> str:
            if root_item_uri.startswith(root_uri):
                rel = root_item_uri[len(root_uri):]
                return f"{target_uri}{rel}" if rel else target_uri
            return root_item_uri
        
        total_deleted = 0
        total_moved = 0
        total_failed = 0
        
        logger.info(
            f"[SyncDiff] Starting sync operations: "
            f"delete={len(diff.deleted_files)}, move={len(diff.added_files) + len(diff.updated_files)}"
        )
        
        if diff.deleted_files:
            logger.info(f"[SyncDiff] Phase 1: Deleting {len(diff.deleted_files)} files from target")
        for i, deleted_file in enumerate(diff.deleted_files, 1):
            try:
                logger.debug(f"[SyncDiff] Deleting file [{i}/{len(diff.deleted_files)}]: {deleted_file}")
                await viking_fs.rm(deleted_file, ctx=self._current_ctx)
                total_deleted += 1
                logger.info(f"[SyncDiff] Deleted file [{i}/{len(diff.deleted_files)}]: {deleted_file}")
            except Exception as e:
                total_failed += 1
                logger.warning(
                    f"[SyncDiff] Failed to delete file [{i}/{len(diff.deleted_files)}]: {deleted_file}, error={e}"
                )
        
        if diff.updated_files:
            logger.info(
                f"[SyncDiff] Phase 2: Removing {len(diff.updated_files)} old files for update"
            )
        for i, updated_file in enumerate(diff.updated_files, 1):
            target_file = map_to_target(updated_file)
            try:
                logger.debug(
                    f"[SyncDiff] Removing old file for update [{i}/{len(diff.updated_files)}]: {target_file}"
                )
                await viking_fs.rm(target_file, ctx=self._current_ctx)
                logger.info(
                    f"[SyncDiff] Removed old file for update [{i}/{len(diff.updated_files)}]: {target_file}"
                )
            except Exception as e:
                logger.warning(
                    f"[SyncDiff] Failed to remove old file [{i}/{len(diff.updated_files)}]: {target_file}, error={e}"
                )
        
        files_to_move = diff.added_files + diff.updated_files
        if files_to_move:
            logger.info(f"[SyncDiff] Phase 3: Moving {len(files_to_move)} files from root to target")
        for i, root_file in enumerate(files_to_move, 1):
            target_file = map_to_target(root_file)
            try:
                logger.debug(
                    f"[SyncDiff] Moving file [{i}/{len(files_to_move)}]: {root_file} -> {target_file}"
                )
                await viking_fs.mv(root_file, target_file, ctx=self._current_ctx)
                total_moved += 1
                logger.info(
                    f"[SyncDiff] Moved file [{i}/{len(files_to_move)}]: {root_file} -> {target_file}"
                )
            except Exception as e:
                total_failed += 1
                logger.warning(
                    f"[SyncDiff] Failed to move file [{i}/{len(files_to_move)}]: "
                    f"{root_file} -> {target_file}, error={e}"
                )
        
        if diff.deleted_dirs:
            logger.info(
                f"[SyncDiff] Phase 4: Deleting {len(diff.deleted_dirs)} directories from target"
            )
        for i, deleted_dir in enumerate(
            sorted(diff.deleted_dirs, key=lambda x: x.count("/"), reverse=True), 1
        ):
            try:
                logger.debug(
                    f"[SyncDiff] Deleting directory [{i}/{len(diff.deleted_dirs)}]: {deleted_dir}"
                )
                await viking_fs.rm(deleted_dir, recursive=True, ctx=self._current_ctx)
                logger.info(
                    f"[SyncDiff] Deleted directory [{i}/{len(diff.deleted_dirs)}]: {deleted_dir}"
                )
            except Exception as e:
                total_failed += 1
                logger.warning(
                    f"[SyncDiff] Failed to delete directory [{i}/{len(diff.deleted_dirs)}]: "
                    f"{deleted_dir}, error={e}"
                )
        
        logger.info(
            f"[SyncDiff] Sync operations completed: "
            f"deleted={total_deleted}, moved={total_moved}, failed={total_failed}"
        )

    async def _collect_children_abstracts(self, children_uris: List[str]) -> List[Dict[str, str]]:
        """Collect .abstract.md from subdirectories."""
        viking_fs = get_viking_fs()
        results = []

        for child_uri in children_uris:
            abstract = await viking_fs.abstract(child_uri, ctx=self._current_ctx)
            dir_name = child_uri.split("/")[-1]
            results.append({"name": dir_name, "abstract": abstract})
        return results

    async def _generate_file_summaries(
        self,
        file_paths: List[str],
        context_type: Optional[str] = None,
        parent_uri: Optional[str] = None,
        enqueue_files: bool = False,
        incremental_update: bool = False,
        target_uri: Optional[str] = None,
        semantic_msg_id: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Concurrently generate file summaries."""
        if not file_paths:
            return []

        async def generate_one_summary(file_path: str) -> Dict[str, str]:
            summary = None
            if incremental_update and target_uri:
                content_changed = await self._check_file_content_changed(
                    file_path, target_uri
                )
                if not content_changed:
                    summary = await self._read_existing_summary(file_path, target_uri)
            
            if summary is None:
                summary = await self._generate_single_file_summary(file_path, ctx=self._current_ctx)
            
            if enqueue_files and context_type and parent_uri:
                try:
                    await self._vectorize_single_file(
                        parent_uri=parent_uri,
                        context_type=context_type,
                        file_path=file_path,
                        summary_dict=summary,
                        semantic_msg_id=semantic_msg_id,
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to vectorize file {file_path}: {e}",
                        exc_info=True,
                    )
            return summary

        tasks = [generate_one_summary(fp) for fp in file_paths]
        return await asyncio.gather(*tasks)

    async def _generate_text_summary(
        self,
        file_path: str,
        file_name: str,
        llm_sem: asyncio.Semaphore,
        ctx: Optional[RequestContext] = None,
    ) -> Dict[str, str]:
        """Generate summary for a single text file (code, documentation, or other text)."""
        viking_fs = get_viking_fs()
        vlm = get_openviking_config().vlm
        active_ctx = ctx or self._current_ctx

        # Read file content (limit length)
        content = await viking_fs.read_file(file_path, ctx=active_ctx)

        # Limit content length (about 10000 tokens)
        max_chars = 30000
        content = await viking_fs.read_file(file_path, ctx=active_ctx)
        if isinstance(content, bytes):
            # Try to decode with error handling for text files
            try:
                content = content.decode("utf-8")
            except UnicodeDecodeError:
                logger.warning(f"Failed to decode file as UTF-8, skipping: {file_path}")
                return {"name": file_name, "summary": ""}

        # Limit content length (about 10000 tokens)
        max_chars = 30000
        if len(content) > max_chars:
            content = content[:max_chars] + "\n...(truncated)"

        # Generate summary
        if not vlm.is_available():
            logger.warning("VLM not available, using empty summary")
            return {"name": file_name, "summary": ""}

        # Detect file type and select appropriate prompt
        file_type = self._detect_file_type(file_name)

        if file_type == FILE_TYPE_CODE:
            code_mode = get_openviking_config().code.code_summary_mode

            if code_mode in ("ast", "ast_llm") and len(content.splitlines()) >= 100:
                from openviking.parse.parsers.code.ast import extract_skeleton

                verbose = code_mode == "ast_llm"
                skeleton_text = extract_skeleton(file_name, content, verbose=verbose)
                if skeleton_text:
                    if code_mode == "ast":
                        return {"name": file_name, "summary": skeleton_text}
                    else:  # ast_llm
                        prompt = render_prompt(
                            "semantic.code_ast_summary",
                            {"file_name": file_name, "skeleton": skeleton_text},
                        )
                        async with llm_sem:
                            summary = await vlm.get_completion_async(prompt)
                        return {"name": file_name, "summary": summary.strip()}
                if skeleton_text is None:
                    logger.info("AST unsupported language, fallback to LLM: %s", file_path)
                else:
                    logger.info("AST empty skeleton, fallback to LLM: %s", file_path)

            # "llm" mode or fallback when skeleton is None/empty
            prompt = render_prompt(
                "semantic.code_summary",
                {"file_name": file_name, "content": content},
            )
            async with llm_sem:
                summary = await vlm.get_completion_async(prompt)
            return {"name": file_name, "summary": summary.strip()}

        elif file_type == FILE_TYPE_DOCUMENTATION:
            prompt_id = "semantic.document_summary"
        else:
            prompt_id = "semantic.file_summary"

        prompt = render_prompt(
            prompt_id,
            {"file_name": file_name, "content": content},
        )

        async with llm_sem:
            summary = await vlm.get_completion_async(prompt)
        return {"name": file_name, "summary": summary.strip()}

    async def _generate_single_file_summary(
        self,
        file_path: str,
        llm_sem: Optional[asyncio.Semaphore] = None,
        ctx: Optional[RequestContext] = None,
    ) -> Dict[str, str]:
        """Generate summary for a single file.

        Args:
            file_path: File path

        Returns:
            {"name": file_name, "summary": summary_content}
        """
        file_name = file_path.split("/")[-1]
        llm_sem = llm_sem or asyncio.Semaphore(self.max_concurrent_llm)
        media_type = get_media_type(file_name, None)
        if media_type == "image":
            return await generate_image_summary(file_path, file_name, llm_sem, ctx=ctx)
        elif media_type == "audio":
            return await generate_audio_summary(file_path, file_name, llm_sem, ctx=ctx)
        elif media_type == "video":
            return await generate_video_summary(file_path, file_name, llm_sem, ctx=ctx)
        else:
            return await self._generate_text_summary(file_path, file_name, llm_sem, ctx=ctx)

    def _extract_abstract_from_overview(self, overview_content: str) -> str:
        """Extract abstract from overview.md."""
        lines = overview_content.split("\n")

        # Skip header lines (starting with #)
        content_lines = []
        in_header = True

        for line in lines:
            if in_header and line.startswith("#"):
                continue
            elif in_header and line.strip():
                in_header = False

            if not in_header:
                # Stop at first ##
                if line.startswith("##"):
                    break
                if line.strip():
                    content_lines.append(line.strip())

        return "\n".join(content_lines).strip()

    async def _generate_overview(
        self,
        dir_uri: str,
        file_summaries: List[Dict[str, str]],
        children_abstracts: List[Dict[str, str]],
    ) -> str:
        """Generate directory's .overview.md (L1).

        Args:
            dir_uri: Directory URI
            file_summaries: File summary list
            children_abstracts: Subdirectory summary list

        Returns:
            Overview content
        """
        import re

        vlm = get_openviking_config().vlm

        if not vlm.is_available():
            logger.warning("VLM not available, using default overview")
            return f"# {dir_uri.split('/')[-1]}\n\nDirectory overview"

        # Build file index mapping and summary string
        file_index_map = {}
        file_summaries_lines = []
        for idx, item in enumerate(file_summaries, 1):
            file_index_map[idx] = item["name"]
            file_summaries_lines.append(f"[{idx}] {item['name']}: {item['summary']}")
        file_summaries_str = "\n".join(file_summaries_lines) if file_summaries_lines else "None"

        # Build subdirectory summary string
        children_abstracts_str = (
            "\n".join(f"- {item['name']}/: {item['abstract']}" for item in children_abstracts)
            if children_abstracts
            else "None"
        )

        # Generate overview
        try:
            prompt = render_prompt(
                "semantic.overview_generation",
                {
                    "dir_name": dir_uri.split("/")[-1],
                    "file_summaries": file_summaries_str,
                    "children_abstracts": children_abstracts_str,
                },
            )

            overview = await vlm.get_completion_async(prompt)

            # Post-process: replace [number] with actual file name
            def replace_index(match):
                idx = int(match.group(1))
                return file_index_map.get(idx, match.group(0))

            overview = re.sub(r"\[(\d+)\]", replace_index, overview)

            return overview.strip()

        except Exception as e:
            logger.error(f"Failed to generate overview for {dir_uri}: {e}", exc_info=True)
            return f"# {dir_uri.split('/')[-1]}\n\nDirectory overview"

    async def _vectorize_directory_simple(
        self,
        uri: str,
        context_type: str,
        abstract: str,
        overview: str,
        ctx: Optional[RequestContext] = None,
        semantic_msg_id: Optional[str] = None,
        lock_resource_uri: Optional[str] = "",
        lock_id: Optional[str] = "",
    ) -> None:
        """Create directory Context and enqueue to EmbeddingQueue."""

        if self._current_msg and getattr(self._current_msg, "skip_vectorization", False):
            logger.info(f"Skipping vectorization for {uri} (requested via SemanticMsg)")
            return

        from openviking.utils.embedding_utils import vectorize_directory_meta
        if semantic_msg_id and lock_resource_uri and lock_id:
            tracker = EmbeddingTaskTracker.get_instance()
            await tracker.increment(
                semantic_msg_id=semantic_msg_id,
            )
            await tracker.increment(
                semantic_msg_id=semantic_msg_id
            )

        active_ctx = ctx or self._current_ctx
        await vectorize_directory_meta(
            uri=uri,
            abstract=abstract,
            overview=overview,
            context_type=context_type,
            ctx=active_ctx,
            semantic_msg_id=semantic_msg_id,
        )

    async def _vectorize_files(
        self,
        uri: str,
        context_type: str,
        file_paths: List[str],
        file_summaries: List[Dict[str, str]],
        ctx: Optional[RequestContext] = None,
    ) -> None:
        """Vectorize files in directory."""
        from openviking.storage.queuefs import get_queue_manager

        queue_manager = get_queue_manager()
        embedding_queue = queue_manager.get_queue(queue_manager.EMBEDDING)

        for file_path, file_summary_dict in zip(file_paths, file_summaries):
            await self._vectorize_single_file(
                parent_uri=uri,
                context_type=context_type,
                file_path=file_path,
                summary_dict=file_summary_dict,
                embedding_queue=embedding_queue,
                ctx=ctx,
            )

    async def _vectorize_single_file(
        self,
        parent_uri: str,
        context_type: str,
        file_path: str,
        summary_dict: Dict[str, str],
        ctx: Optional[RequestContext] = None,
        semantic_msg_id: Optional[str] = None,
        lock_resource_uri: Optional[str] = "",
        lock_id: Optional[str] = "",
    ) -> None:
        """Vectorize a single file using its content or summary."""
        from openviking.utils.embedding_utils import vectorize_file

        if semantic_msg_id and lock_resource_uri and lock_id:
            tracker = EmbeddingTaskTracker.get_instance()
            await tracker.increment(
                semantic_msg_id=semantic_msg_id,
            )

        active_ctx = ctx or self._current_ctx
        await vectorize_file(
            file_path=file_path,
            summary_dict=summary_dict,
            parent_uri=parent_uri,
            context_type=context_type,
            ctx=active_ctx,
            semantic_msg_id=semantic_msg_id,
        )

    def _get_target_file_path(self, current_uri: str, target_uri: str) -> Optional[str]:
        """Get target file path for incremental update."""
        try:
            relative_path = current_uri[len(self._current_msg.uri):] if self._current_msg else ""
            if relative_path.startswith("/"):
                relative_path = relative_path[1:]
            return f"{target_uri}/{relative_path}" if relative_path else target_uri
        except Exception:
            return None

    async def _check_file_content_changed(self, file_path: str, target_uri: str) -> bool:
        """Check if file content has changed."""
        target_path = self._get_target_file_path(file_path, target_uri)
        if not target_path:
            return True
        try:
            viking_fs = get_viking_fs()
            current_content = await viking_fs.read_file(file_path, ctx=self._current_ctx)
            target_content = await viking_fs.read_file(target_path, ctx=self._current_ctx)
            return current_content != target_content
        except Exception:
            return True

    async def _read_existing_summary(self, file_path: str, target_uri: str) -> Optional[Dict[str, str]]:
        """Read existing summary from vector store."""
        target_path = self._get_target_file_path(file_path, target_uri)
        if not target_path:
            return None
        try:
            viking_fs = get_viking_fs()
            vector_store = viking_fs._get_vector_store()
            if not vector_store:
                return None
            records = await vector_store.get_context_by_uri(
                account_id=self._current_ctx.account_id,
                uri=target_path,
                limit=1,
            )
            if records and len(records) > 0:
                record = records[0]
                summary = record.get("summary", "")
                if summary:
                    file_name = file_path.split("/")[-1]
                    return {"name": file_name, "summary": summary}
        except Exception:
            pass
        return None

    async def _check_dir_children_changed(
        self, dir_uri: str, current_files: List[str], current_dirs: List[str], target_uri: str
    ) -> bool:
        """Check if directory children have changed."""
        target_path = self._get_target_file_path(dir_uri, target_uri)
        if not target_path:
            return True
        try:
            viking_fs = get_viking_fs()
            target_entries = await viking_fs.ls(target_path, ctx=self._current_ctx)
            target_files = []
            target_dirs = []
            for entry in target_entries:
                name = entry.get("name", "")
                if not name or name.startswith(".") or name in [".", ".."]:
                    continue
                if entry.get("isDir", False):
                    target_dirs.append(name)
                else:
                    target_files.append(name)
            
            current_file_names = {f.split("/")[-1] for f in current_files}
            target_file_names = set(target_files)
            if current_file_names != target_file_names:
                return True
            
            current_dir_names = {d.split("/")[-1] for d in current_dirs}
            target_dir_names = set(target_dirs)
            if current_dir_names != target_dir_names:
                return True
            
            for current_file in current_files:
                if await self._check_file_content_changed(current_file, target_uri):
                    return True
            return False
        except Exception:
            return True

    async def _read_existing_overview_abstract(
        self, dir_uri: str, target_uri: str
    ) -> tuple[Optional[str], Optional[str]]:
        """Read existing overview and abstract from target directory."""
        target_path = self._get_target_file_path(dir_uri, target_uri)
        if not target_path:
            return None, None
        try:
            viking_fs = get_viking_fs()
            overview = await viking_fs.read_file(f"{target_path}/.overview.md", ctx=self._current_ctx)
            abstract = await viking_fs.read_file(f"{target_path}/.abstract.md", ctx=self._current_ctx)
            return overview, abstract
        except Exception:
            return None, None
