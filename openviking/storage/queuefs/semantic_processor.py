# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""SemanticProcessor: Processes messages from SemanticQueue, generates .abstract.md and .overview.md."""

import asyncio
from typing import Any, Dict, List, Optional, Tuple

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
                f"Processing semantic generation for: {msg.uri} (recursive={msg.recursive})"
            )

            if msg.recursive:
                executor = SemanticDagExecutor(
                    processor=self,
                    context_type=msg.context_type,
                    max_concurrent_llm=self.max_concurrent_llm,
                    ctx=self._current_ctx,
                    incremental_update=msg.is_incremental_update,
                    target_uri_root=msg.target_uri_root if msg.target_uri_root else None,
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
                target_uri_root = msg.target_uri_root if msg.target_uri_root else None

                if incremental_update and target_uri_root:
                    children_changed = await self._check_dir_children_changed(
                        msg.uri, file_paths, children_uris, target_uri_root
                    )
                    if not children_changed:
                        overview, abstract = await self._read_existing_overview_abstract(
                            msg.uri, target_uri_root
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
                    target_uri_root=target_uri_root,
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
        target_uri_root: Optional[str] = None,
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
            target_uri_root=target_uri_root,
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
        target_uri_root: Optional[str] = None,
        semantic_msg_id: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Concurrently generate file summaries."""
        if not file_paths:
            return []

        async def generate_one_summary(file_path: str) -> Dict[str, str]:
            summary = None
            if incremental_update and target_uri_root:
                content_changed = await self._check_file_content_changed(
                    file_path, target_uri_root
                )
                if not content_changed:
                    summary = await self._read_existing_summary(file_path, target_uri_root)
            
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

    def _get_target_file_path(self, current_uri: str, target_uri_root: str) -> Optional[str]:
        """Get target file path for incremental update."""
        try:
            relative_path = current_uri[len(self._current_msg.uri):] if self._current_msg else ""
            if relative_path.startswith("/"):
                relative_path = relative_path[1:]
            return f"{target_uri_root}/{relative_path}" if relative_path else target_uri_root
        except Exception:
            return None

    async def _check_file_content_changed(self, file_path: str, target_uri_root: str) -> bool:
        """Check if file content has changed."""
        target_path = self._get_target_file_path(file_path, target_uri_root)
        if not target_path:
            return True
        try:
            viking_fs = get_viking_fs()
            current_content = await viking_fs.read_file(file_path, ctx=self._current_ctx)
            target_content = await viking_fs.read_file(target_path, ctx=self._current_ctx)
            return current_content != target_content
        except Exception:
            return True

    async def _read_existing_summary(self, file_path: str, target_uri_root: str) -> Optional[Dict[str, str]]:
        """Read existing summary from vector store."""
        target_path = self._get_target_file_path(file_path, target_uri_root)
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
        self, dir_uri: str, current_files: List[str], current_dirs: List[str], target_uri_root: str
    ) -> bool:
        """Check if directory children have changed."""
        target_path = self._get_target_file_path(dir_uri, target_uri_root)
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
                if await self._check_file_content_changed(current_file, target_uri_root):
                    return True
            return False
        except Exception:
            return True

    async def _read_existing_overview_abstract(
        self, dir_uri: str, target_uri_root: str
    ) -> tuple[Optional[str], Optional[str]]:
        """Read existing overview and abstract from target directory."""
        target_path = self._get_target_file_path(dir_uri, target_uri_root)
        if not target_path:
            return None, None
        try:
            viking_fs = get_viking_fs()
            overview = await viking_fs.read_file(f"{target_path}/.overview.md", ctx=self._current_ctx)
            abstract = await viking_fs.read_file(f"{target_path}/.abstract.md", ctx=self._current_ctx)
            return overview, abstract
        except Exception:
            return None, None
