# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import pytest

from openviking.server.identity import RequestContext, Role
from openviking.storage.queuefs.semantic_processor import DiffResult, SemanticProcessor
from openviking_cli.session.user_id import UserIdentifier


class MockVikingFS:
    """Mock VikingFS for testing sync diff callback."""

    def __init__(self):
        self.tree = {}
        self.file_contents = {}
        self.rm_calls = []
        self.mv_calls = []
        self.agfs = None

    def setup_tree(self, tree: dict, file_contents: dict = None):
        self.tree = tree
        self.file_contents = file_contents or {}

    async def ls(self, uri, ctx=None):
        return self.tree.get(uri, [])

    async def rm(self, uri, recursive=False, ctx=None):
        self.rm_calls.append((uri, recursive))
        if uri in self.tree:
            del self.tree[uri]
        keys_to_delete = [k for k in self.tree if k.startswith(uri + "/")]
        for k in keys_to_delete:
            del self.tree[k]
        keys_to_delete = [k for k in self.file_contents if k.startswith(uri)]
        for k in keys_to_delete:
            del self.file_contents[k]

    async def mv(self, old_uri, new_uri, ctx=None):
        self.mv_calls.append((old_uri, new_uri))
        if old_uri in self.file_contents:
            self.file_contents[new_uri] = self.file_contents[old_uri]
            del self.file_contents[old_uri]

    async def read_file(self, uri, ctx=None):
        return self.file_contents.get(uri, b"")


class MockLockManager:
    """Mock ResourceLockManager for testing."""

    def __init__(self):
        self.release_calls = []

    def release_lock(self, resource_uri, lock_id=None):
        self.release_calls.append((resource_uri, lock_id))
        return True


@pytest.fixture
def mock_viking_fs(monkeypatch):
    """Create mock VikingFS instance."""
    mock_fs = MockVikingFS()
    monkeypatch.setattr(
        "openviking.storage.queuefs.semantic_processor.get_viking_fs",
        lambda: mock_fs
    )
    return mock_fs


@pytest.fixture
def processor():
    """Create SemanticProcessor instance with mock context."""
    proc = SemanticProcessor()
    proc._current_ctx = RequestContext(
        user=UserIdentifier("acc1", "user1", "agent1"),
        role=Role.USER
    )
    return proc


@pytest.mark.asyncio
async def test_collect_tree_info_basic(mock_viking_fs, processor):
    """Test basic tree info collection."""
    root_uri = "viking://user/test/repo"
    mock_viking_fs.setup_tree({
        root_uri: [
            {"name": "file1.txt", "isDir": False},
            {"name": "file2.txt", "isDir": False},
            {"name": "subdir", "isDir": True},
        ],
        f"{root_uri}/subdir": [
            {"name": "file3.txt", "isDir": False},
        ],
    })

    tree_info = await processor._collect_tree_info(root_uri)

    assert root_uri in tree_info
    sub_dirs, files = tree_info[root_uri]
    assert len(files) == 2
    assert len(sub_dirs) == 1
    assert f"{root_uri}/subdir" in tree_info


@pytest.mark.asyncio
async def test_collect_tree_info_empty_dir(mock_viking_fs, processor):
    """Test tree info collection for empty directory."""
    root_uri = "viking://user/test/empty"
    mock_viking_fs.setup_tree({root_uri: []})

    tree_info = await processor._collect_tree_info(root_uri)

    assert root_uri in tree_info
    sub_dirs, files = tree_info[root_uri]
    assert len(files) == 0
    assert len(sub_dirs) == 0


@pytest.mark.asyncio
async def test_compute_diff_added_files(mock_viking_fs, processor):
    """Test diff computation for added files."""
    root_uri = "viking://user/test/root"
    target_uri = "viking://user/test/target"

    root_tree = {
        root_uri: (
            [],
            [f"{root_uri}/new_file.txt", f"{root_uri}/common.txt"]
        ),
    }
    target_tree = {
        target_uri: (
            [],
            [f"{target_uri}/common.txt"]
        ),
    }

    mock_viking_fs.setup_tree(
        {},
        {
            f"{root_uri}/common.txt": b"same content",
            f"{target_uri}/common.txt": b"same content",
        }
    )

    diff = await processor._compute_diff(root_tree, target_tree, root_uri, target_uri)

    assert len(diff.added_files) == 1
    assert f"{root_uri}/new_file.txt" in diff.added_files
    assert len(diff.deleted_files) == 0
    assert len(diff.updated_files) == 0


@pytest.mark.asyncio
async def test_compute_diff_deleted_files(mock_viking_fs, processor):
    """Test diff computation for deleted files."""
    root_uri = "viking://user/test/root"
    target_uri = "viking://user/test/target"

    root_tree = {
        root_uri: (
            [],
            [f"{root_uri}/common.txt"]
        ),
    }
    target_tree = {
        target_uri: (
            [],
            [f"{target_uri}/common.txt", f"{target_uri}/old_file.txt"]
        ),
    }

    mock_viking_fs.setup_tree(
        {},
        {
            f"{root_uri}/common.txt": b"same content",
            f"{target_uri}/common.txt": b"same content",
        }
    )

    diff = await processor._compute_diff(root_tree, target_tree, root_uri, target_uri)

    assert len(diff.deleted_files) == 1
    assert f"{target_uri}/old_file.txt" in diff.deleted_files
    assert len(diff.added_files) == 0
    assert len(diff.updated_files) == 0


@pytest.mark.asyncio
async def test_compute_diff_updated_files(mock_viking_fs, processor):
    """Test diff computation for updated files."""
    root_uri = "viking://user/test/root"
    target_uri = "viking://user/test/target"

    root_tree = {
        root_uri: (
            [],
            [f"{root_uri}/modified.txt", f"{root_uri}/unchanged.txt"]
        ),
    }
    target_tree = {
        target_uri: (
            [],
            [f"{target_uri}/modified.txt", f"{target_uri}/unchanged.txt"]
        ),
    }

    mock_viking_fs.setup_tree(
        {},
        {
            f"{root_uri}/modified.txt": b"new content",
            f"{target_uri}/modified.txt": b"old content",
            f"{root_uri}/unchanged.txt": b"same content",
            f"{target_uri}/unchanged.txt": b"same content",
        }
    )

    diff = await processor._compute_diff(root_tree, target_tree, root_uri, target_uri)

    assert len(diff.updated_files) == 1
    assert f"{root_uri}/modified.txt" in diff.updated_files
    assert len(diff.added_files) == 0
    assert len(diff.deleted_files) == 0


@pytest.mark.asyncio
async def test_compute_diff_directories(mock_viking_fs, processor):
    """Test diff computation for directories."""
    root_uri = "viking://user/test/root"
    target_uri = "viking://user/test/target"

    root_tree = {
        root_uri: ([f"{root_uri}/new_dir"], []),
        f"{root_uri}/new_dir": ([], []),
    }
    target_tree = {
        target_uri: ([f"{target_uri}/old_dir"], []),
        f"{target_uri}/old_dir": ([], []),
    }

    diff = await processor._compute_diff(root_tree, target_tree, root_uri, target_uri)

    assert len(diff.added_dirs) == 1
    assert f"{root_uri}/new_dir" in diff.added_dirs
    assert len(diff.deleted_dirs) == 1
    assert f"{target_uri}/old_dir" in diff.deleted_dirs


@pytest.mark.asyncio
async def test_execute_sync_operations(mock_viking_fs, processor):
    """Test executing sync operations."""
    root_uri = "viking://user/test/root"
    target_uri = "viking://user/test/target"

    mock_viking_fs.setup_tree(
        {
            root_uri: [],
            target_uri: [],
        },
        {
            f"{root_uri}/new.txt": b"new",
            f"{root_uri}/updated.txt": b"updated",
            f"{target_uri}/updated.txt": b"old",
            f"{target_uri}/deleted.txt": b"deleted",
        }
    )

    diff = DiffResult(
        added_files=[f"{root_uri}/new.txt"],
        deleted_files=[f"{target_uri}/deleted.txt"],
        updated_files=[f"{root_uri}/updated.txt"],
        added_dirs=[],
        deleted_dirs=[],
    )

    await processor._execute_sync_operations(diff, root_uri, target_uri)

    assert (f"{target_uri}/deleted.txt", False) in mock_viking_fs.rm_calls
    assert (f"{target_uri}/updated.txt", False) in mock_viking_fs.rm_calls
    assert (f"{root_uri}/new.txt", f"{target_uri}/new.txt") in mock_viking_fs.mv_calls
    assert (f"{root_uri}/updated.txt", f"{target_uri}/updated.txt") in mock_viking_fs.mv_calls


@pytest.mark.asyncio
async def test_create_sync_diff_callback_full_flow(mock_viking_fs, processor, monkeypatch):
    """Test full sync diff callback flow."""
    root_uri = "viking://user/test/root"
    target_uri = "viking://user/test/target"
    lock_id = "test-lock-123"

    mock_viking_fs.setup_tree(
        {
            root_uri: [
                {"name": "new_file.txt", "isDir": False},
                {"name": "modified.txt", "isDir": False},
            ],
            target_uri: [
                {"name": "modified.txt", "isDir": False},
                {"name": "old_file.txt", "isDir": False},
            ],
        },
        {
            f"{root_uri}/new_file.txt": b"new content",
            f"{root_uri}/modified.txt": b"modified content",
            f"{target_uri}/modified.txt": b"original content",
            f"{target_uri}/old_file.txt": b"old content",
        }
    )

    mock_lock_manager = MockLockManager()
    monkeypatch.setattr(
        "openviking.resource.resource_lock.ResourceLockManager",
        lambda agfs: mock_lock_manager
    )
    mock_viking_fs.agfs = "mock_agfs"

    callback = processor._create_sync_diff_callback(root_uri, target_uri, lock_id)
    await callback()

    deleted_files = [call[0] for call in mock_viking_fs.rm_calls]
    assert f"{target_uri}/old_file.txt" in deleted_files
    assert f"{target_uri}/modified.txt" in deleted_files

    moved_files = [(src, dst) for src, dst in mock_viking_fs.mv_calls]
    assert (f"{root_uri}/new_file.txt", f"{target_uri}/new_file.txt") in moved_files
    assert (f"{root_uri}/modified.txt", f"{target_uri}/modified.txt") in moved_files

    assert (target_uri, lock_id) in mock_lock_manager.release_calls


@pytest.mark.asyncio
async def test_create_sync_diff_callback_with_directories(mock_viking_fs, processor, monkeypatch):
    """Test sync diff callback with nested directories."""
    root_uri = "viking://user/test/root"
    target_uri = "viking://user/test/target"
    lock_id = "test-lock-456"

    mock_viking_fs.setup_tree(
        {
            root_uri: [
                {"name": "new_dir", "isDir": True},
                {"name": "common_dir", "isDir": True},
            ],
            f"{root_uri}/new_dir": [
                {"name": "file_in_new.txt", "isDir": False},
            ],
            f"{root_uri}/common_dir": [
                {"name": "file.txt", "isDir": False},
            ],
            target_uri: [
                {"name": "common_dir", "isDir": True},
                {"name": "old_dir", "isDir": True},
            ],
            f"{target_uri}/common_dir": [
                {"name": "file.txt", "isDir": False},
            ],
            f"{target_uri}/old_dir": [
                {"name": "old_file.txt", "isDir": False},
            ],
        },
        {
            f"{root_uri}/new_dir/file_in_new.txt": b"new",
            f"{root_uri}/common_dir/file.txt": b"same",
            f"{target_uri}/common_dir/file.txt": b"same",
            f"{target_uri}/old_dir/old_file.txt": b"old",
        }
    )

    mock_lock_manager = MockLockManager()
    monkeypatch.setattr(
        "openviking.resource.resource_lock.ResourceLockManager",
        lambda agfs: mock_lock_manager
    )
    mock_viking_fs.agfs = "mock_agfs"

    callback = processor._create_sync_diff_callback(root_uri, target_uri, lock_id)
    await callback()

    assert (target_uri, lock_id) in mock_lock_manager.release_calls


@pytest.mark.asyncio
async def test_create_sync_diff_callback_error_handling(mock_viking_fs, processor, monkeypatch):
    """Test that callback handles errors gracefully."""
    root_uri = "viking://user/test/root"
    target_uri = "viking://user/test/target"
    lock_id = "test-lock-789"

    mock_viking_fs.setup_tree({})

    mock_lock_manager = MockLockManager()
    monkeypatch.setattr(
        "openviking.resource.resource_lock.ResourceLockManager",
        lambda agfs: mock_lock_manager
    )
    mock_viking_fs.agfs = "mock_agfs"

    callback = processor._create_sync_diff_callback(root_uri, target_uri, lock_id)
    
    await callback()

    assert (target_uri, lock_id) in mock_lock_manager.release_calls


@pytest.mark.asyncio
async def test_diff_result_dataclass():
    """Test DiffResult dataclass behavior."""
    diff = DiffResult(
        added_files=["file1.txt"],
        deleted_files=["file2.txt"],
        updated_files=["file3.txt"],
        added_dirs=["dir1"],
        deleted_dirs=["dir2"],
    )

    assert diff.added_files == ["file1.txt"]
    assert diff.deleted_files == ["file2.txt"]
    assert diff.updated_files == ["file3.txt"]
    assert diff.added_dirs == ["dir1"]
    assert diff.deleted_dirs == ["dir2"]

    empty_diff = DiffResult()
    assert empty_diff.added_files == []
    assert empty_diff.deleted_files == []
    assert empty_diff.updated_files == []
    assert empty_diff.added_dirs == []
    assert empty_diff.deleted_dirs == []


@pytest.mark.asyncio
async def test_create_sync_diff_callback_same_uri(mock_viking_fs, processor, monkeypatch):
    """Test that callback skips diff comparison when root_uri equals target_uri."""
    same_uri = "viking://user/test/same_dir"
    lock_id = "test-lock-same-uri"

    mock_viking_fs.setup_tree(
        {
            same_uri: [
                {"name": "file1.txt", "isDir": False},
                {"name": "file2.txt", "isDir": False},
            ],
        },
        {
            f"{same_uri}/file1.txt": b"content1",
            f"{same_uri}/file2.txt": b"content2",
        }
    )

    mock_lock_manager = MockLockManager()
    monkeypatch.setattr(
        "openviking.resource.resource_lock.ResourceLockManager",
        lambda agfs: mock_lock_manager
    )
    mock_viking_fs.agfs = "mock_agfs"

    callback = processor._create_sync_diff_callback(same_uri, same_uri, lock_id)
    await callback()

    assert len(mock_viking_fs.rm_calls) == 0, "Should not delete any files when URIs are the same"
    assert len(mock_viking_fs.mv_calls) == 0, "Should not move any files when URIs are the same"

    assert (same_uri, lock_id) in mock_lock_manager.release_calls, \
        "Should still release lock even when URIs are the same"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
