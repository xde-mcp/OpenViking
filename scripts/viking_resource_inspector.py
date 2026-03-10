#!/usr/bin/env python3
"""
Viking Resource Inspector - 列出 Viking FS 中 resource URI 对应的 AGFS 信息和向量索引信息

用法:
    python viking_resource_inspector.py <resource_uri> [--config <config_file>]

示例:
    python viking_resource_inspector.py viking://resources/my-project/docs/
    python viking_resource_inspector.py viking://resources/ --config config.yaml
"""

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pydantic import BaseModel, Field


class S3Config(BaseModel):
    """S3 后端配置"""
    bucket: Optional[str] = None
    region: Optional[str] = None
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    endpoint: Optional[str] = None
    prefix: str = ""
    use_ssl: bool = True
    use_path_style: bool = True


class AGFSConfig(BaseModel):
    """AGFS 配置"""
    path: str = "./data"
    port: int = 1833
    url: str = "http://localhost:1833"
    mode: str = "binding-client"
    backend: str = "local"
    timeout: int = 10
    log_level: str = "warn"
    lib_path: Optional[str] = None
    s3: S3Config = Field(default_factory=S3Config)


class VolcengineConfig(BaseModel):
    """火山引擎配置"""
    ak: Optional[str] = None
    sk: Optional[str] = None
    region: Optional[str] = None


class VikingDBConfig(BaseModel):
    """VikingDB 私有化部署配置"""
    host: Optional[str] = None
    headers: Dict[str, str] = Field(default_factory=dict)


class VectorDBBackendConfig(BaseModel):
    """向量数据库后端配置"""
    backend: str = "local"
    name: str = "context"
    path: Optional[str] = None
    url: Optional[str] = None
    distance_metric: str = "cosine"
    dimension: int = 0
    sparse_weight: float = 0.0
    volcengine: VolcengineConfig = Field(default_factory=VolcengineConfig)
    vikingdb: VikingDBConfig = Field(default_factory=VikingDBConfig)


@dataclass
class AGFSInfo:
    """AGFS 文件/目录信息"""
    uri: str
    path: str
    name: str
    size: int
    is_dir: bool
    mod_time: str
    mode: Optional[int] = None
    meta: Optional[Dict[str, Any]] = None
    children: List["AGFSInfo"] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "uri": self.uri,
            "path": self.path,
            "name": self.name,
            "size": self.size,
            "is_dir": self.is_dir,
            "mod_time": self.mod_time,
        }
        if self.mode is not None:
            result["mode"] = self.mode
        if self.meta:
            result["meta"] = self.meta
        if self.children:
            result["children"] = [c.to_dict() for c in self.children]
        return result


@dataclass
class VectorIndexInfo:
    """向量索引信息"""
    id: str
    uri: str
    parent_uri: Optional[str]
    context_type: str
    is_leaf: bool
    level: int
    abstract: Optional[str] = None
    name: Optional[str] = None
    created_at: Optional[str] = None
    active_count: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "id": self.id,
            "uri": self.uri,
            "parent_uri": self.parent_uri,
            "context_type": self.context_type,
            "is_leaf": self.is_leaf,
            "level": self.level,
        }
        if self.abstract:
            result["abstract"] = self.abstract[:200] + "..." if len(self.abstract) > 200 else self.abstract
        if self.name:
            result["name"] = self.name
        if self.created_at:
            result["created_at"] = self.created_at
        if self.active_count is not None:
            result["active_count"] = self.active_count
        return result


class VikingResourceInspector:
    """Viking 资源检查器"""
    
    DEFAULT_ACCOUNT_ID = "default"
    
    def __init__(
        self,
        agfs_config: AGFSConfig,
        vector_config: Optional[VectorDBBackendConfig] = None,
        account_id: Optional[str] = None,
    ):
        self.agfs_config = agfs_config
        self.vector_config = vector_config
        self.account_id = account_id or self.DEFAULT_ACCOUNT_ID
        self._agfs_client = None
        self._vector_backend = None
    
    def _get_agfs_client(self):
        """获取 AGFS 客户端"""
        if self._agfs_client is None:
            from openviking.utils.agfs_utils import create_agfs_client
            self._agfs_client = create_agfs_client(self.agfs_config)
        return self._agfs_client
    
    def _get_vector_backend(self):
        """获取向量索引后端"""
        if self._vector_backend is None and self.vector_config:
            from openviking.storage.viking_vector_index_backend import VikingVectorIndexBackend
            self._vector_backend = VikingVectorIndexBackend(self.vector_config)
        return self._vector_backend
    
    def _uri_to_path(self, uri: str) -> str:
        """将 Viking URI 转换为 AGFS 路径
        
        viking://resources/docs -> /local/{account_id}/resources/docs
        """
        if uri.startswith("viking://"):
            remainder = uri[len("viking://"):].strip("/")
            if not remainder:
                return f"/local/{self.account_id}"
            return f"/local/{self.account_id}/{remainder}"
        elif uri.startswith("/"):
            return uri
        else:
            return f"/local/{self.account_id}/{uri}"
    
    def _path_to_uri(self, path: str) -> str:
        """将 AGFS 路径转换为 Viking URI
        
        /local/{account_id}/resources/docs -> viking://resources/docs
        """
        if path.startswith("viking://"):
            return path
        elif path.startswith("/local/"):
            inner = path[7:].strip("/")
            if not inner:
                return "viking://"
            parts = [p for p in inner.split("/") if p]
            if parts and parts[0] == self.account_id:
                parts = parts[1:]
            if not parts:
                return "viking://"
            return f"viking://{'/'.join(parts)}"
        else:
            return f"viking://{path}"
    
    def get_agfs_info(self, uri: str, recursive: bool = False, depth: int = 0) -> Optional[AGFSInfo]:
        """获取 AGFS 信息"""
        client = self._get_agfs_client()
        path = self._uri_to_path(uri)
        
        try:
            stat_info = client.stat(path)
        except Exception as e:
            print(f"  [AGFS] 无法获取路径信息: {path}, 错误: {e}")
            return None
        
        agfs_info = AGFSInfo(
            uri=uri,
            path=path,
            name=stat_info.get("name", Path(path).name),
            size=stat_info.get("size", 0),
            is_dir=stat_info.get("isDir", False),
            mod_time=stat_info.get("modTime", ""),
            mode=stat_info.get("mode"),
            meta=stat_info.get("meta"),
        )
        
        if agfs_info.is_dir and recursive:
            try:
                entries = client.ls(path)
                for entry in entries:
                    name = entry.get("name", "")
                    if name in [".", ".."]:
                        continue
                    child_uri = self._path_to_uri(f"{path}/{name}")
                    child_info = self.get_agfs_info(child_uri, recursive=True, depth=depth + 1)
                    if child_info:
                        agfs_info.children.append(child_info)
            except Exception as e:
                print(f"  [AGFS] 无法列出目录: {path}, 错误: {e}")
        
        return agfs_info
    
    async def get_vector_index_info(self, uri: str, include_children: bool = True) -> List[VectorIndexInfo]:
        """获取向量索引信息"""
        backend = self._get_vector_backend()
        if backend is None:
            return []
        
        results = []
        
        try:
            print(f"[VectorDB] 查询向量索引: account_id={self.account_id}, uri={uri}")
            records = await backend.get_context_by_uri(
                account_id=self.account_id,
                uri=uri,
                limit=10,
            )
            
            for record in records:
                info = VectorIndexInfo(
                    id=record.get("id", ""),
                    uri=record.get("uri", ""),
                    parent_uri=record.get("parent_uri"),
                    context_type=record.get("context_type", "resource"),
                    is_leaf=record.get("is_leaf", True),
                    level=record.get("level", 2),
                    abstract=record.get("abstract"),
                    name=record.get("name"),
                    created_at=record.get("created_at"),
                    active_count=record.get("active_count"),
                )
                results.append(info)
            
            if include_children and records:
                from openviking.storage.expr import And, In, Eq, PathScope
                child_records = await backend.filter(
                    filter=PathScope("uri", uri, depth=-1),
                    limit=100,
                )
                
                seen_ids = {r.id for r in results}
                for record in child_records:
                    if record.get("id") in seen_ids:
                        continue
                    info = VectorIndexInfo(
                        id=record.get("id", ""),
                        uri=record.get("uri", ""),
                        parent_uri=record.get("parent_uri"),
                        context_type=record.get("context_type", "resource"),
                        is_leaf=record.get("is_leaf", True),
                        level=record.get("level", 2),
                        abstract=record.get("abstract"),
                        name=record.get("name"),
                        created_at=record.get("created_at"),
                        active_count=record.get("active_count"),
                    )
                    results.append(info)
                    
        except Exception as e:
            print(f"  [VectorIndex] 查询向量索引失败: {e}")
        
        return results
    
    async def inspect(
        self,
        uri: str,
        recursive: bool = False,
        include_vector: bool = True,
    ) -> Dict[str, Any]:
        """检查资源 URI 的 AGFS 和向量索引信息"""
        result = {
            "uri": uri,
            "account_id": self.account_id,
            "agfs": None,
            "vector_index": [],
        }
        
        print(f"\n{'='*60}")
        print(f"检查资源 URI: {uri}")
        print(f"Account ID: {self.account_id}")
        print(f"AGFS 路径: {self._uri_to_path(uri)}")
        print(f"{'='*60}\n")
        
        print("[1] 获取 AGFS 信息...")
        agfs_info = self.get_agfs_info(uri, recursive=recursive)
        if agfs_info:
            result["agfs"] = agfs_info.to_dict()
            self._print_agfs_info(agfs_info, indent=0)
        else:
            print("  未找到 AGFS 信息")
        
        print()
        
        if include_vector and self.vector_config:
            print("[2] 获取向量索引信息...")
            vector_info = await self.get_vector_index_info(uri, include_children=recursive)
            result["vector_index"] = [v.to_dict() for v in vector_info]
            self._print_vector_info(vector_info)
        else:
            print("[2] 跳过向量索引查询 (未配置或已禁用)")
        
        return result
    
    def _print_agfs_info(self, info: AGFSInfo, indent: int = 0):
        """打印 AGFS 信息"""
        prefix = "  " * indent
        type_str = "目录" if info.is_dir else "文件"
        size_str = self._format_size(info.size)
        
        print(f"{prefix}├── [{type_str}] {info.name}")
        print(f"{prefix}│   路径: {info.path}")
        print(f"{prefix}│   URI: {info.uri}")
        print(f"{prefix}│   大小: {size_str}")
        print(f"{prefix}│   修改时间: {info.mod_time}")
        
        if info.meta:
            backend_type = info.meta.get("Type", "unknown")
            print(f"{prefix}│   后端类型: {backend_type}")
        
        for child in info.children:
            self._print_agfs_info(child, indent + 1)
    
    def _print_vector_info(self, infos: List[VectorIndexInfo]):
        """打印向量索引信息"""
        if not infos:
            print("  未找到向量索引记录")
            return
        
        print(f"  找到 {len(infos)} 条向量索引记录:\n")
        
        for i, info in enumerate(infos, 1):
            level_str = {0: "L0(摘要)", 1: "L1(概览)", 2: "L2(内容)"}.get(info.level, f"L{info.level}")
            type_str = {"resource": "资源", "memory": "记忆", "skill": "技能"}.get(info.context_type, info.context_type)
            
            print(f"  [{i}] ID: {info.id[:16]}...")
            print(f"      URI: {info.uri}")
            print(f"      类型: {type_str} | 层级: {level_str} | 叶子节点: {info.is_leaf}")
            
            if info.parent_uri:
                print(f"      父 URI: {info.parent_uri}")
            if info.name:
                print(f"      名称: {info.name}")
            if info.abstract:
                abstract_preview = info.abstract[:100] + "..." if len(info.abstract) > 100 else info.abstract
                print(f"      摘要: {abstract_preview}")
            if info.active_count is not None:
                print(f"      活跃计数: {info.active_count}")
            print()
    
    def _format_size(self, size: int) -> str:
        """格式化文件大小"""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size < 1024:
                return f"{size:.2f} {unit}"
            size /= 1024
        return f"{size:.2f} PB"
    
    async def close(self):
        """关闭资源"""
        if self._vector_backend:
            await self._vector_backend.close()


def load_config_from_env() -> tuple[AGFSConfig, Optional[VectorDBBackendConfig]]:
    """从环境变量加载配置"""
    agfs_config = AGFSConfig(
        path=os.getenv("VIKING_AGFS_PATH"),
        url=os.getenv("VIKING_AGFS_URL", "http://localhost:1833"),
        mode=os.getenv("VIKING_AGFS_MODE", "binding-client"),
        backend=os.getenv("VIKING_AGFS_BACKEND", "local"),
    )
    
    vector_config = None
    vector_backend = os.getenv("VIKING_VECTOR_BACKEND", "local")
    
    if vector_backend:
        vector_config = VectorDBBackendConfig(
            backend=vector_backend,
            path=os.getenv("VIKING_VECTOR_PATH"),
            url=os.getenv("VIKING_VECTOR_URL"),
            dimension=int(os.getenv("VIKING_VECTOR_DIMENSION", "1024")),
        )
        
        if vector_backend == "volcengine":
            vector_config.volcengine = VolcengineConfig(
                ak=os.getenv("VIKING_VOLCENGINE_AK"),
                sk=os.getenv("VIKING_VOLCENGINE_SK"),
                region=os.getenv("VIKING_VOLCENGINE_REGION"),
            )
        elif vector_backend == "vikingdb":
            vector_config.vikingdb = VikingDBConfig(
                host=os.getenv("VIKING_VIKINGDB_HOST"),
            )
    
    return agfs_config, vector_config


async def main():
    parser = argparse.ArgumentParser(
        description="Viking Resource Inspector - 列出 Viking FS 中 resource URI 对应的 AGFS 信息和向量索引信息",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s viking://resources/my-project/docs/
  %(prog)s viking://resources/ --recursive
  %(prog)s viking://resources/ --no-vector
  %(prog)s viking://resources/ --account my-account

环境变量:
  VIKING_AGFS_PATH        AGFS 数据存储路径 (默认: ./data)
  VIKING_AGFS_URL         AGFS 服务 URL (默认: http://localhost:1833)
  VIKING_AGFS_MODE        AGFS 客户端模式 (默认: binding-client)
  VIKING_AGFS_LIB_PATH    AGFS binding 库路径 (仅 binding-client 模式)
  VIKING_VECTOR_BACKEND   向量数据库后端 (默认: local)
  VIKING_VECTOR_PATH      本地向量数据库路径
  VIKING_ACCOUNT_ID       账户 ID (默认: default)
        """,
    )
    
    parser.add_argument(
        "uri",
        help="要检查的 Viking resource URI (例如: viking://resources/my-project/)",
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="递归列出子目录/子记录",
    )
    parser.add_argument(
        "--no-vector",
        action="store_true",
        help="不查询向量索引信息",
    )
    parser.add_argument(
        "--account",
        default=os.getenv("VIKING_ACCOUNT_ID", "default"),
        help="账户 ID (默认: default)",
    )
    parser.add_argument(
        "--agfs-url",
        default=os.getenv("VIKING_AGFS_URL", "http://localhost:1833"),
        help="AGFS 服务 URL",
    )
    parser.add_argument(
        "--agfs-path",
        default=os.getenv("VIKING_AGFS_PATH", "./data"),
        help="AGFS 数据存储路径 (默认: ./data)",
    )
    parser.add_argument(
        "--agfs-mode",
        default=os.getenv("VIKING_AGFS_MODE", "binding-client"),
        choices=["http-client", "binding-client"],
        help="AGFS 客户端模式",
    )
    parser.add_argument(
        "--agfs-lib-path",
        default=os.getenv("VIKING_AGFS_LIB_PATH"),
        help="AGFS binding 库路径 (仅 binding-client 模式)",
    )
    parser.add_argument(
        "--vector-backend",
        default=os.getenv("VIKING_VECTOR_BACKEND", "local"),
        help="向量数据库后端类型",
    )
    parser.add_argument(
        "--vector-path",
        default=os.getenv("VIKING_VECTOR_PATH"),
        help="本地向量数据库路径",
    )
    parser.add_argument(
        "-o", "--output",
        help="输出 JSON 文件路径",
    )
    parser.add_argument(
        "-j", "--json",
        action="store_true",
        help="以 JSON 格式输出结果",
    )
    
    args = parser.parse_args()
    
    agfs_config = AGFSConfig(
        path=args.agfs_path,
        url=args.agfs_url,
        mode=args.agfs_mode,
        lib_path=args.agfs_lib_path,
    )
    
    vector_config = None
    if not args.no_vector and args.vector_backend:
        vector_config = VectorDBBackendConfig(
            backend=args.vector_backend,
            path=args.vector_path,
        )
    
    inspector = VikingResourceInspector(
        agfs_config=agfs_config,
        vector_config=vector_config,
        account_id=args.account,
    )
    
    try:
        result = await inspector.inspect(
            uri=args.uri,
            recursive=args.recursive,
            include_vector=not args.no_vector,
        )
        
        if args.json:
            print("\n" + json.dumps(result, indent=2, ensure_ascii=False))
        
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\n结果已保存到: {args.output}")
            
    finally:
        await inspector.close()


if __name__ == "__main__":
    asyncio.run(main())
