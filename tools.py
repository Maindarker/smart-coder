"""安全的最小工具集：仅限工作区内读写，不执行任意命令。

命令执行（run_shell / run_test）留给你后续接 Docker 沙箱 —— 见文末 TODO。
"""
from pathlib import Path

from langchain_core.tools import tool

from config import settings

# TODO: 命令执行 / 测试运行 / Git 操作，应在 Docker 沙箱容器内实现，
#       这里刻意只保留「纯文件读写」这一最安全的最小集。


def _safe_path(p: str) -> Path:
    """把相对路径解析到工作区内，越界则拒绝。"""
    root = settings.workspace_root.resolve()
    path = (root / p).resolve()
    if not path.is_relative_to(root):
        raise PermissionError(f"路径越界: {p}")
    return path


@tool
def list_files(path: str = ".") -> str:
    """列出工作区内某个目录下的文件与子目录（path 为相对路径）。"""
    p = _safe_path(path)
    if not p.is_dir():
        return f"不是目录: {path}"
    skip = {".git", "__pycache__", ".venv", "chroma"}
    items = sorted(
        str(x.relative_to(settings.workspace_root))
        for x in p.iterdir()
        if x.name not in skip
    )
    return "\n".join(items) or "(空目录)"


@tool
def read_file(path: str) -> str:
    """读取工作区内某个文本文件内容（path 为相对路径）。"""
    p = _safe_path(path)
    if not p.is_file():
        return f"文件不存在: {path}"
    if p.stat().st_size > 200_000:
        return f"文件过大，跳过: {path}"
    return p.read_text(encoding="utf-8", errors="replace")


TOOLS = [list_files, read_file]
