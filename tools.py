"""Agent 工具集：文件读写（本机受限）+ 命令执行/测试（Docker 沙箱）。

安全分层：
- 文件读写：list_files / read_file 仅限工作区内，路径越界拒绝
- 命令执行：run_shell / run_test 在非 root 的 Docker 沙箱内运行，默认断网、限时限内存
"""
from pathlib import Path

from langchain_core.tools import tool

from config import settings
from sandbox import CONTAINER_WORKDIR, run_command


def _safe_path(p: str) -> Path:
    """把相对路径解析到工作区内，越界则拒绝。"""
    root = settings.workspace_root.resolve()
    path = (root / p).resolve()
    if not path.is_relative_to(root):
        raise PermissionError(f"路径越界: {p}")
    return path


def _fmt(r: dict) -> str:
    """把沙箱执行结果格式化为给模型的文本。"""
    parts = []
    if r.get("stdout"):
        parts.append(r["stdout"].rstrip())
    if r.get("stderr"):
        parts.append("[stderr]\n" + r["stderr"].rstrip())
    tail = f"[exit_code={r.get('exit_code')}]"
    if r.get("timed_out"):
        tail += " [timed_out]"
    parts.append(tail)
    return "\n".join(parts)


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


@tool
def run_shell(command: str, allow_network: bool = False) -> str:
    """在 Docker 沙箱内执行 shell 命令。默认断网、超时 60s、内存 512m。

    适合：git status/diff/log、ls、grep、编译、运行脚本等。
    需要网络的命令（git clone/push/pull、下载依赖）需 allow_network=True，
    且这类危险操作应经过审批。
    """
    r = run_command(command, network=allow_network)
    return _fmt(r)


@tool
def run_test(path: str = ".") -> str:
    """在 Docker 沙箱内运行 pytest 测试。path 为工作区内相对路径（目录或文件）。"""
    cwd = f"{CONTAINER_WORKDIR}/{path.strip('/')}"
    r = run_command("python -m pytest -q", cwd=cwd, timeout=180)
    return _fmt(r)


TOOLS = [list_files, read_file, run_shell, run_test]
