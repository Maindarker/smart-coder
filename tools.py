"""Agent 工具集：文件读写（本机受限）+ 命令执行/测试（沙箱或本机）+ 长期记忆 + 项目识别。

语言无关性：这里不再假设 Python 项目。
- 工作区路径一律走 workspace.current()（运行时可切换、可按任务绑定）
- 文件遍历统一走 workspace.iter_files()，skip 规则单一来源（依赖树/构建产物/venv）
- run_test 按项目类型自动选命令（npm/go/cargo/mvn/pytest/make…），也可显式覆盖
- describe_project 让模型先搞清楚"这是什么项目、怎么跑"，避免瞎猜

安全分层：
- 文件读写：list_files / read_file 仅限当前工作区内，路径越界拒绝
- 命令执行：run_shell / run_test 走 sandbox.run_command —— docker 模式在非 root
  容器内跑（默认断网、限时限内存），host 模式在本机直跑但每条命令都经人工审批
- 长期记忆：remember_fact / recall_memory 落盘到 agent 目录下按项目隔离的 memory.md
"""
import datetime
from pathlib import Path

from langchain_core.tools import tool

import workspace
from sandbox import CONTAINER_WORKDIR, effective_mode, run_command


def _safe_path(p: str) -> Path:
    """把相对路径解析到当前工作区内，越界则拒绝（工作区是运行时可变的）。"""
    root = workspace.current().resolve()
    path = (root / p).resolve()
    if not path.is_relative_to(root):
        raise PermissionError(f"路径越界: {p}")
    return path


def _fmt(r: dict, *, show_mode: bool = False) -> str:
    """把执行结果格式化为给模型的文本。"""
    parts = []
    if r.get("stdout"):
        parts.append(r["stdout"].rstrip())
    if r.get("stderr"):
        parts.append("[stderr]\n" + r["stderr"].rstrip())
    tail = f"[exit_code={r.get('exit_code')}]"
    if r.get("timed_out"):
        tail += " [timed_out]"
    if show_mode:
        tail += f" [exec_mode={effective_mode()}]"
    parts.append(tail)
    return "\n".join(parts)


@tool
def list_files(path: str = ".") -> str:
    """列出当前工作区内某个目录下的文件与子目录（path 为相对路径）。"""
    root = workspace.current().resolve()
    p = _safe_path(path)
    if not p.is_dir():
        return f"不是目录: {path}"
    items = []
    for x in sorted(p.iterdir(), key=lambda i: i.name):
        if x.name in workspace.SKIP_DIR_NAMES:
            continue
        items.append(str(x.relative_to(root)) + ("/" if x.is_dir() else ""))
    return "\n".join(items) or "(空目录)"


@tool
def read_file(path: str) -> str:
    """读取当前工作区内某个文本文件内容（path 为相对路径）。"""
    p = _safe_path(path)
    if not p.is_file():
        return f"文件不存在: {path}"
    if p.stat().st_size > workspace.MAX_TEXT_BYTES:
        return f"文件过大，跳过: {path}"
    return p.read_text(encoding="utf-8", errors="replace")


@tool
def search_code(query: str, path: str = ".", glob: str = "") -> str:
    """在当前工作区内按关键词搜索代码（纯文本匹配、大小写不敏感、无需本地模型、支持任意语言）。

    轻量模式关闭 RAG 后，这是 agent 定位代码的主要手段；返回 路径:行号: 内容。
    path 可为目录或单个文件；glob 可选，用于限定文件后缀（如 ".ts" 或 ".py,.go"）。
    """
    root = workspace.current().resolve()
    base = _safe_path(path)
    exts: set[str] | None = None
    if glob.strip():
        exts = {
            (g if g.startswith(".") else "." + g).lower()
            for g in glob.replace(" ", "").split(",") if g
        }

    if base.is_file():
        files = [base]
    elif base.is_dir():
        files = workspace.iter_files(
            base, exts=exts, max_bytes=workspace.MAX_TEXT_BYTES
        )
    else:
        return f"路径不存在: {path}"

    needle = query.lower()
    out: list[str] = []
    hits = 0
    for p in files:
        try:
            raw = p.read_bytes()
        except OSError:
            continue
        if b"\x00" in raw[:8192]:        # 疑似二进制，跳过
            continue
        text = raw.decode("utf-8", errors="replace")
        for i, line in enumerate(text.splitlines(), 1):
            if needle in line.lower():
                shown = line.strip()
                if len(shown) > 300:
                    shown = shown[:300] + "…"
                try:
                    shown_path = p.relative_to(root)
                except ValueError:
                    shown_path = p
                out.append(f"{shown_path}:{i}: {shown}")
                hits += 1
                if hits >= 200:
                    out.append("…（结果过多已截断，请缩小关键词范围）")
                    return "\n".join(out)
    return "\n".join(out) if out else "（没有匹配的代码）"


@tool
def run_shell(command: str, allow_network: bool = False) -> str:
    """在当前项目的执行环境里跑 shell 命令（host 模式=本机，docker 模式=沙箱）。

    适合：git status/diff/log、ls、grep、各种语言的编译/构建/包管理器命令等。
    默认断网（docker 模式）或需人工审批（host 模式）；
    需要网络的命令（git clone/push/pull、装依赖）要 allow_network=True，并会被审批。
    """
    r = run_command(command, network=allow_network)
    return _fmt(r, show_mode=True)


@tool
def run_test(path: str = ".", command: str = "") -> str:
    """在当前项目里跑测试。命令按项目类型自动推断，也可用 command 显式指定。

    自动推断：package.json→npm test、go.mod→go test ./...、Cargo.toml→cargo test、
    pyproject/tests→python -m pytest、pom.xml→mvn test、Gemfile→rspec、Makefile→make test。
    path 为工作区内相对路径（在哪个目录里执行，默认项目根）。
    """
    info = workspace.describe_project(workspace.current())
    cmd = (command or "").strip() or info.get("test_cmd")
    if not cmd:
        return (
            f"无法自动推断 {info.get('lang')} 项目的测试命令。"
            f"当前项目：{info.get('path')}，清单文件：{info.get('manifests') or '无'}。"
            "请先调用 describe_project 了解项目结构，再用 run_shell 显式执行测试命令。"
        )
    cwd = f"{CONTAINER_WORKDIR}/{path.strip('/')}" if path.strip("/.") else CONTAINER_WORKDIR
    r = run_command(cmd, cwd=cwd, timeout=600)
    header = f"$ {cmd}"
    if effective_mode() == "docker" and info.get("lang") not in ("python", "generic", "git"):
        header += ("\n⚠️ 当前是 docker 沙箱模式，而沙箱镜像只带 Python 运行时，"
                   f"这条 {info.get('lang')} 命令很可能失败；改用 EXEC_MODE=host 可正常执行。")
    return header + "\n" + _fmt(r, show_mode=True)


@tool
def describe_project(path: str = ".") -> str:
    """了解当前项目是什么、用什么语言、怎么跑测试/构建（动手前先调用它）。

    返回：语言、清单文件、源码构成、测试命令、构建脚本、README 摘要、执行模式。
    """
    base = _safe_path(path) if path.strip("/.") else workspace.current()
    target = base if base.is_dir() else base.parent
    info = workspace.describe_project(target)
    lines = [
        f"路径：{info['path']}",
        f"语言：{info['lang']}"
        + (f"（同时含：{', '.join(info['langs'])}）" if len(info.get("langs") or []) > 1 else "")
        + (f" [按源码构成判断]" if info.get("lang_source") == "detected" else ""),
        f"清单文件：{', '.join(info.get('manifests') or []) or '（无）'}"
        + ("，是 git 仓库" if info.get("is_repo") else ""),
        f"测试命令：{info.get('test_cmd') or '（未能推断，请用 run_shell 指定）'}",
        f"执行模式：{effective_mode()}"
        + ("（沙箱镜像只带 Python 运行时，非 Python 项目建议 EXEC_MODE=host）"
           if effective_mode() == "docker" and info.get("lang") not in ("python", "generic", "git") else ""),
    ]
    if info.get("scripts"):
        lines.append("package.json scripts：" + ", ".join(sorted(info["scripts"])))
    if info.get("make_targets"):
        lines.append("Makefile targets：" + ", ".join(info["make_targets"][:15]))
    if info.get("ext_histogram"):
        hist = ", ".join(f"{k}×{v}" for k, v in info["ext_histogram"].items())
        lines.append("源码后缀分布：" + hist)
    lines.append(f"顶层条目：{', '.join(_top_entries(target))}")
    if info.get("readme"):
        lines.append("README 摘要：\n" + info["readme"][:600])
    return "\n".join(lines)


def _top_entries(d: Path, limit: int = 30) -> list[str]:
    out = []
    try:
        for x in sorted(d.iterdir(), key=lambda i: i.name):
            if x.name in workspace.SKIP_DIR_NAMES:
                continue
            out.append(x.name + ("/" if x.is_dir() else ""))
            if len(out) >= limit:
                break
    except OSError:
        pass
    return out


@tool
def remember_fact(fact: str) -> str:
    """把一条需要长期记住的事实/偏好写入当前项目的记忆库（如用户偏好、项目约定、决定）。

    之后无论哪个会话，只要还在这个项目里，就可用 recall_memory 查回这条事实。
    """
    p = workspace.memory_file()
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with p.open("a", encoding="utf-8") as f:
        f.write(f"- [{ts}] {fact.strip()}\n")
    return f"已记住（项目 {workspace.current_id()}）：{fact.strip()}"


@tool
def recall_memory(query: str = "") -> str:
    """读取当前项目记忆库中的事实/偏好。query 为关键词（留空返回全部）。"""
    p = workspace.memory_file()
    if not p.exists():
        return "（记忆库为空）"
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    if query:
        lines = [l for l in lines if query.lower() in l.lower()]
    return "\n".join(lines) if lines else "（没有匹配的记忆）"


TOOLS = [list_files, read_file, search_code, describe_project,
         run_shell, run_test, remember_fact, recall_memory]
