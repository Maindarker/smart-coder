"""运行时工作区：多项目注册表 + 按任务隔离 + 语言自动识别。

为什么需要这个模块
------------------
`config.workspace_root` 是 **import 时快照**的模块级常量，而
`tools.MEMORY_FILE` / `agent.CHECKPOINT_DB` / `rag.CHROMA_DIR` 都在 import 时
由它派生。只要工作区还是"启动时常量"，界面上加多少项目都没用——必须重启。
所以这里把工作区升级为 **运行时可解析 + 可按任务绑定** 的值：

  1. 项目清单持久化在 agent 自己目录（`<agent>/.agent_cache/projects.json`），
     不往用户仓库里写任何东西；每个项目的 agent 状态放
     `<agent>/.agent_cache/projects/<项目id>/`，互不污染。
  2. 解析优先级：contextvar（当前任务绑定）> 注册表 current > `WORKSPACE_ROOT` 回退。
     contextvar 是线程/协程安全的，所以同一进程里并发跑多个不同项目不会串
     （webapp.py 的 TaskHandle 是每任务一个线程 → 每个线程一个 context）。
  3. 语言自动识别 + 测试命令推断，让 run_test / RAG 不再假设 Python。

对外主要接口
------------
    workspace.current()            -> Path     # 当前该操作哪个代码库
    workspace.current_id()         -> str      # 稳定 id，用于状态命名空间/thread 前缀
    workspace.bind(path)           -> 上下文管理器，任务内绑定工作区
    workspace.state_dir()          -> Path     # 当前项目的 agent 状态目录
    workspace.detect_project(path) -> dict     # 轻量识别（供目录浏览列表用）
    workspace.describe_project(path) -> dict   # 深度识别（清单+扩展名统计+测试命令）
    workspace.iter_files(...)      -> Iterator[Path]  # 全项目共享的文件遍历（统一 skip 规则）
"""
from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import os
import re
import shutil
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

from config import ROOT, settings

# ---------------- 路径常量 ----------------

STATE_ROOT = ROOT / ".agent_cache"                 # agent 自己的状态根目录
PROJECTS_FILE = STATE_ROOT / "projects.json"       # 项目清单
PROJECTS_STATE_DIR = STATE_ROOT / "projects"       # 每项目状态命名空间
CHECKPOINT_DB = STATE_ROOT / "checkpoints.sqlite"  # 所有项目共用，靠 thread_id 前缀隔离

_current_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "active_workspace", default=None
)
_lock = threading.RLock()

# ---------------- 文件遍历：统一的 skip 规则 ----------------

# 依赖树 / 构建产物 / 缓存目录。这些名字在任何语言里都是"不该搜"的。
# 注意：这里【没有】bin / lib / include / share —— 它们是 Go/C/C++/Rust 项目的
# 真实源码目录，早期版本把它们当成本仓库虚拟环境的目录一起跳过了，属于 bug。
SKIP_DIR_NAMES = {
    # VCS / 编辑器
    ".git", ".hg", ".svn", ".idea", ".vscode",
    # Python
    "__pycache__", ".venv", "venv", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    ".tox", ".nox", "site-packages", "dist-packages", ".eggs",
    # JS / TS
    "node_modules", ".next", ".nuxt", ".svelte-kit", ".turbo", "bower_components",
    # 构建产物（跨语言）
    "dist", "build", "out", "target", "coverage", "htmlcov", ".gradle",
    # 依赖 / 工具链
    "vendor", ".dart_tool", "Pods", "DerivedData", ".terraform", ".serverless",
    # agent 自身状态
    "chroma", ".agent_cache", ".cache",
}

# 搜索时应跳过的文件名/后缀（锁文件、压缩产物、sourcemap 等，量大且无信息量）
_SKIP_FILE_NAMES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "composer.lock",
    "Cargo.lock", "poetry.lock", "Pipfile.lock", "go.sum", "Gemfile.lock",
}
_SKIP_FILE_SUFFIXES = (".min.js", ".min.css", ".map", ".pyc", ".pyo", ".so", ".dylib",
                       ".dll", ".class", ".jar", ".whl", ".lock")


# 虚拟环境内部的目录名。**只在工作区根目录本身就是虚拟环境时**才跳这些名字，
# 否则会把 Go/C/C++ 项目的 bin/ lib/ include/ 源码当成 venv 内容跳掉（早期版本的 bug）。
VENV_INTERNAL_DIRS = {"bin", "lib", "include",
                      "share", "Scripts", "Lib", "lib64"}


def _is_venv_dir(p: Path) -> bool:
    """识别虚拟环境目录：根目录有 pyvenv.cfg / Scripts\\activate 即为 venv。"""
    return (p / "pyvenv.cfg").exists() or (p / "Scripts" / "activate").exists()


def iter_files(
    root: Path | str,
    *,
    exts: set[str] | None = None,
    max_bytes: int | None = None,
    limit: int | None = None,
):
    """遍历 root 下的文件，按 SKIP_DIR_NAMES / venv 识别剪枝后逐个 yield。

    用 os.walk + 原地修改 dirnames 做剪枝，比 rglob 后再过滤快得多
    （不会真的走进 node_modules / .venv）。

    exts:      只保留这些后缀（含点，如 {".py", ".ts"}）；None = 全部
    max_bytes: 超过则跳过（不受 limit 影响）
    limit:     最多 yield 多少个文件，防止在超大仓库上失控
    """
    root = Path(root)
    if not root.is_dir():
        return
    # 工作区本身是 venv（例如 agent 拿自己当工作区）时，才额外跳过 venv 内部目录
    skip = SKIP_DIR_NAMES | (
        VENV_INTERNAL_DIRS if _is_venv_dir(root) else set())
    yielded = 0
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        kept: list[str] = []
        for name in dirnames:
            if name in skip:
                continue
            if _is_venv_dir(here / name):
                continue
            kept.append(name)
        dirnames[:] = sorted(kept)          # 原地剪枝 + 稳定顺序

        for name in sorted(filenames):
            if name in _SKIP_FILE_NAMES or name.endswith(_SKIP_FILE_SUFFIXES):
                continue
            p = here / name
            if exts is not None and p.suffix.lower() not in exts:
                continue
            if max_bytes is not None:
                try:
                    if p.stat().st_size > max_bytes:
                        continue
                except OSError:
                    continue
            yield p
            yielded += 1
            if limit is not None and yielded >= limit:
                return


# ---------------- 项目识别 ----------------

# 语言 -> 清单文件（按此顺序决定 primary，越靠前越优先）
MANIFESTS: dict[str, list[str]] = {
    "node": ["package.json"],
    "python": ["pyproject.toml", "setup.py", "requirements.txt", "Pipfile", "poetry.lock"],
    "go": ["go.mod"],
    "rust": ["Cargo.toml"],
    "java": ["pom.xml", "build.gradle", "build.gradle.kts"],
    "ruby": ["Gemfile"],
    "php": ["composer.json"],
    "dotnet": ["*.csproj", "*.fsproj", "*.sln"],
    "cpp": ["CMakeLists.txt"],
}

# 后缀 -> 语言（用于"清单缺失时按源码构成猜"和 RAG 索引范围）
EXT_LANG: dict[str, str] = {
    ".py": "python", ".pyi": "python",
    ".js": "node", ".mjs": "node", ".cjs": "node", ".jsx": "node",
    ".ts": "node", ".tsx": "node", ".vue": "node", ".svelte": "node",
    ".go": "go",
    ".rs": "rust",
    ".java": "java", ".kt": "java", ".kts": "java", ".scala": "java",
    ".rb": "ruby", ".erb": "ruby",
    ".php": "php",
    ".cs": "dotnet", ".fs": "dotnet", ".vb": "dotnet",
    ".c": "cpp", ".h": "cpp", ".cc": "cpp", ".cpp": "cpp", ".hpp": "cpp",
    ".cxx": "cpp", ".hxx": "cpp",
    ".swift": "swift", ".m": "objc", ".mm": "objc",
    ".sh": "shell", ".bash": "shell", ".zsh": "shell", ".ps1": "shell",
    ".sql": "sql", ".lua": "lua", ".dart": "dart", ".ex": "elixir", ".exs": "elixir",
    ".hs": "haskell", ".clj": "clojure", ".erl": "erlang",
    ".html": "web", ".css": "web", ".scss": "web", ".sass": "web", ".less": "web",
}

# 可检索/可索引的源码后缀（RAG 与 search_code 的默认范围）
SOURCE_EXTS: set[str] = set(EXT_LANG) | {
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".env",
    ".md", ".rst", ".txt", ".proto", ".graphql", ".gql", ".tf", ".tfvars",
    ".dockerfile", ".mk", ".cmake", ".gradle", ".properties", ".xml",
}

#: 单文件超过这个大小就不读（超大生成文件/压缩文件）
MAX_TEXT_BYTES = 200_000


def _slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (s or "project")[:24]


def project_id_for(path: Path | str) -> str:
    """由绝对路径推导稳定 id：同名目录重加回来仍复用同一份状态。"""
    real = Path(path).expanduser().resolve()
    digest = hashlib.sha1(str(real).encode("utf-8")).hexdigest()[:8]
    return f"{_slugify(real.name)}-{digest}"


def _matches_manifest(d: Path, pattern: str) -> list[str]:
    if "*" in pattern:
        return sorted(p.name for p in d.glob(pattern))
    return [pattern] if (d / pattern).exists() else []


def detect_project(path: Path | str) -> dict:
    """轻量识别：只看清单文件是否存在（O(十几个 stat)），供目录浏览列表逐项打标。"""
    d = Path(path).expanduser()
    langs: list[str] = []
    manifests: list[str] = []
    for lang, patterns in MANIFESTS.items():
        found: list[str] = []
        for pat in patterns:
            found.extend(_matches_manifest(d, pat))
        if found:
            langs.append(lang)
            manifests.extend(found)
    is_repo = (d / ".git").exists()
    return {
        "lang": langs[0] if langs else ("git" if is_repo else "generic"),
        "langs": langs,
        "manifests": manifests,
        "is_repo": is_repo,
        "is_project": bool(langs or is_repo),
    }


def _ext_histogram(path: Path, limit: int = 20_000) -> dict[str, int]:
    """源码后缀直方图（受 limit 保护），用于清单缺失时判断项目类型。"""
    hist: dict[str, int] = {}
    for p in iter_files(path, exts=SOURCE_EXTS, limit=limit):
        hist[p.suffix.lower()] = hist.get(p.suffix.lower(), 0) + 1
    return hist


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _makefile_targets(path: Path) -> list[str]:
    """粗略抽取 Makefile 里的 target（用于推断 make test / make build）。"""
    f = path / "Makefile"
    if not f.is_file():
        return []
    targets: list[str] = []
    try:
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines()[:500]:
            m = re.match(r"^([A-Za-z0-9_.\-]+)\s*:(?!=)", line)
            if m and not m.group(1).startswith("."):
                targets.append(m.group(1))
    except OSError:
        return []
    return targets[:40]


def test_command(info: dict) -> str | None:
    """按项目类型推断"跑测试"的命令；推断不出来返回 None（让模型改用 run_shell）。"""
    lang = info.get("lang")
    path = Path(info["path"])
    scripts = info.get("scripts") or {}

    if lang == "node":
        if "test" in scripts:
            return "npm test"
        if "build" in scripts:
            return "npm run build"          # 没有测试脚本时退化为构建（也能暴露错误）
        return None
    if lang == "python":
        has_tests = (path / "tests").is_dir() or (path / "test").is_dir()
        if not has_tests and not (path / "pyproject.toml").exists():
            has_tests = any(
                p.name.startswith("test_") and p.suffix == ".py"
                for p in iter_files(path, exts={".py"}, limit=400)
            )
        return "python -m pytest -q" if has_tests else None
    if lang == "go":
        return "go test ./..."
    if lang == "rust":
        return "cargo test"
    if lang == "java":
        if (path / "pom.xml").exists():
            return "mvn -q test"
        return "./gradlew test" if (path / "gradlew").exists() else None
    if lang == "ruby":
        return "bundle exec rspec" if (path / "spec").is_dir() else "bundle exec rake test"
    if lang == "php":
        return "composer test" if "test" in scripts else None
    if lang == "dotnet":
        return "dotnet test"

    targets = info.get("make_targets") or []
    for candidate in ("test", "check", "ci"):
        if candidate in targets:
            return f"make {candidate}"
    return None


def describe_project(path: Path | str, *, deep: bool = True) -> dict:
    """深度识别：清单 + 源码构成 + 测试命令 + 构建脚本，告诉模型"这是什么项目、怎么跑"。

    describe_project 工具与 /api/projects 都用它，所以结果要能直接塞进 prompt：
    字段刻意保持扁平、值都可序列化。
    """
    d = Path(path).expanduser()
    info = detect_project(d)
    info["path"] = str(d)
    info["exists"] = d.is_dir()
    info["readme"] = None
    info["scripts"] = {}
    info["make_targets"] = []
    if not d.is_dir():
        info["test_cmd"] = None
        return info

    for name in ("README.md", "README.rst", "README.txt", "readme.md", "README"):
        f = d / name
        if f.is_file():
            try:
                info["readme"] = f.read_text(
                    encoding="utf-8", errors="replace")[:1500]
            except OSError:
                pass
            break

    pkg = _read_json(d / "package.json")
    info["scripts"] = {k: str(v)
                       for k, v in (pkg.get("scripts") or {}).items()}
    info["make_targets"] = _makefile_targets(d)

    if deep:
        hist = _ext_histogram(d)
        info["ext_histogram"] = dict(
            sorted(hist.items(), key=lambda kv: kv[1], reverse=True)[:15]
        )
        # 清单缺失 / 清单语言没源码时，用扩展名直方图纠正 primary
        by_lang: dict[str, int] = {}
        for ext, n in hist.items():
            lang = EXT_LANG.get(ext)
            if lang:
                by_lang[lang] = by_lang.get(lang, 0) + n
        dominant = max(by_lang.items(), key=lambda kv: kv[1])[
            0] if by_lang else None
        declared = info["lang"]
        if dominant and (declared in ("generic", "git") or by_lang.get(declared, 0) == 0):
            info["lang"] = dominant
            info["lang_source"] = "detected"

    info["test_cmd"] = test_command(info)
    return info


# ---------------- 项目注册表 ----------------

def _empty_registry() -> dict:
    return {"current": None, "projects": []}


def _load() -> dict:
    """读注册表；首次使用（或文件损坏）时把 WORKSPACE_ROOT 注册成默认项目。"""
    with _lock:
        data = _read_json(PROJECTS_FILE)
        if not isinstance(data, dict) or "projects" not in data:
            data = _empty_registry()
        data.setdefault("current", None)
        data.setdefault("projects", [])
        if not data["projects"]:
            default = Path(settings.workspace_root).expanduser().resolve()
            if default.is_dir():
                pid = project_id_for(default)
                data["projects"].append({
                    "id": pid, "name": default.name or str(default),
                    "path": str(default), "added_at": _now(),
                })
                data["current"] = pid
                _write(data)
        return data


def _write(data: dict) -> None:
    with _lock:
        STATE_ROOT.mkdir(parents=True, exist_ok=True)
        tmp = PROJECTS_FILE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False,
                       indent=2), encoding="utf-8")
        tmp.replace(PROJECTS_FILE)      # 原子替换，避免写一半被读到


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _find(data: dict, pid: str) -> dict | None:
    return next((p for p in data["projects"] if p["id"] == pid), None)


def _with_meta(p: dict) -> dict:
    """给注册表条目补上运行时信息（是否存在、语言），供 API/前端直接用。"""
    path = Path(p["path"])
    info = detect_project(path)
    return {**p, "exists": path.is_dir(), "lang": info["lang"],
            "langs": info["langs"], "is_repo": info["is_repo"]}


def list_projects() -> dict:
    data = _load()
    return {
        "projects": [_with_meta(p) for p in data["projects"]],
        "current": current_id(),
    }


def normalize_dir(path: str) -> Path:
    """校验一个用户输入的目录路径，返回解析后的绝对路径；非法则抛 ValueError。"""
    if not path or not str(path).strip():
        raise ValueError("路径不能为空")
    p = Path(str(path).strip()).expanduser()
    if not p.is_absolute():
        raise ValueError("请填绝对路径（例如 /Users/you/code/my-project）")
    p = p.resolve()
    if not p.exists():
        raise ValueError(f"目录不存在：{p}")
    if not p.is_dir():
        raise ValueError(f"不是目录：{p}")
    if p == Path(p.anchor):
        raise ValueError("不允许把整个文件系统根目录作为工作区")
    if not os.access(p, os.R_OK):
        raise ValueError(f"目录不可读：{p}")
    return p


def add_project(path: str, name: str | None = None, *, select: bool = True) -> dict:
    """把目录加入项目清单（同路径幂等复用 id），默认立即切为当前项目。"""
    d = normalize_dir(path)
    with _lock:
        data = _load()
        pid = project_id_for(d)
        existing = _find(data, pid)
        if existing is None:
            entry = {"id": pid, "name": (name or "").strip() or d.name or str(d),
                     "path": str(d), "added_at": _now()}
            data["projects"].append(entry)
        else:
            entry = existing
            if name and name.strip():
                entry["name"] = name.strip()
            entry["path"] = str(d)       # 目录被移动后重新指回来
        if select:
            data["current"] = pid
        _write(data)
        return _with_meta(entry)


def remove_project(pid: str) -> bool:
    """从清单移除（不动磁盘上的项目和状态目录，避免误删数据）。"""
    with _lock:
        data = _load()
        before = len(data["projects"])
        data["projects"] = [p for p in data["projects"] if p["id"] != pid]
        if data.get("current") == pid:
            data["current"] = data["projects"][0]["id"] if data["projects"] else None
        changed = len(data["projects"]) != before
        if changed:
            _write(data)
        return changed


def select_project(pid: str) -> dict:
    with _lock:
        data = _load()
        p = _find(data, pid)
        if p is None:
            raise ValueError(f"项目不存在：{pid}")
        data["current"] = pid
        _write(data)
        return _with_meta(p)


def current_project() -> dict | None:
    """当前选中的注册表条目（不含 contextvar 绑定），没有则 None。"""
    with _lock:
        data = _load()
        p = _find(data, data.get("current") or "")
        return _with_meta(p) if p else None


# ---------------- 工作区解析 ----------------

def current() -> Path:
    """当前该操作哪个代码库：contextvar 绑定 > 注册表 current > WORKSPACE_ROOT 回退。"""
    bound = _current_var.get()
    if bound:
        return Path(bound)
    proj = current_project()
    if proj and proj.get("exists"):
        return Path(proj["path"])
    if proj:                              # 注册过但目录没了：仍返回它，让报错信息指对地方
        return Path(proj["path"])
    return Path(settings.workspace_root).expanduser().resolve()


def current_id() -> str:
    """当前工作区的稳定 id，用于状态命名空间与 thread_id 前缀。"""
    bound = _current_var.get()
    if bound:
        return project_id_for(bound)
    proj = current_project()
    if proj:
        return proj["id"]
    return project_id_for(current())


def state_dir(pid: str | None = None) -> Path:
    """当前（或指定）项目的 agent 状态目录，已创建。"""
    d = PROJECTS_STATE_DIR / (pid or current_id())
    d.mkdir(parents=True, exist_ok=True)
    return d


def memory_file() -> Path:
    """【遗留】早期长期记忆文件的位置，按项目隔离。

    长期记忆现已改走 LangGraph Store（见 memory.py）：这个路径只在
    `memory.migrate_from_memory_md()` 里被读取一次，用于把老内容导入 Store。
    新代码不要再往这里写。
    """
    return state_dir() / "memory.md"


def chroma_dir() -> Path:
    """向量库目录：按项目隔离，放在 agent 目录而非用户仓库。"""
    return state_dir() / "chroma"


def thread_key(thread_id: str) -> str:
    """给 thread_id 加项目前缀：一个 sqlite 承载所有项目，graph 无需重编译。"""
    return f"{current_id()}::{thread_id}"


@contextlib.contextmanager
def bind(path: Path | str):
    """在当前线程/上下文中临时绑定工作区，退出时恢复。

    webapp.py 每个任务一个线程，在这里绑定后：即使并发任务跑不同项目，
    各自看到的工作区也是对的（contextvar 按线程隔离，不用加全局锁）。
    """
    token = _current_var.set(str(Path(path).expanduser().resolve()))
    try:
        yield Path(_current_var.get())
    finally:
        _current_var.reset(token)


# ---------------- 旧版状态迁移 ----------------

MIGRATION_MARKER = STATE_ROOT / ".migrated_v2"


def _legacy_root() -> Path:
    """改造前的状态目录：就是当时那个工作区根目录下的 .agent_cache。"""
    return Path(settings.workspace_root).expanduser() / ".agent_cache"


def migrate_legacy_state() -> list[str]:
    """把单项目时代的状态认领到新结构下（一次性、幂等、失败不阻断启动）。

    改造前的布局（状态写在工作区根目录，且会话/记忆都不分项目）：
        <工作区>/.agent_cache/checkpoints.sqlite   thread_id 没有项目前缀
        <工作区>/.agent_cache/memory.md            全局一份记忆
    现在改成 agent 目录下按项目隔离。老数据必须迁到"当时的那个工作区"名下
    （即 settings.workspace_root，也就是注册表里的默认项目），否则升级后
    旧的长期记忆和会话历史会读不到。

    向量库（旧的 <工作区>/chroma）不迁移：索引按新后缀集合重建一次即可，比搬运更可靠。

    返回人类可读的说明列表（供启动日志打印）。
    """
    if MIGRATION_MARKER.exists():
        return []

    notes: list[str] = []
    pid = project_id_for(settings.workspace_root)
    target = PROJECTS_STATE_DIR / pid
    legacy = _legacy_root()
    failed = False

    # 1) 会话历史：旧 workspace 目录下的 sqlite 搬进 agent 目录
    legacy_db = legacy / "checkpoints.sqlite"
    try:
        if legacy_db.is_file() and legacy_db.resolve() != CHECKPOINT_DB.resolve():
            if not CHECKPOINT_DB.exists():
                STATE_ROOT.mkdir(parents=True, exist_ok=True)
                for suffix in ("", "-wal", "-shm"):
                    src = Path(str(legacy_db) + suffix)
                    if src.exists():
                        shutil.move(str(src), str(CHECKPOINT_DB) + suffix)
                notes.append(f"已把旧会话库从 {legacy_db.parent} 迁到 {STATE_ROOT}")
            else:
                notes.append(f"检测到旧会话库 {legacy_db}，但新库已存在，未覆盖（数据留在原处）")
    except OSError as e:
        failed = True
        notes.append(f"旧会话库搬迁失败（不影响使用）：{e}")

    # 2) 长期记忆：旧 memory.md 归到默认项目名下
    try:
        for src in (legacy / "memory.md", STATE_ROOT / "memory.md"):
            dst = target / "memory.md"
            if src.is_file() and src.resolve() != dst.resolve() and not dst.exists():
                target.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
                notes.append(
                    f"已把长期记忆归到项目 {pid} 名下（{len(dst.read_text(encoding='utf-8').splitlines())} 条）")
    except OSError as e:
        failed = True
        notes.append(f"长期记忆迁移失败（不影响使用）：{e}")

    # 3) 会话历史加项目前缀：旧的裸 thread_id（main / me ...）以前属于唯一那个工作区
    if CHECKPOINT_DB.exists():
        try:
            conn = sqlite3.connect(str(CHECKPOINT_DB), timeout=5)
            try:
                bare = [r[0] for r in conn.execute(
                    "SELECT DISTINCT thread_id FROM checkpoints WHERE thread_id NOT LIKE '%::%'")]
                for tid in bare:
                    for table in ("checkpoints", "writes"):
                        conn.execute(
                            f"UPDATE {table} SET thread_id = ? WHERE thread_id = ?",
                            (f"{pid}::{tid}", tid))
                conn.commit()
                if bare:
                    notes.append(f"已把 {len(bare)} 个旧会话（{', '.join(bare[:5])}…）"
                                 f"归到项目 {pid} 名下")
            finally:
                conn.close()
        except sqlite3.Error as e:
            failed = True
            notes.append(f"旧会话加项目前缀失败（不影响使用）：{e}")

    if not failed:
        STATE_ROOT.mkdir(parents=True, exist_ok=True)
        MIGRATION_MARKER.write_text(
            f"{_now()} 迁移完成，项目={pid}\n" + "\n".join(notes) + "\n", encoding="utf-8")
    return notes
