"""内核级围栏（OS confinement）：把命令真正关起来，而不是"看起来危险才拦"。

**为什么需要它**（审计文档 §2.4 之后的结论）：`shellrisk.py` 那类文本判定属于
**consent 层**（决定"要不要问人"），不是 **enforcement 层**（真正拦住）。任何能"算出"
命令的写法（`python -c`、`base64 -d | sh`、变量拼接、别名）都能绕过文本分析。
业界主流做法（Codex CLI / Claude Code / DSH 本身）都是同一个结构：

    内核级围栏做强制执行  +  审批只用于"越界升级"  +  可逆性兜底

本模块就是那个"内核级围栏"，对齐 DSH 的 `dsh-sandbox-local` / Codex 的 `sandbox_mode`：

| 后端 | 平台 | 机制 |
|---|---|---|
| `seatbelt` | macOS | `sandbox-exec`（Apple Seatbelt / SBPL）：allow-default + `(deny file-write*)` + 可写根白名单 |
| `bwrap` | Linux | bubblewrap：只读宿主 root + 私有 /proc + 可写工作区 bind + `--unshare-net` |
| `none` | 其它/不可用 | 只在显式选择 `danger-full-access` 时才会出现 |

**策略词汇只有"文件效果 + 网络"**（与 DSH 一致：它连网络都不管，本模块额外管住网络，
因为这是 Codex 的默认姿态，也更符合本项目的 `allow_network` 语义）：

- `read-only`：一切写入被拒（连工作区也不能写）
- `workspace-write`：只允许写「工作区根 + 平台临时目录」（白名单路径全部 canonicalize，
  因为 Seatbelt 匹配的是**解析后**的真实路径，`/tmp` 实际是 `/private/tmp`）
- `danger-full-access`：不围栏（由审批兜底，必须是人明确放行的升级）
- `network=False`：额外禁网（Seatbelt `(deny network*)` / bwrap `--unshare-net`）

**fail closed**：拿不到可用后端时抛 `SandboxUnavailable`，**拒绝裸跑**，而不是悄悄降级成
无隔离执行（DSH 的 `SANDBOX_UNAVAILABLE` 就是这个立场）。想接受无隔离，必须显式把
`EXEC_MODE` 设成 `host`（那是一条会逐条审批的、明确知情的选择）。
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

#: 文件效果策略（与 DSH 的 SandboxMode 同名同义）
READ_ONLY = "read-only"
WORKSPACE_WRITE = "workspace-write"
DANGER_FULL_ACCESS = "danger-full-access"
MODES = (READ_ONLY, WORKSPACE_WRITE, DANGER_FULL_ACCESS)


class SandboxUnavailable(RuntimeError):
    """请求了围栏但没有可用后端 —— 拒绝在无隔离状态下执行。"""


@dataclass(frozen=True)
class Policy:
    """一次调用的围栏策略：文件效果 + 网络。"""

    mode: str
    workspace: Path
    network: bool = False

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"未知 sandbox mode: {self.mode!r}")


@dataclass(frozen=True)
class Confinement:
    """包装结果：真正要 spawn 的 argv + 由谁提供围栏 + 强制完整度。"""

    argv: list[str]
    backend: str        # "seatbelt" | "bwrap" | "none"
    enforcement: str    # "full" | "partial" | "none"（bwrap 是 full；其它后端可如实降级）
    mode: str
    network: bool
    profile_path: str = ""   # seatbelt 的临时 SBPL 文件（用完请调 cleanup()）

    @property
    def confined(self) -> bool:
        return self.backend != "none"

    def cleanup(self) -> None:
        """删掉临时档案文件（在进程 spawn 完成之后调用即可）。"""
        if self.profile_path:
            try:
                os.unlink(self.profile_path)
            except OSError:
                pass


# ---------------- 可写根 ----------------


def writable_roots(policy: Policy) -> list[str]:
    """`workspace-write` 允许写入的根（canonical、去重、去嵌套）。

    Seatbelt 匹配的是解析后的真实路径（`/tmp` 就是 `/private/tmp`），所以这里必须 canonicalize；
    同时给工作区根保留 `resolve()` 后的形态，符号链接指向的目录才不会被误判成越界。
    """
    if policy.mode == READ_ONLY:
        return []
    if policy.mode == DANGER_FULL_ACCESS:
        return []            # 不围栏，无所谓白名单
    roots: list[Path] = []
    for cand in (policy.workspace, Path("/tmp"), Path(tempfile.gettempdir())):
        try:
            roots.append(Path(cand).resolve())
        except OSError:      # pragma: no cover —— 极端情况下 resolve 失败就跳过
            continue
    seen: list[Path] = []
    for r in roots:
        if not any(r == s or r.is_relative_to(s) for s in seen):
            seen.append(r)
    return [str(r) for r in seen]


#: Seatbelt 里必须放行的字符设备（写 /dev/null 是最常见的无害写入）
_DEV_LITERALS = ("/dev/null", "/dev/stdout", "/dev/stderr", "/dev/tty", "/dev/dtracehelper")


def seatbelt_profile(policy: Policy) -> str:
    """生成 SBPL 档案：allow-default + deny file-write* + 可写根白名单（+ 可选禁网）。"""
    lines = ["(version 1)", "(allow default)", "(deny file-write*)"]
    grants = [f'(literal "{p}")' for p in _DEV_LITERALS]
    grants += [f'(subpath "{r}")' for r in writable_roots(policy)]
    lines.append("(allow file-write*")
    lines += [f"  {g}" for g in grants]
    lines.append(")")
    if not policy.network:
        lines.append("(deny network*)")
    return "\n".join(lines) + "\n"


def bwrap_argv(policy: Policy) -> list[str]:
    """bubblewrap 参数（Linux）：只读宿主 root + 私有 /dev、PID /proc + 可写工作区 bind。

    私有 PID 命名空间还有安全意义：看不到宿主进程，`/proc/<pid>/root` 这类魔术链接
    就没法绕过 bind 挂载（bwrap 官方文档与 DSH 的 bwrap 档案都强调了这一点）。
    """
    ws = str(Path(policy.workspace).resolve())
    argv = [
        "bwrap",
        "--ro-bind", "/", "/",
        "--dev", "/dev",
        "--proc", "/proc",
        "--unshare-pid",
        "--die-with-parent",
        "--new-session",
        "--tmpfs", "/tmp",
        "--bind", ws, ws,
    ]
    if not policy.network:
        argv.append("--unshare-net")
    return argv


# ---------------- 后端选择 ----------------


_probe_cache: dict[str, tuple[str, str] | None] = {}


def _probe(argv: list[str]) -> bool:
    try:
        proc = subprocess.run(argv, capture_output=True, timeout=10)
    except Exception:  # noqa: BLE001 —— 探测失败一律视为不可用
        return False
    return proc.returncode == 0


def select_backend(*, refresh: bool = False) -> tuple[str, str] | None:
    """探测并缓存一个可用后端，返回 (backend, enforcement)；不可用返回 None。

    enforcement：
    - `seatbelt` / `bwrap` 是内核级完整围栏 → "full"
    - 其它情况一律 None（宁可拒绝执行，也不谎报"已隔离"）
    """
    key = sys.platform
    if not refresh and key in _probe_cache:
        return _probe_cache[key]

    verdict: tuple[str, str] | None = None
    if sys.platform == "darwin":
        exe = shutil.which("sandbox-exec") or "/usr/bin/sandbox-exec"
        # 功能探测：能跑通一个空档案才算可用（Apple 已把 sandbox-exec 标记为 deprecated，
        # 所以不能只看文件存在 —— DSH 也是这么做功能探测的）
        if Path(exe).exists() and _probe([exe, "-p", "(version 1)(allow default)", "/usr/bin/true"]):
            verdict = ("seatbelt", "full")
    elif sys.platform.startswith("linux"):
        if shutil.which("bwrap") and _probe([shutil.which("bwrap"), "--ro-bind", "/", "/",
                                             "--dev", "/dev", "--proc", "/proc", "/bin/true"]):
            verdict = ("bwrap", "full")

    _probe_cache[key] = verdict
    return verdict


def backend_note() -> str:
    """给模型/用户看的一行环境说明。"""
    sel = select_backend()
    if sel is None:
        return ("围栏不可用（未找到可用的 sandbox-exec / bubblewrap）——"
                "命令会被拒绝执行，除非显式把 EXEC_MODE 设成 host 接受无隔离")
    return f"{sel[0]} 内核围栏（enforcement={sel[1]}）"


# ---------------- 包装 ----------------


def wrap(argv: list[str], policy: Policy) -> Confinement:
    """把 argv 包成"受围栏约束的 argv"。拿不到后端且策略要求围栏时抛 SandboxUnavailable。"""
    if policy.mode == DANGER_FULL_ACCESS:
        return Confinement(argv=list(argv), backend="none", enforcement="none",
                           mode=policy.mode, network=True)

    sel = select_backend()
    if sel is None:
        raise SandboxUnavailable(
            "请求了内核围栏但没有可用后端：macOS 需要 sandbox-exec，Linux 需要 bubblewrap(bwrap)。"
            "拒绝在无隔离状态下执行命令。"
            "（确实要无隔离运行，请显式设置 EXEC_MODE=host，那是一条逐条审批的知情选择。）"
        )
    backend, enforcement = sel
    if backend == "seatbelt":
        profile = seatbelt_profile(policy)
        fd, path = tempfile.mkstemp(prefix="dsh-sbpl-", suffix=".sb")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(profile)
        return Confinement(argv=["/usr/bin/sandbox-exec", "-f", path, *argv],
                           backend=backend, enforcement=enforcement,
                           mode=policy.mode, network=policy.network, profile_path=path)
    wrapped = bwrap_argv(policy) + ["--", *argv]
    return Confinement(argv=wrapped, backend=backend, enforcement=enforcement,
                       mode=policy.mode, network=policy.network)


#: 内核拒绝的 stderr 特征（用于把"命令本身失败"和"被围栏拦下"区分开的**启发式**提示）
DENIAL_SIGNATURES = (
    "Operation not permitted",
    "sandbox-exec:",
    "bwrap:",
)


def looks_denied(stderr: str, exit_code: int) -> bool:
    """粗略判断这次失败是否来自围栏拒绝（仅用于给模型一个提示，不作为安全判定）。"""
    if exit_code == 0 or not stderr:
        return False
    return any(sig in stderr for sig in DENIAL_SIGNATURES)


def describe(policy: Policy) -> str:
    """一行人类可读的策略描述（进 prompt / 审批弹窗）。"""
    if policy.mode == DANGER_FULL_ACCESS:
        return "无围栏（danger-full-access：命令可读写整机、可联网）"
    roots = writable_roots(policy)
    net = "可联网" if policy.network else "禁网"
    if policy.mode == READ_ONLY:
        return f"只读围栏（整个文件系统禁止写入；{net}）"
    return f"工作区围栏（可写：{'、'.join(roots)}；{net}）"
