"""任务级可逆性快照：给"围栏内的破坏"留一条回滚路。

**这是三件套的第三件**（前两件：`confinement.py` 的内核围栏、`middleware.py` 的越界审批）。
三者合起来才能让 `rm -rf` 这类**只落在工作区内**的破坏不必逐条审批：
命令照跑（有围栏兜住越界），出事了能回滚（有快照兜住工作区）。
只靠围栏不够 —— 围栏允许的范围内照样能删光你的项目；只靠审批也不够 ——
人一疲劳就点"批准"。可逆性把"单次误操作的代价"降到最低，这也是 Aider 之类
"每次改动自动 commit"的思路。

实现（只依赖 git，不新增依赖）：

- **任务开始时**（engine.run_task / main.py 调用 `begin()`）记录：
  `HEAD`、`git status --porcelain`、`git diff HEAD`（含已暂存）、未跟踪文件清单，
  落到 `.agent_cache/snapshots/<项目id>/<时间戳>/`（**不写进被操作的仓库**）。
- **任务结束时**（`finish_after()`）把这次任务**实际改了什么**采集回来：
  `after.patch`（本次改动的完整 patch，可复查/可重放）+ `after_status.txt`，
  并把精简摘要交给 finish 节点 —— 最终汇报里的"改了什么"因此是**机器采集的事实**，
  而不是模型自己回忆的（与审计里"`observed` 类字段必须机器采集"是同一条原则）。
- **回滚**（`restore()`，CLI 用 `python main.py --rollback`）：
  - 已跟踪文件的改动回到任务开始前：`git checkout -- .`
  - 任务开始时**存在**的未提交改动：`git apply <dir>/diff.patch`（两步合起来 = 回到任务开始）
  - 任务期间新建的未跟踪文件：只列出、**不自动删**（交人工确认）

**诚实的局限**：

- 只覆盖 git 工作区；**任务中途新建的未跟踪文件被删掉无法复原**（只留了名字，没有内容备份）；
- 不覆盖 `.gitignore` 排除的内容，也不覆盖仓库外的路径（那些由围栏兜）；
- 这不是备份系统：不做内容级快照（那要复制整棵树），只做"能回到任务开始前"这件事；
- **刻意不自动 commit**：Aider 那套"每次改动自动 commit"会把提交写进用户的仓库与历史，
  与本项目"不往被操作的仓库写东西"的约定冲突。这里用等价物替代：
  任务开始自动快照 + 任务改动留成 patch（`after.patch`）+ 一条命令回滚（`--rollback`）。
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import workspace

#: 采集/展示时的文本上限（超大 diff 只留头尾，避免塞爆模型上下文与日志）
MAX_DIFF_CHARS = 20000


@dataclass
class Snapshot:
    """一次任务快照的结果。`available=False` 时退回"逐条审批"的保守姿态。"""

    available: bool
    path: Path | None = None
    head: str = ""
    changed: int = 0                       # 任务开始时已有的改动文件数
    untracked: list[str] = field(default_factory=list)
    note: str = ""                         # 不可用原因 / 摘要

    def describe(self) -> str:
        if not self.available:
            return f"无快照（{self.note}）"
        bits = [f"快照 {self.path}"]
        if self.head:
            bits.append(f"HEAD={self.head[:8]}")
        bits.append(f"开始前已有改动 {self.changed} 个文件")
        if self.untracked:
            bits.append(f"未跟踪文件 {len(self.untracked)} 个")
        return "；".join(bits)

    def rollback_hint(self) -> str:
        if not self.available:
            return "本次没有可回滚快照"
        return (f"回滚：cd {workspace.current()} && git checkout -- . "
                f"（任务开始前的未提交改动可用 git apply {self.path}/diff.patch 找回）")


def _git(args: list[str], cwd: Path, timeout: int = 20) -> tuple[int, str]:
    """跑一条 git 命令，失败不抛异常（快照是尽力而为的兜底，不该阻断任务）。"""
    try:
        proc = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True,
                              text=True, encoding="utf-8", errors="replace", timeout=timeout)
    except Exception as e:  # noqa: BLE001 —— git 不在 / 超时 / 权限问题
        return -1, str(e)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def is_git_repo(root: Path | None = None) -> bool:
    """工作区是否 git 仓库（判定"可回滚"用；只看标记，不起进程）。"""
    root = root or workspace.current()
    try:
        return (Path(root) / ".git").exists()
    except OSError:
        return False


def rollback_available() -> bool:
    """当前工作区是否具备回滚能力（供审批判定：没有回滚能力就要提高审批门槛）。"""
    return is_git_repo() and shutil.which("git") is not None


def _snapshot_dir() -> Path:
    """按时间戳建目录；同一秒内跑两次任务也不互相覆盖（加 -2/-3 后缀）。"""
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = workspace.STATE_ROOT / "snapshots" / workspace.current_id()
    for suffix in [""] + [f"-{i}" for i in range(2, 100)]:
        d = base / f"{ts}{suffix}"
        try:
            d.mkdir(parents=True, exist_ok=False)
            return d
        except FileExistsError:
            continue
    raise OSError(f"快照目录创建失败：{base}/{ts}")


def begin(label: str = "") -> Snapshot:
    """在任务开始时打一个快照（幂等、失败不阻断）。"""
    root = workspace.current()
    if not is_git_repo(root):
        return Snapshot(False, note="工作区不是 git 仓库，无法做可回滚快照")
    if shutil.which("git") is None:
        return Snapshot(False, note="PATH 里没有 git")

    code, head_out = _git(["rev-parse", "HEAD"], root)
    head = head_out.strip().splitlines()[-1].strip() if code == 0 else ""
    code, status = _git(["status", "--porcelain"], root)
    if code != 0:
        return Snapshot(False, note=f"git status 失败：{status.strip()[:120]}")
    code, diff = _git(["diff", "HEAD"], root)          # 含已暂存 + 未暂存
    if code != 0:
        diff = ""

    lines = [ln for ln in status.splitlines() if ln.strip()]
    untracked = [ln[3:].strip() for ln in lines if ln.startswith("??")]

    d = _snapshot_dir()
    (d / "head.txt").write_text(head + "\n", encoding="utf-8")
    (d / "status.txt").write_text(status, encoding="utf-8")
    (d / "diff.patch").write_text(diff, encoding="utf-8")
    (d / "untracked.txt").write_text("\n".join(untracked) + ("\n" if untracked else ""),
                                     encoding="utf-8")
    (d / "workspace.txt").write_text(str(root) + "\n", encoding="utf-8")
    (d / "README.txt").write_text(
        "任务开始时的快照（由 snapshot.begin() 生成）\n"
        f"任务：{label}\n工作区：{root}\nHEAD：{head}\n\n"
        "回滚方式：\n"
        "  git checkout -- .                    # 已跟踪文件回到任务开始前的 HEAD 状态\n"
        f"  git apply {d}/diff.patch             # 找回任务开始时就存在的未提交改动\n"
        f"  cat {d}/untracked.txt                # 任务开始时已存在的未跟踪文件清单\n",
        encoding="utf-8")

    return Snapshot(True, path=d, head=head, changed=len(lines) - len(untracked),
                    untracked=untracked,
                    note=f"任务开始时已有 {len(lines)} 个未提交改动")


# ---------------- 任务期间的"实际改动"（机器采集，喂给汇报与用户）----------------


@dataclass
class Change:
    """任务期间工作区实际发生了什么变化（全部来自 git，不依赖模型自述）。"""

    available: bool
    files: int = 0
    stat: str = ""                 # `git diff --stat`（人类可读）
    diff: str = ""                 # 完整 diff（超长会截断）
    untracked: list[str] = field(default_factory=list)
    truncated: bool = False
    note: str = ""

    def summary(self) -> str:
        """给模型/终端看的一行摘要（例如"改动 3 个文件（+12/-4），新增 1 个未跟踪文件"）。"""
        if not self.available:
            return f"（无法采集改动：{self.note}）"
        if not self.files and not self.untracked:
            return "（本次任务没有改动工作区文件）"
        parts = []
        if self.files:
            parts.append(f"改动 {self.files} 个已跟踪文件")
        if self.untracked:
            parts.append(f"新增 {len(self.untracked)} 个未跟踪文件")
        tail = f"（{'；'.join(parts)}）"
        if self.stat:
            tail += "\n" + self.stat.strip()
        return tail


def current_change() -> Change:
    """采集"当前工作区相对 HEAD 的改动"（任务中途/结束时都能调用）。"""
    root = workspace.current()
    if not is_git_repo(root) or shutil.which("git") is None:
        return Change(False, note="工作区不是 git 仓库或没有 git")
    code, status = _git(["status", "--porcelain"], root)
    if code != 0:
        return Change(False, note=f"git status 失败：{status.strip()[:120]}")
    lines = [ln for ln in status.splitlines() if ln.strip()]
    untracked = [ln[3:].strip() for ln in lines if ln.startswith("??")]
    tracked = len(lines) - len(untracked)

    code, stat = _git(["diff", "HEAD", "--stat"], root)
    code2, diff = _git(["diff", "HEAD"], root)
    truncated = len(diff) > MAX_DIFF_CHARS
    if truncated:
        diff = (diff[:MAX_DIFF_CHARS] +
                f"\n…（diff 过长已截断，完整内容见快照目录；共 {len(diff)} 字符）")
    return Change(True, files=tracked, stat=stat.strip(), diff=diff,
                  untracked=untracked, truncated=truncated)


def finish_after(snap: Snapshot) -> Change:
    """任务结束时留档：把本次改动写成 after.patch / after_status.txt，并返回摘要。

    有了这个 patch，即使工作区后来被下一轮任务或别的进程改坏，
    这次任务的成果仍可复查、可重放（`git apply after.patch`）。
    """
    change = current_change()
    if not snap.available or not snap.path or not change.available:
        return change
    try:
        (snap.path / "after.patch").write_text(change.diff, encoding="utf-8")
        (snap.path / "after_status.txt").write_text(
            (change.stat + "\n").strip() + "\n\n未跟踪文件：\n" + "\n".join(change.untracked) + "\n",
            encoding="utf-8")
    except OSError:      # pragma: no cover —— 写不进去不影响任务结果
        pass
    return change


# ---------------- 回滚 ----------------


def workspace_of(snap_dir: Path | str) -> Path | None:
    """快照记录的工作区路径（**回滚必须按它来**，而不是按"当前选中的项目"）。

    血的教训：restore() 早期版本用 workspace.current() 决定去哪儿 checkout，
    于是在"任务结束后回到别的项目"的场景下会去动**另一个仓库** —— 实测踩到
    （在 agent 会话里因为沙箱拦住了 git 写索引才没造成损失，普通终端里就是直接
    丢掉那个仓库的未提交改动）。快照属于哪个工作区，是快照自己的事实。
    """
    try:
        text = (Path(snap_dir) / "workspace.txt").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return Path(text) if text else None


def latest(pid: str | None = None) -> Path | None:
    """当前项目最近一次快照目录（供 `--rollback` 不带参数时使用）。"""
    root = workspace.STATE_ROOT / "snapshots" / (pid or workspace.current_id())
    if not root.is_dir():
        return None
    dirs = sorted((d for d in root.iterdir() if d.is_dir()), key=lambda d: d.name)
    return dirs[-1] if dirs else None


def recent(limit: int = 5) -> list[dict]:
    """最近几次快照（跨项目），供"当前项目没有快照"时给出提示。"""
    base = workspace.STATE_ROOT / "snapshots"
    if not base.is_dir():
        return []
    out: list[dict] = []
    for pid_dir in base.iterdir():
        if not pid_dir.is_dir():
            continue
        for d in pid_dir.iterdir():
            if d.is_dir() and (d / "head.txt").exists():
                out.append({"path": d, "workspace": workspace_of(d)})
    out.sort(key=lambda item: item["path"].name, reverse=True)
    return out[:limit]


def restore(snap_dir: Path | str) -> dict:
    """把工作区恢复到该快照记录的状态（**会丢弃之后的改动**，CLI 侧会先要确认）。

    两步：`git checkout -- .`（已跟踪 → 快照的 HEAD 状态）
        + `git apply <dir>/diff.patch`（把任务开始前**就存在**的未提交改动贴回来）。
    任务期间新建的未跟踪文件只列出、不自动删。
    """
    d = Path(snap_dir)
    if not (d / "head.txt").exists():
        return {"ok": False, "message": f"不是有效的快照目录：{d}"}
    root = workspace_of(d)
    if root is None or not root.is_dir():
        return {"ok": False,
                "message": f"快照没有记录有效的工作区（{root}），拒绝执行回滚以免动错仓库"}

    before = _read_untracked(d / "untracked.txt")
    code, out = _git(["checkout", "--", "."], root)
    if code != 0:
        return {"ok": False, "message": f"git checkout 失败：{out.strip()[:200]}"}

    patch = d / "diff.patch"
    applied = False
    if patch.exists() and patch.read_text(encoding="utf-8").strip():
        code, out = _git(["apply", str(patch)], root)
        applied = code == 0
        if not applied:
            return {"ok": False,
                    "message": f"已 checkout，但任务开始前的改动没能贴回（git apply 失败）：{out.strip()[:200]}"}

    after = current_change()
    extra = [f for f in after.untracked if f not in before]
    return {"ok": True, "workspace": str(root),
            "head": (d / "head.txt").read_text(encoding="utf-8").strip(),
            "reapplied_pre_task_patch": applied, "new_untracked": extra,
            "message": ("已恢复到任务开始前的状态"
                        + ("（含任务开始时就有的未提交改动）" if applied else "")
                        + (f"；任务期间新建的未跟踪文件仍在，需自行确认：{', '.join(extra[:10])}"
                           if extra else ""))}


def _read_untracked(path: Path) -> list[str]:
    try:
        return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except OSError:
        return []
