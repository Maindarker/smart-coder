"""打开操作系统原生的「选择文件夹」对话框，拿到服务器本机的绝对路径。

为什么需要单独做这一层
----------------------
浏览器确实能弹"选择目录"窗口（`<input webkitdirectory>` / `showDirectoryPicker()`），
但出于安全设计，它们**只暴露文件名/相对路径，绝不暴露绝对路径** ——
后端拿不到"要操作哪个目录"这个最关键的信息，只能让用户手打或走页面内浏览。

而这个 agent 的部署形态是"每人自己机器上跑一个只监听 127.0.0.1 的本地服务"，
浏览器和后端在同一台机器上。所以由**服务端进程**去弹系统原生对话框最直接：
用户看到的是自己系统熟悉的文件夹选择器，服务端拿到的是真实绝对路径，两边都满足。

平台支持
--------
- macOS  : `osascript` 的 `choose folder`（Finder 风格原生窗口）
- Linux  : `zenity --file-selection --directory`，退回 `kdialog`
- Windows: PowerShell 的 `FolderBrowserDialog`
- 都不行（无 GUI 会话 / 远程服务器 / 容器里跑）→ 返回 `unsupported`，
  前端会自动退回"在页面里浏览目录"，功能不丢。

测试/无头环境
-------------
设环境变量 `SMARTCODER_PICK_DIR_CMD`（一个命令模板，`{initial}` 会被替换成初始目录）
即可替换掉真实弹窗，用于自动化测试：
    SMARTCODER_PICK_DIR_CMD="echo /tmp/my-project" python -m uvicorn webapp:app

macOS 抢焦点开关
----------------
macOS 默认**不**先执行 `tell me to activate`（那会白等约 2 秒，见 `_macos_command`）。
若所在机器上面板确实会被浏览器窗口挡住，设 `SMARTCODER_PICK_DIR_ACTIVATE=1` 换回旧行为：
    SMARTCODER_PICK_DIR_ACTIVATE=1 python -m uvicorn webapp:app
"""
from __future__ import annotations

import os
import platform
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field

#: 默认等待用户操作的上限（秒）。超时按"取消"处理，避免请求永久挂住。
DEFAULT_TIMEOUT = 180

#: 覆盖真实弹窗的命令模板（仅测试用）
ENV_OVERRIDE = "SMARTCODER_PICK_DIR_CMD"

#: macOS：弹窗前是否先 `tell me to activate` 抢焦点（默认关；理由见 _macos_command）
ENV_ACTIVATE = "SMARTCODER_PICK_DIR_ACTIVATE"

#: 弹窗标题
_PROMPT = "选择要交给 smart-coder 操作的工程目录"


@dataclass
class PickResult:
    """一次目录选择的结果。

    status 取值：
      ok          —— 选好了，path 是绝对路径
      canceled    —— 用户取消（正常操作，不是错误）
      timeout     —— 等太久没操作
      unsupported —— 当前系统/环境没有可用的原生对话框
      failed      —— 弹窗本身出错（附 message/hint）
    """

    status: str
    path: str | None = None
    message: str = ""
    hint: str = ""
    command: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "ok" and bool(self.path)

    def as_dict(self) -> dict:
        return {"ok": self.ok, "status": self.status, "path": self.path,
                "message": self.message, "hint": self.hint}


# ---------------- 平台探测 ----------------

def _env_on(name: str) -> bool:
    """读一个布尔开关：`1/true/yes/on` 算开（空值、`0/false/no` 都算关）。"""
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _macos_command(initial: str | None, timeout: int) -> list[str]:
    """组装 macOS 的 AppleScript。

    为什么默认【不】抢焦点（历史上的第一行 `tell me to activate` 被删掉了）：
    `me` 指的就是 osascript 自己，而 osascript 是 BackgroundOnly 进程、**永远进不了前台**。
    于是系统会一直等它变成前台应用，等到约 2s 激活超时才肯继续往下走 —— 实测
    「spawn 到即将弹出面板」由 0.1s 变成 2.0~2.3s（18 次全部如此，稳定复现），而这 2s 毫无收益：
    轮询 LaunchServices 的前台标记，面板打开期间它自始至终没进过前台。
    真正让面板显示在前面的不是这一行，所以默认删掉；万一个别机器上面板确实会被
    浏览器挡住，设 SMARTCODER_PICK_DIR_ACTIVATE=1 即可换回旧行为。
    """
    script: list[str] = []
    if _env_on(ENV_ACTIVATE):
        script.append("tell me to activate")
    if initial:
        script.append(f'set p to choose folder with prompt "{_PROMPT}" '
                      f'default location POSIX file "{initial}"')
    else:
        script.append(f'set p to choose folder with prompt "{_PROMPT}"')
    script.append("return POSIX path of p")
    body = "\n".join(script)
    return ["osascript", "-e", f"with timeout of {timeout} seconds\n{body}\nend timeout"]


def _linux_commands(initial: str | None) -> list[list[str]]:
    cmds: list[list[str]] = []
    if shutil.which("zenity"):
        cmd = ["zenity", "--file-selection", "--directory", f"--title={_PROMPT}"]
        if initial:
            cmd += ["--filename", initial.rstrip("/") + "/"]
        cmds.append(cmd)
    if shutil.which("kdialog"):
        cmd = ["kdialog", "--getexistingdirectory"]
        cmd.append(initial or os.path.expanduser("~"))
        cmds.append(cmd)
    return cmds


def _windows_command(initial: str | None) -> list[str]:
    init_line = (f"$d.SelectedPath = '{initial}'" if initial else "")
    ps = (
        "Add-Type -AssemblyName System.Windows.Forms | Out-Null;"
        "$d = New-Object System.Windows.Forms.FolderBrowserDialog;"
        f"$d.Description = '{_PROMPT}';$d.ShowNewFolderButton = $false;"
        f"{init_line}"
        "if ($d.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) "
        "{ Write-Output $d.SelectedPath } else { exit 1 }"
    )
    return ["powershell", "-NoProfile", "-STA", "-Command", ps]


def detect() -> tuple[bool, str, str]:
    """返回 (是否可用, 平台标识, 说明)。"""
    if os.environ.get(ENV_OVERRIDE):
        return True, "override", f"由 {ENV_OVERRIDE} 指定"
    system = platform.system()
    if system == "Darwin":
        if shutil.which("osascript"):
            return True, "macos", "macOS 原生文件夹选择器（osascript）"
        return False, "macos", "找不到 osascript"
    if system == "Linux":
        cmds = _linux_commands(None)
        if cmds:
            return True, "linux", f"{cmds[0][0]} 目录选择器"
        return False, "linux", "没装 zenity / kdialog"
    if system == "Windows":
        if shutil.which("powershell"):
            return True, "windows", "Windows 文件夹选择器（PowerShell）"
        return False, "windows", "找不到 powershell"
    return False, system.lower() or "unknown", f"暂不支持的系统：{system}"


# ---------------- 弹窗 ----------------

def pick_directory(initial: str | None = None,
                   timeout: int = DEFAULT_TIMEOUT) -> PickResult:
    """弹出系统原生目录选择窗口，阻塞直到用户选完/取消/超时。

    initial 为初始定位目录（不存在时忽略，交给系统默认位置）。
    """
    if initial and not os.path.isdir(initial):
        initial = os.path.dirname(initial) if os.path.dirname(initial) else None

    override = os.environ.get(ENV_OVERRIDE)
    if override:
        # 用 shlex 而不是 str.split：模板里常有带引号的参数（如 sh -c 'echo x; exit 1'）
        cmd = shlex.split(override.replace("{initial}", initial or ""))
    else:
        ok, kind, why = detect()
        if not ok:
            return PickResult(
                status="unsupported", message=why,
                hint="改用页面内「浏览目录」方式选择（服务器可能没有图形界面，或不是本机浏览器访问）")
        if kind == "macos":
            cmd = _macos_command(initial, timeout)
        elif kind == "linux":
            cmd = _linux_commands(initial)[0]
        elif kind == "windows":
            cmd = _windows_command(initial)
        else:  # pragma: no cover - detect() 已覆盖
            return PickResult(status="unsupported", message=why)

    try:
        # 额外留 15s 余量：让弹窗自己的超时先触发（macOS 的 with timeout / zenity 自身行为），
        # 这样能拿到"用户没操作"的干净结果；subprocess 的 timeout 只是最后兜底，
        # 防止某个平台的对话框完全不响应时把请求永久挂死。
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout + 15)
    except subprocess.TimeoutExpired:
        return PickResult(status="timeout", command=cmd,
                          message=f"等待超过 {timeout} 秒没有操作",
                          hint="重新点击按钮即可再次弹窗")
    except OSError as e:
        return PickResult(status="failed", command=cmd,
                          message=f"无法启动目录选择器：{e}",
                          hint="改用页面内「浏览目录」方式选择")

    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()

    if _is_timeout_hit(err):
        return PickResult(status="timeout", command=cmd,
                          message=f"等待超过 {timeout} 秒没有操作",
                          hint="重新点击按钮即可再次弹窗")
    if proc.returncode != 0:
        if _is_cancel(err, proc.returncode):
            return PickResult(status="canceled", command=cmd)
        return PickResult(status="failed", command=cmd,
                          message=err.splitlines()[-1] if err else f"选择器退出码 {proc.returncode}",
                          hint="若服务器不在本机（例如 SSH 端口转发），系统弹窗会出现在服务器那台机器上；"
                               "这种情况请改用页面内「浏览目录」")
    if not out:
        # rc=0 却没有输出：选择器被中断或平台实现异常，明确报出来而不是返回一个空路径
        return PickResult(status="failed", command=cmd,
                          message="目录选择器没有返回路径（可能被系统中断）",
                          hint="改用页面内「浏览目录」方式选择")

    path = out.splitlines()[-1].strip().rstrip("/") or "/"
    return PickResult(status="ok", path=path, command=cmd)


def _is_timeout_hit(stderr: str) -> bool:
    """识别"弹窗自己超时"：macOS 的 AppleScript 超时是 -1712，别的平台用文案兜。"""
    low = (stderr or "").lower()
    return "-1712" in low or "timed out" in low


def _is_cancel(stderr: str, returncode: int) -> bool:
    """识别"用户取消"：各平台表现形式不同，但都不该当成错误报给用户。

    注意最后那行"rc==1 且没有任何 stderr"的判定：zenity/kdialog 取消时就是静默返回 1，
    而**真正的失败一定会在 stderr 留信息**（例如 macOS 的 -1743 未授权 Apple Events）。
    如果无脑把 rc==1 当成取消，失败就会被静默吞掉、界面毫无反应。
    """
    low = (stderr or "").lower()
    if "user canceled" in low or "-128" in low:
        return True                       # macOS
    if "cancel" in low:
        return True                       # zenity / kdialog
    return returncode == 1 and not (stderr or "").strip()
