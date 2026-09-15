"""命令执行层：内核围栏（默认）/ 无隔离本机 / Docker 容器。

三种模式由 config.EXEC_MODE 决定：

- **"local"（默认）**：命令在本机执行，但被**内核级围栏**关住
  （macOS Seatbelt / Linux bubblewrap，见 `confinement.py`）：
  只能写「工作区 + 平台临时目录」，默认**禁网**。越界会在内核层失败
  （`Operation not permitted`），模型据此可以用 `escalate` 参数申请放宽，那会触发人工审批。
  拿不到可用后端时**拒绝执行**（fail closed），绝不悄悄降级成无隔离。
- **"host"**：命令在本机直接执行、无任何隔离，安全**完全靠逐条人工审批**兜底。
  这是明确知情的降级选择（例如机器上装不了 bubblewrap），也是从早期版本延续下来的行为。
- **"docker"**：命令在非 root 沙箱容器内执行，默认断网、限时限内存，
  当前工作区以 /workspace 挂载进容器；沙箱镜像目前只带 Python 运行时。

上面两种"没有强制隔离"的情形（host / sandbox_mode=danger-full-access）会被
`fence_available()` 识别出来，审批层据此把 `contained` 级的破坏也纳入审批 ——
因为那时候没人兜得住它们。

工作区是运行时可变的（workspace.py）：挂载源与 host 侧 cwd 都在调用时解析，
所以同一个服务进程里并发跑不同项目、切换项目都不会串。

所有模式返回统一的 dict：`{exit_code, stdout, stderr, timed_out, denied, confinement}`，
其中 `denied` 表示"本次失败疑似来自围栏拒绝"（启发式，只用于给模型一句提示，不作安全判定）。
"""
from __future__ import annotations

import os
import subprocess
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

import confinement
import workspace
from config import settings, ROOT
from confinement import DANGER_FULL_ACCESS, Policy

SANDBOX_IMAGE = "smart-coder-sandbox:latest"
DOCKERFILE = "Dockerfile.sandbox"  # 位于 agent 项目根目录（ROOT），不是被操作的工作区
CONTAINER_WORKDIR = "/workspace"

try:
    import docker
    _DOCKER_AVAILABLE = True
except Exception:  # noqa: BLE001 —— 轻量模式可能根本没装 docker 库
    docker = None
    _DOCKER_AVAILABLE = False

_client = None
_warned_fallback = False


def _docker():
    global _client
    if _client is None:
        _client = docker.from_env()
    return _client


# ---------------- 升级授权（一次审批 → 一次放宽）----------------
#
# 审批在 gate 节点完成（agent.py），但"这次调用被批准放宽到哪一档"要传给真正执行的
# run_command。用 contextvar 传递与 workspace.bind() 同源，避免把内部参数塞进模型可见的工具签名。

_grant: ContextVar[str] = ContextVar("sandbox_grant", default="")


@contextmanager
def granted(level: str):
    """在 with 块内把"本次调用获批的放宽档位"传给 run_command（"" | network | full）。"""
    token = _grant.set((level or "").strip().lower())
    try:
        yield
    finally:
        _grant.reset(token)


def current_grant() -> str:
    return _grant.get()


def escalation_needed(command: str, requested: str = "") -> str:
    """本次调用最终要放宽到哪一档（模型请求与判定结论取更宽的那个）。"""
    from shellrisk import command_verdict
    verdict = command_verdict(command)
    order = {"": 0, "network": 1, "full": 2}
    candidates = [str(requested or "").strip().lower()]
    if verdict.needs:
        candidates.append(verdict.needs)
    return max(candidates, key=lambda c: order.get(c, 2))


# ---------------- 模式与环境说明 ----------------

def effective_mode() -> str:
    """返回实际生效的执行模式。请求 docker 但不可用时回退 host（提示一次）。"""
    global _warned_fallback
    mode = settings.exec_mode if settings.exec_mode in ("docker", "host", "local") else "docker"
    if mode == "docker" and not _DOCKER_AVAILABLE:
        mode = "host"
        if not _warned_fallback:
            _warned_fallback = True
            print("[sandbox] 未安装 docker Python 库，已回退到本机直跑模式（EXEC_MODE=host）。")
    return mode


def fence_available() -> bool:
    """当前是否有**强制隔离**（内核围栏 或 容器）。

    没有隔离时，`contained` 级的破坏（例如 `rm -rf`）也没人兜得住 → 审批层会把它也拦下来问人。
    """
    mode = effective_mode()
    if mode == "docker":
        return True
    if mode == "local":
        return (settings.resolved_sandbox_mode != DANGER_FULL_ACCESS
                and confinement.select_backend() is not None)
    return False


def environment_note() -> str:
    """一行执行环境说明（进 prompt / 审批弹窗 / 日志）。"""
    mode = effective_mode()
    if mode == "docker":
        return "docker 容器（非 root、默认断网、限时限内存）"
    if mode == "local":
        if settings.resolved_sandbox_mode == DANGER_FULL_ACCESS:
            return "本机，无围栏（sandbox_mode=danger-full-access：明确放弃了强制隔离）"
        if confinement.select_backend() is None:
            return ("本机，内核围栏**不可用** → 命令会被拒绝执行"
                    "（要接受无隔离请显式设置 EXEC_MODE=host）")
        policy = _policy_for("", False)
        return f"本机 + {confinement.describe(policy)}"
    return "本机，无围栏（EXEC_MODE=host：逐条人工审批兜底）"


def ensure_sandbox_image() -> None:
    """本地没有沙箱镜像时，用 Dockerfile.sandbox 自动构建一次（幂等，仅 docker 模式用）。"""
    if effective_mode() != "docker":
        return
    client = _docker()
    try:
        client.images.get(SANDBOX_IMAGE)
        return
    except docker.errors.ImageNotFound:
        pass
    print(f"未找到沙箱镜像 {SANDBOX_IMAGE}，正在自动构建（首次约 1~3 分钟）...")
    client.images.build(
        path=str(ROOT),
        dockerfile=DOCKERFILE,
        tag=SANDBOX_IMAGE,
        rm=True,
    )


def _policy_for(escalate: str, network: bool) -> Policy:
    """把（升级档位、网络需求）翻译成一次调用的围栏策略。"""
    if escalate == "full":
        return Policy(mode=DANGER_FULL_ACCESS, workspace=workspace.current(), network=True)
    return Policy(mode=settings.resolved_sandbox_mode, workspace=workspace.current(),
                  network=bool(network or escalate == "network"))


def run_command(
    cmd: str,
    *,
    cwd: str = CONTAINER_WORKDIR,
    timeout: int = 60,
    mem_limit: str = "512m",
    network: bool = False,
    escalate: str = "",
) -> dict:
    """执行一条 shell 命令。

    escalate：本次调用获批的放宽档位（"" | "network" | "full"）。默认取 gate 节点
    在本线程里设置的授权（sandbox.granted(...)），也允许调用方显式传。
    """
    level = (escalate or current_grant() or "").strip().lower()
    if level not in ("", "network", "full"):
        level = "full"          # 不认识的档位一律按最高权限对待（审批已经发生过）
    mode = effective_mode()
    if mode == "docker":
        return _run_docker(cmd, cwd=cwd, timeout=timeout, mem_limit=mem_limit,
                           network=network or level in ("network", "full"))
    if mode == "local":
        return _run_local(cmd, cwd=cwd, timeout=timeout, network=network, escalate=level)
    return _run_host(cmd, cwd=cwd, timeout=timeout)


# ---------------- docker 沙箱 ----------------

def _run_docker(
    cmd: str,
    *,
    cwd: str = CONTAINER_WORKDIR,
    timeout: int = 60,
    mem_limit: str = "512m",
    network: bool = False,
) -> dict:
    ensure_sandbox_image()
    client = _docker()
    container = client.containers.run(
        image=SANDBOX_IMAGE,
        command=["/bin/bash", "-lc", cmd],
        working_dir=cwd,
        user="sandbox",
        mem_limit=mem_limit,
        nano_cpus=1_000_000_000,  # 1 CPU
        network_disabled=not network,
        detach=True,
        tty=False,
        # 挂载【当前】工作区（运行时可切换的项目目录），而不是启动时那个
        volumes={str(workspace.current()): {"bind": CONTAINER_WORKDIR, "mode": "rw"}},
    )
    timed_out = False
    try:
        result = container.wait(timeout=timeout)
        exit_code = result.get("StatusCode", -1)
        stdout = container.logs(stdout=True, stderr=False).decode("utf-8", errors="replace")
        stderr = container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace")
    except Exception as e:  # noqa: BLE001 —— 多为 wait 超时
        timed_out = True
        exit_code = -1
        stdout = ""
        stderr = f"[沙箱执行超时或异常] {e}"
        try:
            container.kill()
        except Exception:  # noqa: BLE001
            pass
    finally:
        try:
            container.remove(force=True)
        except Exception:  # noqa: BLE001
            pass
    return {"exit_code": exit_code, "stdout": stdout, "stderr": stderr,
            "timed_out": timed_out, "denied": False,
            "confinement": "docker 容器（非 root、默认断网）"}


# ---------------- 本机执行（host / local）----------------

def _host_cwd(cwd: str) -> Path:
    """把容器内路径（/workspace/...）翻译成本机工作区路径；越界一律回到工作区根。"""
    root = workspace.current().resolve()
    if cwd == CONTAINER_WORKDIR:
        return root
    if cwd.startswith(CONTAINER_WORKDIR + "/"):
        rel = cwd[len(CONTAINER_WORKDIR) + 1:].lstrip("/")
        cand = (root / rel).resolve()
        if cand.is_relative_to(root):
            return cand
    return root


def _project_venv_bin() -> Path | None:
    """当前项目自带的虚拟环境 bin 目录（若有）。"""
    for name in (".venv", "venv", "env"):
        for sub in ("bin", "Scripts"):        # Scripts 兼容 Windows
            d = workspace.current() / name / sub
            if d.is_dir():
                return d
    return None


def _host_env() -> dict:
    """PATH 顺序（固定"项目优先、agent 兜底"）：

      1. 当前项目自带的 .venv/bin —— 让项目自己的 python/pytest/node 生效；
      2. 原 PATH —— 系统工具链（node/go/cargo/…）；
      3. agent 自己的 venv bin 兜底 —— 只在项目没提供时才解析到 agent 的解释器。
    早期版本把 agent venv 放在最前，会劫持项目自身的 python；对任意项目（尤其
    带自己 venv 的仓库）是错的，所以改成现在的顺序。
    """
    env = os.environ.copy()
    parts: list[str] = []
    proj_bin = _project_venv_bin()
    if proj_bin:
        parts.append(str(proj_bin))
    parts.append(env.get("PATH", ""))
    agent_bin = str(Path(sys.executable).resolve().parent)
    if agent_bin not in parts and str(proj_bin or "") != agent_bin:
        parts.append(agent_bin)
    env["PATH"] = os.pathsep.join(p for p in parts if p)
    return env


def _spawn(argv: list[str], *, cwd: Path, timeout: int) -> dict:
    """真正 spawn，并把超时/异常收敛成统一结构。"""
    try:
        proc = subprocess.run(
            argv, cwd=str(cwd), env=_host_env(), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout,
        )
        return {"exit_code": proc.returncode,
                "stdout": proc.stdout or "",
                "stderr": proc.stderr or "",
                "timed_out": False}
    except subprocess.TimeoutExpired as e:  # noqa: BLE001
        return {"exit_code": -1,
                "stdout": (e.stdout or "") if isinstance(e.stdout, str) else "",
                "stderr": f"[本机执行超时（>{timeout}s）] {(e.stderr or '') if isinstance(e.stderr, str) else ''}",
                "timed_out": True}
    except Exception as e:  # noqa: BLE001
        return {"exit_code": -1, "stdout": "", "stderr": f"[本机执行异常] {e}", "timed_out": False}


def _run_host(cmd: str, *, cwd: str = CONTAINER_WORKDIR, timeout: int = 60) -> dict:
    """无隔离的本机执行（EXEC_MODE=host）：安全完全靠审批层逐条把关。"""
    out = _spawn(["/bin/bash", "-lc", cmd], cwd=_host_cwd(cwd), timeout=timeout)
    out.update({"denied": False, "confinement": "无围栏（EXEC_MODE=host）"})
    return out


def _run_local(cmd: str, *, cwd: str = CONTAINER_WORKDIR, timeout: int = 60,
               network: bool = False, escalate: str = "") -> dict:
    """本机执行 + 内核围栏（默认模式）：越界由内核拦住，越界需求走审批升级。"""
    policy = _policy_for(escalate, network)
    try:
        conf = confinement.wrap(["/bin/bash", "-lc", cmd], policy)
    except confinement.SandboxUnavailable as e:
        # fail closed：宁可这条命令不执行，也不在无隔离状态下跑
        return {"exit_code": 126, "stdout": "", "stderr": f"[内核围栏不可用] {e}",
                "timed_out": False, "denied": True,
                "confinement": f"请求 {policy.mode} 但无可用后端"}
    try:
        out = _spawn(conf.argv, cwd=_host_cwd(cwd), timeout=timeout)
    finally:
        conf.cleanup()
    out["denied"] = conf.confined and confinement.looks_denied(out.get("stderr", ""),
                                                               out.get("exit_code", 0))
    out["confinement"] = confinement.describe(policy)
    return out
