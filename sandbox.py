"""命令执行层：Docker 沙箱（完整模式）或本机直跑（轻量模式，无 Docker）。

模式由 config.EXEC_MODE 决定：
- "docker"（默认，完整模式）：命令在非 root 沙箱容器内执行，默认断网、限时限内存，
  工作区以 /workspace 挂载进容器。docker Python 库缺失或守护进程不可用时自动回退 host。
- "host"（轻量模式，同事机器默认）：命令直接在本机用 bash 执行，继承当前环境；
  安全性由"每条命令人工审批"（agent.py 的 interrupt）+ 危险模式拦截兜底。

两种模式返回统一的 dict：{exit_code, stdout, stderr, timed_out}。
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from config import settings, ROOT

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


def effective_mode() -> str:
    """返回实际生效的执行模式。请求 docker 但不可用时自动回退 host（提示一次）。"""
    global _warned_fallback
    mode = settings.exec_mode
    if mode not in ("docker", "host"):
        mode = "docker"
    if mode == "docker" and not _DOCKER_AVAILABLE:
        mode = "host"
        if not _warned_fallback:
            _warned_fallback = True
            print("[sandbox] 未安装 docker Python 库，已回退到本机直跑模式（EXEC_MODE=host）。")
    return mode


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


def run_command(
    cmd: str,
    *,
    cwd: str = CONTAINER_WORKDIR,
    timeout: int = 60,
    mem_limit: str = "512m",
    network: bool = False,
) -> dict:
    """在沙箱（docker）或本机（host）执行 shell 命令。

    返回 dict：{exit_code, stdout, stderr, timed_out}
    - exit_code：进程退出码；-1 表示超时或执行异常
    - timed_out：是否因超时被强制终止
    """
    if effective_mode() == "host":
        return _run_host(cmd, cwd=cwd, timeout=timeout)
    return _run_docker(cmd, cwd=cwd, timeout=timeout, mem_limit=mem_limit, network=network)


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
        volumes={str(settings.workspace_root): {"bind": CONTAINER_WORKDIR, "mode": "rw"}},
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
    return {"exit_code": exit_code, "stdout": stdout, "stderr": stderr, "timed_out": timed_out}


# ---------------- 本机直跑（轻量模式） ----------------

def _host_cwd(cwd: str) -> Path:
    """把容器内路径（/workspace/...）翻译成本机工作区路径；越界一律回到工作区根。"""
    root = settings.workspace_root.resolve()
    if cwd == CONTAINER_WORKDIR:
        return root
    if cwd.startswith(CONTAINER_WORKDIR + "/"):
        rel = cwd[len(CONTAINER_WORKDIR) + 1:].lstrip("/")
        cand = (root / rel).resolve()
        if cand.is_relative_to(root):
            return cand
    return root


def _run_host(cmd: str, *, cwd: str = CONTAINER_WORKDIR, timeout: int = 60) -> dict:
    """本机 bash 执行。继承当前环境；把 venv/bin 放 PATH 最前，保证 python 解析到当前解释器。

    注意：无容器隔离（网络/权限/资源全放开），安全靠 agent 层逐条人工审批。
    """
    host_cwd = _host_cwd(cwd)
    # 让 `python` / `pytest` 等解析到当前 venv，而不是系统 python
    env = os.environ.copy()
    venv_bin = str(Path(sys.executable).resolve().parent)
    env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
    timed_out = False
    try:
        proc = subprocess.run(
            ["/bin/bash", "-lc", cmd],
            cwd=str(host_cwd),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        return {"exit_code": proc.returncode,
                "stdout": proc.stdout or "",
                "stderr": proc.stderr or "",
                "timed_out": False}
    except subprocess.TimeoutExpired as e:  # noqa: BLE001
        timed_out = True
        return {"exit_code": -1,
                "stdout": (e.stdout or "") if isinstance(e.stdout, str) else "",
                "stderr": f"[本机执行超时（>{timeout}s）] {(e.stderr or '') if isinstance(e.stderr, str) else ''}",
                "timed_out": timed_out}
    except Exception as e:  # noqa: BLE001
        return {"exit_code": -1,
                "stdout": "",
                "stderr": f"[本机执行异常] {e}",
                "timed_out": False}
