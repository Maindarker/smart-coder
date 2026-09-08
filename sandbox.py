"""Docker 沙箱执行封装：命令执行、资源限制、超时、工作区挂载。

安全要点：
- 所有命令在非 root 的 sandbox 容器内执行（镜像内 USER sandbox）
- 工作区以 /workspace 挂载进容器
- 资源限制：内存 / CPU / 网络（默认断网，git 等需要网络时显式开启）
- 超时强制 kill + remove
"""
from __future__ import annotations

import docker
from config import settings

SANDBOX_IMAGE = "smart-coder-sandbox:latest"
CONTAINER_WORKDIR = "/workspace"

_client: docker.DockerClient | None = None


def _docker() -> docker.DockerClient:
    global _client
    if _client is None:
        _client = docker.from_env()
    return _client


def run_command(
    cmd: str,
    *,
    cwd: str = CONTAINER_WORKDIR,
    timeout: int = 60,
    mem_limit: str = "512m",
    network: bool = False,
) -> dict:
    """在沙箱容器中执行 shell 命令。

    返回 dict：{exit_code, stdout, stderr, timed_out}
    - exit_code：进程退出码；-1 表示超时或执行异常
    - timed_out：是否因超时被强制终止
    """
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
