"""Docker 沙箱执行封装：命令执行、资源限制、超时、工作区挂载。

安全要点：
- 所有命令在非 root 的 sandbox 容器内执行（镜像内 USER sandbox）
- 工作区以 /workspace 挂载进容器
- 资源限制：内存 / CPU / 网络（默认断网，git 等需要网络时显式开启）
- 超时强制 kill + remove
- 沙箱镜像缺失时自动构建（首次运行自举，无需手动 docker build）
"""
from __future__ import annotations

import docker
from config import settings, ROOT

SANDBOX_IMAGE = "smart-coder-sandbox:latest"
DOCKERFILE = "Dockerfile.sandbox"  # 位于 agent 项目根目录（ROOT），不是被操作的工作区
CONTAINER_WORKDIR = "/workspace"

_client: docker.DockerClient | None = None


def _docker() -> docker.DockerClient:
    global _client
    if _client is None:
        _client = docker.from_env()
    return _client


def ensure_sandbox_image() -> None:
    """本地没有沙箱镜像时，用 Dockerfile.sandbox 自动构建一次（幂等）。

    镜像已存在时只是毫秒级查询后直接返回；缺失时才触发 docker build，
    保证项目在全新环境 clone 后第一次使用沙箱工具就能自举。
    """
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
    """在沙箱容器中执行 shell 命令。

    返回 dict：{exit_code, stdout, stderr, timed_out}
    - exit_code：进程退出码；-1 表示超时或执行异常
    - timed_out：是否因超时被强制终止
    """
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
