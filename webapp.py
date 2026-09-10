"""本地 Web 服务入口：uvicorn webapp:app

启动：bash run-web.sh   （等价 ./bin/python -m uvicorn webapp:app --port 8000）
- 单 worker 运行（任务注册表在进程内存里），浏览器访问 http://127.0.0.1:8000/
- WORKSPACE_ROOT 决定 agent 操作哪个代码库；换项目 = 改 .env 后重启服务
"""
from __future__ import annotations

import json
import queue
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

import engine
from config import settings

app = FastAPI(title="smart-coder", version="0.1.0")

WEB_DIR = Path(__file__).resolve().parent / "web"
MAX_RUNNING = 4   # 同时运行任务上限（每个任务可能起 Docker 沙箱容器，见 sandbox.py 资源限制）
MAX_KEEP = 50     # 结束后在内存里保留的任务数（含未消费的事件缓冲），超出则清理最老的
_semaphore = threading.BoundedSemaphore(MAX_RUNNING)
_registry: dict[str, "TaskHandle"] = {}


class TaskHandle:
    """一个后台任务：引擎在独立线程里跑，事件进队列，审批按 approval_id 挂起等待。"""

    def __init__(self, task: str, thread_id: str):
        self.task_id = uuid.uuid4().hex
        self.task = task
        self.thread_id = thread_id
        self.events: "queue.Queue[dict]" = queue.Queue()
        self.approvals: dict[str, engine.ApprovalRequest] = {}
        self.finished = False
        self.thread = threading.Thread(
            target=self._run, name=f"task-{self.task_id}", daemon=True
        )

    def _run(self) -> None:
        def approve(req: engine.ApprovalRequest) -> bool:
            # 挂起等待浏览器 POST /approve；超时视为拒绝（引擎线程不会永久阻塞）
            self.approvals[req.id] = req
            try:
                return req.wait(timeout=engine.DEFAULT_APPROVAL_TIMEOUT)
            finally:
                self.approvals.pop(req.id, None)

        try:
            engine.run_task(self.task, self.thread_id,
                            approve=approve, on_event=self.events.put)
        except Exception:  # noqa: BLE001 —— engine 已发 error 事件，这里只保证收尾
            pass
        finally:
            self.finished = True
            self.events.put({"type": "close"})
            _semaphore.release()   # 任务真正跑完才释放并发名额

    def start(self) -> None:
        self.thread.start()


def _prune() -> None:
    """注册表超出 MAX_KEEP 时，清掉已结束的最老任务，避免内存无限增长。"""
    if len(_registry) <= MAX_KEEP:
        return
    for tid, h in list(_registry.items()):
        if h.finished:
            _registry.pop(tid, None)
        if len(_registry) <= MAX_KEEP:
            break


# ---------- API ----------

@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.get("/api/workspace")
def workspace() -> dict:
    """返回当前工作区路径（对应 WORKSPACE_ROOT，即 agent 当前操作的代码库）。"""
    root = settings.workspace_root
    return {"workspace_root": str(root), "exists": root.exists()}


@app.post("/api/tasks")
def create_task(body: dict) -> dict:
    task = (body.get("task") or "").strip()
    if not task:
        raise HTTPException(status_code=400, detail="task 不能为空")
    if not _semaphore.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail=f"已有 {MAX_RUNNING} 个任务在运行，请等当前任务结束后再试",
        )
    _prune()
    h = TaskHandle(task, body.get("thread_id") or uuid.uuid4().hex)
    _registry[h.task_id] = h
    h.start()
    return {"task_id": h.task_id}


@app.get("/api/tasks/{tid}/events")
def task_events(tid: str):
    """SSE 事件流：log / approval / done / error / close。"""
    h = _registry.get(tid)
    if h is None:
        raise HTTPException(status_code=404, detail="task 不存在")

    def gen():
        while True:
            ev = h.events.get()   # 阻塞取事件；SSE 响应跑在 FastAPI 线程池里
            yield f"data: {json.dumps(ev, ensure_ascii=False, default=str)}\n\n"
            if ev.get("type") == "close":
                break

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/api/tasks/{tid}/approve")
def task_approve(tid: str, body: dict) -> dict:
    """审批回复：前端点允许/拒绝后调用，唤醒引擎线程继续执行。"""
    h = _registry.get(tid)
    if h is None:
        raise HTTPException(status_code=404, detail="task 不存在")
    req = h.approvals.get(body.get("approval_id", ""))
    if req is None:
        raise HTTPException(status_code=404, detail="该审批已过期或不存在")
    req.resolve(bool(body.get("approved", False)))
    return {"ok": True}


# 静态前端（必须最后挂载，避免吞掉 /api 路由）
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
