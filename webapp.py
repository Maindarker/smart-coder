"""本地 Web 服务入口：uvicorn webapp:app

启动：bash run-web.sh   （等价 ./bin/python -m uvicorn webapp:app --port 8000）
- 单 worker 运行（任务注册表在进程内存里），浏览器访问 http://127.0.0.1:8000/
- 项目（工作区）在界面里随时添加/切换，不用改 .env、不用重启：
    GET    /api/projects              项目列表 + 当前选中
    POST   /api/projects              添加目录 {path, name?}（幂等，自动切为当前）
    POST   /api/projects/{pid}/select 切换当前项目
    DELETE /api/projects/{pid}        从列表移除（不动磁盘数据）
    GET    /api/projects/{pid}/describe 深度识别（语言/测试命令/源码构成）
    POST   /api/dialog/directory      弹【系统原生】文件夹选择窗口，返回绝对路径
    GET    /api/fs?path=/abs/dir      服务端目录浏览（原生弹窗不可用时的兜底）
- 每个任务在自己的线程里跑，并用 workspace.bind() 绑定它所属的项目，
  所以并发跑不同项目不会互相串。
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
import dirpicker
import workspace
from config import settings
from sandbox import effective_mode

app = FastAPI(title="smart-coder", version="0.2.0")

# 一次性把单项目时代的会话库/记忆认领到新的按项目结构下（幂等；升级首启动时才有输出）
for _note in workspace.migrate_legacy_state():
    print(f"[migrate] {_note}")

WEB_DIR = Path(__file__).resolve().parent / "web"
MAX_RUNNING = 4   # 同时运行任务上限（每个任务可能起 Docker 沙箱容器，见 sandbox.py 资源限制）
MAX_KEEP = 50     # 结束后在内存里保留的任务数（含未消费的事件缓冲），超出则清理最老的
_semaphore = threading.BoundedSemaphore(MAX_RUNNING)
_registry: dict[str, "TaskHandle"] = {}


class TaskHandle:
    """一个后台任务：引擎在独立线程里跑，事件进队列，审批按 approval_id 挂起等待。"""

    def __init__(self, task: str, thread_id: str, project_id: str | None = None,
                 workdir: Path | None = None):
        self.task_id = uuid.uuid4().hex
        self.task = task
        self.thread_id = thread_id
        self.project_id = project_id
        self.workdir = workdir
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
            if self.workdir is not None:
                info = workspace.describe_project(self.workdir, deep=False)
                self.events.put({"type": "log",
                                 "message": f"项目：{info['path']}"
                                            f"（{info['lang']}，测试命令：{info.get('test_cmd') or '未推断出'}）"})
            # workdir 让引擎在本线程上下文里绑定工作区：并发任务跑不同项目也不会串
            engine.run_task(self.task, self.thread_id,
                            approve=approve, on_event=self.events.put,
                            workdir=self.workdir)
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


# ---------- 项目（工作区）管理 ----------

@app.get("/api/health")
def health() -> dict:
    ok, kind, why = dirpicker.detect()
    return {"ok": True, "exec_mode": effective_mode(),
            "rag_enabled": settings.rag_enabled,
            "dialog_picker": {"available": ok, "kind": kind, "detail": why},
            "workspace": str(workspace.current())}


@app.get("/api/projects")
def list_projects() -> dict:
    """所有已添加的项目 + 当前选中项。"""
    data = workspace.list_projects()
    cur = workspace.current()
    data["current_info"] = workspace.describe_project(cur, deep=False) if cur.is_dir() else None
    data["exec_mode"] = effective_mode()
    data["rag_enabled"] = settings.rag_enabled
    ok, kind, why = dirpicker.detect()
    data["dialog_picker"] = {"available": ok, "kind": kind, "detail": why}
    return data


@app.post("/api/projects")
def add_project(body: dict) -> dict:
    """添加一个工程目录（绝对路径）。同路径重复添加是幂等的，并立即切为当前项目。"""
    try:
        entry = workspace.add_project(
            body.get("path", ""), body.get("name"), select=bool(body.get("select", True))
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"project": entry, "current": workspace.current_id()}


@app.post("/api/projects/{pid}/select")
def select_project(pid: str) -> dict:
    try:
        entry = workspace.select_project(pid)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return {"project": entry, "current": workspace.current_id()}


@app.delete("/api/projects/{pid}")
def delete_project(pid: str) -> dict:
    """从列表移除项目。只改注册表，不动磁盘上的代码与 agent 状态目录。"""
    if not workspace.remove_project(pid):
        raise HTTPException(status_code=404, detail="项目不存在")
    return {"ok": True, "current": workspace.current_id()}


@app.get("/api/projects/{pid}/describe")
def describe_project(pid: str) -> dict:
    """深度识别某个项目：语言、源码构成、测试命令、README 摘要。"""
    data = workspace.list_projects()
    p = next((x for x in data["projects"] if x["id"] == pid), None)
    if p is None:
        raise HTTPException(status_code=404, detail="项目不存在")
    return workspace.describe_project(p["path"], deep=True)


@app.get("/api/workspace")
def workspace_info() -> dict:
    """当前工作区（向后兼容的旧接口：早先只有单个 WORKSPACE_ROOT）。"""
    cur = workspace.current()
    info = workspace.describe_project(cur, deep=False) if cur.is_dir() else None
    return {
        "workspace_root": str(cur),
        "exists": cur.is_dir(),
        "project_id": workspace.current_id(),
        "lang": (info or {}).get("lang"),
        "test_cmd": (info or {}).get("test_cmd"),
        "exec_mode": effective_mode(),
    }


# ---------- 系统原生目录选择（像上传文件那样点选本地目录） ----------

#: 同一时刻只允许一个系统弹窗（连点按钮不该弹出一堆窗口）
_dialog_lock = threading.Lock()


@app.post("/api/dialog/directory")
def dialog_directory(body: dict | None = None) -> dict:
    """在服务器所在机器上弹出【系统原生】文件夹选择窗口，返回选中的绝对路径。

    为什么要服务端弹窗：浏览器的 <input webkitdirectory> / showDirectoryPicker()
    出于安全考虑只给文件名、不给绝对路径，后端无法据此定位目录；
    而本服务只监听 127.0.0.1、与浏览器同机，所以由服务端进程弹系统对话框最直接。

    请求会一直挂着直到用户选完/取消/超时（默认 180s）。前端要据此显示"等待中"。
    """
    if not _dialog_lock.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="已经有一个目录选择窗口打开了，请先完成或取消它")
    try:
        body = body or {}
        initial = body.get("initial") or str(workspace.current())
        result = dirpicker.pick_directory(initial, timeout=dirpicker.DEFAULT_TIMEOUT)
    finally:
        _dialog_lock.release()

    if result.status == "unsupported":
        raise HTTPException(status_code=501, detail=f"{result.message}。{result.hint}")
    if result.status == "failed":
        raise HTTPException(status_code=502, detail=f"{result.message}。{result.hint}")
    if result.status == "timeout":
        raise HTTPException(status_code=408, detail=f"{result.message}。{result.hint}")
    if result.status == "canceled":
        return {"ok": False, "status": "canceled"}

    # 选完立刻校验（复用添加项目那套规则：绝对路径、存在、是目录、不是根目录、可读）
    try:
        picked = workspace.normalize_dir(result.path or "")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"这个目录不能用：{e}") from e
    info = workspace.describe_project(picked, deep=False)
    return {"ok": True, "status": "ok", "path": str(picked),
            "name": picked.name, "lang": info["lang"], "test_cmd": info.get("test_cmd")}


# ---------- 服务端目录浏览（页面内兜底方案；服务器无 GUI 时用） ----------

@app.get("/api/fs")
def browse_fs(path: str = "") -> dict:
    """列出某个绝对路径下的子目录，并标注哪些"看起来是项目"。

    这是原生弹窗不可用时的兜底（服务器无图形界面 / 通过 SSH 端口转发访问），
    也是"顺便看看某个目录里有哪些子项目"的入口。
    """
    if path.strip():
        try:
            base = workspace.normalize_dir(path)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
    else:
        base = workspace.current()
        if not base.is_dir():
            base = Path.home()

    registered = {p["path"]: p["id"] for p in workspace.list_projects()["projects"]}
    entries = []
    try:
        children = sorted(base.iterdir(), key=lambda x: (x.name.startswith("."), x.name.lower()))
    except OSError as e:
        raise HTTPException(status_code=400, detail=f"无法读取目录：{e}") from e

    for child in children:
        try:
            if not child.is_dir():       # 只列目录：这里选的是"工程目录"
                continue
        except OSError:
            continue
        info = workspace.detect_project(child)
        entries.append({
            "name": child.name,
            "path": str(child),
            "hidden": child.name.startswith("."),
            "is_project": info["is_project"],
            "lang": info["lang"],
            "manifests": info["manifests"],
            "registered_id": registered.get(str(child)),
        })
        if len(entries) >= 300:
            break

    self_info = workspace.detect_project(base)
    parent = base.parent
    return {
        "path": str(base),
        "parent": str(parent) if parent != base else None,
        "home": str(Path.home()),
        "is_project": self_info["is_project"],
        "lang": self_info["lang"],
        "registered_id": registered.get(str(base)),
        "entries": entries,
    }


# ---------- 任务 ----------

@app.post("/api/tasks")
def create_task(body: dict) -> dict:
    task = (body.get("task") or "").strip()
    if not task:
        raise HTTPException(status_code=400, detail="task 不能为空")

    # 任务绑定到提交时的项目：之后界面切项目也不会影响已在跑的任务
    workdir: Path | None = None
    project_id = body.get("project_id")
    if project_id:
        p = next((x for x in workspace.list_projects()["projects"] if x["id"] == project_id), None)
        if p is None:
            raise HTTPException(status_code=404, detail="项目不存在")
        workdir = Path(p["path"])
    else:
        project_id = workspace.current_id()
        cur = workspace.current()
        workdir = cur if cur.is_dir() else None

    if not _semaphore.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail=f"已有 {MAX_RUNNING} 个任务在运行，请等当前任务结束后再试",
        )
    _prune()
    h = TaskHandle(task, body.get("thread_id") or uuid.uuid4().hex,
                   project_id=project_id, workdir=workdir)
    _registry[h.task_id] = h
    h.start()
    return {"task_id": h.task_id, "project_id": project_id,
            "workspace_root": str(workdir) if workdir else None}


@app.get("/api/tasks/{tid}/events")
def task_events(tid: str):
    """SSE 事件流：log / approval / done / error / close。"""
    h = _registry.get(tid)
    if h is None:
        raise HTTPException(status_code=404, detail="任务不存在")

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
        raise HTTPException(status_code=404, detail="任务不存在")
    req = h.approvals.get(body.get("approval_id", ""))
    if req is None:
        raise HTTPException(status_code=404, detail="该审批已过期或不存在")
    req.resolve(bool(body.get("approved", False)))
    return {"ok": True}


# 静态前端（必须最后挂载，避免吞掉 /api 路由）
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
