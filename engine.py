"""任务执行引擎：把 main.py 的 invoke + 审批循环抽象成可编程 API。

CLI / Web / 桌面都只调 run_task()，区别只在 approve 与 on_event 两个回调：
- approve(req)：决定一次危险操作是否放行（CLI 传 input() 包装；Web 传"等待浏览器点击"的回调）
- on_event(ev)：推送进度事件（Web 里转成 SSE 帧）

多项目相关：
- workdir 参数把「本任务操作哪个代码库」显式传进来，engine 用 workspace.bind() 在
  当前线程的上下文里绑定它。contextvar 是按线程隔离的，所以 webapp 每任务一线程的
  模型下，并发跑不同项目不会互相串（不要改成写全局变量）。
- thread_id 会自动加当前项目前缀（workspace.thread_key），一个 sqlite 承载所有项目的
  会话历史，graph 无需按项目重编译。

设计说明：
- interrupt() / Command(resume) 语义与 main.py 完全一致，只是把 input() 换成可注入回调。
- 成本统计用 tracker.session() 按任务分账（cost.py），并发任务不会混在一起。
"""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from langchain_core.messages import HumanMessage
from langgraph.types import Command

import workspace
from agent import app
from cost import tracker

#: 审批等待上限（秒），超时视为拒绝（浏览器关掉后引擎线程不至于永久阻塞）
DEFAULT_APPROVAL_TIMEOUT = 600.0


@dataclass
class ApprovalRequest:
    """一次待审批的危险操作。

    resolve() 由外部（HTTP 层）调用并唤醒 wait()；wait() 阻塞引擎线程等待人拍板。
    """

    question: str
    tool: str
    args: dict
    reason: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    _done: threading.Event = field(default_factory=threading.Event, repr=False)
    _approved: bool = False

    def resolve(self, approved: bool) -> None:
        self._approved = approved
        self._done.set()

    def wait(self, timeout: Optional[float] = None) -> bool:
        """阻塞直到 resolve()；超时返回 False（视为拒绝）。"""
        self._done.wait(timeout)
        return self._approved


def _find_pending(config: dict) -> Optional[ApprovalRequest]:
    """镜像 main.py 的查找逻辑：从 state.tasks 里找第一个 pending interrupt。"""
    st = app.get_state(config)
    for t in st.tasks or []:
        ints = getattr(t, "interrupts", None)
        if ints:
            v = ints[0].value or {}
            return ApprovalRequest(
                question=v.get("question", "是否允许该操作？"),
                tool=v.get("tool", ""),
                args=v.get("args", {}),
                reason=v.get("reason", ""),
            )
    return None


def run_task(
    task: str,
    thread_id: str = "main",
    approve: Optional[Callable[[ApprovalRequest], bool]] = None,
    on_event: Optional[Callable[[dict], None]] = None,
    workdir: str | Path | None = None,
) -> dict:
    """执行一个任务直到结束，返回最终 state。

    workdir 指定本任务操作哪个代码库（不传则用当前选中的项目）。
    approve 缺省时任何危险操作都直接拒绝。on_event 事件：
      {"type": "approval", approval_id, question, tool, args, reason}   # 需人审批
      {"type": "done",    result, cost}                                 # 任务完成
      {"type": "error",   message, hint}                                # 异常（随后 raise）
    """
    if workdir:
        with workspace.bind(workdir):
            return _run(task, thread_id, approve, on_event)
    return _run(task, thread_id, approve, on_event)


def _run(
    task: str,
    thread_id: str,
    approve: Optional[Callable[[ApprovalRequest], bool]],
    on_event: Optional[Callable[[dict], None]],
) -> dict:
    # 会话历史按项目隔离：同一 thread_id 在不同项目里是两段互不可见的对话
    config = {"configurable": {"thread_id": workspace.thread_key(thread_id)}}
    state = {
        "task": task, "plan": "", "context": "", "result": "",
        "feedback": "", "done": False, "iterations": 0,
        "history_summary": "", "history_summarized": 0,
        "messages": [HumanMessage(content=task)],
    }

    def emit(ev: dict) -> None:
        if on_event:
            on_event(ev)

    with tracker.session():
        try:
            final = app.invoke(state, config=config)
        except Exception as e:  # noqa: BLE001 —— 与 main.py 一致：给出可自愈提示后抛出
            emit({"type": "error",
                  "message": f"{type(e).__name__}: {e}",
                  "hint": "若是旧会话 checkpoint 损坏，清理 .agent_cache/checkpoints.sqlite 后重试"})
            raise

        while True:
            req = _find_pending(config)
            if req is None:
                break
            emit({"type": "approval", "approval_id": req.id, "question": req.question,
                  "tool": req.tool, "args": req.args, "reason": req.reason})
            approved = approve(req) if approve else False
            # 恢复执行：与 main.py 的 Command(resume={"approved": ...}) 一致
            final = app.invoke(Command(resume={"approved": approved}), config=config)

        emit({"type": "done", "result": final.get("result", ""), "cost": tracker.summary()})
    return final
