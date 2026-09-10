"""任务执行引擎：把 main.py 的 invoke + 审批循环抽象成可编程 API。

CLI / Web / 桌面都只调 run_task()，区别只在 approve 与 on_event 两个回调：
- approve(req)：决定一次危险操作是否放行（CLI 传 input() 包装；Web 传“等待浏览器点击”的回调）
- on_event(ev)：推送进度事件（Web 里转成 SSE 帧）

设计说明：
- interrupt() / Command(resume) 语义与 main.py 完全一致，只是把 input() 换成可注入回调。
- 注意：cost.tracker 目前是全局单例（cost.py），本机单用户没问题；
  若将来同一进程并发跑多个任务，需把 tracker 改成 per-task 隔离（见 WEB_IMPLEMENTATION.md §7.1）。
"""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional

from langchain_core.messages import HumanMessage
from langgraph.types import Command

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
) -> dict:
    """执行一个任务直到结束，返回最终 state。

    approve 缺省时任何危险操作都直接拒绝。on_event 事件：
      {"type": "approval", approval_id, question, tool, args, reason}   # 需人审批
      {"type": "done",    result, cost}                                 # 任务完成
      {"type": "error",   message, hint}                                # 异常（随后 raise）
    """
    config = {"configurable": {"thread_id": thread_id}}
    state = {
        "task": task, "plan": "", "context": "", "result": "",
        "feedback": "", "done": False, "iterations": 0,
        "messages": [HumanMessage(content=task)],
    }
    tracker.reset()

    def emit(ev: dict) -> None:
        if on_event:
            on_event(ev)

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
