"""本项目的横切关注点：重试 / 摘要 / 权限控制 / 人工审批（自实现）。

为什么是"自实现"：需求要求"通过 Middleware 添加重试、摘要、权限控制和人工审批"。
框架的 `AgentMiddleware` 只能挂在 `create_agent` 上（`StateGraph.compile()` 没有
middleware 参数），而本项目是手写 StateGraph（见 agent.py）。按既定口径取宽松解释：
用"等价能力 + 明确接缝"自实现，agent.py 只负责编排。

四个接缝（全部可用纯本地单测覆盖，不需要调 API）：
- `call_model()`       重试     ：指数退避 + 抖动；参数/鉴权类错误不重试
- `compact_history()`  摘要     ：长会话折叠进 `history_summary`，替代"只截断最近 N 条"
- `permission_risk()`  权限控制 ：危险命令规则判定（纯函数）
- `review_tool_call()` 人工审批 ：需要时 `interrupt()` 找人拍板，并给出"拒了之后怎么办"

刻意不做的事：**不自动重试工具调用**。`run_shell` / `run_test` 有副作用，
盲目重试等于把命令跑两遍；工具异常仍按原策略回传给模型，由模型决定下一步
（见 agent.py 的 execute）。这与"重试只作用于可安全重复的模型调用"是一致的取舍。
"""
from __future__ import annotations

import random
import re
import time
from typing import NamedTuple

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import interrupt

from config import settings

# ---------------- 重试 ----------------

MAX_MODEL_RETRIES = 2        # 首次之外的额外重试次数
RETRY_BACKOFF_BASE = 1.0     # 退避基数（秒）：1s、2s、4s…
RETRY_BACKOFF_MAX = 8.0      # 单次退避上限，防止畸形增长

# 这些错误重试没有意义（请求本身不对 / 凭据不对），直接抛出、早暴露。
# 只有装了 openai 才有这些类型；缺失时退化为"一切都可重试"。
try:  # pragma: no cover - 取决于运行环境
    from openai import (
        AuthenticationError,
        BadRequestError,
        NotFoundError,
        PermissionDeniedError,
    )

    _NON_RETRYABLE: tuple[type[BaseException], ...] = (
        BadRequestError, AuthenticationError, PermissionDeniedError, NotFoundError,
    )
except Exception:  # noqa: BLE001
    _NON_RETRYABLE = ()


def call_model(llm, messages, *, label: str = "model"):
    """调用模型；可重试错误按指数退避 + 抖动重试，耗尽后抛出带上下文的错误。

    抖动是必要的：webapp 最多 4 个任务并发，固定退避会让它们同时重试、同时再撞墙。
    """
    last: BaseException | None = None
    for attempt in range(MAX_MODEL_RETRIES + 1):
        try:
            return llm.invoke(messages)
        except Exception as e:  # noqa: BLE001 —— 重试策略统一在这里收敛
            if isinstance(e, _NON_RETRYABLE):
                raise
            last = e
            if attempt >= MAX_MODEL_RETRIES:
                break
            delay = min(RETRY_BACKOFF_BASE * (2 ** attempt), RETRY_BACKOFF_MAX)
            delay *= 0.5 + random.random()
            print(f"[retry] {label} 调用失败（{type(e).__name__}: {e}），"
                  f"{delay:.1f}s 后第 {attempt + 1} 次重试")
            time.sleep(delay)
    raise RuntimeError(
        f"模型调用失败（{label}，已重试 {MAX_MODEL_RETRIES} 次）：{type(last).__name__}: {last}"
    ) from last


# ---------------- 摘要 ----------------

#: 消息超过这个条数才触发折叠（短会话完全不受影响）
HISTORY_TRIGGER_MESSAGES = 32
#: 折叠后保留最近这么多条原文（与改造前"只截断最近 16 条"保持一致，避免回归）
HISTORY_KEEP_RECENT = 16

SUMMARY_SYSTEM_PROMPT = (
    "你在压缩一段人机对话历史。把新对话并入已有摘要，输出一份精炼的摘要。"
    "必须保留：用户的目标与偏好、已确认的事实与决定、做过什么及其结果、未完成的事项。"
    "不要加入推测，不要输出摘要之外的任何话。"
)


def _fmt_message(m) -> str | None:
    """把一条消息压成一行历史文本；没有文本内容（如纯工具调用）返回 None。"""
    content = getattr(m, "content", "")
    if not isinstance(content, str) or not content.strip():
        return None
    role = "用户" if isinstance(m, HumanMessage) else "助手"
    return f"{role}：{content.strip()}"


def history_text(state) -> str:
    """给模型看的历史文本：已折叠部分的摘要 + 最近若干条原文。"""
    parts: list[str] = []
    summary = str(state.get("history_summary") or "").strip()
    if summary:
        parts.append("（更早对话的摘要）\n" + summary)
    msgs = state.get("messages") or []
    lines = [t for t in (_fmt_message(m) for m in msgs[-HISTORY_KEEP_RECENT:]) if t]
    if lines:
        parts.append("\n".join(lines))
    return "\n\n".join(parts) or "（无历史）"


def compact_history(state, llm) -> dict | None:
    """长会话摘要：把"还没折叠过的旧消息"并入 `history_summary`。

    返回要写回 state 的更新；没有可折叠内容时返回 None（此时**不产生任何模型调用**）。
    `history_summarized` 记录"已折叠到第几条"，避免每轮把同一批消息反复摘要。
    """
    msgs = state.get("messages") or []
    if len(msgs) <= HISTORY_TRIGGER_MESSAGES:
        return None
    folded = int(state.get("history_summarized") or 0)
    end = len(msgs) - HISTORY_KEEP_RECENT          # 折叠区间 [folded, end)
    if end <= folded:
        return None
    batch = [t for t in (_fmt_message(m) for m in msgs[folded:end]) if t]
    if not batch:
        return {"history_summarized": end}
    prev = str(state.get("history_summary") or "").strip() or "（无）"
    reply = call_model(llm, [
        SystemMessage(SUMMARY_SYSTEM_PROMPT),
        HumanMessage(f"已有摘要：\n{prev}\n\n需要并入的新对话：\n" + "\n".join(batch)),
    ], label="compact")
    return {"history_summary": str(reply.content or "").strip(), "history_summarized": end}


# ---------------- 权限控制 ----------------

#: 危险命令模式：(正则, 风险说明)
DANGEROUS_PATTERNS = [
    (r"\brm\s+(-[a-z]*r[a-z]*f|-[a-z]*f[a-z]*r)", "递归删除文件"),
    (r"\bgit\s+push\b", "git 推送到远程"),
    (r"\bgit\s+reset\s+--hard\b", "git 硬重置（丢失提交）"),
    (r"\bsudo\b", "提权执行"),
    (r"\bchmod\s+777\b", "开放所有权限"),
    (r"\bshutdown\b|\breboot\b", "关机/重启"),
]

#: 这些工具在 host 模式下"每条都要人工审批"（无容器隔离，命令直接落本机）
APPROVAL_REQUIRED_TOOLS = ("run_shell", "run_test")


def permission_risk(tc: dict) -> str | None:
    """危险操作判定（纯函数）。返回风险说明；无需拦截返回 None。

    只对 run_shell 生效：命令内容不可静态预判的工具（如 run_test 的自动推断命令）
    由 approval_reason() 在 host 模式下整体兜住。
    """
    if tc.get("name") != "run_shell":
        return None
    args = tc.get("args") or {}
    if args.get("allow_network"):
        return "命令需要网络访问（可能下载代码或推送变更）"
    cmd = args.get("command", "")
    for pat, why in DANGEROUS_PATTERNS:
        if re.search(pat, cmd):
            return why
    return None


def approval_reason(tc: dict) -> str | None:
    """需要人工审批的原因；None 表示可直接执行。"""
    reason = permission_risk(tc)
    if reason is None and tc.get("name") in APPROVAL_REQUIRED_TOOLS \
            and settings.exec_mode == "host":
        reason = "轻量模式（无 Docker 沙箱）：命令在本机直接执行"
    return reason


# ---------------- 人工审批 ----------------


class Review(NamedTuple):
    """一次工具调用的审批结论。"""

    action: str            # "allow" | "feedback" | "abort"
    message: str = ""


def review_tool_call(tc: dict) -> Review:
    """人工审批中间件：需要审批时 `interrupt()` 挂起等人拍板，再解释结果。

    - `allow`    ：放行，正常执行
    - `feedback` ：用户拒绝但任务继续 —— 把拒绝原因喂回模型，让它换一种做法
    - `abort`    ：用户拒绝的是**危险操作** —— 终止任务（保持既有安全语义）

    注意：本函数依赖 `interrupt()`，只能在图的节点内调用（且图必须带 checkpointer）。
    """
    reason = approval_reason(tc)
    if reason is None:
        return Review("allow")
    decision = interrupt({
        "question": f"是否允许执行以下操作？\n工具：{tc['name']}\n参数：{tc['args']}\n原因：{reason}",
        "tool": tc["name"],
        "args": tc["args"],
        "reason": reason,
    })
    if (decision or {}).get("approved"):
        return Review("allow")
    if permission_risk(tc):
        # 危险操作被拒 → 终止任务，不留给模型"换个写法绕过"的机会。
        # 这句文案被 reflect 的短路判断和 finish 依赖，改动需同步（grep "用户已拒绝"）。
        return Review("abort", f"用户已拒绝执行该危险操作（{reason}），任务已终止。")
    return Review(
        "feedback",
        f"用户拒绝了这条命令（{reason}）。"
        f"请改用不执行该命令的方式完成任务，或先向用户说明必要性。",
    )
