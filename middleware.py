"""本项目的横切关注点：重试 / 摘要 / 权限控制 / 人工审批（自实现）。

为什么是"自实现"：需求要求"通过 Middleware 添加重试、摘要、权限控制和人工审批"。
框架的 `AgentMiddleware` 只能挂在 `create_agent` 上（`StateGraph.compile()` 没有
middleware 参数），而本项目是手写 StateGraph（见 agent.py）。按既定口径取宽松解释：
用"等价能力 + 明确接缝"自实现，agent.py 只负责编排。

四个接缝（全部可用纯本地单测覆盖，不需要调 API）：
- `call_model()`       重试     ：指数退避 + 抖动；参数/鉴权类错误不重试
- `compact_history()`  摘要     ：长会话折叠进 `history_summary`，替代"只截断最近 N 条"
- `permission_risk()`  权限控制 ：危险命令判定（规则在 shellrisk.py，按"围栏兜不兜得住"分级）
- `review_tool_call()` 人工审批 ：**只在越界 / 围栏兜不住 / 没人兜底时**才 `interrupt()` 找人拍板

审批的判据见下方"权限控制"一节：内核围栏（confinement.py）兜住越界、任务快照（snapshot.py）
兜住工作区内的破坏，两者都不适用时才打扰人 —— 这是业界主流（Codex / Claude Code / DSH）的取舍。

刻意不做的事：**不自动重试工具调用**。`run_shell` / `run_test` 有副作用，
盲目重试等于把命令跑两遍；工具异常仍按原策略回传给模型，由模型决定下一步
（见 agent.py 的 execute）。这与"重试只作用于可安全重复的模型调用"是一致的取舍。
"""
from __future__ import annotations

import random
import time
from typing import NamedTuple

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt

import sandbox
import shellrisk
import snapshot
import trace
from config import settings
from cost import tracker

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


def call_model(llm, messages, *, label: str = "model", model: str | None = None):
    """调用模型；可重试错误按指数退避 + 抖动重试，耗尽后抛出带上下文的错误。

    抖动是必要的：webapp 最多 4 个任务并发，固定退避会让它们同时重试、同时再撞墙。

    **模型调用是轨迹里最重要的一环**（trace.py）：整段包在一个 span 里，
    所以日志里能看到 label、模型名、prompt 摘要、耗时、token、重试次数与异常 ——
    这些在节点代码里是拿不到的整体信息（尤其是"重试了几次才成功"）。
    """
    name = model or trace.model_name_of(llm)
    with trace.model_call(label, name, messages) as sp:
        last: BaseException | None = None
        attempts = 0
        for attempt in range(MAX_MODEL_RETRIES + 1):
            attempts = attempt + 1
            before = tracker.totals()          # token 快照：结构化输出拿不到 usage_metadata
            try:
                resp = llm.invoke(messages)
            except Exception as e:  # noqa: BLE001 —— 重试策略统一在这里收敛
                if isinstance(e, _NON_RETRYABLE):
                    sp.annotate(attempts=attempts, model=name)
                    raise
                last = e
                if attempt >= MAX_MODEL_RETRIES:
                    break
                delay = min(RETRY_BACKOFF_BASE * (2 ** attempt), RETRY_BACKOFF_MAX)
                delay *= 0.5 + random.random()
                trace.emit("retry", name=label, model=name, attempt=attempts,
                           error=f"{type(e).__name__}: {e}", delay_ms=round(delay * 1000))
                if trace.current() is None:     # 没开轨迹时的兜底打印（保持老行为）
                    print(f"[retry] {label} 调用失败（{type(e).__name__}: {e}），"
                          f"{delay:.1f}s 后第 {attempts} 次重试")
                time.sleep(delay)
            else:
                usage = trace.usage_of(resp) or trace.usage_delta(before, tracker.totals(), name)
                content = getattr(resp, "content", "")
                sp.annotate(model=name, attempts=attempts,
                            tool_calls=len(getattr(resp, "tool_calls", None) or []) or None,
                            reply_chars=len(content) if isinstance(content, str) and content else None,
                            **usage)
                return resp
        sp.annotate(attempts=attempts, model=name)
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


# ---------------- 权限控制（consent 层）----------------
#
# 判据不是"看起来吓不吓人"，而是"**谁兜得住这条命令的副作用**"：
#
#   内核围栏（confinement.py）兜住"越界"  +  任务快照（snapshot.py）兜住"工作区内被破坏"
#
# 所以只有下面三种情况才打扰人：
#   1. 模型请求越界升级（escalate=network/full）—— 围栏之外，必须人放行；
#   2. 判定为 uncontained —— 网络/远端副作用、提权、设备级，围栏一概兜不住；
#   3. 判定为 contained，但当前**没有围栏或没有可回滚快照** —— 没人兜底。
# 其余情况（沙箱内的 `ls`、`rm -rf`、`chmod`…）直接执行、不再逐条弹窗 ——
# 这正是业界主流（Codex / Claude Code / DSH）的取舍：审批服务于"越界"，而不是"每一条"。
#
# **文件写入工具（write_file / edit_file）走的是同一条判据，结果是"不问"**：
# 它们只能改当前工作区内的文件（工具层 `_safe_path` 是第二道闸，越界直接抛 PermissionError），
# 所以副作用天然落在围栏里，可回滚性交给任务快照 —— 与"围栏内的 rm -rf"完全同类。
# 这也是把"改文件"从 `run_shell`（黑盒，只能保守判定）挪进语义化工具的主要收益：
# 同样的写入，走工具是"区内可回滚的已知副作用"→ 不打扰人；走 shell 是无法判定的字符串。

#: 这些工具在"逐条审批"策略（APPROVAL_POLICY=always）下每条都要人看一眼。
#: 只放**有副作用**的工具：只读工具（read_file / search_code / list_files…）永远不进这个名单。
APPROVAL_REQUIRED_TOOLS = ("run_shell", "run_test", "write_file", "edit_file")

#: 文件写入工具：副作用是"改一个工作区内的文件"，判据与围栏内的破坏同类。
#: 注意它们**没有** escalate/allow_network 参数 —— 越界由 tools._safe_path 直接拒绝，
#: 不提供"用一次审批换一次越界写"的通道（要越界就显式用 run_shell + escalate 走审批）。
FILE_MUTATION_TOOLS = ("write_file", "edit_file")

#: 逐条审批策略的说明文案（shell 分支与文件工具分支共用同一句，避免两处文案漂移）
ALWAYS_POLICY_REASON = "APPROVAL_POLICY=always：逐条审批（无围栏模式的兜底姿态）"


def _always_policy_reason(tc: dict) -> str | None:
    """APPROVAL_POLICY=always 是最保守的姿态：有副作用的工具逐条问人。

    只读工具不在 APPROVAL_REQUIRED_TOOLS 里，所以永远走不到这里。
    """
    if settings.resolved_approval_policy == "always" and tc.get("name") in APPROVAL_REQUIRED_TOOLS:
        return ALWAYS_POLICY_REASON
    return None


def escalation_request(tc: dict) -> str:
    """模型**显式请求**的越界档位："" | "network" | "full"。

    `allow_network=True` 等价于请求 network 档（历史参数，语义一致）。
    不认识的值按最高档处理 —— 仍然要审批，只是问得更宽，不会因此放宽安全。
    """
    args = tc.get("args") or {}
    req = str(args.get("escalate") or "").strip().lower()
    if req in ("network", "full"):
        return req
    if req in ("", "none", "false", "0"):
        return "network" if args.get("allow_network") else ""
    return "full"


def required_level(tc: dict) -> str:
    """批准之后，本次调用实际可以放宽到哪一档（模型请求与判定结论取更宽的）。

    文件写入工具**永远返回 ""**：它们的签名里没有 escalate/allow_network，
    执行侧也没有任何实现。否则模型只要在 args 里塞一个 escalate=full，
    审批弹窗和审计日志就会写成"批准后放宽到 full"——那是骗人的（实际什么都没放宽）。
    """
    if tc.get("name") in FILE_MUTATION_TOOLS:
        return ""
    cmd = (tc.get("args") or {}).get("command")
    return sandbox.escalation_needed(cmd if isinstance(cmd, str) else "",
                                     escalation_request(tc))


def permission_risk(tc: dict) -> str | None:
    """**保守口径**的危险判定（纯函数）：只要判定不是 safe 就给风险说明。

    这是"没有内核围栏"场景要用的口径（host 模式：contained 级的破坏也没人兜住）。
    有围栏时请用 approval_reason()，它按等级区分该不该打扰人。
    """
    args = tc.get("args") or {}
    if args.get("allow_network"):
        return "命令需要网络访问（可能下载代码或推送变更）"
    cmd = args.get("command")
    if not isinstance(cmd, str):
        return None
    return shellrisk.command_risk(cmd)


def approval_reason(tc: dict) -> str | None:
    """需要人工审批的原因；None 表示可以直接执行（沙箱内、有人兜底）。"""
    args = tc.get("args") or {}
    requested = escalation_request(tc)
    if requested:
        if requested == "network":
            return "请求越界升级：放行网络（其余仍受围栏限制）"
        return "请求越界升级：完全取消围栏（danger-full-access）"

    if tc.get("name") in FILE_MUTATION_TOOLS:
        # 文件写入工具：副作用就是"改一个工作区内的文件"，与围栏内的 rm -rf 同类 ——
        # 路径越界由工具层的 _safe_path 直接拒绝（那是第二道闸，不是审批事项），
        # 所以这里判的是"有没有人兜得住"：有围栏 + 有可回滚快照就不打扰人。
        where = str(args.get("path") or "?")
        if not sandbox.fence_available():
            return f"要写入文件 {where}，但当前没有强制隔离，没人兜底"
        if not snapshot.rollback_available():
            return f"要写入文件 {where}，但工作区没有可回滚的 git 快照"
        return _always_policy_reason(tc)

    cmd = args.get("command")
    verdict = (shellrisk.command_verdict(cmd) if isinstance(cmd, str)
               else shellrisk.Verdict(shellrisk.SAFE))

    if verdict.level == shellrisk.UNCONTAINED:
        return f"{verdict.reason}；围栏兜不住这类副作用"
    if verdict.needs:
        # 围栏能兜住它的文件效果，但它需要比当前策略更多的权限（典型是网络）
        # → 当场问一次，批准后只放行需要的那一档。别让它先撞墙失败、再回来申请（白跑一轮）。
        need = "放行网络" if verdict.needs == shellrisk.NEEDS_NETWORK else "完全取消围栏"
        return f"{verdict.reason}；围栏默认不允许，批准后只{need}"
    if verdict.level == shellrisk.CONTAINED:
        if not sandbox.fence_available():
            return f"{verdict.reason}；但当前没有强制隔离，没人兜底"
        if not snapshot.rollback_available():
            return f"{verdict.reason}；工作区没有可回滚的 git 快照"
    return _always_policy_reason(tc)


# ---------------- 人工审批 ----------------


class Review(NamedTuple):
    """一次工具调用的审批结论。"""

    action: str              # "allow" | "feedback" | "abort"
    message: str = ""
    escalation: str = ""     # 批准后本次调用可放宽到哪一档："" | "network" | "full"


def review_tool_call(tc: dict, *, context: str = "") -> Review:
    """人工审批中间件：需要审批时 `interrupt()` 挂起等人拍板，再解释结果。

    - `allow`    ：放行，正常执行（可能带一个放宽档位）
    - `feedback` ：用户拒绝但任务继续 —— 把拒绝原因喂回模型，让它换一种做法
    - `abort`    ：用户拒绝的是**围栏兜不住**的操作（远端副作用/提权/设备）—— 终止任务

    `APPROVAL_POLICY=never`（无人值守）时不弹窗，需要审批的动作**一律自动拒绝**
    （fail closed，与 DSH 的 never 语义一致）。

    注意：本函数依赖 `interrupt()`，只能在图的节点内调用（且图必须带 checkpointer）。
    ⚠️ 而且必须在一个**没有任何副作用**的节点里调用：LangGraph 恢复执行时会重放整个节点，
    若把 `interrupt()` 和"执行工具"放在同一个节点，重放会把已经执行过的工具**再执行一遍**
    （审计 §2.3，已实测）。agent.py 因此把审批单独拆成 gate 节点。
    """
    reason = approval_reason(tc)
    if reason is None:
        return Review("allow")

    level = required_level(tc)
    cmd = (tc.get("args") or {}).get("command")
    verdict = shellrisk.command_verdict(cmd) if isinstance(cmd, str) else shellrisk.Verdict(shellrisk.SAFE)
    hard = verdict.level == shellrisk.UNCONTAINED      # 围栏兜不住的 → 拒绝即终止

    if settings.resolved_approval_policy == "never":
        return Review("feedback",
                      f"[APPROVAL_POLICY=never] 该操作需要人工批准，已自动拒绝：{reason}。"
                      f"请改用不越界的方式完成任务。")

    detail = f"工具：{tc['name']}\n参数：{tc['args']}\n原因：{reason}"
    if level == "network":
        detail += "\n批准后：放行网络（其余仍受围栏限制）"
    elif level == "full":
        detail += "\n批准后：完全取消围栏（danger-full-access，命令可读写整机、可联网）"
    if context:
        detail += f"\n{context}"

    decision = interrupt({
        "question": f"是否允许执行以下操作？\n{detail}",
        "tool": tc["name"],
        "args": tc["args"],
        "reason": reason,
        "escalation": level,        # 新增字段：前端可展示"批准后会放宽到什么程度"
    })
    if (decision or {}).get("approved"):
        return Review("allow", escalation=level)
    if hard:
        # 围栏兜不住的操作被拒 → 终止任务，不留给模型"换个写法绕过"的机会。
        # 拒绝即终止的文案。注意：reflect 的短路判据是 state.aborted（结构化字段），
        # **不再**依赖这句话的字面 —— 这句话现在只用于汇报，改文案不会再让短路静默失效。
        return Review("abort", f"用户已拒绝执行该操作（{reason}），任务已终止。")
    return Review(
        "feedback",
        f"用户拒绝了这条命令（{reason}）。"
        f"请改用不需要越界、也不破坏工作区的方式完成任务，或先向用户说明必要性。",
    )
