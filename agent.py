"""LangGraph 有状态代码助手：plan -> retrieve -> execute -> reflect -> finish。

- plan：Pro 模型（thinking 开）产出步骤计划
- retrieve：RAG 混合检索（向量+BM25+重排），注入相关代码上下文
- execute：Flash 模型 + 工具调用（文件读写 + Docker 沙箱命令/测试）
  - 危险操作（rm -rf / git push / sudo / 联网命令等）触发 interrupt() 人工审批
- reflect：Pro 模型结构化输出（thinking 关），判断是否完成
- finish：Pro 模型总结汇报
- 记忆（两层）：
  - 会话历史：State.messages（add_messages reducer）随 checkpoint 持久化，跨运行回放对话
  - 长期偏好：remember_fact / recall_memory 工具落盘 .agent_cache/memory.md，跨会话可查
"""
import re
import sqlite3
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from config import settings, get_llm
from tools import TOOLS
from rag import retrieve

MAX_ITERATIONS = 5   # 防死循环：plan/execute/reflect 全局循环上限
MAX_TOOL_ROUNDS = 4  # 单次 execute 内最多工具往返次数
RETRIEVE_K = 4       # 注入上下文的代码片段数

# 危险命令模式：(正则, 风险说明)
DANGEROUS_PATTERNS = [
    (r"\brm\s+(-[a-z]*r[a-z]*f|-[a-z]*f[a-z]*r)", "递归删除文件"),
    (r"\bgit\s+push\b", "git 推送到远程"),
    (r"\bgit\s+reset\s+--hard\b", "git 硬重置（丢失提交）"),
    (r"\bsudo\b", "提权执行"),
    (r"\bchmod\s+777\b", "开放所有权限"),
    (r"\bshutdown\b|\breboot\b", "关机/重启"),
]


class AgentState(TypedDict):
    task: str
    plan: str
    context: str   # RAG 检索到的相关代码片段
    result: str
    feedback: str
    done: bool
    iterations: int
    messages: Annotated[list, add_messages]  # 会话历史，随 checkpoint 持久化、跨运行回放


class Decision(BaseModel):
    done: bool = Field(description="任务是否已完成")
    feedback: str = Field(description="未完成时给出下一步具体反馈；完成时可留空")


def _history_text(state: AgentState) -> str:
    """把持久化的会话消息（用户/助手轮次）压缩成文本注入 prompt，实现跨会话记忆。"""
    msgs = state.get("messages") or []
    if not msgs:
        return "（无历史）"
    lines = []
    for m in msgs:
        role = "用户" if isinstance(m, HumanMessage) else "助手"
        content = getattr(m, "content", "")
        if isinstance(content, str) and content.strip():
            lines.append(f"{role}：{content.strip()}")
    return "\n".join(lines[-16:]) or "（无历史）"


def _danger_check(tc: dict) -> str | None:
    """判断一次工具调用是否危险，返回风险说明；不危险返回 None。"""
    if tc.get("name") != "run_shell":
        return None
    args = tc.get("args") or {}
    cmd = args.get("command", "")
    if args.get("allow_network"):
        return "命令需要网络访问（可能下载代码或推送变更）"
    for pat, why in DANGEROUS_PATTERNS:
        if re.search(pat, cmd):
            return why
    return None


def plan(state: AgentState) -> dict:
    llm = get_llm(settings.planner_model, thinking=True)
    msg = llm.invoke([
        SystemMessage("你是资深代码助手。针对任务给出简洁、可执行的步骤计划，直接列步骤。"
                      "若任务涉及用户偏好或历史事实，先结合对话历史/记忆作答。"),
        HumanMessage(f"任务：{state['task']}\n对话历史：\n{_history_text(state)}\n"
                     f"上一轮反馈：{state.get('feedback') or '无'}"),
    ])
    return {"plan": msg.content, "iterations": state.get("iterations", 0) + 1}


def retrieve_node(state: AgentState) -> dict:
    """RAG 检索：根据任务+计划检索相关代码片段，注入上下文。检索失败不阻断主流程。"""
    if not settings.rag_enabled:
        # 轻量模式：不做 RAG（无需本地向量/重排模型），让执行器用工具自行定位代码
        return {"context": ""}
    query = f"{state['task']} {state['plan']}"
    try:
        hits = retrieve(query, k=RETRIEVE_K)
    except Exception as e:  # noqa: BLE001
        return {"context": f"[检索失败: {e}]"}
    if not hits:
        return {"context": ""}
    blocks = [f"### {h['path']}:{h['start']}-{h['end']}\n{h['text']}" for h in hits]
    return {"context": "\n\n".join(blocks)}


def execute(state: AgentState) -> dict:
    llm = get_llm(settings.executor_model, thinking=False).bind_tools(TOOLS)
    hint = ("（未启用 RAG。定位代码请先用 list_files / search_code / read_file 工具，"
            "不要凭空猜路径。）" if not settings.rag_enabled else "（无相关代码上下文）")
    ctx = state.get("context") or hint
    msgs = [
        SystemMessage(
            "你是执行器。用工具完成任务，最后用一句中文总结执行结果。可参考相关代码上下文。"
            "若任务涉及『记住/询问用户偏好或事实』：要记住时调用 remember_fact 写入长期记忆；"
            "要查询时先调用 recall_memory 读取记忆再回答，不要凭空猜测。"
        ),
        HumanMessage(
            f"任务：{state['task']}\n计划：{state['plan']}\n对话历史：\n{_history_text(state)}\n"
            f"反馈：{state.get('feedback') or '无'}\n相关代码上下文：\n{ctx}"
        ),
    ]
    result = ""
    for _ in range(MAX_TOOL_ROUNDS):
        ai = llm.invoke(msgs)
        msgs.append(ai)
        if not ai.tool_calls:
            result = ai.content or ""
            break
        for tc in ai.tool_calls:
            fn = next((t for t in TOOLS if t.name == tc["name"]), None)
            if fn is None:
                obs = f"未知工具: {tc['name']}"
            else:
                # 人工审批：
                #   - 危险操作（rm -rf / git push / sudo …）一律审批；
                #   - host 模式（无 Docker 沙箱）下，任何 run_shell / run_test 命令都逐条审批。
                reason = _danger_check(tc)
                if reason is None and tc["name"] in ("run_shell", "run_test") \
                        and settings.exec_mode == "host":
                    reason = "轻量模式（无 Docker 沙箱）：命令在本机直接执行"
                if reason:
                    decision = interrupt({
                        "question": f"是否允许执行以下操作？\n工具：{tc['name']}\n参数：{tc['args']}\n原因：{reason}",
                        "tool": tc["name"],
                        "args": tc["args"],
                        "reason": reason,
                    })
                    if not (decision or {}).get("approved"):
                        if _danger_check(tc):
                            # 危险操作被拒 → 终止任务（保持原有安全语义）
                            return {"result": f"用户已拒绝执行该危险操作（{reason}），任务已终止。"}
                        # host 模式普通命令被拒 → 不终止任务：把反馈喂回模型，让它换一种方式
                        msgs.append(ToolMessage(
                            content=f"用户拒绝了这条命令（{reason}）。"
                                    f"请改用不执行该命令的方式完成任务，或先向用户说明必要性。",
                            tool_call_id=tc["id"]))
                        continue
                try:
                    obs = fn.invoke(tc["args"])
                except Exception as e:  # noqa: BLE001 —— 工具异常回传给模型重试
                    obs = f"工具调用出错: {e}"
            msgs.append(ToolMessage(content=str(obs), tool_call_id=tc["id"]))
            result = str(obs)
    return {"result": result}


def reflect(state: AgentState) -> dict:
    if "用户已拒绝" in (state.get("result") or ""):
        return {"done": True, "feedback": "用户拒绝了危险操作，任务终止。"}
    llm = get_llm(settings.planner_model, thinking=False)
    decider = llm.with_structured_output(Decision, method="function_calling")
    d = decider.invoke([
        SystemMessage("评估执行结果是否已达成任务目标，未达成时给出针对性反馈。"),
        HumanMessage(
            f"任务：{state['task']}\n计划：{state['plan']}\n对话历史：\n{_history_text(state)}\n"
            f"执行结果：{state['result']}"
        ),
    ])
    return {"done": d.done, "feedback": d.feedback}


def finish(state: AgentState) -> dict:
    llm = get_llm(settings.planner_model, thinking=False)
    summary = llm.invoke([
        SystemMessage("用中文简洁汇报任务完成情况。"),
        HumanMessage(f"任务：{state['task']}\n对话历史：\n{_history_text(state)}\n执行结果：{state['result']}"),
    ])
    # 把最终答复写回 messages，随 checkpoint 持久化，供下一轮对话回放
    return {"result": summary.content, "messages": [AIMessage(content=summary.content)]}


def route(state: AgentState) -> str:
    if state.get("done") or state.get("iterations", 0) >= MAX_ITERATIONS:
        return "finish"
    return "execute"


graph = StateGraph(AgentState)
graph.add_node("plan", plan)
graph.add_node("retrieve", retrieve_node)
graph.add_node("execute", execute)
graph.add_node("reflect", reflect)
graph.add_node("finish", finish)
graph.add_edge(START, "plan")
graph.add_edge("plan", "retrieve")
graph.add_edge("retrieve", "execute")
graph.add_edge("execute", "reflect")
graph.add_conditional_edges("reflect", route, {"execute": "execute", "finish": "finish"})
graph.add_edge("finish", END)

# 会话记忆：checkpoint 持久化到项目内 sqlite，按 thread_id 做会话隔离（.agent_cache 已在 .gitignore）
# 长期偏好：由 remember_fact / recall_memory 工具落盘 .agent_cache/memory.md（见 tools.py）
CHECKPOINT_DB = settings.workspace_root / ".agent_cache" / "checkpoints.sqlite"
CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)
_conn = sqlite3.connect(str(CHECKPOINT_DB), check_same_thread=False)
checkpointer = SqliteSaver(_conn)

app = graph.compile(checkpointer=checkpointer)
