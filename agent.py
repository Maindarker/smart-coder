"""LangGraph 有状态代码助手：plan -> retrieve -> execute -> reflect -> finish。

- plan：Pro 模型（thinking 开）产出步骤计划
- retrieve：RAG 混合检索（向量+BM25+重排），注入相关代码上下文
- execute：Flash 模型 + 工具调用（文件读写 + Docker 沙箱命令/测试）
  - 危险操作（rm -rf / git push / sudo / 联网命令等）触发 interrupt() 人工审批
- reflect：Pro 模型结构化输出（thinking 关），判断是否完成
- finish：Pro 模型总结汇报
- 记忆：SqliteSaver checkpoint 持久化，按 thread_id 会话隔离
"""
import re
import sqlite3
from typing_extensions import TypedDict
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
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


class Decision(BaseModel):
    done: bool = Field(description="任务是否已完成")
    feedback: str = Field(description="未完成时给出下一步具体反馈；完成时可留空")


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
        SystemMessage("你是资深代码助手。针对任务给出简洁、可执行的步骤计划，直接列步骤。"),
        HumanMessage(f"任务：{state['task']}\n上一轮反馈：{state.get('feedback') or '无'}"),
    ])
    return {"plan": msg.content, "iterations": state.get("iterations", 0) + 1}


def retrieve_node(state: AgentState) -> dict:
    """RAG 检索：根据任务+计划检索相关代码片段，注入上下文。检索失败不阻断主流程。"""
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
    ctx = state.get("context") or "（无相关代码上下文）"
    msgs = [
        SystemMessage("你是执行器。用工具完成任务，最后用一句中文总结执行结果。可参考相关代码上下文。"),
        HumanMessage(
            f"任务：{state['task']}\n计划：{state['plan']}\n反馈：{state.get('feedback') or '无'}\n"
            f"相关代码上下文：\n{ctx}"
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
                # 危险操作人工审批
                danger = _danger_check(tc)
                if danger:
                    decision = interrupt({
                        "question": f"是否允许执行以下危险操作？\n工具：{tc['name']}\n参数：{tc['args']}\n风险：{danger}",
                        "tool": tc["name"],
                        "args": tc["args"],
                        "reason": danger,
                    })
                    if not (decision or {}).get("approved"):
                        return {"result": f"用户已拒绝执行该危险操作（{danger}），任务已终止。"}
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
            f"任务：{state['task']}\n计划：{state['plan']}\n执行结果：{state['result']}"
        ),
    ])
    return {"done": d.done, "feedback": d.feedback}


def finish(state: AgentState) -> dict:
    llm = get_llm(settings.planner_model, thinking=False)
    summary = llm.invoke([
        SystemMessage("用中文简洁汇报任务完成情况。"),
        HumanMessage(f"任务：{state['task']}\n执行结果：{state['result']}"),
    ])
    return {"result": summary.content}


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

# 中长期记忆：checkpoint 持久化到项目内 sqlite，按 thread_id 做会话隔离（.agent_cache 已在 .gitignore）
CHECKPOINT_DB = settings.workspace_root / ".agent_cache" / "checkpoints.sqlite"
CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)
_conn = sqlite3.connect(str(CHECKPOINT_DB), check_same_thread=False)
checkpointer = SqliteSaver(_conn)

app = graph.compile(checkpointer=checkpointer)
