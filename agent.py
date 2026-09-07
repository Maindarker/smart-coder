"""LangGraph 有状态代码助手骨架：plan -> execute -> reflect -> finish。

- plan：Pro 模型（thinking 开）产出步骤计划
- execute：Flash 模型 + 工具调用（最小文件工具集）
- reflect：Pro 模型结构化输出（thinking 关），判断是否完成
- finish：Pro 模型总结汇报

TODO 扩展点：
- RAG 检索（向量 + BM25 + 重排）在 execute 前加一个 retrieve 节点；
- 命令执行 / 测试 / Git 工具接入 Docker 沙箱；
- 中长期记忆：把 state 用 SqliteSaver 落盘 + thread_id 会话隔离；
- 危险操作（git push / rm）用 interrupt() 做 human-in-the-loop 审批。
"""
from typing_extensions import TypedDict
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

from config import settings, get_llm
from tools import TOOLS

MAX_ITERATIONS = 5   # 防死循环：plan/execute/reflect 全局循环上限
MAX_TOOL_ROUNDS = 4  # 单次 execute 内最多工具往返次数


class AgentState(TypedDict):
    task: str
    plan: str
    result: str
    feedback: str
    done: bool
    iterations: int


class Decision(BaseModel):
    done: bool = Field(description="任务是否已完成")
    feedback: str = Field(description="未完成时给出下一步具体反馈；完成时可留空")


def plan(state: AgentState) -> dict:
    llm = get_llm(settings.planner_model, thinking=True)
    msg = llm.invoke([
        SystemMessage("你是资深代码助手。针对任务给出简洁、可执行的步骤计划，直接列步骤。"),
        HumanMessage(f"任务：{state['task']}\n上一轮反馈：{state.get('feedback') or '无'}"),
    ])
    return {"plan": msg.content, "iterations": state.get("iterations", 0) + 1}


def execute(state: AgentState) -> dict:
    llm = get_llm(settings.executor_model, thinking=False).bind_tools(TOOLS)
    msgs = [
        SystemMessage("你是执行器。用工具完成任务，最后用一句中文总结执行结果。"),
        HumanMessage(
            f"任务：{state['task']}\n计划：{state['plan']}\n反馈：{state.get('feedback') or '无'}"
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
                try:
                    obs = fn.invoke(tc["args"])
                except Exception as e:  # noqa: BLE001 —— 工具异常回传给模型重试
                    obs = f"工具调用出错: {e}"
            msgs.append(ToolMessage(content=str(obs), tool_call_id=tc["id"]))
            result = str(obs)
    return {"result": result}


def reflect(state: AgentState) -> dict:
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
graph.add_node("execute", execute)
graph.add_node("reflect", reflect)
graph.add_node("finish", finish)
graph.add_edge(START, "plan")
graph.add_edge("plan", "execute")
graph.add_edge("execute", "reflect")
graph.add_conditional_edges("reflect", route, {"execute": "execute", "finish": "finish"})
graph.add_edge("finish", END)

app = graph.compile()
