"""LangGraph 有状态代码助手：plan -> retrieve -> execute -> reflect -> finish。

语言/项目无关：工作区由 workspace.py 在运行时决定（界面里可随时添加/切换项目），
每个任务在自己线程里绑定工作区，所以并发跑不同项目也不会串。

- plan：Pro 模型（thinking 开）产出步骤计划
- retrieve：RAG 混合检索（向量+BM25+重排），注入相关代码上下文（按项目隔离的索引）
- execute：Flash 模型 + 工具调用（文件读写 + 项目识别 + 命令/测试）
  - 危险操作（rm -rf / git push / sudo / 联网命令等）触发 interrupt() 人工审批
- reflect：Pro 模型结构化输出（thinking 关），判断是否完成
- finish：Pro 模型总结汇报
- 记忆（两层，都不写进被操作的用户仓库）：
  - 会话历史：State.messages（add_messages reducer）随 Checkpointer 按 thread_id 持久化
  - 长期偏好：remember_fact / recall_memory 走 LangGraph Store（见 memory.py），
    以 namespace ("memory", 项目id) 隔离，跨会话、跨线程可查
- 横切能力（重试 / 摘要 / 权限控制 / 人工审批）自实现于 middleware.py，本文件只做编排：
  - 重试：所有模型调用走 middleware.call_model（指数退避 + 抖动）
  - 摘要：plan 里先 compact_history 折叠长会话，prompt 统一用 middleware.history_text
  - 权限 + 审批：execute 里每次工具调用先过 middleware.review_tool_call
"""
import sqlite3
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel, Field

from config import settings, get_llm
from middleware import call_model, compact_history, history_text, review_tool_call
from tools import TOOLS
from rag import retrieve
import memory
import workspace

MAX_ITERATIONS = 5   # 全局循环上限：execute→reflect 最多跑 5 轮（由 reflect 计数、route 判定）
MAX_TOOL_ROUNDS = 4  # 单次 execute 内最多工具往返次数
RETRIEVE_K = 4       # 注入上下文的代码片段数

# 危险命令模式与审批判定见 middleware.py（permission_risk / approval_reason / review_tool_call）


class AgentState(TypedDict):
    task: str
    plan: str
    context: str   # RAG 检索到的相关代码片段
    result: str
    feedback: str
    done: bool
    iterations: int
    history_summary: str      # 摘要中间件折叠出的"更早对话摘要"
    history_summarized: int   # 已折叠到第几条，避免重复摘要同一批消息
    messages: Annotated[list, add_messages]  # 会话历史，随 checkpoint 持久化、跨运行回放


class Decision(BaseModel):
    done: bool = Field(description="任务是否已完成")
    feedback: str = Field(description="未完成时给出下一步具体反馈；完成时可留空")


def _project_brief() -> str:
    """当前项目简报（路径/语言/测试命令）。每次都实时解析，所以切换项目后立刻正确。"""
    try:
        info = workspace.describe_project(workspace.current(), deep=False)
        line = f"当前项目：{info['path']}\n语言：{info['lang']}"
        if info.get("manifests"):
            line += f"（清单：{', '.join(info['manifests'])}）"
        line += f"\n测试命令：{info.get('test_cmd') or '（未能推断，改用 run_shell 显式指定）'}"
        return line
    except Exception as e:  # noqa: BLE001 —— 简报失败不该阻断任务
        return f"当前项目：{workspace.current()}（识别失败：{e}）"


def plan(state: AgentState) -> dict:
    # 摘要中间件：长会话先折叠成摘要（消息不够多时不触发，也不会产生额外模型调用）
    folded = compact_history(state, get_llm(settings.executor_model, thinking=False))
    view = {**state, **folded} if folded else state
    llm = get_llm(settings.planner_model, thinking=True)
    msg = call_model(llm, [
        SystemMessage("你是资深代码助手。针对任务给出简洁、可执行的步骤计划，直接列步骤。"
                      "项目可能是任意语言/技术栈，不要假设是 Python；不确定项目结构时，"
                      "计划里应先安排用工具了解项目（describe_project / list_files）。"
                      "若任务涉及用户偏好或历史事实，先结合对话历史/记忆作答。"),
        HumanMessage(f"任务：{state['task']}\n{_project_brief()}\n对话历史：\n{history_text(view)}\n"
                     f"上一轮反馈：{state.get('feedback') or '无'}"),
    ], label="plan")
    # 注意：iterations 不在这里递增。route() 循环回到的是 execute，plan 只在开头跑一次，
    # 早期把计数放在这里会让 MAX_ITERATIONS 永远不生效（死代码）——现在由 reflect 计数。
    out = {"plan": msg.content}
    if folded:
        out.update(folded)
    return out


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
    hint = ("（未启用 RAG。定位代码请先用 describe_project / list_files / search_code / read_file 工具，"
            "不要凭空猜路径，也不要假设项目是 Python。）" if not settings.rag_enabled
            else "（无相关代码上下文）")
    ctx = state.get("context") or hint
    msgs = [
        SystemMessage(
            "你是执行器。用工具完成任务，最后用一句中文总结执行结果。可参考相关代码上下文。"
            "项目可能是任意语言/技术栈：先描述项目再动手，跑测试用 run_test（会自动选命令），"
            "需要别的构建/包管理命令时用 run_shell。不要凭空猜文件路径。"
            "若任务涉及『记住/询问用户偏好或事实』：要记住时调用 remember_fact 写入长期记忆；"
            "要查询时先调用 recall_memory 读取记忆再回答，不要凭空猜测。"
        ),
        HumanMessage(
            f"任务：{state['task']}\n{_project_brief()}\n计划：{state['plan']}\n"
            f"对话历史：\n{history_text(state)}\n"
            f"反馈：{state.get('feedback') or '无'}\n相关代码上下文：\n{ctx}"
        ),
    ]
    result = ""
    for _ in range(MAX_TOOL_ROUNDS):
        ai = call_model(llm, msgs, label="execute")
        msgs.append(ai)
        if not ai.tool_calls:
            result = ai.content or ""
            break
        for tc in ai.tool_calls:
            # 人工审批中间件：危险操作一律审批；host 模式（无沙箱）下命令逐条审批。
            # 拒绝后的两种处置（终止任务 / 反馈给模型换做法）由中间件判定，这里只执行结论。
            review = review_tool_call(tc)
            if review.action == "abort":
                return {"result": review.message}
            if review.action == "feedback":
                msgs.append(ToolMessage(content=review.message, tool_call_id=tc["id"]))
                continue
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
    # 每完成一轮 execute→reflect 计一次，供 route() 判定全局迭代上限。
    # 必须在这里（而不是 plan）递增：route() 循环回到的是 execute，plan 只在开头跑一次。
    # 两条 return 路径都要带上这个计数，否则走短路分支时计数会停住、护栏再次失效。
    n = state.get("iterations", 0) + 1
    if "用户已拒绝" in (state.get("result") or ""):
        return {"done": True, "feedback": "用户拒绝了危险操作，任务终止。", "iterations": n}
    llm = get_llm(settings.planner_model, thinking=False)
    decider = llm.with_structured_output(Decision, method="function_calling")
    d = call_model(decider, [
        SystemMessage("评估执行结果是否已达成任务目标，未达成时给出针对性反馈。"),
        HumanMessage(
            f"任务：{state['task']}\n计划：{state['plan']}\n对话历史：\n{history_text(state)}\n"
            f"执行结果：{state['result']}"
        ),
    ], label="reflect")
    return {"done": d.done, "feedback": d.feedback, "iterations": n}


def finish(state: AgentState) -> dict:
    llm = get_llm(settings.planner_model, thinking=False)
    # 达到迭代上限而任务仍未判定完成时，明确要求如实汇报。
    # 否则模型会"脑补成功"（README 第九节记录过同类事故：空结果被汇报成执行成功）。
    capped = not state.get("done") and state.get("iterations", 0) >= MAX_ITERATIONS
    sys_prompt = "用中文简洁汇报任务完成情况。"
    if capped:
        sys_prompt += (
            f"注意：已用满 {MAX_ITERATIONS} 轮迭代仍未判定任务完成，"
            "必须如实说明已完成到哪一步、还有什么没做完，不得声称任务已完成。"
        )
    summary = call_model(llm, [
        SystemMessage(sys_prompt),
        HumanMessage(f"任务：{state['task']}\n对话历史：\n{history_text(state)}\n执行结果：{state['result']}"),
    ], label="finish")
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

# 两层记忆各用各的机制：
# - Checkpointer：会话历史（State.messages）按 thread_id 持久化，解决"同一会话跨运行"。
#   落盘在【agent 自己目录】的 sqlite，一个 sqlite 承载所有项目，靠 thread_id 项目前缀隔离
#   （见 workspace.thread_key）。
# - Store：跨会话/跨线程的长期信息（用户偏好、项目约定），由 remember_fact / recall_memory
#   读写，解决"跨线程信息"。落盘在 .agent_cache/store.sqlite，靠 namespace
#   ("memory", 项目id) 隔离（见 memory.py）。工具内用 get_store() 取到的就是这里传进去的实例。
CHECKPOINT_DB = workspace.CHECKPOINT_DB
CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)
_conn = sqlite3.connect(str(CHECKPOINT_DB), check_same_thread=False)
checkpointer = SqliteSaver(_conn)

app = graph.compile(checkpointer=checkpointer, store=memory.store)
