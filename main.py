"""CLI 入口：python main.py "你的任务" [--thread=会话id]

支持人工审批：执行到危险操作（rm -rf / git push / sudo / 联网命令等）时，
会暂停并询问是否允许，输入 y 批准、其他任意键拒绝。
"""
import sys

from agent import app
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from cost import tracker


def _ask_approval(question: str) -> bool:
    ans = input(f"\n⚠️  需要人工审批\n{question}\n是否允许？[y/N] ").strip().lower()
    return ans in ("y", "yes")


def main() -> None:
    args = sys.argv[1:]
    thread_id = "main"
    for a in args:
        if a.startswith("--thread="):
            thread_id = a.split("=", 1)[1]
    task = " ".join(a for a in args if not a.startswith("--thread="))
    if not task:
        task = "列出当前项目根目录下的文件，并说明这个项目是做什么的"

    config = {"configurable": {"thread_id": thread_id}}
    # 只重置本轮工作态；messages 走 add_messages reducer 会追加到历史，不会被清空
    state = {"task": task, "plan": "", "context": "", "result": "",
             "feedback": "", "done": False, "iterations": 0,
             "messages": [HumanMessage(content=task)]}

    tracker.reset()  # 每次运行清零统计
    print(f"\n任务：{task}\n会话：{thread_id}\n" + "=" * 60)

    try:
        final = app.invoke(state, config=config)
    except Exception as e:  # noqa: BLE001 —— 给出可自愈的提示，而不是裸 traceback
        print(f"\n❌ 运行出错：{type(e).__name__}: {e}")
        print("如果是旧会话的 checkpoint 历史损坏/不兼容导致，清空会话历史后重试即可：")
        print("  rm -f .agent_cache/checkpoints.sqlite")
        raise

    # 审批循环：只要还有 pending 的 interrupt，就询问用户并恢复
    while True:
        st = app.get_state(config)
        pending = None
        for t in (st.tasks or []):
            ints = getattr(t, "interrupts", None)
            if ints:
                pending = ints[0]
                break
        if pending is None:
            break
        question = (pending.value or {}).get("question", "是否允许该操作？")
        approved = _ask_approval(question)
        final = app.invoke(Command(resume={"approved": approved}), config=config)

    print("\n" + "=" * 60)
    print(final.get("result", ""))
    print("\n" + "=" * 60)
    print("💰 本次成本统计：")
    print(tracker.summary())


if __name__ == "__main__":
    main()
