"""CLI 入口：python main.py "你的任务" [--thread=会话id] [--project=/绝对路径]

支持人工审批：执行到危险操作（rm -rf / git push / sudo / 联网命令等）时，
会暂停并询问是否允许，输入 y 批准、其他任意键拒绝。

--project 可以指向任意语言的任意代码库（不限于 Python）：
  python main.py "跑一下测试" --project=/Users/you/code/my-node-app
不传则用 Web 界面里最后选中的项目（workspace.py 的注册表）。
"""
import sys

import memory
import workspace
from agent import app
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from cost import tracker


def _ask_approval(question: str) -> bool:
    ans = input(f"\n⚠️  需要人工审批\n{question}\n是否允许？[y/N] ").strip().lower()
    return ans in ("y", "yes")


def main() -> None:
    for note in workspace.migrate_legacy_state():
        print(f"[migrate] {note}")
    # 长期记忆从 memory.md 迁到 LangGraph Store（幂等，仅首次有输出）
    for note in memory.migrate_from_memory_md():
        print(f"[migrate] {note}")

    args = sys.argv[1:]
    thread_id = "main"
    project: str | None = None
    for a in args:
        if a.startswith("--thread="):
            thread_id = a.split("=", 1)[1]
        elif a.startswith("--project="):
            project = a.split("=", 1)[1]
    task = " ".join(
        a for a in args if not a.startswith(("--thread=", "--project="))
    )
    if not task:
        task = "先了解这个项目是什么，然后列出根目录下的文件"

    if project:
        entry = workspace.add_project(project)     # 加进注册表并切换为当前项目
        print(f"项目：{entry['name']}（{entry['lang']}）→ {entry['path']}")

    # 会话历史按项目隔离（同一 thread_id 在不同项目里是两段独立对话）
    config = {"configurable": {"thread_id": workspace.thread_key(thread_id)}}
    # 只重置本轮工作态；messages 走 add_messages reducer 会追加到历史，不会被清空
    state = {"task": task, "plan": "", "context": "", "result": "",
             "feedback": "", "done": False, "iterations": 0,
             "history_summary": "", "history_summarized": 0,
             "messages": [HumanMessage(content=task)]}

    print(f"\n任务：{task}\n项目：{workspace.current()}\n会话：{thread_id}\n" + "=" * 60)

    with tracker.session():
        try:
            final = app.invoke(state, config=config)
        except Exception as e:  # noqa: BLE001 —— 给出可自愈的提示，而不是裸 traceback
            print(f"\n❌ 运行出错：{type(e).__name__}: {e}")
            print("如果是旧会话的 checkpoint 历史损坏/不兼容导致，清空会话历史后重试即可：")
            print(f"  rm -f {workspace.CHECKPOINT_DB}")
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
