"""CLI 入口：python main.py "你的任务" [--thread=会话id] [--project=/绝对路径]

安全模型（第九章）：命令默认在**内核围栏**里跑（写 项目目录+临时目录、禁网），
只在越界时才暂停询问（输入 y 批准、其他任意键拒绝）。

运行轨迹（第十四章）：任务执行过程中**边跑边打印**每一步 —— 当前节点、调用的模型、
工具名与参数、耗时、token、异常；同时落一份 JSONL 到
`.agent_cache/traces/<时间戳>_<项目id>_<runid>.jsonl`，事后再看：
    python main.py --trace            # 渲染最近一次任务的轨迹
    python main.py --trace=<文件>      # 渲染指定轨迹文件
    python main.py --no-trace         # 本次不打印、也不落盘（纯净输出）

可逆性：
  - 每次任务开始自动打一个 git 快照（.agent_cache/snapshots/<项目id>/<时间戳>/）
  - 任务结束打印"本次实际改动"（git 机器采集）与快照位置
  - python main.py --rollback            # 恢复到最近一次快照（要确认）
  - python main.py --rollback=<快照目录>  # 恢复到指定快照

--project 可以指向任意语言的任意代码库（不限于 Python）：
  python main.py "跑一下测试" --project=/Users/you/code/my-node-app
不传则用 Web 界面里最后选中的项目（workspace.py 的注册表）。
"""
import sys
import time
from pathlib import Path

import memory
import snapshot
import trace
import workspace
import sandbox
from agent import app
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from config import settings
from cost import tracker


def _ask_approval(question: str) -> bool:
    ans = input(f"\n⚠️  需要人工审批\n{question}\n是否允许？[y/N] ").strip().lower()
    return ans in ("y", "yes")


def _rollback(target: str, assume_yes: bool) -> int:
    """把工作区恢复到某个快照记录的状态（不跑任务）。默认用最近一次快照。"""
    d = Path(target) if target else snapshot.latest()
    if not d or not Path(d).is_dir():
        print("当前项目没有可回滚的快照（.agent_cache/snapshots/<项目id>/ 下没有记录）。")
        others = snapshot.recent()
        if others:
            print("最近的快照（可能是别的项目，用 --rollback=<目录> 指定）：")
            for item in others:
                print(f"  {item['path']}  →  {item['workspace']}")
        return 1
    target = snapshot.workspace_of(d) or workspace.current()
    print(f"将把 {target} 恢复到快照 {d}")
    print("  · 已跟踪文件回到快照时的 HEAD（之后的改动会丢失）")
    print("  · 快照时**已存在**的未提交改动会被贴回")
    print("  · 任务期间新建的未跟踪文件不会被删（会列出来让你确认）")
    if not assume_yes and input("确认继续？[y/N] ").strip().lower() not in ("y", "yes"):
        print("已取消。")
        return 1
    report = snapshot.restore(d)
    print(("✅ " if report.get("ok") else "❌ ") + report.get("message", ""))
    return 0 if report.get("ok") else 1


def _show_trace(target: str) -> int:
    """渲染一份已落盘的轨迹（默认最近一次），供事后复盘。"""
    path = Path(target) if target else trace.latest()
    if not path or not Path(path).is_file():
        print(f"没有找到轨迹文件（目录：{trace.trace_dir()}）。")
        files = trace.list_files(limit=10)
        if files:
            print("最近的轨迹：")
            for f in files:
                print(f"  {f['mtime']}  {f['name']}  {f['task'][:40]}")
        return 1
    print(f"轨迹文件：{path}\n" + "=" * 60)
    print(trace.render_events(path))
    return 0


def main() -> None:
    for note in workspace.migrate_legacy_state():
        print(f"[migrate] {note}")
    # 长期记忆从 memory.md 迁到 LangGraph Store（幂等，仅首次有输出）
    for note in memory.migrate_from_memory_md():
        print(f"[migrate] {note}")

    args = sys.argv[1:]
    thread_id = "main"
    project: str | None = None
    rollback: str | None = None
    show_trace: str | None = None
    no_trace = False
    assume_yes = False
    for a in args:
        if a.startswith("--thread="):
            thread_id = a.split("=", 1)[1]
        elif a.startswith("--project="):
            project = a.split("=", 1)[1]
        elif a == "--rollback":
            rollback = ""
        elif a.startswith("--rollback="):
            rollback = a.split("=", 1)[1]
        elif a == "--trace":
            show_trace = ""
        elif a.startswith("--trace="):
            show_trace = a.split("=", 1)[1]
        elif a == "--no-trace":
            no_trace = True
        elif a in ("--yes", "-y"):
            assume_yes = True
    task = " ".join(
        a for a in args
        if not a.startswith(("--thread=", "--project=", "--rollback", "--trace"))
        and a not in ("--yes", "-y", "--no-trace")
    )

    if project:
        entry = workspace.add_project(project)     # 加进注册表并切换为当前项目
        print(f"项目：{entry['name']}（{entry['lang']}）→ {entry['path']}")

    if rollback is not None:                       # 只做回滚，不跑任务
        sys.exit(_rollback(rollback, assume_yes))

    if show_trace is not None:                     # 只看轨迹，不跑任务
        sys.exit(_show_trace(show_trace))

    if not task:
        task = "先了解这个项目是什么，然后列出根目录下的文件"

    # 会话历史按项目隔离（同一 thread_id 在不同项目里是两段独立对话）
    config = {"configurable": {"thread_id": workspace.thread_key(thread_id)}}
    # 只重置本轮工作态；messages 走 add_messages reducer 会追加到历史，不会被清空
    # 任务级可逆性快照：围栏 + 快照 = 围栏内的破坏可以放手跑、出事能回滚（snapshot.py）
    snap = snapshot.begin(label=task)
    state = {"task": task, "plan": "", "context": "", "result": "",
             "feedback": "", "done": False, "iterations": 0,
             "history_summary": "", "history_summarized": 0,
             # 执行器工作通道（覆盖式通道，每次跑任务都重置）
             # exec_digest 是其中唯一"跨轮累积"的：由 reflect 每轮写回，见 agent._exec_digest()
             "exec_msgs": [], "pending_calls": [], "decisions": {},
             "tool_rounds": 0, "aborted": False, "exec_digest": "",
             "snapshot_path": str(snap.path) if snap.available else "",
             "messages": [HumanMessage(content=task)]}

    print(f"\n任务：{task}\n项目：{workspace.current()}\n会话：{thread_id}")
    print(f"执行环境：{sandbox.environment_note()}")
    print(f"🛟 {snap.describe()}")
    print("=" * 60)

    # 轨迹会话：整个任务期间边跑边打印（node / model / tool / retry / error 每一步），
    # 同时写 JSONL。`--no-trace` 关掉两条通道，`TRACE_CONSOLE=false` 可只落盘不刷屏。
    with trace.session(task=task, thread_id=thread_id, project_id=workspace.current_id(),
                       console=(False if no_trace else None),
                       enabled=(False if no_trace else None)) as tr:
        tr.emit("run.start", task=task, thread=thread_id, project=workspace.current_id(),
                workspace=str(workspace.current()), planner_model=settings.planner_model,
                executor_model=settings.executor_model, rag=settings.rag_enabled,
                exec_mode=sandbox.environment_note(),
                approval_policy=settings.resolved_approval_policy)
        if tr.path:
            tr.emit("log", message=f"轨迹文件：{tr.path}（事后复盘：python main.py --trace）")

        with tracker.session():
            try:
                final = app.invoke(state, config=config)
            except Exception as e:  # noqa: BLE001 —— 给出可自愈的提示，而不是裸 traceback
                tr.emit("error", where="app.invoke", error=f"{type(e).__name__}: {e}")
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
                value = pending.value or {}
                question = value.get("question", "是否允许该操作？")
                tr.emit("approval", name=value.get("tool", ""), args=value.get("args"),
                        reason=value.get("reason"), escalation=value.get("escalation") or None)
                waited = time.time()
                approved = _ask_approval(question)
                tr.emit("approval.result", name=value.get("tool", ""), approved=approved,
                        wait_ms=round((time.time() - waited) * 1000.0, 1))
                final = app.invoke(Command(resume={"approved": approved}), config=config)

            print("\n" + "=" * 60)
            print(final.get("result", ""))
            print("\n" + "=" * 60)
            change = snapshot.finish_after(snap)
            if change.available:
                print(f"📝 本次实际改动（git 采集）：{change.summary()}")
                if change.diff.strip():
                    print("-" * 60)
                    print(change.diff.rstrip()[:4000])
                print("-" * 60)
            if snap.available:
                print(f"🛟 {snap.describe()}")
                print(f"   回滚：python main.py --rollback（或 {snap.rollback_hint()}）")
                print("=" * 60)
            print("💰 本次成本统计：")
            print(tracker.summary())

    # 会话结束后再打印轨迹位置（此时文件名一定已经确定）
    if tr.path:
        print(f"🧭 本次运行轨迹：{tr.path}")
        print(f"   复盘：python main.py --trace    原始 JSONL：{tr.path}")


if __name__ == "__main__":
    main()
