"""跨线程长期记忆：LangGraph Store（替代早期按项目落盘的 memory.md）。

为什么换：
- 早期 remember_fact / recall_memory 直接读写 <agent>/.agent_cache/projects/<项目id>/memory.md，
  属于"自己实现的存储"：没有 namespace、没有统一查询接口、也不受图管理；
- 现在改用 LangGraph Store —— 图用 store= 参数持有它（见 agent.py 的 graph.compile），
  工具内用 get_store() 取用，项目隔离靠 namespace：("memory", 项目id)。
  checkpoint 解决"同一会话跨运行"，Store 解决"跨会话/跨线程的长期信息"，两者分工明确。

并发：SqliteStore 自带 threading.Lock，webapp.py 的 MAX_RUNNING=4 并发任务安全。

迁移：首个进程启动时把历史 memory.md 内容导入 Store（幂等，见 migrate_from_memory_md）。
"""
from __future__ import annotations

import re
import sqlite3
import uuid
from datetime import datetime

from langgraph.config import get_config, get_store
from langgraph.store.sqlite import SqliteStore

import workspace

# 与 checkpoint 分文件存放，互不干扰（checkpoint 由 langgraph 自己建表）
STORE_DB = workspace.STATE_ROOT / "store.sqlite"

MEMORY_NAMESPACE = "memory"   # namespace = (MEMORY_NAMESPACE, 项目id)
RECALL_LIMIT = 200            # Store.search 默认只回 10 条，长期记忆必须放宽

#: 早期 memory.md 的行格式：`- [2026-09-10 14:20:11] 事实内容`
_LEGACY_LINE = re.compile(r"^-\s*\[([^\]]+)\]\s*(.*)$")


def _new_conn() -> sqlite3.Connection:
    STORE_DB.parent.mkdir(parents=True, exist_ok=True)
    # 两个参数都与官方 SqliteStore.from_conn_string 保持一致：
    # - check_same_thread=False：SqliteStore 内部有 lock，允许跨线程用同一连接
    # - isolation_level=None：自动提交模式。SqliteStore 自己用显式 BEGIN/COMMIT 管事务，
    #   若走 sqlite3 默认的隐式事务，它的 BEGIN 会撞上 "cannot start a transaction
    #   within a transaction"（实测踩过）。
    return sqlite3.connect(str(STORE_DB), check_same_thread=False, isolation_level=None)


#: 全局单例。agent.py 用它编译图；工具内实际取到的是同一个对象（见 _active_store）。
store = SqliteStore(_new_conn())
store.setup()   # 建表 + 迁移，幂等（langgraph 要求首次使用前调用）


def namespace(pid: str | None = None) -> tuple[str, str]:
    """按项目隔离的 namespace：一份 Store 承载所有项目，彼此不可见。"""
    return (MEMORY_NAMESPACE, pid or workspace.current_id())


def _active_store(st=None) -> SqliteStore:
    """优先用图注入的 store，图外调用回退到模块单例。

    图注入的那份就是这里编译时用的单例，正常运行时两者相同；
    显式传参主要是给单测留口子（可用临时 Store，不碰真实数据文件）。
    """
    if st is not None:
        return st
    try:
        return get_store() or store
    except Exception:  # noqa: BLE001 —— 图外（无 runtime 上下文）调用
        return store


def _thread_id() -> str | None:
    """记录这条事实来自哪个会话（跨线程溯源）；图外调用返回 None。"""
    try:
        return (get_config().get("configurable") or {}).get("thread_id")
    except Exception:  # noqa: BLE001
        return None


def _put(text: str, *, ts: str, st=None, pid=None, thread_id=None) -> dict:
    """单条写入。key 用"时间戳+随机后缀"，保持与旧 memory.md 一致的追加语义。"""
    key = f"{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"
    value = {"fact": text, "ts": ts, "thread_id": thread_id or _thread_id()}
    _active_store(st).put(namespace(pid), key, value)
    return {"key": key, **value}


def remember(fact: str, *, st=None, pid=None, thread_id=None) -> dict:
    """写入一条长期事实，返回落库后的条目。"""
    text = " ".join((fact or "").split())
    if not text:
        raise ValueError("要记住的内容不能为空")
    return _put(text, ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                st=st, pid=pid, thread_id=thread_id)


def recall(query: str = "", *, st=None, pid=None, limit: int = RECALL_LIMIT) -> list[dict]:
    """读取长期记忆，按时间升序。query 非空时按关键词过滤（与旧行为一致）。"""
    items = _active_store(st).search(namespace(pid), limit=limit)
    rows = [{"key": it.key, **(it.value or {})} for it in items]
    rows.sort(key=lambda r: (str(r.get("ts") or ""), str(r.get("key") or "")))
    q = (query or "").strip().lower()
    if q:
        rows = [r for r in rows if q in str(r.get("fact", "")).lower()]
    return rows


def recall_text(query: str = "", *, st=None, pid=None) -> str:
    """给模型看的文本视图（保持与早期 memory.md 相同的呈现形式）。"""
    rows = recall(query, st=st, pid=pid)
    if not rows:
        return "（没有匹配的记忆）" if (query or "").strip() else "（记忆库为空）"
    return "\n".join(f"- [{r.get('ts', '?')}] {r.get('fact', '')}" for r in rows)


def migrate_from_memory_md(projects: list[dict] | None = None, *, st=None) -> list[str]:
    """把历史 memory.md 导入 Store。幂等：该项目的 namespace 已有内容就跳过。

    旧格式每行是 `- [时间] 事实`；解析不出时间的行按原样导入。
    返回迁移说明（供 CLI / Web 启动时打印）。
    """
    if projects is None:
        try:
            projects = workspace.list_projects().get("projects", [])
        except Exception:  # noqa: BLE001 —— 注册表缺失不应阻断启动
            projects = []

    notes: list[str] = []
    for p in projects:
        pid = p.get("id")
        if not pid:
            continue
        md = workspace.PROJECTS_STATE_DIR / pid / "memory.md"
        if not md.exists():
            continue
        if recall("", st=st, pid=pid):
            continue   # 已经导入过，不重复
        count = 0
        for raw in md.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            m = _LEGACY_LINE.match(line)
            ts = m.group(1).strip() if m else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            fact = " ".join(((m.group(2) if m else line.lstrip("- ")).split()))
            if not fact:
                continue
            _put(fact, ts=ts, st=st, pid=pid, thread_id=None)
            count += 1
        if count:
            notes.append(f"长期记忆迁移：{p.get('name') or pid} 从 memory.md 导入 {count} 条")
    return notes
