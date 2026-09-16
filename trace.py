"""运行轨迹（observability）：把一次任务的每一步**边跑边打印**，同时落成 JSONL 日志。

需求：任务在跑的时候就能看清"现在到哪一步了、调的是哪个模型、调用了什么工具、参数是什么、
耗时多少、花了多少 token、有没有异常"；实时刷得太快看不过来时，还能事后翻本地日志。

三条输出通道（同一条记录，三种消费方式，互不影响）：

1. **控制台**：CLI / Web 服务端进程直接流式打印（每条 `flush=True`，所以是"边执行边出"，
   不是跑完才吐）。渲染逻辑只有 `render()` 一处，CLI 和浏览器用的是同一份文案。
2. **前端**：engine 把记录塞进 `on_event` → SSE → 浏览器里的「运行轨迹」面板实时追加。
3. **文件**：`.agent_cache/traces/<时间戳>_<项目id>_<runid>.jsonl`，一行一个 JSON 事件，
   方便 grep / jq / 事后复盘（也可以直接 `python -m json.tool` 看单条）。

事件模型（kind）：

    run.start / run.end            任务开始、结束（总耗时 / token / 调用次数 / 异常数）
    node.start / node.end          图节点（plan / retrieve / execute_model / gate / execute_tools / reflect / finish）
    model.start / model.end        一次模型调用（label、模型名、prompt 摘要、耗时、token）
    tool.start / tool.end          一次工具调用（工具名、参数、耗时、结果预览、异常）
    review                         审批判定（allow / feedback / abort 及其原因）
    approval / approval.result     挂起等人拍板、以及人的结论
    retry / error / log / rag      重试、异常、散点信息、RAG 检索结果
    snapshot                       任务级可逆快照的建立与结束

设计要点：

- **contextvar 会话**：Web 端同一进程里最多 4 个任务并发、每任务一个线程，所以轨迹要按线程隔离
  （与 cost.py 的分账、workspace.py 的工作区绑定同一套思路）。对外只有 `session()` 一个入口，
  任务外的 emit 全部是 no-op，任何模块都不需要判断"现在有没有在记轨迹"。
- **span 记耗时**：模型调用 / 工具调用 / 图节点各包一个 span —— 进入发 `.start`、出来发 `.end`
  （带 `duration_ms` 与 `status`），异常时附上 `error` 字段并**原样抛出**（记轨迹绝不吞异常）。
- **默认只记摘要**：日志里存 prompt 的角色/长度摘要、工具结果的预览，避免把整个仓库内容写进磁盘；
  需要完整 prompt / 完整工具输出时设 `TRACE_FULL_CONTENT=true`。
- 记轨迹本身**不改变任务语义**：所有打点都在 try/except 里，sink 抛异常也不会影响主流程
  （观测系统不该把被观测的流程搞挂）。
"""
from __future__ import annotations

import contextlib
import contextvars
import functools
import json
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from config import settings

#: 控制台/前端单条日志里参数、结果预览的最大长度（完整内容只在 TRACE_FULL_CONTENT 时落盘）
PREVIEW_CHARS = 400

#: 事件图标：控制台与浏览器共用，方便一扫就知道这一步是什么
_ICON = {
    "run": "▶",
    "node": "◆",
    "model": "🧠",
    "tool": "🔧",
    "review": "⚖️",
    "approval": "⏸",
    "rag": "🔍",
    "retry": "⚠️",
    "error": "❌",
    "snapshot": "🛟",
    "log": "·",
}


# ---------------- 小工具 ----------------


def trace_dir() -> Path:
    """轨迹目录（可在 .env 里用 TRACE_DIR 覆盖）。"""
    return Path(settings.trace_dir)


def _one_line(text: Any, limit: int = PREVIEW_CHARS) -> str:
    """压成一行并截断 —— 控制台一行一条、JSONL 一事件一行。"""
    s = str(text).replace("\r\n", "\n").replace("\r", "\n").strip()
    if len(s) > limit:
        s = s[:limit] + f"…（共 {len(s)} 字符，已截断）"
    return s.replace("\n", "⏎")


def _num(n: Any) -> str:
    try:
        return f"{int(n):,}"
    except (TypeError, ValueError):
        return str(n)


def _dur(ms: Any) -> str:
    """耗时的可读渲染：短于 1 秒给 ms，长于 1 秒给秒。"""
    try:
        f = float(ms)
    except (TypeError, ValueError):
        return "?"
    return f"{f:.0f}ms" if f < 1000 else f"{f / 1000:.2f}s"


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else f"{text[:limit]}…（共 {len(text)} 字符）"


#: 公开别名：其它模块要把一段文本压成一行日志时用它（`trace.brief(...)`）
brief = _one_line


def _jsonable(value: Any) -> Any:
    """把任意对象转成能进 JSON 的形式（工具参数里可能有 Path / 枚举之类）。"""
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except (TypeError, ValueError):
        if isinstance(value, dict):
            return {str(k): _jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_jsonable(v) for v in value]
        return str(value)


def clip_args(args: Any, limit: int = PREVIEW_CHARS) -> Any:
    """工具参数的落盘形式：结构保留，超长字符串值截断（避免把整个文件内容写进日志）。"""
    args = _jsonable(args)
    if isinstance(args, dict):
        out = {}
        for k, v in args.items():
            if isinstance(v, str) and len(v) > limit:
                out[k] = f"{v[:limit]}…（共 {len(v)} 字符）"
            else:
                out[k] = v
        return out
    if isinstance(args, str) and len(args) > limit:
        return f"{args[:limit]}…（共 {len(args)} 字符）"
    return args


def args_brief(args: Any) -> str:
    """参数的一行摘要（控制台里最好读的形式：key=值）。"""
    args = clip_args(args, limit=120)
    if isinstance(args, dict):
        if not args:
            return "（无参数）"
        bits = []
        for k, v in args.items():
            if isinstance(v, str):
                bits.append(f'{k}="{_one_line(v, 120)}"')
            else:
                bits.append(f"{k}={json.dumps(v, ensure_ascii=False, default=str)}")
        return _one_line(" ".join(bits), PREVIEW_CHARS)
    return _one_line(args, PREVIEW_CHARS)


def message_digest(messages: Sequence[Any] | None) -> dict:
    """prompt 摘要：消息条数、角色构成、总字符数（默认不落盘完整 prompt）。"""
    msgs = list(messages or [])
    roles: list[str] = []
    total = 0
    for m in msgs:
        role = getattr(m, "type", None) or type(m).__name__
        content = getattr(m, "content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, default=str)
        roles.append(f"{role}:{len(content)}")
        total += len(content)
    out: dict = {"messages": len(msgs), "prompt_chars": total, "roles": ",".join(roles)}
    if settings.trace_full_content:
        out["prompt"] = [
            _clip(str(getattr(m, "content", "")), 4000) for m in msgs
        ]
    return out


def tool_result_fields(obs: Any, limit: int = PREVIEW_CHARS) -> dict:
    """工具结果的落盘字段：长度 + 预览（首行），完整内容按开关决定是否记录。"""
    text = obs if isinstance(obs, str) else str(obs)
    fields = {"result_chars": len(text), "result_preview": _one_line(text, limit)}
    if settings.trace_full_content:
        fields["result"] = _clip(text, 20000)
    return fields


def model_name_of(llm: Any) -> str:
    """从 LangChain 模型对象上取模型名；取不到就退化成类名（测试替身也能记）。"""
    for attr in ("model_name", "model"):
        v = getattr(llm, attr, None)
        if isinstance(v, str) and v:
            return v
    return type(llm).__name__


def usage_of(response: Any) -> dict:
    """从模型响应里取 token 用量（usage_metadata / response_metadata）。"""
    meta = getattr(response, "usage_metadata", None)
    if not meta and getattr(response, "response_metadata", None):
        meta = (response.response_metadata or {}).get("token_usage")
    if not isinstance(meta, dict):
        return {}
    out = {
        "input_tokens": meta.get("input_tokens", meta.get("prompt_tokens")),
        "output_tokens": meta.get("output_tokens", meta.get("completion_tokens")),
        "total_tokens": meta.get("total_tokens"),
    }
    if out["total_tokens"] is None and out["input_tokens"] is not None:
        out["total_tokens"] = (out["input_tokens"] or 0) + (out["output_tokens"] or 0)
    return {k: int(v) for k, v in out.items() if v is not None}


def usage_delta(before: dict | None, after: dict | None, model: str | None = None) -> dict:
    """两次成本快照的差值 —— 结构化输出（`.with_structured_output()`）拿不到 usage_metadata，
    但挂在模型上的 LangChain callback 仍然记了账，所以用差值补齐 token 信息。

    `before` / `after` 是 `cost.CostTracker.totals()` 的形状：{calls, input, output, total}。
    """
    if not before or not after:
        return {}
    out: dict = {}
    for key, field in (("input", "input_tokens"), ("output", "output_tokens"),
                       ("total", "total_tokens")):
        delta = int(after.get(key, 0)) - int(before.get(key, 0))
        if delta:
            out[field] = delta
    if out:
        out["tokens_from"] = f"callbacks:{model}" if model else "callbacks"
    return out


# ---------------- 记录渲染（CLI 与浏览器共用同一份文案） ----------------


def render(rec: dict) -> str:
    """把一条记录渲染成一行给人看的中文文案。

    刻意做成一行的：流式刷屏时一行 = 一步，扫一眼就知道走到哪了。
    """
    kind = rec.get("kind", "")
    name = rec.get("name") or rec.get("node") or rec.get("tool") or rec.get("label") or "-"
    head = f"[{rec.get('time', '')}]"
    if kind == "run.start":
        return (f"{head} ▶ 任务开始 · 项目 {rec.get('project') or '-'} · 线程 {rec.get('thread') or '-'}\n"
                f"          任务：{_one_line(rec.get('task') or '', 200)}\n"
                f"          模型：规划 {rec.get('planner_model') or '-'} / 执行 {rec.get('executor_model') or '-'}"
                f" · RAG {'开' if rec.get('rag') else '关'}\n"
                f"          环境：{_one_line(rec.get('exec_mode') or '-', 200)}")
    if kind == "run.end":
        return (f"{head} ▶ 任务结束 · 耗时 {_dur(rec.get('duration_ms'))}"
                f" · 模型 {rec.get('model_calls', 0)} 次 / 工具 {rec.get('tool_calls', 0)} 次"
                f" · token {_num(rec.get('total_tokens', 0))}"
                f"（入 {_num(rec.get('input_tokens', 0))} / 出 {_num(rec.get('output_tokens', 0))}）"
                f" · 异常 {rec.get('errors', 0)}"
                + (f" · 重试 {rec['retries']} 次" if rec.get("retries") else "")
                + (f"\n          📄 轨迹文件：{rec['trace_file']}" if rec.get("trace_file") else ""))
    if kind == "node.start":
        return f"{head} ◆ 节点 [{name}] 开始"
    if kind == "node.end":
        tail = f" · {_one_line(rec['summary'], 200)}" if rec.get("summary") else ""
        return (f"{head} ◆ 节点 [{name}] 结束 · {_dur(rec.get('duration_ms'))}{tail}"
                + ("  ⚠️ 重放（审批恢复）" if rec.get("replay_note") else ""))
    if kind == "model.start":
        return (f"{head} 🧠 模型 [{name}] {rec.get('model') or '?'} · 请求中"
                f"（{rec.get('messages', 0)} 条消息 / {_num(rec.get('prompt_chars', 0))} 字符）")
    if kind == "model.end":
        tok = ""
        if rec.get("total_tokens") is not None:
            tok = (f" · token {_num(rec.get('total_tokens'))}"
                   f"（入 {_num(rec.get('input_tokens', 0))} / 出 {_num(rec.get('output_tokens', 0))}）")
        elif rec.get("tokens_from"):
            tok = " · token 由回调统计（结构化输出不返回用量）"
        ok = f"✔ {rec.get('model') or name}"
        if rec.get("status") == "error":
            return f"{head} ❌ 模型 [{name}] 失败 · {_dur(rec.get('duration_ms'))} · {rec.get('error')}"
        return (f"{head} 🧠 模型 [{name}] {ok} · 耗时 {_dur(rec.get('duration_ms'))}{tok}"
                + (f" · 产出 {rec.get('tool_calls')} 个工具调用" if rec.get("tool_calls") else "")
                + (f" · 重试 {rec['retries']} 次" if rec.get("retries") else ""))
    if kind == "tool.start":
        return (f"{head} 🔧 工具 [{name}] 调用 · {args_brief(rec.get('args'))}"
                + (f" · 放宽档位={rec['escalation']}" if rec.get("escalation") else ""))
    if kind == "tool.end":
        if rec.get("status") == "error":
            return (f"{head} ❌ 工具 [{name}] 异常 · {_dur(rec.get('duration_ms'))}"
                    f" · {rec.get('error')}")
        return (f"{head} 🔧 工具 [{name}] 完成 · {_dur(rec.get('duration_ms'))}"
                f" · 结果 {_num(rec.get('result_chars', 0))} 字符"
                + (f" · {_one_line(rec.get('result_preview'), 300)}"
                   if rec.get("result_preview") else ""))
    if kind == "review":
        action = {"allow": "✅ 放行（无需打扰人）", "feedback": "↩️ 拒绝并反馈给模型",
                  "abort": "⛔ 拒绝并终止任务"}.get(rec.get("action"), rec.get("action"))
        why = f" · 原因：{_one_line(rec.get('reason'), 160)}" if rec.get("reason") else ""
        return f"{head} ⚖️ 审批判定 [{name}] {action}{why}"
    if kind == "approval":
        return (f"{head} ⏸ 等待人工审批 [{name}] · {_one_line(rec.get('reason'), 160)}"
                f" · 参数 {args_brief(rec.get('args'))}")
    if kind == "approval.result":
        return (f"{head} ▶ 审批结论 [{name}] "
                f"{'已批准' if rec.get('approved') else '已拒绝'}"
                + (f"（等待 {_dur(rec.get('wait_ms'))}）" if rec.get("wait_ms") is not None else ""))
    if kind == "rag":
        if rec.get("disabled"):
            return f"{head} 🔍 RAG 检索 · 已关闭（RAG_ENABLED=false，改用 search_code 等工具定位代码）"
        return (f"{head} 🔍 RAG 检索 · 命中 {rec.get('hits', 0)} 段"
                + (f" · {', '.join(rec.get('files') or [])}" if rec.get("files") else ""))
    if kind == "retry":
        return (f"{head} ⚠️ [{name}] 第 {rec.get('attempt')} 次失败（{rec.get('error')}），"
                f"{_dur(rec.get('delay_ms'))} 后重试")
    if kind == "snapshot":
        return f"{head} 🛟 快照 {rec.get('state')} · {_one_line(rec.get('detail'), 200)}"
    if kind == "error":
        return f"{head} ❌ {rec.get('where') or '运行'} 出错 · {_one_line(rec.get('error'), 300)}"
    if kind == "log":
        return f"{head} · {_one_line(rec.get('message'), 300)}"
    return f"{head} · {kind} {json.dumps({k: v for k, v in rec.items() if k not in ('ts', 'time', 'seq')}, ensure_ascii=False, default=str)[:300]}"


# ---------------- 核心：Tracer ----------------


class Span:
    """一次"有开始有结束"的操作（模型调用 / 工具调用 / 图节点）。

    用法：

        with trace.model_call("plan", model="deepseek-v4-pro", messages=msgs) as sp:
            resp = llm.invoke(msgs)
            sp.annotate(**trace.usage_of(resp))

    进入时发 `<kind>.start`，退出时发 `<kind>.end`（带 `duration_ms`、`status`）；
    体内抛异常时附 `error` 字段并**原样抛出** —— 观测层不吞异常、不改变控制流。
    """

    def __init__(self, tracer: "Tracer | None", kind: str, name: str, fields: dict):
        self.tracer = tracer
        self.kind = kind
        self.name = name
        self.fields = dict(fields)
        self.extra: dict = {}
        self.duration_ms: float | None = None
        self._t0 = 0.0

    def annotate(self, **fields) -> "Span":
        """补充要写进 `.end` 的字段（token、结果预览、工具调用数……）。"""
        self.extra.update({k: v for k, v in fields.items() if v is not None})
        return self

    @property
    def enabled(self) -> bool:
        return self.tracer is not None

    def __enter__(self) -> "Span":
        self._t0 = time.perf_counter()
        if self.tracer is not None:
            self.tracer.emit(f"{self.kind}.start", name=self.name, **self.fields)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.duration_ms = (time.perf_counter() - self._t0) * 1000.0
        if self.tracer is not None:
            fields = dict(self.fields)
            fields.update(self.extra)
            fields.update(name=self.name, duration_ms=round(self.duration_ms, 1))
            if exc is not None:
                fields["status"] = "error"
                fields["error"] = f"{type(exc).__name__}: {exc}"
            else:
                fields.setdefault("status", "ok")
            self.tracer.emit(f"{self.kind}.end", **fields)
        return False        # 绝不吞异常


class Tracer:
    """一次任务的轨迹记录器：写 JSONL + 打印控制台 + 推给 sink（前端 SSE）。"""

    def __init__(
        self,
        *,
        task: str = "",
        thread_id: str = "",
        project_id: str = "",
        run_id: str | None = None,
        directory: Path | str | None = None,
        console: bool | None = None,
        enabled: bool | None = None,
        sinks: Iterable[Callable[[dict, str], None]] = (),
    ) -> None:
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.task = task
        self.thread_id = thread_id
        self.project_id = project_id or "default"
        self.dir = Path(directory) if directory else trace_dir()
        self.console = settings.trace_console if console is None else bool(console)
        self._enabled = enabled
        self.sinks = list(sinks)
        self._seq = 0
        self._t0 = time.time()
        self._lock = threading.Lock()
        self._fh = None
        self._closed = False
        self._path: Path | None = None
        #: 计数：run.end 里汇报（也方便事后一眼看成本）
        self.stats = {"model_calls": 0, "tool_calls": 0, "errors": 0, "retries": 0,
                      "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    # ---- 落盘 ----

    @property
    def enabled(self) -> bool:
        """是否落盘 JSONL（TRACE_ENABLED=false 或 CLI `--no-trace` 时只打印、不写文件）。"""
        return bool(settings.trace_enabled) if self._enabled is None else bool(self._enabled)

    @property
    def path(self) -> Path | None:
        return self._path

    def _open(self) -> None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.dir.mkdir(parents=True, exist_ok=True)
        self._path = self.dir / f"{stamp}_{self.project_id}_{self.run_id}.jsonl"
        self._fh = self._path.open("a", encoding="utf-8")

    def _write(self, rec: dict) -> None:
        if not self.enabled:
            return
        try:
            if self._fh is None:
                self._open()
            assert self._fh is not None
            self._fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            self._fh.flush()          # 边跑边落盘：进程被杀也能看到已经跑到哪一步
        except Exception:  # noqa: BLE001 —— 观测失败绝不影响任务
            pass

    # ---- 事件 ----

    def emit(self, kind: str, **fields) -> dict:
        with self._lock:
            self._seq += 1
            rec: dict = {
                "ts": round(time.time(), 3),
                "time": datetime.now().strftime("%H:%M:%S.%f")[:-3],
                "run": self.run_id,
                "project": self.project_id,
                "seq": self._seq,
                "elapsed_ms": round((time.time() - self._t0) * 1000.0, 1),
                "kind": kind,
            }
            rec.update({k: _jsonable(v) for k, v in fields.items() if v is not None})
            self._count(rec)
            line = ""
            try:
                line = render(rec)
            except Exception:  # noqa: BLE001 —— 渲染失败也不能丢事件
                line = f"[{rec['time']}] · {kind}"
            self._write(rec)
            if self.console:
                try:
                    print(line, flush=True)
                except Exception:  # noqa: BLE001
                    pass
            for sink in self.sinks:
                try:
                    sink(rec, line)
                except Exception:  # noqa: BLE001 —— 前端断了不该影响任务
                    pass
            return rec

    def _count(self, rec: dict) -> None:
        kind = rec.get("kind")
        if kind == "model.end":
            self.stats["model_calls"] += 1
            for key in ("input_tokens", "output_tokens", "total_tokens"):
                self.stats[key] += int(rec.get(key) or 0)
        elif kind == "tool.end":
            self.stats["tool_calls"] += 1
        elif kind == "retry":
            # 重试单独计数：它是"发生过但被救回来"的异常，和最终失败的 error 不是一回事
            self.stats["retries"] += 1
        if rec.get("status") == "error" or kind == "error":
            self.stats["errors"] += 1

    def close(self, **fields) -> None:
        """收尾：写 run.end 并关文件（重复调用安全 —— 第二次直接返回）。

        总耗时默认取"从第一条事件到现在"（engine 会传更精确的任务耗时覆盖它）。
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self.emit("run.end", model_calls=self.stats["model_calls"],
                  tool_calls=self.stats["tool_calls"], errors=self.stats["errors"],
                  retries=self.stats["retries"],
                  input_tokens=self.stats["input_tokens"],
                  output_tokens=self.stats["output_tokens"],
                  total_tokens=self.stats["total_tokens"],
                  duration_ms=self.stats.get("duration_ms")
                  or round((time.time() - self._t0) * 1000.0, 1),
                  trace_file=str(self._path) if self._path else None,
                  **fields)
        with self._lock:
            if self._fh is not None:
                try:
                    self._fh.close()
                finally:
                    self._fh = None


# ---------------- 会话（contextvar） ----------------

_current: contextvars.ContextVar[Tracer | None] = contextvars.ContextVar("agent_trace", default=None)


def current() -> Tracer | None:
    """当前上下文（线程）里正在生效的轨迹记录器；没有则 None。"""
    return _current.get()


@contextlib.contextmanager
def session(
    *,
    task: str = "",
    thread_id: str = "",
    project_id: str = "",
    directory: Path | str | None = None,
    console: bool | None = None,
    enabled: bool | None = None,
    sinks: Iterable[Callable[[dict, str], None]] = (),
):
    """开一段轨迹会话：块内所有打点都记到这个任务上，退出时自动 run.end + 关文件。

    contextvar 是按线程隔离的，所以 Web 端并发任务各记各的，不会串（同 cost.tracker.session）。
    """
    tr = Tracer(task=task, thread_id=thread_id, project_id=project_id,
                directory=directory, console=console, enabled=enabled, sinks=sinks)
    token = _current.set(tr)
    try:
        yield tr
    finally:
        _current.reset(token)
        tr.close()


def emit(kind: str, **fields) -> dict | None:
    """记一条事件（没有会话时是 no-op，调用方无需判断）。"""
    tr = _current.get()
    return tr.emit(kind, **fields) if tr is not None else None


def span(kind: str, name: str, **fields) -> Span:
    return Span(_current.get(), kind, name, fields)


def model_call(label: str, model: str | None = None, messages: Sequence[Any] | None = None,
               **extra) -> Span:
    """模型调用 span：开始记 prompt 摘要，结束时补 token / 耗时。"""
    fields = dict(extra)
    if model:
        fields["model"] = model
    fields.update(message_digest(messages))
    return Span(_current.get(), "model", label, fields)


def tool_call(name: str, args: Any = None, **extra) -> Span:
    """工具调用 span：参数在这个时间点快照下来（避免事后被改）。"""
    fields = dict(extra)
    fields["args"] = clip_args(args)
    return Span(_current.get(), "tool", name, fields)


def node(name: str):
    """装饰图节点：自动记录 `node.start` / `node.end`（含耗时与产出摘要）。

    `functools.wraps` 保留签名，LangGraph 解析节点参数（state / config）不受影响。
    """
    def deco(fn):
        @functools.wraps(fn)
        def traced(state, *a, **kw):
            tr = _current.get()
            if tr is None:                      # 没开轨迹：零开销直通
                return fn(state, *a, **kw)
            t0 = time.perf_counter()
            tr.emit("node.start", name=name, task_stage=name)
            try:
                out = fn(state, *a, **kw)
            except BaseException as e:  # noqa: BLE001 —— 记完再原样抛
                tr.emit("node.end", name=name, status="error",
                        error=f"{type(e).__name__}: {e}",
                        duration_ms=round((time.perf_counter() - t0) * 1000.0, 1))
                raise
            tr.emit("node.end", name=name, duration_ms=round((time.perf_counter() - t0) * 1000.0, 1),
                    summary=update_summary(out))
            return out

        return traced

    return deco


#: 节点结束时记进日志的字段（体积可控：只记关键结论，不记整个 state）
_SUMMARY_KEYS = ("plan", "context", "result", "feedback", "done", "iterations",
                 "tool_rounds", "aborted", "changes", "history_summary")


def update_summary(out: Any) -> str:
    """把节点的返回值压成一句人话（executor 的整段 state 不该进日志）。"""
    if not isinstance(out, dict):
        return _one_line(out, 200)
    bits: list[str] = []
    for k in _SUMMARY_KEYS:
        if k not in out:
            continue
        v = out[k]
        if isinstance(v, str):
            v = _one_line(v, 160)
        bits.append(f"{k}={v}")
    calls = out.get("pending_calls")
    if isinstance(calls, list) and calls:
        bits.append("待调用=" + ", ".join(str(c.get("name")) for c in calls))
    return " · ".join(bits) if bits else "（无状态变更）"


# ---------------- 轨迹文件的事后消费（Web 端「查看 JSONL」/ CLI 复盘） ----------------


def list_files(limit: int = 50) -> list[dict]:
    """轨迹文件列表（新的在前），供界面/CLI 事后翻看。"""
    d = trace_dir()
    if not d.is_dir():
        return []
    items = []
    for p in sorted(d.glob("*.jsonl"), key=lambda x: x.stat().st_mtime, reverse=True)[:limit]:
        try:
            st = p.stat()
        except OSError:
            continue
        head = _first_event(p)
        items.append({
            "name": p.name,
            "path": str(p),
            "size": st.st_size,
            "mtime": datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "task": head.get("task", ""),
            "project": head.get("project", ""),
            "run": head.get("run", ""),
        })
    return items


def _first_event(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return {}


def resolve(name: str) -> Path:
    """把文件名解析成轨迹目录下的真实路径（拒绝目录穿越）。"""
    d = trace_dir().resolve()
    p = (d / name).resolve()
    if p.parent != d or p.suffix != ".jsonl":
        raise ValueError(f"非法的轨迹文件名: {name}")
    return p


def read_events(path: Path | str, limit: int = 2000) -> list[dict]:
    """读取轨迹文件（默认只返回最后 limit 条，避免超大文件把浏览器拖死）。"""
    out: list[dict] = []
    p = Path(path)
    if not p.is_file():
        return out
    with p.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(out) > limit:
                out.pop(0)
    return out


def render_events(path: Path | str, limit: int = 2000) -> str:
    """把轨迹文件渲染成可读文本（CLI 复盘 / 界面预览用）。"""
    return "\n".join(render(rec) for rec in read_events(path, limit=limit))


def latest() -> Path | None:
    files = list_files(limit=1)
    return Path(files[0]["path"]) if files else None
