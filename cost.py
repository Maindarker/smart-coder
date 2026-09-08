"""LLM 调用成本统计：按模型分组记录调用次数与 token 用量。

通过 LangChain callback 在每次 LLM 结束时自动累计，无需侵入各节点代码。
金额估算：可在 PRICING 里按模型填每百万 token 价格（元），留空则不显示金额。
"""
from __future__ import annotations

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

# 可选：每百万 token 价格（元）。填入后 summary 会附带金额估算；留空则只显示 token 数。
# 价格请以 DeepSeek 官方为准，这里只做示例。
PRICING: dict[str, dict[str, float]] = {
    # "deepseek-v4-pro": {"input": 0.0, "output": 0.0},
    # "deepseek-v4-flash": {"input": 0.0, "output": 0.0},
}


class CostTracker(BaseCallbackHandler):
    """累计所有 LLM 调用的 token 用量，按模型分组。"""

    def __init__(self) -> None:
        self.models: dict[str, dict] = {}

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        model = None
        for gen in response.generations:
            for g in gen:
                msg = getattr(g, "message", None)
                if msg is None:
                    continue
                meta = getattr(msg, "response_metadata", None) or {}
                model = meta.get("model_name") or model
                usage = getattr(msg, "usage_metadata", None) or {}
                self._add(model, usage)
        if model is None:
            self._add("unknown", {})

    def _add(self, model: str | None, usage: dict) -> None:
        key = model or "unknown"
        m = self.models.setdefault(key, {"calls": 0, "input": 0, "output": 0, "total": 0})
        m["calls"] += 1
        m["input"] += usage.get("input_tokens", 0)
        m["output"] += usage.get("output_tokens", 0)
        m["total"] += usage.get("total_tokens", 0)

    def reset(self) -> None:
        self.models.clear()

    def summary(self) -> str:
        if not self.models:
            return "（无 LLM 调用）"
        lines = []
        total_calls = total_tokens = 0
        for model, m in sorted(self.models.items()):
            line = (f"  {model}: {m['calls']} 次调用 | "
                    f"输入 {m['input']} / 输出 {m['output']} / 共 {m['total']} tokens")
            price = PRICING.get(model)
            if price:
                cost = (m["input"] / 1e6 * price.get("input", 0)
                        + m["output"] / 1e6 * price.get("output", 0))
                line += f" ≈ ¥{cost:.4f}"
            lines.append(line)
            total_calls += m["calls"]
            total_tokens += m["total"]
        lines.append(f"  合计: {total_calls} 次调用, {total_tokens} tokens")
        return "\n".join(lines)


# 全局单例：config.get_llm() 会把它挂到每个 LLM 上，整次运行自动累计
tracker = CostTracker()
