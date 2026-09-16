"""集中配置：从 .env 读取 DeepSeek API 配置与模型分工。"""
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from langchain_openai import ChatOpenAI

from cost import tracker

# 项目根目录 = 本文件所在目录
ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

# HuggingFace 模型权重缓存放到项目内（而非默认的 ~/.cache/huggingface），保持项目自包含。
# 必须在导入 sentence_transformers / transformers 之前设置；rag.py 中这些是懒加载，故此处生效。
HF_CACHE = ROOT / ".cache" / "huggingface"
os.environ.setdefault("HF_HOME", str(HF_CACHE))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # 国内镜像加速，可被外部环境变量覆盖
# 模型已缓存 → 离线加载（更快、消除未认证警告）；
# 全新机器没缓存 → 不设 OFFLINE，首次使用 embedding/重排时自动联网下载（约 200MB）。
if (HF_CACHE / "hub").exists():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    deepseek_api_key: str
    deepseek_base_url: str = "https://api.deepseek.com"
    planner_model: str = "deepseek-v4-pro"    # 规划 / 反思用强推理模型
    executor_model: str = "deepseek-v4-flash"  # 执行 / 工具调用用快模型

    # 默认工作区（**回退值**，不是唯一工作区）。
    # 支持任意项目后，真正"当前操作哪个代码库"由 workspace.py 在运行期决定：
    #   contextvar（任务绑定）> 注册表 current（界面里选中的项目）> 这里的回退值
    # 所以这里只在"一个项目都还没添加、也没有注册表"时兜底，默认 = agent 自身目录。
    workspace_root: Path = Field(
        default_factory=lambda: Path(os.environ.get("WORKSPACE_ROOT", str(ROOT)))
    )

    # ---- 运行模式：两个正交旋钮（业界口径，见 README 第九章）----
    # 旋钮 A · 强制隔离在哪：【local】内核围栏（macOS Seatbelt / Linux bubblewrap，
    #   命令真的被关在工作区里）｜"host" 无隔离（逐条审批兜底）｜"docker" 容器隔离。
    # 旋钮 B · sandbox_mode：read-only | workspace-write | danger-full-access（文件效果+网络）
    # 旋钮 C · approval_policy：always（逐条问）| on-escalation（只在越界时才问，默认）
    #                           | never（越界直接拒，适合无人值守）
    # B/C 留空即按 A 推导（preset）：
    #   local  → workspace-write + on-escalation   ← 业界主流：普通命令不问，越界才问
    #   docker → danger-full-access + on-escalation（隔离由容器提供，沙箱镜像内默认为断网）
    #   host   → danger-full-access + always       （没有围栏，只能靠人逐条把关）
    # rag_enabled=False 时不做 RAG 向量检索（省掉 torch/chroma/本地模型 ≈3~5GB），
    # agent 改用 list_files / search_code / read_file 工具自行定位代码。
    rag_enabled: bool = True
    exec_mode: str = "local"
    sandbox_mode: str = ""
    approval_policy: str = ""

    @property
    def resolved_sandbox_mode(self) -> str:
        """生效的文件效果策略（显式配置优先，否则按 exec_mode 推导）。"""
        mode = (self.sandbox_mode or "").strip().lower()
        if mode in ("read-only", "workspace-write", "danger-full-access"):
            return mode
        return "workspace-write" if self.exec_mode == "local" else "danger-full-access"

    @property
    def resolved_approval_policy(self) -> str:
        """生效的审批策略（显式配置优先，否则按 exec_mode 推导）。"""
        policy = (self.approval_policy or "").strip().lower()
        if policy in ("always", "on-escalation", "never"):
            return policy
        return "always" if self.exec_mode == "host" else "on-escalation"

    @property
    def preset_name(self) -> str:
        """一行展示当前组合，写进 prompt / 审批弹窗 / 日志。"""
        return (f"{self.exec_mode}"
                f"（sandbox={self.resolved_sandbox_mode}, "
                f"approval={self.resolved_approval_policy}）")

    # RAG 索引是否自动跟随文件变动重建（按 REINDEX_TTL 间隔检测，见 rag.py）。
    # 关闭后索引只在首次/切换项目时建立，适合超大仓库。
    rag_auto_reindex: bool = True

    # ---- 运行轨迹（observability，见 trace.py）----
    # 三件事各管各的：控制台流式打印 / JSONL 落盘 / 预览长度。
    trace_enabled: bool = True        # 落盘 .agent_cache/traces/*.jsonl
    trace_console: bool = True        # 边跑边往终端打印（Web 服务端进程里也打）
    trace_full_content: bool = False  # 连完整 prompt / 完整工具输出一起记（默认只记摘要与预览）
    trace_dir: Path = Field(default_factory=lambda: ROOT / ".agent_cache" / "traces")


settings = Settings()


def get_llm(model: str | None = None, temperature: float = 0.0, thinking: bool = False) -> ChatOpenAI:
    """构造 DeepSeek V4 的 ChatOpenAI。

    实测要点（langchain-openai 1.6.x + DeepSeek V4）：
    - 字段已改名为 openai_api_base / openai_api_key / model_name；
    - V4 默认开启 thinking 模式，该模式「不支持强制的 tool_choice」，
      所以 with_structured_output(..., method="function_calling") 必须先 thinking=False；
    - json_schema 响应格式 DeepSeek 不支持，结构化输出统一走 function_calling。
    """
    extra = {} if thinking else {"thinking": {"type": "disabled"}}
    return ChatOpenAI(
        model_name=model or settings.executor_model,
        openai_api_key=settings.deepseek_api_key,
        openai_api_base=settings.deepseek_base_url,
        temperature=temperature,
        extra_body=extra,
        callbacks=[tracker],
    )
