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

    # ---- 运行模式开关 ----
    # 默认 host（任意项目）：命令在本机直接执行，不假设语言，每条命令人工审批。
    #   - exec_mode="host"：无容器隔离，安全靠 agent.py 的逐条审批兜底；
    #     任意语言的项目（node/go/rust/…）都能直接跑，无需为每种语言建镜像。
    #   - exec_mode="docker"：命令在非 root 沙箱容器内执行、默认断网，
    #     但沙箱镜像目前只带 Python 运行时（见 Dockerfile.sandbox），
    #     跑非 Python 项目需要在 .env 里改回 host 或自行扩展镜像。
    # rag_enabled=False 时不做 RAG 向量检索（省掉 torch/chroma/本地模型 ≈3~5GB），
    # agent 改用 list_files / search_code / read_file 工具自行定位代码。
    rag_enabled: bool = True
    exec_mode: str = "host"   # "host"（默认，任意项目） | "docker"（Python 项目强隔离）

    # RAG 索引是否自动跟随文件变动重建（按 REINDEX_TTL 间隔检测，见 rag.py）。
    # 关闭后索引只在首次/切换项目时建立，适合超大仓库。
    rag_auto_reindex: bool = True


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
