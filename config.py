"""集中配置：从 .env 读取 DeepSeek API 配置与模型分工。"""
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict
from langchain_openai import ChatOpenAI

# 项目根目录 = 本文件所在目录
ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    deepseek_api_key: str
    deepseek_base_url: str = "https://api.deepseek.com"
    planner_model: str = "deepseek-v4-pro"    # 规划 / 反思用强推理模型
    executor_model: str = "deepseek-v4-flash"  # 执行 / 工具调用用快模型

    # 安全：agent 只允许在这个目录内读写文件
    workspace_root: Path = ROOT


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
    )
