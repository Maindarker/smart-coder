# 🤖 「 代码辅助 Agent实战」

> **从环境搭建到端到端落地的完整实战记录**——包含每一步的动机、每个依赖包的用途、所有踩过的坑与解决思路，以及可以跑通的完整代码。

> 适用读者：想用 LangGraph 搭建有状态 Agent、想接入 DeepSeek API、想做代码库 RAG / Docker 沙箱安全的开发者。文中所有结论均来自真实运行验证。

---

## 📑 目录

1. [为什么做这件事](#一为什么做这件事)
2. [总体架构设计](#二总体架构设计)
3. [环境准备与依赖选型](#三环境准备与依赖选型)
4. [摸清 DeepSeek V4 的"脾气"](#四摸清-deepseek-v4-的脾气)
5. [最小可运行骨架：4 个文件跑通 Agent Loop](#五最小可运行骨架)
6. [安全层：Docker 沙箱的构建与连环坑](#六安全层docker-沙箱的构建与连环坑)
7. [RAG 代码检索引擎：向量 + BM25 + 重排](#七rag-代码检索引擎)
8. [记忆持久化：SqliteSaver 与 thread_id](#八记忆持久化)
9. [危险操作审批：interrupt 人机协同](#九危险操作审批)
10. [测试与验证实录](#十测试与验证实录)
11. [经验总结与踩坑清单](#十一经验总结与踩坑清单)
12. [支持任意项目：多项目工作区](#十二支持任意项目多项目工作区)
13. [Web 界面：项目管理与人工审批](#十三web-界面项目管理与人工审批)

---

## 一、为什么做这件事

大模型的代码能力已经很惊艳，但要让它真正"干活"——修改代码、跑测试、搜索大仓库、操作 Git——单靠"对话式补全"是不够的。我们需要的是一个**能自主规划、调用工具、反思纠错的有状态 Agent**。

本项目的目标是从零构建一个代码辅助 Agent（起名 `smart-coder`），要求：

- ✅ 能理解用户任务并拆解计划
- ✅ 能**安全地**执行命令（命令全部跑在 Docker 沙箱里，非 root、断网、限资源）
- ✅ 能对**代码库做检索**（向量 + BM25 混合 + 重排，而非只靠 LLM 上下文硬猜）
- ✅ 记得住任务状态（中断后能续跑，多会话隔离）
- ✅ 危险操作（`rm -rf`、`git push` 等）必须**人工审批**

最终交付的是一个在本地可运行的 CLI Agent：`python main.py "你的任务"`。

---

## 二、总体架构设计

### 2.1 为什么选 LangGraph 而不是 LangChain AgentExecutor

LangChain 的 `AgentExecutor` 是一个"黑盒"循环：你给它 Agent 和工具，它自己决定怎么调。简单，但**不透明、难控制**——你很难插入"反思"节点、很难在中间停住等人审批、很难精确控制重试逻辑。

LangGraph 把 Agent 建模成一张**有向状态图（StateGraph）**：节点是"动作"（plan / execute / reflect...），边是"流转条件"。好处是：

- 🔄 **显式循环**：`reflect` 不满意可以走回 `execute`，次数可限
- 🌿 **条件分支**：根据状态决定走哪条边
- 💾 **状态管理**：每个节点读写共享的 State，天然可持久化（checkpoint）
- ✋ **human-in-the-loop**：`interrupt()` 可以在任意节点"冻结"执行等人审批

### 2.2 六个模块的分工

| 模块           | 选型                                 | 职责                                                  |
| -------------- | ------------------------------------ | ----------------------------------------------------- |
| **LLM**        | DeepSeek V4（Pro / Flash 双模型）    | Pro 负责规划/反思（强推理），Flash 负责执行（快、省） |
| **Agent Loop** | LangGraph StateGraph                 | `plan → retrieve → execute → reflect → finish`        |
| **工具集**     | LangChain `@tool`                    | 文件读写（本机受限）+ 命令/测试（Docker 沙箱）        |
| **RAG**        | ChromaDB + rank-bm25 + cross-encoder | 代码库索引、混合检索、重排                            |
| **记忆**       | SqliteSaver + 偏好文件                | 会话历史回放 + 长期偏好记忆（跨会话）                |
| **安全层**     | Docker（非 root 容器）               | 命令隔离 + 危险操作 `interrupt()` 审批                |

### 2.3 Agent Loop 设计

```
┌────────┐     ┌──────────┐     ┌─────────┐     ┌─────────┐
│  plan  │ ──▶ │ retrieve │ ──▶ │ execute │ ──▶ │ reflect │
└────────┘     └──────────┘     └─────────┘     └─────────┘
                                                   │
                          ┌────────────────────────┘
                          ▼ 完成 或 超过最大轮数
                     ┌─────────┐
                     │ finish  │
                     └─────────┘
```

- **plan**：Pro 模型（开 thinking）产出步骤计划
- **retrieve**：RAG 从代码库检索相关片段注入上下文（避免让 LLM 瞎猜代码位置）
- **execute**：Flash 模型 + 工具循环（工具调用 → 观察结果 → 再调用），最多 4 轮
- **reflect**：Pro 模型结构化输出 `{done, feedback}`，判断任务是否完成
- **finish**：Pro 模型汇总成给用户看的结论

---

## 三、环境准备与依赖选型

### 3.1 运行环境：两套方案，先选一套

代码把「**命令在哪执行**」和「**代码怎么被检索**」拆成了两个**互相独立**的开关
（`config.py` 的 `exec_mode` / `rag_enabled`，在 `.env` 里配），所以运行环境有两套推荐搭配：

|                        | 方案 A · 轻量模式（**推荐默认**）                  | 方案 B · 完整模式                              |
| ---------------------- | ------------------------------------------------- | ---------------------------------------------- |
| `.env` 两个开关        | `EXEC_MODE=host` + `RAG_ENABLED=false`            | `EXEC_MODE=docker` + `RAG_ENABLED=true`        |
| 依赖文件               | `requirements.txt`                                | `requirements-full.txt`                        |
| 额外前置               | **无**，Python 3.11 就够                          | Docker Desktop（沙箱镜像首用自动构建）         |
| 命令在哪跑             | 你的本机 shell（继承当前环境，项目自带 venv 优先；只有超时兜底，无 CPU/内存限制）| 非 root 容器：默认断网、1 CPU、512MB、60s 超时 |
| 安全靠什么兜底         | **每条 `run_shell`/`run_test` 都弹人工审批** + 危险命令拦截 | 容器隔离 + 危险命令拦截（普通命令不逐条审批） |
| 能跑什么项目           | **任意语言**（node / go / rust / java …）         | 目前只有 **Python**（镜像只带 Python 运行时）  |
| 代码怎么定位           | agent 用 `describe_project`/`list_files`/`search_code`/`read_file` 自己找 | 先 RAG：向量 + BM25 混合检索 → 重排 → 相关代码块注入上下文 |
| 体积（macOS 实测）     | 约几百 MB（无 torch / chroma）                    | site-packages 1.4GB，另需首次下载约 239MB 模型权重 |

> 两套方案的共同点：系统 macOS / Linux / Windows 都行（Docker 能跑就支持方案 B），
> Python 3.11，密钥只填 `.env` 里的 `DEEPSEEK_API_KEY`。
> 方案 A 就是仓库里 `.env.example` 的那套配置：拷一份、填上 key 即可。
>
> ⚠️ 注意 `RAG_ENABLED` 要**显式写 false**：`config.py` 里这个字段的默认值是 `true`
> （`EXEC_MODE` 的默认值已经是 `host`，两者不对称），不写就会去连向量库、而轻量模式没装
> `chromadb`，每个任务都会往上下文里塞一句"检索失败"。

**为什么把轻量模式当默认**：这个 agent 面向"任意语言、任意项目"，
而沙箱镜像（`Dockerfile.sandbox`）只带 Python 运行时 —— 拿 docker 当默认，
同事第一次帮一个前端项目跑 `npm test` 就会撞墙。轻量模式把隔离手段从"容器兜底"
换成"**逐条人工审批**兜底"：`agent.py` 里 host 模式下任何 `run_shell` / `run_test`
都会先 `interrupt` 问一句（拒绝普通命令不终止任务、agent 会换个方式重试；
拒绝危险命令则终止任务，见第九节）。安全语义没丢，只是从"机器拦"变成"人看一眼"。
RAG 同理：开它要装 torch/chroma 并首次下载模型，而关掉后 agent 靠
`search_code` 等工具自己找代码，对多数任务已经够用。

**两个开关其实互相独立，4 种组合都合法**，上表只是两套推荐搭配：

- `host` + `RAG_ENABLED=true`：想在本机跑、但想要向量检索 → 装 `requirements-full.txt` 即可；
- `docker` + `RAG_ENABLED=false`：只想要强隔离、不想下模型（要求项目是 Python）；
- 混装时**不需要**改代码：`rag.py` 与 `sandbox.py` 里的重依赖都是**函数内懒加载**的
  （`chromadb` / `sentence_transformers` / `rank_bm25` 在 `rag.py` 函数里，
  `docker` 在 `sandbox.py` 的 `try/except ImportError` 里）。

**回退与排错**：请求了 docker 却没装 docker Python 库 → 启动时打印一次提示并
**自动回退 host**（`sandbox.effective_mode()`）；装了库但 Docker Desktop 没启动 →
命令会报错，用 `GET /api/health` 的 `exec_mode` 字段（Web 页面顶部也会显示）看**实际生效**的模式。

### 3.2 依赖包清单与用途

先做一次 `pip list` 盘点，分清"框架层"和"工具层"。下面这张表是**每个包为什么在这里**：

| 包                            | 版本        | 为什么需要它                                                           |
| ----------------------------- | ----------- | ---------------------------------------------------------------------- |
| `langgraph`                   | 1.2.11      | Agent 状态图核心：StateGraph / interrupt / checkpoint                  |
| `langchain-core`              | 1.6.2       | 消息模型（SystemMessage 等）、Runnable、`@tool` 装饰器                 |
| `langchain-openai`            | 1.6.0       | 用 OpenAI 兼容协议接 DeepSeek（`ChatOpenAI`）                          |
| `langgraph-checkpoint`        | 4.2.0       | 记忆抽象（真正落盘在 sqlite 子包里，见下）                             |
| `chromadb`                    | 1.5.9       | 向量库，持久化代码 chunk 的 embedding                                  |
| `sentence-transformers`       | 6.0.1       | 本地 embedding 模型 + cross-encoder 重排模型                           |
| `torch` / `transformers`      | 2.14 / 5.16 | sentence-transformers 的底层引擎                                       |
| `tiktoken`                    | 0.14.0      | token 计数                                                             |
| `pydantic-settings`           | 2.15.0      | 从 `.env` 读配置（API Key、模型分工）                                  |
| `python-dotenv`               | 1.2.3       | 加载 `.env`                                                            |
| `docker`                      | 7.2.0       | Python 调用 Docker 创建沙箱容器（见"踩坑"章节，原装的 docker-py 太老） |
| `rank-bm25`                   | 0.2.2       | **后补**：BM25 稀疏检索，与向量检索做混合                              |
| `langchain-chroma`            | 1.1.0       | **后补**：LangChain ↔ Chroma 集成封装                                  |
| `langgraph-checkpoint-sqlite` | 3.1.1       | **后补**：SqliteSaver 被拆成独立包，需单独装                           |
| `pytest`                      | 9.1.1       | 沙箱内跑 Python 测试用                                                 |

> 💡 经验：`langgraph 1.x` 已经把 checkpoint 后端拆成独立包（`langgraph-checkpoint-sqlite` / `-postgres`），`SqliteSaver` 不在核心包里，很多人会在这里踩"ModuleNotFoundError"。

> 📌 **上表是"每个包为什么在"，但并不是全都要装**：真正钉死版本的清单是
> `requirements.txt`（方案 A）和 `requirements-full.txt`（方案 B = A + 重依赖），
> 两者不一致时以这两个文件为准。
> **只有方案 B 才需要的包**：`chromadb`、`sentence-transformers`、`torch`/`transformers`、
> `rank-bm25`、`langchain-chroma`、`docker` —— 它们在代码里全是**懒加载**：
> 关掉 RAG、用 host 模式时根本不会被 import，所以不装也能跑。
> 反过来，`uvicorn`（起 Web 服务）、`langgraph-checkpoint-sqlite`（记忆落盘）、
> `pytest` 两个方案都要，别漏。
> （`tiktoken` 是早期版本的遗留项，现在的 `cost.py` 直接用模型回报的 usage，已不再需要。）

### 3.3 安装命令（可直接复制）

两条路都是「**先建 venv → 再装对应那份 requirements**」，区别只在装哪个文件、`.env` 里两个开关怎么配。
两份 requirements 都把版本钉死了，所以**不要**再手抄包名（手抄容易漏、也容易装错 docker SDK 版本）。

**① 公共步骤（两套方案都要）**

```bash
# 创建并激活虚拟环境（目录名随便取，这里用 agent_env）
python3.11 -m venv agent_env
source agent_env/bin/activate        # macOS / Linux
agent_env\Scripts\activate           # Windows（cmd / PowerShell）
# 激活成功后提示符前会出现 (agent_env) 字样

# 升级 pip（旧版 resolver 解析新依赖容易失败）
python -m pip install --upgrade pip
```

**② 方案 A · 轻量模式（默认，约几百 MB，无需 Docker）**

```bash
pip install -r requirements.txt
```

`.env` 最少填 key，两个开关建议**显式写上**（`RAG_ENABLED` 的代码默认值是 `true`，见 3.1 节末尾的提醒）：

```ini
DEEPSEEK_API_KEY=sk-你的key
EXEC_MODE=host          # 命令在本机跑，逐条人工审批
RAG_ENABLED=false       # 不做向量检索，agent 用工具自己找代码
```

到这里就能用了，**不需要** Docker、**不需要**下载任何模型：

```bash
python main.py "这个项目的测试怎么跑？跑一下" --project=/绝对路径/你的项目   # CLI
bash run-web.sh                                                          # 或浏览器版
```

**③ 方案 B · 完整模式（Docker 沙箱 + 本地 RAG）**

```bash
pip install -r requirements-full.txt   # = requirements.txt + chromadb/sentence-transformers/torch/docker…
docker info >/dev/null && echo "Docker 就绪"   # 需要 Docker Desktop 已经在跑
```

`.env` 换一套开关：

```ini
DEEPSEEK_API_KEY=sk-你的key
EXEC_MODE=docker        # 命令在非 root 容器内跑：默认断网、1 CPU、512MB、60s 超时
RAG_ENABLED=true        # 向量 + BM25 混合检索后重排，相关代码块注入上下文
```

> 沙箱镜像**不用手动 build**：docker 模式下第一次执行命令时会用 `Dockerfile.sandbox`
> 自动构建（首次约 1~3 分钟），见 6.2 / 6.6 节。

几点说明：

- **体积**：`sentence-transformers` 会自动带上 `torch` / `transformers`，属正常现象。
  本机（macOS Apple Silicon）实测 site-packages 共 **1.4GB**，其中重的那几块是
  torch 587MB + transformers 112MB + scipy 99MB + sklearn 48MB + numpy 36MB + chromadb 6.5MB
  ≈ **890MB**，方案 A 就是把这 890MB 省掉；Linux 上 torch 还会带 CUDA 轮子，3~5GB 很常见。
- **模型权重不是 pip 包**：开 RAG 后首次调用 embedding / cross-encoder 时才从 Hugging Face
  下载（两者合计约 **239MB**，落在项目内 `.cache/huggingface`）。`config.py` 已经默认走
  `hf-mirror.com` 镜像并设好 `HF_HOME`，国内网络不用额外配置，细节见 7.3 节。
- **两套方案随时互切**：改 `.env` 两个开关即可，代码不用动；依赖是"装多了不影响"——
  方案 B 装了重依赖后也能跑 `EXEC_MODE=host`。想让环境也瘦回去：
  `pip uninstall -y torch transformers sentence-transformers chromadb langchain-chroma rank-bm25 docker`。
- **手工装（不用 requirements 文件）时的坑**：某些环境会把 `docker` 装成 2016 年的
  `docker-py 1.10.6`，必须升到 7.x（原因与报错见 6.5 节坑 5）；
  `requirements-full.txt` 里已经钉了 `docker==7.2.0`，所以走文件安装不用管。
- **Windows / Linux 装不上 torch 时**：按 `requirements-full.txt` 里的注释去掉版本号，让 pip 按平台挑。
- 别忘了在项目根目录建 `.env`（完整字段见 5.1 节），并把 `.env` 加进 `.gitignore`。

### 3.4 环境自检

装完先做三件事确认"地基"是稳的：

```python
# 1) 关键模块能否导入（两套方案都要）
import langgraph, langchain_openai, fastapi

# 方案 B（完整模式）再多验这几个；方案 A 没装它们，这里报 ModuleNotFoundError 是正常的
import chromadb, sentence_transformers, docker

# 2) LangGraph 的最小图能否编译 + 运行
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class S(TypedDict):
    x: int

def add(s: S) -> S:
    return {"x": s["x"] + 1}

g = StateGraph(S)
g.add_node("add", add)
g.add_edge(START, "add")
g.add_edge("add", END)
app = g.compile()
assert app.invoke({"x": 0})["x"] == 1

# 3) 确认实际生效的执行模式与 RAG 开关
import sandbox
from config import settings
print(sandbox.effective_mode(), settings.rag_enabled)
# 期望：方案 A → host False；方案 B → docker True
# （请求了 docker 但没装 docker 库时会自动回退成 host，见 3.1 节）
```

如果这几步都过，说明 LangGraph 主链路是好的，后面所有问题都集中在**接入层**（API 兼容、网络、容器）。

---

## 四、摸清 DeepSeek V4 的"脾气"

这是整个项目**信息差最大**的部分。DeepSeek V4 的 API 与常见教程里写的有三处不同，如果不实测，代码跑起来全是 400 报错。

### 4.1 模型名已经换代

网上大量教程还在教 `deepseek-chat` / `deepseek-reasoner`，但这两个模型名已经**停用**（2026-07-24 起），官方迁移到 V4 命名：

| 用途                        | 模型名                         |
| --------------------------- | ------------------------------ |
| 快 / 便宜（执行、工具调用） | `deepseek-v4-flash`            |
| 强推理（规划、反思）        | `deepseek-v4-pro`              |
| 视觉（实验）                | `deepseek-v4-flash-vision-exp` |

验证方式：直接调一次 API 看返回，模型名对不对立刻见分晓。

### 4.2 Thinking 模式与结构化输出冲突 ⚠️

V4 模型**默认开启 thinking 模式**（先产出一段推理再回答）。实测发现：

- ✅ thinking 模式下，普通工具调用（`bind_tools`）**可用**
- ❌ thinking 模式下，**强制 `tool_choice` 会被拒**：`Thinking mode does not support this tool_choice`
- ❌ `response_format=json_schema` 直接被拒：`This response_format type is unavailable now`

> 🤔 **为什么 `bind_tools` 可以、`with_structured_output` 却必挂？**
>
> 关键在于 `tool_choice` 是否被**强制**：
>
> - `bind_tools` 只是把工具**注册**给模型，`tool_choice` 不指定（auto），模型"可调可不调、可自由选"——所以 thinking 模式放行 ✅
> - `with_structured_output(method="function_calling")` 的目的是拿**稳定的结构化结果**：LangChain 把你的 Pydantic schema 包装成一个"伪工具"，并**强制** `tool_choice` 必须调用它——只有模型把内容填进 tool_call 的 `args`，LangChain 才能解析出结构体。强制调用恰好撞上 thinking 模式的限制 ❌
>
> 类比：`bind_tools` = 把工具箱放模型面前"这些工具你可以用"；`with_structured_output` = 逼模型按固定格式交卷"必须把这个 JSON 填好"。

因此 `with_structured_output(method="function_calling")` 在默认（thinking 开启）配置下必挂。解法是在请求体里**显式关掉 thinking**：

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model_name="deepseek-v4-flash",
    openai_api_key=...,
    openai_api_base="https://api.deepseek.com",
    temperature=0,
    extra_body={"thinking": {"type": "disabled"}},  # 关 thinking
)

# 现在结构化输出才可用，且只能用 function_calling，不能用 json_schema
from pydantic import BaseModel, Field

class Decision(BaseModel):
    done: bool = Field(description="任务是否已完成")
    feedback: str = Field(description="反馈")

decider = llm.with_structured_output(Decision, method="function_calling")
```

### 4.3 langchain-openai 1.6.x 的字段改名

新版 `ChatOpenAI` 把参数改了名：

| 旧写法（网上教程） | 新字段（1.6.x）    |
| ------------------ | ------------------ |
| `model=`           | `model_name=`      |
| `api_key=`         | `openai_api_key=`  |
| `base_url=`        | `openai_api_base=` |

实测：旧写法**仍能构造成功**（有别名映射），但读配置时必须用新字段名，否则 `llm.base_url` 会抛 `AttributeError`——这种"能建但读不了"的坑最难排查。

### 4.4 小结：一条能跑通 DeepSeek 的工厂函数

把这些坑收敛成一个工厂函数（后续所有节点都复用它）：

```python
def get_llm(model=None, temperature=0.0, thinking=False):
    """thinking=True：规划/反思；thinking=False：执行/结构化输出。"""
    extra = {} if thinking else {"thinking": {"type": "disabled"}}
    return ChatOpenAI(
        model_name=model or settings.executor_model,
        openai_api_key=settings.deepseek_api_key,
        openai_api_base=settings.deepseek_base_url,
        temperature=temperature,
        extra_body=extra,
    )
```

> 小坑：在 stdin 里用 `load_dotenv()` 无参调用会触发 `find_dotenv()` 断言失败，务必传显式路径 `load_dotenv("/abs/path/.env")`。

---

## 五、最小可运行骨架

> 先跑通"薄"骨架，再往上加东西——这是本项目贯彻的最小原则。骨架只有 4 个文件，但 LangGraph 的完整闭环（含结构化输出、条件路由、防死循环）都在里面。

### 5.1 config.py —— 集中配置与模型工厂

```python
# config.py（节选）
import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from langchain_openai import ChatOpenAI

ROOT = Path(__file__).resolve().parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(ROOT / ".env"), extra="ignore")
    deepseek_api_key: str
    deepseek_base_url: str = "https://api.deepseek.com"
    planner_model: str = "deepseek-v4-pro"      # 规划/反思用强推理模型
    executor_model: str = "deepseek-v4-flash"   # 执行用快模型
    workspace_root: Path = ROOT                 # 安全边界：只能操作这个目录

settings = Settings()
```

> 📌 **后续演进（见第十二节）**：`workspace_root` 已降级为**回退值**。要支持"界面上随时添加/切换
> 任意项目"，工作区就不能是 import 时快照的常量 —— 现在由 `workspace.py` 在运行期解析：
> `contextvar（当前任务绑定）> 注册表 current（界面选中）> workspace_root（回退）`。

对应根目录下创建 `.env`：

```ini
DEEPSEEK_API_KEY=sk-xxx
DEEPSEEK_BASE_URL=https://api.deepseek.com
PLANNER_MODEL=deepseek-v4-pro
EXECUTOR_MODEL=deepseek-v4-flash
```

> 🔒 安全第一课：`.env` 必须进 `.gitignore`，绝不允许提交 API Key。

### 5.2 tools.py —— 最小工具集（先不碰沙箱）

先用两个"绝对安全"的文件工具验证链路，命令执行等安全层就绪再加：

```python
from pathlib import Path
from langchain_core.tools import tool
from config import settings

def _safe_path(p: str) -> Path:
    """路径越界防护：只允许 workspace 内。"""
    root = settings.workspace_root.resolve()
    path = (root / p).resolve()
    if not path.is_relative_to(root):
        raise PermissionError(f"路径越界: {p}")
    return path

@tool
def list_files(path: str = ".") -> str:
    """列出工作区内某个目录的文件与子目录。"""
    ...

@tool
def read_file(path: str) -> str:
    """读取工作区内某个文本文件（限 200KB）。"""
    ...

TOOLS = [list_files, read_file]
```

### 5.3 agent.py —— StateGraph 闭环

骨架版的 Loop 是 `plan → execute → reflect → finish`（RAG 的 `retrieve` 是后面加的）。几个设计要点：

- **State 用 TypedDict**，节点函数返回"要更新的字段"的字典即可
- **reflect 用结构化输出**（上面 4.2 的 `Decision` 模型），把"是否完成"变成可路由的判断
- **防死循环**：`MAX_ITERATIONS` 硬上限 + `iterations` 计数

```python
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    task: str
    plan: str
    result: str
    feedback: str
    done: bool
    iterations: int

def route(state: AgentState) -> str:
    if state.get("done") or state.get("iterations", 0) >= MAX_ITERATIONS:
        return "finish"
    return "execute"   # 不满意就回到 execute 再来一轮

graph = StateGraph(AgentState)
graph.add_node("plan", plan)
graph.add_node("execute", execute)
graph.add_node("reflect", reflect)
graph.add_node("finish", finish)
graph.add_edge(START, "plan")
graph.add_edge("plan", "execute")
graph.add_edge("execute", "reflect")
graph.add_conditional_edges("reflect", route, {"execute": "execute", "finish": "finish"})
graph.add_edge("finish", END)
app = graph.compile()
```

### 5.4 main.py —— CLI 入口

```python
python main.py "列出项目根目录的文件，说明项目是做什么的"
```

这个薄骨架已经能跑通一轮完整的"规划→执行→反思→汇报"。但注意：此时 execute 没有真工具，只能靠 LLM 自己的知识——所以下一步立刻上**安全沙箱**，把"执行"做实。

---

## 六、安全层：Docker 沙箱的构建与连环坑

> 本章是"坑最多、最有价值"的一章。代码量不大，但环境问题环环相扣，几乎每一步都踩了一个真实的坑。建议收藏。

### 6.1 设计思路

> 📌 **本章讲的是方案 B（`EXEC_MODE=docker`）**。默认的轻量模式（方案 A）没有容器隔离：
> 命令在本机直跑，安全靠"每条 `run_shell`/`run_test` 人工审批 + 危险模式拦截"兜底
> （两套方案的取舍见 3.1 节）。只想用方案 A 的话，本章可以跳过。

让 Agent 在宿主机上直接 `subprocess` 跑命令、又**不设人工审批**，等于裸奔。所以安全设计是分层的：

1. **命令执行进容器**：用非 root 用户 `sandbox` 跑
2. **资源受限**：内存 512M、CPU 1 核、默认断网、超时强制 kill
3. **工作区挂载**：只把目标项目目录挂成 `/workspace`，容器内改的就是宿主机上的项目
4. **危险命令审批**：`rm -rf` / `git push` / `sudo` 等命中规则 → 停下来问人（第九章）

### 6.2 Dockerfile.sandbox

```dockerfile
FROM python:3.11-slim
# 非 root 用户（禁止提权、禁止写系统目录）
RUN useradd -m -u 1000 sandbox \
    && mkdir -p /workspace \
    && chown sandbox:sandbox /workspace
# 沙箱内常用工具（按需最小集）
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir pytest
WORKDIR /workspace
USER sandbox
CMD ["/bin/bash"]
```

### 6.3 sandbox.py —— docker-py 封装

核心就一个函数：把命令扔进容器执行，返回 stdout/stderr/退出码：

```python
import docker
from config import settings

def run_command(cmd, *, cwd="/workspace", timeout=60,
                mem_limit="512m", network=False):
    client = docker.from_env()
    container = client.containers.run(
        image="smart-coder-sandbox:latest",
        command=["/bin/bash", "-lc", cmd],
        working_dir=cwd,
        user="sandbox",
        mem_limit=mem_limit,
        nano_cpus=1_000_000_000,      # 1 CPU
        network_disabled=not network,  # 默认断网！
        detach=True,
        tty=False,
        volumes={str(settings.workspace_root): {"bind": "/workspace", "mode": "rw"}},
    )
    try:
        result = container.wait(timeout=timeout)      # 超时在此抛异常
        stdout = container.logs(stdout=True, stderr=False).decode("utf-8", errors="replace")
        stderr = container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace")
        return {"exit_code": result.get("StatusCode", -1), "stdout": stdout,
                "stderr": stderr, "timed_out": False}
    except Exception:
        container.kill()
        return {"exit_code": -1, "stdout": "", "stderr": "超时或异常",
                "timed_out": True}
    finally:
        container.remove(force=True)
```

### 6.4 工具接入

```python
@tool
def run_shell(command: str, allow_network: bool = False) -> str:
    """在 Docker 沙箱内执行 shell 命令（默认断网、超时 60s）。"""
    r = run_command(command, network=allow_network)
    return 格式化结果(r)   # 把 stdout/stderr/exit_code 拼成文本

@tool
def run_test(path: str = ".") -> str:
    """在沙箱内运行 pytest。"""
    r = run_command("python -m pytest -q", cwd=f"/workspace/{path}", timeout=180)
    return 格式化结果(r)
```

> 📌 **后续演进（见第十二节）**：`run_test` 不再写死 pytest，改为按项目类型自动选命令
> （`npm test` / `go test ./...` / `cargo test` / `mvn test` / `make test` / pytest…），
> 并支持 `command` 参数显式覆盖；`_safe_path` 的边界也从固定 `workspace_root` 改为
> 运行期解析的当前项目目录。

### 6.5 🕳️ 连环坑与解决记录（重点）

**坑 1：docker build 报 `operation not permitted`**

```
failed to update builder last activity time: open ~/.docker/buildx/activity/...: operation not permitted
```

原因：docker buildx 要写宿主机的 `~/.docker/buildx` 状态目录。解决：确认这是沙箱权限问题（不是代码问题），给 docker 命令放行该目录访问。

**坑 2：Docker Hub 直连超时**

```
failed to fetch oauth token: Post "https://auth.docker.io/token": dial tcp ...: i/o timeout
```

原因：`auth.docker.io` / `registry-1.docker.io` 直连不通（典型国内网络）。先测几个候选加速器能否匿名拉取：

```bash
# 能连通的表现是 HTTP 401（registry 要求 token）或 200，000 就是不通
for m in https://docker.m.daocloud.io https://docker.1ms.run https://dockerproxy.net; do
  curl -s -o /dev/null -w "$m -> %{http_code}\n" --max-time 8 "$m/v2/"
done
# 再实测 docker pull（更准）
docker pull docker.1ms.run/library/python:3.11-slim
```

结论：`docker.m.daocloud.io` 已不支持匿名拉取（401），**`docker.1ms.run` / `dockerproxy.net` / `hub.rat.dev` 可用**。

**坑 3：改了 `~/.docker/daemon.json` 把 Docker Desktop 引擎弄卡死**

我最初想通过写 host 侧的 `~/.docker/daemon.json` 加 `registry-mirrors`，结果重启后 Docker 引擎 3 分钟起不来，`docker info` 一直超时。原因：**Docker Desktop for Mac 的引擎配置并不读 host 的 `~/.docker/daemon.json`**（那是 Linux 原生 Docker 的位置），改它反而干扰了引擎。

解决：**回滚 daemon.json + 彻底重启引擎**（quit → kill 残留进程 → 重新 open）。最后采用更干净的路子——**不改 daemon 配置，直接在 Dockerfile 的 `FROM` 里写加速器地址**：

```dockerfile
FROM docker.1ms.run/library/python:3.11-slim   # 等价 library/python:3.11-slim，但走国内可达的源
```

> 💡 教训：优先用"进程内可配置"的方案（改 Dockerfile），不要轻易动全局 daemon 配置。

**坑 4：`docker pull` 报 `Keychain Error (-67674)`**

```
error getting credentials - err: exit status 1, out: `Keychain Error. (-67674)`
```

原因：Docker 配置了 osxkeychain credential helper，拉镜像时要访问 macOS 钥匙串，被沙箱拦截。解决：给 docker 命令完整权限（钥匙串属于系统资源）。**对你本机跑没有任何影响**，这是沙箱环境特有约束。

**坑 5：docker-py 居然是 2016 年的老包**

`pip list` 里是 `docker-py 1.10.6`（老包名），代码里 `import docker` 用的是它。症状：

```python
# 坑 5a：from_env() 报 URLSchemeUnknown: Not supported URL scheme http+docker
# 坑 5b：升级到 7.2.0 后报
TypeError: load_config() got an unexpected keyword argument 'config_dict'
```

5a 是版本太老不认识新协议；5b 是 `docker-py` 和 `docker` 两个包**残留冲突**（旧包的单文件 `auth.py` 覆盖了新包的 `auth/` 目录）。解决：彻底卸载再装干净的：

```bash
pip uninstall -y docker docker-py docker-pycreds
pip install docker        # 装到 7.2.0
# 验证签名里出现 config_dict 参数即正常
python -c "import inspect; from docker import auth; print(inspect.signature(auth.load_config))"
# → (config_path=None, config_dict=None, credstore_env=None)
```

### 6.6 验证沙箱

```bash
python -c "
import sandbox
print(sandbox.run_command('echo hello && whoami && python --version'))
"
# hello
# sandbox        ← 非 root，安全 ✅
# Python 3.11.16
```

跑通这行，安全层的地基就算打好了。

---

## 七、RAG 代码检索引擎

> 📌 **本章讲的是方案 B（`RAG_ENABLED=true`）**。默认的轻量模式不开 RAG，跳过本章不影响主流程：
> `RAG_ENABLED=false` 时 `retrieve_node` 直接返回空上下文，改由执行器用
> `describe_project` / `list_files` / `search_code` / `read_file` 自己定位代码（见 3.1 节）。

> 代码仓库动辄几万行，全塞进 LLM 上下文既不现实也烧钱。RAG 的目标：**给定任务，从代码库里捞出最相关的几个片段**喂给执行器。本项目实现的是"向量 + BM25 混合 + 重排"三段式。

### 7.1 检索链路

```
用户任务
   │
   ├─① 向量检索（ChromaDB，语义相似，top 20）
   ├─② BM25 检索（rank-bm25，关键词命中，top 20）
   │
   └─③ RRF 融合（Reciprocal Rank Fusion 合并两份候选）
        │
        └─④ Cross-Encoder 重排（[query, chunk] 精细打分，取 top k）
             │
             └─▶ 注入 execute 节点的上下文
```

为什么混合？向量检索擅长"语义相似但用词不同"，BM25 擅长"精准关键词"（如函数名 `run_command`、报错里的类名）。RRF 是个便宜的融合公式：`score = Σ 1/(60 + rank)`。最后 cross-encoder 对融合后的候选逐对打分，质量最高。

### 7.2 实现要点（rag.py）

```python
def index_codebase(root=None):
    """扫描代码 → 行级切分（40 行/chunk，重叠 10 行）→ embedding → 写 Chroma + 建 BM25"""
    corpus, meta, ids = [], [], []
    for path in _iter_code_files(root):        # 只扫 *.py
        ...
    embeddings = embedder.encode(corpus, normalize_embeddings=True).tolist()
    collection.add(ids=ids, embeddings=embeddings, documents=corpus, metadatas=meta)
    _bm25 = BM25Okapi([_tokenize(t) for t in corpus])

def retrieve(query, k=5):
    # ① Chroma 向量查询 top 20
    # ② BM25 打分取 top 20
    # ③ RRF 融合
    # ④ cross-encoder 重排取 top k
    return [{"path":..., "start":..., "end":..., "text":..., "score":...}]
```

代码 chunk 切分先用了朴素的"按行切分（40 行、重叠 10 行）"，好处是通用（Python / JS / TS 都适用）。进阶可以做 AST 感知切分（按函数/类切），后续值得投入。

### 7.3 🕳️ 模型"该放哪"的争论与解法

embedding 和 cross-encoder 是 **sentence-transformers 的本地模型**，不花 API 钱、不联网推理——但它们的**权重文件**需要从 Hugging Face Hub 下载。

**第一个坑：模型默认下载到 `~/.cache/huggingface`（真实主机Home目录）**，不在项目/venv 里。要做到项目自包含，用环境变量把缓存指到项目内：

```python
# config.py
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))      # 权重放项目内
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")               # 国内镜像加速
os.environ.setdefault("HF_HUB_OFFLINE", "1")                                # 已缓存则离线加载
```

三个变量的作用：

| 变量             | 解决什么问题                                                                              |
| ---------------- | ----------------------------------------------------------------------------------------- |
| `HF_HOME`        | 默认权重缓存到真实主机Home目录 → 改到项目内 `.cache/huggingface`                          |
| `HF_ENDPOINT`    | huggingface.co 直连超时 → 走 `hf-mirror.com` 镜像                                         |
| `HF_HUB_OFFLINE` | 模型已缓存后仍联网检查、且打印 `unauthenticated requests` 警告 → 离线加载（更快、无警告） |

> 注意：`HF_*` 变量必须在 `import sentence_transformers` **之前**设置才生效，所以放在被所有模块先 import 的 `config.py` 里；`rag.py` 对模型是懒加载，正好配合。

**第二个坑：CrossEncoder 的 `predict()` 卡死不动。**

症状：`Loading weights: 100%` 之后进程卡住。诊断后发现根因是**模型文件下载不完整**（缓存里有 `.incomplete` 残留、`modules.json` 等缺失），加载时它尝试联网补文件，而 huggingface.co 直连不通 → 假死。

解决：设置 `HF_ENDPOINT=hf-mirror` 让补下走镜像，或直接删掉残缺缓存重新下载一遍。顺带建议把 `HF_HUB_OFFLINE=1` 加上——模型完整后根本不该再联网。

### 7.4 集成 retrieve 节点

在 Agent Loop 里加一个 `retrieve` 节点，位置在 `plan` 之后、`execute` 之前：

```python
def retrieve_node(state):
    hits = rag.retrieve(f"{state['task']} {state['plan']}", k=4)
    blocks = [f"### {h['path']}:{h['start']}-{h['end']}\n{h['text']}" for h in hits]
    return {"context": "\n\n".join(blocks)}   # 注入 execute 的提示词

graph.add_node("retrieve", retrieve_node)
graph.add_edge("plan", "retrieve")
graph.add_edge("retrieve", "execute")
```

execute 的系统提示词里带上"相关代码上下文：\n{context}"，Flash 模型就能直接引用真实代码，而不是凭记忆猜。

---

## 八、记忆持久化

### 8.1 为什么需要 checkpoint

LangGraph 的 State 默认在内存里，进程一结束就丢。加上 checkpoint 后：

- 任务跑到一半（比如等人审批时）**进程退出也能恢复**
- 用 `thread_id` 区分不同会话，互不串扰

### 8.2 🕳️ 坑：SqliteSaver 不在 langgraph 核心包

`from langgraph.checkpoint.sqlite import SqliteSaver` 直接报 `ModuleNotFoundError`。原因：`langgraph-checkpoint` 4.x 把存储后端拆成了独立包，SQLite 需要另装：

```bash
pip install langgraph-checkpoint-sqlite
```

### 8.3 集成

```python
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

CHECKPOINT_DB = settings.workspace_root / ".agent_cache" / "checkpoints.sqlite"
CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)
_conn = sqlite3.connect(str(CHECKPOINT_DB), check_same_thread=False)

app = graph.compile(checkpointer=SqliteSaver(_conn))
```

调用时带上会话 id：

```python
app.invoke(state, config={"configurable": {"thread_id": "会话A"}})
app.invoke(state, config={"configurable": {"thread_id": "会话B"}})   # 隔离
```

> 📌 **后续演进（见第十二节）**：`checkpoints.sqlite` 已从"工作区目录下"移到 **agent 自己目录**
> （避免污染被协助的代码库），并靠 `thread_id` 前缀（`项目id::会话id`）实现项目级隔离。
> 单项目时代的旧会话会在首次启动时自动加上默认项目前缀（一次性迁移，见 `workspace.migrate_legacy_state`）。

> `.agent_cache/`、`chroma/` 都是运行时产物，记得进 `.gitignore`。

### 8.4 长期记忆：会话历史 + 偏好文件

checkpoint 只负责"落盘状态"，本身并不会"记住用户说过的话"。本项目在它之上又加了两层，让 agent 能跨会话回忆起用户的偏好：

| 层       | 载体                              | 机制                                             | 作用                   |
| -------- | --------------------------------- | ------------------------------------------------ | ---------------------- |
| 会话历史 | `State.messages`（`add_messages`） | 随 checkpoint 持久化，每轮追加用户话 / 助手答    | 同线程内回放上下文     |
| 长期偏好 | `.agent_cache/memory.md`          | `remember_fact` 写入、`recall_memory` 查询       | 跨线程可召回事实/偏好  |

> 📌 **后续演进（见第十二节）**：记忆文件改为按项目隔离的
> `.agent_cache/projects/<项目id>/memory.md`，"记住我的偏好"不会再跨项目串味。

- 每次运行 `main.py` 只把新提问以 `HumanMessage` 追加进 `messages`，最终答复以 `AIMessage` 写回，不会覆盖历史。
- "记住 xx / 我偏好 xx"这类任务，`execute` 节点会调 `remember_fact` 落盘；询问时先 `recall_memory` 再作答，不靠猜。
- `memory.md` 是全局的，所以**不必带 `--thread`** 也能查回偏好；`--thread` 只用于开多个互相隔离的会话。

示例：

```bash
python main.py "记住：我偏好用 pytest 而不是 unittest"
python main.py "我现在偏好什么测试框架？"   # → 你偏好使用 pytest
```

---

## 九、危险操作审批

### 9.1 思路：危险命令清单 + interrupt

不是所有命令都该让 Agent 自主执行。定义危险模式白名单：

```python
DANGEROUS_PATTERNS = [
    (r"\brm\s+(-[a-z]*r[a-z]*f|-[a-z]*f[a-z]*r)", "递归删除文件"),
    (r"\bgit\s+push\b", "git 推送到远程"),
    (r"\bgit\s+reset\s+--hard\b", "git 硬重置（丢失提交）"),
    (r"\bsudo\b", "提权执行"),
    (r"\bshutdown\b|\breboot\b", "关机/重启"),
]
```

接着用一个 `_danger_check` 函数把"模式清单"翻译成"这次调用要不要拦"——注意只对 `run_shell` 生效，且**联网命令一律视为危险**（要联网意味着可能下载/推送代码）：

```python
import re

def _danger_check(tc: dict) -> str | None:
    """判断一次工具调用是否危险，返回风险说明；不危险返回 None。"""
    if tc.get("name") != "run_shell":   # 文件读写类工具直接放行
        return None
    args = tc.get("args") or {}
    cmd = args.get("command", "")
    if args.get("allow_network"):       # 要联网的默认按危险处理
        return "命令需要网络访问（可能下载代码或推送变更）"
    for pat, why in DANGEROUS_PATTERNS: # 逐条匹配上面的危险模式
        if re.search(pat, cmd):
            return why
    return None
```

在 execute 的工具循环里，每次调用 `run_shell` 前先做检查，命中就 `interrupt()` 冻结图执行：

```python
from langgraph.types import interrupt

if danger := _danger_check(tc):      # tc 是这次工具调用
    decision = interrupt({
        "question": f"是否允许执行以下危险操作？\n工具：{tc['name']}\n参数：{tc['args']}\n风险：{danger}",
    })
    if not (decision or {}).get("approved"):
        return {"result": f"用户已拒绝执行该危险操作（{danger}），任务已终止。"}
```

CLI 侧（main.py）循环检测 pending 的 interrupt 并询问用户：

```python
from langgraph.types import Command

while True:
    st = app.get_state(config)
    pending = 找到 pending 的 interrupt(st)
    if not pending:
        break
    approved = input(f"{pending.value['question']}\n是否允许？[y/N] ") in ("y", "yes")
    final = app.invoke(Command(resume={"approved": approved}), config=config)
```

### 9.2 🕳️ 坑：拒绝之后 Agent 居然"谎报执行成功"

这是个隐蔽 bug，现象：用户输入 `N` 拒绝 `rm -rf`，最终汇报却是"命令成功、退出码 0"。

**根因**（两个叠加）：

1. 拒绝分支原来用 `continue`，跳过了 `result` 变量的更新 → execute 返回空结果
2. finish 节点拿到空结果，模型"脑补"了一段执行成功的话（LLM 幻觉）

**修复**：

1. 拒绝后**直接 `return`**，把"用户已拒绝"作为明确结果返回，不再给模型编造的机会
2. reflect 节点识别 `"用户已拒绝"` 字样就直接 `done=True`，不再让 Pro 评估"是否完成"（避免误判未完成而空转循环）

顺带验证了修复效果：同样拒绝场景下，Pro 调用从 3 次降到 2 次（reflect 短路省了一次）。

---

## 十、测试与验证实录

> 下面是完整的测试用例。每个用例固定两块：**① 执行命令**（可直接复制运行）、**② 终端输出**（下方留了空代码块，把你在终端跑出来的真实结果原样粘贴进去即可，保留格式与缩进）。

### 10.1 用例速查表

| #   | 场景         | 核心验证点                                   | 命令                                                 |
| --- | ------------ | -------------------------------------------- | ---------------------------------------------------- |
| 1   | 冒烟测试     | plan→retrieve→execute→reflect→finish 全链路  | `python main.py "列出项目文件并说明项目作用"`        |
| 2   | 沙箱命令执行 | 命令在 Docker 容器内运行、结果回传           | `python main.py "在沙箱执行 python -c 'print(6*7)'"` |
| 3   | RAG 代码检索 | retrieve 检索到真实代码片段并注入            | `python main.py "sandbox.py 怎么限制超时和内存？"`   |
| 4   | 审批 · 批准  | 危险命令拦截 → 输入 `y` → 沙箱内执行         | `python main.py "执行 rm -rf /tmp/x"` + `y`          |
| 5   | 审批 · 拒绝  | 危险命令拦截 → 输入 `n` → 报告已拒绝、不执行 | `python main.py "执行 rm -rf /tmp/x"` + `n`          |
| 6   | 会话记忆     | `thread_id` 会话隔离、状态可恢复             | 两次 `--thread=xxx` 连续提问                         |

### 10.2 用例 1：冒烟测试（完整链路）

**目的**：验证五个节点都能跑通，LLM 与 RAG 模型正常加载。

```bash
python main.py "列出项目根目录的文件，一句话说明项目是做什么的"
```

**终端输出**：

```text
任务：列出项目根目录的文件，一句话说明这个项目是做什么的
会话：main
============================================================
Loading weights: 100%|█████████████████████| 103/103 [00:00<00:00, 7097.54it/s]
Loading weights: 100%|█████████████████████| 105/105 [00:00<00:00, 7528.37it/s]

============================================================
项目根目录文件已列出，这是一个基于 LangGraph + DeepSeek V4 的有状态代码助手系统，通过规划、检索、执行、反思流程，结合 RAG 和 Docker 沙箱，用自然语言完成代码任务并汇报结果。
```

### 10.3 用例 2：沙箱命令执行

**目的**：验证 `run_shell` 真正在 Docker 沙箱里执行命令并回传结果。

```bash
python main.py "在沙箱里执行 python -c 'print(6*7)'，把结果告诉我"
```

**终端输出**：

```text
任务：在沙箱里执行 python -c 'print(6*7)'，把结果告诉我
会话：main
============================================================
Loading weights: 100%|█████████████████████| 103/103 [00:00<00:00, 6428.67it/s]
Loading weights: 100%|█████████████████████| 105/105 [00:00<00:00, 7633.67it/s]

============================================================
任务完成，输出结果为 42。
```

### 10.4 用例 3：RAG 代码检索

**目的**：验证 retrieve 节点检索到真实代码、execute 能引用它回答（可观察回答是否提到 `timeout`/`mem_limit` 等真实参数）。

```bash
python main.py "结合代码说明 sandbox.py 是怎么做超时和资源限制的"
```

**终端输出**：

```text
任务：sandbox.py 里是怎么做超时和资源限制的？结合代码回答
会话：main
============================================================
Loading weights: 100%|█████████████████████| 103/103 [00:00<00:00, 5393.02it/s]
Loading weights: 100%|█████████████████████| 105/105 [00:00<00:00, 5112.69it/s]

============================================================
sandbox.py 通过 Docker 容器实现超时和资源限制：

**超时**：`container.wait(timeout=timeout)`（默认 60 秒）阻塞等待容器退出，超时则捕获异常、设置 `timed_out=True`、`exit_code=-1`，并调用 `container.kill()` 强制终止。

**资源限制**：在 `client.containers.run()` 中通过容器参数设置——`mem_limit="512m"` 限制内存 512MB，`nano_cpus=1_000_000_000` 限制 1 个 CPU 核心，`network_disabled=not network` 默认断网，`user="sandbox"` 以非 root 用户运行。

**清理**：`finally` 块中 `container.remove(force=True)` 强制删除容器，避免残留。返回 `{exit_code, stdout, stderr, timed_out}` 字典。
```

### 10.5 用例 4：危险操作审批 —— 批准（y）

**目的**：验证 `interrupt()` 拦截 `rm -rf`，输入 `y` 后命令在沙箱内执行。

```bash
python main.py "用 run_shell 执行 rm -rf /tmp/test_approve，告诉我结果"   # 提示后输入 y
```

**终端输出**：

```text
任务：用 run_shell 执行 rm -rf /tmp/nonexist_test，告诉我结果
会话：main
============================================================
Loading weights: 100%|█████████████████████| 103/103 [00:00<00:00, 6466.30it/s]
Loading weights: 100%|█████████████████████| 105/105 [00:00<00:00, 7533.78it/s]

⚠️  需要人工审批
是否允许执行以下危险操作？
工具：run_shell
参数：{'command': 'rm -rf /tmp/nonexist_test'}
风险：递归删除文件
是否允许？[y/N] N

============================================================
已执行 `rm -rf /tmp/nonexist_test`，命令成功，退出码 0，无输出。目标本就不存在，`-f` 静默忽略，无报错。
```

### 10.6 用例 5：危险操作审批 —— 拒绝（n）

**目的**：验证输入 `n` 后 agent **如实报告"已拒绝"且不执行命令**（此场景曾出现"谎报执行成功"bug，修复后应输出"任务已终止/未执行"）。

```bash
python main.py "用 run_shell 执行 rm -rf /tmp/test_reject，告诉我结果"    # 提示后输入 n
```

**终端输出**：

```text
任务：用 run_shell 执行 rm -rf /tmp/nonexist_test，告诉我结果
会话：main
============================================================
Loading weights: 100%|█████████████████████| 103/103 [00:00<00:00, 5754.50it/s]
Loading weights: 100%|█████████████████████| 105/105 [00:00<00:00, 6805.25it/s]

⚠️  需要人工审批
是否允许执行以下危险操作？
工具：run_shell
参数：{'command': 'rm -rf /tmp/nonexist_test'}
风险：递归删除文件
是否允许？[y/N] N

============================================================
任务未执行，用户拒绝运行递归删除命令。
```

### 10.7 用例 6：会话记忆（thread_id）

**目的**：验证 checkpoint 落盘与 `thread_id` 会话隔离——第二条命令能"记得"第一条里交代的事。

```bash
python main.py "记住：我偏好用 pytest 而不是 unittest" --thread=me

# 终端输出如下：
任务：记住：我偏好用 pytest 而不是 unittest
会话：me
============================================================
Loading weights: 100%|█████████████████████| 103/103 [00:00<00:00, 5271.54it/s]
Loading weights: 100%|█████████████████████| 105/105 [00:00<00:00, 6501.55it/s]

============================================================
已记住：您偏好使用 pytest 而非 unittest。
```

```bash
python main.py "我现在偏好什么测试框架？" --thread=me

# 终端输出如下：
任务：我现在偏好什么测试框架？
会话：me
============================================================
Loading weights: 100%|█████████████████████| 103/103 [00:00<00:00, 6999.68it/s]
Loading weights: 100%|█████████████████████| 105/105 [00:00<00:00, 4992.60it/s]

============================================================
你偏好使用 **pytest** 作为测试框架。
```

### 10.8 附加：环境自检（30 秒确认依赖就绪）

**目的**：给读者一个"不跑完整任务也能确认环境 OK"的快速命令。

```bash
# ① 沙箱是否可用（应输出 sandbox 表示非 root）
python -c "import sandbox; print(sandbox.run_command('whoami'))"

## 终端输出：{'exit_code': 0, 'stdout': 'sandbox\n', 'stderr': '', 'timed_out': False}
```

```bash
# ② RAG 能否索引代码库（应输出 chunk 数量）
python -c "import rag; print('chunks =', rag.index_codebase())"

# 终端输出：
Loading weights: 100%|█████████████████████| 103/103 [00:00<00:00, 7291.61it/s]
chunks = 25
```

---

## 十一、经验总结与踩坑清单

### 11.1 一句话经验

> **别信教程的版本号，一切以实测为准。** 本项目 80% 的坑来自"文档/教程没跟上版本"：DeepSeek 模型改名、langchain-openai 字段改名、docker-py 版本、langgraph-checkpoint 拆包、huggingface 默认缓存位置……每个都只能靠跑一次真实调用暴露出来。

### 11.2 完整踩坑清单（按阶段汇总）

**① 接入层（DeepSeek API）**

1. `deepseek-chat/reasoner` 已停用 → 用 `deepseek-v4-flash` / `deepseek-v4-pro`
2. thinking 模式不支持强制 `tool_choice` → 结构化输出前关 thinking
3. `json_schema` 响应格式不支持 → 统一 `method="function_calling"`
4. langchain-openai 字段改名：`model→model_name`、`base_url→openai_api_base`
5. stdin 下 `load_dotenv()` 断言失败 → 传显式路径

**② Docker / 网络**

6. buildx 写 `~/.docker` 被沙箱拦 → 放行该目录
7. Docker Hub 直连超时 → 加速器，最后用 `FROM docker.1ms.run/...`
8. `daemon.json` 误改导致引擎卡死 → 回滚 + 彻底重启；且 Docker Desktop 不读 host 的 daemon.json
9. docker pull `Keychain Error` → credential helper 访问钥匙串被拦（沙箱特有）
10. docker SDK 是老 `docker-py 1.10.6` → 升级 `docker 7.2.0` 并清理残留冲突

**③ 模型 / HuggingFace**

11. 权重默认下到 `~/.cache/huggingface`（家目录） → `HF_HOME` 指到项目内
12. huggingface.co 直连超时 → `HF_ENDPOINT=hf-mirror.com`
13. CrossEncoder `predict()` 假死 → 缓存不完整，删了重下 / 补镜像
14. `unauthenticated requests` 警告 → `HF_HUB_OFFLINE=1` 离线加载

**④ LangGraph**

15. `SqliteSaver` 找不到 → `langgraph-checkpoint-sqlite` 是独立包
16. interrupt 拒绝后 agent 谎报成功 → 拒绝分支直接 return + reflect 识别拒绝态

### 11.3 目录结构一览

```
agent_env/                  ← 本身是 Python venv
├── .env                    # API Key、模型分工、HF 配置（已 gitignore）
├── .gitignore
├── config.py               # Settings + get_llm() + HF_* 环境变量
├── workspace.py            # 多项目注册表 + 运行期工作区解析 + 语言识别 + 共享文件遍历
├── tools.py                # list_files / read_file / search_code / describe_project
│                           #   + run_shell / run_test（按语言自动选命令）+ 记忆工具
├── sandbox.py              # 命令执行：host 本机直跑（默认）/ docker 沙箱
├── rag.py                  # 代码索引（多语言）+ 混合检索 + 重排，按项目隔离
├── agent.py                # StateGraph：plan→retrieve→execute→reflect→finish
├── engine.py               # 任务引擎：绑定工作区 + 审批循环 + 事件回调（CLI/Web 共用）
├── cost.py                 # LLM 成本统计（contextvar 按任务分账）
├── main.py                 # CLI 入口（--project= 可指定任意项目）
├── webapp.py               # FastAPI 服务：项目增删切换 + 系统弹窗/目录浏览 + SSE 任务流
├── dirpicker.py            # 系统原生"选择文件夹"弹窗（macOS/Linux/Windows）
├── web/index.html          # 前端：项目下拉框 + 添加工程目录（系统弹窗）+ 审批弹窗
├── run-web.sh              # 一键启动
├── Dockerfile.sandbox      # 沙箱镜像（python:3.11-slim + git + pytest，仅 Python 项目用）
├── md/                     # 使用/维护文档（gitignore）
├── .cache/huggingface/     # 本地模型权重（gitignore）
├── chroma/                 # ⚠️ 旧版遗留向量库（新版写 .agent_cache/projects/<id>/chroma）
└── .agent_cache/           # agent 自己的状态（gitignore）
    ├── projects.json       #   项目清单（界面上添加的目录都记在这）
    ├── checkpoints.sqlite  #   所有项目的会话历史，靠 thread_id 前缀隔离
    └── projects/<项目id>/  #   每个项目的 memory.md 与 chroma/（互不污染）
```

---

## 十二、支持任意项目：多项目工作区

最初这个 Agent 的假设是"操作 Python 代码库"，要辅助别的语言得改好几处硬编码。
落地时换了个更彻底的做法：**不针对某种语言做适配，而是把"工作区"从启动时常量
升级为运行时可切换的值，并让语言/测试命令自动识别**。这样任意语言、任意目录都能用。

### 12.1 原来有哪 5 处语言/单项目假设

| # | 位置 | 原来的假设 | 现在 |
| --- | --- | --- | --- |
| 1 | `rag.py` | `root.rglob("*.py")`，只索引 Python | 多语言后缀集合（js/ts/vue/go/rs/java/c/cpp/… + 配置文档），统一走 `workspace.iter_files` |
| 2 | `tools.py` `run_test` | 写死 `python -m pytest` | 按清单文件自动推断测试命令，支持 `command` 覆盖 |
| 3 | `Dockerfile.sandbox` | 只有 Python 运行时 | 默认执行模式改为 `EXEC_MODE=host`（任意语言可用）；docker 保留给 Python 项目强隔离 |
| 4 | `tools.py` / `rag.py` 的 skip 集合 | 把 `bin/lib/include/share` 当成虚拟环境目录整个跳过 | 只有工作区根**本身是 venv** 时才跳这些名字（见 12.3） |
| 5 | `config.py` + `tools/agent/rag` 的模块级常量 | `workspace_root` import 时快照，不可变 | `workspace.py` 运行期解析 + contextvar 按任务绑定 |

### 12.2 架构：工作区怎么"活"起来

```
config.settings.workspace_root          ← 只是回退值（一个项目都没添加时兜底）
        ▲
        │  workspace.current() 解析顺序
        │
   ┌────┴─────────────────────────────────────────┐
   │ 1. contextvar（engine.run_task 用 bind() 绑）  │  ← 每个任务线程独立，天然并发安全
   │ 2. 注册表 current（界面上选中的项目）           │
   │ 3. settings.workspace_root（回退）             │
   └──────────────────────────────────────────────┘
```

- **为什么必须 contextvar**：Web 端 `MAX_RUNNING=4`，4 个任务可并发。如果只是"改个全局变量"
  来表示当前项目，两个任务跑不同项目就会互相串。contextvar 按线程隔离，
  正好对上"每个任务一个线程"的模型，不用加锁。
- **为什么 checkpointer 不用重建**：所有项目共用一个 `checkpoints.sqlite`，
  `thread_id` 加项目前缀（`项目id::会话id`）即可隔离，`app` 依然是 import 时编译一次。
- **成本统计也按任务分账**：`cost.tracker` 改成 contextvar 感知的代理
  （`tracker.session()`），否则并发任务的 token 会混在一起。

### 12.3 顺手修掉的两个隐藏 bug

**① skip 集合误伤源码目录。** 原 skip 集合含 `bin/lib/include/share/site-packages`
—— 那是"本仓库自己就是 venv"的历史包袱。对 Go/C/C++/Rust 项目，`include/`、`lib/`、`bin/`
是**真实源码目录**，会被静默整个跳过，搜索直接漏结果。现在的规则：

```
工作区根有 pyvenv.cfg（本身就是 venv）→ 额外跳过 bin/lib/include/share/Scripts
任意位置的 .venv/venv（含 pyvenv.cfg）→ 跳过整个目录
其他项目                          → bin/lib/include 照常搜索
```

**② RAG 索引不跟项目走。** BM25 与 chunk 元数据是进程级单例，界面切换项目后
如果不重建，检索会把**上一个项目**的代码片段塞进 prompt。现在索引与项目 id 绑定
（切项目必定重建），并按 `REINDEX_TTL`(30s) + 文件指纹自动跟随文件变动
（`RAG_AUTO_REINDEX=true`）。

### 12.4 状态不再写进用户仓库

| | 改造前 | 改造后 |
| --- | --- | --- |
| checkpoint | `<工作区>/.agent_cache/checkpoints.sqlite` | `<agent>/.agent_cache/checkpoints.sqlite`（thread_id 带项目前缀） |
| 长期记忆 | `<工作区>/.agent_cache/memory.md`（全局一份） | `<agent>/.agent_cache/projects/<项目id>/memory.md` |
| 向量库 | `<工作区>/chroma/` | `<agent>/.agent_cache/projects/<项目id>/chroma/` |
| 项目清单 | 无（只有 `.env` 里一个 `WORKSPACE_ROOT`） | `<agent>/.agent_cache/projects.json` |

被协助的代码库里**不会再出现 `.agent_cache/` 或 `chroma/`**。
升级时旧数据会自动"认领"到默认项目名下（一次性、幂等，见
`workspace.migrate_legacy_state()`）：记忆文件搬过去、裸 `thread_id`（`main`/`me`…）
加上项目前缀，旧会话历史继续可用。

### 12.5 项目识别：让 agent 自己搞清"这是什么项目"

新增 `describe_project` 工具（和 `/api/projects/<id>/describe` 接口），
返回语言、清单文件、源码后缀直方图、自动推断的测试命令、`package.json` scripts、
Makefile targets、README 摘要。plan 与 execute 节点每次都把这份**项目简报**注入 prompt，
所以 agent 不会对着 Go 项目猜"用 pytest 跑一下"。

识别优先级：清单文件（`package.json`/`go.mod`/`Cargo.toml`/`pyproject.toml`…）
→ 若声明语言没有源码，则按后缀直方图纠正（例如只有 `.ts` 却漏了 `package.json`）。

测试命令推断表：

| 项目类型 | 判定依据 | 命令 |
| --- | --- | --- |
| node | `package.json` 有 `scripts.test` | `npm test` |
| python | `tests/` 或 `test_*.py` 或 `pyproject.toml` | `python -m pytest -q` |
| go | `go.mod` | `go test ./...` |
| rust | `Cargo.toml` | `cargo test` |
| java | `pom.xml` / `gradlew` | `mvn -q test` / `./gradlew test` |
| ruby | `Gemfile` | `bundle exec rspec` |
| dotnet | `*.csproj` / `*.sln` | `dotnet test` |
| 其他 | `Makefile` 里有 `test`/`check`/`ci` | `make test` |
| 兜底 | — | 返回提示，让模型用 `run_shell` 显式指定 |

### 12.6 实测（非 Python 项目端到端）

用一个离线可跑的 Node fixture（`npm test` → `node test/run.js`，且故意留了个 bug）验证：

```
[提交] task_id=7ab4… project=node-demo-2c642f9c workspace=/private/tmp/…/node-demo
[日志] 项目：/private/tmp/…/node-demo（node，测试命令：npm test）
[审批#1] 工具=run_test 原因=轻量模式（无 Docker 沙箱）：命令在本机直接执行
         参数={"path": "."}   → 已自动批准
===== 任务完成 =====
项目是 Node.js（ESM），测试命令为 `npm test`（实际执行 `node test/run.js`）。
运行测试后 1 个用例失败：`add(2,3)` 期望 5，实际 -1。
原因是 `src/math.js` 第 2 行把加法写成了减法 `return a - b;`。建议改为 `return a + b;`
----- 成本 ----- 合计: 9 次调用, 21096 tokens
```

验证到位的点：任务绑定到非 Python 项目、`run_test` 自动选 `npm test`、
host 模式逐条审批经 HTTP 往返恢复执行、多语言 RAG 确实索引了 `.js`
（chroma 里是 `package.json` / `src/math.js` / `test/run.js` 三个 chunk）、
被协助的目录里**没有**多出任何 agent 文件。

---

## 十三、Web 界面：项目管理与人工审批

界面（`web/index.html` + `webapp.py`）不再只是"输入任务 + 看结果"，顶部多了一条项目栏：

```
🤖 smart-coder   [ 项目下拉框 ▾ ]   [＋ 添加工程目录]  [浏览…]  [⟳]
node · /Users/you/code/my-app · 测试：npm test · 执行模式：host
```

### 13.1 「添加工程目录」为什么必须由服务端弹窗

直觉方案是用浏览器的目录选择能力（`<input type="file" webkitdirectory>` 或
`showDirectoryPicker()`），做得跟"上传文件"一样。**但这条路拿不到可用路径**：
出于安全设计，浏览器只暴露文件名/相对路径，**绝不暴露绝对路径**，
后端拿不到"要操作哪个目录"这个最关键的信息 —— 那正是这个 Agent 的立身之本。

好在部署形态帮了忙：本服务只监听 `127.0.0.1`，**浏览器与后端在同一台机器**。
于是改成由**服务端进程**去弹操作系统原生的文件夹选择窗口（新增 `dirpicker.py`）：

| 平台 | 命令 | 体验 |
| --- | --- | --- |
| macOS | `osascript` → `choose folder` | Finder 风格原生窗口，返回 POSIX 绝对路径 |
| Linux | `zenity --file-selection --directory`（退回 `kdialog`） | GTK 原生窗口 |
| Windows | PowerShell `FolderBrowserDialog` | 系统文件夹对话框 |

两边都满足：用户看到的是自己系统熟悉的"选文件夹"窗口，服务端拿到的是真实绝对路径。

**几个必须处理的细节**（都是实测踩出来的）：

- **取消不能当报错**：macOS 取消返回 `-128`、zenity 取消是静默 `exit 1`，
  都按 `canceled` 处理；但**只有 `rc==1 且没有任何 stderr` 才算取消** ——
  真正的失败一定会在 stderr 留信息（例如 `-1743` 未授权 Apple Events），
  无脑把 `rc==1` 当取消会把失败静默吞掉、界面毫无反应。
- **超时要分两层**：弹窗自己带超时（AppleScript `with timeout of N seconds`），
  触发时错误码是 `-1712`，要单独识别成"等待超时"；`subprocess` 的 timeout 只做最后兜底
  （多留 15s），防止某个平台的对话框完全不响应时把 HTTP 请求永久挂死。
- **只允许一个弹窗**：连点按钮不该弹出一堆窗口，用锁挡住并返回 `429`。
- **`rc=0` 却没有输出**：说明选择器被中断，明确报错而不是返回一个空路径。
- **别用 `tell me to activate` 抢焦点**：`me` 指的就是 `osascript` 自己，而它是
  `BackgroundOnly` 进程、**永远进不了前台**，系统会一直等它变前台等到约 **2s** 激活超时
  才继续 —— 实测「spawn 到即将弹出面板」由 `0.1s` 变成 `2.0~2.3s`（18 次稳定复现），
  而这 2s 毫无收益（面板打开期间 LaunchServices 的前台应用始终没变，面板该在前台就在前台）。
  所以默认不执行它；个别机器上真被浏览器挡住时，设 `SMARTCODER_PICK_DIR_ACTIVATE=1` 换回旧行为。
- **请求耗时 = 面板 + 人手时间**：这个接口是**同步阻塞**的（挂到用户选完/取消，最多 180s），
  所以浏览器 DevTools 里它总是显示几秒 —— 那不是服务端慢：接口自身的处理
  （锁 + `normalize_dir` + `describe_project(deep=False)`）实测只有 **8～17ms**，
  时间全花在原生面板上。
- **选完立刻校验**：复用添加项目那套规则（绝对路径、存在、是目录、非根目录、可读），
  非法路径在加入列表前就被拒绝。

### 13.2 兜底：页面内浏览

服务器没有图形界面（容器/无头）或通过 SSH 端口转发访问时，系统弹窗会开在服务器那台机器上，
用户根本看不到。所以保留了**页面内浏览**（`GET /api/fs`）作为替代路径：

- 前端启动时读 `/api/health` 的能力探测，若确定弹不出窗口，主按钮直接走页面内浏览并说明原因；
- 弹窗失败/超时（`501/502/408`）时自动切到页面内浏览，并把原因显示出来；
- 也可以在弹窗不可用时点「浏览…」手动进入。

页面内浏览会逐层列出目录，并标注哪些"看起来是 node/go/python 项目"（命中清单文件）、
哪些"已添加"；还支持直接粘贴绝对路径跳转。

### 13.3 接口一览

| 接口 | 作用 |
| --- | --- |
| `POST /api/dialog/directory` | 弹**系统原生**文件夹选择窗口，返回校验过的绝对路径（阻塞至选完/取消/超时） |
| `GET /api/projects` | 项目列表 + 当前选中 + 项目简报 + 执行模式 + 弹窗能力探测 |
| `POST /api/projects` | 添加目录 `{path, name?}`（同路径幂等，自动切为当前） |
| `POST /api/projects/{id}/select` | 切换当前项目 |
| `DELETE /api/projects/{id}` | 从列表移除（**不动磁盘数据**） |
| `GET /api/projects/{id}/describe` | 深度识别：语言/测试命令/源码构成/README |
| `GET /api/fs?path=/abs` | 页面内目录浏览（兜底方案，只列目录并标注是否像项目） |
| `GET /api/workspace` | 当前工作区（旧接口，保留兼容） |
| `POST /api/tasks` | 提交任务，带 `project_id`（**提交时绑定**，之后切项目不影响在跑的任务） |
| `GET /api/tasks/{id}/events` | SSE：`log` / `approval` / `done` / `error` / `close` |
| `POST /api/tasks/{id}/approve` | 审批回复，唤醒引擎线程 |

安全边界：路径必须是**存在的绝对路径**，拒绝 `/`、相对路径、文件路径与不可读目录；
文件读写工具始终被限制在**当前项目目录**内（`_safe_path` 越界即拒绝）。

自动化测试（无头环境）可以用环境变量把弹窗替换成一个命令：

```bash
SMARTCODER_PICK_DIR_CMD='echo /tmp/my-project' python -m uvicorn webapp:app
```

macOS 上如果发现系统面板被浏览器窗口挡住（默认不再调用 `tell me to activate`，因为它会白等约 2 秒），
可以用开关换回带抢焦点的旧行为：

```bash
SMARTCODER_PICK_DIR_ACTIVATE=1 python -m uvicorn webapp:app
```

命令行同样支持任意项目：

```bash
python main.py "这个项目怎么跑测试？跑一下"                     # 用界面里最后选中的项目
python main.py "跑一下测试" --project=/Users/you/code/my-app    # 登记并切换到该目录
```

---

## 后续展望

- [ ] **文件写入/编辑工具**：目前只有读工具 + `run_shell`，改代码得靠 shell 命令（如 sed），
      "修 bug"类任务不趁手；加 `write_file` / `apply_patch` 会明显提升实用性
- [ ] 多语言沙箱镜像：按项目类型选镜像（`node:20-slim` / `golang` / `rust`），兼顾隔离与通用
- [ ] AST 感知的代码切分（按函数/类），提升检索精度
- [ ] 并行任务的资源配额（`MAX_RUNNING=4`，单个任务会占用多轮 LLM 调用）
- [ ] Git 操作的专属工具（而非裸 `git` 命令 + 审批）
- [ ] 项目级配置记忆（例如"这个项目测试要加 `--experimental-vm-modules`"写进 projects.json）
