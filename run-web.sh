#!/usr/bin/env bash
# 本机 Web 模式一键启动：
#   bash run-web.sh            # 默认端口 8000
#   PORT=9000 bash run-web.sh  # 自定义端口
# 然后浏览器打开 http://127.0.0.1:8000/
# 首次运行会自动安装 fastapi（仓库 venv 里已带 uvicorn）。
set -euo pipefail
cd "$(dirname "$0")"

if [[ -x ./.venv/bin/python ]]; then
  PY=./.venv/bin/python    # 同事标准安装：python3.11 -m venv .venv
elif [[ -x ./bin/python ]]; then
  PY=./bin/python          # 本仓库自身就是 venv
else
  PY=python3
fi

# 首次运行：缺核心依赖 → 给出安装指引；只缺 fastapi/uvicorn → 自动补装
if ! "$PY" -c "import langgraph, langchain_openai, dotenv" >/dev/null 2>&1; then
  echo "缺少核心依赖，请先安装："
  echo "  $PY -m pip install -r requirements.txt"
  echo "（完整模式：pip install -r requirements-full.txt）"
  exit 1
fi
if ! "$PY" -c "import fastapi, uvicorn" >/dev/null 2>&1; then
  echo "== 首次运行：安装 fastapi / uvicorn =="
  "$PY" -m pip install fastapi uvicorn
fi

PORT="${PORT:-8000}"
echo "== 浏览器打开 http://127.0.0.1:${PORT}/ （Ctrl+C 停止）=="
echo "== 当前工作区：$(grep -E '^WORKSPACE_ROOT=' .env 2>/dev/null | cut -d= -f2- || echo '(未设置=agent自身目录)')"
exec "$PY" -m uvicorn webapp:app --host 127.0.0.1 --port "$PORT"
