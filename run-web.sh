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
# 当前项目由 workspace.py 的注册表决定，界面上可随时添加/切换（不再需要改 .env 重启）
"$PY" - <<'PY' || true
import workspace
p = workspace.current_project()
if p:
    print(f"== 当前项目：{p['name']}（{p['lang']}）-> {p['path']}")
else:
    print("== 还没有添加项目：打开页面后点「＋ 添加工程目录」")
print(f"== 已登记 {len(workspace.list_projects()['projects'])} 个项目，"
      f"可在页面顶部下拉框切换 ==")
PY
exec "$PY" -m uvicorn webapp:app --host 127.0.0.1 --port "$PORT"
