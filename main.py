"""CLI 入口：python main.py "你的任务" """
import sys

from agent import app


def main() -> None:
    task = " ".join(sys.argv[1:]) or "列出当前项目根目录下的文件，并说明这个项目是做什么的"
    print(f"\n任务：{task}\n" + "=" * 60)
    final = app.invoke({
        "task": task, "plan": "", "result": "",
        "feedback": "", "done": False, "iterations": 0,
    })
    print("\n" + "=" * 60)
    print(final["result"])


if __name__ == "__main__":
    main()
