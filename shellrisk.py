"""命令危险性判定：按"命令位置"识别，并给出**沙箱兜不兜得住**的等级。

**定位先说清**：本模块是 **consent 层**（决定"要不要问人"），**不是 enforcement 层**。
真正的强制隔离在 `confinement.py`（macOS Seatbelt / Linux bubblewrap）：任何能"算出"命令的
写法（`python -c`、`base64 -d | sh`、变量拼接、别名）都能绕过文本分析，所以文本判定
只负责"减少噪音 + 决定该不该弹窗"，拦住命令是内核围栏的事。这也是 Codex / Claude Code /
DSH 的共同结构。

**为什么要分级**（审计 §2.4 之后的结论）：早期把"看起来危险"当判据，于是
`grep -rn "rm -rf" .` 被拦、`rm -r -f` 反而漏。有内核围栏之后，正确的判据变成
"**这条命令的副作用，围栏能不能兜住**"：

| 等级 | 含义 | 判定后怎么处理 |
|---|---|---|
| `safe` | 无可执行副作用 | 直接跑（沙箱内），不打扰人 |
| `contained` | 破坏只落在工作区内 —— 围栏 + 任务快照能兜住 | 有围栏就直接跑（可回滚）；无围栏/无快照才问人 |
| `uncontained` | 围栏兜不住：网络/远端副作用、提权、设备、管道执行代码 | **必须人工审批**（`needs` 说明至少需要哪种放宽） |

判定步骤（"少误报"的部分仍然重要）：

1. **预递归**：先挖出 `$(...)` 与反引号里的内容各自判一遍，外层用占位符替换后继续分析；
2. **引号感知分词**：`shlex`(posix + punctuation_chars) —— `echo 'rm -rf'` 里的 `rm -rf`
   只是 echo 的一个参数，不是命令；
3. **命令位置判定**：只有"简单命令的第一个词"是命令（跳过 `VAR=val`、穿透
   sudo/env/timeout/nohup/xargs/command/… 与 `bash -c`/`eval`/`find -exec`），
   重定向目标（`> file`）不算命令位置；
4. **兜底**：命令无法完整解析时（引号不闭合等）退回旧的保守正则，按 `uncontained` 处理。

**根本局限（必须知道）**：`python -c "shutil.rmtree(...)"`、`node -e "..."` 这类
"解释器 + 代码字符串"能执行任意操作，静态判定不可能覆盖 —— 这正是必须有内核围栏的原因。
"""
from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass

#: 命令替换的最大递归深度（超过就放弃深入，交回外层规则）
MAX_DEPTH = 4

#: 等级
SAFE = "safe"
CONTAINED = "contained"          # 沙箱 + 快照能兜住
UNCONTAINED = "uncontained"      # 沙箱兜不住，必须审批
LEVELS = (SAFE, CONTAINED, UNCONTAINED)

#: 放宽需求：批准之后至少要把沙箱放宽到哪一档
NEEDS_NETWORK = "network"
NEEDS_FULL = "full"

#: 能起"新简单命令"的分隔符
SEPARATORS = {";", ";;", "&&", "||", "|", "&", "(", ")"}
PIPE = "|"
#: 重定向：其后是文件名，不是命令位置
REDIRECTS = {">", ">>", "<", "<<", ">&", "<&", ">|", "&>", "&>>"}

#: shell 解释器：带 -c 时递归解析其字符串参数
SHELLS = {"bash", "sh", "zsh", "dash", "ksh", "csh", "tcsh", "fish", "ash"}
#: 这些包装器不改变"后面那个词才是真正的命令"，跳过它们继续找
WRAPPERS = {"env", "nohup", "setsid", "time", "nice", "ionice", "stdbuf",
            "command", "builtin", "exec", "xargs", "timeout", "watch"}
#: 直接判定为提权的前缀命令
PRIVILEGE = {"sudo", "doas"}
#: 这些选项后面跟一个值，跳过选项时要一并跳过（如 `nice -n 10`）
VALUE_OPTS = {"-n", "-u", "-o", "-e", "-C", "-s", "-I", "-P", "-c"}

#: 无论带什么参数都需要网络的命令（围栏默认禁网 → 需要一次审批放行"网络"档）
ALWAYS_NETWORK = {
    "curl", "wget", "http", "https", "aria2c", "ssh", "scp", "sftp", "rsync",
    "nc", "ncat", "netcat", "telnet", "ping", "dig", "nslookup", "host", "traceroute",
    "gh", "glab", "hub", "svn",
}

#: 只有这些子命令才需要网络（`npm test`、`go build` 之类不该被打扰）
NETWORK_SUBCOMMANDS = {
    "pip": {"install", "download", "wheel"}, "pip3": {"install", "download", "wheel"},
    "npm": {"install", "i", "ci", "add", "update", "view", "info", "publish", "exec", "create"},
    "yarn": {"install", "add", "upgrade", "create", "publish"},
    "pnpm": {"install", "add", "update", "create", "publish"},
    "bun": {"install", "add", "update", "create"},
    "go": {"get", "install", "mod"},
    "cargo": {"install", "add", "update", "publish", "fetch", "search"},
    "gem": {"install", "update", "fetch"}, "composer": {"install", "require", "update"},
    "apt": {"install", "update", "upgrade", "dist-upgrade", "full-upgrade"},
    "apt-get": {"install", "update", "upgrade", "dist-upgrade", "full-upgrade"},
    "brew": {"install", "upgrade", "update", "tap", "reinstall"},
    "port": {"install", "upgrade"},
    "docker": {"pull", "push", "build", "run", "login", "search"},
    "podman": {"pull", "push", "build", "run", "login", "search"},
    "mvn": {"dependency", "deploy", "release"}, "gradle": {"build", "dependencies"},
}

_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_TIMEOUT_ARG = re.compile(r"^\d+(\.\d+)?[smhd]?$")

#: 兜底用的旧规则（命令无法完整解析时启用，按"宁可多问"处理）
FALLBACK_PATTERNS = [
    (r"\brm\s+(-[a-z]*r[a-z]*f|-[a-z]*f[a-z]*r)", "递归删除文件"),
    (r"\bgit\s+push\b", "git 推送到远程"),
    (r"\bgit\s+reset\s+--hard\b", "git 硬重置（丢失提交）"),
    (r"\bsudo\b", "提权执行"),
    (r"\bchmod\s+777\b", "开放所有权限"),
    (r"\bshutdown\b|\breboot\b", "关机/重启"),
]


@dataclass(frozen=True)
class Verdict:
    """一次命令判定的结论。"""

    level: str                      # safe | contained | uncontained
    reason: str = ""
    needs: str = ""                 # 批准后最少需要的放宽："" | network | full

    @property
    def risky(self) -> bool:
        return self.level != SAFE


_SAFE = Verdict(SAFE)

#: 严重度排序：一条命令里有多个片段时，取最严重的那个
_LEVEL_RANK = {SAFE: 0, CONTAINED: 1, UNCONTAINED: 2}
_NEEDS_RANK = {"": 0, NEEDS_NETWORK: 1, NEEDS_FULL: 2}


def _worse(a: Verdict | None, b: Verdict | None) -> Verdict | None:
    """两个判定里取更严重的（先看等级，再看放宽需求）。"""
    if a is None:
        return b
    if b is None:
        return a
    return a if (_LEVEL_RANK[a.level], _NEEDS_RANK[a.needs]) >= \
                (_LEVEL_RANK[b.level], _NEEDS_RANK[b.needs]) else b


def command_verdict(command: str) -> Verdict:
    """判定一条 shell 命令（纯函数）。返回 Verdict；安全时 level == "safe"。"""
    if not command or not command.strip():
        return _SAFE
    return _risk(command) or _SAFE


def command_risk(command: str) -> str | None:
    """**保守口径**：只要不是 safe 就给风险说明（供"没有内核围栏"的场景使用）。

    host 模式（EXEC_MODE=host）没有围栏，contained 级的破坏也拦不住，
    所以那里用这个更严的口径；有围栏时请用 command_verdict() 分级判断。
    """
    v = command_verdict(command)
    return (v.reason or None) if v.risky else None


# ---------------- 第 1 步：命令替换 ----------------


def _substitutions(text: str) -> list[tuple[int, int, str]]:
    """找出 `$(...)` 与 `` `...` `` 的区间：[(start, end, inner)]。

    自己做括号配平，而不是正则 —— `$(rm -rf "$(pwd)")` 这种嵌套正则处理不了。
    """
    found: list[tuple[int, int, str]] = []
    i = 0
    while i < len(text):
        if text.startswith("$(", i):
            depth, j = 1, i + 2
            while j < len(text) and depth:
                depth += (text[j] == "(") - (text[j] == ")")
                j += 1
            if depth == 0:                       # 配平成功才认，否则按普通文本处理
                found.append((i, j, text[i + 2:j - 1]))
                i = j
                continue
        elif text[i] == "`":
            j = text.find("`", i + 1)
            if j != -1:
                found.append((i, j + 1, text[i + 1:j]))
                i = j + 1
                continue
        i += 1
    return found


def _mask_substitutions(text: str, spans: list[tuple[int, int, str]]) -> str:
    """把命令替换换成占位符，保持外层词法结构不变（已单独判过，这里不再重复判）。"""
    out, last = [], 0
    for s, e, _ in spans:
        out.append(text[last:s])
        out.append(" __SUBST__ ")
        last = e
    out.append(text[last:])
    return "".join(out)


# ---------------- 第 2~3 步：分词 + 命令位置 ----------------


def _risk(text: str, depth: int = 0, piped: bool = False) -> Verdict | None:
    """递归判定：piped 表示这条命令是否从管道读标准输入。"""
    if depth > MAX_DEPTH:
        return None
    # 1) 命令替换里的内容，各自当命令判一遍
    spans = _substitutions(text)
    worst: Verdict | None = None
    for _, _, inner in spans:
        worst = _worse(worst, _risk(inner, depth + 1))
    # 2) 引号感知分词
    try:
        lex = shlex.shlex(_mask_substitutions(text, spans), posix=True,
                          punctuation_chars=";&|()<>")
        lex.whitespace_split = True
        lex.commenters = ""                      # `#` 在本项目里更可能是文件名的一部分
        tokens = list(lex)
    except ValueError:
        # 引号不闭合 / 无法解析 → 退回旧正则，保持"宁可多问"
        return _fallback(text)

    # 3) 逐段找"命令位置"
    cmd_pos, piped_next = True, False
    i, total = 0, len(tokens)
    while i < total:
        tok = tokens[i]
        if tok in SEPARATORS:                    # 分隔符之后又是一个命令位置
            piped_next = tok == PIPE
            cmd_pos = True
            i += 1
            continue
        if tok in REDIRECTS:                     # `> file`：目标是文件名，不是命令
            cmd_pos = False
            i += 1
            continue
        if not cmd_pos:
            i += 1
            continue
        words = []
        while i < total and tokens[i] not in SEPARATORS and tokens[i] not in REDIRECTS:
            words.append(tokens[i])
            i += 1
        worst = _worse(worst, _simple_risk(words, depth, piped_next))
        cmd_pos = False
    return worst


def _skip_options(words: list[str], i: int) -> int:
    """跳过 `-x --long` 形式的选项（带值的选项连值一起跳）。"""
    while i < len(words) and words[i].startswith("-") and words[i] != "--":
        opt = words[i]
        i += 1
        if opt in VALUE_OPTS and i < len(words):
            i += 1
    return i


def _simple_risk(words: list[str], depth: int, piped: bool = False) -> Verdict | None:
    """判定一条"简单命令"（一段不带分隔符的词序列）。"""
    i = 0
    while i < len(words) and _ASSIGN.match(words[i]):     # 前置 VAR=val
        i += 1
    # 穿透包装器，直到找到真正的命令词
    while i < len(words):
        base = os.path.basename(words[i])
        if base in PRIVILEGE:
            # 提权：围栏挡不住"换一套身份绕过围栏"，必须人拍板
            return Verdict(UNCONTAINED, "提权执行（可能绕过围栏）", NEEDS_FULL)
        if base == "env":                                # env 还能带 VAR=val
            i = _skip_options(words, i + 1)
            while i < len(words) and _ASSIGN.match(words[i]):
                i += 1
            continue
        if base in WRAPPERS:
            i = _skip_options(words, i + 1)
            if i < len(words) and _TIMEOUT_ARG.match(words[i]):   # timeout 5 …
                i += 1
            continue
        break
    if i >= len(words):
        return None

    cmd = os.path.basename(words[i])
    rest = words[i + 1:]

    # ---- shell 解释器：`bash -c "…"` 递归；`… | sh` 无从预知内容 ----
    if cmd in SHELLS:
        for j, w in enumerate(rest):
            if w == "-c" or (w.startswith("-") and not w.startswith("--") and "c" in w[1:]):
                if j + 1 < len(rest):
                    return _risk(rest[j + 1], depth + 1)
        if piped:
            return Verdict(UNCONTAINED,
                           "把管道内容当脚本执行（如 curl … | sh），内容无法预知",
                           NEEDS_NETWORK)
        return None
    if cmd == "eval":
        return _risk(" ".join(rest), depth + 1)

    # ---- 网络需求：围栏默认禁网，所以这类命令要当场审批（而不是先失败再申请）----
    if cmd in ALWAYS_NETWORK:
        return Verdict(CONTAINED, f"{cmd} 需要网络", NEEDS_NETWORK)
    if cmd in NETWORK_SUBCOMMANDS:
        subs = [w for w in rest if not w.startswith("-")]
        # 裸 `yarn` / `npm`（不带子命令）等价于 install
        if (subs and subs[0] in NETWORK_SUBCOMMANDS[cmd]) or (not subs and cmd in ("npm", "yarn", "pnpm")):
            return Verdict(CONTAINED, f"{cmd} {subs[0] if subs else 'install'} 需要联网拉取依赖",
                           NEEDS_NETWORK)

    # ---- 具体规则：判据是"围栏兜不兜得住" ----
    if cmd == "rm":
        flags = [w for w in rest if w.startswith("-")]
        short = "".join(f[1:] for f in flags if not f.startswith("--"))
        recursive = "r" in short or "R" in short or "--recursive" in flags
        force = "f" in short or "--force" in flags
        if recursive and force:
            # 只能删到围栏允许的范围（工作区 + 临时区）→ contained，靠任务快照兜可逆性
            return Verdict(CONTAINED, "递归删除文件（围栏内，可回滚）")
        return None

    if cmd == "git":
        j = 0
        while j < len(rest):                       # 跳过 git 全局选项（-C <路径> 带值）
            if rest[j] in ("-C", "-c", "--git-dir", "--work-tree", "--namespace"):
                j += 2
                continue
            if rest[j].startswith("-"):
                j += 1
                continue
            break
        sub = rest[j] if j < len(rest) else ""
        subrest = rest[j + 1:]
        if sub == "push":
            # 远端副作用 + 凭据，围栏一概兜不住 → 必须审批（只需放行网络）
            if any(f.startswith("--force") or f == "-f" for f in subrest):
                return Verdict(UNCONTAINED, "git 强制推送到远程（远端副作用）", NEEDS_NETWORK)
            return Verdict(UNCONTAINED, "git 推送到远程（远端副作用）", NEEDS_NETWORK)
        if sub == "reset" and "--hard" in subrest:
            return Verdict(CONTAINED, "git 硬重置（丢失提交，可回滚）")
        if sub == "clean" and any(f.startswith("-") and not f.startswith("--") and "f" in f[1:]
                                  for f in subrest):
            return Verdict(CONTAINED, "git 清理未跟踪文件（围栏内）")
        return None

    if cmd == "chmod":
        args = [w for w in rest if not w.startswith("-")]
        if args and args[0] in ("777", "a+rwx"):
            return Verdict(CONTAINED, "开放所有权限（围栏内）")
        return None

    if cmd in ("shutdown", "reboot", "halt", "poweroff"):
        return Verdict(UNCONTAINED, "关机/重启（系统级，围栏兜不住）", NEEDS_FULL)
    if cmd.startswith("mkfs"):
        return Verdict(UNCONTAINED, "格式化磁盘（设备级，围栏兜不住）", NEEDS_FULL)
    if cmd == "dd" and any(w.startswith("of=/dev/") for w in rest):
        return Verdict(UNCONTAINED, "裸写块设备（设备级，围栏兜不住）", NEEDS_FULL)

    # `find … -exec rm -rf {} +`：-exec 之后是一段新的命令
    if cmd == "find":
        for j, w in enumerate(rest):
            if w in ("-exec", "-execdir", "-ok", "-okdir") and j + 1 < len(rest):
                return _simple_risk(rest[j + 1:], depth + 1)
        return None
    return None


def _fallback(text: str) -> Verdict | None:
    """无法解析时的保守兜底：沿用改造前的正则，按"必须问人"处理。"""
    for pattern, why in FALLBACK_PATTERNS:
        if re.search(pattern, text):
            return Verdict(UNCONTAINED, f"{why}（命令无法完整解析，按保守规则处理）")
    return None
