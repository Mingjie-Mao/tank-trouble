#!/usr/bin/env python3
"""Export visible Claude Code dialogue from local project JSONL files."""

import argparse
import datetime as dt
import hashlib
import html
import json
import os
import re
from pathlib import Path


PROJECT_SLUG = "-Users-cichlidfish-tank-trouble"
SKIP_USER_PREFIXES = (
    "<local-command-caveat>",
    "<local-command-stdout>",
    "<command-name>",
    "<system-reminder>",
)


def _timestamp(value):
    if not value:
        return "时间未知"
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    return parsed.astimezone(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _tag(text, name):
    match = re.search(rf"<{name}>(.*?)</{name}>", text, re.DOTALL)
    return html.unescape(match.group(1).strip()) if match else ""


def _notification_text(text):
    parts = []
    summary = _tag(text, "summary")
    event = _tag(text, "event")
    output_file = _tag(text, "output-file")
    if summary:
        parts.append(f"**任务：** {summary}")
    if event:
        parts.append(f"**结果：**\n\n{event}")
    if output_file:
        parts.append(f"**输出文件：** `{output_file}`")
    return "\n\n".join(parts) or text.strip()


def _text_blocks(content):
    if isinstance(content, str):
        return [content]
    if not isinstance(content, list):
        return []
    return [
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]


def read_session(path):
    entries = []
    seen = set()
    first_timestamp = None
    last_timestamp = None
    line_count = 0
    counts = {"user": 0, "assistant": 0, "notification": 0, "event": 0}

    with path.open(encoding="utf-8") as source:
        for line in source:
            line_count += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            timestamp = record.get("timestamp")
            if timestamp:
                first_timestamp = first_timestamp or timestamp
                last_timestamp = timestamp
            message = record.get("message") or {}
            role = message.get("role")
            content = message.get("content")

            if role == "user":
                for text in _text_blocks(content):
                    text = text.strip()
                    if not text or text.startswith(SKIP_USER_PREFIXES):
                        continue
                    if text.startswith("<task-notification>"):
                        kind = "notification"
                        text = _notification_text(text)
                    elif text.startswith("[Request interrupted by user]"):
                        kind = "event"
                    elif text.startswith("<"):
                        continue
                    else:
                        kind = "user"
                    key = (kind, timestamp, text)
                    if key not in seen:
                        seen.add(key)
                        entries.append((timestamp or "", kind, text))
                        counts[kind] += 1

            elif role == "assistant":
                for text in _text_blocks(content):
                    text = text.strip()
                    if not text:
                        continue
                    key = ("assistant", timestamp, text)
                    if key not in seen:
                        seen.add(key)
                        entries.append((timestamp or "", "assistant", text))
                        counts["assistant"] += 1

    entries.sort(key=lambda item: item[0])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "path": path,
        "entries": entries,
        "counts": counts,
        "line_count": line_count,
        "byte_count": path.stat().st_size,
        "sha256": digest,
        "first_timestamp": first_timestamp,
        "last_timestamp": last_timestamp,
    }


def render(sessions, output):
    generated = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    lines = [
        "# Claude Code 完整可见对话上下文（Tank Trouble）",
        "",
        f"> 生成时间：{generated}",
        "> 来源：本机 Claude Code 项目 JSONL；按 UTC 时间顺序导出。",
        "",
        "## 使用方式",
        "",
        "- 新 agent 应先读 `docs/HANDOFF_COMPLETE_CONTEXT.md` 获取当前结论和最新进度。",
        "- 需要追溯某项设计为什么出现、用户原话或失败实验讨论时，再搜索本文件。",
        "- 根目录旧文件 `restored_session_20260705-0708.md` 只覆盖第一份会话，可留作历史校验，",
        "  但不再作为主要恢复入口。",
        "",
        "## 保留与排除范围",
        "",
        "本文件保留全部可见用户消息、Claude 最终文字回复、请求中断事件，以及后台任务通知中的",
        "任务摘要和实验结果。为避免噪声和泄露无关机器细节，不包含 Claude 的内部 `thinking`、",
        "原始工具调用参数、完整工具返回、队列元数据、附件元数据、标题生成记录和 `/model` 等本地命令。",
        "这些原始信息仍保存在下列 JSONL 中，可按 SHA-256 校验。",
        "",
        "## 原始来源清单",
        "",
        "| 会话 | 时间范围 | 原始大小 | 可见用户 | Agent 回复 | 后台通知 | SHA-256 |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]

    for index, session in enumerate(sessions, 1):
        counts = session["counts"]
        start = _timestamp(session["first_timestamp"])
        end = _timestamp(session["last_timestamp"])
        path = f"~/.claude/projects/{PROJECT_SLUG}/{session['path'].name}"
        lines.append(
            f"| {index}：`{path}` | {start} — {end} | "
            f"{session['byte_count'] / 1024 / 1024:.1f} MiB | "
            f"{counts['user']} | {counts['assistant']} | "
            f"{counts['notification']} | `{session['sha256']}` |"
        )

    labels = {
        "user": "👤 用户",
        "assistant": "🤖 Claude Code",
        "notification": "⚙️ 后台任务通知",
        "event": "⏸️ 会话事件",
    }
    for index, session in enumerate(sessions, 1):
        lines.extend([
            "",
            "---",
            "",
            f"# 会话 {index}：{session['path'].stem}",
            "",
            f"原始时间范围：{_timestamp(session['first_timestamp'])} — "
            f"{_timestamp(session['last_timestamp'])}",
        ])
        for timestamp, kind, text in session["entries"]:
            lines.extend([
                "",
                "---",
                "",
                f"## {labels[kind]}（{_timestamp(timestamp)}）",
                "",
                text,
            ])

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path.home() / ".claude" / "projects" / PROJECT_SLUG,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "docs"
        / "CLAUDE_CODE_COMPLETE_CONTEXT.md",
    )
    args = parser.parse_args()
    paths = sorted(
        args.source_dir.glob("*.jsonl"),
        key=lambda path: path.stat().st_mtime,
    )
    if not paths:
        raise SystemExit(f"没有找到 Claude Code 会话：{args.source_dir}")
    sessions = sorted(
        (read_session(path) for path in paths),
        key=lambda item: item["first_timestamp"] or "",
    )
    render(sessions, args.output)
    total_entries = sum(len(session["entries"]) for session in sessions)
    print(f"已生成 {args.output}：{len(sessions)} 个会话，{total_entries} 条可见记录")


if __name__ == "__main__":
    main()
