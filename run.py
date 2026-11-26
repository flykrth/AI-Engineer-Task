"""CLI entrypoint for temporal WhatsApp team chat state compilation and grounded Q&A."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from google.genai import errors as genai_errors
import httpx
import openai
from pydantic import BaseModel, Field

from src.compiler import compile_project_state_with_meta
from src.llm import BaseLLMAdapter, LLMError, create_llm_adapter
from src.parser import ParsedChat, parse_chat_file
from src.schemas import EvidenceRef, ProjectState

_SYSTEM_PROMPT = """You answer questions about an exported WhatsApp group chat.

Rules:
1. Read all messages in strict chronological order from start to finish. Later decisions, reassignments, and deadline corrections override earlier ones.
2. Answer in clear English or Hinglish matching the question and chat context.
3. Base every claim strictly on the provided transcript lines. Do not invent facts or assume unconfirmed future plans or promises were completed.
4. If the chat never settled the answer, if the relevant message was deleted, or if the topic is not in the chat, set supported=false and set answer to "the chat doesn't say".
5. For every supported claim, include exact evidence entries using the message's msg_id, line_start, line_end, timestamp_raw, sender, and a short exact quote."""


class EvidenceItem(BaseModel):
    """Exact transcript citation supporting an answer."""

    msg_id: str
    line_start: int
    line_end: int
    timestamp_raw: str
    sender: str | None = None
    quote: str


class Slice1Answer(BaseModel):
    """Structured answer returned by Slice 1 Q&A."""

    answer: str
    supported: bool
    evidence: list[EvidenceItem] = Field(default_factory=list)


def format_transcript(chat: ParsedChat) -> str:
    """Format parsed messages chronologically with explicit ID and line metadata."""
    lines: list[str] = []
    for msg in chat.messages:
        sender_label = msg.sender if msg.sender is not None else "system"
        header = (
            f"[{msg.msg_id} | L{msg.line_start}-L{msg.line_end} | "
            f"{msg.timestamp_raw} | {sender_label}]"
        )
        suffix = " [edited]" if msg.is_edited else ""
        if msg.kind == "deleted":
            suffix = f"{suffix} [deleted]"
        lines.append(f"{header} {msg.text}{suffix}")
    return "\n".join(lines)


def answer_question(
    chat: ParsedChat,
    question: str,
    adapter: BaseLLMAdapter,
) -> Slice1Answer:
    """Prompt the configured LLM adapter to answer one question over the parsed chat."""
    transcript = format_transcript(chat)
    prompt = (
        f"Parsed WhatsApp chat ({len(chat.messages)} messages, "
        f"detected date order {chat.date_order}):\n\n"
        f"{transcript}\n\n"
        f"Question: {question}\n"
    )
    result = adapter.generate_structured(
        prompt,
        Slice1Answer,
        system_prompt=_SYSTEM_PROMPT,
        temperature=0.0,
    )
    if not result.supported and "the chat doesn't say" not in result.answer.lower():
        return result.model_copy(update={"answer": "the chat doesn't say"})
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Compile temporal project state and answer grounded questions about an exported WhatsApp group chat."
    )
    parser.add_argument(
        "--chat",
        type=Path,
        required=True,
        help="Path to the exported WhatsApp .txt chat file.",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Question to answer from the chat.",
    )
    parser.add_argument(
        "--state-only",
        action="store_true",
        help="Compile and display the current project state without answering a question.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force recompiling ProjectState even if a cached version exists.",
    )
    parser.add_argument(
        "--provider",
        choices=("gemini", "openai", "ollama"),
        default=None,
        help="Optional LLM provider override (defaults to LLM_PROVIDER in .env).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Print structured JSON output instead of formatted text.",
    )
    return parser


def _format_evidence_ref(ref: EvidenceRef) -> str:
    """Format an EvidenceRef as [msg_id | line_ref | timestamp_raw | sender]."""
    sender_label = ref.sender if ref.sender is not None else "system"
    line_ref = (
        f"L{ref.line_start}"
        if ref.line_start == ref.line_end
        else f"L{ref.line_start}-L{ref.line_end}"
    )
    return f"[{ref.msg_id} | {line_ref} | {ref.timestamp_raw} | {sender_label}]"


def _format_citation_list(refs: list[EvidenceRef]) -> str:
    if not refs:
        return "[no verified citations]"
    return ", ".join(_format_evidence_ref(ref) for ref in refs)


def format_state_text(
    state: ProjectState,
    adapter: BaseLLMAdapter,
    *,
    from_cache: bool,
) -> str:
    """Render a readable plain-text 'State of the project' summary grouped by section."""
    source_label = "cache" if from_cache else "compiled"
    parts = [
        "State of the project",
        f"Provider: {adapter.provider_name} ({adapter.chat_model}) | Source: {source_label}",
        (
            f"Messages: {state.total_messages} | Date order: {state.date_order} | "
            f"Last message: {state.last_message_timestamp or 'n/a'}"
        ),
        (
            "Participants: "
            + (", ".join(state.participants) if state.participants else "none")
        ),
        "",
        "1. Current decisions",
    ]

    if not state.decisions:
        parts.append("  (none)")
    else:
        for decision in state.decisions:
            parts.append(
                f"  - {decision.title} ({decision.topic_key}): "
                f"{decision.current_value} [{decision.status}]"
            )
            if len(decision.history) > 1:
                prior_chain = " -> ".join(
                    f"{entry.value} ({entry.action}, {entry.timestamp_raw})"
                    for entry in decision.history[:-1]
                )
                parts.append(f"    Prior values: {prior_chain}")
            parts.append(f"    Citations: {_format_citation_list(decision.evidence)}")

    parts.extend(["", "2. Tasks and ownership"])
    if not state.tasks:
        parts.append("  (none)")
    else:
        for task in state.tasks:
            owner_label = task.current_owner or "unassigned"
            note_part = f" | note: {task.status_note}" if task.status_note else ""
            parts.append(
                f"  - {task.title} ({task.task_key}): "
                f"owner={owner_label} | status={task.status}{note_part}"
            )
            if len(task.history) > 1:
                history_chain = " -> ".join(
                    f"{entry.action} ({entry.owner or 'unassigned'}, {entry.timestamp_raw})"
                    for entry in task.history
                )
                parts.append(f"    History: {history_chain}")
            parts.append(f"    Citations: {_format_citation_list(task.evidence)}")

    parts.extend(["", "3. Deadlines"])
    if not state.deadlines:
        parts.append("  (none)")
    else:
        for deadline in state.deadlines:
            rule_part = (
                f" | rule: {deadline.dependency_rule}"
                if deadline.dependency_rule
                else ""
            )
            parts.append(
                f"  - {deadline.title} ({deadline.item_key}): "
                f"{deadline.current_date_text} [{deadline.status}]{rule_part}"
            )
            if len(deadline.history) > 1:
                prior_shifts = " -> ".join(
                    f"{entry.value} ({entry.timestamp_raw})"
                    for entry in deadline.history[:-1]
                )
                parts.append(f"    Prior shifts: {prior_shifts}")
            parts.append(f"    Citations: {_format_citation_list(deadline.evidence)}")

    parts.extend(["", "4. Open, unassigned, or unconfirmed items"])
    open_lines: list[str] = []
    for item in state.open_items:
        if item.status == "resolved":
            continue
        owner_part = f" | owner: {item.owner}" if item.owner else ""
        open_lines.append(
            f"  - [issue:{item.status}] {item.title} ({item.issue_key}): "
            f"{item.description}{owner_part}"
        )
        open_lines.append(f"    Citations: {_format_citation_list(item.evidence)}")

    for task in state.tasks:
        if task.status in ("unassigned", "unconfirmed", "blocked"):
            owner_part = f"owner={task.current_owner}" if task.current_owner else "unassigned"
            note_part = f" | {task.status_note}" if task.status_note else ""
            open_lines.append(
                f"  - [task:{task.status}] {task.title} ({task.task_key}): "
                f"{owner_part}{note_part}"
            )
            open_lines.append(f"    Citations: {_format_citation_list(task.evidence)}")

    for decision in state.decisions:
        if decision.status in ("uncertain", "proposed"):
            open_lines.append(
                f"  - [decision:{decision.status}] {decision.title} ({decision.topic_key}): "
                f"{decision.current_value}"
            )
            open_lines.append(f"    Citations: {_format_citation_list(decision.evidence)}")

    if not open_lines:
        parts.append("  (none)")
    else:
        parts.extend(open_lines)

    return "\n".join(parts)


def _format_text_output(
    question: str,
    chat: ParsedChat,
    result: Slice1Answer,
    adapter: BaseLLMAdapter,
) -> str:
    """Render a concise plain-text answer and evidence block."""
    parts = [
        f"Provider: {adapter.provider_name} ({adapter.chat_model})",
        f"Chat lines: {chat.total_source_lines} | Messages: {len(chat.messages)} | Date order: {chat.date_order}",
        f"Question: {question}",
        f"Answer: {result.answer}",
    ]
    if result.evidence:
        parts.append("Evidence:")
        for item in result.evidence:
            sender_label = item.sender if item.sender is not None else "system"
            line_ref = (
                f"L{item.line_start}"
                if item.line_start == item.line_end
                else f"L{item.line_start}-L{item.line_end}"
            )
            parts.append(
                f"  - [{item.msg_id} | {line_ref} | {item.timestamp_raw} | {sender_label}] "
                f'"{item.quote}"'
            )
    return "\n".join(parts)


def run_cli(
    argv: list[str] | None = None,
    *,
    adapter_factory: Any = create_llm_adapter,
    cache_dir: Path | None = None,
) -> int:
    """Execute the CLI workflow and return an exit status code."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    has_question = bool(args.question and args.question.strip())
    if not args.state_only and not has_question:
        print(
            "Error: either --question or --state-only is required.",
            file=sys.stderr,
        )
        return 2

    try:
        chat = parse_chat_file(args.chat)
        adapter = adapter_factory(provider=args.provider)
        if args.state_only:
            raw_text = args.chat.read_text(encoding="utf-8")
            state, from_cache, _ = compile_project_state_with_meta(
                chat,
                adapter,
                raw_text=raw_text,
                cache_dir=cache_dir,
                rebuild=args.rebuild,
            )
            if args.as_json:
                print(state.model_dump_json(indent=2))
            else:
                print(format_state_text(state, adapter, from_cache=from_cache))
            return 0

        result = answer_question(chat, args.question, adapter)
    except (
        FileNotFoundError,
        ValueError,
        LLMError,
        genai_errors.APIError,
        openai.OpenAIError,
        httpx.HTTPError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.as_json:
        payload = {
            "question": args.question,
            "chat_file": str(args.chat),
            "provider": adapter.provider_name,
            "model": adapter.chat_model,
            "date_order": chat.date_order,
            "total_source_lines": chat.total_source_lines,
            "total_messages": len(chat.messages),
            "answer": result.answer,
            "supported": result.supported,
            "evidence": [item.model_dump() for item in result.evidence],
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(_format_text_output(args.question, chat, result, adapter))
    return 0


def main() -> int:
    """CLI entrypoint."""
    return run_cli()


if __name__ == "__main__":
    raise SystemExit(main())
