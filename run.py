"""Slice 1 CLI entrypoint for grounded WhatsApp team chat Q&A."""

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

from src.llm import BaseLLMAdapter, LLMError, create_llm_adapter
from src.parser import ParsedChat, parse_chat_file

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
    """Create the Slice 1 CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Ask grounded questions about an exported WhatsApp group chat."
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
        required=True,
        help="Question to answer from the chat.",
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
) -> int:
    """Execute the Slice 1 CLI workflow and return an exit status code."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        chat = parse_chat_file(args.chat)
        adapter = adapter_factory(provider=args.provider)
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
