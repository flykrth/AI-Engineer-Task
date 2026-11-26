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
import uvicorn

from src.cli import (
    create_console,
    render_project_state,
    render_verified_answer,
    run_interactive_session,
)
from src.compiler import compile_project_state_with_meta
from src.llm import LLMError, create_llm_adapter
from src.parser import parse_chat_file
from src.qa import answer_and_verify
from src.retriever import HybridRetriever


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Compile temporal project state and answer grounded questions "
            "about an exported WhatsApp group chat."
        )
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
        help="Question to answer from the chat (skips interactive mode).",
    )
    parser.add_argument(
        "--state-only",
        action="store_true",
        help="Compile and display the current project state without entering interactive mode.",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Launch the FastAPI single-page web dashboard.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host interface to bind when running with --web (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind when running with --web (default: 8000).",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuilding cached ProjectState and retrieval embeddings even if cached versions exist.",
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
        help="Print structured JSON output for --question or --state-only.",
    )
    return parser


def _launch_web_dashboard(
    chat_path: Path,
    *,
    host: str,
    port: int,
    provider: str | None,
    rebuild: bool,
    cache_dir: Path | None,
    adapter_factory: Any,
) -> None:
    from src.web.app import create_app

    app = create_app(
        chat_path,
        provider=provider,
        rebuild=rebuild,
        cache_dir=cache_dir,
        adapter_factory=adapter_factory,
    )
    uvicorn.run(app, host=host, port=port)


def run_cli(
    argv: list[str] | None = None,
    *,
    adapter_factory: Any = create_llm_adapter,
    cache_dir: Path | None = None,
    input_fn: Any = None,
) -> int:
    """Execute the CLI workflow and return an exit status code."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    has_question_arg = args.question is not None
    if has_question_arg and not args.question.strip():
        print("Error: --question must not be empty.", file=sys.stderr)
        return 2

    if args.web and (args.state_only or has_question_arg or args.as_json):
        print(
            "Error: --web cannot be combined with --question, --state-only, or --json.",
            file=sys.stderr,
        )
        return 2

    if args.state_only and has_question_arg:
        print(
            "Error: --state-only cannot be combined with --question.",
            file=sys.stderr,
        )
        return 2

    if not args.state_only and not has_question_arg and not args.web:
        if args.as_json:
            print(
                "Error: --json requires --question or --state-only.",
                file=sys.stderr,
            )
            return 2
        return run_interactive_session(
            args.chat,
            provider=args.provider,
            rebuild=args.rebuild,
            adapter_factory=adapter_factory,
            cache_dir=cache_dir,
            input_fn=input_fn,
        )

    if args.web:
        if not 1 <= args.port <= 65535:
            print("Error: --port must be between 1 and 65535.", file=sys.stderr)
            return 2
        try:
            _launch_web_dashboard(
                args.chat,
                host=args.host,
                port=args.port,
                provider=args.provider,
                rebuild=args.rebuild,
                cache_dir=cache_dir,
                adapter_factory=adapter_factory,
            )
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
        return 0

    try:
        chat = parse_chat_file(args.chat)
        raw_text = args.chat.read_text(encoding="utf-8")
        adapter = adapter_factory(provider=args.provider)
        state, from_cache, _ = compile_project_state_with_meta(
            chat,
            adapter,
            raw_text=raw_text,
            cache_dir=cache_dir,
            rebuild=args.rebuild,
        )
        if args.state_only:
            if args.as_json:
                print(state.model_dump_json(indent=2))
            else:
                render_project_state(
                    create_console(),
                    state,
                    adapter,
                    from_cache=from_cache,
                    chat_path=args.chat,
                )
            return 0

        assert args.question is not None
        retriever = HybridRetriever(
            chat,
            adapter,
            raw_text=raw_text,
            cache_dir=cache_dir,
            rebuild=args.rebuild,
        )
        retrieval = retriever.retrieve(args.question)
        result = answer_and_verify(
            args.question,
            chat,
            state,
            retrieval,
            adapter,
        )
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
            "partial_status_only": result.partial_status_only,
            "verification_status": result.verification_status,
            "rejected_msg_ids": result.rejected_msg_ids,
            "evidence": [item.model_dump() for item in result.evidence],
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        render_verified_answer(
            create_console(),
            args.question,
            chat,
            result,
            adapter,
            retrieval=retrieval,
        )
    return 0


def main() -> int:
    """CLI entrypoint."""
    return run_cli()


if __name__ == "__main__":
    raise SystemExit(main())
