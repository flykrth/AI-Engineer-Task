"""Rich terminal dashboard and interactive REPL for WhatsApp project state and Q&A."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
import sys
from typing import Any

from google.genai import errors as genai_errors
import httpx
import openai
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.compiler import compile_project_state_with_meta
from src.llm import BaseLLMAdapter, LLMError, create_llm_adapter
from src.parser import ParsedChat, parse_chat_file
from src.qa import answer_and_verify
from src.retriever import HybridRetriever, RetrievalResult
from src.schemas import (
    DecisionRecord,
    DeadlineRecord,
    EvidenceRef,
    OpenItemRecord,
    ProjectState,
    TaskRecord,
    VerifiedAnswer,
)

REPL_PROMPT = "Ask a question (commands: state, rebuild, help, exit): "

_STATUS_STYLES: dict[str, str] = {
    "confirmed": "bold green",
    "completed": "bold green",
    "resolved": "bold green",
    "verified": "bold green",
    "assigned": "cyan",
    "in_progress": "bold cyan",
    "reversed": "bold yellow",
    "proposed": "yellow",
    "tentative": "yellow",
    "rule_based": "cyan",
    "uncertain": "bold magenta",
    "unconfirmed": "bold yellow",
    "deferred": "yellow",
    "open": "bold yellow",
    "unassigned": "bold red",
    "blocked": "bold red",
    "dropped": "dim",
    "superseded": "dim",
    "verified partial": "bold yellow",
    "abstained": "bold red",
}


def create_console(*, force_terminal: bool | None = None) -> Console:
    """Create a Rich console tuned for TTY display or unbroken non-TTY output."""
    is_tty = sys.stdout.isatty() if force_terminal is None else force_terminal
    return Console(
        file=sys.stdout,
        force_terminal=force_terminal,
        width=None if is_tty else 240,
        highlight=False,
        soft_wrap=not is_tty,
    )


def format_evidence_ref(ref: EvidenceRef) -> str:
    """Format an EvidenceRef as [msg_id | line_ref | timestamp_raw | sender]."""
    sender_label = ref.sender if ref.sender is not None else "system"
    line_ref = (
        f"L{ref.line_start}"
        if ref.line_start == ref.line_end
        else f"L{ref.line_start}-L{ref.line_end}"
    )
    return f"[{ref.msg_id} | {line_ref} | {ref.timestamp_raw} | {sender_label}]"


def _status_text(label: str) -> Text:
    style = _STATUS_STYLES.get(label, "bold white")
    return Text(f"[{label}]", style=style)


def _citation_text(refs: Sequence[EvidenceRef]) -> Text:
    if not refs:
        return Text("[no verified citations]", style="dim")
    text = Text()
    for idx, ref in enumerate(refs):
        if idx > 0:
            text.append("\n")
        text.append(format_evidence_ref(ref), style="bright_blue")
    return text


def _build_table(title: str, console: Console) -> Table:
    return Table(
        title=title,
        title_justify="left",
        title_style="bold",
        box=box.ROUNDED,
        show_lines=True,
        expand=console.is_terminal,
    )


def _render_decisions_section(console: Console, decisions: Sequence[DecisionRecord]) -> None:
    if not decisions:
        console.print(Text("1. Current decisions", style="bold"))
        console.print(Text("  (none)", style="dim"))
        return

    table = _build_table("1. Current decisions", console)
    table.add_column("Topic", style="bold", overflow="fold")
    table.add_column("Current decision", overflow="fold")
    table.add_column("Status", no_wrap=True)
    table.add_column("History", overflow="fold")
    table.add_column("Citations", overflow="fold", no_wrap=not console.is_terminal)

    for decision in decisions:
        topic_cell = Text.assemble(
            (decision.title, "bold"),
            "\n",
            (f"({decision.topic_key})", "dim"),
        )
        history_cell = Text("-", style="dim")
        if len(decision.history) > 1:
            prior_chain = " -> ".join(
                f"{entry.value} ({entry.action}, {entry.timestamp_raw})"
                for entry in decision.history[:-1]
            )
            history_cell = Text(f"Prior values: {prior_chain}")
        elif decision.history and decision.history[0].note:
            history_cell = Text(decision.history[0].note)

        table.add_row(
            topic_cell,
            Text(decision.current_value),
            _status_text(decision.status),
            history_cell,
            _citation_text(decision.evidence),
        )
    console.print(table)


def _render_tasks_section(console: Console, tasks: Sequence[TaskRecord]) -> None:
    if not tasks:
        console.print(Text("2. Tasks and ownership", style="bold"))
        console.print(Text("  (none)", style="dim"))
        return

    table = _build_table("2. Tasks and ownership", console)
    table.add_column("Task", style="bold", overflow="fold")
    table.add_column("Owner", overflow="fold")
    table.add_column("Status", no_wrap=True)
    table.add_column("Notes and history", overflow="fold")
    table.add_column("Citations", overflow="fold", no_wrap=not console.is_terminal)

    for task in tasks:
        task_cell = Text.assemble(
            (task.title, "bold"),
            "\n",
            (f"({task.task_key})", "dim"),
        )
        owner_label = task.current_owner or "unassigned"
        owner_cell = Text(
            owner_label,
            style="bold red" if task.current_owner is None else "",
        )

        detail_parts: list[str] = []
        if task.status_note:
            detail_parts.append(f"Note: {task.status_note}")
        if len(task.history) > 1:
            history_chain = " -> ".join(
                f"{entry.action} ({entry.owner or 'unassigned'}, {entry.timestamp_raw})"
                for entry in task.history
            )
            detail_parts.append(f"History: {history_chain}")
        details_cell = Text("\n".join(detail_parts)) if detail_parts else Text("-", style="dim")

        table.add_row(
            task_cell,
            owner_cell,
            _status_text(task.status),
            details_cell,
            _citation_text(task.evidence),
        )
    console.print(table)


def _render_deadlines_section(console: Console, deadlines: Sequence[DeadlineRecord]) -> None:
    if not deadlines:
        console.print(Text("3. Deadlines", style="bold"))
        console.print(Text("  (none)", style="dim"))
        return

    table = _build_table("3. Deadlines", console)
    table.add_column("Milestone", style="bold", overflow="fold")
    table.add_column("Current date", overflow="fold")
    table.add_column("Status", no_wrap=True)
    table.add_column("Rule and shifts", overflow="fold")
    table.add_column("Citations", overflow="fold", no_wrap=not console.is_terminal)

    for deadline in deadlines:
        milestone_cell = Text.assemble(
            (deadline.title, "bold"),
            "\n",
            (f"({deadline.item_key})", "dim"),
        )
        rule_parts: list[str] = []
        if deadline.dependency_rule:
            rule_parts.append(f"rule: {deadline.dependency_rule}")
        if len(deadline.history) > 1:
            prior_shifts = " -> ".join(
                f"{entry.value} ({entry.timestamp_raw})"
                for entry in deadline.history[:-1]
            )
            rule_parts.append(f"Prior shifts: {prior_shifts}")
        rule_cell = Text("\n".join(rule_parts)) if rule_parts else Text("-", style="dim")

        table.add_row(
            milestone_cell,
            Text(deadline.current_date_text),
            _status_text(deadline.status),
            rule_cell,
            _citation_text(deadline.evidence),
        )
    console.print(table)


def _collect_open_rows(
    state: ProjectState,
) -> list[tuple[str, str, str, str, list[EvidenceRef]]]:
    rows: list[tuple[str, str, str, str, list[EvidenceRef]]] = []
    item: OpenItemRecord
    for item in state.open_items:
        if item.status == "resolved":
            continue
        detail = item.description
        if item.owner:
            detail = f"{detail} (owner: {item.owner})"
        rows.append((f"issue:{item.status}", item.title, item.issue_key, detail, item.evidence))

    for task in state.tasks:
        if task.status in ("unassigned", "unconfirmed", "blocked"):
            owner_part = f"owner: {task.current_owner}" if task.current_owner else "unassigned"
            detail = f"{owner_part}; {task.status_note}" if task.status_note else owner_part
            rows.append((f"task:{task.status}", task.title, task.task_key, detail, task.evidence))

    for decision in state.decisions:
        if decision.status in ("uncertain", "proposed"):
            rows.append(
                (
                    f"decision:{decision.status}",
                    decision.title,
                    decision.topic_key,
                    decision.current_value,
                    decision.evidence,
                )
            )
    return rows


def _render_open_section(console: Console, state: ProjectState) -> None:
    rows = _collect_open_rows(state)
    if not rows:
        console.print(Text("4. Open, unassigned, or unconfirmed items", style="bold"))
        console.print(Text("  (none)", style="dim"))
        return

    table = _build_table("4. Open, unassigned, or unconfirmed items", console)
    table.add_column("Category", no_wrap=True)
    table.add_column("Item", style="bold", overflow="fold")
    table.add_column("Details", overflow="fold")
    table.add_column("Citations", overflow="fold", no_wrap=not console.is_terminal)

    for tag, title, key, detail, evidence in rows:
        item_cell = Text.assemble((title, "bold"), "\n", (f"({key})", "dim"))
        table.add_row(
            _status_text(tag),
            item_cell,
            Text(detail),
            _citation_text(evidence),
        )
    console.print(table)


def render_project_state(
    console: Console,
    state: ProjectState,
    adapter: BaseLLMAdapter,
    *,
    from_cache: bool,
    chat_path: Path | None = None,
) -> None:
    """Render the Rich 'State of the project' dashboard."""
    source_label = "cache" if from_cache else "compiled"
    resolved_chat = str(chat_path) if chat_path is not None else (state.source_path or "unknown")
    participants = ", ".join(state.participants) if state.participants else "none"

    header_body = Text()
    header_body.append(f"Chat file: {resolved_chat}\n")
    header_body.append(
        f"Provider: {adapter.provider_name} ({adapter.chat_model}) | Source: {source_label}\n"
    )
    header_body.append(
        f"Messages: {state.total_messages} | Date order: {state.date_order} | "
        f"Last message: {state.last_message_timestamp or 'n/a'}\n"
    )
    header_body.append(f"Participants: {participants}")

    console.print(
        Panel(
            header_body,
            title=Text("State of the project", style="bold"),
            title_align="left",
            border_style="cyan",
            expand=console.is_terminal,
        )
    )
    _render_decisions_section(console, state.decisions)
    _render_tasks_section(console, state.tasks)
    _render_deadlines_section(console, state.deadlines)
    _render_open_section(console, state)


def _answer_status_label(result: VerifiedAnswer) -> str:
    if not result.supported:
        return "abstained"
    if result.partial_status_only:
        return "verified partial"
    return "verified"


def render_verified_answer(
    console: Console,
    question: str,
    chat: ParsedChat,
    result: VerifiedAnswer,
    adapter: BaseLLMAdapter,
    *,
    retrieval: RetrievalResult | None = None,
) -> None:
    """Render a verified Q&A response panel and cited evidence table."""
    status_label = _answer_status_label(result)
    border_style = (
        "red" if status_label == "abstained" else "yellow" if status_label == "verified partial" else "green"
    )

    meta = Text(
        f"Provider: {adapter.provider_name} ({adapter.chat_model}) | "
        f"Chat lines: {chat.total_source_lines} | Messages: {len(chat.messages)} | "
        f"Date order: {chat.date_order}",
        style="dim",
    )
    if retrieval is not None:
        retrieval_mode = "hybrid (bm25 + embeddings)" if retrieval.used_dense else "bm25 fallback"
        meta.append(f" | Retrieval: {retrieval_mode}")
    console.print(meta)

    body = Text()
    body.append(f"Question: {question}\n")
    body.append(f"Answer: {result.answer}\n")
    body.append("Status: ")
    body.append_text(_status_text(status_label))
    body.append(f" (verification_status: {result.verification_status})")
    if result.rejected_msg_ids:
        body.append(
            f"\nFiltered unverified IDs: {', '.join(result.rejected_msg_ids)}",
            style="dim",
        )

    console.print(
        Panel(
            body,
            title=Text(f"Answer [{status_label}]", style="bold"),
            title_align="left",
            border_style=border_style,
            expand=console.is_terminal,
        )
    )

    if result.evidence:
        console.print(Text("Evidence:", style="bold"))
        table = Table(
            box=box.SIMPLE_HEAVY,
            show_lines=True,
            expand=console.is_terminal,
        )
        table.add_column("Citation", overflow="fold", no_wrap=not console.is_terminal)
        table.add_column("Quote", overflow="fold")
        for ref in result.evidence:
            table.add_row(
                Text(format_evidence_ref(ref), style="bright_blue"),
                Text(f'"{ref.quote}"'),
            )
        console.print(table)
    else:
        console.print(Text("Evidence: none (abstained)", style="dim"))


def render_help(console: Console) -> None:
    """Render available interactive REPL commands."""
    help_text = Text()
    help_text.append("Enter any question in English or Hinglish to query the chat.\n")
    help_text.append("Commands:\n", style="bold")
    help_text.append("  state    Re-render the current State of the project dashboard\n")
    help_text.append("  rebuild  Force recompiling ProjectState and re-indexing embeddings\n")
    help_text.append("  help     Show this command summary\n")
    help_text.append("  exit     Quit the interactive session (also quit, Ctrl+C, or EOF)")
    console.print(
        Panel(
            help_text,
            title=Text("Interactive commands", style="bold"),
            title_align="left",
            border_style="blue",
            expand=console.is_terminal,
        )
    )


def run_interactive_session(
    chat_path: Path,
    *,
    provider: str | None = None,
    rebuild: bool = False,
    adapter_factory: Any = create_llm_adapter,
    cache_dir: Path | None = None,
    console: Console | None = None,
    input_fn: Callable[[str], str] | None = None,
) -> int:
    """Run the default Rich interactive CLI session for a WhatsApp chat export."""
    active_console = console if console is not None else create_console()
    prompt_fn = input_fn if input_fn is not None else input

    try:
        chat = parse_chat_file(chat_path)
        raw_text = chat_path.read_text(encoding="utf-8")
        adapter = adapter_factory(provider=provider)
        state, from_cache, _ = compile_project_state_with_meta(
            chat,
            adapter,
            raw_text=raw_text,
            cache_dir=cache_dir,
            rebuild=rebuild,
        )
        retriever = HybridRetriever(
            chat,
            adapter,
            raw_text=raw_text,
            cache_dir=cache_dir,
            rebuild=rebuild,
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

    render_project_state(
        active_console,
        state,
        adapter,
        from_cache=from_cache,
        chat_path=chat_path,
    )
    if retriever.dense_fallback:
        reason = retriever.dense_fallback_reason or "embedding provider unavailable"
        active_console.print(
            Text(
                f"[notice] Dense embeddings unavailable ({reason}); using BM25 lexical fallback.",
                style="yellow",
            )
        )

    while True:
        try:
            raw_input_line = prompt_fn(REPL_PROMPT)
        except (EOFError, KeyboardInterrupt):
            active_console.print(Text("\nExiting interactive session.", style="dim"))
            return 0

        if input_fn is None and not sys.stdin.isatty():
            active_console.print()

        command = raw_input_line.strip()
        if not command:
            continue

        lowered = command.lower()
        if lowered in ("exit", "quit", ":q", "/exit", "/quit"):
            active_console.print(Text("Exiting interactive session.", style="dim"))
            return 0

        if lowered in ("help", "?", "/help"):
            render_help(active_console)
            continue

        if lowered == "state":
            render_project_state(
                active_console,
                state,
                adapter,
                from_cache=from_cache,
                chat_path=chat_path,
            )
            continue

        if lowered == "rebuild":
            try:
                chat = parse_chat_file(chat_path)
                raw_text = chat_path.read_text(encoding="utf-8")
                state, from_cache, _ = compile_project_state_with_meta(
                    chat,
                    adapter,
                    raw_text=raw_text,
                    cache_dir=cache_dir,
                    rebuild=True,
                )
                retriever = HybridRetriever(
                    chat,
                    adapter,
                    raw_text=raw_text,
                    cache_dir=cache_dir,
                    rebuild=True,
                )
            except (
                FileNotFoundError,
                ValueError,
                LLMError,
                genai_errors.APIError,
                openai.OpenAIError,
                httpx.HTTPError,
            ) as exc:
                active_console.print(Text(f"[error] Rebuild failed: {exc}", style="bold red"))
                continue

            render_project_state(
                active_console,
                state,
                adapter,
                from_cache=from_cache,
                chat_path=chat_path,
            )
            active_console.print(
                Text("Project state and retrieval index rebuilt.", style="green")
            )
            continue

        try:
            retrieval = retriever.retrieve(command)
            result = answer_and_verify(command, chat, state, retrieval, adapter)
        except (
            ValueError,
            LLMError,
            genai_errors.APIError,
            openai.OpenAIError,
            httpx.HTTPError,
        ) as exc:
            active_console.print(Text(f"[error] Query failed: {exc}", style="bold red"))
            continue

        render_verified_answer(
            active_console,
            command,
            chat,
            result,
            adapter,
            retrieval=retrieval,
        )
