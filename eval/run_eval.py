"""Reproducible evaluation runner for the temporal chat Q&A pipeline."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.compiler import compile_project_state_with_meta
from src.llm import BaseLLMAdapter, create_llm_adapter
from src.parser import ParsedChat, ParsedMessage, parse_chat_file
from src.qa import answer_and_verify
from src.retriever import HybridRetriever, RetrievalResult
from src.schemas import EvidenceRef, ProjectState, VerifiedAnswer
from src.verifier import ABSTENTION_ANSWER


def _resolve_repo_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _normalize_path_key(path: Path) -> str:
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def check_evidence_refs(
    refs: Sequence[EvidenceRef],
    msg_by_id: dict[str, ParsedMessage],
    *,
    context_label: str,
) -> list[str]:
    """Verify that every EvidenceRef matches an actual ParsedMessage record."""
    errors: list[str] = []
    for ref in refs:
        msg = msg_by_id.get(ref.msg_id)
        if msg is None:
            errors.append(f"{context_label}: unknown msg_id '{ref.msg_id}'")
            continue
        if ref.line_start != msg.line_start or ref.line_end != msg.line_end:
            errors.append(
                f"{context_label}: line mismatch for {ref.msg_id} "
                f"({ref.line_start}-{ref.line_end} != {msg.line_start}-{msg.line_end})"
            )
        if ref.timestamp_raw != msg.timestamp_raw or ref.timestamp_iso != msg.timestamp_iso:
            errors.append(
                f"{context_label}: timestamp mismatch for {ref.msg_id} "
                f"({ref.timestamp_raw!r} != {msg.timestamp_raw!r})"
            )
        if ref.sender != msg.sender:
            errors.append(
                f"{context_label}: sender mismatch for {ref.msg_id} "
                f"({ref.sender!r} != {msg.sender!r})"
            )
        if ref.quote != msg.text:
            errors.append(f"{context_label}: quote mismatch for {ref.msg_id}")
    return errors


def check_project_state_invariants(
    state: ProjectState, chat: ParsedChat
) -> list[str]:
    """Verify all citations inside ProjectState and its history[] against ParsedChat."""
    msg_by_id = {m.msg_id: m for m in chat.messages}
    errors: list[str] = []

    for dec in state.decisions:
        errors.extend(
            check_evidence_refs(
                dec.evidence, msg_by_id, context_label=f"decision:{dec.topic_key}"
            )
        )
        for idx, entry in enumerate(dec.history):
            errors.extend(
                check_evidence_refs(
                    entry.evidence,
                    msg_by_id,
                    context_label=f"decision:{dec.topic_key}:history[{idx}]",
                )
            )

    for task in state.tasks:
        errors.extend(
            check_evidence_refs(
                task.evidence, msg_by_id, context_label=f"task:{task.task_key}"
            )
        )
        for idx, entry in enumerate(task.history):
            errors.extend(
                check_evidence_refs(
                    entry.evidence,
                    msg_by_id,
                    context_label=f"task:{task.task_key}:history[{idx}]",
                )
            )

    for dl in state.deadlines:
        errors.extend(
            check_evidence_refs(
                dl.evidence, msg_by_id, context_label=f"deadline:{dl.item_key}"
            )
        )
        for idx, entry in enumerate(dl.history):
            errors.extend(
                check_evidence_refs(
                    entry.evidence,
                    msg_by_id,
                    context_label=f"deadline:{dl.item_key}:history[{idx}]",
                )
            )

    for item in state.open_items:
        errors.extend(
            check_evidence_refs(
                item.evidence, msg_by_id, context_label=f"open_item:{item.issue_key}"
            )
        )
        for idx, entry in enumerate(item.history):
            errors.extend(
                check_evidence_refs(
                    entry.evidence,
                    msg_by_id,
                    context_label=f"open_item:{item.issue_key}:history[{idx}]",
                )
            )

    return errors


def check_answer_invariants(
    result: VerifiedAnswer, chat: ParsedChat
) -> list[str]:
    """Check deterministic citation and abstention invariants on a VerifiedAnswer."""
    msg_by_id = {m.msg_id: m for m in chat.messages}
    errors = check_evidence_refs(result.evidence, msg_by_id, context_label="answer")

    if result.supported:
        if not result.evidence:
            errors.append("supported=True but evidence is empty")
        else:
            kinds = [msg_by_id[ref.msg_id].kind for ref in result.evidence if ref.msg_id in msg_by_id]
            if kinds and all(kind == "deleted" for kind in kinds):
                errors.append("supported=True but all cited messages are deleted placeholders")
    else:
        if result.answer != ABSTENTION_ANSWER:
            errors.append(
                f"supported=False but answer is {result.answer!r} instead of {ABSTENTION_ANSWER!r}"
            )
        if result.evidence:
            errors.append("supported=False but evidence is non-empty")

    if not result.verification_status.strip():
        errors.append("verification_status is empty")

    return errors


def evaluate_fact_hints(
    item: dict[str, Any], result: VerifiedAnswer
) -> dict[str, Any]:
    """Compute deterministic pre-audit substring and citation hints (aid only)."""
    lowered = result.answer.lower()
    expected_supported = item.get("expected_supported")
    expected_partial = item.get("expected_partial_status_only")

    fact_groups: list[list[str]] = list(item.get("fact_hint_groups") or [])
    matched_groups: list[list[str]] = []
    missing_groups: list[list[str]] = []
    for group in fact_groups:
        if any(term.lower() in lowered for term in group):
            matched_groups.append(group)
        else:
            missing_groups.append(group)

    forbidden_terms: list[str] = list(item.get("forbidden_substrings") or [])
    forbidden_hits = [term for term in forbidden_terms if term.lower() in lowered]

    returned_ids = [ref.msg_id for ref in result.evidence]
    expected_ids_any: list[str] = list(item.get("expected_msg_ids_any") or [])
    matched_expected_ids = [mid for mid in returned_ids if mid in expected_ids_any]

    supported_ok = (
        expected_supported is None or result.supported == bool(expected_supported)
    )
    partial_ok = (
        expected_partial is None
        or result.partial_status_only == bool(expected_partial)
    )
    citation_overlap_ok = (
        not expected_ids_any or len(matched_expected_ids) > 0
    )

    all_hints_pass = (
        supported_ok
        and not missing_groups
        and not forbidden_hits
        and (citation_overlap_ok if result.supported else True)
    )

    return {
        "supported_matches_expected": supported_ok,
        "partial_matches_expected": partial_ok,
        "fact_groups_matched": len(matched_groups),
        "fact_groups_total": len(fact_groups),
        "missing_fact_groups": missing_groups,
        "forbidden_hits": forbidden_hits,
        "expected_msg_ids_any": expected_ids_any,
        "matched_expected_msg_ids": matched_expected_ids,
        "citation_overlap_ok": citation_overlap_ok,
        "all_hints_pass": all_hints_pass,
    }


def _summarize_retrieval(retrieval: RetrievalResult) -> dict[str, Any]:
    return {
        "used_dense": retrieval.used_dense,
        "dense_fallback": retrieval.dense_fallback,
        "dense_fallback_reason": retrieval.dense_fallback_reason,
        "cache_hit": retrieval.cache_hit,
        "top_hits": [
            {
                "chunk_index": hit.chunk_index,
                "start_idx": hit.start_idx,
                "end_idx": hit.end_idx,
                "rrf_score": round(hit.rrf_score, 6),
                "bm25_rank": hit.bm25_rank,
                "dense_rank": hit.dense_rank,
            }
            for hit in retrieval.hits
        ],
        "block_msg_spans": [
            f"{block.msg_ids[0]}-{block.msg_ids[-1]}"
            for block in retrieval.blocks
            if block.msg_ids
        ],
    }


def _escape_md_cell(text: str) -> str:
    return " ".join(text.replace("|", "\\|").splitlines()).strip()


def _format_markdown_report(report: dict[str, Any]) -> str:
    lines: list[str] = [
        f"# Evaluation report: {report['suite_name']}",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Provider / chat model / embed model: `{report['provider']}` / `{report['chat_model']}` / `{report['embed_model']}`",
        f"- Mode: `{'state-only' if report['state_only'] else 'full-qa'}`",
        f"- Rebuild requested: `{report['rebuild']}`",
        "",
        "## Compiled project states",
        "",
    ]

    for chat_entry in report["chats"]:
        state = chat_entry["project_state"]
        lines.extend(
            [
                f"### Chat: `{chat_entry['chat_file']}`",
                "",
                f"- Messages: `{chat_entry['total_messages']}` (source lines: `{chat_entry['total_source_lines']}`, date_order: `{chat_entry['date_order']}`)",
                f"- State cache hit: `{chat_entry['state_cache_hit']}`",
                f"- State invariant errors: `{len(chat_entry['state_invariant_errors'])}`",
                f"- Counts: decisions={len(state['decisions'])}, tasks={len(state['tasks'])}, deadlines={len(state['deadlines'])}, open_items={len(state['open_items'])}",
                "",
                "Decisions:",
            ]
        )
        if state["decisions"]:
            for dec in state["decisions"]:
                refs = ", ".join(r["msg_id"] for r in dec["evidence"])
                lines.append(
                    f"- `{dec['topic_key']}` ({dec['status']}): {dec['current_value']} [{refs}]"
                )
        else:
            lines.append("- (none)")

        lines.append("")
        lines.append("Tasks:")
        if state["tasks"]:
            for task in state["tasks"]:
                owner = task["current_owner"] or "unassigned"
                note = f" ({task['status_note']})" if task["status_note"] else ""
                refs = ", ".join(r["msg_id"] for r in task["evidence"])
                lines.append(
                    f"- `{task['task_key']}` [{task['status']}] owner={owner}{note} [{refs}]"
                )
        else:
            lines.append("- (none)")

        lines.append("")
        lines.append("Deadlines:")
        if state["deadlines"]:
            for dl in state["deadlines"]:
                rule = f" (rule: {dl['dependency_rule']})" if dl.get("dependency_rule") else ""
                refs = ", ".join(r["msg_id"] for r in dl["evidence"])
                lines.append(
                    f"- `{dl['item_key']}` [{dl['status']}]: {dl['current_date_text']}{rule} [{refs}]"
                )
        else:
            lines.append("- (none)")

        lines.append("")
        lines.append("Open items:")
        if state["open_items"]:
            for item in state["open_items"]:
                owner = f" owner={item['owner']}" if item.get("owner") else ""
                refs = ", ".join(r["msg_id"] for r in item["evidence"])
                lines.append(
                    f"- `{item['issue_key']}` [{item['status']}{owner}]: {item['description']} [{refs}]"
                )
        else:
            lines.append("- (none)")
        lines.append("")

    results = report.get("results", [])
    if results:
        summary = report["summary"]
        lines.extend(
            [
                "## Q&A results (pre-audit diagnostic summary)",
                "",
                f"- Total questions: `{summary['total_questions']}`",
                f"- Structural invariants passed: `{summary['invariants_passed']} / {summary['total_questions']}`",
                f"- Deterministic hint checks passed: `{summary['hints_passed']} / {summary['total_questions']}` (manual audit still required)",
                f"- Rejected fabricated/invalid msg_ids count: `{summary['total_rejected_msg_ids']}`",
                "",
                "| # | Question | Expected (by hand) | System answer | Evidence IDs | Status / Flags | Hint check |",
                "|---|---|---|---|---|---|---|",
            ]
        )
        for row in results:
            ev_ids = ", ".join(r["msg_id"] for r in row["evidence"]) or "none"
            flags = (
                f"supported={row['supported']}, "
                f"partial={row['partial_status_only']}, "
                f"ver={row['verification_status']}, "
                f"rejected={row['rejected_msg_ids']}"
            )
            hint = row["hint_check"]
            hint_label = (
                "PASS"
                if hint["all_hints_pass"]
                else f"CHECK (missing={hint['missing_fact_groups']}, forbidden={hint['forbidden_hits']})"
            )
            lines.append(
                "| "
                + " | ".join(
                    [
                        _escape_md_cell(str(row["id"])),
                        _escape_md_cell(row["question"]),
                        _escape_md_cell(row["expected_answer"]),
                        _escape_md_cell(row["answer"]),
                        _escape_md_cell(ev_ids),
                        _escape_md_cell(flags),
                        _escape_md_cell(hint_label),
                    ]
                )
                + " |"
            )
        lines.append("")

    return "\n".join(lines)


def run_evaluation(
    *,
    suite_path: Path | None = None,
    chat_override: Path | None = None,
    state_only: bool = False,
    rebuild: bool = False,
    provider: str | None = None,
    output_dir: Path = Path("eval_outputs"),
    output_name: str | None = None,
    top_k: int = 5,
    expand_n: int = 3,
    adapter_factory: Callable[..., BaseLLMAdapter] = create_llm_adapter,
    cache_dir: Path | None = None,
) -> tuple[dict[str, Any], Path, Path]:
    """Execute the evaluation suite or state-only run and write JSON + markdown reports."""
    if suite_path is None and chat_override is None:
        raise ValueError("Either suite_path or chat_override must be provided.")

    suite_data: dict[str, Any]
    if suite_path is not None:
        resolved_suite = _resolve_repo_path(suite_path)
        suite_data = json.loads(resolved_suite.read_text(encoding="utf-8"))
        suite_name = str(suite_data.get("suite_name") or resolved_suite.stem)
        default_chat = suite_data.get("default_chat_file")
        raw_questions: list[dict[str, Any]] = list(suite_data.get("questions") or [])
    else:
        assert chat_override is not None
        suite_name = f"state_{chat_override.stem}"
        default_chat = str(chat_override)
        raw_questions = []

    filter_key = (
        _normalize_path_key(_resolve_repo_path(chat_override))
        if chat_override is not None
        else None
    )

    questions_to_run: list[dict[str, Any]] = []
    chat_paths_ordered: list[Path] = []
    seen_chat_keys: set[str] = set()

    if raw_questions:
        for item in raw_questions:
            raw_cf = item.get("chat_file") or default_chat
            if not raw_cf:
                raise ValueError(f"Question '{item.get('id')}' has no chat_file.")
            resolved_cf = _resolve_repo_path(raw_cf)
            cf_key = _normalize_path_key(resolved_cf)
            if filter_key is not None and cf_key != filter_key:
                continue
            questions_to_run.append({**item, "_resolved_chat": resolved_cf, "_chat_key": cf_key})
            if cf_key not in seen_chat_keys:
                seen_chat_keys.add(cf_key)
                chat_paths_ordered.append(resolved_cf)
    elif chat_override is not None:
        resolved_cf = _resolve_repo_path(chat_override)
        chat_paths_ordered.append(resolved_cf)

    if not chat_paths_ordered:
        raise ValueError("No matching chat files or questions selected to run.")

    adapter = adapter_factory(provider=provider)

    compiled_chats: dict[str, dict[str, Any]] = {}
    chat_reports: list[dict[str, Any]] = []

    for chat_path in chat_paths_ordered:
        chat = parse_chat_file(chat_path)
        raw_text = chat_path.read_text(encoding="utf-8")
        state, state_cache_hit, state_cache_path = compile_project_state_with_meta(
            chat,
            adapter,
            raw_text=raw_text,
            cache_dir=cache_dir,
            rebuild=rebuild,
        )
        state_errors = check_project_state_invariants(state, chat)
        retriever: HybridRetriever | None = None
        if not state_only and questions_to_run:
            retriever = HybridRetriever(
                chat,
                adapter,
                raw_text=raw_text,
                cache_dir=cache_dir,
                rebuild=rebuild,
            )

        rel_chat_label = (
            str(chat_path.relative_to(REPO_ROOT))
            if chat_path.is_relative_to(REPO_ROOT)
            else str(chat_path)
        )
        ckey = _normalize_path_key(chat_path)
        compiled_chats[ckey] = {
            "chat": chat,
            "state": state,
            "retriever": retriever,
            "label": rel_chat_label,
        }
        chat_reports.append(
            {
                "chat_file": rel_chat_label,
                "date_order": chat.date_order,
                "total_source_lines": chat.total_source_lines,
                "total_messages": len(chat.messages),
                "state_cache_hit": state_cache_hit,
                "state_cache_path": str(state_cache_path),
                "state_invariant_errors": state_errors,
                "project_state": state.model_dump(),
            }
        )

    results: list[dict[str, Any]] = []
    if not state_only:
        for item in questions_to_run:
            bundle = compiled_chats[item["_chat_key"]]
            chat = bundle["chat"]
            state = bundle["state"]
            retriever = bundle["retriever"]
            assert retriever is not None

            retrieval = retriever.retrieve(
                item["question"], top_k=top_k, expand_n=expand_n
            )
            verified = answer_and_verify(
                item["question"], chat, state, retrieval, adapter
            )
            inv_errors = check_answer_invariants(verified, chat)
            hint_check = evaluate_fact_hints(item, verified)

            results.append(
                {
                    "id": item.get("id", f"Q{len(results) + 1}"),
                    "category": item.get("category", "general"),
                    "chat_file": bundle["label"],
                    "question": item["question"],
                    "expected_answer": item.get("expected_answer", ""),
                    "expected_supported": item.get("expected_supported"),
                    "expected_partial_status_only": item.get(
                        "expected_partial_status_only"
                    ),
                    "answer": verified.answer,
                    "supported": verified.supported,
                    "partial_status_only": verified.partial_status_only,
                    "verification_status": verified.verification_status,
                    "rejected_msg_ids": verified.rejected_msg_ids,
                    "evidence": [ref.model_dump() for ref in verified.evidence],
                    "retrieval": _summarize_retrieval(retrieval),
                    "invariant_ok": len(inv_errors) == 0,
                    "invariant_errors": inv_errors,
                    "hint_check": hint_check,
                }
            )

    invariants_passed = sum(1 for r in results if r["invariant_ok"])
    hints_passed = sum(1 for r in results if r["hint_check"]["all_hints_pass"])
    total_rejected = sum(len(r["rejected_msg_ids"]) for r in results)
    state_invariants_ok = all(
        len(c["state_invariant_errors"]) == 0 for c in chat_reports
    )

    report: dict[str, Any] = {
        "suite_name": suite_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider": adapter.provider_name,
        "chat_model": adapter.chat_model,
        "embed_model": adapter.embed_model,
        "state_only": state_only,
        "rebuild": rebuild,
        "chats": chat_reports,
        "results": results,
        "summary": {
            "total_chats": len(chat_reports),
            "state_invariants_ok": state_invariants_ok,
            "total_questions": len(results),
            "invariants_passed": invariants_passed,
            "hints_passed": hints_passed,
            "total_rejected_msg_ids": total_rejected,
        },
    }

    resolved_out_dir = _resolve_repo_path(output_dir)
    resolved_out_dir.mkdir(parents=True, exist_ok=True)
    stem = output_name or (
        f"{suite_name}_state_only" if state_only else suite_name
    )
    if chat_override is not None and suite_path is not None and not output_name:
        stem = f"{suite_name}_{chat_override.stem}"
    json_path = resolved_out_dir / f"{stem}.json"
    md_path = resolved_out_dir / f"{stem}.md"

    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    md_path.write_text(_format_markdown_report(report), encoding="utf-8")
    return report, json_path, md_path


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for the evaluation runner."""
    parser = argparse.ArgumentParser(
        description="Run evaluation benchmarks against the temporal chat Q&A pipeline."
    )
    parser.add_argument(
        "--suite",
        type=Path,
        default=None,
        help="Path to evaluation suite JSON file (for example eval/team_chat_eval.json).",
    )
    parser.add_argument(
        "--chat",
        type=Path,
        default=None,
        help="Optional chat file path to filter a multi-chat suite or run --state-only directly.",
    )
    parser.add_argument(
        "--state-only",
        action="store_true",
        help="Compile ProjectState only without running Q&A questions.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuilding compiled ProjectState and embedding caches.",
    )
    parser.add_argument(
        "--provider",
        choices=("gemini", "openai", "ollama"),
        default=None,
        help="Optional LLM provider override.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval_outputs"),
        help="Directory for JSON and markdown output reports (default: eval_outputs).",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Optional base filename stem for output reports.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top RRF chunks to retrieve per question (default: 5).",
    )
    parser.add_argument(
        "--expand-n",
        type=int,
        default=3,
        help="Number of adjacent messages to expand around retrieved chunks (default: 3).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run evaluation CLI and print concise summary."""
    args = build_arg_parser().parse_args(argv)
    if args.suite is None and args.chat is None:
        print(
            "Error: provide --suite or --chat (with --state-only).",
            file=sys.stderr,
        )
        return 2
    if args.suite is None and not args.state_only:
        print(
            "Error: --chat without --suite requires --state-only.",
            file=sys.stderr,
        )
        return 2

    report, json_path, md_path = run_evaluation(
        suite_path=args.suite,
        chat_override=args.chat,
        state_only=args.state_only,
        rebuild=args.rebuild,
        provider=args.provider,
        output_dir=args.output_dir,
        output_name=args.output_name,
        top_k=args.top_k,
        expand_n=args.expand_n,
    )
    summary = report["summary"]
    print(
        f"Completed '{report['suite_name']}' using {report['provider']}/{report['chat_model']}: "
        f"chats={summary['total_chats']}, state_invariants_ok={summary['state_invariants_ok']}, "
        f"questions={summary['total_questions']}, "
        f"invariants_passed={summary['invariants_passed']}/{summary['total_questions']}, "
        f"hints_passed={summary['hints_passed']}/{summary['total_questions']}, "
        f"rejected_ids={summary['total_rejected_msg_ids']}"
    )
    print(f"Reports saved: {json_path} and {md_path}")
    all_ok = summary["state_invariants_ok"] and (
        summary["invariants_passed"] == summary["total_questions"]
    )
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
