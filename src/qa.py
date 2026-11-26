"""Grounded Q&A synthesizer combining ProjectState and retrieved evidence blocks."""

from __future__ import annotations

from collections.abc import Sequence

from src.llm import BaseLLMAdapter
from src.parser import ParsedChat
from src.retriever import RetrievalResult
from src.schemas import DraftAnswer, EvidenceRef, ProjectState, VerifiedAnswer
from src.verifier import ABSTENTION_ANSWER, ValidationFailure, verify_with_retry

_QA_SYSTEM_PROMPT = """You answer questions about an exported WhatsApp team chat using compiled ProjectState and raw chronological evidence blocks.

Rules:
1. Language: Answer in clear English or Hinglish matching the user's question.
2. Chronology and reversals: Read ProjectState history[] and raw evidence in chronological order. Later decisions, owner reassignments, and deadline updates override earlier ones. When asked whether something changed, trace the timeline from initial choice through revisions to the latest state. When a deadline depends on a rule or a decision evolved across updates, cite evidence_msg_ids covering BOTH the rule/origin message and the latest update.
3. Promises and ETAs are NOT completions: Never treat future promises or target ETAs (such as "kal tak", "by Sunday", "ready by <date>", "pakka", or in-progress percentages like "70%") as completed work.
4. Missing, deleted, or post-chat facts: If the answer depends on deleted message content, unshared URLs/details, or events after last_message_timestamp, set supported=false, partial_status_only=false, answer="the chat doesn't say", and evidence_msg_ids=[].
5. Partially tracked or unconfirmed outcomes: If a task or decision was discussed (with an owner, partial progress, or ETA) but completion or finalization was never confirmed before the chat ended, set supported=true, partial_status_only=true, explicitly include "the chat doesn't say whether <task> finished" (or "the chat doesn't say whether <topic> was finalized") alongside the latest verified owner/status/ETA, and cite the supporting msg_ids.
6. Citations: Output ONLY exact msg_id strings (for example "M017", "M042") in evidence_msg_ids, taken strictly from the provided ProjectState annotations or raw evidence blocks. Never output timestamps, line numbers, or senders inside evidence_msg_ids, never invent a msg_id, and do NOT write raw internal msg_id tokens (such as "M017") inside the prose answer string."""


def _format_ref_ids(refs: Sequence[EvidenceRef]) -> str:
    ids = list(dict.fromkeys(ref.msg_id for ref in refs))
    return ", ".join(ids) if ids else "none"


def format_state_for_qa(state: ProjectState) -> str:
    """Format ProjectState with history[] and grounded msg_ids for Q&A synthesis."""
    lines: list[str] = [
        f"Last message timestamp: {state.last_message_timestamp or 'unknown'} "
        f"(date_order={state.date_order}, total_messages={state.total_messages})",
        (
            "Participants: "
            + (", ".join(state.participants) if state.participants else "none")
        ),
        "",
        "Decisions:",
    ]

    if not state.decisions:
        lines.append("  (none)")
    else:
        for dec in state.decisions:
            lines.append(
                f"  - {dec.title} ({dec.topic_key}): current={dec.current_value!r} "
                f"| status={dec.status} | msg_ids=[{_format_ref_ids(dec.evidence)}]"
            )
            if dec.history:
                hist_parts = [
                    f"{h.timestamp_raw} [{h.action}] {h.value!r}"
                    + (f" ({h.note})" if h.note else "")
                    + f" [{_format_ref_ids(h.evidence)}]"
                    for h in dec.history
                ]
                lines.append("    history[]: " + " -> ".join(hist_parts))

    lines.extend(["", "Tasks:"])
    if not state.tasks:
        lines.append("  (none)")
    else:
        for task in state.tasks:
            owner_label = task.current_owner or "unassigned"
            note_part = f" | status_note={task.status_note!r}" if task.status_note else ""
            lines.append(
                f"  - {task.title} ({task.task_key}): current_owner={owner_label} "
                f"| status={task.status}{note_part} | msg_ids=[{_format_ref_ids(task.evidence)}]"
            )
            if task.history:
                hist_parts = [
                    f"{h.timestamp_raw} [{h.action}] owner={h.owner or 'unassigned'} "
                    f"value={h.value!r} [{_format_ref_ids(h.evidence)}]"
                    for h in task.history
                ]
                lines.append("    history[]: " + " -> ".join(hist_parts))

    lines.extend(["", "Deadlines:"])
    if not state.deadlines:
        lines.append("  (none)")
    else:
        for dl in state.deadlines:
            rule_part = (
                f" | dependency_rule={dl.dependency_rule!r}"
                if dl.dependency_rule
                else ""
            )
            lines.append(
                f"  - {dl.title} ({dl.item_key}): current_date={dl.current_date_text!r} "
                f"| status={dl.status}{rule_part} | msg_ids=[{_format_ref_ids(dl.evidence)}]"
            )
            if dl.history:
                hist_parts = [
                    f"{h.timestamp_raw} [{h.action}] {h.value!r}"
                    + (f" (rule={h.note!r})" if h.note else "")
                    + f" [{_format_ref_ids(h.evidence)}]"
                    for h in dl.history
                ]
                lines.append("    history[]: " + " -> ".join(hist_parts))

    lines.extend(["", "Open or unresolved items:"])
    if not state.open_items:
        lines.append("  (none)")
    else:
        for item in state.open_items:
            owner_part = f" | owner={item.owner}" if item.owner else ""
            lines.append(
                f"  - {item.title} ({item.issue_key}): status={item.status}{owner_part} "
                f"| description={item.description!r} | msg_ids=[{_format_ref_ids(item.evidence)}]"
            )
            if item.history:
                hist_parts = [
                    f"{h.timestamp_raw} [{h.action}] {h.value!r} [{_format_ref_ids(h.evidence)}]"
                    for h in item.history
                ]
                lines.append("    history[]: " + " -> ".join(hist_parts))

    return "\n".join(lines)


def _collect_context_msg_ids(
    state: ProjectState, retrieval: RetrievalResult
) -> list[str]:
    seen: dict[str, None] = {}
    for block in retrieval.blocks:
        for mid in block.msg_ids:
            seen[mid] = None
    for dec in state.decisions:
        for ref in dec.evidence:
            seen[ref.msg_id] = None
        for entry in dec.history:
            for ref in entry.evidence:
                seen[ref.msg_id] = None
    for task in state.tasks:
        for ref in task.evidence:
            seen[ref.msg_id] = None
        for entry in task.history:
            for ref in entry.evidence:
                seen[ref.msg_id] = None
    for dl in state.deadlines:
        for ref in dl.evidence:
            seen[ref.msg_id] = None
        for entry in dl.history:
            for ref in entry.evidence:
                seen[ref.msg_id] = None
    for item in state.open_items:
        for ref in item.evidence:
            seen[ref.msg_id] = None
        for entry in item.history:
            for ref in entry.evidence:
                seen[ref.msg_id] = None
    return list(seen.keys())


def _build_qa_prompt(
    question: str,
    state: ProjectState,
    retrieval: RetrievalResult,
) -> str:
    state_summary = format_state_for_qa(state)
    evidence_text = (
        retrieval.formatted_context.strip()
        if retrieval.formatted_context.strip()
        else "(no retrieved raw evidence blocks)"
    )
    return (
        "Compiled ProjectState summary:\n"
        f"{state_summary}\n\n"
        "Expanded chronological raw evidence blocks:\n"
        f"{evidence_text}\n\n"
        f"Question: {question}\n\n"
        "Return a structured DraftAnswer JSON object. Cite ONLY exact msg_id values "
        "from the context above in evidence_msg_ids."
    )


def _build_corrective_prompt(
    question: str,
    state: ProjectState,
    retrieval: RetrievalResult,
    failed_draft: DraftAnswer,
    failure: ValidationFailure,
) -> str:
    base_prompt = _build_qa_prompt(question, state, retrieval)
    valid_context_ids = _collect_context_msg_ids(state, retrieval)
    return (
        f"{base_prompt}\n\n"
        "CORRECTION REQUIRED: Your previous draft failed deterministic verification.\n"
        f"Failure code: {failure.code}\n"
        f"Failure detail: {failure.message}\n"
        f"Previous draft answer: {failed_draft.answer!r}\n"
        f"Previous evidence_msg_ids: {failed_draft.evidence_msg_ids}\n"
        f"Valid msg_ids visible in context: {valid_context_ids}\n\n"
        "Fix this strictly:\n"
        "- Only cite exact msg_ids from the visible context that are NOT deleted placeholders.\n"
        "- Never claim a task marked unconfirmed, unassigned, in_progress, or blocked is finished, completed, deployed, live, merged, or done unless a non-deleted cited message explicitly reports completion.\n"
        "- If completion is unconfirmed, set supported=true, partial_status_only=true, state 'the chat doesn't say whether <task> finished' alongside verified owner/status/ETA, and cite valid msg_ids.\n"
        f"- If the chat does not support an answer, set supported=false, partial_status_only=false, answer={ABSTENTION_ANSWER!r}, evidence_msg_ids=[]."
    )


def generate_draft_answer(
    question: str,
    state: ProjectState,
    retrieval: RetrievalResult,
    adapter: BaseLLMAdapter,
) -> DraftAnswer:
    """Synthesize an unverified DraftAnswer from ProjectState and retrieved blocks."""
    prompt = _build_qa_prompt(question, state, retrieval)
    return adapter.generate_structured(
        prompt,
        DraftAnswer,
        system_prompt=_QA_SYSTEM_PROMPT,
        temperature=0.0,
    )


def answer_and_verify(
    question: str,
    chat: ParsedChat,
    state: ProjectState,
    retrieval: RetrievalResult,
    adapter: BaseLLMAdapter,
) -> VerifiedAnswer:
    """Run grounded Q&A synthesis with deterministic verification and at most 1 repair retry."""
    initial_draft = generate_draft_answer(question, state, retrieval, adapter)

    def _repair_once(
        failed_draft: DraftAnswer, failure: ValidationFailure
    ) -> DraftAnswer:
        repair_prompt = _build_corrective_prompt(
            question, state, retrieval, failed_draft, failure
        )
        return adapter.generate_structured(
            repair_prompt,
            DraftAnswer,
            system_prompt=_QA_SYSTEM_PROMPT,
            temperature=0.0,
        )

    return verify_with_retry(
        initial_draft,
        chat,
        state,
        question=question,
        repair_fn=_repair_once,
    )
