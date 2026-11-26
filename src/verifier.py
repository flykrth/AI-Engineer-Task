"""Deterministic citation resolution, abstention normalization, and safety guards."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

from pydantic import BaseModel

from src.parser import ParsedChat, ParsedMessage
from src.schemas import (
    DraftAnswer,
    EvidenceRef,
    ProjectState,
    TaskRecord,
    VerifiedAnswer,
)

ABSTENTION_ANSWER = "the chat doesn't say"
_INCOMPLETE_TASK_STATUSES = frozenset(
    {"unconfirmed", "unassigned", "in_progress", "blocked"}
)

_CLAUSE_SPLIT_RE = re.compile(
    r"[.!?\n;]+|,\s*(?:but|while|however|though|whereas|and)\b",
    re.IGNORECASE,
)
_COMPLETION_PHRASE_RE = re.compile(
    r"\b(?:"
    r"finished|"
    r"completed|"
    r"complete|"
    r"deployed|"
    r"is\s+live|"
    r"went\s+live|"
    r"now\s+live|"
    r"are\s+live|"
    r"merged|"
    r"done|"
    r"shipped|"
    r"launched|"
    r"bought\s+it|"
    r"purchased|"
    r"khatam\s+ho\s+gaya|"
    r"pura\s+ho\s+gaya|"
    r"ho\s+gaya"
    r")\b",
    re.IGNORECASE,
)
_UNCERTAINTY_OR_NEGATION_RE = re.compile(
    r"\b(?:"
    r"doesn'?t\s+say|does\s+not\s+say|didn'?t\s+say|did\s+not\s+say|"
    r"never\s+confirmed|not\s+confirmed|unconfirmed|unclear|unknown|"
    r"cannot\s+confirm|can'?t\s+confirm|no\s+confirmation|"
    r"whether|if\s+it|if\s+the|"
    r"not\s+yet|yet\s+to|not|no|never|neither|nor|"
    r"isn'?t|aren'?t|wasn'?t|weren'?t|hasn'?t|haven'?t|hadn'?t|didn'?t|don'?t|won'?t|"
    r"pending|remaining|incomplete|unfinished|in\s+progress|ongoing|wip|"
    r"almost|nearly|partially|partly|"
    r"promised|promise|eta|expected\s+by|target|planned|scheduled|"
    r"will\s+be|going\s+to|would\s+be|should\s+be|might\s+be|could\s+be|"
    r"need\s+a|needed|required|chahiye|allowed|"
    r"kal\s+tak|tak\s+ready|pakka|kar\s+raha|kar\s+rahi|karunga|karungi|dunga|dungi|"
    r"bacha|baaki|nahi|nhi"
    r")\b",
    re.IGNORECASE,
)
_PARTIAL_PERCENT_RE = re.compile(
    r"\b([0-9]{1,2})\s*%\s*(?:done|complete|completed|finished|ready)?\b",
    re.IGNORECASE,
)
_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_TASK_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "was",
        "were",
        "with",
        "task",
        "work",
        "item",
        "feature",
        "status",
    }
)


class ValidationFailure(BaseModel):
    """Structured explanation of why a DraftAnswer failed deterministic verification."""

    code: str
    message: str


def _normalize_match_token(tok: str) -> str:
    """Lightly stem common English suffixes when at least 4 stem chars remain."""
    for suffix in ("ment", "tion", "ing", "ed", "es", "s"):
        if tok.endswith(suffix) and len(tok) - len(suffix) >= 4:
            return tok[: -len(suffix)]
    return tok


def _content_tokens(text: str) -> set[str]:
    return {
        _normalize_match_token(tok)
        for tok in _TOKEN_RE.findall(text.lower())
        if len(tok) >= 2 and tok not in _TASK_STOPWORDS
    }


def _task_tokens(task: TaskRecord) -> set[str]:
    parts = [
        task.task_key.replace("_", " "),
        task.title,
    ]
    return _content_tokens(" ".join(parts))


def _phrase_asserts_completion(clause: str) -> bool:
    """Return True if a clause unambiguously asserts completion without hedging or negation."""
    stripped = clause.strip()
    if not stripped:
        return False
    if _UNCERTAINTY_OR_NEGATION_RE.search(stripped):
        return False
    for pct_match in _PARTIAL_PERCENT_RE.finditer(stripped):
        if int(pct_match.group(1)) < 100:
            return False
    return bool(_COMPLETION_PHRASE_RE.search(stripped))


def _task_evidence_ids(task: TaskRecord) -> set[str]:
    ids = {ref.msg_id for ref in task.evidence}
    for entry in task.history:
        for ref in entry.evidence:
            ids.add(ref.msg_id)
    return ids


def _find_matching_tasks(
    clause: str,
    question: str,
    cited_ids: set[str],
    tasks: Sequence[TaskRecord],
) -> list[TaskRecord]:
    """Identify tasks targeted by a completion clause via lexical or evidence match."""
    clause_tokens = _content_tokens(clause)
    direct_matches: list[TaskRecord] = []
    for task in tasks:
        t_tokens = _task_tokens(task)
        if t_tokens & clause_tokens:
            direct_matches.append(task)
    if direct_matches:
        return direct_matches

    # Fall back to question tokens or cited evidence IDs when answer uses pronouns ("it is done").
    question_tokens = _content_tokens(question)
    indirect_matches: list[TaskRecord] = []
    for task in tasks:
        t_tokens = _task_tokens(task)
        if (t_tokens & question_tokens) or (_task_evidence_ids(task) & cited_ids):
            indirect_matches.append(task)
    return indirect_matches


def _quote_reports_completion(
    msg: ParsedMessage,
    target_task: TaskRecord,
    all_tasks: Sequence[TaskRecord],
) -> bool:
    """Check whether a non-deleted message reports unambiguous completion for target_task."""
    if msg.kind in ("deleted", "media_omitted"):
        return False
    clauses = [c.strip() for c in _CLAUSE_SPLIT_RE.split(msg.text) if c.strip()]
    if not clauses:
        return False

    target_tokens = _task_tokens(target_task)
    other_tasks = [t for t in all_tasks if t.task_key != target_task.task_key]

    for clause in clauses:
        if not _phrase_asserts_completion(clause):
            continue
        clause_tokens = _content_tokens(clause)
        if target_tokens & clause_tokens:
            return True
        mentions_other_only = any(
            (_task_tokens(other) & clause_tokens) for other in other_tasks
        )
        if not mentions_other_only:
            return True
    return False


def _detect_completion_overclaim(
    answer: str,
    question: str,
    valid_msgs: Sequence[ParsedMessage],
    state: ProjectState,
) -> ValidationFailure | None:
    """Guard 3: block unambiguous completion claims for incomplete ProjectState tasks."""
    if not state.tasks:
        return None

    cited_ids = {m.msg_id for m in valid_msgs}
    clauses = [c.strip() for c in _CLAUSE_SPLIT_RE.split(answer) if c.strip()]
    for clause in clauses:
        if not _phrase_asserts_completion(clause):
            continue
        matched_tasks = _find_matching_tasks(clause, question, cited_ids, state.tasks)
        for task in matched_tasks:
            if task.status not in _INCOMPLETE_TASK_STATUSES:
                continue
            has_completion_evidence = any(
                _quote_reports_completion(msg, task, state.tasks) for msg in valid_msgs
            )
            if not has_completion_evidence:
                return ValidationFailure(
                    code="completion_overclaim",
                    message=(
                        f"Answer asserts completion in '{clause}', but ProjectState marks "
                        f"task '{task.title}' ({task.task_key}) as '{task.status}' and none "
                        "of the cited messages report completion."
                    ),
                )
    return None


def resolve_evidence_refs(
    evidence_msg_ids: Sequence[str], chat: ParsedChat
) -> tuple[list[EvidenceRef], list[str], list[ParsedMessage]]:
    """Validate msg_ids against ParsedChat, returning authentic refs and rejected IDs."""
    msg_by_id = {m.msg_id: m for m in chat.messages}
    msg_index_by_id = {m.msg_id: idx for idx, m in enumerate(chat.messages)}

    valid_ids: list[str] = []
    seen_valid: set[str] = set()
    rejected_ids: list[str] = []
    seen_rejected: set[str] = set()

    for raw_id in evidence_msg_ids:
        cleaned = raw_id.strip()
        if cleaned in msg_by_id:
            if cleaned not in seen_valid:
                seen_valid.add(cleaned)
                valid_ids.append(cleaned)
        else:
            rejected_label = cleaned if cleaned else raw_id
            if rejected_label not in seen_rejected:
                seen_rejected.add(rejected_label)
                rejected_ids.append(rejected_label)

    valid_ids.sort(key=lambda mid: msg_index_by_id[mid])
    valid_msgs = [msg_by_id[mid] for mid in valid_ids]
    refs = [
        EvidenceRef(
            msg_id=msg.msg_id,
            line_start=msg.line_start,
            line_end=msg.line_end,
            timestamp_raw=msg.timestamp_raw,
            timestamp_iso=msg.timestamp_iso,
            sender=msg.sender,
            quote=msg.text,
        )
        for msg in valid_msgs
    ]
    return refs, rejected_ids, valid_msgs


def verify_draft(
    draft: DraftAnswer,
    chat: ParsedChat,
    state: ProjectState,
    *,
    question: str = "",
) -> tuple[VerifiedAnswer, ValidationFailure | None]:
    """Deterministically verify citations, abstention rules, and completion claims."""
    refs, rejected_ids, valid_msgs = resolve_evidence_refs(
        draft.evidence_msg_ids, chat
    )

    if not draft.supported:
        return (
            VerifiedAnswer(
                answer=ABSTENTION_ANSWER,
                supported=False,
                partial_status_only=False,
                evidence=[],
                verification_status="abstained",
                rejected_msg_ids=rejected_ids,
            ),
            None,
        )

    # Guard 1: empty verified citations after ID resolution.
    if not refs:
        return (
            VerifiedAnswer(
                answer=ABSTENTION_ANSWER,
                supported=False,
                partial_status_only=False,
                evidence=[],
                verification_status="failed_empty_citations",
                rejected_msg_ids=rejected_ids,
            ),
            ValidationFailure(
                code="empty_citations",
                message=(
                    "Supported answer has zero valid evidence_msg_ids after checking "
                    f"against ParsedChat (rejected_msg_ids={rejected_ids})."
                ),
            ),
        )

    # Guard 2: deleted-only citations cannot support positive answers.
    if all(msg.kind == "deleted" for msg in valid_msgs):
        return (
            VerifiedAnswer(
                answer=ABSTENTION_ANSWER,
                supported=False,
                partial_status_only=False,
                evidence=[],
                verification_status="failed_deleted_only_citations",
                rejected_msg_ids=rejected_ids,
            ),
            ValidationFailure(
                code="deleted_only_citations",
                message=(
                    "All cited messages are deleted placeholders whose original content "
                    "is unavailable."
                ),
            ),
        )

    # Guard 3: completion overclaim on incomplete tasks.
    completion_failure = _detect_completion_overclaim(
        draft.answer, question, valid_msgs, state
    )
    if completion_failure is not None:
        return (
            VerifiedAnswer(
                answer=ABSTENTION_ANSWER,
                supported=False,
                partial_status_only=False,
                evidence=[],
                verification_status=f"failed_{completion_failure.code}",
                rejected_msg_ids=rejected_ids,
            ),
            completion_failure,
        )

    status = "verified_with_stripped_ids" if rejected_ids else "verified"
    return (
        VerifiedAnswer(
            answer=draft.answer.strip(),
            supported=True,
            partial_status_only=draft.partial_status_only,
            evidence=refs,
            verification_status=status,
            rejected_msg_ids=rejected_ids,
        ),
        None,
    )


def verify_with_retry(
    initial_draft: DraftAnswer,
    chat: ParsedChat,
    state: ProjectState,
    *,
    question: str = "",
    repair_fn: Callable[[DraftAnswer, ValidationFailure], DraftAnswer] | None = None,
) -> VerifiedAnswer:
    """Verify a draft answer, running at most 1 corrective repair attempt on guard failure."""
    verified, failure = verify_draft(initial_draft, chat, state, question=question)
    if failure is None:
        return verified

    if repair_fn is None:
        return verified.model_copy(
            update={"verification_status": f"forced_abstention:{failure.code}"}
        )

    repaired_draft = repair_fn(initial_draft, failure)
    repaired_verified, repaired_failure = verify_draft(
        repaired_draft, chat, state, question=question
    )

    combined_rejected = list(
        dict.fromkeys([*verified.rejected_msg_ids, *repaired_verified.rejected_msg_ids])
    )
    if repaired_failure is None:
        next_status = (
            "abstained_after_retry"
            if not repaired_verified.supported
            else "verified_after_retry"
        )
        return repaired_verified.model_copy(
            update={
                "verification_status": next_status,
                "rejected_msg_ids": combined_rejected,
            }
        )

    return VerifiedAnswer(
        answer=ABSTENTION_ANSWER,
        supported=False,
        partial_status_only=False,
        evidence=[],
        verification_status=f"forced_abstention:{repaired_failure.code}",
        rejected_msg_ids=combined_rejected,
    )
