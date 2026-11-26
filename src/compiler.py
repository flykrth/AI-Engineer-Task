"""Chronological sliding-window state extractor, deterministic reducer, and cache."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from src.llm import BaseLLMAdapter
from src.parser import ParsedChat, ParsedMessage
from src.schemas import (
    SCHEMA_VERSION,
    DeadlineDelta,
    DeadlineRecord,
    DeadlineStatus,
    DecisionDelta,
    DecisionRecord,
    DecisionStatus,
    EvidenceRef,
    OpenIssueDelta,
    OpenItemRecord,
    OpenItemStatus,
    ProjectState,
    StateHistoryEntry,
    TaskDelta,
    TaskRecord,
    TaskStatus,
    WindowExtractionResult,
)

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"

_FIRST_PERSON_TOKENS = {
    "i",
    "me",
    "my",
    "myself",
    "main",
    "mein",
    "mera",
    "meri",
    "mere",
    "hum",
    "humne",
}
_UNASSIGNED_OWNER_TOKENS = {
    "",
    "none",
    "null",
    "unassigned",
    "nobody",
    "no one",
    "tbd",
    "n/a",
    "unknown",
    "anyone",
}
_NOISE_KEY_SUFFIXES = (
    "_decisions",
    "_decision",
    "_deadlines",
    "_deadline",
    "_issues",
    "_issue",
    "_items",
    "_item",
    "_tasks",
    "_task",
    "_status",
    "_feature",
    "_module",
    "_work",
    "_due_date",
    "_date",
)
_FILLER_TOKENS = {
    "task",
    "tasks",
    "decision",
    "decisions",
    "deadline",
    "deadlines",
    "issue",
    "issues",
    "item",
    "items",
    "feature",
    "features",
    "module",
    "work",
    "status",
    "date",
    "due",
    "method",
    "option",
    "options",
    "choice",
    "selection",
    "integration",
    "implementation",
    "setup",
    "system",
    "plan",
    "process",
    "handling",
    "submission",
    "gateway",
    "provider",
    "platform",
    "service",
    "channel",
    "approach",
    "strategy",
    "flow",
    "the",
    "for",
    "of",
    "and",
    "or",
    "to",
    "in",
    "on",
}
_TOKEN_SYNONYMS = {
    "auth": "authentication",
    "db": "database",
    "config": "configuration",
    "repo": "repository",
    "env": "environment",
    "impl": "implementation",
    "spec": "specification",
}

_EXTRACTION_SYSTEM_PROMPT = """You extract chronological state changes from a window of an exported WhatsApp team chat.

Rules:
1. Extract ONLY project-relevant events clearly stated in this chat window. Never invent facts, owners, or dates. Ignore off-topic social or personal banter (greetings, food, weather, jokes, memes, movies, or unrelated personal errands) that does not affect the team's project decisions, tasks, deadlines, or blockers.
2. Cite ONLY exact msg_id values present in the current window (e.g. "M017"). Never invent or alter a msg_id.
3. Resolve first-person pronouns ("I", "me", "my", "Main", "mein", "mera", "meri") to the exact sender name on that message header. Never output "Main" or "I" as an owner.
4. Use lowercase snake_case keys (topic_key, task_key, item_key, issue_key). If an entity already appears in the Active Canonical Key Registry, ALWAYS reuse that exact key when updating, replacing, or reversing it rather than inventing a new synonym or "_alternative" / "_v2" key.
5. Decisions:
   - Use "proposed" for suggestions or one person's preference.
   - Use "confirmed" ONLY when there is explicit, unambiguous group agreement on that exact topic (e.g. "final", "agreed", "decided", "+1", or confirmed purchase).
   - Use "reversed" when a later message replaces an earlier decision; put the NEW active choice in "value" and note what was replaced in "rationale". Do NOT emit a separate DecisionDelta just for a problem or failure observation unless an option is abandoned or a replacement is chosen.
   - Use "dropped" when an option is abandoned without replacement.
   - Use "uncertain" (NEVER "confirmed") when a short or playful reply (such as "obviously", "yes", "sure", or a laughing emoji reaction) immediately follows two different interleaved topics or questions from different speakers (making it ambiguous which message it answered). When one person states a preference and the only response is an ambiguous or laughing reply interleaved with another topic, classify the decision as "uncertain" (never "confirmed") and also emit an OpenIssueDelta ("opened") noting it is still unconfirmed.
6. Tasks:
   - CRITICAL: Never combine multiple deliverables into one TaskDelta. Whenever one line or message assigns "TaskA + TaskB" or "TaskA and TaskB" (for example, "Alice - backend + docs"), ALWAYS emit TWO separate TaskDelta items with separate atomic task_keys (e.g. "backend" and "docs") so later partial handoffs, drops, or completions track independently.
   - Use "assigned" for initial ownership and "reassigned" whenever ownership transfers, swaps, or a participant takes over an existing task (including phrases like "I'll take X", "X is mine now", "X ab mere paas", or "you focus on Y instead" -> emit "reassigned" with owner resolved to the new owner). Only reassign once the new owner accepts or claims it.
   - Use "dropped_unassigned" when an owner drops a task or asks someone else to take it and nobody claims it (set owner=null).
   - Use "progress_updated" for partial progress (e.g., "60% done", "v1 pushed", "only X remaining"). Never mark partial work as "completed".
   - Use "promised_eta" for future promises, target dates, or ETAs (e.g., "kal tak", "by Sunday", "ready by <date>", "kal pakka") and for promises mocked sarcastically by teammates; NEVER mark a promise or ETA as "completed". Do not emit task promised_eta for pure deliverable deadline reminders that belong in deadlines.
   - Use "completed" ONLY when the work is explicitly finished, merged, or done.
   - Use "blocked" when work is blocked by missing access, dependencies, or external hurdles.
7. Deadlines and dependencies:
   - Extract EVERY milestone or deliverable deadline mentioned; if one message states both an anchor milestone date and a dependent deliverable date/rule, emit separate DeadlineDelta items for each.
   - Use "set" for a firm initial date, "changed" every time a date moves or is corrected, "tentative" for uncertain dates (e.g. ending with "?"), and "rule_defined" when a relative rule is given (put the relative offset in dependency_rule).
8. Open items:
   - Use "opened" for unresolved questions, unclaimed work raised in discussion, unconfirmed options (such as shared design options without clear unanimous approval), or pending meeting agenda topics. Set owner=null when ownership itself is unresolved or under debate.
   - Use "deferred" when postponed to a later meeting/date, and "resolved" when settled.
9. Media, edited, and deleted markers:
   - Read adjacent messages around "<Media omitted>" to understand screenshots, diagrams, receipts, or design options, and cite the explanatory message IDs.
   - Ignore "[deleted]" placeholders; never guess deleted content."""


@dataclass(frozen=True)
class _AnchoredDelta:
    """Internal wrapper pairing a delta with its chronological message anchor."""

    anchor_max: int
    anchor_min: int
    seq_order: int
    canonical_key: str
    valid_ids: tuple[str, ...]
    delta: DecisionDelta | TaskDelta | DeadlineDelta | OpenIssueDelta


def format_message_line(msg: ParsedMessage) -> str:
    """Format one parsed message with ID, line span, timestamp, sender, and markers."""
    sender_label = msg.sender if msg.sender is not None else "system"
    header = (
        f"[{msg.msg_id} | L{msg.line_start}-L{msg.line_end} | "
        f"{msg.timestamp_raw} | {sender_label}]"
    )
    suffix = " [edited]" if msg.is_edited else ""
    if msg.kind == "deleted":
        suffix = f"{suffix} [deleted]"
    return f"{header} {msg.text}{suffix}"


def format_messages_window(messages: Sequence[ParsedMessage]) -> str:
    """Format a sequence of parsed messages into a chronological transcript block."""
    return "\n".join(format_message_line(msg) for msg in messages)


def build_message_windows(
    messages: Sequence[ParsedMessage],
    *,
    window_size: int = 25,
    overlap: int = 6,
) -> list[list[ParsedMessage]]:
    """Split chronological messages into overlapping sliding windows."""
    if window_size < 1:
        raise ValueError("window_size must be at least 1.")
    if overlap < 0 or overlap >= window_size:
        raise ValueError("overlap must be >= 0 and strictly less than window_size.")
    if not messages:
        return []

    msg_list = list(messages)
    if len(msg_list) <= window_size:
        return [msg_list]

    windows: list[list[ParsedMessage]] = []
    step = window_size - overlap
    start = 0
    while start < len(msg_list):
        end = min(len(msg_list), start + window_size)
        window_start = max(0, end - window_size) if end == len(msg_list) else start
        windows.append(msg_list[window_start:end])
        if end >= len(msg_list):
            break
        start += step
    return windows


def extract_participants(chat: ParsedChat) -> list[str]:
    """Return unique participant names ordered by first appearance."""
    seen: set[str] = set()
    participants: list[str] = []
    for msg in chat.messages:
        if msg.sender and msg.sender not in seen:
            seen.add(msg.sender)
            participants.append(msg.sender)
    return participants


def compute_chat_sha256(chat: ParsedChat, raw_text: str | None = None) -> str:
    """Compute a deterministic SHA-256 hex digest for the chat input."""
    if raw_text is not None:
        return hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
    if chat.source_path:
        path = Path(chat.source_path)
        if path.is_file():
            return hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
    serialized = chat.model_dump_json()
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def compute_cache_key(
    *,
    chat_sha256: str,
    schema_version: str,
    provider: str,
    chat_model: str,
    window_size: int,
    overlap: int,
) -> str:
    """Compute the content-addressed cache key for compiled ProjectState."""
    payload = json.dumps(
        {
            "chat_sha256": chat_sha256,
            "schema_version": schema_version,
            "provider": provider,
            "chat_model": chat_model,
            "window_size": window_size,
            "overlap": overlap,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_key(raw_key: str) -> str:
    """Normalize a raw entity key into clean lowercase snake_case without noise suffixes."""
    cleaned = re.sub(r"[^a-z0-9]+", "_", raw_key.strip().lower()).strip("_")
    if not cleaned:
        return "unnamed"

    changed = True
    while changed:
        changed = False
        for suffix in _NOISE_KEY_SUFFIXES:
            if cleaned.endswith(suffix) and len(cleaned) > len(suffix):
                cleaned = cleaned[: -len(suffix)].rstrip("_")
                changed = True
                break
    return cleaned or "unnamed"


def _singularize_token(token: str) -> str:
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def _core_tokens(key: str) -> tuple[str, ...]:
    tokens: list[str] = []
    for raw in key.split("_"):
        if not raw or raw in _FILLER_TOKENS:
            continue
        singular = _singularize_token(raw)
        if singular in _FILLER_TOKENS:
            continue
        canonical = _TOKEN_SYNONYMS.get(singular, singular)
        if canonical not in tokens:
            tokens.append(canonical)
    return tuple(tokens)


def resolve_canonical_key(raw_key: str, registered_keys: Sequence[str]) -> str:
    """Match a normalized key against existing registered keys using safe token aliases."""
    norm = normalize_key(raw_key)
    if norm in registered_keys:
        return norm

    cand_tokens = _core_tokens(norm)
    cand_set = set(cand_tokens)

    for existing in registered_keys:
        existing_tokens = _core_tokens(existing)
        if cand_tokens and existing_tokens:
            if set(existing_tokens) == cand_set:
                return existing
            if len(cand_tokens) == 1 and len(existing_tokens) == 1:
                a, b = cand_tokens[0], existing_tokens[0]
                if min(len(a), len(b)) >= 4 and (a.startswith(b) or b.startswith(a)):
                    return existing
    return norm


def _resolve_owner_name(
    raw_owner: str | None,
    evidence_msgs: Sequence[ParsedMessage],
    participants: Sequence[str],
) -> str | None:
    """Normalize owner names, resolving first-person pronouns to message senders."""
    if raw_owner is None:
        return None
    stripped = raw_owner.strip()
    lowered = stripped.lower()
    if lowered in _UNASSIGNED_OWNER_TOKENS:
        return None

    if lowered in _FIRST_PERSON_TOKENS:
        for msg in evidence_msgs:
            if msg.sender:
                return msg.sender
        return None

    by_lower = {p.lower(): p for p in participants}
    if lowered in by_lower:
        return by_lower[lowered]
    return stripped


def _norm_text_for_compare(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value.lower())).strip()


def _text_token_overlap(a: str | None, b: str | None) -> float:
    ta = set(_norm_text_for_compare(a).split())
    tb = set(_norm_text_for_compare(b).split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), len(tb))


def _deduplicate_anchored_deltas(
    items: list[_AnchoredDelta], kind: str
) -> list[_AnchoredDelta]:
    """Deduplicate identical and overlapping-window extractions for one delta category."""
    kept: list[_AnchoredDelta] = []
    for item in items:
        merged = False
        item_ids = set(item.valid_ids)
        for idx, prev in enumerate(kept):
            if prev.canonical_key != item.canonical_key:
                continue
            prev_ids = set(prev.valid_ids)
            if not (prev_ids & item_ids):
                continue

            same_action = prev.delta.action == item.delta.action
            if kind == "decision" and not same_action:
                assert isinstance(prev.delta, DecisionDelta) and isinstance(
                    item.delta, DecisionDelta
                )
                same_resolved_value = _norm_text_for_compare(
                    prev.delta.value
                ) == _norm_text_for_compare(item.delta.value)
                both_decided = {prev.delta.action, item.delta.action} <= {
                    "confirmed",
                    "reversed",
                }
                same_anchor_equiv = (
                    {prev.delta.action, item.delta.action}
                    <= {"proposed", "confirmed", "reversed"}
                    and prev.anchor_max == item.anchor_max
                )
                if not (same_resolved_value and (both_decided or same_anchor_equiv)):
                    continue
            elif kind == "deadline" and not same_action:
                assert isinstance(prev.delta, DeadlineDelta) and isinstance(
                    item.delta, DeadlineDelta
                )
                same_date = _norm_text_for_compare(
                    prev.delta.date_text
                ) == _norm_text_for_compare(item.delta.date_text)
                same_anchor_deadline = (
                    {prev.delta.action, item.delta.action}
                    <= {"set", "changed", "rule_defined"}
                    and prev.anchor_max == item.anchor_max
                )
                if not (same_date and same_anchor_deadline):
                    continue
            elif not same_action:
                continue

            compatible = False
            if kind == "decision":
                assert isinstance(prev.delta, DecisionDelta) and isinstance(
                    item.delta, DecisionDelta
                )
                compatible = (
                    _norm_text_for_compare(prev.delta.value)
                    == _norm_text_for_compare(item.delta.value)
                    or _text_token_overlap(prev.delta.value, item.delta.value) >= 0.5
                    or prev.anchor_max == item.anchor_max
                )
            elif kind == "task":
                assert isinstance(prev.delta, TaskDelta) and isinstance(
                    item.delta, TaskDelta
                )
                owner_a = _norm_text_for_compare(prev.delta.owner)
                owner_b = _norm_text_for_compare(item.delta.owner)
                compatible = (
                    owner_a == owner_b
                    or (prev.anchor_max == item.anchor_max and (not owner_a or not owner_b))
                )
            elif kind == "deadline":
                assert isinstance(prev.delta, DeadlineDelta) and isinstance(
                    item.delta, DeadlineDelta
                )
                compatible = (
                    _norm_text_for_compare(prev.delta.date_text)
                    == _norm_text_for_compare(item.delta.date_text)
                    or _text_token_overlap(prev.delta.date_text, item.delta.date_text) >= 0.5
                    or prev.anchor_max == item.anchor_max
                )
            elif kind == "open_item":
                assert isinstance(prev.delta, OpenIssueDelta) and isinstance(
                    item.delta, OpenIssueDelta
                )
                compatible = (
                    _text_token_overlap(prev.delta.description, item.delta.description)
                    >= 0.4
                    or prev.anchor_max == item.anchor_max
                )

            if compatible:
                union_ids = tuple(dict.fromkeys([*prev.valid_ids, *item.valid_ids]))
                merged_delta = _merge_delta_fields(prev.delta, item.delta, union_ids)
                kept[idx] = _AnchoredDelta(
                    anchor_max=max(prev.anchor_max, item.anchor_max),
                    anchor_min=min(prev.anchor_min, item.anchor_min),
                    seq_order=prev.seq_order,
                    canonical_key=prev.canonical_key,
                    valid_ids=union_ids,
                    delta=merged_delta,
                )
                merged = True
                break
        if not merged:
            kept.append(item)
    return kept


def _merge_delta_fields(
    a: DecisionDelta | TaskDelta | DeadlineDelta | OpenIssueDelta,
    b: DecisionDelta | TaskDelta | DeadlineDelta | OpenIssueDelta,
    union_ids: tuple[str, ...],
) -> DecisionDelta | TaskDelta | DeadlineDelta | OpenIssueDelta:
    if isinstance(a, DecisionDelta) and isinstance(b, DecisionDelta):
        decision_rank = {
            "proposed": 1,
            "uncertain": 2,
            "dropped": 2,
            "confirmed": 3,
            "reversed": 4,
        }
        chosen_action = (
            a.action
            if decision_rank.get(a.action, 0) >= decision_rank.get(b.action, 0)
            else b.action
        )
        return a.model_copy(
            update={
                "action": chosen_action,
                "rationale": a.rationale or b.rationale,
                "evidence_msg_ids": list(union_ids),
            }
        )
    if isinstance(a, TaskDelta) and isinstance(b, TaskDelta):
        return a.model_copy(
            update={
                "owner": a.owner or b.owner,
                "status_note": a.status_note or b.status_note,
                "evidence_msg_ids": list(union_ids),
            }
        )
    if isinstance(a, DeadlineDelta) and isinstance(b, DeadlineDelta):
        deadline_rank = {"tentative": 1, "set": 2, "changed": 3, "rule_defined": 4}
        chosen_action = (
            a.action
            if deadline_rank.get(a.action, 0) >= deadline_rank.get(b.action, 0)
            else b.action
        )
        return a.model_copy(
            update={
                "action": chosen_action,
                "dependency_rule": a.dependency_rule or b.dependency_rule,
                "evidence_msg_ids": list(union_ids),
            }
        )
    assert isinstance(a, OpenIssueDelta) and isinstance(b, OpenIssueDelta)
    return a.model_copy(
        update={
            "owner": a.owner or b.owner,
            "description": a.description if len(a.description) >= len(b.description) else b.description,
            "evidence_msg_ids": list(union_ids),
        }
    )


def _build_evidence_refs(
    valid_ids: Sequence[str],
    msg_by_id: dict[str, ParsedMessage],
    msg_index_by_id: dict[str, int],
) -> list[EvidenceRef]:
    ordered_ids = sorted(valid_ids, key=lambda mid: msg_index_by_id[mid])
    refs: list[EvidenceRef] = []
    for mid in ordered_ids:
        msg = msg_by_id[mid]
        refs.append(
            EvidenceRef(
                msg_id=msg.msg_id,
                line_start=msg.line_start,
                line_end=msg.line_end,
                timestamp_raw=msg.timestamp_raw,
                timestamp_iso=msg.timestamp_iso,
                sender=msg.sender,
                quote=msg.text,
            )
        )
    return refs


def _merge_unique_evidence(
    existing: Sequence[EvidenceRef], incoming: Sequence[EvidenceRef]
) -> list[EvidenceRef]:
    by_id: dict[str, EvidenceRef] = {ref.msg_id: ref for ref in existing}
    for ref in incoming:
        if ref.msg_id not in by_id:
            by_id[ref.msg_id] = ref
    return sorted(by_id.values(), key=lambda ref: (ref.timestamp_iso, ref.line_start, ref.msg_id))


def _prepare_anchored_deltas(
    raw_deltas: Sequence[DecisionDelta | TaskDelta | DeadlineDelta | OpenIssueDelta],
    msg_by_id: dict[str, ParsedMessage],
    msg_index_by_id: dict[str, int],
    participants: Sequence[str],
    kind: str,
) -> list[_AnchoredDelta]:
    anchored: list[_AnchoredDelta] = []
    registered_keys: list[str] = []

    raw_items: list[tuple[int, int, int, tuple[str, ...], DecisionDelta | TaskDelta | DeadlineDelta | OpenIssueDelta]] = []
    for seq_idx, delta in enumerate(raw_deltas):
        valid_ids: list[str] = []
        seen_ids: set[str] = set()
        for mid in delta.evidence_msg_ids:
            cleaned_mid = mid.strip()
            if cleaned_mid in msg_by_id and cleaned_mid not in seen_ids:
                seen_ids.add(cleaned_mid)
                valid_ids.append(cleaned_mid)
        if not valid_ids:
            continue

        indices = [msg_index_by_id[mid] for mid in valid_ids]
        raw_items.append((max(indices), min(indices), seq_idx, tuple(valid_ids), delta))

    # Sort chronologically before canonical key registration so earlier keys win.
    raw_items.sort(key=lambda item: (item[0], item[1], item[2]))

    for anchor_max, anchor_min, seq_idx, valid_ids, delta in raw_items:
        evidence_msgs = [msg_by_id[mid] for mid in valid_ids]
        if isinstance(delta, DecisionDelta):
            raw_key = delta.topic_key
            norm_delta: DecisionDelta | TaskDelta | DeadlineDelta | OpenIssueDelta = delta
        elif isinstance(delta, TaskDelta):
            raw_key = delta.task_key
            resolved_owner = _resolve_owner_name(delta.owner, evidence_msgs, participants)
            if delta.action == "dropped_unassigned":
                resolved_owner = None
            norm_delta = delta.model_copy(update={"owner": resolved_owner})
        elif isinstance(delta, DeadlineDelta):
            raw_key = delta.item_key
            norm_delta = delta
        else:
            raw_key = delta.issue_key
            resolved_owner = _resolve_owner_name(delta.owner, evidence_msgs, participants)
            norm_delta = delta.model_copy(update={"owner": resolved_owner})

        canonical_key = resolve_canonical_key(raw_key, registered_keys)
        if canonical_key not in registered_keys:
            registered_keys.append(canonical_key)

        anchored.append(
            _AnchoredDelta(
                anchor_max=anchor_max,
                anchor_min=anchor_min,
                seq_order=seq_idx,
                canonical_key=canonical_key,
                valid_ids=valid_ids,
                delta=norm_delta,
            )
        )

    deduped = _deduplicate_anchored_deltas(anchored, kind)
    deduped.sort(key=lambda item: (item.anchor_max, item.anchor_min, item.seq_order))
    return deduped


def _append_history_entry(
    history: list[StateHistoryEntry], entry: StateHistoryEntry
) -> None:
    if history:
        last = history[-1]
        if (
            last.timestamp_iso == entry.timestamp_iso
            and last.action == entry.action
            and _norm_text_for_compare(last.value) == _norm_text_for_compare(entry.value)
            and _norm_text_for_compare(last.owner) == _norm_text_for_compare(entry.owner)
        ):
            history[-1] = last.model_copy(
                update={
                    "note": last.note or entry.note,
                    "evidence": _merge_unique_evidence(last.evidence, entry.evidence),
                }
            )
            return
    history.append(entry)


def _reduce_decisions(
    anchored: list[_AnchoredDelta],
    msg_by_id: dict[str, ParsedMessage],
    msg_index_by_id: dict[str, int],
) -> list[DecisionRecord]:
    records: dict[str, DecisionRecord] = {}
    for item in anchored:
        delta = item.delta
        assert isinstance(delta, DecisionDelta)
        anchor_msg = msg_by_id[
            max(item.valid_ids, key=lambda mid: msg_index_by_id[mid])
        ]
        refs = _build_evidence_refs(item.valid_ids, msg_by_id, msg_index_by_id)
        history_entry = StateHistoryEntry(
            timestamp_raw=anchor_msg.timestamp_raw,
            timestamp_iso=anchor_msg.timestamp_iso,
            action=delta.action,
            value=delta.value,
            note=delta.rationale,
            evidence=refs,
        )

        existing = records.get(item.canonical_key)
        if existing is None:
            status: DecisionStatus = delta.action
            records[item.canonical_key] = DecisionRecord(
                topic_key=item.canonical_key,
                title=delta.title,
                current_value=delta.value,
                status=status,
                history=[history_entry],
                evidence=refs,
            )
            continue

        same_value = _norm_text_for_compare(existing.current_value) == _norm_text_for_compare(
            delta.value
        )
        if same_value and existing.history:
            last = existing.history[-1]
            if (
                last.action in ("confirmed", "reversed")
                and delta.action in ("confirmed", "reversed")
                and (
                    last.timestamp_iso == history_entry.timestamp_iso
                    or {r.msg_id for r in last.evidence} & {r.msg_id for r in refs}
                )
            ):
                merged_action: DecisionStatus = (
                    "confirmed"
                    if "confirmed" in (last.action, delta.action)
                    else delta.action
                )
                existing.history[-1] = last.model_copy(
                    update={
                        "action": merged_action,
                        "note": last.note or history_entry.note,
                        "evidence": _merge_unique_evidence(last.evidence, refs),
                    }
                )
                existing.status = merged_action
                existing.evidence = _merge_unique_evidence(existing.evidence, refs)
                continue

        _append_history_entry(existing.history, history_entry)
        new_status: DecisionStatus = delta.action
        if delta.action == "proposed" and existing.status == "confirmed" and same_value:
            new_status = "confirmed"

        new_evidence = (
            _merge_unique_evidence(existing.evidence, refs) if same_value else refs
        )
        existing.current_value = delta.value
        existing.status = new_status
        existing.evidence = new_evidence

    return list(records.values())


def _reduce_tasks(
    anchored: list[_AnchoredDelta],
    msg_by_id: dict[str, ParsedMessage],
    msg_index_by_id: dict[str, int],
) -> list[TaskRecord]:
    records: dict[str, TaskRecord] = {}
    for item in anchored:
        delta = item.delta
        assert isinstance(delta, TaskDelta)
        anchor_msg = msg_by_id[
            max(item.valid_ids, key=lambda mid: msg_index_by_id[mid])
        ]
        refs = _build_evidence_refs(item.valid_ids, msg_by_id, msg_index_by_id)
        history_entry = StateHistoryEntry(
            timestamp_raw=anchor_msg.timestamp_raw,
            timestamp_iso=anchor_msg.timestamp_iso,
            action=delta.action,
            value=delta.status_note or delta.action,
            owner=delta.owner,
            note=delta.status_note,
            evidence=refs,
        )

        existing = records.get(item.canonical_key)
        if existing is None:
            owner = None if delta.action == "dropped_unassigned" else delta.owner
            status = _initial_task_status(delta.action, owner)
            records[item.canonical_key] = TaskRecord(
                task_key=item.canonical_key,
                title=delta.title,
                current_owner=owner,
                status=status,
                status_note=(delta.status_note or "").strip(),
                history=[history_entry],
                evidence=refs,
            )
            continue

        _append_history_entry(existing.history, history_entry)
        _apply_task_delta(existing, delta, refs, anchor_iso=anchor_msg.timestamp_iso)

    return list(records.values())


def _initial_task_status(action: str, owner: str | None) -> TaskStatus:
    if action == "dropped_unassigned":
        return "unassigned"
    if action in ("assigned", "reassigned"):
        return "assigned" if owner is not None else "unassigned"
    if action == "progress_updated":
        return "in_progress"
    if action == "promised_eta":
        return "unconfirmed"
    if action == "blocked":
        return "blocked"
    if action == "completed":
        return "completed"
    return "unconfirmed"


_REOPEN_NOTE_RE = re.compile(
    r"\b(?:remaining|pending|left|incomplete|unfinished|reopen(?:ed)?|wip|todo|bacha|baaki)\b"
    r"|\b(?:[1-9]?\d)%",
    re.IGNORECASE,
)


def _apply_task_delta(
    record: TaskRecord,
    delta: TaskDelta,
    refs: list[EvidenceRef],
    *,
    anchor_iso: str,
) -> None:
    action = delta.action
    note = (delta.status_note or "").strip()

    if action == "dropped_unassigned":
        record.current_owner = None
        record.status = "unassigned"
        record.status_note = note
        record.evidence = refs
        return

    if action in ("assigned", "reassigned"):
        record.current_owner = delta.owner
        record.status = "assigned" if delta.owner is not None else "unassigned"
        record.status_note = note
        record.evidence = refs
        return

    if action == "completed":
        if delta.owner is not None:
            record.current_owner = delta.owner
        record.status = "completed"
        record.status_note = note
        record.evidence = refs
        return

    if action == "blocked":
        if record.current_owner is None and delta.owner is not None:
            record.current_owner = delta.owner
        if record.status != "completed":
            record.status = "blocked"
        if note:
            record.status_note = note
        record.evidence = refs
        return

    if action == "progress_updated":
        if record.current_owner is None and delta.owner is not None:
            record.current_owner = delta.owner
        later_than_last = bool(
            record.history and anchor_iso > record.history[-2].timestamp_iso
            if len(record.history) >= 2
            else False
        )
        explicitly_reopened = later_than_last and bool(_REOPEN_NOTE_RE.search(note))
        if record.status != "completed" or explicitly_reopened:
            record.status = "in_progress"
        if note:
            record.status_note = note
        record.evidence = _merge_unique_evidence(record.evidence, refs)
        return

    if action == "promised_eta":
        if record.current_owner is None and delta.owner is not None:
            record.current_owner = delta.owner
        if record.status != "completed":
            record.status = "unconfirmed"
        if note:
            record.status_note = note
        record.evidence = refs


def _reduce_deadlines(
    anchored: list[_AnchoredDelta],
    msg_by_id: dict[str, ParsedMessage],
    msg_index_by_id: dict[str, int],
) -> list[DeadlineRecord]:
    records: dict[str, DeadlineRecord] = {}
    for item in anchored:
        delta = item.delta
        assert isinstance(delta, DeadlineDelta)
        anchor_msg = msg_by_id[
            max(item.valid_ids, key=lambda mid: msg_index_by_id[mid])
        ]
        refs = _build_evidence_refs(item.valid_ids, msg_by_id, msg_index_by_id)
        history_entry = StateHistoryEntry(
            timestamp_raw=anchor_msg.timestamp_raw,
            timestamp_iso=anchor_msg.timestamp_iso,
            action=delta.action,
            value=delta.date_text,
            note=delta.dependency_rule,
            evidence=refs,
        )
        status: DeadlineStatus = (
            "tentative"
            if delta.action == "tentative"
            else "rule_based"
            if delta.action == "rule_defined"
            else "confirmed"
        )

        existing = records.get(item.canonical_key)
        if existing is None:
            records[item.canonical_key] = DeadlineRecord(
                item_key=item.canonical_key,
                title=delta.title,
                current_date_text=delta.date_text,
                status=status,
                dependency_rule=delta.dependency_rule,
                history=[history_entry],
                evidence=refs,
            )
            continue

        _append_history_entry(existing.history, history_entry)
        existing.current_date_text = delta.date_text
        existing.status = status
        if delta.dependency_rule:
            existing.dependency_rule = delta.dependency_rule
        existing.evidence = refs

    return list(records.values())


def _reduce_open_items(
    anchored: list[_AnchoredDelta],
    msg_by_id: dict[str, ParsedMessage],
    msg_index_by_id: dict[str, int],
) -> list[OpenItemRecord]:
    records: dict[str, OpenItemRecord] = {}
    for item in anchored:
        delta = item.delta
        assert isinstance(delta, OpenIssueDelta)
        anchor_msg = msg_by_id[
            max(item.valid_ids, key=lambda mid: msg_index_by_id[mid])
        ]
        refs = _build_evidence_refs(item.valid_ids, msg_by_id, msg_index_by_id)
        history_entry = StateHistoryEntry(
            timestamp_raw=anchor_msg.timestamp_raw,
            timestamp_iso=anchor_msg.timestamp_iso,
            action=delta.action,
            value=delta.description,
            owner=delta.owner,
            evidence=refs,
        )
        status: OpenItemStatus = (
            "resolved"
            if delta.action == "resolved"
            else "deferred"
            if delta.action == "deferred"
            else "open"
        )

        existing = records.get(item.canonical_key)
        if existing is None:
            records[item.canonical_key] = OpenItemRecord(
                issue_key=item.canonical_key,
                title=delta.title,
                status=status,
                description=delta.description,
                owner=delta.owner,
                history=[history_entry],
                evidence=refs,
            )
            continue

        _append_history_entry(existing.history, history_entry)
        existing.status = status
        if delta.description.strip():
            existing.description = delta.description.strip()
        if delta.owner is not None:
            existing.owner = delta.owner
        existing.evidence = refs

    return list(records.values())


def reduce_deltas_to_state(
    chat: ParsedChat,
    deltas: Sequence[WindowExtractionResult] | WindowExtractionResult,
    *,
    chat_sha256: str | None = None,
    schema_version: str = SCHEMA_VERSION,
) -> ProjectState:
    """Deterministically merge chronological deltas into a grounded ProjectState."""
    batch_list = [deltas] if isinstance(deltas, WindowExtractionResult) else list(deltas)
    participants = extract_participants(chat)
    msg_by_id = {m.msg_id: m for m in chat.messages}
    msg_index_by_id = {m.msg_id: idx for idx, m in enumerate(chat.messages)}

    raw_decisions: list[DecisionDelta] = []
    raw_tasks: list[TaskDelta] = []
    raw_deadlines: list[DeadlineDelta] = []
    raw_open_items: list[OpenIssueDelta] = []

    for batch in batch_list:
        raw_decisions.extend(batch.decisions)
        raw_tasks.extend(batch.tasks)
        raw_deadlines.extend(batch.deadlines)
        raw_open_items.extend(batch.open_items)

    anchored_decisions = _prepare_anchored_deltas(
        raw_decisions, msg_by_id, msg_index_by_id, participants, "decision"
    )
    anchored_tasks = _prepare_anchored_deltas(
        raw_tasks, msg_by_id, msg_index_by_id, participants, "task"
    )
    anchored_deadlines = _prepare_anchored_deltas(
        raw_deadlines, msg_by_id, msg_index_by_id, participants, "deadline"
    )
    anchored_open_items = _prepare_anchored_deltas(
        raw_open_items, msg_by_id, msg_index_by_id, participants, "open_item"
    )

    last_msg = chat.messages[-1] if chat.messages else None
    resolved_sha = chat_sha256 or compute_chat_sha256(chat)

    return ProjectState(
        source_path=chat.source_path,
        chat_sha256=resolved_sha,
        schema_version=schema_version,
        date_order=chat.date_order,
        total_messages=len(chat.messages),
        last_message_timestamp=last_msg.timestamp_raw if last_msg else None,
        last_message_timestamp_iso=last_msg.timestamp_iso if last_msg else None,
        participants=participants,
        decisions=_reduce_decisions(anchored_decisions, msg_by_id, msg_index_by_id),
        tasks=_reduce_tasks(anchored_tasks, msg_by_id, msg_index_by_id),
        deadlines=_reduce_deadlines(anchored_deadlines, msg_by_id, msg_index_by_id),
        open_items=_reduce_open_items(anchored_open_items, msg_by_id, msg_index_by_id),
    )


class _CanonicalRegistry:
    """Forward-passing registry of canonical keys and latest values across windows."""

    def __init__(self, participants: Sequence[str]) -> None:
        self.participants = list(participants)
        self.decisions: dict[str, str] = {}
        self.tasks: dict[str, str] = {}
        self.deadlines: dict[str, str] = {}
        self.open_items: dict[str, str] = {}

    def format_for_prompt(self) -> str:
        """Render active canonical keys so subsequent windows reuse identical keys."""
        lines = [
            f"Known participants: {', '.join(self.participants) if self.participants else 'none'}"
        ]
        lines.append("Active canonical keys from earlier windows (REUSE exact keys):")
        lines.append(self._format_section("decisions", self.decisions))
        lines.append(self._format_section("tasks", self.tasks))
        lines.append(self._format_section("deadlines", self.deadlines))
        lines.append(self._format_section("open_items", self.open_items))
        return "\n".join(lines)

    @staticmethod
    def _format_section(label: str, items: dict[str, str]) -> str:
        if not items:
            return f"  {label}: (none yet)"
        rendered = ", ".join(f"{k} [{summary}]" for k, summary in items.items())
        return f"  {label}: {rendered}"

    def absorb_window_result(
        self,
        result: WindowExtractionResult,
        window_msg_by_id: dict[str, ParsedMessage],
    ) -> WindowExtractionResult:
        """Restrict citations to window IDs, canonicalize keys, and update the registry."""
        valid_ids = set(window_msg_by_id)

        def _clean_ids(raw_ids: Sequence[str]) -> list[str]:
            return list(
                dict.fromkeys(
                    mid.strip() for mid in raw_ids if mid.strip() in valid_ids
                )
            )

        decisions: list[DecisionDelta] = []
        for delta in result.decisions:
            ids = _clean_ids(delta.evidence_msg_ids)
            if not ids:
                continue
            key = resolve_canonical_key(delta.topic_key, list(self.decisions))
            self.decisions[key] = f"{delta.action}: {delta.value}"
            decisions.append(delta.model_copy(update={"topic_key": key, "evidence_msg_ids": ids}))

        tasks: list[TaskDelta] = []
        for delta in result.tasks:
            ids = _clean_ids(delta.evidence_msg_ids)
            if not ids:
                continue
            key = resolve_canonical_key(delta.task_key, list(self.tasks))
            msgs = [window_msg_by_id[mid] for mid in ids]
            owner = _resolve_owner_name(delta.owner, msgs, self.participants)
            if delta.action == "dropped_unassigned":
                owner = None
            owner_label = owner or "unassigned"
            self.tasks[key] = f"{delta.action} ({owner_label})"
            tasks.append(
                delta.model_copy(
                    update={"task_key": key, "owner": owner, "evidence_msg_ids": ids}
                )
            )

        deadlines: list[DeadlineDelta] = []
        for delta in result.deadlines:
            ids = _clean_ids(delta.evidence_msg_ids)
            if not ids:
                continue
            key = resolve_canonical_key(delta.item_key, list(self.deadlines))
            rule_part = f"; rule={delta.dependency_rule}" if delta.dependency_rule else ""
            self.deadlines[key] = f"{delta.date_text} ({delta.action}{rule_part})"
            deadlines.append(
                delta.model_copy(update={"item_key": key, "evidence_msg_ids": ids})
            )

        open_items: list[OpenIssueDelta] = []
        for delta in result.open_items:
            ids = _clean_ids(delta.evidence_msg_ids)
            if not ids:
                continue
            key = resolve_canonical_key(delta.issue_key, list(self.open_items))
            msgs = [window_msg_by_id[mid] for mid in ids]
            owner = _resolve_owner_name(delta.owner, msgs, self.participants)
            self.open_items[key] = f"{delta.action}"
            open_items.append(
                delta.model_copy(
                    update={"issue_key": key, "owner": owner, "evidence_msg_ids": ids}
                )
            )

        return WindowExtractionResult(
            decisions=decisions,
            tasks=tasks,
            deadlines=deadlines,
            open_items=open_items,
        )


def _build_window_prompt(
    window: Sequence[ParsedMessage],
    *,
    window_index: int,
    total_windows: int,
    date_order: str,
    registry: _CanonicalRegistry,
) -> str:
    transcript = format_messages_window(window)
    valid_id_range = f"{window[0].msg_id} to {window[-1].msg_id}"
    return (
        f"Window {window_index + 1} of {total_windows} "
        f"(detected date order: {date_order}, valid msg_ids in this window: {valid_id_range}).\n\n"
        f"{registry.format_for_prompt()}\n\n"
        "Chronological chat messages:\n"
        f"{transcript}\n\n"
        "Extract all decisions, tasks, deadlines, and open items updated in this window. "
        "Split any compound 'A + B' or 'A and B' task assignments into separate atomic TaskDelta entries. "
        "Keep proposals 'uncertain' or 'proposed' if the only positive reply is an ambiguous short word interleaved with a separate topic."
    )


def compile_project_state_with_meta(
    chat: ParsedChat,
    adapter: BaseLLMAdapter,
    *,
    raw_text: str | None = None,
    window_size: int = 25,
    overlap: int = 6,
    cache_dir: Path | None = None,
    rebuild: bool = False,
) -> tuple[ProjectState, bool, Path]:
    """Compile ProjectState from ParsedChat with SHA-256 caching and return cache metadata."""
    resolved_cache_dir = cache_dir if cache_dir is not None else DEFAULT_CACHE_DIR
    chat_sha256 = compute_chat_sha256(chat, raw_text=raw_text)
    cache_key = compute_cache_key(
        chat_sha256=chat_sha256,
        schema_version=SCHEMA_VERSION,
        provider=adapter.provider_name,
        chat_model=adapter.chat_model,
        window_size=window_size,
        overlap=overlap,
    )
    cache_path = resolved_cache_dir / f"project_state_{cache_key[:24]}.json"

    if not rebuild and cache_path.is_file():
        try:
            cached = ProjectState.model_validate_json(
                cache_path.read_text(encoding="utf-8")
            )
            return cached, True, cache_path
        except ValueError:
            pass

    windows = build_message_windows(
        chat.messages, window_size=window_size, overlap=overlap
    )
    registry = _CanonicalRegistry(extract_participants(chat))
    extracted_windows: list[WindowExtractionResult] = []

    for idx, window in enumerate(windows):
        prompt = _build_window_prompt(
            window,
            window_index=idx,
            total_windows=len(windows),
            date_order=chat.date_order,
            registry=registry,
        )
        raw_result = adapter.generate_structured(
            prompt,
            WindowExtractionResult,
            system_prompt=_EXTRACTION_SYSTEM_PROMPT,
            temperature=0.0,
        )
        window_msg_by_id = {m.msg_id: m for m in window}
        extracted_windows.append(registry.absorb_window_result(raw_result, window_msg_by_id))

    state = reduce_deltas_to_state(
        chat,
        extracted_windows,
        chat_sha256=chat_sha256,
        schema_version=SCHEMA_VERSION,
    )

    resolved_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(state.model_dump_json(indent=2), encoding="utf-8")
    return state, False, cache_path


def compile_project_state(
    chat: ParsedChat,
    adapter: BaseLLMAdapter,
    *,
    raw_text: str | None = None,
    window_size: int = 25,
    overlap: int = 6,
    cache_dir: Path | None = None,
    rebuild: bool = False,
) -> ProjectState:
    """Compile ProjectState from ParsedChat using sliding-window extraction and caching."""
    state, _, _ = compile_project_state_with_meta(
        chat,
        adapter,
        raw_text=raw_text,
        window_size=window_size,
        overlap=overlap,
        cache_dir=cache_dir,
        rebuild=rebuild,
    )
    return state
