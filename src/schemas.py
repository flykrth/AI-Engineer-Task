"""Typed delta and compiled project state schemas for temporal chat compilation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from src.parser import DateOrder

SCHEMA_VERSION = "1.0"

DecisionAction = Literal["proposed", "confirmed", "reversed", "dropped", "uncertain"]
TaskAction = Literal[
    "assigned",
    "reassigned",
    "dropped_unassigned",
    "progress_updated",
    "completed",
    "blocked",
    "promised_eta",
]
DeadlineAction = Literal["set", "changed", "rule_defined", "tentative"]
OpenIssueAction = Literal["opened", "resolved", "deferred"]

DecisionStatus = Literal["confirmed", "proposed", "reversed", "dropped", "uncertain"]
TaskStatus = Literal[
    "assigned",
    "in_progress",
    "completed",
    "unassigned",
    "blocked",
    "unconfirmed",
]
DeadlineStatus = Literal["confirmed", "tentative", "rule_based", "superseded"]
OpenItemStatus = Literal["open", "deferred", "resolved"]


class EvidenceRef(BaseModel):
    """Grounded citation pointing to an exact parsed WhatsApp message."""

    msg_id: str
    line_start: int
    line_end: int
    timestamp_raw: str
    timestamp_iso: str
    sender: str | None = None
    quote: str


class DecisionDelta(BaseModel):
    """Single chronological event affecting a project decision."""

    topic_key: str = Field(
        description="Canonical lowercase snake_case key identifying the decision topic."
    )
    title: str = Field(description="Short human-readable title for the decision topic.")
    action: DecisionAction
    value: str = Field(description="Proposed, confirmed, reversed, or dropped option.")
    rationale: str | None = Field(
        default=None,
        description="Optional short reason stated in the chat for this update.",
    )
    evidence_msg_ids: list[str] = Field(
        default_factory=list,
        description="Message IDs from the current window supporting this delta.",
    )


class TaskDelta(BaseModel):
    """Single chronological event affecting task ownership or progress."""

    task_key: str = Field(
        description="Canonical lowercase snake_case key identifying the task."
    )
    title: str = Field(description="Short human-readable task title.")
    action: TaskAction
    owner: str | None = Field(
        default=None,
        description="Resolved participant name, or null when dropped or unassigned.",
    )
    status_note: str | None = Field(
        default=None,
        description="Optional concise progress, blocker, or ETA note from the messages.",
    )
    evidence_msg_ids: list[str] = Field(
        default_factory=list,
        description="Message IDs from the current window supporting this delta.",
    )


class DeadlineDelta(BaseModel):
    """Single chronological event affecting a deadline or scheduling dependency."""

    item_key: str = Field(
        description="Canonical lowercase snake_case key identifying the milestone or deliverable."
    )
    title: str = Field(description="Short human-readable milestone title.")
    action: DeadlineAction
    date_text: str = Field(
        description="Date or timing text stated in the chat (e.g., '26 Nov' or 'by Sunday')."
    )
    dependency_rule: str | None = Field(
        default=None,
        description="Optional relative rule linking this deadline to another milestone.",
    )
    evidence_msg_ids: list[str] = Field(
        default_factory=list,
        description="Message IDs from the current window supporting this delta.",
    )


class OpenIssueDelta(BaseModel):
    """Single chronological event opening, deferring, or resolving an open question or issue."""

    issue_key: str = Field(
        description="Canonical lowercase snake_case key identifying the open issue or question."
    )
    title: str = Field(description="Short human-readable issue title.")
    action: OpenIssueAction
    description: str = Field(
        description="Concise summary of the unresolved item, agenda point, or decision gap."
    )
    owner: str | None = Field(
        default=None,
        description="Optional participant responsible for or raising the item.",
    )
    evidence_msg_ids: list[str] = Field(
        default_factory=list,
        description="Message IDs from the current window supporting this delta.",
    )


class WindowExtractionResult(BaseModel):
    """Structured delta extraction output produced by the LLM for one chat window."""

    decisions: list[DecisionDelta] = Field(default_factory=list)
    tasks: list[TaskDelta] = Field(default_factory=list)
    deadlines: list[DeadlineDelta] = Field(default_factory=list)
    open_items: list[OpenIssueDelta] = Field(default_factory=list)


class StateHistoryEntry(BaseModel):
    """Chronological revision entry recorded inside a compiled state record."""

    timestamp_raw: str
    timestamp_iso: str
    action: str
    value: str
    owner: str | None = None
    note: str | None = None
    evidence: list[EvidenceRef] = Field(default_factory=list)


class DecisionRecord(BaseModel):
    """Compiled current state and revision history for one decision topic."""

    topic_key: str
    title: str
    current_value: str
    status: DecisionStatus
    history: list[StateHistoryEntry] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)


class TaskRecord(BaseModel):
    """Compiled current ownership, status, and history for one task."""

    task_key: str
    title: str
    current_owner: str | None = None
    status: TaskStatus
    status_note: str = ""
    history: list[StateHistoryEntry] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)


class DeadlineRecord(BaseModel):
    """Compiled current date, dependency rule, and history for one deadline."""

    item_key: str
    title: str
    current_date_text: str
    status: DeadlineStatus
    dependency_rule: str | None = None
    history: list[StateHistoryEntry] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)


class OpenItemRecord(BaseModel):
    """Compiled current state and history for one open question or unresolved issue."""

    issue_key: str
    title: str
    status: OpenItemStatus
    description: str
    owner: str | None = None
    history: list[StateHistoryEntry] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)


class ProjectState(BaseModel):
    """Complete event-sourced project state compiled from a WhatsApp chat export."""

    source_path: str | None = None
    chat_sha256: str
    schema_version: str = SCHEMA_VERSION
    date_order: DateOrder = "DD/MM"
    total_messages: int
    last_message_timestamp: str | None = None
    last_message_timestamp_iso: str | None = None
    participants: list[str] = Field(default_factory=list)
    decisions: list[DecisionRecord] = Field(default_factory=list)
    tasks: list[TaskRecord] = Field(default_factory=list)
    deadlines: list[DeadlineRecord] = Field(default_factory=list)
    open_items: list[OpenItemRecord] = Field(default_factory=list)


class DraftAnswer(BaseModel):
    """Unverified LLM draft answer citing only raw message IDs."""

    reasoning_trace: str = Field(
        description="Step-by-step chronological check of ProjectState history and raw evidence blocks."
    )
    supported: bool = Field(
        description="True when the chat contains verified information answering the question; False when missing, deleted, or post-chat."
    )
    partial_status_only: bool = Field(
        default=False,
        description="True when the task/topic was discussed but completion was never confirmed before the chat ended.",
    )
    answer: str = Field(
        description="Grounded English or Hinglish answer, or 'the chat doesn't say' when unsupported."
    )
    evidence_msg_ids: list[str] = Field(
        default_factory=list,
        description="List of exact msg_id strings from the chat (e.g. ['M017', 'M042']). Never include timestamps or line numbers.",
    )


class VerifiedAnswer(BaseModel):
    """Deterministically verified answer with grounded EvidenceRef citations."""

    answer: str
    supported: bool
    partial_status_only: bool = False
    evidence: list[EvidenceRef] = Field(default_factory=list)
    verification_status: str
    rejected_msg_ids: list[str] = Field(default_factory=list)
