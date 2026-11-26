"""Unit tests for grounded Q&A synthesis and deterministic citation verification."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from src.llm import BaseLLMAdapter, LLMConfig
from src.parser import ParsedChat, parse_chat_text
from src.qa import answer_and_verify, format_state_for_qa
from src.retriever import RetrievalResult
from src.schemas import (
    DeadlineRecord,
    DecisionRecord,
    DraftAnswer,
    EvidenceRef,
    OpenItemRecord,
    ProjectState,
    StateHistoryEntry,
    TaskRecord,
)
from src.verifier import ABSTENTION_ANSWER


class ScriptedDraftAdapter(BaseLLMAdapter):
    """Fake adapter returning scripted DraftAnswer responses in sequence."""

    def __init__(self, drafts: Sequence[DraftAnswer]) -> None:
        super().__init__(
            LLMConfig(
                provider="gemini",
                api_key="test-key",
                chat_model="gemini-3.5-flash-lite",
                embed_model="gemini-embedding-001",
            )
        )
        self._drafts = list(drafts)
        self.prompts: list[str] = []

    def _is_transient_error(self, exc: Exception) -> bool:
        return False

    def _raw_generate_text(
        self, prompt: str, *, system_prompt: str | None, temperature: float
    ) -> str:
        return ""

    def _raw_generate_structured(
        self,
        prompt: str,
        schema: type[Any],
        *,
        system_prompt: str | None,
        temperature: float,
    ) -> tuple[Any, str]:
        idx = min(len(self.prompts), len(self._drafts) - 1)
        self.prompts.append(prompt)
        draft = self._drafts[idx]
        return draft, draft.model_dump_json()

    def _raw_embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]


def _sample_chat_and_state() -> tuple[ParsedChat, ProjectState, RetrievalResult]:
    raw = "\n".join(
        [
            "01/10/26, 09:00 AM - Alice: Billing gateway assigned to Maya using Stripe.",
            "02/10/26, 10:15 AM - Maya: Switch billing to WireTransfer; Stripe needs business docs.",
            "03/10/26, 11:30 AM - Liam: Stripe sandbox works! I'll take billing now.",
            "04/10/26, 01:00 PM - Alice: This message was deleted",
            "05/10/26, 04:00 PM - Bob: Auth module done, merged to main. Analytics 60%.",
            "06/10/26, 05:30 PM - Alice: Deploying service on CloudHost, ready by 12 Oct.",
            "07/10/26, 06:45 PM - Maya: Export task still unassigned, nobody picked it up.",
        ]
    )
    chat = parse_chat_text(raw)
    refs_by_id = {
        m.msg_id: EvidenceRef(
            msg_id=m.msg_id,
            line_start=m.line_start,
            line_end=m.line_end,
            timestamp_raw=m.timestamp_raw,
            timestamp_iso=m.timestamp_iso,
            sender=m.sender,
            quote=m.text,
        )
        for m in chat.messages
    }

    state = ProjectState(
        chat_sha256="dummy_sha256",
        total_messages=len(chat.messages),
        last_message_timestamp=chat.messages[-1].timestamp_raw,
        last_message_timestamp_iso=chat.messages[-1].timestamp_iso,
        participants=["Alice", "Maya", "Liam", "Bob"],
        decisions=[
            DecisionRecord(
                topic_key="billing_method",
                title="Billing method",
                current_value="Stripe sandbox",
                status="confirmed",
                history=[
                    StateHistoryEntry(
                        timestamp_raw=chat.messages[0].timestamp_raw,
                        timestamp_iso=chat.messages[0].timestamp_iso,
                        action="proposed",
                        value="Stripe",
                        evidence=[refs_by_id["M001"]],
                    ),
                    StateHistoryEntry(
                        timestamp_raw=chat.messages[1].timestamp_raw,
                        timestamp_iso=chat.messages[1].timestamp_iso,
                        action="reversed",
                        value="WireTransfer",
                        evidence=[refs_by_id["M002"]],
                    ),
                    StateHistoryEntry(
                        timestamp_raw=chat.messages[2].timestamp_raw,
                        timestamp_iso=chat.messages[2].timestamp_iso,
                        action="reversed",
                        value="Stripe sandbox",
                        evidence=[refs_by_id["M003"]],
                    ),
                ],
                evidence=[refs_by_id["M003"]],
            )
        ],
        tasks=[
            TaskRecord(
                task_key="auth",
                title="Auth module",
                current_owner="Bob",
                status="completed",
                status_note="merged to main",
                history=[
                    StateHistoryEntry(
                        timestamp_raw=chat.messages[4].timestamp_raw,
                        timestamp_iso=chat.messages[4].timestamp_iso,
                        action="completed",
                        value="merged to main",
                        owner="Bob",
                        evidence=[refs_by_id["M005"]],
                    )
                ],
                evidence=[refs_by_id["M005"]],
            ),
            TaskRecord(
                task_key="analytics",
                title="Analytics",
                current_owner="Bob",
                status="in_progress",
                status_note="60%",
                evidence=[refs_by_id["M005"]],
            ),
            TaskRecord(
                task_key="cloudhost_deployment",
                title="CloudHost deployment",
                current_owner="Alice",
                status="unconfirmed",
                status_note="ready by 12 Oct",
                evidence=[refs_by_id["M006"]],
            ),
            TaskRecord(
                task_key="export",
                title="Export task",
                current_owner=None,
                status="unassigned",
                status_note="nobody picked it up",
                evidence=[refs_by_id["M007"]],
            ),
        ],
        deadlines=[
            DeadlineRecord(
                item_key="launch_pack",
                title="Launch pack",
                current_date_text="10 Oct",
                status="rule_based",
                dependency_rule="2 days before demo",
                evidence=[refs_by_id["M006"]],
            )
        ],
        open_items=[
            OpenItemRecord(
                issue_key="export_owner",
                title="Export ownership",
                status="open",
                description="Export task still unassigned",
                evidence=[refs_by_id["M007"]],
            )
        ],
    )
    retrieval = RetrievalResult(
        query="test",
        formatted_context="\n".join(
            f"[{m.msg_id} | L{m.line_start}-L{m.line_end} | {m.timestamp_raw} | {m.sender}] {m.text}"
            for m in chat.messages
        ),
    )
    return chat, state, retrieval


def test_valid_msg_ids_resolve_exact_parsed_chat_records_and_strip_unknowns() -> None:
    chat, state, retrieval = _sample_chat_and_state()
    adapter = ScriptedDraftAdapter(
        [
            DraftAnswer(
                reasoning_trace="Liam took billing with Stripe sandbox in M003; M999 is invalid.",
                supported=True,
                partial_status_only=False,
                answer="Liam owns billing using Stripe sandbox.",
                evidence_msg_ids=["M003", "M999", "M003"],
            )
        ]
    )

    verified = answer_and_verify(
        "Who owns billing and what method?", chat, state, retrieval, adapter
    )

    assert len(adapter.prompts) == 1
    assert verified.supported is True
    assert verified.verification_status == "verified_with_stripped_ids"
    assert verified.rejected_msg_ids == ["M999"]
    assert len(verified.evidence) == 1
    ref = verified.evidence[0]
    assert ref.msg_id == "M003"
    assert ref.line_start == 3
    assert ref.line_end == 3
    assert ref.timestamp_raw == "03/10/26, 11:30 AM"
    assert ref.sender == "Liam"
    assert ref.quote == "Stripe sandbox works! I'll take billing now."


def test_fabricated_msg_ids_trigger_one_retry_then_force_abstention() -> None:
    chat, state, retrieval = _sample_chat_and_state()

    # Case A: First draft has fabricated ID, corrective retry succeeds with real ID.
    adapter_recovered = ScriptedDraftAdapter(
        [
            DraftAnswer(
                reasoning_trace="Guessed M888.",
                supported=True,
                partial_status_only=False,
                answer="Liam owns billing.",
                evidence_msg_ids=["M888"],
            ),
            DraftAnswer(
                reasoning_trace="Corrected to real M003.",
                supported=True,
                partial_status_only=False,
                answer="Liam owns billing using Stripe sandbox.",
                evidence_msg_ids=["M003"],
            ),
        ]
    )
    res_recovered = answer_and_verify(
        "Who owns billing?", chat, state, retrieval, adapter_recovered
    )
    assert len(adapter_recovered.prompts) == 2
    assert "empty_citations" in adapter_recovered.prompts[1]
    assert res_recovered.supported is True
    assert res_recovered.verification_status == "verified_after_retry"
    assert res_recovered.rejected_msg_ids == ["M888"]
    assert [r.msg_id for r in res_recovered.evidence] == ["M003"]

    # Case B: Both initial and retry drafts return fabricated IDs -> forced abstention after 1 retry.
    adapter_persistent_bad = ScriptedDraftAdapter(
        [
            DraftAnswer(
                reasoning_trace="Bad ID 1.",
                supported=True,
                partial_status_only=False,
                answer="Liam owns billing.",
                evidence_msg_ids=["M888"],
            ),
            DraftAnswer(
                reasoning_trace="Bad ID 2.",
                supported=True,
                partial_status_only=False,
                answer="Liam owns billing.",
                evidence_msg_ids=["M777"],
            ),
            DraftAnswer(
                reasoning_trace="Should never reach a 3rd attempt.",
                supported=True,
                partial_status_only=False,
                answer="Should not be called.",
                evidence_msg_ids=["M003"],
            ),
        ]
    )
    res_forced = answer_and_verify(
        "Who owns billing?", chat, state, retrieval, adapter_persistent_bad
    )
    assert len(adapter_persistent_bad.prompts) == 2
    assert res_forced.supported is False
    assert res_forced.answer == ABSTENTION_ANSWER
    assert res_forced.evidence == []
    assert res_forced.verification_status == "forced_abstention:empty_citations"
    assert res_forced.rejected_msg_ids == ["M888", "M777"]


def test_deleted_only_citations_force_abstention() -> None:
    chat, state, retrieval = _sample_chat_and_state()
    adapter = ScriptedDraftAdapter(
        [
            DraftAnswer(
                reasoning_trace="Trying to guess deleted content.",
                supported=True,
                partial_status_only=False,
                answer="Alice deleted the staging credentials.",
                evidence_msg_ids=["M004"],
            ),
            DraftAnswer(
                reasoning_trace="Still citing only M004.",
                supported=True,
                partial_status_only=False,
                answer="Alice deleted a note on 4 Oct.",
                evidence_msg_ids=["M004"],
            ),
        ]
    )

    verified = answer_and_verify(
        "What did Alice delete on 4 Oct?", chat, state, retrieval, adapter
    )
    assert len(adapter.prompts) == 2
    assert "deleted_only_citations" in adapter.prompts[1]
    assert verified.supported is False
    assert verified.answer == ABSTENTION_ANSWER
    assert verified.evidence == []
    assert verified.verification_status == "forced_abstention:deleted_only_citations"


def test_completion_overclaim_on_unconfirmed_and_in_progress_tasks_blocked() -> None:
    chat, state, retrieval = _sample_chat_and_state()

    # Initial draft falsely claims CloudHost deployment is live/completed; retry fixes to partial status.
    adapter_repaired = ScriptedDraftAdapter(
        [
            DraftAnswer(
                reasoning_trace="Claiming ETA as done.",
                supported=True,
                partial_status_only=False,
                answer="Yes, Alice deployed the service on CloudHost and it is live.",
                evidence_msg_ids=["M006"],
            ),
            DraftAnswer(
                reasoning_trace="Correcting to partial status without overclaiming completion.",
                supported=True,
                partial_status_only=True,
                answer=(
                    "The chat doesn't say whether CloudHost deployment finished. "
                    "Alice said she is deploying on CloudHost with an expected ready date of 12 Oct."
                ),
                evidence_msg_ids=["M006"],
            ),
        ]
    )
    res_repaired = answer_and_verify(
        "Did Alice finish deploying on CloudHost?",
        chat,
        state,
        retrieval,
        adapter_repaired,
    )
    assert len(adapter_repaired.prompts) == 2
    assert "completion_overclaim" in adapter_repaired.prompts[1]
    assert res_repaired.supported is True
    assert res_repaired.partial_status_only is True
    assert "the chat doesn't say whether" in res_repaired.answer.lower()
    assert [r.msg_id for r in res_repaired.evidence] == ["M006"]
    assert res_repaired.verification_status == "verified_after_retry"

    # Persistent overclaim on in_progress analytics (citing M005 which says Auth done, Analytics 60%)
    # must still be caught and forced to abstain.
    adapter_persistent_overclaim = ScriptedDraftAdapter(
        [
            DraftAnswer(
                reasoning_trace="Claiming analytics is done.",
                supported=True,
                partial_status_only=False,
                answer="Analytics is done and completed by Bob.",
                evidence_msg_ids=["M005"],
            ),
            DraftAnswer(
                reasoning_trace="Still claiming analytics is finished.",
                supported=True,
                partial_status_only=False,
                answer="Bob finished the analytics task.",
                evidence_msg_ids=["M005"],
            ),
        ]
    )
    res_blocked = answer_and_verify(
        "Did Bob finish analytics?",
        chat,
        state,
        retrieval,
        adapter_persistent_overclaim,
    )
    assert len(adapter_persistent_overclaim.prompts) == 2
    assert res_blocked.supported is False
    assert res_blocked.answer == ABSTENTION_ANSWER
    assert res_blocked.evidence == []
    assert res_blocked.verification_status == "forced_abstention:completion_overclaim"


def test_completed_task_and_negated_incomplete_task_pass_without_false_positive() -> None:
    chat, state, retrieval = _sample_chat_and_state()
    adapter = ScriptedDraftAdapter(
        [
            DraftAnswer(
                reasoning_trace="Auth is done, analytics is 60%.",
                supported=True,
                partial_status_only=True,
                answer=(
                    "Auth is done and merged by Bob, while analytics is not done yet (only 60% complete)."
                ),
                evidence_msg_ids=["M005"],
            )
        ]
    )

    verified = answer_and_verify(
        "Are auth and analytics finished?", chat, state, retrieval, adapter
    )
    assert len(adapter.prompts) == 1
    assert verified.supported is True
    assert verified.verification_status == "verified"
    assert [r.msg_id for r in verified.evidence] == ["M005"]


def test_partial_unconfirmed_outcome_and_unsupported_normalization() -> None:
    chat, state, retrieval = _sample_chat_and_state()

    # Partial unconfirmed status preserves "the chat doesn't say whether ..." with citations.
    partial_adapter = ScriptedDraftAdapter(
        [
            DraftAnswer(
                reasoning_trace="CloudHost deployment has only an ETA.",
                supported=True,
                partial_status_only=True,
                answer=(
                    "The chat doesn't say whether CloudHost deployment finished; "
                    "Alice was deploying it with an ETA of 12 Oct."
                ),
                evidence_msg_ids=["M006"],
            )
        ]
    )
    partial_res = answer_and_verify(
        "Did Alice finish deploying on CloudHost?",
        chat,
        state,
        retrieval,
        partial_adapter,
    )
    assert partial_res.supported is True
    assert partial_res.partial_status_only is True
    assert "the chat doesn't say whether" in partial_res.answer.lower()
    assert [r.msg_id for r in partial_res.evidence] == ["M006"]

    # Explicit unsupported draft normalizes answer to exact abstention phrase.
    unsupported_adapter = ScriptedDraftAdapter(
        [
            DraftAnswer(
                reasoning_trace="Not mentioned anywhere.",
                supported=False,
                partial_status_only=False,
                answer="No mention of budget approval in the messages.",
                evidence_msg_ids=["M001"],
            )
        ]
    )
    unsupported_res = answer_and_verify(
        "What was the approved budget?",
        chat,
        state,
        retrieval,
        unsupported_adapter,
    )
    assert len(unsupported_adapter.prompts) == 1
    assert unsupported_res.supported is False
    assert unsupported_res.partial_status_only is False
    assert unsupported_res.answer == ABSTENTION_ANSWER
    assert unsupported_res.evidence == []
    assert unsupported_res.verification_status == "abstained"


def test_format_state_for_qa_includes_history_rules_and_msg_ids() -> None:
    _, state, _ = _sample_chat_and_state()
    summary = format_state_for_qa(state)

    assert "Last message timestamp: 07/10/26, 06:45 PM" in summary
    assert "history[]:" in summary
    assert "[M001]" in summary and "[M002]" in summary and "[M003]" in summary
    assert "dependency_rule='2 days before demo'" in summary
    assert "current_owner=unassigned | status=unassigned" in summary


def test_multitask_swap_citation_and_version_dot_anaphora_no_false_overclaim() -> None:
    raw = "\n".join(
        [
            "10/08/2026, 09:10:00 - Vikram: Task split: Ananya - scanner module, Bob - auth.",
            "18/08/2026, 10:15:22 - Tanya: Scanner module main complete kar dungi. Ananya tum hotspot fallback dekh lena.",
            "25/08/2026, 19:30:22 - Tanya: Scanner module done and tagged in v1.4!",
            "06/09/2026, 16:40:00 - Ananya: Hotspot fallback ready by 13 Sep.",
        ]
    )
    chat = parse_chat_text(raw)
    refs_by_id = {
        m.msg_id: EvidenceRef(
            msg_id=m.msg_id,
            line_start=m.line_start,
            line_end=m.line_end,
            timestamp_raw=m.timestamp_raw,
            timestamp_iso=m.timestamp_iso,
            sender=m.sender,
            quote=m.text,
        )
        for m in chat.messages
    }
    state = ProjectState(
        chat_sha256="sha-anaphora",
        total_messages=len(chat.messages),
        last_message_timestamp=chat.messages[-1].timestamp_raw,
        last_message_timestamp_iso=chat.messages[-1].timestamp_iso,
        participants=["Vikram", "Ananya", "Bob", "Tanya"],
        tasks=[
            TaskRecord(
                task_key="scanner_module",
                title="Scanner module",
                current_owner="Tanya",
                status="completed",
                status_note="done and tagged in v1.4",
                history=[
                    StateHistoryEntry(
                        timestamp_raw=chat.messages[1].timestamp_raw,
                        timestamp_iso=chat.messages[1].timestamp_iso,
                        action="reassigned",
                        value="reassigned",
                        owner="Tanya",
                        evidence=[refs_by_id["M002"]],
                    )
                ],
                evidence=[refs_by_id["M003"]],
            ),
            TaskRecord(
                task_key="hotspot_fallback",
                title="Hotspot fallback",
                current_owner="Ananya",
                status="unconfirmed",
                status_note="ready by 13 Sep",
                history=[
                    StateHistoryEntry(
                        timestamp_raw=chat.messages[1].timestamp_raw,
                        timestamp_iso=chat.messages[1].timestamp_iso,
                        action="assigned",
                        value="assigned",
                        owner="Ananya",
                        evidence=[refs_by_id["M002"]],
                    )
                ],
                evidence=[refs_by_id["M004"]],
            ),
        ],
    )
    retrieval = RetrievalResult(
        query="scanner",
        formatted_context="\n".join(
            f"[{m.msg_id} | L{m.line_start}-L{m.line_end} | {m.timestamp_raw} | {m.sender}] {m.text}"
            for m in chat.messages
        ),
    )

    # Same-sentence and two-sentence anaphora ("it is finished and tagged in v1.4") citing M002 swap
    # must resolve to scanner_module (completed), NOT bleed to hotspot_fallback (unconfirmed).
    good_adapter = ScriptedDraftAdapter(
        [
            DraftAnswer(
                reasoning_trace="Tanya took over scanner module in M002 and finished it in M003.",
                supported=True,
                partial_status_only=False,
                answer=(
                    "Tanya took over the scanner module from Ananya, and it is finished and tagged in v1.4. "
                    "She completed and tagged it cleanly."
                ),
                evidence_msg_ids=["M002", "M003"],
            )
        ]
    )
    ok = answer_and_verify(
        "Who took over the scanner module and is it finished?",
        chat,
        state,
        retrieval,
        good_adapter,
    )
    assert len(good_adapter.prompts) == 1
    assert ok.supported is True
    assert ok.verification_status == "verified"

    # Overclaiming completion on hotspot_fallback MUST still be blocked.
    bad_adapter = ScriptedDraftAdapter(
        [
            DraftAnswer(
                reasoning_trace="Wrongly claiming hotspot fallback finished.",
                supported=True,
                partial_status_only=False,
                answer="Hotspot fallback is finished by Ananya.",
                evidence_msg_ids=["M002", "M004"],
            ),
            DraftAnswer(
                reasoning_trace="Still claiming done.",
                supported=True,
                partial_status_only=False,
                answer="Ananya completed hotspot fallback.",
                evidence_msg_ids=["M004"],
            ),
        ]
    )
    blocked = answer_and_verify(
        "Did Ananya finish hotspot fallback?",
        chat,
        state,
        retrieval,
        bad_adapter,
    )
    assert blocked.supported is False
    assert blocked.verification_status == "forced_abstention:completion_overclaim"
