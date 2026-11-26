"""Mocked unit tests for the temporal state compiler, deterministic reducer, and cache."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.compiler import (
    build_message_windows,
    compile_project_state,
    compile_project_state_with_meta,
    normalize_key,
    reduce_deltas_to_state,
    resolve_canonical_key,
)
from src.llm import BaseLLMAdapter, LLMConfig
from src.parser import parse_chat_text
from src.schemas import (
    DeadlineDelta,
    DecisionDelta,
    OpenIssueDelta,
    TaskDelta,
    WindowExtractionResult,
)


class ScriptedWindowAdapter(BaseLLMAdapter):
    """Fake LLM adapter returning pre-scripted WindowExtractionResult per window call."""

    def __init__(
        self,
        responder: Callable[[int, str], WindowExtractionResult],
        *,
        chat_model: str = "gemini-3.5-flash-lite",
    ) -> None:
        super().__init__(
            LLMConfig(
                provider="gemini",
                api_key="test-key",
                chat_model=chat_model,
                embed_model="gemini-embedding-001",
            )
        )
        self._responder = responder
        self.calls: list[str] = []

    def _is_transient_error(self, exc: Exception) -> bool:
        return False

    def _raw_generate_text(
        self, prompt: str, *, system_prompt: str | None, temperature: float
    ) -> str:
        return "{}"

    def _raw_generate_structured(
        self,
        prompt: str,
        schema: type[Any],
        *,
        system_prompt: str | None,
        temperature: float,
    ) -> tuple[Any, str]:
        idx = len(self.calls)
        self.calls.append(prompt)
        result = self._responder(idx, prompt)
        return result, result.model_dump_json()

    def _raw_embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]


def test_triple_reversal_preserves_full_history_and_latest_state(
    tmp_path: Path,
) -> None:
    raw = "\n".join(
        [
            "01/10/26, 10:00 AM - Alice: Billing gateway will be handled by Maya via Stripe.",
            "05/10/26, 11:00 AM - Maya: Stripe requires business registration we lack.",
            "05/10/26, 11:05 AM - Alice: Final decision: drop Stripe, use manual bank transfer.",
            "10/10/26, 09:00 AM - Liam: Stripe sandbox mode works without business docs.",
            "10/10/26, 09:10 AM - Alice: Confirmed, switch back to Stripe sandbox. Scrap bank transfer.",
            "10/10/26, 09:12 AM - Liam: I'll own billing gateway now, Maya focus on docs.",
            "12/10/26, 04:00 PM - Liam: Billing webhook still remaining.",
        ]
    )
    chat = parse_chat_text(raw)

    def responder(idx: int, prompt: str) -> WindowExtractionResult:
        assert idx == 0
        return WindowExtractionResult(
            decisions=[
                DecisionDelta(
                    topic_key="billing_gateway",
                    title="Billing gateway",
                    action="proposed",
                    value="Stripe",
                    evidence_msg_ids=["M001"],
                ),
                DecisionDelta(
                    topic_key="billing_method_decision",
                    title="Billing gateway",
                    action="reversed",
                    value="Manual bank transfer",
                    rationale="Stripe requires business registration",
                    evidence_msg_ids=["M002", "M003"],
                ),
                DecisionDelta(
                    topic_key="billing_gateway",
                    title="Billing gateway",
                    action="reversed",
                    value="Stripe sandbox",
                    rationale="Works without business registration",
                    evidence_msg_ids=["M004", "M005"],
                ),
            ],
            tasks=[
                TaskDelta(
                    task_key="billing_gateway_task",
                    title="Billing gateway",
                    action="assigned",
                    owner="Maya",
                    evidence_msg_ids=["M001"],
                ),
                TaskDelta(
                    task_key="billing_gateway",
                    title="Billing gateway",
                    action="reassigned",
                    owner="Liam",
                    evidence_msg_ids=["M006"],
                ),
                TaskDelta(
                    task_key="billing_gateway",
                    title="Billing gateway",
                    action="progress_updated",
                    owner="Liam",
                    status_note="webhook remaining",
                    evidence_msg_ids=["M007"],
                ),
            ],
        )

    adapter = ScriptedWindowAdapter(responder)
    state = compile_project_state(
        chat, adapter, raw_text=raw, cache_dir=tmp_path / ".cache"
    )

    assert len(state.decisions) == 1
    decision = state.decisions[0]
    assert decision.topic_key == "billing_gateway"
    assert decision.current_value == "Stripe sandbox"
    assert decision.status == "reversed"
    assert [entry.value for entry in decision.history] == [
        "Stripe",
        "Manual bank transfer",
        "Stripe sandbox",
    ]
    assert [ref.msg_id for ref in decision.evidence] == ["M004", "M005"]

    assert len(state.tasks) == 1
    task = state.tasks[0]
    assert task.task_key == "billing_gateway"
    assert task.current_owner == "Liam"
    assert task.status == "in_progress"
    assert task.status_note == "webhook remaining"
    assert [entry.owner for entry in task.history] == ["Maya", "Liam", "Liam"]


def test_partial_handoff_swap_and_dropped_unassigned_task(tmp_path: Path) -> None:
    raw = "\n".join(
        [
            "01/10/26, 09:00 AM - Alice: Task split:",
            "Ben - auth module",
            "Liam - search index",
            "Main - release script",
            "04/10/26, 02:00 PM - Liam: Search is too big, someone else take search please.",
            "06/10/26, 06:00 PM - Ben: Auth is 60% done on branch wip-auth, can Liam finish it?",
            "06/10/26, 06:05 PM - Liam: Sure, I'll take auth. Ben you do DB schema then.",
            "06/10/26, 06:06 PM - Ben: Deal, DB schema is mine.",
            "09/10/26, 08:00 PM - Liam: Auth done and merged.",
        ]
    )
    chat = parse_chat_text(raw)

    def responder(idx: int, prompt: str) -> WindowExtractionResult:
        return WindowExtractionResult(
            tasks=[
                TaskDelta(
                    task_key="auth",
                    title="Auth module",
                    action="assigned",
                    owner="Ben",
                    evidence_msg_ids=["M001"],
                ),
                TaskDelta(
                    task_key="search_index",
                    title="Search index",
                    action="assigned",
                    owner="Liam",
                    evidence_msg_ids=["M001"],
                ),
                TaskDelta(
                    task_key="release_script",
                    title="Release script",
                    action="assigned",
                    owner="Main",
                    evidence_msg_ids=["M001"],
                ),
                TaskDelta(
                    task_key="search_index",
                    title="Search index",
                    action="dropped_unassigned",
                    owner="Liam",
                    status_note="dropped, needs new owner",
                    evidence_msg_ids=["M002"],
                ),
                TaskDelta(
                    task_key="auth",
                    title="Auth module",
                    action="progress_updated",
                    owner="Ben",
                    status_note="60% done on branch wip-auth",
                    evidence_msg_ids=["M003"],
                ),
                TaskDelta(
                    task_key="auth",
                    title="Auth module",
                    action="reassigned",
                    owner="Liam",
                    status_note="handed off from Ben at 60%",
                    evidence_msg_ids=["M004"],
                ),
                TaskDelta(
                    task_key="db_schema",
                    title="DB schema",
                    action="assigned",
                    owner="Ben",
                    evidence_msg_ids=["M004", "M005"],
                ),
                TaskDelta(
                    task_key="authentication",
                    title="Auth module",
                    action="completed",
                    owner="Liam",
                    status_note="done and merged",
                    evidence_msg_ids=["M006"],
                ),
            ]
        )

    adapter = ScriptedWindowAdapter(responder)
    state = compile_project_state(
        chat, adapter, raw_text=raw, cache_dir=tmp_path / ".cache"
    )
    tasks_by_key = {t.task_key: t for t in state.tasks}

    assert tasks_by_key["auth"].current_owner == "Liam"
    assert tasks_by_key["auth"].status == "completed"
    assert [h.action for h in tasks_by_key["auth"].history] == [
        "assigned",
        "progress_updated",
        "reassigned",
        "completed",
    ]
    assert tasks_by_key["search_index"].current_owner is None
    assert tasks_by_key["search_index"].status == "unassigned"
    assert tasks_by_key["release_script"].current_owner == "Alice"
    assert tasks_by_key["db_schema"].current_owner == "Ben"
    assert tasks_by_key["db_schema"].status == "assigned"


def test_promised_eta_never_marks_task_completed_and_cannot_downgrade_completed() -> None:
    raw = "\n".join(
        [
            "01/10/26, 09:00 AM - Alice: Ben - test suite",
            "02/10/26, 09:00 PM - Ben: Tests will be done by tomorrow, promise.",
            "02/10/26, 09:01 PM - Maya: Sure Ben, \"by tomorrow\"",
            "05/10/26, 10:00 PM - Ben: Tests still pending, I'll finish by Sunday.",
            "06/10/26, 10:00 AM - Alice: Maya - packaging script",
            "07/10/26, 11:00 AM - Maya: Packaging script done and tagged.",
            "08/10/26, 10:00 AM - Alice: Liam - billing module",
            "09/10/26, 02:00 PM - Liam: Checkout integrated and working.",
            "11/10/26, 05:00 PM - Liam: Billing webhook remaining.",
        ]
    )
    chat = parse_chat_text(raw)

    deltas = WindowExtractionResult(
        tasks=[
            TaskDelta(
                task_key="test_suite",
                title="Test suite",
                action="assigned",
                owner="Ben",
                evidence_msg_ids=["M001"],
            ),
            TaskDelta(
                task_key="test_suite",
                title="Test suite",
                action="promised_eta",
                owner="Ben",
                status_note="promised by tomorrow (sarcastic reply)",
                evidence_msg_ids=["M002", "M003"],
            ),
            TaskDelta(
                task_key="test_suite",
                title="Test suite",
                action="promised_eta",
                owner="Ben",
                status_note="still pending, ETA Sunday",
                evidence_msg_ids=["M004"],
            ),
            TaskDelta(
                task_key="packaging_script",
                title="Packaging script",
                action="assigned",
                owner="Maya",
                evidence_msg_ids=["M005"],
            ),
            TaskDelta(
                task_key="packaging_script",
                title="Packaging script",
                action="completed",
                owner="Maya",
                status_note="done and tagged",
                evidence_msg_ids=["M006"],
            ),
            # Late/stray progress update referencing the same completed message must not downgrade.
            TaskDelta(
                task_key="packaging_script",
                title="Packaging script",
                action="promised_eta",
                owner="Maya",
                status_note="late duplicate note",
                evidence_msg_ids=["M006"],
            ),
            TaskDelta(
                task_key="billing_module",
                title="Billing module",
                action="assigned",
                owner="Liam",
                evidence_msg_ids=["M007"],
            ),
            TaskDelta(
                task_key="billing_module",
                title="Billing module",
                action="completed",
                owner="Liam",
                status_note="checkout working",
                evidence_msg_ids=["M008"],
            ),
            # Strictly later message explicitly reporting remaining work reopens to in_progress.
            TaskDelta(
                task_key="billing_module",
                title="Billing module",
                action="progress_updated",
                owner="Liam",
                status_note="webhook remaining",
                evidence_msg_ids=["M009"],
            ),
        ]
    )

    state = reduce_deltas_to_state(chat, deltas, chat_sha256="test-sha")
    by_key = {t.task_key: t for t in state.tasks}

    assert by_key["test_suite"].status == "unconfirmed"
    assert by_key["test_suite"].status != "completed"
    assert by_key["packaging_script"].status == "completed"
    assert by_key["billing"].status == "in_progress"
    assert by_key["billing"].status_note == "webhook remaining"


def test_overlapping_windows_deduplicate_and_pass_canonical_keys(
    tmp_path: Path,
) -> None:
    lines = [
        f"0{min(9, (i // 3) + 1)}/10/26, 10:{i:02d} AM - Alice: Message {i + 1}"
        for i in range(10)
    ]
    lines[4] = "02/10/26, 10:04 AM - Alice: Final database choice is Postgres."
    lines[8] = "03/10/26, 10:08 AM - Alice: Wait, switching database to DuckDB."
    raw = "\n".join(lines)
    chat = parse_chat_text(raw)

    windows = build_message_windows(chat.messages, window_size=6, overlap=2)
    assert len(windows) == 2
    # M005 and M006 sit in the overlap between window 0 and window 1.
    overlap_ids = {m.msg_id for m in windows[0]} & {m.msg_id for m in windows[1]}
    assert overlap_ids == {"M005", "M006"}

    seen_prompts: list[str] = []

    def responder(idx: int, prompt: str) -> WindowExtractionResult:
        seen_prompts.append(prompt)
        if idx == 0:
            return WindowExtractionResult(
                decisions=[
                    DecisionDelta(
                        topic_key="database_engine",
                        title="Database engine",
                        action="confirmed",
                        value="Postgres",
                        evidence_msg_ids=["M005"],
                    )
                ]
            )
        # Window 1 should see database_engine in the canonical key registry prompt,
        # and even if it emits M005 again in overlap, it must deduplicate cleanly.
        assert "database_engine" in prompt
        return WindowExtractionResult(
            decisions=[
                DecisionDelta(
                    topic_key="database_engine_decision",
                    title="Database engine",
                    action="confirmed",
                    value="Postgres",
                    evidence_msg_ids=["M005"],
                ),
                DecisionDelta(
                    topic_key="database_engine",
                    title="Database engine",
                    action="reversed",
                    value="DuckDB",
                    evidence_msg_ids=["M009"],
                ),
            ]
        )

    adapter = ScriptedWindowAdapter(responder)
    state = compile_project_state(
        chat,
        adapter,
        raw_text=raw,
        window_size=6,
        overlap=2,
        cache_dir=tmp_path / ".cache",
    )

    assert len(seen_prompts) == 2
    assert len(state.decisions) == 1
    db = state.decisions[0]
    assert db.current_value == "DuckDB"
    assert [(h.action, h.value) for h in db.history] == [
        ("confirmed", "Postgres"),
        ("reversed", "DuckDB"),
    ]


def test_hallucinated_msg_ids_are_stripped_and_ungrounded_deltas_dropped(
    tmp_path: Path,
) -> None:
    raw = (
        "01/10/26, 09:00 AM - Alice: Launch date is 20 Nov.\n"
        "02/10/26, 09:00 AM - Bob: Report is due 2 days before launch (18 Nov).\n"
    )
    chat = parse_chat_text(raw)

    # Direct reducer check with mixed valid/invalid IDs and all-invalid IDs.
    mixed = WindowExtractionResult(
        deadlines=[
            DeadlineDelta(
                item_key="launch_date",
                title="Launch date",
                action="set",
                date_text="20 Nov",
                evidence_msg_ids=["M999", "M001", "FAKE_ID"],
            ),
            DeadlineDelta(
                item_key="invented_deadline",
                title="Invented",
                action="set",
                date_text="31 Dec",
                evidence_msg_ids=["M999", "BAD"],
            ),
        ],
        open_items=[
            OpenIssueDelta(
                issue_key="ungrounded_issue",
                title="Ghost issue",
                action="opened",
                description="Never happened",
                evidence_msg_ids=["M404"],
            )
        ],
    )
    reduced = reduce_deltas_to_state(chat, mixed, chat_sha256="sha")
    assert len(reduced.deadlines) == 1
    assert reduced.deadlines[0].item_key == "launch"
    assert [ref.msg_id for ref in reduced.deadlines[0].evidence] == ["M001"]
    assert reduced.deadlines[0].evidence[0].quote == "Launch date is 20 Nov."
    assert reduced.open_items == []

    # Extractor-level window check: out-of-window IDs are also dropped per window.
    def responder(idx: int, prompt: str) -> WindowExtractionResult:
        return WindowExtractionResult(
            deadlines=[
                DeadlineDelta(
                    item_key="report_deadline",
                    title="Report",
                    action="rule_defined",
                    date_text="18 Nov",
                    dependency_rule="2 days before launch",
                    evidence_msg_ids=["M002", "M999"],
                ),
                DeadlineDelta(
                    item_key="hallucinated_item",
                    title="Hallucinated",
                    action="set",
                    date_text="25 Nov",
                    evidence_msg_ids=["M099"],
                ),
            ]
        )

    adapter = ScriptedWindowAdapter(responder)
    compiled = compile_project_state(
        chat, adapter, raw_text=raw, cache_dir=tmp_path / ".cache"
    )
    assert len(compiled.deadlines) == 1
    assert compiled.deadlines[0].dependency_rule == "2 days before launch"
    assert [ref.msg_id for ref in compiled.deadlines[0].evidence] == ["M002"]


def test_cache_hit_and_rebuild_behavior(tmp_path: Path) -> None:
    raw = "01/10/26, 09:00 AM - Alice: Use FastAPI for the service.\n"
    chat = parse_chat_text(raw)
    cache_dir = tmp_path / ".cache"

    def responder(idx: int, prompt: str) -> WindowExtractionResult:
        return WindowExtractionResult(
            decisions=[
                DecisionDelta(
                    topic_key="backend_framework",
                    title="Backend framework",
                    action="confirmed",
                    value="FastAPI",
                    evidence_msg_ids=["M001"],
                )
            ]
        )

    adapter = ScriptedWindowAdapter(responder)

    state1, from_cache1, cache_path1 = compile_project_state_with_meta(
        chat, adapter, raw_text=raw, cache_dir=cache_dir, rebuild=False
    )
    assert from_cache1 is False
    assert cache_path1.is_file()
    assert len(adapter.calls) == 1

    # Second call with same content + model hits cache without invoking adapter.
    state2, from_cache2, cache_path2 = compile_project_state_with_meta(
        chat, adapter, raw_text=raw, cache_dir=cache_dir, rebuild=False
    )
    assert from_cache2 is True
    assert cache_path2 == cache_path1
    assert state2 == state1
    assert len(adapter.calls) == 1

    # rebuild=True forces recompilation.
    state3, from_cache3, _ = compile_project_state_with_meta(
        chat, adapter, raw_text=raw, cache_dir=cache_dir, rebuild=True
    )
    assert from_cache3 is False
    assert state3 == state1
    assert len(adapter.calls) == 2

    # Different model name creates a distinct cache entry and compiles fresh.
    other_adapter = ScriptedWindowAdapter(responder, chat_model="gemini-other")
    _, from_cache4, cache_path4 = compile_project_state_with_meta(
        chat, other_adapter, raw_text=raw, cache_dir=cache_dir, rebuild=False
    )
    assert from_cache4 is False
    assert cache_path4 != cache_path1
    assert len(other_adapter.calls) == 1


def test_key_normalization_and_alias_matching() -> None:
    assert normalize_key("Payment-Method_decision") == "payment_method"
    assert normalize_key("Auth Task") == "auth"
    assert normalize_key("Report_Due_Date") == "report"

    registered = ["payments", "db_schema", "listings_ui", "auth"]
    assert resolve_canonical_key("payment_method", registered) == "payments"
    assert resolve_canonical_key("payments_integration", registered) == "payments"
    assert resolve_canonical_key("database_schema_work", registered) == "db_schema"
    assert resolve_canonical_key("ui_listings", registered) == "listings_ui"
    assert resolve_canonical_key("authentication", registered) == "auth"
    # Distinct topics must NOT collide.
    assert resolve_canonical_key("auth_tests", registered) == "auth_tests"
    assert resolve_canonical_key("seller_dashboard", registered) == "seller_dashboard"


def test_compiler_scaling_over_1000_lines_dedup_and_key_preservation(
    tmp_path: Path,
) -> None:
    from datetime import datetime, timedelta
    import time

    start_dt = datetime(2026, 7, 1, 9, 0)
    raw_lines: list[str] = []
    # Build 1,100 lines (~900+ messages plus multi-line continuations) across 5 weeks.
    for msg_idx in range(1, 920):
        ts = (start_dt + timedelta(minutes=msg_idx * 45)).strftime("%d/%m/%y, %I:%M %p")
        sender = ["Dev", "Ira", "Kunal", "Sana", "Tarun", "Zoya"][msg_idx % 6]
        if msg_idx == 20:
            raw_lines.append(f"{ts} - {sender}: Motor drive controller set to L298N.")
        elif msg_idx == 24:
            raw_lines.append(f"{ts} - {sender}: Telemetry link assigned to Ira.")
        elif msg_idx == 450:
            raw_lines.append(f"{ts} - {sender}: Drop L298N, motor controller reversed to BTS7960.")
        elif msg_idx == 780:
            raw_lines.append(f"{ts} - {sender}: Final reversal: motor controller is VESC CAN.")
            raw_lines.append("Controller calibration starts tomorrow.")
        elif msg_idx % 5 == 0:
            raw_lines.append(f"{ts} - {sender}: Multi-line workshop note {msg_idx}:")
            raw_lines.append(f"detail line A for note {msg_idx}")
        else:
            raw_lines.append(f"{ts} - {sender}: General banter message {msg_idx}.")

    raw = "\n".join(raw_lines)
    chat = parse_chat_text(raw)
    assert chat.total_source_lines >= 1000
    assert len(chat.messages) == 919

    windows = build_message_windows(chat.messages, window_size=25, overlap=6)
    assert len(windows) >= 40
    covered_ids = {m.msg_id for win in windows for m in win}
    assert len(covered_ids) == len(chat.messages)

    seen_prompts: list[str] = []

    def responder(idx: int, prompt: str) -> WindowExtractionResult:
        seen_prompts.append(prompt)
        decisions: list[DecisionDelta] = []
        tasks: list[TaskDelta] = []
        # Emit overlapping deltas whenever key boundary messages appear in the window.
        if "[M020 " in prompt:
            decisions.append(
                DecisionDelta(
                    topic_key="motor_controller",
                    title="Motor controller",
                    action="confirmed",
                    value="L298N",
                    evidence_msg_ids=["M020"],
                )
            )
        if "[M024 " in prompt:
            tasks.append(
                TaskDelta(
                    task_key="telemetry_link",
                    title="Telemetry link",
                    action="assigned",
                    owner="Ira",
                    evidence_msg_ids=["M024"],
                )
            )
        if "[M450 " in prompt:
            assert "motor_controller" in prompt
            decisions.append(
                DecisionDelta(
                    topic_key="motor_controller_decision",
                    title="Motor controller",
                    action="reversed",
                    value="BTS7960",
                    evidence_msg_ids=["M450"],
                )
            )
        if "[M780 " in prompt:
            assert "motor_controller" in prompt
            decisions.append(
                DecisionDelta(
                    topic_key="motor_controller_choice",
                    title="Motor controller",
                    action="reversed",
                    value="VESC CAN",
                    evidence_msg_ids=["M780"],
                )
            )
        return WindowExtractionResult(decisions=decisions, tasks=tasks)

    t0 = time.perf_counter()
    adapter = ScriptedWindowAdapter(responder)
    state = compile_project_state(
        chat,
        adapter,
        raw_text=raw,
        window_size=25,
        overlap=6,
        cache_dir=tmp_path / ".cache",
    )
    elapsed = time.perf_counter() - t0

    assert elapsed < 5.0
    assert len(seen_prompts) == len(windows)
    assert len(state.decisions) == 1
    assert state.decisions[0].topic_key == "motor_controller"
    assert state.decisions[0].current_value == "VESC CAN"
    # Overlap windows covering M020/M450/M780 must deduplicate cleanly into exactly 3 history entries.
    assert [h.value for h in state.decisions[0].history] == [
        "L298N",
        "BTS7960",
        "VESC CAN",
    ]
    assert len(state.tasks) == 1
    assert state.tasks[0].task_key == "telemetry_link"
    assert state.tasks[0].current_owner == "Ira"


def test_overlapping_windows_deduplicate_same_anchor_action_variants(
    tmp_path: Path,
) -> None:
    lines = [
        f"0{min(9, (i // 3) + 1)}/10/26, 10:{i:02d} AM - Alice: Message {i + 1}"
        for i in range(10)
    ]
    lines[4] = "02/10/26, 10:04 AM - Alice: Use CloudHost for deployment. Decided."
    lines[5] = "02/10/26, 10:05 AM - Bob: Final report due 23 Nov."
    raw = "\n".join(lines)
    chat = parse_chat_text(raw)

    def responder(idx: int, prompt: str) -> WindowExtractionResult:
        if idx == 0:
            return WindowExtractionResult(
                decisions=[
                    DecisionDelta(
                        topic_key="deployment_host",
                        title="Deployment Host",
                        action="confirmed",
                        value="CloudHost",
                        evidence_msg_ids=["M004", "M005"],
                    )
                ],
                deadlines=[
                    DeadlineDelta(
                        item_key="report_due",
                        title="Report due",
                        action="set",
                        date_text="23 Nov",
                        evidence_msg_ids=["M006"],
                    )
                ],
            )
        # Overlap window extracts same anchor with weaker/variant action ("proposed" and "changed").
        return WindowExtractionResult(
            decisions=[
                DecisionDelta(
                    topic_key="deployment_host",
                    title="Deployment Host",
                    action="proposed",
                    value="CloudHost",
                    evidence_msg_ids=["M005"],
                )
            ],
            deadlines=[
                DeadlineDelta(
                    item_key="report_due",
                    title="Report due",
                    action="changed",
                    date_text="23 Nov",
                    dependency_rule="3 days before demo",
                    evidence_msg_ids=["M006"],
                )
            ],
        )

    adapter = ScriptedWindowAdapter(responder)
    state = compile_project_state(
        chat,
        adapter,
        raw_text=raw,
        window_size=6,
        overlap=2,
        cache_dir=tmp_path / ".cache",
    )
    assert len(state.decisions) == 1
    assert state.decisions[0].status == "confirmed"
    assert len(state.decisions[0].history) == 1
    assert len(state.deadlines) == 1
    assert state.deadlines[0].current_date_text == "23 Nov"
    assert state.deadlines[0].dependency_rule == "3 days before demo"
    assert len(state.deadlines[0].history) == 1
