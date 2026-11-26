"""Unit tests for the CLI entrypoint covering Phase 3 retrieval and verified Q&A."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from run import run_cli
from src.llm import BaseLLMAdapter, LLMConfig, MissingAPIKeyError
from src.schemas import (
    DeadlineDelta,
    DecisionDelta,
    DraftAnswer,
    OpenIssueDelta,
    ProjectState,
    TaskDelta,
    WindowExtractionResult,
)

TEAM_CHAT_PATH = Path(__file__).resolve().parent.parent / "data" / "team_chat.txt"


class FakePipelineAdapter(BaseLLMAdapter):
    """Deterministic fake adapter handling compiler windows, embeddings, and Q&A drafts."""

    def __init__(
        self,
        window_result: WindowExtractionResult,
        drafts: Sequence[DraftAnswer] | None = None,
    ) -> None:
        super().__init__(
            LLMConfig(
                provider="gemini",
                api_key="test-key",
                chat_model="gemini-3.5-flash-lite",
                embed_model="gemini-embedding-001",
            )
        )
        self.window_result = window_result
        self._drafts = list(drafts or [])
        self.compiler_calls = 0
        self.qa_calls = 0
        self.embed_calls: list[list[str]] = []

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
        if schema is WindowExtractionResult:
            self.compiler_calls += 1
            return self.window_result, self.window_result.model_dump_json()
        if schema is DraftAnswer:
            idx = min(self.qa_calls, len(self._drafts) - 1)
            self.qa_calls += 1
            draft = self._drafts[idx]
            return draft, draft.model_dump_json()
        raise AssertionError(f"Unexpected schema requested: {schema}")

    def _raw_embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls.append(list(texts))
        return [[float(idx + 1), 1.0] for idx, _ in enumerate(texts)]


def _make_sample_chat_and_extraction() -> tuple[str, WindowExtractionResult]:
    chat_text = "\n".join(
        [
            "01/10/26, 09:00 AM - Alice: Use ProviderA for gateway.",
            "02/10/26, 09:00 AM - Alice: Actually switch to ProviderB, drop ProviderA.",
            "03/10/26, 10:00 AM - Bob: I'll take the API task.",
            "04/10/26, 11:00 AM - Alice: Release is 10 Nov. Docs due 2 days before release (8 Nov).",
            "05/10/26, 12:00 PM - Alice: Still need an owner for analytics.",
            "06/10/26, 01:00 PM - Bob: This message was deleted",
            "07/10/26, 02:00 PM - Alice: Deploying dashboard on CloudHost, ready by 14 Nov.",
        ]
    )
    extraction = WindowExtractionResult(
        decisions=[
            DecisionDelta(
                topic_key="gateway",
                title="Gateway choice",
                action="proposed",
                value="ProviderA",
                evidence_msg_ids=["M001"],
            ),
            DecisionDelta(
                topic_key="gateway",
                title="Gateway choice",
                action="reversed",
                value="ProviderB",
                evidence_msg_ids=["M002"],
            ),
        ],
        tasks=[
            TaskDelta(
                task_key="api",
                title="API implementation",
                action="assigned",
                owner="Bob",
                evidence_msg_ids=["M003"],
            ),
            TaskDelta(
                task_key="analytics",
                title="Analytics",
                action="dropped_unassigned",
                owner=None,
                status_note="needs an owner",
                evidence_msg_ids=["M005"],
            ),
            TaskDelta(
                task_key="dashboard_deployment",
                title="Dashboard deployment",
                action="promised_eta",
                owner="Alice",
                status_note="ready by 14 Nov",
                evidence_msg_ids=["M007"],
            ),
        ],
        deadlines=[
            DeadlineDelta(
                item_key="docs",
                title="Documentation",
                action="rule_defined",
                date_text="8 Nov",
                dependency_rule="2 days before release",
                evidence_msg_ids=["M004"],
            )
        ],
        open_items=[
            OpenIssueDelta(
                issue_key="analytics_owner",
                title="Analytics ownership",
                action="opened",
                description="Still need an owner for analytics",
                evidence_msg_ids=["M005"],
            )
        ],
    )
    return chat_text, extraction


def test_run_cli_phase3_qa_json_text_and_rebuild(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    raw, extraction = _make_sample_chat_and_extraction()
    chat_file = tmp_path / "chat.txt"
    chat_file.write_text(raw, encoding="utf-8")
    cache_dir = tmp_path / ".cache"

    fake = FakePipelineAdapter(
        extraction,
        drafts=[
            DraftAnswer(
                reasoning_trace="ProviderB replaced ProviderA in M002; M999 is invalid.",
                supported=True,
                partial_status_only=False,
                answer="The team switched from ProviderA to ProviderB.",
                evidence_msg_ids=["M002", "M999"],
            ),
            DraftAnswer(
                reasoning_trace="ProviderB confirmed in M002.",
                supported=True,
                partial_status_only=False,
                answer="The team switched from ProviderA to ProviderB.",
                evidence_msg_ids=["M002"],
            ),
            DraftAnswer(
                reasoning_trace="ProviderB confirmed in M002.",
                supported=True,
                partial_status_only=False,
                answer="The team switched from ProviderA to ProviderB.",
                evidence_msg_ids=["M002"],
            ),
        ],
    )

    # First run compiles state and builds embedding cache.
    exit_json = run_cli(
        ["--chat", str(chat_file), "--question", "What gateway did we choose?", "--json"],
        adapter_factory=lambda provider=None: fake,
        cache_dir=cache_dir,
    )
    assert exit_json == 0
    assert fake.compiler_calls == 1
    assert len(fake.embed_calls) == 2  # 1 index batch + 1 query embedding
    data = json.loads(capsys.readouterr().out)
    assert data["supported"] is True
    assert data["partial_status_only"] is False
    assert data["verification_status"] == "verified_with_stripped_ids"
    assert data["rejected_msg_ids"] == ["M999"]
    assert data["provider"] == "gemini"
    assert data["total_messages"] == 7
    assert data["evidence"][0]["msg_id"] == "M002"
    assert data["evidence"][0]["line_start"] == 2
    assert data["evidence"][0]["sender"] == "Alice"

    # Second run hits both ProjectState cache and embedding index cache (only query embed runs).
    exit_text = run_cli(
        ["--chat", str(chat_file), "--question", "What gateway did we choose?"],
        adapter_factory=lambda provider=None: fake,
        cache_dir=cache_dir,
    )
    assert exit_text == 0
    assert fake.compiler_calls == 1
    assert len(fake.embed_calls) == 3  # +1 query embedding only
    text_out = capsys.readouterr().out
    assert "Answer: The team switched from ProviderA to ProviderB." in text_out
    assert "Evidence:" in text_out
    assert "[M002 | L2 | 02/10/26, 09:00 AM | Alice]" in text_out

    # Third run with --rebuild forwards --rebuild to both compiler and retriever index caches.
    exit_rebuild = run_cli(
        ["--chat", str(chat_file), "--question", "What gateway did we choose?", "--rebuild"],
        adapter_factory=lambda provider=None: fake,
        cache_dir=cache_dir,
    )
    assert exit_rebuild == 0
    assert fake.compiler_calls == 2
    assert len(fake.embed_calls) == 5  # +1 rebuilt index batch +1 query embedding
    capsys.readouterr()


def test_run_cli_unsupported_and_verifier_abstention(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    raw, extraction = _make_sample_chat_and_extraction()
    chat_file = tmp_path / "chat.txt"
    chat_file.write_text(raw, encoding="utf-8")
    cache_dir = tmp_path / ".cache"

    # Draft claims deleted message content -> 1 retry -> still deleted-only -> forced abstention.
    fake = FakePipelineAdapter(
        extraction,
        drafts=[
            DraftAnswer(
                reasoning_trace="Guessing deleted note.",
                supported=True,
                partial_status_only=False,
                answer="Bob deleted the secret API key.",
                evidence_msg_ids=["M006"],
            ),
            DraftAnswer(
                reasoning_trace="Still citing deleted message.",
                supported=True,
                partial_status_only=False,
                answer="Bob deleted a secret note.",
                evidence_msg_ids=["M006"],
            ),
        ],
    )

    exit_code = run_cli(
        ["--chat", str(chat_file), "--question", "What did Bob delete?", "--json"],
        adapter_factory=lambda provider=None: fake,
        cache_dir=cache_dir,
    )
    assert exit_code == 0
    assert fake.qa_calls == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["supported"] is False
    assert payload["answer"] == "the chat doesn't say"
    assert payload["evidence"] == []
    assert payload["verification_status"] == "forced_abstention:deleted_only_citations"


def test_run_cli_missing_key_returns_nonzero_clean_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def failing_factory(provider: str | None = None) -> BaseLLMAdapter:
        raise MissingAPIKeyError("GEMINI_API_KEY is not set.")

    exit_code = run_cli(
        ["--chat", str(TEAM_CHAT_PATH), "--question", "Who owns payments?"],
        adapter_factory=failing_factory,
    )
    assert exit_code == 1
    err = capsys.readouterr().err
    assert "GEMINI_API_KEY is not set." in err


def test_run_cli_missing_action_returns_argument_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = run_cli(["--chat", str(TEAM_CHAT_PATH)])
    assert exit_code == 2
    err = capsys.readouterr().err
    assert "--question" in err and "--state-only" in err


def test_run_cli_state_only_text_json_and_rebuild(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    raw, extraction = _make_sample_chat_and_extraction()
    chat_file = tmp_path / "sample.txt"
    chat_file.write_text(raw, encoding="utf-8")
    fake = FakePipelineAdapter(extraction)
    cache_dir = tmp_path / ".cache"

    # First run compiles and writes cache.
    exit_text = run_cli(
        ["--chat", str(chat_file), "--state-only"],
        adapter_factory=lambda provider=None: fake,
        cache_dir=cache_dir,
    )
    assert exit_text == 0
    assert fake.compiler_calls == 1
    out_text = capsys.readouterr().out
    assert "State of the project" in out_text
    assert "Source: compiled" in out_text
    assert "1. Current decisions" in out_text
    assert "Prior values: ProviderA" in out_text
    assert "2. Tasks and ownership" in out_text
    assert "owner=Bob | status=assigned" in out_text
    assert "3. Deadlines" in out_text
    assert "rule: 2 days before release" in out_text
    assert "4. Open, unassigned, or unconfirmed items" in out_text
    assert "[M002 | L2 | 02/10/26, 09:00 AM | Alice]" in out_text

    # Second run without --rebuild hits cache and prints clean ProjectState JSON.
    exit_json = run_cli(
        ["--chat", str(chat_file), "--state-only", "--json"],
        adapter_factory=lambda provider=None: fake,
        cache_dir=cache_dir,
    )
    assert exit_json == 0
    assert fake.compiler_calls == 1
    parsed_state = ProjectState.model_validate_json(capsys.readouterr().out)
    assert parsed_state.decisions[0].current_value == "ProviderB"
    assert len(parsed_state.decisions[0].history) == 2

    # Third run with --rebuild bypasses cache and recompiles.
    exit_rebuild = run_cli(
        ["--chat", str(chat_file), "--state-only", "--rebuild"],
        adapter_factory=lambda provider=None: fake,
        cache_dir=cache_dir,
    )
    assert exit_rebuild == 0
    assert fake.compiler_calls == 2
    assert "Source: compiled" in capsys.readouterr().out
