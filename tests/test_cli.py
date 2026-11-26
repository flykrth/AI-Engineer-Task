"""Unit tests for the CLI entrypoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from run import EvidenceItem, Slice1Answer, format_transcript, run_cli
from src.llm import BaseLLMAdapter, LLMConfig, MissingAPIKeyError
from src.parser import parse_chat_file
from src.schemas import (
    DeadlineDelta,
    DecisionDelta,
    OpenIssueDelta,
    ProjectState,
    TaskDelta,
    WindowExtractionResult,
)

TEAM_CHAT_PATH = Path(__file__).resolve().parent.parent / "data" / "team_chat.txt"


class FakeAdapter(BaseLLMAdapter):
    """Deterministic fake adapter for CLI testing."""

    def __init__(self, response: Slice1Answer) -> None:
        super().__init__(
            LLMConfig(
                provider="gemini",
                api_key="test-key",
                chat_model="gemini-3.5-flash-lite",
                embed_model="gemini-embedding-001",
            )
        )
        self.response = response
        self.last_prompt: str = ""

    def _is_transient_error(self, exc: Exception) -> bool:
        return False

    def _raw_generate_text(
        self, prompt: str, *, system_prompt: str | None, temperature: float
    ) -> str:
        return self.response.answer

    def _raw_generate_structured(
        self,
        prompt: str,
        schema: type[Any],
        *,
        system_prompt: str | None,
        temperature: float,
    ) -> tuple[Any, str]:
        self.last_prompt = prompt
        return self.response, self.response.model_dump_json()

    def _raw_embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]


class FakeCompilerAdapter(BaseLLMAdapter):
    """Deterministic fake adapter returning WindowExtractionResult objects for state CLI tests."""

    def __init__(self, window_result: WindowExtractionResult) -> None:
        super().__init__(
            LLMConfig(
                provider="gemini",
                api_key="test-key",
                chat_model="gemini-3.5-flash-lite",
                embed_model="gemini-embedding-001",
            )
        )
        self.window_result = window_result
        self.structured_calls = 0

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
        self.structured_calls += 1
        return self.window_result, self.window_result.model_dump_json()

    def _raw_embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]


def test_format_transcript_includes_ids_lines_timestamps_and_senders() -> None:
    chat = parse_chat_file(TEAM_CHAT_PATH)
    transcript = format_transcript(chat)

    assert "[M001 | L1-L1 | 03/10/26, 9:14 PM | system]" in transcript
    assert "[M017 | L17-L22 | 05/10/26, 8:02 PM | Aarav] Task split karte hain:" in transcript
    assert "[M036 | L41-L41 | 16/10/26, 9:07 PM | Meera]" in transcript
    assert "[edited]" in transcript
    assert "[deleted]" in transcript


def test_run_cli_json_and_text_modes(capsys: pytest.CaptureFixture[str]) -> None:
    fake = FakeAdapter(
        Slice1Answer(
            answer="Kabir owns payments using Razorpay test mode; webhook is remaining.",
            supported=True,
            evidence=[
                EvidenceItem(
                    msg_id="M068",
                    line_start=72,
                    line_end=72,
                    timestamp_raw="02/11/26, 8:50 PM",
                    sender="Kabir",
                    quote="payments ab mere paas, Meera tum report pe focus karo",
                )
            ],
        )
    )

    exit_code = run_cli(
        ["--chat", str(TEAM_CHAT_PATH), "--question", "Who owns payments?", "--json"],
        adapter_factory=lambda provider=None: fake,
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["supported"] is True
    assert data["provider"] == "gemini"
    assert data["total_messages"] == 102
    assert data["evidence"][0]["line_start"] == 72

    exit_code_text = run_cli(
        ["--chat", str(TEAM_CHAT_PATH), "--question", "Who owns payments?"],
        adapter_factory=lambda provider=None: fake,
    )
    assert exit_code_text == 0
    captured_text = capsys.readouterr()
    assert "Answer: Kabir owns payments" in captured_text.out
    assert "[M068 | L72 | 02/11/26, 8:50 PM | Kabir]" in captured_text.out


def test_run_cli_unsupported_answer_normalization(
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake = FakeAdapter(
        Slice1Answer(
            answer="I cannot verify that from the messages.",
            supported=False,
            evidence=[],
        )
    )

    exit_code = run_cli(
        ["--chat", str(TEAM_CHAT_PATH), "--question", "What did Aarav delete?"],
        adapter_factory=lambda provider=None: fake,
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Answer: the chat doesn't say" in out


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
    chat_file = tmp_path / "sample.txt"
    chat_file.write_text(
        "\n".join(
            [
                "01/10/26, 09:00 AM - Alice: Use ProviderA for gateway.",
                "02/10/26, 09:00 AM - Alice: Actually switch to ProviderB, drop ProviderA.",
                "03/10/26, 10:00 AM - Bob: I'll take the API task.",
                "04/10/26, 11:00 AM - Alice: Release is 10 Nov. Docs due 2 days before release (8 Nov).",
                "05/10/26, 12:00 PM - Alice: Still need an owner for analytics.",
            ]
        ),
        encoding="utf-8",
    )

    fake = FakeCompilerAdapter(
        WindowExtractionResult(
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
    )
    cache_dir = tmp_path / ".cache"

    # First run compiles and writes cache.
    exit_text = run_cli(
        ["--chat", str(chat_file), "--state-only"],
        adapter_factory=lambda provider=None: fake,
        cache_dir=cache_dir,
    )
    assert exit_text == 0
    assert fake.structured_calls == 1
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
    assert fake.structured_calls == 1
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
    assert fake.structured_calls == 2
    assert "Source: compiled" in capsys.readouterr().out
