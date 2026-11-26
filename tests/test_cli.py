"""Unit tests for the Slice 1 CLI entrypoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from run import EvidenceItem, Slice1Answer, format_transcript, run_cli
from src.llm import BaseLLMAdapter, LLMConfig, MissingAPIKeyError
from src.parser import parse_chat_file

TEAM_CHAT_PATH = Path(__file__).resolve().parent.parent / "data" / "team_chat.txt"


class FakeAdapter(BaseLLMAdapter):
    """Deterministic fake adapter for CLI testing."""

    def __init__(self, response: Slice1Answer) -> None:
        super().__init__(
            LLMConfig(
                provider="gemini",
                api_key="test-key",
                chat_model="gemini-3.8-flash",
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
