"""Unit tests for the evaluation suite format and reproducible evaluation runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eval.run_eval import (
    check_answer_invariants,
    check_project_state_invariants,
    main as eval_main,
    run_evaluation,
)
from src.llm import BaseLLMAdapter, LLMConfig
from src.parser import parse_chat_text
from src.schemas import (
    DraftAnswer,
    TaskDelta,
    VerifiedAnswer,
    WindowExtractionResult,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TEAM_CHAT_EVAL_PATH = REPO_ROOT / "eval" / "team_chat_eval.json"


class MockEvalAdapter(BaseLLMAdapter):
    """Mock adapter returning valid extraction deltas and draft answers for runner tests."""

    def __init__(self) -> None:
        super().__init__(
            LLMConfig(
                provider="gemini",
                api_key="test-key",
                chat_model="gemini-3.5-flash-lite",
                embed_model="gemini-embedding-001",
            )
        )

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
        if schema is WindowExtractionResult:
            res = WindowExtractionResult(
                tasks=[
                    TaskDelta(
                        task_key="payments",
                        title="Payments",
                        action="assigned",
                        owner="Kabir",
                        status_note="webhook remaining",
                        evidence_msg_ids=["M001"],
                    )
                ]
            )
            return res, res.model_dump_json()
        if schema is DraftAnswer:
            if "deleted message" in prompt.lower() or "github repo" in prompt.lower():
                draft = DraftAnswer(
                    reasoning_trace="Information missing or deleted.",
                    supported=False,
                    partial_status_only=False,
                    answer="the chat doesn't say",
                    evidence_msg_ids=[],
                )
            else:
                draft = DraftAnswer(
                    reasoning_trace="Found valid M001 and stripped M999.",
                    supported=True,
                    partial_status_only=True,
                    answer="Kabir owns payments using Razorpay test mode; webhook is still remaining.",
                    evidence_msg_ids=["M001", "M999"],
                )
            return draft, draft.model_dump_json()
        raise AssertionError(f"Unexpected schema: {schema}")

    def _raw_embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, float(idx + 1)] for idx, _ in enumerate(texts)]


def test_team_chat_eval_suite_schema_and_hygiene() -> None:
    raw = TEAM_CHAT_EVAL_PATH.read_text(encoding="utf-8")
    assert "\u2014" not in raw
    data = json.loads(raw)
    questions = data["questions"]
    assert len(questions) == 15
    assert [q["id"] for q in questions] == [f"Q{i}" for i in range(1, 16)]
    for q in questions:
        chat_path = REPO_ROOT / q["chat_file"]
        assert chat_path.is_file()
        assert q["question"].strip()
        assert q["expected_answer"].strip()


def test_run_evaluation_full_and_state_only_with_mock_adapter(tmp_path: Path) -> None:
    mini_chat = tmp_path / "mini_chat.txt"
    mini_chat.write_text(
        "\n".join(
            [
                "01/10/26, 09:00 AM - Kabir: payments Razorpay test mode webhook remaining",
                "02/10/26, 09:05 AM - Aarav: This message was deleted",
            ]
        ),
        encoding="utf-8",
    )
    suite_file = tmp_path / "mini_suite.json"
    suite_file.write_text(
        json.dumps(
            {
                "suite_name": "mini_test",
                "default_chat_file": str(mini_chat),
                "questions": [
                    {
                        "id": "Q1",
                        "category": "payments",
                        "chat_file": str(mini_chat),
                        "question": "Who owns payments?",
                        "expected_answer": "Kabir owns payments with webhook remaining.",
                        "expected_supported": True,
                        "expected_msg_ids_any": ["M001"],
                        "fact_hint_groups": [["kabir"], ["razorpay"], ["webhook"]],
                    },
                    {
                        "id": "Q2",
                        "category": "abstention",
                        "chat_file": str(mini_chat),
                        "question": "What did the deleted message say?",
                        "expected_answer": "the chat doesn't say",
                        "expected_supported": False,
                        "expected_msg_ids_any": [],
                        "fact_hint_groups": [["the chat doesn't say"]],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    out_dir = tmp_path / "eval_outputs"
    cache_dir = tmp_path / ".cache"

    report, json_path, md_path = run_evaluation(
        suite_path=suite_file,
        output_dir=out_dir,
        cache_dir=cache_dir,
        adapter_factory=lambda provider=None: MockEvalAdapter(),
    )
    assert json_path.is_file()
    assert md_path.is_file()
    assert report["summary"]["total_questions"] == 2
    assert report["summary"]["invariants_passed"] == 2
    assert report["summary"]["hints_passed"] == 2
    assert report["summary"]["total_rejected_msg_ids"] == 1
    assert report["results"][0]["rejected_msg_ids"] == ["M999"]
    assert report["results"][1]["supported"] is False
    assert "\u2014" not in md_path.read_text(encoding="utf-8")

    # State-only mode skips Q&A and writes state report.
    state_report, state_json, state_md = run_evaluation(
        chat_override=mini_chat,
        state_only=True,
        output_dir=out_dir,
        cache_dir=cache_dir,
        adapter_factory=lambda provider=None: MockEvalAdapter(),
    )
    assert state_json.is_file()
    assert state_md.is_file()
    assert state_report["state_only"] is True
    assert state_report["results"] == []
    assert state_report["summary"]["state_invariants_ok"] is True


def test_invariant_helpers_catch_mismatches() -> None:
    chat = parse_chat_text("01/10/26, 09:00 AM - Alice: Task one done\n")
    bad_verified = VerifiedAnswer(
        answer="Task one done",
        supported=True,
        partial_status_only=False,
        evidence=[],
        verification_status="verified",
    )
    errors = check_answer_invariants(bad_verified, chat)
    assert any("evidence is empty" in err for err in errors)

    # CLI validation error when neither --suite nor --chat provided.
    assert eval_main([]) == 2
    assert eval_main(["--chat", "data/team_chat.txt"]) == 2
    assert check_project_state_invariants.__name__ == "check_project_state_invariants"
