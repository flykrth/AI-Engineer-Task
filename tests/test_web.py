"""Unit and integration tests for the FastAPI web backend and static dashboard."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from src.llm import BaseLLMAdapter, LLMConfig
from src.schemas import (
    DeadlineDelta,
    DecisionDelta,
    DraftAnswer,
    OpenIssueDelta,
    TaskDelta,
    WindowExtractionResult,
)
from src.web.app import create_app

STATIC_DIR = Path(__file__).resolve().parent.parent / "src" / "web" / "static"


class FakeWebAdapter(BaseLLMAdapter):
    """Deterministic fake adapter for testing FastAPI endpoints without live APIs."""

    def __init__(
        self,
        window_result: WindowExtractionResult,
        drafts: Sequence[DraftAnswer] | None = None,
        *,
        fail_embeddings: bool = False,
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
        self.fail_embeddings = fail_embeddings
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
        if self.fail_embeddings:
            raise RuntimeError("Simulated embedding API failure")
        return [[float(idx + 1), 1.0] for idx, _ in enumerate(texts)]


def _make_web_sample_chat() -> tuple[str, WindowExtractionResult]:
    chat_text = "\n".join(
        [
            "01/10/26, 09:00 AM - Alice: Use ProviderA for payments.",
            "02/10/26, 09:30 AM - Alice: Actually switch to ProviderB <This message was edited>",
            "03/10/26, 10:00 AM - Bob: <Media omitted>",
            "04/10/26, 10:01 AM - Bob: That image shows the staging pipeline.",
            "05/10/26, 11:00 AM - Carol: Literal <script>alert(1)</script> snippet in text.",
            "06/10/26, 12:00 PM - Alice: Release is 18 Nov, docs 2 days earlier (16 Nov).",
            "07/10/26, 01:00 PM - Bob: This message was deleted",
            "08/10/26, 02:00 PM - Carol: Analytics still has no owner.",
        ]
    )
    extraction = WindowExtractionResult(
        decisions=[
            DecisionDelta(
                topic_key="payments",
                title="Payment provider",
                action="proposed",
                value="ProviderA",
                evidence_msg_ids=["M001"],
            ),
            DecisionDelta(
                topic_key="payments",
                title="Payment provider",
                action="reversed",
                value="ProviderB",
                evidence_msg_ids=["M002"],
            ),
        ],
        tasks=[
            TaskDelta(
                task_key="staging_pipeline",
                title="Staging pipeline",
                action="progress_updated",
                owner="Bob",
                status_note="diagram shared",
                evidence_msg_ids=["M004"],
            ),
            TaskDelta(
                task_key="analytics",
                title="Analytics",
                action="dropped_unassigned",
                owner=None,
                status_note="needs an owner",
                evidence_msg_ids=["M008"],
            ),
        ],
        deadlines=[
            DeadlineDelta(
                item_key="docs",
                title="Documentation",
                action="rule_defined",
                date_text="16 Nov",
                dependency_rule="2 days before release",
                evidence_msg_ids=["M006"],
            )
        ],
        open_items=[
            OpenIssueDelta(
                issue_key="analytics_owner",
                title="Analytics ownership",
                action="opened",
                description="Analytics still has no owner.",
                evidence_msg_ids=["M008"],
            )
        ],
    )
    return chat_text, extraction


def test_web_endpoints_state_messages_and_rebuild(tmp_path: Path) -> None:
    raw, extraction = _make_web_sample_chat()
    chat_file = tmp_path / "chat.txt"
    chat_file.write_text(raw, encoding="utf-8")
    fake = FakeWebAdapter(extraction)

    app = create_app(
        chat_file,
        adapter_factory=lambda provider=None: fake,
        cache_dir=tmp_path / ".cache",
    )
    client = TestClient(app)

    # GET / serves index.html cleanly
    root_resp = client.get("/")
    assert root_resp.status_code == 200
    assert "text/html" in root_resp.headers["content-type"]
    assert "Team chat project state" in root_resp.text
    assert "/static/app.js" in root_resp.text

    # GET /static/app.js and /static/styles.css serve cleanly
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/styles.css").status_code == 200

    # GET /api/health
    health_resp = client.get("/api/health")
    assert health_resp.status_code == 200
    health = health_resp.json()
    assert health["status"] == "ok"
    assert health["provider"] == "gemini"
    assert health["total_messages"] == 8
    assert health["from_cache"] is False
    assert health["dense_available"] is True
    assert health["dense_fallback"] is False

    # GET /api/state
    state_resp = client.get("/api/state")
    assert state_resp.status_code == 200
    state_json = state_resp.json()
    assert state_json["provider"] == "gemini"
    assert state_json["model"] == "gemini-3.5-flash-lite"
    assert state_json["from_cache"] is False
    assert state_json["total_source_lines"] == 8
    assert state_json["chat_file"] == str(chat_file)
    assert state_json["decisions"][0]["current_value"] == "ProviderB"
    assert len(state_json["decisions"][0]["history"]) == 2
    assert state_json["tasks"][1]["status"] == "unassigned"

    # GET /api/messages preserves literal <Media omitted>, <script>, edited, and deleted markers
    msg_resp = client.get("/api/messages")
    assert msg_resp.status_code == 200
    messages = msg_resp.json()["messages"]
    assert len(messages) == 8
    assert messages[1]["is_edited"] is True
    assert messages[2]["kind"] == "media_omitted"
    assert messages[2]["text"] == "<Media omitted>"
    assert messages[4]["text"] == "Literal <script>alert(1)</script> snippet in text."
    assert messages[6]["kind"] == "deleted"

    # POST /api/rebuild forces recompile and re-index
    assert fake.compiler_calls == 1
    assert len(fake.embed_calls) == 1
    rebuild_resp = client.post("/api/rebuild")
    assert rebuild_resp.status_code == 200
    rebuilt = rebuild_resp.json()
    assert rebuilt["from_cache"] is False
    assert fake.compiler_calls == 2
    assert len(fake.embed_calls) == 2


def test_web_ask_verified_partial_abstained_and_validation(tmp_path: Path) -> None:
    raw, extraction = _make_web_sample_chat()
    chat_file = tmp_path / "chat.txt"
    chat_file.write_text(raw, encoding="utf-8")

    fake = FakeWebAdapter(
        extraction,
        drafts=[
            # 1. Verified answer
            DraftAnswer(
                reasoning_trace="ProviderB replaced ProviderA in M002.",
                supported=True,
                partial_status_only=False,
                answer="The team switched from ProviderA to ProviderB.",
                evidence_msg_ids=["M002"],
            ),
            # 2. Partial status only answer
            DraftAnswer(
                reasoning_trace="M004 shows staging diagram shared, no completion confirmed.",
                supported=True,
                partial_status_only=True,
                answer="Bob is working on the staging pipeline, and the chat doesn't say whether it finished.",
                evidence_msg_ids=["M004"],
            ),
            # 3 & 4. Deleted-only citation triggers retry and forced abstention
            DraftAnswer(
                reasoning_trace="Guessing deleted message M007.",
                supported=True,
                partial_status_only=False,
                answer="Bob deleted a deployment key.",
                evidence_msg_ids=["M007"],
            ),
            DraftAnswer(
                reasoning_trace="Still citing deleted message M007.",
                supported=True,
                partial_status_only=False,
                answer="Bob deleted a secret note.",
                evidence_msg_ids=["M007"],
            ),
        ],
    )

    app = create_app(
        chat_file,
        adapter_factory=lambda provider=None: fake,
        cache_dir=tmp_path / ".cache",
    )
    client = TestClient(app)

    # Blank question returns HTTP 422
    blank_resp = client.post("/api/ask", json={"question": "   "})
    assert blank_resp.status_code == 422

    # 1. Verified
    resp1 = client.post("/api/ask", json={"question": "What payment provider was chosen?"})
    assert resp1.status_code == 200
    d1 = resp1.json()
    assert d1["supported"] is True
    assert d1["partial_status_only"] is False
    assert d1["verification_status"] == "verified"
    assert d1["used_dense"] is True
    assert d1["dense_fallback"] is False
    assert d1["evidence"][0]["msg_id"] == "M002"

    # 2. Partial
    resp2 = client.post("/api/ask", json={"question": "Did Bob finish staging?"})
    assert resp2.status_code == 200
    d2 = resp2.json()
    assert d2["supported"] is True
    assert d2["partial_status_only"] is True
    assert d2["verification_status"] == "verified"
    assert d2["evidence"][0]["msg_id"] == "M004"

    # 3. Forced abstention on deleted-only draft
    resp3 = client.post("/api/ask", json={"question": "What did Bob delete?"})
    assert resp3.status_code == 200
    d3 = resp3.json()
    assert d3["supported"] is False
    assert d3["answer"] == "the chat doesn't say"
    assert d3["evidence"] == []
    assert d3["verification_status"] == "forced_abstention:deleted_only_citations"


def test_web_ask_reports_dense_fallback_when_embeddings_fail(tmp_path: Path) -> None:
    raw, extraction = _make_web_sample_chat()
    chat_file = tmp_path / "chat.txt"
    chat_file.write_text(raw, encoding="utf-8")

    fake = FakeWebAdapter(
        extraction,
        drafts=[
            DraftAnswer(
                reasoning_trace="M002 confirms ProviderB.",
                supported=True,
                partial_status_only=False,
                answer="ProviderB is selected.",
                evidence_msg_ids=["M002"],
            )
        ],
        fail_embeddings=True,
    )

    app = create_app(
        chat_file,
        adapter_factory=lambda provider=None: fake,
        cache_dir=tmp_path / ".cache",
    )
    client = TestClient(app)
    resp = client.post("/api/ask", json={"question": "Payment provider?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["supported"] is True
    assert data["used_dense"] is False
    assert data["dense_fallback"] is True


def test_web_static_assets_safety_and_hygiene() -> None:
    forbidden_terms = [
        "innerHTML",
        "outerHTML",
        "insertAdjacentHTML",
        "http://",
        "https://",
        "CampusCart",
        "Aarav",
        "Priya",
        "Rohan",
        "Meera",
        "Kabir",
        "Razorpay",
        "Namecheap",
    ]
    for filename in ("index.html", "styles.css", "app.js"):
        content = (STATIC_DIR / filename).read_text(encoding="utf-8")
        # Ensure zero emojis or em-dashes (pure ASCII check).
        non_ascii = [ch for ch in content if ord(ch) > 127]
        assert not non_ascii, f"Unexpected non-ASCII characters in {filename}: {non_ascii}"
        assert "\u2014" not in content
        for term in forbidden_terms:
            assert term not in content, f"Forbidden term {term!r} found in {filename}"
