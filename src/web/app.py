"""FastAPI application providing ProjectState, transcript evidence, and grounded Q&A APIs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import threading
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from google.genai import errors as genai_errors
import httpx
import openai
from pydantic import BaseModel, Field, field_validator

from src.compiler import compile_project_state_with_meta
from src.llm import BaseLLMAdapter, LLMError, create_llm_adapter
from src.parser import ParsedChat, parse_chat_file
from src.qa import answer_and_verify
from src.retriever import HybridRetriever
from src.schemas import ProjectState

STATIC_DIR = Path(__file__).resolve().parent / "static"


class AskRequest(BaseModel):
    """Request body for grounded Q&A queries."""

    question: str = Field(..., description="User question in English or Hinglish.")

    @field_validator("question")
    @classmethod
    def _validate_non_empty_question(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Question must not be blank.")
        return cleaned


@dataclass
class _AppRuntime:
    """Thread-safe container holding compiled state and retrieval index."""

    chat_path: Path
    provider: str | None
    cache_dir: Path | None
    adapter_factory: Any
    adapter: BaseLLMAdapter
    chat: ParsedChat
    raw_text: str
    state: ProjectState
    from_cache: bool
    retriever: HybridRetriever
    lock: threading.RLock = field(default_factory=threading.RLock)

    def build_state_payload(self) -> dict[str, Any]:
        payload = self.state.model_dump()
        payload.update(
            {
                "provider": self.adapter.provider_name,
                "model": self.adapter.chat_model,
                "from_cache": self.from_cache,
                "total_source_lines": self.chat.total_source_lines,
                "chat_file": str(self.chat_path),
            }
        )
        return payload

    def build_health_payload(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "chat_file": str(self.chat_path),
            "provider": self.adapter.provider_name,
            "model": self.adapter.chat_model,
            "embed_model": self.adapter.embed_model,
            "date_order": self.chat.date_order,
            "total_source_lines": self.chat.total_source_lines,
            "total_messages": len(self.chat.messages),
            "from_cache": self.from_cache,
            "dense_available": self.retriever.dense_available,
            "dense_fallback": self.retriever.dense_fallback,
        }


def _load_runtime_artifacts(
    chat_path: Path,
    *,
    provider: str | None,
    rebuild: bool,
    cache_dir: Path | None,
    adapter_factory: Any,
    adapter: BaseLLMAdapter | None = None,
) -> tuple[BaseLLMAdapter, ParsedChat, str, ProjectState, bool, HybridRetriever]:
    chat = parse_chat_file(chat_path)
    raw_text = chat_path.read_text(encoding="utf-8")
    resolved_adapter = adapter if adapter is not None else adapter_factory(provider=provider)
    state, from_cache, _ = compile_project_state_with_meta(
        chat,
        resolved_adapter,
        raw_text=raw_text,
        cache_dir=cache_dir,
        rebuild=rebuild,
    )
    retriever = HybridRetriever(
        chat,
        resolved_adapter,
        raw_text=raw_text,
        cache_dir=cache_dir,
        rebuild=rebuild,
    )
    return resolved_adapter, chat, raw_text, state, from_cache, retriever


def create_app(
    chat_path: Path,
    *,
    provider: str | None = None,
    rebuild: bool = False,
    cache_dir: Path | None = None,
    adapter_factory: Any = create_llm_adapter,
) -> FastAPI:
    """Create and configure the FastAPI dashboard application for a WhatsApp chat."""
    resolved_path = Path(chat_path)
    adapter, chat, raw_text, state, from_cache, retriever = _load_runtime_artifacts(
        resolved_path,
        provider=provider,
        rebuild=rebuild,
        cache_dir=cache_dir,
        adapter_factory=adapter_factory,
    )
    runtime = _AppRuntime(
        chat_path=resolved_path,
        provider=provider,
        cache_dir=cache_dir,
        adapter_factory=adapter_factory,
        adapter=adapter,
        chat=chat,
        raw_text=raw_text,
        state=state,
        from_cache=from_cache,
        retriever=retriever,
    )

    app = FastAPI(
        title="Team Chat Project State and Grounded Q&A",
        version="0.1.0",
    )
    app.state.runtime = runtime

    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", response_class=FileResponse)
    def serve_index() -> FileResponse:
        index_path = STATIC_DIR / "index.html"
        if not index_path.is_file():
            raise HTTPException(status_code=404, detail="Dashboard index.html not found.")
        return FileResponse(index_path, media_type="text/html; charset=utf-8")

    @app.get("/api/health")
    def get_health() -> dict[str, Any]:
        with runtime.lock:
            return runtime.build_health_payload()

    @app.get("/api/state")
    def get_state() -> dict[str, Any]:
        with runtime.lock:
            return runtime.build_state_payload()

    @app.get("/api/messages")
    def get_messages() -> dict[str, Any]:
        with runtime.lock:
            return {
                "chat_file": str(runtime.chat_path),
                "date_order": runtime.chat.date_order,
                "total_source_lines": runtime.chat.total_source_lines,
                "total_messages": len(runtime.chat.messages),
                "messages": [msg.model_dump() for msg in runtime.chat.messages],
            }

    @app.post("/api/ask")
    def post_ask(payload: AskRequest) -> dict[str, Any]:
        with runtime.lock:
            try:
                retrieval = runtime.retriever.retrieve(payload.question)
                verified = answer_and_verify(
                    payload.question,
                    runtime.chat,
                    runtime.state,
                    retrieval,
                    runtime.adapter,
                )
            except (
                ValueError,
                LLMError,
                genai_errors.APIError,
                openai.OpenAIError,
                httpx.HTTPError,
            ) as exc:
                raise HTTPException(status_code=502, detail=str(exc)) from exc

            result = verified.model_dump()
            result.update(
                {
                    "question": payload.question,
                    "used_dense": retrieval.used_dense,
                    "dense_fallback": retrieval.dense_fallback,
                    "dense_fallback_reason": retrieval.dense_fallback_reason,
                }
            )
            return result

    @app.post("/api/rebuild")
    def post_rebuild() -> dict[str, Any]:
        with runtime.lock:
            try:
                _, chat_next, raw_next, state_next, from_cache_next, retriever_next = (
                    _load_runtime_artifacts(
                        runtime.chat_path,
                        provider=runtime.provider,
                        rebuild=True,
                        cache_dir=runtime.cache_dir,
                        adapter_factory=runtime.adapter_factory,
                        adapter=runtime.adapter,
                    )
                )
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except (
                ValueError,
                LLMError,
                genai_errors.APIError,
                openai.OpenAIError,
                httpx.HTTPError,
            ) as exc:
                raise HTTPException(status_code=502, detail=str(exc)) from exc

            runtime.chat = chat_next
            runtime.raw_text = raw_next
            runtime.state = state_next
            runtime.from_cache = from_cache_next
            runtime.retriever = retriever_next
            return runtime.build_state_payload()

    return app
