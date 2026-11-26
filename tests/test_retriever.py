"""Unit tests for hybrid BM25 + embedding retrieval and context expansion."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from src.llm import BaseLLMAdapter, LLMConfig
from src.parser import parse_chat_text
from src.retriever import (
    RRF_K,
    HybridRetriever,
    build_chunks,
    tokenize_text,
)


class MockEmbeddingAdapter(BaseLLMAdapter):
    """Fake adapter with controllable embed_texts behavior and call tracking."""

    def __init__(
        self,
        embed_fn: Callable[[list[str]], list[list[float]]],
        *,
        provider: str = "gemini",
        embed_model: str = "gemini-embedding-001",
    ) -> None:
        super().__init__(
            LLMConfig(
                provider=provider,  # type: ignore[arg-type]
                api_key="test-key",
                chat_model="gemini-3.5-flash-lite",
                embed_model=embed_model,
            )
        )
        self._embed_fn = embed_fn
        self.embed_calls: list[list[str]] = []

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
        return None, "{}"

    def _raw_embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls.append(list(texts))
        return self._embed_fn(texts)


def _make_synthetic_chat(num_messages: int = 24) -> str:
    lines: list[str] = []
    for idx in range(1, num_messages + 1):
        minute = (idx - 1) % 60
        sender = ["Alice", "Bob", "Maya", "Liam"][idx % 4]
        if idx == 4:
            text = "Uploaded billing wireframe."
        elif idx == 5:
            text = "<Media omitted>"
        elif idx == 6:
            text = "Screenshot above shows invoice layout without tax field."
        elif idx == 11:
            text = "Check branch auth-wip on portal.example.org before release."
        elif idx == 15:
            text = "Kal tak database migration pakka ho jayega."
        elif idx == 23:
            text = "Final status: checkout integration still pending."
        else:
            text = f"Routine update number {idx} for sprint tracking."
        lines.append(f"01/10/26, 10:{minute:02d} AM - {sender}: {text}")
    return "\n".join(lines)


def test_tokenize_english_hinglish_and_compound_tokens() -> None:
    tokens = tokenize_text(
        "Check auth-wip on campuscart.in, login/signup kal tak pakka!"
    )
    assert "auth-wip" in tokens
    assert "auth" in tokens
    assert "wip" in tokens
    assert "campuscart.in" in tokens
    assert "campuscart" in tokens
    assert "in" in tokens
    assert "login/signup" in tokens
    assert "login" in tokens
    assert "signup" in tokens
    assert "kal" in tokens
    assert "pakka" in tokens


def test_build_chunks_formats_metadata_and_markers() -> None:
    raw = "\n".join(
        [
            "01/10/26, 09:00 AM - Alice: First line",
            "01/10/26, 09:01 AM - Bob: Edited line <This message was edited>",
            "01/10/26, 09:02 AM - Maya: This message was deleted",
            "01/10/26, 09:03 AM - Liam: Fourth line",
            "01/10/26, 09:04 AM - Alice: Fifth line",
            "01/10/26, 09:05 AM - Bob: Sixth line",
            "01/10/26, 09:06 AM - Maya: Seventh line",
        ]
    )
    chat = parse_chat_text(raw)
    chunks = build_chunks(chat.messages, chunk_size=6, stride=3)

    assert len(chunks) == 2
    assert chunks[0].msg_ids == ["M001", "M002", "M003", "M004", "M005", "M006"]
    assert chunks[1].msg_ids == ["M004", "M005", "M006", "M007"]
    assert "[M002 | L2-L2 | 01/10/26, 09:01 AM | Bob] Edited line [edited]" in chunks[0].text
    assert "[M003 | L3-L3 | 01/10/26, 09:02 AM | Maya] This message was deleted [deleted]" in chunks[0].text


def test_hybrid_rrf_combines_lexical_and_dense_rankings(tmp_path: Path) -> None:
    raw = "\n".join(
        [
            "01/10/26, 09:00 AM - Alice: alpha budget note",
            "01/10/26, 09:01 AM - Bob: alpha budget follow up",
            "01/10/26, 09:02 AM - Maya: alpha planning details",
            "01/10/26, 09:03 AM - Liam: gamma deployment window",
            "01/10/26, 09:04 AM - Alice: beta alpha hybrid topic",
            "01/10/26, 09:05 AM - Bob: beta server schedule",
            "01/10/26, 09:06 AM - Maya: unrelated lunch note",
            "01/10/26, 09:07 AM - Liam: unrelated coffee chat",
            "01/10/26, 09:08 AM - Alice: delta conceptual match only",
        ]
    )
    chat = parse_chat_text(raw)

    # Chunk 0: M001-M003 (3 "alpha" hits -> strong BM25, weak dense)
    # Chunk 1: M003-M005 (2 "alpha" hits -> medium BM25, strong dense)
    # Chunk 2: M005-M007 (1 "alpha" hit -> weaker BM25, weak dense)
    # Chunk 3: M007-M009 (0 lexical hits -> no BM25, strongest dense)
    def embed_fn(texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for t in texts:
            if "query phrase" in t:
                # Query vector
                vectors.append([1.0, 0.0])
            elif "delta conceptual match" in t:
                # Chunk 3: highest dense (cos=1.0), no lexical
                vectors.append([1.0, 0.0])
            elif "gamma deployment window" in t:
                # Chunk 1: second dense (cos~0.89), AND matches lexical "alpha"
                vectors.append([0.89, 0.45])
            elif "beta server schedule" in t:
                vectors.append([0.3, 0.95])
            else:
                vectors.append([0.1, 0.99])
        return vectors

    adapter = MockEmbeddingAdapter(embed_fn)
    retriever = HybridRetriever(
        chat,
        adapter,
        raw_text=raw,
        chunk_size=3,
        stride=2,
        cache_dir=tmp_path / ".cache",
    )

    result = retriever.retrieve(
        "alpha query phrase", top_k=4, expand_n=0, include_tail=False
    )
    assert result.used_dense is True
    assert result.dense_fallback is False
    # Chunk 1 has strong dense (rank 2) + lexical match (rank 2) => beats single-signal chunks.
    top_hit = result.hits[0]
    assert top_hit.chunk_index == 1
    assert top_hit.bm25_rank is not None
    assert top_hit.dense_rank == 2
    expected_rrf = 1.0 / (RRF_K + top_hit.bm25_rank) + 1.0 / (RRF_K + top_hit.dense_rank)
    assert top_hit.rrf_score == pytest.approx(expected_rrf)


def test_context_expansion_couples_media_omitted_and_merges_without_duplicates(
    tmp_path: Path,
) -> None:
    raw = _make_synthetic_chat(24)
    chat = parse_chat_text(raw)

    # Chunk 0: M001..M006, Chunk 1: M004..M009 (contains <Media omitted> M005 + M006 explanation)
    # Chunk 2: M007..M012, Chunk 3: M010..M015
    def embed_fn(texts: list[str]) -> list[list[float]]:
        vecs: list[list[float]] = []
        for t in texts:
            if "wireframe" in t or "invoice" in t:
                vecs.append([1.0, 0.0])
            elif "auth-wip" in t:
                vecs.append([0.7, 0.7])
            else:
                vecs.append([0.0, 1.0])
        return vecs

    adapter = MockEmbeddingAdapter(embed_fn)
    retriever = HybridRetriever(
        chat,
        adapter,
        raw_text=raw,
        chunk_size=6,
        stride=3,
        cache_dir=tmp_path / ".cache",
    )

    # Query hits chunk containing "invoice" (M006) and expands +/- 3 so M004, M005 (<Media omitted>),
    # M006 (explanation), and adjacent chunks merge cleanly without duplicate messages.
    result = retriever.retrieve(
        "invoice layout wireframe", top_k=2, expand_n=3, include_tail=True, tail_size=5
    )

    all_block_msg_ids = [mid for block in result.blocks for mid in block.msg_ids]
    assert len(all_block_msg_ids) == len(set(all_block_msg_ids))
    # Contiguous chronological ordering across all blocks.
    indices = [idx for block in result.blocks for idx in range(block.start_idx, block.end_idx + 1)]
    assert indices == sorted(indices)

    # Media omitted message (M005) and its adjacent explanation (M004, M006) stay in the same block.
    media_blocks = [b for b in result.blocks if "M005" in b.msg_ids]
    assert len(media_blocks) == 1
    assert "M004" in media_blocks[0].msg_ids
    assert "M006" in media_blocks[0].msg_ids
    assert "<Media omitted>" in media_blocks[0].formatted_text
    assert "invoice layout without tax field" in media_blocks[0].formatted_text

    # Uncovered trailing tail (last 5 messages M020..M024) is also included cleanly.
    assert any("M024" in b.msg_ids and "M020" in b.msg_ids for b in result.blocks)


def test_adjacent_span_merging_and_tail_already_covered(tmp_path: Path) -> None:
    raw = _make_synthetic_chat(12)
    chat = parse_chat_text(raw)

    def embed_fn(texts: list[str]) -> list[list[float]]:
        return [[1.0, float(i)] for i, _ in enumerate(texts)]

    adapter = MockEmbeddingAdapter(embed_fn)
    retriever = HybridRetriever(
        chat,
        adapter,
        raw_text=raw,
        chunk_size=6,
        stride=3,
        cache_dir=tmp_path / ".cache",
    )

    # With 12 messages and expand_n=3, top_k=2 covers overlapping/adjacent spans through end.
    result = retriever.retrieve("checkout pending", top_k=3, expand_n=3, include_tail=True, tail_size=5)
    assert len(result.blocks) == 1
    assert result.blocks[0].start_idx == 0
    assert result.blocks[0].end_idx == 11
    assert len(result.blocks[0].msg_ids) == 12


def test_embedding_cache_hit_and_rebuild(tmp_path: Path) -> None:
    raw = _make_synthetic_chat(15)
    chat = parse_chat_text(raw)

    def embed_fn(texts: list[str]) -> list[list[float]]:
        return [[float(idx + 1), 1.0] for idx, _ in enumerate(texts)]

    adapter = MockEmbeddingAdapter(embed_fn)
    cache_dir = tmp_path / ".cache"

    r1 = HybridRetriever(
        chat,
        adapter,
        raw_text=raw,
        chunk_size=6,
        stride=3,
        cache_dir=cache_dir,
        rebuild=False,
    )
    assert r1.index_cache_hit is False
    assert r1.dense_available is True
    assert len(adapter.embed_calls) == 1

    # Second instance with same chat/provider/model should hit cache (0 index embed calls).
    r2 = HybridRetriever(
        chat,
        adapter,
        raw_text=raw,
        chunk_size=6,
        stride=3,
        cache_dir=cache_dir,
        rebuild=False,
    )
    assert r2.index_cache_hit is True
    assert r2.dense_available is True
    assert len(adapter.embed_calls) == 1

    # Rebuild=True forces recomputing chunk embeddings.
    r3 = HybridRetriever(
        chat,
        adapter,
        raw_text=raw,
        chunk_size=6,
        stride=3,
        cache_dir=cache_dir,
        rebuild=True,
    )
    assert r3.index_cache_hit is False
    assert r3.dense_available is True
    assert len(adapter.embed_calls) == 2

    # Different embed_model invalidates cache key.
    other_model_adapter = MockEmbeddingAdapter(
        embed_fn, embed_model="text-embedding-3-small"
    )
    r4 = HybridRetriever(
        chat,
        other_model_adapter,
        raw_text=raw,
        chunk_size=6,
        stride=3,
        cache_dir=cache_dir,
        rebuild=False,
    )
    assert r4.index_cache_hit is False
    assert len(other_model_adapter.embed_calls) == 1


def test_graceful_bm25_fallback_on_index_and_query_embedding_failures(
    tmp_path: Path,
) -> None:
    raw = _make_synthetic_chat(18)
    chat = parse_chat_text(raw)

    # Case 1: Index-time API exception falls back cleanly to BM25.
    def failing_index_embed(texts: list[str]) -> list[list[float]]:
        raise RuntimeError("Embedding endpoint unavailable")

    failing_adapter = MockEmbeddingAdapter(failing_index_embed)
    r_index_fail = HybridRetriever(
        chat,
        failing_adapter,
        raw_text=raw,
        cache_dir=tmp_path / ".cache_fail",
    )
    assert r_index_fail.dense_available is False
    assert r_index_fail.dense_fallback is True
    res1 = r_index_fail.retrieve("portal.example.org auth-wip", top_k=2, expand_n=1)
    assert res1.used_dense is False
    assert res1.dense_fallback is True
    assert "Embedding endpoint unavailable" in (res1.dense_fallback_reason or "")
    assert any("M011" in b.msg_ids for b in res1.blocks)

    # Case 2: Empty / zero-norm vectors fall back cleanly to BM25.
    zero_adapter = MockEmbeddingAdapter(lambda texts: [[0.0, 0.0] for _ in texts])
    r_zero = HybridRetriever(
        chat,
        zero_adapter,
        raw_text=raw,
        cache_dir=tmp_path / ".cache_zero",
    )
    assert r_zero.dense_available is False
    assert r_zero.dense_fallback is True
    res2 = r_zero.retrieve("database migration pakka", top_k=2, expand_n=1)
    assert res2.used_dense is False
    assert res2.dense_fallback is True
    assert any("M015" in b.msg_ids for b in res2.blocks)

    # Case 3: Index succeeds, query embedding fails -> falls back to BM25 for that query.
    def fail_on_single_query(texts: list[str]) -> list[list[float]]:
        if len(texts) == 1:
            raise RuntimeError("Query embedding rate limited")
        return [[1.0, float(i + 1)] for i, _ in enumerate(texts)]

    query_fail_adapter = MockEmbeddingAdapter(fail_on_single_query)
    r_query_fail = HybridRetriever(
        chat,
        query_fail_adapter,
        raw_text=raw,
        cache_dir=tmp_path / ".cache_qfail",
    )
    assert r_query_fail.dense_available is True
    res3 = r_query_fail.retrieve("invoice layout", top_k=2, expand_n=1)
    assert res3.used_dense is False
    assert res3.dense_fallback is True
    assert "Query embedding rate limited" in (res3.dense_fallback_reason or "")
    assert any("M006" in b.msg_ids for b in res3.blocks)
