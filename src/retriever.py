"""Hybrid BM25 and API-embedding retriever with RRF and chronological context expansion."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi

from src.compiler import (
    DEFAULT_CACHE_DIR,
    compute_chat_sha256,
    format_messages_window,
)
from src.llm import BaseLLMAdapter
from src.parser import ParsedChat, ParsedMessage

INDEX_VERSION = "1.0"
RRF_K = 60

_COMPOUND_TOKEN_RE = re.compile(r"[^\W_]+(?:[._/-][^\W_]+)*", re.UNICODE)
_SPLIT_SEPARATOR_RE = re.compile(r"[._/-]+")


class ChatChunk(BaseModel):
    """Overlapping chronological message window used as the retrieval unit."""

    chunk_index: int
    start_idx: int
    end_idx: int
    msg_ids: list[str]
    line_start: int
    line_end: int
    text: str


class RetrievalHit(BaseModel):
    """Single ranked chunk produced by BM25, dense embeddings, or hybrid RRF."""

    chunk_index: int
    start_idx: int
    end_idx: int
    rrf_score: float
    bm25_rank: int | None = None
    dense_rank: int | None = None
    bm25_score: float = 0.0
    dense_score: float | None = None


class EvidenceBlock(BaseModel):
    """Merged contiguous chronological block of expanded message matches."""

    start_idx: int
    end_idx: int
    line_start: int
    line_end: int
    msg_ids: list[str]
    messages: list[ParsedMessage]
    formatted_text: str
    includes_tail: bool = False


class RetrievalResult(BaseModel):
    """Complete retrieval output including fused hits, expanded blocks, and flags."""

    query: str
    hits: list[RetrievalHit] = Field(default_factory=list)
    blocks: list[EvidenceBlock] = Field(default_factory=list)
    used_dense: bool = False
    dense_fallback: bool = False
    dense_fallback_reason: str | None = None
    cache_hit: bool = False
    cache_path: str | None = None
    formatted_context: str = ""


def tokenize_text(text: str) -> list[str]:
    """Tokenize English and Hinglish text while preserving and splitting compound tokens."""
    tokens: list[str] = []
    for match in _COMPOUND_TOKEN_RE.finditer(text.lower()):
        compound = match.group(0)
        tokens.append(compound)
        if _SPLIT_SEPARATOR_RE.search(compound):
            for part in _SPLIT_SEPARATOR_RE.split(compound):
                if part:
                    tokens.append(part)
    return tokens


def build_chunks(
    messages: Sequence[ParsedMessage],
    *,
    chunk_size: int = 6,
    stride: int = 3,
) -> list[ChatChunk]:
    """Build overlapping chronological message chunks formatted with full metadata."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1.")
    if stride < 1 or stride > chunk_size:
        raise ValueError("stride must be between 1 and chunk_size inclusive.")
    if not messages:
        return []

    msg_list = list(messages)
    total = len(msg_list)
    chunks: list[ChatChunk] = []

    start = 0
    while start < total:
        end = min(total, start + chunk_size)
        window = msg_list[start:end]
        chunks.append(
            ChatChunk(
                chunk_index=len(chunks),
                start_idx=start,
                end_idx=end - 1,
                msg_ids=[m.msg_id for m in window],
                line_start=window[0].line_start,
                line_end=window[-1].line_end,
                text=format_messages_window(window),
            )
        )
        if end >= total:
            break
        start += stride
    return chunks


def compute_embedding_cache_key(
    *,
    chat_sha256: str,
    provider: str,
    embed_model: str,
    index_version: str = INDEX_VERSION,
    chunk_size: int = 6,
    stride: int = 3,
) -> str:
    """Compute the content-addressed cache key for chunk embeddings."""
    payload = json.dumps(
        {
            "chat_sha256": chat_sha256,
            "provider": provider,
            "embed_model": embed_model,
            "index_version": index_version,
            "chunk_size": chunk_size,
            "stride": stride,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalize_embedding_matrix(
    vectors: Sequence[Sequence[float]], expected_count: int
) -> np.ndarray:
    """Validate and L2-normalize an embedding vector batch into float32 matrix."""
    if len(vectors) != expected_count:
        raise ValueError(
            f"Expected {expected_count} embedding vectors, got {len(vectors)}."
        )
    if expected_count == 0:
        return np.empty((0, 0), dtype=np.float32)

    first_dim = len(vectors[0])
    if first_dim == 0:
        raise ValueError("Embedding vectors are empty.")
    for idx, vec in enumerate(vectors):
        if len(vec) != first_dim:
            raise ValueError(
                f"Inconsistent embedding dimension at index {idx}: {len(vec)} != {first_dim}."
            )

    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("Embedding matrix contains non-finite values or invalid shape.")

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.all(norms <= 1e-12):
        raise ValueError("All embedding vectors have zero norm.")

    safe_norms = np.where(norms > 1e-12, norms, 1.0)
    return matrix / safe_norms


def _merge_index_spans(
    spans: Sequence[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Merge overlapping or adjacent inclusive index spans into sorted disjoint spans."""
    if not spans:
        return []
    ordered = sorted(spans, key=lambda pair: (pair[0], pair[1]))
    merged: list[tuple[int, int]] = [ordered[0]]
    for start, end in ordered[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + 1:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _span_is_covered(
    target: tuple[int, int], covered_spans: Sequence[tuple[int, int]]
) -> bool:
    """Return True if target [start, end] is completely covered by merged spans."""
    t_start, t_end = target
    for c_start, c_end in covered_spans:
        if c_start <= t_start and t_end <= c_end:
            return True
    return False


def _rank_bm25(
    bm25_scores: Sequence[float],
    tokenized_chunks: Sequence[Sequence[str]],
    query_tokens: Sequence[str],
) -> tuple[dict[int, int], list[int]]:
    """Rank chunks matching at least one query token; fall back to overlap on tiny corpora."""
    query_set = set(query_tokens)
    overlap_counts = [
        len(query_set & set(chunk_tokens)) for chunk_tokens in tokenized_chunks
    ]
    matched_indices = [idx for idx, count in enumerate(overlap_counts) if count > 0]
    if not matched_indices:
        return {}, [0] * len(tokenized_chunks)

    matched_indices.sort(
        key=lambda idx: (-float(bm25_scores[idx]), -overlap_counts[idx], idx)
    )
    ranks: dict[int, int] = {}
    current_rank = 1
    prev_key: tuple[float, int] | None = None
    for pos, idx in enumerate(matched_indices, start=1):
        key = (round(float(bm25_scores[idx]), 12), overlap_counts[idx])
        if prev_key is None or key != prev_key:
            current_rank = pos
            prev_key = key
        ranks[idx] = current_rank
    return ranks, overlap_counts


def _rank_dense(dense_scores: np.ndarray) -> dict[int, int]:
    """Assign 1-based dense ranks from cosine similarity scores."""
    indices = list(range(len(dense_scores)))
    indices.sort(key=lambda idx: (-float(dense_scores[idx]), idx))
    ranks: dict[int, int] = {}
    current_rank = 1
    prev_score: float | None = None
    for pos, idx in enumerate(indices, start=1):
        score = round(float(dense_scores[idx]), 12)
        if prev_score is None or score != prev_score:
            current_rank = pos
            prev_score = score
        ranks[idx] = current_rank
    return ranks


class HybridRetriever:
    """Hybrid BM25 and dense embedding retriever with RRF and +/- N expansion."""

    def __init__(
        self,
        chat: ParsedChat,
        adapter: BaseLLMAdapter | None = None,
        *,
        raw_text: str | None = None,
        chunk_size: int = 6,
        stride: int = 3,
        cache_dir: Path | None = None,
        rebuild: bool = False,
    ) -> None:
        self.chat = chat
        self.adapter = adapter
        self.chunk_size = chunk_size
        self.stride = stride
        self.cache_dir = cache_dir if cache_dir is not None else DEFAULT_CACHE_DIR
        self.chat_sha256 = compute_chat_sha256(chat, raw_text=raw_text)

        self.chunks = build_chunks(
            chat.messages, chunk_size=chunk_size, stride=stride
        )
        self._tokenized_chunks = [tokenize_text(chunk.text) for chunk in self.chunks]
        self._bm25 = BM25Okapi(self._tokenized_chunks) if self.chunks else None

        self._embeddings: np.ndarray | None = None
        self.dense_available = False
        self.dense_fallback = False
        self.dense_fallback_reason: str | None = None
        self.index_cache_hit = False
        self.cache_path: Path | None = None

        self._init_dense_index(rebuild=rebuild)

    def _cache_paths(self) -> tuple[Path, Path]:
        assert self.adapter is not None
        key = compute_embedding_cache_key(
            chat_sha256=self.chat_sha256,
            provider=self.adapter.provider_name,
            embed_model=self.adapter.embed_model,
            index_version=INDEX_VERSION,
            chunk_size=self.chunk_size,
            stride=self.stride,
        )
        prefix = key[:24]
        return (
            self.cache_dir / f"embeddings_{prefix}.npz",
            self.cache_dir / f"chunks_{prefix}.meta.json",
        )

    def _try_load_cached_embeddings(
        self, matrix_path: Path, meta_path: Path
    ) -> np.ndarray | None:
        if not (matrix_path.is_file() and meta_path.is_file()):
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            cached_chunks = meta.get("chunks", [])
            if len(cached_chunks) != len(self.chunks):
                return None
            for expected, recorded in zip(self.chunks, cached_chunks):
                if (
                    recorded.get("chunk_index") != expected.chunk_index
                    or recorded.get("msg_ids") != expected.msg_ids
                ):
                    return None
            with np.load(matrix_path, allow_pickle=False) as archive:
                matrix = np.asarray(archive["embeddings"], dtype=np.float32)
            if (
                matrix.ndim != 2
                or matrix.shape[0] != len(self.chunks)
                or matrix.shape[1] == 0
                or not np.isfinite(matrix).all()
            ):
                return None
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            if np.all(norms <= 1e-12):
                return None
            return matrix
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _save_cached_embeddings(
        self, matrix_path: Path, meta_path: Path, matrix: np.ndarray
    ) -> None:
        assert self.adapter is not None
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        meta_payload: dict[str, Any] = {
            "chat_sha256": self.chat_sha256,
            "provider": self.adapter.provider_name,
            "embed_model": self.adapter.embed_model,
            "index_version": INDEX_VERSION,
            "chunk_size": self.chunk_size,
            "stride": self.stride,
            "chunks": [
                {
                    "chunk_index": c.chunk_index,
                    "start_idx": c.start_idx,
                    "end_idx": c.end_idx,
                    "msg_ids": c.msg_ids,
                    "line_start": c.line_start,
                    "line_end": c.line_end,
                }
                for c in self.chunks
            ],
        }
        with matrix_path.open("wb") as handle:
            np.savez(handle, embeddings=matrix)
        meta_path.write_text(
            json.dumps(meta_payload, indent=2), encoding="utf-8"
        )

    def _init_dense_index(self, *, rebuild: bool) -> None:
        if not self.chunks:
            return
        if self.adapter is None:
            self.dense_fallback = True
            self.dense_fallback_reason = "No embedding adapter configured."
            return

        matrix_path, meta_path = self._cache_paths()
        self.cache_path = matrix_path

        if not rebuild:
            cached = self._try_load_cached_embeddings(matrix_path, meta_path)
            if cached is not None:
                self._embeddings = cached
                self.dense_available = True
                self.index_cache_hit = True
                return

        try:
            raw_vectors = self.adapter.embed_texts([c.text for c in self.chunks])
            matrix = _normalize_embedding_matrix(raw_vectors, len(self.chunks))
            self._embeddings = matrix
            self.dense_available = True
            self.index_cache_hit = False
            self._save_cached_embeddings(matrix_path, meta_path, matrix)
        except Exception as exc:
            self._embeddings = None
            self.dense_available = False
            self.dense_fallback = True
            self.dense_fallback_reason = str(exc)

    def _embed_query_vector(self, query: str) -> tuple[np.ndarray | None, str | None]:
        if not self.dense_available or self.adapter is None or self._embeddings is None:
            return None, self.dense_fallback_reason

        try:
            vectors = self.adapter.embed_texts([query])
            matrix = _normalize_embedding_matrix(vectors, 1)
        except Exception as exc:
            return None, str(exc)

        query_vec = matrix[0]
        if query_vec.shape[0] != self._embeddings.shape[1]:
            return (
                None,
                f"Query embedding dimension {query_vec.shape[0]} != index dimension {self._embeddings.shape[1]}.",
            )
        return query_vec, None

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        expand_n: int = 3,
        include_tail: bool = True,
        tail_size: int = 5,
    ) -> RetrievalResult:
        """Retrieve top-k RRF-fused chunks and merge +/- N expanded evidence blocks."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if expand_n < 0:
            raise ValueError("expand_n must be >= 0.")

        if not self.chunks:
            return RetrievalResult(
                query=query,
                used_dense=False,
                dense_fallback=self.dense_fallback,
                dense_fallback_reason=self.dense_fallback_reason,
                cache_hit=self.index_cache_hit,
                cache_path=str(self.cache_path) if self.cache_path else None,
            )

        query_tokens = tokenize_text(query)
        assert self._bm25 is not None
        bm25_scores = (
            self._bm25.get_scores(query_tokens)
            if query_tokens
            else np.zeros(len(self.chunks), dtype=np.float64)
        )
        bm25_ranks, overlap_counts = _rank_bm25(
            bm25_scores, self._tokenized_chunks, query_tokens
        )

        query_vec, query_fallback_reason = self._embed_query_vector(query)
        used_dense = query_vec is not None
        dense_fallback = self.dense_fallback or not used_dense
        fallback_reason = query_fallback_reason or self.dense_fallback_reason

        dense_scores: np.ndarray | None = None
        dense_ranks: dict[int, int] = {}
        if used_dense and query_vec is not None and self._embeddings is not None:
            dense_scores = self._embeddings @ query_vec
            dense_ranks = _rank_dense(dense_scores)

        hits: list[RetrievalHit] = []
        for idx, chunk in enumerate(self.chunks):
            r_bm25 = bm25_ranks.get(idx)
            r_dense = dense_ranks.get(idx)
            score = 0.0
            if r_bm25 is not None:
                score += 1.0 / (RRF_K + r_bm25)
            if r_dense is not None:
                score += 1.0 / (RRF_K + r_dense)
            hits.append(
                RetrievalHit(
                    chunk_index=chunk.chunk_index,
                    start_idx=chunk.start_idx,
                    end_idx=chunk.end_idx,
                    rrf_score=score,
                    bm25_rank=r_bm25,
                    dense_rank=r_dense,
                    bm25_score=float(bm25_scores[idx]),
                    dense_score=float(dense_scores[idx])
                    if dense_scores is not None
                    else None,
                )
            )

        hits.sort(
            key=lambda h: (
                -h.rrf_score,
                -overlap_counts[h.chunk_index],
                h.chunk_index,
            )
        )
        top_hits = hits[: min(top_k, len(hits))]

        max_msg_idx = len(self.chat.messages) - 1
        expanded_spans = [
            (
                max(0, hit.start_idx - expand_n),
                min(max_msg_idx, hit.end_idx + expand_n),
            )
            for hit in top_hits
        ]
        merged_spans = _merge_index_spans(expanded_spans)

        tail_span: tuple[int, int] | None = None
        if include_tail and tail_size > 0 and self.chat.messages:
            candidate_tail = (max(0, len(self.chat.messages) - tail_size), max_msg_idx)
            if not _span_is_covered(candidate_tail, merged_spans):
                tail_span = candidate_tail
                merged_spans = _merge_index_spans([*merged_spans, candidate_tail])

        blocks: list[EvidenceBlock] = []
        for s_idx, e_idx in merged_spans:
            block_msgs = self.chat.messages[s_idx : e_idx + 1]
            includes_tail = bool(
                tail_span is not None and s_idx <= tail_span[1] and e_idx >= tail_span[0]
            )
            blocks.append(
                EvidenceBlock(
                    start_idx=s_idx,
                    end_idx=e_idx,
                    line_start=block_msgs[0].line_start,
                    line_end=block_msgs[-1].line_end,
                    msg_ids=[m.msg_id for m in block_msgs],
                    messages=block_msgs,
                    formatted_text=format_messages_window(block_msgs),
                    includes_tail=includes_tail,
                )
            )

        formatted_sections = [
            (
                f"Evidence block {idx} ({block.msg_ids[0]}-{block.msg_ids[-1]}, "
                f"L{block.line_start}-L{block.line_end}):\n{block.formatted_text}"
            )
            for idx, block in enumerate(blocks, start=1)
        ]

        return RetrievalResult(
            query=query,
            hits=top_hits,
            blocks=blocks,
            used_dense=used_dense,
            dense_fallback=dense_fallback,
            dense_fallback_reason=fallback_reason if dense_fallback else None,
            cache_hit=self.index_cache_hit,
            cache_path=str(self.cache_path) if self.cache_path else None,
            formatted_context="\n\n".join(formatted_sections),
        )
