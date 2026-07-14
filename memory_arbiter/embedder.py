from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple

# Bump when embed_text input construction, truncation strategy, or pipeline
# semantics change.  Part of embedding_space_id — changing it forces a rebuild.
EMBEDDING_PIPELINE_VERSION = 1

EncodeFn = Callable[[str], list[float]]
TokenizeFn = Callable[[str], list[int]]


@dataclass
class EmbedResult:
    """Result of a token-safe embedding call."""
    embedding: list[float]
    truncated: bool
    original_tokens: int
    used_tokens: int


@dataclass
class ManagedEmbedder:
    """Embedder with model identity and token-safe helpers."""
    encode_raw: EncodeFn
    tokenize: TokenizeFn
    model_digest: str
    embedding_space_id: str
    n_ctx: int
    reserved_tokens: int = 64
    warnings: list[str] = field(default_factory=list)

    def embed_text(
        self,
        prefix: str,
        body: str,
        max_body_chars: Optional[int] = None,
    ) -> EmbedResult:
        """Unified token-safe embedding (design doc §1.1b).

        Counts full prefix+body tokens for diagnostics, then truncates body
        if total exceeds the model context budget.  All memory/query/section
        embedding must go through this method.
        """
        full_text = prefix + body
        original_tokens = len(self.tokenize(full_text))

        body_candidate = body
        if max_body_chars is not None and len(body_candidate) > max_body_chars:
            body_candidate = body_candidate[:max_body_chars]

        token_budget = self.n_ctx - self.reserved_tokens
        candidate_tokens = len(self.tokenize(prefix + body_candidate))

        used_tokens = candidate_tokens
        if candidate_tokens > token_budget:
            lo, hi = 0, len(body_candidate)
            best = ""
            while lo <= hi:
                mid = (lo + hi) // 2
                t = len(self.tokenize(prefix + body_candidate[:mid]))
                if t <= token_budget:
                    best = body_candidate[:mid]
                    used_tokens = t
                    lo = mid + 1
                else:
                    hi = mid - 1
            body_candidate = best
            if not best:
                used_tokens = len(self.tokenize(prefix))

        final_text = prefix + body_candidate
        truncated = original_tokens > used_tokens or len(body_candidate) < len(body)

        try:
            embedding = self.encode_raw(final_text)
        except Exception:
            embedding = self.encode_raw(prefix)
            used_tokens = len(self.tokenize(prefix))
            truncated = True

        return EmbedResult(
            embedding=embedding,
            truncated=truncated,
            original_tokens=original_tokens,
            used_tokens=used_tokens,
        )


def compute_model_digest(model_path: str) -> str:
    """SHA-256 of the model file content."""
    h = hashlib.sha256()
    with open(model_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_embedding_space_id(
    model_digest: str,
    dim: int,
    pipeline_version: int,
    effective_config: dict[str, Any],
) -> str:
    """Stable vector-space identity from canonical JSON of config payload."""
    payload = {
        "provider": "gguf",
        "model_sha256": model_digest,
        "dim": dim,
        "pipeline_version": pipeline_version,
        "effective_config": effective_config,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_embedder(
    model_path: str,
    expected_dim: int,
    n_ctx: int = 2048,
    reserved_tokens: int = 64,
) -> Tuple[Optional[ManagedEmbedder], list[str]]:
    """Build a managed GGUF embedder with token-safe helpers.

    Returns (ManagedEmbedder, []) on success, (None, warnings) on failure.
    Never raises.
    """
    warnings: list[str] = []
    if not model_path or not model_path.strip():
        return None, []
    try:
        from llama_cpp import Llama
    except ImportError:
        warnings.append("llama-cpp-python not installed; auto-embedding disabled. pip install llama-cpp-python")
        return None, warnings
    if not os.path.exists(model_path):
        warnings.append(f"GGUF model not found: {model_path}; auto-embedding disabled.")
        return None, warnings
    try:
        llm = Llama(model_path=model_path, embedding=True, verbose=False, n_ctx=n_ctx)

        def encode(text: str) -> list[float]:
            data = llm.create_embedding(text)["data"][0]["embedding"]
            return [float(x) for x in data]

        def tokenize(text: str) -> list[int]:
            return llm.tokenize(text.encode("utf-8"), add_bos=False)

        sample = encode("dimension probe")
        if len(sample) != expected_dim:
            warnings.append(f"GGUF dim {len(sample)} != config vec.dim {expected_dim}; auto-embedding disabled.")
            return None, warnings

        model_digest = compute_model_digest(model_path)
        effective_config = {
            "n_ctx": n_ctx,
            "reserved_tokens": reserved_tokens,
            "max_section_chars_default": 3600,
        }
        space_id = compute_embedding_space_id(
            model_digest, expected_dim, EMBEDDING_PIPELINE_VERSION, effective_config
        )

        return ManagedEmbedder(
            encode_raw=encode,
            tokenize=tokenize,
            model_digest=model_digest,
            embedding_space_id=space_id,
            n_ctx=n_ctx,
            reserved_tokens=reserved_tokens,
            warnings=warnings,
        ), warnings
    except Exception as exc:
        warnings.append(f"GGUF embedder load failed: {exc}; auto-embedding disabled.")
        return None, warnings
