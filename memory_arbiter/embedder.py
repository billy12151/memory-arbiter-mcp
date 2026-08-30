from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from .timeutil import utc_now_iso

# Bump when embed_text input construction, truncation strategy, or pipeline
# semantics change.  Part of embedding_space_id — changing it forces a rebuild.
# v2: evidence offsets are derived from a normalization-to-source character
# map (not reverse-searched), and unit text preserves source punctuation
# spacing. Rotating the space id forces every v1 row to be rebuilt before
# evidence recall is re-enabled.
# v2.1 (no bump — vectors unchanged): the token budget clamps to
# min(n_ctx - reserved_tokens, n_batch) so truncation happens in embed_text
# instead of silently inside llama-cpp-python's embed(); the bytes reaching
# the model are identical for realistic inputs (the one exception: a
# byte-fallback character landing exactly on the cut boundary can shift the
# cut by one trailing token — the same tail-truncation regime as before), so
# the space id must NOT rotate.
EMBEDDING_PIPELINE_VERSION = 2

EncodeFn = Callable[[str], list[float]]
TokenizeFn = Callable[[str], list[int]]
# Rebuilds a CPU-only (encode, tokenize) pair from scratch; returns instead
# of raising so the caller can treat failure as "no degrade possible".
CpuRebuildFn = Callable[[], tuple["EncodeFn", "TokenizeFn"]]


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
    # llama-cpp-python's embed() silently truncates encode input to n_batch
    # tokens; embed_text's budget must clamp to the same ceiling (see
    # token_budget()). build_embedder copies the constructed instance's real
    # value here instead of assuming the library default.
    n_batch: int = 512
    warnings: list[str] = field(default_factory=list)
    last_encode_error: str | None = None
    # Device self-healing: a GPU-backed instance that starts failing at
    # runtime (driver error, device removed) degrades ONCE to a freshly
    # built CPU instance; llama.cpp cannot migrate a loaded context between
    # devices, so the only recovery is a rebuild. Restart re-probes the GPU.
    gpu_backed: bool = False
    device_degraded: bool = False
    device_degraded_at: str | None = None
    _cpu_rebuild: CpuRebuildFn | None = None
    # llama-cpp-python's GGUF inference (create_embedding AND tokenize) is not
    # thread-safe: concurrent calls on one Llama instance deadlock. The async
    # split worker runs embed on a background thread while the main thread may
    # embed a search query on the same instance. This lock serialises every
    # llama-cpp call so the two never overlap. Embed is already CPU-bound and
    # single-threaded by nature, so serialising costs nothing — it only makes
    # the existing "one caller at a time" invariant explicit. The runtime
    # GPU→CPU degrade also runs under this lock (only embed_text calls it),
    # so a rebuild can never race a concurrent encode.
    _embed_lock: threading.Lock = field(default_factory=threading.Lock)

    def token_budget(self) -> int:
        """Max tokens one embed_text call may send to the model.

        Two gates apply and the tighter one wins: the model context window
        (n_ctx - reserved_tokens, the semantic ceiling) and the library batch
        ceiling (n_batch, what one llama-cpp-python embed() forward can
        actually chew — inputs beyond it are silently truncated there).
        """
        return min(self.n_ctx - self.reserved_tokens, self.n_batch)

    def embed_text(
        self,
        prefix: str,
        body: str,
        max_body_chars: int | None = None,
    ) -> EmbedResult:
        """Unified token-safe embedding (design doc §1.1b).

        Counts full prefix+body tokens for diagnostics, then truncates body
        if total exceeds the model context budget.  All evidence-unit, query,
        and workspace-canonical embedding must go through this method.

        Thread-safety: the whole body runs under ``_embed_lock`` because both
        ``tokenize`` and ``encode_raw`` hit the underlying ``Llama`` instance,
        whose GGUF inference is not thread-safe.
        """
        with self._embed_lock:
            # Join prefix and body with a newline boundary so tokenizers don't merge
            # the trailing token of the prefix with the leading token of the body
            # (e.g. subject "cat" + content "dog" must not become "catdog").  An
            # empty prefix yields a leading newline only when the body is non-empty,
            # which models handle identically to the bare body.
            sep = "\n" if prefix and body else ""
            full_text = prefix + sep + body
            original_tokens = len(self.tokenize(full_text))

            body_candidate = body
            if max_body_chars is not None and len(body_candidate) > max_body_chars:
                body_candidate = body_candidate[:max_body_chars]

            token_budget = self.token_budget()
            candidate_tokens = len(self.tokenize(prefix + sep + body_candidate))

            used_tokens = candidate_tokens
            if candidate_tokens > token_budget:
                lo, hi = 0, len(body_candidate)
                best = ""
                while lo <= hi:
                    mid = (lo + hi) // 2
                    t = len(self.tokenize(prefix + sep + body_candidate[:mid]))
                    if t <= token_budget:
                        best = body_candidate[:mid]
                        used_tokens = t
                        lo = mid + 1
                    else:
                        hi = mid - 1
                body_candidate = best
                if not best:
                    used_tokens = len(self.tokenize(prefix))

            final_text = prefix + sep + body_candidate
            truncated = original_tokens > used_tokens or len(body_candidate) < len(body)

            try:
                embedding = self.encode_raw(final_text)
            except Exception as exc:
                if self._maybe_degrade_to_cpu(str(exc)):
                    # One retry on the fresh CPU instance; the failing call
                    # heals instead of returning a sentinel for a transient
                    # device fault.
                    try:
                        embedding = self.encode_raw(final_text)
                        return EmbedResult(
                            embedding=embedding,
                            truncated=truncated,
                            original_tokens=original_tokens,
                            used_tokens=used_tokens,
                        )
                    except Exception as retry_exc:
                        exc = retry_exc
                # The model-level failure will likely recur on the bare prefix too.
                # embed_text is a Never-raises surface: record the error and return a
                # sentinel result so the caller can surface a warning instead of
                # propagating an exception up the MCP tool call.
                self.last_encode_error = str(exc)
                try:
                    prefix_tokens = len(self.tokenize(prefix))
                except Exception:
                    # Tokenize on an unrecoverable instance must not break the
                    # Never-raises contract either.
                    prefix_tokens = 0
                return EmbedResult(
                    embedding=[],
                    truncated=True,
                    original_tokens=original_tokens,
                    used_tokens=prefix_tokens,
                )

            return EmbedResult(
                embedding=embedding,
                truncated=truncated,
                original_tokens=original_tokens,
                used_tokens=used_tokens,
            )

    def tokenize_locked(self, text: str) -> list[int]:
        """Tokenize under ``_embed_lock`` for out-of-band callers.

        llama-cpp instances are not thread-safe even for tokenize (concurrent
        tokenize + encode on one instance deadlocks). embed_text serialises
        internally, but deep diagnostics tokenizing on the shared LIVE
        instance must take the same lock — and go through this method so a
        post-degrade closure swap is always seen.
        """
        with self._embed_lock:
            return self.tokenize(text)

    def _maybe_degrade_to_cpu(self, reason: str) -> bool:
        """One-shot runtime GPU→CPU degrade. Caller must hold ``_embed_lock``.

        Only fires for instances that loaded onto a GPU and whose startup
        dimension probe succeeded (a working model that later started
        failing). Swapping ``encode_raw``/``tokenize`` drops the last
        references to the broken GPU instance, so its memory is reclaimed.
        The one-shot latch closes even on a failed rebuild: retrying a
        rebuild on every call would multiply latency for no recovery gain.
        """
        if not self.gpu_backed or self.device_degraded or self._cpu_rebuild is None:
            return False
        self.device_degraded = True
        try:
            encode, tokenize = self._cpu_rebuild()
        except Exception as exc:
            self.warnings.append(
                f"GPU encode failed ({reason}); CPU degrade rebuild also failed "
                f"({exc}) — embedder stays unavailable until restart"
            )
            return False
        self.encode_raw = encode
        self.tokenize = tokenize
        self.gpu_backed = False
        self.device_degraded_at = utc_now_iso()
        self.warnings.append(
            f"GPU encode failed ({reason}); degraded to CPU inference at "
            f"{self.device_degraded_at} — restart the service to re-probe the GPU"
        )
        return True


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
    max_section_chars: int = 3600,
) -> tuple[ManagedEmbedder | None, list[str]]:
    """Build a managed GGUF embedder with token-safe helpers.

    Device policy lives here, not in callers: when the installed
    llama-cpp-python wheel carries a GPU backend (llama_supports_gpu_offload)
    the model is constructed with ``n_gpu_layers=-1``; any construction or
    startup-probe failure falls back to a CPU instance. A GPU instance that
    fails later at runtime degrades once via ManagedEmbedder's self-heal.
    llm-level GPU offload roughly quadruples embedding throughput on Apple
    Silicon; on GPU-less hosts the probe returns False and nothing changes.

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
        def construct(offload: bool) -> Any:
            kwargs: dict[str, Any] = {
                "model_path": model_path,
                "embedding": True,
                "verbose": False,
                "n_ctx": n_ctx,
            }
            if offload:
                kwargs["n_gpu_layers"] = -1
            return Llama(**kwargs)

        def make_closures(instance: Any) -> tuple[EncodeFn, TokenizeFn]:
            def encode(text: str) -> list[float]:
                data = instance.create_embedding(text)["data"][0]["embedding"]
                if not isinstance(data, list):
                    return []
                out: list[float] = []
                for x in data:
                    if isinstance(x, (int, float)):
                        out.append(float(x))
                return out

            def tokenize(text: str) -> list[int]:
                return [int(token) for token in instance.tokenize(text.encode("utf-8"), add_bos=False)]

            return encode, tokenize

        gpu_backend = False
        try:
            import llama_cpp
            gpu_backend = bool(llama_cpp.llama_supports_gpu_offload())
        except Exception:
            gpu_backend = False

        llm: Any = None
        used_gpu = False
        if gpu_backend:
            try:
                llm = construct(True)
                used_gpu = True
            except Exception as exc:
                warnings.append(f"GPU offload construction failed ({exc}); falling back to CPU")
        if llm is None:
            llm = construct(False)

        encode, tokenize = make_closures(llm)

        try:
            sample = encode("dimension probe")
        except Exception as exc:
            if not used_gpu:
                # A CPU-side probe failure is a real load failure; let the
                # outer handler record it (old behavior).
                raise
            # A GPU-side fault can surface as a probe exception (device OOM
            # at first eval, GPU reset between construct and probe); retry
            # once on CPU before declaring the embedder unavailable.
            warnings.append(f"GPU dimension probe failed ({exc}); retrying on CPU")
            llm = construct(False)
            used_gpu = False
            encode, tokenize = make_closures(llm)
            sample = encode("dimension probe")
        if len(sample) != expected_dim and used_gpu:
            # A GPU-side fault can also masquerade as a dimension mismatch;
            # retry once on CPU before declaring a config error.
            warnings.append(
                f"GPU dimension probe returned {len(sample)} dims (expected {expected_dim}); retrying on CPU"
            )
            llm = construct(False)
            used_gpu = False
            encode, tokenize = make_closures(llm)
            sample = encode("dimension probe")
        if len(sample) != expected_dim:
            warnings.append(f"GGUF dim {len(sample)} != config vec.dim {expected_dim}; auto-embedding disabled.")
            return None, warnings

        model_digest = compute_model_digest(model_path)
        # All output-affecting config must be captured here so that changing any
        # of them yields a different embedding_space_id and forces a rebuild.
        # These values must come from the caller's real Settings, NOT literals,
        # otherwise the space-id invariant silently breaks (design doc §1.1b).
        #
        # Deliberately NOT captured (decision 2026-08-31, user-approved):
        # - n_gpu_layers (device selection): CPU vs GPU vectors differ only at
        #   numeric-noise level (cosine >= 0.9997 measured, against workspace
        #   match thresholds at 0.25 scale). Capturing it would flip the space
        #   id whenever a host's GPU availability changes and force a full
        #   evidence rebuild for noise.
        # - n_batch: not user-configurable today (library default); it shapes
        #   the token budget but cannot drift per host. If either ever becomes
        #   configurable, revisit and capture it here.
        effective_config = {
            "n_ctx": n_ctx,
            "reserved_tokens": reserved_tokens,
            "max_section_chars": max_section_chars,
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
            n_batch=int(getattr(llm, "n_batch", 512) or 512),
            warnings=warnings,
            gpu_backed=used_gpu,
            _cpu_rebuild=(lambda: make_closures(construct(False))) if used_gpu else None,
        ), warnings
    except Exception as exc:
        warnings.append(f"GGUF embedder load failed: {exc}; auto-embedding disabled.")
        return None, warnings
