from __future__ import annotations

import os
from typing import Callable, Optional, Tuple

EncodeFn = Callable[[str], list[float]]


def build_embedder(model_path: str, expected_dim: int) -> Tuple[Optional[EncodeFn], list[str]]:
    """Build a GGUF embedding function, degrading to warnings instead of raising."""
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
        llm = Llama(model_path=model_path, embedding=True, verbose=False)

        def encode(text: str) -> list[float]:
            data = llm.create_embedding(text)["data"][0]["embedding"]
            return [float(x) for x in data]

        sample = encode("dimension probe")
        if len(sample) != expected_dim:
            warnings.append(f"GGUF dim {len(sample)} != config vec.dim {expected_dim}; auto-embedding disabled.")
            return None, warnings
        return encode, warnings
    except Exception as exc:
        warnings.append(f"GGUF embedder load failed: {exc}; auto-embedding disabled.")
        return None, warnings
