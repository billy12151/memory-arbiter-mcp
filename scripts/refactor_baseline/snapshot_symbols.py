"""Capture a refactor safety-net snapshot of the public+private symbol surface.

Records, for the facade modules that Phase 3/4/5 will turn into packages:
  - every attribute name (dir())
  - each callable's inspect.signature (so a moved/renamed param is caught)
  - each callable's __module__ and __qualname__ (so a delegation that changes
    the defining module is an explicit, whitelisted decision — R6)

The snapshot is JSON so a later phase can diff "before" vs "after" and fail
if a name vanished, a signature drifted, or a __module__ changed unexpectedly.

Usage:
    .venv/bin/python scripts/refactor_baseline/snapshot_symbols.py \
        --out scripts/refactor_baseline/symbols_before.json
"""
from __future__ import annotations

import argparse
import inspect
import json
from typing import Any

# Facade modules whose symbol surface must survive the split (R6/R8).
TARGET_MODULES = [
    "memory_arbiter.db",
    "memory_arbiter.tools",
    "memory_arbiter.search",
    "memory_arbiter.doctor",
]

# Facade classes whose method surface must survive (delegation keeps names).
TARGET_CLASSES = [
    ("memory_arbiter.db", "MemoryDB"),
    ("memory_arbiter.tools", "MemoryTools"),
]


def _describe(obj: Any) -> dict[str, Any]:
    desc: dict[str, Any] = {
        "kind": type(obj).__name__,
        "module": getattr(obj, "__module__", None),
        "qualname": getattr(obj, "__qualname__", None),
    }
    if callable(obj):
        try:
            desc["signature"] = str(inspect.signature(obj))
        except (TypeError, ValueError):
            desc["signature"] = None
    return desc


def snapshot_module(module_name: str) -> dict[str, Any]:
    module = __import__(module_name, fromlist=["*"])
    attrs: dict[str, Any] = {}
    for name in dir(module):
        # Skip dunder import machinery; keep private single-underscore names
        # because tests and cross-module code depend on them (R8).
        if name.startswith("__") and name.endswith("__"):
            continue
        try:
            attrs[name] = _describe(getattr(module, name))
        except Exception as exc:  # pragma: no cover - defensive
            attrs[name] = {"kind": "unreadable", "error": str(exc)}
    return attrs


def snapshot_class(module_name: str, class_name: str) -> dict[str, Any]:
    module = __import__(module_name, fromlist=[class_name])
    cls = getattr(module, class_name)
    methods: dict[str, Any] = {}
    for name in dir(cls):
        if name.startswith("__") and name.endswith("__"):
            continue
        try:
            methods[name] = _describe(getattr(cls, name))
        except Exception as exc:  # pragma: no cover - defensive
            methods[name] = {"kind": "unreadable", "error": str(exc)}
    return methods


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    snapshot = {
        "modules": {m: snapshot_module(m) for m in TARGET_MODULES},
        "classes": {
            f"{mod}.{cls}": snapshot_class(mod, cls) for mod, cls in TARGET_CLASSES
        },
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(snapshot, fh, ensure_ascii=False, indent=2, sort_keys=True)
    counts = {
        "modules": {m: len(a) for m, a in snapshot["modules"].items()},
        "classes": {c: len(a) for c, a in snapshot["classes"].items()},
    }
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
