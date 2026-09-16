"""eval/ fixtures integrity — P0-1 c1 (plan §2.3/§2.7-5).

Guards the shipped corpus files, not the export tool: qid coverage, key
uniqueness (content-addressed), label vocabulary, and distractor hygiene
(no twin-bucket rows, no overlap with targets).
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FIXTURES = REPO / "eval" / "fixtures" / "recall"
LABEL_VOCAB = {"relevant", "borderline", "irrelevant"}


def _load(name: str) -> list[dict]:
    return [
        json.loads(line)
        for line in (FIXTURES / name).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_queries_cover_34_with_expected_kinds() -> None:
    data = json.loads((FIXTURES / "queries.json").read_text(encoding="utf-8"))
    queries = data["queries"]
    assert data["corpus_version"] == "recall-v1"
    assert len(queries) == 34
    kinds: dict[str, int] = {}
    for row in queries:
        kinds[row["kind"]] = kinds.get(row["kind"], 0) + 1
    assert kinds == {"paraphrase": 12, "lookup": 10, "legal": 8, "far": 4}


def test_targets_content_addressed_and_unique() -> None:
    targets = _load("targets.jsonl")
    keys = [row["fixture_key"] for row in targets]
    assert len(keys) == len(set(keys)), "fixture_key collision"
    assert all(row["fixture_key"].startswith("t-") for row in targets)
    assert all(len(row["content_sha"]) == 64 for row in targets)
    assert all(row["fixture_key"] == "t-" + row["content_sha"][:12] for row in targets)


def test_labels_reference_existing_targets_only() -> None:
    targets = _load("targets.jsonl")
    labels = _load("labels.jsonl")
    known = {row["fixture_key"] for row in targets}
    for row in labels:
        assert row["label"] in {"relevant", "borderline"}, row
        assert row["fixture_key"] in known, row
    labeled_qids = {row["qid"] for row in labels}
    # 当年口径：A/B 组 22 qid 有非无关标注；C/D 组 12 qid 未标=默认无关
    assert len(labeled_qids) == 22


def test_distractors_exclude_twin_bucket_and_targets() -> None:
    targets = _load("targets.jsonl")
    distractors = _load("distractors.jsonl")
    target_shas = {row["content_sha"] for row in targets}
    for row in distractors:
        assert row["workspace_canonical"] != "mema-twin", row["fixture_key"]
        assert row["content_sha"] not in target_shas, row["fixture_key"]
        assert row["status_at_export"] == "active", row["fixture_key"]


def test_manifest_matches_files() -> None:
    manifest = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["targets"] == len(_load("targets.jsonl"))
    assert manifest["distractors"] == len(_load("distractors.jsonl"))
    assert manifest["labeled_qids"] == 22
