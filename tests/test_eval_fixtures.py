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


SIM_FIXTURES = REPO / "eval" / "fixtures" / "similarity"
SIM_LABELS = {"true_near_dup", "clearly_different", "same_entity_diff_attr", "opposite_semantics"}


def _sim_cases() -> list[dict]:
    return [
        json.loads(line)
        for line in (SIM_FIXTURES / "cases.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_similarity_corpus_shape_and_naturalness() -> None:
    cases = _sim_cases()
    counts: dict[str, int] = {}
    for case in cases:
        counts[case["label"]] = counts.get(case["label"], 0) + 1
    assert counts == {label: 12 for label in SIM_LABELS}
    assert len({case["group"] for case in cases}) == 12
    # 考题必须进得了考场：near 变体与锚 content 不得字节相同（防重门会拦截，
    # 那测的是 dedup gate 不是相似提示）。语料不凑门——subject 相似度保持
    # 自然梯度，触不触发由机制表现（首跑实证：提示率 4/12，0.95 门漏掉
    # 0.83-0.93 自然后缀档，为阈值调优供数据）。
    import difflib

    ratios = []
    for case in cases:
        if case["label"] == "true_near_dup":
            assert case["anchor"]["content"] != case["variant"]["content"]
            assert case["anchor"]["tags"] == case["variant"]["tags"]
            ratios.append(
                difflib.SequenceMatcher(
                    None, case["anchor"]["subject"], case["variant"]["subject"],
                ).ratio()
            )
    assert any(r >= 0.95 for r in ratios), "梯度缺高相似档"
    assert any(r < 0.8 for r in ratios), "梯度缺自由改述档（凑门回归哨兵）"
    for case in cases:
        if case["label"] == "clearly_different":
            assert set(case["anchor"]["tags"]).isdisjoint(case["variant"]["tags"])
        if case["label"] == "same_entity_diff_attr":
            # 形态哨兵（sweep 对抗出的污染教训）：同实体异属性类的变体
            # 不得是 near 形态（subject 高相似 + tags 全同）——标签必须与形态一致
            def _tag_jac(a: list[str], b: list[str]) -> float:
                sa = {t.casefold().strip() for t in a if t.strip()}
                sb = {t.casefold().strip() for t in b if t.strip()}
                return len(sa & sb) / len(sa | sb) if sa | sb else 0.0

            jac = _tag_jac(case["anchor"]["tags"], case["variant"]["tags"])
            assert jac <= 0.6, (case["case_id"], jac)


CONFLICT_FIXTURES = REPO / "eval" / "fixtures" / "conflict"
CONFLICT_LABELS = {"true_conflict", "coexist", "noise"}
CONFLICT_SHAPES = {"scan_evolution", "governed_negative", "write_opposition"}


def _conflict_pairs() -> list[dict]:
    return [
        json.loads(line)
        for line in (CONFLICT_FIXTURES / "pairs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_conflict_pairs_composition_and_integrity() -> None:
    pairs = _conflict_pairs()
    assert len(pairs) == 66
    for pair in pairs:
        assert pair["label"] in CONFLICT_LABELS, pair["pair_id"]
        assert pair["shape"] in CONFLICT_SHAPES, pair["pair_id"]
        left, right = pair["left"], pair["right"]
        # 成组硬约束：对内成员同 workspace（semantic notice 拒绝混合 workspace 快照）
        assert left["workspace"] == right["workspace"], pair["pair_id"]
        # 重放保真：成员正文非空且互不字节相同（防重门会拦截）
        assert left["content"] and right["content"], pair["pair_id"]
        assert left["content"] != right["content"], pair["pair_id"]
    labels = {p["label"] for p in pairs}
    assert labels == CONFLICT_LABELS
    # owner 判定来源的对不得为空（ground truth 主体）
    owner_sourced = [p for p in pairs if p["label_source"].startswith("owner_")]
    assert len(owner_sourced) >= 30


def test_conflict_overrides_audit_trail_exists() -> None:
    overrides = json.loads(
        (CONFLICT_FIXTURES / "label_overrides.json").read_text(encoding="utf-8"),
    )
    assert "cf-coexist-796-797" in overrides  # 改判 true_conflict（演进取代未标注）
    pair_ids = {p["pair_id"] for p in _conflict_pairs()}
    dropped = {pid for pid, (label, _) in overrides.items() if label is None}
    assert not (dropped & pair_ids), "剔除对不得留在正式对集"
