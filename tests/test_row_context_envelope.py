"""mDeBERTa 判定输入上下文化（0.17.1 owner ②）：row_window 契约行为测试。

钉死的契约：
- A-cross 判定输入= subject + 对立行 + 前后各 1 句窗口
  （semantic_judge.row_window，对立行在窗口内且不截断）；
- 判定输入无 context 标记词——行上下文 envelope（row_context_text）是
  抽槽范式的辅助件，已随该范式整体删除（0.17.0 E1/E2 → 0.17.1 退役）；
- 裸行文本保留给守卫/锚/notice 值（grounding 契约不变）。
"""
from __future__ import annotations

from typing import Any

import tests.test_vnext_evidence as tv

# ── 共享场景构造 ───────────────────────────────────────────────────────────────

_OWN_CONTENT = (
    "# 本机配置\n"
    "连接池上限为 99，队列长度为 99。\n"
    "连接池上限为 100，队列长度为 100。\n"
)
_PEER_CONTENT = "# 服务配置\n连接池上限为 300，队列长度为 4。"


def _peer_hit(peer: dict[str, Any], row_id: int, text: str, distance: float,
              start: int = 0, end: int | None = None) -> dict[str, Any]:
    return {
        "memory_id": int(peer["id"]), "id": row_id, "kind": "text", "text": text,
        "start_offset": start, "end_offset": len(text) if end is None else end,
        "distance": distance, "memory_row_version": 1,
    }


# ── A-cross 判定输入带 row_window、无 context 标记词 ───────────────────────────

def test_cross_dispatch_bare_sentence_pairs_no_context(tmp_path, monkeypatch) -> None:
    """0.17.1: the judge sees BARE row texts — the row-context envelope was a
    slot-extraction aid and dies with that paradigm (owner 范围收窄)."""
    tools = tv.make_tools(tmp_path, semantic_enabled=True)
    tools.settings.semantic_conflict_on_write = "off"
    peer1 = tools.memory_write(content=_PEER_CONTENT, subject="ctx-peer", tags=[])["data"]
    new = tools.memory_write(content=_OWN_CONTENT, subject="ctx-own", tags=[])["data"]
    assert tools.wait_semantic_worker_drained(timeout=5)
    captured: list[tuple[str, str]] = []

    class Capture:
        @staticmethod
        def judge_pairs(pairs):
            captured.extend(pairs)
            from memory_arbiter.semantic_judge import PairVerdict
            return [PairVerdict("no_conflict",
                                {"conflict": 0.0, "no_conflict": 1.0, "possible_conflict": 0.0},
                                None, "test") for _ in pairs]

    sentence = "连接池上限为 300，队列长度为 4。"
    sentence_start = _PEER_CONTENT.index(sentence)
    a_hits = [_peer_hit(peer1, 101, sentence, 0.1,
                        start=sentence_start,
                        end=sentence_start + len(sentence))]
    monkeypatch.setattr(
        tools.db, "row_knn",
        lambda embedding, **kw: ([{"memory_id": int(peer1["id"]), "id": 1, "kind": "subject",
                                   "text": "ctx-peer", "distance": 0.5, "memory_row_version": 1}]
                                 if kw.get("subject_rows_only") else list(a_hits)),
    )
    import memory_arbiter.pipeline.gates as _gates
    monkeypatch.setattr(
        _gates, "candidate_cos_gate",
        lambda own, hits, vecs: ([(h, 0.85) for h in hits], [], []),
    )
    monkeypatch.setattr(tools, "_ensure_semantic_backend", lambda: Capture)

    receipt = tools._process_semantic_conflict_job(new["id"], tv._job_snapshot(tools, new["id"]))

    assert captured, "A-cross must consult the judge on the sentence pair"
    # 0.17.1 owner ②：判定输入= subject+对立行+前后句窗口——对立行在窗口内、
    # 无 context 标记词（该机制已随抽槽范式退役）
    for text_a, text_b in captured:
        assert "context" not in text_a and "context" not in text_b
    assert any(sentence in text_b for text_a, text_b in captured), \
        "peer 对立行必须出现在窗口内"
