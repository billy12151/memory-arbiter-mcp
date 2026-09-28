"""0.17.0 E1/E2（owner 2026-09-25 方案，mema 行上下文 envelope）：检测行上下文行为测试。

钉死的契约：
- row_context_text 构造：标题+前/后邻行、空行/分隔行/标题自身跳过、分部截断、空返回；
- A-cross 两侧 envelope 带 context、quote 仍是主行原文（grounding 契约不变）；
- internal keeper 不带 context（范围外，B 通道回归钉保持逐位一致）；
- C 通道 peer 侧带 context、claim 侧不带（D3 翻案）；
- pair-v10 渲染：带 context 出「上下文」段、无 context 渲染零变化（向后兼容）；
- 值取自上下文被 grounding 挡住（qwen_unverified）——D4 契约机制性成立。
"""
from __future__ import annotations

from typing import Any

import pytest

import tests.test_vnext_evidence as tv
from memory_arbiter.rowseg import row_context_text

# pair-v10 系统提示词逐位钉（对抗 review P3：子串缺席防不住其他措辞漂移）。
# 改动系统词必须显式 bump PAIR_PROMPT_VERSION 并更新此钉。


# ── row_context_text 构造 ─────────────────────────────────────────────────────

_CONTENT = (
    "# 连接池配置\n"
    "超时阈值为 500ms。\n"
    "上限调整为 200。\n"
    "队列为 4。\n"
    "\n"
    "# 其他节\n"
    "回滚窗口为 30 分钟。\n"
)


def _span(text: str) -> tuple[int, int]:
    start = _CONTENT.index(text)
    return start, start + len(text)


def test_row_context_text_assembly() -> None:
    start, end = _span("上限调整为 200。")
    assert row_context_text(_CONTENT, start, end) == (
        "连接池配置 / 超时阈值为 500ms。 / 队列为 4。"
    )


def test_row_context_text_skips_blank_heading_and_separator_lines() -> None:
    content = (
        "# 标题\n"
        "| 名 | 值 |\n"
        "|---|---|\n"
        "| 上限 | 200 |\n"
        "\n"
        "下一段内容够长可以成行。\n"
    )
    start = content.index("| 上限 | 200 |")
    end = start + len("| 上限 | 200 |")
    # 前邻行跳过分隔行取表头（原始行文本，同 block）；后方向跨空行屏障不再取
    # 邻行（对抗 review P3：异 block 归属污染）。
    assert row_context_text(content, start, end) == "标题 / | 名 | 值 |"


def test_row_context_text_blank_line_blocks_cross_table_neighbors() -> None:
    content = (
        "# 表一\n"
        "| a | 1 |\n"
        "| b | 2 |\n"
        "\n"
        "# 表二\n"
        "| c | 3 |\n"
    )
    start = content.index("| c | 3 |")
    end = start + len("| c | 3 |")
    # 跨空行屏障后 prev 不再取表一的数据行（实测过的归属污染形态）。
    assert row_context_text(content, start, end) == "表二"


def test_row_context_text_caps_per_part_and_total() -> None:
    heading = "#" + " 标" * 60  # 远超 80 字
    prev = "前" * 200
    nxt = "后" * 200
    content = f"{heading}\n{prev}\n主行内容一句话。\n{nxt}\n"
    start = content.index("主行内容一句话。")
    end = start + len("主行内容一句话。")
    ctx = row_context_text(content, start, end)
    parts = ctx.split(" / ")
    assert len(parts[0]) <= 80 and len(parts[1]) <= 110 and len(parts[2]) <= 110
    assert len(ctx) <= 300


def test_row_context_text_empty_cases() -> None:
    assert row_context_text("", 0, 0) == ""
    # 单行无标题无邻行 → 空（调用方不设 context 键）
    assert row_context_text("只有一句。", 0, len("只有一句。")) == ""
    # 跨行句：所在源行不算邻行
    content = "前一句在这。\n跨行的主句跨越了\n两行排版。\n后句。\n"
    start = content.index("跨行的主句")
    end = content.index("两行排版。") + len("两行排版。")
    assert row_context_text(content, start, end) == "前一句在这。 / 后句。"


# ── 共享场景构造 ───────────────────────────────────────────────────────────────

_OWN_CONTENT = (
    "# 本机配置\n"
    "连接池上限为 99，队列长度为 99。\n"
    "连接池上限为 100，队列长度为 100。\n"
)
_PEER_CONTENT = "# 服务配置\n连接池上限为 300，队列长度为 4。"


class _EnvCapture:
    """按 env 原样捕获 classify_pair 调用，返回可配置的四字段抽取。"""

    calls: "list[tuple[dict[str, Any], dict[str, Any]]]" = []
    reply: dict[str, str] = {
        "attribute_a": "连接池上限", "value_a": "99",
        "attribute_b": "连接池上限", "value_b": "100",
    }

    @classmethod
    def reset(cls) -> None:
        cls.calls = []
        cls.reply = {
            "attribute_a": "连接池上限", "value_a": "99",
            "attribute_b": "连接池上限", "value_b": "100",
        }

    @classmethod
    def classify_pair(cls, left: dict[str, Any], right: "dict[str, Any]", **kw: Any) -> ModelSignal:
        cls.calls.append((dict(left), dict(right)))
        return ModelSignal(True, "attribute_value_extraction", None, "", dict(cls.reply), None)


def _peer_hit(peer: dict[str, Any], row_id: int, text: str, distance: float,
              start: int = 0, end: int | None = None) -> dict[str, Any]:
    return {
        "memory_id": int(peer["id"]), "id": row_id, "kind": "text", "text": text,
        "start_offset": start, "end_offset": len(text) if end is None else end,
        "distance": distance, "memory_row_version": 1,
    }


# ── A-cross 两侧 envelope 带 context；quote 保持主行；internal 不带 ─────────────

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
    for text_a, text_b in captured:
        assert "context" not in text_a and "context" not in text_b
    # the peer side is exactly the row sentence
    assert any(text_b == sentence for text_a, text_b in captured)


