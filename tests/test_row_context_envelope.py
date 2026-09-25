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
from memory_arbiter.semantic_conflict import (
    AttributeValueExtraction,
    LocalGGUFSemanticBackend,
    ModelSignal,
    PAIR_PROMPT_VERSION,
    _PAIR_PROMPT,
    _PAIR_PROMPT_EN,
    evaluate_single_direction_extraction,
)

# pair-v10 系统提示词逐位钉（对抗 review P3：子串缺席防不住其他措辞漂移）。
# 改动系统词必须显式 bump PAIR_PROMPT_VERSION 并更新此钉。
_PAIR_PROMPT_SHA = "82d6543575d5ce085f53b2c7f75392c6af6905a52e792e5799d8d910e6276959"
_PAIR_PROMPT_EN_SHA = "db00b8c41cd749284a705f9fa07b62e33b4506a645ba9a4186a624300b5acc56"


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

def test_cross_dispatch_envs_carry_context(tmp_path, monkeypatch) -> None:
    tools = tv.make_tools(tmp_path, semantic_enabled=True)
    tools.settings.semantic_conflict_on_write = "off"
    peer1 = tools.memory_write(content=_PEER_CONTENT, subject="ctx-peer", tags=[])["data"]
    new = tools.memory_write(content=_OWN_CONTENT, subject="ctx-own", tags=[])["data"]
    assert tools.wait_semantic_worker_drained(timeout=5)
    _EnvCapture.reset()
    monkeypatch.setattr(tools, "_ensure_semantic_backend", lambda: _EnvCapture)

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

    receipt = tools._process_semantic_conflict_job(new["id"], tv._job_snapshot(tools, new["id"]))

    cross_calls = [
        (left, right) for left, right in _EnvCapture.calls
        if "context" in left or "context" in right
    ]
    assert cross_calls, "A-cross Qwen dispatch must carry row context on the envelopes"
    left, right = cross_calls[0]
    # 两条 own 主行都可能成为被派发侧——各自的 context 预期不同但都确定。
    expected_own_context = {
        "连接池上限为 99，队列长度为 99。": "本机配置 / 连接池上限为 100，队列长度为 100。",
        "连接池上限为 100，队列长度为 100。": "本机配置 / 连接池上限为 99，队列长度为 99。",
    }
    assert left["quote"] in expected_own_context
    assert left["context"] == expected_own_context[left["quote"]]
    assert right["context"] == "服务配置"
    assert right["quote"] == sentence
    # internal keeper（范围外）不带 context；无 claims → C 不派发
    internal_calls = [
        (left, right) for left, right in _EnvCapture.calls
        if "context" not in left and "context" not in right
    ]
    assert internal_calls, "internal keepers ride without context (scope pin)"
    assert receipt["qwen_budget"]["internal"] == 1


# ── C 通道：peer 侧带 context、claim 侧不带（D3 翻案）──────────────────────────

def test_channel_c_right_env_context_claim_side_none(tmp_path, monkeypatch) -> None:
    tools = tv.make_tools(tmp_path, semantic_enabled=True)
    tools.settings.semantic_conflict_on_write = "off"
    peer1 = tools.memory_write(content=_PEER_CONTENT, subject="c-ctx-peer", tags=[])["data"]
    new = tools.memory_write(
        content="连接池上限为 99。", subject="c-ctx-own", tags=[],
        claims=[{"attr": "连接池上限", "value": "99"}],
    )["data"]
    assert tools.wait_semantic_worker_drained(timeout=5)
    _EnvCapture.reset()
    _EnvCapture.reply = {
        "attribute_a": "连接池上限", "value_a": "99",
        "attribute_b": "连接池上限", "value_b": "300",
    }
    monkeypatch.setattr(tools, "_ensure_semantic_backend", lambda: _EnvCapture)
    sentence = "连接池上限为 300，队列长度为 4。"
    sentence_start = _PEER_CONTENT.index(sentence)
    hits = [_peer_hit(peer1, 201, sentence, 0.2,
                      start=sentence_start, end=sentence_start + len(sentence))]
    monkeypatch.setattr(tools.db, "row_knn", lambda embedding, **kw: list(hits))
    monkeypatch.setattr(
        tools.db.evidence, "row_vectors_for_ids",
        lambda ids, conn=None: {int(i): [0.4, 0.9] for i in ids},
    )

    result = tools._evidence.check_claim_sentence_conflicts(
        int(new["id"]), tv._job_snapshot(tools, new["id"]),
        skip_peers=None, allowed_memory_ids=[int(peer1["id"])],
        notices_used=0, budget_sink=None, deadline_fn=None,
    )

    assert result["notices"] == 1, "grounded value difference must land the C notice"
    assert _EnvCapture.calls, "channel C must have dispatched Qwen once"
    left, right = _EnvCapture.calls[0]
    assert "context" not in left, "claim side carries no sentence context (D3)"
    assert right["context"] == "服务配置"
    assert right["quote"] == sentence


# ── pair-v10 渲染：带 context 出段、无 context 零变化 ──────────────────────────

def test_pair_text_renders_context_block() -> None:
    base_left = {"subject": "s", "quote": "上限调整为 200。"}
    base_right = {"subject": "s", "quote": "连接池上限为 500。"}
    plain = LocalGGUFSemanticBackend._pair_text(dict(base_left), dict(base_right))
    assert "上下文" not in plain, "无 context 时渲染必须与 pair-v9 逐位兼容"

    with_ctx = LocalGGUFSemanticBackend._pair_text(
        {**base_left, "context": "本机配置"},
        {**base_right, "context": "服务配置"},
    )
    assert "上下文（仅供判断属性归属" in with_ctx
    assert "A上下文=本机配置" in with_ctx and "B上下文=服务配置" in with_ctx
    assert with_ctx.index("上下文") < with_ctx.index("A证据原文="), "语境段在证据原文之前"

    en = LocalGGUFSemanticBackend._pair_text(
        {"subject": "s", "quote": "The limit is 200.", "context": "service config"},
        {"subject": "s", "quote": "The limit is 500."},
    )
    assert "Context (attribute ownership only" in en

    assert PAIR_PROMPT_VERSION == "pair-v10"
    assert _PAIR_PROMPT.count("例：") == 1, "示例标记唯一，_strip_pair_example 契约不变"
    # pair-v10：系统提示词与 v9 逐位一致（0.6B 对系统措辞敏感——slow 校准对
    # 实证，加一行指令即扰动长值对抽取）；契约只存在于渲染出的 context 段。
    # sha 钉（对抗 review P3）：子串缺席防不住其他措辞漂移——改系统词必须显式换版本。
    import hashlib

    assert hashlib.sha256(_PAIR_PROMPT.encode()).hexdigest() == _PAIR_PROMPT_SHA
    assert hashlib.sha256(_PAIR_PROMPT_EN.encode()).hexdigest() == _PAIR_PROMPT_EN_SHA
    assert "上下文" not in _PAIR_PROMPT and "context" not in _PAIR_PROMPT_EN.lower()
    assert "属性与值必须取自下方证据原文" in with_ctx
    assert "must come from the evidence below" in en
    # 对抗 review P2：truncation retry 渲染同步缩 context（120/侧）——
    # 否则双长行+满 context 的 retry 形态被 n_ctx 守卫确定性关死。
    retry_text = LocalGGUFSemanticBackend._pair_text(
        {**base_left, "context": "甲" * 300},
        {**base_right, "context": "乙" * 300},
        quote_cap=240, context_cap=120,
    )
    assert "甲" * 120 in retry_text and "甲" * 121 not in retry_text
    assert "乙" * 120 in retry_text and "乙" * 121 not in retry_text


# ── D4 契约：值取自上下文被 grounding 挡住 ─────────────────────────────────────

def test_value_from_context_fails_grounding() -> None:
    # Qwen 从上下文捞了值（300 只在 context、不在 quote）→ 必须 unverified
    left = {"quote": "上限调整为 200。", "context": "历史配置为 300。"}
    right = {"quote": "连接池上限为 500。", "context": "服务配置"}
    extraction = AttributeValueExtraction(
        attribute_a="连接池上限", value_a="200",
        attribute_b="连接池上限", value_b="300",
    )
    gate = evaluate_single_direction_extraction(extraction, left, right)
    assert gate.state == "review_candidate" and gate.reason == "qwen_unverified"

    # 正向：值都取自主行、属性靠上下文恢复 → notice_ready
    ok = AttributeValueExtraction(
        attribute_a="连接池上限", value_a="200",
        attribute_b="连接池上限", value_b="500",
    )
    good = evaluate_single_direction_extraction(ok, left, right)
    assert good.state == "notice_ready"
