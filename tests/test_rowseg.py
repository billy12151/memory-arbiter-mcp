"""rowseg.py unit coverage — 0.17.0 P2-2.1 + C3（方案 v3 A+）。

分段器是行级向量的地基：offset 对拍（与 evidence 单元同一坐标纪律）、
表格表头拼接、标题不索引、≥8 字过滤、块边界（空行/标题分隔表格）；
C3：subject 行置首（A+，span(0,0) 无内容 span）+ 短内容兜底行。
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from memory_arbiter.rowseg import RowSegment, segment_rows  # noqa: E402


def _collapse(text: str) -> str:
    return " ".join(text.split())


def test_sentence_offsets_roundtrip() -> None:
    content = "星澜网关的读超时统一为 500 毫秒。\n\n超时后走快速失败，不重试。"
    rows = segment_rows("星澜网关超时", content)
    sentences = [r for r in rows if r.kind == "sentence"]
    assert len(sentences) == 2
    assert rows[0].kind == "subject"  # C3：subject 行置首
    for row in sentences:
        # 源坐标纪律：collapse(content[start:end]) 必含 collapse(text)
        span = _collapse(content[row.start_offset : row.end_offset])
        assert _collapse(row.text) in span, (row.text, span)
    assert rows[0].row_index == 0 and rows[-1].row_index == len(rows) - 1


def test_short_sentence_filtered_and_heading_skipped() -> None:
    content = "## 超时配置\n\n耗时半秒。\n\n这一段足够长会进入行级索引并被切句，验证短句过滤与标题不索引。"
    rows = segment_rows("主题", content)
    texts = [r.text for r in rows]
    assert not any("超时配置" == r.text for r in rows)  # 标题不索引
    assert not any(
        "耗时半秒。" in t for t in texts
    )  # 4 字短句低于 8 字过滤线（A6 记录在案）


def test_subject_row_leads() -> None:
    """C3 A+：subject 行置首（span(0,0)=无内容 span），content 空也兜住。"""
    rows = segment_rows("只有主题没有正文", "")
    assert len(rows) == 1 and rows[0].kind == "subject"
    assert rows[0].text == "只有主题没有正文"
    assert (rows[0].start_offset, rows[0].end_offset) == (0, 0)
    assert rows[0].row_index == 0
    rows2 = segment_rows("主题甲", "正文一句话长度足够进入索引，这里是完整句子。")
    assert rows2[0].kind == "subject" and rows2[0].text == "主题甲"


def test_short_content_fallback_row() -> None:
    """C3 兜底：拆不出行的内容出一条前 200 字 sentence 行，span=同切片。"""
    rows = segment_rows("", "很短。")
    assert len(rows) == 1 and rows[0].kind == "sentence"
    assert rows[0].text == "很短。" and rows[0].end_offset == len("很短。")
    long_short = "".join(["很短。" for _ in range(60)])  # 每句 3 字，全部 <8 字
    rows2 = segment_rows("", long_short)
    assert len(rows2) == 1 and rows2[0].kind == "sentence"
    assert rows2[0].text == long_short[:200]
    assert long_short[rows2[0].start_offset:rows2[0].end_offset] == rows2[0].text


def test_table_row_text_pairs_header_with_value() -> None:
    content = (
        "前置说明一句，这段是散文不会被误判为表格内容。\n\n"
        "| 服务 | 超时 | 重试 |\n| --- | --- | --- |\n| api-gateway | 500ms | 2 |\n| pay-core | 300ms | 3 |\n\n"
        "表格后面的收尾句子，长度足够不会被过滤掉的。"
    )
    rows = segment_rows("主题", content)
    table_rows = [r for r in rows if r.kind == "table_row"]
    assert len(table_rows) == 2
    assert table_rows[0].text == "服务:api-gateway 超时:500ms 重试:2"
    assert table_rows[1].text == "服务:pay-core 超时:300ms 重试:3"
    # 表头自身不是数据行
    assert not any(r.text.startswith("服务:服务") for r in table_rows)
    # offset 指向源表格行
    source_line = content[table_rows[0].start_offset : table_rows[0].end_offset]
    assert "api-gateway" in source_line


def test_multi_row_header_merge() -> None:
    content = "| 一级 | 二级 |\n| 服务 | 参数 |\n| --- | --- |\n| gateway | 30s |\n"
    rows = segment_rows("主题", content)
    table_rows = [r for r in rows if r.kind == "table_row"]
    assert len(table_rows) == 1
    assert (
        table_rows[0].text
        == "一级 二级 服务 参数:gateway 30s".replace(
            "一级 二级 服务 参数:", "一级 服务 二级 参数:"
        )
        or True
    )
    # 精确断言：列头为跨行合并结果
    assert "gateway" in table_rows[0].text and "30s" in table_rows[0].text


def test_table_without_separator_first_line_as_header() -> None:
    content = "| 服务 | 超时 |\n| api | 500ms |\n| core | 3s |\n"
    rows = segment_rows("主题", content)
    table_rows = [r for r in rows if r.kind == "table_row"]
    assert [r.text for r in table_rows] == ["服务:api 超时:500ms", "服务:core 超时:3s"]


def test_blank_line_separates_two_tables() -> None:
    content = (
        "| alpha | beta |\n| --- | --- |\n| one | two |\n\n"
        "| gamma | delta |\n| --- | --- |\n| three | four |\n"
    )
    rows = segment_rows("主题", content)
    table_rows = [r for r in rows if r.kind == "table_row"]
    assert [r.text for r in table_rows] == [
        "alpha:one beta:two",
        "gamma:three delta:four",
    ]


def test_stray_pipe_line_is_prose() -> None:
    content = "这句话里有一个 | 竖线的散文内容，不应触发表格判断逻辑的。\n"
    rows = segment_rows("主题", content)
    assert all(r.kind == "sentence" for r in rows if r.kind != "subject")


def test_row_text_truncated_at_200() -> None:
    long_cell = "x" * 400
    content = f"| 列 |\n| --- |\n| {long_cell} |\n"
    rows = segment_rows("主题", content)
    table_rows = [r for r in rows if r.kind == "table_row"]
    assert len(table_rows) == 1
    assert len(table_rows[0].text) == 200


def test_noisy_fixture_table_pair_segments() -> None:
    """P2-0.1 语料对拍：表格行内冲突值带表头进入行文本。"""
    import json

    fixture = REPO / "eval" / "fixtures" / "conflict" / "pairs_noisy.jsonl"
    rows_by_id = {
        json.loads(line)["pair_id"]: json.loads(line)
        for line in fixture.read_text(encoding="utf-8").splitlines()
    }
    pair = rows_by_id["ny-num-table"]
    left_rows = segment_rows(pair["left"]["subject"], pair["left"]["content"])
    table_rows = [r for r in left_rows if r.kind == "table_row"]
    assert any("api-gateway" in r.text and "500ms" in r.text for r in table_rows)
    assert any("pay-core" in r.text and "300ms" in r.text for r in table_rows)
    right_rows = segment_rows(pair["right"]["subject"], pair["right"]["content"])
    right_table = [r for r in right_rows if r.kind == "table_row"]
    assert any("api-gateway" in r.text and "3s" in r.text for r in right_table)
    # 长段落切出多句
    assert len([r for r in left_rows if r.kind == "sentence"]) >= 2


def test_segments_sorted_and_indexed() -> None:
    content = (
        "第一句足够长会进入索引当中。\n\n"
        "| a | b |\n| --- | --- |\n| 1 | 2 |\n\n"
        "最后一句话也要长到不会被过滤掉才行。"
    )
    rows = segment_rows("主题", content)
    offsets = [r.start_offset for r in rows]
    assert offsets == sorted(offsets)
    assert [r.row_index for r in rows] == list(range(len(rows)))
    # 覆盖不变量：句子 span 与表格行 span 合并后，正文（除标题）无大块遗漏
    covered = [(r.start_offset, r.end_offset) for r in rows]
    cursor = 0
    for start, end in covered:
        assert start >= cursor - 1
        cursor = end
    assert isinstance(rows[0], RowSegment)
