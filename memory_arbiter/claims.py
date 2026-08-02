"""Deterministic structured-claim extraction for v0.9.

The extractor deliberately stays lexical: it finds explicit key/value, table,
number+unit, and semantic-version claims without calling a model.  Semantic
equivalence and winner selection remain the host LLM's responsibility.
"""
from __future__ import annotations

import re
from typing import Any, Mapping, MutableMapping, Optional


_CJK_BOOKKEEPING = {
    "执行时间", "任务来源", "改动文件", "发版任务", "更新时间",
    "更新", "修正", "已完成", "补查", "执行结果", "完成情况", "变更记录",
    "提交记录", "关联文件", "创建时间", "修改时间", "检查结果", "处理结果",
}
_GENERIC_SUBJECTS = {
    "release", "releases", "notes", "release notes", "配置", "说明", "文档",
    "总结", "计划", "记录", "更新", "changelog", "readme", "todo",
}
_REFERENCE_ATTRIBUTES = {"id", "ref", "pr", "issue", "commit"}
_UNIT_MAP = {
    "毫秒": "ms", "ms": "ms",
    "秒": "s", "sec": "s", "secs": "s", "second": "s", "seconds": "s", "s": "s",
    "qps": "qps", "mb": "mb", "gb": "gb", "kb": "kb", "%": "%",
}

_FENCED_CODE_RE = re.compile(r"```[^\n]*\n?.*?```", re.DOTALL)
_QUOTED_OR_PLAIN_KV_RE = re.compile(
    r'(?<![\w.])(?:"(?P<qkey>[A-Za-z][A-Za-z0-9_.-]{0,63})"|'
    r"'(?P<skey>[A-Za-z][A-Za-z0-9_.-]{0,63})'|"
    r'(?P<key>[A-Za-z][A-Za-z0-9_.-]{0,63}))\s*[:=]\s*'
    r'(?:"(?P<qval>[^"\n]{1,200})"|'
    r"'(?P<sval>[^'\n]{1,200})'|"
    r'(?P<val>[^\s,;，；|)\]}]{1,200}))'
)
_CJK_KV_RE = re.compile(
    r"(?<![\u4e00-\u9fff])(?P<key>[\u4e00-\u9fff]{1,6})\s*[:：=]\s*"
    r"(?P<val>[A-Za-z0-9_./@+%-]+)(?![\u4e00-\u9fff])"
)
_NUM_UNIT_RE = re.compile(
    r"(?<![\w\u4e00-\u9fff])(?P<key>[A-Za-z_][A-Za-z0-9_.-]{0,9}|[\u4e00-\u9fff]{1,10})"
    r"(?:\s*[:：])?\s+(?P<num>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>毫秒|秒|ms|secs?|seconds?|s|qps|mb|gb|kb|%)?"
    r"(?![\w\u4e00-\u9fff])",
    re.IGNORECASE,
)
_VERSION_RE = re.compile(r"(?<![\w.])v?(?P<value>\d+\.\d+\.\d+)(?![\w.])", re.IGNORECASE)
_SCOPE_RE = re.compile(r"(?im)^\s*scope\s*[:=]\s*([^\s|,;]+)")
_COMMIT_RE = re.compile(r"^[0-9a-f]{7,40}$", re.IGNORECASE)
_FILE_LINE_RE = re.compile(r"^[^\s]+:\d+$")
_NUMBER_RE = re.compile(r"^\d+(?:\.\d+)?(?:\s*[A-Za-z%]+)?$")
_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")


def canon_token(value: Any) -> str:
    """Canonicalise entity/attribute/scope without semantic aliasing."""
    if value is None:
        return ""
    text = re.sub(r"\s+", " ", str(value).strip()).lower()
    return text.strip("。，,;；:：.、· \t-—")


def canon_scope(value: Any) -> str:
    return canon_token(value)


def canon_value(value: Any, *, value_type: Optional[str] = None) -> str:
    text = str(value or "").strip().strip("\"'").strip().lower()
    if value_type == "version" and text.startswith("v"):
        text = text[1:]
    match = re.fullmatch(
        r"(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>毫秒|秒|ms|secs?|seconds?|s|qps|mb|gb|kb|%)",
        text,
        re.IGNORECASE,
    )
    if match:
        unit = _UNIT_MAP.get(match.group("unit").lower(), match.group("unit").lower())
        return f"{match.group('num')} {unit}"
    return text


def infer_value_type(value: str) -> str:
    if _SEMVER_RE.fullmatch(value):
        return "version"
    if value in {"true", "false", "enabled", "disabled", "yes", "no"}:
        return "bool"
    if _NUMBER_RE.fullmatch(value):
        return "number"
    return "string"


def _masked_content(content: str) -> str:
    """Blank fenced code while preserving offsets."""
    chars = list(content)
    for match in _FENCED_CODE_RE.finditer(content):
        chars[match.start():match.end()] = [" " if ch != "\n" else "\n" for ch in match.group(0)]
    return "".join(chars)


def _record_value(record: Any, name: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def resolve_entity(record: Any) -> tuple[str, str]:
    metadata = _record_value(record, "metadata", {}) or {}
    if not isinstance(metadata, Mapping):
        metadata = {}
    explicit = canon_token(metadata.get("entity"))
    if explicit:
        return explicit, "metadata"
    subject = canon_token(_record_value(record, "subject"))
    if subject and subject not in _GENERIC_SUBJECTS:
        return subject, "subject"
    tags = _record_value(record, "tags", []) or []
    if isinstance(tags, str):
        tags = [tags]
    if tags:
        candidate = canon_token(tags[0])
        if candidate and candidate not in _GENERIC_SUBJECTS:
            return candidate, "tag"
    return "default", "default"


def resolve_scope(record: Any, content: str) -> str:
    metadata = _record_value(record, "metadata", {}) or {}
    if isinstance(metadata, Mapping):
        explicit = canon_scope(metadata.get("scope"))
        if explicit:
            return explicit
    match = _SCOPE_RE.search(content)
    return canon_scope(match.group(1)) if match else ""


def _is_reference(attribute: str, value: str) -> bool:
    if value.startswith(("http://", "https://")):
        return True
    # A bare URL is tokenised by P1 as ``https: //host/path``.  Treat that
    # representation as a reference too, otherwise ``https`` becomes a bogus
    # configuration attribute whenever prose contains a link.
    if attribute in {"http", "https"} and value.startswith("//"):
        return True
    if _COMMIT_RE.fullmatch(value) or _FILE_LINE_RE.fullmatch(value):
        return True
    if attribute in _REFERENCE_ATTRIBUTES and re.fullmatch(r"#?\d+", value):
        return True
    return False


def _evidence(content: str, start: int, end: int, radius: int = 60) -> str:
    return content[max(0, start - radius):min(len(content), end + radius)].strip()


def extract_claims(
    record: Any,
    diagnostics: Optional[MutableMapping[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Extract canonical claim dicts from a MemoryRecord or mapping.

    ``diagnostics`` is optional and receives entity_source, ambiguous_keys,
    rejected_reference_count, and extractor_counts.  Ambiguous same-key groups
    are omitted as a whole.
    """
    diag: MutableMapping[str, Any] = diagnostics if diagnostics is not None else {}
    content = str(_record_value(record, "content", "") or "")
    entity, entity_source = resolve_entity(record)
    diag["entity_source"] = entity_source
    if not entity:
        diag.update({"skipped_reason": "missing_entity", "ambiguous_keys": [], "extractor_counts": {}})
        return []
    scope = resolve_scope(record, content)
    masked = _masked_content(content)
    candidates: list[dict[str, Any]] = []
    occupied: list[tuple[int, int]] = []
    rejected_refs = 0

    def overlaps(start: int, end: int) -> bool:
        return any(start < old_end and end > old_start for old_start, old_end in occupied)

    def add(attribute: Any, raw_value: Any, start: int, end: int, rule: str,
            explicit_type: Optional[str] = None) -> None:
        nonlocal rejected_refs
        attr = canon_token(attribute)
        if not attr or attr in _CJK_BOOKKEEPING:
            return
        value = canon_value(raw_value, value_type=explicit_type)
        if not value or _is_reference(attr, value):
            if value:
                rejected_refs += 1
            return
        value_type = explicit_type or infer_value_type(value)
        candidates.append({
            "entity": entity,
            "attribute": attr,
            "scope": scope,
            "value": value,
            "raw_value": str(raw_value),
            "value_type": value_type,
            "extractor_rule": rule,
            "evidence": _evidence(content, start, end),
            "start_offset": start,
            "end_offset": end,
        })
        occupied.append((start, end))

    # P1/P1-CJK: same priority, both before table/number/version.
    for match in _QUOTED_OR_PLAIN_KV_RE.finditer(masked):
        key = match.group("qkey") or match.group("skey") or match.group("key")
        value = match.group("qval") or match.group("sval") or match.group("val")
        add(key, value, match.start(), match.end(), "p1_kv")
    for match in _CJK_KV_RE.finditer(masked):
        if not overlaps(match.start(), match.end()):
            add(match.group("key"), match.group("val"), match.start(), match.end(), "p1_cjk_kv")

    # P2: first two table cells only; skip separator/header rows.
    lines = masked.splitlines(keepends=True)
    line_start = 0
    for line_index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("|") and stripped.count("|") >= 3:
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            if len(cells) >= 2:
                key, value = cells[0], cells[1]
                next_is_separator = False
                if line_index + 1 < len(lines):
                    next_stripped = lines[line_index + 1].strip()
                    if next_stripped.startswith("|"):
                        next_cells = [
                            cell.strip() for cell in next_stripped.strip("|").split("|")
                        ]
                        next_is_separator = (
                            len(next_cells) >= 2
                            and all(re.fullmatch(r":?-{2,}:?", cell) for cell in next_cells[:2])
                        )
                if (key and value and
                        not next_is_separator and
                        not re.fullmatch(r":?-{2,}:?", key) and
                        not re.fullmatch(r":?-{2,}:?", value) and
                        canon_token(key) not in {"key", "attribute", "字段", "属性"}):
                    start = line_start + line.find(key)
                    end = line_start + line.rfind(value) + len(value)
                    if not overlaps(start, end):
                        add(key, value, start, end, "p2_table")
        line_start += len(line)

    for match in _NUM_UNIT_RE.finditer(masked):
        if overlaps(match.start(), match.end()):
            continue
        raw = match.group("num") + (match.group("unit") or "")
        add(match.group("key"), raw, match.start(), match.end(), "p3_num_unit", "number")

    for match in _VERSION_RE.finditer(masked):
        if not overlaps(match.start(), match.end()):
            add("version", match.group("value"), match.start(), match.end(), "p4_version", "version")

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for claim in candidates:
        key = (claim["entity"], claim["attribute"], claim["scope"])
        grouped.setdefault(key, []).append(claim)

    claims: list[dict[str, Any]] = []
    ambiguous: list[str] = []
    for key, rows in grouped.items():
        values = {row["value"] for row in rows}
        if len(values) > 1:
            ambiguous.append(".".join(part for part in key if part))
            continue
        # Deterministic first occurrence; higher-priority extractors ran first.
        claims.append(min(rows, key=lambda row: int(row["start_offset"])))
    claims.sort(key=lambda row: (int(row["start_offset"]), row["attribute"]))
    extractor_counts: dict[str, int] = {}
    for claim in claims:
        rule = str(claim["extractor_rule"])
        extractor_counts[rule] = extractor_counts.get(rule, 0) + 1
    diag.update({
        "entity": entity,
        "scope": scope,
        "ambiguous_keys": ambiguous,
        "ambiguous_key_count": len(ambiguous),
        "rejected_reference_count": rejected_refs,
        "extractor_counts": extractor_counts,
    })
    return claims
