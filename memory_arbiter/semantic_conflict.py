"""Local extraction protocol and isolated GGUF process supervision.

The model extracts comparable attribute/value fields from candidate evidence
pairs. Deterministic gates may accept, reject, or request extraction; advisory
semantic notices still require an agent to read both memories before acting.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import time
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol

from .difference_classifier import _cn_to_int
from .constants import (
    EMBED_PREFIX_STS,
    CLAIM_ATTR_TAU,
    SEMANTIC_PAIR_MAX_ATTEMPTS,
    SEMANTIC_PAIR_RETRY_MAX_TOKENS,
    SEMANTIC_PAIR_RETRY_CONTEXT_CHARS,
    SEMANTIC_PAIR_RETRY_QUOTE_CHARS,
)

ACTION_TYPES = {
    "value_changed",
    "scope_changed",
    "polarity_changed",
    "source_of_truth_changed",
    "lifecycle_changed",
    "policy_changed",
    "uncertain",
}
NON_ACTION_TYPES = {"equivalent", "compatible", "unrelated"}

_REPLACEMENT_TERMS = [
    "以后以", "替换", "改为", "不再采用", "之前不对", "旧设计", "新设计",
    "新口径", "旧口径", "下线", "不公开", "公开", "不采用", "采用",
    "默认", "必须", "只用", "不能", "不要", "不应", "不是", "而不是",
    "移除", "删除", "主路径", "active-only",
]
_DONE_TERMS = ["已完成", "已经完成", "已修复", "已经修复", "已发布", "已经发布", "已处理", "完成并"]
_NEGATION_TERMS = ["不", "不要", "不能", "不应", "不采用", "不是", "而不是", "移除", "删除", "下线", "不公开", "禁止"]
_LIST_TERMS = ["包括", "包含", "场景", "列表"]
_TOKEN_RE = re.compile(r"[A-Za-z0-9_\-.]+|[一-鿿]{2,}")
_STOPWORDS = {
    "用户", "今天", "中午", "想吃", "项目", "系统", "已经", "应该", "可以",
    "默认", "内容", "记忆", "方案", "设计", "模型", "本地", "腾讯云",
    "一个", "两个", "这个", "那个", "使用", "采用", "固定", "不要", "不能",
    "不应", "不是", "已经完成",
}

def evidence_is_cjk(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """True when either side's evidence/metadata carries CJK — the pair
    prompt follows the evidence language (pair-v8); mixed pairs stay
    Chinese (the Chinese prompt is the calibrated default)."""
    for env in (left, right):
        for key in ("quote", "content", "subject"):
            if _CJK_RE.search(str(env.get(key) or "")):
                return True
    return False
# pair-v6 note: the prompt text is deliberately identical to pair-v5. Two
# few-shot variants teaching long-evidence fragment selection were tried and
# rejected by experiment (2026-09-09 calibration matrix): any added example
# broke side attribution on the Tier1 calibration pair (the 0.5B adopted
# positional heuristics from the example, e.g. copying the B-side opening into
# value_a) — same lesson as the rejected "compress to the core value" wording.
# pair-v7 (2026-09-16): system prompt still untouched (the v6 lesson stands);
# the only change is an optional rule-candidate-values line in the USER turn
# (present only when decide_evidence extracted values on both
# sides.
# Since 0.15.14 (A2) decoding is grammar-free: no response_format anywhere on
# this path (its per-token grammar evaluation halved decode throughput, and
# even on the retry it cost 3-5x the alternative — see _pair_retry_feedback);
# the value/attribute caps are enforced post-hoc (L3 truncation + grounding
# gates). One retry is kept for schema/truncation/empty-field failures, built
# from the SPECIFIC violation the first attempt tripped with the offending
# output deliberately NOT echoed back (echo locks the 0.5B into the failed
# copy state; both measured 2026-09-11). A queue gate (A6) may skip the
# retry while other requests wait.


_WORKSPACE_RESPONSE_FORMAT = {
    "type": "json_object",
    "schema": {
        "type": "object",
        "properties": {
            "candidate": {"type": ["string", "null"]},
            "relation": {
                "type": "string",
                "enum": ["alias", "typo", "same_project", "same_family", "related", "unrelated", "uncertain"],
            },
            "confidence": {"type": "number"},
            "evidence": {"type": "string", "maxLength": 200},
        },
        "required": ["candidate", "relation", "confidence", "evidence"],
        "additionalProperties": False,
    },
}

_WORKSPACE_PROMPT = """你是 mema 的 workspace 归一候选建议器，只输出 JSON，不要解释。
输入是一个新记忆的 workspace 原文 + 短证据(标题/关键句) + 若干候选 workspace。
任务：判断该 workspace 是否应归一到某个候选，只做建议，不做最终裁决。
字段：candidate(建议归一到的候选名，或 null)，relation(alias|typo|same_project|same_family|related|unrelated|uncertain)，confidence(0..1)，evidence(一句话理由)。
规则：同一项目不同写法/错别字/中英名互指 → alias/typo，高 confidence。
同客户不同子域(售后/运维/培训/回访)、仅主题相关 → related/same_family，中低 confidence。
明显无关 → unrelated，candidate=null。
拿不准 → uncertain，candidate=null，低 confidence。"""


@dataclass
class PairEvidence:
    common_tokens: list[str]
    char_cosine: float
    token_cosine: float
    replacement: bool
    todo_done: bool
    contains_diff: bool
    polarity_diff: bool
    list_value_diff: bool
    duplicate_guard: bool
    compatible_guard: bool
    only_left: list[str] = field(default_factory=list)
    only_right: list[str] = field(default_factory=list)


@dataclass
class EvidenceDecision:
    action: str
    reason: str
    anchors: list[str] = field(default_factory=list)
    left_value: str | None = None
    right_value: str | None = None


@dataclass(frozen=True)
class PairGateResult:
    state: str
    reason: str
    attribute: str | None = None
    value_a: str | None = None
    value_b: str | None = None
    grounded: bool = False


class SemanticBackend(Protocol):
    """0.17.1: the judge backend protocol (mDeBERTa). classify_pair/
    suggest_workspace_candidate died with the GGUF engine."""

    def judge_pairs(self, pairs: "list[tuple[str, str]]") -> "list[Any]":
        ...

    def load(self) -> None:
        ...

    def status(self) -> dict[str, Any]:
        ...

    def unload(self, timeout: float = 30.0, disable: bool = False) -> dict[str, Any]:
        ...

    def set_disabled(self, disabled: bool) -> None:
        ...


def _tokens(text: str) -> set[str]:
    out: set[str] = set()
    for token in _TOKEN_RE.findall((text or "").lower()):
        token = token.strip("_.-")
        if len(token) >= 2 and token not in _STOPWORDS:
            out.add(token)
    return out


def _char_ngrams(text: str, n: int = 3) -> set[str]:
    compact = re.sub(r"\s+", "", (text or "").lower())
    return {compact[i:i + n] for i in range(max(0, len(compact) - n + 1))}


def _cosine(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / math.sqrt(len(left) * len(right))


def vector_cosine(left: list[float] | None, right: list[float] | None) -> float:
    """Cosine over two embedding vectors (soft-ordering score, v0.15.12 C4).

    Distinct from the token-set ``_cosine`` above: this one scores a pair of
    subject+tags hint vectors. Mismatched lengths or either side missing
    score 0 — the caller treats that as "no overlap signal", ranked last,
    never an error.
    """
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = math.sqrt(sum(a * a for a in left))
    norm_right = math.sqrt(sum(b * b for b in right))
    if norm_left == 0.0 or norm_right == 0.0:
        return 0.0
    return dot / (norm_left * norm_right)


_EVIDENCE_ALIASES = {
    "pgsql": "postgresql",
    "postgres": "postgresql",
    "sqlite_vec": "sqlite-vec",
}
# B-C1 evaluation (2026-08-28): same bare-substring shape as the old
# coexistence_veto dimension markers, but deliberately left quote-based —
# decide_evidence runs at the recall/triage stage before any extraction
# exists, so there is no attribute to align against, and these are explicit
# scope phrases (environment/region/platform pairs) rather than generic
# dimension words that commonly appear in unrelated prose.
_EXPLICIT_SCOPES = (
    ("测试环境", "生产环境"),
    ("移动端", "管理后台"),
    ("公开api", "内部api"),
    ("中国区", "海外区"),
)
# Pre-Qwen vetoes (2026-09-17, owner-directed; see decide_evidence):
# ops markers (timestamp-suffixed slugs / test-marker wording) and
# decision-vs-observation asymmetry. Calibration: governed-negative corpus —
# 12 positives zero false-veto, 17/43 negatives hit (42% of owner's own
# dismissal intuitions), targets 17918 (ops marker) and 633/625 (decision vs
# evaluation) all captured. The marker pattern deliberately excludes the
# "conflict-test" wording — that baseline pair is a true conflict per owner.
_OPS_MARKER_RE = re.compile(
    r"-\d{10,}|self[- ]?test.{0,40}marker|test[- ]?marker|测试标记",
    re.IGNORECASE,
)
_DECISION_RE = re.compile(
    r"用户.{0,4}(?:确认|修正|拍板|裁定|要求|提出|原则|决定|共识)"
)
_EVALUATION_RE = re.compile(
    r"测试|评测|实测|样本|tuning|跑了|实验|测得|结论是|verify|verified|measured",
    re.IGNORECASE,
)
# Process-record veto (2026-09-17, owner principle extended): review rounds,
# design→release progression and re-verification notes are workflow records,
# not competing claims — the bidirectional gate only stopped them by luck
# (inconsistent extractions), and single-direction let three through. A
# same-object-id rule was measured and DROPPED: "任务 id=123 预算 5000 vs
# 500" is exactly the real conflict this product exists to catch.
# 0.17.0 P2-1.1: bare 「复审」 killed real config pairs ("每半年复审/每季度
# 复审" is a POLICY statement, not a review record — 11th-round live kill);
# it now requires an artefact context. 「复验」 same shape; re-verif gains a
# left boundary so "pre-verify" no longer matches.
_PROCESS_REVIEW_RE = re.compile(
    r"review findings|follow[- ]?up review|adversarial review|review for id"
    r"|评审结论|评审意见|评审记录|审查结论"
    r"|(?:文档|方案|设计|配置|规则)复审|复审(?:结论|记录)",
    re.IGNORECASE,
)
_PROCESS_REVERIFY_RE = re.compile(
    r"重启后.{0,8}再次?验证|(?:文档|方案|设计|配置)复验|复验(?:结论|记录)"
    r"|(?<![a-z])re[- ]?verif|verify again",
    re.IGNORECASE,
)
_PROCESS_DESIGN_RE = re.compile(
    r"设计文档|最终设计|design[_ -]?v\d|方案[_ -]?v\d", re.IGNORECASE,
)
_PROCESS_RELEASE_RE = re.compile(
    r"发版完成|已发布|已上线|released|deployment complete", re.IGNORECASE,
)

# 0.17.0 P2-6.3: lineage-version evolution veto (E1). Both sides carrying a
# LINEAGE marker (design doc / spec / 第N版) with DIFFERENT version numbers
# is a supersedes relationship, not a competing claim — the value delta the
# gates see is the intentional revision. Same-lineage product-version deltas
# (E2) deliberately stay candidates. Known limits recorded in the plan:
# enumerated lineage vocabulary, Chinese product names may miss extraction,
# cross-references like「文档1.0升级到2.0」inside one side can confuse.
_LINEAGE_MARKER_RE = re.compile(
    r"(?:设计文档|最终设计|方案|文档|规格|spec|design)\s*v(\d+(?:\.\d+)*)"
    r"|(?:设计文档|最终设计|方案|文档|规格|spec|design)[^\n]{0,3}?版本\s*(\d+(?:\.\d+)*)"
    r"|第\s*([一二三四五六七八九十\d]+)\s*版",
    re.IGNORECASE,
)


def _lineage_primary_version(text: str) -> "tuple[int, ...] | None":
    """The text's self-declared lineage versions as numeric tuples (highest
    wins at comparison); None when it carries no lineage marker at all — the
    veto needs BOTH sides self-declared. Cross-references (「v2.0 取代 v1.0」)
    resolve by MAX version: the superseding doc owns the pair."""
    matches = _LINEAGE_MARKER_RE.findall(text or "")
    if not matches:
        return None
    versions: list[tuple[int, ...]] = []
    for dotted, versioned, nth in matches:
        text_version = dotted or versioned
        if text_version:
            parsed_version = tuple(int(part) for part in text_version.split("."))
            # 「v2」与「v2.0」是同一版本的拼写变体（semver 同代）——去尾零
            # 归一后再比较，否则同代真冲突在判定链两层（decide_evidence +
            # gates._subject_version_primary）被元组不等静默豁免（R2 review）。
            while len(parsed_version) > 1 and parsed_version[-1] == 0:
                parsed_version = parsed_version[:-1]
            versions.append(parsed_version)
        elif nth:
            parsed: "int | None" = int(nth) if nth.isdigit() else None
            if parsed is None:
                from .difference_classifier import _cn_to_int
                parsed = _cn_to_int(nth)
            if parsed is not None:
                versions.append((parsed,))
    if not versions:
        return None
    return max(versions)
# 0.17.0 D1 (owner 2026-09-23): version-like claims attrs are timeline
# evolution, never opposing claims — the claims-channel counterpart of the
# lineage veto above, applied to the attr axis. Pattern match (regex+vocab
# blend, same style as _LINEAGE_MARKER_RE), NOT an enumerated allow-list:
# the word-list route was retired once already (0.16.4 evolution domain
# replaced it); pure vocab cannot cover variants like 客户端版本/release notes.
# Word boundaries on the English vocabulary (adversarial review P2): plain
# substrings matched build_command / tags / release_channel / docker-build
# steps — real config conflicts got silently exempted. CJK terms need no
# boundary (they carry their own).
_VERSIONAL_ATTR_RE = re.compile(
    r"版本|发版|\b(?:version|commit|revision|release|tag|build)\b",
    re.IGNORECASE,
)


def attr_is_versional(attr_norm: str) -> bool:
    """True when a claims attr carries version semantics (release/commit/tag).

    A value delta on such an attr is expected evolution — two memories
    recording different versions of the same artifact — so the claims channel
    skips it and counts the skip (versional_vetoed) to keep the exemption
    observable instead of a black hole. Deliberately narrow: numeric attrs in
    general are NOT covered (that lesson belongs to the internal channel).
    """
    return bool(_VERSIONAL_ATTR_RE.search(attr_norm or ""))


_VALUE_RE = re.compile(
    r"(?<![\w.])v?\d+(?:\.\d+){0,2}\s*"
    r"(?:ms|s|秒|分钟|小时|个工作日|工作日|个自然日|自然日|日|天|%|mb|gb|kb|条|次|核|g"
    r"|(?:business|working|calendar)?\s*days?|workdays?)?",
    re.IGNORECASE,
)

# Gate-v2 G4 sentence prefilter (方案词表, 与 decide 相邻): a row must carry
# an extractable value, a negation, a time anchor, an assignment shape, or
# BE a table row (gated in pipeline/gates.row_prefilter by kind) to
# originate a KNN query — 64% of real-library rows pass (标定 #9: 值 41% +
# 仅否定 23%), cutting the rest of the KNN spend. The assignment shapes
# (对抗 review P1-5) cover TEXT-value oppositions the numeric _VALUE_RE
# cannot see: "export-format 取值为 csv" / "上传方式=非 dist/*" / "利率
# 上限为 LPR 四倍" are mainline conflict shapes, not the pure-prose gap.
# Known accepted gap (方案 §5): "X 使用 A vs X 使用 B" / "岗位定级 P5"
# forms without any marker stay un-originated on the WRITE path; the scan
# path skips this prefilter by design and the Agent judges them there.
_NEGATION_WORDS = r"(?:不|没|非|未|无|难道|决不|except|never|not|isn't|doesn't|won't|can't)"
_TIME_ANCHOR = r"(?:星期[一二三四五六日天]|周[一二三四五六日天]|昨天|今天|明天|\d{4}-\d{2}-\d{2}|\d{1,2}月\d{1,2}[日号])"
_ASSIGNMENT_SHAPE = r"(?:取值|配置为|设置为|默认为|等于|上限|下限|阈值|限额|配额|=)"
_SENT_PREFILTER = re.compile(
    rf"{_VALUE_RE.pattern}|{_NEGATION_WORDS}|{_TIME_ANCHOR}|{_ASSIGNMENT_SHAPE}",
    re.IGNORECASE,
)


def _numeric_stripped_skeleton(text: str) -> str:
    """Value-stripped normalized skeleton: what remains when every numeric
    value (with its unit suffix) is removed. Duplicate detection compares
    these for full equality — equal numeric sets alone must not read as a
    duplicate when the non-numeric tokens still differ."""
    return _normalize_evidence_text(_VALUE_RE.sub("", text or ""))


_OPERATOR_SIGNATURE_RE = re.compile(r"[<>≥≤≠]=?|(?<!\d)[+\-−](?=\d)")


def _operator_signature(text: str) -> frozenset[str]:
    """Comparator/sign tokens that normalization strips but contradictions
    hinge on (``>= 100ms`` vs ``< 100ms``, ``+5%`` vs ``-5%``). Date hyphens
    sit between two digits and are deliberately not treated as signs."""
    return frozenset(_OPERATOR_SIGNATURE_RE.findall((text or "").casefold()))


def _normalize_evidence_text(text: str) -> str:
    value = (text or "").casefold()
    value = re.sub(r"qwen\s*([0-9]+(?:\.[0-9]+)*)", r"qwen\1", value)
    value = re.sub(r"(\d+(?:\.\d+)?)\s*秒", r"\1s", value)
    for alias, canonical in _EVIDENCE_ALIASES.items():
        value = re.sub(rf"\b{re.escape(alias)}\b", canonical, value)
    value = re.sub(r"[\s_\-]+", "", value)
    return re.sub(r"[^a-z0-9\u4e00-\u9fff.%]+", "", value)


def _normalized_values(text: str) -> list[str]:
    values: list[str] = []
    for match in _VALUE_RE.finditer((text or "").casefold()):
        value = re.sub(r"\s+", "", match.group(0)).replace("秒", "s")
        if value.startswith("v") and len(value) > 1 and value[1].isdigit():
            value = value[1:]
        # Canonical unit conversion (2026-09-16): "500ms" vs "0.5s" is a
        # duplicate, not a conflict — the candidate trigger must compare
        # canonical forms, or equivalent values burn a Qwen pair.
        values.append(canonical_unit_value(value))
    return values


def coexistence_veto(
    left: dict[str, Any],
    right: dict[str, Any],
    forward: "Any | None" = None,
    reverse: "Any | None" = None,
) -> str | None:
    """Return a deterministic coexistence reason code, or None when unknown.
    0.17.1: extraction params retained for the single remaining internal
    caller (decide_evidence passes none — quote-substring legacy path)."""
    left_text = str(left.get("quote") or left.get("content") or "").casefold()
    right_text = str(right.get("quote") or right.get("content") or "").casefold()
    if forward is not None or reverse is not None:
        left_dimension = " ".join(filter(None, (
            getattr(forward, "attribute_a", "") if forward is not None else "",
            getattr(reverse, "attribute_b", "") if reverse is not None else "",
        ))).casefold()
        right_dimension = " ".join(filter(None, (
            getattr(forward, "attribute_b", "") if forward is not None else "",
            getattr(reverse, "attribute_a", "") if reverse is not None else "",
        ))).casefold()
    else:
        left_dimension, right_dimension = left_text, right_text
    dimension_markers = {
        "coexist_environment_mismatch": (("测试环境", "生产环境"), ("staging", "production"), ("dev", "prod")),
        "coexist_region_mismatch": (("中国区", "海外区"), ("us-east", "eu-west")),
        "coexist_version_mismatch": (("v1", "v2"), ("旧版本", "新版本")),
        "coexist_object_mismatch": (("移动端", "管理后台"), ("客户端", "服务端")),
        "coexist_observation_time_mismatch": (("昨天", "今天"), ("上周", "本周"), ("当时", "目前")),
        "coexist_historical_current": (("历史记录", "当前"), ("曾经", "现在")),
        "coexist_metric_mismatch": (("平均", "峰值"), ("总计", "单项"), ("总数", "其中")),
    }
    for code, pairs in dimension_markers.items():
        if any(
            (a in left_dimension and b in right_dimension)
            or (b in left_dimension and a in right_dimension)
            for a, b in pairs
        ):
            return code
    if forward is not None:
        va, vb = normalize_value(getattr(forward, "value_a", "")), normalize_value(getattr(forward, "value_b", ""))
        version_shape = re.compile(r"^(?:v\d+(?:\.\d+)*|\d+\.\d+\.\d+(?:\.\d+)*)$", re.IGNORECASE)
        if version_shape.match(va) and version_shape.match(vb):
            return "coexist_version_value_evolution"
    evolution = (
        "替换为", "替换成", "升级为", "升级到", "迁移到", "迁移至", "不再采用",
        "改为", "改成", "切换到", "切换为", "换成", "变为", "变更为", "调整为",
        "更新为", "更改为",
    )
    if any(term in left_text or term in right_text for term in evolution) and any(
        marker in left_text or marker in right_text
        for marker in ("旧", "新", "此前", "现在", "当前", "之前", "原来", "原先")
    ):
        return "coexist_explicit_evolution"
    return None


def decide_evidence(left_text: str, right_text: str) -> EvidenceDecision:
    """Classify a short pair using only narrow, explainable evidence."""
    left_norm = _normalize_evidence_text(left_text)
    right_norm = _normalize_evidence_text(right_text)
    if left_norm and left_norm == right_norm:
        # Normalization strips comparators and signs: ">= 100ms" vs "< 100ms"
        # and "+5%" vs "-5%" normalize equal but contradict — never let them
        # die here as duplicates.
        if _operator_signature(left_text) == _operator_signature(right_text):
            return EvidenceDecision("ignore", "equivalent_value")

    left_lower = (left_text or "").casefold()
    right_lower = (right_text or "").casefold()
    for left_scope, right_scope in _EXPLICIT_SCOPES:
        if (
            (left_scope in left_lower and right_scope in right_lower)
            or (right_scope in left_lower and left_scope in right_lower)
        ):
            return EvidenceDecision("ignore", "explicit_scope_mismatch")

    # 2026-09-17 (owner): two pre-Qwen vetoes over the content texts — the
    # real-library governed negatives showed both shapes slipping through to
    # the judge and being dismissed by hand every time.
    # Ops markers: timestamp-suffixed slugs (jingleai-bridge-self-test-
    # 1783837684) and test-marker wording are operational data, never
    # business claims. Deliberately NOT matching "conflict-test" — the
    # conflict-test baseline pair (export-format json vs csv) is a true
    # same-attribute difference per owner ruling.
    if _OPS_MARKER_RE.search(left_text or "") or _OPS_MARKER_RE.search(right_text or ""):
        return EvidenceDecision("ignore", "ops_marker")
    # Decision vs observation: an owner ruling on one side ("用户确认/修正/
    # 拍板/要求…") against an evaluation on the other ("测试/实测/样本/
    # tuning…") is evidence-versus-verdict, not two claims fighting —
    # "用户确认 5000 元" vs "实测结论上限 500 元" must not read as a
    # conflict. Known boundary (owner-accepted): implementation drift
    # ("用户要求 3 秒" vs "实测配置 5 秒") is filtered too; catching drift
    # would need its own channel.
    if (
        (_DECISION_RE.search(left_text or "") and _EVALUATION_RE.search(right_text or ""))
        or (_DECISION_RE.search(right_text or "") and _EVALUATION_RE.search(left_text or ""))
    ):
        return EvidenceDecision("ignore", "decision_vs_observation")
    # Process records: one side being workflow prose (review findings /
    # follow-up review / re-verification after restart) or a design↔release
    # pair marks the pair as iterations of one effort. Calibration: the
    # three single-direction false positives (review rounds, design→release,
    # re-verify) all captured, 12 positives zero false-veto.
    if _PROCESS_REVIEW_RE.search(left_text or "") or _PROCESS_REVIEW_RE.search(right_text or ""):
        return EvidenceDecision("ignore", "process_record")
    if _PROCESS_REVERIFY_RE.search(left_text or "") or _PROCESS_REVERIFY_RE.search(right_text or ""):
        return EvidenceDecision("ignore", "process_record")
    if (
        (_PROCESS_DESIGN_RE.search(left_text or "") and _PROCESS_RELEASE_RE.search(right_text or ""))
        or (_PROCESS_DESIGN_RE.search(right_text or "") and _PROCESS_RELEASE_RE.search(left_text or ""))
    ):
        return EvidenceDecision("ignore", "process_record")

    # 0.17.0 P2-6.3 (E1): both sides self-declare a lineage version and the
    # PRIMARY (max) versions differ → supersedes evolution, never a competing
    # claim. Equal primaries fall through (same-generation text); E2
    # product-version deltas deliberately stay candidates.
    _left_lineage = _lineage_primary_version(left_text or "")
    _right_lineage = _lineage_primary_version(right_text or "")
    if _left_lineage is not None and _right_lineage is not None and _left_lineage != _right_lineage:
        return EvidenceDecision("ignore", "lineage_version_evolution")

    left_values = _normalized_values(left_text)
    right_values = _normalized_values(right_text)
    left_skeleton = _VALUE_RE.sub("", left_lower)
    right_skeleton = _VALUE_RE.sub("", right_lower)
    common = sorted(_tokens(left_skeleton) & _tokens(right_skeleton))
    char_similarity = _cosine(_char_ngrams(left_skeleton), _char_ngrams(right_skeleton))
    if left_values and right_values and left_values != right_values and (
        common or char_similarity >= 0.45
    ):
        # Numeric deltas are recall evidence only. Ports, PIDs, status codes,
        # durations, and similar values need an extracted attribute before they
        # can become a user-visible notice.
        return EvidenceDecision(
            "check", "numeric_value_candidate", common,
            ", ".join(left_values), ", ".join(right_values),
        )

    evidence = pair_text_evidence(left_text, right_text)
    if evidence.todo_done:
        return EvidenceDecision("notify", "todo_resolved", evidence.common_tokens)
    if evidence.contains_diff:
        return EvidenceDecision("notify", "polarity_changed", evidence.common_tokens)
    if evidence.polarity_diff and evidence.char_cosine >= 0.45:
        return EvidenceDecision("notify", "polarity_changed", evidence.common_tokens)
    if evidence.duplicate_guard:
        return EvidenceDecision("ignore", "equivalent_value")
    if evidence.compatible_guard:
        return EvidenceDecision("ignore", "compatible_evidence")
    if evidence.char_cosine >= 0.20 or evidence.token_cosine > 0 or evidence.common_tokens:
        return EvidenceDecision("check", "semantic_similarity_only", evidence.common_tokens)
    return EvidenceDecision("ignore", "insufficient_local_evidence")


# Direct deterministic conflict path (2026-09-16, owner-directed): when the
# rule layer has already extracted exactly one canonical value per side and
# the value-stripped keys are near-identical, the pair IS the
# same-attribute-different-value shape — no Qwen inference needed. The
# notice stays advisory (the agent triages it), so the threshold errs
# toward missing (falls through to Qwen), never toward noise.
DIRECT_VALUE_KEY_COSINE = 0.96
_VERSION_TOKEN_RE = re.compile(r"v\d", re.IGNORECASE)


def direct_value_verdict(
    unit_text: str, hit_text: str, decision: EvidenceDecision, *,
    embedder: Any = None, key_cosine: float | None = None,
) -> "tuple[str, str, str] | None":
    """Return (attribute, value_a, value_b) for a deterministic conflict, or
    None to leave the pair on the Qwen path.

    Guards, in order: the numeric channel must hold exactly one value per
    side (multi-value sentences are pairwise-ambiguous); the coexistence
    veto (dimension/evolution markers) still applies; version-bearing
    quotes (v1/v2) skip — a version ordinal reads as a bare value and would
    turn evolution into a false direct conflict; the value-stripped key
    cosine must clear DIRECT_VALUE_KEY_COSINE (identical skeletons shortcut
    at 1.0 without an embed call).
    """
    if decision.reason != "numeric_value_candidate":
        return None
    if not decision.left_value or not decision.right_value:
        return None
    if ", " in decision.left_value or ", " in decision.right_value:
        return None
    if _VERSION_TOKEN_RE.search(unit_text) or _VERSION_TOKEN_RE.search(hit_text):
        return None
    if coexistence_veto({"quote": unit_text}, {"quote": hit_text}) is not None:
        return None
    key_a = _numeric_stripped_skeleton(unit_text)
    key_b = _numeric_stripped_skeleton(hit_text)
    if not key_a or not key_b:
        return None
    if key_a == key_b:
        cosine = 1.0
    elif key_cosine is not None:
        cosine = key_cosine
    elif embedder is not None:
        vec_a = embedder.embed_text(prefix=EMBED_PREFIX_STS, body=key_a)
        vec_b = embedder.embed_text(prefix=EMBED_PREFIX_STS, body=key_b)
        cosine = vector_cosine(list(vec_a.embedding), list(vec_b.embedding))
    else:
        return None
    if cosine < DIRECT_VALUE_KEY_COSINE:
        return None
    attribute = normalize_attribute(key_a).rstrip("为的是")[:_MAX_ATTRIBUTE_CHARS]
    if not attribute:
        return None
    return attribute, str(decision.left_value), str(decision.right_value)


def is_cross_evolution(decision: EvidenceDecision) -> bool:

    """0.16.4 §1/§0.5: cross-memory evolution-domain pairs.

    ``notify`` shapes (todo state transitions, polarity snapshots) across
    memories are a timeline phenomenon, not a semantic conflict ("v1 chose
    A, later B looked better" is normal). Both producers — the scan pipeline
    and the write-time KNN loop — exclude them through THIS single predicate
    so the exclusion logic can never fork into two copies. Same-memory
    internal pairs are NOT affected: the in-memory contradiction duty stays
    (internal notify goes through the Qwen final review instead, §2).
    """
    return decision.action == "notify"


def pair_text_evidence(left_text: str, right_text: str) -> PairEvidence:
    left_tokens = _tokens(left_text)
    right_tokens = _tokens(right_text)
    common = left_tokens & right_tokens
    only_left = left_tokens - right_tokens
    only_right = right_tokens - left_tokens
    joined = f"{left_text}\n{right_text}"
    replacement = any(term in joined for term in _REPLACEMENT_TERMS)
    left_lower = (left_text or "").lower()
    right_lower = (right_text or "").lower()
    left_is_todo = "待办" in left_lower or "todo" in left_lower
    right_is_todo = "待办" in right_lower or "todo" in right_lower
    left_done = any(term in left_text for term in _DONE_TERMS)
    right_done = any(term in right_text for term in _DONE_TERMS)
    # Direction-agnostic: a todo on one side marked done on the other. The pair
    # is unordered at the gate (the caller may pass new/old either way), so we
    # accept both orientations. The common-token guard prevents unrelated
    # todo/done statements from pairing.
    todo_done = bool(common) and (
        (left_is_todo and right_done) or (right_is_todo and left_done)
    )
    contains_diff = (
        ("包含" in left_text and ("不包含" in right_text or "移除" in right_text))
        or ("包含" in right_text and ("不包含" in left_text or "移除" in left_text))
    )
    left_neg = any(term in left_text for term in _NEGATION_TERMS)
    right_neg = any(term in right_text for term in _NEGATION_TERMS)
    polarity_diff = left_neg != right_neg and bool(common)
    list_context = any(term in joined for term in _LIST_TERMS)
    char_cosine = _cosine(_char_ngrams(left_text), _char_ngrams(right_text))
    token_cosine = _cosine(left_tokens, right_tokens)
    list_value_diff = (
        bool(common)
        and bool(only_left)
        and bool(only_right)
        and (replacement or polarity_diff or list_context or char_cosine >= 0.45)
    )
    lower = joined.lower()
    compatible_guard = False
    duplicate_guard = False

    def numeric_values(text: str) -> set[str]:
        values = set(re.findall(r"\d+(?:\.\d+)?\s*(?:mb|gb|kb|%|ms|s|秒|分钟|小时|个工作日|工作日|个自然日|自然日|日|天|周|(?:business|working|calendar)?\s*days?|workdays?|核|g)?", text.lower()))
        normalized = set()
        for value in values:
            # Same canonicalisation as the candidate channel: equal values
            # in different exact units ("500ms" vs "0.5s") are duplicates.
            normalized.add(canonical_unit_value(re.sub(r"\s+", "", value).replace("gb", "g")))
        return normalized

    # Same concrete values + high lexical overlap is a duplicate, not a conflict.
    # 0.15.8: the lexical rider (char_cosine>=0.45) is replaced by full
    # equality of the value-stripped skeleton. Equal numeric sets alone say
    # nothing — two memories sharing a date (2026-09-08 …) or any common
    # numbers keep equal numeric sets while differing in every non-numeric
    # token (json vs csv), and the cosine rider silently swallowed those real
    # conflicts (the write-time silent-drop root cause R1).
    if (
        numeric_values(left_text)
        and numeric_values(left_text) == numeric_values(right_text)
        and _numeric_stripped_skeleton(left_text) == _numeric_stripped_skeleton(right_text)
    ):
        duplicate_guard = True

    # Alias/name statements that share the same concrete alias tokens are duplicates.
    if ("mema" in lower and "迷码" in lower) and ("alias" in lower or "cli" in lower or "命令" in lower or "中文名" in lower):
        duplicate_guard = True

    # A negated excluded alternative plus a shared positive value can be compatible
    # support, but never suppress explicit replacement/old-new wording.
    explicit_replacement = any(term in joined for term in ["以后以", "替换", "改为", "不再采用", "之前不对", "旧设计", "新设计", "新口径", "旧口径", "下线", "而不是", "不公开", "公开"])
    if polarity_diff and not explicit_replacement and len(common) >= 2 and only_right and not contains_diff:
        compatible_guard = True
    # Comparators and signs survive neither normalization nor the value
    # strip: pairs whose only difference is an operator (">= 100ms" vs
    # "< 100ms", "+5%" vs "-5%") contradict and must never be suppressed
    # by the duplicate/compatible guards.
    if _operator_signature(left_text) != _operator_signature(right_text):
        duplicate_guard = False
        compatible_guard = False
    return PairEvidence(
        common_tokens=sorted(common),
        char_cosine=char_cosine,
        token_cosine=token_cosine,
        replacement=replacement,
        todo_done=todo_done,
        contains_diff=contains_diff,
        polarity_diff=polarity_diff,
        list_value_diff=list_value_diff,
        duplicate_guard=duplicate_guard,
        compatible_guard=compatible_guard,
        only_left=sorted(only_left),
        only_right=sorted(only_right),
    )


_MAX_ATTRIBUTE_CHARS = 80
# value_is_grounded bounds (pair-v7 owner 口径)：值过长/词过多视为"抄句子"
# 而非抽取值——grounding 的反 copy-the-sentence 规则。
_MAX_VALUE_CHARS = 64
_MAX_VALUE_WORDS = 12
_ATTRIBUTE_ALIASES = {
    "dbselection": "databasechoice",
    "databaseengine": "databasechoice",
    "数据库选型": "数据库选择",
    "数据库引擎": "数据库选择",
}
_VALUE_ALIASES = {
    "pgsql": "postgresql",
    "postgres": "postgresql",
}
# A former "sqlitevec" -> "sqlite-vec" entry was deleted as a dead item:
# _mechanical_normalize strips hyphens/underscores/dots before this lookup,
# so every spelling ("sqlite-vec", "sqlite_vec", "sqlite vec") already
# converges to "sqlitevec" and the alias only re-labelled the output form.
# Any non-identity re-labelling is unsafe at the persistence boundary:
# db/conflicts.py merges conflict value groups keyed by the stored
# normalized_value string, so a group persisted as "sqlitevec" would split
# away from newly written "sqlite-vec" values. Without the entry,
# normalize_value output is byte-identical to the pre-alias baseline.


def _mechanical_normalize(value: str) -> str:
    normalized = (value or "").casefold().strip()
    normalized = re.sub(r"(?<=\d)[,_](?=\d)", "", normalized)
    normalized = re.sub(r"(\d+(?:\.\d+)?)\s*秒\b", r"\1s", normalized)
    normalized = re.sub(r"(\d+(?:\.\d+)?)\s*毫秒\b", r"\1ms", normalized)
    # Unit-spelling compaction mirroring the recall guard's equivalences so
    # "8GB" and "8G" normalize equal at the post-gate too.
    normalized = re.sub(r"(?<=\d)\s*(gb|kb|mb|tb)(?![a-z0-9])", lambda match: match.group(1)[0], normalized)
    # Strip separators/punctuation but PRESERVE a decimal point between digits,
    # so 1.5 and 15 do not collapse. Keep word chars and ".", drop "_", then
    # remove any "." not flanked by digits (so dotnet/a.b are unaffected).
    normalized = re.sub(r"[^\w.]+", "", normalized, flags=re.UNICODE)
    normalized = normalized.replace("_", "")
    normalized = re.sub(r"(?<!\d)\.|\.(?!\d)", "", normalized)
    return normalized


_ATTRIBUTE_VALUE_RUN_RE = re.compile(r"\d+(?:\.\d+)*")


def normalize_attribute(value: str) -> str:
    normalized = _mechanical_normalize(value)
    # The prompt contract forbids values inside attributes ("不包含具体值"),
    # but the 0.5B bleeds the threshold in ("单笔退款超过 500 元" vs "…5000
    # 元") and the strict bidirectional mirror then reads the two directions
    # as different attributes (eval cf-oppo-05, 2026-09-16). Strip numeric
    # runs so one attribute with different embedded values canonicalises
    # together; the VALUE fields still carry the actual difference.
    normalized = _ATTRIBUTE_VALUE_RUN_RE.sub("", normalized)
    return _ATTRIBUTE_ALIASES.get(normalized, normalized)


_VALUE_APPROX_PREFIX_RE = re.compile(r"^(?:大约|大概|约|近)")
_VALUE_MEASURE_GE_RE = re.compile(r"(?<=\d)个")

# Exact physical unit conversion (2026-09-16, owner-directed): time → ms,
# size → kb, Chinese magnitude suffixes fold into the number, 块 is a
# colloquial 元. Only EXACT relations convert — 月/年 (variable length)
# deliberately stay unconverted. 工作日/自然日 convert as natural 24h days
# (owner 2026-09-17, cf-oppo-12): in the OPPOSITION direction either reading
# (natural day vs 8h workday) keeps the values unequal, so identification is
# unaffected by the ambiguity; the only behavioural change is that
# "3 个工作日 vs 72 小时" now reads as the same value on the duplicate side.
# Covers both the raw spellings (gb) and the _mechanical_normalize compacted
# forms (g); English business/working/calendar days normalize with spaces
# stripped, so the keys are the concatenated spellings.
_UNIT_TO_CANONICAL: dict[str, "tuple[Decimal, str]"] = {
    "ms": (Decimal(1), "ms"),
    "s": (Decimal(1000), "ms"), "秒": (Decimal(1000), "ms"),
    "sec": (Decimal(1000), "ms"), "secs": (Decimal(1000), "ms"),
    "second": (Decimal(1000), "ms"), "seconds": (Decimal(1000), "ms"),
    "分钟": (Decimal(60_000), "ms"), "min": (Decimal(60_000), "ms"),
    "mins": (Decimal(60_000), "ms"), "minute": (Decimal(60_000), "ms"),
    "minutes": (Decimal(60_000), "ms"),
    "小时": (Decimal(3_600_000), "ms"), "h": (Decimal(3_600_000), "ms"),
    "hr": (Decimal(3_600_000), "ms"), "hrs": (Decimal(3_600_000), "ms"),
    "hour": (Decimal(3_600_000), "ms"), "hours": (Decimal(3_600_000), "ms"),
    "天": (Decimal(86_400_000), "ms"), "日": (Decimal(86_400_000), "ms"),
    "工作日": (Decimal(86_400_000), "ms"), "个工作日": (Decimal(86_400_000), "ms"),
    "自然日": (Decimal(86_400_000), "ms"), "个自然日": (Decimal(86_400_000), "ms"),
    "day": (Decimal(86_400_000), "ms"), "days": (Decimal(86_400_000), "ms"),
    "businessday": (Decimal(86_400_000), "ms"), "businessdays": (Decimal(86_400_000), "ms"),
    "workday": (Decimal(86_400_000), "ms"), "workdays": (Decimal(86_400_000), "ms"),
    "workingday": (Decimal(86_400_000), "ms"), "workingdays": (Decimal(86_400_000), "ms"),
    "calendarday": (Decimal(86_400_000), "ms"), "calendardays": (Decimal(86_400_000), "ms"),
    "周": (Decimal(604_800_000), "ms"), "week": (Decimal(604_800_000), "ms"),
    "weeks": (Decimal(604_800_000), "ms"),
    "kb": (Decimal(1), "kb"), "k": (Decimal(1), "kb"),
    "mb": (Decimal(1024), "kb"), "m": (Decimal(1024), "kb"),
    "gb": (Decimal(1024 ** 2), "kb"), "g": (Decimal(1024 ** 2), "kb"),
    "tb": (Decimal(1024 ** 3), "kb"), "t": (Decimal(1024 ** 3), "kb"),
    "元": (Decimal(1), "元"), "块": (Decimal(1), "元"), "块钱": (Decimal(1), "元"),
    "千": (Decimal(1000), ""), "万": (Decimal(10_000), ""),
}
_VALUE_NUM_UNIT_RE = re.compile(r"^(\d+(?:\.\d+)?)([a-z]+|[一-鿿]+)?$")

# 0.17.0 P2-1.2: Chinese duration words never match _VALUE_NUM_UNIT_RE (they
# start with a numeral character, not a digit), so "半秒 vs 500毫秒" failed
# normalization and surfaced as a FALSE conflict (baseline ny-neg-samevalue,
# live evidence). The fold runs on the whole anchored value only — prose is
# never rewritten here. 刻钟 folds through minutes; 点 stays a bare suffix
# (clock time compares fine as "2点"/"10点" strings once numerals fold).
_CN_DURATION_VALUE_RE = re.compile(
    r"^(半|[零一二三四五六七八九十百千万两]+)(毫秒|秒|分钟|小时|天|周|点|刻钟)$"
)


def _fold_chinese_duration(normalized: str) -> str:
    match = _CN_DURATION_VALUE_RE.match(normalized)
    if not match:
        return normalized
    numeral, unit = match.group(1), match.group(2)
    if numeral == "半":
        count = "0.5"
    else:
        parsed = _cn_to_int(numeral)
        if parsed is None:
            return normalized
        count = str(parsed)
    if unit == "刻钟":
        # 「半刻钟」numeral=半 → count="0.5"，int("0.5") 曾直接 ValueError
        # （归一层放行「刻钟」而准入层 _CN_HALF_UNIT_RE 刻意不收，两层口径
        # 矛盾，R2 review）。按浮点折算：整刻钟保持整数拼写，半刻钟=7.5分钟。
        minutes = float(count) * 15
        return f"{int(minutes) if minutes == int(minutes) else minutes}分钟"
    return f"{count}{unit}"


def canonical_unit_value(normalized: str) -> str:
    """Canonicalise a normalized "number+unit" value; non-matching shapes
    and unconvertible units pass through unchanged (identity by default —
    the persistence-boundary rule from _VALUE_ALIASES)."""
    match = _VALUE_NUM_UNIT_RE.match(normalized)
    if not match or not match.group(2):
        return normalized
    conversion = _UNIT_TO_CANONICAL.get(match.group(2))
    if conversion is None:
        return normalized
    factor, base = conversion
    value = Decimal(match.group(1)) * factor
    number = format(value.normalize(), "f")
    return f"{number}{base}"


def normalize_value(value: str) -> str:
    normalized = _mechanical_normalize(value)
    normalized = _fold_chinese_duration(normalized)
    normalized = _VALUE_ALIASES.get(normalized, normalized)
    # Direction-dependent fillers the 0.5B attaches to an otherwise identical
    # value ("近 90 天" vs "90 天", "5个工作日" vs "5工作日") used to fail the
    # strict bidirectional mirror as bidirectional_mapping_mismatch even
    # though both directions agreed (eval cf-oppo-10/12, 2026-09-16).
    # Approximator prefixes and the measure word 个 carry no value semantics.
    normalized = _VALUE_APPROX_PREFIX_RE.sub("", normalized)
    normalized = _VALUE_MEASURE_GE_RE.sub("", normalized)
    return canonical_unit_value(normalized)


def _bounded_short_value(value: str, quote: str) -> bool:
    """Reject copied whole quotes/sentences while allowing compact slot values."""
    compact = value.strip()
    source = quote.strip()
    if not compact or len(compact) > _MAX_VALUE_CHARS:
        return False
    if len(re.findall(r"\S+", compact)) > _MAX_VALUE_WORDS:
        return False
    value_norm = _mechanical_normalize(compact)
    quote_norm = _mechanical_normalize(source)
    if not value_norm or value_norm == quote_norm:
        return False
    # Sentence punctuation is a strong signal that the model copied a clause
    # instead of extracting a name, number, state, or short policy value.
    if re.search(r"[。！？!?；;](?:[\"'”’）)]*)$", compact):
        return False
    # Continuation marks at the end (，、：；—) are the same signal from the
    # 64-char hard cut: the cap (grammar era: decode-level maxLength; since
    # 0.15.14: L3 truncation) guillotines an over-long copy at exactly 64
    # chars, which often lands mid-clause (observed live:
    # "…不替 Agent 挑重要命中，").
    if re.search(r"[，、：；—…]$", compact):
        return False
    folded_value = compact.casefold()
    folded_quote = source.casefold()
    if folded_value in folded_quote:
        # Guillotine detection, the other tell: an exact-copy fragment whose
        # every occurrence is immediately followed by a word character is the
        # hard-cut head of a longer token/clause ("重新确" before "认"), not a
        # value the model chose. Boundaries at punctuation/space/end are fine.
        for occurrence in re.finditer(re.escape(folded_value), folded_quote):
            end = occurrence.end()
            if end >= len(folded_quote) or not re.match(r"\w", folded_quote[end]):
                break
        else:
            return False
    return True


def value_is_grounded(value: str, quote: str) -> bool:
    """Accept only bounded short values with mechanical grounding in the quote."""
    if not quote or not _bounded_short_value(value, quote):
        return False
    if value.casefold() in quote.casefold():
        return True
    target = normalize_value(value)
    if not target:
        return False
    # Compare against bounded quote tokens/phrases; this permits case, spacing,
    # punctuation, numeric formatting, units and explicit aliases, not paraphrase.
    # The numeric branch carries common CJK units so "90天" grounds in
    # "近 90 天" (eval cf-oppo-10/12, 2026-09-16): Qwen's spacing varies with
    # direction, and without the unit the piece degrades to a bare "90".
    pieces = re.findall(
        r"[A-Za-z][A-Za-z0-9_.-]*"
        r"|\d[\d,_.]*\s*个?\s*(?:ms|s|毫秒|秒|分钟|小时|工作日|天|日|周|月|年|%|mb|gb|kb|元|次|条|倍|人|台|核)?"
        r"|[\u4e00-\u9fff]{1,24}",
        quote,
    )
    return any(normalize_value(piece) == target for piece in pieces)


def notice_dedupe_key(
    left_id: int,
    right_id: int,
    left_version: int,
    right_version: int,
    notice_type: str,
) -> str:
    left = (int(left_id), int(left_version))
    right = (int(right_id), int(right_version))
    (a_id, a_version), (b_id, b_version) = sorted(
        [left, right], key=lambda item: item[0]
    )
    raw = f"semantic:{a_id}:{a_version}:{b_id}:{b_version}:{notice_type}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
