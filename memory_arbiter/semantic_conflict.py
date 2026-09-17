"""Local extraction protocol and isolated GGUF process supervision.

The model extracts comparable attribute/value fields from candidate evidence
pairs. Deterministic gates may accept, reject, or request extraction; advisory
semantic notices still require an agent to read both memories before acting.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import multiprocessing
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol

from .constants import (
    SEMANTIC_N_CTX,
    SEMANTIC_PAIR_MAX_ATTEMPTS,
    SEMANTIC_PAIR_RETRY_MAX_TOKENS,
    SEMANTIC_PAIR_RETRY_QUOTE_CHARS,
)
from .tokens import estimate_tokens

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

PAIR_PROMPT_VERSION = "pair-v8"

_PAIR_PROMPT = """你只做条件抽槽，直接以 { 开头输出一个 JSON 对象，不要解释、复述输入或裁决。
对象必须恰好包含四个字符串字段：attribute_a、value_a、attribute_b、value_b。
attribute 是两侧正在回答的最小可比较问题，不包含具体值、时间、环境或版本；value 是原证据中该属性的具体取值，取原文中的连续片段，长度不超过 64 字、不超过 12 个词；原句过长时截取最能体现取值差异的连续片段，禁止整句照抄，value 不得以句号、叹号、分号等句末标点结尾。
无论是否能可靠抽取，都必须输出全部四个字符串字段，不得省略字段。无法可靠抽取时将对应字段写成字符串 "__unknown__"；不要输出 null、conflict、coexistence、winner、confidence 或额外字段。
例：A=生产数据库使用 MySQL。B=生产数据库使用 SQLite。
输出：{"attribute_a":"数据库选型","value_a":"MySQL","attribute_b":"数据库选型","value_b":"SQLite"}"""

# pair-v8 (2026-09-16): English mirror of the protocol for non-CJK evidence.
# Field names stay identical (downstream parsing is schema-fixed); only the
# instructions and the few-shot change language. Chinese instructions on
# English evidence made the 0.5B bleed the threshold into the attribute
# (eval: "Refund requests over 5000/500 CNY" pair, job completed, zero
# notice). The retry feedback turn stays Chinese (schema-level, measured
# wording — changing it is a separate calibration).
_PAIR_PROMPT_EN = """You only do conditional slot extraction. Output a single JSON object starting with { — no explanation, no restating the input, no verdicts.
The object must contain exactly four string fields: attribute_a, value_a, attribute_b, value_b.
attribute is the minimal comparable question both sides answer; it must not contain concrete values, times, environments, or versions. value is that attribute's concrete value in the evidence: a contiguous fragment copied from the source, at most 64 chars and 12 words; for long sentences copy the shortest fragment that carries the value difference; never copy a whole sentence; a value must not end with sentence punctuation.
Always output all four string fields, even when extraction is unreliable; write the string "__unknown__" for fields you cannot reliably extract. Do not output null, conflict, coexistence, winner, confidence, or extra fields.
Example: A=The production database uses MySQL. B=The production database uses SQLite.
Output: {"attribute_a":"database engine","value_a":"MySQL","attribute_b":"database engine","value_b":"SQLite"}"""

_CJK_RE = re.compile(r"[一-鿿]")


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
# (_pair_text), present only when decide_evidence extracted values on both
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
class WorkspaceCandidateSignal:
    candidate: str | None
    relation: str
    confidence: float | None
    evidence: str
    raw: str = ""
    error: str | None = None


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
class ModelSignal:
    candidate: bool
    candidate_type: str
    confidence: float | None
    raw: str
    parsed: dict[str, Any] | None
    error: str | None = None
    # A1 ring instrumentation (0.15.14): per-call token accounting so the
    # parent's pair-timing ring can separate long decodes from queue waits.
    # Filled only by backends that actually ran the model; None on errors.
    prompt_tokens: int | None = None
    generated_tokens: int | None = None
    retried: bool = False


@dataclass
class EvidenceDecision:
    action: str
    reason: str
    anchors: list[str] = field(default_factory=list)
    left_value: str | None = None
    right_value: str | None = None


@dataclass(frozen=True)
class AttributeValueExtraction:
    attribute_a: str
    value_a: str
    attribute_b: str
    value_b: str


@dataclass(frozen=True)
class PairGateResult:
    state: str
    reason: str
    attribute: str | None = None
    value_a: str | None = None
    value_b: str | None = None
    grounded: bool = False


class SemanticBackend(Protocol):
    def classify_pair(
        self, left: dict[str, Any], right: dict[str, Any],
        *,
        deadline_monotonic: float | None = None,
        retry_allowed: bool = True,
    ) -> ModelSignal:
        ...

    def suggest_workspace_candidate(
        self,
        ws_raw: str,
        evidence: dict[str, Any],
        candidates: list[str],
        *,
        deadline_monotonic: float | None = None,
    ) -> WorkspaceCandidateSignal:
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
_VALUE_RE = re.compile(
    r"(?<![\w.])v?\d+(?:\.\d+){0,2}\s*"
    r"(?:ms|s|秒|分钟|小时|个工作日|工作日|个自然日|自然日|日|天|%|mb|gb|kb|条|次|核|g"
    r"|(?:business|working|calendar)?\s*days?|workdays?)?",
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
        vec_a = embedder.embed_text(prefix="", body=key_a)
        vec_b = embedder.embed_text(prefix="", body=key_b)
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


def _extract_first_json_object(raw: str) -> str | None:
    """Return the first balanced top-level ``{...}`` object in *raw*, or None.

    Small models frequently emit JSON with nested objects/arrays or trailing
    prose. A non-greedy ``\\{.*?\\}`` regex stops at the first ``}``, which
    truncates nested payloads (``{"a": {"b": 1}}`` -> ``{"a": {"b": 1}``) and
    silently drops the candidate. Brace-balanced extraction is robust to
    nesting while still returning only the first object so trailing text does
    not break ``json.loads``. Strings are tracked so braces inside string
    literals do not affect depth.
    """
    text = raw or ""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


_EXTRACTION_FIELDS = {"attribute_a", "value_a", "attribute_b", "value_b"}
_UNKNOWN_SENTINEL = "__unknown__"
_MAX_ATTRIBUTE_CHARS = 80
# Extracted values are slot values, never prose.  Keep the protocol bound well
# below the quote/attribute limits so a model cannot make an entire sentence
# appear "grounded" merely by copying it verbatim.
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


def extraction_from_text(raw: str) -> tuple[AttributeValueExtraction | None, str | None]:
    """Strictly parse the bounded four-field conflict extraction protocol."""
    snippet = _extract_first_json_object(raw or "")
    if not snippet:
        return None, "missing_json"
    text = raw or ""
    bracket, brace = text.find("["), text.find("{")
    if bracket != -1 and (brace == -1 or bracket < brace):
        # The first structural token is an array (bare or prose-prefixed): extra
        # top-level structure the protocol rejects, not an object to dig out of
        # prose (spec §15.4).
        return None, "invalid_schema:array_wrapper"

    try:
        parsed = json.loads(snippet, object_pairs_hook=_reject_duplicate_fields)
    except Exception as exc:
        return None, f"invalid_json:{exc}"
    if not isinstance(parsed, dict) or set(parsed) != _EXTRACTION_FIELDS:
        return None, "invalid_schema"
    limits = {
        "attribute_a": _MAX_ATTRIBUTE_CHARS, "attribute_b": _MAX_ATTRIBUTE_CHARS,
        "value_a": _MAX_VALUE_CHARS, "value_b": _MAX_VALUE_CHARS,
    }
    values: dict[str, str] = {}
    for field_name, limit in limits.items():
        value = parsed[field_name]
        if not isinstance(value, str):
            return None, "invalid_schema"
        value = value.strip()
        if not value or len(value) > limit or "\n" in value or "\r" in value:
            return None, f"invalid_{field_name}"
        values[field_name] = value
    if any(value.casefold() == _UNKNOWN_SENTINEL for value in values.values()):
        # The protocol-legal "cannot reliably extract" marker: a valid model
        # response, but never a usable extraction for any gate (spec §15.4).
        return None, "unknown_field"
    return AttributeValueExtraction(**values), None


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


def coexistence_veto(
    left: dict[str, Any],
    right: dict[str, Any],
    forward: AttributeValueExtraction | None = None,
    reverse: AttributeValueExtraction | None = None,
) -> str | None:
    """Return a deterministic coexistence reason code, or None when unknown."""
    left_text = str(left.get("quote") or left.get("content") or "").casefold()
    right_text = str(right.get("quote") or right.get("content") or "").casefold()
    # 2026-08-28 (B-C1): dimension markers used to be matched as bare
    # substrings of the whole quote, so an unrelated "平均"/"峰值" (etc.)
    # anywhere in the prose vetoed a genuine same-attribute conflict and the
    # notice was silently dropped (false negative). When extractions are
    # supplied (the notice gate always passes them) a marker pair only counts
    # as a dimension difference when each marker appears in the corresponding
    # side's extracted attribute text — markers that occur only in the quote
    # body no longer veto, so more real conflicts surface as notices. Callers
    # without extractions keep the legacy quote-substring behaviour.
    if forward is not None or reverse is not None:
        # forward was extracted left→right, reverse right→left.
        left_dimension = " ".join(filter(None, (
            forward.attribute_a if forward is not None else "",
            reverse.attribute_b if reverse is not None else "",
        ))).casefold()
        right_dimension = " ".join(filter(None, (
            forward.attribute_b if forward is not None else "",
            reverse.attribute_a if reverse is not None else "",
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
    # 2026-09-17 (owner): a value pair that is BOTH version ordinals is
    # release-to-release evolution, never a conflict — "v0.2.1 vs v0.2.2
    # 发版记录" pairs read as a 版本号 difference and slip past the literal
    # v1/v2 marker pair. Version shape = v-prefixed with any dotted tail
    # (v2, v0.2, v0.2.1) or an unprefixed 3+ segment ordinal (0.2.1,
    # 0.16.7); a SINGLE dotted number (2.5) is an ordinary rate/quantity
    # value and must not be vetoed. Both sides must be version-shaped.
    if forward is not None:
        va, vb = normalize_value(forward.value_a), normalize_value(forward.value_b)
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
        # Requires real replacement wording — a bare "由" is an ordinary
        # passive/agent marker in Chinese and must not co-trigger the veto.
        return "coexist_explicit_evolution"
    return None


def evaluate_pair_extractions(
    forward: AttributeValueExtraction | None,
    reverse: AttributeValueExtraction | None,
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    require_bidirectional: bool,
) -> PairGateResult:
    """Apply scan/notice gates; reverse is extracted from input order B→A."""
    valid = [item for item in (forward, reverse) if item is not None]
    if not valid:
        return PairGateResult("review_candidate", "qwen_unverified")
    if require_bidirectional and (forward is None or reverse is None):
        return PairGateResult("review_candidate", "bidirectional_extraction_required")

    def one_direction(item: AttributeValueExtraction) -> bool:
        return (
            normalize_attribute(item.attribute_a) == normalize_attribute(item.attribute_b)
            and normalize_value(item.value_a) != normalize_value(item.value_b)
        )

    if not any(one_direction(item) for item in valid):
        return PairGateResult("review_candidate", "not_same_attribute_different_value")
    if forward is None or reverse is None:
        item = valid[0]
        return PairGateResult(
            "review_candidate", "single_direction_only", normalize_attribute(item.attribute_a),
            normalize_value(item.value_a), normalize_value(item.value_b), False,
        )
    if not (one_direction(forward) and one_direction(reverse)):
        return PairGateResult("review_candidate", "direction_invalid")
    if not (
        normalize_attribute(forward.attribute_a) == normalize_attribute(reverse.attribute_b)
        and normalize_attribute(forward.attribute_b) == normalize_attribute(reverse.attribute_a)
        and normalize_value(forward.value_a) == normalize_value(reverse.value_b)
        and normalize_value(forward.value_b) == normalize_value(reverse.value_a)
    ):
        return PairGateResult("review_candidate", "bidirectional_mapping_mismatch")
    quote_a = str(left.get("quote") or left.get("content") or "")
    quote_b = str(right.get("quote") or right.get("content") or "")
    grounded = value_is_grounded(forward.value_a, quote_a) and value_is_grounded(forward.value_b, quote_b)
    if not grounded:
        return PairGateResult("review_candidate", "qwen_unverified")
    veto = coexistence_veto(left, right, forward, reverse)
    if veto:
        return PairGateResult("review_candidate", veto, grounded=True)
    return PairGateResult(
        "notice_ready", "same_attribute_different_grounded_value",
        normalize_attribute(forward.attribute_a), normalize_value(forward.value_a),
        normalize_value(forward.value_b), True,
    )


def signal_extraction(signal: Any) -> AttributeValueExtraction | None:
    """Pull the validated four-field extraction out of a ModelSignal, if any."""
    parsed = getattr(signal, "parsed", None)
    if not isinstance(parsed, dict):
        return None
    try:
        return AttributeValueExtraction(**{key: parsed[key] for key in (
            "attribute_a", "value_a", "attribute_b", "value_b",
        )})
    except (KeyError, TypeError):
        return None


def model_signal_from_text(raw: str) -> ModelSignal:
    extraction, error = extraction_from_text(raw)
    if extraction is None:
        if error and (error.startswith("invalid_json") or error == "missing_json"):
            kind = "invalid_json"
        elif error == "unknown_field":
            # Explicit "cannot extract": distinguish it from protocol violations
            # so diagnostics separate model output from technical failure.
            kind = "unknown_field"
        else:
            kind = "invalid_schema"
        return ModelSignal(False, kind, None, raw or "", None, error)
    parsed = {
        "attribute_a": extraction.attribute_a, "value_a": extraction.value_a,
        "attribute_b": extraction.attribute_b, "value_b": extraction.value_b,
    }
    candidate = (
        normalize_attribute(extraction.attribute_a) == normalize_attribute(extraction.attribute_b)
        and normalize_value(extraction.value_a) != normalize_value(extraction.value_b)
    )
    return ModelSignal(candidate, "attribute_value_extraction", None, raw or "", parsed)


_WS_RELATIONS = {"alias", "typo", "same_project", "same_family", "related", "unrelated", "uncertain"}


def workspace_candidate_from_text(raw: str, candidates: list[str]) -> "WorkspaceCandidateSignal":
    """Parse a workspace-suggester JSON blob into a WorkspaceCandidateSignal.

    Guards the candidate against hallucination: a suggested candidate that is
    not in the provided list is dropped (candidate=None, relation=uncertain).
    """
    snippet = _extract_first_json_object(raw or "")
    if not snippet:
        return WorkspaceCandidateSignal(None, "uncertain", None, raw or "", raw or "", "missing_json")
    try:
        parsed = json.loads(snippet)
    except (ValueError, TypeError) as exc:
        return WorkspaceCandidateSignal(None, "uncertain", None, raw or "", raw or "", str(exc))
    if not isinstance(parsed, dict):
        return WorkspaceCandidateSignal(None, "uncertain", None, raw or "", raw or "", "not_object")

    candidate = parsed.get("candidate")
    candidate = str(candidate).strip() if candidate else None
    # Anti-hallucination: only accept a candidate the caller actually offered.
    if candidate and candidates and candidate not in candidates:
        candidate = None
    relation = str(parsed.get("relation") or "uncertain").strip().lower()
    if relation not in _WS_RELATIONS:
        relation = "uncertain"
    if candidate is None and relation in {"alias", "typo", "same_project"}:
        relation = "uncertain"
    conf_raw = parsed.get("confidence")
    try:
        confidence = float(conf_raw) if conf_raw is not None else None
    except (TypeError, ValueError):
        confidence = None
    # Clamp a hallucinated out-of-range confidence into [0,1] so downstream
    # threshold gates can't be tricked by an inflated value like 5.0.
    if confidence is not None:
        confidence = max(0.0, min(1.0, confidence))
    evidence = str(parsed.get("evidence") or "").strip()
    return WorkspaceCandidateSignal(candidate, relation, confidence, evidence, raw or "", None)


def _pair_retry_strategy(error: str | None) -> str | None:
    """Map a failed pair extraction's error to a feedback-retry strategy.

    Returns None when a retry cannot help: legal negatives (unknown_field),
    backend failures, and successful extractions never reach here — the caller
    only consults this for invalid_json/invalid_schema signals.
    """
    if not error:
        return None
    if error.startswith(("invalid_value_", "invalid_attribute_")):
        return "over_limit"
    if error == "missing_json" or error.startswith("invalid_json"):
        return "truncated"
    if error.startswith("invalid_schema"):
        return "schema"
    return None


def _extra_field_names(raw: str) -> list[str]:
    """Top-level keys of the parsed object that the four-field protocol rejects
    (the 0.5B writes __unknown__ / event_time / workspace_canonical as FIELD
    names when it confuses metadata with the schema)."""
    snippet = _extract_first_json_object(raw or "")
    if not snippet:
        return []
    try:
        parsed = json.loads(snippet)
    except ValueError:
        return []
    if not isinstance(parsed, dict):
        return []
    return [str(key) for key in parsed if key not in _EXTRACTION_FIELDS]


def _field_state(raw: str, error: str) -> str:
    """What actually went wrong with the named field, read from the raw output.

    ``invalid_<field>`` is a single error code covering three different causes,
    and the feedback must name the REAL one: the pre-0.15.14 wording assumed
    "empty/whitespace/newline" because grammar-era maxLength kept over-long
    values from ever reaching a retry. With grammar-free decoding an over-long
    copy lands here instead — and telling the 0.5B "your field is empty" when
    it just copied 257 chars sends the retry to guess a value out of thin air
    (observed live: the retry answered {"attribute_a":"历史待办","value_a":"无"}).
    """
    field_name = error.removeprefix("invalid_")
    snippet = _extract_first_json_object(raw or "")
    field_value: Any = None
    if snippet:
        try:
            parsed = json.loads(snippet)
        except ValueError:
            parsed = None
        if isinstance(parsed, dict):
            field_value = parsed.get(field_name)
    if not isinstance(field_value, str):
        # Not a string at all (missing object, wrong type, unparseable raw):
        # only a fresh, well-formed attempt can fix this.
        return "shape"
    limit = _MAX_ATTRIBUTE_CHARS if field_name.startswith("attribute_") else _MAX_VALUE_CHARS
    if len(field_value.strip()) > limit:
        return "too_long"
    if not field_value.strip() or "\n" in field_value or "\r" in field_value:
        return "empty"
    return "shape"


def _pair_retry_feedback(strategy: str, error: str, raw: str = "") -> str:
    """Feedback turn for the retry — names the SPECIFIC violation.

    Measured (2026-09-11, real-model fixtures): the generic four-field reminder
    cannot fix the "extra field" family at all (0/6), while naming the actual
    offending keys with no echo recovers 5/5 in 0.5-0.9 s. The failed output
    is deliberately NOT echoed: with the raw in context the 0.5B stays locked
    in the copy state and re-produces it (0/9 across echo variants). Wording
    for the fragment rules mirrors the pair-v6 prompt's own instruction (a
    "compress the value" phrasing was tried and rejected: it flattened
    opposing values into equality).
    """
    if strategy == "over_limit":
        field_name = error.removeprefix("invalid_")
        state = _field_state(raw, error)
        if state == "too_long":
            if field_name.startswith("attribute_"):
                return (
                    f"上次输出的 {field_name} 超过 80 字。attribute 是两侧正在回答的最小可比较"
                    "问题，不包含具体值，只写短词，直接以 { 开头输出完整 JSON。"
                )
            return (
                f"上次输出的 {field_name} 超过 64 字（照抄了整段原文）。value 只保留最能体现"
                "取值差异的最短连续片段，长度不超过 64 字、不超过 12 个词，禁止整句照抄，"
                "两个 value 不得写成同一段；直接以 { 开头输出完整 JSON。"
            )
        if state == "empty":
            if field_name.startswith("attribute_"):
                return (
                    f"上次输出的 {field_name} 为空、是纯空白或含换行。attribute 是两侧正在回答的"
                    "最小可比较问题，不包含具体值，长度不超过 80 字，直接以 { 开头输出完整 JSON。"
                )
            return (
                f"上次输出的 {field_name} 为空、是纯空白或含换行。value 是原证据中该属性的具体"
                "取值片段（不超过 64 字、不超过 12 个词），截取最能体现取值差异的连续片段，"
                "禁止整句照抄，直接以 { 开头输出完整 JSON。"
            )
        return (
            f"上次输出的 {field_name} 不合协议。它必须是原证据中的短取值片段：value 不超过 "
            "64 字、不超过 12 个词，attribute 不超过 80 字，禁止整句照抄，"
            "直接以 { 开头输出完整 JSON。"
        )
    if strategy == "truncated":
        return (
            "上次输出未完成就被截断。value 只截取最短的取值片段（不超过 64 字），"
            "禁止照抄整句，直接以 { 开头输出完整 JSON。"
        )
    extra = _extra_field_names(raw)
    if extra:
        quoted = "、".join(f"“{name}”" for name in extra)
        return (
            f"上次输出多出了字段 {quoted}。这些不是字段名——只有 attribute_a、value_a、"
            "attribute_b、value_b 四个字段合法，其余一律不要输出；直接以 { 开头输出完整 JSON。"
        )
    return (
        "上次输出的 JSON 结构不合协议：必须恰好包含 attribute_a、value_a、attribute_b、value_b "
        "四个字符串字段，不要输出其他字段或数组，直接以 { 开头输出完整 JSON。"
    )


def _reject_duplicate_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    decoded: dict[str, Any] = {}
    for key, value in pairs:
        if key in decoded:
            raise ValueError(f"duplicate field: {key}")
        decoded[key] = value
    return decoded


# Clause/word boundaries an over-long value may be cut at (L3, second-round
# review M1). Cutting anywhere else would manufacture a beheaded fragment
# ("…挑重要命") that grounds as a verbatim substring yet no longer reads as
# a value.
_L3_CUT_BOUNDARIES = "，、：；。！？!?…—,;:()（）"


def _l3_cut_value(value: str) -> str | None:
    """Cut an over-long value at the last clause or word boundary inside the cap.

    Clause punctuation wins over whitespace even when a space sits later in
    the window: the live sample's 64-char window ends "…命中，法规 RA", and a
    space-first cut would leave the mid-word tail "法规" while the comma at 58
    terminates the clause cleanly. Returns None when the window holds no
    boundary at all — a mid-run cut would invent exactly the guillotine
    fragment the grounding gates exist to reject, so the pair stays on the
    retry path instead.
    """
    window = value[:_MAX_VALUE_CHARS]
    index = max((i for i, ch in enumerate(window) if ch in _L3_CUT_BOUNDARIES), default=-1)
    if index < 0:
        index = max((i for i, ch in enumerate(window) if ch.isspace()), default=-1)
    if index < 0:
        return None
    cut = window[:index].strip().rstrip(_L3_CUT_BOUNDARIES).strip()
    return cut or None


def _l3_truncated_signal(raw: str) -> ModelSignal | None:
    """L3 post-hoc leniency (A2, 0.15.14): a value field over the 64-char cap
    is cut to a clause boundary inside the cap instead of invalidating the
    whole output — this replaces the grammar-era decoding-level maxLength.
    Only the exact-four-field shape with cuttable values qualifies; anything
    else (extra/missing fields, bad attributes, embedded newlines, no clause
    boundary inside the window) stays on the retry path. Attributes are never
    truncated: an over-long attribute is a malformed question, not a value the
    evidence supports.

    L3 relaxes the LENGTH caps only — the strict parser's other protocol
    rejections still apply: an array-wrapped raw (first structural token is
    "[", spec §15.4) and duplicated fields (ambiguous which value was meant)
    are refused here too instead of being silently accepted by plain
    json.loads (first-round review M1).

    Second-round review M1/M2 fixes: the cut lands on a clause/word boundary
    (never mid-clause), and a pair whose two values normalize EQUAL after
    cutting is refused — otherwise a real conflict whose divergence sits
    beyond the cap would silently become a definitive
    not_same_attribute_different_value negative."""
    snippet = _extract_first_json_object(raw or "")
    if not snippet:
        return None
    text = raw or ""
    bracket, brace = text.find("["), text.find("{")
    if bracket != -1 and (brace == -1 or bracket < brace):
        return None
    try:
        parsed = json.loads(snippet, object_pairs_hook=_reject_duplicate_fields)
    except (ValueError, TypeError):
        return None
    if not isinstance(parsed, dict) or set(parsed) != _EXTRACTION_FIELDS:
        return None
    fields: dict[str, str] = {}
    for name in ("attribute_a", "attribute_b"):
        value = parsed.get(name)
        if not isinstance(value, str):
            return None
        value = value.strip()
        if not value or len(value) > _MAX_ATTRIBUTE_CHARS or "\n" in value or "\r" in value:
            return None
        fields[name] = value
    cut_any = False
    for name in ("value_a", "value_b"):
        value = parsed.get(name)
        if not isinstance(value, str):
            return None
        value = value.strip()
        if not value or "\n" in value or "\r" in value:
            return None
        if len(value) > _MAX_VALUE_CHARS:
            cut = _l3_cut_value(value)
            if cut is None:
                return None
            value = cut
            cut_any = True
        fields[name] = value
    if not cut_any:
        return None
    if normalize_value(fields["value_a"]) == normalize_value(fields["value_b"]):
        # Truncation collapsed the distinguishing part: refuse so the pair
        # retries / degrades instead of reading as a clean negative.
        return None
    return model_signal_from_text(json.dumps(fields, ensure_ascii=False))


class LocalGGUFSemanticBackend:
    def __init__(
        self, model_path: Path, *, n_ctx: int = SEMANTIC_N_CTX, n_threads: int = 4,
        n_batch: int = 128, n_gpu_layers: int = 0,
    ) -> None:
        self.model_path = Path(model_path).expanduser()
        self.n_ctx = int(n_ctx)
        self.n_threads = int(n_threads)
        self.n_batch = int(n_batch)
        # 0 = CPU-only (llama-cpp-python's own default); -1 = full offload.
        # A3 (0.15.14): GPU helps prefill regardless of decode length (~1.1x
        # mean on short decodes, 1.15-1.25x on long ones — eval 2026-09-11).
        self.n_gpu_layers = int(n_gpu_layers)
        self._llm: Any = None
        self._cond = threading.Condition(threading.Lock())
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._last_error: str | None = None
        self._loaded_at: float | None = None
        self._inflight = 0
        self._unloading = False
        self._loading = False
        self._disabled = False
        self._generation = 0
        self._pair_retried = 0
        self._pair_retry_recovered = 0
        self._pair_l3_truncated = 0
        self._gpu_fallback = False
        # 2026-09-17 (0.16.8): decode-parameter family routing. Qwen3-style
        # models must drop the "\n\n" stop (their <think>\n\n shell would be
        # cut at the second token) and carry a /no_think prefix. Detected from
        # the GGUF architecture field on load; a legacy-parametered Qwen3
        # self-heals behaviourally (see classify_pair).
        self._qwen3_style = False
        self._think_strikes = 0
        self._family_autodetected = 0

    def _build_llm(self) -> Any:
        if not self.model_path.exists():
            raise FileNotFoundError(str(self.model_path))
        from llama_cpp import Llama
        kwargs: dict[str, Any] = {
            "model_path": str(self.model_path),
            "n_ctx": self.n_ctx,
            "n_threads": self.n_threads,
            "n_gpu_layers": self.n_gpu_layers,
            "verbose": False,
        }
        if self.n_batch > 0:
            kwargs["n_batch"] = self.n_batch
        try:
            llm = Llama(**kwargs)
            # Family routing (0.16.8): general.architecture is a mandatory
            # GGUF field (qwen3 / qwen2 / …) — safer than general.name.
            # Probe failures keep the conservative legacy decode params; the
            # behavioural self-heal in classify_pair is the backstop.
            architecture = str((getattr(llm, "metadata", None) or {}).get("general.architecture") or "")
            self._qwen3_style = "qwen3" in architecture.lower()
            return llm
        except Exception:
            # A3 fallback (second-round review L3): with the offload default
            # flipped on, a host whose Metal/GPU init fails would otherwise
            # re-attempt the failing load on EVERY pair (no backoff, invisible
            # in counters). Mirror the embedder's self-healing policy: fall
            # back to CPU once, remember it, and report it in status().
            if self.n_gpu_layers == 0:
                raise
            kwargs["n_gpu_layers"] = 0
            llm = Llama(**kwargs)
            self._gpu_fallback = True
            return llm

    def _ensure_llm(self) -> Any:
        with self._cond:
            while self._unloading or self._loading:
                self._cond.wait()
            if self._disabled:
                raise RuntimeError("semantic backend disabled")
            if self._llm is not None:
                return self._llm
            self._loading = True
        try:
            with self._load_lock:
                llm = self._build_llm()
        except BaseException:
            with self._cond:
                self._loading = False
                self._cond.notify_all()
            raise
        with self._cond:
            self._llm = llm
            self._loaded_at = time.time()
            self._last_error = None
            self._loading = False
            self._cond.notify_all()
            if self._disabled or self._unloading:
                # Caller requested unload/disable while native loading was in progress.
                self._llm = None
                self._loaded_at = None
                self._generation += 1
                raise RuntimeError("semantic backend disabled")
            return self._llm

    def load(self) -> None:
        self._ensure_llm()

    @staticmethod
    def _memory_text(record: dict[str, Any]) -> str:
        subject = record.get("subject") or ""
        tags = ", ".join(record.get("tags") or []) if isinstance(record.get("tags"), list) else str(record.get("tags") or "")
        raw_metadata = record.get("metadata")
        metadata: dict[str, Any] = raw_metadata if isinstance(raw_metadata, dict) else {}
        fields = (
            ("subject", subject), ("tags", tags),
            ("workspace_canonical", record.get("workspace_canonical")),
            ("memory_id", record.get("memory_id")), ("version", record.get("version")),
            ("event_time", record.get("event_time")),
            ("entity", metadata.get("entity")), ("scope", metadata.get("scope")),
        )
        return "; ".join(
            f"{key}={value}" for key, value in fields if value not in (None, "", [])
        )

    @classmethod
    def _pair_text(cls, left: dict[str, Any], right: dict[str, Any], *, quote_cap: int = 400) -> str:
        """Serialize metadata first and leave both bounded quotes nearest the output."""
        # 400 chars = the local-text segmenter's unit cap, so an evidence unit
        # reaches the model whole (no second truncation); longer text only
        # invites the 0.5B to copy whole clauses into values (the top
        # qwen_invalid_output source before pair-v5). Truncation retries pass a
        # smaller cap: the copy-prone tail is cut and the freed n_ctx budget
        # funds a larger max_tokens for the retry.
        left_quote = str(left.get("quote") or left.get("content") or "")[:quote_cap]
        right_quote = str(right.get("quote") or right.get("content") or "")[:quote_cap]
        cjk = evidence_is_cjk(left, right)
        # pair-v7: when the rule layer already extracted a value difference
        # (decide_evidence's numeric channel), name it as a locating hint.
        # The 0.5B then confirms an attribute around known values instead of
        # re-deriving them — free-running extraction swallowed the threshold
        # into the attribute and vetoed real conflicts (eval cf-oppo-04/05,
        # 2026-09-16). The hint sits BEFORE the quotes: the bounded quotes
        # stay nearest the output (see docstring), and values must still be
        # copied from the evidence text, never from the hint.
        hint = ""
        left_value = str(left.get("rule_value") or "").strip()
        right_value = str(right.get("rule_value") or "").strip()
        if left_value and right_value:
            hint = (
                f"规则层候选值差（仅供定位属性，value 必须取自证据原文）："
                f"A候选值={left_value} B候选值={right_value}\n"
                if cjk else
                f"Rule-layer candidate value difference (locating hint only; "
                f"values must be copied from the evidence text): "
                f"A candidate={left_value} B candidate={right_value}\n"
            )
        if cjk:
            return (
                f"A metadata: {cls._memory_text(left)}\n"
                f"B metadata: {cls._memory_text(right)}\n"
                f"{hint}"
                "只根据以下证据原文抽取 attribute/value：\n"
                f"A证据原文={left_quote}\nB证据原文={right_quote}"
            )
        return (
            f"A metadata: {cls._memory_text(left)}\n"
            f"B metadata: {cls._memory_text(right)}\n"
            f"{hint}"
            "Extract attribute/value from the evidence text only:\n"
            f"A evidence={left_quote}\nB evidence={right_quote}"
        )

    def _acquire_llm_for_call(self) -> Any | None:
        with self._cond:
            while self._unloading or self._loading:
                self._cond.wait()
            if self._disabled:
                return None
            if self._llm is not None:
                self._inflight += 1
                return self._llm
            self._loading = True
            self._inflight += 1
        try:
            with self._load_lock:
                llm = self._build_llm()
        except BaseException:
            with self._cond:
                self._loading = False
                self._inflight = max(0, self._inflight - 1)
                self._cond.notify_all()
            raise
        with self._cond:
            self._llm = llm
            self._loaded_at = time.time()
            self._last_error = None
            self._loading = False
            if self._disabled or self._unloading:
                self._llm = None
                self._loaded_at = None
                self._generation += 1
                self._inflight = max(0, self._inflight - 1)
                self._cond.notify_all()
                return None
            self._cond.notify_all()
            return self._llm

    def _release_llm_for_call(self) -> None:
        with self._cond:
            self._inflight = max(0, self._inflight - 1)
            self._cond.notify_all()

    def classify_pair(
        self, left: dict[str, Any], right: dict[str, Any],
        *,
        deadline_monotonic: float | None = None,
        retry_allowed: bool = True,
    ) -> ModelSignal:
        llm: Any | None = None
        acquired = False
        # Attempt accounting lives outside the try so the except branch can
        # carry whatever the failed run consumed (first-round review L3):
        # a raised retry attempt must not reset the pair's usage to None.
        retried = False
        prompt_tokens_total = 0
        generated_tokens_total = 0
        try:
            llm = self._acquire_llm_for_call()
            if llm is None:
                return ModelSignal(False, "backend_unavailable", None, "", None, "disabled")
            acquired = True
            # Grammar-free decoding since 0.15.14 (A2/L0): plain chat
            # completion WITHOUT response_format — the per-token grammar
            # evaluation halved decode throughput; the caps it used to enforce
            # are post-hoc now (L3 truncation + grounding). Seeding (the
            # matrix's L2) was rejected in integration — see the PAIR_PROMPT
            # note. max_tokens 384 covers the protocol worst case (~330
            # tokens); "</s>" was a dead stop: Qwen2.5's EOS (<|im_end|>)
            # comes from the chat template.
            max_tokens = 384
            cjk = evidence_is_cjk(left, right)
            # 0.16.8: Qwen3-style models need the /no_think soft switch in
            # the user turn (default thinking burns the whole 384 budget and
            # answers nothing) and must NOT stop on "\n\n" (the empty think
            # shell emits it at token two). Qwen2.5 keeps both legacy params
            # byte-for-byte — owner compatibility constraint.
            nothink = "/no_think\n" if self._qwen3_style else ""
            messages: list[dict[str, str]] = [
                {"role": "system", "content": _PAIR_PROMPT if cjk else _PAIR_PROMPT_EN},
                {"role": "user", "content": (
                    f"{nothink}输入: {self._pair_text(left, right)}\n输出:"
                    if cjk else
                    f"{nothink}Input: {self._pair_text(left, right)}\nOutput:"
                )},
            ]
            decode_stop = None if self._qwen3_style else ["\n\n"]
            # A6 queue gate: with other requests waiting, one attempt only —
            # an invalid output then fails fast instead of doubling the wait
            # of everything behind it.
            attempts = 1 if not retry_allowed else max(1, SEMANTIC_PAIR_MAX_ATTEMPTS)
            signal: ModelSignal | None = None
            for attempt in range(attempts):
                # Every attempt is grammar-free (see the PAIR_PROMPT note): the
                # retry differs only in its feedback turn.
                with self._infer_lock:
                    out = llm.create_chat_completion(
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=0.0,
                        top_p=0.9,
                        stop=decode_stop,
                    )
                usage = out.get("usage") or {}
                prompt_tokens_total += int(usage.get("prompt_tokens") or 0)
                generated_tokens_total += int(usage.get("completion_tokens") or 0)
                raw = str(out["choices"][0]["message"]["content"] or "")
                signal = model_signal_from_text(raw)
                # Behavioural family self-heal (0.16.8): a Qwen3-style model
                # loaded under legacy params emits exactly "<think" and dies
                # at the "\n\n" stop. Two CONSECUTIVE strikes (a single one
                # could be a legacy model echoing think-tag prose) flip this
                # backend to Qwen3 params and rerun the pair once; a clean
                # output resets the streak.
                if not self._qwen3_style:
                    if signal.parsed is None and raw.lstrip().startswith("<think"):
                        with self._cond:
                            self._think_strikes += 1
                            if self._think_strikes >= 2:
                                self._qwen3_style = True
                                self._family_autodetected += 1
                        if self._qwen3_style:
                            messages = [
                                messages[0],
                                {"role": "user", "content": f"/no_think\n{messages[1]['content']}"},
                            ]
                            decode_stop = None
                            with self._infer_lock:
                                out = llm.create_chat_completion(
                                    messages=messages,
                                    max_tokens=max_tokens,
                                    temperature=0.0,
                                    top_p=0.9,
                                )
                            usage = out.get("usage") or {}
                            prompt_tokens_total += int(usage.get("prompt_tokens") or 0)
                            generated_tokens_total += int(usage.get("completion_tokens") or 0)
                            raw = str(out["choices"][0]["message"]["content"] or "")
                            signal = model_signal_from_text(raw)
                    elif signal.parsed is not None:
                        with self._cond:
                            self._think_strikes = 0
                strategy = None
                if signal.candidate_type in {"invalid_json", "invalid_schema"}:
                    l3_signal = _l3_truncated_signal(raw)
                    if l3_signal is not None:
                        with self._cond:
                            self._pair_l3_truncated += 1
                        signal = l3_signal
                    else:
                        strategy = _pair_retry_strategy(signal.error)
                if strategy is None or attempt + 1 >= attempts:
                    if retried and signal.candidate_type == "attribute_value_extraction":
                        with self._cond:
                            self._pair_retry_recovered += 1
                    signal.prompt_tokens = prompt_tokens_total
                    signal.generated_tokens = generated_tokens_total
                    signal.retried = retried
                    return signal
                # One protocol invalid output earns a single retry: same input,
                # plus a feedback turn naming the specific violation (measured
                # 2026-09-11 — no echo: an echoed bad output keeps the 0.5B in
                # its copy state; naming the real offending keys beats the
                # generic four-field reminder). Truncation retries shrink the
                # quotes and widen the output budget (freed n_ctx funds it).
                feedback = _pair_retry_feedback(strategy, signal.error or "", raw)
                if strategy == "truncated":
                    retry_max_tokens = SEMANTIC_PAIR_RETRY_MAX_TOKENS
                    # Re-read the family flag: the self-heal above may have
                    # flipped it mid-call.
                    retry_nothink = "/no_think\n" if self._qwen3_style else ""
                    retry_messages = [
                        messages[0],
                        {"role": "user", "content": (
                            f"{retry_nothink}输入: {self._pair_text(left, right, quote_cap=SEMANTIC_PAIR_RETRY_QUOTE_CHARS)}\n输出:"
                            if cjk else
                            f"{retry_nothink}Input: {self._pair_text(left, right, quote_cap=SEMANTIC_PAIR_RETRY_QUOTE_CHARS)}\nOutput:"
                        )},
                        {"role": "user", "content": feedback},
                    ]
                else:
                    retry_max_tokens = max_tokens
                    retry_messages = [
                        messages[0],
                        {"role": "user", "content": messages[1]["content"]},
                        {"role": "user", "content": feedback},
                    ]
                # n_ctx guard (dense CJK runs ~1 token/char, so the echoed raw
                # can push the retry past the window — llama-cpp-python then
                # raises ValueError and a clean qwen_invalid_output turns into
                # a backend_error). estimate_tokens under-reads dense CJK by
                # ~30%, hence the 1.3x margin plus chat-template headroom.
                prompt_estimate = int(
                    estimate_tokens("".join(message["content"] for message in retry_messages)) * 1.3
                ) + 64
                if prompt_estimate + retry_max_tokens >= self.n_ctx:
                    signal.prompt_tokens = prompt_tokens_total
                    signal.generated_tokens = generated_tokens_total
                    signal.retried = retried
                    return signal
                retried = True
                with self._cond:
                    self._pair_retried += 1
                max_tokens = retry_max_tokens
                messages = retry_messages
            # Unreachable (the loop always returns); the loop runs at least
            # once because attempts is clamped to >= 1.
            assert signal is not None
            return signal
        except Exception as exc:
            with self._cond:
                self._last_error = str(exc)
            return ModelSignal(
                False, "backend_error", None, "", None, str(exc),
                prompt_tokens=prompt_tokens_total or None,
                generated_tokens=generated_tokens_total or None,
                retried=retried,
            )
        finally:
            if acquired:
                self._release_llm_for_call()

    def unload(self, timeout: float = 30.0, disable: bool = False) -> dict[str, Any]:
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._cond:
            if disable:
                self._disabled = True
            self._unloading = True
            while self._inflight > 0 or self._loading:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._unloading = False
                    self._cond.notify_all()
                    return {
                        "ok": False,
                        "unloaded": False,
                        "timeout": True,
                        "inflight": self._inflight,
                        "loading": self._loading,
                        "retry_hint": "retry unload after current semantic inference or load completes",
                        "generation": self._generation,
                    }
                self._cond.wait(remaining)
            self._llm = None
            self._loaded_at = None
            self._generation += 1
            self._unloading = False
            self._cond.notify_all()
            return {
                "ok": True,
                "unloaded": True,
                "timeout": False,
                "inflight": 0,
                "retry_hint": None,
                "generation": self._generation,
            }

    def set_disabled(self, disabled: bool) -> None:
        with self._cond:
            self._disabled = bool(disabled)
            self._cond.notify_all()

    def suggest_workspace_candidate(
        self,
        ws_raw: str,
        evidence: dict[str, Any],
        candidates: list[str],
    ) -> "WorkspaceCandidateSignal":
        """Suggest whether ws_raw should normalize to one of `candidates`.

        A *suggester*, never the arbiter (636 §6): the caller decides how to use
        this given isolation + rule vetoes. Degrades to a safe uncertain/no-op on
        any backend error — never raises.
        """
        llm: Any | None = None
        acquired = False
        try:
            llm = self._acquire_llm_for_call()
            if llm is None:
                return WorkspaceCandidateSignal(None, "uncertain", None, "", "", "disabled")
            acquired = True
            title = str(evidence.get("title") or "")
            keys = " / ".join(evidence.get("key_sentences") or [])[:300]
            cand_str = ", ".join(candidates) if candidates else "(无)"
            text = (
                f"workspace原文: {ws_raw}\n标题: {title}\n关键句: {keys}\n候选: {cand_str}"
            )
            with self._infer_lock:
                # max_tokens headroom as in classify_pair (truncated JSON was
                # the top qwen_invalid_output source); "</s>" dropped — Qwen2.5
                # EOS (<|im_end|>) comes from the chat template.
                out = llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": _WORKSPACE_PROMPT},
                        {"role": "user", "content": f"输入:\n{text}\n输出:"},
                    ],
                    max_tokens=384,
                    temperature=0.0,
                    top_p=0.9,
                    stop=["\n\n"],
                    response_format=_WORKSPACE_RESPONSE_FORMAT,
                )
            raw = out["choices"][0]["message"]["content"]
            return workspace_candidate_from_text(raw, candidates)
        except Exception as exc:
            with self._cond:
                self._last_error = str(exc)
            return WorkspaceCandidateSignal(None, "uncertain", None, "", "", str(exc))
        finally:
            if acquired:
                self._release_llm_for_call()

    def pair_retry_stats(self) -> dict[str, Any]:
        """Retry/L3/fallback counters for the parent to piggyback on responses."""
        with self._cond:
            return {
                "pair_retried": self._pair_retried,
                "pair_retry_recovered": self._pair_retry_recovered,
                "pair_l3_truncated": self._pair_l3_truncated,
                "gpu_fallback": self._gpu_fallback,
                "model_family": "qwen3" if self._qwen3_style else "legacy",
                "family_autodetected": self._family_autodetected,
            }

    def status(self) -> dict[str, Any]:
        with self._cond:
            state = "resident" if self._llm is not None else "unloaded"
            return {
                "backend": "local_gguf",
                "model_path": str(self.model_path),
                "model_exists": self.model_path.exists(),
                "model_state": state,
                "loaded_at": self._loaded_at,
                "last_error": self._last_error,
                "inflight": self._inflight,
                "unloading": self._unloading,
                "disabled": self._disabled,
                "generation": self._generation,
                "n_ctx": self.n_ctx,
                "n_gpu_layers": self.n_gpu_layers,
                "gpu_fallback": self._gpu_fallback,
                "prompt_version": PAIR_PROMPT_VERSION,
                "pair_retried": self._pair_retried,
                "pair_retry_recovered": self._pair_retry_recovered,
                "pair_l3_truncated": self._pair_l3_truncated,
                "model_family": "qwen3" if self._qwen3_style else "legacy",
                "family_autodetected": self._family_autodetected,
            }


def _semantic_inference_process(conn: Any, config: dict[str, Any]) -> None:
    """Child entry point. It owns only llama.cpp state, never MemoryDB state."""
    backend = LocalGGUFSemanticBackend(
        Path(config["model_path"]),
        n_ctx=int(config["n_ctx"]),
        n_threads=int(config["n_threads"]),
        n_batch=int(config["n_batch"]),
        n_gpu_layers=int(config.get("n_gpu_layers", 0)),
    )
    try:
        while True:
            request = conn.recv()
            command = request.get("command")
            if command == "shutdown":
                return
            try:
                if command == "load":
                    backend.load()
                    result: Any = {"loaded": True}
                elif command == "classify_pair":
                    result = backend.classify_pair(
                        request["left"], request["right"],
                        retry_allowed=bool(request.get("retry_allowed", True)),
                    )
                    conn.send({
                        "ok": True,
                        "result": result,
                        # Piggyback retry/L3 counters so the parent's status()
                        # exposes them without a separate (potentially
                        # blocking) status RPC. Envelope keys are additive.
                        "backend_status": backend.pair_retry_stats(),
                    })
                    continue
                elif command == "suggest_workspace_candidate":
                    result = backend.suggest_workspace_candidate(
                        request["workspace"], request["evidence"], request["candidates"],
                    )
                else:
                    raise ValueError(f"unknown semantic child command: {command}")
                conn.send({"ok": True, "result": result})
            except BaseException as exc:
                conn.send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    except (EOFError, BrokenPipeError, OSError):
        return
    finally:
        # A3 (0.15.14): llama-cpp-python 0.3.34 loads Metal buffers that
        # SIGABRT the process inside __cxa_finalize on interpreter exit unless
        # the model is explicitly freed first (verified mitigation: unload +
        # del + gc — see the 0.15.13.1 teardown rule and llama-cpp-metal-exit
        # crash notes). SIGTERM termination skips destructors and is immune.
        try:
            backend.unload()
        except Exception:
            pass
        del backend
        gc.collect()
        conn.close()


class IsolatedGGUFSemanticBackend:
    """Single-flight process supervisor with a real inference hard timeout."""

    def __init__(
        self,
        model_path: Path,
        *,
        n_ctx: int = SEMANTIC_N_CTX,
        n_threads: int = 4,
        n_batch: int = 128,
        n_gpu_layers: int = 0,
        hard_timeout_ms: int = 30_000,
        load_timeout_ms: int = 120_000,
        process_target: Any = _semantic_inference_process,
    ) -> None:
        self.model_path = Path(model_path).expanduser()
        self.n_ctx = int(n_ctx)
        self.n_threads = int(n_threads)
        self.n_batch = int(n_batch)
        self.n_gpu_layers = int(n_gpu_layers)
        self.hard_timeout_ms = max(1, int(hard_timeout_ms))
        self.load_timeout_ms = max(1, int(load_timeout_ms))
        self._process_target = process_target
        self._ctx = multiprocessing.get_context("spawn")
        self._request_lock = threading.Lock()
        self._schedule_cond = threading.Condition()
        self._schedule_queues: dict[str, deque[tuple[object, float | None]]] = {
            "notice": deque(), "workspace": deque(),
        }
        self._schedule_active = False
        self._schedule_last_class = "workspace"
        self._schedule_max_pending = 64
        self._state_lock = threading.RLock()
        self._process: Any = None
        self._conn: Any = None
        self._disabled = False
        self._generation = 0
        self._restarts = 0
        self._timed_out = 0
        self._last_error: str | None = None
        self._loaded_at: float | None = None
        self._inflight_started: float | None = None
        self._child_loaded = False
        self._last_child_backend_status: dict[str, Any] | None = None

    def _start_locked(self) -> None:
        if self._disabled:
            raise RuntimeError("semantic backend disabled")
        if self._process is not None and self._process.is_alive():
            return
        if self._process is not None:
            self._terminate_locked(count_restart=True)
        parent, child = self._ctx.Pipe(duplex=True)
        process = self._ctx.Process(
            target=self._process_target,
            args=(child, {
                "model_path": str(self.model_path),
                "n_ctx": self.n_ctx,
                "n_threads": self.n_threads,
                "n_batch": self.n_batch,
                "n_gpu_layers": self.n_gpu_layers,
            }),
            name="memory-arbiter-semantic-inference",
            daemon=True,
        )
        try:
            process.start()
        except BaseException:
            parent.close()
            child.close()
            try:
                process.close()
            except (AttributeError, ValueError):
                pass
            raise
        child.close()
        self._process = process
        self._conn = parent
        self._generation += 1
        self._child_loaded = False

    def _terminate_locked(self, *, count_restart: bool) -> None:
        process, conn = self._process, self._conn
        self._process = None
        self._conn = None
        self._loaded_at = None
        self._child_loaded = False
        if conn is not None:
            try:
                conn.close()
            except OSError:
                pass
        if process is not None:
            if process.is_alive():
                process.terminate()
            process.join(timeout=2.0)
            if process.is_alive():  # pragma: no cover - defensive OS fallback
                process.kill()
                process.join(timeout=1.0)
            try:
                process.close()
            except (AttributeError, ValueError):
                pass
            if count_restart:
                self._restarts += 1

    def _admit_request(self, request_class: str, deadline_monotonic: float | None) -> object | None:
        now = time.monotonic()
        if deadline_monotonic is not None and deadline_monotonic <= now:
            return None
        token = object()
        cls = "workspace" if request_class == "workspace" else "notice"
        with self._schedule_cond:
            total_pending = sum(len(queue) for queue in self._schedule_queues.values())
            if total_pending >= self._schedule_max_pending:
                return None
            self._schedule_queues[cls].append((token, deadline_monotonic))
            while True:
                if deadline_monotonic is not None and deadline_monotonic <= time.monotonic():
                    try:
                        self._schedule_queues[cls].remove((token, deadline_monotonic))
                    except ValueError:
                        pass
                    self._schedule_cond.notify_all()
                    return None
                if not self._schedule_active and self._select_next_request_locked() is token:
                    self._schedule_active = True
                    self._schedule_last_class = cls
                    self._schedule_queues[cls].popleft()
                    self._schedule_cond.notify_all()
                    return token
                if deadline_monotonic is None:
                    self._schedule_cond.wait()
                else:
                    remaining = max(0.0, deadline_monotonic - time.monotonic())
                    if remaining <= 0:
                        continue
                    self._schedule_cond.wait(remaining)

    def _select_next_request_locked(self) -> object | None:
        now = time.monotonic()
        for queue in self._schedule_queues.values():
            while queue and queue[0][1] is not None and queue[0][1] <= now:
                queue.popleft()
        notice = self._schedule_queues["notice"]
        workspace = self._schedule_queues["workspace"]
        if notice and workspace:
            next_class = "workspace" if self._schedule_last_class == "notice" else "notice"
            return self._schedule_queues[next_class][0][0]
        if notice:
            return notice[0][0]
        if workspace:
            return workspace[0][0]
        return None

    def _release_admission(self) -> None:
        with self._schedule_cond:
            self._schedule_active = False
            self._schedule_cond.notify_all()

    def _request(
        self, command: str, *, request_class: str = "notice",
        deadline_monotonic: float | None = None, **payload: Any,
    ) -> Any:
        # Fast rejection matters for synchronous workspace suggestions while a
        # prior inference owns the single-flight lock. Re-check after acquiring
        # it to close the race with disable.
        with self._state_lock:
            if self._disabled:
                raise RuntimeError("semantic backend disabled")
        admission = self._admit_request(request_class, deadline_monotonic)
        if admission is None:
            raise TimeoutError("semantic request admission deadline expired")
        try:
            acquired = self._request_lock.acquire(timeout=0)
            if not acquired:  # pragma: no cover - scheduler invariant guard
                raise RuntimeError("semantic scheduler admitted concurrent request")
            # Strict max_concurrency=1, including load.
            with self._state_lock:
                self._start_locked()
                conn = self._conn
                process = self._process
                self._inflight_started = time.monotonic()
            try:
                if command != "load" and not self._child_loaded:
                    load_timeout = self._remaining_timeout_ms(deadline_monotonic, self.load_timeout_ms)
                    self._exchange(conn, "load", load_timeout)
                    self._child_loaded = True
                    self._loaded_at = time.time()
                request_timeout = self._remaining_timeout_ms(
                    deadline_monotonic,
                    self.load_timeout_ms if command == "load" else self.hard_timeout_ms,
                )
                result = self._exchange(conn, command, request_timeout, **payload)
                if command == "load":
                    self._child_loaded = True
                    self._loaded_at = time.time()
                self._last_error = None
                return result
            except (EOFError, BrokenPipeError, OSError) as exc:
                with self._state_lock:
                    self._last_error = f"semantic child exited: {exc}"
                    self._terminate_locked(count_restart=True)
                raise RuntimeError(self._last_error) from exc
            finally:
                with self._state_lock:
                    self._inflight_started = None
                    if process is not None and self._process is process and not process.is_alive():
                        self._terminate_locked(count_restart=True)
        finally:
            if 'acquired' in locals() and acquired:
                self._request_lock.release()
            self._release_admission()

    @staticmethod
    def _remaining_timeout_ms(deadline_monotonic: float | None, configured_ms: int) -> int:
        if deadline_monotonic is None:
            return configured_ms
        remaining_ms = int((deadline_monotonic - time.monotonic()) * 1000)
        if remaining_ms <= 0:
            raise TimeoutError("semantic request deadline expired before inference")
        return max(1, min(configured_ms, remaining_ms))

    def _exchange(self, conn: Any, command: str, timeout_ms: int, **payload: Any) -> Any:
        try:
                conn.send({"command": command, **payload})
                timeout = max(0.001, timeout_ms / 1000.0)
                deadline = time.monotonic() + timeout
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        with self._state_lock:
                            self._timed_out += 1
                            phase = "load" if command == "load" else "inference"
                            self._last_error = f"semantic {phase} hard timeout after {timeout_ms}ms"
                            if self._conn is conn:
                                self._terminate_locked(count_restart=True)
                        raise TimeoutError(self._last_error)
                    if conn.poll(min(remaining, 0.05)):
                        break
                    with self._state_lock:
                        if self._conn is not conn:
                            raise EOFError("semantic child connection was closed")
                response = conn.recv()
                if not response.get("ok"):
                    self._last_error = str(response.get("error") or "semantic child error")
                    raise RuntimeError(self._last_error)
                child_status = response.get("backend_status")
                if isinstance(child_status, dict):
                    # classify_pair piggybacks retry counters; older children
                    # simply omit the key and the parent stays at the default.
                    with self._state_lock:
                        self._last_child_backend_status = child_status
                return response.get("result")
        except TimeoutError:
            raise

    def load(self) -> None:
        self._request("load")

    def _effective_retry_allowed(self, retry_allowed: bool) -> bool:
        """A6 queue gate, scheduler half: when other requests already wait on
        this single-flight scheduler, the child runs one attempt only. The
        caller's retry_allowed carries the semantic-worker job queue — the
        other half of the same gate."""
        with self._schedule_cond:
            return retry_allowed and not any(
                len(queue) for queue in self._schedule_queues.values()
            )

    def classify_pair(
        self, left: dict[str, Any], right: dict[str, Any],
        *,
        deadline_monotonic: float | None = None,
        retry_allowed: bool = True,
    ) -> ModelSignal:
        try:
            result = self._request(
                "classify_pair", left=left, right=right,
                deadline_monotonic=deadline_monotonic,
                retry_allowed=self._effective_retry_allowed(retry_allowed),
            )
            return result if isinstance(result, ModelSignal) else ModelSignal(False, "backend_error", None, "", None, "invalid child response")
        except Exception as exc:
            return ModelSignal(False, "backend_error", None, "", None, str(exc))

    def suggest_workspace_candidate(
        self, ws_raw: str, evidence: dict[str, Any], candidates: list[str],
        *, deadline_monotonic: float | None = None,
    ) -> WorkspaceCandidateSignal:
        try:
            result = self._request(
                "suggest_workspace_candidate", request_class="workspace",
                deadline_monotonic=deadline_monotonic,
                workspace=ws_raw, evidence=evidence, candidates=candidates,
            )
            return result if isinstance(result, WorkspaceCandidateSignal) else WorkspaceCandidateSignal(None, "uncertain", None, "", "", "invalid child response")
        except Exception as exc:
            return WorkspaceCandidateSignal(None, "uncertain", None, "", "", str(exc))

    def unload(self, timeout: float = 30.0, disable: bool = False) -> dict[str, Any]:
        if disable:
            # Admission closes before waiting on the single-flight lock. A timed
            # out unload must still leave queued/new callers unable to start.
            with self._state_lock:
                self._disabled = True
        acquired = self._request_lock.acquire(timeout=max(0.0, float(timeout)))
        if not acquired:
            return {"ok": False, "unloaded": False, "timeout": True, "inflight": 1, "retry_hint": "retry after inference completes", "generation": self._generation}
        try:
            with self._state_lock:
                self._terminate_locked(count_restart=False)
                return {"ok": True, "unloaded": True, "timeout": False, "inflight": 0, "retry_hint": None, "generation": self._generation}
        finally:
            self._request_lock.release()

    def set_disabled(self, disabled: bool) -> None:
        with self._state_lock:
            self._disabled = bool(disabled)

    def force_terminate(self) -> dict[str, Any]:
        """Process-exit cleanup; may interrupt the sole in-flight request."""
        with self._state_lock:
            self._disabled = True
            self._terminate_locked(count_restart=False)
            return {"ok": True, "unloaded": True, "forced": True, "generation": self._generation}

    def status(self) -> dict[str, Any]:
        with self._state_lock:
            process = self._process
            age_ms = None
            if self._inflight_started is not None:
                age_ms = int((time.monotonic() - self._inflight_started) * 1000)
            child_status = dict(self._last_child_backend_status or {})
            return {
                "backend": "local_gguf_process",
                "model_path": str(self.model_path),
                "model_exists": self.model_path.exists(),
                "model_state": "resident" if process is not None and process.is_alive() else "unloaded",
                "loaded_at": self._loaded_at,
                "last_error": self._last_error,
                "inflight": 1 if self._inflight_started is not None else 0,
                "inflight_age_ms": age_ms,
                "disabled": self._disabled,
                "generation": self._generation,
                "child_pid": process.pid if process is not None and process.is_alive() else None,
                "child_restarts": self._restarts,
                "timed_out_jobs": self._timed_out,
                "max_concurrency": 1,
                # Observability for the 0.15.8 inference-window/prompt fixes:
                # lets an operator confirm a restarted host really runs the
                # widened context and the new prompt without guessing.
                "n_ctx": self.n_ctx,
                "n_gpu_layers": self.n_gpu_layers,
                "prompt_version": PAIR_PROMPT_VERSION,
                # Retry/L3 counters piggybacked from the child's classify_pair
                # responses; absent until the first pair completes.
                "pair_retried": child_status.get("pair_retried"),
                "pair_retry_recovered": child_status.get("pair_retry_recovered"),
                "pair_l3_truncated": child_status.get("pair_l3_truncated"),
                # Decode-family routing (0.16.8): which parameter set the child
                # actually runs, and whether it had to self-heal from legacy
                # params (metadata probe missed).
                "model_family": child_status.get("model_family"),
                "family_autodetected": child_status.get("family_autodetected"),
                # True once the child's GPU offload failed and it fell back to
                # CPU (A3 self-healing; visible only after a classify_pair).
                "gpu_fallback": child_status.get("gpu_fallback"),
            }


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
