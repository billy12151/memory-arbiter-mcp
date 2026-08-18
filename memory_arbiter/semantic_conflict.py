"""Local Qwen protocol parsing and isolated GGUF process supervision.

The model filters only short evidence pairs routed as ``check``. Deterministic
``notify`` decisions bypass it, and the Agent remains the terminal reviewer.
"""
from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol

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

_PAIR_PROMPT = """你是 mema 的候选降噪器，只输出 JSON，不要解释。
字段：label(conflict|not_conflict|uncertain), same_fact_slot(boolean), confidence(number 0..1)。

核心规则：默认 not_conflict。只有两句在同一时间和范围内回答同一个具体问题，且答案不能同时成立，才是 conflict。
同主题不等于同事实槽位；一个说技术选型、另一个说连接池、参数或能力，是 not_conflict。
同义词、重复、兼容补充、不同环境或范围也是 not_conflict。
旧口径明确替换新口径、采用与不采用、公开与不公开、同一数值不同、todo 与已完成，是 conflict。
确实同槽位但证据不足才用 uncertain。

例1 A=数据库使用 PostgreSQL。 B=PostgreSQL 连接池上限为20。 -> {"label":"not_conflict","same_fact_slot":false,"confidence":0.95}
例2 A=系统支持 memory find。 B=系统支持 memory read。 -> {"label":"not_conflict","same_fact_slot":false,"confidence":0.9}
例3 A=数据库使用 pgsql。 B=数据库使用 PostgreSQL。 -> {"label":"not_conflict","same_fact_slot":true,"confidence":0.98}
例4 A=旧方案采用 Redis。 B=新方案不再采用 Redis，改用 SQLite。 -> {"label":"conflict","same_fact_slot":true,"confidence":0.95}
例5 A=TODO 完成数据库迁移。 B=数据库迁移已经完成。 -> {"label":"conflict","same_fact_slot":true,"confidence":0.95}"""


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
    candidate: Optional[str]
    relation: str
    confidence: Optional[float]
    evidence: str
    raw: str = ""
    error: Optional[str] = None


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
    confidence: Optional[float]
    raw: str
    parsed: dict[str, Any] | None
    error: Optional[str] = None


@dataclass
class EvidenceDecision:
    action: str
    reason: str
    anchors: list[str] = field(default_factory=list)
    left_value: Optional[str] = None
    right_value: Optional[str] = None


class SemanticBackend(Protocol):
    def classify_pair(self, left: dict[str, Any], right: dict[str, Any]) -> ModelSignal:
        ...

    def suggest_workspace_candidate(
        self,
        ws_raw: str,
        evidence: dict[str, Any],
        candidates: list[str],
    ) -> WorkspaceCandidateSignal:
        ...

    def load(self) -> None:
        ...

    def status(self) -> dict[str, Any]:
        ...

    def unload(self, timeout: float = 30.0, disable: bool = False) -> dict[str, Any]:
        ...

    def maybe_unload_if_idle(self) -> dict[str, Any]:
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


_EVIDENCE_ALIASES = {
    "pgsql": "postgresql",
    "postgres": "postgresql",
    "sqlite_vec": "sqlite-vec",
}
_EXPLICIT_SCOPES = (
    ("测试环境", "生产环境"),
    ("移动端", "管理后台"),
    ("公开api", "内部api"),
    ("中国区", "海外区"),
)
_VALUE_RE = re.compile(
    r"(?<![\w.])v?\d+(?:\.\d+){0,2}\s*(?:ms|s|秒|分钟|小时|天|%|mb|gb|kb|条|次|核|g)?",
    re.IGNORECASE,
)


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
        values.append(value)
    return values


def decide_evidence(left_text: str, right_text: str) -> EvidenceDecision:
    """Classify a short pair using only narrow, explainable evidence."""
    left_norm = _normalize_evidence_text(left_text)
    right_norm = _normalize_evidence_text(right_text)
    if left_norm and left_norm == right_norm:
        return EvidenceDecision("ignore", "equivalent_value")

    left_lower = (left_text or "").casefold()
    right_lower = (right_text or "").casefold()
    for left_scope, right_scope in _EXPLICIT_SCOPES:
        if (
            (left_scope in left_lower and right_scope in right_lower)
            or (right_scope in left_lower and left_scope in right_lower)
        ):
            return EvidenceDecision("ignore", "explicit_scope_mismatch")

    left_values = _normalized_values(left_text)
    right_values = _normalized_values(right_text)
    left_skeleton = _VALUE_RE.sub("", left_lower)
    right_skeleton = _VALUE_RE.sub("", right_lower)
    common = sorted(_tokens(left_skeleton) & _tokens(right_skeleton))
    char_similarity = _cosine(_char_ngrams(left_skeleton), _char_ngrams(right_skeleton))
    if left_values and right_values and left_values != right_values and (
        common or char_similarity >= 0.45
    ):
        return EvidenceDecision(
            "notify", "numeric_value_changed", common,
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
    # accept both orientations. The common-token guard below (applied in
    # The common-token guard prevents unrelated todo/done statements pairing.
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
        values = set(re.findall(r"\d+(?:\.\d+)?\s*(?:mb|gb|kb|%|ms|s|秒|核|g)?", text.lower()))
        normalized = set()
        for value in values:
            normalized.add(re.sub(r"\s+", "", value).replace("gb", "g"))
        return normalized

    # Same concrete values + high lexical overlap is a duplicate, not a conflict.
    if numeric_values(left_text) and numeric_values(left_text) == numeric_values(right_text) and char_cosine >= 0.45:
        duplicate_guard = True

    # Alias/name statements that share the same concrete alias tokens are duplicates.
    if ("mema" in lower and "迷码" in lower) and ("alias" in lower or "cli" in lower or "命令" in lower or "中文名" in lower):
        duplicate_guard = True

    # A negated excluded alternative plus a shared positive value can be compatible
    # support, but never suppress explicit replacement/old-new wording.
    explicit_replacement = any(term in joined for term in ["以后以", "替换", "改为", "不再采用", "之前不对", "旧设计", "新设计", "新口径", "旧口径", "下线", "而不是", "不公开", "公开"])
    if polarity_diff and not explicit_replacement and len(common) >= 2 and only_right and not contains_diff:
        compatible_guard = True
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


def _extract_first_json_object(raw: str) -> Optional[str]:
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


def model_signal_from_text(raw: str) -> ModelSignal:
    parsed: dict[str, Any] | None = None
    snippet = _extract_first_json_object(raw or "")
    if not snippet:
        return ModelSignal(False, "invalid_json", None, raw or "", None, "missing_json")
    protocol_fields = {"label", "same_fact_slot", "confidence"}

    def reject_duplicate_protocol_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        decoded_object: dict[str, Any] = {}
        seen_protocol_fields: set[str] = set()
        for key, value in pairs:
            if key in protocol_fields:
                if key in seen_protocol_fields:
                    raise ValueError(f"duplicate protocol field: {key}")
                seen_protocol_fields.add(key)
            decoded_object[key] = value
        return decoded_object

    try:
        decoded = json.loads(snippet, object_pairs_hook=reject_duplicate_protocol_fields)
    except Exception as exc:
        return ModelSignal(False, "invalid_json", None, raw or "", None, str(exc))
    if not isinstance(decoded, dict):
        return ModelSignal(False, "invalid_schema", None, raw or "", None, "invalid_schema")
    parsed = decoded

    required = {"label", "same_fact_slot", "confidence"}
    label = parsed.get("label")
    same_slot = parsed.get("same_fact_slot")
    confidence = parsed.get("confidence")
    schema_valid = (
        required.issubset(parsed)
        and type(same_slot) is bool
        and isinstance(label, str)
        and label in {"conflict", "not_conflict", "uncertain"}
        and isinstance(confidence, (int, float))
        and not isinstance(confidence, bool)
        and math.isfinite(confidence)
        and 0.0 <= confidence <= 1.0
    )
    if schema_valid and label in {"conflict", "uncertain"}:
        schema_valid = same_slot is True
    if not schema_valid:
        return ModelSignal(False, "invalid_schema", None, raw or "", parsed, "invalid_schema")
    assert isinstance(label, str)
    assert isinstance(confidence, (int, float)) and not isinstance(confidence, bool)
    return ModelSignal(label in {"conflict", "uncertain"}, label, float(confidence), raw or "", parsed)


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


class LocalGGUFSemanticBackend:
    def __init__(self, model_path: Path, *, n_ctx: int = 1024, n_threads: int = 4, n_batch: int = 128):
        self.model_path = Path(model_path).expanduser()
        self.n_ctx = int(n_ctx)
        self.n_threads = int(n_threads)
        self.n_batch = int(n_batch)
        self._llm: Any = None
        self._cond = threading.Condition(threading.Lock())
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._last_error: Optional[str] = None
        self._loaded_at: Optional[float] = None
        self._inflight = 0
        self._unloading = False
        self._loading = False
        self._disabled = False
        self._generation = 0

    def _build_llm(self) -> Any:
        if not self.model_path.exists():
            raise FileNotFoundError(str(self.model_path))
        from llama_cpp import Llama  # type: ignore
        kwargs = {
            "model_path": str(self.model_path),
            "n_ctx": self.n_ctx,
            "n_threads": self.n_threads,
            "verbose": False,
        }
        if self.n_batch > 0:
            kwargs["n_batch"] = self.n_batch
        return Llama(**kwargs)

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
        content = record.get("content") or ""
        return f"subject: {subject}\ntags: {tags}\ncontent: {content}".strip()

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

    def classify_pair(self, left: dict[str, Any], right: dict[str, Any]) -> ModelSignal:
        llm: Any | None = None
        acquired = False
        try:
            llm = self._acquire_llm_for_call()
            if llm is None:
                return ModelSignal(False, "backend_unavailable", None, "", None, "disabled")
            acquired = True
            text = f"A: {self._memory_text(left)}\nB: {self._memory_text(right)}"
            with self._infer_lock:
                out = llm.create_chat_completion(
                    messages=[{"role": "system", "content": _PAIR_PROMPT}, {"role": "user", "content": f"输入: {text}\n输出:"}],
                    max_tokens=120,
                    temperature=0.0,
                    top_p=0.9,
                    stop=["\n\n", "</s>"],
                )
            raw = out["choices"][0]["message"]["content"]
            return model_signal_from_text(raw)
        except Exception as exc:
            with self._cond:
                self._last_error = str(exc)
            return ModelSignal(False, "backend_error", None, "", None, str(exc))
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

    def maybe_unload_if_idle(self) -> dict[str, Any]:
        with self._cond:
            if self._inflight > 0 or self._unloading or self._loading:
                return {
                    "ok": False,
                    "unloaded": False,
                    "reason": "busy",
                    "inflight": self._inflight,
                    "generation": self._generation,
                }
            was_loaded = self._llm is not None
            if was_loaded:
                self._llm = None
                self._loaded_at = None
                self._generation += 1
            return {
                "ok": True,
                "unloaded": was_loaded,
                "reason": "idle" if was_loaded else "already_unloaded",
                "inflight": 0,
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
                out = llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": _WORKSPACE_PROMPT},
                        {"role": "user", "content": f"输入:\n{text}\n输出:"},
                    ],
                    max_tokens=120,
                    temperature=0.0,
                    top_p=0.9,
                    stop=["\n\n", "</s>"],
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
            }


def _semantic_inference_process(conn: Any, config: dict[str, Any]) -> None:
    """Child entry point. It owns only llama.cpp state, never MemoryDB state."""
    backend = LocalGGUFSemanticBackend(
        Path(config["model_path"]),
        n_ctx=int(config["n_ctx"]),
        n_threads=int(config["n_threads"]),
        n_batch=int(config["n_batch"]),
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
                    result = backend.classify_pair(request["left"], request["right"])
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
        conn.close()


class IsolatedGGUFSemanticBackend:
    """Single-flight process supervisor with a real inference hard timeout."""

    def __init__(
        self,
        model_path: Path,
        *,
        n_ctx: int = 1024,
        n_threads: int = 4,
        n_batch: int = 128,
        hard_timeout_ms: int = 30_000,
        load_timeout_ms: int = 120_000,
        process_target: Any = _semantic_inference_process,
    ) -> None:
        self.model_path = Path(model_path).expanduser()
        self.n_ctx = int(n_ctx)
        self.n_threads = int(n_threads)
        self.n_batch = int(n_batch)
        self.hard_timeout_ms = max(1, int(hard_timeout_ms))
        self.load_timeout_ms = max(1, int(load_timeout_ms))
        self._process_target = process_target
        self._ctx = multiprocessing.get_context("spawn")
        self._request_lock = threading.Lock()
        self._state_lock = threading.RLock()
        self._process: Any = None
        self._conn: Any = None
        self._disabled = False
        self._generation = 0
        self._restarts = 0
        self._timed_out = 0
        self._last_error: Optional[str] = None
        self._loaded_at: Optional[float] = None
        self._inflight_started: Optional[float] = None
        self._child_loaded = False

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

    def _request(self, command: str, **payload: Any) -> Any:
        # Fast rejection matters for synchronous workspace suggestions while a
        # prior inference owns the single-flight lock. Re-check after acquiring
        # it to close the race with disable.
        with self._state_lock:
            if self._disabled:
                raise RuntimeError("semantic backend disabled")
        with self._request_lock:  # Strict max_concurrency=1, including load.
            with self._state_lock:
                self._start_locked()
                conn = self._conn
                process = self._process
                self._inflight_started = time.monotonic()
            try:
                if command != "load" and not self._child_loaded:
                    self._exchange(conn, "load", self.load_timeout_ms)
                    self._child_loaded = True
                    self._loaded_at = time.time()
                result = self._exchange(
                    conn, command,
                    self.load_timeout_ms if command == "load" else self.hard_timeout_ms,
                    **payload,
                )
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
                return response.get("result")
        except TimeoutError:
            raise

    def load(self) -> None:
        self._request("load")

    def classify_pair(self, left: dict[str, Any], right: dict[str, Any]) -> ModelSignal:
        try:
            result = self._request("classify_pair", left=left, right=right)
            return result if isinstance(result, ModelSignal) else ModelSignal(False, "backend_error", None, "", None, "invalid child response")
        except Exception as exc:
            return ModelSignal(False, "backend_error", None, "", None, str(exc))

    def suggest_workspace_candidate(
        self, ws_raw: str, evidence: dict[str, Any], candidates: list[str],
    ) -> WorkspaceCandidateSignal:
        try:
            result = self._request(
                "suggest_workspace_candidate",
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

    def maybe_unload_if_idle(self) -> dict[str, Any]:
        if not self._request_lock.acquire(blocking=False):
            return {"ok": False, "unloaded": False, "reason": "busy", "inflight": 1, "generation": self._generation}
        try:
            with self._state_lock:
                was_loaded = self._process is not None
                self._terminate_locked(count_restart=False)
                return {"ok": True, "unloaded": was_loaded, "reason": "idle" if was_loaded else "already_unloaded", "inflight": 0, "generation": self._generation}
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
