"""evidence 判定批/预算/视图层（从 evidence.py 搬出，拆分批 ⑤ 纯移动）。

_JudgeBatch/_judge_outcome/_judge_pair_compat/_pair_diff_anchor/_JudgePairView/
_JobJudgeBudget；evidence.py re-export 保活（test_write_time_prefilter 直取
_judge_pair_compat、test_judge_budget_pool 直取 _JobJudgeBudget）。
"""
from __future__ import annotations

import hashlib
import hashlib
from typing import Any, TYPE_CHECKING

from ..constants import SEMANTIC_MAX_EXAMINED_PAIRS
from ..semantic_judge import PairVerdict

if TYPE_CHECKING:
    pass


class _JudgeBatch:
    """0.17.1 (owner 拍板: 攒批进本版): collect judge inputs through a phase's
    gates first (closure / version drift / budget / deadline — the pre-judge
    funnel is unchanged), then fire ONE batched judge_pairs call and drain
    the verdicts. Chunks are fixed-size SEMANTIC_MDEBERTA_BATCH slices; the
    deadline is enforced per pair at collection time (pass 1), not between
    chunks.

    The queue never crosses jobs — each job builds its own instance (R2-3
    no-cross-job-state rule)."""

    def __init__(self) -> None:
        self._items: list[dict[str, Any]] = []

    def add(self, **item: Any) -> None:
        self._items.append(item)

    def __len__(self) -> int:
        return len(self._items)

    def drain(self, judge_fn: "Any", *, stop_probe: "Any = None") -> list[dict[str, Any]]:
        """judge_fn(list[(a, b)]) -> list[PairVerdict]; returns the queued
        items with ``verdict`` attached, in queue order. 逐片契约（R1 实施
        review P2-2）：每片返回数与该片对数逐一校验，违约片连同其后的全部
        item 补 error verdict——此前各片的结果保留，zip 永不跨片错位。
        ``stop_probe``（0.17.1 重标合一）：片间让路探针，返回 True 即停发
        余片——余 item 补 error verdict（忙时公平性，对齐 INTEGRATION 的
        worker-yield 宣称）；已发片结果照常回填。"""
        out: list[dict[str, Any]] = list(self._items)
        self._items = []
        if not out:
            return out
        pairs = [(item.get("judge_text_a", item["text_a"]),
                  item.get("judge_text_b", item["text_b"])) for item in out]
        # 分块恒用常量批档（judge_fn 是闭包，backend.batch_size 探测恒落空
        # 的死分支已删）；设备分档 auto 在 config 解析时定型，子进程内部自
        # 会再按 batch_size 切，此处只做调用切片。
        from ..constants import SEMANTIC_MDEBERTA_BATCH
        chunk = max(1, int(SEMANTIC_MDEBERTA_BATCH))
        verdicts: list[Any] = []
        fail_start: int | None = None
        fail_reason = ""
        for start in range(0, len(pairs), chunk):
            # 0.17.1 重标合一：片间让路探针（方案 §2.2.5）——忙时 63 片不再
            # 一口气越墙，对齐 INTEGRATION「batch 返回后 worker 让路」宣称。
            if stop_probe is not None and start > 0 and stop_probe():
                fail_start, fail_reason = start, "job budget expired mid-batch"
                break
            part = judge_fn(pairs[start : start + chunk])
            if len(part) != len(pairs[start : start + chunk]):
                fail_start = start
                fail_reason = (
                    f"judge returned {len(part)} verdicts "
                    f"for {len(pairs[start:start + chunk])} pairs"
                )
                break
            verdicts.extend(part)
        if fail_start is not None:
            # 对齐保证：fail_start == len(verdicts)（此前各片逐片校验满额）。
            from ..semantic_judge import PairVerdict
            bad = PairVerdict("no_conflict", {}, None, "mdeberta:unavailable", error=fail_reason)
            verdicts.extend([bad] * (len(out) - fail_start))
        for item, verdict in zip(out, verdicts):
            item["verdict"] = verdict
        return out


def _judge_outcome(verdict: Any, min_prob: float) -> str:
    """§3.3 decision table (owner 2026-09-28): conflict ≥ min_prob → 'notice';
    conflict below → 'below_threshold'; possible → 'possible'; no_conflict →
    'clear'. Technical failures surface via verdict.error (the caller's
    fail-open path) and never reach this function."""
    if getattr(verdict, "error", None):
        return "error"
    if verdict.label == "conflict":
        return "notice" if float(verdict.probs.get("conflict", 0.0)) >= min_prob else "below_threshold"
    if verdict.label == "possible_conflict":
        return "possible"
    return "clear"


def _judge_pair_compat(backend: Any, text_a: str, text_b: str) -> Any:
    """Legacy-backend bridge: an extraction-shaped backend exposing only
    classify_pair(env_a, env_b) maps its candidate bool to a conflict /
    no_conflict verdict (test fixtures assert on notice outcomes, which this
    preserves). Returns None when the backend cannot serve."""
    try:
        signal = backend.classify_pair({"quote": text_a}, {"quote": text_b})
    except Exception:
        return None
    if getattr(signal, "candidate", False):
        return PairVerdict(
            "conflict", {"conflict": 1.0, "no_conflict": 0.0, "possible_conflict": 0.0},
            None, "test-backend",
        )
    return PairVerdict(
        "no_conflict", {"conflict": 0.0, "no_conflict": 1.0, "possible_conflict": 0.0},
        None, "test-backend",
    )


def _pair_diff_anchor(text_a: str, text_b: str) -> str:
    """§3.4 slot 差异锚 (owner 拍板): no extraction → a reproducible pair
    hash as the slot attribute. Same pair ⇒ same anchor; different
    oppositions under one subject never collide on a slot. NFKC + whitespace
    collapse so cosmetic reflow cannot fork the identity."""
    import unicodedata

    def canon(text: str) -> str:
        return " ".join(unicodedata.normalize("NFKC", text or "").split())

    digest = hashlib.sha256(
        (canon(text_a) + "\x1f" + canon(text_b)).encode("utf-8"),
    ).hexdigest()
    return digest[:12]


class _JudgePairView:
    """The notice-site payload view of one judged pair — what the judge saw
    and concluded, for model_signal in the notice payload."""

    @staticmethod
    def signal(verdict: Any) -> dict[str, Any]:
        return {
            "label": verdict.label,
            "probs": {k: round(float(v), 4) for k, v in verdict.probs.items()},
            "mechanism": verdict.mechanism,
            "model_version": verdict.model_version,
        }


class _JobJudgeBudget:
    """0.17.1 重标合一（owner 拍板 2026-10-03，方案 §2.3）：internal 帽撤销，
    单一 job 全局判定池——internal keepers 与 A-cross 按提交顺序消费同一个池，
    internal-first 排序保证 land-first（E10①）。receipt 分账按来源保留。

    半池守恒（harness 实测回归修复，2026-10-03）：internal 消费上限
    ⌈total/2⌉——无上限的 internal-first 在行密集语料（数值/表格记忆的
    O(n²) keeper）上会吃光全池，跨记忆对全部 capped→backlog（eval 冲突
    道实测：106 写 0 notice、internal 9390 行、backlog 顶帽驱逐）。跨记忆
    是召回主通道（0.372→0.163 前车之鉴），保底半池；internal 超出份额
    静默消失（池语义不变）。正常写入（internal 1~5 对）不受影响。

    前身 _JobQwenBudget（0.17.0 Q1 owner D1）：internal 保护帽 ≤3 + A-cross
    吃余量——帽的原始理由是 Qwen 一对 ~1.6s、internal keepers 曾吃光共享预算
    （recall 0.372→0.163）；mDeBERTa 批前向 ~0.1s/16 对后前提不成立，帽退役，
    land-first 语义由批量内排序继承。池仍 per-job 实例（R2-3：共享对象上
    不挂跨 job 可变状态）。"""

    def __init__(self, total: int = SEMANTIC_MAX_EXAMINED_PAIRS) -> None:
        self.total = max(1, int(total))
        self.internal_used = 0
        self.a_cross_used = 0

    @property
    def internal_share(self) -> int:
        """internal 可用份额 = ⌈total/2⌉（cross 同样保底 ⌊total/2⌋）。"""
        return max(1, self.total - self.total // 2)

    @property
    def remaining(self) -> int:
        return max(0, self.total - self.internal_used - self.a_cross_used)

    def spend(self, source: str) -> bool:
        """单一总池消费 + internal 半池份额。池/份额耗尽返回 False，收尾
        语义按来源分：internal 静默消失、A-cross 记 pairs_examined_capped
        进 backlog。"""
        if source == "internal":
            if self.internal_used >= self.internal_share or self.remaining <= 0:
                return False
            self.internal_used += 1
        else:
            if self.remaining <= 0:
                return False
            self.a_cross_used += 1
        return True

    def receipt_block(self) -> "dict[str, Any] | None":
        """§3.3 conditional receipt block — absent entirely when nothing was
        deducted (zero values never appear)."""
        block: dict[str, Any] = {}
        if self.internal_used:
            block["internal"] = self.internal_used
        if self.a_cross_used:
            block["a_cross"] = self.a_cross_used
        return block or None

    @property
    def pairs_examined(self) -> int:
        """Job-global pairs_examined (§3.3): internal + A-cross."""
        return self.internal_used + self.a_cross_used
