"""mema-twin write routing (0.16.2 plan §1.2, owner rule recorded as #976).

``mema-twin`` is the persona bucket: only the mema-twin agent itself may
write it. Any other caller whose write resolves to that bucket — including
authorized governance moves — is redirected to ``mema-twin-dev`` so persona
material stays cabined while remaining reachable in the twin family. The
twin itself (client or agent_id = "mema-twin") is unaffected. E6 untouched:
this is the default write destination, not the autonomous-move rule.
"""
from __future__ import annotations

from .ws_keys import _mechanical_ws_key as _mechanical_key  # noqa: F401  (split re-export)

TWIN_BUCKET = "mema-twin"
TWIN_DEV_BUCKET = "mema-twin-dev"
TWIN_IDENTITY = "mema-twin"

# 拆分批 ①（2026-10-04）：A9 时代的「有意复制」由顶层 ws_keys.py 收拢取代
# ——顶层叶子不触发 db/__init__ 副作用（A9 的复制理由就此消解），两侧现在
# 同源。_mechanical_key 名字保留（消费方 + tests/test_twin_variant_guard 的
# 等价钉继续指向它）。


def twin_redirect_target(
    canonical: str, *, client: str | None, agent_id: str | None,
) -> str | None:
    """Return the redirect destination when this write must be re-bucketed.

    ``canonical`` is the RESOLVED target bucket (post-alias/normalization);
    only the twin bucket triggers. Identity mirrors the agent_id attribution
    rule: trusted request identity only, never process env. A missing
    identity is NOT the twin — redirect applies.

    R3 实施审查 P1（2026-10-04）：比较改 casefold+strip 规范化键——此前
    字面比较下，"Mema-Twin" 等大小写变体在 confirm 侧绕过防毒守卫（raw
    变体不触发守卫），而 alias 落库键走同一规范化、精确命中保护键，毒行
    照样落库并劫持 twin 本体写入。规范键比较让守卫/redirect tail/hint
    三处消费方一次性闭合（真库实证复现于 mema 实施记录）。

    A9（0.17.1 修复批）：判定键再升级为**机械键**——casefold+strip 仍漏
    分隔符变体（``mema_twin`` / ``mematwin``）：解析器 1b 折叠会把这些
    变体折到已注册的拼写，攻击者先注册变体即可让 twin 本体写入落进
    攻击者桶（实测：twin 写 mema-twin → canonical=mema_twin，可读出
    persona）。机械键比较堵住该通道（与注册守卫配套）。
    """
    if _mechanical_key(canonical) != _mechanical_key(TWIN_BUCKET):
        return None
    if client == TWIN_IDENTITY or agent_id == TWIN_IDENTITY:
        return None
    return TWIN_DEV_BUCKET


def protected_bucket_variant(name: str) -> bool:
    """A9：``name`` 是 PROTECTED_WORKSPACES 的机械等价变体（非原名）。

    注册/改名路径用它拒绝变体名落库（否则变体先注册 → 解析器把保护桶
    原名折到变体 → 保护桶被劫持）。非保护桶（如 agent-lane/agent_lane）
    不受影响。
    """
    from .constants import PROTECTED_WORKSPACES

    key = _mechanical_key(name)
    if not key:
        return False
    return any(
        key == _mechanical_key(protected) and str(name).strip() != protected
        for protected in PROTECTED_WORKSPACES
    )
