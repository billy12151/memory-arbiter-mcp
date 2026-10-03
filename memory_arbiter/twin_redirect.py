"""mema-twin write routing (0.16.2 plan §1.2, owner rule recorded as #976).

``mema-twin`` is the persona bucket: only the mema-twin agent itself may
write it. Any other caller whose write resolves to that bucket — including
authorized governance moves — is redirected to ``mema-twin-dev`` so persona
material stays cabined while remaining reachable in the twin family. The
twin itself (client or agent_id = "mema-twin") is unaffected. E6 untouched:
this is the default write destination, not the autonomous-move rule.
"""
from __future__ import annotations

TWIN_BUCKET = "mema-twin"
TWIN_DEV_BUCKET = "mema-twin-dev"
TWIN_IDENTITY = "mema-twin"


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
    """
    if canonical.strip().casefold() != TWIN_BUCKET:
        return None
    if client == TWIN_IDENTITY or agent_id == TWIN_IDENTITY:
        return None
    return TWIN_DEV_BUCKET
