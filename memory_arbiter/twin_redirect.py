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
    only exact ``mema-twin`` triggers. Identity mirrors the agent_id
    attribution rule: trusted request identity only, never process env. A
    missing identity is NOT the twin — redirect applies.
    """
    if canonical != TWIN_BUCKET:
        return None
    if client == TWIN_IDENTITY or agent_id == TWIN_IDENTITY:
        return None
    return TWIN_DEV_BUCKET
