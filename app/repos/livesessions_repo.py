"""
WHAT:
    Live-shopping data lookups from PDF §7 — is anyone live right now,
    what's being shown, and a recap of a finished session. The ONLY
    file allowed to query `livesessions` directly.

WHY THIS APPROACH:
    Unlike orders/bargains/carts, live sessions aren't private to one
    user — anyone can ask "is anyone live?" So there's no buyerId/userId
    scoping here, and no cross-user security test for this repo (nothing
    to leak between users, since none of this is user-specific data).

FLOW:
    Called by Phase 5's feature logic and, later, Phase 7's tool-calling
    layer.

LOGIC:
    get_live_now() only returns sessions with status "live" — PDF §7.1's
    exact ask ("is anyone live right now"), not scheduled or ended ones.
    get_session_recap() is intentionally allowed to return a session in
    ANY status (not just "ended") — PDF §7.3's example ("what did I miss
    in that live") implies a finished session, but nothing stops someone
    asking for a recap mid-session either, and there's no harm in
    answering that too.

MECHANISM:
    Same two-layer safety pattern as every repo: Step 3.3 allowlist as
    the projection, Step 3.4 sanitizer on every result.
"""

import structlog

from app.db.connection import get_database
from app.repos.base import to_object_id
from app.security.field_allowlist import get_projection
from app.security.sanitizer import sanitize_document, sanitize_documents

logger = structlog.get_logger()

COLLECTION = "livesessions"


async def get_live_now(limit: int = 5) -> list[dict]:
    """PDF §7.1 — which sessions are live right now, if any."""
    db = get_database()
    projection = get_projection(COLLECTION)
    cursor = (
        db[COLLECTION]
        .find({"status": "live"}, projection)
        .sort("startTime", -1)
        .limit(limit)
    )
    docs = await cursor.to_list(length=limit)
    return sanitize_documents(COLLECTION, docs)


async def get_session_products(session_id: str) -> dict | None:
    """PDF §7.2 — which products are being featured, in what order."""
    object_id = to_object_id(session_id)
    if object_id is None:
        return None

    db = get_database()
    projection = get_projection(COLLECTION)
    doc = await db[COLLECTION].find_one({"_id": object_id}, projection)
    doc = sanitize_document(COLLECTION, doc)
    if doc is None:
        return None

    return {
        "sessionId": session_id,
        "title": doc.get("title"),
        "status": doc.get("status"),
        "productSequence": doc.get("productSequence", []),
    }


async def get_session_recap(session_id: str) -> dict | None:
    """PDF §7.3 — summary of a session (peak viewers, revenue, products,
    recent comments). Works for any status, not just "ended" — see
    module docstring."""
    object_id = to_object_id(session_id)
    if object_id is None:
        return None

    db = get_database()
    projection = get_projection(COLLECTION)
    doc = await db[COLLECTION].find_one({"_id": object_id}, projection)
    doc = sanitize_document(COLLECTION, doc)
    if doc is None:
        return None

    return {
        "sessionId": session_id,
        "title": doc.get("title"),
        "status": doc.get("status"),
        "peakViewers": doc.get("peakViewers"),
        "revenue": doc.get("revenue"),
        "productCount": len(doc.get("products", []) or []),
        "comments": doc.get("comments", []),
    }