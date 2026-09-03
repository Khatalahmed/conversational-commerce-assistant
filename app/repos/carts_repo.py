"""
WHAT:
    Cart-related data lookups from PDF §6.1 (cart contents) and §6.2
    (cart total). This is the ONLY file allowed to query `carts`
    directly.

WHY THIS APPROACH:
    Unlike orders/bargains (many documents per user), each user has
    exactly ONE cart document, matched by the "user" field (note: named
    "user", not "buyerId" — a schema naming difference worth knowing).
    So there's no history to page through — just "find the one cart
    that belongs to this user."

FLOW:
    Called by Phase 5's feature logic and, later, Phase 7's tool-calling
    layer.

LOGIC:
    An empty or missing cart is NOT an error — it's a completely normal
    state for a user who hasn't added anything yet (Phase 4.3's
    graceful-empty-result rule).

    THERE IS NO get_cart_total() HERE ANY MORE, and the reason is worth
    keeping. It existed for PDF §6.2 and summed cartPrice across the
    line items, documented as "ASSUMPTION: cartPrice is already the LINE
    TOTAL". Checked against the real carts collection, that assumption
    is false: a line reading qty=5, cartPrice=381 sits against a product
    whose discounted price is 400, so cartPrice is the PER-UNIT price
    agreed when the item was added. The function under-reported any line
    with qty > 1 - ₹381 for a line worth ₹1,905 - and two of the
    thirteen real carts have such a line.

    It was never reached: the cart total the assistant actually says out
    loud is computed by _enrich_cart() in agent/tool_executor.py, which
    multiplies by qty and is correct. Rather than fix a second, unused
    implementation of the same sum and leave two answers in the
    codebase, it is deleted. If PDF §6.2 ever needs its own entry point
    again, it belongs next to _enrich_cart's arithmetic, not beside it.

MECHANISM:
    Same two-layer safety pattern as every repo: Step 3.3 allowlist as
    the projection, Step 3.4 sanitizer on the result.
"""

import structlog

from app.db.connection import get_database
from app.repos.base import to_object_id
from app.security.field_allowlist import get_projection
from app.security.sanitizer import sanitize_document

logger = structlog.get_logger()

COLLECTION = "carts"


async def _find_cart(user_id: str) -> dict | None:
    """Internal helper — fetches the one cart for this user, with both
    safety layers applied. Returns None if none exists yet."""
    object_id = to_object_id(user_id)
    if object_id is None:
        return None
    db = get_database()
    projection = get_projection(COLLECTION)
    doc = await db[COLLECTION].find_one({"user": object_id}, projection)
    return sanitize_document(COLLECTION, doc)


async def get_cart(user_id: str) -> dict:
    """PDF §6.1 — all items currently in the user's cart."""
    doc = await _find_cart(user_id)
    items = (doc or {}).get("items", []) or []
    return {"items": items, "itemCount": len(items)}
