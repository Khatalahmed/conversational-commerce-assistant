"""
WHAT:
    Coupon validity checking from PDF §6.3 — READ-ONLY. Only ever
    reports whether a code WOULD work; never applies it to anything.
    The ONLY file allowed to query `coupons` directly.

WHY THIS APPROACH:
    Coupons aren't owned by one user (anyone can ask "does SALE100
    work?"), but "have I already used this" IS user-specific. `usedBy`
    (every OTHER user who's redeemed this code) is deliberately excluded
    from the Step 3.3 allowlist — it's other users' data. So instead of
    fetching that whole array, we run a separate, narrow existence
    check: "does usedBy CONTAIN this one user_id" — confirming the
    current user's own usage without ever exposing who else used it.

FLOW:
    Called by Phase 5's feature logic. minSpend is checked against a
    cart total the CALLER supplies: the tool schema takes an optional
    cart_total, and the assistant fills it from get_cart, whose total is
    computed by _enrich_cart() in agent/tool_executor.py. That is the
    only place a cart total is worked out - see the note there.

LOGIC:
    Valid only if ALL of: exists, isActive, today is within
    [startDate, endDate], minSpend is met (if a cart total was
    provided), and this user hasn't hit maxUsagePerUser. Any failing
    check returns valid=False with a specific reason.
    NOTE: compared against a NAIVE UTC datetime (no timezone), matching
    how the MongoDB driver returns dates by default — mixing
    timezone-aware and naive datetimes in a comparison crashes Python.

MECHANISM:
    Main fetch uses the standard allowlist + sanitizer pattern. The
    per-user usage check is a SEPARATE count_documents() query matching
    on `usedBy` containing the user's ObjectId — Mongo matches array
    containment automatically on an equality query like this — without
    that field ever appearing in the returned/sanitized document.
"""

import re
from datetime import datetime, timezone

import structlog

from app.db.connection import get_database
from app.repos.base import to_object_id
from app.security.field_allowlist import get_projection
from app.security.sanitizer import sanitize_document

logger = structlog.get_logger()

COLLECTION = "coupons"


def _now_naive_utc() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


async def check_coupon_validity(
    code: str, user_id: str, cart_total: float | None = None
) -> dict:
    """PDF §6.3 — is this code valid right now, for this user, and what
    would it save? READ-ONLY: never applies the coupon."""
    db = get_database()
    projection = get_projection(COLLECTION)

    doc = await db[COLLECTION].find_one(
        {"code": {"$regex": f"^{re.escape(code)}$", "$options": "i"}},
        projection,
    )
    doc = sanitize_document(COLLECTION, doc)

    if doc is None:
        return {"code": code, "valid": False, "reason": "not_found"}

    if not doc.get("isActive"):
        return {"code": code, "valid": False, "reason": "inactive"}

    now = _now_naive_utc()
    start_date = doc.get("startDate")
    end_date = doc.get("endDate")
    if start_date and now < start_date:
        return {"code": code, "valid": False, "reason": "not_started_yet"}
    if end_date and now > end_date:
        return {"code": code, "valid": False, "reason": "expired"}

    min_spend = doc.get("minSpend")
    if min_spend and cart_total is not None and cart_total < min_spend:
        return {
            "code": code,
            "valid": False,
            "reason": "min_spend_not_met",
            "minSpend": min_spend,
            "cartTotal": cart_total,
        }

    max_usage = doc.get("maxUsagePerUser")
    if max_usage is not None:
        user_object_id = to_object_id(user_id)
        if user_object_id is not None:
            already_used = await db[COLLECTION].count_documents(
                {"code": doc.get("code"), "usedBy": user_object_id}
            )
            if already_used >= max_usage:
                return {"code": code, "valid": False, "reason": "usage_limit_reached"}

    return {
        "code": doc.get("code"),
        "valid": True,
        "discountType": doc.get("discountType"),
        "discountValue": doc.get("discountValue"),
        "maxDiscount": doc.get("maxDiscount"),
        "minSpend": min_spend,
    }