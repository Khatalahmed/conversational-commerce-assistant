"""
WHAT:
    Seller-side data lookups from PDF §11 — payout status, sales
    performance, coupon performance, pending bargain count. Reads
    `payouts`, `users`, `coupons`, `bargains`. Everything here is scoped
    by sellerId, same cross-user protection pattern as orders_repo, just
    for the seller's own business data instead of a buyer's orders.

WHY THIS APPROACH:
    get_coupon_performance() scopes by sellerId AND code TOGETHER -
    worth calling out specifically: without the sellerId check, one
    seller could ask about a coupon code belonging to a DIFFERENT
    seller and see its performance stats. Coupon codes aren't
    guaranteed unique/private the way an orderId or bargainId is, so
    this scoping matters just as much here as buyerId does in
    orders_repo.

OUT OF SCOPE FOR THE DEMO - A DECISION, NOT AN OVERSIGHT:
    None of these four is in TOOL_REGISTRY, so none is reachable from
    chat. That is deliberate. Everything the demo shows is buyer-side -
    the system prompt opens "You are the shopping assistant", and
    every question in docs/demo-script.md is a buyer's. Four more tool
    schemas would be re-sent on every round of every conversation, for a
    capability no demo question touches, in a service where the tool
    schemas already dominate the prompt budget.

    The code stays because PDF §11 is a real client requirement and the
    tests below prove these queries work against real data. What is
    missing is the wiring, not the capability.

WHAT WIRING IT UP WOULD TAKE, so the next person does not get it wrong:
    1. seller_id MUST NOT be a tool parameter. Every function here takes
       it first, and exposing that shape directly would let the model
       ask for ANY seller's revenue, payout amounts and coupon
       performance - the exact cross-user leak that keeping user_id out
       of all 35 schemas exists to prevent. It has to be the verified
       JWT subject, injected server-side, the way orders_repo's user_id
       already is.
    2. Because TOOL_REGISTRY injects the verified id as `user_id=` and
       these take `seller_id=`, that means thin adapters, not a direct
       registration.
    3. Gate on account_repo.is_seller() first, per PDF §5.9 - a buyer
       asking about payouts should be told this is a seller feature, not
       handed an empty result that reads like a bug.
    4. Add a system-prompt rule, or the model will offer seller features
       to buyers the same way it once offered to place bargains.

FLOW:
    Called by app/tests/test_seller_repo.py. Not reachable from chat -
    see above.

LOGIC:
    get_payout_status() takes BOTH seller_id and order_ref, matching
    PDF §11.1's example exactly ("When do I get paid for order
    ORD0000000000000002?") - a seller has many payouts, so we need to
    know which one.

MECHANISM:
    Same two-layer safety pattern as every repo: Step 3.3 allowlist as
    the projection, Step 3.4 sanitizer on every result, across all
    three collections this file touches.
"""

import re

import structlog

from app.db.connection import get_database
from app.repos.base import to_object_id
from app.security.field_allowlist import get_projection
from app.security.sanitizer import sanitize_document

logger = structlog.get_logger()

PAYOUTS_COLLECTION = "payouts"
USERS_COLLECTION = "users"
COUPONS_COLLECTION = "coupons"
BARGAINS_COLLECTION = "bargains"


async def get_payout_status(seller_id: str, order_ref: str) -> dict | None:
    """PDF §11.1 — payout status for one specific order."""
    object_id = to_object_id(seller_id)
    if object_id is None:
        return None

    db = get_database()
    projection = get_projection(PAYOUTS_COLLECTION)
    doc = await db[PAYOUTS_COLLECTION].find_one(
        {"sellerId": object_id, "orderRef": order_ref}, projection
    )
    doc = sanitize_document(PAYOUTS_COLLECTION, doc)
    if doc is None:
        return None

    return {
        "orderRef": doc.get("orderRef"),
        "status": doc.get("status"),
        "sellerAmount": doc.get("sellerAmount"),
        "payoutMode": doc.get("payoutMode"),
    }


async def get_sales_performance(seller_id: str) -> dict | None:
    """PDF §11.2 — overall sales performance summary."""
    object_id = to_object_id(seller_id)
    if object_id is None:
        return None

    db = get_database()
    projection = get_projection(USERS_COLLECTION)
    doc = await db[USERS_COLLECTION].find_one({"_id": object_id}, projection)
    doc = sanitize_document(USERS_COLLECTION, doc)
    if doc is None:
        return None

    return {
        "productsSoldCount": doc.get("productsSoldCount"),
        "monthlyRevenue": doc.get("monthlyRevenue"),
        "yearlyRevenue": doc.get("yearlyRevenue"),
    }


async def get_coupon_performance(seller_id: str, code: str) -> dict | None:
    """PDF §11.3 — how a seller's OWN coupon is performing. Scoped by
    sellerId + code together - see module docstring on why this
    matters."""
    object_id = to_object_id(seller_id)
    if object_id is None:
        return None

    db = get_database()
    projection = get_projection(COUPONS_COLLECTION)
    doc = await db[COUPONS_COLLECTION].find_one(
        {
            "sellerId": object_id,
            "code": {"$regex": f"^{re.escape(code)}$", "$options": "i"},
        },
        projection,
    )
    doc = sanitize_document(COUPONS_COLLECTION, doc)
    if doc is None:
        return None

    return {
        "code": doc.get("code"),
        "views": doc.get("views"),
        "viewsThisWeek": doc.get("viewsThisWeek"),
        "ordersUsed": doc.get("ordersUsed"),
        "totalDiscountGiven": doc.get("totalDiscountGiven"),
    }


async def get_pending_bargain_count(seller_id: str) -> dict:
    """PDF §11.4 — how many pending bargain offers await this seller."""
    object_id = to_object_id(seller_id)
    if object_id is None:
        return {"pendingCount": 0}

    db = get_database()
    count = await db[BARGAINS_COLLECTION].count_documents(
        {"sellerId": object_id, "status": "pending"}
    )
    return {"pendingCount": count}