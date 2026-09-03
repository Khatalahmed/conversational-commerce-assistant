"""
WHAT:
    Self-service account/profile lookups from PDF §10 — default
    address, unread notifications, saved items, followers/following.
    Reads `users`, `addresses`, `notifications`. Everything here is
    scoped to the LOGGED-IN user looking at their OWN data — this is
    different from products_repo/reviews_repo's user lookups, which are
    about looking up OTHER people (sellers) publicly.

WHY THIS APPROACH:
    Includes is_seller() as a small shared helper — Phase 5.9 needs to
    detect "is this user a seller" before offering seller-only features,
    and this is the natural place for it since it's just a `users` read.

FLOW:
    Called by Phase 5's feature logic and, later, Phase 7's tool-calling
    layer. seller_repo.py will import is_seller() from here rather than
    duplicating the same check.

LOGIC:
    get_default_address() falls back to the user's FIRST address (by
    creation date) if none is explicitly marked isDefault - a
    reasonable real-world behavior, flagged clearly in the response via
    isFallback so the feature layer can phrase it appropriately
    ("your address on file" vs "your default address").
    get_followers_following() reports resolvedCount separately from the
    RAW array length, since the gap-closure patch scanned reference
    fields like buyerId/sellerId/hostId but NOT followers/following
    arrays specifically - some entries may not resolve to a real user
    document yet. This is handled gracefully, not as an error.

MECHANISM:
    Same two-layer safety pattern as every repo: Step 3.3 allowlist as
    the projection, Step 3.4 sanitizer on every result, across all
    three collections this file touches.
"""

import structlog

from app.db.connection import get_database
from app.repos.base import to_object_id
from app.security.field_allowlist import get_projection
from app.security.sanitizer import sanitize_document, sanitize_documents

logger = structlog.get_logger()

USERS_COLLECTION = "users"
ADDRESSES_COLLECTION = "addresses"
NOTIFICATIONS_COLLECTION = "notifications"


async def is_seller(user_id: str) -> bool:
    """Shared helper — Phase 5.9 uses this to gate seller-only features."""
    object_id = to_object_id(user_id)
    if object_id is None:
        return False
    db = get_database()
    doc = await db[USERS_COLLECTION].find_one(
        {"_id": object_id}, {"sellerStatus": 1}
    )
    return bool(doc and doc.get("sellerStatus"))


async def get_default_address(user_id: str) -> dict | None:
    """PDF §10.1 — falls back to the user's first address if none is
    explicitly marked default (see module docstring)."""
    object_id = to_object_id(user_id)
    if object_id is None:
        return None

    db = get_database()
    projection = get_projection(ADDRESSES_COLLECTION)

    doc = await db[ADDRESSES_COLLECTION].find_one(
        {"user": object_id, "isDefault": True}, projection
    )
    is_fallback = False
    if doc is None:
        doc = await db[ADDRESSES_COLLECTION].find_one(
            {"user": object_id}, projection, sort=[("createdAt", 1)]
        )
        is_fallback = True

    doc = sanitize_document(ADDRESSES_COLLECTION, doc)
    if doc is None:
        return None

    return {
        "label": doc.get("label"),
        "type": doc.get("type"),
        "city": doc.get("city"),
        "state": doc.get("state"),
        "pincode": doc.get("pincode"),
        "isFallback": is_fallback,
    }


async def get_unread_notifications(user_id: str, sample_size: int = 3) -> dict:
    """PDF §10.2 — unread count plus a few recent samples."""
    object_id = to_object_id(user_id)
    if object_id is None:
        return {"unreadCount": 0, "samples": []}

    db = get_database()
    query = {"userId": object_id, "isRead": False}

    unread_count = await db[NOTIFICATIONS_COLLECTION].count_documents(query)

    projection = get_projection(NOTIFICATIONS_COLLECTION)
    cursor = (
        db[NOTIFICATIONS_COLLECTION]
        .find(query, projection)
        .sort("createdAt", -1)
        .limit(sample_size)
    )
    docs = await cursor.to_list(length=sample_size)
    docs = sanitize_documents(NOTIFICATIONS_COLLECTION, docs)

    return {
        "unreadCount": unread_count,
        "samples": [{"title": d.get("title"), "message": d.get("message")} for d in docs],
    }


async def get_saved_items(user_id: str) -> dict | None:
    """PDF §10.3 — saved product/bit IDs. Returns IDs only, not full
    product details (image/price) - deliberately kept single-purpose;
    the Phase 5 feature layer enriches these via products_repo when
    building the final chat response, rather than this repo reaching
    into a second collection it doesn't own."""
    object_id = to_object_id(user_id)
    if object_id is None:
        return None

    db = get_database()
    projection = get_projection(USERS_COLLECTION)
    doc = await db[USERS_COLLECTION].find_one({"_id": object_id}, projection)
    doc = sanitize_document(USERS_COLLECTION, doc)
    if doc is None:
        return None

    return {
        "savedProductIds": doc.get("savedProducts", []) or [],
        "savedBitIds": doc.get("savedBits", []) or [],
    }


async def get_followers_following(user_id: str, kind: str) -> dict | None:
    """PDF §10.4 — resolves follower/following IDs into usernames.
    kind must be "followers" or "following". See module docstring on
    resolvedCount vs raw count."""
    if kind not in ("followers", "following"):
        raise ValueError("kind must be 'followers' or 'following'")

    object_id = to_object_id(user_id)
    if object_id is None:
        return None

    db = get_database()
    user_projection = get_projection(USERS_COLLECTION)
    doc = await db[USERS_COLLECTION].find_one({"_id": object_id}, user_projection)
    doc = sanitize_document(USERS_COLLECTION, doc)
    if doc is None:
        return None

    raw_ids = doc.get(kind, []) or []
    if not raw_ids:
        return {"kind": kind, "rawCount": 0, "resolvedCount": 0, "users": []}

    resolved_cursor = db[USERS_COLLECTION].find(
        {"_id": {"$in": raw_ids}}, user_projection
    )
    resolved_docs = await resolved_cursor.to_list(length=len(raw_ids))
    resolved_docs = sanitize_documents(USERS_COLLECTION, resolved_docs)

    return {
        "kind": kind,
        "rawCount": len(raw_ids),
        "resolvedCount": len(resolved_docs),
        "users": [
            {
                "username": d.get("username"),
                "businessName": (d.get("sellerProfile") or {}).get("businessName"),
            }
            for d in resolved_docs
        ],
    }