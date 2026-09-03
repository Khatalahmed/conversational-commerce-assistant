"""
WHAT:
    Review/rating summary (PDF §9.1) and seller trust indicators (§9.2)
    from PDF §9. The ONLY file allowed to query `reviews` directly. Also
    reads `users` for §9.2, same reasoning as products_repo's
    get_seller_info — no per-user scoping needed (reviews/seller stats
    are public), so reaching into users directly here is fine.

WHY THIS APPROACH:
    get_product_reviews() computes the average rating itself using
    MongoDB's aggregation pipeline (a $group stage) rather than pulling
    every review into Python and averaging there — this scales
    correctly even if a product eventually has thousands of reviews,
    since the average is computed by the database, not by loading
    everything into memory first.

FLOW:
    Called by Phase 5's feature logic and, later, Phase 7's tool-calling
    layer.

LOGIC:
    get_product_reviews() returns both the aggregate (average rating,
    review count) AND a handful of individual comments (matching PDF
    §9.1's example: "Rated 4.6★ from 10 reviews. Most say: ...") — most
    recent first, capped by `sample_size`.
    get_seller_trust_info() reuses the exact same trust-signal fields
    PDF §9.2 and §11.2 both reference (customerRating, productsSoldCount,
    followerCount) — same shape as products_repo's seller lookup, just
    addressed directly by seller ID rather than via a product.

MECHANISM:
    Same two-layer safety pattern as every repo: Step 3.3 allowlist as
    the projection (used inside the aggregation's $project stage too),
    Step 3.4 sanitizer on every result.
"""

import structlog

from app.db.connection import get_database
from app.repos.base import to_object_id
from app.security.field_allowlist import get_projection
from app.security.sanitizer import sanitize_document, sanitize_documents

logger = structlog.get_logger()

REVIEWS_COLLECTION = "reviews"
USERS_COLLECTION = "users"


async def get_product_reviews(product_id: str, sample_size: int = 5) -> dict:
    """PDF §9.1 — average rating, review count, and a handful of recent
    comments. Returns a clean zero-review result if none exist yet —
    not an error, a brand new product legitimately has no reviews."""
    object_id = to_object_id(product_id)
    if object_id is None:
        return {"productId": product_id, "reviewCount": 0, "averageRating": None, "comments": []}

    db = get_database()

    # Aggregation computes the average IN the database, not in Python -
    # scales correctly regardless of how many reviews a product has.
    agg_cursor = db[REVIEWS_COLLECTION].aggregate([
        {"$match": {"productId": object_id}},
        {"$group": {"_id": None, "averageRating": {"$avg": "$rating"}, "count": {"$sum": 1}}},
    ])
    agg_result = await agg_cursor.to_list(length=1)

    if not agg_result:
        return {"productId": product_id, "reviewCount": 0, "averageRating": None, "comments": []}

    projection = get_projection(REVIEWS_COLLECTION)
    cursor = (
        db[REVIEWS_COLLECTION]
        .find({"productId": object_id}, projection)
        .sort("createdAt", -1)
        .limit(sample_size)
    )
    docs = await cursor.to_list(length=sample_size)
    docs = sanitize_documents(REVIEWS_COLLECTION, docs)

    return {
        "productId": product_id,
        "reviewCount": agg_result[0]["count"],
        "averageRating": round(agg_result[0]["averageRating"], 1),
        "comments": [d.get("comment") for d in docs if d.get("comment")],
    }


async def get_seller_trust_info(seller_id: str) -> dict | None:
    """PDF §9.2 — rating, follower count, sales history for a seller."""
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
        "sellerId": seller_id,
        "businessName": (doc.get("sellerProfile") or {}).get("businessName"),
        "customerRating": doc.get("customerRating"),
        "followerCount": doc.get("followerCount"),
        "productsSoldCount": doc.get("productsSoldCount"),
    }