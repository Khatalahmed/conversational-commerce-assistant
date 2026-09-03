"""
WHAT:
    Short-video ("Bits") data lookups from PDF §8 — trending videos,
    tagged products in a video, and hashtag search. The ONLY file
    allowed to query `bits` directly.

WHY THIS APPROACH:
    Like livesessions, Bits are public content, not private to one
    user — no buyerId/userId scoping, no cross-user security test here.

FLOW:
    Called by Phase 5's feature logic and, later, Phase 7's tool-calling
    layer.

LOGIC:
    get_trending_bits() filters to isTrending=True first per PDF §8.1's
    exact wording ("trending videos"), sorted by viewCount as a
    tiebreaker/fallback ordering.
    search_by_hashtag() is case-insensitive and accepts the hashtag with
    or without a leading "#", since a user might type either "jeans" or
    "#jeans" in chat.

MECHANISM:
    Same two-layer safety pattern as every repo: Step 3.3 allowlist as
    the projection, Step 3.4 sanitizer on every result.
"""

import re

import structlog

from app.db.connection import get_database
from app.repos.base import to_object_id
from app.security.field_allowlist import get_projection
from app.security.sanitizer import sanitize_document, sanitize_documents

logger = structlog.get_logger()

COLLECTION = "bits"


async def get_trending_bits(limit: int = 5) -> list[dict]:
    """PDF §8.1 — currently trending short videos."""
    db = get_database()
    projection = get_projection(COLLECTION)
    cursor = (
        db[COLLECTION]
        .find({"isTrending": True}, projection)
        .sort("viewCount", -1)
        .limit(limit)
    )
    docs = await cursor.to_list(length=limit)
    return sanitize_documents(COLLECTION, docs)


async def get_tagged_products(bit_id: str) -> dict | None:
    """PDF §8.2 — which products are tagged inside this specific video."""
    object_id = to_object_id(bit_id)
    if object_id is None:
        return None

    db = get_database()
    projection = get_projection(COLLECTION)
    doc = await db[COLLECTION].find_one({"_id": object_id}, projection)
    doc = sanitize_document(COLLECTION, doc)
    if doc is None:
        return None

    return {
        "bitId": bit_id,
        "title": doc.get("title"),
        "products": doc.get("products", []),
    }


async def search_by_hashtag(hashtag: str, limit: int = 10) -> list[dict]:
    """PDF §8.3 — find videos by hashtag or topic. Accepts the tag with
    or without a leading '#', case-insensitive."""
    normalized = hashtag if hashtag.startswith("#") else f"#{hashtag}"

    db = get_database()
    projection = get_projection(COLLECTION)
    cursor = (
        db[COLLECTION]
        .find(
            {"hashtags": {"$regex": f"^{re.escape(normalized)}$", "$options": "i"}},
            projection,
        )
        .limit(limit)
    )
    docs = await cursor.to_list(length=limit)
    return sanitize_documents(COLLECTION, docs)