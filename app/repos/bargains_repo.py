"""
WHAT:
    All bargaining-related data lookups from PDF §5 (eligibility, status,
    counter-offer, suggested amount). §5.1 ("what is bargaining") is
    static FAQ text with no database involved, so it belongs in Phase
    5's feature layer, not here.

WHY THIS APPROACH:
    check_bargain_eligibility() and suggest_offer_amount() are NOT
    scoped to a user — they're properties of the PRODUCT (anyone can
    ask "can I bargain on this?" before ever making an offer). But
    get_bargain_status() and get_counter_offer() ARE scoped by buyerId,
    same cross-user protection pattern as orders_repo, since those
    reveal a specific negotiation between one buyer and one seller.

FLOW:
    Called by Phase 5's feature logic and, later, Phase 7's tool-calling
    layer.

LOGIC:
    get_bargain_status()/get_counter_offer() accept EITHER a specific
    bargain_id OR a product_id (returning the user's most recent bargain
    on that product) — matching how a real user actually refers to a
    bargain in chat ("my offer on the blue jacket"), since they'd never
    know or type a raw bargain ID.
    suggest_offer_amount() uses bargainSettings.maximumDiscount and
    products.price per the PDF's own stated data source for §5.5.

MECHANISM:
    Same two-layer safety pattern as every other repo: Step 3.3
    allowlist as the MongoDB projection, Step 3.4 sanitizer on every
    result, for both `bargains` and `products` documents fetched here.
"""

import structlog

from app.db.connection import get_database
from app.repos.base import to_object_id
from app.security.field_allowlist import get_projection
from app.security.sanitizer import sanitize_document

logger = structlog.get_logger()

BARGAINS_COLLECTION = "bargains"
PRODUCTS_COLLECTION = "products"


async def check_bargain_eligibility(product_id: str) -> dict | None:
    """PDF §5.2 — does this product allow bargaining, and by how much?"""
    object_id = to_object_id(product_id)
    if object_id is None:
        return None

    db = get_database()
    projection = get_projection(PRODUCTS_COLLECTION)
    doc = await db[PRODUCTS_COLLECTION].find_one({"_id": object_id}, projection)
    doc = sanitize_document(PRODUCTS_COLLECTION, doc)
    if doc is None:
        return None

    settings = doc.get("bargainSettings") or {}
    max_discount = settings.get("maximumDiscount")
    return {
        "productId": product_id,
        "bargainingAllowed": bool(settings),
        "autoAcceptDiscount": settings.get("autoAcceptDiscount"),
        "maximumDiscount": max_discount,
    }


async def suggest_offer_amount(product_id: str) -> dict | None:
    """PDF §5.5 — a reasonable offer, based on the seller's own discount
    settings and the product's price (per PDF's stated data source)."""
    object_id = to_object_id(product_id)
    if object_id is None:
        return None

    db = get_database()
    projection = get_projection(PRODUCTS_COLLECTION)
    doc = await db[PRODUCTS_COLLECTION].find_one({"_id": object_id}, projection)
    doc = sanitize_document(PRODUCTS_COLLECTION, doc)
    if doc is None:
        return None

    price = doc.get("price")
    max_discount = (doc.get("bargainSettings") or {}).get("maximumDiscount")
    if price is None or max_discount is None:
        return {"productId": product_id, "suggestionAvailable": False}

    suggested = round(price * (1 - max_discount / 100))
    return {
        "productId": product_id,
        "suggestionAvailable": True,
        "price": price,
        "maximumDiscount": max_discount,
        "suggestedOffer": suggested,
    }


async def _find_bargain_scoped(user_id: str, extra_query: dict) -> dict | None:
    """Internal helper — every user-scoped bargain lookup goes through
    this. Sorts newest-first and takes one, so it works correctly
    whether extra_query pins an exact bargain_id (only one match anyway)
    or just a product_id (returns the most recent bargain on it)."""
    buyer_object_id = to_object_id(user_id)
    if buyer_object_id is None:
        return None

    db = get_database()
    query = {"buyerId": buyer_object_id, **extra_query}
    projection = get_projection(BARGAINS_COLLECTION)
    cursor = (
        db[BARGAINS_COLLECTION]
        .find(query, projection)
        .sort("createdAt", -1)
        .limit(1)
    )
    docs = await cursor.to_list(length=1)
    if not docs:
        return None
    return sanitize_document(BARGAINS_COLLECTION, docs[0])


async def get_bargain_status(
    user_id: str, bargain_id: str | None = None, product_id: str | None = None
) -> dict | None:
    """PDF §5.3 — accepts EITHER a bargain_id OR a product_id (returns
    the user's most recent bargain on that product)."""
    if bargain_id:
        object_id = to_object_id(bargain_id)
        if object_id is None:
            return None
        doc = await _find_bargain_scoped(user_id, {"_id": object_id})
    elif product_id:
        product_object_id = to_object_id(product_id)
        if product_object_id is None:
            return None
        doc = await _find_bargain_scoped(user_id, {"productId": product_object_id})
    else:
        return None

    if doc is None:
        return None

    return {
        "productName": (doc.get("productSnapshot") or {}).get("name"),
        "status": doc.get("status"),
        "offeredPrice": doc.get("offeredPrice"),
        "currentPrice": doc.get("currentPrice"),
        "expiresAt": doc.get("expiresAt"),
        "hasCounterOffer": bool(doc.get("counterOffer")),
    }


async def get_counter_offer(
    user_id: str, bargain_id: str | None = None, product_id: str | None = None
) -> dict | None:
    """PDF §5.4 — the seller's counter-offer, if one exists. Real
    sandbox data confirms this ISN'T always the case (only 1/18
    bargains) — every caller must handle counterOfferExists=False."""
    if bargain_id:
        object_id = to_object_id(bargain_id)
        if object_id is None:
            return None
        doc = await _find_bargain_scoped(user_id, {"_id": object_id})
    elif product_id:
        product_object_id = to_object_id(product_id)
        if product_object_id is None:
            return None
        doc = await _find_bargain_scoped(user_id, {"productId": product_object_id})
    else:
        return None

    if doc is None:
        return None

    counter = doc.get("counterOffer")
    if not counter:
        return {
            "productName": (doc.get("productSnapshot") or {}).get("name"),
            "counterOfferExists": False,
        }

    return {
        "productName": (doc.get("productSnapshot") or {}).get("name"),
        "counterOfferExists": True,
        "price": counter.get("price"),
        "message": counter.get("message"),
        "offeredAt": counter.get("offeredAt"),
    }

async def count_bargains(user_id: str) -> int:
    """How many offers this user has made, in total.

    Exists for the same reason orders_repo.count_orders does: a capped
    list handed over with no total reads AS the total. Measured there on
    a real account - 67 orders, answered as "you have 8".
    """
    buyer_object_id = to_object_id(user_id)
    if buyer_object_id is None:
        return 0
    db = get_database()
    return await db[BARGAINS_COLLECTION].count_documents({"buyerId": buyer_object_id})


async def list_bargains(user_id: str, limit: int = 5) -> list[dict]:
    """Every offer this user has made, newest first.

    WHY THIS DID NOT EXIST, AND WHY THAT WAS A HOLE. Every other bargain
    lookup takes a bargain_id or a product_id, which is right for "what
    happened to my offer on the blue kurta" and useless for "what
    happened to my offers" - a user has neither id to hand. Asked that,
    the assistant could only ask which product, and the eval case that
    caught it was retired as unanswerable rather than fixed.

    Bargaining is the feature the marketplace is built around, so a buyer being
    unable to ask about their own offers was the gap worth closing.

    SCOPED BY buyerId, like every other user-facing bargain query - the
    id is server-injected and never comes from the model.
    """
    buyer_object_id = to_object_id(user_id)
    if buyer_object_id is None:
        return []

    db = get_database()
    projection = get_projection(BARGAINS_COLLECTION)
    cursor = (
        db[BARGAINS_COLLECTION]
        .find({"buyerId": buyer_object_id}, projection)
        .sort("createdAt", -1)
        .limit(limit)
    )
    docs = await cursor.to_list(length=limit)
    return [sanitize_document(BARGAINS_COLLECTION, doc) for doc in docs]
