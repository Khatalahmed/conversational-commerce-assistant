"""
WHAT:
    All 7 product-related data lookups from PDF §4 (Product Discovery).
    This is the ONLY file in the whole app allowed to query the
    `products` collection directly. By agreed design (unlike orders),
    two functions here (get_recommendations, get_seller_info) also read
    from `users` directly, since products aren't owned by one user the
    way orders are, and splitting those two into a separate not-yet-built
    users_repo would leave PDF §4.6/§4.7 unfinished for no real benefit
    right now.

WHY THIS APPROACH:
    Unlike orders, products are public — any user can browse any
    product, so there's no per-user scoping/cross-user test here. The
    interesting design work is instead in accepting ALREADY-STRUCTURED
    filters (category, price range, color, size) rather than raw text —
    turning natural language like "shirts under 500" into these filters
    is Phase 6/7's job (the LLM), once it exists. This repo function
    just needs to query correctly once it receives clean filter values.

FLOW:
    Called by Phase 5's feature logic and, later, Phase 7's tool-calling
    layer, which will have already extracted structured filters from
    the user's natural-language message via the LLM.

LOGIC:
    - search_products() and get_trending_products() both exclude
      isSold=True products by default (ASSUMPTION — flagged for
      confirmation: showing a sold-out item in general browsing/trending
      would be a bad experience; direct lookups below do NOT apply this
      filter, since a direct question about a specific product should
      always get an answer even if it's sold).
    - Category/subCategory matching is case-insensitive (NL extraction
      might produce "men" vs "Men" — we shouldn't lose a match over
      casing).
    - Price filtering uses discountedPrice, matching the PDF §4.2
      example exactly ("products under 500" -> discountedPrice <= 500).

MECHANISM:
    Same two-layer safety pattern as orders_repo: every query uses the
    Step 3.3 allowlist as its MongoDB projection, and every result
    passes through the Step 3.4 sanitizer before returning — for BOTH
    `products` and `users` documents fetched in this file.
"""

import re

import structlog
from pymongo.errors import OperationFailure

from app.agent.embeddings import (
    EmbeddingModelMismatch,
    EmbeddingUnavailable,
    embed_query,
    embedding_model_tag,
)
from app.config.settings import get_settings
from app.db.connection import get_database
from app.repos.base import to_object_id
from app.security.field_allowlist import get_projection
from app.security.sanitizer import sanitize_document, sanitize_documents

logger = structlog.get_logger()

COLLECTION = "products"
USERS_COLLECTION = "users"

# ── the marketplace's existing semantic-search layer ───────────────────────────
# These were already built, populated and indexed in the Atlas cluster
# (named "marketplace-semantic-search") by the marketplace team - 143 product
# embeddings for 143 products, kept current by a change-stream sync.
# Nothing here provisions them; this repo just stops ignoring them.
EMBEDDINGS_COLLECTION = "product_embeddings"
TEXT_INDEX = "product_text_index"      # Atlas Search, full-text
VECTOR_INDEX = "product_vector_index"  # Atlas Vector Search, 384-dim cosine

# Fields the full-text index actually covers. Taken from the live index
# definition rather than guessed - searching a path the index does not
# map returns nothing, silently.
TEXT_SEARCH_PATHS = ["name", "tags", "searchKeywords", "category", "subCategory"]


def _case_insensitive_exact(value: str) -> dict:
    """Builds a MongoDB regex query that matches a field exactly,
    ignoring case — so "men" and "Men" both match category "Men"."""
    return {"$regex": f"^{re.escape(value)}$", "$options": "i"}


async def _find_product(product_id: str) -> dict | None:
    """Internal helper — fetches one product by ID, applying both
    safety layers. Used by every direct-lookup function below."""
    object_id = to_object_id(product_id)
    if object_id is None:
        return None  # malformed ID -> nothing found, no crash

    db = get_database()
    projection = get_projection(COLLECTION)
    doc = await db[COLLECTION].find_one({"_id": object_id}, projection)
    return sanitize_document(COLLECTION, doc)


async def search_products(
    category: str | None = None,
    sub_category: str | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
    color: str | None = None,
    size: str | None = None,
    in_stock_only: bool = False,
    limit: int = 20,
) -> list[dict]:
    """PDF §4.1 (category search) + §4.2 (price filtering), combined
    into one flexible filter function — Phase 5.2.1's structured filter
    schema. All arguments are optional; only the ones actually passed
    narrow the search."""
    query: dict = {"isSold": False}  # see module docstring ASSUMPTION note

    if category:
        query["category"] = _case_insensitive_exact(category)
    if sub_category:
        query["subCategory"] = _case_insensitive_exact(sub_category)

    if min_price is not None or max_price is not None:
        price_query: dict = {}
        if min_price is not None:
            price_query["$gte"] = min_price
        if max_price is not None:
            price_query["$lte"] = max_price
        query["discountedPrice"] = price_query  # PDF §4.2 uses discountedPrice, not price

    if color or size:
        # $elemMatch requires BOTH conditions to match the SAME array
        # entry — without it, MongoDB would accept a product that has
        # the right color in one variant and the right size in another.
        variant_match: dict = {}
        if color:
            variant_match["color"] = _case_insensitive_exact(color)
        if size:
            variant_match["size"] = _case_insensitive_exact(size)
        if in_stock_only:
            variant_match["isOutOfStock"] = False
        query["variants"] = {"$elemMatch": variant_match}
    elif in_stock_only:
        query["totalStock"] = {"$gt": 0}

    db = get_database()
    projection = get_projection(COLLECTION)
    cursor = db[COLLECTION].find(query, projection).limit(limit)
    docs = await cursor.to_list(length=limit)
    return sanitize_documents(COLLECTION, docs)


async def get_variant_stock(product_id: str, color: str, size: str) -> dict | None:
    """PDF §4.3 — is this specific color/size in stock, and how many?"""
    doc = await _find_product(product_id)
    if doc is None:
        return None

    variants = doc.get("variants", []) or []
    # Search the variants array in Python (not MongoDB) since we already
    # fetched the full product — no need for a second query.
    match = next(
        (
            v for v in variants
            if v.get("color", "").lower() == color.lower()
            and v.get("size", "").lower() == size.lower()
        ),
        None,
    )
    if match is None:
        return {"productId": product_id, "color": color, "size": size, "found": False}

    return {
        "productId": product_id,
        "color": color,
        "size": size,
        "found": True,
        "stock": match.get("stock", 0),
        "isOutOfStock": match.get("isOutOfStock", True),
    }


async def get_product_detail(product_id: str) -> dict | None:
    """PDF §4.4 — full description, images, condition, return policy."""
    doc = await _find_product(product_id)
    if doc is None:
        return None
    return {
        "productId": product_id,
        "name": doc.get("name"),
        "description": doc.get("description"),
        "images": doc.get("images", []),
        "condition": doc.get("condition"),
        "price": doc.get("price"),
        "discountedPrice": doc.get("discountedPrice"),
        "shipping": doc.get("shipping"),
        "variants": doc.get("variants", []),
    }


async def get_trending_products(limit: int = 5) -> list[dict]:
    """PDF §4.5 — currently trending / top-pick products."""
    db = get_database()
    projection = get_projection(COLLECTION)
    query = {"isSold": False}
    cursor = (
        db[COLLECTION]
        .find(query, projection)
        # isTopPick first, then view/like count — matches "top-pick" and
        # "trending" both being covered by this one function per the PDF.
        .sort([("isTopPick", -1), ("viewCount", -1), ("likeCount", -1)])
        .limit(limit)
    )
    docs = await cursor.to_list(length=limit)
    return sanitize_documents(COLLECTION, docs)


async def get_recommendations(user_id: str, limit: int = 5) -> list[dict]:
    """PDF §4.6 — based on the user's own saved/liked products. Falls
    back to trending if the user has no saved/liked history yet, or
    doesn't exist — never returns an error for a new user."""
    object_id = to_object_id(user_id)
    if object_id is None:
        return await get_trending_products(limit)

    db = get_database()
    user_projection = get_projection(USERS_COLLECTION)
    user_doc = await db[USERS_COLLECTION].find_one({"_id": object_id}, user_projection)
    user_doc = sanitize_document(USERS_COLLECTION, user_doc)

    if user_doc is None:
        return await get_trending_products(limit)  # unknown user -> generic fallback

    saved = user_doc.get("savedProducts", []) or []
    liked = user_doc.get("likedProducts", []) or []
    seed_ids = list(saved) + list(liked)

    if not seed_ids:
        return await get_trending_products(limit)  # no history yet -> generic fallback

    # Find what categories the user's saved/liked products belong to,
    # so we can recommend MORE of what they've already shown interest in.
    seed_cursor = db[COLLECTION].find({"_id": {"$in": seed_ids}}, {"category": 1})
    seed_docs = await seed_cursor.to_list(length=len(seed_ids))
    categories = list({d["category"] for d in seed_docs if d.get("category")})

    if not categories:
        return await get_trending_products(limit)

    projection = get_projection(COLLECTION)
    query = {
        "category": {"$in": categories},
        "_id": {"$nin": seed_ids},  # don't recommend what they already saved/liked
        "isSold": False,
    }
    cursor = db[COLLECTION].find(query, projection).sort("likeCount", -1).limit(limit)
    docs = await cursor.to_list(length=limit)
    return sanitize_documents(COLLECTION, docs)


async def get_seller_info(product_id: str) -> dict | None:
    """PDF §4.7 — who's selling this, and are they trustworthy?"""
    doc = await _find_product(product_id)
    if doc is None or not doc.get("sellerId"):
        return None

    db = get_database()
    seller_projection = get_projection(USERS_COLLECTION)
    seller_doc = await db[USERS_COLLECTION].find_one(
        {"_id": doc["sellerId"]}, seller_projection
    )
    seller_doc = sanitize_document(USERS_COLLECTION, seller_doc)
    if seller_doc is None:
        return None

    seller_profile = seller_doc.get("sellerProfile", {}) or {}
    return {
        "productId": product_id,
        "businessName": seller_profile.get("businessName"),
        "customerRating": seller_doc.get("customerRating"),
        "followerCount": seller_doc.get("followerCount"),
    }

async def get_distinct_categories() -> dict:
    """Returns the REAL category/subCategory values actually used on
    products - not the categories collection's display menu, which we
    discovered uses different naming (e.g. "Home Decor" vs the real
    products.category value "Home"). This is what Phase 6/7 should
    ground the LLM's filter extraction against, since it matches
    exactly what search_products() filters on."""
    db = get_database()
    categories = await db[COLLECTION].distinct("category")
    sub_categories = await db[COLLECTION].distinct("subCategory")
    return {
        "categories": sorted(c for c in categories if c),
        "subCategories": sorted(s for s in sub_categories if s),
    }

async def get_products_by_ids(product_ids: list) -> list[dict]:
    """Batch lookup - turns a list of product ObjectIds (e.g. from a
    live session's productSequence, or a Bit's tagged products) into
    real product data in ONE query, instead of one query per product.
    Used by the Live Shopping and Bits feature layers to enrich raw ID
    lists before responding to the user."""
    if not product_ids:
        return []

    db = get_database()
    projection = get_projection(COLLECTION)
    cursor = db[COLLECTION].find({"_id": {"$in": product_ids}}, projection)
    docs = await cursor.to_list(length=len(product_ids))
    return sanitize_documents(COLLECTION, docs)

async def _search_products_by_name_regex(name: str, limit: int) -> list[dict]:
    """The original substring match. Kept ONLY as a fallback for a
    cluster without the Atlas Search index (a fresh environment, a
    restored snapshot). It cannot use an index - a case-insensitive
    regex forces a collection scan - so it is the slow path by
    definition, not a preference."""
    db = get_database()
    projection = get_projection(COLLECTION)
    query = {"name": {"$regex": re.escape(name), "$options": "i"}, "isSold": False}
    cursor = db[COLLECTION].find(query, projection).limit(limit)
    docs = await cursor.to_list(length=limit)
    return sanitize_documents(COLLECTION, docs)


async def search_products_by_name(name: str, limit: int = 5) -> list[dict]:
    """Find a product the user named, using the marketplace's Atlas Search index.

    WHY NOT THE REGEX IT REPLACES. The old query was
    {"name": {"$regex": ..., "$options": "i"}}, and a case-insensitive
    regex cannot use an index - every search scanned the whole
    collection. It also only ever matched the NAME, so "buddha statue"
    missed a product tagged "buddha" but named something else, and a
    typo missed everything.

    product_text_index was already built and READY on this cluster and
    covers name, tags, searchKeywords, category and subCategory. Using
    it is faster AND better recall, at no cost. Measured on the real
    catalogue: "watch" now also finds "Smartwatch" (the regex found it
    only by accident of substring matching, unranked), and "budha" finds
    Buddha, which the regex could not do at all.

    Falls back to the old scan if the index is absent, so a cluster
    without it degrades in speed rather than breaking outright.
    """
    db = get_database()
    pipeline = [
        # $search MUST be the first stage in the pipeline.
        {
            "$search": {
                "index": TEXT_INDEX,
                # THREE CLAUSES, EACH EARNING ITS PLACE - measured against
                # the real catalogue, not assumed:
                #   text     - ordinary token match, scores highest
                #   wildcard - the analyzer treats "Smartwatch" as ONE
                #              token, so a plain text search for "watch"
                #              misses it. This is the only clause that
                #              catches a word inside a compound word.
                #   fuzzy 1  - typo tolerance ("budha" -> Buddha,
                #              "shrit" -> Shirt). maxEdits is 1, NOT 2:
                #              at 2 the catalogue returned "Spider Fetch
                #              T Shirt" and "Bamboo towel" for "watch",
                #              and an assistant presents its top hits as
                #              real answers, so noise is worse than a
                #              miss.
                "compound": {
                    "should": [
                        {"text": {"query": name, "path": TEXT_SEARCH_PATHS}},
                        {
                            "wildcard": {
                                "query": f"*{name}*",
                                "path": TEXT_SEARCH_PATHS,
                                "allowAnalyzedField": True,
                            }
                        },
                        {
                            "text": {
                                "query": name,
                                "path": TEXT_SEARCH_PATHS,
                                "fuzzy": {"maxEdits": 1},
                            }
                        },
                    ],
                    "minimumShouldMatch": 1,
                },
            }
        },
        {"$match": {"isSold": False}},
        {"$limit": limit},
        # Same allowlist as every other read - the projection is not
        # optional just because this is an aggregation.
        {"$project": get_projection(COLLECTION)},
    ]

    try:
        docs = [doc async for doc in db[COLLECTION].aggregate(pipeline)]
    except OperationFailure as exc:
        logger.warning(
            "atlas_search_unavailable_falling_back_to_scan",
            index=TEXT_INDEX,
            detail=str(exc)[:200],
        )
        return await _search_products_by_name_regex(name, limit)

    return sanitize_documents(COLLECTION, docs)


async def find_similar_products(product_id: str, limit: int = 5) -> list[dict]:
    """Semantically similar products, via the marketplace's existing vector index.

    HOW THIS WORKS WITHOUT KNOWING THEIR EMBEDDING MODEL. Normally
    $vectorSearch needs the QUERY embedded by the same model that
    produced the stored vectors - embed with a different model and
    cosine similarity is meaningless, silently. We do not know which
    model the marketplace used (nothing in the database records it).

    So the query vector here is not computed at all: it is the STORED
    embedding of a product we already have. Stored-vs-stored is the same
    model by construction, so the comparison is valid whatever that
    model turns out to be. Free-text semantic search ("something cosy
    for winter") still needs their answer; "more like this one" does not.

    Sold products are dropped AFTER the vector stage rather than before,
    since $vectorSearch cannot be preceded by a $match.
    """
    object_id = to_object_id(product_id)
    if object_id is None:
        return []

    db = get_database()
    seed = await db[EMBEDDINGS_COLLECTION].find_one(
        {"_id": object_id}, get_projection(EMBEDDINGS_COLLECTION)
    )
    if not seed or not seed.get("embedding"):
        # Not every product necessarily has an embedding yet - the sync
        # is eventually consistent. Absence is normal, not an error.
        logger.info("no_embedding_for_product", product_id=product_id)
        return []

    pipeline = [
        {
            "$vectorSearch": {
                "index": VECTOR_INDEX,
                "path": "embedding",
                "queryVector": seed["embedding"],
                # numCandidates is the approximate-search breadth; the
                # +1 on limit is because the seed always returns itself
                # at similarity 1.0 and is dropped below.
                "numCandidates": max(100, (limit + 1) * 10),
                "limit": limit + 1,
            }
        },
        {"$project": {"score": {"$meta": "vectorSearchScore"}}},
    ]

    try:
        hits = [doc async for doc in db[EMBEDDINGS_COLLECTION].aggregate(pipeline)]
    except OperationFailure as exc:
        logger.warning(
            "vector_search_unavailable", index=VECTOR_INDEX, detail=str(exc)[:200]
        )
        return []

    ranked_ids = [h["_id"] for h in hits if h["_id"] != object_id][:limit]
    if not ranked_ids:
        return []

    # get_products_by_ids applies the products allowlist and sanitizer -
    # the embeddings collection never supplies content, only an ordering.
    products = await get_products_by_ids(ranked_ids)
    by_id = {p["_id"]: p for p in products if not p.get("isSold")}

    # $in does not preserve order, and here the order IS the relevance.
    return [by_id[pid] for pid in ranked_ids if pid in by_id]

async def search_products_semantically(query: str, limit: int = 5) -> list[dict]:
    """Free-text search by MEANING: "something cosy for winter".

    THE DIFFERENCE FROM find_similar_products. That one reuses a stored
    vector as its query, so it needs no model and works whatever embedded
    the catalogue. This one has to embed the QUERY, which means the model
    must match the one that embedded the documents.

    A MISMATCH IS THE DANGEROUS CASE, not a missing model. Comparing a
    query from model A against documents from model B returns numbers
    that look completely normal - ranked, plausible, in the usual 0-0.4
    range - and are noise. Nothing errors. The results read as merely
    mediocre, which is exactly how a broken search survives a review.

    So the stored documents carry an embeddingModel tag, and this refuses
    to run unless it matches. Untagged documents mean a catalogue from
    the original pipeline, which was never identified - also a refusal.
    """
    if not query or not query.strip():
        return []

    db = get_database()

    # Check the catalogue BEFORE spending an embedding call on a query
    # we would not be able to use.
    sample = await db[EMBEDDINGS_COLLECTION].find_one(
        {}, {"embeddingModel": 1, "embedding": 1}
    )
    if sample is None:
        logger.info("semantic_search_no_embeddings")
        return []

    active_tag = embedding_model_tag()
    stored_tag = sample.get("embeddingModel")

    # THE UNTAGGED CASE. Documents from the original pipeline carry no
    # embeddingModel field at all, so they can never match any tag - and
    # with a read-only credential they cannot be tagged. LEGACY_EMBEDDING_
    # ASSUMED is the operator asserting which model produced them, backed
    # by scripts/identify_embedding_model.py reproducing those exact
    # vectors.
    #
    # It only helps when the assertion names the model we are ACTUALLY
    # running. Otherwise it falls through to the mismatch below, which is
    # the point: a stale assertion must not outlive the model it was
    # written for.
    if stored_tag is None:
        assumed = get_settings().legacy_embedding_assumed
        if assumed and assumed == active_tag:
            # Logged on every search, not once at startup. This is a
            # human claim standing in for a stored fact, and the log is
            # the only trace of that when results are later questioned.
            logger.info(
                "semantic_search_untagged_catalogue_assumed",
                assumed=assumed,
                basis="LEGACY_EMBEDDING_ASSUMED",
                caveat="documents carry no tag; match is asserted, not stored",
            )
            stored_tag = assumed

    if stored_tag != active_tag:
        logger.error(
            "semantic_search_model_mismatch",
            stored=stored_tag or "(untagged - original pipeline)",
            expected=active_tag,
            consequence="refusing to search rather than return ranked noise",
        )
        raise EmbeddingModelMismatch(
            f"The catalogue was embedded by {stored_tag or 'an unidentified model'}, "
            f"but queries would be embedded by {active_tag}. Either run "
            f"scripts/identify_embedding_model.py and configure the matching "
            f"model, run scripts/reembed_catalogue.py, or use "
            f"find_similar_products, which needs no model."
        )

    try:
        query_vector = await embed_query(query)
    except EmbeddingUnavailable as exc:
        logger.warning("semantic_search_unavailable", detail=str(exc)[:200])
        return []

    pipeline = [
        {
            "$vectorSearch": {
                "index": VECTOR_INDEX,
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": max(100, limit * 20),
                "limit": limit,
            }
        },
        {"$project": {"score": {"$meta": "vectorSearchScore"}}},
    ]

    try:
        hits = [doc async for doc in db[EMBEDDINGS_COLLECTION].aggregate(pipeline)]
    except OperationFailure as exc:
        logger.warning(
            "semantic_search_index_unavailable",
            index=VECTOR_INDEX,
            detail=str(exc)[:200],
        )
        return []

    ranked_ids = [h["_id"] for h in hits]
    if not ranked_ids:
        return []

    products = await get_products_by_ids(ranked_ids)
    by_id = {p["_id"]: p for p in products if not p.get("isSold")}
    # $in does not preserve order, and here the order IS the relevance.
    return [by_id[pid] for pid in ranked_ids if pid in by_id]
