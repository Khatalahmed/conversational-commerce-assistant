"""
WHAT:
    Defines, per collection, EXACTLY which fields the chatbot is allowed
    to read from the database and return in a response. Anything not
    listed here is invisible to the rest of the app by default.

WHY THIS APPROACH:
    An allowlist (only listed fields pass through) is safer than a
    blocklist (block known-bad fields), because a newly added database
    field is hidden by default here, rather than leaking by default
    until someone remembers to blocklist it.

FLOW:
    1. Every repository function (Phase 4) will import the allowlist for
       its collection and use it to build a MongoDB "projection" — i.e.
       telling MongoDB to only RETURN these fields in the first place,
       not fetch everything and filter afterward.
    2. As a second layer (Phase 3.4's sanitizer), any response also gets
       stripped of anything not on this list, in case a repo function
       has a bug and over-fetches.

LOGIC:
    Nested fields (e.g. "pricing.total" inside orders) are written with
    dot notation, matching how MongoDB projections work natively.
    Fields explicitly marked EXCLUDED below are commented for clarity,
    even though omission alone already hides them — the comments exist
    so future developers immediately understand WHY something is
    missing, not just that it is.

MECHANISM:
    Each collection maps to a plain Python list of field name strings.
    Repos will pass these lists directly into MongoDB's `projection`
    argument, e.g. collection.find({}, {"password": 0}) is the WRONG
    (blocklist) pattern — we instead build a dict like
    {"orderId": 1, "status": 1, ...} which is the correct allowlist
    pattern MongoDB projections support.
"""

# ── ORDERS ──────────────────────────────────────────────────────────
# EXCLUDED: deliveryAddress (full address + phone), payment (method/
# status is borderline but paidAt/method not needed for chatbot use
# cases per PDF), invoice (url is fine per PDF §3.5 — INCLUDED below),
# buyerNote (free text, not part of any PDF requirement), location, __v.
ORDERS_ALLOWED_FIELDS = [
    "orderId",
    "buyerId",
    "sellerId",
    "items",
    "pricing",
    "status",
    "statusHistory",
    "tracking",
    "orderType",
    "deliveryType",
    "dates",
    "invoice.url",
    "invoice.generatedAt",
    "review",
    "createdAt",
    "updatedAt",
    "deliveryAddress.city",
    "deliveryAddress.state",
]

# ── PRODUCTS ─────────────────────────────────────────────────────────
# EXCLUDED: stepData (internal seller-flow scratch data, not
# customer-facing), __v.
PRODUCTS_ALLOWED_FIELDS = [
    "SKU",
    "sellerId",
    "name",
    "description",
    "category",
    "subCategory",
    "categoryType",
    "productType",
    "price",
    "discountedPrice",
    "totalStock",
    "condition",
    "status",
    "isSold",
    "isTopPick",
    "images",
    "variants",
    "shipping",
    "bargainSettings",
    "tags",
    "searchKeywords",
    "viewCount",
    "likeCount",
    "saveCount",
    "shareCount",
    "analytics",
    "createdAt",
    "updatedAt",
]

# ── BARGAINS ─────────────────────────────────────────────────────────
# EXCLUDED: buyerNote, sellerNote (free-text, not part of any PDF
# requirement), __v.
BARGAINS_ALLOWED_FIELDS = [
    "productId",
    "buyerId",
    "sellerId",
    "originalPrice",
    "productSnapshot",
    "offeredPrice",
    "currentPrice",
    "discountPercentage",
    "status",
    "variant",
    "quantity",
    "orderPlaced",
    "expiresAt",
    "autoAccepted",
    "counterOffer",
    "respondedAt",
    "createdAt",
    "updatedAt",
]

# ── CARTS ────────────────────────────────────────────────────────────
CARTS_ALLOWED_FIELDS = [
    "user",
    "items",
    "coupon",
    "discount",
    "liveSessionId",
    "createdAt",
    "updatedAt",
]

# ── COUPONS ──────────────────────────────────────────────────────────
COUPONS_ALLOWED_FIELDS = [
    "code",
    "name",
    "discountType",
    "discountValue",
    "minSpend",
    "maxDiscount",
    "applicableProducts",
    "sellerId",
    "isActive",
    "startDate",
    "endDate",
    "maxUsagePerUser",
    "ordersUsed",
    "totalDiscountGiven",
    "views",
    "viewsThisWeek",
]

# ── LIVESESSIONS ─────────────────────────────────────────────────────
# EXCLUDED: comments[].userId is technically another user's ID exposed
# via a public livestream comment — acceptable since livestream comments
# are inherently public-facing within the app, same as products/reviews.
LIVESESSIONS_ALLOWED_FIELDS = [
    "channelName",
    "title",
    "description",
    "hashtags",
    "thumbnail",
    "hostId",
    "products",
    "productSequence",
    "status",
    "isActive",
    "isTrending",
    "scheduledStartTime",
    "startTime",
    "endTime",
    "duration",
    "viewersCount",
    "peakViewers",
    "views",
    "likeCount",
    "revenue",
    "comments",
    "createdAt",
    "updatedAt",
]

# ── BITS ─────────────────────────────────────────────────────────────
BITS_ALLOWED_FIELDS = [
    "title",
    "description",
    "video",
    "thumbnail",
    "hashtags",
    "products",
    "userId",
    "likeCount",
    "viewCount",
    "shareCount",
    "isActive",
    "isTrending",
    "createdAt",
]

# ── REVIEWS ──────────────────────────────────────────────────────────
REVIEWS_ALLOWED_FIELDS = [
    "productId",
    "reviewerId",
    "rating",
    "comment",
    "createdAt",
]

# ── ADDRESSES ────────────────────────────────────────────────────────
# EXCLUDED: line1, line2 (exact street address — PDF §10.1 example only
# shows label/city/state, not full street), phone, coordinates.
ADDRESSES_ALLOWED_FIELDS = [
    "user",
    "type",
    "label",
    "city",
    "state",
    "pincode",
    "isDefault",
]

# ── NOTIFICATIONS ────────────────────────────────────────────────────
NOTIFICATIONS_ALLOWED_FIELDS = [
    "userId",
    "type",
    "title",
    "message",
    "actionUrl",
    "actionLabel",
    "isRead",
    "createdAt",
]

# ── USERS ────────────────────────────────────────────────────────────
# EXCLUDED (per your explicit constraints): password, refreshToken,
# refreshTokenExpiry, phone, email, dob, countryCode, isAdmin,
# sellerProfile.bankDetails, sellerProfile.documents, sellerProfile.
# address (full billing/pickup address), sellerProfile.rejectionMessage,
# globalBargainSettings (internal), searchHistory, __v.
USERS_ALLOWED_FIELDS = [
    "username",
    "profilePic",
    "categoryType",
    "sellerStatus",
    "sellerProfile.businessName",
    "followerCount",
    "followers",
    "following",
    "reviewsCount",
    "productsSoldCount",
    "customerRating",
    # monthlyRevenue / yearlyRevenue ARE DELIBERATELY ABSENT.
    #
    # They were here for seller_repo.get_sales_performance (PDF §11.2),
    # which is not wired into TOOL_REGISTRY and cannot be reached from
    # chat - so this list was letting a seller's takings out of the
    # database for EVERY query against `users`, to serve one function
    # nobody can call.
    #
    # That is backwards for the outer layer of a defence-in-depth
    # design. This list is the thing that is supposed to still hold when
    # an inner layer has a bug: a repo that forgets its projection, or a
    # new tool that returns a raw user document. Narrow by default, and
    # widen deliberately.
    #
    # WIRING SELLER TOOLS UP MEANS ADDING THEM BACK - and gating them,
    # see the header of app/repos/seller_repo.py. get_sales_performance
    # returns None for both until then.
    "savedBits",
    "savedProducts",
    "likedProducts",
    "createdAt",
]

# ── PAYOUTS ──────────────────────────────────────────────────────────
# EXCLUDED: payoutDestination (bank/UPI details — per your explicit
# constraint), __v.
PAYOUTS_ALLOWED_FIELDS = [
    "orderRef",
    "orderTotal",
    "commission",
    "sellerAmount",
    "status",
    "payoutMode",
    "sellerId",
    "createdAt",
    "updatedAt",
]

# ── CATEGORIES ───────────────────────────────────────────────────────
# Fully public reference data, no sensitivity concerns.
CATEGORIES_ALLOWED_FIELDS = [
    "name",
    "slug",
    "image",
    "subCategories",
]

# ── ANALYTICS/VIEW COLLECTIONS (productviews, liveviews, reelviews) ──
# Not directly user-facing per the PDF (no requirement reads these
# directly for chat responses), but included for completeness in case
# a future feature (e.g. Phase 5.2's recommendation logic) needs them.
# userId is present since Product Discovery §4.6 explicitly needs a
# user's own view history to power recommendations for THAT SAME user.
PRODUCTVIEWS_ALLOWED_FIELDS = [
    "userId",
    "productId",
    "durationMs",
    "completedView",
    "timestamp",
]


# ── PRODUCT_EMBEDDINGS ───────────────────────────────────────────────
# the marketplace's own semantic-search layer, kept in sync by their pipeline.
# The 384-float `embedding` IS listed, because the repo reads it to use
# as a $vectorSearch query vector - but it never reaches a response: the
# vector stage returns only _id + score, and the real product documents
# are then fetched through the products allowlist like any other lookup.
# EXCLUDED: embeddingHash (internal change-detection), updatedAt.
PRODUCT_EMBEDDINGS_ALLOWED_FIELDS = [
    "embedding",
]

# Master lookup so repos (Phase 4) can fetch the right list by name.
COLLECTION_ALLOWLISTS = {
    "orders": ORDERS_ALLOWED_FIELDS,
    "products": PRODUCTS_ALLOWED_FIELDS,
    "bargains": BARGAINS_ALLOWED_FIELDS,
    "carts": CARTS_ALLOWED_FIELDS,
    "coupons": COUPONS_ALLOWED_FIELDS,
    "livesessions": LIVESESSIONS_ALLOWED_FIELDS,
    "bits": BITS_ALLOWED_FIELDS,
    "reviews": REVIEWS_ALLOWED_FIELDS,
    "addresses": ADDRESSES_ALLOWED_FIELDS,
    "notifications": NOTIFICATIONS_ALLOWED_FIELDS,
    "users": USERS_ALLOWED_FIELDS,
    "payouts": PAYOUTS_ALLOWED_FIELDS,
    "categories": CATEGORIES_ALLOWED_FIELDS,
    "productviews": PRODUCTVIEWS_ALLOWED_FIELDS,
    "product_embeddings": PRODUCT_EMBEDDINGS_ALLOWED_FIELDS,
}


def get_projection(collection_name: str) -> dict:
    """
    Converts an allowlist into the dict shape MongoDB's `projection`
    argument expects: {"fieldName": 1, ...} meaning "only return these."
    Always includes "_id" implicitly (MongoDB includes it by default
    unless explicitly excluded, which is fine — _id is not sensitive).
    """
    fields = COLLECTION_ALLOWLISTS.get(collection_name)
    if fields is None:
        raise ValueError(
            f"No allowlist defined for collection '{collection_name}'. "
            f"Every collection we query MUST have an explicit allowlist "
            f"— refusing to guess."
        )
    return {field: 1 for field in fields}