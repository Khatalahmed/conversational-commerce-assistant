"""
WHAT:
    Executes a tool call the LLM requested, injecting the verified
    user_id server-side, and TRIMS large results down to only what the
    LLM actually needs before sending them back into the conversation.

WHY THIS APPROACH:
    Real testing revealed a distinct problem from Phase 3's data
    safety: search_products returning 20 full product documents (each
    with complete image arrays, full variant/image trees, shipping
    details) produced over 51,000 tokens for ONE tool result -
    blowing well past the free tier's token-per-minute limit.
    Phase 3's allowlist governs what's safe to leave the DATABASE; this
    trimming step is a SEPARATE concern - what's actually USEFUL to
    hand the LLM, matching what Phase 5's response templates reference
    (name, price, category - never raw image URLs).

FLOW:
    orchestrator.py calls execute_tool() for every tool_call the LLM
    requests, then sends the JSON-safe, TRIMMED result back to the LLM.

LOGIC:
    Only list-returning / detail-returning tools need trimming - order
    and bargain results are already small and specific per Phase 5's
    own response shapes. LLM_LIST_LIMIT_CAP hard-caps how many results
    are sent regardless of what the model requests, so a single tool
    call can never again blow the token budget this way.

    TRIMMERS DO TWO JOBS, NOT ONE. Shrinking is the obvious one. The
    second is making a result USEFUL: a live session's productSequence
    and a Bit's products are raw ObjectIds, and no amount of them lets
    the model say "the blue jacket is up next". Those two trimmers are
    async and enrich the IDs into real names and prices via
    products_repo.get_products_by_ids() - one batched query, not one
    per product.

    Live-session and Bit comments carry the commenting user's userId.
    Public or not, it is an internal ID the model is instructed never
    to surface, so the comment trimmer keeps only username and text.

MECHANISM:
    TOOL_REGISTRY (from tools.py) provides the real function and
    whether it needs user_id. to_json_safe() converts MongoDB types.
    _TRIMMERS maps specific tool names to a function that strips a
    result down before it's ever serialized; a trimmer may be sync or
    async, and execute_tool awaits it when needed.
"""

import inspect
import json
from datetime import datetime, timezone

import structlog
from bson import ObjectId

from app.agent.tools import TOOL_REGISTRY
from app.config.settings import get_settings
from app.repos import bargains_repo, orders_repo, products_repo
from app.repos.base import to_object_id

logger = structlog.get_logger()

# Hard cap - regardless of what limit the LLM requests, list results
# are capped here. Prevents a repeat of the 51k-token overflow.
LLM_LIST_LIMIT_CAP = 8

# Comments are free text of unbounded length and count. A handful is
# enough for the model to characterise the mood of a session.
MAX_COMMENTS = 3

# WHAT A FAILED LOOKUP TELLS THE MODEL, and why it is a sentence of
# instruction rather than a status.
#
# This used to read "This lookup failed - please try again." Measured, on
# a real question: search_products_semantically refused (the catalogue's
# embeddings carry no model tag, so searching would return ranked noise),
# the model received that bland failure, and answered "a few warm options
# to choose from: knit sweater, fleece hoodie, wool cardigan" - none of
# which exist in the catalogue. It had filled a gap from its own training
# data, which is the single thing this whole project claims never happens.
#
# The system prompt already says to answer only from tool results, but it
# covers a tool that finds NOTHING; a tool that BREAKS reads to the model
# as an absence of information rather than a prohibition, and being
# helpful is its default. The correction has to travel with the failure
# itself, in the tool result the model is looking at, not in a rule
# several thousand tokens earlier.
TOOL_FAILED = {
    "error": "This lookup is unavailable right now.",
    "instruction": (
        "Tell the user you cannot check this at the moment. Do NOT answer "
        "from your own knowledge, and do NOT name products, prices or "
        "orders that did not come from a tool result."
    ),
}


def to_json_safe(value):
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: to_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_json_safe(v) for v in value]
    return value


def _trim_product_list(products: list) -> list:
    """Used for search_products / get_trending_products /
    get_recommendations - strips each product down to only what Phase
    5's response templates actually reference (name + price), dropping
    images, full variant trees, shipping, tags, analytics entirely."""
    # DEDUPED ON (name, price). The real catalogue contains several
    # products with the same name at the same price - three "jeans" at
    # Rs.800 appeared in a measured similar-products result. They are
    # distinct rows in the database, but to a shopper they are the same
    # line repeated three times, and a list that repeats itself reads as
    # broken rather than thorough.
    #
    # A DIFFERENT price is kept: same name, different price is a real
    # choice the user can act on. Only the genuinely indistinguishable
    # collapse.
    seen = set()
    out = []
    for p in products or []:
        key = ((p.get("name") or "").strip().lower(), p.get("discountedPrice"))
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "product_id": str(p.get("_id")),
            "name": p.get("name"),
            "price": p.get("price"),
            "discountedPrice": p.get("discountedPrice"),
            "category": p.get("category"),
        })
        if len(out) >= LLM_LIST_LIMIT_CAP:
            break
    return out


def _trim_product_detail(detail: dict | None) -> dict | None:
    """Used for get_product_detail - keeps the fields PDF §4.4's spec
    needs, drops raw image URLs and per-variant image arrays (the LLM
    never needs to read a URL to describe a product in text)."""
    if detail is None:
        return None
    variants = detail.get("variants", []) or []
    return {
        "productId": detail.get("productId"),
        "name": detail.get("name"),
        "description": detail.get("description"),
        "condition": detail.get("condition"),
        "price": detail.get("price"),
        "discountedPrice": detail.get("discountedPrice"),
        "shipping": detail.get("shipping"),
        "variantSummary": [
            {"color": v.get("color"), "size": v.get("size"), "stock": v.get("stock")}
            for v in variants
        ],
    }


DELIVERED_OR_DONE = {"delivered", "cancelled", "returned", "refunded"}


def _trim_order_history(orders: list) -> list:
    """get_order_history - the raw documents are large and, on one
    measured call, 1,574 tokens for five orders: buyerId, sellerId,
    statusHistory, review, deliveryType and the full address all travel
    for an answer that needs an item, a status and a date.

    IT ALSO COMPUTES WHETHER A DELIVERY IS LATE. A real order came back
    with expectedDelivery of 27 July while the date was 26 August, and
    the assistant reported it flatly as "expected by 27 July 2026" -
    true, and useless. A model cannot notice that without knowing
    today's date, which it does not, so the comparison is done here and
    handed over as a fact. Only for orders still in progress: a
    delivered or cancelled order is not "late".
    """
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    out = []

    for order in (orders or [])[:LLM_LIST_LIMIT_CAP]:
        dates = order.get("dates") or {}
        expected = dates.get("expectedDelivery")
        status = (order.get("status") or "").lower()

        row = {
            "orderId": order.get("orderId"),
            "status": order.get("status"),
            "items": [
                {"name": i.get("name"), "qty": i.get("qty")}
                for i in (order.get("items") or [])[:4]
            ],
            "total": (order.get("pricing") or {}).get("total"),
            "expectedDelivery": expected,
        }

        if isinstance(expected, datetime) and status not in DELIVERED_OR_DONE:
            days_late = (now - expected).days
            if days_late > 0:
                row["isOverdue"] = True
                row["daysLate"] = days_late
        out.append(row)

    return out


def _trim_bargain_list(bargains: list) -> list:
    """list_bargains - the raw documents carry buyerId, sellerId, the
    whole productSnapshot including its image URL, variant and quantity,
    for an answer that needs a product, a price and what happened.

    IT ALSO SAYS WHETHER A PENDING OFFER HAS LAPSED, for exactly the
    reason _trim_order_history computes isOverdue: expiresAt is a date,
    the model does not know today's, and "expires 12 July" reported
    flatly in September is true and useless. Only for offers still
    pending - an accepted or expired one has already resolved.
    """
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    out = []

    for bargain in (bargains or [])[:LLM_LIST_LIMIT_CAP]:
        expires = bargain.get("expiresAt")
        status = (bargain.get("status") or "").lower()
        row = {
            "productName": (bargain.get("productSnapshot") or {}).get("name"),
            "offeredPrice": bargain.get("offeredPrice"),
            "originalPrice": bargain.get("originalPrice"),
            "status": bargain.get("status"),
            "hasCounterOffer": bool(bargain.get("counterOffer")),
            "orderPlaced": bool(bargain.get("orderPlaced")),
            "expiresAt": expires,
        }
        # THE DATE ALONE READS BADLY IN A CHAT BUBBLE, and a measured
        # answer proved it: handed expiresAt raw, the model printed
        # "expired at 2026-07-09T09:15:32" into a phone message. Same
        # defect _trim_order_history solved with daysLate - the model
        # cannot subtract from a date it does not have, so the number it
        # actually wants to say is computed here.
        if isinstance(expires, datetime):
            days = (now - expires).days
            if expires > now:
                row["expiresInDays"] = -days
            elif status in ("pending", "expired"):
                # The count is given for BOTH, because for an expired
                # offer the expiry IS what happened - and the first
                # version of this skipped it, leaving the model to print
                # "expired on 2026-07-09 09:15" beside a sibling offer
                # that correctly read "lapsed 54 days ago".
                row["lapsedDaysAgo"] = days
                # ...but only PENDING gets the flag. An offer already
                # marked expired says so in its status; one still marked
                # pending whose date has passed is the surprising state
                # the model cannot infer on its own.
                if status == "pending":
                    row["hasLapsed"] = True
        out.append(row)

    return out


def _trim_comments(comments: list | None) -> list:
    """Keeps username + text only. userId is deliberately dropped - it
    is an internal ID, and the system prompt forbids surfacing those."""
    recent = (comments or [])[-MAX_COMMENTS:]
    out = []
    for comment in recent:
        if isinstance(comment, dict):
            text = comment.get("text") or comment.get("comment")
            if text:
                out.append({"username": comment.get("username"), "text": text})
        elif isinstance(comment, str):
            out.append({"username": None, "text": comment})
    return out


def _trim_live_sessions(sessions: list) -> list:
    """get_live_now - drops thumbnails, product arrays and full comment
    threads; a "what's live" answer needs a title and an audience size."""
    return [
        {
            "session_id": str(s.get("_id")),
            "title": s.get("title"),
            "status": s.get("status"),
            "viewersCount": s.get("viewersCount"),
            "isTrending": s.get("isTrending"),
        }
        for s in (sessions or [])[:LLM_LIST_LIMIT_CAP]
    ]


def _trim_bits(bits: list) -> list:
    """get_trending_bits / search_by_hashtag - drops the video and
    thumbnail URLs entirely; the model describes a Bit in words and
    never needs to read a CDN link."""
    return [
        {
            "bit_id": str(b.get("_id")),
            "title": b.get("title"),
            "hashtags": (b.get("hashtags") or [])[:5],
            "likeCount": b.get("likeCount"),
            "viewCount": b.get("viewCount"),
        }
        for b in (bits or [])[:LLM_LIST_LIMIT_CAP]
    ]


def _trim_session_recap(recap: dict | None) -> dict | None:
    """get_session_recap - the comments array is unbounded, so it is
    replaced by a count plus the last few, and the raw product list by
    the count the repo already computed."""
    if recap is None:
        return None
    comments = recap.get("comments") or []
    return {
        "sessionId": recap.get("sessionId"),
        "title": recap.get("title"),
        "status": recap.get("status"),
        "peakViewers": recap.get("peakViewers"),
        "revenue": recap.get("revenue"),
        "productCount": recap.get("productCount"),
        "commentCount": len(comments),
        "recentComments": _trim_comments(comments),
    }


def _unresolved(requested: list, resolved: list) -> int:
    """How many referenced products no longer exist.

    NOT a rare edge case: 19 of 79 live sessions in the sandbox
    reference products that have since been deleted, so a session can
    legitimately list a product the catalogue cannot resolve. Reporting
    the gap explicitly stops the model promising "1 product" it is then
    unable to name - it can say the details are no longer available
    instead, which is the truth.
    """
    return max(len(requested) - len(resolved), 0)


async def _enrich_session_products(result: dict | None) -> dict | None:
    """get_session_products - productSequence is a raw ObjectId array.
    Resolved to real names and prices in ONE batched query so the model
    can actually name what is being featured."""
    if result is None:
        return None
    requested = list(result.get("productSequence") or [])[:LLM_LIST_LIMIT_CAP]
    products = await products_repo.get_products_by_ids(requested)
    return {
        "sessionId": result.get("sessionId"),
        "title": result.get("title"),
        "status": result.get("status"),
        "productCount": len(result.get("productSequence") or []),
        "products": _trim_product_list(products),
        "unavailableCount": _unresolved(requested, products),
    }


async def _enrich_tagged_products(result: dict | None) -> dict | None:
    """get_tagged_products - same reasoning as _enrich_session_products,
    for the products tagged inside a Bit."""
    if result is None:
        return None
    requested = list(result.get("products") or [])[:LLM_LIST_LIMIT_CAP]
    products = await products_repo.get_products_by_ids(requested)
    return {
        "bitId": result.get("bitId"),
        "title": result.get("title"),
        "productCount": len(result.get("products") or []),
        "products": _trim_product_list(products),
        "unavailableCount": _unresolved(requested, products),
    }


async def _enrich_cart(cart: dict | None) -> dict | None:
    """get_cart - items reference products by raw ObjectId, which the
    model cannot turn into "a Photo Frame and a Kurta". Resolved to real
    names in ONE batched query, same as the live-session and Bit
    trimmers.

    THE TOTAL IS COMPUTED HERE, and this is now the only place that
    computes it. carts_repo used to carry a get_cart_total() for PDF
    §6.2 which summed cartPrice WITHOUT multiplying by qty; measured
    against the real carts collection that is wrong (a line reading
    qty=5, cartPrice=381 sits against a unit price of 400, so cartPrice
    is per-unit). It was never called from here anyway - the items
    already carry cartPrice and qty, and a second tool round would cost
    a full prompt to learn something we can add up - so it was deleted
    rather than fixed, leaving one implementation instead of two that
    disagree.
    """
    if cart is None:
        return None
    items = cart.get("items") or []
    product_ids = [i.get("product") for i in items if i.get("product")]
    products = {p["_id"]: p for p in await products_repo.get_products_by_ids(product_ids)}

    lines, subtotal = [], 0
    for item in items:
        price = item.get("cartPrice") or 0
        qty = item.get("qty") or 0
        subtotal += price * qty
        product = products.get(item.get("product")) or {}
        lines.append({
            "name": product.get("name") or "a product no longer available",
            "variant": item.get("variant"),
            "qty": qty,
            "price": price,
        })

    return {
        "items": lines,
        "itemCount": len(lines),
        "subtotal": subtotal,
        "discount": cart.get("discount", 0),
        "total": subtotal - (cart.get("discount") or 0),
    }


async def _enrich_saved_items(saved: dict | None) -> dict | None:
    """get_saved_items - returns bare id lists, which tell the model
    nothing it can say out loud. Products are resolved to names; Bits are
    only counted, since naming them would need a second collection and
    "3 saved videos" is enough to answer the question."""
    if saved is None:
        return None
    product_ids = list(saved.get("savedProductIds") or [])[:LLM_LIST_LIMIT_CAP]
    products = await products_repo.get_products_by_ids(product_ids)
    return {
        "savedProducts": _trim_product_list(products),
        "savedProductCount": len(saved.get("savedProductIds") or []),
        "savedBitCount": len(saved.get("savedBitIds") or []),
    }


def _trim_follow_list(result: dict | None) -> dict | None:
    """get_followers_or_following - the repo resolves EVERY id in the
    list into a user document, so a popular account returns an unbounded
    array for a question that is usually about the number.

    The counts are kept whole and only the named sample is capped:
    "1,240 followers, including alice and bob" is the answer; listing
    1,240 usernames is not. rawCount and resolvedCount both survive
    because the gap between them is real - a follower whose account has
    since been deleted cannot be named, and the model should say so
    rather than promise a name it does not have.
    """
    if result is None:
        return None
    users = result.get("users") or []
    return {
        "kind": result.get("kind"),
        "rawCount": result.get("rawCount"),
        "resolvedCount": result.get("resolvedCount"),
        "sample": [
            {"username": u.get("username"), "businessName": u.get("businessName")}
            for u in users[:LLM_LIST_LIMIT_CAP]
        ],
    }


# Maps a tool name to a function that trims its raw repo result before
# it's sent to the LLM. Tools not listed here pass through unchanged -
# order/bargain/review results are already small and specific.
_TRIMMERS = {
    "get_order_history": _trim_order_history,
    "search_products": _trim_product_list,
    "get_trending_products": _trim_product_list,
    "get_recommendations": _trim_product_list,
    "get_product_detail": _trim_product_detail,
    "search_products_by_name": _trim_product_list,
    "find_similar_products": _trim_product_list,
    "search_products_semantically": _trim_product_list,
    # Live shopping
    "get_live_now": _trim_live_sessions,
    "get_session_recap": _trim_session_recap,
    "get_session_products": _enrich_session_products,  # async
    # Bits
    "get_trending_bits": _trim_bits,
    "search_by_hashtag": _trim_bits,
    "get_tagged_products": _enrich_tagged_products,  # async
    # Cart / account
    "get_cart": _enrich_cart,                        # async
    "get_saved_items": _enrich_saved_items,           # async
    "get_followers_or_following": _trim_follow_list,
    "list_bargains": _trim_bargain_list,
}


# A SECOND PASS, AFTER TRIMMING, FOR THE COUNT A WINDOW CANNOT SHOW.
#
# The trimmers shrink a list; they cannot say how much was left behind,
# because by then the rest is gone - and the model, handed eight orders
# and no total, reports eight. Measured on a real account: 67 orders,
# answered as "you have 8 orders". Faithful to what it was given, and
# wrong.
#
# Summarisers run after the trimmer, receive the verified user_id, and
# wrap the trimmed list with the totals only a fresh query knows. A map
# rather than an if, so the next list with the same problem - carts,
# Bits - is one line.
async def _summarise_orders(trimmed: list, user_id: str) -> dict:
    total = await orders_repo.count_orders(user_id)
    return {
        "orders": trimmed,
        # Named so the difference is unmissable: the model has to work to
        # confuse "showing" with "totalOrders".
        "showing": len(trimmed),
        "totalOrders": total,
        "note": (
            f"Showing the {len(trimmed)} most recent of {total} orders."
            if total > len(trimmed)
            else None
        ),
    }


async def _summarise_bargains(trimmed: list, user_id: str) -> dict:
    total = await bargains_repo.count_bargains(user_id)
    return {
        "offers": trimmed,
        "showing": len(trimmed),
        "totalOffers": total,
        "note": (
            f"Showing the {len(trimmed)} most recent of {total} offers."
            if total > len(trimmed)
            else None
        ),
    }


_SUMMARISERS = {
    "get_order_history": _summarise_orders,
    # Same defect, same fix: a capped list with no total reads as the
    # total. One line, because the map exists.
    "list_bargains": _summarise_bargains,
}


# THE APP'S VIEW OF THE SAME RESULT, ALONGSIDE THE MODEL'S.
#
# The client asked for products the user can tap. The obvious route is to
# let the model write the link into its sentence, and it is the wrong
# one: the model would be COMPOSING the reference, which means it can
# eventually attach the right id to the wrong name, and a card that opens
# the wrong product is worse than no card at all.
#
# So the ids travel beside the prose instead of inside it. The model
# never sees a URL and never writes one; the client is handed exact
# (product_id, name, price) rows lifted straight from the tool result and
# builds its own navigation. The system prompt's "never show internal
# IDs" rule stays exactly as it is, because the model still never shows
# one.
#
# Keyed by tool name rather than sniffed from the result's shape, for the
# same reason _TRIMMERS is: the shapes are known here, and a heuristic
# would eventually pick up an order line or a cart row and offer to open
# a product page for something that is not a product.
#
# None means the trimmed result IS the list; a string names the key the
# list sits under.
_PRODUCT_LIST_SOURCES = {
    "search_products": None,
    "search_products_by_name": None,
    "search_products_semantically": None,
    "find_similar_products": None,
    "get_trending_products": None,
    "get_recommendations": None,
    "get_session_products": "products",
    "get_tagged_products": "products",
    "get_saved_items": "savedProducts",
}


def _product_cards(tool_name: str, result) -> list[dict]:
    """The products this tool result surfaced, in the shape a client
    needs to draw a tappable card.

    NO URL IS BUILT HERE, deliberately. the storefront is a marketing site
    with no product pages - /product/<id> renders its router's 404 - so
    there is nowhere on the web for a link to point. The app routes on
    product_id through its own navigation stack, and a web link can be
    added later without touching this function, by whoever owns the
    route that would serve it.

    No images either: the trimmers drop them on purpose, and fetching
    thumbnails would need a second query. A follow-up, not a blocker -
    the allowlist already permits the field.
    """
    if tool_name == "get_product_detail":
        # The odd one out - a single product, and _trim_product_detail
        # names its id "productId" where the list trimmer says
        # "product_id". Normalised here rather than changing either,
        # since both shapes are what their own readers already expect.
        if isinstance(result, dict) and result.get("productId"):
            return [{
                "product_id": str(result["productId"]),
                "name": result.get("name"),
                "price": result.get("price"),
                "discountedPrice": result.get("discountedPrice"),
            }]
        return []

    if tool_name not in _PRODUCT_LIST_SOURCES:
        return []

    key = _PRODUCT_LIST_SOURCES[tool_name]
    items = result if key is None else (result or {}).get(key)
    if not isinstance(items, list):
        return []

    cards = []
    for item in items:
        # A product whose id did not survive cannot be opened, so it is
        # not offered as something to tap.
        if not isinstance(item, dict) or not item.get("product_id"):
            continue
        cards.append({
            "product_id": item["product_id"],
            "name": item.get("name"),
            "price": item.get("price"),
            "discountedPrice": item.get("discountedPrice"),
        })
    return cards


# Said once per process rather than per answer. A silent "no pictures"
# is exactly the kind of invisible state that wastes an afternoon.
_image_guard_announced = False


def _images_enabled() -> bool:
    global _image_guard_announced
    if get_settings().product_image_query:
        return True
    if not _image_guard_announced:
        _image_guard_announced = True
        logger.warning(
            "product_images_withheld",
            reason="PRODUCT_IMAGE_QUERY is empty, and the catalogue holds "
                   "full-resolution originals - one measured at 8.5MB for a "
                   "46px thumbnail",
            fix="enable Bunny Optimizer on the pull zone, then set "
                "PRODUCT_IMAGE_QUERY='?width=96&quality=75'",
        )
    return False


async def attach_thumbnails(cards: list[dict]) -> list[dict]:
    """One batched query for the images the trimmers threw away.

    THE TRIMMERS ARE RIGHT TO DROP THEM. A model describes a product in
    words and has no use for a CDN URL, so carrying image arrays through
    the whole pipeline would spend tokens on something it ignores - that
    is the 51k-token lesson this module was built around.

    A card is the opposite case: a row with no picture reads as
    unfinished beside a real product screen. So the image is fetched
    once, for the handful of cards actually being sent, at the end of the
    turn rather than on every tool call - one indexed $in over at most
    LLM_LIST_LIMIT_CAP ids.

    A missing thumbnail is never a reason to lose the card: every failure
    here degrades to no image rather than no products.

    NO IMAGE IS SERVED UNTIL THE CDN CAN RESIZE ONE. The catalogue holds
    full-resolution originals: one measured product image is 8.5 MB, a
    6144x8192 photograph, rendered into a 46-pixel square. Eight of those
    is roughly 10 MB for a single answer, on a phone, in a market where
    mobile data is the cost that matters.

    Bunny's Optimizer fixes it at the CDN, and product_image_query is
    where its parameters go. Left empty - which it is until someone
    enables the Optimizer on the pull zone - this returns the cards
    without images rather than shipping the originals. The guard is the
    default because the failure is silent and expensive: the app works
    perfectly and quietly burns a data plan.
    """
    if not cards:
        return cards

    # Checked BEFORE the query, so a withheld image costs no round trip.
    if not _images_enabled():
        return cards

    ids = [oid for oid in (to_object_id(c["product_id"]) for c in cards) if oid is not None]
    if not ids:
        return cards

    try:
        docs = await products_repo.get_products_by_ids(ids)
    except Exception as exc:
        logger.warning("product_thumbnail_lookup_failed", error=str(exc))
        return cards

    suffix = get_settings().product_image_query

    first_image = {}
    for doc in docs:
        images = doc.get("images") or []
        if images and isinstance(images[0], dict) and images[0].get("url"):
            first_image[str(doc.get("_id"))] = images[0]["url"] + suffix

    for card in cards:
        card["image"] = first_image.get(card["product_id"])
    return cards


async def execute_tool(
    tool_name: str, arguments: dict, user_id: str, collector: list | None = None
) -> str:
    if tool_name not in TOOL_REGISTRY:
        logger.warning("unknown_tool_requested", tool_name=tool_name)
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    func, needs_user_id = TOOL_REGISTRY[tool_name]
    kwargs = dict(arguments) if arguments else {}
    if needs_user_id:
        kwargs["user_id"] = user_id  # ALWAYS server-injected, never from the LLM

    # Hard-cap any "limit" argument regardless of what the LLM requested.
    if "limit" in kwargs and isinstance(kwargs["limit"], int):
        kwargs["limit"] = min(kwargs["limit"], LLM_LIST_LIMIT_CAP)

    logger.info("tool_executing", tool_name=tool_name, arguments=arguments)
    try:
        result = await func(**kwargs)
    except Exception as exc:
        logger.error("tool_execution_failed", tool_name=tool_name, error=str(exc))
        return json.dumps(TOOL_FAILED)

    trimmer = _TRIMMERS.get(tool_name)
    if trimmer is not None:
        try:
            # A trimmer may be async when it needs a second query to make
            # the result useful (ID enrichment) rather than only smaller.
            result = (
                await trimmer(result)
                if inspect.iscoroutinefunction(trimmer)
                else trimmer(result)
            )
        except Exception as exc:
            logger.error("tool_result_trim_failed", tool_name=tool_name, error=str(exc))
            return json.dumps(TOOL_FAILED)

    summariser = _SUMMARISERS.get(tool_name)
    if summariser is not None:
        try:
            result = await summariser(result, user_id)
        except Exception as exc:
            # A missing total is worse than no answer only if it lies -
            # so drop the summary and return the list unchanged.
            logger.error("tool_result_summary_failed", tool_name=tool_name, error=str(exc))

    # Products the app can open, collected beside the string the model
    # reads. A no-op when nobody passed a collector, so /chat and the
    # tests take exactly the path they took before.
    if collector is not None:
        cards = _product_cards(tool_name, result)
        if cards:
            collector.append(cards)

    return json.dumps(to_json_safe(result), default=str)
