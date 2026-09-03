"""
Seeds a DEMO database with realistic but entirely fictional the marketplace data.

WHY THIS EXISTS
    Reading the real staging database in tests is harmless - the data
    never leaves Atlas. Sending it to an LLM is a different act: the
    free tiers state that submitted content may be used to improve their
    products and may be reviewed by humans, and Google's terms
    explicitly advise against submitting personal information to
    non-paid services. Staging holds real customers - 134 orders with
    delivery cities, 59 with courier tracking numbers.

    So anything that reaches an LLM (demos, the live test, manual /chat
    poking) should run against this dataset instead. Repo-level tests
    stay pointed at real staging, because they exercise the real schema
    and its real messiness, and nothing they read is transmitted
    anywhere.

WHAT IT WRITES
    Enough to demo every feature area the assistant covers: a buyer with
    orders in every status, bargains mid-negotiation, a cart, coupons,
    live sessions, Bits, reviews, and a SECOND buyer whose order exists
    solely so the cross-user leak test has something real to fail
    against.

    Field names and shapes are copied from the real collections, not
    invented - a demo database with a subtly different schema would
    "work" here and break on staging, which is the worst possible
    outcome for a POC.

SAFETY
    Refuses to run against MONGODB_URI, and refuses any database named
    like the production one. The seeding user needs write access, which
    the read-only chatbot account deliberately does not have.

USAGE
    Set DEMO_MONGODB_URI to a cluster you can write to (a free Atlas M0
    is fine), then:

        uv run python scripts/seed_demo_data.py

    Then point the app at it by setting in .env:

        MONGODB_DATABASE=marketplace_demo

    ...and switch back to `marketplace` for repo tests.
"""

import asyncio
import os
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from urllib.parse import quote_plus  # noqa: E402

from bson import ObjectId  # noqa: E402
from motor.motor_asyncio import AsyncIOMotorClient  # noqa: E402
from pymongo.errors import InvalidURI  # noqa: E402

from app.config.settings import get_settings  # noqa: E402

DEMO_DATABASE = "marketplace_demo"

# The marketplace staging cluster, identified by hostname. Any URI pointing at
# it is refused outright - it holds real customer data, and the chatbot
# account against it is read-only by design.
STAGING_HOST_MARKER = "marketplace-semantic-search"

# Deterministic, so re-seeding produces the same ids and any demo script
# or bookmark keeps working across runs.
random.seed(20260820)


def oid(n: int) -> ObjectId:
    """Stable ObjectId from a small integer - readable in logs, and the
    same on every re-seed."""
    return ObjectId(f"{n:024x}")


NOW = datetime.now(timezone.utc).replace(tzinfo=None)


def days_ago(n: int) -> datetime:
    return NOW - timedelta(days=n)


def days_ahead(n: int) -> datetime:
    return NOW + timedelta(days=n)


# ── Identities ───────────────────────────────────────────────────────
# All fictional. The demo buyer is who you hold a token for.
DEMO_BUYER = oid(1)
OTHER_BUYER = oid(2)
SELLER_A = oid(10)
SELLER_B = oid(11)


def _img(name: str) -> dict:
    return {"public_id": f"demo/{name}", "url": f"https://example.invalid/{name}.jpg"}


USERS = [
    {
        "_id": DEMO_BUYER,
        "username": "asha.demo",
        "profilePic": _img("asha"),
        "categoryType": "buyer",
        "sellerStatus": "none",
        "sellerProfile": {"businessName": ""},
        "followerCount": 4,
        "followers": [OTHER_BUYER],
        "following": [SELLER_A, SELLER_B],
        "likedProducts": [oid(101), oid(104)],
        "savedProducts": [oid(103)],
        "savedBits": [oid(401)],
        "customerRating": 0,
        "reviewsCount": 2,
        "productsSoldCount": 0,
        "monthlyRevenue": 0,
        "yearlyRevenue": 0,
        "createdAt": days_ago(300),
    },
    {
        "_id": OTHER_BUYER,
        "username": "vikram.demo",
        "profilePic": _img("vikram"),
        "categoryType": "buyer",
        "sellerStatus": "none",
        "sellerProfile": {"businessName": ""},
        "followerCount": 1,
        "followers": [],
        "following": [SELLER_A],
        "likedProducts": [],
        "savedProducts": [],
        "savedBits": [],
        "customerRating": 0,
        "reviewsCount": 0,
        "productsSoldCount": 0,
        "monthlyRevenue": 0,
        "yearlyRevenue": 0,
        "createdAt": days_ago(120),
    },
    {
        "_id": SELLER_A,
        "username": "kalaghar",
        "profilePic": _img("kalaghar"),
        "categoryType": "seller",
        "sellerStatus": "approved",
        "sellerProfile": {"businessName": "Kala Ghar Home Decor"},
        "followerCount": 1280,
        "followers": [DEMO_BUYER, OTHER_BUYER],
        "following": [],
        "likedProducts": [],
        "savedProducts": [],
        "savedBits": [],
        "customerRating": 4,
        "reviewsCount": 214,
        "productsSoldCount": 890,
        "monthlyRevenue": 184000,
        "yearlyRevenue": 2100000,
        "createdAt": days_ago(700),
    },
    {
        "_id": SELLER_B,
        "username": "threadline",
        "profilePic": _img("threadline"),
        "categoryType": "seller",
        "sellerStatus": "approved",
        "sellerProfile": {"businessName": "Threadline Apparel"},
        "followerCount": 640,
        "followers": [DEMO_BUYER],
        "following": [],
        "likedProducts": [],
        "savedProducts": [],
        "savedBits": [],
        "customerRating": 4,
        "reviewsCount": 96,
        "productsSoldCount": 410,
        "monthlyRevenue": 96000,
        "yearlyRevenue": 1150000,
        "createdAt": days_ago(500),
    },
]


def product(
    n, name, desc, category, sub, price, discounted, seller, *,
    variants, tags, stock=None, top_pick=False, sold=False, bargain=(10, 25),
):
    auto, maximum = bargain
    return {
        "_id": oid(n),
        "SKU": f"DEMO-{n}",
        "sellerId": seller,
        "name": name,
        "description": desc,
        "category": category,
        "subCategory": sub,
        "categoryType": "physical",
        "productType": {
            "hasSize": any(v.get("size") for v in variants),
            "hasColor": any(v.get("color") for v in variants),
            "hasVariants": bool(variants),
        },
        "price": price,
        "discountedPrice": discounted,
        "totalStock": stock if stock is not None else sum(v["stock"] for v in variants),
        "condition": "new",
        "status": "active",
        "isSold": sold,
        "soldAt": None,
        "isTopPick": top_pick,
        "topPickExpiresAt": None,
        "orderAcceptingType": "instant",
        "images": [_img(f"p{n}")],
        "variants": [
            {**v, "isOutOfStock": v["stock"] <= 0, "images": []} for v in variants
        ],
        "shipping": {
            "freeShipping": discounted >= 999,
            "estimatedDeliveryDays": 5,
            "codAvailable": True,
            "returnPolicy": "7 day return",
        },
        "bargainSettings": {"autoAcceptDiscount": auto, "maximumDiscount": maximum},
        "tags": tags,
        "searchKeywords": tags,
        "comments": [],
        "customSpecs": [],
        "viewCount": random.randint(40, 900),
        "likeCount": random.randint(3, 120),
        "saveCount": random.randint(1, 60),
        "shareCount": random.randint(0, 30),
        "analytics": {
            "totalSales": random.randint(0, 40),
            "totalRevenue": random.randint(0, 60000),
            "averageRating": random.randint(3, 5),
            "totalReviews": random.randint(0, 30),
        },
        "createdAt": days_ago(random.randint(20, 200)),
        "updatedAt": days_ago(random.randint(1, 19)),
    }


PRODUCTS = [
    product(101, "Brass Meditating Buddha Showpiece",
            "Hand-finished brass Buddha for a shelf or entryway.",
            "Home Decor", "Showpieces & Figurines", 2499, 1799, SELLER_A,
            variants=[{"color": "Antique Gold", "size": "Medium", "stock": 12}],
            tags=["buddha", "brass", "showpiece", "statue"], top_pick=True),
    product(102, "12x18 Customised Photo Frame",
            "Solid wood frame, personalised print included.",
            "Home Decor", "Photo Frames", 1299, 899, SELLER_A,
            variants=[{"color": "Walnut", "size": "12x18", "stock": 20},
                      {"color": "Black", "size": "12x18", "stock": 6}],
            tags=["frame", "photo", "wooden", "gift"]),
    product(103, "Minimal Wall Clock",
            "Silent sweep movement, matte dial.",
            "Home Decor", "Clocks", 1899, 1499, SELLER_A,
            variants=[{"color": "Charcoal", "size": "12 inch", "stock": 9}],
            tags=["clock", "wall", "minimal"]),
    product(104, "Terracotta Vase Set",
            "Set of two hand-thrown terracotta vases.",
            "Home Decor", "Table Decor", 1599, 1199, SELLER_A,
            variants=[{"color": "Terracotta", "size": "Set of 2", "stock": 0}],
            tags=["vase", "terracotta", "pottery"]),
    product(105, "Oxford Cotton Formal Shirt",
            "Full-sleeve oxford cotton, wrinkle resistant.",
            "Men's Fashion", "Formal Shirts", 1999, 1299, SELLER_B,
            variants=[{"color": "Sky Blue", "size": "M", "stock": 14},
                      {"color": "Sky Blue", "size": "L", "stock": 8},
                      {"color": "White", "size": "M", "stock": 0}],
            tags=["shirt", "formal", "cotton", "oxford"]),
    product(106, "Relaxed Denim Jacket",
            "Mid-wash denim, relaxed fit, brass buttons.",
            "Men's Fashion", "Winter Wear", 3499, 2499, SELLER_B,
            variants=[{"color": "Indigo", "size": "M", "stock": 5},
                      {"color": "Indigo", "size": "L", "stock": 3}],
            tags=["jacket", "denim", "winter"], top_pick=True),
    product(107, "Everyday Cotton T-Shirt",
            "240 GSM combed cotton, pre-shrunk.",
            "Men's Fashion", "T-shirts & Polos", 899, 599, SELLER_B,
            variants=[{"color": "Black", "size": "M", "stock": 30},
                      {"color": "Olive", "size": "L", "stock": 18}],
            tags=["tshirt", "cotton", "casual"]),
    product(108, "Block Print Cotton Kurti",
            "Hand block printed, three-quarter sleeve.",
            "Women's Fashion", "Kurtis", 1799, 1249, SELLER_B,
            variants=[{"color": "Indigo", "size": "S", "stock": 7},
                      {"color": "Indigo", "size": "M", "stock": 11}],
            tags=["kurti", "cotton", "blockprint", "ethnic"]),
    product(109, "Chikankari Straight Kurta",
            "Lucknowi chikankari on cotton mul.",
            "Women's Fashion", "Ethnic Wear", 2899, 2199, SELLER_B,
            variants=[{"color": "Ivory", "size": "M", "stock": 4}],
            tags=["kurta", "chikankari", "ethnic"]),
    product(110, "Smartwatch Active 2",
            "AMOLED display, 7-day battery, SpO2.",
            "Electronics", "Televisions", 5999, 3999, SELLER_A,
            variants=[{"color": "Midnight", "size": "44mm", "stock": 15}],
            tags=["smartwatch", "watch", "wearable", "fitness"]),
    product(111, "400 TC Cotton Bedsheet Set",
            "King size, sateen weave, two pillow covers.",
            "Bed & Bath", "Bedsheets", 2499, 1699, SELLER_A,
            variants=[{"color": "Sage", "size": "King", "stock": 10}],
            tags=["bedsheet", "cotton", "king"]),
    product(112, "Bamboo Bath Towel",
            "600 GSM bamboo blend, quick dry.",
            "Bed & Bath", "Towels", 1199, 799, SELLER_A,
            variants=[{"color": "Stone", "size": "30x60", "stock": 0}],
            tags=["towel", "bamboo", "bath"], sold=True),
]

CATEGORIES = [
    {"_id": oid(200 + i), "name": name, "slug": name.lower().replace(" ", "-"),
     "image": _img(f"cat{i}"),
     "subCategories": [{"name": s, "slug": s.lower().replace(" ", "-")} for s in subs]}
    for i, (name, subs) in enumerate([
        ("Home Decor", ["Showpieces & Figurines", "Photo Frames", "Clocks", "Table Decor"]),
        ("Men's Fashion", ["Formal Shirts", "T-shirts & Polos", "Winter Wear"]),
        ("Women's Fashion", ["Kurtis", "Ethnic Wear"]),
        ("Electronics", ["Televisions"]),
        ("Bed & Bath", ["Bedsheets", "Towels"]),
    ])
]


# The path an order walks. A status implies every step before it.
STATUS_SEQUENCE = ["pending", "confirmed", "shipped", "delivered"]

STATUS_NOTES = {
    "pending": "Order placed",
    "confirmed": "Payment confirmed, seller notified",
    "shipped": "Handed to courier",
    "delivered": "Delivered",
    "cancelled": "Cancelled by buyer",
}


def _status_history(status: str, days: int) -> list[dict]:
    """Every step up to `status`, timestamped between order date and now.

    Cancellation is not a stage of the sequence - it is a departure from
    it - so a cancelled order shows the steps it did complete, then the
    cancellation.
    """
    if status == "cancelled":
        reached = ["pending", "confirmed"]
    else:
        reached = STATUS_SEQUENCE[:STATUS_SEQUENCE.index(status) + 1]

    steps = len(reached) + (1 if status == "cancelled" else 0)
    # Spread the steps over the order's life so far, leaving the last
    # stage a little short of "now" for anything already finished.
    span = max(days - 1, 1)
    history = [
        {"status": s,
         "timestamp": days_ago(max(days - round(span * i / max(steps - 1, 1)), 0)),
         "note": STATUS_NOTES[s], "_id": ObjectId()}
        for i, s in enumerate(reached)
    ]
    if status == "cancelled":
        history.append({
            "status": "cancelled", "timestamp": days_ago(max(days - 1, 0)),
            "note": STATUS_NOTES["cancelled"], "_id": ObjectId(),
        })
    return history


def _order_dates(status: str, days: int) -> dict:
    """orderPlaced always; expectedDelivery only where it still means
    something.

    A delivered order from five months ago with an expected delivery
    three days from now is the kind of contradiction that survives
    review and then gets read aloud in a demo.
    """
    dates = {"orderPlaced": days_ago(days)}
    if status == "delivered":
        # Landed a few days after it shipped, comfortably in the past.
        dates["expectedDelivery"] = days_ago(max(days - 4, 0))
        dates["delivered"] = days_ago(max(days - 4, 0))
    elif status == "cancelled":
        pass  # never had one that mattered
    else:
        dates["expectedDelivery"] = days_ahead(random.randint(2, 6))
    return dates


def order(n, buyer, seller, status, items, *, invoice=False, days=10,
          courier=None, cancelled=False, city=None):
    subtotal = sum(i["total"] for i in items)
    doc = {
        "_id": oid(n),
        "orderId": f"ZTC{n:06d}",
        "buyerId": buyer,
        "sellerId": seller,
        "sellerIds": [seller],
        "liveSessionId": None,
        "items": items,
        "orderType": "regular",
        "deliveryType": "standard",
        "pricing": {
            "subtotal": subtotal, "discount": 0, "shipping": 0,
            "tax": round(subtotal * 0.05), "total": subtotal + round(subtotal * 0.05),
        },
        "status": status,
        # THE TRAIL, NOT JUST THE DESTINATION. orders_repo surfaces
        # statusHistory to the LLM, so an order that reached "delivered"
        # while its history says only "pending" produces an answer that
        # contradicts itself mid-sentence.
        "statusHistory": _status_history(status, days),
        "dates": _order_dates(status, days),
        "review": {"images": []},
        "deliveryAddress": dict(city or {"city": "Pune", "state": "Maharashtra"}),
        "createdAt": days_ago(days),
        "updatedAt": days_ago(max(days - 2, 0)),
    }
    if cancelled:
        doc["dates"]["cancelled"] = days_ago(max(days - 1, 0))
    if courier:
        doc["tracking"] = {
            "courier": courier,
            "awb": f"DEMO{n}{random.randint(1000, 9999)}",
            "trackingUrl": f"https://example.invalid/track/DEMO{n}",
            "estimatedDelivery": days_ahead(3),
        }
    if invoice:
        doc["invoice"] = {
            "url": f"https://example.invalid/invoice/ZTC{n:06d}.pdf",
            "generatedAt": days_ago(max(days - 3, 0)),
        }
    return doc


def item(product_doc, qty=1, variant_index=0):
    v = product_doc["variants"][variant_index]
    price = product_doc["discountedPrice"]
    return {
        "product": product_doc["_id"], "name": product_doc["name"],
        "image": product_doc["images"][0]["url"],
        "variant": {"color": v.get("color"), "size": v.get("size")},
        "qty": qty, "price": price, "total": price * qty,
        "bitId": None, "sellerId": product_doc["sellerId"],
        "bargainId": None, "_id": ObjectId(),
    }


P = {p["_id"]: p for p in PRODUCTS}

ORDERS = [
    order(301, DEMO_BUYER, SELLER_A, "pending", [item(P[oid(101)])], days=1),
    order(302, DEMO_BUYER, SELLER_B, "confirmed", [item(P[oid(105)], 2)], days=4),
    order(303, DEMO_BUYER, SELLER_B, "shipped", [item(P[oid(106)])], days=7,
          courier="Delhivery"),
    order(304, DEMO_BUYER, SELLER_A, "delivered", [item(P[oid(103)])], days=25,
          courier="Blue Dart", invoice=True),
    order(305, DEMO_BUYER, SELLER_A, "delivered",
          [item(P[oid(111)]), item(P[oid(112)])], days=60, courier="Ekart"),
    order(306, DEMO_BUYER, SELLER_B, "cancelled", [item(P[oid(107)])], days=40,
          cancelled=True),
    # Belongs to someone else on purpose: the cross-user security test
    # needs a real order that must NEVER surface for the demo buyer.
    order(310, OTHER_BUYER, SELLER_A, "shipped", [item(P[oid(110)])], days=3,
          courier="Delhivery"),
]

# ── Order history with some depth to it ──────────────────────────────
# WHY THIS IS GENERATED RATHER THAN LISTED. The seven orders above are
# hand-written because each one pins a specific case: every status the
# app handles, a cancellation, a multi-item order, and the cross-user
# fixture. Those must not drift, so they stay explicit.
#
# What they cannot give you is HISTORY. "What did I buy in June?" and
# "how much have I spent?" both need a spread of dates and amounts, and
# seven orders clustered in the last two months answer neither - the
# assistant returns something technically correct and visibly thin.
#
# So the rest is generated across six months. Deterministic: random is
# seeded at the top of this file, so re-running produces the same demo
# and a rehearsed script stays valid.
DELIVERY_CITIES = [
    {"city": "Pune", "state": "Maharashtra"},
    {"city": "Mumbai", "state": "Maharashtra"},
    {"city": "Bengaluru", "state": "Karnataka"},
    {"city": "Hyderabad", "state": "Telangana"},
    {"city": "Jaipur", "state": "Rajasthan"},
    {"city": "Kolkata", "state": "West Bengal"},
]
COURIERS = ["Delhivery", "Blue Dart", "Ekart", "XpressBees", "DTDC"]

# Sellable products only. A "delivered" order for an item marked sold
# out is fine (it sold), but building every order from the same handful
# makes the history read as obviously fake.
_SELLABLE = [p for p in PRODUCTS if not p.get("isSold")]


def _status_for_age(days: int) -> str:
    """What an order of this age would plausibly be.

    ORDER STATUS IS A FUNCTION OF TIME, and a demo that ignores that
    looks wrong to anyone who reads it closely - a "pending" order from
    four months ago is the kind of detail a client notices while you are
    explaining something else.
    """
    if days <= 2:
        return random.choice(["pending", "confirmed"])
    if days <= 6:
        return random.choice(["confirmed", "shipped"])
    if days <= 12:
        return random.choice(["shipped", "delivered"])
    # Older than a fortnight: delivered, with the occasional cancellation.
    return "cancelled" if random.random() < 0.12 else "delivered"


def _generated_orders(start_n: int, count: int) -> list[dict]:
    out = []
    # Spread across ~6 months, oldest first, with jitter so the dates do
    # not fall on a suspiciously regular cadence.
    for i in range(count):
        n = start_n + i
        # WEIGHTED TOWARDS RECENT, not spread evenly. Two reasons, and
        # they agree: people order more in the recent past than six
        # months ago, and an even spread leaves almost everything
        # "delivered" - which starves the single most-demoed question,
        # "where's my order?", of anything actually in transit.
        #
        # The exponent is what does it: squaring the tail compresses the
        # last quarter of the sequence into the last fortnight, so
        # roughly a third of the history is still moving.
        fraction = i / max(count - 1, 1)
        days = int(175 * ((1 - fraction) ** 2.2)) + random.randint(-3, 3)
        days = max(days, 1)
        status = _status_for_age(days)

        # Most orders are one item; a few are not, because real baskets
        # are not uniform and the UI needs to handle both.
        qty_of_items = 1 if random.random() < 0.72 else 2
        chosen = random.sample(_SELLABLE, qty_of_items)
        items = [
            item(p, qty=1 if random.random() < 0.8 else 2) for p in chosen
        ]

        delivered = status == "delivered"
        shipped_or_later = status in ("shipped", "delivered")
        out.append(
            order(
                n, DEMO_BUYER, chosen[0]["sellerId"], status, items,
                days=days,
                # Only orders that actually moved have a courier, and
                # only completed ones have an invoice. Attaching both to
                # everything is the tell that data was generated.
                courier=random.choice(COURIERS) if shipped_or_later else None,
                invoice=delivered,
                cancelled=status == "cancelled",
                city=random.choice(DELIVERY_CITIES),
            )
        )
    return out


ORDERS += _generated_orders(311, 21)

BARGAINS = [
    {
        "_id": oid(320), "productId": oid(106), "buyerId": DEMO_BUYER,
        "sellerId": SELLER_B, "originalPrice": 2499, "offeredPrice": 1999,
        "currentPrice": 1999, "discountPercentage": 20.0, "status": "pending",
        "productSnapshot": {"name": P[oid(106)]["name"],
                            "image": P[oid(106)]["images"][0]["url"]},
        "variant": {"color": "Indigo", "size": "M"}, "quantity": 1,
        "orderPlaced": False, "expiresAt": days_ahead(2), "autoAccepted": False,
        "createdAt": days_ago(1), "updatedAt": days_ago(1),
    },
    {
        "_id": oid(321), "productId": oid(101), "buyerId": DEMO_BUYER,
        "sellerId": SELLER_A, "originalPrice": 1799, "offeredPrice": 1200,
        "currentPrice": 1450, "discountPercentage": 19.4, "status": "countered",
        "productSnapshot": {"name": P[oid(101)]["name"],
                            "image": P[oid(101)]["images"][0]["url"]},
        "variant": {"color": "Antique Gold", "size": "Medium"}, "quantity": 1,
        "orderPlaced": False, "expiresAt": days_ahead(1), "autoAccepted": False,
        "counterOffer": {"price": 1450, "note": "Best I can do on this piece.",
                         "at": days_ago(1)},
        "respondedAt": days_ago(1), "createdAt": days_ago(2), "updatedAt": days_ago(1),
    },
    {
        "_id": oid(322), "productId": oid(108), "buyerId": DEMO_BUYER,
        "sellerId": SELLER_B, "originalPrice": 1249, "offeredPrice": 1120,
        "currentPrice": 1120, "discountPercentage": 10.3, "status": "accepted",
        "productSnapshot": {"name": P[oid(108)]["name"],
                            "image": P[oid(108)]["images"][0]["url"]},
        "variant": {"color": "Indigo", "size": "M"}, "quantity": 1,
        "orderPlaced": False, "expiresAt": days_ahead(3), "autoAccepted": True,
        "respondedAt": days_ago(2), "createdAt": days_ago(2), "updatedAt": days_ago(2),
    },
]

CARTS = [{
    "_id": oid(330), "user": DEMO_BUYER,
    "items": [
        {"product": oid(102), "variant": {"color": "Walnut", "size": "12x18"},
         "qty": 2, "cartPrice": 899, "bargainId": None,
         "_id": ObjectId(), "addedAt": days_ago(1)},
        {"product": oid(109), "variant": {"color": "Ivory", "size": "M"},
         "qty": 1, "cartPrice": 2199, "bargainId": None,
         "_id": ObjectId(), "addedAt": days_ago(0)},
    ],
    "coupon": None, "discount": 0, "liveSessionId": None,
    "createdAt": days_ago(3), "updatedAt": NOW,
}]

COUPONS = [
    {"_id": oid(340), "code": "DEMO10", "name": "10% off Home Decor",
     "discountType": "percentage", "discountValue": 10, "maxDiscount": 500,
     "minSpend": 999, "startDate": days_ago(10), "endDate": days_ahead(20),
     "applicableProducts": [oid(101), oid(102), oid(103)], "sellerId": SELLER_A,
     "isActive": True, "views": 240, "viewsThisWeek": 31, "ordersUsed": 12,
     "totalDiscountGiven": 4200, "maxUsagePerUser": 1,
     "lastViewReset": days_ago(7), "usedBy": [],
     "createdAt": days_ago(10), "updatedAt": days_ago(1)},
    {"_id": oid(341), "code": "EXPIRED5", "name": "Old winter offer",
     "discountType": "percentage", "discountValue": 5, "maxDiscount": None,
     "minSpend": 499, "startDate": days_ago(120), "endDate": days_ago(60),
     "applicableProducts": [], "sellerId": SELLER_B, "isActive": False,
     "views": 90, "viewsThisWeek": 0, "ordersUsed": 40,
     "totalDiscountGiven": 3100, "maxUsagePerUser": 1,
     "lastViewReset": days_ago(60), "usedBy": [],
     "createdAt": days_ago(120), "updatedAt": days_ago(60)},
]

LIVESESSIONS = [
    {"_id": oid(350), "channelName": "kalaghar-live", "hostId": SELLER_A,
     "queuePosition": None, "title": "Festive Home Decor Drop",
     "description": "Brass, frames and clocks - live picks.",
     "scheduledStartTime": days_ago(0), "viewerActivity": {}, "duration": 0,
     "status": "live", "viewersCount": 184, "products": [oid(101), oid(103)],
     "productSequence": [oid(101), oid(103), oid(104)], "hashtags": ["homedecor", "festive"],
     "thumbnail": _img("live350"), "peakViewers": 210, "revenue": 18400, "views": 940,
     "isActive": True, "stepData": {"step1": {"products": []}}, "likes": [], "likeCount": 132,
     "comments": [{"username": "asha.demo", "text": "Is the Buddha still available?"},
                  {"username": "vikram.demo", "text": "Clock looks great"}],
     "startTime": days_ago(0), "endTime": None, "isTrending": True,
     "createdAt": days_ago(2), "updatedAt": NOW},
    {"_id": oid(351), "channelName": "threadline-live", "hostId": SELLER_B,
     "queuePosition": None, "title": "Winter Layering Edit",
     "description": "Jackets and shirts.", "scheduledStartTime": days_ago(6),
     "viewerActivity": {}, "duration": 3600, "status": "ended", "viewersCount": 0,
     "products": [oid(106), oid(105)], "productSequence": [oid(106), oid(105)],
     "hashtags": ["winter", "menswear"], "thumbnail": _img("live351"),
     "peakViewers": 96, "revenue": 22300, "views": 610, "isActive": False,
     "stepData": {"step1": {"products": []}}, "likes": [], "likeCount": 74,
     "comments": [{"username": "asha.demo", "text": "Loved the denim jacket"}],
     "startTime": days_ago(6), "endTime": days_ago(6), "isTrending": False,
     "createdAt": days_ago(8), "updatedAt": days_ago(6)},
]

BITS = [
    {"_id": oid(401), "title": "Styling a brass Buddha", "description": "Three shelf looks.",
     "video": _img("bit401"), "thumbnail": _img("bit401t"),
     "hashtags": ["#homedecor", "#styling"], "products": [oid(101), oid(104)],
     "userId": SELLER_A, "likeCount": 420, "viewCount": 8100, "shareCount": 60,
     "isActive": True, "isTrending": True, "createdAt": days_ago(5)},
    {"_id": oid(402), "title": "Denim jacket, 3 ways", "description": "Winter layering.",
     "video": _img("bit402"), "thumbnail": _img("bit402t"),
     "hashtags": ["#winter", "#menswear", "#styling"], "products": [oid(106)],
     "userId": SELLER_B, "likeCount": 260, "viewCount": 5400, "shareCount": 31,
     "isActive": True, "isTrending": True, "createdAt": days_ago(9)},
    {"_id": oid(403), "title": "Block print, up close", "description": "How it's made.",
     "video": _img("bit403"), "thumbnail": _img("bit403t"),
     "hashtags": ["#ethnic", "#handmade"], "products": [oid(108), oid(109)],
     "userId": SELLER_B, "likeCount": 130, "viewCount": 2200, "shareCount": 12,
     "isActive": True, "isTrending": False, "createdAt": days_ago(14)},
]

REVIEWS = [
    {"_id": oid(420), "productId": oid(103), "reviewerId": DEMO_BUYER, "rating": 5,
     "comment": "Silent as promised, looks smart on the wall.",
     "createdAt": days_ago(20), "updatedAt": days_ago(20)},
    {"_id": oid(421), "productId": oid(111), "reviewerId": DEMO_BUYER, "rating": 4,
     "comment": "Soft cotton, colour slightly lighter than the photo.",
     "createdAt": days_ago(55), "updatedAt": days_ago(55)},
    {"_id": oid(422), "productId": oid(101), "reviewerId": OTHER_BUYER, "rating": 5,
     "comment": "Beautiful finish, heavier than expected in a good way.",
     "createdAt": days_ago(30), "updatedAt": days_ago(30)},
]

ADDRESSES = [
    {"_id": oid(430), "user": DEMO_BUYER, "type": "home", "label": "Home",
     "city": "Pune", "state": "Maharashtra", "pincode": "411001",
     "isDefault": True, "createdAt": days_ago(200), "updatedAt": days_ago(200)},
    {"_id": oid(431), "user": DEMO_BUYER, "type": "work", "label": "Office",
     "city": "Pune", "state": "Maharashtra", "pincode": "411045",
     "isDefault": False, "createdAt": days_ago(150), "updatedAt": days_ago(150)},
    {"_id": oid(432), "user": OTHER_BUYER, "type": "home", "label": "Home",
     "city": "Nagpur", "state": "Maharashtra", "pincode": "440001",
     "isDefault": True, "createdAt": days_ago(100), "updatedAt": days_ago(100)},
]

NOTIFICATIONS = [
    {"_id": oid(440), "userId": DEMO_BUYER, "type": "order",
     "title": "Your order has shipped",
     "message": "ZTC000303 is on its way with Delhivery.",
     "actionUrl": "/orders/ZTC000303", "actionLabel": "Track",
     "isRead": False, "createdAt": days_ago(2), "updatedAt": days_ago(2)},
    {"_id": oid(441), "userId": DEMO_BUYER, "type": "bargain",
     "title": "Seller countered your offer",
     "message": "Kala Ghar countered at Rs.1450.",
     "actionUrl": "/bargains", "actionLabel": "View",
     "isRead": False, "createdAt": days_ago(1), "updatedAt": days_ago(1)},
    {"_id": oid(442), "userId": DEMO_BUYER, "type": "promo",
     "title": "10% off Home Decor", "message": "Use DEMO10 before it expires.",
     "actionUrl": "/coupons", "actionLabel": "Shop",
     "isRead": True, "createdAt": days_ago(6), "updatedAt": days_ago(6)},
]

PAYOUTS = [
    {"_id": oid(450), "orderRef": "ZTC000304", "sellerId": SELLER_A,
     "orderTotal": 1574, "commission": 157, "sellerAmount": 1417,
     "status": "paid", "payoutMode": "bank",
     "createdAt": days_ago(20), "updatedAt": days_ago(18)},
    {"_id": oid(451), "orderRef": "ZTC000303", "sellerId": SELLER_B,
     "orderTotal": 2624, "commission": 262, "sellerAmount": 2362,
     "status": "pending", "payoutMode": "bank",
     "createdAt": days_ago(5), "updatedAt": days_ago(5)},
]


def _synthetic_embedding(doc: dict, dims: int = 384) -> list[float]:
    """A deterministic 384-dim vector built from a product's category and
    tags.

    THIS IS NOT A REAL EMBEDDING MODEL, and must not be described as one.
    the marketplace's pipeline produces the genuine vectors and nothing records
    which model it uses, so we cannot reproduce them. What this does is
    encode REAL structure - category, subCategory, tags - into a vector
    so that products which genuinely belong together score close under
    cosine similarity, and unrelated ones do not.

    That is enough for `find_similar_products` to be demonstrable and
    testable on demo data. It is NOT enough to judge the quality of
    the marketplace's actual semantic search, and it would be dishonest to present
    a demo run on these vectors as evidence of that.

    Same dimension and metric as the real index (384, cosine), so the
    query path being exercised is identical.
    """
    import hashlib
    import math

    vec = [0.0] * dims
    weighted = (
        [(1.0, doc["category"]), (0.8, doc["subCategory"])]
        + [(0.6, t) for t in doc.get("tags", [])]
    )
    for weight, token in weighted:
        h = hashlib.sha256(token.lower().encode()).digest()
        for i in range(8):
            idx = int.from_bytes(h[i * 2:i * 2 + 2], "big") % dims
            vec[idx] += weight * (1 if h[i] % 2 == 0 else -1)

    # Per-product jitter, so no two products are exactly identical.
    hp = hashlib.sha256(str(doc["_id"]).encode()).digest()
    for i in range(16):
        idx = int.from_bytes(hp[i * 2:i * 2 + 2], "big") % dims
        vec[idx] += 0.05 * (1 if hp[i] % 2 == 0 else -1)

    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


PRODUCT_EMBEDDINGS = [
    {"_id": p["_id"], "embedding": _synthetic_embedding(p),
     "embeddingHash": f"demo-{p['_id']}", "updatedAt": NOW}
    for p in PRODUCTS
]

COLLECTIONS = {
    "users": USERS, "products": PRODUCTS, "categories": CATEGORIES,
    "orders": ORDERS, "bargains": BARGAINS, "carts": CARTS,
    "coupons": COUPONS, "livesessions": LIVESESSIONS, "bits": BITS,
    "reviews": REVIEWS, "addresses": ADDRESSES, "notifications": NOTIFICATIONS,
    "payouts": PAYOUTS, "product_embeddings": PRODUCT_EMBEDDINGS,
}

TEXT_INDEX_DEFINITION = {
    "mappings": {
        "dynamic": False,
        "fields": {
            "name": {"type": "string"},
            "category": {"type": "string"},
            "subCategory": {"type": "string"},
            "tags": {"type": "string"},
            "searchKeywords": {"type": "string"},
            "variants": {"type": "document", "fields": {"color": {"type": "string"}}},
        },
    }
}


def _escape_userinfo(uri: str) -> str:
    """Percent-encodes the username and password inside a Mongo URI.

    WHY THIS IS NEEDED. A password containing @ : / ? # or % is legal as
    a password and illegal as raw URI text - pymongo rejects the whole
    string with "must be escaped according to RFC 3986", which reads
    like the URI is malformed rather than like one character needs
    encoding. Rather than send someone away to urllib.quote_plus their
    own password, we do it here.

    ONLY CALLED AFTER PARSING HAS ALREADY FAILED. Escaping an already
    correct URI would double-encode it (%40 -> %2540) and break a
    working credential, so this never runs on the happy path.

    Splits on the LAST "@": a host never contains one, but a password
    very well might, so splitting on the first would cut in the wrong
    place.
    """
    scheme, _, rest = uri.partition("://")
    if not rest or "@" not in rest:
        return uri
    userinfo, _, hostpart = rest.rpartition("@")
    user, sep, password = userinfo.partition(":")
    if not sep:
        return uri
    return f"{scheme}://{quote_plus(user)}:{quote_plus(password)}@{hostpart}"


def _target_uri() -> str:
    # Accepted either way. The env var keeps the credential out of shell
    # history; the argument avoids a shell-specific export step, which
    # differs between PowerShell and bash and is easy to get wrong.
    uri = (
        sys.argv[1] if len(sys.argv) > 1 else os.environ.get("DEMO_MONGODB_URI", "")
    ).strip()
    if not uri:
        raise SystemExit(
            "No demo cluster URI given.\n\n"
            "Pass it as an argument:\n"
            '    uv run python scripts/seed_demo_data.py "mongodb+srv://USER:PASS@host/"\n\n'
            "...or set DEMO_MONGODB_URI first.\n\n"
            "It must point at a cluster you can WRITE to - a free Atlas M0\n"
            "works - and must NOT be the marketplace staging cluster: that account\n"
            "is read-only by design, and staging holds real customer data."
        )

    # Refuse the real cluster even if someone pastes it in by mistake.
    #
    # KEYED ON THE STAGING HOSTNAME, NOT ON .env. The first version of
    # this compared the target against settings.mongodb_uri, which was
    # wrong the moment .env was legitimately repointed at the demo
    # cluster: the guard then read the demo URI as "the real one" and
    # blocked re-seeding. A guard that fires on correct usage gets
    # disabled by whoever hits it, which is worse than no guard.
    #
    # The staging host does not move, so matching on it is both stable
    # and independent of however .env happens to be configured today.
    host = _host_of(uri)
    if STAGING_HOST_MARKER in host:
        raise SystemExit(
            f"That URI points at the marketplace staging cluster ({host}).\n"
            "Refusing to seed - it holds real customer data and the\n"
            "chatbot account is read-only by design. Use a separate cluster."
        )

    if uri.strip() == get_settings().mongodb_uri.strip():
        # Not a problem: it just means the app is already pointed at the
        # cluster being seeded, which is exactly the demo setup.
        print("note: seeding the same cluster the app currently reads\n")
    return uri



def _host_of(uri: str) -> str:
    """The HOSTNAME only - no credentials, no path, no query string.

    The query string matters here. A URI like

        mongodb+srv://u:p@cluster0.example.net?appName=the marketplace-Semantic-Search

    has no slash before "?", so splitting on "/" alone leaves the appName
    glued to the host - and a guard looking for the staging cluster's name
    then matched a COMPLETELY DIFFERENT cluster whose appName happened to
    mention it. The guard refused a legitimate destination and blamed the
    user. Strip "?" as well as "/".
    """
    return uri.split("@")[-1].split("/")[0].split("?")[0].lower()

async def main() -> None:
    uri = _target_uri()
    try:
        client = AsyncIOMotorClient(uri)
    except InvalidURI:
        # Almost always a password with a reserved character in it.
        escaped = _escape_userinfo(uri)
        if escaped == uri:
            raise
        print("note: percent-encoded the username/password for you\n")
        client = AsyncIOMotorClient(escaped)
    db = client[DEMO_DATABASE]

    await client.admin.command("ping")
    print(f"connected -> database '{DEMO_DATABASE}'\n")

    for name, docs in COLLECTIONS.items():
        await db[name].delete_many({})
        if docs:
            await db[name].insert_many(docs)
        print(f"  {name:<14} {len(docs):>3} docs")

    # Indexes the app actually relies on. products has only _id on
    # staging, which is fine at 143 rows and not fine later - seeding
    # them here also documents what production should have.
    print("\nindexes:")
    await db.orders.create_index([("buyerId", 1), ("createdAt", -1)])
    await db.products.create_index([("category", 1), ("isSold", 1)])
    await db.bargains.create_index([("buyerId", 1), ("productId", 1)])
    await db.carts.create_index([("user", 1)])
    await db.reviews.create_index([("productId", 1)])
    print("  regular indexes created")

    # Atlas Search. Needs Atlas (any tier incl. free M0); a plain
    # mongod does not have it, and the repo falls back to a regex scan
    # in that case - slower, still correct.
    try:
        await db.products.create_search_index(
            {"name": "product_text_index", "definition": TEXT_INDEX_DEFINITION}
        )
        print("  product_text_index requested (takes ~1 min to build)")
        await db.product_embeddings.create_search_index(
            {
                "name": "product_vector_index",
                "type": "vectorSearch",
                "definition": {
                    "fields": [
                        {"type": "vector", "path": "embedding",
                         "numDimensions": 384, "similarity": "cosine"}
                    ]
                },
            }
        )
        print("  product_vector_index requested (takes ~1 min to build)")
    except Exception as exc:
        print(f"  search indexes NOT created: {type(exc).__name__}")
        print("    -> search_products_by_name falls back to a regex scan.")
        print("    -> Expected on a non-Atlas server; fine for a demo.")

    print(
        "\nDone. Point the app at it with:\n"
        f"    MONGODB_URI=<this demo cluster>\n"
        f"    MONGODB_DATABASE={DEMO_DATABASE}\n\n"
        f"Demo buyer id: {DEMO_BUYER}\n"
        "Generate a token for them with scripts/generate_test_token.py\n"
        "(it reads whichever database is configured).\n\n"
        "NOTE: the product embeddings here are SYNTHETIC - derived from\n"
        "category and tags, NOT from the marketplace's model (nothing records which\n"
        "model that is). Similar-product search is therefore demonstrable\n"
        "and testable on this data, but a demo run on these vectors is NOT\n"
        "evidence about the quality of the marketplace's real semantic search."
    )
    client.close()


asyncio.run(main())
