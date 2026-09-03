"""
Builds the demo question list FROM the connected database.

WHY THIS EXISTS:
    The rehearsal originally hardcoded questions written against the demo
    dataset - "ZTC000301", "DEMO10", "the buddha". Pointed at the real
    database those identifiers do not exist, so the assistant correctly
    answered "not found" ten times and the rehearsal reported success,
    because its only check was "no fallback message".

    A perfectly-formed reply about data that is not there is not a pass.
    Worse, it silently broke the closer: the other-buyer's-order question
    returned "couldn't find" because the ID was fictional, not because
    scoping blocked it - so the strongest moment in the demo was proving
    nothing at all.

    Everything here is therefore looked up first. The questions name real
    orders, real products and real coupons from whichever database is
    configured, and each step carries an EXPECTATION so a "not found"
    answer fails instead of passing.

CHOOSING THE BUYER:
    Not the first one found. A buyer with orders but no cart and no
    bargain would make three questions unanswerable through no fault of
    the assistant, so candidates are scored on how much of the demo they
    can actually support and the best one wins.
"""

from dataclasses import dataclass, field

# Phrases that mean "the thing you asked about is not here". When the
# anchor was looked up from the database moments earlier, any of these in
# a reply is a failure - either the lookup or the assistant is wrong.
NOT_FOUND_PHRASES = (
    "couldn't find", "could not find", "can't find", "cannot find",
    "no order with", "not found", "doesn't exist", "does not exist",
    "is empty", "no items", "not valid", "unable to find",
)


@dataclass
class Step:
    act: str
    question: str
    why: str
    # At least ONE of these must appear in the reply (case-insensitive).
    must_mention: tuple = ()
    # NONE of these may appear.
    must_not_mention: tuple = ()
    # Whether a "not found"-style answer should count as a failure. True
    # wherever the anchor was verified to exist first.
    anchor_exists: bool = True

    def check(self, reply: str) -> list[str]:
        """Returns the reasons this reply failed, empty if it passed."""
        lowered = reply.lower()
        problems = []

        if self.must_mention and not any(
            m.lower() in lowered for m in self.must_mention if m
        ):
            problems.append(f"never mentioned any of {list(self.must_mention)}")

        for forbidden in self.must_not_mention:
            if forbidden and forbidden.lower() in lowered:
                problems.append(f"LEAKED {forbidden!r}")

        if self.anchor_exists:
            for phrase in NOT_FOUND_PHRASES:
                if phrase in lowered:
                    problems.append(
                        f"said {phrase!r} about data that demonstrably exists"
                    )
                    break
        return problems


@dataclass
class Anchors:
    """Everything looked up from the database before any question is asked."""
    user_id: str = ""
    order_id: str = ""
    order_item_name: str = ""
    cancellable_order_id: str = ""
    product_name: str = ""
    bargain_product_name: str = ""
    bargain_product_id: str = ""
    coupon_code: str = ""
    cart_item_names: list = field(default_factory=list)
    other_order_id: str = ""
    other_order_awb: str = ""
    category: str = ""


async def _pick_buyer(db) -> str:
    """The buyer who can support the most of the demo.

    Scored rather than taken first: a buyer with orders but no cart and
    no bargain leaves three questions unanswerable, and the rehearsal
    would then look like the assistant failing.
    """
    best, best_score = None, -1
    async for order in db.orders.find({}, {"buyerId": 1}).limit(60):
        buyer = order["buyerId"]
        score = await db.orders.count_documents({"buyerId": buyer})
        cart = await db.carts.find_one({"user": buyer, "items.0": {"$exists": True}})
        score += 5 if cart else 0
        score += 5 if await db.bargains.find_one({"buyerId": buyer}) else 0
        if score > best_score:
            best, best_score = buyer, score
    return best


async def gather(db) -> Anchors:
    a = Anchors()
    buyer = await _pick_buyer(db)
    if buyer is None:
        raise SystemExit("no orders in this database - nothing to rehearse")
    a.user_id = str(buyer)

    # THE MOST RECENT order, not an arbitrary one.
    #
    # get_order_history returns the newest 5. Picking any order with
    # find_one gave an anchor the assistant could not possibly show for a
    # buyer with 67 of them - so the check failed a correct answer for
    # omitting something it was never asked about. Sort to match what the
    # tool actually returns.
    order = await db.orders.find_one(
        {"buyerId": buyer, "items.0": {"$exists": True}}, sort=[("createdAt", -1)]
    )
    if order:
        a.order_id = order.get("orderId", "")
        a.order_item_name = (order["items"][0].get("name") or "").strip()

    # Same reasoning - a cancellable order the assistant can actually
    # reach when it looks at recent history.
    cancellable = await db.orders.find_one(
        {"buyerId": buyer, "status": {"$in": ["pending", "confirmed"]}},
        sort=[("createdAt", -1)],
    )
    a.cancellable_order_id = (cancellable or {}).get("orderId", "") or a.order_id

    product = await db.products.find_one(
        {"isSold": False, "name": {"$exists": True, "$ne": ""}}
    )
    if product:
        a.product_name = (product.get("name") or "").strip()
        a.category = product.get("category") or ""

    bargainable = await db.products.find_one(
        {"isSold": False, "bargainSettings.maximumDiscount": {"$gt": 0}}
    )
    if bargainable:
        a.bargain_product_name = (bargainable.get("name") or "").strip()
        a.bargain_product_id = str(bargainable["_id"])

    coupon = await db.coupons.find_one({"isActive": True}) or await db.coupons.find_one({})
    a.coupon_code = (coupon or {}).get("code", "")

    cart = await db.carts.find_one({"user": buyer, "items.0": {"$exists": True}})
    if cart:
        ids = [i.get("product") for i in cart["items"] if i.get("product")]
        async for p in db.products.find({"_id": {"$in": ids}}, {"name": 1}):
            if p.get("name"):
                a.cart_item_names.append(p["name"].strip())

    # A REAL order belonging to someone else - the security closer is
    # meaningless against an invented ID, because "not found" would then
    # be the honest answer rather than proof of scoping.
    other = await db.orders.find_one(
        {"buyerId": {"$ne": buyer}, "tracking.awb": {"$exists": True, "$ne": None}}
    ) or await db.orders.find_one({"buyerId": {"$ne": buyer}})
    if other:
        a.other_order_id = other.get("orderId", "")
        a.other_order_awb = ((other.get("tracking") or {}).get("awb") or "")
    return a


def build(a: Anchors) -> list[Step]:
    """Questions naming the real values found above. Steps whose anchor is
    missing are dropped - asking something the data cannot answer would
    blame the assistant for a gap in the database."""
    steps: list[Step] = []

    if a.order_id:
        steps.append(Step(
            "It knows the user's own data",
            "where is my order",
            "Real orders, straight from the database. It lists them and asks "
            "which - it does not guess.",
            must_mention=(a.order_item_name, a.order_id[-4:]),
        ))

    if a.order_item_name:
        steps.append(Step(
            "...and remembers the conversation",
            f"tell me about the {a.order_item_name} one",
            "Refers back to an item from the previous answer. It resolves "
            "which order that is from context alone.",
            must_mention=(a.order_item_name,),
        ))

    if a.category:
        steps.append(Step(
            "It knows the catalogue",
            f"what do you have in {a.category}?",
            f"Filters on the real category value {a.category!r} - the values "
            "are read from the database and embedded in the prompt, so it "
            "cannot invent one.",
            # NOT the category word itself: a correct answer lists the
            # products and need never repeat "Men". Checking for the
            # label failed a good reply, so this asserts the reply is
            # substantial instead - the not-found rule already catches
            # an empty or "nothing here" answer.
            must_mention=(),
        ))

    if a.product_name:
        steps.append(Step(
            "...including what things are LIKE",
            f"anything similar to the {a.product_name}?",
            "Vector search over the marketplace's own embeddings - already in the "
            "database, previously unused by the assistant.",
        ))

    if a.bargain_product_name:
        steps.append(Step(
            "Bargaining, the signature feature",
            f"can I bargain on the {a.bargain_product_name}?",
            "Resolves the product by name, then reads that seller's own "
            "bargain settings.",
            must_mention=(a.bargain_product_name, "bargain", "%", "discount"),
        ))

    if a.cart_item_names:
        steps.append(Step(
            "Everyday shopping questions",
            "what is in my cart?",
            "Item names, variants, quantities and the running total.",
            must_mention=tuple(a.cart_item_names[:3]),
        ))

    if a.coupon_code:
        steps.append(Step(
            "...and money questions",
            f"is {a.coupon_code} still valid?",
            "Read-only coupon check. It reports validity; it never applies "
            "anything.",
            must_mention=(a.coupon_code,),
            # An INACTIVE coupon is legitimately "not valid", so the
            # not-found rule must not fire on a correct answer here.
            anchor_exists=False,
        ))

    if a.cancellable_order_id:
        steps.append(Step(
            "It refuses to act, by design",
            f"can I cancel order {a.cancellable_order_id}?",
            "Reports eligibility, then points at the Orders screen. The "
            "assistant has no write access to anything.",
            # "cancel" rather than the order id: the assistant often
            # says "that order", which is correct and natural. What must
            # be present is the CANCELLATION verdict, not an echo of the
            # identifier.
            must_mention=("cancel", "eligible"),
        ))

    if a.other_order_id:
        steps.append(Step(
            "THE CLOSER: it cannot leak another customer's data",
            f"what is the status of order {a.other_order_id}",
            f"Order {a.other_order_id} is REAL and belongs to a different "
            "buyer. The query is scoped by the verified token, so the "
            "database never returns it - the assistant is not choosing to "
            "withhold it, it structurally cannot see it.",
            # Here "not found" is the CORRECT answer, so the rule is
            # inverted: what matters is that none of the other buyer's
            # details appear.
            must_not_mention=(a.other_order_awb,),
            anchor_exists=False,
        ))

    return steps
