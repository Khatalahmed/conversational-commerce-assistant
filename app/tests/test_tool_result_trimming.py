"""
WHAT:
    Tests the two trimmers that shape what the model SEES - product lists
    and order history.

WHY THESE TWO:
    Both were written after watching real answers, not from theory.

    Product lists repeated themselves: a measured similar-products result
    returned "jeans" three times at Rs.800, because the catalogue holds
    several genuinely distinct rows that are indistinguishable to a
    shopper. A list that repeats itself reads as broken.

    Order history was 1,574 tokens of raw documents for five orders, and
    carried expectedDelivery dates a month in the past with nothing
    marking them late - so the assistant reported "expected by 27 July"
    on 26 August, which is true and useless. A model cannot spot that: it
    does not know today's date. The comparison has to happen here.

FLOW:
    Pure unit tests over the trimmer functions. No database, no LLM.
"""

from datetime import datetime, timedelta, timezone

from app.agent.tool_executor import _trim_order_history, _trim_product_list

NOW = datetime.now(timezone.utc).replace(tzinfo=None)


def product(name, price, discounted=None, _id="1" * 24):
    return {
        "_id": _id, "name": name, "price": price,
        "discountedPrice": discounted if discounted is not None else price,
        "category": "Men",
    }


def order(order_id, status, expected_days_ago=None, items=("thing",)):
    dates = {}
    if expected_days_ago is not None:
        dates["expectedDelivery"] = NOW - timedelta(days=expected_days_ago)
    return {
        "orderId": order_id, "status": status, "dates": dates,
        "items": [{"name": n, "qty": 1} for n in items],
        "pricing": {"total": 100},
        # Fields the model never needs, present to prove they are dropped.
        "buyerId": "x", "sellerId": "y", "statusHistory": [{"status": "pending"}],
        "review": {"images": []}, "deliveryAddress": {"city": "Pune"},
    }


class TestProductListDeduping:
    def test_identical_name_and_price_collapse(self):
        rows = _trim_product_list([
            product("jeans", 1000, 800, "a" * 24),
            product("jeans", 1000, 800, "b" * 24),
            product("jeans", 1000, 800, "c" * 24),
        ])
        assert len(rows) == 1, "the same line three times reads as broken"

    def test_same_name_different_price_is_kept(self):
        """A real choice the shopper can act on, not a duplicate."""
        rows = _trim_product_list([
            product("jeans", 1000, 800, "a" * 24),
            product("jeans", 1500, 1200, "b" * 24),
        ])
        assert len(rows) == 2

    def test_case_and_whitespace_do_not_defeat_it(self):
        rows = _trim_product_list([
            product("Jeans ", 1000, 800, "a" * 24),
            product("jeans", 1000, 800, "b" * 24),
        ])
        assert len(rows) == 1

    def test_distinct_products_all_survive(self):
        rows = _trim_product_list([
            product("jeans", 1000, 800, "a" * 24),
            product("Baggy Blue Street", 2099, 1449, "b" * 24),
            product("Denim Model", 1449, 1349, "c" * 24),
        ])
        assert len(rows) == 3

    def test_the_hard_cap_still_applies(self):
        rows = _trim_product_list([product(f"p{i}", 100 + i) for i in range(50)])
        assert len(rows) <= 8

    def test_empty_input_is_safe(self):
        assert _trim_product_list([]) == []
        assert _trim_product_list(None) == []


class TestOrderHistoryTrimming:
    def test_internal_fields_are_dropped(self):
        row = _trim_order_history([order("ORD1", "confirmed")])[0]
        for gone in ("buyerId", "sellerId", "statusHistory", "review",
                     "deliveryAddress"):
            assert gone not in row

    def test_what_the_answer_needs_survives(self):
        row = _trim_order_history([order("ORD1", "confirmed", items=("clock",))])[0]
        assert row["orderId"] == "ORD1"
        assert row["status"] == "confirmed"
        assert row["items"][0]["name"] == "clock"
        assert row["total"] == 100


class TestOverdueDetection:
    def test_a_late_in_progress_order_is_flagged(self):
        """The model cannot work this out - it does not know the date."""
        row = _trim_order_history([order("ORD1", "confirmed", expected_days_ago=29)])[0]
        assert row["isOverdue"] is True
        assert row["daysLate"] == 29

    def test_a_future_delivery_is_not_flagged(self):
        future = {
            **order("ORD1", "confirmed"),
            "dates": {"expectedDelivery": NOW + timedelta(days=3)},
        }
        row = _trim_order_history([future])[0]
        assert "isOverdue" not in row

    def test_a_delivered_order_is_never_late(self):
        """It arrived. Whether it beat its estimate is not the question a
        shopper is asking, and calling it overdue would be wrong."""
        row = _trim_order_history([order("ORD1", "delivered", expected_days_ago=40)])[0]
        assert "isOverdue" not in row

    def test_a_cancelled_order_is_never_late(self):
        row = _trim_order_history([order("ORD1", "cancelled", expected_days_ago=90)])[0]
        assert "isOverdue" not in row

    def test_a_missing_date_does_not_crash(self):
        row = _trim_order_history([order("ORD1", "confirmed", expected_days_ago=None)])[0]
        assert "isOverdue" not in row
        assert row["expectedDelivery"] is None


class TestOrderCountIsNotTheWindowSize:
    """A capped list with no total reads as the total.

    Measured on a real account: 67 orders, of which the model was handed
    eight, and it answered "you have 8 orders". Not a hallucination - a
    faithful reading of a truncated list nobody had labelled. The
    summariser exists to label it.
    """

    async def test_the_total_is_reported_alongside_the_window(self, db, real_order):
        import json
        from app.agent.tool_executor import execute_tool
        from app.repos import orders_repo

        user_id = str(real_order["buyerId"])
        true_total = await orders_repo.count_orders(user_id)

        result = json.loads(await execute_tool("get_order_history", {}, user_id))

        assert result["totalOrders"] == true_total
        assert result["showing"] == len(result["orders"])
        assert result["showing"] <= true_total

    async def test_the_note_says_so_when_the_list_is_truncated(self, db, real_order):
        """The counts alone are two numbers in a JSON blob. The sentence
        is what the model actually reads."""
        import json
        from app.agent.tool_executor import execute_tool
        from app.repos import orders_repo

        user_id = str(real_order["buyerId"])
        total = await orders_repo.count_orders(user_id)
        result = json.loads(await execute_tool("get_order_history", {}, user_id))

        if total > result["showing"]:
            assert result["note"] is not None
            assert str(total) in result["note"]
        else:
            assert result["note"] is None

    async def test_an_unknown_buyer_counts_zero_rather_than_raising(self):
        from app.repos import orders_repo

        assert await orders_repo.count_orders("not-an-object-id") == 0


class TestBargainListTrimming:
    """list_bargains returns the raw documents - buyerId, sellerId, the
    whole productSnapshot including its image URL, variant, quantity -
    for an answer that needs a product, a price and what happened."""

    def _bargain(self, status="pending", expires_days_ago=None, counter=None):
        from datetime import timedelta
        doc = {
            "productSnapshot": {"name": "Cotton Kurta", "image": "https://cdn/x.jpg"},
            "offeredPrice": 450, "originalPrice": 599, "status": status,
            "orderPlaced": False,
            # Present to prove they are dropped.
            "buyerId": "b", "sellerId": "s", "variant": {"color": "blue"},
            "quantity": 1, "discountPercentage": 25,
        }
        if counter:
            doc["counterOffer"] = counter
        if expires_days_ago is not None:
            doc["expiresAt"] = NOW - timedelta(days=expires_days_ago)
        return doc

    def test_only_the_answerable_fields_survive(self):
        from app.agent.tool_executor import _trim_bargain_list
        row = _trim_bargain_list([self._bargain()])[0]
        assert set(row) == {
            "productName", "offeredPrice", "originalPrice", "status",
            "hasCounterOffer", "orderPlaced", "expiresAt",
        }
        assert row["productName"] == "Cotton Kurta"
        assert "image" not in str(row), "the CDN url must not reach the model"

    def test_a_counter_offer_is_reduced_to_a_flag(self):
        from app.agent.tool_executor import _trim_bargain_list
        rows = _trim_bargain_list([self._bargain(counter={"price": 500})])
        assert rows[0]["hasCounterOffer"] is True
        assert _trim_bargain_list([self._bargain()])[0]["hasCounterOffer"] is False

    def test_a_pending_offer_past_its_expiry_is_marked_lapsed(self):
        """Same reasoning as isOverdue on orders: expiresAt is a date and
        the model does not know today's, so "expires 12 July" reported in
        September is true and useless."""
        from app.agent.tool_executor import _trim_bargain_list
        row = _trim_bargain_list([self._bargain(expires_days_ago=30)])[0]
        assert row["hasLapsed"] is True

    def test_a_resolved_offer_is_never_marked_lapsed(self):
        from app.agent.tool_executor import _trim_bargain_list
        for status in ("accepted", "auto_accepted", "expired"):
            row = _trim_bargain_list([self._bargain(status, expires_days_ago=30)])[0]
            assert "hasLapsed" not in row, f"{status} has already resolved"

    def test_a_future_expiry_is_not_lapsed(self):
        from app.agent.tool_executor import _trim_bargain_list
        row = _trim_bargain_list([self._bargain(expires_days_ago=-5)])[0]
        assert "hasLapsed" not in row

    def test_the_list_is_capped(self):
        from app.agent.tool_executor import LLM_LIST_LIMIT_CAP, _trim_bargain_list
        rows = _trim_bargain_list([self._bargain()] * 40)
        assert len(rows) == LLM_LIST_LIMIT_CAP

    def test_none_is_survived(self):
        from app.agent.tool_executor import _trim_bargain_list
        assert _trim_bargain_list(None) == []


class TestBargainExpiryIsCountedNotJustDated:
    """Measured: handed expiresAt raw, the model wrote "expired at
    2026-07-09T09:15:32" into a phone chat bubble. The orders trimmer
    already solved this shape with daysLate."""

    def _b(self, status, days_ago):
        from datetime import timedelta
        return {
            "productSnapshot": {"name": "saree"}, "offeredPrice": 518,
            "originalPrice": 699, "status": status,
            "expiresAt": NOW - timedelta(days=days_ago),
        }

    def test_a_lapsed_offer_says_how_long_ago(self):
        from app.agent.tool_executor import _trim_bargain_list
        row = _trim_bargain_list([self._b("pending", 55)])[0]
        assert row["hasLapsed"] is True
        assert row["lapsedDaysAgo"] == 55

    def test_a_live_offer_says_how_long_is_left(self):
        from app.agent.tool_executor import _trim_bargain_list
        row = _trim_bargain_list([self._b("pending", -3)])[0]
        assert row["expiresInDays"] == 3
        assert "hasLapsed" not in row

    def test_an_expired_offer_is_counted_but_not_flagged(self):
        """For an expired offer the expiry IS what happened, so the count
        belongs - but the flag does not: its status already says so."""
        from app.agent.tool_executor import _trim_bargain_list
        row = _trim_bargain_list([self._b("expired", 55)])[0]
        assert row["lapsedDaysAgo"] == 55
        assert "hasLapsed" not in row

    def test_an_accepted_offer_gets_no_countdown_at_all(self):
        """It resolved another way - the expiry date is irrelevant."""
        from app.agent.tool_executor import _trim_bargain_list
        row = _trim_bargain_list([self._b("accepted", 55)])[0]
        assert "hasLapsed" not in row
        assert "lapsedDaysAgo" not in row
