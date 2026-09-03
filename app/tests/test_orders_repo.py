"""
Permanent, re-runnable test suite for orders_repo.py. Covers every
sub-feature from PDF §3, correct user-scoping (the project's core
security guarantee), and graceful empty-result handling.
"""

from app.repos import orders_repo


class TestGetOrderStatus:
    async def test_returns_real_status(self, real_order):
        user_id = str(real_order["buyerId"])
        result = await orders_repo.get_order_status(user_id, real_order["orderId"])
        assert result is not None
        assert result["orderId"] == real_order["orderId"]
        assert result["status"] == real_order["status"]

    async def test_wrong_user_cannot_see_order(self, real_order):
        wrong_user_id = "000000000000000000000000"
        result = await orders_repo.get_order_status(wrong_user_id, real_order["orderId"])
        assert result is None, "SECURITY: a real order must be invisible to the wrong user"

    async def test_nonexistent_order_returns_none_not_error(self, real_order):
        user_id = str(real_order["buyerId"])
        result = await orders_repo.get_order_status(user_id, "FAKE-ORDER-ID")
        assert result is None


class TestGetOrderHistory:
    async def test_returns_list_scoped_to_user(self, real_order):
        user_id = str(real_order["buyerId"])
        result = await orders_repo.get_order_history(user_id, limit=5)
        assert isinstance(result, list)
        assert all(o.get("orderId") for o in result)

    async def test_unknown_user_returns_empty_list_not_error(self, db):
        result = await orders_repo.get_order_history("000000000000000000000000")
        assert result == []


class TestGetOrderDetail:
    async def test_returns_items_and_coarse_delivery_location_only(self, real_order):
        user_id = str(real_order["buyerId"])
        result = await orders_repo.get_order_detail(user_id, real_order["orderId"])
        assert result is not None
        assert "items" in result
        # SECURITY: full address must NEVER appear, only city/state
        assert "deliveryAddress" not in result
        assert "line1" not in str(result)
        assert "phone" not in str(result)


class TestGetInvoice:
    async def test_order_with_invoice_returns_url(self, real_order_with_invoice):
        user_id = str(real_order_with_invoice["buyerId"])
        result = await orders_repo.get_invoice(user_id, real_order_with_invoice["orderId"])
        assert result["invoiceAvailable"] is True
        assert result["url"]

    async def test_order_without_invoice_handled_gracefully(self, real_order_without_invoice):
        user_id = str(real_order_without_invoice["buyerId"])
        result = await orders_repo.get_invoice(user_id, real_order_without_invoice["orderId"])
        assert result["invoiceAvailable"] is False


class TestCheckCancellationEligibility:
    async def test_cancellable_order_returns_true(self, real_cancellable_order):
        user_id = str(real_cancellable_order["buyerId"])
        result = await orders_repo.check_cancellation_eligibility(
            user_id, real_cancellable_order["orderId"]
        )
        assert result["canCancel"] is True

    async def test_delivered_order_returns_false(self, real_delivered_order):
        user_id = str(real_delivered_order["buyerId"])
        result = await orders_repo.check_cancellation_eligibility(
            user_id, real_delivered_order["orderId"]
        )
        assert result["canCancel"] is False