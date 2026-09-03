"""
Permanent test suite for seller_repo.py (PDF §11). Confirms
cross-seller coupon protection - one seller cannot see another
seller's coupon performance, even with the exact right code.
"""

from app.repos import seller_repo


class TestGetPayoutStatus:
    async def test_returns_real_payout(self, real_payout):
        result = await seller_repo.get_payout_status(
            str(real_payout["sellerId"]), real_payout["orderRef"]
        )
        assert result is not None
        assert result["sellerAmount"] == real_payout["sellerAmount"]


class TestGetSalesPerformance:
    async def test_returns_real_seller_stats(self, real_seller):
        result = await seller_repo.get_sales_performance(str(real_seller["_id"]))
        assert result is not None
        assert "productsSoldCount" in result


class TestGetCouponPerformance:
    async def test_correct_seller_can_see_own_coupon(self, real_seller_coupon):
        result = await seller_repo.get_coupon_performance(
            str(real_seller_coupon["sellerId"]), real_seller_coupon["code"]
        )
        assert result is not None
        assert result["code"] == real_seller_coupon["code"]

    async def test_wrong_seller_cannot_see_coupon(self, real_seller_coupon):
        wrong_seller_id = "000000000000000000000000"
        result = await seller_repo.get_coupon_performance(
            wrong_seller_id, real_seller_coupon["code"]
        )
        assert result is None, "SECURITY: a real coupon must be invisible to the wrong seller"


class TestGetPendingBargainCount:
    async def test_returns_valid_count_structure(self, real_seller):
        result = await seller_repo.get_pending_bargain_count(str(real_seller["_id"]))
        assert "pendingCount" in result
        assert result["pendingCount"] >= 0