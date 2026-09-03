"""
Permanent test suite for carts_repo.py + coupons_repo.py (PDF §6).
Confirms the read-only coupon-validity guarantee: the repo NEVER
applies a coupon, only reports validity.
"""

from app.repos import carts_repo, coupons_repo


class TestGetCart:
    async def test_returns_real_items(self, real_cart_with_items):
        user_id = str(real_cart_with_items["user"])
        result = await carts_repo.get_cart(user_id)
        assert result["itemCount"] == len(real_cart_with_items["items"])

    async def test_no_cart_returns_empty_gracefully(self, db):
        result = await carts_repo.get_cart("000000000000000000000000")
        assert result == {"items": [], "itemCount": 0}


class TestCheckCouponValidity:
    async def test_nonexistent_code_returns_not_found(self, db):
        result = await coupons_repo.check_coupon_validity("NO-SUCH-CODE-XYZ", "000000000000000000000000")
        assert result["valid"] is False
        assert result["reason"] == "not_found"

    async def test_inactive_coupon_correctly_rejected(self, real_inactive_coupon):
        result = await coupons_repo.check_coupon_validity(
            real_inactive_coupon["code"], "000000000000000000000000"
        )
        assert result["valid"] is False
        assert result["reason"] == "inactive"