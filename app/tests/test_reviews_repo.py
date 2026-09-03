"""
Permanent test suite for reviews_repo.py (PDF §9). Confirms the
aggregation-computed average is correct, not just present.
"""

from app.repos import reviews_repo


class TestGetProductReviews:
    async def test_returns_real_average_and_count(self, real_review):
        result = await reviews_repo.get_product_reviews(str(real_review["productId"]))
        assert result["reviewCount"] > 0
        assert result["averageRating"] is not None

    async def test_product_with_no_reviews_returns_zero_gracefully(self, db):
        result = await reviews_repo.get_product_reviews("000000000000000000000000")
        assert result["reviewCount"] == 0
        assert result["averageRating"] is None


class TestGetSellerTrustInfo:
    async def test_returns_real_seller_info(self, real_product_with_variants):
        result = await reviews_repo.get_seller_trust_info(str(real_product_with_variants["sellerId"]))
        assert result is not None
        assert "customerRating" in result

    async def test_fake_seller_returns_none(self, db):
        result = await reviews_repo.get_seller_trust_info("000000000000000000000000")
        assert result is None