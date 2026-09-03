"""
Permanent test suite for products_repo.py. Covers PDF §4's sub-features
and the real edge cases discovered during Phase 4 development (case-
insensitive category matching, exact variant lookup, graceful fallback
for recommendations).
"""

from app.repos import products_repo


class TestSearchProducts:
    async def test_category_filter_returns_matches(self, real_product_with_variants):
        category = real_product_with_variants["category"]
        results = await products_repo.search_products(category=category, limit=5)
        assert len(results) > 0
        assert all(p["category"].lower() == category.lower() for p in results)

    async def test_no_matches_returns_empty_list_not_error(self, db):
        results = await products_repo.search_products(category="NoSuchCategoryXYZ")
        assert results == []


class TestGetVariantStock:
    async def test_real_variant_is_found(self, real_product_with_variants):
        variant = real_product_with_variants["variants"][0]
        result = await products_repo.get_variant_stock(
            str(real_product_with_variants["_id"]), variant["color"], variant["size"]
        )
        assert result["found"] is True
        assert result["stock"] == variant.get("stock", 0)

    async def test_fake_variant_returns_found_false(self, real_product_with_variants):
        result = await products_repo.get_variant_stock(
            str(real_product_with_variants["_id"]), "nonexistent-color", "nonexistent-size"
        )
        assert result["found"] is False


class TestGetProductDetail:
    async def test_returns_real_product(self, real_product_with_variants):
        result = await products_repo.get_product_detail(str(real_product_with_variants["_id"]))
        assert result["name"] == real_product_with_variants["name"]

    async def test_fake_product_returns_none(self, db):
        result = await products_repo.get_product_detail("000000000000000000000000")
        assert result is None


class TestGetRecommendations:
    async def test_unknown_user_falls_back_to_trending_not_empty(self, db):
        results = await products_repo.get_recommendations("000000000000000000000000")
        assert len(results) > 0, "Fallback to trending must never return empty for a bad user_id"


class TestGetSellerInfo:
    async def test_returns_seller_for_real_product(self, real_product_with_variants):
        result = await products_repo.get_seller_info(str(real_product_with_variants["_id"]))
        assert result is not None
        assert "businessName" in result