"""
Permanent test suite for bits_repo.py (PDF §8). No user-scoping - Bits
are public content, so no cross-user security test for this repo.
"""

from app.repos import bits_repo


class TestGetTrendingBits:
    async def test_returns_a_list_without_error(self, db):
        result = await bits_repo.get_trending_bits()
        assert isinstance(result, list)


class TestGetTaggedProducts:
    async def test_returns_real_products(self, real_bit_with_products):
        result = await bits_repo.get_tagged_products(str(real_bit_with_products["_id"]))
        assert len(result["products"]) == len(real_bit_with_products["products"])

    async def test_fake_bit_returns_none(self, db):
        result = await bits_repo.get_tagged_products("000000000000000000000000")
        assert result is None


class TestSearchByHashtag:
    async def test_finds_results_without_leading_hash(self, real_bit_with_hashtags):
        real_tag = real_bit_with_hashtags["hashtags"][0].lstrip("#")
        results = await bits_repo.search_by_hashtag(real_tag)
        assert len(results) > 0

    async def test_no_matches_returns_empty_list(self, db):
        results = await bits_repo.search_by_hashtag("nosuchhashtagxyz123")
        assert results == []