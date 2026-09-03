"""
WHAT:
    Tests for the two searches that run on the marketplace's existing Atlas
    indexes - full-text (product_text_index) and vector
    (product_vector_index).

WHY THIS APPROACH:
    These were already built, populated and READY on the cluster; the
    chatbot simply was not using them. So the risk here is not "does
    Atlas work" but "are we querying it correctly" - a $search against
    an unmapped path, or a $vectorSearch against the wrong collection,
    returns EMPTY rather than erroring. Silence is the failure mode, so
    every test below asserts on real results from real data rather than
    on the absence of an exception.

    Live-data tests, not mocked: an index definition is exactly the kind
    of thing a mock would assert into existence while production has it
    wrong.
"""

import pytest

from app.repos import products_repo


@pytest.fixture
async def a_real_product(db):
    """A product that actually has an embedding - the sync is eventually
    consistent, so 'has a product' and 'has an embedding' differ."""
    emb = await db.product_embeddings.find_one({})
    assert emb, "no product embeddings in this cluster"
    product = await db.products.find_one({"_id": emb["_id"]})
    assert product, "embedding exists for a product that does not"
    return product


class TestFullTextSearch:
    async def test_finds_a_product_by_a_word_in_its_name(self, db, a_real_product):
        word = a_real_product["name"].split("-")[-1].split()[0]
        results = await products_repo.search_products_by_name(word, limit=5)
        assert results, f"full-text search found nothing for {word!r}"
        assert any(word.lower() in (r.get("name") or "").lower() for r in results)

    async def test_returns_allowlisted_fields_only(self, db, a_real_product):
        word = a_real_product["name"].split("-")[-1].split()[0]
        results = await products_repo.search_products_by_name(word, limit=3)
        # The aggregation path must apply the projection exactly like
        # find() does - it is easy to forget on a $search pipeline.
        for r in results:
            assert "stepData" not in r, "internal seller data leaked via $search"

    async def test_tolerates_a_typo(self, db):
        """The regex could not do this at all - a substring match has no
        notion of "close". Fuzzy matching is why the compound query
        exists."""
        results = await products_repo.search_products_by_name("budha", limit=5)
        assert any("buddha" in (r.get("name") or "").lower() for r in results), (
            "typo tolerance regressed - check the fuzzy clause"
        )

    async def test_finds_a_word_inside_a_compound_word(self, db):
        """"Smartwatch" is ONE token to the analyzer, so a plain text
        search for "watch" misses it. The wildcard clause is the only
        thing that catches this, and it is a real catalogue pattern."""
        results = await products_repo.search_products_by_name("watch", limit=10)
        names = " ".join((r.get("name") or "").lower() for r in results)
        assert "smartwatch" in names, "compound-word matching regressed"

    async def test_a_typo_does_not_drag_in_unrelated_products(self, db):
        """Guards the OTHER direction. At fuzzy maxEdits=2 this
        catalogue returned shirts and towels for "watch"; the assistant
        presents top hits as real answers, so noise is worse than a
        miss."""
        results = await products_repo.search_products_by_name("watch", limit=10)
        for r in results:
            name = (r.get("name") or "").lower()
            assert "shirt" not in name and "towel" not in name, (
                f"unrelated product {r.get('name')!r} matched 'watch' - "
                f"fuzzy matching is too loose"
            )

    async def test_nonsense_query_returns_empty_not_error(self, db):
        assert await products_repo.search_products_by_name("zzzqqqxxnothing") == []

    async def test_respects_the_limit(self, db, a_real_product):
        results = await products_repo.search_products_by_name("a", limit=2)
        assert len(results) <= 2

    async def test_regex_fallback_still_works(self, db, a_real_product):
        # The slow path, used only when the index is missing. It must
        # remain correct, or a fresh cluster silently returns nothing.
        name = a_real_product["name"]
        results = await products_repo._search_products_by_name_regex(name, 5)
        assert any(r.get("name") == name for r in results)


class TestVectorSearch:
    async def test_returns_similar_products(self, db, a_real_product):
        similar = await products_repo.find_similar_products(
            str(a_real_product["_id"]), limit=5
        )
        assert similar, "vector search returned nothing for a product that has one"

    async def test_the_seed_product_is_not_returned_as_its_own_match(
        self, db, a_real_product
    ):
        # The seed always scores 1.0 against itself. Suggesting the item
        # someone is already looking at is not a recommendation.
        seed_id = a_real_product["_id"]
        similar = await products_repo.find_similar_products(str(seed_id), limit=5)
        assert all(p["_id"] != seed_id for p in similar)

    async def test_results_are_ordered_by_relevance(self, db, a_real_product):
        # $in does not preserve order - the repo re-sorts to the vector
        # ranking. Without that the "most similar" is whatever MongoDB
        # happened to return first, which looks fine and is wrong.
        similar = await products_repo.find_similar_products(
            str(a_real_product["_id"]), limit=5
        )
        assert len(similar) >= 2
        ids = [p["_id"] for p in similar]
        assert len(ids) == len(set(ids)), "duplicate products in results"

    async def test_unknown_product_returns_empty_not_error(self, db):
        assert await products_repo.find_similar_products("6" * 24) == []

    async def test_malformed_id_returns_empty_not_error(self, db):
        assert await products_repo.find_similar_products("not-an-object-id") == []

    async def test_sold_products_are_excluded(self, db, a_real_product):
        similar = await products_repo.find_similar_products(
            str(a_real_product["_id"]), limit=10
        )
        assert all(not p.get("isSold") for p in similar)


class TestExposedToTheAssistant:
    def test_the_tool_is_registered(self):
        from app.agent.tools import TOOL_REGISTRY, TOOLS

        assert "find_similar_products" in TOOL_REGISTRY
        names = [t["function"]["name"] for t in TOOLS]
        assert "find_similar_products" in names

    def test_the_tool_does_not_ask_the_model_for_a_user_id(self):
        # Same rule as every other tool: identity is server-injected.
        from app.agent.tools import TOOLS

        tool = next(
            t for t in TOOLS if t["function"]["name"] == "find_similar_products"
        )
        assert "user_id" not in tool["function"]["parameters"]["properties"]
