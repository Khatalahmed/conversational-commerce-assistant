"""
WHAT:
    Tests the product rows that travel BESIDE the answer - the ones a
    client turns into something tappable.

WHY THIS EXISTS SEPARATELY FROM THE TRIMMERS:
    The trimmers shape what the MODEL reads. These cards are what the APP
    draws, and the whole point of keeping them apart is that the model
    never composes a product reference: it is handed no URL, writes no
    URL, and so cannot attach the right id to the wrong name. A card that
    opens the wrong product is worse than no card, so the tests below are
    mostly about what must NOT become a card.

FLOW:
    Pure unit tests over _product_cards, plus one that the collector is
    actually filled by execute_tool. No database, no LLM.
"""

import json

import pytest

from app.agent.tool_executor import _product_cards, execute_tool


def product(name="kurta", price=1000, discounted=800, _id="a" * 24):
    return {
        "product_id": _id, "name": name, "price": price,
        "discountedPrice": discounted, "category": "Men",
    }


class TestWhatBecomesACard:
    def test_a_bare_list_result_is_read_directly(self):
        cards = _product_cards("get_trending_products", [product(), product(_id="b" * 24)])
        assert [c["product_id"] for c in cards] == ["a" * 24, "b" * 24]

    def test_a_nested_list_is_read_from_its_key(self):
        cards = _product_cards(
            "get_session_products",
            {"sessionId": "s1", "products": [product()], "unavailableCount": 2},
        )
        assert len(cards) == 1
        assert cards[0]["name"] == "kurta"

    def test_saved_items_uses_its_own_key(self):
        cards = _product_cards(
            "get_saved_items",
            {"savedProducts": [product()], "savedBitCount": 3},
        )
        assert len(cards) == 1

    def test_product_detail_is_normalised_to_the_list_shape(self):
        """_trim_product_detail names its id "productId" where the list
        trimmer says "product_id". A client should not have to know."""
        cards = _product_cards("get_product_detail", {
            "productId": "c" * 24, "name": "kurta", "price": 1000,
            "discountedPrice": 800, "variantSummary": [],
        })
        assert cards == [{
            "product_id": "c" * 24, "name": "kurta",
            "price": 1000, "discountedPrice": 800,
        }]


class TestWhatMustNotBecomeACard:
    """The failure this design exists to prevent is a card that opens
    something the answer was not about."""

    def test_an_order_result_yields_nothing(self):
        """Order lines carry a name and a price and look enough like
        products to fool a shape-sniffing heuristic. They are keyed out
        by tool name instead, and this is the test that says why."""
        orders = {"orders": [{"orderId": "ORD1", "items": [{"name": "kurta", "qty": 1}],
                              "total": 800}], "showing": 1, "totalOrders": 9}
        assert _product_cards("get_order_history", orders) == []

    def test_a_cart_result_yields_nothing(self):
        cart = {"items": [{"name": "kurta", "qty": 1, "price": 800}], "total": 800}
        assert _product_cards("get_cart", cart) == []

    def test_a_product_without_an_id_is_dropped(self):
        """It cannot be opened, so it is not offered as something to tap."""
        rows = [product(), {"name": "ghost", "price": 100}]
        cards = _product_cards("search_products", rows)
        assert len(cards) == 1
        assert cards[0]["name"] == "kurta"

    def test_a_failed_lookup_yields_nothing(self):
        from app.agent.tool_executor import TOOL_FAILED
        assert _product_cards("search_products", TOOL_FAILED) == []

    def test_none_and_junk_are_survived(self):
        assert _product_cards("search_products", None) == []
        assert _product_cards("get_session_products", None) == []
        assert _product_cards("search_products", "not a list") == []

    def test_no_card_ever_carries_a_url(self):
        """the storefront has no product pages - /product/<id> renders its
        router's 404 - so a link built here would point at nothing. The
        app navigates on product_id through its own stack."""
        cards = _product_cards("search_products", [product()])
        for card in cards:
            assert set(card) == {"product_id", "name", "price", "discountedPrice"}
            assert not any("http" in str(v) for v in card.values())


class TestTheCollectorIsActuallyFilled:
    async def test_execute_tool_appends_one_batch_per_call(self, monkeypatch):
        async def fake_trending(limit=5):
            return [{"_id": "a" * 24, "name": "kurta", "price": 1000,
                     "discountedPrice": 800, "category": "Men"}]

        monkeypatch.setitem(
            __import__("app.agent.tools", fromlist=["TOOL_REGISTRY"]).TOOL_REGISTRY,
            "get_trending_products", (fake_trending, False),
        )

        batches: list = []
        raw = await execute_tool("get_trending_products", {}, "u1", collector=batches)

        assert len(batches) == 1
        assert batches[0][0]["product_id"] == "a" * 24
        # And the model's own view is untouched by any of this.
        assert "product_id" in json.loads(raw)[0]

    async def test_no_collector_means_no_work_and_no_error(self, monkeypatch):
        """/chat used to call execute_tool with three arguments and still
        does - the feature costs the path that does not use it nothing."""
        async def fake_trending(limit=5):
            return [{"_id": "a" * 24, "name": "kurta", "price": 1000}]

        monkeypatch.setitem(
            __import__("app.agent.tools", fromlist=["TOOL_REGISTRY"]).TOOL_REGISTRY,
            "get_trending_products", (fake_trending, False),
        )
        assert json.loads(await execute_tool("get_trending_products", {}, "u1"))


def images_on(monkeypatch):
    """Turns thumbnails on. They are OFF by default now - see
    TestImagesAreWithheldUntilTheCdnCanResize - so anything testing the
    attach path has to enable them explicitly."""
    from types import SimpleNamespace

    from app.agent import tool_executor
    monkeypatch.setattr(
        tool_executor, "get_settings",
        lambda: SimpleNamespace(product_image_query="?width=96&quality=75"),
    )


class TestImagesAreWithheldUntilTheCdnCanResize:
    """The catalogue holds full-resolution originals. One measured
    product image is 8.5 MB - a 6144x8192 photograph - rendered into a
    46-pixel square, and a trending answer pulled roughly 10 MB for eight
    of them.

    Bunny Optimizer is not enabled on the pull zone, so there is nothing
    useful to append and nothing safe to send. Withheld by default
    because the failure is silent: the app works perfectly and quietly
    burns a mobile data plan."""

    async def test_no_image_is_attached_when_the_query_is_empty(self, monkeypatch):
        from types import SimpleNamespace

        from app.agent import tool_executor
        monkeypatch.setattr(
            tool_executor, "get_settings",
            lambda: SimpleNamespace(product_image_query=""),
        )
        cards = await tool_executor.attach_thumbnails([product()])
        assert "image" not in cards[0]
        assert cards[0]["name"] == "kurta", "the card itself still ships"

    async def test_the_database_is_not_even_queried(self, monkeypatch):
        """A withheld image should cost nothing, not a wasted round trip."""
        from types import SimpleNamespace

        from app.agent import tool_executor

        async def boom(ids):
            raise AssertionError("no lookup should happen when images are off")

        monkeypatch.setattr(
            tool_executor, "get_settings",
            lambda: SimpleNamespace(product_image_query=""),
        )
        monkeypatch.setattr(tool_executor.products_repo, "get_products_by_ids", boom)
        await tool_executor.attach_thumbnails([product()])

    async def test_setting_the_query_turns_them_back_on(self, monkeypatch):
        """The whole point: this is a config change, not a code change."""
        from bson import ObjectId

        from app.agent import tool_executor
        images_on(monkeypatch)

        async def fake(ids):
            return [{"_id": ObjectId("a" * 24),
                     "images": [{"url": "https://cdn.example/a.jpg"}]}]

        monkeypatch.setattr(tool_executor.products_repo, "get_products_by_ids", fake)
        cards = await tool_executor.attach_thumbnails([product()])
        assert cards[0]["image"] == "https://cdn.example/a.jpg?width=96&quality=75"


class TestThumbnails:
    """The trimmers drop images because the model cannot use a CDN URL.
    The card is the one consumer that can, so it fetches them itself -
    and must survive every way that lookup can go wrong, because a
    missing picture is not a reason to lose the product."""

    async def test_the_first_image_is_attached(self, monkeypatch):
        images_on(monkeypatch)
        from bson import ObjectId

        from app.agent import tool_executor

        async def fake_by_ids(ids):
            return [{"_id": ObjectId("a" * 24), "images": [
                {"url": "https://cdn.example/one.jpg", "public_id": "p1"},
                {"url": "https://cdn.example/two.jpg", "public_id": "p2"},
            ]}]

        monkeypatch.setattr(tool_executor.products_repo, "get_products_by_ids", fake_by_ids)
        cards = await tool_executor.attach_thumbnails([product()])
        assert cards[0]["image"] == (
            "https://cdn.example/one.jpg?width=96&quality=75"
        ), "the FIRST image, with the resize parameters appended"

    async def test_a_product_with_no_images_gets_none(self, monkeypatch):
        images_on(monkeypatch)
        from bson import ObjectId

        from app.agent import tool_executor

        async def fake_by_ids(ids):
            return [{"_id": ObjectId("a" * 24), "images": []}]

        monkeypatch.setattr(tool_executor.products_repo, "get_products_by_ids", fake_by_ids)
        cards = await tool_executor.attach_thumbnails([product()])
        assert cards[0]["image"] is None
        assert cards[0]["name"] == "kurta", "the card survives the missing picture"

    async def test_a_failed_lookup_keeps_every_card(self, monkeypatch):
        images_on(monkeypatch)
        from app.agent import tool_executor

        async def boom(ids):
            raise RuntimeError("cluster unreachable")

        monkeypatch.setattr(tool_executor.products_repo, "get_products_by_ids", boom)
        cards = await tool_executor.attach_thumbnails([product(), product(_id="b" * 24)])
        assert len(cards) == 2
        assert all("image" not in c for c in cards)

    async def test_an_unparseable_id_does_not_reach_the_database(self, monkeypatch):
        images_on(monkeypatch)
        from app.agent import tool_executor

        called = False

        async def fake_by_ids(ids):
            nonlocal called
            called = True
            return []

        monkeypatch.setattr(tool_executor.products_repo, "get_products_by_ids", fake_by_ids)
        cards = await tool_executor.attach_thumbnails([product(_id="not-an-object-id")])
        assert cards == [product(_id="not-an-object-id")]
        assert not called, "no query is worth making for an id Mongo cannot parse"

    async def test_no_cards_means_no_query(self, monkeypatch):
        images_on(monkeypatch)
        from app.agent import tool_executor

        async def boom(ids):
            raise AssertionError("should not be called")

        monkeypatch.setattr(tool_executor.products_repo, "get_products_by_ids", boom)
        assert await tool_executor.attach_thumbnails([]) == []
