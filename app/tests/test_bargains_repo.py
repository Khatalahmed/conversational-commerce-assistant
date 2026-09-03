"""
Permanent test suite for bargains_repo.py - the signature feature.
Deliberately targets the ONE real bargain with a counter-offer (rather
than leaving it to chance) plus the wrong-user security guarantee.
"""

from app.repos import bargains_repo


class TestCheckBargainEligibility:
    async def test_returns_settings_for_bargainable_product(self, real_bargain_with_counter_offer):
        result = await bargains_repo.check_bargain_eligibility(
            str(real_bargain_with_counter_offer["productId"])
        )
        assert result["bargainingAllowed"] is True


class TestGetBargainStatus:
    async def test_returns_real_bargain_by_id(self, real_bargain_with_counter_offer):
        user_id = str(real_bargain_with_counter_offer["buyerId"])
        result = await bargains_repo.get_bargain_status(
            user_id, bargain_id=str(real_bargain_with_counter_offer["_id"])
        )
        assert result is not None
        assert result["hasCounterOffer"] is True

    async def test_lookup_by_product_id_also_works(self, real_bargain_with_counter_offer):
        user_id = str(real_bargain_with_counter_offer["buyerId"])
        result = await bargains_repo.get_bargain_status(
            user_id, product_id=str(real_bargain_with_counter_offer["productId"])
        )
        assert result is not None

    async def test_wrong_user_cannot_see_bargain(self, real_bargain_with_counter_offer):
        wrong_user_id = "000000000000000000000000"
        result = await bargains_repo.get_bargain_status(
            wrong_user_id, bargain_id=str(real_bargain_with_counter_offer["_id"])
        )
        assert result is None, "SECURITY: a real bargain must be invisible to the wrong user"


class TestGetCounterOffer:
    async def test_real_counter_offer_returns_details(self, real_bargain_with_counter_offer):
        user_id = str(real_bargain_with_counter_offer["buyerId"])
        result = await bargains_repo.get_counter_offer(
            user_id, bargain_id=str(real_bargain_with_counter_offer["_id"])
        )
        assert result["counterOfferExists"] is True
        assert result["price"] is not None

    async def test_no_counter_offer_handled_gracefully(self, real_bargain_without_counter_offer):
        user_id = str(real_bargain_without_counter_offer["buyerId"])
        result = await bargains_repo.get_counter_offer(
            user_id, bargain_id=str(real_bargain_without_counter_offer["_id"])
        )
        assert result["counterOfferExists"] is False

class TestListBargains:
    """The capability that did not exist. Every other bargain lookup
    needs a bargain_id or a product_id, so "what happened to my offers"
    - a buyer with neither id to hand - was unanswerable, and the eval
    case covering it had to be retired rather than fixed."""

    async def test_returns_the_users_own_offers(self, real_bargain_with_counter_offer):
        user_id = str(real_bargain_with_counter_offer["buyerId"])
        rows = await bargains_repo.list_bargains(user_id)
        assert rows, "a buyer with a known bargain must get at least one back"
        assert all(r.get("productSnapshot") is not None for r in rows)

    async def test_a_wrong_user_sees_nothing(self, real_bargain_with_counter_offer):
        rows = await bargains_repo.list_bargains("000000000000000000000000")
        assert rows == [], (
            "SECURITY: bargains are a negotiation between one buyer and one "
            "seller - listing must be scoped exactly like the single lookups"
        )

    async def test_an_unparseable_user_id_returns_empty_not_everything(self):
        """The failure mode worth naming: a bad id falling through to an
        unscoped query would hand one buyer every offer in the system."""
        assert await bargains_repo.list_bargains("not-an-object-id") == []

    async def test_newest_first(self, real_bargain_with_counter_offer):
        user_id = str(real_bargain_with_counter_offer["buyerId"])
        rows = await bargains_repo.list_bargains(user_id, limit=8)
        dates = [r.get("createdAt") for r in rows if r.get("createdAt")]
        assert dates == sorted(dates, reverse=True)

    async def test_the_limit_is_honoured(self, real_bargain_with_counter_offer):
        user_id = str(real_bargain_with_counter_offer["buyerId"])
        assert len(await bargains_repo.list_bargains(user_id, limit=1)) <= 1

    async def test_the_count_is_the_whole_total_not_the_page(
        self, real_bargain_with_counter_offer
    ):
        user_id = str(real_bargain_with_counter_offer["buyerId"])
        total = await bargains_repo.count_bargains(user_id)
        page = await bargains_repo.list_bargains(user_id, limit=1)
        assert total >= len(page)

    # Takes `db` because the first assertion uses a VALID ObjectId, so it
    # reaches the database rather than short-circuiting in to_object_id.
    # The fixture is what marks this needs_db - without it CI runs a
    # database test with no database.
    async def test_counting_an_unknown_user_is_zero_not_an_error(self, db):
        assert await bargains_repo.count_bargains("000000000000000000000000") == 0
        assert await bargains_repo.count_bargains("nonsense") == 0
