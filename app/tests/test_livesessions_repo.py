"""
Permanent test suite for livesessions_repo.py (PDF §7). No user-scoping
here by design - live sessions are public, so there's no cross-user
security test for this repo, unlike orders/bargains.
"""

from app.repos import livesessions_repo


class TestGetLiveNow:
    async def test_returns_a_list_without_error(self, db):
        result = await livesessions_repo.get_live_now()
        assert isinstance(result, list)  # 0 live right now is a valid real state


class TestGetSessionProducts:
    async def test_returns_real_product_sequence(self, real_session_with_products):
        result = await livesessions_repo.get_session_products(
            str(real_session_with_products["_id"])
        )
        assert result["title"] == real_session_with_products["title"]
        assert len(result["productSequence"]) == len(real_session_with_products["productSequence"])

    async def test_fake_session_returns_none(self, db):
        result = await livesessions_repo.get_session_products("000000000000000000000000")
        assert result is None


class TestGetSessionRecap:
    async def test_returns_real_peak_viewers(self, real_session_with_products):
        result = await livesessions_repo.get_session_recap(
            str(real_session_with_products["_id"])
        )
        assert result["peakViewers"] == real_session_with_products.get("peakViewers")