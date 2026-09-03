"""
WHAT:
    Tests for the per-user /chat rate limiter - the unit itself, on BOTH
    backends, and its enforcement through the real endpoint.

WHY THIS APPROACH:
    A limiter has two ways to be wrong and only one is noticeable. Too
    loose and it does not protect anything; too tight and it rejects
    real users, which looks like an outage. Both directions are asserted
    here, along with the two properties that make it worth having at
    all: that it keys on the VERIFIED user (so one client cannot spend
    another's allowance) and that it runs AFTER authentication.

    EVERY CONTRACT TEST RUNS TWICE, once against the in-process table and
    once against Redis (via fakeredis, so no server is needed). The
    interface is identical between backends, which is precisely how a
    difference in BEHAVIOUR hides - and this limiter exists to be
    correct in the deployment that has more than one worker, which is
    the one no test would otherwise touch.

    The test that matters most is test_the_allowance_is_global: it is
    the bug that motivated Redis here at all. Per-process counts meant
    `uvicorn --workers 4` granted four times the configured allowance,
    silently, while every test still passed.

    Time is injected rather than slept through - a test that waits 60
    seconds for a window to roll gets deleted the first time someone is
    in a hurry.
"""

from datetime import datetime, timedelta, timezone

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport

from app.api.main import app
from app.api.routes import chat as chat_route
from app.config.settings import Settings
from app.db import redis_client
from app.security import rate_limit
from app.security.rate_limit import RateLimitExceeded, check_rate_limit
from app.security.auth import create_test_token

USER = "0123456789abcdef01234567"
OTHER_USER = "0123456789abcdef01234568"


def _settings(**overrides) -> Settings:
    base = {
        "mongodb_uri": "mongodb://localhost/test",
        "llm_api_key": "test-key",
        "jwt_secret": "test-secret-padded-out-to-sixty-four-bytes-for-hs512-ok!!",
    }
    return Settings(**{**base, **overrides})


@pytest.fixture
def limits(monkeypatch):
    """Configures the limiter without touching the real .env."""

    def _apply(requests: int, window_seconds: int = 60):
        settings = _settings(
            chat_rate_limit_requests=requests,
            chat_rate_limit_window_seconds=window_seconds,
        )
        monkeypatch.setattr(rate_limit, "get_settings", lambda: settings)
        return settings

    return _apply


@pytest.fixture
def frozen_clock(monkeypatch):
    """Controllable time, so a rolling window can be tested in
    milliseconds instead of minutes.

    It governs BOTH backends: the Redis path scores each request with
    the app's clock rather than Redis TIME, precisely so that one frozen
    clock drives the whole test and the two backends cannot drift apart
    under it.
    """

    class Clock:
        def __init__(self):
            self.now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        def advance(self, seconds: float):
            self.now += timedelta(seconds=seconds)

    clock = Clock()

    class FrozenDatetime:
        @staticmethod
        def now(tz=None):
            return clock.now

    monkeypatch.setattr(rate_limit, "datetime", FrozenDatetime)
    return clock


@pytest.fixture(params=["memory", "redis"])
async def any_backend(request, monkeypatch):
    """Both backends must satisfy the same contract."""
    rate_limit.reset()
    if request.param == "memory":
        monkeypatch.setattr(redis_client, "_client", None)
        monkeypatch.setattr(redis_client, "_checked", True)
        yield "memory"
    else:
        from fakeredis.aioredis import FakeRedis

        client = FakeRedis(decode_responses=True)
        monkeypatch.setattr(redis_client, "_client", client)
        monkeypatch.setattr(redis_client, "_checked", True)
        yield "redis"
        await client.aclose()


@pytest.fixture
async def redis_backend(monkeypatch):
    """Redis only, for the properties the in-process table cannot have."""
    from fakeredis.aioredis import FakeRedis

    rate_limit.reset()
    client = FakeRedis(decode_responses=True)
    monkeypatch.setattr(redis_client, "_client", client)
    monkeypatch.setattr(redis_client, "_checked", True)
    yield client
    await client.aclose()


@pytest.fixture
def memory_backend(monkeypatch):
    """The in-process table - what runs when REDIS_URL is unset."""
    rate_limit.reset()
    monkeypatch.setattr(redis_client, "_client", None)
    monkeypatch.setattr(redis_client, "_checked", True)
    return None


class TestAllowanceIsEnforced:
    async def test_requests_up_to_the_allowance_are_permitted(self, any_backend, limits):
        limits(requests=5)
        for _ in range(5):
            await check_rate_limit(USER)  # must not raise

    async def test_the_request_past_the_allowance_is_refused(self, any_backend, limits):
        limits(requests=3)
        for _ in range(3):
            await check_rate_limit(USER)
        with pytest.raises(RateLimitExceeded):
            await check_rate_limit(USER)

    async def test_a_refused_request_does_not_consume_a_slot(
        self, any_backend, limits, frozen_clock
    ):
        # Otherwise a client hammering the endpoint would keep pushing
        # its own recovery further away - a lockout, not a limit.
        limits(requests=2, window_seconds=60)
        await check_rate_limit(USER)
        await check_rate_limit(USER)

        frozen_clock.advance(30)
        for _ in range(5):
            with pytest.raises(RateLimitExceeded):
                await check_rate_limit(USER)

        # The two original requests still expire on their own schedule.
        frozen_clock.advance(31)
        await check_rate_limit(USER)

    async def test_zero_disables_the_limiter(self, any_backend, limits):
        limits(requests=0)
        for _ in range(500):
            await check_rate_limit(USER)


class TestWindowIsSliding:
    async def test_allowance_returns_as_the_window_rolls(
        self, any_backend, limits, frozen_clock
    ):
        limits(requests=2, window_seconds=60)
        await check_rate_limit(USER)
        frozen_clock.advance(30)
        await check_rate_limit(USER)

        with pytest.raises(RateLimitExceeded):
            await check_rate_limit(USER)

        # 61s after the FIRST request, only that one has expired.
        frozen_clock.advance(31)
        await check_rate_limit(USER)
        with pytest.raises(RateLimitExceeded):
            await check_rate_limit(USER)

    async def test_a_full_window_of_silence_restores_everything(
        self, any_backend, limits, frozen_clock
    ):
        limits(requests=3, window_seconds=60)
        for _ in range(3):
            await check_rate_limit(USER)

        frozen_clock.advance(61)
        for _ in range(3):
            await check_rate_limit(USER)

    async def test_a_fixed_window_boundary_cannot_be_exploited(
        self, any_backend, limits, frozen_clock
    ):
        """The reason this is a SLIDING window.

        With a fixed window resetting on a clock boundary, a client can
        spend its whole allowance just before the reset and its whole
        next allowance just after - double the intended rate, in an
        instant. A sliding window has no such boundary.
        """
        limits(requests=5, window_seconds=60)
        for _ in range(5):
            await check_rate_limit(USER)

        frozen_clock.advance(59.9)
        with pytest.raises(RateLimitExceeded):
            await check_rate_limit(USER)


class TestPerUserIsolation:
    async def test_one_users_traffic_does_not_spend_anothers_allowance(
        self, any_backend, limits
    ):
        limits(requests=2)
        await check_rate_limit(USER)
        await check_rate_limit(USER)
        with pytest.raises(RateLimitExceeded):
            await check_rate_limit(USER)

        # A different user is entirely unaffected.
        await check_rate_limit(OTHER_USER)
        await check_rate_limit(OTHER_USER)


class TestRetryAfter:
    async def test_retry_after_reflects_when_a_slot_frees_up(
        self, any_backend, limits, frozen_clock
    ):
        limits(requests=1, window_seconds=60)
        await check_rate_limit(USER)

        frozen_clock.advance(20)
        with pytest.raises(RateLimitExceeded) as exc_info:
            await check_rate_limit(USER)

        # 60s window, 20s elapsed -> roughly 40s remaining.
        assert 39 <= exc_info.value.retry_after_seconds <= 42

    async def test_retry_after_is_never_zero_or_negative(
        self, any_backend, limits, frozen_clock
    ):
        # A client told to wait 0 seconds retries instantly, which is
        # the behaviour the limiter exists to prevent.
        limits(requests=1, window_seconds=60)
        await check_rate_limit(USER)
        frozen_clock.advance(59.999)
        with pytest.raises(RateLimitExceeded) as exc_info:
            await check_rate_limit(USER)
        assert exc_info.value.retry_after_seconds >= 1


# ── The reason Redis is here at all ──────────────────────────────────

class TestSharedAcrossWorkers:
    async def test_the_allowance_is_global_not_per_worker(
        self, redis_backend, limits
    ):
        """THE BUG THIS BACKEND EXISTS TO FIX.

        With per-process counts, `uvicorn --workers 4` ran four
        independent limiters that each granted the full allowance, so a
        configured 20 was really 80 - and nothing said so, because a
        test suite runs in one process.

        reset() wipes this process's table, which is the closest thing
        to "the next request lands on a different worker". Under the old
        limiter that handed back a fresh allowance. Under Redis the
        count is where every worker can see it, so it does not.
        """
        limits(requests=3, window_seconds=60)
        await check_rate_limit(USER)
        await check_rate_limit(USER)

        rate_limit.reset()  # a different worker, with no local history
        await check_rate_limit(USER)

        rate_limit.reset()  # and another
        with pytest.raises(RateLimitExceeded):
            await check_rate_limit(USER)

    async def test_concurrent_requests_cannot_both_claim_the_last_slot(
        self, redis_backend, limits
    ):
        """The decision has to be atomic, not read-then-write.

        Ten requests fired at once against an allowance of three must
        produce at most three successes - never four because two of them
        read the same count before either wrote.
        """
        import asyncio

        limits(requests=3, window_seconds=60)
        results = await asyncio.gather(
            *(check_rate_limit(USER) for _ in range(10)), return_exceptions=True
        )
        allowed = [r for r in results if r is None]
        refused = [r for r in results if isinstance(r, RateLimitExceeded)]

        assert len(allowed) <= 3, "granted more than the allowance under concurrency"
        assert len(allowed) + len(refused) == 10, "something failed for another reason"

    async def test_the_window_key_carries_a_ttl(self, redis_backend, limits):
        """Otherwise every user who ever called /chat is stored forever."""
        limits(requests=5, window_seconds=60)
        await check_rate_limit(USER)

        ttl_ms = await redis_backend.pttl(rate_limit.REDIS_KEY_PREFIX + USER)
        assert 0 < ttl_ms <= 60_000


class TestRedisFailureDegradesRatherThanRefusing:
    async def test_an_unreachable_redis_falls_back_to_the_local_table(
        self, limits, monkeypatch
    ):
        """Fail-open-to-per-process, deliberately.

        Refusing every request because a cache is down converts a cache
        outage into a total outage. Falling back leaves exactly the
        protection this file had before Redis existed: bounded per
        worker, which is degraded rather than absent.
        """

        class BrokenRedis:
            def pipeline(self, transaction=True):
                raise ConnectionError("redis is down")

        rate_limit.reset()
        monkeypatch.setattr(redis_client, "_client", BrokenRedis())
        monkeypatch.setattr(redis_client, "_checked", True)

        limits(requests=2, window_seconds=60)
        await check_rate_limit(USER)
        await check_rate_limit(USER)

        # Still limited - by the in-process table, not by nothing.
        with pytest.raises(RateLimitExceeded):
            await check_rate_limit(USER)
        assert rate_limit.tracked_users() == 1


class TestMemoryIsBounded:
    """In-process only: Redis expires each key itself, so there is no
    unbounded table there to sweep."""

    async def test_idle_users_are_evicted_once_over_the_cap(
        self, memory_backend, limits, frozen_clock, monkeypatch
    ):
        limits(requests=5, window_seconds=60)
        monkeypatch.setattr(rate_limit, "MAX_TRACKED_USERS", 10)

        for i in range(10):
            await check_rate_limit(f"user-{i}")
        assert rate_limit.tracked_users() == 10

        # All ten fall out of the window, then a new arrival triggers
        # the sweep.
        frozen_clock.advance(61)
        await check_rate_limit("late-arrival")
        assert rate_limit.tracked_users() < 10


# ── Through the real endpoint ────────────────────────────────────────

@pytest_asyncio.fixture
async def client(db):
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


class TestEnforcedOnTheEndpoint:
    async def test_chat_returns_429_with_retry_after(
        self, client, real_order, limits, monkeypatch
    ):
        limits(requests=2)
        headers = {"Authorization": f"Bearer {create_test_token(str(real_order['buyerId']))}"}

        # The two ALLOWED requests would otherwise reach the real LLM
        # and spend quota on a test that has nothing to do with the
        # model. Stubbed out: what is under test is the third request
        # being refused, and that never reaches the orchestrator at all.
        async def _no_llm(user_id, message, history=None, products_out=None,
                          attribution_out=None):
            return "stubbed reply", [{"role": "assistant", "content": "stubbed reply"}]

        monkeypatch.setattr(chat_route, "run_conversation", _no_llm)

        await client.post("/chat", headers=headers, json={"message": "a", "session_id": "rl1"})
        await client.post("/chat", headers=headers, json={"message": "b", "session_id": "rl1"})
        third = await client.post(
            "/chat", headers=headers, json={"message": "c", "session_id": "rl1"}
        )

        assert third.status_code == 429
        assert "Retry-After" in third.headers
        assert int(third.headers["Retry-After"]) >= 1

    async def test_an_unauthenticated_flood_does_not_spend_a_users_allowance(
        self, client, real_order, limits
    ):
        """Authentication runs FIRST, by construction.

        If the limiter ran before auth it would have to key on something
        unverified, and anyone could exhaust another user's allowance
        just by claiming their id.
        """
        limits(requests=2)
        for _ in range(20):
            response = await client.post(
                "/chat",
                headers={"Authorization": "Bearer forged-token"},
                json={"message": "flood", "session_id": "rl2"},
            )
            assert response.status_code == 401

        # The real user's allowance is untouched.
        assert rate_limit.tracked_users() == 0
