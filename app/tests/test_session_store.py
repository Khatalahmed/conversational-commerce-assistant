"""
WHAT:
    Tests for conversation memory, run against BOTH backends - the
    in-process dict and Redis (via fakeredis, so no server is needed).

WHY THIS APPROACH:
    The interface is identical between backends, which is precisely how a
    difference in BEHAVIOUR hides. Every contract test below is
    parametrised over both, so "works in dev, breaks in production" is
    caught here rather than in production.

    The test that matters most is survives_a_restart: it simulates the
    exact failure that motivated Redis at all. Editing a file under
    `uvicorn --reload` restarted the process, every conversation was
    erased, and the next "anything similar?" answered "similar to what?".
    Under Redis that must not happen; under the dict it provably does,
    and that is asserted too - a fallback whose limitation is
    undocumented gets deployed by accident.

FLOW:
    Pure unit tests. No database, no network, no LLM.
"""

import json

import pytest

from app.db import redis_client
from app.memory import session_store

USER = "000000000000000000000001"
OTHER_USER = "000000000000000000000002"


def conversation(*user_messages: str) -> list:
    msgs = [{"role": "system", "content": "You are the marketplace assistant."}]
    for text in user_messages:
        msgs.append({"role": "user", "content": text})
        msgs.append({"role": "assistant", "content": f"answer to {text}"})
    return msgs


@pytest.fixture
async def redis_backend(monkeypatch):
    """A real redis-py client API backed by an in-memory fake."""
    from fakeredis.aioredis import FakeRedis

    client = FakeRedis(decode_responses=True)
    session_store.reset_for_tests()
    monkeypatch.setattr(redis_client, "_client", client)
    monkeypatch.setattr(redis_client, "_checked", True)
    yield client
    await client.aclose()


@pytest.fixture
def memory_backend(monkeypatch):
    """The in-process dict - what runs when REDIS_URL is unset."""
    session_store.reset_for_tests()
    monkeypatch.setattr(redis_client, "_client", None)
    monkeypatch.setattr(redis_client, "_checked", True)
    return None


@pytest.fixture(params=["memory", "redis"])
async def any_backend(request, monkeypatch):
    """Both backends must satisfy the same contract."""
    session_store.reset_for_tests()
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


class TestContractHoldsOnBothBackends:
    async def test_a_new_session_has_no_history(self, any_backend):
        assert await session_store.get_session(USER, "brand-new") is None

    async def test_save_then_get_round_trips(self, any_backend):
        msgs = conversation("tell me about the blue jacket")
        await session_store.save_session(USER, "s1", msgs)

        stored = await session_store.get_session(USER, "s1")
        assert stored is not None
        contents = [m.get("content") for m in stored]
        assert "tell me about the blue jacket" in contents
        assert stored[0]["role"] == "system"

    async def test_another_users_session_id_is_not_inherited(self, any_backend):
        """session_id is client-supplied and guessable, and history holds
        prior tool results. Reusing someone else's must read as new."""
        await session_store.save_session(USER, "guessable", conversation("my secret list"))

        stolen = await session_store.get_session(OTHER_USER, "guessable")
        assert stolen is None

    async def test_clear_removes_it(self, any_backend):
        await session_store.save_session(USER, "s2", conversation("hello"))
        await session_store.clear_session(USER, "s2")
        assert await session_store.get_session(USER, "s2") is None

    async def test_history_is_trimmed_to_the_turn_limit(self, any_backend):
        msgs = conversation(*[f"message {i}" for i in range(20)])
        await session_store.save_session(USER, "s3", msgs)

        stored = await session_store.get_session(USER, "s3")
        user_turns = [m for m in stored if m.get("role") == "user"]
        assert len(user_turns) == session_store.MAX_TURNS
        # The most RECENT turns survive, not the oldest.
        assert user_turns[-1]["content"] == "message 19"
        # And the system prompt is never trimmed away.
        assert stored[0]["role"] == "system"

    async def test_tool_exchanges_survive_the_round_trip(self, any_backend):
        """Structured tool calls must come back intact - Redis stores
        JSON, and a lossy encode would break the NEXT API call rather
        than failing here."""
        msgs = [
            {"role": "system", "content": "prompt"},
            {"role": "user", "content": "where is my order"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "get_order_history", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": '{"orders": []}'},
            {"role": "assistant", "content": "You have no orders."},
        ]
        await session_store.save_session(USER, "s4", msgs)

        stored = await session_store.get_session(USER, "s4")
        call = next(m for m in stored if m.get("tool_calls"))
        assert call["tool_calls"][0]["id"] == "call_1"
        assert call["tool_calls"][0]["function"]["name"] == "get_order_history"
        result = next(m for m in stored if m.get("role") == "tool")
        assert json.loads(result["content"]) == {"orders": []}


class TestTheReasonRedisExists:
    async def test_redis_survives_a_restart(self, redis_backend):
        """THE test. A process restart must not erase a conversation.

        Simulated by clearing all in-process state - which is exactly
        what a restart does - while Redis keeps its data, as a real
        server would across a deploy.
        """
        await session_store.save_session(USER, "s5", conversation("tell me about the buddha"))

        session_store._sessions.clear()  # the "restart"

        stored = await session_store.get_session(USER, "s5")
        assert stored is not None, "history did not survive the restart"
        assert "tell me about the buddha" in [m.get("content") for m in stored]

    async def test_the_in_process_fallback_does_not_survive_a_restart(
        self, memory_backend
    ):
        """Asserted deliberately. This limitation is real and is what
        gets deployed when REDIS_URL is forgotten - documenting it in a
        test means nobody discovers it from a confused user instead."""
        await session_store.save_session(USER, "s6", conversation("tell me about the buddha"))
        assert await session_store.get_session(USER, "s6") is not None

        session_store._sessions.clear()  # the "restart"

        assert await session_store.get_session(USER, "s6") is None

    async def test_redis_sets_an_expiry(self, redis_backend):
        """Without a TTL, abandoned conversations accumulate forever -
        the same unbounded-growth problem the dict has, just on someone
        else's disk."""
        await session_store.save_session(USER, "s7", conversation("hi"))
        ttl = await redis_backend.ttl(session_store.REDIS_KEY_PREFIX + f"{USER}:s7")
        assert 0 < ttl <= session_store.SESSION_TIMEOUT_MINUTES * 60

    async def test_the_expiry_is_refreshed_on_every_write(self, redis_backend):
        """The timeout should measure INACTIVITY, not total conversation
        age - a long chat must not be cut off mid-way."""
        key = session_store.REDIS_KEY_PREFIX + f"{USER}:s8"
        await session_store.save_session(USER, "s8", conversation("one"))
        await redis_backend.expire(key, 5)          # simulate time passing
        await session_store.save_session(USER, "s8", conversation("one", "two"))
        assert await redis_backend.ttl(key) > 5


class TestDegradesRatherThanBreaks:
    async def test_a_failing_read_returns_no_history_not_an_error(
        self, redis_backend, monkeypatch
    ):
        """Losing the cache should cost the user CONTEXT, not their
        answer. Raising here would turn a cache blip into a 500."""
        async def boom(*a, **k):
            raise ConnectionError("redis is down")

        monkeypatch.setattr(redis_backend, "get", boom)
        assert await session_store.get_session(USER, "s9") is None

    async def test_a_failing_write_does_not_raise(self, redis_backend, monkeypatch):
        # The user has already been answered by this point; failing now
        # would discard a good response over a cache problem.
        async def boom(*a, **k):
            raise ConnectionError("redis is down")

        monkeypatch.setattr(redis_backend, "set", boom)
        await session_store.save_session(USER, "s10", conversation("hi"))

    async def test_corrupt_stored_data_is_discarded_not_raised(
        self, redis_backend
    ):
        await redis_backend.set(
            session_store.REDIS_KEY_PREFIX + f"{USER}:s11", "not valid json{"
        )
        assert await session_store.get_session(USER, "s11") is None

    async def test_missing_redis_url_falls_back_to_memory(self, monkeypatch):
        """The POC must still run with nothing but a MongoDB URI."""
        from app.config.settings import Settings

        session_store.reset_for_tests()
        settings = Settings(
            mongodb_uri="mongodb://localhost/test",
            llm_api_key="k",
            jwt_secret="s" * 64,
            redis_url=None,
        )
        monkeypatch.setattr(redis_client, "get_settings", lambda: settings)

        assert await redis_client.get_redis() is None
        await session_store.save_session(USER, "s12", conversation("hi"))
        assert await session_store.get_session(USER, "s12") is not None

    async def test_an_upstash_rest_url_is_rejected_with_a_clear_reason(
        self, monkeypatch
    ):
        """Upstash shows REST credentials first, so this WILL be pasted
        in by someone. redis-py cannot speak that protocol, and its own
        error says nothing about the endpoint being the wrong kind."""
        from app.config.settings import Settings

        session_store.reset_for_tests()
        settings = Settings(
            mongodb_uri="mongodb://localhost/test",
            llm_api_key="k",
            jwt_secret="s" * 64,
            redis_url="https://my-db-12345.upstash.io",
        )
        monkeypatch.setattr(redis_client, "get_settings", lambda: settings)

        assert await redis_client.get_redis() is None
        # ...and the app still works, on the fallback.
        await session_store.save_session(USER, "s13", conversation("hi"))
        assert await session_store.get_session(USER, "s13") is not None

    async def test_blank_redis_url_is_treated_as_unset(self):
        # A copied .env.example gives "" - which must not be handed to
        # redis.from_url() as if it were a real URL.
        from app.config.settings import Settings

        settings = Settings(
            mongodb_uri="mongodb://localhost/test",
            llm_api_key="k",
            jwt_secret="s" * 64,
            redis_url="   ",
        )
        assert settings.redis_url is None
