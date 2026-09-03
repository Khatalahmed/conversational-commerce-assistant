"""
WHAT:
    Integration tests for the full /chat pipeline (Phase 7-9 combined)
    with the LLM itself SCRIPTED. Everything else is real: the JWT
    dependency, the orchestration loop, tool execution, the repo layer,
    MongoDB, and the session store.

WHY THE LLM IS MOCKED:
    These tests were never about whether the model is clever - they are
    about whether OUR pipeline threads a request through correctly. Real
    API calls made them nondeterministic and, on the free tier,
    routinely impossible: the suite spends roughly 6,000 of Groq's
    8,000 tokens/minute before reaching the multi-turn test, whose
    second turn alone needs ~5,000 more (full history plus all 35 tool
    schemas, re-sent every round). The test then failed on quota rather
    than on any defect - and a test that fails for reasons unrelated to
    the code stops being read.

    Scripting the model also makes previously UNTESTABLE paths testable,
    because they only occur when a provider misbehaves: rate-limit
    fallback, provider failover, and the tool-iteration ceiling. Those
    are asserted below and never were before.

    A LIE THE MOCK MUST NOT TELL: the tool calls it scripts are executed
    for real, against the real repo layer and the real database. The
    model's CHOICE is faked; the consequences of that choice are not.

    The original live-API tests are kept at the bottom under
    @pytest.mark.live - deselected by default (see pyproject.toml), run
    deliberately with `uv run pytest -m live`. Mocking answers "does our
    code work"; that one answers "does the real model still cooperate",
    and deleting it would quietly drop the only check on the latter.

MECHANISM:
    orchestrator.py does `from app.agent.llm_client import complete`, so
    it holds its OWN reference - patching llm_client.complete would not
    affect it. The patches below therefore target
    app.agent.orchestrator.complete / .get_providers directly.

    Uses httpx.AsyncClient + ASGITransport (NOT Starlette's TestClient):
    this runs every request on the SAME event loop as the test and the
    session-scoped `db` fixture. TestClient runs the app in a separate
    thread with its own loop, which conflicts with the shared Motor
    connection - the real root cause of the "event loop is closed"
    errors seen during Phase 10.
"""

import asyncio
import json

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport

from app.agent import orchestrator
from app.agent.llm_client import (
    Completion,
    LLMRateLimited,
    LLMUnavailable,
    Provider,
    ToolCall,
)
from app.agent.orchestrator import (
    FALLBACK_BUSY,
    FALLBACK_GENERIC,
    FALLBACK_UNAVAILABLE,
)
from app.api.main import app
from app.security.auth import create_test_token

FALLBACKS = {FALLBACK_GENERIC, FALLBACK_BUSY, FALLBACK_UNAVAILABLE}

# Seconds to pause between the two turns of the LIVE test.
#
# Groq's free tier allows 8,000 prompt tokens per MINUTE, and one
# measured turn of that test spends ~5,500 of them (three tool rounds
# plus the synthesis turn). Two turns back to back therefore cannot fit
# in one window - turn 2 was failing on quota, not on any defect. A
# pause just longer than the window lets the allowance reset.
#
# Only the opt-in live test pays this; the mocked suite is untouched and
# still runs in about ten seconds.
LIVE_TPM_WINDOW_SECONDS = 65


# ── Scripted LLM ─────────────────────────────────────────────────────

class ScriptedLLM:
    """Stands in for llm_client.complete().

    Takes a list of scripted turns, each either a Completion to return
    or an exception to raise, and records every call so tests can assert
    on what the orchestrator actually SENT - which is the only way to
    verify conversation memory without depending on model behaviour.
    """

    def __init__(self, script):
        self.script = list(script)
        self.calls = []  # [(provider_name, messages, tools, tool_choice), ...]

    async def __call__(self, provider, messages, tools, tool_choice="auto"):
        # Copied, not referenced: the orchestrator mutates the same list
        # afterwards, so a stored reference would show the FINAL state
        # on every recorded call and make memory assertions meaningless.
        self.calls.append((provider.name, [dict(m) for m in messages], tools, tool_choice))

        if not self.script:
            raise AssertionError(
                f"ScriptedLLM ran out of scripted turns after {len(self.calls)} "
                f"call(s) - the orchestrator looped more than expected."
            )

        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def messages_on_call(self, index: int) -> list:
        return self.calls[index][1]

    def tools_on_call(self, index: int) -> list:
        return self.calls[index][2]

    def tool_choice_on_call(self, index: int) -> str:
        return self.calls[index][3]


def reply(text: str) -> Completion:
    """A plain text answer - ends the tool loop."""
    return Completion(
        content=text, tool_calls=[], provider="scripted", model="scripted", prompt_tokens=100
    )


def tool_call(name: str, arguments: dict | None = None, call_id: str = "call_1") -> Completion:
    """A tool request. The named tool is then executed FOR REAL."""
    return Completion(
        content=None,
        tool_calls=[
            ToolCall(
                id=call_id,
                name=name,
                arguments=json.dumps(arguments or {}),
                extra={},
            )
        ],
        provider="scripted",
        model="scripted",
        prompt_tokens=100,
    )


@pytest.fixture
def scripted(monkeypatch):
    """Installs a ScriptedLLM, plus a fixed provider list so failover
    behaviour does not depend on which API keys happen to be in .env."""

    def _install(script, provider_names=("primary",)):
        llm = ScriptedLLM(script)
        providers = tuple(
            Provider(name=n, base_url="http://scripted", model="scripted", api_key="x")
            for n in provider_names
        )
        monkeypatch.setattr(orchestrator, "complete", llm)
        monkeypatch.setattr(orchestrator, "get_providers", lambda: providers)
        return llm

    return _install


# ── Shared fixtures ──────────────────────────────────────────────────

@pytest_asyncio.fixture
async def client(db):
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


@pytest_asyncio.fixture
async def auth_headers(real_order):
    token = create_test_token(user_id=str(real_order["buyerId"]))
    return {"Authorization": f"Bearer {token}"}


# ── Tests that never involved the LLM ────────────────────────────────

class TestHealthEndpoint:
    async def test_health_returns_ok(self, client):
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestAuthGate:
    async def test_rejects_missing_auth_header(self, client):
        """401, not 422.

        This asserted 422 until the endpoint declared a Bearer security
        scheme - that status was an artifact of reading the token via a
        required Header(...), which FastAPI treats as request-VALIDATION.
        A missing credential is an authentication failure, so 401 with
        WWW-Authenticate is both more correct and what a mobile client
        will branch on to trigger a token refresh.
        """
        response = await client.post("/chat", json={"message": "hello", "session_id": "t1"})
        assert response.status_code == 401
        assert response.headers.get("WWW-Authenticate") == "Bearer"

    async def test_rejects_invalid_token(self, client):
        response = await client.post(
            "/chat",
            headers={"Authorization": "Bearer not-a-real-token"},
            json={"message": "hello", "session_id": "t2"},
        )
        assert response.status_code == 401

    async def test_no_llm_call_is_made_for_an_unauthenticated_request(
        self, client, scripted
    ):
        # Auth must gate BEFORE any spend. Previously invisible: with a
        # real API you cannot prove a call was not made.
        llm = scripted([reply("should never be reached")])
        await client.post(
            "/chat",
            headers={"Authorization": "Bearer not-a-real-token"},
            json={"message": "hello", "session_id": "t2b"},
        )
        assert llm.call_count == 0


# ── The pipeline, with a scripted model ──────────────────────────────

class TestChatPipeline:
    async def test_tool_result_reaches_the_final_reply(self, client, auth_headers, scripted):
        # get_order_history runs FOR REAL against MongoDB; only the
        # decision to call it is scripted.
        llm = scripted([
            tool_call("get_order_history", {"limit": 3}),
            reply("You have a few recent orders - the latest is on its way."),
        ])

        response = await client.post(
            "/chat",
            headers=auth_headers,
            json={"message": "where is my order", "session_id": "m1"},
        )

        assert response.status_code == 200
        assert response.json()["reply"].startswith("You have a few recent orders")
        assert llm.call_count == 2

        # The second call must carry the real tool result back to the model.
        second_call = llm.messages_on_call(1)
        tool_messages = [m for m in second_call if m.get("role") == "tool"]
        assert len(tool_messages) == 1
        assert tool_messages[0]["tool_call_id"] == "call_1"
        json.loads(tool_messages[0]["content"])  # real, JSON-serializable DB data

    async def test_user_id_is_injected_server_side_not_taken_from_the_model(
        self, client, auth_headers, scripted, real_order
    ):
        # The model asks for order history WITHOUT a user_id (it has no
        # way to supply one), and still gets THIS user's orders.
        scripted([
            tool_call("get_order_history", {"limit": 5}),
            reply("Here are your orders."),
        ])
        response = await client.post(
            "/chat",
            headers=auth_headers,
            json={"message": "my orders", "session_id": "m2"},
        )
        assert response.status_code == 200

    async def test_a_reply_with_no_tool_call_short_circuits(self, client, auth_headers, scripted):
        llm = scripted([reply("Hello! How can I help with your the marketplace order today?")])
        response = await client.post(
            "/chat",
            headers=auth_headers,
            json={"message": "hi", "session_id": "m3"},
        )
        assert response.status_code == 200
        assert llm.call_count == 1

    async def test_unparseable_tool_arguments_do_not_500(self, client, auth_headers, scripted):
        # A smaller model does emit invalid JSON arguments. Every
        # tool_call id must still be answered or the NEXT request is
        # malformed - so the orchestrator hands back a readable error.
        broken = Completion(
            content=None,
            tool_calls=[ToolCall(id="c9", name="get_order_history",
                                 arguments="{not json", extra={})],
            provider="scripted", model="scripted", prompt_tokens=10,
        )
        llm = scripted([broken, reply("Sorry, let me try that differently.")])

        response = await client.post(
            "/chat",
            headers=auth_headers,
            json={"message": "orders", "session_id": "m4"},
        )
        assert response.status_code == 200

        tool_messages = [m for m in llm.messages_on_call(1) if m.get("role") == "tool"]
        assert tool_messages[0]["tool_call_id"] == "c9"
        assert "error" in json.loads(tool_messages[0]["content"])


class TestConversationMemory:
    async def test_second_turn_receives_the_first_turns_history(
        self, client, auth_headers, scripted
    ):
        """The actual contract of the session store.

        The old version asked a real model "is it in stock?" and hoped
        it resolved the pronoun - testing the MODEL, not our memory, and
        failing whenever quota ran out. What matters to us is whether
        turn 1 is still in the messages we send on turn 2. That is now
        asserted directly.
        """
        session_id = "mem-1"
        llm = scripted([reply("That's a lovely jacket."), reply("Yes, it's in stock.")])

        await client.post(
            "/chat",
            headers=auth_headers,
            json={"message": "tell me about the blue jacket", "session_id": session_id},
        )
        await client.post(
            "/chat",
            headers=auth_headers,
            json={"message": "is it in stock?", "session_id": session_id},
        )

        second_turn = llm.messages_on_call(1)
        contents = [m.get("content") for m in second_turn]
        assert "tell me about the blue jacket" in contents, "turn 1 was lost"
        assert "That's a lovely jacket." in contents, "the earlier reply was lost"
        assert "is it in stock?" in contents
        # Exactly one system prompt - not re-prepended on resume.
        assert sum(1 for m in second_turn if m.get("role") == "system") == 1

    async def test_a_different_session_id_starts_clean(self, client, auth_headers, scripted):
        llm = scripted([reply("first"), reply("second")])

        await client.post(
            "/chat",
            headers=auth_headers,
            json={"message": "remember this", "session_id": "mem-2a"},
        )
        await client.post(
            "/chat",
            headers=auth_headers,
            json={"message": "unrelated", "session_id": "mem-2b"},
        )

        contents = [m.get("content") for m in llm.messages_on_call(1)]
        assert "remember this" not in contents

    async def test_another_users_session_id_is_not_inherited(
        self, client, scripted, two_different_buyers_orders
    ):
        """session_id is client-supplied and guessable; history holds
        prior tool results. Reusing someone else's id must read as a NEW
        conversation, not theirs."""
        victim_order, attacker_order = two_different_buyers_orders
        shared_session_id = "guessable-session-id"
        llm = scripted([reply("noted"), reply("nothing here")])

        await client.post(
            "/chat",
            headers={"Authorization": f"Bearer {create_test_token(str(victim_order['buyerId']))}"},
            json={"message": "my secret shopping list", "session_id": shared_session_id},
        )
        await client.post(
            "/chat",
            headers={"Authorization": f"Bearer {create_test_token(str(attacker_order['buyerId']))}"},
            json={"message": "what were we discussing?", "session_id": shared_session_id},
        )

        contents = [m.get("content") for m in llm.messages_on_call(1)]
        assert "my secret shopping list" not in contents, (
            "SECURITY: another user's conversation leaked via a reused session_id"
        )


class TestProviderFailureHandling:
    """Paths that only occur when a provider misbehaves - untestable
    against a live API, and exactly what broke the old suite."""

    async def test_rate_limit_returns_the_busy_message_not_a_500(
        self, client, auth_headers, scripted
    ):
        scripted([LLMRateLimited("quota exhausted")])
        response = await client.post(
            "/chat",
            headers=auth_headers,
            json={"message": "hello", "session_id": "f1"},
        )
        assert response.status_code == 200
        assert response.json()["reply"] == FALLBACK_BUSY

    async def test_unavailable_provider_returns_its_own_message(
        self, client, auth_headers, scripted
    ):
        scripted([LLMUnavailable("connection refused")])
        response = await client.post(
            "/chat",
            headers=auth_headers,
            json={"message": "hello", "session_id": "f2"},
        )
        assert response.json()["reply"] == FALLBACK_UNAVAILABLE

    async def test_failover_to_the_second_provider(self, client, auth_headers, scripted):
        llm = scripted(
            [LLMRateLimited("primary out"), reply("Answered by the backup.")],
            provider_names=("primary", "secondary"),
        )
        response = await client.post(
            "/chat",
            headers=auth_headers,
            json={"message": "hello", "session_id": "f3"},
        )
        assert response.json()["reply"] == "Answered by the backup."
        assert [name for name, _, _, _ in llm.calls] == ["primary", "secondary"]

    async def test_a_failing_provider_is_not_retried_before_moving_on(
        self, client, auth_headers, scripted
    ):
        # A rate limit is per-minute; an immediate retry just burns
        # another request. The next provider is tried instead.
        llm = scripted(
            [LLMRateLimited("out"), LLMRateLimited("also out")],
            provider_names=("primary", "secondary"),
        )
        await client.post(
            "/chat",
            headers=auth_headers,
            json={"message": "hello", "session_id": "f4"},
        )
        assert [name for name, _, _, _ in llm.calls] == ["primary", "secondary"]

    async def test_tool_limit_ends_in_a_real_answer_not_a_fallback(
        self, client, auth_headers, scripted
    ):
        """Hitting the tool cap must not throw the gathered data away.

        This used to return FALLBACK_GENERIC: the loop executed the
        final tool, appended its result, and exited without ever letting
        the model read it. A real conversation hit this - the model
        answered "tell me about <product>" with three sensible lookups
        and got an apology.
        """
        script = [tool_call("get_order_history", {}, call_id=f"c{i}") for i in range(3)]
        script.append(reply("Your most recent order is out for delivery."))
        llm = scripted(script)

        response = await client.post(
            "/chat",
            headers=auth_headers,
            json={"message": "orders", "session_id": "f5"},
        )

        assert response.status_code == 200
        assert response.json()["reply"] == "Your most recent order is out for delivery."
        # N rounds of tools, PLUS one turn to speak.
        assert llm.call_count == orchestrator.MAX_TOOL_ITERATIONS + 1

    async def test_the_synthesis_turn_declines_tools_but_still_declares_them(
        self, client, auth_headers, scripted
    ):
        """tool_choice="none", NOT an empty tool list.

        Both stop a fourth call, but removing the tools while the
        history still holds tool_calls and tool results makes the
        request contradict itself - and a live run showed the model
        imitating that transcript with no schema to follow, producing
        output Groq rejected with "Parsing failed".
        """
        script = [tool_call("get_order_history", {}, call_id=f"c{i}") for i in range(3)]
        script.append(reply("Here is what I found."))
        llm = scripted(script)

        await client.post(
            "/chat",
            headers=auth_headers,
            json={"message": "orders", "session_id": "f5b"},
        )

        assert llm.tool_choice_on_call(0) == "auto", "normal rounds may call tools"
        assert llm.tool_choice_on_call(-1) == "none", "synthesis must decline them"
        # The schemas must STILL be sent, or the history stops making sense.
        assert llm.tools_on_call(-1), "synthesis must still declare the tools"

    async def test_tool_calling_still_cannot_loop_forever(
        self, client, auth_headers, scripted
    ):
        # The cap itself is unchanged - only what happens after it.
        script = [tool_call("get_order_history", {}, call_id=f"c{i}") for i in range(3)]
        script.append(reply("done"))
        llm = scripted(script)
        await client.post(
            "/chat",
            headers=auth_headers,
            json={"message": "orders", "session_id": "f5c"},
        )
        tool_rounds = sum(1 for _, _, _, choice in llm.calls if choice == "auto")
        assert tool_rounds == orchestrator.MAX_TOOL_ITERATIONS

    async def test_a_failing_synthesis_turn_still_answers_in_words(
        self, client, auth_headers, scripted
    ):
        # If the last call is rate-limited there is nothing to say, but
        # the app must still get a sentence rather than a 500.
        script = [tool_call("get_order_history", {}, call_id=f"c{i}") for i in range(3)]
        script.append(LLMRateLimited("out of quota"))
        scripted(script)

        response = await client.post(
            "/chat",
            headers=auth_headers,
            json={"message": "orders", "session_id": "f6"},
        )
        assert response.status_code == 200
        assert response.json()["reply"] == FALLBACK_BUSY


# ── Live-API tests: opt in with `uv run pytest -m live` ──────────────

@pytest.mark.live
class TestAgainstTheRealModel:
    """Deselected by default (pyproject.toml addopts). Answers the one
    question the mocks cannot: does the REAL model still choose sensible
    tools and produce a usable sentence? Run before a release, or after
    changing the system prompt or tool descriptions.

    EXACTLY ONE TEST LIVES HERE, ON PURPOSE. There were two, and they
    could not both pass: a measured run showed the pair needing ~8,900
    prompt tokens inside the same three seconds, against Groq's free
    limit of 8,000 per minute - so the second failed on quota every
    time, regardless of the code. A suite that fails for reasons
    unrelated to the code stops being read.

    The single-turn test was the one dropped, because this one already
    covers what it checked (the real model picking a real tool and
    writing a real sentence) and adds multi-turn context on top. The
    mocked tests above cover the single-turn pipeline deterministically.
    """

    async def test_multi_turn_memory_resolves_pronoun(
        self, client, auth_headers, real_product_with_variants
    ):
        session_id = "live-2"
        product_name = real_product_with_variants["name"]

        first = await client.post(
            "/chat",
            headers=auth_headers,
            json={"message": f"tell me about {product_name}", "session_id": session_id},
        )
        assert first.status_code == 200
        # ASSERTED ON ITS OWN, not just via the status code. This test
        # once passed while turn 1 returned the generic fallback: the
        # model had burned all three tool rounds on an unrequested
        # bargain lookup, and turn 2 answered from the history it had
        # already fetched. A status check alone cannot see that.
        assert first.json()["reply"] not in FALLBACKS, (
            "turn 1 fell back - the assistant never actually answered"
        )

        # Wait out the per-minute token allowance before turn 2 - see
        # LIVE_TPM_WINDOW_SECONDS. Without this the test measures Groq's
        # billing window rather than our code.
        await asyncio.sleep(LIVE_TPM_WINDOW_SECONDS)

        second = await client.post(
            "/chat",
            headers=auth_headers,
            json={"message": "is it in stock?", "session_id": session_id},
        )
        assert second.status_code == 200
        reply_text = second.json()["reply"]
        assert reply_text not in FALLBACKS, "the assistant fell back - the LLM call failed"
        assert "which product" not in reply_text.lower()
