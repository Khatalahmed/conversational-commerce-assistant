"""
WHAT:
    Tests for the streamed delivery path - llm_client.stream(), the
    orchestrator's progress events, and the /chat/stream endpoint.

WHY THESE, SPECIFICALLY:
    Streaming is a DELIVERY change, not a behaviour change, and the way
    it goes wrong is by quietly becoming a behaviour change. Three
    things are therefore pinned:

    1. stream() returns the same Completion complete() does. That is
       what lets the orchestration loop stay single - if it drifts, the
       two endpoints answer differently and only one of them is tested
       anywhere else.

    2. Tool calls survive being cut into fragments. The id and name
       arrive in the first frame for an index and the arguments dribble
       in across many, so a naive reader ends up with a tool call whose
       arguments are half a JSON object. Live, that surfaces as the
       assistant randomly failing to look things up.

    3. Nothing is retried once a token has reached the user. The
       buffered path can throw a bad response away; the streamed one has
       already spent it, and retrying paints a second answer on top of
       half of the first.

FLOW:
    Unit tests with a faked HTTP client, plus endpoint tests over
    ASGITransport with the orchestrator stubbed. No database, no LLM,
    no network.
"""

import json

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport

from app.agent import llm_client
from app.agent.llm_client import (
    LLMRateLimited,
    LLMUnavailable,
    Provider,
    stream,
)
from app.api.main import app
from app.api.routes import chat as chat_route
from app.security.auth import create_test_token

PROVIDER = Provider("test", "https://example.com/v1", "test-model", "key")


def sse(**payload) -> str:
    """One frame as a provider sends it."""
    return "data: " + json.dumps(payload)


def text_frame(text: str) -> str:
    return sse(choices=[{"delta": {"content": text}, "finish_reason": None}])


class _FakeStreamResponse:
    def __init__(self, lines, status_code=200, body=None):
        self.status_code = status_code
        self._lines = lines
        self._body = body or {}
        self.text = json.dumps(self._body)

    async def aread(self):
        return self.text.encode()

    def json(self):
        return self._body

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _FakeStreamContext:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *exc_info):
        return False


class _FakeClient:
    def __init__(self, response):
        self._response = response
        self.last_body = None

    def stream(self, method, url, headers=None, json=None):
        self.last_body = json
        return _FakeStreamContext(self._response)


@pytest.fixture
def fake_stream(monkeypatch):
    """Serves a canned SSE transcript to stream()."""

    def _apply(lines, status_code=200, body=None):
        client = _FakeClient(_FakeStreamResponse(lines, status_code, body))
        monkeypatch.setattr(llm_client, "_get_http_client", lambda: client)
        return client

    return _apply


class TestTextStreaming:
    async def test_fragments_are_delivered_as_they_arrive(self, fake_stream):
        fake_stream([text_frame("Hello"), text_frame(" there"), "data: [DONE]"])

        seen = []
        completion = await stream(PROVIDER, [], [], on_text=seen.append)

        assert seen == ["Hello", " there"]
        assert completion.content == "Hello there"

    async def test_the_completion_matches_the_buffered_shape(self, fake_stream):
        """A caller that ignores on_text must get exactly what complete()
        would have returned - that equivalence is what keeps one loop."""
        fake_stream([text_frame("Hi"), "data: [DONE]"])

        completion = await stream(PROVIDER, [], [])

        assert completion.content == "Hi"
        assert completion.tool_calls == []
        assert completion.provider == "test"
        assert completion.model == "test-model"

    async def test_usage_is_requested_and_read(self, fake_stream):
        """prompt_tokens is absent from a streamed response unless asked
        for, and it is the number that shows the prompt cache working."""
        client = fake_stream(
            [text_frame("Hi"), sse(choices=[], usage={"prompt_tokens": 1729})]
        )

        completion = await stream(PROVIDER, [], [])

        assert client.last_body["stream"] is True
        assert client.last_body["stream_options"] == {"include_usage": True}
        assert completion.prompt_tokens == 1729

    async def test_an_unparseable_frame_does_not_lose_the_answer(self, fake_stream):
        fake_stream([text_frame("Good"), "data: {not json", text_frame(" answer")])

        completion = await stream(PROVIDER, [], [])

        assert completion.content == "Good answer"

    async def test_no_content_reads_as_none_not_empty_string(self, fake_stream):
        """The orchestrator tests `completion.content or ''` for empty
        answers, and treats None and '' alike - but the buffered path
        returns None, so this one must too."""
        fake_stream(["data: [DONE]"])

        assert (await stream(PROVIDER, [], [])).content is None


class TestToolCallsSurviveFragmentation:
    async def test_arguments_are_reassembled_across_frames(self, fake_stream):
        fake_stream([
            sse(choices=[{"delta": {"tool_calls": [
                {"index": 0, "id": "call_1", "type": "function",
                 "function": {"name": "get_order_status", "arguments": ""}}
            ]}}]),
            sse(choices=[{"delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": '{"order_'}}
            ]}}]),
            sse(choices=[{"delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": 'id": "ORD1"}'}}
            ]}}]),
            "data: [DONE]",
        ])

        completion = await stream(PROVIDER, [], [])

        assert len(completion.tool_calls) == 1
        call = completion.tool_calls[0]
        assert call.id == "call_1"
        assert call.name == "get_order_status"
        # The orchestrator json.loads() this; half an object would be a
        # parse failure and a wasted round.
        assert json.loads(call.arguments) == {"order_id": "ORD1"}

    async def test_parallel_tool_calls_stay_separate(self, fake_stream):
        fake_stream([
            sse(choices=[{"delta": {"tool_calls": [
                {"index": 0, "id": "a", "function": {"name": "get_cart", "arguments": "{}"}},
                {"index": 1, "id": "b", "function": {"name": "get_live_now", "arguments": "{}"}},
            ]}}]),
            "data: [DONE]",
        ])

        completion = await stream(PROVIDER, [], [])

        assert [c.name for c in completion.tool_calls] == ["get_cart", "get_live_now"]
        assert [c.id for c in completion.tool_calls] == ["a", "b"]

    async def test_provider_specific_fields_survive_the_round_trip(self, fake_stream):
        """Google rejects its own tool call on the next turn if the
        thought_signature it issued is not echoed back - the same reason
        ToolCall.extra exists on the buffered path."""
        fake_stream([
            sse(choices=[{"delta": {"tool_calls": [
                {"index": 0, "id": "a", "function": {"name": "get_cart", "arguments": "{}"},
                 "extra_content": {"google": {"thought_signature": "sig"}}}
            ]}}]),
            "data: [DONE]",
        ])

        completion = await stream(PROVIDER, [], [])

        assert completion.tool_calls[0].to_message_part()["extra_content"] == {
            "google": {"thought_signature": "sig"}
        }

    async def test_a_nameless_fragment_is_dropped(self, fake_stream):
        """Better no tool call than one the registry cannot look up."""
        fake_stream([
            sse(choices=[{"delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": "{}"}}
            ]}}]),
            "data: [DONE]",
        ])

        assert (await stream(PROVIDER, [], [])).tool_calls == []


class TestFailuresBeforeAnyBytes:
    async def test_a_rate_limit_still_raises_the_same_type(self, fake_stream):
        """An HTTP error arrives before the body does, so failover works
        exactly as it does on the buffered path."""
        fake_stream([], status_code=429, body={"error": {"message": "slow down"}})

        with pytest.raises(LLMRateLimited):
            await stream(PROVIDER, [], [])

    async def test_a_dead_provider_still_raises_unavailable(self, fake_stream):
        fake_stream([], status_code=404, body={"error": {"message": "no model"}})

        with pytest.raises(LLMUnavailable):
            await stream(PROVIDER, [], [])


class TestNothingIsRetriedAfterTheFirstToken:
    async def test_a_mid_stream_failure_ends_the_answer(self, monkeypatch):
        """Retrying would paint a second answer over half of the first.

        The user has already read those words; the only honest thing
        left is to stop, which the endpoint reports as an error rather
        than silently restarting somewhere else.
        """
        from app.agent import orchestrator

        monkeypatch.setattr(
            orchestrator, "get_providers", lambda: (PROVIDER, PROVIDER)
        )

        calls = {"n": 0}

        async def exploding_stream(provider, messages, tools, tool_choice, on_text=None):
            calls["n"] += 1
            on_text("half a sen")
            raise LLMUnavailable("connection dropped")

        monkeypatch.setattr(orchestrator, "stream", exploding_stream)

        seen = []
        with pytest.raises(LLMUnavailable):
            await orchestrator._complete_with_failover(
                [], "user-1", on_text=seen.append
            )

        assert seen == ["half a sen"]
        assert calls["n"] == 1, "a second provider was tried after the user saw text"

    async def test_failover_still_happens_when_nothing_was_streamed(self, monkeypatch):
        """The guard must not disable failover outright - only after the
        point of no return."""
        from app.agent import orchestrator

        good = Provider("second", "https://example.com/v1", "m", "k")
        monkeypatch.setattr(orchestrator, "get_providers", lambda: (PROVIDER, good))

        tried = []

        async def flaky_stream(provider, messages, tools, tool_choice, on_text=None):
            tried.append(provider.name)
            if provider.name == "test":
                raise LLMUnavailable("dead")
            on_text("answer")
            return llm_client.Completion("answer", [], provider.name, "m", 0)

        monkeypatch.setattr(orchestrator, "stream", flaky_stream)

        completion = await orchestrator._complete_with_failover(
            [], "user-1", on_text=lambda _: None
        )

        assert tried == ["test", "second"]
        assert completion.provider == "second"


# ── The endpoint ─────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def client(db):
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


def parse_frames(body: str) -> list[tuple[str, dict]]:
    """The same parse the demo page does, kept honest by being written
    twice against the same bytes."""
    out = []
    for frame in body.split("\n\n"):
        name, data = None, None
        for line in frame.split("\n"):
            if line.startswith("event:"):
                name = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data = line[len("data:"):].strip()
        if name and data:
            out.append((name, json.loads(data)))
    return out


class TestChatStreamEndpoint:
    async def test_status_then_tokens_then_done(self, client, real_order, monkeypatch):
        async def fake_conversation(user_id, message, history=None, on_event=None):
            on_event({"type": "status", "tool": "get_order_history",
                      "label": "Looking up your orders"})
            for piece in ("It ", "arrives ", "Tuesday."):
                on_event({"type": "token", "text": piece})
            return "It arrives Tuesday.", [{"role": "assistant", "content": "It arrives Tuesday."}]

        monkeypatch.setattr(chat_route, "run_conversation", fake_conversation)
        headers = {"Authorization": f"Bearer {create_test_token(str(real_order['buyerId']))}"}

        response = await client.post(
            "/chat/stream", headers=headers,
            json={"message": "where is my order", "session_id": "st-1"},
        )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

        frames = parse_frames(response.text)
        assert [name for name, _ in frames] == [
            "status", "token", "token", "token", "done"
        ]
        assert frames[0][1]["label"] == "Looking up your orders"
        # The reply in `done` must equal what the tokens spelled, or a
        # client that renders tokens sees the text change under it.
        streamed = "".join(p["text"] for name, p in frames if name == "token")
        assert frames[-1][1]["reply"] == streamed

    async def test_a_reply_with_no_tokens_still_reaches_the_user(
        self, client, real_order, monkeypatch
    ):
        """The fallbacks - rate limited, provider down - are returned as
        plain text without ever being streamed. done carries them, which
        is why the client needs no special case."""
        async def fallback_only(user_id, message, history=None, on_event=None):
            return "I'm handling a lot of requests right now", []

        monkeypatch.setattr(chat_route, "run_conversation", fallback_only)
        headers = {"Authorization": f"Bearer {create_test_token(str(real_order['buyerId']))}"}

        response = await client.post(
            "/chat/stream", headers=headers,
            json={"message": "hi", "session_id": "st-2"},
        )

        frames = parse_frames(response.text)
        assert [name for name, _ in frames] == ["done"]
        assert "handling a lot of requests" in frames[0][1]["reply"]

    async def test_an_unexpected_failure_is_a_sentence_not_a_severed_stream(
        self, client, real_order, monkeypatch
    ):
        async def boom(user_id, message, history=None, on_event=None):
            raise ValueError("something nobody planned for")

        monkeypatch.setattr(chat_route, "run_conversation", boom)
        headers = {"Authorization": f"Bearer {create_test_token(str(real_order['buyerId']))}"}

        response = await client.post(
            "/chat/stream", headers=headers,
            json={"message": "hi", "session_id": "st-3"},
        )

        frames = parse_frames(response.text)
        assert [name for name, _ in frames] == ["error"]
        assert "try rephrasing" in frames[0][1]["message"]

    async def test_it_is_behind_the_same_auth_as_chat(self, client):
        response = await client.post(
            "/chat/stream",
            headers={"Authorization": "Bearer forged"},
            json={"message": "hi", "session_id": "st-4"},
        )
        assert response.status_code == 401

    async def test_it_is_behind_the_same_rate_limit_as_chat(
        self, client, real_order, monkeypatch
    ):
        """A streaming endpoint that skipped the limiter would be the
        cheapest way to exhaust the LLM quota."""
        from app.config.settings import Settings
        from app.security import rate_limit

        settings = Settings(
            mongodb_uri="mongodb://localhost/test",
            llm_api_key="k",
            jwt_secret="s" * 64,
            chat_rate_limit_requests=1,
        )
        monkeypatch.setattr(rate_limit, "get_settings", lambda: settings)

        async def quick(user_id, message, history=None, on_event=None):
            return "ok", []

        monkeypatch.setattr(chat_route, "run_conversation", quick)
        headers = {"Authorization": f"Bearer {create_test_token(str(real_order['buyerId']))}"}
        body = {"message": "hi", "session_id": "st-5"}

        assert (await client.post("/chat/stream", headers=headers, json=body)).status_code == 200
        second = await client.post("/chat/stream", headers=headers, json=body)
        assert second.status_code == 429
