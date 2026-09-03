"""
WHAT:
    Tests that the tool calls of ONE round run together, and that running
    them together did not quietly reorder anything.

WHY THE ORDERING TESTS ARE THE POINT:
    Making the calls concurrent is a few lines. What concurrency breaks
    is everything downstream that assumed "the order they finished" and
    "the order the model asked" were the same sentence:

      - a provider handed tool replies that do not line up with its own
        calls rejects the NEXT request, so a reordered history is a bug
        that surfaces one turn later than it was caused;

      - _deliver_products takes the LAST product batch, meaning the tool
        the answer was written from. Under a shared collector that
        silently becomes whichever query finished last, so the cards
        would follow network timing rather than the model.

    So the fake tools below deliberately finish in the WRONG order: the
    one called first is the slowest. Sequential code cannot fail these
    tests, and a careless parallel version cannot pass them.

FLOW:
    A scripted LLM (no network) asks for several tools in one completion.
    The tools are fakes with controlled durations. No database.
"""

import asyncio
import json
import time

import pytest

from app.agent import orchestrator
from app.agent.llm_client import Completion, Provider, ToolCall

PROVIDERS = (Provider(name="scripted", base_url="http://x", model="m", api_key="k"),)

SYSTEM = [{"role": "system", "content": "test"}]


def calls(*specs) -> Completion:
    """One completion asking for several tools at once."""
    return Completion(
        content=None,
        tool_calls=[
            ToolCall(id=cid, name=name, arguments=args, extra={})
            for cid, name, args in specs
        ],
        provider="scripted", model="m", prompt_tokens=10,
    )


def answer(text="done") -> Completion:
    return Completion(content=text, tool_calls=[], provider="scripted",
                      model="m", prompt_tokens=10)


class Scripted:
    def __init__(self, *steps):
        self.steps = list(steps)
        self.seen = []

    async def __call__(self, provider, messages, tools, tool_choice="auto"):
        self.seen.append([dict(m) for m in messages])
        assert self.steps, "orchestrator looped more than the script allows"
        return self.steps.pop(0)


@pytest.fixture
def wired(monkeypatch):
    """Scripted LLM, fixed providers, and no thumbnail lookup."""
    async def no_thumbs(cards):
        return cards

    monkeypatch.setattr(orchestrator, "get_providers", lambda: PROVIDERS)
    monkeypatch.setattr(orchestrator, "attach_thumbnails", no_thumbs)

    def _install(*steps):
        llm = Scripted(*steps)
        monkeypatch.setattr(orchestrator, "complete", llm)
        return llm

    return _install


def fake_tool(delay, payload):
    async def run(**kwargs):
        await asyncio.sleep(delay)
        return payload
    return run


def registry(monkeypatch, **tools):
    from app.agent import tools as tools_mod
    for name, func in tools.items():
        monkeypatch.setitem(tools_mod.TOOL_REGISTRY, name, (func, False))


def product(pid, name):
    return {"_id": pid, "name": name, "price": 100, "discountedPrice": 100,
            "category": "Men"}


class TestTheyOverlap:
    async def test_three_lookups_take_one_lookup_of_time(self, wired, monkeypatch):
        registry(
            monkeypatch,
            get_cart=fake_tool(0.15, {"items": []}),
            get_saved_items=fake_tool(0.15, {"savedProducts": []}),
            get_unread_notifications=fake_tool(0.15, []),
        )
        wired(
            calls(("a", "get_cart", "{}"),
                  ("b", "get_saved_items", "{}"),
                  ("c", "get_unread_notifications", "{}")),
            answer(),
        )

        started = time.perf_counter()
        reply, _ = await orchestrator.run_conversation("u1", "q", history=list(SYSTEM))
        elapsed = time.perf_counter() - started

        assert reply == "done"
        assert elapsed < 0.30, (
            f"three 0.15s lookups took {elapsed:.2f}s - that is sequential"
        )


class TestNothingGotReordered:
    async def test_tool_replies_line_up_with_the_calls(self, wired, monkeypatch):
        """The first call is the SLOWEST, so completion order is the
        reverse of call order. A provider rejects the next request when
        the history does not match its own call list."""
        registry(
            monkeypatch,
            get_cart=fake_tool(0.20, {"items": ["slow"]}),
            get_saved_items=fake_tool(0.10, {"savedProducts": []}),
            get_unread_notifications=fake_tool(0.01, ["fast"]),
        )
        wired(
            calls(("a", "get_cart", "{}"),
                  ("b", "get_saved_items", "{}"),
                  ("c", "get_unread_notifications", "{}")),
            answer(),
        )

        _, messages = await orchestrator.run_conversation("u1", "q", history=list(SYSTEM))
        ids = [m["tool_call_id"] for m in messages if m.get("role") == "tool"]
        assert ids == ["a", "b", "c"], "tool replies must follow the model's call order"

    async def test_cards_come_from_the_last_called_not_the_last_finished(
        self, wired, monkeypatch
    ):
        """THE SUBTLE ONE. Two product tools in a round: the first is
        slow, the second fast. _deliver_products takes the last batch,
        which has to mean the one the model asked for last."""
        registry(
            monkeypatch,
            search_products=fake_tool(0.20, [product("a" * 24, "SLOW-FIRST-CALLED")]),
            get_trending_products=fake_tool(0.01, [product("b" * 24, "FAST-LAST-CALLED")]),
        )
        wired(
            calls(("a", "search_products", "{}"),
                  ("b", "get_trending_products", "{}")),
            answer(),
        )

        out: list = []
        await orchestrator.run_conversation(
            "u1", "q", history=list(SYSTEM), products_out=out
        )
        assert [c["name"] for c in out] == ["FAST-LAST-CALLED"]


class TestFailuresStayContained:
    async def test_an_escaped_exception_still_answers_every_call_id(
        self, wired, monkeypatch
    ):
        """execute_tool answers its own failures, so an exception reaching
        the gather means one escaped it. Every tool_call id must still
        come back or the NEXT request is invalid."""
        async def exploding(name, args, user_id, collector=None):
            if name == "get_cart":
                raise RuntimeError("escaped")
            return json.dumps({"ok": name})

        monkeypatch.setattr(orchestrator, "execute_tool", exploding)
        wired(
            calls(("a", "get_cart", "{}"), ("b", "get_saved_items", "{}")),
            answer(),
        )

        _, messages = await orchestrator.run_conversation("u1", "q", history=list(SYSTEM))
        tools = [m for m in messages if m.get("role") == "tool"]
        assert [m["tool_call_id"] for m in tools] == ["a", "b"]
        assert "unavailable" in json.loads(tools[0]["content"])["error"].lower()
        assert json.loads(tools[1]["content"]) == {"ok": "get_saved_items"}

    async def test_a_malformed_call_is_answered_without_being_run(
        self, wired, monkeypatch
    ):
        ran = []

        async def counted(**kwargs):
            ran.append(1)
            return {"items": []}

        registry(monkeypatch, get_cart=counted, get_saved_items=counted)
        wired(
            calls(("a", "get_cart", "not json at all"),
                  ("b", "get_saved_items", "{}")),
            answer(),
        )

        _, messages = await orchestrator.run_conversation("u1", "q", history=list(SYSTEM))
        tools = [m for m in messages if m.get("role") == "tool"]
        assert [m["tool_call_id"] for m in tools] == ["a", "b"]
        assert "not a valid JSON object" in tools[0]["content"]
        assert len(ran) == 1, "the malformed call must not reach a tool"
