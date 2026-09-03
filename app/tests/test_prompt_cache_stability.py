"""
WHAT:
    Guards the cacheable prefix of every LLM request - the tool schemas
    and the system prompt - against becoming unstable.

WHY THIS MATTERS MORE THAN IT LOOKS:
    Groq caches prompts automatically, with no code changes, and gives
    cached prefix tokens a 50% discount AND excludes them from rate
    limits. That second part is why a measured demo run pushed ~113,000
    tokens through a nominally 8,000-tokens-per-minute account without
    ever being throttled.

    Caching is a PREFIX match: one byte different anywhere before the
    varying part and everything after it misses. So a single
    datetime.now() in the system prompt, or an unsorted list from
    MongoDB, silently doubles the bill and reinstates the rate limit.

    Nothing raises when that happens. The assistant keeps working, just
    more expensively and more slowly, and the cause is invisible unless
    someone thinks to look. That is exactly the kind of regression a
    test should catch, because a human never will.

WHAT IS CHECKED:
    - the system prompt is byte-identical across builds
    - the tool schemas serialize deterministically, in a stable order
    - no timestamp, UUID or ObjectId is embedded in the prompt
    - the one DYNAMIC input (real category values from MongoDB) is
      sorted, since MongoDB's distinct() gives no order guarantee

FLOW:
    Mostly pure unit tests - the categories lookup is stubbed, so these
    run without a database. One DB-backed test covers the real sort.
"""

import hashlib
import json
import re

import pytest

from app.agent import orchestrator
from app.agent.tools import TOOLS

FAKE_CATEGORIES = {
    "categories": ["Bed & Bath", "Electronics", "Home Decor"],
    "subCategories": ["Bedsheets", "Clocks", "Televisions"],
}


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.fixture
def stub_categories(monkeypatch):
    """Removes the database from the equation so these tests measure the
    PROMPT's determinism, not MongoDB's."""

    async def _categories():
        return FAKE_CATEGORIES

    monkeypatch.setattr(
        orchestrator.products_repo, "get_distinct_categories", _categories
    )


class TestTheSystemPromptIsStable:
    async def test_two_builds_are_byte_identical(self, stub_categories):
        first = await orchestrator.build_system_prompt()
        second = await orchestrator.build_system_prompt()
        assert digest(first) == digest(second), (
            "the system prompt changed between builds - every LLM call will "
            "miss the prompt cache"
        )

    async def test_no_timestamp_or_id_is_embedded(self, stub_categories):
        """The classic cache killer. A date, a clock time, a request id or
        an ObjectId in the prompt makes every single call unique."""
        prompt = await orchestrator.build_system_prompt()
        found = re.findall(
            r"\d{4}-\d{2}-\d{2}"          # a date
            r"|\d{2}:\d{2}:\d{2}"          # a clock time
            r"|[0-9a-f]{24}"               # a Mongo ObjectId
            r"|[0-9a-f]{8}-[0-9a-f]{4}",   # a UUID
            prompt,
        )
        assert not found, f"volatile values in the system prompt: {found}"

    async def test_the_prompt_still_contains_its_grounding_data(
        self, stub_categories
    ):
        # Stability is worthless if achieved by dropping the real values -
        # the model needs them to avoid inventing search filters.
        prompt = await orchestrator.build_system_prompt()
        assert "Home Decor" in prompt
        assert "Televisions" in prompt


class TestTheToolSchemasAreStable:
    def test_serialization_is_deterministic(self):
        assert json.dumps(TOOLS) == json.dumps(TOOLS)

    def test_tool_order_does_not_depend_on_a_set_or_dict_ordering(self):
        """TOOLS is a literal list, and must stay one. Building it from a
        set or by iterating a dict of handlers would reorder between runs
        and invalidate the cache on every process restart."""
        names = [t["function"]["name"] for t in TOOLS]
        assert names == [t["function"]["name"] for t in TOOLS]
        assert len(names) == len(set(names)), "duplicate tool names"

    def test_every_tool_has_the_fields_the_providers_require(self):
        for tool in TOOLS:
            assert tool.get("type") == "function"
            fn = tool["function"]
            assert fn.get("name") and fn.get("description")
            assert "parameters" in fn


class TestTheDynamicInputIsSorted:
    async def test_real_categories_come_back_sorted(self, db):
        """The ONE genuinely dynamic part of the prompt.

        MongoDB's distinct() makes no ordering guarantee, so without an
        explicit sort the same catalogue could render two different
        prompts - and the cache would miss on alternate calls for no
        visible reason.
        """
        from app.repos import products_repo

        result = await products_repo.get_distinct_categories()
        assert result["categories"] == sorted(result["categories"])
        assert result["subCategories"] == sorted(result["subCategories"])


class TestTheCacheIsMeasuredNotAssumed:
    """The module above guards the prefix from drifting, and warm_cache.py
    pre-warms it - but until Completion carried cached_tokens, NOTHING in
    the running service could tell whether either worked. A stray byte
    would double the bill and reinstate the rate limit while the
    assistant kept answering perfectly, which is exactly the failure that
    stays invisible without a number.

    Measured against the real prefix on azure/gpt-5-mini: 1,792 of 1,897
    prompt tokens cached, 94%, stable across consecutive calls.
    """

    def test_the_nested_cached_count_is_read(self):
        from app.agent.llm_client import _cached_tokens
        assert _cached_tokens(
            {"prompt_tokens": 1897, "prompt_tokens_details": {"cached_tokens": 1792}}
        ) == 1792

    def test_a_provider_that_does_not_cache_reads_as_zero(self):
        from app.agent.llm_client import _cached_tokens
        assert _cached_tokens({"prompt_tokens": 1897}) == 0

    def test_a_null_details_object_reads_as_zero(self):
        """At least one provider sends the key as null rather than
        omitting it, which `or {}` has to absorb."""
        from app.agent.llm_client import _cached_tokens
        assert _cached_tokens({"prompt_tokens_details": None}) == 0
        assert _cached_tokens({"prompt_tokens_details": {"cached_tokens": None}}) == 0

    def test_a_details_object_of_the_wrong_shape_does_not_raise(self):
        from app.agent.llm_client import _cached_tokens
        assert _cached_tokens({"prompt_tokens_details": []}) == 0
        assert _cached_tokens({}) == 0

    def test_completion_defaults_to_zero_so_old_constructors_still_work(self):
        """Completion is built positionally in several tests with five
        arguments. The new field must not break them."""
        from app.agent.llm_client import Completion
        completion = Completion("answer", [], "scripted", "m", 100)
        assert completion.cached_tokens == 0
