"""
WHAT:
    Tests the LLM judge that answers the one grounding question a regex
    cannot: did the reply name a product no tool returned?

WHY THE PARSING TESTS MATTER:
    The judge's verdict FAILS a case, so a parsing slip either lets a
    hallucination through or fails an honest answer. Models wrap JSON in
    backticks, add a sentence before it, or answer in prose - and a judge
    that reports "did not return JSON" whenever the model was merely
    chatty is a judge that gets switched off.

FLOW:
    Pure unit tests over _parse and the no-evidence shortcut. The live
    test - does the real model actually CATCH an invented product - is
    marked `live` and deselected by default.
"""

import pytest

from app.evals.judge import Judgement, _parse, judge_products


class TestReadingTheVerdict:
    def test_plain_json(self):
        assert _parse('{"invented": ["knit sweater"]}') == (["knit sweater"], "")

    def test_an_empty_list_is_a_clean_bill(self):
        assert _parse('{"invented": []}') == ([], "")

    def test_fenced_json_is_unwrapped(self):
        """Common enough that treating it as a failure would make the
        judge unusable."""
        assert _parse('```json\n{"invented": ["ghost"]}\n```') == (["ghost"], "")
        assert _parse('```\n{"invented": []}\n```') == ([], "")

    def test_json_buried_in_a_sentence_is_still_found(self):
        text = 'Here is my finding:\n{"invented": ["wool cardigan"]}\nHope that helps.'
        assert _parse(text) == (["wool cardigan"], "")

    def test_blank_entries_are_dropped(self):
        assert _parse('{"invented": ["ok", "", "  "]}') == (["ok"], "")

    def test_prose_with_no_json_is_an_error_not_a_verdict(self):
        invented, error = _parse("Everything looked fine to me.")
        assert invented == []
        assert error, "silently reading prose as a pass would hide a miss"

    def test_json_of_the_wrong_shape_is_an_error(self):
        assert _parse('["knit sweater"]')[1]
        assert _parse('{"result": "fine"}')[1]
        assert _parse("")[1]


class TestNoEvidenceMeansNoVerdict:
    async def test_a_reply_with_no_lookups_is_not_accused(self):
        """An answer that names no products - "your cart is empty", a
        refusal - has nothing to compare against. Silence, not an
        accusation, and no LLM call spent either."""
        verdict = await judge_products("Your cart is empty.", [])
        assert verdict == Judgement(allowed_count=0)
        assert verdict.ok


def looked_up(*names):
    """A search that returned exactly these products."""
    import json
    return [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "t1", "type": "function",
             "function": {"name": "search_products", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "t1", "content": json.dumps(
            [{"product_id": str(i), "name": n, "price": 599}
             for i, n in enumerate(names)])},
    ]


@pytest.mark.live
class TestTheJudgeCatchesTheIncidentItWasBuiltFor:
    """The question the unit tests cannot answer: does the real model
    actually notice? Deselected by default - run with `pytest -m live`.

    THE FIRST VERSION OF THIS TEST WAS TOO EASY AND PASSED A BROKEN
    JUDGE. It gave the invented products PRICES, which makes them
    obviously offers. The measured incident had none - "a few warm
    options to choose from: knit sweater, fleece hoodie, wool cardigan" -
    and against that the judge returned a clean bill, because the prompt
    told it to ignore "generic words" and unpriced items read as generic.

    So the payload here is the incident verbatim. A judge that only
    catches the easy shape is worse than none: it certifies the exact
    failure it was written to find.
    """

    async def test_the_unpriced_suggestion_form_is_caught(self, db):
        verdict = await judge_products(
            "Here are a few warm options to choose from: a knit sweater, a "
            "fleece hoodie, and a wool cardigan.",
            looked_up("Cotton Kurta", "Handloom Dupatta"),
        )
        assert not verdict.error, verdict.error
        assert len(verdict.invented) >= 2, f"judge let it through: {verdict.raw!r}"

    async def test_an_honest_answer_is_not_accused(self, db):
        verdict = await judge_products(
            "I found a Cotton Kurta at ₹599 and a Handloom Dupatta at ₹1,250.",
            looked_up("Cotton Kurta", "Handloom Dupatta"),
        )
        assert not verdict.error, verdict.error
        assert verdict.invented == [], f"false positive: {verdict.raw!r}"

    async def test_saying_it_found_nothing_is_not_an_invention(self, db):
        """The prompt's whole point is that a refusal is the CORRECT
        behaviour. Punishing it would push the next prompt version back
        toward guessing."""
        verdict = await judge_products(
            "I couldn't find anything warm in the catalogue right now.",
            looked_up("Cotton Kurta"),
        )
        assert verdict.invented == [], f"false positive: {verdict.raw!r}"

    async def test_asking_a_clarifying_question_is_not_an_invention(self, db):
        verdict = await judge_products(
            "Which category are you after - clothing, home, or beauty?",
            looked_up("Cotton Kurta"),
        )
        assert verdict.invented == [], f"false positive: {verdict.raw!r}"
