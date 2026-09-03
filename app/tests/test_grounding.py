"""
WHAT:
    Tests the mechanical grounding checks - the ones that decide whether
    an answer's claims trace back to a tool result.

WHY THESE MATTER MORE THAN THE EVAL SUITE AROUND THEM:
    The eval runner is a script; if it miscounts, someone notices a
    strange number. These functions are the thing a live grounding
    verifier would call before an answer reaches a customer, so a false
    NEGATIVE here is a hallucination shipped, and a false POSITIVE is a
    correct answer blocked.

    The second one is why the money check is deliberately soft: a cart
    subtotal is arithmetic over tool results rather than a quote from
    one, and a checker that cannot tell those apart would flag honest
    answers forever until someone turned it off.

FLOW:
    Pure unit tests. No database, no LLM.
"""

import json

from app.agent import grounding


def tool(payload) -> dict:
    return {"role": "tool", "tool_call_id": "t1", "content": json.dumps(payload)}


ORDERS = tool({
    "orders": [
        {"orderId": "ORD0000000000000001", "status": "shipped",
         "items": [{"name": "Cotton Shirt", "qty": 1}], "total": 1499},
    ],
    "showing": 1, "totalOrders": 9,
})


class TestOrderIdsAreHardEvidence:
    """An order id is never arrived at by arithmetic - it was returned
    or it was invented."""

    def test_a_quoted_order_id_is_grounded(self):
        report = grounding.check(
            "Your order ORD0000000000000001 has shipped.", [ORDERS]
        )
        assert report.checked_order_ids == 1
        assert report.ungrounded_order_ids == []
        assert report.ok

    def test_an_invented_order_id_is_a_hard_failure(self):
        report = grounding.check("Your order ORD0000000000000003 shipped.", [ORDERS])
        assert report.ungrounded_order_ids == ["ORD0000000000000003"]
        assert not report.ok
        assert "invented order id" in report.hard_failures[0]

    def test_an_id_inside_a_longer_string_still_counts_as_returned(self):
        """Tool results carry ids inside sentences too - a note, a
        tracking blurb. Returned is returned."""
        messages = [tool({"note": "Refund raised for ORD0000000000000001 today"})]
        report = grounding.check("That's ORD0000000000000001.", messages)
        assert report.ok


class TestMoneyIsSoftEvidence:
    def test_a_quoted_price_is_grounded(self):
        report = grounding.check("The total was ₹1,499.", [ORDERS])
        assert report.checked_amounts == 1
        assert report.unmatched_amounts == []

    def test_indian_digit_grouping_is_normalised(self):
        messages = [tool({"price": 125500})]
        report = grounding.check("It is ₹1,25,500.", messages)
        assert report.unmatched_amounts == []

    def test_rupees_written_as_rs_is_read_too(self):
        messages = [tool({"price": 800})]
        assert grounding.check("Rs. 800 for that one.", messages).unmatched_amounts == []

    def test_an_unmatched_amount_is_reported_but_does_not_fail(self):
        """It could be a hallucinated price, or an honest subtotal. The
        checker says what it saw and lets a human decide."""
        report = grounding.check("That comes to ₹4,999.", [ORDERS])
        assert report.unmatched_amounts == ["4,999"]
        assert report.ok, "a derived amount must not fail an answer"

    def test_a_boolean_does_not_ground_a_claim_of_one(self):
        """bool is an int subclass in Python, so isTopPick=True would
        otherwise silently vouch for '₹1'."""
        messages = [tool({"isTopPick": True, "isSold": False})]
        report = grounding.check("It costs ₹1.", messages)
        assert report.unmatched_amounts == ["1"]


class TestPromisingAnActionIsAlwaysWrong:
    """The assistant is read-only at the database role. A sentence
    claiming otherwise is wrong no matter what the tools returned."""

    def test_claiming_a_cancellation_fails(self):
        report = grounding.check("I've cancelled that order for you.", [ORDERS])
        assert not report.ok
        assert "promised an action" in report.hard_failures[0]

    def test_offering_to_notify_fails(self):
        report = grounding.check("I'll let you know when it ships.", [ORDERS])
        assert not report.ok

    def test_reporting_what_the_user_can_do_is_fine(self):
        report = grounding.check(
            "It can still be cancelled - you can do that from the order "
            "screen in the app.",
            [ORDERS],
        )
        assert report.ok


class TestItSurvivesMessyInput:
    def test_an_empty_reply_reports_nothing(self):
        assert grounding.check("", [ORDERS]).ok

    def test_a_malformed_tool_payload_is_skipped_not_raised(self):
        messages = [{"role": "tool", "tool_call_id": "x", "content": "not json"}]
        report = grounding.check("Order ORD0000000000000001.", messages)
        assert report.tool_results_seen == 0
        assert report.ungrounded_order_ids == ["ORD0000000000000001"]

    def test_non_tool_messages_are_not_treated_as_evidence(self):
        """The USER can name an order id. That is not the assistant
        having looked it up."""
        messages = [{"role": "user", "content": "where is ORD0000000000000001"}]
        report = grounding.check("ORD0000000000000001 has shipped.", messages)
        assert report.ungrounded_order_ids == ["ORD0000000000000001"]

    def test_no_messages_at_all(self):
        assert grounding.check("Hello.", []).ok


class TestTheMoneyPatternDoesNotSeeThingsThatAreNotMoney:
    """Every false positive here costs the soft findings a little
    credibility, and a report nobody trusts is a report nobody reads."""

    def test_rs_inside_an_ordinary_word_is_not_a_price(self):
        """Measured: "cancel within 24 hours 2 days after delivery" was
        reported as an ungrounded claim of Rs 2, because "rs" sits
        inside "hours" and the pattern was case-insensitive."""
        report = grounding.check(
            "You can cancel within 24 hours 2 days after it ships.", [ORDERS]
        )
        assert report.unmatched_amounts == []
        assert report.checked_amounts == 0

    def test_the_same_trap_in_orders_and_sellers(self):
        for word in ("orders 5", "sellers 9", "yours 3"):
            report = grounding.check(f"About your {word} of them.", [ORDERS])
            assert report.checked_amounts == 0, f"{word!r} read as money"

    def test_a_real_rs_price_is_still_read(self):
        messages = [tool({"price": 800})]
        assert grounding.check("Rs. 800 for that.", messages).checked_amounts == 1
        assert grounding.check("Rs 800 for that.", messages).unmatched_amounts == []


def exchange(tool_name, payload, call_id="t1"):
    """One assistant tool_call plus its result - the shape the
    orchestrator actually appends, since attribute() has to join the two
    to learn a tool's NAME."""
    return [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": call_id, "type": "function",
             "function": {"name": tool_name, "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": call_id, "content": json.dumps(payload)},
    ]


class TestAttributionNamesTheSource:
    def test_a_price_is_traced_to_the_tool_that_returned_it(self):
        messages = exchange("get_trending_products", [{"name": "kurta", "price": 599}])
        claims = grounding.attribute("The kurta is ₹599.", messages)
        assert len(claims) == 1
        assert claims[0]["tool"] == "get_trending_products"
        assert claims[0]["kind"] == "price"

    def test_an_order_id_is_traced(self):
        messages = exchange("get_order_history",
                            {"orders": [{"orderId": "ORD0000000000000001"}]})
        claims = grounding.attribute("Order ORD0000000000000001 shipped.", messages)
        assert [c["kind"] for c in claims] == ["order_id"]
        assert claims[0]["tool"] == "get_order_history"

    def test_the_right_tool_is_named_when_several_ran(self):
        messages = (
            exchange("get_cart", {"total": 111}, "a")
            + exchange("get_order_history", {"orders": [{"total": 222}]}, "b")
        )
        claims = grounding.attribute("Cart ₹111, order ₹222.", messages)
        assert [(c["text"], c["tool"]) for c in claims] == [
            ("₹111", "get_cart"), ("₹222", "get_order_history"),
        ]


class TestTheSpansAreUsableByAClient:
    """The UI slices the reply by these offsets. If they are wrong it
    marks the wrong words, which is worse than marking nothing."""

    def test_slicing_the_reply_returns_exactly_the_claim(self):
        messages = exchange("get_cart", {"total": 1199})
        reply = "Your cart comes to ₹1,199 including shipping."
        for claim in grounding.attribute(reply, messages):
            assert reply[claim["start"]:claim["end"]] == claim["text"]

    def test_the_span_covers_the_rupee_sign_not_just_the_digits(self):
        messages = exchange("get_cart", {"total": 599})
        claims = grounding.attribute("It is ₹599.", messages)
        assert claims[0]["text"] == "₹599"

    def test_the_same_price_twice_gets_two_separate_spans(self):
        """Marking by searching for the text would highlight both from
        one match, or the first one twice."""
        messages = exchange("get_cart", {"total": 599})
        reply = "₹599 now, ₹599 later."
        claims = grounding.attribute(reply, messages)
        assert len(claims) == 2
        assert claims[0]["start"] != claims[1]["start"]
        assert [reply[c["start"]:c["end"]] for c in claims] == ["₹599", "₹599"]

    def test_claims_come_back_in_reading_order(self):
        messages = exchange("get_order_history",
                            {"orders": [{"orderId": "ORD0000000000000001",
                                         "total": 1499}]})
        reply = "Order ORD0000000000000001 came to ₹1,499."
        claims = grounding.attribute(reply, messages)
        assert [c["start"] for c in claims] == sorted(c["start"] for c in claims)


class TestAttributionClaimsNothingItCannotProve:
    def test_an_amount_no_tool_returned_is_not_attributed(self):
        """The derived subtotal that check() reports as soft must not be
        underlined as if a tool vouched for it."""
        messages = exchange("get_cart", {"total": 1199})
        claims = grounding.attribute("That works out around ₹4,999.", messages)
        assert claims == []

    def test_a_reply_with_no_messages_attributes_nothing(self):
        assert grounding.attribute("It costs ₹599.", []) == []

    def test_an_empty_reply_attributes_nothing(self):
        assert grounding.attribute("", exchange("get_cart", {"total": 1})) == []

    def test_a_result_whose_call_was_never_named_is_survived(self):
        """A tool message with no matching assistant turn - defensive,
        since a truncated history is a real shape."""
        messages = [{"role": "tool", "tool_call_id": "orphan",
                     "content": json.dumps({"total": 599})}]
        claims = grounding.attribute("It is ₹599.", messages)
        assert claims == [] or claims[0]["tool"] is None


class TestDigitGroupsEndInADigit:
    """Measured while building attribution: "₹111, order ₹222" produced
    the span "₹111," because the digit class allowed a trailing comma.
    It underlined the punctuation and reported the amount as "111,"."""

    def test_a_trailing_comma_is_not_part_of_the_price(self):
        messages = exchange("get_cart", {"total": 111})
        claims = grounding.attribute("Cart ₹111, and more.", messages)
        assert claims[0]["text"] == "₹111"

    def test_indian_grouping_still_matches_whole(self):
        messages = exchange("get_cart", {"total": 125500})
        claims = grounding.attribute("It is ₹1,25,500 total.", messages)
        assert claims[0]["text"] == "₹1,25,500"

    def test_the_soft_finding_no_longer_carries_a_comma(self):
        report = grounding.check("Around ₹4,999, roughly.", [ORDERS])
        assert report.unmatched_amounts == ["4,999"]


class TestTheLastFourDigitsForm:
    """What the product actually writes. The prompt says to list orders
    by "last 4 of the ID" and never to show internal ids, so a real reply
    says "order ending 4417" - measured, a five-order answer produced
    ZERO attributable ids before this pattern existed."""

    ORDERS_TWO = exchange("get_order_history", {"orders": [
        {"orderId": "ORD7420000000004417", "status": "confirmed"},
        {"orderId": "ORD7420000000008230", "status": "cancelled"},
    ]})

    def test_a_suffix_is_traced_to_the_order_tool(self):
        claims = grounding.attribute(
            "Your order ending 4417 is confirmed.", self.ORDERS_TWO
        )
        assert [(c["kind"], c["text"]) for c in claims] == [
            ("order_suffix", "4417")
        ]
        assert claims[0]["tool"] == "get_order_history"

    def test_each_listed_order_is_attributed_separately(self):
        reply = "- ending 4417 — confirmed\n- ending 8230 — cancelled"
        claims = grounding.attribute(reply, self.ORDERS_TWO)
        assert len(claims) == 2
        assert [reply[c["start"]:c["end"]] for c in claims] == ["4417", "8230"]

    def test_a_suffix_matching_no_order_is_not_attributed(self):
        assert grounding.attribute("Order ending 9999.", self.ORDERS_TWO) == []

    def test_only_order_shaped_strings_can_vouch_for_a_suffix(self):
        """A product description ending in those digits must not stand in
        for an order the user never placed."""
        messages = exchange("search_products",
                            [{"name": "Frame model 4417", "price": 500}])
        assert grounding.attribute("Your order ending 4417.", messages) == []

    def test_check_reports_an_invented_suffix_without_failing_the_answer(self):
        report = grounding.check("Order ending 9999 shipped.", self.ORDERS_TWO)
        assert report.unmatched_order_suffixes == ["9999"]
        assert report.ok, (
            "promoting this to a hard failure would re-score the whole eval "
            "baseline - that must be its own measured change"
        )


class TestTheSuffixIsFoundByValueNotByWording:
    """Two consecutive live runs wrote "order ending 4417" and
    "order …4417" for the same data. Any pattern over the prose is a
    guess about wording that changes between calls, so the real suffixes
    are searched for directly."""

    ORDERS = exchange("get_order_history", {"orders": [
        {"orderId": "ORD7420000000004417", "status": "confirmed"},
    ]})

    def test_the_ellipsis_form_is_found(self):
        claims = grounding.attribute("Your order …4417 is confirmed.", self.ORDERS)
        assert [c["text"] for c in claims] == ["4417"]

    def test_the_word_form_is_found_by_the_same_code(self):
        claims = grounding.attribute("Your order ending 4417.", self.ORDERS)
        assert [c["text"] for c in claims] == ["4417"]

    def test_a_hash_form_would_also_be_found(self):
        assert grounding.attribute("Order #4417.", self.ORDERS)[0]["text"] == "4417"

    def test_digits_inside_a_longer_number_are_not_matched(self):
        """A year or a bigger figure that merely contains the suffix must
        not be underlined as an order reference."""
        assert grounding.attribute("Expected 24417 units.", self.ORDERS) == []
        assert grounding.attribute("Back in 44170 days.", self.ORDERS) == []

    def test_a_full_id_and_its_own_suffix_do_not_double_mark(self):
        reply = "Order ORD7420000000004417 is confirmed."
        claims = grounding.attribute(reply, self.ORDERS)
        assert len(claims) == 1, "the id and its last four overlap"
        assert claims[0]["kind"] == "order_id"


class TestTheNamesATooolActuallyReturned:
    """The allowed set an LLM judge is constrained to. Everything that
    gets in here is a name the assistant may legitimately say; anything
    that wrongly gets in licenses a hallucination."""

    def test_product_lists_contribute_their_names(self):
        messages = exchange("search_products",
                            [{"product_id": "a", "name": "Cotton Kurta"},
                             {"product_id": "b", "name": "Silk Saree"}])
        assert grounding.product_names(messages) == {"Cotton Kurta", "Silk Saree"}

    def test_order_items_contribute_their_names(self):
        messages = exchange("get_order_history", {"orders": [
            {"orderId": "ORD0000000000000001",
             "items": [{"name": "digital clock", "qty": 1}]},
        ]})
        assert grounding.product_names(messages) == {"digital clock"}

    def test_bargains_use_their_own_spelling(self):
        messages = exchange("get_bargain_status", {"productName": "Cotton Shirt"})
        assert grounding.product_names(messages) == {"Cotton Shirt"}

    def test_people_and_sessions_are_not_products(self):
        """A username or a live-session title must not become a name the
        assistant may present as something to buy."""
        messages = (
            exchange("get_followers_or_following",
                     {"sample": [{"username": "rutuja", "businessName": "the marketplace Co"}]}, "a")
            + exchange("get_live_now", [{"session_id": "s", "title": "Friday Haul"}], "b")
            + exchange("get_trending_bits", [{"bit_id": "b", "title": "Unboxing"}], "c")
        )
        assert grounding.product_names(messages) == set()

    def test_blank_and_missing_names_are_skipped(self):
        messages = exchange("search_products",
                            [{"name": "  "}, {"name": None}, {"name": "Real One"}])
        assert grounding.product_names(messages) == {"Real One"}

    def test_nothing_looked_up_means_an_empty_set(self):
        assert grounding.product_names([]) == set()
