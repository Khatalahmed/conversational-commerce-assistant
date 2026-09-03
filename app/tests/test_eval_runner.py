"""
WHAT:
    Tests the scoring and the run-to-run comparison - the parts of the
    eval suite that decide what a number MEANS.

WHY:
    The suite exists so a prompt edit becomes a reversible decision
    rather than a hunch, and that rests entirely on two pieces of logic:
    whether a case passed, and which cases moved since last time. Both
    run without an LLM, so both can be tested properly - and if either
    is wrong, every score the suite has ever printed was wrong with it.

FLOW:
    Pure unit tests over Result and compare(). No database, no LLM.
"""

from dataclasses import asdict

from app.evals.runner import Result, compare


def result(case_id="c1", **overrides) -> Result:
    """A passing result, with failures opted into one at a time."""
    base = dict(
        case_id=case_id, question="q", reply="a real answer",
        tools_called=["get_order_history"], latency_s=1.0,
        answered=True, tools_ok=True,
    )
    base.update(overrides)
    return Result(**base)


class TestWhatCountsAsPassing:
    def test_a_clean_run_passes(self):
        assert result().passed
        assert result().reasons == []

    def test_a_fallback_reply_fails(self):
        r = result(answered=False)
        assert not r.passed
        assert "fell back" in r.reasons[0]

    def test_looking_nowhere_useful_fails(self):
        r = result(tools_ok=False)
        assert not r.passed
        assert "looked nowhere useful" in r.reasons

    def test_a_forbidden_tool_fails(self):
        r = result(forbidden_used=["search_products"])
        assert not r.passed
        assert "used forbidden search_products" in r.reasons

    def test_a_hard_grounding_failure_fails(self):
        r = result(hard_failures=["invented order id ORD999"])
        assert not r.passed
        assert "invented order id ORD999" in r.reasons

    def test_a_crash_fails_and_says_so(self):
        r = result(error="TimeoutError: took too long")
        assert not r.passed
        assert r.reasons[0].startswith("crashed:")

    def test_soft_findings_do_not_fail_a_case(self):
        """A derived subtotal is reported for review, not counted
        against the score. If that ever changes, honest answers start
        failing and the suite gets switched off."""
        r = result(soft_findings=["amount 4,999 not found in any tool result"])
        assert r.passed
        assert r.reasons == []


def payload(label, *results) -> dict:
    return {
        "label": label, "passed": sum(1 for r in results if r.passed),
        "total": len(results), "results": [asdict(r) for r in results],
    }


class TestComparingTwoRuns:
    def test_a_newly_failing_case_is_reported_as_broken(self, capsys):
        was = payload("v1", result("a"), result("b"))
        now = payload("v2", result("a"), result("b", answered=False))

        compare(now, was)
        out = capsys.readouterr().out
        assert "BROKE  b" in out
        assert "FIXED" not in out

    def test_a_newly_passing_case_is_reported_as_fixed(self, capsys):
        was = payload("v1", result("a", tools_ok=False))
        now = payload("v2", result("a"))

        compare(now, was)
        assert "FIXED  a" in capsys.readouterr().out

    def test_an_unchanged_run_says_so_rather_than_printing_nothing(self, capsys):
        same = payload("v1", result("a"), result("b"))
        compare(same, same)
        assert "no change in which cases pass" in capsys.readouterr().out

    def test_a_case_absent_from_the_previous_run_is_not_a_regression(self):
        """Adding a case to the golden set must not read as the prompt
        having broken something."""
        was = payload("v1", result("a"))
        now = payload("v2", result("a"), result("brand-new", answered=False))

        import io
        import contextlib

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            compare(now, was)
        assert "BROKE" not in buffer.getvalue()


class TestAProviderOutageIsNotAPromptRegression:
    """A baseline run really did die on 'azure: ReadError' partway
    through, and because the training-data filter leaves one provider
    standing against the real database there is no failover. Everything
    after the drop scored as "fell back instead of answering" - a prompt
    regression the suite invented. These pin the fix."""

    def test_an_unreachable_provider_does_not_count_as_failing(self):
        r = result(not_assessed=True, answered=False)
        assert not r.passed
        assert r.reasons == ["provider unreachable - not assessed"]

    def test_the_score_excludes_both_halves_of_the_fraction(self):
        from app.evals.runner import report_payload

        payload = report_payload(
            [result("a"), result("b"), result("c", not_assessed=True, answered=False)],
            "x",
        )
        assert (payload["passed"], payload["total"]) == (2, 2), (
            "an unassessed case must leave the denominator, not just the "
            "numerator - 2/3 would read as a 67% score that nobody earned"
        )
        assert payload["not_assessed"] == 1

    def test_a_case_skipped_in_either_run_is_never_reported_as_broken(self, capsys):
        was = payload("v1", result("a"))
        now = payload("v2", result("a", not_assessed=True, answered=False))

        compare(now, was)
        out = capsys.readouterr().out
        assert "BROKE" not in out
        assert "no change in which cases pass" in out
