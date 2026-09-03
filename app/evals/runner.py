"""
WHAT:
    Runs the golden set against the real model and scores each answer.

WHY A SCORE AT ALL:
    Every rule in build_system_prompt is justified by a one-off
    observation - someone asked something, watched it go wrong, and
    added a sentence. That is how the prompt got good, and it is also
    why nobody can currently change it with confidence: there is no way
    to tell whether a wording that fixes one question broke three
    others. A number makes prompt edits reversible decisions instead of
    superstition.

WHAT A CASE HAS TO SATISFY:
    answered   it produced a real reply, not one of the fallbacks
    tools      it looked in a place that could hold the answer
    forbidden  it avoided the routes a case rules out
    grounded   nothing in the answer traces back to nowhere

    Grounding HARD failures fail the case. Soft findings - money amounts
    that might be legitimately derived - are printed for a human and do
    not move the score. See grounding.py for why that line is drawn
    there.

ON QUOTA:
    Every case is a full conversation: system prompt plus 35 tool
    schemas, times however many tool rounds it takes. The live tests in
    this repo already recorded what happens when several of those land
    inside one minute, so the runner paces itself between cases and can
    be pointed at a subset.
"""

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from app.agent import grounding
from app.agent.orchestrator import (
    FALLBACK_BUSY,
    FALLBACK_UNAVAILABLE,
    FALLBACKS,
    build_system_prompt,
    run_conversation,
)
from app.evals.cases import CASES, Case, resolve
from app.evals.judge import judge_products
from app.repos import products_repo


@dataclass
class Result:
    case_id: str
    question: str
    reply: str = ""
    tools_called: list = field(default_factory=list)
    latency_s: float = 0.0
    error: str = ""
    answered: bool = False
    # The provider was unreachable, not the prompt at fault. Kept OUT of
    # the score entirely - see NOT ASSESSED below.
    not_assessed: bool = False
    tools_ok: bool = True
    forbidden_used: list = field(default_factory=list)
    hard_failures: list = field(default_factory=list)
    soft_findings: list = field(default_factory=list)
    # Only populated when --judge ran. A judged run therefore applies a
    # check an unjudged one does not, so the two are NOT comparable -
    # compare judged against judged.
    invented_products: list = field(default_factory=list)
    judge_error: str = ""

    @property
    def passed(self) -> bool:
        return (
            not self.not_assessed
            and not self.error
            and self.answered
            and self.tools_ok
            and not self.forbidden_used
            and not self.hard_failures
            and not self.invented_products
        )

    @property
    def reasons(self) -> list:
        out = []
        if self.not_assessed:
            return ["provider unreachable - not assessed"]
        if self.error:
            out.append(f"crashed: {self.error}")
        if not self.answered:
            out.append("fell back instead of answering")
        if not self.tools_ok:
            out.append("looked nowhere useful")
        out += [f"used forbidden {t}" for t in self.forbidden_used]
        out += self.hard_failures
        out += [f"named a product no tool returned: {p!r}"
                for p in self.invented_products]
        return out


async def pick_fixtures() -> str:
    """A real product name for the {product} cases.

    Chosen the way the demo chips are - long enough to be a thing rather
    than a colour, short enough to be one product - so a case never
    depends on a name that has since been delisted.
    """
    docs = await products_repo.get_trending_products(limit=8)
    for doc in docs:
        name = (doc.get("name") or "").strip()
        if 8 <= len(name) <= 38:
            return name
    return (docs[0].get("name") or "").strip() if docs else "kurta"


async def run_case(
    case: Case, user_id: str, system_prompt: str, use_judge: bool = False
) -> Result:
    result = Result(case_id=case.id, question=case.question)
    tools: list = []

    def on_event(event: dict) -> None:
        if event.get("type") == "status":
            tools.append(event.get("tool"))

    started = time.perf_counter()
    try:
        reply, messages = await run_conversation(
            user_id,
            case.question,
            # A FRESH history per case. Cases must not be able to pass
            # because an earlier one already fetched the data.
            history=[{"role": "system", "content": system_prompt}],
            on_event=on_event,
        )
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        result.latency_s = time.perf_counter() - started
        return result

    result.latency_s = time.perf_counter() - started
    result.reply = reply
    result.tools_called = tools
    result.answered = bool(reply) and reply not in FALLBACKS

    # THE PROVIDER BEING DOWN IS NOT A VERDICT ON THE PROMPT. Measured:
    # a baseline run died partway through on 'azure: ReadError', and
    # because the training-data filter leaves exactly one provider
    # standing against the real database there was no failover - so
    # every remaining case scored as "fell back instead of answering"
    # and the run reported a prompt regression that had not happened.
    if reply in (FALLBACK_UNAVAILABLE, FALLBACK_BUSY):
        result.not_assessed = True

    called = set(tools)
    result.tools_ok = not case.expect_any or bool(called & case.expect_any)
    result.forbidden_used = sorted(called & case.forbid)

    report = grounding.check(reply, messages)
    result.hard_failures = report.hard_failures
    result.soft_findings = [
        f"amount {a} not found in any tool result" for a in report.unmatched_amounts
    ]
    result.soft_findings += [
        f"order ending {s} matches no order returned" for s in report.unmatched_order_suffixes
    ]

    # THE PART A REGEX CANNOT DO. Costs a second LLM call, so it is
    # opt-in - but it is the only check covering the failure this
    # project was built around, which was a product NAME.
    if use_judge:
        verdict = await judge_products(reply, messages)
        result.invented_products = verdict.invented
        result.judge_error = verdict.error

    return result


async def run(
    user_id: str,
    only: set | None = None,
    limit: int | None = None,
    delay: float = 2.0,
    retry_delay: float = 20.0,
    checkpoint: str | None = None,
    judge: bool = False,
) -> list[Result]:
    product = await pick_fixtures()
    cases = resolve(CASES, product)
    if only:
        cases = [c for c in cases if c.id in only]
    if limit:
        cases = cases[:limit]

    # Built ONCE. Every case gets a copy, so they stay independent
    # without paying for the category lookup thirty times.
    system_prompt = await build_system_prompt()

    results = []
    for index, case in enumerate(cases, 1):
        print(f"  [{index}/{len(cases)}] {case.id} ... ", end="", flush=True)
        result = await run_case(case, user_id, system_prompt, use_judge=judge)

        if result.not_assessed:
            # One retry, after a pause long enough for a rate-limit
            # window to roll over or a dropped connection to re-open.
            print("provider down, retrying ... ", end="", flush=True)
            await asyncio.sleep(retry_delay)
            result = await run_case(case, user_id, system_prompt, use_judge=judge)

        results.append(result)
        verdict = "SKIP" if result.not_assessed else ("PASS" if result.passed else "FAIL")
        print(
            f"{verdict}  ({result.latency_s:.1f}s)"
            + ("" if result.passed else "  " + "; ".join(result.reasons))
        )

        # Written after EVERY case. The first baseline run lost eight
        # minutes of completed work to a provider drop at case four,
        # because nothing reached disk until the very end.
        if checkpoint:
            save(report_payload(results, "partial"), checkpoint, quiet=True)

        if index < len(cases):
            await asyncio.sleep(delay)
    return results


def report_payload(results: list[Result], label: str) -> dict:
    """The scored summary. NOT-ASSESSED cases are excluded from both
    halves of the fraction - a run where the provider dropped twice is
    28/33, not 28/35, because nobody learned anything about those two."""
    assessed = [r for r in results if not r.not_assessed]
    return {
        "label": label,
        "run_at": datetime.now(timezone.utc).isoformat(),
        "passed": sum(1 for r in assessed if r.passed),
        "total": len(assessed),
        "not_assessed": sum(1 for r in results if r.not_assessed),
        "results": [asdict(r) for r in results],
    }


def report(results: list[Result], label: str) -> dict:
    assessed = [r for r in results if not r.not_assessed]
    skipped = [r for r in results if r.not_assessed]
    passed = [r for r in assessed if r.passed]
    failed = [r for r in assessed if not r.passed]
    latencies = sorted(r.latency_s for r in assessed if not r.error)

    print("\n" + "=" * 72)
    print(f"  {label}: {len(passed)}/{len(assessed)} passed "
          f"({100 * len(passed) / max(len(assessed), 1):.0f}%)")
    if skipped:
        print(f"  NOT ASSESSED: {len(skipped)} "
              f"({', '.join(r.case_id for r in skipped)}) - provider unreachable")
    if latencies:
        print(f"  latency  median {latencies[len(latencies) // 2]:.1f}s   "
              f"slowest {latencies[-1]:.1f}s")
    print("=" * 72)

    if failed:
        print("\nFAILED")
        for r in failed:
            print(f"  {r.case_id:20} {'; '.join(r.reasons)}")
            print(f"  {'':20} tools: {', '.join(r.tools_called) or '(none)'}")

    stuck = [r for r in results if r.judge_error]
    if stuck:
        print(f"\n  JUDGE UNAVAILABLE on {len(stuck)} case(s): "
              f"{stuck[0].judge_error}")

    soft = [(r.case_id, f) for r in results for f in r.soft_findings]
    if soft:
        print("\nTO REVIEW (amounts that may be derived rather than quoted)")
        for case_id, finding in soft:
            print(f"  {case_id:20} {finding}")

    return report_payload(results, label)


def save(payload: dict, path: str, quiet: bool = False) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    if not quiet:
        print(f"\nwritten to {path}")


def compare(current: dict, previous: dict) -> None:
    """The point of scoring: what a prompt edit changed."""
    was = {r["case_id"]: r for r in previous.get("results", [])}
    now = {r["case_id"]: r for r in current.get("results", [])}

    def assessed(row):
        return not row.get("not_assessed")

    def ok(row):
        return (
            not row["error"] and row["answered"] and row["tools_ok"]
            and not row["forbidden_used"] and not row["hard_failures"]
            and not row.get("invented_products")
        )

    # Only cases assessed in BOTH runs can have moved. A provider drop is
    # not a prompt regression and must never be printed as one.
    both = [c for c in now if c in was and assessed(now[c]) and assessed(was[c])]
    fixed = [c for c in both if ok(now[c]) and not ok(was[c])]
    broke = [c for c in both if not ok(now[c]) and ok(was[c])]

    print(f"\nvs {previous.get('label', 'previous')} "
          f"({previous.get('passed')}/{previous.get('total')})")
    for case_id in broke:
        print(f"  BROKE  {case_id}")
    for case_id in fixed:
        print(f"  FIXED  {case_id}")
    if not fixed and not broke:
        print("  no change in which cases pass")
