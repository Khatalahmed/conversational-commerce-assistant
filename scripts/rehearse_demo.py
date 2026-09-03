"""
Runs the client demo end to end, records what the assistant ACTUALLY
says, and writes it to docs/demo-script.md as a run sheet.

    uv run python scripts/rehearse_demo.py

WHY:
    "Hope it works live" is not a plan. This runs every question in
    order against the demo database, captures the real reply and the
    real token cost, and fails loudly if any answer comes back as a
    fallback - so the script you take into the meeting is one you have
    already seen work.

    Token cost is measured because it is the binding constraint: Groq's
    free tier allows 8,000 prompt tokens per MINUTE, shared across
    everyone. The summary at the end says whether the demo fits.

PACING:
    A real client fires questions quickly; the free tier cannot. Each
    question waits PAUSE_SECONDS so the rehearsal reflects a paced demo
    rather than measuring the rate limiter. On a paid key, set
    PAUSE_SECONDS=0.
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent import orchestrator  # noqa: E402
from app.agent.orchestrator import (  # noqa: E402
    FALLBACK_BUSY,
    FALLBACK_GENERIC,
    FALLBACK_UNAVAILABLE,
    run_conversation,
)
from app.agent.llm_client import get_providers  # noqa: E402
from app.config.settings import get_settings  # noqa: E402
from demo_script_builder import build, gather  # noqa: E402
from app.db.connection import (  # noqa: E402
    close_mongo_connection,
    connect_to_mongo,
    get_database,
)
from app.repos.base import to_object_id  # noqa: E402

FALLBACKS = {FALLBACK_GENERIC, FALLBACK_BUSY, FALLBACK_UNAVAILABLE}

# WHICH fallback it was, because they mean opposite things to whoever is
# reading this output at 9am.
#
# "returned a fallback" was all this used to say, and it cost a
# confusing half-hour: two rehearsals run back to back both failed on
# the last questions, which reads exactly like a broken assistant. It
# was the token-per-minute limit. One rehearsal spends ~61,000 prompt
# tokens against Azure's 100,000/minute, so a second one inside the same
# minute cannot finish - and the code was never at fault.
FALLBACK_DIAGNOSIS = {
    FALLBACK_BUSY: (
        "rate limited - NOT a code failure. One rehearsal spends ~61k prompt "
        "tokens and the limit is 100k/minute, so a second run inside a minute "
        "will do this. Wait a minute and re-run."
    ),
    FALLBACK_UNAVAILABLE: (
        "the provider could not be reached at all. Run scripts/check_azure.py - "
        "an expired credit looks exactly like this."
    ),
    FALLBACK_GENERIC: (
        "the model returned nothing usable. Re-run once; if it repeats, it is "
        "worth investigating before demoing."
    ),
}
PAUSE_SECONDS = int(os.environ.get("PAUSE_SECONDS", "25"))
OUTPUT = Path("docs/demo-script.md")

def safe(text: str) -> str:
    """Windows consoles are cp1252 and raise on the rupee sign."""
    return text.encode("ascii", "replace").decode()


async def main() -> None:
    settings = get_settings()
    if settings.mongodb_database == "marketplace":
        # OPT-IN, GATED ON THE ACTUAL GUARANTEE - not on being asked nicely.
        #
        # The refusal exists because a rehearsal sends real customer
        # orders, delivery cities and tracking numbers through an LLM, and
        # the free tiers reserve the right to train on that. Azure OpenAI
        # does not: prompts and completions are contractually excluded
        # from model training.
        #
        # So the flag alone is not enough. It also CHECKS that a
        # no-training provider is actually primary, because a flag that
        # trusted the operator's word would still ship real data to Groq
        # the day someone removed the Azure key from .env.
        if "--real-data" not in sys.argv:
            raise SystemExit(
                "Refusing to rehearse against the REAL database.\n\n"
                "A rehearsal sends every reply through an LLM, and staging\n"
                "holds real customer data.\n\n"
                "  Safe:  set MONGODB_DATABASE=marketplace_demo\n"
                "  Or:    pass --real-data, allowed ONLY while a no-training\n"
                "         provider (Azure OpenAI) is primary."
            )

        primary = get_providers()[0]
        if primary.kind != "azure":
            raise SystemExit(
                f"--real-data refused: the primary provider is "
                f"'{primary.name}', not Azure.\n\n"
                "Real customer data may only go to a provider contractually\n"
                "excluded from training on it. Configure AZURE_OPENAI_* in\n"
                ".env, or rehearse against marketplace_demo."
            )

        print("  !! REAL customer data, going to Azure OpenAI (no training).")
        print("     Azure still retains content for 30 days for abuse")
        print("     monitoring, and flagged content may be reviewed.\n")

    await connect_to_mongo()
    db = get_database()
    # EVERY question is built from data looked up right now, in this
    # database. Hardcoded ids belonged to the demo dataset and became ten
    # polite "not found" answers the moment this ran against real data -
    # which the old pass condition happily reported as success.
    anchors = await gather(db)
    script = build(anchors)
    user_id = anchors.user_id

    # NAMED, NOT NUMBERED. The --real-data gate above already warns that
    # this is real customer data; it does not say WHOSE afternoon of
    # orders is about to be read out loud, and "buyer: 698e25..." never
    # will.
    _buyer = await db.users.find_one({"_id": to_object_id(user_id)}, {"username": 1})
    print(f"  buyer:   {(_buyer or {}).get('username') or '(no username)'}  [{user_id}]")
    print(f"  anchors: order={anchors.order_id or '-'}  "
          f"product={anchors.product_name or '-'!r}  "
          f"coupon={anchors.coupon_code or '-'}  "
          f"other-buyer-order={anchors.other_order_id or '-'}")
    print(f"  {len(script)} questions built from this database\n")

    # WRAPS stream(), NOT complete(), BECAUSE THAT IS WHAT THE DEMO USES.
    #
    # The rehearsal used to call run_conversation with no callback, which
    # takes the buffered path - the one /chat uses. The demo page moved
    # to /chat/stream, so the pre-flight check was verifying a code path
    # nobody would watch and leaving the one the client sees untested.
    # Passing on_event below puts the rehearsal on the same path as the
    # browser, and this wrapper follows it there to keep counting tokens.
    spent = {"tokens": 0}
    original = orchestrator.stream

    async def counting(provider, messages, tools, tool_choice="auto", on_text=None):
        completion = await original(
            provider, messages, tools, tool_choice, on_text=on_text
        )
        spent["tokens"] += completion.prompt_tokens or 0
        return completion

    orchestrator.stream = counting

    history = None
    rows, failures, total = [], [], 0

    for index, step in enumerate(script, start=1):
        if index > 1 and PAUSE_SECONDS:
            await asyncio.sleep(PAUSE_SECONDS)

        spent["tokens"] = 0
        # Collected exactly as the demo page collects them, so the two
        # assertions below are about what a viewer actually sees.
        streamed, statuses = [], []

        def on_event(event):
            if event["type"] == "token":
                streamed.append(event["text"])
            elif event["type"] == "status":
                statuses.append(event["label"])

        reply, history = await run_conversation(
            user_id, step.question, history=history, on_event=on_event
        )
        cost = spent["tokens"]
        total += cost

        # "No fallback" was the entire test before, which happily passed a
        # reply of "I couldn't find that order" - a well-formed answer
        # about data that was not there. Each step now states what its
        # answer must actually contain.
        problems = (
            [f"returned a fallback - {FALLBACK_DIAGNOSIS[reply]}"]
            if reply in FALLBACKS
            else step.check(reply)
        )

        # TWO THINGS THE DEMO PAGE DEPENDS ON, checked here rather than
        # discovered in front of a client.
        #
        # An answer that arrives in one lump still renders - the page
        # falls back to the reply carried by `done` - but it renders as
        # the dead pause streaming was added to remove, and nothing else
        # would report that.
        #
        # The page draws the tokens and THEN replaces them with `reply`.
        # If those two disagree the text visibly rewrites itself
        # mid-sentence, which reads as a glitch.
        if not problems and not streamed:
            problems = ["answered without streaming a single token"]
        elif "".join(streamed) != reply and reply not in FALLBACKS:
            problems = ["streamed text does not match the final reply"]

        if problems:
            failures.append((step.question, problems))

        print(f"\n[{index}/{len(script)}] {step.act}")
        print(f"  Q: {step.question}")
        print(f"  A: {safe(reply)[:400]}")
        first_status = statuses[0] if statuses else "-"
        print(f"  ({cost} prompt tokens | {len(streamed)} tokens streamed "
              f"| first status: {first_status})")
        for problem in problems:
            print(f"  FAIL: {problem}")

        rows.append((step.act, step.question, step.why, reply, cost, problems))

    orchestrator.stream = original
    await close_mongo_connection()

    OUTPUT.parent.mkdir(exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        f.write("# the marketplace Assistant - Demo Run Sheet\n\n")
        f.write(
            "Every answer below was produced by an actual run against the demo\n"
            "database. Ask the questions in this order; the sequence builds.\n\n"
            "**Use the same `session_id` throughout** - several questions depend\n"
            "on the ones before them.\n\n"
        )
        # BUILT, NOT FIXED PROSE. This block used to assume the demo ran
        # on invented data: it told the operator to confirm "OK demo
        # data" and to mint a token with generate_test_token.py. Run
        # against marketplace both instructions are wrong - the first inverts
        # what to check for, and the second refuses without --real-data.
        real = get_settings().mongodb_database == "marketplace"
        f.write(
            "## Before you start\n\n"
            "Run these in order. Each prevents a specific way the demo breaks.\n\n"
            "```\n"
            "uv run python scripts/inspect_data.py     # names the buyer you will demo as\n"
            "uv run python scripts/check_redis.py      # must say: OK connected\n"
            "uv run python scripts/warm_cache.py       # must say: CACHE WARM\n"
            "uv run uvicorn app.api.main:app           # NO --reload\n"
            "```\n\n"
            "Then open **http://127.0.0.1:8000/demo** and click an account.\n"
            "There is no token to paste - the picker signs you in.\n\n"
            "| Step | Prevents |\n"
            "|---|---|\n"
            "| `inspect_data` | not noticing WHOSE data is about to be on screen |\n"
            "| `check_redis` | memory vanishing mid-demo, so follow-ups lose context |\n"
            "| `warm_cache` | the first question being slow and full price |\n"
            "| no `--reload` | a file save restarting the server mid-answer |\n\n"
            "Warm the cache a minute or two before you begin, not the night\n"
            "before - it expires after 2 hours idle.\n\n"
        )
        if real:
            f.write(
                "> **This run was against `marketplace` - REAL customer data.** The\n"
                "> accounts in the picker are real marketplace buyers, and the orders\n"
                "> quoted below are theirs. Fine on a laptop you control; never\n"
                "> behind a URL. Set `DEMO_UI_ENABLED=false` before deploying.\n\n"
            )
        f.write("---\n\n")
        for i, (act, q, why, reply, cost, failed) in enumerate(rows, start=1):
            f.write(f"## {i}. {act}\n\n")
            f.write(f"**Ask:** `{q}`\n\n")
            f.write("**It answered:**\n\n")
            for line in reply.splitlines():
                f.write(f"> {line}\n")
            f.write(f"\n**Why this lands:** {why}\n\n")
            if failed:
                f.write("> [!] FAILED the last rehearsal:\n")
                for problem in failed:
                    f.write(f"> - {problem}\n")
                f.write("\n")
            f.write(f"*Cost: {cost} prompt tokens*\n\n---\n\n")

        f.write("## Token budget\n\n")
        f.write(f"- Total for the full demo: **{total:,} prompt tokens**\n")
        f.write(f"- Average per question: **{total // max(len(rows), 1):,}**\n")
        f.write("- Groq free tier: **8,000 per minute**, shared across all users\n\n")
        minutes = total / 8000
        f.write(
            f"At the free tier this demo needs about **{minutes:.1f} minutes** of\n"
            "elapsed time no matter how fast anyone types. On a paid key it runs\n"
            "at conversation speed.\n"
        )

    print(f"\n{'=' * 60}")
    print(f"total: {total:,} prompt tokens  |  free-tier floor: {total / 8000:.1f} min")
    print(f"written: {OUTPUT}")
    if failures:
        print(f"\n!! {len(failures)} question(s) FAILED - not demo-ready:")
        for question, problems in failures:
            print(f"   - {question}")
            for problem in problems:
                print(f"       {problem}")
    else:
        print(
            "\nAll questions answered, and every answer actually mentioned "
            "what it was asked about."
        )


asyncio.run(main())
