"""
Runs the golden set against the real model and prints a score.

Every case is a full conversation, so a whole run costs real quota and
several minutes. Start small:

    uv run python scripts/run_evals.py --limit 3
    uv run python scripts/run_evals.py --only order-where,trending
    uv run python scripts/run_evals.py --label prompt-v7

Comparing two runs is the point of the thing - edit the prompt, run
again, and see which cases moved:

    uv run python scripts/run_evals.py --label v8 --against evals/prompt-v7.json

Results land in evals/<label>.json.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db.connection import (  # noqa: E402
    close_mongo_connection,
    connect_to_mongo,
    get_database,
)
from app.evals import runner  # noqa: E402


async def pick_user() -> str:
    """A buyer with real orders, so the account cases have something to
    find. A user with no history would make half the set pass by
    answering "you have none", which is true and proves nothing."""
    db = get_database()
    order = await db["orders"].find_one({}, {"buyerId": 1})
    if not order:
        raise SystemExit("No orders in this database - the account cases "
                         "would all be vacuous. Point .env at real data.")
    return str(order["buyerId"])


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="run", help="names the output file")
    parser.add_argument("--only", default="", help="comma-separated case ids")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--delay", type=float, default=2.0,
                        help="seconds between cases, to stay under rate limits")
    parser.add_argument("--against", default="", help="a previous results file")
    parser.add_argument(
        "--judge", action="store_true",
        help="also ask an LLM whether the answer named a product no tool "
             "returned. Doubles the LLM calls, and applies a check an "
             "unjudged run does not - compare judged runs to judged runs.",
    )
    parser.add_argument("--out-dir", default="evals")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Partial results land here after every case, so a provider drop
    # costs one case rather than the whole run.
    checkpoint = str(out_dir / f"{args.label}.partial.json")

    await connect_to_mongo()
    try:
        user_id = await pick_user()
        only = {c.strip() for c in args.only.split(",") if c.strip()}
        print(f"user {user_id}  ·  delay {args.delay}s\n")
        results = await runner.run(user_id, only=only or None,
                                   limit=args.limit, delay=args.delay,
                                   checkpoint=checkpoint, judge=args.judge)
    finally:
        await close_mongo_connection()

    payload = runner.report(results, args.label)

    if args.against:
        previous = Path(args.against)
        if previous.exists():
            runner.compare(payload, json.loads(previous.read_text(encoding="utf-8")))
        else:
            print(f"\n(no previous run at {previous})")

    runner.save(payload, str(out_dir / f"{args.label}.json"))
    Path(checkpoint).unlink(missing_ok=True)

    # Non-zero on failure, so this can gate a release if anyone wants it
    # to. Not wired into CI: CI has no Atlas access and no LLM quota.
    sys.exit(0 if payload["passed"] == payload["total"] else 1)


asyncio.run(main())
