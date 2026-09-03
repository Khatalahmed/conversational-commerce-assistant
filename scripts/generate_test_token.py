"""
DEV-ONLY TOOL - NOT part of the deployed app. Generates a test JWT so
you can demo the /chat endpoint while waiting for the real marketplace
backend signing secret (Phase 3.1). The moment the real secret arrives,
update .env's JWT_SECRET and this becomes unnecessary - the app itself
NEVER issues tokens, only verifies them, matching the real production
design where the main the marketplace backend is the only token issuer.

WHY THIS SCRIPT ASKS BEFORE TOUCHING THE REAL DATABASE:
    It used to open with `db.orders.find_one({})` and mint a token for
    whoever bought that order. Against MONGODB_DATABASE=marketplace that is a
    real marketplace customer, chosen at random, and everything demoed
    afterwards was their purchase history and their delivery city -
    without anyone in the room realising, because the script printed
    only a raw ObjectId.

    IT IS ALSO NOT STABLE. find_one({}) has no filter and no sort, so
    Mongo returns whatever it finds first in storage order, and that
    order is not guaranteed: a compaction, a document move or a restore
    can change it. So the identity being demoed is not merely "a real
    customer" but "an arbitrary real customer who may quietly become a
    different one" - which also breaks docs/demo-script.md, since the
    order IDs in it belong to whoever was first last time.

    Both problems disappear against marketplace_demo, where the buyers are
    invented. Real data now needs --real-data, matching the convention
    rehearse_demo.py already uses, and the script says out loud whose
    account it just handed you either way.
"""

import asyncio
import os

import sys
from pathlib import Path

# Dev script lives in scripts/, so the project root must be on sys.path
# for `import app` to resolve when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from app.config.settings import get_settings  # noqa: E402
from app.db.connection import close_mongo_connection, connect_to_mongo, get_database  # noqa: E402
from app.security.auth import create_test_token  # noqa: E402

TOKEN_MINUTES = 120


async def main():
    settings = get_settings()
    real = settings.mongodb_database == settings.production_database_name

    # THE CHECK HAPPENS BEFORE THE CONNECTION, so refusing costs nothing
    # and cannot half-do anything.
    if real and "--real-data" not in sys.argv:
        raise SystemExit(
            f"Refusing to mint a token against the REAL database "
            f"('{settings.production_database_name}').\n\n"
            "This picks an arbitrary REAL marketplace customer and hands you their\n"
            "account. Everything you then demo - orders, delivery city,\n"
            "cart - is that person's, and they did not agree to be in a\n"
            "demo.\n\n"
            "  Safe:  set MONGODB_DATABASE=marketplace_demo in .env\n"
            "         (the demo cluster is already seeded)\n"
            "  Or:    pass --real-data if you genuinely need a real account."
        )

    await connect_to_mongo()
    db = get_database()

    # Unchanged, and still arbitrary - see the module docstring. What is
    # new is that the arbitrariness is now visible in the output.
    order = await db.orders.find_one({})
    if order is None:
        await close_mongo_connection()
        raise SystemExit(
            f"No orders in database '{settings.mongodb_database}', so there is "
            f"no buyer to mint a token for.\n"
            f"Seed it first:  uv run python scripts/seed_demo_data.py"
        )

    user_id = str(order["buyerId"])
    buyer = await db.users.find_one({"_id": order["buyerId"]}, {"username": 1})
    address = (order.get("deliveryAddress") or {})
    order_count = await db.orders.count_documents({"buyerId": order["buyerId"]})
    await close_mongo_connection()

    token = create_test_token(user_id=user_id, expires_in_minutes=TOKEN_MINUTES)

    # WHO, NOT JUST WHICH ID. A raw ObjectId tells the person running
    # this nothing, which is exactly how a real customer's history ended
    # up on screen unnoticed. A username and a city are recognisable.
    print(f"database : {settings.mongodb_database}" + ("   << REAL CUSTOMER DATA" if real else ""))
    print(f"account  : {(buyer or {}).get('username') or '(no username)'}")
    print(f"user_id  : {user_id}")
    print(f"they have: {order_count} orders"
          + (f", delivering to {address.get('city')}, {address.get('state')}"
             if address.get("city") else ""))

    if real:
        print()
        print("  !! This is a REAL marketplace customer, picked by storage order")
        print("     rather than chosen. Anything you demo is their data, and")
        print("     which customer it is can change without the code changing.")
        print("     Use MONGODB_DATABASE=marketplace_demo unless you need real data.")

    print(f"\nToken (valid {TOKEN_MINUTES} minutes):\n{token}")
    # PORT, NOT A HARDCODED 8000. This script has no way to know which
    # port uvicorn was started on, and printing the wrong one sends
    # whoever is following along to a page that does not answer. PORT is
    # the same variable the Dockerfile uses, so there is one convention
    # rather than two:  PORT=8010 uv run python scripts/generate_test_token.py
    port = os.environ.get("PORT", "8000")
    print(f"\nPaste it at http://127.0.0.1:{port}/demo, or into the 'Authorize'")
    print(f"button at http://127.0.0.1:{port}/docs")
    if "PORT" not in os.environ:
        print("(assuming port 8000 - set PORT if uvicorn is on another one)")


asyncio.run(main())
