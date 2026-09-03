"""
Prints what the assistant can actually see, against whichever database
.env currently points at. No LLM calls - free, instant, repeatable.

Use it to sanity-check a demo database after seeding, or to look at real
staging without going through the chat endpoint.

    uv run python scripts/inspect_data.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config.settings import get_settings  # noqa: E402
from app.db.connection import (  # noqa: E402
    close_mongo_connection,
    connect_to_mongo,
    get_database,
)
from app.repos import (  # noqa: E402
    bargains_repo,
    carts_repo,
    orders_repo,
    products_repo,
)


def rule(title: str) -> None:
    print(f"\n{'-' * 68}\n{title}\n{'-' * 68}")



def _host_of(uri: str) -> str:
    """The HOSTNAME only - no credentials, no path, no query string.

    The query string matters here. A URI like

        mongodb+srv://u:p@cluster0.example.net?appName=the marketplace-Semantic-Search

    has no slash before "?", so splitting on "/" alone leaves the appName
    glued to the host - and a guard looking for the staging cluster's name
    then matched a COMPLETELY DIFFERENT cluster whose appName happened to
    mention it. The guard refused a legitimate destination and blamed the
    user. Strip "?" as well as "/".
    """
    return uri.split("@")[-1].split("/")[0].split("?")[0].lower()

async def main() -> None:
    settings = get_settings()
    host = _host_of(settings.mongodb_uri)

    await connect_to_mongo()
    db = get_database()

    rule("WHERE THIS IS POINTED")
    print(f"  cluster:  {host}")
    print(f"  database: {settings.mongodb_database}")
    # CHECK THE CLUSTER, NOT JUST THE DATABASE NAME.
    #
    # This originally tested `mongodb_database == "marketplace"` alone, and
    # printed "safe for LLM calls" while pointed at the real marketplace
    # cluster - the database name happened to say demo, so the check
    # passed. A safety banner that can say "safe" when it is not is
    # worse than having none, because it is trusted.
    #
    # The staging HOSTNAME is the reliable signal: it identifies the
    # cluster holding real customers regardless of which database inside
    # it is selected.
    on_real_cluster = "marketplace-semantic-search" in host.lower()
    on_real_database = settings.mongodb_database == "marketplace"

    if on_real_cluster and on_real_database:
        print("  !! REAL customer data - do not send this to a free-tier LLM")
    elif on_real_cluster:
        print(f"  !! REAL cluster, but database '{settings.mongodb_database}'")
        print("     If that database does not exist here, nothing will load.")
        print("     For real data set MONGODB_DATABASE=marketplace; for the demo")
        print("     point MONGODB_URI back at the demo cluster.")
    elif on_real_database:
        print("  ?  Database is named 'marketplace' on a non-the marketplace cluster.")
        print("     Probably a copy - check before assuming it is safe.")
    else:
        print("  OK  demo data - safe for LLM calls")

    rule("COLLECTIONS")
    for name in sorted(await db.list_collection_names()):
        print(f"  {name:<22} {await db[name].count_documents({}):>5}")

    rule("SEARCH INDEXES")
    for coll in ("products", "product_embeddings"):
        try:
            idx = await db[coll].list_search_indexes().to_list(length=None)
            for i in idx:
                print(f"  {coll}.{i['name']:<24} {i.get('type','search'):<13} {i.get('status')}")
            if not idx:
                print(f"  {coll}: none")
        except Exception as exc:
            print(f"  {coll}: unavailable ({type(exc).__name__})")

    rule("PRODUCT SEARCH  -  Atlas Search vs the old regex")
    for query in ("buddha", "budha", "watch", "frame"):
        new = await products_repo.search_products_by_name(query, limit=5)
        old = await products_repo._search_products_by_name_regex(query, 5)
        print(f"\n  {query!r}   atlas={len(new)}  regex={len(old)}")
        for p in new:
            print(f"      {p.get('name')}")
        if not new:
            print("      (nothing)")

    rule("SIMILAR PRODUCTS  -  vector search")
    seed = await products_repo.search_products_by_name("buddha", limit=1)
    if seed:
        print(f"  similar to: {seed[0].get('name')}")
        for p in await products_repo.find_similar_products(str(seed[0]["_id"]), limit=5):
            print(f"      {p.get('name')}  (Rs.{p.get('discountedPrice')})")
    else:
        print("  no seed product found")

    rule("A BUYER'S VIEW  -  what the assistant would retrieve")
    order = await db.orders.find_one({})
    if not order:
        print("  no orders in this database")
    else:
        user_id = str(order["buyerId"])
        # WHO, NOT JUST WHICH ID. This buyer is whoever find_one({})
        # happened to return, and against the real database that is an
        # arbitrary REAL customer. A bare ObjectId gives the person
        # reading this output no way to notice that; a username and a
        # city do.
        buyer = await db.users.find_one({"_id": order["buyerId"]}, {"username": 1})
        city = (order.get("deliveryAddress") or {}).get("city")
        print(f"  buyer: {(buyer or {}).get('username') or '(no username)'}"
              f"{f' - {city}' if city else ''}  [{user_id}]\n")

        history = await orders_repo.get_order_history(user_id, limit=10)
        print(f"  orders ({len(history)}):")
        for o in history:
            print(f"      {o.get('orderId')}  {o.get('status'):<10}"
                  f"  Rs.{(o.get('pricing') or {}).get('total')}")

        cart = await carts_repo.get_cart(user_id)
        # Same arithmetic the assistant uses - see _enrich_cart in
        # agent/tool_executor.py. cartPrice is PER UNIT, so qty matters.
        subtotal = sum(
            (i.get("cartPrice") or 0) * (i.get("qty") or 0) for i in cart["items"]
        )
        print(f"\n  cart: {cart['itemCount']} items, subtotal Rs.{subtotal}")

        bargain = await db.bargains.find_one({"buyerId": order["buyerId"]})
        if bargain:
            status = await bargains_repo.get_bargain_status(
                user_id, product_id=str(bargain["productId"])
            )
            print(f"  bargain: {status}")
        else:
            print("  bargain: none for this buyer")

        print("\n  cross-user check - another buyer's order must NOT resolve:")
        other = await db.orders.find_one({"buyerId": {"$ne": order["buyerId"]}})
        if other:
            leaked = await orders_repo.get_order_status(user_id, other["orderId"])
            print(f"      {other['orderId']} -> {leaked}"
                  f"   {'OK blocked' if leaked is None else 'XX LEAKED'}")
        else:
            print("      (only one buyer in this database)")

    print()
    await close_mongo_connection()


asyncio.run(main())
