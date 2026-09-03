"""
Copies the marketplace's REAL product catalogue into the demo database, and
repoints the synthetic orders, carts, bargains and reviews at it.

    uv run python scripts/import_real_catalogue.py "<DEMO_CLUSTER_URI>"           # dry run
    uv run python scripts/import_real_catalogue.py "<DEMO_CLUSTER_URI>" --apply   # write

WHY THIS IS SAFE, AND WHERE THE LINE IS:
    A product catalogue is not personal data. Names, prices, categories,
    images and stock are what the marketplace app shows to anyone who opens it.
    Copying those into the demo database means the client sees THEIR OWN
    products in the demo - recognisable, real prices - instead of twelve
    items invented for the purpose.

    Everything that identifies a PERSON stays synthetic:

      COPIED      products, product_embeddings, categories
      NOT COPIED  users, orders, carts, bargains, addresses,
                  notifications, payouts, reviews, bits, livesessions

    sellerId is REMAPPED. A real product carries the ObjectId of a real
    seller account, and that is a reference to a person - so every
    imported product is reassigned to one of the two synthetic demo
    sellers. No real user id reaches the demo database at all.

    Bits and live sessions are skipped even though they look catalogue-ish:
    their comment arrays carry real usernames, and their hostId/userId are
    real accounts.

WHAT THIS FIXES BEYOND REALISM:
    find_similar_products currently runs on SYNTHETIC embeddings I
    generated from category and tags. Importing the real ones means the
    demo runs on the marketplace's actual vectors - the semantic search stops being
    a simulation of the feature and becomes the feature.

SAFETY:
    Reads staging with the read-only account (it could not write there if
    it tried). Refuses to write anywhere that looks like the real cluster.
    Dry run unless --apply is passed.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bson import ObjectId  # noqa: E402
from motor.motor_asyncio import AsyncIOMotorClient  # noqa: E402

from app.config.settings import get_settings  # noqa: E402

DEMO_DATABASE = "marketplace_demo"
STAGING_HOST_MARKER = "marketplace-semantic-search"

# Must match seed_demo_data.py - these are the synthetic sellers every
# imported product gets reassigned to.
SELLER_A = ObjectId(f"{10:024x}")
SELLER_B = ObjectId(f"{11:024x}")

COPY = ["products", "product_embeddings", "categories"]
NEVER_COPY = [
    "users", "orders", "carts", "bargains", "addresses",
    "notifications", "payouts", "reviews", "bits", "livesessions",
]



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

def line() -> None:
    print("-" * 66)


async def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    apply = "--apply" in sys.argv

    if not args:
        raise SystemExit(
            "Give the DEMO cluster URI as an argument.\n\n"
            '  uv run python scripts/import_real_catalogue.py "mongodb+srv://..."\n\n'
            "MONGODB_URI in .env must currently point at the REAL cluster,\n"
            "since that is where the catalogue is read from."
        )

    demo_uri = args[0]
    settings = get_settings()
    source_uri = settings.mongodb_uri

    source_host = _host_of(source_uri)
    demo_host = _host_of(demo_uri)

    if STAGING_HOST_MARKER not in source_host.lower():
        raise SystemExit(
            f"MONGODB_URI points at {source_host}, not the real marketplace cluster.\n"
            "Point it at the staging cluster - that is the catalogue source."
        )
    if STAGING_HOST_MARKER in demo_host.lower():
        raise SystemExit(
            "The DESTINATION is the real marketplace cluster. Refusing to write.\n"
            "Pass the demo cluster URI instead."
        )

    # Catch the placeholder being pasted verbatim. It reads like
    # something to copy, and the dry run used to accept it happily.
    if "your-demo-cluster-uri" in demo_uri.lower() or "://" not in demo_uri:
        raise SystemExit(
            f"That is not a connection string: {demo_uri!r}\n\n"
            "Paste your actual demo cluster URI, the mongodb+srv://... one."
        )

    source = AsyncIOMotorClient(source_uri)["marketplace"]
    demo_client = AsyncIOMotorClient(demo_uri, serverSelectionTimeoutMS=8000)
    demo = demo_client[DEMO_DATABASE]

    # REACH THE DESTINATION EVEN ON A DRY RUN.
    #
    # It previously only read from the source and printed a plan, so a
    # dry run against an unreachable destination looked like a complete
    # success - and then --apply died on the first write. A dry run that
    # cannot fail the way the real run fails is not a rehearsal.
    try:
        await demo_client.admin.command("ping")
    except Exception as exc:
        demo_client.close()
        raise SystemExit(
            f"Cannot reach the destination cluster ({demo_host}).\n"
            f"  {type(exc).__name__}: {str(exc)[:160]}\n\n"
            "Check the URI, the password, and that your IP is allowed in\n"
            "Atlas Network Access."
        )

    line()
    print("IMPORT REAL CATALOGUE" + ("" if apply else "   (DRY RUN)"))
    line()
    print(f"  from: {source_host}  db 'marketplace'      (read-only)")
    print(f"  to:   {demo_host}  db '{DEMO_DATABASE}'")
    print()

    products = await source.products.find({}).to_list(length=None)
    embeddings = await source.product_embeddings.find({}).to_list(length=None)
    categories = await source.categories.find({}).to_list(length=None)

    real_seller_ids = {p.get("sellerId") for p in products if p.get("sellerId")}

    line()
    print("WOULD COPY" if not apply else "COPYING")
    line()
    print(f"  products            {len(products):>5}")
    print(f"  product_embeddings  {len(embeddings):>5}")
    print(f"  categories          {len(categories):>5}")
    print()
    print(f"  {len(real_seller_ids)} real seller ids -> remapped to 2 synthetic sellers")
    print()
    print("  sample of what arrives:")
    for p in products[:5]:
        print(f"    {p.get('name')!r}  Rs.{p.get('discountedPrice')}  [{p.get('category')}]")

    line()
    print("WOULD NOT COPY (stays synthetic)")
    line()
    for name in NEVER_COPY:
        count = await source[name].count_documents({})
        print(f"  {name:<16} {count:>5} real docs  - left untouched")
    print()
    print("  Nothing identifying a person is read into the demo database.")

    if not apply:
        line()
        print("  Dry run. Re-run with --apply to write.")
        line()
        demo_client.close()
        return

    # ---- write ----
    seller_cycle = [SELLER_A, SELLER_B]
    cleaned = []
    for index, product in enumerate(products):
        doc = dict(product)
        doc.pop("stepData", None)      # internal seller-flow scratch data
        doc.pop("comments", None)      # free text from real users
        doc["sellerId"] = seller_cycle[index % 2]
        cleaned.append(doc)

    await demo.products.delete_many({})
    await demo.products.insert_many(cleaned)

    await demo.product_embeddings.delete_many({})
    if embeddings:
        await demo.product_embeddings.insert_many(embeddings)

    await demo.categories.delete_many({})
    if categories:
        await demo.categories.insert_many(
            [{k: v for k, v in c.items()} for c in categories]
        )

    print()
    line()
    print("REPOINTING SYNTHETIC DOCS AT THE REAL CATALOGUE")
    line()

    # The synthetic orders/carts/bargains referenced invented product ids.
    # Those products no longer exist, so every reference is rewritten to a
    # real one - deterministically, so re-running gives the same demo.
    picks = sorted(cleaned, key=lambda p: str(p["_id"]))[:12]
    if not picks:
        raise SystemExit("no products imported - nothing to repoint")

    def pick(n: int) -> dict:
        return picks[n % len(picks)]

    updated = 0
    for n, order in enumerate(await demo.orders.find({}).to_list(length=None)):
        items = []
        for i, item in enumerate(order.get("items") or []):
            product = pick(n + i)
            price = product.get("discountedPrice") or product.get("price") or 0
            qty = item.get("qty", 1)
            items.append({
                **item,
                "product": product["_id"],
                "name": product.get("name"),
                "price": price,
                "total": price * qty,
                "sellerId": product["sellerId"],
            })
        subtotal = sum(i["total"] for i in items)
        await demo.orders.update_one(
            {"_id": order["_id"]},
            {"$set": {
                "items": items,
                "pricing": {**order.get("pricing", {}), "subtotal": subtotal,
                            "tax": round(subtotal * 0.05),
                            "total": subtotal + round(subtotal * 0.05)},
                "sellerId": items[0]["sellerId"] if items else order.get("sellerId"),
            }},
        )
        updated += 1
    print(f"  orders repointed    {updated}")

    for n, cart in enumerate(await demo.carts.find({}).to_list(length=None)):
        items = []
        for i, item in enumerate(cart.get("items") or []):
            product = pick(n + i + 3)
            items.append({**item, "product": product["_id"],
                          "cartPrice": product.get("discountedPrice") or 0})
        await demo.carts.update_one({"_id": cart["_id"]}, {"$set": {"items": items}})
    print("  carts repointed     ok")

    for n, bargain in enumerate(await demo.bargains.find({}).to_list(length=None)):
        product = pick(n + 6)
        original = product.get("discountedPrice") or product.get("price") or 1000
        await demo.bargains.update_one(
            {"_id": bargain["_id"]},
            {"$set": {
                "productId": product["_id"],
                "sellerId": product["sellerId"],
                "originalPrice": original,
                "offeredPrice": int(original * 0.8),
                "currentPrice": int(original * 0.9),
                "productSnapshot": {
                    "name": product.get("name"),
                    "image": (product.get("images") or [{}])[0].get("url", ""),
                },
            }},
        )
    print("  bargains repointed  ok")

    for n, review in enumerate(await demo.reviews.find({}).to_list(length=None)):
        await demo.reviews.update_one(
            {"_id": review["_id"]}, {"$set": {"productId": pick(n)["_id"]}}
        )
    print("  reviews repointed   ok")

    print()
    line()
    print("  Done. The demo now shows the marketplace's real products with")
    print("  entirely synthetic customers, orders and carts.")
    print()
    print("  Next:")
    print("    - point MONGODB_URI back at the DEMO cluster")
    print("    - MONGODB_DATABASE=marketplace_demo")
    print("    - uv run python scripts/inspect_data.py    (expect: OK demo data)")
    print("    - uv run python scripts/rehearse_demo.py   (names have changed)")
    line()
    demo_client.close()


asyncio.run(main())
