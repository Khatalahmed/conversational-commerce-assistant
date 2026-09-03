"""
Re-embeds a product catalogue with Azure text-embedding-3-small, and
rebuilds the Atlas vector index to match.

    uv run python scripts/reembed_catalogue.py "<WRITABLE_URI>" <database>
    uv run python scripts/reembed_catalogue.py "<WRITABLE_URI>" <database> --apply

WHY:
    The existing product_embeddings were produced by a model nothing
    records - 384 dimensions, unit-normalised, otherwise unidentified.
    find_similar_products works around that by using a STORED vector as
    the query, which needs no model. Free-text search cannot: "something
    cosy for winter" has to be embedded, and embedding it with the wrong
    model returns ranked nonsense that looks like a working search.

    Re-embedding with a known model removes the unknown. Every vector is
    then one you produced, with a tag saying so.

    Cost is not a factor at this size: 143 products is roughly 14,000
    tokens, about $0.0003 at text-embedding-3-small's $0.02/M.

*** READ THIS BEFORE RUNNING IT ON A LIVE DATABASE ***

    the marketplace's own pipeline watches products via a change stream and
    re-embeds them with the OLD model. If it is still running when this
    finishes, the first product anyone edits gets a 384-dim vector
    written back into a 1536-dim collection - and the index will reject
    it or the search will silently skip it.

    So on a live database this is a MIGRATION, not a script: stop or
    update that pipeline first. Against the demo database, none of that
    applies.

SAFETY:
    Dry run unless --apply. Refuses the real cluster unless
    --i-have-stopped-the-sync-pipeline is also passed, which is
    deliberately awkward to type.
"""

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motor.motor_asyncio import AsyncIOMotorClient  # noqa: E402

from app.agent.embeddings import (  # noqa: E402
    embed_texts,
    embedding_model_tag,
    embeddings_configured,
    product_text,
)
from app.config.settings import get_settings  # noqa: E402

STAGING_HOST_MARKER = "marketplace-semantic-search"
VECTOR_INDEX = "product_vector_index"
EMBEDDINGS_COLLECTION = "product_embeddings"

# text-embedding-3-small, per Azure's published rate.
USD_PER_MILLION_TOKENS = 0.02


def line() -> None:
    print("-" * 68)


def _host_of(uri: str) -> str:
    return uri.split("@")[-1].split("/")[0].split("?")[0].lower()


async def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    apply = "--apply" in sys.argv
    sync_stopped = "--i-have-stopped-the-sync-pipeline" in sys.argv

    # DEFAULTS TO THE DEMO CLUSTER FROM .env.
    #
    # MONGODB_URI holds one cluster at a time, so the demo URI kept being
    # lost when someone switched to real data - and the real one got
    # pasted in its place. Reading DEMO_MONGODB_URI removes the step
    # where a URI has to be found and retyped at all.
    demo_uri = get_settings().demo_mongodb_uri
    uri = args[0] if args else demo_uri
    database = args[1] if len(args) > 1 else "marketplace_demo"

    if not uri:
        raise SystemExit(
            "No cluster given, and DEMO_MONGODB_URI is not set in .env.\n\n"
            "Either:\n"
            "  set DEMO_MONGODB_URI in .env (recommended - it survives\n"
            "  switching MONGODB_URI back and forth), or pass it:\n\n"
            '    uv run python scripts/reembed_catalogue.py "<URI>" <database>\n\n'
            "It must be a cluster you can WRITE to. The chatbot's own\n"
            "account is read-only by design."
        )

    host = _host_of(uri)
    if not args:
        print(f"  using DEMO_MONGODB_URI from .env ({host}, db '{database}')\n")

    if not embeddings_configured():
        raise SystemExit(
            "AZURE_EMBEDDING_DEPLOYMENT is not set in .env.\n\n"
            "Deploy text-embedding-3-small in your Azure OpenAI resource,\n"
            "then set AZURE_EMBEDDING_DEPLOYMENT to the DEPLOYMENT name\n"
            "(which need not match the model name)."
        )

    if STAGING_HOST_MARKER in host and not sync_stopped:
        raise SystemExit(
            f"That is the live the marketplace cluster ({host}).\n\n"
            "the marketplace's change-stream pipeline re-embeds edited products with\n"
            "the OLD 384-dim model. If it is still running, the first product\n"
            "anyone edits after this writes a 384-dim vector into a 1536-dim\n"
            "collection, and search silently degrades.\n\n"
            "Stop or update that pipeline, then re-run with:\n"
            "    --i-have-stopped-the-sync-pipeline\n\n"
            "Or run this against the demo database first."
        )

    settings = get_settings()
    client = AsyncIOMotorClient(uri, serverSelectionTimeoutMS=8000)
    db = client[database]

    try:
        await client.admin.command("ping")
    except Exception as exc:
        client.close()
        raise SystemExit(f"Cannot reach {host}: {type(exc).__name__}: {str(exc)[:160]}")

    line()
    print("RE-EMBED CATALOGUE" + ("" if apply else "   (DRY RUN)"))
    line()
    print(f"  cluster    {host}")
    print(f"  database   {database}")
    print(f"  model      {settings.azure_embedding_deployment} "
          f"({settings.azure_embedding_dimensions} dims)")
    print(f"  tag        {embedding_model_tag()}")
    print()

    products = await db.products.find({}).to_list(length=None)
    if not products:
        client.close()
        raise SystemExit(f"no products in {database}.products - nothing to embed")

    texts = [product_text(p) for p in products]
    # ~4 chars per token is rough but stable enough to size a decision
    # that turns out to cost a fraction of a cent either way.
    tokens = sum(len(t) for t in texts) // 4
    cost = tokens / 1_000_000 * USD_PER_MILLION_TOKENS

    existing = await db[EMBEDDINGS_COLLECTION].count_documents({})
    sample = await db[EMBEDDINGS_COLLECTION].find_one({})
    current_dims = len(sample["embedding"]) if sample and sample.get("embedding") else 0
    current_tag = (sample or {}).get("embeddingModel", "(none - pre-migration)")

    line()
    print("CURRENT STATE")
    line()
    print(f"  products              {len(products):>6}")
    print(f"  existing embeddings   {existing:>6}  ({current_dims} dims)")
    print(f"  existing model tag    {current_tag}")
    print()

    line()
    print("WOULD WRITE" if not apply else "WRITING")
    line()
    print(f"  embeddings to create  {len(products):>6}  "
          f"({settings.azure_embedding_dimensions} dims)")
    print(f"  estimated tokens      {tokens:>6,}")
    print(f"  estimated cost        ${cost:.4f}")
    print()
    print("  text that gets embedded (first product):")
    print(f"    {texts[0][:200]!r}")
    print()
    print(f"  the '{VECTOR_INDEX}' index will be DROPPED and recreated at "
          f"{settings.azure_embedding_dimensions} dims,")
    print("  because Atlas fixes the width in the index definition.")

    if not apply:
        line()
        print("  Dry run. Nothing written. Re-run with --apply.")
        line()
        client.close()
        return

    print()
    line()
    print("EMBEDDING")
    line()
    vectors = await embed_texts(texts)
    print(f"  received {len(vectors)} vectors of {len(vectors[0])} dims")

    docs = [
        {
            "_id": product["_id"],
            "embedding": vector,
            # The tag is the whole safety mechanism: it is what lets a
            # search refuse a catalogue it did not embed, rather than
            # silently comparing across two vector spaces.
            "embeddingModel": embedding_model_tag(),
            "updatedAt": datetime.now(timezone.utc).replace(tzinfo=None),
        }
        for product, vector in zip(products, vectors)
    ]

    await db[EMBEDDINGS_COLLECTION].delete_many({})
    await db[EMBEDDINGS_COLLECTION].insert_many(docs)
    print(f"  wrote {len(docs)} embedding documents")

    print()
    line()
    print("REBUILDING THE VECTOR INDEX")
    line()
    try:
        await db[EMBEDDINGS_COLLECTION].drop_search_index(VECTOR_INDEX)
        print(f"  dropped {VECTOR_INDEX}")
        # Atlas deletes asynchronously; recreating too soon is rejected.
        await asyncio.sleep(10)
    except Exception as exc:
        print(f"  no existing index to drop ({type(exc).__name__})")

    await db[EMBEDDINGS_COLLECTION].create_search_index(
        {
            "name": VECTOR_INDEX,
            "type": "vectorSearch",
            "definition": {
                "fields": [
                    {
                        "type": "vector",
                        "path": "embedding",
                        "numDimensions": settings.azure_embedding_dimensions,
                        "similarity": "cosine",
                    }
                ]
            },
        }
    )
    print(f"  requested {VECTOR_INDEX} at "
          f"{settings.azure_embedding_dimensions} dims (builds in ~1 min)")

    print()
    line()
    print("  Done. Wait for the index to report READY, then:")
    print("    uv run python scripts/inspect_data.py")
    print()
    print("  Free-text semantic search is now available; find_similar_products")
    print("  keeps working and now runs on vectors you produced.")
    line()
    client.close()


asyncio.run(main())
