"""
WHAT:
    Identifies which model embedded the existing catalogue, by evidence
    rather than by inference from the dimension count.

WHY THIS EXISTS:
    Everything anyone has ever written about the original pipeline says
    the same thing: 384 dimensions, unit-normalised, otherwise
    unidentified. "384 + unit-normalised = almost certainly MiniLM" is a
    reasonable guess and a terrible foundation, because guessing wrong
    does not fail - it returns ranked, plausible, meaningless results.

THE TRICK THAT MAKES THIS POSSIBLE:
    We do not need the pipeline source. We have its OUTPUT: 143 stored
    vectors, sitting next to the 143 products they were built from. If a
    candidate model re-embeds a product's text and lands on essentially
    the stored vector, that model IS the model. The same model over the
    same text reproduces itself at ~0.999 cosine; a different model
    scores ~0.0-0.3. That gap is the whole mechanism - there is no
    ambiguous middle to argue about.

WHAT IT ALSO IDENTIFIES:
    The TEXT RECIPE. The model is only half the answer: MiniLM over
    "name" and MiniLM over "name + description + tags" produce different
    vectors. So this crosses every candidate model against every
    plausible recipe and reports the grid. The winning CELL gives both.

READ-ONLY. Opens the catalogue with the app's own credential, reads
products and product_embeddings, writes nothing. Safe against
production.

USAGE:
    uv run python scripts/identify_embedding_model.py
    uv run python scripts/identify_embedding_model.py --sample 20
    uv run python scripts/identify_embedding_model.py --model BAAI/bge-small-en-v1.5

NOTE ON DEPENDENCIES:
    sentence-transformers is already a DEV dependency. Running this
    downloads each candidate's weights (~90MB for MiniLM) into a local
    cache. It adds nothing to the deployed container - deciding whether
    to do THAT is precisely what this script exists to inform.
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motor.motor_asyncio import AsyncIOMotorClient  # noqa: E402

from app.config.settings import get_settings  # noqa: E402

EMBEDDINGS_COLLECTION = "product_embeddings"
PRODUCTS_COLLECTION = "products"

# The usual suspects for 384-dim, unit-normalised, cosine vectors.
# MiniLM first because it is the overwhelming default; the others exist
# so that a NEGATIVE result on MiniLM is informative rather than a dead
# end.
CANDIDATE_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-MiniLM-L12-v2",
    "sentence-transformers/paraphrase-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
    "thenlper/gte-small",
]

# How the pipeline might have assembled the text it embedded, ordered
# from most to least likely. Each is a list of product fields joined by
# a space; list fields are flattened first.
TEXT_RECIPES = {
    "name": ["name"],
    "name+category": ["name", "category"],
    "name+description": ["name", "description"],
    "name+category+description": ["name", "category", "description"],
    "name+description+tags": ["name", "description", "tags"],
    "name+category+subCategory+description+tags": [
        "name", "category", "subCategory", "description", "tags",
    ],
    # What THIS app would build, from app.agent.embeddings.product_text.
    "app_product_text": ["__app__"],
}

# Above this, the candidate REPRODUCED the stored vector: same model,
# same text. Identical runs land at 0.999+; float32 storage and
# tokenizer version drift account for the margin.
CONFIRMED = 0.98

# Between these, something is right but not everything - typically the
# correct model over a DIFFERENT text recipe. Reported loudly, because
# it means keep going with that model and other recipes.
SUGGESTIVE = 0.70

# GEOMETRY IS ONLY EVIDENCE IF IT DISCRIMINATES, and an absolute
# threshold cannot tell you whether it did.
#
# Measured on this catalogue: all-MiniLM-L6-v2 scores 0.8717 and
# all-MiniLM-L12-v2 scores 0.8773. Two DIFFERENT models, five
# thousandths apart. That is not both of them being right - it is the
# floor. Any competent sentence encoder agrees that dresses sit near
# dresses, so a clothing catalogue produces high correlation for
# everything, and a fixed cutoff of "0.75 means something" reads that
# floor as a discovery.
#
# So what counts is the MARGIN between the best model and the best
# OTHER model. A model that genuinely produced these vectors should
# stand clear of its neighbours; models separated by less than this are
# indistinguishable, whatever their absolute scores.
GEOMETRY_MARGIN = 0.10
GEOMETRY_STRONG_ABS = 0.75


def line(char="-"):
    print(char * 72)


def _field_text(product: dict, field: str) -> str:
    if field == "__app__":
        from app.agent.embeddings import product_text
        return product_text(product)
    if field in ("tags", "searchKeywords"):
        return " ".join(w for w in (product.get(field) or []) if w)
    return str(product.get(field) or "")


def build_text(product: dict, recipe: list[str]) -> str:
    parts = [_field_text(product, f).strip() for f in recipe]
    return " ".join(p for p in parts if p)


def cosine(a, b) -> float:
    """Plain Python: no numpy import for five lines of arithmetic."""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


async def load_pairs(sample: int):
    """Products next to the vectors that were built from them."""
    settings = get_settings()
    client = AsyncIOMotorClient(settings.mongodb_uri)
    db = client[settings.mongodb_database]

    stored = [
        doc async for doc in db[EMBEDDINGS_COLLECTION]
        .find({}, {"embedding": 1, "embeddingModel": 1})
        .limit(sample)
    ]
    if not stored:
        print("  No documents in product_embeddings - nothing to identify.")
        client.close()
        return []

    tags = {d.get("embeddingModel") for d in stored}
    only_tag = tags.pop() if len(tags) == 1 else tags
    print(f"  stored vectors  {len(stored)}")
    print(f"  dimensions      {len(stored[0]['embedding'])}")
    print(f"  tag             {only_tag or '(untagged - original pipeline)'}")
    mags = [sum(x * x for x in d["embedding"]) ** 0.5 for d in stored]
    note = "(unit-normalised)" if max(mags) < 1.001 else ""
    print(f"  magnitude       {min(mags):.4f} - {max(mags):.4f}  {note}")

    ids = [d["_id"] for d in stored]
    products = {
        p["_id"]: p
        async for p in db[PRODUCTS_COLLECTION].find({"_id": {"$in": ids}})
    }
    client.close()

    pairs = [
        (products[d["_id"]], d["embedding"])
        for d in stored
        if d["_id"] in products
    ]
    missing = len(stored) - len(pairs)
    if missing:
        print(f"  WARNING: {missing} embeddings have no matching product - skipped")
    print(f"  usable pairs    {len(pairs)}")
    return pairs


def _dimension_of(model) -> int:
    """get_sentence_embedding_dimension was renamed in sentence-transformers 6."""
    getter = getattr(model, "get_embedding_dimension", None) or         model.get_sentence_embedding_dimension
    return getter()


def _upper_triangle(vectors) -> list[float]:
    """Every pairwise cosine, flattened - the SHAPE of a vector space."""
    n = len(vectors)
    return [
        cosine(vectors[i], vectors[j])
        for i in range(n)
        for j in range(i + 1, n)
    ]


def pearson(a: list[float], b: list[float]) -> float:
    n = len(a)
    if n < 2:
        return 0.0
    ma, mb = sum(a) / n, sum(b) / n
    da = [x - ma for x in a]
    db = [y - mb for y in b]
    num = sum(x * y for x, y in zip(da, db))
    den = (sum(x * x for x in da) ** 0.5) * (sum(y * y for y in db) ** 0.5)
    return num / den if den else 0.0


def score_model(model_name: str, pairs, dims: int):
    """Mean cosine per recipe, for one candidate model."""
    from sentence_transformers import SentenceTransformer

    print(f"\n  loading {model_name} ...", flush=True)
    try:
        model = SentenceTransformer(model_name)
    except Exception as exc:
        print(f"    SKIPPED - could not load: {type(exc).__name__}: {exc}")
        return {}

    width = _dimension_of(model)
    if width != dims:
        print(f"    SKIPPED - produces {width} dims, catalogue is {dims}")
        return {}

    stored_shape = _upper_triangle([stored for _, stored in pairs])

    results = {}
    for recipe_name, recipe in TEXT_RECIPES.items():
        texts = [build_text(p, recipe) for p, _ in pairs]
        # normalize_embeddings=True because the stored vectors are unit
        # length. Cosine ignores magnitude anyway, so this only affects
        # how the numbers read - but it keeps them comparable.
        vectors = [
            list(map(float, v))
            for v in model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            )
        ]

        direct = sum(
            cosine(v, stored) for v, (_, stored) in zip(vectors, pairs)
        ) / len(pairs)

        # THE SECOND TEST, and the one that survives an unknown text
        # recipe. `direct` asks "is this the same vector?" - which needs
        # the model AND the exact string to match. This asks "is this the
        # same SPACE?": if products A and B sit close together in the
        # stored vectors, do they sit close together here too?
        #
        # That relationship is a property of the MODEL. Change the text
        # the pipeline embedded and every absolute vector moves, but the
        # catalogue's shape - which products resemble which - largely
        # holds. So a right-model/wrong-text candidate scores ~0.3 on
        # direct and still 0.7+ here, while a wrong model scores ~0 on
        # both. That is the distinction absolute cosine cannot draw.
        geometry = pearson(_upper_triangle(vectors), stored_shape)

        results[recipe_name] = {"direct": direct, "geometry": geometry}
    return results


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Identify the catalogue's embedding model by evidence."
    )
    parser.add_argument(
        "--sample", type=int, default=10,
        help="How many products to test against (default 10).",
    )
    parser.add_argument(
        "--model", action="append", dest="models",
        help="Test only this model. Repeatable.",
    )
    args = parser.parse_args()

    line("=")
    print("IDENTIFYING THE CATALOGUE'S EMBEDDING MODEL")
    line("=")
    print()
    print("CATALOGUE")
    line()
    pairs = await load_pairs(args.sample)
    if not pairs:
        return 1

    dims = len(pairs[0][1])
    candidates = args.models or CANDIDATE_MODELS

    print()
    line()
    print(f"TESTING {len(candidates)} MODELS x {len(TEXT_RECIPES)} TEXT RECIPES")
    line()
    print("  DIRECT   = mean cosine vs the stored vector. Same model AND")
    print("             same text reproduces itself at ~0.999.")
    print("  GEOMETRY = does this model agree with the stored vectors about")
    print("             which products resemble which? Survives an unknown")
    print("             text recipe, so it separates 'wrong model' from")
    print("             'right model, text I did not guess'.")

    grid = {}
    for name in candidates:
        scores = score_model(name, pairs, dims)
        if scores:
            grid[name] = scores
            bd = max(scores, key=lambda r: scores[r]["direct"])
            bg = max(scores, key=lambda r: scores[r]["geometry"])
            print(f"    direct   {scores[bd]['direct']:.4f}  via {bd}")
            print(f"    geometry {scores[bg]['geometry']:.4f}  via {bg}")

    print()
    line("=")
    print("RESULT")
    line("=")

    if not grid:
        print()
        print("  No candidate could even be loaded at the right width.")
        print("  The catalogue was embedded by something outside this list.")
        print("  Next step is the pipeline source in the backend repo.")
        return 1

    flat = sorted(
        (
            (cell["direct"], cell["geometry"], model, recipe)
            for model, scores in grid.items()
            for recipe, cell in scores.items()
        ),
        reverse=True,
    )
    by_geometry = sorted(
        (
            (cell["geometry"], cell["direct"], model, recipe)
            for model, scores in grid.items()
            for recipe, cell in scores.items()
        ),
        reverse=True,
    )

    print()
    print("  Top 5 by DIRECT match:")
    print()
    for direct, geom, model, recipe in flat[:5]:
        print(f"    direct {direct:.4f}  geometry {geom:+.4f}  {model}  [{recipe}]")

    print()
    print("  Top 5 by GEOMETRY:")
    print()
    for geom, direct, model, recipe in by_geometry[:5]:
        print(f"    geometry {geom:+.4f}  direct {direct:.4f}  {model}  [{recipe}]")

    top_score, _, top_model, top_recipe = flat[0]
    top_geom, top_geom_direct, geom_model, geom_recipe = by_geometry[0]
    print()

    if top_score >= CONFIRMED:
        print(f"  CONFIRMED: {top_model}")
        print(f"  Text recipe: {top_recipe}")
        print(f"  Mean cosine {top_score:.4f} against the stored vectors -")
        print("  that is reproduction, not resemblance. Set in .env:")
        print()
        print("    EMBEDDING_BACKEND=local")
        print(f"    LOCAL_EMBEDDING_MODEL={top_model}")
        print(f"    LOCAL_EMBEDDING_DIMENSIONS={dims}")
        print(f"    LEGACY_EMBEDDING_ASSUMED=local:{top_model}:{dims}")
        return 0

    if top_score >= SUGGESTIVE:
        print(f"  PARTIAL: {top_model} at {top_score:.4f} [{top_recipe}]")
        print("  High enough that the MODEL is probably right and the TEXT")
        print("  RECIPE is wrong. Add the recipe the pipeline actually used")
        print("  to TEXT_RECIPES and re-run.")
        print()
        print("  Do NOT configure anything on a partial match. Query vectors")
        print("  would sit near, but not in, the catalogue's space - which")
        print("  degrades ranking silently, the exact failure being avoided.")
        return 1

    print(f"  NOT IDENTIFIED by direct match. Best was {top_score:.4f}")
    print(f"  ({top_model}) - noise-level. None of these models, over any")
    print("  of these text recipes, produced the stored vectors.")
    print()

    # The runner-up must be a DIFFERENT MODEL: the same model over a
    # slightly different recipe is not an independent comparison.
    rival = next(
        ((g, m) for g, _, m, _ in by_geometry if m != geom_model),
        None,
    )
    margin = top_geom - rival[0] if rival else top_geom

    print(f"  Geometry: best {top_geom:+.4f} ({geom_model})")
    if rival:
        print(f"            next {rival[0]:+.4f} ({rival[1]})")
        print(f"            margin {margin:+.4f}"
              f"  (needs {GEOMETRY_MARGIN:+.2f} to mean anything)")
    print()

    if rival and margin < GEOMETRY_MARGIN:
        print("  GEOMETRY IS NOT EVIDENCE HERE. Two different models score")
        print("  within a rounding error of each other, so the correlation")
        print("  is measuring the catalogue, not the model - any encoder")
        print("  puts similar products near each other. High and")
        print("  undiscriminating is the floor, not a finding.")
    elif top_geom >= GEOMETRY_STRONG_ABS:
        print(f"  GEOMETRY POINTS SOMEWHERE: {top_geom:+.4f}, clear of the")
        print(f"  next model by {margin:+.4f}")
        print(f"    {geom_model}  [{geom_recipe}]")
        print()
        print("  This model agrees with the stored vectors about which")
        print("  products resemble which, while not reproducing them. The")
        print("  usual cause is the right model family over text this script")
        print("  does not know how to rebuild - an extra field, different")
        print("  cleaning, a prefix like 'query: ' or 'passage: '.")
        print()
        print("  Add the real recipe to TEXT_RECIPES and re-run. Do NOT")
        print("  configure on geometry alone: it says the space is similar,")
        print("  not that it is the same one, and 'similar' is exactly the")
        print("  failure that returns plausible nonsense.")
    else:
        print(f"  Geometry finds nothing either (best {top_geom:+.4f}).")
        print("  The stored space is unrelated to every candidate here.")

    print()
    print("  Free-text search cannot be made safe from this. Either get the")
    print("  pipeline source from the backend repo, or take the upgrade")
    print("  route (pipeline writes 1536 + tag, then backfill).")
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
