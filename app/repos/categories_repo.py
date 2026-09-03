"""
WHAT:
    Lists all real categories and subcategories from the `categories`
    collection. Fills the gap flagged in the Phase 5.2 spec: grounds
    Phase 6/7's LLM category extraction against REAL names from your
    data, instead of letting it guess ("Jeans" vs the actual
    "jeans-trousers").

WHY THIS APPROACH:
    This is reference/lookup data, not a live search - small enough
    that a single "get everything" function is the right shape, rather
    than building filtering into it.

DELIBERATELY NOT A TOOL, AND NOT DEAD EITHER:
    Nothing in TOOL_REGISTRY points here, which looks like an oversight
    and is not. The assistant does need real category names - it would
    otherwise search for "Jeans" against a catalogue that says
    "jeans-trousers" - but it gets them from
    products_repo.get_distinct_categories(), whose result is baked
    straight into the system prompt once per conversation. Grounding the
    model should always have is worth a string in the prompt; it is not
    worth a tool round-trip, which costs a full prompt to learn
    something we could have told it for free (see the note in
    agent/tools.py).

    The two differ in SOURCE, which is why both exist: this reads the
    `categories` collection - the taxonomy as the marketplace team defines it
    - while get_distinct_categories reads the values products actually
    carry. When those disagree, the products win, because they are what
    a search can match. scripts/check_categories.py prints this one so
    the disagreement is visible.

FLOW:
    Called by scripts/check_categories.py. Not reachable from chat.

MECHANISM:
    Same two-layer safety pattern as every repo, though categories are
    fully public reference data with no sensitive fields at all.
"""

import structlog

from app.db.connection import get_database
from app.security.field_allowlist import get_projection
from app.security.sanitizer import sanitize_documents

logger = structlog.get_logger()

COLLECTION = "categories"


async def get_all_categories() -> list[dict]:
    """Returns every category with its subcategory names."""
    db = get_database()
    projection = get_projection(COLLECTION)
    cursor = db[COLLECTION].find({}, projection)
    docs = await cursor.to_list(length=None)
    docs = sanitize_documents(COLLECTION, docs)

    return [
        {
            "name": doc.get("name"),
            "subCategories": [
                sc.get("name") for sc in (doc.get("subCategories") or [])
            ],
        }
        for doc in docs
    ]