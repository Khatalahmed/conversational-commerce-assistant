"""
WHAT:
    The last checkpoint before any database document reaches a chatbot
    response. Independently re-checks every field against the Step 3.3
    allowlist and strips anything not explicitly listed — regardless of
    how the document was originally fetched.

WHY THIS APPROACH:
    "Defense in depth" - security shouldn't rely on one perfect layer.
    Step 3.3's allowlist controls what we ASK MongoDB for (the query
    projection). This file doesn't trust that request was built
    correctly - it re-verifies the actual data independently, so even a
    future bug (wrong projection, a repo function queried without one
    "just to debug," a typo) still can't leak a forbidden field.

FLOW:
    1. A repo function (Phase 4) fetches a document from MongoDB.
    2. Before returning it to feature logic (Phase 5), it passes the
       document through sanitize_document() here.
    3. Anything not on the allowlist for that collection is silently
       removed - and logged, so a developer notices during testing if
       something unexpected got stripped (a sign the repo's projection
       has a bug worth fixing, even though the leak itself was still
       prevented).

LOGIC:
    Allowlist fields can be "flat" (e.g. "status" - keep this field's
    entire value, whatever shape it is) or "dotted/nested" (e.g.
    "invoice.url" - only keep that specific sub-field, strip any other
    sibling field inside "invoice" that isn't also explicitly listed).
    "_id" is always kept automatically - it's never sensitive and is
    needed for reference, matching MongoDB's own default behavior.

MECHANISM:
    We first convert the flat allowlist (a list of strings, some with
    dots) into a nested tree structure, then walk the real document
    against that tree recursively - keeping only what the tree permits
    at each level.
"""

import structlog

from app.security.field_allowlist import COLLECTION_ALLOWLISTS

logger = structlog.get_logger()


def _build_allowlist_tree(fields: list[str]) -> dict:
    """
    Converts a flat list like ["status", "invoice.url",
    "invoice.generatedAt"] into a nested tree:
        {"status": True, "invoice": {"url": True, "generatedAt": True}}
    `True` means "keep this field's value completely, as-is."
    A nested dict means "only keep these specific sub-keys inside."
    """
    tree: dict = {}
    for field in fields:
        top, _, rest = field.partition(".")
        if not rest:
            # No nesting for this field - fully allowed, whole value.
            if tree.get(top) is not True:
                tree.setdefault(top, True)
            continue
        if tree.get(top) is True:
            continue  # already fully open, no need to restrict further
        subtree = tree.setdefault(top, {})
        if not isinstance(subtree, dict):
            subtree = {}
            tree[top] = subtree
        child = _build_allowlist_tree([rest])
        subtree.update(child)
    return tree


def _sanitize_value(value, tree_node):
    """Applies one tree node to one value, recursively."""
    if tree_node is True:
        return value
    if value is None:
        return None
    if not isinstance(value, dict):
        # Allowlist expected a nested object here but got something
        # else - drop it defensively rather than guess, and log it so
        # it's visible during development, not a silent surprise.
        logger.warning(
            "sanitizer_unexpected_shape",
            expected="dict",
            got=type(value).__name__,
        )
        return None
    cleaned = {}
    for key, sub_node in tree_node.items():
        if key in value:
            cleaned[key] = _sanitize_value(value[key], sub_node)
    return cleaned


def sanitize_document(collection_name: str, document: dict | None) -> dict | None:
    """
    THE defense-in-depth safety net. Call this on every document before
    it leaves the data access layer, no exceptions.
    """
    if document is None:
        return None

    fields = COLLECTION_ALLOWLISTS.get(collection_name)
    if fields is None:
        raise ValueError(
            f"No allowlist defined for collection '{collection_name}'. "
            f"Refusing to return unrecognized-collection data."
        )

    tree = _build_allowlist_tree(fields)
    cleaned = {}

    if "_id" in document:
        cleaned["_id"] = document["_id"]

    for key, sub_node in tree.items():
        if key in document:
            cleaned[key] = _sanitize_value(document[key], sub_node)

    dropped = set(document.keys()) - set(cleaned.keys())
    if dropped:
        logger.warning(
            "sanitizer_stripped_fields",
            collection=collection_name,
            dropped_fields=sorted(dropped),
        )

    return cleaned


def sanitize_documents(collection_name: str, documents: list[dict]) -> list[dict]:
    """Convenience wrapper for a list of documents (e.g. search results)."""
    return [sanitize_document(collection_name, doc) for doc in documents]