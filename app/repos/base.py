"""
WHAT:
    Tiny shared utility used by every repository file in Phase 4.

WHY THIS APPROACH:
    Every repo needs to safely convert a string ID (from a verified JWT,
    or a chat-extracted order/product ID) into MongoDB's ObjectId type.
    A malformed ID should never crash the app - it should just mean
    "not found," handled the same way everywhere, rather than each of
    the 9 repo files reinventing this slightly differently.

FLOW:
    Every repo function calls to_object_id() before querying MongoDB.

LOGIC:
    If the string isn't a valid 24-character hex ObjectId, we return
    None instead of raising - callers treat None as "no match," which
    naturally produces the graceful "not found" behavior Phase 4.3
    requires, with no special-case error handling needed at every call site.

MECHANISM:
    bson.ObjectId raises InvalidId on bad input - we catch that
    specific exception (and TypeError, for non-string input) and
    convert it into a clean None return instead of letting it propagate.
"""

import structlog
from bson import ObjectId
from bson.errors import InvalidId

logger = structlog.get_logger()


def to_object_id(value: str) -> ObjectId | None:
    try:
        return ObjectId(value)
    except (InvalidId, TypeError):
        # Log at warning level - an invalid ID reaching this point
        # usually means either bad input upstream or a bug worth
        # investigating, but it's never a reason to crash the chatbot.
        logger.warning("invalid_object_id", value=str(value))
        return None