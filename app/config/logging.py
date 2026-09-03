"""
WHAT:
    This file configures structlog (our structured logging library) and
    defines a custom "processor" — a step that runs on every single log
    call — which automatically hides sensitive field values before the
    log line is ever printed or written anywhere.

WHY THIS APPROACH:
    We can't rely on every developer (including future-you, in a hurry,
    six months from now) to remember "don't log the password field" every
    single time. Instead, we build ONE central rule that applies to every
    log call automatically, everywhere in the app, forever — so forgetting
    is no longer possible.

FLOW:
    1. App starts up and calls configure_logging() once (we'll wire this
       into the app's startup in a later phase).
    2. Every other file then does: `logger = structlog.get_logger()` and
       calls things like `logger.info("order_fetched", order_id=..., ...)`.
    3. Before that log line is actually printed, our redact_sensitive_data
       processor runs automatically and scrubs any risky field names.
    4. The final output is one line of JSON — safe to grep, safe to ship
       to a log dashboard later (Phase 13).

LOGIC:
    - SENSITIVE_KEYS is our "blocklist" for logging specifically (note:
      this is DIFFERENT from Phase 3's allowlist, which controls what the
      chatbot returns to USERS. This one controls what we as developers
      are allowed to see in our OWN logs/terminal/dashboard).
    - Matching is case-insensitive and matches on PARTIAL key names too
      (e.g. "userPhone" or "phone_number" both get caught by "phone"),
      because real-world field names vary and we'd rather over-redact
      than under-redact.
    - SAFE_KEYS is the narrow exception to that greediness: a short set of
      EXACT key names that contain a blocklisted substring but hold no
      secret (notably "prompt_tokens", which contains "token"). Exact
      matching keeps the exception from widening into a loophole.
    - Redaction is recursive through BOTH dicts and sequences: a sensitive
      key buried in a nested dict, or inside a list of documents, is
      redacted just the same as a top-level field. Lists matter as much as
      dicts here because secrets usually travel in collections (a list of
      users, an order's items, a repo's find().to_list() result).
    - Dataclasses and namedtuples are walked by FIELD NAME as well, since
      those names are what every rule here judges against. Not
      hypothetical: Provider in agent/llm_client.py is a dataclass holding
      api_key. Arbitrary objects are deliberately left alone — see the
      note on the dataclass branch for why the line is drawn there.
    - MAX_REDACTION_DEPTH bounds all of the above, so a reference cycle
      can never turn a log call into a RecursionError.

MECHANISM:
    - structlog processors are just plain functions that receive the log
      event dict and return a (possibly modified) version of it — they
      run in a chain, in order, before the final render step.
    - We put our redaction processor EARLY in that chain, so nothing
      downstream (including the final JSON renderer) ever sees the real
      value.
"""

import dataclasses
import logging
import sys

import structlog

# Partial, case-insensitive matches. Add to this list any time a new
# sensitive field name shows up anywhere in the schema (see Phase 3.3's
# allowlist for the current known set: password, refreshToken,
# bankDetails, payoutDestination, phone, email).
SENSITIVE_KEYS = (
    "password",
    "token",
    "secret",
    "bank",
    "payout",
    "phone",
    "email",
    "upi",
    "account_number",
    "accountnumber",
    "ifsc",
    # Credentials. Added alongside object introspection below: Provider
    # (llm_client.py) is a dataclass whose fields include api_key, so the
    # moment we started reading object fields by name, "api_key" became a
    # name that could reach a log line. Note the entry is "api_key"/
    # "apikey" and NOT bare "key" - "key" would swallow the legitimate
    # searchKeywords field from the products collection.
    "api_key",
    "apikey",
    "authorization",
    "credential",
)

# How deep to follow a nested structure before giving up.
#
# WHY A LIMIT EXISTS AT ALL: dicts and lists nest a handful of levels in
# practice, but OBJECT graphs hold back-references (a child pointing at
# its parent), and a cycle would recurse until Python raised
# RecursionError - from inside a log call, which is the worst place to
# raise. 8 is far deeper than any real logged payload here.
MAX_REDACTION_DEPTH = 8

# Keys that CONTAIN a blocklisted substring but carry no sensitive value.
#
# WHY THIS EXISTS: the substring rule above is deliberately greedy, and
# that is right for "userPhone"/"phone_number" — but "token" also lives
# inside "prompt_tokens", so the LLM cost telemetry the orchestrator logs
# on every single call was being written as "***REDACTED***". The one
# metric this project's whole token-budget design is tuned against was
# invisible in its own logs.
#
# Matched as EXACT, lowercased keys, checked BEFORE the blocklist — so
# the over-redact bias is untouched for anything not listed here
# verbatim. "auth_token" is still redacted; so is "prompt_tokens_raw".
# These four are the OpenAI-compatible `usage` field names plus our own
# max_tokens config value, i.e. counts of tokens, never a token itself.
SAFE_KEYS = frozenset(
    {
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "max_tokens",
    }
)


def _is_sensitive_key(key: str) -> bool:
    # Lowercase comparison so "Password", "PASSWORD", "userPassword" all match.
    key_lower = key.lower()
    if key_lower in SAFE_KEYS:
        return False
    return any(sensitive in key_lower for sensitive in SENSITIVE_KEYS)


def _redact_value(value, depth: int = 0):
    if depth >= MAX_REDACTION_DEPTH:
        # Deeper than any real payload - stop rather than risk a cycle.
        return "***TRUNCATED***"

    # If the value itself is a nested dict, recurse into it and redact
    # any sensitive keys found inside, instead of only checking top level.
    if isinstance(value, dict):
        return {
            k: (
                "***REDACTED***"
                if _is_sensitive_key(k)
                else _redact_value(v, depth + 1)
            )
            for k, v in value.items()
        }
    # A namedtuple is a tuple, but its ELEMENTS are named - and iterating
    # it as a plain sequence throws those names away, so a field called
    # "password" would sail through as an unlabelled string. _asdict()
    # recovers the names, which is the only thing that makes the values
    # judgeable at all. Checked before the sequence branch below, since
    # it would otherwise match there first.
    if isinstance(value, tuple) and hasattr(value, "_asdict"):
        return _redact_value(value._asdict(), depth)

    # Dataclass instances are DECLARED data holders - exactly the shape
    # that carries a secret in a named field. Provider (llm_client.py)
    # holds api_key; reading its fields by name is what lets the rule
    # above catch it. dataclasses.asdict() is deliberately NOT used: it
    # recurses and deep-copies on its own, producing a finished structure
    # that would never pass through this function's redaction at all.
    #
    # SCOPED TO DATACLASSES ON PURPOSE. Introspecting every object with a
    # __dict__ would mean walking a Motor client or an httpx.Response the
    # moment one reached a log call - enormous, useless output, and a far
    # richer source of reference cycles. A dataclass is a deliberate
    # declaration that "this is data"; anything else is left as-is and
    # renders through its own repr, as it does today.
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = {f.name: getattr(value, f.name, None) for f in dataclasses.fields(value)}
        return _redact_value(fields, depth)

    # Lists and tuples must be walked too. WHY: a secret is very often
    # held in a COLLECTION of documents rather than one - a list of users,
    # an order's items, a repo function returning find().to_list(). With
    # dict-only recursion, logging [{"password": ...}] wrote the password
    # out in full, because the list itself is not a dict and was returned
    # untouched. Any sequence is therefore a path to a nested dict.
    #
    # str/bytes are deliberately NOT included: they are sequences too, and
    # iterating them would shred every string into characters. The common
    # real case here is a plain list of strings (providers=["groq", ...]),
    # which passes through unchanged because a str hits neither branch.
    if isinstance(value, (list, tuple)):
        # Rebuilt as a plain list/tuple, NOT type(value)(...), which
        # assumes a constructor taking an iterable and is not true of
        # every sequence subclass. This processor runs inside every log
        # call - it degrades an exotic type to a plain one rather than
        # ever raising from a log line.
        redacted = [_redact_value(item, depth + 1) for item in value]
        return tuple(redacted) if isinstance(value, tuple) else redacted
    return value


def redact_sensitive_data(logger, method_name, event_dict):
    """
    A structlog processor: runs on every log call. Receives the full
    dict of everything passed to the log call and returns a cleaned copy.
    """
    cleaned = {}
    for key, value in event_dict.items():
        if _is_sensitive_key(key):
            cleaned[key] = "***REDACTED***"
        else:
            cleaned[key] = _redact_value(value)
    return cleaned


def configure_logging() -> None:
    """
    Call this ONCE, at app startup. Sets up structlog to output one line
    of JSON per log call, passing every call through our redaction step
    first.
    """
    # Configure Python's built-in logging as the final output destination
    # (structlog builds on top of it rather than replacing it entirely).
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.INFO,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            redact_sensitive_data,  # our custom redaction step
            structlog.processors.JSONRenderer(),  # final output: one JSON line
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )