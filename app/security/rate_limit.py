"""
WHAT:
    A per-user request limiter for /chat. Counts each user's requests
    over a rolling window and refuses further ones once they exceed the
    configured allowance. Shared across workers via Redis when one is
    configured, per-process otherwise.

WHY THIS APPROACH:
    /chat is a public endpoint that spends real money on every call.
    Nothing currently stops one client - a retry loop, a leaked token, a
    buggy release of the mobile app - from issuing requests as fast as
    the network allows. On the free LLM tier that drains the shared
    quota in seconds, and EVERY other user then gets "I'm handling a lot
    of requests right now" until it resets.

    KEYED ON THE VERIFIED user_id, NOT ON IP. Mobile traffic arrives
    through carrier NAT, so thousands of unrelated users can share one
    address - an IP limit would throttle a whole network segment while
    still letting a single abusive account spread across addresses. The
    user_id comes from a signature-verified JWT (see security/auth.py),
    so it cannot be spoofed the way a header can.

WHY IT IS SHARED STATE, AND WHY THAT WAS A BUG:
    The counts used to live in a module-level dict, which is per-PROCESS.
    Under `uvicorn --workers 4` that is four independent limiters, each
    granting the full allowance, so the real limit was 4 x 20 and the
    configured number meant nothing. Worse, it was silently wrong: every
    test passes on one worker, and the gap only opens in the deployment
    where the limiter actually matters.

    Redis fixes it because a sliding window is exactly a sorted set -
    timestamps as scores, expired ones dropped by range, the count read
    back. Every worker reads and writes the same key, so the allowance
    is the allowance.

WHAT THIS DOES NOT DO:
    It bounds ONE user, not total load: twenty users each within their
    allowance can still exhaust the LLM quota between them, because that
    quota is per-service, not per-user. That case is already survivable
    - the orchestrator answers with FALLBACK_BUSY rather than failing -
    and fixing it properly means a global concurrency budget or a paid
    tier, not a bigger limiter. This closes the "one client burns it
    all" hole, which is the one an attacker or a bug actually reaches
    for.

FLOW:
    api/dependencies.py awaits this AFTER get_current_user_id, so the
    limiter always sees a verified identity. Unauthenticated requests
    never reach it - they are rejected earlier and cost nothing but a
    signature check.

LOGIC:
    A sliding window, not a fixed one. A fixed window resets on a clock
    boundary, which lets a client send a full allowance at 11:59:59 and
    another at 12:00:00 - double the intended rate, at the worst
    possible moment. Keeping the timestamps and expiring them
    individually costs a few bytes per request and has no such edge.

    WHEN REDIS IS UNREACHABLE the request is checked against the
    in-process table instead of being refused. Fail-open-to-per-process,
    not fail-closed: a cache outage would otherwise take the whole
    assistant down, and per-worker limiting is exactly what this file
    did before - degraded, not absent. Same reasoning the session store
    already applies to history.

MECHANISM:
    Redis path: one transaction per request against a sorted set -
    prune by score, add this request, read the count, read the oldest,
    refresh the TTL. If the count comes back over the allowance, the
    entry just added is removed again, so a REFUSED request never
    consumes a slot (otherwise a client hammering the endpoint pushes
    its own recovery further away - a lockout, not a limit).

    Adding first and compensating afterwards, rather than deciding first
    and adding after, is what makes the decision atomic without a Lua
    script: two concurrent requests can never both be told yes for the
    same free slot. The cost is that they can both be told no when only
    one should have been - over-strict during a burst, never
    over-permissive, which is the direction that matters here.

    Timestamps come from the APP's clock, not Redis TIME, so that the
    frozen-clock tests govern both backends identically. Worker clocks
    are NTP-synced in practice and a few milliseconds of skew is nothing
    against a 60-second window.

    In-process path: one deque of timestamps per user, oldest evicted on
    read. Same behaviour, one process wide.

    Rejections carry Retry-After either way, computed from when the
    oldest request in the window expires, so a client is told exactly
    how long to wait instead of guessing.
"""

from collections import deque
from datetime import datetime, timedelta, timezone
from secrets import token_hex

import structlog

from app.config.settings import get_settings
from app.db.redis_client import get_redis

logger = structlog.get_logger()

# Bounds how many users' histories are held at once. Applies only to the
# in-process table: Redis expires each key itself, so an abandoned user
# there costs nothing once their window lapses.
MAX_TRACKED_USERS = 10_000

REDIS_KEY_PREFIX = "assistant:ratelimit:"

_requests: dict[str, deque] = {}


class RateLimitExceeded(Exception):
    """Raised when a user is over their allowance. Carries how long to
    wait, so the caller can send a truthful Retry-After."""

    def __init__(self, retry_after_seconds: int):
        self.retry_after_seconds = retry_after_seconds
        super().__init__(f"Rate limit exceeded, retry in {retry_after_seconds}s")


def _refuse(user_id: str, used: int, allowance: int, retry_after: int) -> None:
    logger.warning(
        "rate_limit_exceeded",
        user_id=user_id,
        requests_in_window=used,
        allowance=allowance,
        retry_after_seconds=retry_after,
    )
    raise RateLimitExceeded(retry_after)


# -- In-process fallback ----------------------------------------------

def _prune(history: deque, cutoff: datetime) -> None:
    """Drops timestamps that have left the window."""
    while history and history[0] <= cutoff:
        history.popleft()


def _evict_idle_users(now: datetime, window: timedelta) -> None:
    """Bounds memory. Runs only when the table is over its cap, since
    the common path should not pay to walk every tracked user."""
    if len(_requests) <= MAX_TRACKED_USERS:
        return

    cutoff = now - window
    idle = [user for user, history in _requests.items() if not history or history[-1] <= cutoff]
    for user in idle:
        del _requests[user]

    # Still over? Drop the least recently active - they are closest to
    # expiring anyway, and a limiter that grows without bound is a
    # denial of service against ourselves.
    overflow = len(_requests) - MAX_TRACKED_USERS
    if overflow > 0:
        oldest = sorted(_requests, key=lambda u: _requests[u][-1])[:overflow]
        for user in oldest:
            del _requests[user]

    logger.info("rate_limit_table_pruned", idle=len(idle), remaining=len(_requests))


def _check_in_process(
    user_id: str, allowance: int, window: timedelta, now: datetime
) -> None:
    history = _requests.setdefault(user_id, deque())
    _prune(history, now - window)

    if len(history) >= allowance:
        # Oldest request in the window decides when a slot frees up.
        retry_after = max(1, int((history[0] + window - now).total_seconds()) + 1)
        _refuse(user_id, len(history), allowance, retry_after)

    history.append(now)
    _evict_idle_users(now, window)


# -- Redis-backed sliding window --------------------------------------

async def _check_redis(
    client, user_id: str, allowance: int, window_ms: int, now_ms: int
) -> None:
    key = REDIS_KEY_PREFIX + user_id
    # Unique per request: a sorted set holds each member once, so two
    # requests sharing a millisecond would otherwise collapse into a
    # single entry and quietly hand out a free slot.
    member = f"{now_ms}-{token_hex(6)}"

    async with client.pipeline(transaction=True) as pipe:
        pipe.zremrangebyscore(key, "-inf", now_ms - window_ms)
        pipe.zadd(key, {member: now_ms})
        pipe.zcard(key)
        pipe.zrange(key, 0, 0, withscores=True)
        # Refreshed on every request, so an abandoned key disappears one
        # window after its owner's last call rather than living forever.
        pipe.pexpire(key, window_ms)
        _, _, used, oldest, _ = await pipe.execute()

    if used <= allowance:
        return

    # Over the line - take back the slot this request just claimed.
    try:
        await client.zrem(key, member)
    except Exception as exc:
        # Left behind, this entry slides out of the window on its own
        # within one window. Worth a log, not worth failing a request.
        logger.warning("rate_limit_slot_not_reclaimed", error=str(exc)[:200])

    oldest_ms = oldest[0][1] if oldest else now_ms
    retry_after = max(1, int((oldest_ms + window_ms - now_ms) / 1000) + 1)
    _refuse(user_id, used - 1, allowance, retry_after)


# -- Public interface -------------------------------------------------

async def check_rate_limit(user_id: str) -> None:
    """Records one request for this user, or raises RateLimitExceeded.

    Call once per request, AFTER the JWT has been verified.
    """
    settings = get_settings()
    allowance = settings.chat_rate_limit_requests

    # 0 disables the limiter outright - used by tests, and an escape
    # hatch if it ever needs turning off in production without a deploy.
    if allowance <= 0:
        return

    window = timedelta(seconds=settings.chat_rate_limit_window_seconds)
    now = datetime.now(timezone.utc)

    client = await get_redis()
    if client is not None:
        try:
            await _check_redis(
                client,
                user_id,
                allowance,
                int(window.total_seconds() * 1000),
                int(now.timestamp() * 1000),
            )
        except RateLimitExceeded:
            raise
        except Exception as exc:
            # See the fail-open note in the module docstring: the
            # in-process table below still bounds this client per worker.
            logger.error(
                "rate_limit_redis_failed",
                error=str(exc)[:200],
                consequence="falling back to the per-process limiter for this request",
            )
        else:
            return

    _check_in_process(user_id, allowance, window, now)


def reset() -> None:
    """Clears the in-process table. For tests only - it deliberately
    does not touch Redis, which is shared with whatever else is pointed
    at that server."""
    _requests.clear()


def tracked_users() -> int:
    """How many users the IN-PROCESS table holds. Exposed for tests and
    health checks; under Redis the counts live in Redis and this reads
    0, which is the honest answer for this process."""
    return len(_requests)
