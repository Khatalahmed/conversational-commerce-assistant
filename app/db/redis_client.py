"""
WHAT:
    The ONE Redis connection the app shares, and the ONE place that
    decides whether Redis is available at all.

WHY THIS EXISTS:
    Two features need the same connection for the same reason: they hold
    state that must be identical across every worker process.
    Conversation memory (app/memory/session_store.py) and the /chat rate
    limiter (app/security/rate_limit.py) both degrade to per-process
    state without it, and both degrade in ways that are invisible until
    there is more than one worker.

    They were not both resolving it before - the session store had a
    private client and the limiter had none at all. Resolving it twice
    would mean two connection pools, two copies of the Upstash trap
    below, and two different possible answers to "is Redis up?" inside
    one request.

    It sits in db/ next to the Mongo connection because it is the same
    kind of thing: process-wide infrastructure, opened once and borrowed
    by everything else.

WHY IT IS STILL OPTIONAL:
    A POC must run on nothing but a MongoDB URI. An unset REDIS_URL, or
    an unreachable server, degrades the callers rather than refusing to
    start - turning a cache outage into a total outage is a worse
    failure than the one being avoided. Each caller documents what it
    loses, and both log loudly.

FLOW:
    get_redis() resolves and caches the client (or None) on first use.
    close_redis() at app shutdown, mirroring close_mongo_connection().

MECHANISM:
    Lazy singleton, the same pattern as db/connection.py and
    agent/llm_client.py. `_checked` is tracked SEPARATELY from `_client`
    because None is a real, cacheable answer here - "we looked, there is
    no Redis" must not re-probe on every single request.
"""

import structlog

from app.config.settings import get_settings

logger = structlog.get_logger()

_client = None
_checked = False


async def get_redis():
    """The shared client, or None when Redis is not available.

    Resolved ONCE and cached, including the negative answer. A failure
    here is deliberately not fatal: the callers degrade to per-process
    state, which is worse than Redis and far better than a service that
    will not start.
    """
    global _client, _checked
    if _checked:
        return _client

    _checked = True
    url = get_settings().redis_url
    if not url:
        logger.warning(
            "redis_not_configured",
            reason="no REDIS_URL configured",
            consequence="conversation history is lost on restart and not shared "
                        "across workers; the /chat rate limit is enforced per "
                        "worker, so the effective allowance is workers x limit",
        )
        return None

    # A VERY EASY MISTAKE, CAUGHT EXPLICITLY. Upstash shows its REST
    # credentials most prominently - UPSTASH_REDIS_REST_URL is an
    # https:// endpoint for their HTTP API, and redis-py does not speak
    # it. Handed one, the driver fails with something unhelpful about
    # the connection, and the real problem (wrong endpoint entirely, and
    # the REST token is not the Redis password) is nowhere in the error.
    if url.startswith(("http://", "https://")):
        logger.error(
            "redis_url_looks_like_a_rest_endpoint",
            hint="REDIS_URL must be the Redis-protocol URL (rediss://...:6379), "
                 "not UPSTASH_REDIS_REST_URL. Find it under Connect -> Redis in "
                 "the Upstash console; the REST token is NOT the Redis password.",
            consequence="falling back to per-process state",
        )
        return None

    try:
        import redis.asyncio as redis

        client = redis.from_url(url, decode_responses=True)
        await client.ping()
        _client = client
        logger.info("redis_connected")
    except Exception as exc:
        logger.error(
            "redis_unavailable",
            error=str(exc)[:200],
            consequence="falling back to per-process state",
        )
        _client = None
    return _client


async def close_redis() -> None:
    """Call once at app shutdown, mirroring close_mongo_connection()."""
    global _client, _checked
    if _client is not None:
        await _client.aclose()
        logger.info("redis_closed")
    _client = None
    _checked = False


def reset_for_tests() -> None:
    """Drops the cached client and forces the next get_redis() to
    re-resolve. Tests that want a fake client patch _client and _checked
    directly afterwards."""
    global _client, _checked
    _client = None
    _checked = False
