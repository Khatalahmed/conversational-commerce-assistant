"""
WHAT:
    This file owns the ONE MongoDB connection ("client") the entire app
    will use. It creates that connection once when the app starts, and
    every other file (repositories in Phase 4) will borrow the database
    handle from here instead of creating their own separate connections.

WHY THIS APPROACH:
    Opening a new connection for every single database query is slow and
    wasteful — MongoDB connections take real time to establish (network
    handshake, auth, etc.). A single shared, reused connection pool is
    the standard, correct way to talk to a database in a real backend
    that will serve many users/requests concurrently.

FLOW:
    1. App starts up (this will be wired into Phase 9's API startup).
    2. connect_to_mongo() is called once — opens the client, and
       immediately runs a "ping" to confirm the connection actually
       works (fails loudly here if it doesn't, rather than failing
       later on a random user's first query).
    3. Every repository function (Phase 4) calls get_database() to get
       the same shared database handle.
    4. On shutdown, close_mongo_connection() cleanly releases the
       connection.

LOGIC:
    We use `motor`, the ASYNC MongoDB driver, not the plain `pymongo` we
    used a moment ago just for the manual read-only test. Motor is what
    the real app needs, because a chatbot serving many users at once
    must not let one slow database query freeze every other user's
    conversation — async allows the app to handle other requests while
    waiting on a query.

MECHANISM:
    - AsyncIOMotorClient is Motor's connection pool object — created
      once, reused everywhere.
    - We read the database name from settings rather than hardcoding
      "marketplace," so this stays correct even when we eventually swap to the
      production connection string (Phase 2, Step H of the replication
      plan) which might use a different configured default database name.
    - A module-level variable (_client) holds the single shared instance;
      get_database() raises a clear error if someone calls it before
      connect_to_mongo() has run, instead of silently returning nothing.
"""

import structlog
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.config.settings import get_settings

logger = structlog.get_logger()

# Module-level holders for the single shared client/database instance.
# Starts as None; get set once by connect_to_mongo() at app startup.
_client: AsyncIOMotorClient | None = None
_database: AsyncIOMotorDatabase | None = None

# The database name inside the cluster. Still not read from the URI -
# the URI doesn't reliably include it - but no longer hardcoded either:
# it comes from settings so a DEMO database can be swapped in via .env.
#
# WHY THAT MATTERS. Reading real customer data in tests is harmless -
# it never leaves Atlas. Sending it to an LLM is not: the free tiers
# state that submitted content may be used to improve their products
# and may be seen by human reviewers. Pointing LLM-facing work at a
# seeded database is what separates those two cases.
DEFAULT_DATABASE_NAME = "marketplace"


async def connect_to_mongo() -> None:
    """
    Call this ONCE, at app startup. Opens the shared connection and
    verifies it actually works before letting the app continue.
    """
    global _client, _database

    settings = get_settings()

    # Create the connection pool. This doesn't actually connect yet —
    # Motor connects lazily, on first real use — which is why we
    # immediately follow this with a ping to force a real connection
    # attempt right now, at startup, not later during a user's request.
    _client = AsyncIOMotorClient(settings.mongodb_uri)
    _database = _client[settings.mongodb_database]

    try:
        # "ping" is a lightweight built-in MongoDB command used
        # specifically to test connectivity without touching real data.
        await _client.admin.command("ping")
        logger.info("mongodb_connected", database=settings.mongodb_database)
    except Exception as exc:
        logger.error("mongodb_connection_failed", error=str(exc))
        # Re-raise so the app refuses to start on a broken DB connection,
        # rather than starting up "successfully" and failing later.
        raise


async def close_mongo_connection() -> None:
    """
    Call this ONCE, at app shutdown. Cleanly releases the connection.
    """
    global _client
    if _client is not None:
        _client.close()
        logger.info("mongodb_connection_closed")


def get_database() -> AsyncIOMotorDatabase:
    """
    Every repository function (Phase 4) calls this to get the shared
    database handle. Raises clearly if called before startup connected.
    """
    if _database is None:
        raise RuntimeError(
            "Database not connected yet. connect_to_mongo() must be "
            "called at app startup before any repository function runs."
        )
    return _database


async def check_database_health() -> dict:
    """
    On-demand check: is the database actually reachable right now?
    Returns a small dict, used by the /health endpoint (Phase 9) and by
    cloud monitoring (Phase 11).

    DELIBERATELY VAGUE. /health is unauthenticated - anyone who can
    reach the service can read whatever this returns. A driver
    exception carries the cluster hostname, replica-set members, and
    sometimes the username from the connection string, so the detail
    goes to the LOGS (where operators can see it) and only a status
    goes over the wire. The database NAME is withheld for the same
    reason: it tells a stranger nothing useful about liveness.
    """
    if _client is None:
        logger.warning("health_check_before_connect")
        return {"status": "disconnected"}

    try:
        await _client.admin.command("ping")
        return {"status": "connected"}
    except Exception as exc:
        logger.error("health_check_failed", error=str(exc))
        return {"status": "error"}