"""
Checks whether conversation memory is actually backed by Redis, and
proves it with a real write/read round trip.

    uv run python scripts/check_redis.py

Prints a diagnosis rather than a stack trace: the common failures here
(no URL, the Upstash REST URL pasted by mistake, a wrong password, TLS)
all look similar from the driver and different from each other in what
you have to do about them.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config.settings import get_settings  # noqa: E402
from app.db.redis_client import close_redis, get_redis  # noqa: E402
from app.memory import session_store  # noqa: E402

USER = "check-redis-script"
SESSION = "round-trip"



def _host_of(uri: str) -> str:
    """The HOSTNAME only - no credentials, no path, no query string.

    The query string matters here. A URI like

        mongodb+srv://u:p@cluster0.example.net?appName=the marketplace-Semantic-Search

    has no slash before "?", so splitting on "/" alone leaves the appName
    glued to the host - and a guard looking for the staging cluster's name
    then matched a COMPLETELY DIFFERENT cluster whose appName happened to
    mention it. The guard refused a legitimate destination and blamed the
    user. Strip "?" as well as "/".

    THE SCHEME MUST GO FIRST, and used not to. Splitting on "@" only
    removes credentials when there ARE credentials; "redis://localhost:6379"
    has none, so the first "/" reached belonged to "://" and the host came
    back as "redis:". Every credential-less URI reported its own scheme as
    its hostname - which made the local-Redis branch below unreachable and
    printed an Upstash TLS warning about localhost.
    """
    without_scheme = uri.split("://")[-1]
    return without_scheme.split("@")[-1].split("/")[0].split("?")[0].lower()

def line() -> None:
    print("-" * 62)


async def main() -> None:
    settings = get_settings()
    url = settings.redis_url

    line()
    print("REDIS CONFIGURATION")
    line()

    if not url:
        print("  REDIS_URL: not set")
        print()
        print("  Conversation memory is using the IN-PROCESS fallback.")
        print("  That works, but history is erased on every restart or")
        print("  deploy, and is not shared between workers.")
        print()
        print("  Add this to .env:")
        print("    REDIS_URL=rediss://default:PASSWORD@YOUR-DB.upstash.io:6379")
        return

    scheme = url.split("://")[0] if "://" in url else "(none)"
    # Split on "@" so a password is never printed. A URL without "@"
    # has no credentials in it, so showing it whole is safe.
    host = _host_of(url)
    is_local = host.startswith(("localhost", "127.0.0.1", "::1", "host.docker.internal"))
    print(f"  scheme: {scheme}://")
    print(f"  host:   {host}")
    print()

    if is_local:
        print("  !  This is a LOCAL Redis, on your own machine.")
        print()
        print("     Good enough for development - it survives uvicorn")
        print("     reloads, which is the problem it was added to fix.")
        print()
        print("     It will NOT work once deployed: a cloud host has no")
        print("     Redis on its localhost, so the app would silently fall")
        print("     back to in-process memory and lose conversations on")
        print("     every deploy again. Swap in the Upstash rediss:// URL")
        print("     before shipping.")
        print()

    if scheme in ("http", "https"):
        print("  X  That is the Upstash REST endpoint, not the Redis one.")
        print()
        print("     UPSTASH_REDIS_REST_URL / _TOKEN are for Upstash's HTTP")
        print("     API. This app speaks the Redis protocol, so it needs")
        print("     the other URL - and the REST token is NOT the Redis")
        print("     password.")
        print()
        print("     Upstash console -> your database -> Connect ->")
        print("     choose 'redis-cli' (not '@upstash/redis').")
        return

    # Only a REMOTE redis:// is suspicious. Plain redis:// is correct for
    # localhost - warning about TLS there would be advice to ignore, and
    # warnings people learn to ignore stop working.
    if scheme == "redis" and not is_local:
        # PROVIDER-NEUTRAL, deliberately. This used to say "Upstash
        # requires rediss://", which was true and became misleading the
        # moment the URL pointed at Redis Cloud instead - a warning that
        # names the wrong vendor reads as stale and gets ignored. What
        # matters is not whose server it is: session history carries real
        # order IDs and tracking numbers, and without TLS they cross the
        # public internet in the clear.
        print("  !  Scheme is redis:// - NO TLS - to a remote host.")
        print()
        print("     Session history holds prior tool results: order IDs,")
        print("     tracking numbers, delivery cities. Without TLS those")
        print("     travel unencrypted.")
        print()
        print("     Fine against the demo dataset. NOT fine while")
        print("     MONGODB_DATABASE holds real customers - enable TLS on")
        print("     the database and switch to rediss:// (two s's).")
        print()
        print("     Trying anyway.")
        print()

    line()
    print("CONNECTING")
    line()

    session_store.reset_for_tests()
    client = await get_redis()

    if client is None:
        print("  X  Could not connect. The logged error above says why.")
        print()
        print("     Common causes:")
        print("       - password wrong or truncated when copied")
        print("       - rediss:// needed but redis:// used")
        print("       - port not 6379")
        print("       - outbound TCP blocked by a firewall/VPN")
        return

    print("  OK connected")
    print()

    line()
    print("ROUND TRIP")
    line()

    probe = [
        {"role": "system", "content": "probe"},
        {"role": "user", "content": "does this survive?"},
        {"role": "assistant", "content": "checking"},
    ]
    await session_store.save_session(USER, SESSION, probe)

    # Wipe every trace of in-process state. Anything read back after
    # this came from Redis, which is the whole point of the check.
    session_store._sessions.clear()

    restored = await session_store.get_session(USER, SESSION)
    await session_store.clear_session(USER, SESSION)

    if restored and any(m.get("content") == "does this survive?" for m in restored):
        print("  OK wrote, cleared local memory, read it back from Redis")
        print()
        print("  Conversation memory now survives restarts and is shared")
        print("  across workers.")
    else:
        print("  X  Wrote to Redis but could not read it back.")
        print("     Check the key is not being evicted (maxmemory policy)")
        print("     and that the database is not read-only.")

    await close_redis()
    print()


asyncio.run(main())
