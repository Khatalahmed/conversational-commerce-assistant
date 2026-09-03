"""
WHAT:
    Stores conversation history PER SESSION, so "is it in stock?" after
    "tell me about the blue jacket" can resolve "it". Two backends behind
    one interface: Redis when REDIS_URL is configured, an in-process dict
    otherwise.

WHY REDIS, AND WHY IT IS OPTIONAL:
    The in-process dict was correct and well-bounded for a single
    process, and wrong for anything else - which we watched happen live.
    Editing a file under `uvicorn --reload` restarted the process, every
    stored conversation vanished, and the next "anything similar?"
    answered "similar to what?". The assistant was behaving correctly;
    its memory had simply been erased underneath it.

    That is not a development-only annoyance. The same erasure happens on
    every production deploy, and with more than one worker it gets worse
    and non-deterministic: a follow-up lands on a worker that never saw
    the first message, so the assistant remembers only when the load
    balancer happens to cooperate. That failure reads as "the AI is
    flaky" and gets debugged in the wrong place entirely.

    Redis is still OPTIONAL because a POC must run with nothing but a
    MongoDB URI. Without REDIS_URL the old dict is used, with the old
    limitation, and the app says so at startup rather than pretending.

    STILL NOT WRITTEN TO MONGODB. The project's read-only rule is
    unchanged: conversation history never touches the marketplace database.

FLOW:
    Phase 9's /chat endpoint calls get_session() before run_conversation
    (to pass in prior history) and save_session() after.

LOGIC:
    Two limits, per Phase 8.2:
    - MAX_TURNS: keeps only the most recent N user-initiated turns,
      trimmed at TURN boundaries rather than message count, so a
      tool_calls/tool-response pair is never split apart - which would
      make the next API call invalid.
    - SESSION_TIMEOUT_MINUTES: an untouched session expires. Under Redis
      this is a native TTL refreshed on every write; under the dict it is
      a lazy check on read, as before.

    OWNERSHIP: a session is keyed by the VERIFIED user_id together with
    the session_id, never by session_id alone. A session_id is
    client-supplied and guessable, and history contains prior tool
    results (real orders, tracking numbers). Keying on it alone would let
    anyone who guessed another user's session_id inherit that
    conversation - routing around the JWT scoping, the field allowlist
    and the sanitizer at once. Binding the key to the token's user_id
    closes that path.

MECHANISM:
    Redis stores each session as a JSON list under "assistant:session:{key}"
    with an expiry. Everything the orchestrator puts in `messages` is
    JSON-safe already - tool results are serialized strings and tool
    calls are plain dicts - so no custom encoder is needed.

    The CONNECTION is not owned here. db/redis_client.py resolves it
    once for the whole process and hands the same client to the /chat
    rate limiter, which needs shared state for the same reason this
    does. One pool, one answer to "is Redis up?", one place that knows
    about the Upstash REST-URL trap.
"""

import json
from datetime import datetime, timedelta, timezone

import structlog

from app.db import redis_client
from app.db.redis_client import get_redis

logger = structlog.get_logger()

MAX_TURNS = 10
SESSION_TIMEOUT_MINUTES = 30

# Only applies to the in-process fallback. Redis expires keys itself, so
# it needs no ceiling: an abandoned session costs nothing once its TTL
# lapses. The dict has no such mechanism for sessions never read again.
MAX_SESSIONS = 1000

REDIS_KEY_PREFIX = "assistant:session:"

_sessions: dict[str, dict] = {}


def _key(user_id: str, session_id: str) -> str:
    """ALWAYS scoped by the verified user - see the OWNERSHIP note in the
    module docstring for why session_id alone is not safe to key on."""
    return f"{user_id}:{session_id}"


# ── Trimming (backend-independent) ───────────────────────────────────

def _trim_to_recent_turns(messages: list, max_turns: int = MAX_TURNS) -> list:
    """Trims at TURN boundaries (each starting with a user message) so a
    tool_calls/tool-response pair is never split across the cut."""
    system = messages[0] if messages and messages[0].get("role") == "system" else None
    rest = messages[1:] if system else messages

    turns = []
    current_turn: list = []
    for msg in rest:
        if msg.get("role") == "user":
            if current_turn:
                turns.append(current_turn)
            current_turn = [msg]
        else:
            current_turn.append(msg)
    if current_turn:
        turns.append(current_turn)

    flattened = [m for turn in turns[-max_turns:] for m in turn]
    return ([system] if system else []) + flattened


# ── In-process fallback ──────────────────────────────────────────────

def _is_expired(session: dict, now: datetime) -> bool:
    return now - session["last_active"] > timedelta(minutes=SESSION_TIMEOUT_MINUTES)


def _purge(now: datetime) -> None:
    """Bounds memory. Runs on WRITE, because that is the only moment we
    know new memory is being claimed - a read-triggered sweep would never
    fire for the abandoned sessions that are precisely the problem."""
    expired = [k for k, s in _sessions.items() if _is_expired(s, now)]
    for key in expired:
        del _sessions[key]

    overflow = len(_sessions) - MAX_SESSIONS
    if overflow > 0:
        oldest = sorted(_sessions, key=lambda k: _sessions[k]["last_active"])[:overflow]
        for key in oldest:
            del _sessions[key]

    if expired or overflow > 0:
        logger.info(
            "sessions_purged", expired=len(expired),
            evicted=max(overflow, 0), remaining=len(_sessions),
        )


# ── Public interface ─────────────────────────────────────────────────

async def get_session(user_id: str, session_id: str) -> list | None:
    """Stored history for THIS user's session, or None if new or expired
    - either way the caller treats it as a fresh conversation. A
    session_id belonging to another user is invisible: it hashes to a
    different key, so it reads as "new", not as theirs."""
    key = _key(user_id, session_id)
    client = await get_redis()

    if client is not None:
        try:
            raw = await client.get(REDIS_KEY_PREFIX + key)
        except Exception as exc:
            # A cache read failing means "no history", not "fail the
            # request" - the user loses context, not their answer.
            logger.error("session_read_failed", error=str(exc)[:200])
            return None
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.error("session_corrupt_discarded", session_id=session_id)
            return None

    session = _sessions.get(key)
    if session is None:
        return None

    now = datetime.now(timezone.utc)
    if _is_expired(session, now):
        logger.info("session_expired", session_id=session_id)
        del _sessions[key]
        return None
    return session["messages"]


async def save_session(user_id: str, session_id: str, messages: list) -> None:
    """Persists history for next time, trimmed to the turn limit."""
    trimmed = _trim_to_recent_turns(messages)
    key = _key(user_id, session_id)
    client = await get_redis()

    if client is not None:
        try:
            await client.set(
                REDIS_KEY_PREFIX + key,
                json.dumps(trimmed),
                # Refreshed on every write, so the timeout measures
                # inactivity rather than total conversation age.
                ex=SESSION_TIMEOUT_MINUTES * 60,
            )
        except Exception as exc:
            # Losing the write costs the NEXT turn its context. Still not
            # worth failing a request the user already got an answer to.
            logger.error("session_write_failed", error=str(exc)[:200])
        else:
            logger.info("session_saved", session_id=session_id, message_count=len(trimmed))
        return

    now = datetime.now(timezone.utc)
    _sessions[key] = {"messages": trimmed, "last_active": now}
    _purge(now)
    logger.info("session_saved", session_id=session_id, message_count=len(trimmed))


async def clear_session(user_id: str, session_id: str) -> None:
    key = _key(user_id, session_id)
    client = await get_redis()
    if client is not None:
        try:
            await client.delete(REDIS_KEY_PREFIX + key)
        except Exception as exc:
            logger.error("session_delete_failed", error=str(exc)[:200])
        return
    _sessions.pop(key, None)


async def session_count() -> int:
    """Exposed for health checks and tests - the store itself stays
    private. Under Redis this SCANs, so it is a diagnostic, not something
    to call per request."""
    client = await get_redis()
    if client is not None:
        try:
            return len([k async for k in client.scan_iter(f"{REDIS_KEY_PREFIX}*")])
        except Exception as exc:
            logger.error("session_count_failed", error=str(exc)[:200])
            return 0
    return len(_sessions)


def reset_for_tests() -> None:
    """Clears in-process state and forces backend re-resolution."""
    _sessions.clear()
    redis_client.reset_for_tests()
