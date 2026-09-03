"""
WHAT:
    The two ways to hold a conversation. /chat returns the finished
    reply in one JSON response; /chat/stream sends the same reply as it
    is produced, over Server-Sent Events.

WHY BOTH:
    /chat is the contract Phase 12's mobile app was written against, and
    a streaming endpoint is a different shape of client code. Changing
    it would mean the integration has to land at the same moment as this
    does. So /chat is untouched and /chat/stream is added beside it -
    the mobile app adopts streaming when it is ready to, not when we
    are.

    They are NOT two implementations. Both call the same
    run_conversation; the only difference is that the streaming one
    passes a callback. There is no second orchestration loop to keep in
    step, which is the failure mode this shape exists to avoid.

WHY STREAMING AT ALL:
    A measured answer takes 8-10 seconds, and the handover notes advise
    talking over the silence. Most of that is spent on tool rounds that
    produce no readable text, so streaming TOKENS alone would not fill
    it - the first token cannot arrive until the lookups are done. What
    fills it is the status events: the assistant says what it is looking
    up while it looks it up, and then types the answer out.

MECHANISM:
    run_conversation runs as a task and pushes events into a queue; the
    response generator drains that queue into SSE frames. A queue rather
    than yielding directly because the callback is called from deep
    inside the loop, which knows nothing about HTTP.
"""

import asyncio
import json

import structlog
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.agent.orchestrator import FALLBACK_GENERIC, run_conversation
from app.api.dependencies import rate_limited_user_id
from app.api.schemas import ChatRequest, ChatResponse
from app.memory import session_store

logger = structlog.get_logger()

router = APIRouter()

# Sentinel closing the queue. An object() rather than None, so a genuine
# event can never be mistaken for the end of the stream.
_STREAM_END = object()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, user_id: str = Depends(rate_limited_user_id)):
    # user_id comes from the verified JWT, never the request body - and it
    # scopes the SESSION as well as the data. session_id alone is
    # client-supplied and guessable; see session_store's OWNERSHIP note.
    history = await session_store.get_session(user_id, request.session_id)
    # The mobile app calls THIS endpoint, not the streaming one, so the
    # product rows have to arrive in the response body rather than only
    # as an SSE event.
    products: list = []
    attribution: list = []
    reply, messages = await run_conversation(
        user_id, request.message, history=history,
        products_out=products, attribution_out=attribution,
    )
    await session_store.save_session(user_id, request.session_id, messages)
    return ChatResponse(
        reply=reply, session_id=request.session_id,
        products=products, attribution=attribution,
    )


def _frame(event: dict) -> str:
    """One SSE frame. The event name goes on its own line so a client can
    addEventListener per kind instead of switching on a field."""
    payload = {k: v for k, v in event.items() if k != "type"}
    return f"event: {event['type']}\ndata: {json.dumps(payload)}\n\n"


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest, user_id: str = Depends(rate_limited_user_id)
):
    """The same conversation, delivered as it happens.

    Events, in the order a client sees them:
      status    {"tool": "...", "label": "Looking up your orders"}
      token     {"text": "..."} - fragments of the answer
      products  {"products": [{"product_id": "...", "name": "...", ...}]}
      attribution {"claims": [{"start": 12, "end": 18, "text": "₹1,199",
                   "kind": "price", "tool": "get_order_history"}]}
      done      {"reply": "...", "session_id": "..."}
      error     {"message": "..."} - a sentence, never a stack trace

    products ARRIVES BEFORE done, and at most once. It is sent only when
    the reply is a real answer - never under one of the fallbacks, where
    rows to tap would contradict the apology above them. A client that
    ignores the event renders exactly what it rendered before.

    done CARRIES THE WHOLE REPLY, not just a terminator. A client can
    render the tokens as they arrive and then replace what it drew with
    the authoritative text - which is what makes the fallback paths work
    unchanged. When the model never produced a token (a rate limit, a
    dead provider), those replies are still real sentences and still
    need to reach the user; they arrive in done rather than as tokens,
    and the client needs no special case for them.
    """
    history = await session_store.get_session(user_id, request.session_id)
    queue: asyncio.Queue = asyncio.Queue()

    def on_event(event: dict) -> None:
        # Called synchronously from inside the orchestration loop, on
        # this same event loop - so put_nowait is both safe and correct.
        # An unbounded queue because the producer is the LLM's own token
        # rate: applying backpressure here would slow the answer down to
        # the speed of the slowest reader.
        queue.put_nowait(event)

    async def converse() -> None:
        try:
            reply, messages = await run_conversation(
                user_id, request.message, history=history, on_event=on_event
            )
            await session_store.save_session(user_id, request.session_id, messages)
            queue.put_nowait(
                {"type": "done", "reply": reply, "session_id": request.session_id}
            )
        except asyncio.CancelledError:
            # The client hung up. Nothing to report to nobody.
            raise
        except Exception:
            # run_conversation answers every provider failure in words,
            # so reaching here means something genuinely unexpected. The
            # user still gets a sentence rather than a severed stream.
            logger.exception("chat_stream_failed", user_id=user_id)
            queue.put_nowait({"type": "error", "message": FALLBACK_GENERIC})
        finally:
            queue.put_nowait(_STREAM_END)

    async def events():
        task = asyncio.create_task(converse())
        try:
            while True:
                event = await queue.get()
                if event is _STREAM_END:
                    break
                yield _frame(event)
        finally:
            # A disconnected client should not leave an LLM call and a
            # session write running behind it.
            if not task.done():
                task.cancel()

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # Nginx and several PaaS proxies buffer responses by default,
            # which would hold every frame until the answer finished -
            # turning a streaming endpoint back into a slow blocking one,
            # and only once it is deployed.
            "X-Accel-Buffering": "no",
        },
    )
