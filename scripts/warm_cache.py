"""
Warms the LLM prompt cache before a demo, and proves it worked.

    uv run python scripts/warm_cache.py

WHY:
    Groq and Azure both cache prompt prefixes automatically - 50% off
    cached tokens on Groq,
    and cached tokens do not count toward the rate limit. But the cache
    expires after 2 hours of disuse, so the FIRST question after an idle
    period pays full price and is visibly slower.

    That first question is the one the client sees. This sends a throwaway
    request carrying the exact same prefix the demo will use, so the real
    first question lands on a warm cache.

HOW IT PROVES IT:
    Azure and OpenAI report the cache directly, as
    usage.prompt_tokens_details.cached_tokens - a measurement rather than
    a deduction, and what this prefers when it is present. Measured on
    the real prefix: 1,792 of 1,897 prompt tokens cached, 94%.

    Groq does not report it, so there the prefix is sent TWICE and
    `prompt_time` is reported instead. A cold prefix takes several times
    longer to process than a cached one, and the drop between call one
    and call two is the cache engaging, observed rather than assumed.

WHAT MAKES THIS CORRECT:
    Caching is a PREFIX match, so a warm-up only helps if it sends the
    byte-identical prefix. This builds the system prompt through the same
    build_system_prompt() the orchestrator uses and sends the same TOOLS
    list - it does not approximate either. tool_choice="none" keeps it to
    a single cheap round with no tool execution.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402

from app.agent.llm_client import (  # noqa: E402
    MAX_RESPONSE_TOKENS,
    build_request,
    get_providers,
)
from app.agent.orchestrator import build_system_prompt  # noqa: E402
from app.agent.tools import TOOLS  # noqa: E402
from app.db.connection import close_mongo_connection, connect_to_mongo  # noqa: E402

WARM_UP_ROUNDS = 2


async def warm(provider, system_prompt: str) -> None:
    payload = {
        "model": provider.model,
        # The prefix the demo will send, not a stand-in for it.
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "hello"},
        ],
        "tools": TOOLS,
        # No tool call: one cheap round is enough to populate the cache.
        "tool_choice": "none",
        "max_tokens": MAX_RESPONSE_TOKENS,
    }

    print(f"\n  {provider.name} ({provider.model})")
    timings = []
    cache_hits = []

    async with httpx.AsyncClient(timeout=60) as client:
        for attempt in range(1, WARM_UP_ROUNDS + 1):
            try:
                # THROUGH build_request, NOT HAND-ROLLED. This used to
                # post to "{base_url}/chat/completions" with a Bearer
                # header, which is correct for Groq and wrong for Azure
                # in three separate ways at once - deployment in the
                # path, "api-key" header, max_completion_tokens. The
                # result was a flat 404 that read like a missing model.
                #
                # It failed SILENTLY in the place it mattered: this
                # script is the pre-demo warm-up, so the cache was never
                # warmed and the first question - the one the client
                # watches - paid full price anyway. Sharing the real
                # request builder means a fourth provider quirk cannot
                # reintroduce this.
                url, headers, body = build_request(provider, payload)
                response = await client.post(url, headers=headers, json=body)
            except httpx.HTTPError as exc:
                print(f"    call {attempt}: unreachable ({type(exc).__name__})")
                return

            if response.status_code != 200:
                detail = response.text[:120].replace("\n", " ")
                print(f"    call {attempt}: HTTP {response.status_code} - {detail}")
                return

            usage = response.json().get("usage") or {}
            prompt_tokens = usage.get("prompt_tokens", 0)
            # Groq-specific; absent elsewhere, in which case we can still
            # confirm the prefix was accepted but not that it cached.
            prompt_time = usage.get("prompt_time")
            timings.append(prompt_time)

            # Azure/OpenAI report the cache DIRECTLY rather than leaving
            # it to be inferred from timing - a measurement instead of a
            # deduction, so prefer it when it is there.
            cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
            cache_hits.append(cached)

            if cached is not None:
                print(
                    f"    call {attempt}: {prompt_tokens} prompt tokens, "
                    f"{cached} cached"
                )
            elif prompt_time is None:
                print(f"    call {attempt}: {prompt_tokens} prompt tokens")
            else:
                print(
                    f"    call {attempt}: {prompt_tokens} prompt tokens, "
                    f"prompt_time {prompt_time:.3f}s"
                )

    # A REPORTED CACHE COUNT BEATS AN INFERRED ONE, so check it first.
    # Azure returns cached_tokens directly; Groq does not, and there the
    # drop in prompt_time between the two calls is the only evidence
    # available. Reporting "does not report prompt_time" while holding an
    # exact cached-token count, as this did, throws away the better
    # measurement and reads as a failure.
    if len(cache_hits) >= 2 and cache_hits[1] is not None:
        warmed = cache_hits[1]
        if warmed > 0:
            print(f"    -> CACHE WARM ({warmed} of {prompt_tokens} prefix tokens cached)")
        else:
            print("    -> nothing cached; the prefix may be below this "
                  "provider's minimum cacheable length")
    elif len(timings) >= 2 and all(t is not None for t in timings[:2]):
        cold, warm_time = timings[0], timings[1]
        if warm_time < cold * 0.7:
            speedup = cold / warm_time if warm_time else 0
            print(f"    -> CACHE WARM ({speedup:.1f}x faster to process the prefix)")
        else:
            print("    -> no clear speed-up; the prefix may not be cacheable here")
    else:
        print("    -> sent, but this provider reports neither cached tokens "
              "nor prompt_time")


async def main() -> None:
    # The system prompt embeds real category values from the database, so
    # a connection is needed to reproduce the exact prefix.
    await connect_to_mongo()
    system_prompt = await build_system_prompt()
    await close_mongo_connection()

    providers = get_providers()
    print("-" * 62)
    print("WARMING PROMPT CACHE")
    print("-" * 62)
    print(f"  prefix: system prompt + {len(TOOLS)} tool schemas")

    for provider in providers:
        await warm(provider, system_prompt)

    print()
    print("-" * 62)
    print("  Run this a minute or two before the demo starts.")
    print("  The cache holds for 2 hours of use; it expires after 2 hours idle.")
    print("  Re-run it if the demo is delayed, or if you change the system")
    print("  prompt or the tool list - either invalidates the prefix.")
    print("-" * 62)


asyncio.run(main())
