"""
Verifies the Azure OpenAI setup end to end, and reports which of the
four values is wrong when it is not working.

    uv run python scripts/check_azure.py

WHY THIS EXISTS:
    Azure needs FOUR values that must all agree, and each one fails
    differently and misleadingly:

      endpoint     wrong host -> DNS error, looks like no internet
      api key      wrong -> 401, looks like the resource is missing
      deployment   wrong -> 404 that reads "model not found", when the
                   model is fine and only the DEPLOYMENT name is wrong
      api version  wrong -> 400 about an unsupported parameter, which
                   looks like a bug in our request

    The deployment one is the trap: the deployment name is chosen by
    whoever created it and need NOT match the model name, so a 404
    saying "gpt-5-mini not found" can appear while gpt-5-mini is
    deployed perfectly well under a different name.

    So this makes a real tool-calling request - not a ping - because
    tool calling is what the assistant actually does, and a model that
    answers plain chat but mishandles 32 tool schemas would pass a
    simpler check and fail in the demo.
"""

import asyncio
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.llm_client import (  # noqa: E402
    LLMBadRequest,
    LLMUnavailable,
    build_request,
    complete,
    get_providers,
)
from app.agent.orchestrator import build_system_prompt  # noqa: E402
from app.agent.tools import TOOLS  # noqa: E402
from app.config.settings import get_settings  # noqa: E402
from app.db.connection import (  # noqa: E402
    close_mongo_connection,
    connect_to_mongo,
)


def line() -> None:
    print("-" * 64)


async def main() -> None:
    settings = get_settings()

    line()
    print("AZURE OPENAI CHECK")
    line()

    if not settings.azure_openai_configured:
        print("  Not configured. All three of these are required:")
        for name, value in [
            ("AZURE_OPENAI_ENDPOINT", settings.azure_openai_endpoint),
            ("AZURE_OPENAI_API_KEY", settings.azure_openai_api_key),
            ("AZURE_OPENAI_DEPLOYMENT", settings.azure_openai_deployment),
        ]:
            print(f"    {'set  ' if value else 'MISSING'}  {name}")
        print()
        print("  Without them the app falls back to Groq, which is on the")
        print("  free tier: 8,000 tokens/minute, and its terms permit")
        print("  training on what you send.")
        return

    azure = next((p for p in get_providers() if p.kind == "azure"), None)
    if azure is None:
        print("  Configured but not registered as a provider - that is a bug.")
        return

    print(f"  endpoint    {settings.azure_openai_endpoint}")
    print(f"  deployment  {settings.azure_openai_deployment}")
    print(f"  version     {settings.azure_openai_api_version}")
    print()

    url, _, _ = build_request(azure, {"max_tokens": 1, "messages": []})
    print("  request URL (key never printed):")
    print(f"    {url}")
    print()

    line()
    print("REAL TOOL-CALLING REQUEST")
    line()

    # THE REAL SYSTEM PROMPT, not an approximation of one.
    #
    # An earlier version sent "You are a shopping assistant. Use tools."
    # and then warned that the model had answered in words instead of
    # calling get_order_history. That verdict was wrong: the rule telling
    # it to call get_order_history lives in the REAL prompt, which the
    # check was not sending. It was testing something the app never does
    # and reporting the difference as a fault.
    #
    # Costs a database connection, since the prompt embeds live category
    # values - worth it for a check whose whole job is to be trusted.
    await connect_to_mongo()
    system_prompt = await build_system_prompt()
    await close_mongo_connection()

    try:
        completion = await complete(
            azure,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "where is my order"},
            ],
            TOOLS,
        )
    except LLMBadRequest as exc:
        print(f"  X  400 Bad Request: {exc}")
        print()
        print("     Usually the API VERSION, or a parameter this model")
        print("     does not accept. The gpt-5 series renamed max_tokens")
        print("     to max_completion_tokens - build_request() handles")
        print("     that, so check AZURE_OPENAI_API_VERSION first.")
        return
    except LLMUnavailable as exc:
        detail = str(exc)
        print(f"  X  {detail[:200]}")
        print()
        if "404" in detail:
            print("     404 usually means the DEPLOYMENT NAME is wrong -")
            print("     not the model. Check Foundry -> Deployments for the")
            print(f"     exact name; you have '{settings.azure_openai_deployment}'.")
        elif "401" in detail or "403" in detail:
            print("     401/403 means the KEY is wrong or belongs to another")
            print("     resource. Copy KEY 1 from the resource page again.")
        else:
            print("     Check the endpoint host and your network.")
        return

    print(f"  OK connected")
    print(f"     prompt tokens: {completion.prompt_tokens}")
    print(f"     tools offered: {len(TOOLS)}")
    if completion.tool_calls:
        print(f"     tool chosen:   {completion.tool_calls[0].name}")
        print()
        print("  OK the model reads the tool schemas and picks correctly.")
    else:
        print(f"     replied:       {(completion.content or '')[:80]}")
        print()
        print("  !  Answered in words instead of calling a tool. Not fatal,")
        print("     but 'where is my order' should trigger get_order_history -")
        print("     worth checking before relying on it for a demo.")

    print()
    line()
    # DERIVED, NOT ASSERTED. This line used to read "Azure is primary;
    # the others remain as failover" unconditionally - which became
    # false the moment the provider filter started dropping the
    # training-permitted ones, and printed four rows below a log line
    # saying they had been excluded. A summary that contradicts the
    # state it summarises teaches the reader the wrong thing, and this
    # particular wrong thing is "you have a fallback" when you do not.
    providers = get_providers()
    print("  Provider order:", " -> ".join(p.name for p in providers))

    if len(providers) == 1:
        print(f"  {providers[0].name} ONLY - there is no fallback.")
        print("     An outage there is a full outage: users get")
        print('     "I can\'t reach my assistant service at the moment."')
        print("     Set BACKUP_LLM_* to add a second provider - see .env.example.")
    else:
        trains = [p.name for p in providers[1:] if p.trains_on_prompts]
        print(f"  {providers[0].name} is primary; "
              f"{', '.join(p.name for p in providers[1:])} behind it.")
        if trains:
            print(f"     NOTE: {', '.join(trains)} may train on prompts. Safe for")
            print("     demo data; they are dropped automatically if")
            print("     MONGODB_DATABASE is pointed at real customers.")

    _report_credit_expiry()


def _report_credit_expiry() -> None:
    """Says how many days of Azure are left.

    Shown in the PRE-FLIGHT check rather than a calendar reminder,
    because this is the script run before every demo - which is exactly
    when finding out matters and exactly when a calendar entry from four
    weeks ago will not be looked at.
    """
    from app.config.settings import get_settings

    settings = get_settings()
    raw = settings.azure_credit_expires
    if not raw:
        return

    # PARSED BY SETTINGS, NOT HERE. The running service warns about this
    # date too now - a startup log line and /health - and two
    # implementations of "how many days left" would eventually disagree
    # about the answer.
    days = settings.credit_days_remaining
    if days is None:
        print(f"\n  !  AZURE_CREDIT_EXPIRES={raw!r} is not an ISO date (YYYY-MM-DD).")
        return

    expires = date.fromisoformat(raw.strip())
    print()
    if days < 0:
        print(f"  X  Azure credit EXPIRED {abs(days)} day(s) ago ({expires}).")
        _print_remedies()
    elif days <= 14:
        print(f"  !  Azure credit expires in {days} day(s), on {expires}.")
        _print_remedies()
    else:
        print(f"  Azure credit: {days} days left (expires {expires}).")
    line()


def _print_remedies() -> None:
    """The three ways out, best first.

    Printed at the moment the problem is discovered rather than left in
    a document, because this script is what gets run before a demo and
    "there is no fallback" without a next step is just bad news.
    """
    print("     Fixes, best first:")
    print("       1. Renew the Azure credit.")
    print("       2. Point BACKUP_LLM_* at another provider whose terms")
    print("          exclude training, with BACKUP_LLM_TRAINS_ON_PROMPTS=false.")
    print("          A .env change and a restart - no code change.")
    print("       3. For a demo only: MONGODB_DATABASE=marketplace_demo, which")
    print("          re-enables Groq and Gemini automatically. Demo data,")
    print("          but a working assistant.")


asyncio.run(main())
