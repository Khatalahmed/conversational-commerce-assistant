"""
WHAT:
    The ONE place that talks to an LLM. Owns the ordered list of
    providers, the HTTP call itself, and the small set of error types
    the rest of the app reasons about.

WHY THIS APPROACH:
    Two providers, one wire format. Groq and Google both expose an
    OpenAI-COMPATIBLE chat-completions endpoint, which means the same
    `messages` list, the same tool schemas, and the same response shape
    work on either - verified against our real 34 schemas before this
    was written. That compatibility is what makes failover safe: the
    conversation history stored by Phase 8 is provider-neutral, so a
    request can switch providers mid-conversation without rewriting
    anything it already said.

    WHY RAW httpx AND NOT AN SDK: the OpenAI-compatible surface we use
    is one POST. A vendor SDK would bring its own exception hierarchy
    per provider, so the orchestrator would have to catch two parallel
    sets of "rate limited" - and the groq SDK in particular hardcodes
    its URL path, so it cannot be pointed at Google at all. One small
    client here means ONE error taxonomy for every provider, and httpx
    was already a dependency.

FLOW:
    orchestrator.py asks get_providers() for the ordered list, then
    calls complete() on each in turn until one answers.

LOGIC:
    WHY THERE IS A FALLBACK AT ALL: Groq's free tier caps at 8,000
    tokens/minute, and one round of our conversation costs ~1,450 of
    them (783 for the tool schemas alone, re-sent every round). That is
    roughly five rounds a minute for the whole app - a single
    multi-turn conversation can exhaust it. Google's free tier is a
    SEPARATE quota at a slightly lower per-round cost (~1,300), so the
    fallback roughly doubles the budget rather than only catching
    outages.

    Order is deliberate: Groq first because it is measurably faster
    (~0.8s vs ~2.1s per call). Google is tried only when Groq will not
    answer, and the app runs on Groq alone when no Google key is set -
    a fallback that is mandatory to boot is not a fallback.

    MODELS ARE PINNED, NOT ALIASED. "gemini-flash-latest" would move
    underneath us silently. That matters more than usual here: the
    model this file originally used (llama-3.3-70b-versatile) vanished
    from the account with a 404, and gemini-2.5-flash was already
    deprecated when we probed it. Two dead model names in one sitting
    is why every call logs the provider and model that served it.

MECHANISM:
    A single shared httpx.AsyncClient (connection pooling, same
    lazy-singleton pattern as app/db/connection.py), and a mapping from
    HTTP status to the three error types below. ASYNC throughout, for
    the same reason Phase 2 chose Motor: a blocking call here would
    freeze every other user's request for the seconds an LLM takes.
"""

import json
from dataclasses import dataclass
from functools import lru_cache

import httpx
import structlog

from app.config.settings import get_settings

logger = structlog.get_logger()

# A single turn makes up to MAX_TOOL_ITERATIONS calls, so this is a
# per-CALL ceiling, not a per-message one.
REQUEST_TIMEOUT_SECONDS = 30.0
# COVERS REASONING TOKENS TOO, which is why it is not 1024.
#
# gpt-5-mini is a reasoning model, and on Azure max_completion_tokens
# budgets reasoning AND visible output together. Measured: a trivial
# prompt spent 128 tokens reasoning before writing anything. With three
# tool results in context it exceeded 1024 entirely - returning EMPTY
# content with finish_reason="length", which surfaced as a blank chat
# bubble and, on the synthesis turn, as a fallback for a question the
# model had fully researched.
#
# Raising the ceiling costs nothing: billing is on tokens used, not on
# the cap.
MAX_RESPONSE_TOKENS = 4096

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "openai/gpt-oss-120b"

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
GEMINI_MODEL = "gemini-3.6-flash"


# -- Errors ----------------------------------------------------------
# Three types, because the orchestrator answers each one differently:
# "wait a moment", "try again shortly", "rephrase that".

class LLMError(Exception):
    """Base for every provider failure."""


class LLMRateLimited(LLMError):
    """Quota exhausted. Deliberately NOT retried on the same provider -
    these limits are per-minute, so an immediate retry just burns
    another request. Failing over to the next provider IS worthwhile,
    since its quota is entirely separate."""


class LLMUnavailable(LLMError):
    """Unreachable, timed out, 5xx, bad credentials, or a model that no
    longer exists. Retrying the same provider will not help; another
    provider might."""


class NoProviderConfigured(LLMUnavailable):
    """No provider is usable for the configured data.

    A SUBCLASS OF LLMUnavailable ON PURPOSE. It used to be a bare
    RuntimeError raised from get_providers(), which the orchestrator
    calls OUTSIDE its try block - so this one case escaped the three
    handlers and surfaced to the mobile app as a raw 500 with a stack
    trace, while every other provider failure produced a sentence.

    It is exactly the case most likely to happen on the day the Azure
    credit lapses and somebody removes the dead key: real data still
    configured, every remaining provider one that may train on prompts,
    and the correct answer is to serve nobody. That is still a total
    outage - but a total outage that apologises in words and says why in
    the logs, rather than one that 500s.
    """


class LLMBadRequest(LLMError):
    """The provider rejected the request itself. `code` carries the
    provider's own error code so the orchestrator can recognise
    "tool_use_failed" - a smaller model wrapping a tool call in a
    malformed tag - which IS worth retrying, unlike a genuinely
    malformed request, which is not."""

    def __init__(self, message: str, code: str | None = None):
        super().__init__(message)
        self.code = code


# -- Provider list ---------------------------------------------------

@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    model: str
    api_key: str
    # "openai" for anything OpenAI-COMPATIBLE (Groq, Gemini's compat
    # endpoint); "azure" for Azure OpenAI, which differs in three ways
    # that all have to be handled together - see build_request().
    kind: str = "openai"
    api_version: str | None = None
    # Whether this provider's terms permit training on what we send it.
    # Not cosmetic: it decides eligibility when the configured database
    # holds real customers - see get_providers().
    trains_on_prompts: bool = True


@lru_cache
def get_providers() -> tuple[Provider, ...]:
    """Ordered by preference. Google is included ONLY if its key is
    configured - see settings.gemini_api_key, which is optional."""
    settings = get_settings()

    providers = []

    # AZURE FIRST WHEN CONFIGURED. Not a preference - three measured
    # differences, in descending order of how much they matter here:
    #
    #   privacy   Azure commits that prompts are not used to train
    #             models. That is the ONLY reason demoing against real
    #             the marketplace customer orders is defensible; the free tiers
    #             explicitly reserve the right to train on what we send.
    #   quota     100,000 TPM against Groq free tier's 8,000. A measured
    #             10-question demo spends 113,000 prompt tokens.
    #   model     a frontier-lab model rather than an open-weight one.
    #             The malformed tool calls that forced the retry logic in
    #             orchestrator.py came from the latter.
    #
    # Groq and Gemini remain as fallbacks - but only where what we send
    # them is not somebody's real order. See the filter below.
    if settings.azure_openai_configured:
        providers.append(
            Provider(
                name="azure",
                base_url=settings.azure_openai_endpoint,
                model=settings.azure_openai_deployment,
                api_key=settings.azure_openai_api_key,
                kind="azure",
                api_version=settings.azure_openai_api_version,
                trains_on_prompts=False,
            )
        )

    # THE MITIGATION FOR AZURE BEING ALONE. Placed directly after Azure
    # because that is what it is for: while real customer data is
    # configured, Azure is the only provider the filter below leaves
    # standing, and its credit has an expiry date. A second provider
    # whose terms also exclude training keeps the chain two deep, and
    # arrives entirely from .env - see settings.backup_llm_*.
    #
    # It is NOT assumed compliant. backup_llm_trains_on_prompts defaults
    # to True, so an operator who configures a backup without asserting
    # anything about its terms gets a provider that behaves exactly like
    # Groq: fine for the demo dataset, dropped for real customers.
    if settings.backup_llm_configured:
        providers.append(
            Provider(
                name="backup",
                base_url=settings.backup_llm_base_url,
                model=settings.backup_llm_model,
                api_key=settings.backup_llm_api_key,
                kind=settings.backup_llm_kind,
                api_version=settings.backup_llm_api_version,
                trains_on_prompts=settings.backup_llm_trains_on_prompts,
            )
        )

    if settings.llm_api_key:
        providers.append(
            Provider("groq", GROQ_BASE_URL, GROQ_MODEL, settings.llm_api_key)
        )
    if settings.gemini_api_key:
        providers.append(
            Provider("gemini", GEMINI_BASE_URL, GEMINI_MODEL, settings.gemini_api_key)
        )
    else:
        logger.info("llm_fallback_not_configured", reason="no GEMINI_API_KEY in .env")

    # THE CHAIN FOLLOWS THE DATA, NOT A FLAG.
    #
    # Reading the real `marketplace` database means every tool result carries
    # somebody's actual order - delivery city, courier tracking number,
    # purchase history. Those may only reach a provider contractually
    # excluded from training on them.
    #
    # It was previously enough that Azure was FIRST. That is a guarantee
    # about ordering, not about destination: one rate limit or 500 from
    # Azure and the failover would hand the same customer data to Groq,
    # which reserves the right to train on it. A safety property that
    # holds only while nothing goes wrong is not a safety property.
    #
    # Keyed on the DATABASE rather than a --real-data flag, so it cannot
    # be forgotten: whoever points .env at real customers gets the
    # restriction automatically, and whoever works against the demo
    # dataset keeps the cheap providers.
    if settings.mongodb_database == settings.production_database_name:
        eligible = tuple(p for p in providers if not p.trains_on_prompts)
        excluded = [p.name for p in providers if p.trains_on_prompts]
        if not eligible:
            raise NoProviderConfigured(
                f"MONGODB_DATABASE is '{settings.production_database_name}' (real customer "
                f"data), but no provider is contractually excluded from "
                f"training on prompts. Configure AZURE_OPENAI_*, or set "
                f"BACKUP_LLM_* with BACKUP_LLM_TRAINS_ON_PROMPTS=false, or "
                f"point MONGODB_DATABASE at the demo dataset."
            )
        if excluded:
            logger.info(
                "training_permitted_providers_excluded",
                excluded=excluded,
                reason="MONGODB_DATABASE holds real customer data",
                remaining=[p.name for p in eligible],
            )
        providers = list(eligible)

    if not providers:
        raise NoProviderConfigured(
            "No LLM provider is configured. Set AZURE_OPENAI_* (recommended - "
            "it is contractually excluded from training on prompts), BACKUP_LLM_* "
            "for any other OpenAI-compatible endpoint, or LLM_API_KEY for Groq."
        )

    logger.info("llm_providers_configured", providers=[p.name for p in providers])
    return tuple(providers)


# -- Is the chain actually survivable? --------------------------------

@dataclass(frozen=True)
class ChainStatus:
    """What the provider chain looks like right now, and how worried to
    be about it."""

    providers: tuple[str, ...]
    real_data: bool
    redundant: bool
    credit_days: int | None
    level: str            # "ok" | "at_risk" | "critical"
    reasons: tuple[str, ...]


CREDIT_AT_RISK_DAYS = 14
CREDIT_CRITICAL_DAYS = 3


def assess_chain() -> ChainStatus:
    """Reports whether one provider failing would take the service down.

    WHY THIS IS COMPUTED RATHER THAN REMEMBERED. The dangerous
    configuration is not exotic - it is the DEFAULT one: point
    MONGODB_DATABASE at real customers and the training filter leaves
    Azure alone in the chain, on a credit with an expiry date. Nothing
    in the running service noticed. The date lived in one .env line that
    only scripts/check_azure.py ever read, so the service could be three
    days from a total outage and say nothing at startup, nothing to
    monitoring, and nothing in a log anybody greps.

    "at_risk" IS EXPECTED TO FIRE TODAY, and that is the point: a single
    non-training provider on borrowed time IS a service at risk, and the
    signal clears the moment BACKUP_LLM_* gives the chain somewhere to
    fall back to. A status that only ever says "ok" would be decoration.
    """
    settings = get_settings()
    credit_days = settings.credit_days_remaining
    real_data = settings.mongodb_database == settings.production_database_name

    try:
        providers = get_providers()
    except NoProviderConfigured as exc:
        return ChainStatus(
            providers=(),
            real_data=real_data,
            redundant=False,
            credit_days=credit_days,
            level="critical",
            reasons=(str(exc),),
        )

    names = tuple(p.name for p in providers)
    redundant = len(names) > 1
    reasons: list[str] = []
    level = "ok"

    if not redundant:
        level = "at_risk"
        reasons.append(
            f"only one usable provider ({names[0]}) - any outage there is a full "
            f"outage. Set BACKUP_LLM_* to add a second."
            + (
                " Groq and Gemini are excluded because MONGODB_DATABASE holds real "
                "customer data and their terms permit training on prompts."
                if real_data
                else ""
            )
        )

    if credit_days is not None:
        if credit_days < 0:
            level = "critical"
            reasons.append(f"the Azure credit expired {abs(credit_days)} day(s) ago")
        elif credit_days <= CREDIT_CRITICAL_DAYS and not redundant:
            level = "critical"
            reasons.append(
                f"the Azure credit expires in {credit_days} day(s) and nothing else "
                f"can serve this data"
            )
        elif credit_days <= CREDIT_AT_RISK_DAYS:
            level = "critical" if level == "critical" else "at_risk"
            reasons.append(f"the Azure credit expires in {credit_days} day(s)")

    return ChainStatus(
        providers=names,
        real_data=real_data,
        redundant=redundant,
        credit_days=credit_days,
        level=level,
        reasons=tuple(reasons),
    )


def log_provider_preflight() -> None:
    """Says at STARTUP what would otherwise only be discovered by a user
    getting an apology. Called from the app lifespan."""
    status = assess_chain()
    log = {
        "ok": logger.info,
        "at_risk": logger.warning,
        "critical": logger.error,
    }[status.level]
    log(
        "llm_chain_preflight",
        level=status.level,
        providers=list(status.providers),
        redundant=status.redundant,
        real_data=status.real_data,
        credit_days_remaining=status.credit_days,
        reasons=list(status.reasons),
    )


def build_request(provider: Provider, payload: dict) -> tuple[str, dict, dict]:
    """Returns (url, headers, payload) for one provider.

    THREE THINGS DIFFER ON AZURE, and missing any one of them produces a
    confusing failure rather than an obvious one:

      1. The DEPLOYMENT is in the URL PATH, not the "model" body field.
         Azure has no notion of picking a model per request - you address
         a deployment you created, and its name need not match the model.
         Get it wrong and you get a 404 that reads like the model does
         not exist.

      2. Auth is the "api-key" HEADER, not "Authorization: Bearer".
         Sending Bearer gets a 401 that looks like a bad key.

      3. The gpt-5 series renamed max_tokens to max_completion_tokens.
         Sending the old name is a 400 - and this is the one most likely
         to be missed, because it is a MODEL-generation difference that
         happens to surface here, not an Azure-vs-OpenAI difference.

    An api-version is also mandatory: Azure pins behaviour to a dated
    string rather than versioning by model alone.
    """
    if provider.kind != "azure":
        return (
            f"{provider.base_url}/chat/completions",
            {"Authorization": f"Bearer {provider.api_key}"},
            payload,
        )

    body = dict(payload)
    if "max_tokens" in body:
        body["max_completion_tokens"] = body.pop("max_tokens")

    base = provider.base_url.rstrip("/")
    url = (
        f"{base}/openai/deployments/{provider.model}/chat/completions"
        f"?api-version={provider.api_version}"
    )
    return url, {"api-key": provider.api_key}, body


# -- Response shape --------------------------------------------------
# Deliberately explicit rather than a generic dict-to-object conversion:
# the orchestrator relies on tool_calls ALWAYS existing (as an empty
# list when there are none), and a generic converter would raise
# AttributeError on a response that simply omitted the key.

# Keys we understand in a tool_call. ANYTHING ELSE is provider-specific
# and must survive a round trip untouched - see ToolCall.extra.
_KNOWN_TOOL_CALL_KEYS = {"id", "type", "function", "index"}


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string - the orchestrator parses/validates it
    extra: dict     # provider-specific fields, echoed back verbatim

    def to_message_part(self) -> dict:
        """Rebuilds the wire form for the conversation history.

        WHY `extra` EXISTS AT ALL: the two providers are
        OpenAI-COMPATIBLE, not OpenAI-identical. Google attaches
        extra_content.google.thought_signature to every tool call it
        emits, and REJECTS that same call on the next turn if the
        signature is not echoed back - so dropping unknown fields
        silently broke multi-round conversations on the fallback
        provider. Preserving unknown keys generically, rather than
        special-casing Google, means the next provider's proprietary
        field costs nothing to support.
        """
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
            **self.extra,
        }


def _cached_tokens(usage: dict) -> int:
    """Cached prefix tokens out of a usage payload.

    Nested under prompt_tokens_details on the OpenAI-compatible shape
    Azure returns. Providers that do not cache omit the whole object,
    and at least one sends it as null rather than leaving it out - both
    read as zero rather than raising.
    """
    details = usage.get("prompt_tokens_details") or {}
    if not isinstance(details, dict):
        return 0
    return int(details.get("cached_tokens") or 0)


@dataclass(frozen=True)
class Completion:
    content: str | None
    tool_calls: list[ToolCall]
    provider: str
    model: str
    prompt_tokens: int
    # HOW MANY OF THOSE PROMPT TOKENS WERE ALREADY CACHED.
    #
    # The prefix - system prompt plus 35 tool schemas - is re-sent on
    # every round, and both Azure and Groq cache it automatically. There
    # is a whole test module guarding that prefix against drift
    # (test_prompt_cache_stability) and a script that warms it before a
    # demo, and until now NOTHING measured whether any of it worked: a
    # single stray byte would silently double the bill and reinstate the
    # rate limit, with the assistant still answering perfectly.
    #
    # Defaulted, because not every provider reports it. Zero therefore
    # means "not reported OR a genuine miss" - which is why it is logged
    # beside prompt_tokens rather than alone.
    cached_tokens: int = 0


# -- The call --------------------------------------------------------

_client: httpx.AsyncClient | None = None


def _get_http_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS)
    return _client


async def close_llm_client() -> None:
    """Call once at app shutdown, mirroring close_mongo_connection()."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
        logger.info("llm_client_closed")


def _raise_for_status(provider: Provider, response: httpx.Response) -> None:
    """Maps an HTTP status onto our three error types."""
    if response.status_code == 200:
        return

    status = response.status_code
    code = None
    try:
        body = response.json()
        if isinstance(body, list) and body:
            body = body[0]  # Google returns a single-element list on error
        error = (body or {}).get("error", {})
        raw_code = error.get("code")
        code = raw_code if isinstance(raw_code, str) else None
        detail = str(error.get("message", response.text))[:200]
    except Exception:
        detail = response.text[:200]

    if status == 429:
        raise LLMRateLimited(f"{provider.name}: quota exhausted - {detail}")
    if status == 400:
        raise LLMBadRequest(f"{provider.name}: {detail}", code=code)
    # 401/403 (bad key), 404 (model retired), 5xx (their side) all mean
    # the same thing to us: this provider cannot serve us right now.
    raise LLMUnavailable(f"{provider.name}: HTTP {status} - {detail}")


def _build_payload(
    provider: Provider, messages: list, tools: list, tool_choice: str
) -> dict:
    """The request body, shared by the buffered and streamed calls so
    the two can never disagree about what was asked for."""
    payload = {
        "model": provider.model,
        "messages": messages,
        "max_tokens": MAX_RESPONSE_TOKENS,
    }
    # tool_choice="none" means "the tools exist, but answer in words
    # this time" - used by the orchestrator's final synthesis turn.
    #
    # THE TOOLS STAY DECLARED even then, and that is the whole point.
    # Dropping them while the history still contains tool_calls and
    # tool results produced a request that contradicted itself: the
    # transcript showed tools being used, the payload said there were
    # none. The model imitated the pattern anyway and emitted a call it
    # had no schema for, which Groq rejected outright with "Parsing
    # failed. The model generated output that could not be parsed."
    #
    # An empty list is still honoured as "no tools at all", since
    # providers reject "tools": [] as malformed and a tool_choice
    # without tools is meaningless.
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice

    return payload


def _log_if_truncated(provider: Provider, finish_reason, content) -> None:
    """TRUNCATION USED TO BE SILENT. finish_reason was never read, so a
    response cut off mid-sentence - or cut off before it started, as
    happens when reasoning eats the whole budget - looked identical to a
    short answer. Logged rather than raised: a truncated reply is still
    usually usable, and the caller decides."""
    if finish_reason == "length":
        logger.warning(
            "llm_response_truncated",
            provider=provider.name,
            model=provider.model,
            max_response_tokens=MAX_RESPONSE_TOKENS,
            content_empty=not (content or "").strip(),
        )


async def complete(
    provider: Provider, messages: list, tools: list, tool_choice: str = "auto"
) -> Completion:
    """One chat-completion call against ONE provider. Raises one of the
    three LLMError types above; never returns a partial result."""
    payload = _build_payload(provider, messages, tools, tool_choice)
    url, headers, body = build_request(provider, payload)

    try:
        response = await _get_http_client().post(url, headers=headers, json=body)
    except httpx.TimeoutException as exc:
        raise LLMUnavailable(
            f"{provider.name}: timed out after {REQUEST_TIMEOUT_SECONDS}s"
        ) from exc
    except httpx.HTTPError as exc:
        raise LLMUnavailable(f"{provider.name}: {type(exc).__name__}") from exc

    _raise_for_status(provider, response)

    try:
        body = response.json()
        message = body["choices"][0]["message"]
    except (ValueError, KeyError, IndexError) as exc:
        raise LLMUnavailable(f"{provider.name}: unreadable response shape") from exc

    tool_calls = [
        ToolCall(
            id=tc["id"],
            name=tc["function"]["name"],
            arguments=tc["function"].get("arguments") or "{}",
            extra={k: v for k, v in tc.items() if k not in _KNOWN_TOOL_CALL_KEYS},
        )
        for tc in (message.get("tool_calls") or [])
    ]

    _log_if_truncated(
        provider,
        (body.get("choices") or [{}])[0].get("finish_reason"),
        message.get("content"),
    )

    usage = body.get("usage") or {}
    return Completion(
        content=message.get("content"),
        tool_calls=tool_calls,
        provider=provider.name,
        model=provider.model,
        prompt_tokens=usage.get("prompt_tokens", 0),
        cached_tokens=_cached_tokens(usage),
    )


# -- The same call, streamed ------------------------------------------

async def stream(
    provider: Provider,
    messages: list,
    tools: list,
    tool_choice: str = "auto",
    on_text=None,
) -> Completion:
    """One chat-completion call, delivered incrementally.

    RETURNS THE SAME Completion complete() DOES. That is the whole
    design: the orchestrator's loop is unchanged and cannot tell which
    transport it used, so streaming is a delivery detail rather than a
    second implementation of the conversation. on_text is called with
    each text fragment as it arrives; text also accumulates into the
    returned Completion, so a caller that ignores the callback gets
    exactly the buffered behaviour.

    WHAT STREAMING BUYS, AND WHERE. Only the TEXT turns benefit - a
    round that comes back as tool calls produces nothing a user could
    read. That is fine, because those rounds are where the orchestrator
    emits its own progress events instead: the silence during a lookup
    is filled by saying what is being looked up.

    FAILOVER STOPS AT THE FIRST TOKEN. An HTTP error arrives before any
    body does, so a dead provider still falls through to the next one
    exactly as before. Once bytes have reached the user they cannot be
    recalled, so a mid-stream failure ends the answer rather than
    silently restarting it somewhere else and repeating half a sentence.

    TOOL CALLS ARRIVE IN PIECES. The id and name land in the first
    fragment for an index and the arguments accumulate across many, so
    they are reassembled by index here - a partial arguments string is
    not valid JSON and must never reach the orchestrator's parser.
    """
    payload = _build_payload(provider, messages, tools, tool_choice)
    payload["stream"] = True
    # Usage is otherwise absent from a streamed response, and the
    # prompt-token count is what tells us the cache is working. Providers
    # that do not know this option ignore it.
    payload["stream_options"] = {"include_usage": True}

    url, headers, body = build_request(provider, payload)

    content_parts: list[str] = []
    partial_calls: dict[int, dict] = {}
    finish_reason = None
    prompt_tokens = 0
    cached_tokens = 0

    try:
        async with _get_http_client().stream(
            "POST", url, headers=headers, json=body
        ) as response:
            if response.status_code != 200:
                # The body has not been read yet in streaming mode, and
                # _raise_for_status needs it to explain what went wrong.
                await response.aread()
                _raise_for_status(provider, response)

            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if not data or data == "[DONE]":
                    continue

                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    # A single unreadable frame is not worth abandoning a
                    # good answer for.
                    logger.warning("stream_chunk_unparseable", provider=provider.name)
                    continue

                usage = chunk.get("usage") or {}
                if usage.get("prompt_tokens"):
                    prompt_tokens = usage["prompt_tokens"]
                    cached_tokens = _cached_tokens(usage)

                choices = chunk.get("choices") or []
                if not choices:
                    # The usage-only frame some providers send last.
                    continue
                choice = choices[0]
                finish_reason = choice.get("finish_reason") or finish_reason

                delta = choice.get("delta") or {}
                text = delta.get("content")
                if text:
                    content_parts.append(text)
                    if on_text is not None:
                        on_text(text)

                for fragment in delta.get("tool_calls") or []:
                    index = fragment.get("index", 0)
                    call = partial_calls.setdefault(
                        index, {"id": None, "name": None, "arguments": [], "extra": {}}
                    )
                    if fragment.get("id"):
                        call["id"] = fragment["id"]
                    function = fragment.get("function") or {}
                    if function.get("name"):
                        call["name"] = function["name"]
                    if function.get("arguments"):
                        call["arguments"].append(function["arguments"])
                    call["extra"].update(
                        {
                            k: v
                            for k, v in fragment.items()
                            if k not in _KNOWN_TOOL_CALL_KEYS
                        }
                    )
    except httpx.TimeoutException as exc:
        raise LLMUnavailable(
            f"{provider.name}: timed out after {REQUEST_TIMEOUT_SECONDS}s"
        ) from exc
    except httpx.HTTPError as exc:
        raise LLMUnavailable(f"{provider.name}: {type(exc).__name__}") from exc

    content = "".join(content_parts)
    _log_if_truncated(provider, finish_reason, content)

    tool_calls = [
        ToolCall(
            id=call["id"] or f"call_{index}",
            name=call["name"] or "",
            arguments="".join(call["arguments"]) or "{}",
            extra=call["extra"],
        )
        for index, call in sorted(partial_calls.items())
        # A fragment that never carried a name is not a callable
        # request - dropping it beats handing the orchestrator a tool
        # call it cannot look up.
        if call["name"]
    ]

    return Completion(
        content=content or None,
        tool_calls=tool_calls,
        provider=provider.name,
        model=provider.model,
        prompt_tokens=prompt_tokens,
        cached_tokens=cached_tokens,
    )
