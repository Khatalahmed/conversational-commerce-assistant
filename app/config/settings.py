"""
WHAT:
    This file defines the Settings class — the ONE place in the whole
    project that reads values from the .env file. Every other file that
    needs a secret (MongoDB URI, LLM API key, JWT secret) will import
    from here instead of reading environment variables directly.

WHY THIS APPROACH:
    If every file read os.environ on its own, a typo in a variable name
    would fail silently deep inside some feature, maybe only when a real
    user hits that exact code path. By centralizing it into one class
    with required fields, the app instead fails LOUDLY and IMMEDIATELY
    at startup, naming exactly which value is missing or empty.

FLOW:
    1. App starts up.
    2. Something calls get_settings() for the first time.
    3. pydantic-settings reads the .env file and tries to fill in every
       field declared below.
    4. If a required field is missing or empty, this raises a clear
       error right here, before the app does anything else.
    5. If everything is present, get_settings() returns one Settings
       object that every other module can safely reuse.

LOGIC:
    A field is only accepted if it exists AND is not just blank/whitespace.
    Right now (Phase 1), all three values are intentionally still empty in
    .env — so running this file right now is EXPECTED to fail. That failure
    is proof the safety check works. We'll fill in real values as we reach
    Phase 2 (MongoDB), Phase 3 (JWT secret), and Phase 6 (LLM key).

MECHANISM:
    - BaseSettings (from pydantic-settings) automatically matches each
      field name to an environment variable of the same name, uppercased
      (e.g. field "mongodb_uri" reads the .env line "MONGODB_URI=...").
    - @field_validator adds our own extra rule: reject empty/whitespace
      values, since pydantic alone would accept an empty string as valid.
    - @lru_cache makes get_settings() build the Settings object only
      once, then reuse it — avoids re-reading the .env file on every call.
"""

from datetime import date
from functools import lru_cache

# HS256 signs with the secret directly, so 256 bits of entropy is the
# floor worth having - the same width as the hash it feeds.
MIN_HS_SECRET_CHARS = 32
# Catches "aaaaaaaa...": long, and worth almost nothing.
MIN_HS_SECRET_DISTINCT_CHARS = 8

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Algorithms we are willing to verify with. "none" is absent and must stay
# absent: it means "unsigned", and accepting it would let anyone hand us a
# token claiming to be any user at all.
#
# A SINGLE algorithm is configured rather than a list, to close the
# algorithm-confusion attack: if we accepted both RS256 and HS256 at once,
# an attacker could take the backend's PUBLIC key (which is not secret) and
# use it as an HMAC key to forge a token we would then happily verify.
ALLOWED_JWT_ALGORITHMS = frozenset(
    {"HS256", "HS384", "HS512", "RS256", "RS384", "RS512", "ES256", "ES384"}
)


class Settings(BaseSettings):
    # Tells pydantic-settings WHERE to read values from, and to ignore
    # any .env variables we haven't explicitly declared as a field below
    # (keeps this class as the single source of truth for what's "real").
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # The three secrets we currently expect. "..." means "required" —
    # pydantic will error if this field ends up unset after reading .env.
    mongodb_uri: str = Field(
        ...,
        description="MongoDB Atlas connection string (read-only service account).",
    )
    # OPTIONAL since Azure became the provider. It was required back when
    # Groq WAS the only provider; keeping it required would make
    # Azure-only - the configuration we actually want while real customer
    # data is in play - impossible to express.
    llm_api_key: str | None = Field(
        default=None,
        description="API key for Groq. Optional: leave blank to run on Azure alone.",
    )
    jwt_secret: str = Field(
        ...,
        description="Secret used to verify JWTs issued by the main the marketplace backend.",
    )

    # Which database inside the cluster to read. Configurable so a DEMO
    # dataset can be swapped in for anything that sends data to an LLM,
    # without touching code. Defaults to the real one.
    mongodb_database: str = Field(
        default="marketplace",
        description="Database name inside the cluster (e.g. 'marketplace' or 'marketplace_demo').",
    )

    # WHICH database counts as production, and therefore triggers the
    # provider filter in llm_client.get_providers().
    #
    # A SETTING RATHER THAN A LITERAL, because this value and
    # MONGODB_DATABASE have to agree. Hardcoded, the two drift the
    # moment either is renamed - and the filter then silently stops
    # matching, which is precisely the failure the filter exists to
    # prevent. Wrong here, the service still starts and still answers;
    # it just quietly stops protecting anything.
    production_database_name: str = Field(
        default="marketplace",
        description="Database holding real customer records. While "
                    "MONGODB_DATABASE equals this, providers whose terms "
                    "permit training on prompts are dropped from the chain.",
    )

    # OPTIONAL. The DEMO cluster, kept alongside the real one.
    #
    # WHY BOTH ARE STORED. MONGODB_URI holds one cluster at a time, so
    # switching to real data overwrote the demo URI and it was simply
    # lost - three separate attempts to run a script against "the demo
    # cluster" failed because the value no longer existed anywhere, and
    # the real URI got passed instead. Twice a guard caught it; the
    # third time it was the read-only account that did.
    #
    # Nothing reads this at runtime. It exists so the value survives a
    # switch, and so scripts can offer it rather than asking someone to
    # find it again.
    demo_mongodb_uri: str | None = Field(
        default=None,
        description="The writable DEMO cluster URI. Not used at runtime - "
                    "kept so switching MONGODB_URI does not lose it.",
    )

    # OPTIONAL. A shared code the demo surface demands before it shows
    # anything.
    #
    # WHY IT EXISTS. /demo/accounts lists real buyers and /demo/token
    # signs in as any of them with no password. On the machine running
    # the demo that grants nothing - whoever opens it already has .env.
    # Behind a URL it is a customer directory with one-click
    # impersonation, which is the single thing standing between a
    # shareable link and a privacy incident.
    #
    # UNSET = NO GATE, so local development and laptop demos are
    # unchanged. Set it for anything reachable, and hand the code to
    # whoever should get in alongside the link.
    #
    # Not a password and not per-user: one shared secret, compared in
    # constant time, that turns "anyone who finds the URL" into "anyone
    # you told". That is the whole claim - it does not make the data
    # less real, and a leaked code is a leaked door.
    demo_access_code: str | None = Field(
        default=None,
        description="Shared code required by /demo before it shows accounts. "
                    "Unset = no gate (local use). Set it for any reachable URL.",
    )

    # OPTIONAL. Whether to serve the demo chat page at /demo.
    #
    # DEFAULT ON, so the demo and local development need no extra
    # configuration - that page IS the product as far as a client
    # watching is concerned, and a flag that has to be found before
    # anything can be shown is a flag that wastes the first ten minutes.
    #
    # TURN IT OFF FOR A PUBLIC DEPLOYMENT. It cannot answer without a
    # valid the marketplace token, so it is an unnecessary surface rather than a
    # leak: it advertises the service to anyone who finds the URL and
    # invites pasting a bearer token into a web page. The app logs which
    # state it is in at every startup, because leaving it on is
    # otherwise invisible.
    demo_ui_enabled: bool = Field(
        default=True,
        description="Serve the demo chat page at /demo. Set false for public "
                    "deployments.",
    )

    # WHAT TURNS A PRODUCT PHOTO INTO A THUMBNAIL, and why this is empty.
    #
    # Product cards show a 46px image. The catalogue's images are the
    # full-resolution originals - a measured one is 901 KB - so eight
    # cards is roughly 7 MB for a single answer, on a phone, in a market
    # where mobile data is the cost that matters.
    #
    # Bunny's Optimizer would fix it at the CDN with ?width=96, but it is
    # NOT currently enabled on the pull zone: ?width=96, ?quality=75 and
    # ?class=thumb all return the identical 901,302 bytes. So there is
    # nothing useful to append yet, and appending a parameter the CDN
    # ignores would only look like it worked.
    #
    # EMPTY MEANS NO IMAGE IS SERVED AT ALL - it is a switch, not just a
    # suffix. Measured after the first version shipped the originals: one
    # product image is 8.5 MB, a 6144x8192 photograph, drawn at 46 pixels,
    # and one trending answer pulled roughly 10 MB for eight cards.
    #
    # Off by default because that failure is silent. The cards render,
    # the app works, and the only symptom is somebody's data plan. A
    # picture is worth having only once it can be sent small.
    #
    # Once the Optimizer is switched on, turning images back on is a
    # config change rather than a code one - set it to
    # "?width=96&quality=75" and every card gets light without a redeploy.
    product_image_query: str = Field(
        default="",
        description="Query string appended to product image URLs to request a "
                    "CDN-resized thumbnail, e.g. '?width=96&quality=75'. EMPTY "
                    "MEANS NO IMAGES ARE SERVED - the catalogue holds "
                    "full-resolution originals (one measured at 8.5MB), so "
                    "leave this empty until Bunny Optimizer is enabled on the "
                    "pull zone.",
    )

    # OPTIONAL. Where state SHARED BETWEEN WORKERS lives - conversation
    # history and the /chat rate-limit counts both.
    #
    # Unset means per-process: history is wiped by every restart or
    # deploy, and a user mid-conversation suddenly gets "which product?"
    # out of nowhere. With several workers both break silently - a
    # follow-up can land on a worker that never saw the first message,
    # and each worker grants the full rate-limit allowance, so the real
    # limit is (workers x chat_rate_limit_requests).
    #
    # Optional rather than required because a POC must run on nothing but
    # a MongoDB URI. The app logs loudly at startup when it falls back.
    redis_url: str | None = Field(
        default=None,
        description="Redis URL for state shared across workers - conversation "
                    "memory and rate-limit counts (e.g. rediss://...upstash.io:6379). "
                    "Unset = per-process.",
    )

    # ── Azure OpenAI (optional, becomes the PRIMARY provider) ────────
    # All four are required together - an endpoint without a deployment
    # name is unusable, because on Azure the deployment is part of the
    # request PATH rather than a "model" field in the body.
    #
    # Preferred over Groq when configured, for three measured reasons:
    # 100,000 TPM instead of 8,000; a contractual commitment that prompts
    # are not used to train models (which is what makes demoing against
    # real customer data defensible at all); and a frontier-lab model
    # rather than an open-weight one, which is where the malformed tool
    # calls were coming from.
    azure_openai_endpoint: str | None = Field(
        default=None,
        description="e.g. https://your-resource.openai.azure.com/",
    )
    azure_openai_api_key: str | None = Field(default=None)
    azure_openai_deployment: str | None = Field(
        default=None,
        description="The DEPLOYMENT name, not the model name - they can differ.",
    )
    # OPTIONAL. When the Azure credit or subscription lapses.
    #
    # Recorded because the consequence is sharp and silent: the provider
    # filter drops Groq and Gemini whenever real customer data is
    # configured, so on the day this expires there is no fallback left -
    # every request returns "I can't reach my assistant service". That is
    # the correct behaviour (far better than quietly shipping customer
    # orders to a provider that trains on them) but it is a total outage,
    # and nobody should first learn about it during a demo.
    #
    # ISO date, e.g. 2030-01-01. Unset disables the warning.
    azure_credit_expires: str | None = Field(
        default=None,
        description="ISO date the Azure credit runs out, e.g. 2030-01-01.",
    )

    # ── Azure embeddings (optional, enables free-text semantic search) ──
    # A SEPARATE deployment from the chat one - embeddings and chat are
    # different models and must each be deployed under their own name.
    #
    # Without this, find_similar_products still works: it reuses a
    # product's STORED vector as the query, which needs no model. Only
    # free-text search ("something cosy for winter") needs to embed the
    # query, and therefore needs to match whatever embedded the
    # catalogue.
    azure_embedding_deployment: str | None = Field(
        default=None,
        description="Azure deployment name for text-embedding-3-small.",
    )
    azure_embedding_dimensions: int = Field(
        default=1536,
        description="Vector width. MUST match the Atlas vector index and the "
                    "stored documents - a mismatch is rejected by Atlas.",
    )

    azure_openai_api_version: str = Field(
        default="2024-12-01-preview",
        description="Azure pins API behaviour to a dated version string.",
    )

    # ── Local embeddings (optional, for matching a legacy catalogue) ───
    # WHY THIS EXISTS. The catalogue in MONGODB_DATABASE was embedded by
    # the backend's original pipeline: 384 dimensions, unit-normalised,
    # and never identified in code or docs. Azure's text-embedding-3-small
    # cannot produce those vectors at any width, so free-text search over
    # that catalogue is impossible from Azure alone.
    #
    # There are two ways out, and this is the one that needs no write
    # access: run the SAME model the pipeline ran, locally, for queries
    # only. The catalogue is left exactly as it is.
    #
    # DO NOT SET THESE FROM A GUESS. scripts/identify_embedding_model.py
    # re-embeds known products and compares against their stored vectors;
    # only a ~0.99 match identifies a model. A wrong model here does not
    # error - it returns ranked nonsense, which is the entire failure
    # mode the embeddingModel tag exists to prevent.
    embedding_backend: str = Field(
        default="azure",
        description="Which model embeds QUERIES: 'azure' or 'local'.",
    )
    local_embedding_model: str | None = Field(
        default=None,
        description="sentence-transformers model id, e.g. "
                    "sentence-transformers/all-MiniLM-L6-v2. Required when "
                    "EMBEDDING_BACKEND=local.",
    )
    local_embedding_dimensions: int = Field(
        default=384,
        description="Vector width of the local model. MUST match the Atlas "
                    "vector index and the stored documents.",
    )

    # THE ESCAPE HATCH, AND WHY IT IS DELIBERATELY AWKWARD.
    # The guard compares each stored document's embeddingModel tag
    # against ours. The legacy documents have NO such field, so they can
    # never match - and with a read-only credential they cannot be
    # tagged. This setting is the operator asserting, out of band, "I ran
    # the identification script and untagged documents in this database
    # are THIS model".
    #
    # It is a full tag string rather than a boolean on purpose: writing
    # local:sentence-transformers/all-MiniLM-L6-v2:384 is a claim about a
    # specific model, so it stops matching the moment the configured
    # model changes. A boolean would silently keep permitting a search
    # after someone swapped the model underneath it.
    legacy_embedding_assumed: str | None = Field(
        default=None,
        description="Tag to assume for UNTAGGED embedding documents. Only "
                    "set from a confirmed identify_embedding_model.py run.",
    )

    @field_validator("embedding_backend")
    @classmethod
    def _known_backend(cls, value: str) -> str:
        allowed = {"azure", "local"}
        cleaned = (value or "").strip().lower()
        if cleaned not in allowed:
            raise ValueError(
                f"EMBEDDING_BACKEND must be one of {sorted(allowed)}, got {value!r}"
            )
        return cleaned

    # ── Backup provider (optional) ───────────────────────────────────
    # EXISTS BECAUSE AZURE IS OTHERWISE A SINGLE POINT OF FAILURE WITH A
    # DATE ON IT. While MONGODB_DATABASE holds real customers, Groq and
    # Gemini are dropped from the chain (they may train on prompts), so
    # Azure is the only provider left. When its credit lapses, every
    # request returns "I can't reach my assistant service".
    #
    # Before these settings existed, the fix was a CODE change - a new
    # Provider(...) in llm_client.py - which is the worst possible thing
    # to need on the morning it happens. Now it is four lines of .env
    # and a restart.
    #
    # Any OpenAI-COMPATIBLE chat-completions endpoint works, which is
    # most of them: OpenAI's own API, a second Azure resource in another
    # region or subscription, AWS Bedrock's compatible endpoint, Google
    # Vertex. Set BACKUP_LLM_KIND=azure for an Azure-shaped one (the
    # deployment goes in the path, auth is the api-key header) and give
    # it BACKUP_LLM_API_VERSION too.
    backup_llm_base_url: str | None = Field(
        default=None,
        description="OpenAI-compatible base URL for the backup provider.",
    )
    backup_llm_api_key: str | None = Field(default=None)
    backup_llm_model: str | None = Field(
        default=None,
        description="Model name, or the DEPLOYMENT name when kind is azure.",
    )
    backup_llm_kind: str = Field(
        default="openai",
        description="'openai' for any OpenAI-compatible endpoint, 'azure' for Azure.",
    )
    backup_llm_api_version: str | None = Field(
        default=None,
        description="Required only when backup_llm_kind is 'azure'.",
    )
    # DEFAULTS TO TRUE, AND THAT DEFAULT IS THE SAFE ONE.
    #
    # This is a contractual claim about someone else's terms, and we
    # cannot verify it. Assuming "does not train" by default would mean a
    # mistyped or half-understood .env silently ships real customer
    # orders to a provider that reserves the right to learn from them -
    # the single failure this whole design exists to prevent.
    #
    # So the operator has to assert it explicitly. Left alone, the backup
    # is treated exactly like Groq: usable for the demo dataset, dropped
    # the moment real customer data is configured.
    backup_llm_trains_on_prompts: bool = Field(
        default=True,
        description="Set false ONLY if the provider's terms exclude training on "
                    "submitted content. Left true, the backup is dropped whenever "
                    "MONGODB_DATABASE holds real customer data.",
    )

    # ── JWT verification shape ───────────────────────────────────────
    # These describe the tokens the MAIN PLATFORM BACKEND issues. We do not
    # get to choose them - they have to match whatever that backend
    # already does, and until the team confirms, the defaults below are
    # our best guess. They are settings rather than constants in code
    # precisely BECAUSE they are a guess: when the real answer arrives it
    # is a one-line .env change, not a code change and redeploy.
    jwt_algorithm: str = Field(
        default="HS256",
        description="Signing algorithm used by the marketplace backend. HS256 = shared "
                    "secret; RS256/ES256 = JWT_SECRET holds their PUBLIC key (PEM).",
    )
    jwt_user_id_claims: str = Field(
        default="user_id,userId,sub,_id",
        description="Comma-separated claim names to look for the user ID in, in "
                    "priority order. The first one PRESENT in the token wins.",
    )
    jwt_audience: str | None = Field(
        default=None,
        description="Expected 'aud' claim. When set, a token minted for a "
                    "different audience is rejected. Unset = not checked.",
    )
    jwt_issuer: str | None = Field(
        default=None,
        description="Expected 'iss' claim. When set, a token from a different "
                    "issuer is rejected. Unset = not checked.",
    )

    # "PRESENT BUT BLANK" MEANS UNSET, NOT EMPTY. .env.example ships
    # these as bare "JWT_AUDIENCE=" lines, so a copied file yields "",
    # and "" is not None - without this, verify_token would switch
    # audience checking ON with an empty expected value and reject every
    # token the backend ever sends.
    @field_validator(
        "jwt_audience", "jwt_issuer", "redis_url",
        "azure_openai_endpoint", "azure_openai_api_key", "azure_openai_deployment",
        mode="before",
    )
    @classmethod
    def blank_means_unset(cls, value):
        if value is None or not str(value).strip():
            return None
        return str(value).strip()

    # Same reasoning, different remedy: a blank algorithm or claim list
    # falls back to the documented default rather than erroring, so a
    # half-filled .env still starts.
    @field_validator("jwt_algorithm", "jwt_user_id_claims", mode="before")
    @classmethod
    def blank_means_default(cls, value, info):
        if value is None or not str(value).strip():
            return cls.model_fields[info.field_name].default
        return value

    @field_validator("jwt_algorithm")
    @classmethod
    def known_algorithm(cls, value: str) -> str:
        normalized = (value or "").strip().upper()
        if normalized not in ALLOWED_JWT_ALGORITHMS:
            raise ValueError(
                f"JWT_ALGORITHM '{value}' is not supported. Use one of: "
                f"{', '.join(sorted(ALLOWED_JWT_ALGORITHMS))}. "
                f"('none' is refused on purpose - it means unsigned.)"
            )
        return normalized

    @field_validator("mongodb_database")
    @classmethod
    def clean_database_name(cls, value: str) -> str:
        """Strips whitespace and a trailing inline comment.

        WHY A SETTING NEEDS DEFENDING FROM ITS OWN .env FILE. The two
        readers of that file do not agree. python-dotenv, which
        pydantic-settings uses, strips an inline comment - so

            MONGODB_DATABASE=marketplace          # or marketplace_demo

        reads as "marketplace" locally. Docker's `--env-file` has no such
        notion and passes the entire remainder of the line, so the
        container receives "marketplace          # or marketplace_demo" and the same
        file means two different things depending on how the app was
        started.

        THAT IS NOT MERELY UNTIDY. This exact value decides whether Groq
        and Gemini are dropped from the provider chain: the filter in
        llm_client.get_providers() compares it to "marketplace" to decide
        whether the database holds real customers. A mangled value is
        not equal to "marketplace", so a service genuinely pointed at real
        customer orders would classify itself as demo data and keep the
        providers whose terms permit training on prompts.

        Measured, the first failure was loud - pymongo rejects a
        database name containing a space - and that is luck, not design.
        Normalising here means the safety property depends on what was
        MEANT rather than on which parser happened to read the file.
        """
        cleaned = value.split("#", 1)[0].strip()
        if not cleaned:
            raise ValueError(
                "MONGODB_DATABASE is empty. It decides whether the app treats "
                "the data as real customers, so it cannot be guessed."
            )
        return cleaned

    @field_validator("backup_llm_kind")
    @classmethod
    def known_backup_kind(cls, value: str) -> str:
        normalized = (value or "openai").strip().lower()
        if normalized not in ("openai", "azure"):
            raise ValueError(
                f"BACKUP_LLM_KIND '{value}' is not supported. Use 'openai' for any "
                f"OpenAI-compatible endpoint, or 'azure' for Azure OpenAI."
            )
        return normalized

    @field_validator("jwt_user_id_claims")
    @classmethod
    def at_least_one_claim(cls, value: str) -> str:
        if not [c.strip() for c in (value or "").split(",") if c.strip()]:
            raise ValueError(
                "JWT_USER_ID_CLAIMS is empty - there would be no way to tell "
                "which user a verified token belongs to."
            )
        return value

    @property
    def azure_openai_configured(self) -> bool:
        """All three secrets present. Partial configuration is treated as
        NOT configured rather than as an error: a half-filled .env should
        fall back to the other providers, not refuse to start."""
        return bool(
            self.azure_openai_endpoint
            and self.azure_openai_api_key
            and self.azure_openai_deployment
        )

    @property
    def jwt_secret_weakness(self) -> str | None:
        """Why this signing secret is too weak, or None if it is fine.

        WHAT IS AT STAKE. The secret verifies every token, and a token is
        the only thing that says who is asking. Recover it and you can
        mint a token for any user id and read that customer's orders,
        cart and delivery city - the field allowlist and the sanitizer
        are irrelevant, because the request is indistinguishable from
        the real user's. It is the one credential where a weakness
        defeats every other layer at once.

        HS256 signs with this value directly, so its strength IS the
        secret's length and variety; the usual guidance is at least as
        much entropy as the hash output, 256 bits, i.e. 32 bytes.

        NOT CHECKED FOR RS256/ES256, where JWT_SECRET holds the
        backend's PUBLIC key. That is not secret at all - it is meant to
        be published - so length tells you nothing about safety.

        REPORTED, NOT ENFORCED - see log_secret_preflight(). This value
        has to match the main the marketplace backend exactly, so it is not ours
        to change. Refusing to start would take the assistant down over
        a decision made in another team's codebase, which is a worse
        outcome than saying so loudly at every startup.
        """
        if not self.jwt_algorithm.startswith("HS"):
            return None

        secret = self.jwt_secret.strip()
        if len(secret) < MIN_HS_SECRET_CHARS:
            return (
                f"JWT_SECRET is {len(secret)} characters; {MIN_HS_SECRET_CHARS} or "
                f"more is expected for {self.jwt_algorithm}. A short secret can be "
                f"brute-forced offline, and recovering it means being able to forge "
                f"a token for any user."
            )
        if len(set(secret)) < MIN_HS_SECRET_DISTINCT_CHARS:
            return (
                f"JWT_SECRET is long but uses only {len(set(secret))} distinct "
                f"characters, so it has far less entropy than its length suggests."
            )
        return None

    @property
    def backup_llm_configured(self) -> bool:
        """All three essentials present, same partial-configuration rule
        as azure_openai_configured: half-filled is treated as absent
        rather than as an error."""
        return bool(
            self.backup_llm_base_url
            and self.backup_llm_api_key
            and self.backup_llm_model
        )

    @property
    def credit_days_remaining(self) -> int | None:
        """Days until AZURE_CREDIT_EXPIRES, negative once past, None when
        unset or unparseable.

        Lives here rather than in scripts/check_azure.py, which is where
        it used to live alone: a date only the pre-flight script knows
        about is a date the running service cannot warn anybody about.
        """
        if not self.azure_credit_expires:
            return None
        try:
            expires = date.fromisoformat(self.azure_credit_expires.strip())
        except ValueError:
            return None
        return (expires - date.today()).days

    @property
    def jwt_user_id_claim_list(self) -> tuple[str, ...]:
        """The claim names, parsed and cleaned, in priority order."""
        return tuple(c.strip() for c in self.jwt_user_id_claims.split(",") if c.strip())

    # ── /chat rate limiting ──────────────────────────────────────────
    # Bounds ONE user's request rate, keyed on the verified JWT subject.
    # See security/rate_limit.py for why this is per-user rather than
    # per-IP, and what it deliberately does not protect against.
    #
    # The default is generous for a person - a human conversation is a
    # handful of messages a minute - while still stopping a retry loop
    # or a leaked token from issuing thousands. Set to 0 to disable.
    chat_rate_limit_requests: int = Field(
        default=20,
        description="Max /chat requests per user per window. 0 disables the limiter.",
    )
    chat_rate_limit_window_seconds: int = Field(
        default=60,
        description="Length of the rolling rate-limit window, in seconds.",
    )

    @field_validator("chat_rate_limit_window_seconds")
    @classmethod
    def positive_window(cls, value: int) -> int:
        if value <= 0:
            raise ValueError(
                "CHAT_RATE_LIMIT_WINDOW_SECONDS must be positive. To turn the "
                "limiter off, set CHAT_RATE_LIMIT_REQUESTS=0 instead."
            )
        return value

    # OPTIONAL, and deliberately so. This is the FALLBACK provider's key —
    # the app must still start without it, running on Groq alone, because
    # a fallback that is mandatory to boot is not a fallback. Note it is
    # also absent from the not_empty validator below: "unset" and "set but
    # blank" both mean the same thing here (no fallback configured), and
    # neither is an error worth refusing to start over.
    gemini_api_key: str | None = Field(
        default=None,
        description="API key for the fallback LLM provider (Google AI Studio, free tier). "
                    "Optional — when unset, the app runs on the primary provider only.",
    )

    # Extra validation layer: catches the case where a variable EXISTS in
    # .env but is left blank (e.g. "MONGODB_URI=" with nothing after it) —
    # which plain pydantic would otherwise accept as a valid empty string.
    # gemini_api_key is intentionally NOT listed — see its comment above.
    @field_validator("mongodb_uri", "jwt_secret")
    @classmethod
    def not_empty(cls, value: str, info):
        if not value or not value.strip():
            raise ValueError(
                f"'{info.field_name.upper()}' is present in .env but EMPTY. "
                f"Fill it in before starting the app."
            )
        return value


# Cached so the .env file is only parsed once per app run, and every
# module that calls get_settings() gets back the exact same object.
@lru_cache
def get_settings() -> Settings:
    return Settings()