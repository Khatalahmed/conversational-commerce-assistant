# Conversational Commerce Assistant

A read-only conversational assistant that sits on top of a commerce
platform's live MongoDB and answers questions in natural language —
*"where's my order?"*, *"any blue jackets under ₹2000?"*,
*"can I bargain on this?"*

The client sends a message plus the user's existing JWT; the service
resolves that against real data through 35 read-only tools and answers
from what they return. Nothing is ever written.

> **About this repository.** This is a de-identified version of a system
> built for a client. The platform name, database identifiers, customer
> records and operational documents have been removed or replaced with
> synthetic equivalents. The engineering is unchanged. Published to show
> how it was built — not licensed for reuse.

---

## The parts worth reading

Most of this is ordinary FastAPI and MongoDB. Four things are not.

### Every claim in an answer is traced back to a query

An LLM that reads real order data will eventually state a number that
came from nowhere. [`agent/grounding.py`](app/agent/grounding.py) walks
the finished reply against the actual tool payloads and reports what
cannot be accounted for: an order ID that no lookup returned, a price
that matches nothing, a promise to perform an action the assistant
cannot perform.

It also emits **attribution** — character offsets into the reply paired
with the tool that vouched for each span, so a client can underline the
evidence:

```json
{"start": 12, "end": 18, "text": "₹1,199", "kind": "price", "tool": "get_order_history"}
```

Offsets rather than the matched text, because a reply that says "₹599"
twice would otherwise mark both occurrences from one match. Only prices
and order IDs are attributed — a product *name* cannot be located in
prose without NER, so the honest thing is to mark what can actually be
stood behind.

### Identity is not a model parameter

`user_id` appears in **none** of the 35 tool schemas. The model has no
vocabulary in which to request another user's data; the verified JWT
subject is injected server-side in
[`agent/tool_executor.py`](app/agent/tool_executor.py).

This is enforced, not documented:
[`test_tool_registry.py`](app/tests/test_tool_registry.py) fails the
build on a schema with no implementation, an implementation with no
schema, or any schema that lets the model name *whose* data it wants.

### The provider chain follows the data

Multi-provider failover is standard advice — until you notice the
fallback is the hole. The primary provider was contractually excluded
from training on prompts. The fallbacks were not. Everything worked
perfectly right up to the first rate limit, at which point failover
would have handed real customer records to a provider permitted to
train on them.

Being *first* in a chain is a guarantee about ordering, not about
destination. So the chain is filtered by the data it will carry:

```python
if settings.mongodb_database == settings.production_database_name:
    eligible = tuple(p for p in providers if not p.trains_on_prompts)
```

Point the service at production data and non-compliant providers are
removed entirely — it refuses to start rather than degrade unsafely.
Point it at the synthetic dataset and the cheap providers come back.
Keyed on the database rather than a flag, so it cannot be forgotten.

### Three independent data-safety layers

1. **Read-only role** at the database user — not just in code.
2. **Field allowlist used as the Mongo projection**, so disallowed
   fields never leave the database
   ([`security/field_allowlist.py`](app/security/field_allowlist.py)).
3. **An independent sanitizer** that re-strips anything not allowlisted
   on the way out, and *raises* on an unknown collection rather than
   passing it through ([`security/sanitizer.py`](app/security/sanitizer.py)).

Tool results are also treated as data, never instructions — product
names, descriptions and comments are written by sellers and shoppers.
See [`test_prompt_injection.py`](app/tests/test_prompt_injection.py).

---

## How a request flows

```
POST /chat  (Bearer JWT)
   │
   ├─ verify token ───────────────► user_id            security/auth.py
   ├─ rate limit (Redis)                               security/rate_limit.py
   ├─ load session history                             memory/session_store.py
   ▼
run_conversation                                       agent/orchestrator.py
   │
   └─ up to 3 tool rounds:
        ├─ LLM call, failover across providers         agent/llm_client.py
        └─ tool calls for a round run concurrently     asyncio.gather
              ├─ user_id injected server-side          agent/tool_executor.py
              ├─ repo query: projection = allowlist    repos/*.py
              ├─ sanitizer strips anything else        security/sanitizer.py
              └─ trim + enrich for the token budget
      ▼
   synthesis turn (tools withheld, must answer in words)
      ▼
   grounding + attribution ──────► reply, product cards, claims
```

**Why a synthesis turn.** The loop appends tool *results* and then
re-checks its condition, so on the final iteration the model never saw
what it just asked for — the query was paid for and then apologised
over. Prompting for fewer tool calls was tried first and measured as
useless. Fixing the harness worked: N rounds of tools, plus a turn to
speak.

### Two endpoints, one implementation

| Endpoint | Shape |
|---|---|
| `POST /chat` | JSON in, JSON out |
| `POST /chat/stream` | JSON in, Server-Sent Events out |

Both call the same `run_conversation`; the only difference is that the
streaming one passes a callback. There is no second orchestration loop
to keep in step, which is the failure mode this shape exists to avoid.

An answer cannot begin until the lookups finish, so streaming *tokens*
does nothing for the first several seconds. What fills them is `status`
events naming each lookup as it runs — the wait is filled by saying
what is being looked up rather than by a spinner.

---

## Layout

```
app/
├── agent/       tool-calling loop, 35 tool schemas, provider client,
│                grounding/attribution, embeddings
├── api/         FastAPI app, /chat + /chat/stream + /health, demo UI
├── config/      settings loader, structured logging with redaction
├── db/          shared async MongoDB and Redis connections
├── evals/       golden-case runner and an LLM judge
├── memory/      per-session conversation history (Redis or in-process)
├── repos/       data access, one file per collection, allowlisted
├── security/    JWT verification, field allowlists, sanitizer, rate limit
└── tests/       526 tests
scripts/         dev-only helpers, not part of the deployed app
```

### Deliberately not reachable from chat

Not everything in `repos/` is wired to a tool, and the gaps are choices:

| Code | Why it is not a tool |
|---|---|
| `seller_repo` | The product surface is buyer-side end to end. Four more schemas would be re-sent on every round for a capability nothing reaches. The file documents what wiring it up safely requires — chiefly that `seller_id` must never become a model parameter. |
| `categories_repo` | The assistant needs real category names, but gets them from the system prompt, which is cheaper than a tool round-trip. |

---

## Running it

```bash
uv sync
cp .env.example .env      # then fill in the values
uv run uvicorn app.api.main:app --reload
```

Interactive docs at `/docs`, a chat surface at `/demo`.

| Variable | Required | Purpose |
|---|---|---|
| `MONGODB_URI` | yes | Connection string (read-only service account) |
| `JWT_SECRET` | yes | Must match the issuing backend's signing secret |
| `LLM_API_KEY` | yes* | Primary provider |
| `AZURE_OPENAI_*` | no | Preferred provider; excluded from training on prompts |
| `BACKUP_LLM_*` | no | Any other OpenAI-compatible endpoint |
| `REDIS_URL` | no | Shared session history and rate-limit counts |
| `DEMO_UI_ENABLED` | no | Serves `/demo`. Defaults **true**; set false for anything public |

\* at least one provider must be configured.

**Without `REDIS_URL`** the app still starts and answers correctly, but
session history and rate-limit counts are held per process — silently
wrong the moment there is a second worker, since a follow-up landing on
another worker reads as a new conversation and each worker grants the
full rate-limit allowance. It degrades deliberately rather than
refusing to start, and logs the consequence at startup.

---

## Tests

```bash
uv run pytest                    # all 526
uv run pytest -m "not needs_db"  # the 410 that need no database
```

The `needs_db` marker is applied automatically by `conftest.py` to
anything requesting the `db` fixture — derived rather than written by
hand, so it cannot drift as tests are added. CI runs the hermetic
subset plus a container build; it cannot run the rest, because the
database allow-lists IP addresses and CI runners have none.

The hermetic 410 are not the leftovers. They are the security
invariants, the provider-selection rules, the rate limiter on both
backends, the streaming transport, log redaction and the tool-surface
checks — the places where a regression is silent.

There is also an eval suite with an LLM judge
([`app/evals/`](app/evals)) that scores answers on whether they looked
in the right place, invented nothing, and promised nothing they cannot
do. Results are compared run over run.

---

## Deployment

```bash
docker build -t commerce-assistant .
```

~270MB, unprivileged user, no `.env` or dev scripts inside — CI asserts
all three. See [docs/deployment.md](docs/deployment.md) for
configuration, why `WEB_CONCURRENCY` defaults to 1, and how to read
`/health`.

---

## License

None. Published for demonstration; all rights reserved.
