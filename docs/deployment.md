# Deploying the assistant

Phase 11. Everything here has been run and verified locally; the one
step that has not is pointing it at a real host, which needs an account
this project does not have.

---

## Build and run it

```bash
docker build -t commerce-assistant .
docker run -d --name marketplace -p 8000:8000 \
  -e MONGODB_URI="..." \
  -e MONGODB_DATABASE="marketplace" \
  -e JWT_SECRET="..." \
  -e AZURE_OPENAI_ENDPOINT="..." \
  -e AZURE_OPENAI_API_KEY="..." \
  -e AZURE_OPENAI_DEPLOYMENT="gpt-5-mini" \
  -e REDIS_URL="rediss://..." \
  commerce-assistant
```

The image is ~270MB, runs as an unprivileged user (`marketplace`, uid 10001),
and carries no `.env`, no tests and no dev scripts. Verified:

```
$ curl -s localhost:8000/health
{"status":"ok","database":{"status":"connected"},"llm":{"status":"warning","redundant":false}}
```

---

## Do not use `docker run --env-file .env`

The two readers of that file disagree, and the disagreement is not
cosmetic.

`python-dotenv` — which the app uses locally — strips an inline comment.
Docker's `--env-file` does not, and passes the whole remainder of the
line. So this:

```
MONGODB_DATABASE=marketplace          # or marketplace_demo
```

means `marketplace` when you run uvicorn and
`marketplace          # or marketplace_demo` when you run the container.

**Why that matters more than a typo.** That exact value decides whether
Groq and Gemini are dropped from the provider chain — the filter
compares it to `"marketplace"` to decide whether the data is real customers.
A mangled value is not equal to `"marketplace"`, so a service genuinely
pointed at real orders would classify itself as demo data and keep the
providers whose terms permit training on prompts.

The app now normalises this setting (strips whitespace and any inline
comment) so the safety property depends on what was meant rather than
on which parser read the file. **Pass variables explicitly anyway**, or
keep a separate comment-free file for Docker. Relying on the
normalisation is relying on a backstop.

---

## Workers, and the one thing that must be true first

`WEB_CONCURRENCY` defaults to **1**, deliberately.

Conversation memory and the `/chat` rate limit are both shared through
Redis, so several workers are correct — *once `REDIS_URL` is actually
set*. Without it, both silently fall back to per-process state:

- a follow-up question lands on a worker that never saw the first
  message, so the assistant answers "similar to what?" — which reads as
  the AI being flaky rather than as an infrastructure problem
- each worker grants the full rate-limit allowance, so a configured 20
  is really `workers × 20`

Neither failure announces itself. Set `REDIS_URL`, confirm it with
`scripts/check_redis.py`, then raise `WEB_CONCURRENCY`.

---

## Health checks

`/health` is unauthenticated and deliberately vague — it returns a
status, never hostnames, database names or dates. Detail goes to the
logs.

```json
{"status": "ok", "database": {"status": "connected"},
 "llm": {"status": "warning", "redundant": false}}
```

| `llm.status` | Meaning | Should it page anyone |
|---|---|---|
| `ok` | Two or more usable providers | No |
| `warning` | One provider, or the Azure credit is within 14 days | No — it answers every request today |
| `unavailable` | Credit expired, or no provider can serve this data | Yes — this also sets `status: degraded` |

Only `unavailable` degrades the overall status, so an orchestrator
restarting on `status != ok` reacts to real outages and not to the
standing single-provider warning. The container's own `HEALTHCHECK`
uses the same endpoint.

Expect `warning` on day one: while real customer data is configured,
Azure is the only eligible provider, and a trial credit has an expiry date.
See `BACKUP_LLM_*` for the fix.

---

## What the mobile app needs (Phase 12)

A base URL and the existing the marketplace JWT. Nothing else changes: the token
is the one the app already issues at login, and `JWT_SECRET` here must
match the main backend's signing secret exactly.

| Endpoint | Shape |
|---|---|
| `POST /chat` | JSON in, JSON out. The original contract, unchanged. |
| `POST /chat/stream` | Server-Sent Events: `status`, `token`, `done`, `error`. |

Both take `{"message": "...", "session_id": "..."}` with
`Authorization: Bearer <jwt>`, and both are behind the same auth and
rate limit. Adopt streaming whenever the client is ready — the
non-streaming endpoint is not deprecated and runs the identical loop.

**Set `DEMO_UI_ENABLED=false`.** The `/demo` chat page defaults to on so
demos and local development need no configuration, and that default is
wrong for a public URL: it advertises the service to anyone who finds it
and invites pasting a bearer token into a web page. It cannot answer
without a valid the marketplace token, so this is an unnecessary door rather than
a leak — but it is a door nobody remembers mounting. Startup logs
`demo_ui_enabled` or `demo_ui_disabled` every time, so it is never a
guess.

**There is no CORS middleware, and that is correct for a native app.**
CORS is a browser rule; a native client is unaffected, and the `/demo`
page is served by this same app on the same origin. If a *web* client
on another origin is ever added, that is the point to add
`CORSMiddleware` with an explicit allow-list — not `*`, since every
request carries a bearer token.

---

## Before it faces anyone

- `scripts/check_azure.py` — provider chain, and days of credit left
- `scripts/check_redis.py` — proves a real write/read round trip
- `uv run pytest` — the full suite, from a machine whose IP Atlas allows

CI runs the 274 tests that need no database (`-m "not needs_db"`) plus
a container build. It cannot run the rest: Atlas allow-lists IPs and
GitHub runners have none. Run the full suite locally before a release.
