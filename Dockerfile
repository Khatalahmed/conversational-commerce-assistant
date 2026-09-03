# Appuser conversational assistant.
#
# WHY TWO STAGES: the builder needs uv and a compiler cache to resolve
# and install dependencies; the thing that actually runs needs neither.
# Shipping them means shipping a package manager, its cache and a build
# toolchain into production, which is both larger and more to attack.
#
# WHAT IS DELIBERATELY NOT IN HERE:
#   .env       Configuration arrives as environment variables at run
#              time. Baking secrets into a layer puts a live MongoDB URI
#              and a JWT signing secret into every copy of the image,
#              including any registry it is pushed to, permanently -
#              layers are not removed by deleting the file later.
#   scripts/   Dev-only by the project's own definition, and two of them
#              WRITE (seed_demo_data.py, import_real_catalogue.py). A
#              read-only service should not carry tools that are not.
#              Run diagnostics from a checkout against the same .env.
#   tests/     Excluded via .dockerignore; they need a live sandbox and
#              have no business in a runtime image.

FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

# COMPILE_BYTECODE trades a slower build for a faster first request -
# worth it when the container may be scaled from zero. LINK_MODE=copy
# because uv's default hardlinking does not work across the mount used
# for the cache below.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# /app, NOT /build, AND THE PATH IS LOAD-BEARING.
#
# uv writes the venv at $WORKDIR/.venv, and every console script in it
# gets an absolute shebang pointing back at that interpreter. Built in
# /build and copied to /app, `uvicorn` starts with
# `#!/build/.venv/bin/python` - a path that does not exist in the
# runtime stage. The container then dies with "uvicorn: not found",
# which reads like a missing dependency and is nothing of the sort.
#
# Building at the final path makes the shebangs correct by construction.
WORKDIR /app

# Dependencies before source, so editing app/ does not re-resolve and
# re-download every package - the layer cache holds as long as
# pyproject.toml and uv.lock are unchanged.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# --frozen, NOT `uv sync` alone: the lockfile is the record of what was
# tested. Letting the build re-resolve means the image can quietly ship
# a different dependency tree than the suite ran against, and the first
# anyone hears of it is in production.


FROM python:3.13-slim-bookworm AS runtime

# NOT ROOT. A read-only assistant has no reason to run privileged, and
# the difference matters the day something in the dependency tree turns
# out to be exploitable.
RUN useradd --create-home --uid 10001 appuser

WORKDIR /app

COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --chown=appuser:appuser app ./app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    # ONE WORKER BY DEFAULT, AND THAT DEFAULT IS THE SAFE ONE.
    #
    # Conversation memory and the /chat rate limit are shared through
    # Redis, so several workers are correct - but only once REDIS_URL is
    # actually set. Without it both fall back to per-process state, and
    # the failure is silent: a follow-up lands on a worker that never
    # saw the first message, and each worker grants the full rate-limit
    # allowance. Defaulting to 1 means an image run without Redis is
    # slow rather than subtly wrong. Raise it once Redis is configured.
    WEB_CONCURRENCY=1

USER appuser

EXPOSE 8000

# Uses the app's own /health, which reports the database AND the LLM
# provider chain - so an orchestrator restarting this container is
# reacting to the same signal an operator would. Start period covers
# the Atlas handshake that connect_to_mongo does at startup; the app
# deliberately refuses to start on a broken database connection.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,os,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8000')+'/health', timeout=4).status==200 else 1)"

# `sh -c` because PORT and WEB_CONCURRENCY have to be expanded at run
# time, and `exec` because without it the shell stays PID 1 and swallows
# SIGTERM - so uvicorn would be killed rather than asked to stop, and
# the lifespan shutdown that closes Mongo, Redis and the HTTP client
# would be skipped on every single deploy.
#
# --no-access-log because the app already logs the interesting part of
# every request through structlog (jwt_verified, tool_executing,
# llm_call_served). A second, differently-formatted line per request
# would be noise in the same stream.
CMD ["sh", "-c", "exec uvicorn app.api.main:app --host 0.0.0.0 --port ${PORT} --workers ${WEB_CONCURRENCY} --no-access-log"]
