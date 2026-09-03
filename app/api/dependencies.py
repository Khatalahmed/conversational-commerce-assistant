"""
WHAT:
    FastAPI dependency that extracts and verifies the JWT from the
    Authorization header, returning the verified user_id. Every
    protected route uses this - never trusts a user_id from the
    request body itself, per Phase 3.1's core rule.

WHY THIS APPROACH:
    Centralizing this as ONE dependency means every route that needs
    identity gets it the same, safe way - impossible to accidentally
    add a new route that skips verification.

FLOW:
    FastAPI runs this automatically before the route function, on
    every request to a route that declares it as a dependency.
"""

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.security.auth import InvalidTokenError, verify_token
from app.security.rate_limit import RateLimitExceeded, check_rate_limit


# Declares the endpoint as Bearer-authenticated in the OpenAPI schema.
# The behaviour is the same as reading the header by hand, but /docs then
# renders an "Authorize" padlock: a token is pasted ONCE and applied to
# every request, instead of retyping "Bearer eyJ..." into a header box on
# each call. That matters for demos more than for correctness.
#
# auto_error=False so a MISSING header reaches our own check below rather
# than FastAPI raising first - we want one consistent 401 with a message
# that says what was expected, not two different shapes of failure.
bearer_scheme = HTTPBearer(
    auto_error=False,
    description="JWT issued by the marketplace mobile app's login.",
)


async def get_current_user_id(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> str:
    if credentials is None or not credentials.credentials:
        # 401, not 422. A missing credential is an authentication
        # failure, not a malformed request body - and 422 was only ever
        # an accident of declaring it as a required Header(...).
        raise HTTPException(
            status_code=401,
            detail="Missing or malformed Authorization header. Expected: Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        return verify_token(credentials.credentials.strip())
    except InvalidTokenError as exc:
        raise HTTPException(
            status_code=401, detail=str(exc), headers={"WWW-Authenticate": "Bearer"}
        )

async def rate_limited_user_id(user_id: str = Depends(get_current_user_id)) -> str:
    """Verified identity, plus an enforced request allowance.

    ORDER MATTERS AND IS ENFORCED BY THIS SIGNATURE. The limiter depends
    ON get_current_user_id, so authentication always runs first. That is
    not a stylistic preference:
      - the limiter keys on the verified subject, and an unverified one
        could be set to anything, letting an attacker exhaust another
        user's allowance by claiming to be them;
      - an unauthenticated flood costs one signature check and is
        rejected before it can consume anyone's budget.

    Routes should depend on THIS rather than on get_current_user_id, so
    a new endpoint cannot quietly skip the limiter.
    """
    try:
        await check_rate_limit(user_id)
    except RateLimitExceeded as exc:
        raise HTTPException(
            status_code=429,
            detail="Too many requests - please slow down and try again shortly.",
            # Tells the client exactly how long to wait, instead of
            # leaving it to guess or hammer the endpoint harder.
            headers={"Retry-After": str(exc.retry_after_seconds)},
        )
    return user_id
