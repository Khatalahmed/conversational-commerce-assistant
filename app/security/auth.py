"""
WHAT:
    This file verifies JWT tokens (the login tokens the marketplace mobile app
    already issues to logged-in users) and safely extracts the user's ID
    from them. It also includes a small TEST-ONLY helper to generate
    fake tokens locally, since we don't have the real signing secret yet.

WHY THIS APPROACH:
    Per our Phase 3.1 design: the chat message text itself must NEVER be
    trusted for identity — only a cryptographically verified token proves
    who is really asking. This file is what actually enforces that rule
    in code. Every future API request (Phase 9) will pass its token
    through verify_token() before any feature logic runs.

FLOW:
    1. Mobile app sends a chat request with a JWT attached (Phase 9).
    2. Our API layer calls verify_token(token) before doing anything else.
    3. If the signature is invalid, expired, or malformed, this raises a
       clear error and the request is rejected immediately.
    4. If valid, it returns the user's ID, which then gets passed into
       every repository function (Phase 4) to scope that user's data.

LOGIC:
    A token is only accepted if ALL of these are true:
      - It was signed with our known secret/key, using the ONE algorithm
        we are configured to accept (proves it's genuine, not forged).
      - It carries an expiry, and has not passed it (a token with no
        expiry would be a permanent key if it ever leaked).
      - It matches the expected audience and issuer, when those are
        configured.
      - It actually contains a user ID claim we can extract.

    THE OPEN QUESTION, AND HOW IT IS HANDLED. We still do not know what
    the real marketplace backend calls its user-ID claim, or which algorithm
    it signs with. Both used to be hardcoded guesses in this file, which
    meant being wrong about either required a code change. They are now
    SETTINGS (JWT_ALGORITHM, JWT_USER_ID_CLAIMS) with the guesses as
    defaults, so the answer lands in .env instead.

    Until it does, verify_token tries several plausible claim names in
    priority order and LOGS which one matched - so the first real token
    to reach this service answers the question in the logs rather than
    simply being rejected.

MECHANISM:
    - PyJWT's jwt.encode() creates a signed token (our TEST-ONLY helper
      uses this to simulate what the real backend does).
    - jwt.decode() verifies the signature and expiry automatically,
      raising specific exception types we catch and translate into
      clear, predictable errors for the rest of our app to handle.
"""

from datetime import datetime, timedelta, timezone

import jwt
import structlog

from app.config.settings import get_settings

logger = structlog.get_logger()

def log_secret_preflight() -> None:
    """Says at STARTUP whether the signing secret is strong enough.

    Called from the app lifespan, next to the LLM provider preflight and
    for the same reason: a weakness nobody is told about is a weakness
    nobody fixes. There is no request that would reveal this and no test
    that can - the secret is configuration, and the only moment anyone
    is reading is when the service comes up.
    """
    settings = get_settings()
    weakness = settings.jwt_secret_weakness
    if weakness:
        logger.error(
            "jwt_secret_weak",
            algorithm=settings.jwt_algorithm,
            detail=weakness,
            consequence="a forged token would be indistinguishable from a real "
                        "login, and would bypass every other data-safety layer",
        )
    else:
        logger.info("jwt_secret_ok", algorithm=settings.jwt_algorithm)


class InvalidTokenError(Exception):
    """Raised whenever a token fails verification for any reason."""
    pass


def _extract_user_id(payload: dict) -> str:
    """Finds the user ID among the configured candidate claim names.

    WHY A LIST RATHER THAN ONE NAME: we do not yet know what the real
    the marketplace backend calls this claim - "user_id", "userId", "sub" and
    "_id" are all plausible, and picking one wrong means every real
    token is rejected as "valid but has no user". Trying several in a
    fixed priority order makes the service work on first contact with a
    real token instead of failing until someone edits code.

    THIS IS NOT A SECURITY SHORTCUT. It runs only AFTER the signature
    has been verified, so every claim read here was written by the
    backend that holds the signing key - not by the caller. The choice
    is about which trusted field to read, not whether to trust it.

    Once the team confirms the real name, set JWT_USER_ID_CLAIMS to that
    single value and the guessing stops.
    """
    settings = get_settings()

    for claim in settings.jwt_user_id_claim_list:
        value = payload.get(claim)
        if value:
            # Logged because it is the ANSWER to the open question: the
            # first real token to arrive tells us, in the logs, which
            # claim the backend actually uses.
            logger.info("jwt_user_id_claim_matched", claim=claim)
            return str(value)

    logger.warning(
        "jwt_verification_failed",
        reason="no_user_id_claim",
        looked_for=list(settings.jwt_user_id_claim_list),
        # The claim NAMES present in the token, never their values - this
        # is exactly the breadcrumb needed to configure the right one.
        claims_present=sorted(payload.keys()),
    )
    raise InvalidTokenError(
        "Token is valid but contains no recognized user ID claim."
    )


def verify_token(token: str) -> str:
    """
    Verifies a JWT's signature, expiry, and (when configured) audience
    and issuer, then returns the user ID inside it. Raises
    InvalidTokenError on ANY problem — expired, tampered, malformed, or
    missing a usable user ID claim.
    """
    settings = get_settings()

    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            # EXACTLY ONE algorithm, from validated config. Never a list
            # mixing HS* and RS*: with both accepted, an attacker can
            # HMAC-sign a token using the backend's PUBLIC key (which is
            # not secret) and we would verify it as genuine.
            algorithms=[settings.jwt_algorithm],
            audience=settings.jwt_audience,
            issuer=settings.jwt_issuer,
            options={
                # A token with no expiry is valid forever - if one ever
                # leaks, it is a permanent key to that user's account.
                # Require the claim rather than silently accepting its
                # absence, which is PyJWT's default.
                "require": ["exp"],
                "verify_aud": settings.jwt_audience is not None,
                "verify_iss": settings.jwt_issuer is not None,
            },
        )
    except jwt.ExpiredSignatureError:
        logger.warning("jwt_verification_failed", reason="expired")
        raise InvalidTokenError("Token has expired.")
    except jwt.MissingRequiredClaimError as exc:
        logger.warning("jwt_verification_failed", reason="missing_claim", detail=str(exc))
        raise InvalidTokenError("Token is missing a required claim (exp).")
    except jwt.InvalidAudienceError:
        logger.warning("jwt_verification_failed", reason="wrong_audience")
        raise InvalidTokenError("Token was not issued for this service.")
    except jwt.InvalidIssuerError:
        logger.warning("jwt_verification_failed", reason="wrong_issuer")
        raise InvalidTokenError("Token came from an unexpected issuer.")
    except jwt.InvalidSignatureError:
        logger.warning("jwt_verification_failed", reason="bad_signature")
        raise InvalidTokenError("Token signature is invalid — possibly tampered.")
    except jwt.InvalidAlgorithmError:
        # The token was signed with something other than what we accept.
        # Nearly always a config mismatch with the backend, not an
        # attack - and the fix is JWT_ALGORITHM, so say so in the log.
        logger.warning(
            "jwt_verification_failed",
            reason="algorithm_mismatch",
            configured=settings.jwt_algorithm,
        )
        raise InvalidTokenError("Token uses an unexpected signing algorithm.")
    except jwt.InvalidTokenError as exc:
        # Catches any other malformed-token case PyJWT recognizes.
        logger.warning("jwt_verification_failed", reason="malformed", detail=str(exc))
        raise InvalidTokenError("Token is malformed or invalid.")

    user_id = _extract_user_id(payload)
    logger.info("jwt_verified", user_id=user_id)
    return user_id


# ── TEST-ONLY HELPER — remove or move to app/tests/ once real tokens
# ── from the actual the marketplace backend are available. Simulates what the
# ── real backend does when it issues a login token.
def create_test_token(user_id: str, expires_in_minutes: int = 30) -> str:
    settings = get_settings()

    # Signing needs the PRIVATE key; JWT_SECRET holds the backend's
    # PUBLIC key under RS*/ES*, which cannot sign. Fail with the reason
    # rather than a cryptography-library stack trace.
    if not settings.jwt_algorithm.startswith("HS"):
        raise RuntimeError(
            f"Cannot mint a test token under {settings.jwt_algorithm}: asymmetric "
            f"algorithms sign with a PRIVATE key, and JWT_SECRET holds the "
            f"backend's public key. Ask the marketplace backend team for a real "
            f"sample token instead."
        )

    # Mint under the FIRST configured claim name, so a test token always
    # matches whatever verify_token is currently looking for.
    payload = {
        settings.jwt_user_id_claim_list[0]: user_id,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=expires_in_minutes),
    }
    if settings.jwt_audience:
        payload["aud"] = settings.jwt_audience
    if settings.jwt_issuer:
        payload["iss"] = settings.jwt_issuer

    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)