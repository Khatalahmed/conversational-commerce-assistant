"""
WHAT:
    Unit tests for JWT verification (app/security/auth.py) — the single
    thing standing between a request and another user's order history.

WHY THIS APPROACH:
    The claim name and signing algorithm the real marketplace backend uses are
    still unconfirmed, so these tests pin the BEHAVIOUR around that
    unknown rather than the unknown itself: that any configured claim
    name works, that a wrong algorithm is refused, that "none" can never
    be used, and that a token without an expiry is rejected outright.

FLOW:
    Pure unit tests. No database, no network — settings are constructed
    directly and patched in, so nothing here depends on the real .env.
"""

from datetime import datetime, timedelta, timezone

import jwt
import pytest

from app.config.settings import Settings
from app.security import auth
from app.security.auth import InvalidTokenError, verify_token

# 64 bytes: long enough for HS512 too, so PyJWT does not warn about weak
# HMAC key length on every single token minted here.
SECRET = "test-secret-not-a-real-one-padded-to-sixty-four-bytes-for-hs512!"
USER_ID = "0123456789abcdef01234567"


def _settings(**overrides) -> Settings:
    base = {
        "mongodb_uri": "mongodb://localhost/test",
        "llm_api_key": "test-key",
        "jwt_secret": SECRET,
    }
    return Settings(**{**base, **overrides})


@pytest.fixture
def configured(monkeypatch):
    """Swaps in a Settings object without touching the real .env."""

    def _apply(**overrides):
        settings = _settings(**overrides)
        monkeypatch.setattr(auth, "get_settings", lambda: settings)
        return settings

    return _apply


def _token(claims: dict, *, secret=SECRET, algorithm="HS256", expires_in=30):
    payload = dict(claims)
    if expires_in is not None:
        payload["exp"] = datetime.now(timezone.utc) + timedelta(minutes=expires_in)
    return jwt.encode(payload, secret, algorithm=algorithm)


class TestClaimNameIsConfigurable:
    """The real backend's claim name is still unknown - the point of the
    candidate list is that the service works on first contact with a
    real token instead of rejecting it."""

    @pytest.mark.parametrize("claim", ["user_id", "userId", "sub", "_id"])
    def test_each_default_candidate_claim_is_accepted(self, configured, claim):
        configured()
        assert verify_token(_token({claim: USER_ID})) == USER_ID

    def test_priority_order_decides_when_several_are_present(self, configured):
        configured(jwt_user_id_claims="userId,sub")
        # Both present; the earlier-listed claim must win.
        token = _token({"sub": "wrong-one", "userId": USER_ID})
        assert verify_token(token) == USER_ID

    def test_narrowing_to_the_confirmed_claim_rejects_the_others(self, configured):
        # What you do once the backend team answers: pin it to one name.
        configured(jwt_user_id_claims="buyerId")
        assert verify_token(_token({"buyerId": USER_ID})) == USER_ID
        with pytest.raises(InvalidTokenError):
            verify_token(_token({"user_id": USER_ID}))

    def test_no_recognized_claim_is_rejected(self, configured):
        configured()
        with pytest.raises(InvalidTokenError, match="no recognized user ID claim"):
            verify_token(_token({"email": "someone@example.com"}))

    def test_user_id_is_returned_as_a_string(self, configured):
        # A backend could plausibly send a numeric id; repo functions
        # feed this straight into to_object_id(), which expects a str.
        configured()
        assert verify_token(_token({"user_id": 12345})) == "12345"


class TestAlgorithmIsConfigurable:
    def test_token_signed_with_a_different_algorithm_is_rejected(self, configured):
        configured(jwt_algorithm="HS512")
        with pytest.raises(InvalidTokenError):
            verify_token(_token({"user_id": USER_ID}, algorithm="HS256"))

    def test_matching_algorithm_is_accepted(self, configured):
        configured(jwt_algorithm="HS512")
        token = _token({"user_id": USER_ID}, algorithm="HS512")
        assert verify_token(token) == USER_ID

    def test_none_algorithm_cannot_be_configured(self):
        # "none" means unsigned. If it were ever configurable, anyone
        # could hand us a token claiming to be any user.
        with pytest.raises(ValueError, match="not supported"):
            _settings(jwt_algorithm="none")

    def test_unsigned_token_is_rejected(self, configured):
        configured()
        unsigned = jwt.encode({"user_id": USER_ID}, key="", algorithm="none")
        with pytest.raises(InvalidTokenError):
            verify_token(unsigned)

    def test_test_token_helper_refuses_asymmetric_algorithms(self, configured):
        # JWT_SECRET holds a PUBLIC key under RS*/ES*, which cannot sign.
        configured(jwt_algorithm="RS256")
        with pytest.raises(RuntimeError, match="private key|PRIVATE key"):
            auth.create_test_token(user_id=USER_ID)


class TestSignatureAndExpiry:
    def test_wrong_secret_is_rejected(self, configured):
        configured()
        with pytest.raises(InvalidTokenError, match="tampered"):
            verify_token(_token({"user_id": USER_ID}, secret="a-different-secret-also-padded-to-sixty-four-bytes-abcdefgh"))

    def test_tampered_token_is_rejected(self, configured):
        configured()
        token = _token({"user_id": USER_ID})
        with pytest.raises(InvalidTokenError):
            verify_token(token[:-5] + "XXXXX")

    def test_expired_token_is_rejected(self, configured):
        configured()
        with pytest.raises(InvalidTokenError, match="expired"):
            verify_token(_token({"user_id": USER_ID}, expires_in=-5))

    def test_token_without_an_expiry_is_rejected(self, configured):
        # A token that never expires is a permanent key to that account
        # if it ever leaks. PyJWT accepts a missing exp by default; this
        # asserts we do not.
        configured()
        with pytest.raises(InvalidTokenError, match="required claim"):
            verify_token(_token({"user_id": USER_ID}, expires_in=None))

    def test_garbage_is_rejected_without_crashing(self, configured):
        configured()
        for junk in ["", "not-a-token", "a.b.c", "Bearer x"]:
            with pytest.raises(InvalidTokenError):
                verify_token(junk)


class TestAudienceAndIssuer:
    def test_unset_means_not_checked(self, configured):
        configured()  # no audience/issuer configured
        token = _token({"user_id": USER_ID, "aud": "anything", "iss": "whoever"})
        assert verify_token(token) == USER_ID

    def test_blank_env_value_is_treated_as_unset(self, configured):
        # A copied .env.example gives "" - which must NOT switch the
        # check on with an empty expected value.
        configured(jwt_audience="", jwt_issuer="   ")
        assert verify_token(_token({"user_id": USER_ID})) == USER_ID

    def test_wrong_audience_is_rejected(self, configured):
        configured(jwt_audience="mobile-app")
        token = _token({"user_id": USER_ID, "aud": "some-other-service"})
        with pytest.raises(InvalidTokenError, match="not issued for this service"):
            verify_token(token)

    def test_correct_audience_is_accepted(self, configured):
        configured(jwt_audience="mobile-app")
        token = _token({"user_id": USER_ID, "aud": "mobile-app"})
        assert verify_token(token) == USER_ID

    def test_wrong_issuer_is_rejected(self, configured):
        configured(jwt_issuer="auth-backend")
        token = _token({"user_id": USER_ID, "iss": "impostor"})
        with pytest.raises(InvalidTokenError, match="unexpected issuer"):
            verify_token(token)


class TestRoundTripWithTheTestHelper:
    def test_helper_mints_a_token_verify_token_accepts(self, configured):
        configured()
        assert verify_token(auth.create_test_token(user_id=USER_ID)) == USER_ID

    def test_helper_follows_the_configured_claim_name(self, configured):
        configured(jwt_user_id_claims="buyerId")
        decoded = jwt.decode(
            auth.create_test_token(user_id=USER_ID),
            SECRET,
            algorithms=["HS256"],
        )
        assert decoded["buyerId"] == USER_ID
