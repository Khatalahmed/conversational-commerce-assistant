"""
WHAT:
    Tests the log-redaction processor in app/config/logging.py — both
    that secrets ARE hidden, and that benign telemetry is NOT.

WHY THIS APPROACH:
    Redaction has two distinct failure modes and only one of them is
    loud. Under-redaction leaks a secret into a log dashboard. OVER-
    redaction silently destroys the metric you were trying to read —
    which is exactly what happened to "prompt_tokens" (it contains the
    substring "token"), hiding the LLM cost data this project's whole
    token-budget design is tuned against. Both directions are asserted
    here so neither can regress.

FLOW:
    Pure unit tests — no database, no network, no fixtures. Unlike the
    rest of this suite these run anywhere, including CI.
"""

from app.config.logging import _is_sensitive_key, redact_sensitive_data

REDACTED = "***REDACTED***"


class TestSensitiveKeysAreRedacted:
    def test_obvious_secrets_are_caught(self):
        for key in ("password", "jwt_secret", "refreshToken", "ifsc"):
            assert _is_sensitive_key(key), f"{key!r} must be treated as sensitive"

    def test_matching_is_case_insensitive(self):
        for key in ("PASSWORD", "Password", "userPassword"):
            assert _is_sensitive_key(key)

    def test_partial_matches_still_caught(self):
        # The deliberate over-redact bias: these are why substring
        # matching exists in the first place.
        for key in ("userPhone", "phone_number", "billing_email", "bankDetails"):
            assert _is_sensitive_key(key)

    def test_safe_key_exemption_does_not_widen_into_a_loophole(self):
        # SAFE_KEYS is matched EXACTLY. A key that merely CONTAINS an
        # exempted name must still be redacted.
        for key in ("auth_token", "prompt_tokens_raw", "x_max_tokens"):
            assert _is_sensitive_key(key), f"{key!r} must not inherit the exemption"


class TestBenignTelemetryIsNotRedacted:
    def test_token_counts_survive(self):
        # The actual bug: every one of these contains "token".
        for key in ("prompt_tokens", "completion_tokens", "total_tokens", "max_tokens"):
            assert not _is_sensitive_key(key), f"{key!r} is a count, not a secret"

    def test_ordinary_keys_survive(self):
        for key in ("user_id", "tool_name", "provider", "session_id", "attempt"):
            assert not _is_sensitive_key(key)


class TestProcessorEndToEnd:
    def test_real_orchestrator_log_call_keeps_its_token_count(self):
        # Mirrors the "llm_call_served" call in agent/orchestrator.py.
        event = {
            "event": "llm_call_served",
            "provider": "groq",
            "fell_back": False,
            "prompt_tokens": 1453,
        }
        cleaned = redact_sensitive_data(None, "info", event)
        assert cleaned["prompt_tokens"] == 1453
        assert cleaned["provider"] == "groq"

    def test_secret_in_the_same_call_is_still_hidden(self):
        cleaned = redact_sensitive_data(
            None, "info", {"prompt_tokens": 10, "refreshToken": "abc123"}
        )
        assert cleaned["prompt_tokens"] == 10
        assert cleaned["refreshToken"] == REDACTED

    def test_nested_dict_is_redacted_recursively(self):
        cleaned = redact_sensitive_data(
            None,
            "info",
            {"user": {"username": "ahmed", "password": "hunter2"}},
        )
        assert cleaned["user"]["username"] == "ahmed"
        assert cleaned["user"]["password"] == REDACTED


class TestSequencesAreWalked:
    """Secrets usually travel in a COLLECTION - a list of users, an
    order's items, a repo's find().to_list(). Dict-only recursion let
    every one of those through untouched."""

    def test_list_of_dicts_is_redacted(self):
        cleaned = redact_sensitive_data(
            None, "info", {"users": [{"username": "ahmed", "password": "hunter2"}]}
        )
        assert cleaned["users"][0]["username"] == "ahmed"
        assert cleaned["users"][0]["password"] == REDACTED

    def test_dict_inside_list_inside_dict(self):
        cleaned = redact_sensitive_data(
            None, "info", {"payload": {"accounts": [{"ifsc": "ABC0001"}]}}
        )
        assert cleaned["payload"]["accounts"][0]["ifsc"] == REDACTED

    def test_deeply_nested_list_of_lists(self):
        cleaned = redact_sensitive_data(
            None, "info", {"batch": [[{"refreshToken": "abc"}]]}
        )
        assert cleaned["batch"][0][0]["refreshToken"] == REDACTED

    def test_plain_string_list_passes_through_unchanged(self):
        # The common real case (providers=["groq", "gemini"]). A str is a
        # sequence too - it must NOT be shredded into characters.
        cleaned = redact_sensitive_data(
            None, "info", {"providers": ["groq", "gemini"], "reason": "rate_limit"}
        )
        assert cleaned["providers"] == ["groq", "gemini"]
        assert cleaned["reason"] == "rate_limit"

    def test_tuple_stays_a_tuple(self):
        cleaned = redact_sensitive_data(None, "info", {"tried": ("groq", "gemini")})
        assert cleaned["tried"] == ("groq", "gemini")
        assert isinstance(cleaned["tried"], tuple)

    def test_namedtuple_fields_are_redacted_by_name(self):
        # Iterated as a plain sequence, a namedtuple's field NAMES are
        # lost and "password" becomes an unlabelled string that no rule
        # can catch. _asdict() is what keeps it judgeable.
        from collections import namedtuple

        Usage = namedtuple("Usage", ["model", "password"])
        cleaned = redact_sensitive_data(
            None, "info", {"usage": Usage(model="gpt", password="hunter2")}
        )
        assert cleaned["usage"]["model"] == "gpt"
        assert cleaned["usage"]["password"] == REDACTED
        assert "hunter2" not in str(cleaned)

    def test_empty_sequences_are_safe(self):
        cleaned = redact_sensitive_data(None, "info", {"a": [], "b": ()})
        assert cleaned["a"] == []
        assert cleaned["b"] == ()


class TestObjectsAreIntrospected:
    """A secret in an OBJECT field is invisible unless the object's field
    names are read. The live case is Provider (agent/llm_client.py), a
    frozen dataclass holding api_key."""

    def test_the_real_provider_dataclass_does_not_leak_its_api_key(self):
        from app.agent.llm_client import Provider

        provider = Provider(
            name="groq",
            base_url="https://api.groq.com/openai/v1",
            model="openai/gpt-oss-120b",
            api_key="gsk_supersecret123",
        )
        cleaned = redact_sensitive_data(None, "info", {"provider": provider})

        assert cleaned["provider"]["name"] == "groq"
        assert cleaned["provider"]["model"] == "openai/gpt-oss-120b"
        assert cleaned["provider"]["api_key"] == REDACTED
        assert "gsk_supersecret123" not in str(cleaned)

    def test_dataclass_nested_in_a_list(self):
        from app.agent.llm_client import Provider

        providers = [Provider("groq", "url", "model", "gsk_secret")]
        cleaned = redact_sensitive_data(None, "info", {"providers": providers})
        assert cleaned["providers"][0]["api_key"] == REDACTED

    def test_plain_objects_are_left_alone(self):
        # Only DATACLASSES are introspected. Walking every object with a
        # __dict__ would mean serializing a Motor client or an httpx
        # response the moment one reached a log call.
        class Connection:
            def __init__(self):
                self.password = "hunter2"

        conn = Connection()
        cleaned = redact_sensitive_data(None, "info", {"conn": conn})
        assert cleaned["conn"] is conn  # untouched, renders via its own repr


class TestDepthLimit:
    def test_deep_nesting_is_truncated_not_crashed(self):
        payload = current = {}
        for _ in range(50):
            current["next"] = {}
            current = current["next"]
        current["password"] = "hunter2"

        cleaned = redact_sensitive_data(None, "info", payload)
        assert "hunter2" not in str(cleaned)
        assert "***TRUNCATED***" in str(cleaned)

    def test_reference_cycle_does_not_raise(self):
        # The reason the depth limit exists: without it this recurses
        # until Python raises RecursionError, from inside a log call.
        node: dict = {"name": "root"}
        node["self"] = node
        cleaned = redact_sensitive_data(None, "info", {"node": node})
        assert "***TRUNCATED***" in str(cleaned)

    def test_realistic_payload_is_not_truncated(self):
        # Guards the other direction: the limit must sit well clear of
        # anything genuinely logged, or it silently eats real data.
        order = {"items": [{"variants": [{"stock": {"count": 3}}]}]}
        cleaned = redact_sensitive_data(None, "info", {"order": order})
        assert cleaned["order"]["items"][0]["variants"][0]["stock"]["count"] == 3
