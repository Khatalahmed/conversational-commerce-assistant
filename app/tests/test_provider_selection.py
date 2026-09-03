"""
WHAT:
    Tests that the LLM provider chain follows the SENSITIVITY OF THE DATA,
    not a flag and not the ordering of a list.

WHY THIS MATTERS:
    Reading the real `marketplace` database means every tool result carries
    somebody's actual order - delivery city, courier tracking number,
    purchase history. Groq and Google's free tiers reserve the right to
    train on what they are sent; Azure OpenAI is contractually excluded.

    It used to be enough that Azure was FIRST in the chain. That is a
    guarantee about ordering, not about destination: one rate limit from
    Azure and the failover handed the same customer data to Groq. A
    safety property that holds only while nothing goes wrong is not a
    safety property, and rate limits are exactly the thing that goes
    wrong - we spent a whole session watching them.

    These tests pin the stronger version: against real data, a
    training-permitted provider is not merely last, it is ABSENT.

FLOW:
    Pure unit tests. get_providers() is lru_cached, so every test clears
    the cache first - otherwise the first test's settings would decide
    every later result.
"""

import pytest

from app.agent import llm_client
from app.agent.llm_client import NoProviderConfigured
from app.config.settings import Settings


def _settings(**overrides) -> Settings:
    base = {
        "mongodb_uri": "mongodb://localhost/test",
        "llm_api_key": "groq-key",
        "jwt_secret": "s" * 64,
        "gemini_api_key": "gemini-key",
        "azure_openai_endpoint": "https://example.openai.azure.com",
        "azure_openai_api_key": "azure-key",
        "azure_openai_deployment": "gpt-5-mini",
    }
    return Settings(**{**base, **overrides})


@pytest.fixture
def configured(monkeypatch):
    def _apply(**overrides):
        settings = _settings(**overrides)
        monkeypatch.setattr(llm_client, "get_settings", lambda: settings)
        llm_client.get_providers.cache_clear()
        return settings

    yield _apply
    llm_client.get_providers.cache_clear()


class TestAgainstRealCustomerData:
    def test_training_permitted_providers_are_removed_entirely(self, configured):
        configured(mongodb_database="marketplace")
        names = [p.name for p in llm_client.get_providers()]

        assert names == ["azure"], (
            f"real customer data could reach {[n for n in names if n != 'azure']}"
        )

    def test_being_first_is_not_enough(self, configured):
        """The distinction this whole module exists for.

        Azure was already first. The failover chain behind it was the
        problem, so 'azure is primary' must not be the assertion.
        """
        configured(mongodb_database="marketplace")
        providers = llm_client.get_providers()

        assert providers[0].name == "azure"
        assert len(providers) == 1, "a fallback survives that may train on prompts"

    def test_every_remaining_provider_is_excluded_from_training(self, configured):
        configured(mongodb_database="marketplace")
        assert all(not p.trains_on_prompts for p in llm_client.get_providers())

    def test_it_refuses_to_run_at_all_without_a_no_training_provider(self, configured):
        """Failing CLOSED. If someone removes the Azure key while .env
        still points at real customers, the alternative is silently
        sending that data to Groq - so this raises instead."""
        configured(
            mongodb_database="marketplace",
            azure_openai_endpoint=None,
            azure_openai_api_key=None,
        )
        # NoProviderConfigured, not RuntimeError: it is a subclass of
        # LLMUnavailable so the orchestrator's existing handler turns it
        # into a sentence. As a bare RuntimeError it escaped every
        # handler and reached the mobile app as a 500 - and this is the
        # exact state the day an expired Azure key gets deleted.
        with pytest.raises(NoProviderConfigured, match="no provider is contractually"):
            llm_client.get_providers()


class TestAgainstDemoData:
    def test_the_cheap_providers_are_kept(self, configured):
        """Demo data is invented, so there is nothing to protect - and
        the free providers are useful for cheap testing."""
        configured(mongodb_database="marketplace_demo")
        names = [p.name for p in llm_client.get_providers()]

        assert "azure" in names
        assert "groq" in names
        assert "gemini" in names

    def test_azure_still_leads_when_configured(self, configured):
        configured(mongodb_database="marketplace_demo")
        assert llm_client.get_providers()[0].name == "azure"

    def test_it_works_with_no_azure_at_all(self, configured):
        # A POC must still run on the free tier alone.
        configured(
            mongodb_database="marketplace_demo",
            azure_openai_endpoint=None,
            azure_openai_api_key=None,
        )
        names = [p.name for p in llm_client.get_providers()]
        assert names[0] == "groq"


class TestAzureOnly:
    """The configuration while the Azure credit is live: no Groq key, no
    Gemini key, nothing that trains on prompts configured at all."""

    def test_azure_alone_is_a_valid_configuration(self, configured):
        configured(
            mongodb_database="marketplace_demo",
            llm_api_key=None,
            gemini_api_key=None,
        )
        assert [p.name for p in llm_client.get_providers()] == ["azure"]

    def test_azure_alone_works_against_real_data_too(self, configured):
        configured(mongodb_database="marketplace", llm_api_key=None, gemini_api_key=None)
        assert [p.name for p in llm_client.get_providers()] == ["azure"]

    def test_no_provider_at_all_is_refused_loudly(self, configured):
        """Better a clear error at startup than a 500 on a user's first
        message. Groq's key used to be REQUIRED, which made this state
        unreachable; now that it is optional, it has to be checked."""
        configured(
            mongodb_database="marketplace_demo",
            llm_api_key=None,
            gemini_api_key=None,
            azure_openai_endpoint=None,
            azure_openai_api_key=None,
        )
        with pytest.raises(NoProviderConfigured, match="No LLM provider is configured"):
            llm_client.get_providers()

    def test_groq_returns_the_moment_its_key_comes_back(self, configured):
        """Deliberately reversible. The Azure credit runs out after a
        month; restoring the fallback should be a .env edit, not a code
        change - which is why the failover machinery is kept."""
        configured(mongodb_database="marketplace_demo", llm_api_key="groq-key")
        assert "groq" in [p.name for p in llm_client.get_providers()]


class TestTheBackupProvider:
    """The mitigation for Azure being alone on a dated credit.

    Adding a second non-training provider used to mean editing
    llm_client.py - a code change, needed on the morning the credit
    lapses. These pin the .env-only path, including the part that stops
    it becoming a hole.
    """

    BACKUP = {
        "backup_llm_base_url": "https://compliant.example.com/v1",
        "backup_llm_api_key": "backup-key",
        "backup_llm_model": "some-model",
    }

    def test_an_asserted_non_training_backup_survives_real_data(self, configured):
        configured(
            mongodb_database="marketplace",
            **self.BACKUP,
            backup_llm_trains_on_prompts=False,
        )
        names = [p.name for p in llm_client.get_providers()]
        assert names == ["azure", "backup"], "the backup must sit directly behind azure"

    def test_it_makes_the_chain_survive_azure_going_away(self, configured):
        """The whole point: on expiry day, with the Azure key removed,
        real data still has somewhere compliant to go."""
        configured(
            mongodb_database="marketplace",
            azure_openai_endpoint=None,
            azure_openai_api_key=None,
            **self.BACKUP,
            backup_llm_trains_on_prompts=False,
        )
        assert [p.name for p in llm_client.get_providers()] == ["backup"]

    def test_an_unasserted_backup_is_dropped_from_real_data(self, configured):
        """THE DEFAULT IS THE SAFE ONE.

        backup_llm_trains_on_prompts defaults to True because we cannot
        verify somebody else's terms. An operator who fills in the three
        BACKUP_LLM_* values and says nothing about training gets a
        provider treated exactly like Groq - useful for demo data, gone
        the moment real customers are configured. Defaulting the other
        way would turn a hurried .env edit into the exact leak this
        design exists to prevent.
        """
        configured(mongodb_database="marketplace", **self.BACKUP)
        assert [p.name for p in llm_client.get_providers()] == ["azure"]

    def test_an_unasserted_backup_is_still_used_for_demo_data(self, configured):
        configured(mongodb_database="marketplace_demo", **self.BACKUP)
        assert "backup" in [p.name for p in llm_client.get_providers()]

    def test_a_half_filled_backup_is_treated_as_absent(self, configured):
        """Same rule as azure_openai_configured: partial configuration
        must not raise, or a typo becomes an outage."""
        configured(
            mongodb_database="marketplace_demo",
            backup_llm_base_url="https://compliant.example.com/v1",
            backup_llm_api_key=None,
            backup_llm_model=None,
        )
        assert "backup" not in [p.name for p in llm_client.get_providers()]

    def test_an_azure_shaped_backup_carries_its_api_version(self, configured):
        """kind=azure changes the URL shape and the auth header, and an
        api_version that failed to travel would 404 confusingly."""
        configured(
            mongodb_database="marketplace_demo",
            **self.BACKUP,
            backup_llm_kind="azure",
            backup_llm_api_version="2024-12-01-preview",
        )
        backup = next(p for p in llm_client.get_providers() if p.name == "backup")
        url, headers, _ = llm_client.build_request(backup, {"model": backup.model})
        assert "/openai/deployments/some-model/chat/completions" in url
        assert "api-version=2024-12-01-preview" in url
        assert "api-key" in headers


class TestChainAssessment:
    """What startup logging and /health read. It has to fire on the
    configuration we are actually running, or it is decoration."""

    def test_a_lone_provider_on_real_data_reads_as_at_risk(self, configured):
        configured(mongodb_database="marketplace", azure_credit_expires="2099-01-01")
        status = llm_client.assess_chain()
        assert status.level == "at_risk"
        assert status.redundant is False
        assert any("only one usable provider" in r for r in status.reasons)

    def test_adding_a_compliant_backup_clears_the_warning(self, configured):
        configured(
            mongodb_database="marketplace",
            azure_credit_expires="2099-01-01",
            backup_llm_base_url="https://compliant.example.com/v1",
            backup_llm_api_key="backup-key",
            backup_llm_model="some-model",
            backup_llm_trains_on_prompts=False,
        )
        status = llm_client.assess_chain()
        assert status.level == "ok"
        assert status.redundant is True

    def test_an_expired_credit_is_critical(self, configured):
        configured(mongodb_database="marketplace", azure_credit_expires="2020-01-01")
        assert llm_client.assess_chain().level == "critical"

    def test_no_usable_provider_is_critical_rather_than_an_exception(self, configured):
        """assess_chain is called by /health, which must answer even when
        the chain cannot serve anyone."""
        configured(
            mongodb_database="marketplace",
            azure_openai_endpoint=None,
            azure_openai_api_key=None,
        )
        status = llm_client.assess_chain()
        assert status.level == "critical"
        assert status.providers == ()

    def test_an_unparseable_expiry_date_does_not_crash_startup(self, configured):
        configured(mongodb_database="marketplace_demo", azure_credit_expires="next tuesday")
        assert llm_client.assess_chain().credit_days is None


class TestTheFlagIsNotWhatProtectsUs:
    def test_no_flag_is_required_for_the_restriction_to_apply(self, configured):
        """Keyed on the DATABASE, deliberately.

        A --real-data flag protects only the person who remembers to
        think about it. Whoever points .env at real customers gets the
        restriction whether or not they were thinking about providers.
        """
        configured(mongodb_database="marketplace")
        assert [p.name for p in llm_client.get_providers()] == ["azure"]
