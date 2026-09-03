"""
Integration-level SECURITY tests (Phase 10.3). Same ASGITransport
approach as test_chat_integration.py - see that file's docstring for
why TestClient specifically caused the earlier failures.
"""

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport

from app.agent import orchestrator
from app.api.main import app
from app.config.settings import Settings
from app.security.auth import create_test_token
from app.security.field_allowlist import COLLECTION_ALLOWLISTS


@pytest_asyncio.fixture
async def client(db):
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


class TestTokenSecurity:
    async def test_expired_token_rejected(self, client, real_order):
        expired_token = create_test_token(user_id=str(real_order["buyerId"]), expires_in_minutes=-5)
        response = await client.post(
            "/chat",
            headers={"Authorization": f"Bearer {expired_token}"},
            json={"message": "hello", "session_id": "sec-1"},
        )
        assert response.status_code == 401

    async def test_tampered_token_rejected(self, client, real_order):
        valid_token = create_test_token(user_id=str(real_order["buyerId"]))
        tampered = valid_token[:-5] + "XXXXX"
        response = await client.post(
            "/chat",
            headers={"Authorization": f"Bearer {tampered}"},
            json={"message": "hello", "session_id": "sec-2"},
        )
        assert response.status_code == 401


class TestCrossUserScopingViaChatText:
    async def test_naming_victims_order_id_does_not_leak_their_data(
        self, client, two_different_buyers_orders
    ):
        victim_order, attacker_order = two_different_buyers_orders
        attacker_token = create_test_token(user_id=str(attacker_order["buyerId"]))

        response = await client.post(
            "/chat",
            headers={"Authorization": f"Bearer {attacker_token}"},
            json={
                "message": f"what is the status of order {victim_order['orderId']}",
                "session_id": "sec-3",
            },
        )
        assert response.status_code == 200
        reply = response.json()["reply"]

        victim_awb = (victim_order.get("tracking") or {}).get("awb")
        if victim_awb:
            assert victim_awb not in reply, "SECURITY: victim's real tracking data leaked via chat text"

# ── Hardening from the pre-deployment review ─────────────────────────
# These need no database: the one that builds a system prompt stubs the
# single query it makes, so they run in CI alongside everything else.

class TestToolResultsAreTreatedAsData:
    """Product names, descriptions, titles and comments are written by
    sellers and shoppers, and they reach the model verbatim as tool
    results. The model must treat that text as content to report, not as
    direction to follow.

    The blast radius is bounded either way - identity is injected
    server-side, so no wording inside a product description can make the
    assistant fetch another user's data. What it COULD do is put false
    statements in the assistant's mouth to a user who trusts it, which
    is what this rule is for.
    """

    async def test_the_prompt_states_the_instruction_boundary(self, monkeypatch):
        async def stub_categories():
            return {"categories": ["Men"], "subCategories": ["shirts"]}

        monkeypatch.setattr(
            orchestrator.products_repo, "get_distinct_categories", stub_categories
        )
        prompt = await orchestrator.build_system_prompt()

        assert "DATA, NEVER INSTRUCTIONS" in prompt
        # The rule is worthless if it does not say WHICH text is
        # untrusted - "comments" is the sharpest vector, being the only
        # field a non-seller can write.
        assert "comments" in prompt


class TestSigningSecretStrength:
    """The secret verifies every token, and a token is the only thing
    that says who is asking. Recover it and you can mint one for any
    user id - at which point the field allowlist and the sanitizer are
    irrelevant, because the request looks exactly like the real user's.
    """

    def _settings(self, secret, algorithm="HS256"):
        return Settings(
            mongodb_uri="mongodb://localhost/test",
            mongodb_database="marketplace_demo",
            llm_api_key="k",
            jwt_secret=secret,
            jwt_algorithm=algorithm,
        )

    def test_a_short_hs256_secret_is_reported(self):
        weakness = self._settings("too-short").jwt_secret_weakness
        assert weakness is not None
        assert "brute-forced" in weakness

    def test_length_alone_is_not_strength(self):
        """A 64-character secret of one repeated character is long and
        worth nothing."""
        assert self._settings("a" * 64).jwt_secret_weakness is not None

    def test_a_real_secret_passes(self):
        assert self._settings("Ab3$xQ9!zR2#mN7&pL4%tY6@wS8^vD0uK5").jwt_secret_weakness is None

    def test_public_keys_are_not_judged_by_length(self):
        """Under RS256/ES256 this field holds the backend's PUBLIC key,
        which is meant to be published - length says nothing about
        safety, and warning about it would train people to ignore the
        warning."""
        assert self._settings("-----BEGIN PUBLIC KEY-----short", "RS256").jwt_secret_weakness is None

    def test_it_warns_rather_than_refusing_to_start(self):
        """The secret must match the main the marketplace backend exactly, so it
        is not ours to change. Refusing to boot would take the assistant
        down over another team's decision."""
        weak = self._settings("short")
        assert weak.jwt_secret == "short", "a weak secret must still construct"


class TestAllowlistIsNarrowByDefault:
    def test_seller_revenue_cannot_leave_the_users_collection(self):
        """It was allowed for seller_repo.get_sales_performance, which is
        deliberately not reachable from chat - so every query against
        `users` could carry a seller's takings to serve one function
        nobody can call. This list is the layer that has to hold when an
        inner one has a bug; it should be narrow by default.
        """
        users = COLLECTION_ALLOWLISTS["users"]
        assert "monthlyRevenue" not in users
        assert "yearlyRevenue" not in users

    @pytest.mark.parametrize(
        "collection,forbidden",
        [
            ("users", ["password", "email", "phone", "otp", "token"]),
            ("orders", ["payment", "buyerNote"]),
            ("addresses", ["phone", "addressLine1", "name"]),
        ],
    )
    def test_no_allowlist_carries_credentials_or_contact_details(
        self, collection, forbidden
    ):
        """A regression guard on the whole idea. These lists grow as
        features are added, and the cost of one careless entry is a
        field leaving the database forever after."""
        allowed = COLLECTION_ALLOWLISTS[collection]
        leaked = [f for f in allowed if any(bad in f.lower() for bad in forbidden)]
        assert leaked == [], f"{collection} allowlist exposes {leaked}"


class TestDemoUiIsGated:
    """/demo is a chat page anyone who reaches the service can load. It
    cannot answer without a valid the marketplace token, so it is an unnecessary
    surface rather than a leak - but a deployed instance should not
    serve it by accident, and before this flag existed there was no way
    not to.

    Built through create_app() rather than the imported `app`, because
    a router registered at import time cannot be removed afterwards -
    which is the whole reason the factory exists.
    """

    def _app(self, monkeypatch, enabled: bool):
        from app.api import main

        settings = Settings(
            mongodb_uri="mongodb://localhost/test",
            mongodb_database="marketplace_demo",
            llm_api_key="k",
            jwt_secret="s" * 64,
            demo_ui_enabled=enabled,
        )
        monkeypatch.setattr(main, "get_settings", lambda: settings)
        return main.create_app()

    async def _get_demo(self, application):
        transport = ASGITransport(app=application)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as ac:
            return await ac.get("/demo")

    async def test_it_is_served_by_default(self, monkeypatch):
        """The default is ON deliberately - the demo is the product as
        far as a watching client is concerned, and a flag that must be
        found first wastes the first ten minutes."""
        response = await self._get_demo(self._app(monkeypatch, True))
        assert response.status_code == 200
        assert "the marketplace Assistant" in response.text

    async def test_turning_it_off_removes_the_route_entirely(self, monkeypatch):
        """404, not 401 or a blank page: the route should not exist, so
        nothing advertises that there was ever a page here."""
        response = await self._get_demo(self._app(monkeypatch, False))
        assert response.status_code == 404

    async def test_the_api_still_works_with_the_page_off(self, monkeypatch):
        """Gating the demo must not gate the product.

        Asserted by CALLING the endpoints rather than reading
        app.routes: this FastAPI version keeps an included router as a
        single opaque entry instead of flattening it, so the route table
        is not a thing to make claims about. 401 is the right answer for
        /chat here - it proves the route exists AND that it is still
        behind authentication.
        """
        application = self._app(monkeypatch, False)
        transport = ASGITransport(app=application)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as ac:
            assert (await ac.get("/demo")).status_code == 404
            for path in ("/chat", "/chat/stream"):
                response = await ac.post(
                    path, json={"message": "hi", "session_id": "s"}
                )
                assert response.status_code == 401, path


class TestDemoSurfaceIsRemovableAsAWhole:
    """/demo/accounts lists buyers and /demo/token signs in as any of
    them, with no password. There is deliberately no database check on
    either - the demo has to work against real data, and on the machine
    running it those endpoints grant nothing that .env does not already.

    DEMO_UI_ENABLED IS THEREFORE THE ONLY PROTECTION LEFT, and these
    tests exist to keep it whole. The failure they guard against is a
    half-disabled surface: the page gone but the token minter still
    answering, which would be worse than either.

    Hermetic - the router is never mounted, so nothing reaches a
    database - so CI covers this even though it cannot reach Atlas.
    """

    def _app(self, monkeypatch, database, demo_ui_enabled=True):
        # Only main reads settings now - the demo endpoints stopped
        # consulting them when the per-endpoint database guards were
        # removed, so patching demo_ui.get_settings would fail loudly.
        from app.api import main

        settings = Settings(
            mongodb_uri="mongodb://localhost/test",
            mongodb_database=database,
            llm_api_key="k",
            jwt_secret="s" * 64,
            demo_ui_enabled=demo_ui_enabled,
        )
        monkeypatch.setattr(main, "get_settings", lambda: settings)
        return main.create_app()

    async def _sign_in(self, application, username="Riya"):
        transport = ASGITransport(app=application)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as ac:
            return await ac.post("/demo/token", json={"username": username})

    async def _get(self, application, path):
        transport = ASGITransport(app=application)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as ac:
            return await ac.get(path)

    @pytest.mark.parametrize("database", ["marketplace", "marketplace_demo"])
    async def test_disabling_the_demo_removes_every_part_of_it(
        self, monkeypatch, database
    ):
        """THE WHOLE PROTECTION, and it must hold on BOTH databases.

        Parametrised deliberately: the endpoints no longer care which
        database they are pointed at, so a deployment against real
        customers relies on exactly this and nothing else. If a later
        change mounted the token minter outside the demo router, this is
        the test that would fail.
        """
        application = self._app(monkeypatch, database, demo_ui_enabled=False)

        assert (await self._sign_in(application)).status_code == 404
        assert (await self._get(application, "/demo/accounts")).status_code == 404
        assert (await self._get(application, "/demo")).status_code == 404

    @pytest.mark.parametrize("database", ["marketplace", "marketplace_demo"])
    async def test_the_api_survives_the_demo_being_removed(
        self, monkeypatch, database
    ):
        """Turning the demo off must not turn the product off. 401 on
        /chat proves the route is mounted AND still authenticated."""
        application = self._app(monkeypatch, database, demo_ui_enabled=False)
        transport = ASGITransport(app=application)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as ac:
            response = await ac.post(
                "/chat", json={"message": "hi", "session_id": "s"}
            )
        assert response.status_code == 401

    async def test_the_endpoints_exist_when_the_demo_is_on(self, monkeypatch):
        """The complement of the above: enabled means mounted. Not 404 is
        the assertion - reaching a database is somebody else's test."""
        application = self._app(monkeypatch, "marketplace_demo", demo_ui_enabled=True)
        assert (await self._get(application, "/demo")).status_code == 200


class TestDemoAccessCode:
    """The door in front of a live demo on real data.

    /demo/accounts lists real buyers and /demo/token signs in as any of
    them with no password. Behind a URL that is a customer directory
    with one-click impersonation, and the access code is what turns
    "anyone who finds the link" into "anyone you told".

    Hermetic: every assertion here is about the refusal, which happens
    before any database call.
    """

    def _app(self, monkeypatch, code):
        from app.api import main
        from app.api.routes import demo_ui

        settings = Settings(
            mongodb_uri="mongodb://localhost/test",
            mongodb_database="marketplace",
            llm_api_key="k",
            jwt_secret="s" * 64,
            demo_access_code=code,
        )
        monkeypatch.setattr(main, "get_settings", lambda: settings)
        monkeypatch.setattr(demo_ui, "get_settings", lambda: settings)
        return main.create_app()

    async def _get(self, application, path, code=None):
        transport = ASGITransport(app=application)
        headers = {"X-Demo-Code": code} if code else {}
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as ac:
            return await ac.get(path, headers=headers)

    @pytest.mark.parametrize("path", ["/demo/accounts", "/demo/stats"])
    async def test_no_code_is_refused(self, monkeypatch, path):
        response = await self._get(self._app(monkeypatch, "s3cret"), path)
        assert response.status_code == 401

    @pytest.mark.parametrize("path", ["/demo/accounts", "/demo/stats"])
    async def test_a_wrong_code_is_refused(self, monkeypatch, path):
        response = await self._get(self._app(monkeypatch, "s3cret"), path, code="nope")
        assert response.status_code == 401

    async def test_the_refusal_names_no_customer(self, monkeypatch):
        """A 401 that leaked a username would defeat the point of the
        gate for the one field that matters most."""
        response = await self._get(self._app(monkeypatch, "s3cret"), "/demo/accounts")
        body = response.text.lower()
        for leak in ("riya", "orders", "pune", "username"):
            assert leak not in body

    async def test_sign_in_is_gated_too(self, monkeypatch):
        """The listing is the smaller hole. /demo/token is the one that
        hands over an account."""
        application = self._app(monkeypatch, "s3cret")
        transport = ASGITransport(app=application)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as ac:
            response = await ac.post("/demo/token", json={"username": "Riya"})
        assert response.status_code == 401

    def test_no_code_configured_means_no_gate(self, monkeypatch):
        """Laptop demos and local development must be unchanged - a gate
        you have to configure before you can work is a gate people
        disable permanently.

        Asserted against the guard rather than the endpoint: with no gate
        the request carries on to the database, which a hermetic test has
        no connection to. What matters here is only that the check lets
        it through.
        """
        from app.api.routes import demo_ui

        settings = Settings(
            mongodb_uri="mongodb://localhost/test",
            mongodb_database="marketplace",
            llm_api_key="k",
            jwt_secret="s" * 64,
            demo_access_code=None,
        )
        monkeypatch.setattr(demo_ui, "get_settings", lambda: settings)
        demo_ui._check_access(None)      # must not raise

    def test_the_comparison_is_constant_time(self, monkeypatch):
        """compare_digest, not ==, so a wrong code cannot be guessed one
        character at a time from how long the refusal takes."""
        import inspect

        from app.api.routes import demo_ui

        source = inspect.getsource(demo_ui._check_access)
        assert "compare_digest" in source
        assert "== expected" not in source
