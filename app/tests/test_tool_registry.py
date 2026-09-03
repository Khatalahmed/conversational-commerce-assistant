"""
WHAT:
    Invariants over the tool surface itself - the 34 schemas in
    agent/tools.py, the registry that maps them to repo functions, and
    the trimmers that shrink their results.

WHY THIS FILE EXISTS:
    Four repo functions were written, tested and then never registered
    (seller_repo's four, categories_repo, carts_repo.get_cart_total,
    account_repo.get_followers_following). Every one of them had a
    passing test. None of them was reachable from chat, because a test
    that calls a repo function directly proves the query works and says
    nothing about whether the assistant can ever ask for it.

    The gap is structural, so the guard is too: the checks below are
    about the SHAPE of the tool surface rather than about any one tool.
    They catch a schema with no implementation, an implementation with
    no schema, and a trimmer pointed at a tool that no longer exists.

    THE SECURITY INVARIANT IS THE IMPORTANT ONE.
    test_no_schema_accepts_an_identity_parameter is what makes the
    remaining unwired code safe to wire up later. seller_repo's four
    functions all take seller_id FIRST, and registering them as-is would
    hand the model a parameter for "whose revenue do you want" - the
    exact cross-user leak that keeping user_id out of every schema
    exists to prevent. Whoever wires them up will trip this test before
    they ship it.

FLOW:
    Pure unit tests over the module-level structures. No database, no
    network, no LLM.
"""

import inspect

import pytest

from app.agent.tool_executor import _TRIMMERS
from app.agent.tools import TOOL_REGISTRY, TOOLS

SCHEMA_NAMES = [tool["function"]["name"] for tool in TOOLS]

# Anything that names WHOSE data is being asked for. The verified JWT
# subject is the only acceptable source, and tool_executor injects it -
# so none of these may ever appear in a schema the model fills in.
IDENTITY_PARAMETERS = {
    "user_id", "userId", "user",
    "seller_id", "sellerId",
    "buyer_id", "buyerId",
    "customer_id", "customerId",
    "account_id", "accountId",
}

# THE ONE DELIBERATE EXCEPTION, and the line it draws is worth stating
# because seller_repo sits right on the other side of it.
#
# get_seller_trust_info(seller_id) looks a seller up as a PUBLIC ENTITY,
# the way get_product_detail looks up a product: rating, followers,
# sales count - what any buyer sees on the storefront before deciding to
# buy. The seller here is the SUBJECT of the question, not the person
# asking it, so there is nothing to scope to the caller.
#
# seller_repo.get_sales_performance takes the same seller_id and returns
# monthlyRevenue. The difference is not who is asking - it is what the
# field allowlist lets out of the database. That is exactly why wiring
# seller_repo up with seller_id as a tool parameter would be a leak
# while this entry is not, and why the entry is listed here explicitly
# rather than the rule being softened to let both through.
PUBLIC_SUBJECT_PARAMETERS = {"get_seller_trust_info.seller_id"}


class TestEverySchemaIsReachable:
    def test_every_schema_has_an_implementation(self):
        """A schema with no registry entry is a tool the model will call
        and get "Unknown tool" back from."""
        assert set(SCHEMA_NAMES) - set(TOOL_REGISTRY) == set()

    def test_every_registered_tool_has_a_schema(self):
        """The other direction, and the one that actually bit us: a repo
        function wired into the registry but never described to the
        model is code that cannot run."""
        assert set(TOOL_REGISTRY) - set(SCHEMA_NAMES) == set()

    def test_no_tool_name_is_declared_twice(self):
        duplicates = {n for n in SCHEMA_NAMES if SCHEMA_NAMES.count(n) > 1}
        assert duplicates == set()

    def test_every_trimmer_targets_a_real_tool(self):
        """A trimmer keyed on a renamed tool silently stops running, and
        the result it was shrinking goes to the model whole."""
        assert set(_TRIMMERS) - set(SCHEMA_NAMES) == set()


class TestIdentityIsNeverTheModelsToChoose:
    def test_no_schema_accepts_an_identity_parameter(self):
        """The single rule the whole data-safety design rests on.

        If the model can name whose data it wants, every other layer -
        the read-only role, the field allowlist, the sanitizer - is
        protecting the wrong person's row perfectly.
        """
        offenders = []
        for tool in TOOLS:
            function = tool["function"]
            properties = (function.get("parameters") or {}).get("properties") or {}
            for parameter in properties:
                qualified = f"{function['name']}.{parameter}"
                if parameter in IDENTITY_PARAMETERS and qualified not in PUBLIC_SUBJECT_PARAMETERS:
                    offenders.append(qualified)
        assert offenders == [], (
            "identity must come from the verified JWT, never from the model: "
            f"{offenders}"
        )

    def test_the_exception_list_has_not_gone_stale(self):
        """An exception for a schema that no longer exists is not
        harmless - it is a standing permission nobody reviews."""
        declared = {
            f"{t['function']['name']}.{p}"
            for t in TOOLS
            for p in ((t["function"].get("parameters") or {}).get("properties") or {})
        }
        assert PUBLIC_SUBJECT_PARAMETERS <= declared

    @pytest.mark.parametrize("name", sorted(TOOL_REGISTRY))
    def test_a_tool_needing_user_id_actually_accepts_one(self, name):
        """needs_user_id drives a keyword injection, so a mismatch is a
        TypeError at call time - during a real conversation, surfacing
        as a flat "this lookup failed"."""
        func, needs_user_id = TOOL_REGISTRY[name]
        parameters = inspect.signature(func).parameters
        if needs_user_id:
            assert "user_id" in parameters, f"{name} is flagged needs_user_id but has no user_id"
        else:
            assert "user_id" not in parameters, (
                f"{name} takes user_id but is not flagged needs_user_id - it would "
                f"be called without one"
            )

    @pytest.mark.parametrize("name", sorted(TOOL_REGISTRY))
    def test_no_schema_parameter_is_unknown_to_its_function(self, name):
        """Every property the model may fill in has to be a real keyword
        argument, or the call raises."""
        func, _ = TOOL_REGISTRY[name]
        parameters = inspect.signature(func).parameters
        schema = next(t for t in TOOLS if t["function"]["name"] == name)["function"]
        properties = (schema.get("parameters") or {}).get("properties") or {}
        unknown = [p for p in properties if p not in parameters]
        assert unknown == [], f"{name} declares parameters its function cannot take: {unknown}"

    @pytest.mark.parametrize("name", sorted(TOOL_REGISTRY))
    def test_every_required_argument_is_either_supplied_or_injected(self, name):
        """A function argument with no default, absent from the schema
        and not injected, can never be filled in."""
        func, needs_user_id = TOOL_REGISTRY[name]
        schema = next(t for t in TOOLS if t["function"]["name"] == name)["function"]
        properties = (schema.get("parameters") or {}).get("properties") or {}

        missing = [
            arg
            for arg, param in inspect.signature(func).parameters.items()
            if param.default is inspect.Parameter.empty
            and arg not in properties
            and not (needs_user_id and arg == "user_id")
        ]
        assert missing == [], f"{name} can never receive: {missing}"
