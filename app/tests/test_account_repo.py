"""
Permanent test suite for account_repo.py (PDF §10) - the logged-in
user's own profile data.
"""

from app.repos import account_repo


class TestIsSeller:
    async def test_real_seller_returns_true(self, real_seller):
        result = await account_repo.is_seller(str(real_seller["_id"]))
        assert result is True

    async def test_unknown_user_returns_false_not_error(self, db):
        result = await account_repo.is_seller("000000000000000000000000")
        assert result is False


class TestGetDefaultAddress:
    async def test_returns_real_address(self, real_address):
        result = await account_repo.get_default_address(str(real_address["user"]))
        assert result is not None
        assert result["city"] == real_address["city"]

    async def test_unknown_user_returns_none_not_error(self, db):
        result = await account_repo.get_default_address("000000000000000000000000")
        assert result is None


class TestGetUnreadNotifications:
    async def test_returns_valid_structure(self, real_address):
        # any real user_id works here - structure is what's being checked
        result = await account_repo.get_unread_notifications(str(real_address["user"]))
        assert "unreadCount" in result
        assert isinstance(result["samples"], list)


class TestGetFollowersFollowing:
    async def test_invalid_kind_raises_value_error(self, real_seller):
        try:
            await account_repo.get_followers_following(str(real_seller["_id"]), kind="invalid")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    async def test_resolves_real_followers_into_usernames(self, real_user_with_followers):
        result = await account_repo.get_followers_following(
            str(real_user_with_followers["_id"]), kind="followers"
        )
        assert result is not None
        assert result["rawCount"] == len(real_user_with_followers["followers"])
        # resolvedCount may be LOWER: a follower whose account was
        # deleted cannot be named. That gap is reported, not hidden.
        assert result["resolvedCount"] <= result["rawCount"]

    async def test_no_internal_ids_reach_the_caller(self, real_user_with_followers):
        """The system prompt forbids surfacing internal IDs, so they must
        not be in the result for the model to surface in the first
        place."""
        result = await account_repo.get_followers_following(
            str(real_user_with_followers["_id"]), kind="followers"
        )
        assert all(set(u) <= {"username", "businessName"} for u in result["users"])


class TestGetFollowersFollowingIsReachableFromChat:
    """The repo function above was correct and TESTED for weeks while
    being unreachable, because nothing registered it as a tool. These
    two assert the wiring, which is the part that was missing."""

    async def test_it_is_registered_with_server_side_identity(self):
        from app.agent.tools import TOOL_REGISTRY

        func, needs_user_id = TOOL_REGISTRY["get_followers_or_following"]
        assert func is account_repo.get_followers_following
        assert needs_user_id is True

    async def test_the_result_is_trimmed_before_the_model_sees_it(self):
        """The repo resolves EVERY follower. A popular account would
        otherwise put an unbounded array into the prompt."""
        from app.agent.tool_executor import LLM_LIST_LIMIT_CAP, _trim_follow_list

        many = {
            "kind": "followers",
            "rawCount": 500,
            "resolvedCount": 480,
            "users": [{"username": f"user{i}", "businessName": None} for i in range(480)],
        }
        trimmed = _trim_follow_list(many)

        assert len(trimmed["sample"]) == LLM_LIST_LIMIT_CAP
        # The COUNTS survive whole - they are the actual answer.
        assert trimmed["rawCount"] == 500
        assert trimmed["resolvedCount"] == 480