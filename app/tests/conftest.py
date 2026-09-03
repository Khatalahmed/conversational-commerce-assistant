"""
WHAT:
    Shared pytest fixtures for the whole test suite - one real MongoDB
    connection per test session, plus fixtures that pull REAL,
    dynamically-discovered data from the sandbox (a real order, a real
    bargain with a counter-offer, etc.) rather than hardcoded IDs that
    would silently break the moment sandbox data changes.

WHY THIS APPROACH:
    Every fixture here follows the exact same "find real data
    dynamically" pattern we used throughout Phases 4-9's throwaway
    scripts - Phase 10 just makes that pattern PERMANENT instead of
    rewriting it fresh in every test file.

MECHANISM:
    pytest-asyncio's fixture decorator lets fixtures themselves be
    async (needed since our repos are all async). scope="session" on
    the db connection means it connects ONCE for the entire test run,
    not once per test - matching how the real app behaves.
"""

import pytest
import pytest_asyncio

from app.db import redis_client
from app.db.connection import close_mongo_connection, connect_to_mongo, get_database
from app.memory import session_store
from app.security import rate_limit


def pytest_collection_modifyitems(items):
    """Marks every test that needs a live MongoDB as `needs_db`.

    WHY IT IS DERIVED RATHER THAN WRITTEN DOWN. Roughly a hundred tests
    reach the sandbox, and a marker applied by hand to each is a marker
    that will be forgotten on the hundred-and-first - at which point CI
    either starts failing for a reason unrelated to the change, or
    quietly stops covering something. The `db` fixture is already the
    honest signal: request it and you need a database. This reads that
    signal instead of asking anyone to repeat it.

    It lets CI run the hermetic majority with `-m "not needs_db"`.
    GitHub's runners get a fresh IP on every job and Atlas only accepts
    allow-listed addresses, so the alternative is opening the cluster to
    the internet - which is not a trade worth making to run tests.
    """
    for item in items:
        if "db" in getattr(item, "fixturenames", ()):
            item.add_marker(pytest.mark.needs_db)


@pytest.fixture(autouse=True)
def _clean_rate_limit_state():
    """Every test starts with an empty rate-limit table.

    The limiter is per-user and process-global, and the suite drives
    dozens of /chat calls as the SAME buyer within a few seconds - so
    without this, tests would begin failing with 429s purely because of
    how many ran before them. Autouse rather than opt-in: a test author
    should not have to know the limiter exists to write a passing test.
    """
    rate_limit.reset()
    session_store.reset_for_tests()

    # PIN THE SESSION STORE TO IN-PROCESS MEMORY FOR EVERY TEST.
    #
    # Two reasons, both found the hard way. Hermeticity: without this the
    # suite reads and writes whatever real Redis happens to be in .env,
    # so a developer's local data and the tests interfere with each
    # other. Speed: reset_for_tests() clears the cached backend, so every
    # single test would otherwise reconnect and PING - which took the
    # suite from 11 seconds to 89.
    #
    # Tests that WANT Redis monkeypatch these two attributes themselves,
    # and monkeypatch wins over what is set here. Pinning them ALSO pins
    # the rate limiter, which now shares the same connection - so the
    # limiter's own tests choose their backend the same way.
    redis_client._client = None
    redis_client._checked = True

    yield
    rate_limit.reset()
    session_store.reset_for_tests()


@pytest_asyncio.fixture(scope="session")
async def db():
    await connect_to_mongo()
    yield get_database()
    await close_mongo_connection()


@pytest_asyncio.fixture
async def real_order(db):
    order = await db.orders.find_one({})
    assert order, "No orders found in sandbox"
    return order


@pytest_asyncio.fixture
async def real_order_with_invoice(db):
    order = await db.orders.find_one({"invoice.url": {"$exists": True}})
    assert order, "No order with an invoice found in sandbox"
    return order


@pytest_asyncio.fixture
async def real_order_without_invoice(db):
    order = await db.orders.find_one({"invoice": {"$exists": False}})
    assert order, "No order without an invoice found in sandbox"
    return order


@pytest_asyncio.fixture
async def real_cancellable_order(db):
    order = await db.orders.find_one({"status": {"$in": ["pending", "confirmed"]}})
    assert order, "No cancellable order found in sandbox"
    return order


@pytest_asyncio.fixture
async def real_delivered_order(db):
    order = await db.orders.find_one({"status": "delivered"})
    assert order, "No delivered order found in sandbox"
    return order

@pytest_asyncio.fixture
async def real_product_with_variants(db):
    product = await db.products.find_one({"variants.0": {"$exists": True}})
    assert product, "No product with variants found in sandbox"
    return product


@pytest_asyncio.fixture
async def real_bargain_with_counter_offer(db):
    bargain = await db.bargains.find_one({"counterOffer": {"$exists": True}})
    assert bargain, "No bargain with a counter-offer found in sandbox"
    return bargain


@pytest_asyncio.fixture
async def real_bargain_without_counter_offer(db):
    bargain = await db.bargains.find_one({"counterOffer": {"$exists": False}})
    assert bargain, "No bargain without a counter-offer found in sandbox"
    return bargain

@pytest_asyncio.fixture
async def real_cart_with_items(db):
    cart = await db.carts.find_one({"items.0": {"$exists": True}})
    assert cart, "No cart with items found in sandbox"
    return cart


@pytest_asyncio.fixture
async def real_inactive_coupon(db):
    coupon = await db.coupons.find_one({"isActive": False})
    assert coupon, "No inactive coupon found in sandbox"
    return coupon


@pytest_asyncio.fixture
async def real_session_with_products(db):
    session = await db.livesessions.find_one({"productSequence.0": {"$exists": True}})
    assert session, "No live session with products found in sandbox"
    return session

@pytest_asyncio.fixture
async def real_bit_with_hashtags(db):
    bit = await db.bits.find_one({"hashtags.0": {"$exists": True}})
    assert bit, "No bit with hashtags found in sandbox"
    return bit


@pytest_asyncio.fixture
async def real_bit_with_products(db):
    bit = await db.bits.find_one({"products.0": {"$exists": True}})
    assert bit, "No bit with tagged products found in sandbox"
    return bit


@pytest_asyncio.fixture
async def real_review(db):
    review = await db.reviews.find_one({})
    assert review, "No reviews found in sandbox"
    return review

@pytest_asyncio.fixture
async def real_seller(db):
    seller = await db.users.find_one({"sellerStatus": {"$exists": True, "$ne": None}})
    assert seller, "No seller found in sandbox"
    return seller


@pytest_asyncio.fixture
async def real_user_with_followers(db):
    user = await db.users.find_one({"followers.0": {"$exists": True}})
    assert user, "No user with followers found in sandbox"
    return user


@pytest_asyncio.fixture
async def real_address(db):
    address = await db.addresses.find_one({})
    assert address, "No address found in sandbox"
    return address


@pytest_asyncio.fixture
async def real_payout(db):
    payout = await db.payouts.find_one({})
    assert payout, "No payout found in sandbox"
    return payout


@pytest_asyncio.fixture
async def real_seller_coupon(db):
    coupon = await db.coupons.find_one({})
    assert coupon, "No coupon found in sandbox"
    return coupon

@pytest_asyncio.fixture
async def two_different_buyers_orders(db):
    orders = await db.orders.find({}).to_list(length=20)
    first = orders[0]
    other = next((o for o in orders if str(o["buyerId"]) != str(first["buyerId"])), None)
    assert other, "Need at least 2 different buyers' orders in sandbox to run this test"
    return first, other