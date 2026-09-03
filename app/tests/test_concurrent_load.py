"""
Lightweight concurrency test for Phase 10.4. Deliberately scoped to the
DATABASE layer only (not the LLM) - Groq's free tier has a real 30
requests/minute limit, so a heavy concurrent LLM load test would just
prove we can hit that limit, not reveal anything about our own code.
Confirms the async connection pool (Phase 2.3) handles genuinely
simultaneous requests without crashing or mixing up results between
them - the part of "handling concurrent users" actually within our
control to verify at this stage.
"""

import asyncio

from app.repos import orders_repo


class TestConcurrentDatabaseAccess:
    async def test_ten_simultaneous_requests_dont_crash_or_mix_data(self, db):
        orders = await db.orders.find({}).to_list(length=10)
        assert len(orders) >= 5, "Need at least 5 real orders in sandbox for this test"

        # Fire multiple real repo calls at once, for DIFFERENT users,
        # simulating multiple people using the chatbot at the same
        # moment - the actual real-world scenario Phase 2's async
        # design exists to handle correctly.
        tasks = [
            orders_repo.get_order_status(str(o["buyerId"]), o["orderId"])
            for o in orders[:5]
        ]
        results = await asyncio.gather(*tasks)

        # Each result must correctly match ITS OWN order - proves no
        # cross-talk/mixing happened between simultaneous requests.
        for order, result in zip(orders[:5], results):
            assert result is not None
            assert result["orderId"] == order["orderId"]

    async def test_fifty_read_requests_complete_without_error(self, db):
        # Simple throughput sanity check - not a formal benchmark, just
        # confirming nothing degrades or errors under a batch of reads.
        tasks = [orders_repo.get_order_history("000000000000000000000000") for _ in range(50)]
        results = await asyncio.gather(*tasks)
        assert len(results) == 50
        assert all(r == [] for r in results)