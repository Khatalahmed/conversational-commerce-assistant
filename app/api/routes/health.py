"""
WHAT:
    Wires Phase 2.5's health check into an actual endpoint - used
    locally now, and by the cloud host's monitoring in Phase 11.

WHY THE LLM IS REPORTED HERE:
    A reachable database is not the same as a working assistant. While
    real customer data is configured the provider chain is one deep, on
    a credit with an expiry date - so the service can be perfectly
    healthy by every other measure and days from answering nothing at
    all. Monitoring should learn that before users do.

DELIBERATELY VAGUE, for the same reason check_database_health is:
    /health is unauthenticated, so anything returned here is readable by
    anyone who can reach the service. A level is enough for a monitor to
    alert on. The provider names, the day the credit lapses and the
    reasons all go to the LOGS, where operators can see them and
    strangers cannot - telling the internet the exact date this service
    stops working is not a liveness signal.
"""

from fastapi import APIRouter

from app.agent.llm_client import assess_chain
from app.db.connection import check_database_health

router = APIRouter()

# A chain that is "at_risk" still answers every request today, so it must
# not read as an outage to a monitor that pages someone. It is reported
# as ok-with-a-warning; only "critical" degrades the overall status.
_LLM_WIRE_STATUS = {"ok": "ok", "at_risk": "warning", "critical": "unavailable"}


@router.get("/health")
async def health():
    db_health = await check_database_health()
    chain = assess_chain()
    llm_status = _LLM_WIRE_STATUS[chain.level]

    degraded = db_health["status"] != "connected" or chain.level == "critical"
    return {
        "status": "degraded" if degraded else "ok",
        "database": db_health,
        "llm": {"status": llm_status, "redundant": chain.redundant},
    }