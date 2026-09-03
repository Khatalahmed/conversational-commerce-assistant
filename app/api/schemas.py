"""
WHAT:
    Request/response shapes for the /chat and /health endpoints.

WHY THIS APPROACH:
    Pydantic validates incoming requests automatically - a malformed
    request (missing message, empty session_id) gets rejected with a
    clear 422 error before it ever reaches our orchestration logic,
    matching Phase 9.3's requirement.
"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(..., min_length=1, max_length=200)


class ProductCard(BaseModel):
    """One product the answer drew on, for the client to render as a
    tappable row. Carries no URL: the app navigates on product_id
    through its own stack, and the storefront has no product pages to link
    to. See tool_executor._product_cards for why the id travels beside
    the prose rather than inside it."""

    product_id: str
    name: str | None = None
    price: float | None = None
    discountedPrice: float | None = None
    # First catalogue image, or None for the four products that have
    # none. A card renders without it rather than not at all.
    image: str | None = None


class Attribution(BaseModel):
    """One claim in the reply, and the tool that vouched for it.

    Character offsets into the reply rather than the text alone, so a
    client marks the span by slicing. A reply that says "₹599" twice
    would otherwise have both occurrences marked from a single match.

    Only prices and order ids. A product NAME cannot be located in prose
    without named entity recognition, so the honest thing is to mark what
    can actually be stood behind - see agent/grounding.attribute.
    """

    start: int
    end: int
    text: str
    kind: str   # "price" | "order_id"
    tool: str


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    # Defaulted, so a client written against the old shape keeps
    # deserialising this response unchanged.
    products: list[ProductCard] = Field(default_factory=list)
    attribution: list[Attribution] = Field(default_factory=list)