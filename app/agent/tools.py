"""
WHAT:
    Formal tool definitions the LLM can choose to call, covering Phase
    5's top-3 priority areas: Order Management, Product Discovery,
    Bargaining, Live Shopping, Bits and Reviews (35 tools total).
    Each tool maps to an exact repo
    function from Phase 4.

WHY THIS APPROACH:
    Starting with a focused subset rather than all ~41 repo functions
    at once - a smaller/faster model picks the right tool more reliably
    from a smaller, well-organized set than from everything at once.
    Live Shopping, Bits, Reviews, Account, and Seller-side tools can be
    added the same way once this is proven working.

    DESCRIPTIONS ARE DELIBERATELY TERSE. These schemas are re-sent on
    EVERY round of the tool-calling loop, and they measured as the
    single largest slice of the prompt - 783 of 1,453 tokens per round,
    more than the entire system prompt. On Groq's free tier (8,000
    tokens/minute) that is the difference between roughly five and
    eight rounds a minute. So each description carries only what
    DISTINGUISHES this tool from its neighbours; anything else is
    weight paid for on every round.

    IN PARTICULAR, CROSS-TOOL ORDERING RULES LIVE IN THE SYSTEM PROMPT,
    NOT HERE. "Call get_order_history first", "find the product before
    asking about its bargain" - those describe how tools relate to each
    other, they are already stated once in orchestrator.py's system
    prompt, and repeating them per-tool was paying for the same
    sentence twice. Rule of thumb: what ONE tool does belongs here;
    how tools SEQUENCE belongs in the system prompt.

FLOW:
    app/agent/orchestrator.py sends this TOOLS list to the LLM on every
    call. When the LLM decides to call one, tool_executor.py looks it
    up in TOOL_REGISTRY to find the real repo function to run.

LOGIC:
    CRITICAL: user_id is NEVER a parameter exposed to the LLM in any
    tool schema below - per Phase 3's core security rule, identity only
    ever comes from the verified JWT, injected by our own code
    (tool_executor.py), never something the model is asked to supply.
    TOOL_REGISTRY's needs_user_id flag drives that injection.
    get_distinct_categories is deliberately NOT exposed as a callable
    tool - its results are embedded directly into the system prompt
    (orchestrator.py) once per conversation, since it's grounding data
    the model should always have, not something worth a round-trip.

    No "limit" parameter carries a stated default: tool_executor.py
    hard-caps every limit at LLM_LIST_LIMIT_CAP regardless of what the
    model asks for, so documenting a default here would be tokens spent
    on a number our own code overrides anyway.

MECHANISM:
    TOOLS follows the OpenAI-compatible function-calling schema format,
    which both providers in llm_client.py accept unmodified.
"""

from app.repos import (
    account_repo,
    bargains_repo,
    bits_repo,
    carts_repo,
    coupons_repo,
    livesessions_repo,
    orders_repo,
    products_repo,
    reviews_repo,
)

_ORDER_ID = {"order_id": {"type": "string"}}

TOOLS = [
    # -- ORDERS ------------------------------------------------------
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Current status and stage history of one order.",
            "parameters": {
                "type": "object",
                "properties": _ORDER_ID,
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_delivery_estimate",
            "description": "Expected delivery date for one order.",
            "parameters": {
                "type": "object",
                "properties": _ORDER_ID,
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_order_history",
            "description": "The user's recent orders. Use when no order ID was given.",
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_order_detail",
            "description": "Items, pricing breakdown and delivery city/state for one order.",
            "parameters": {
                "type": "object",
                "properties": _ORDER_ID,
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_invoice",
            "description": "Invoice link for one order, if one exists.",
            "parameters": {
                "type": "object",
                "properties": _ORDER_ID,
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_tracking",
            "description": "Courier and AWB tracking for one order.",
            "parameters": {
                "type": "object",
                "properties": _ORDER_ID,
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_cancellation_eligibility",
            # "Informational only" stays: it is a safety rule about what
            # this tool does NOT do, not a sequencing hint.
            "description": "Whether one order can still be cancelled. Informational only - never cancels.",
            "parameters": {
                "type": "object",
                "properties": _ORDER_ID,
                "required": ["order_id"],
            },
        },
    },
    # -- PRODUCTS ----------------------------------------------------
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "Find products by category, price, colour, size or stock. Use only the exact category values listed in your instructions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string"},
                    "sub_category": {"type": "string"},
                    "min_price": {"type": "number"},
                    "max_price": {"type": "number"},
                    "color": {"type": "string"},
                    "size": {"type": "string"},
                    "in_stock_only": {"type": "boolean"},
                    "limit": {"type": "integer"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_products_by_name",
            "description": "Find a product by name, when the user names it rather than filtering.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_variant_stock",
            "description": "Stock for one colour/size of a product.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "string"},
                    "color": {"type": "string"},
                    "size": {"type": "string"},
                },
                "required": ["product_id", "color", "size"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_product_detail",
            "description": "Description, condition, price and variants of one product.",
            "parameters": {
                "type": "object",
                "properties": {"product_id": {"type": "string"}},
                "required": ["product_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_products_semantically",
            "description": "Find products by DESCRIPTION or vibe rather than name or filters - \"something cosy for winter\", \"a gift for my mother\".",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_similar_products",
            "description": "Products similar to one the user is already looking at. Use for \"something like this\", \"alternatives\", \"cheaper version\".",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["product_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_trending_products",
            "description": "Trending and top-pick products.",
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recommendations",
            # THE FALLBACK IS PART OF THE CONTRACT, so it belongs in the
            # description. Measured: asked "What should I buy?" the model
            # called nothing and replied with four narrowing questions -
            # a fair reading of "from their saved and liked history",
            # which sounds like it needs a history to work. The repo
            # falls back to trending for a user with none, so there was
            # always an answer to give; only the schema said otherwise.
            "description": "Personalised picks for the current user. Takes no "
                           "required arguments and always returns products - "
                           "based on their saved and liked items, falling back "
                           "to trending when they have none yet.",
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_seller_info",
            "description": "Who sells a product, their rating and follower count.",
            "parameters": {
                "type": "object",
                "properties": {"product_id": {"type": "string"}},
                "required": ["product_id"],
            },
        },
    },
    # -- BARGAINING --------------------------------------------------
    {
        "type": "function",
        "function": {
            "name": "check_bargain_eligibility",
            "description": "Whether a product allows bargaining, and by how much.",
            "parameters": {
                "type": "object",
                "properties": {"product_id": {"type": "string"}},
                "required": ["product_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "suggest_offer_amount",
            "description": "A reasonable bargain offer for a product.",
            "parameters": {
                "type": "object",
                "properties": {"product_id": {"type": "string"}},
                "required": ["product_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            # NEEDS NOTHING FROM THE USER, and the description says so
            # because the sibling tools do not. Measured on
            # get_recommendations: a schema that sounded like it required
            # context made the model ask four questions instead of
            # calling it. Both bargain lookups below genuinely need an
            # id, so without this line the model reasonably assumes this
            # one does too.
            "name": "list_bargains",
            "description": "Every offer the user has made, newest first, and "
                           "what happened to each. Takes no required arguments "
                           "- use this for 'what happened to my offers', where "
                           "get_bargain_status needs a specific product.",
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_bargain_status",
            "description": "Status of the user's own bargain offer, by bargain ID or product ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "bargain_id": {"type": "string"},
                    "product_id": {"type": "string"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_counter_offer",
            "description": "The seller's counter-offer on the user's bargain, by bargain ID or product ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "bargain_id": {"type": "string"},
                    "product_id": {"type": "string"},
                },
                "required": [],
            },
        },
    },
    # -- LIVE SHOPPING -----------------------------------------------
    {
        "type": "function",
        "function": {
            "name": "get_live_now",
            "description": "Live selling sessions happening right now.",
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_session_products",
            "description": "Which products a live session is featuring, in order.",
            "parameters": {
                "type": "object",
                "properties": {"session_id": {"type": "string"}},
                "required": ["session_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_session_recap",
            "description": "Summary of a live session: viewers, revenue, reaction.",
            "parameters": {
                "type": "object",
                "properties": {"session_id": {"type": "string"}},
                "required": ["session_id"],
            },
        },
    },
    # -- BITS (short video) ------------------------------------------
    {
        "type": "function",
        "function": {
            "name": "get_trending_bits",
            "description": "Trending short videos (Bits).",
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_by_hashtag",
            "description": "Find Bits by hashtag or topic. The leading # is optional.",
            "parameters": {
                "type": "object",
                "properties": {
                    "hashtag": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["hashtag"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_tagged_products",
            "description": "Products tagged inside one Bit.",
            "parameters": {
                "type": "object",
                "properties": {"bit_id": {"type": "string"}},
                "required": ["bit_id"],
            },
        },
    },
    # -- REVIEWS & TRUST ---------------------------------------------
    # -- CART, COUPONS, ACCOUNT --------------------------------------
    # Added after a demo rehearsal: asked "what is in my cart?", the
    # assistant said it could not see the cart - while carts_repo sat
    # there, tested and working, simply not exposed. For a SHOPPING
    # assistant that is the worst possible answer, so these close the
    # obvious everyday questions a real user asks first.
    {
        "type": "function",
        "function": {
            "name": "get_cart",
            "description": "What is in the user's cart, with item names and the total.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_coupon_validity",
            "description": "Whether a coupon code is currently valid, and its discount. Read-only - never applies it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "cart_total": {"type": "number"},
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_unread_notifications",
            "description": "The user's unread notification count and a few recent ones.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_saved_items",
            "description": "Products and Bits the user has saved.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_followers_or_following",
            # The ENUM is load-bearing, not decoration. The repo raises
            # ValueError on any other value, which execute_tool would
            # turn into a flat "this lookup failed" - a dead end the
            # model cannot recover from. Constraining it in the schema
            # means the provider rejects the bad value before we run.
            "description": "How many people follow the user, or who the user follows, by username.",
            "parameters": {
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": ["followers", "following"],
                    },
                },
                "required": ["kind"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_default_address",
            "description": "The user's default delivery city and state.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_product_reviews",
            "description": "Average rating, review count and recent comments for a product.",
            "parameters": {
                "type": "object",
                "properties": {"product_id": {"type": "string"}},
                "required": ["product_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_seller_trust_info",
            "description": "A seller's rating, followers and sales history, by seller ID.",
            "parameters": {
                "type": "object",
                "properties": {"seller_id": {"type": "string"}},
                "required": ["seller_id"],
            },
        },
    },
]

# Maps each tool name to its real repo function and whether user_id must
# be injected by OUR code (never supplied by the LLM).
TOOL_REGISTRY = {
    "get_order_status": (orders_repo.get_order_status, True),
    "get_delivery_estimate": (orders_repo.get_delivery_estimate, True),
    "get_order_history": (orders_repo.get_order_history, True),
    "get_order_detail": (orders_repo.get_order_detail, True),
    "get_invoice": (orders_repo.get_invoice, True),
    "get_tracking": (orders_repo.get_tracking, True),
    "check_cancellation_eligibility": (orders_repo.check_cancellation_eligibility, True),
    "search_products": (products_repo.search_products, False),
    "get_variant_stock": (products_repo.get_variant_stock, False),
    "get_product_detail": (products_repo.get_product_detail, False),
    "search_products_semantically": (products_repo.search_products_semantically, False),
    "find_similar_products": (products_repo.find_similar_products, False),
    "get_trending_products": (products_repo.get_trending_products, False),
    "get_recommendations": (products_repo.get_recommendations, True),
    "get_seller_info": (products_repo.get_seller_info, False),
    "check_bargain_eligibility": (bargains_repo.check_bargain_eligibility, False),
    "suggest_offer_amount": (bargains_repo.suggest_offer_amount, False),
    "list_bargains": (bargains_repo.list_bargains, True),
    "get_bargain_status": (bargains_repo.get_bargain_status, True),
    "get_counter_offer": (bargains_repo.get_counter_offer, True),
    "search_products_by_name": (products_repo.search_products_by_name, False),
    # Live shopping, Bits and reviews are all PUBLIC data - none of
    # these take user_id, which is why every flag below is False.
    "get_live_now": (livesessions_repo.get_live_now, False),
    "get_session_products": (livesessions_repo.get_session_products, False),
    "get_session_recap": (livesessions_repo.get_session_recap, False),
    "get_trending_bits": (bits_repo.get_trending_bits, False),
    "search_by_hashtag": (bits_repo.search_by_hashtag, False),
    "get_tagged_products": (bits_repo.get_tagged_products, False),
    "get_cart": (carts_repo.get_cart, True),
    "check_coupon_validity": (coupons_repo.check_coupon_validity, True),
    "get_unread_notifications": (account_repo.get_unread_notifications, True),
    "get_saved_items": (account_repo.get_saved_items, True),
    "get_default_address": (account_repo.get_default_address, True),
    # Someone ELSE'S follower list is not reachable: user_id is injected
    # from the JWT and `kind` is the only thing the model supplies.
    "get_followers_or_following": (account_repo.get_followers_following, True),
    "get_product_reviews": (reviews_repo.get_product_reviews, False),
    "get_seller_trust_info": (reviews_repo.get_seller_trust_info, False),
}
