"""
WHAT:
    Mechanical checks that an answer only claims things a tool actually
    returned. No LLM, no network - given a reply and the message history
    it came from, it reports what could not be traced back to a lookup.

WHY THIS IS THE CENTRAL CHECK FOR THIS PROJECT:
    The claim the whole system rests on is that it never invents order
    data, prices or products. That claim is currently defended by prompt
    sentences and one tool-failure instruction, and verified by nobody.
    A measured run already showed the failure it is meant to catch: a
    refused semantic search, and an answer naming a "knit sweater",
    "fleece hoodie" and "wool cardigan" that do not exist.

WHY IT LIVES IN app/agent/ AND NOT app/evals/:
    It started under app/evals/, because the eval runner was its first
    caller. It moved the moment the orchestrator needed it for response
    attribution: production importing from an eval package is the wrong
    dependency direction, and the eval suite depending on the agent is
    the right one. Same code, both callers, arrow pointing the correct
    way.

TWO READINGS OF THE SAME PATTERNS:
    check() asks "is there anything here nobody vouched for".
    attribute() asks "who vouched for this".
    One reports gaps, the other sources, and they must agree - so they
    share ORDER_ID and MONEY rather than growing a second pair that can
    drift apart.

WHAT IT CAN AND CANNOT SEE - the honest boundary:
    An order id is never computed: "ORD0000000000000001" either came
    from a lookup or was invented, so an unmatched one is a HARD
    failure.

    A money amount is different. Most are quoted straight from a tool
    result, but some are legitimately derived - a cart subtotal, a
    discount, "about 2,000 cheaper". So an unmatched amount is reported
    as a SOFT finding for a human to look at, never as a proven
    hallucination.

    A product NAME cannot be extracted from prose at all without named
    entity recognition. That gap is real, and it is exactly the part the
    eval suite hands to an LLM judge instead of pretending to solve here.
"""

import json
import re
from dataclasses import dataclass, field

# "ORD" then digits. Distinctive enough that a match is never a
# coincidence, and never something a model would arrive at by arithmetic.
ORDER_ID = re.compile(r"\bORD\d{6,}\b")

# THE FORM THE PRODUCT ACTUALLY EMITS. The prompt tells the model to
# list orders by "last 4 of the ID" and never to show internal ids, so a
# real reply says "order ending 1378" and the pattern above fires almost
# never - measured, a five-order answer produced zero attributable ids.
# Attribution that cannot see the product's own vocabulary is decoration.
ORDER_SUFFIX = re.compile(r"\bending\s+(\d{4})\b", re.IGNORECASE)

# Money as the assistant is told to write it - the prompt says prices in
# rupees, and measured replies use "₹1,199", "₹1,25,500" and "Rs. 800".
#
# \b BEFORE Rs IS LOad-BEARING, and its absence was a real false
# positive: case-insensitively, "rs" matches inside "hours", so a reply
# saying "cancel within 24 hours 2 days after..." was reported as an
# ungrounded claim of ₹2. Soft findings are only worth reading if they
# are mostly true, so the boundary matters more than it looks.
# The digits are \d+(?:,\d+)* and NOT [\d][\d,]*, which would swallow a
# trailing comma: "₹111, order ₹222" gave the span "₹111," - underlining
# the punctuation, and reporting the amount as "111,". Digit groups must
# END in a digit.
MONEY = re.compile(r"(?:₹|\bRs\.?\s?)\s*(\d+(?:,\d+)*)", re.IGNORECASE)

# First-person promises of things this assistant cannot do. The prompt
# forbids OFFERING an action, and these are the shapes a slip takes.
ACTION_PROMISES = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bi(?:'ve| have| will|'ll) (?:now )?cancel",
        r"\bi(?:'ve| have| will|'ll) (?:now )?place[d]?\b",
        r"\bi(?:'ve| have| will|'ll) (?:now )?add(?:ed)?\b.{0,20}\bcart\b",
        r"\bi(?:'ve| have| will|'ll) (?:now )?appl(?:y|ied)\b",
        r"\bi(?:'ve| have| will|'ll) (?:now )?sen[dt]\b",
        r"\bi(?:'ll| will) (?:notify|let you know|keep you (?:posted|updated))\b",
        r"\bi(?:'ve| have) (?:now )?updated\b",
    )
]


@dataclass
class Grounding:
    """What could not be traced back to a tool result."""

    ungrounded_order_ids: list[str] = field(default_factory=list)
    unmatched_amounts: list[str] = field(default_factory=list)
    unmatched_order_suffixes: list[str] = field(default_factory=list)
    action_promises: list[str] = field(default_factory=list)
    checked_order_ids: int = 0
    checked_amounts: int = 0
    tool_results_seen: int = 0

    @property
    def hard_failures(self) -> list[str]:
        """Findings that are wrong on their own terms, with no innocent
        reading: an invented order id, or a promise to act."""
        out = [f"invented order id {o}" for o in self.ungrounded_order_ids]
        out += [f"promised an action: {p!r}" for p in self.action_promises]
        return out

    @property
    def ok(self) -> bool:
        return not self.hard_failures


def tool_payloads(messages: list[dict]) -> list:
    """Every tool result in the history, parsed back from JSON.

    A result that will not parse is skipped rather than raised on: the
    checks below are a report, and a malformed payload is the
    orchestrator's problem, not a reason to lose the whole run.
    """
    out = []
    for message in messages or []:
        if message.get("role") != "tool":
            continue
        try:
            out.append(json.loads(message.get("content") or "null"))
        except (TypeError, ValueError):
            continue
    return out


def _walk(node, strings: set, numbers: set) -> None:
    if isinstance(node, dict):
        for value in node.values():
            _walk(value, strings, numbers)
    elif isinstance(node, list):
        for value in node:
            _walk(value, strings, numbers)
    elif isinstance(node, str):
        strings.add(node)
        # An order id inside a longer string still counts as returned.
        strings.update(ORDER_ID.findall(node))
    elif isinstance(node, bool):
        # bool is an int subclass - excluded so True does not "ground"
        # a claim of 1.
        return
    elif isinstance(node, (int, float)):
        numbers.add(float(node))


def observed(messages: list[dict]) -> tuple[set, set]:
    """Every string and every number any tool actually returned."""
    strings: set = set()
    numbers: set = set()
    for payload in tool_payloads(messages):
        _walk(payload, strings, numbers)
    return strings, numbers


# Keys under which a trimmed tool result carries a PRODUCT's name.
# "name" covers the product trimmers, order items and cart lines;
# "productName" is the bargain results' spelling. Usernames, business
# names, session titles and Bit titles are deliberately absent - they are
# not products, and letting them into the allowed set would license the
# model to name one as if it were.
PRODUCT_NAME_KEYS = ("name", "productName")


def product_names(messages: list[dict]) -> set:
    """Every product name any tool actually returned.

    THE HALF OF GROUNDING THAT CANNOT BE DONE WITH A REGEX. A price or an
    order id has a shape; a product name is arbitrary text, so it cannot
    be located in prose without named entity recognition. What CAN be
    done mechanically is the other direction - collecting the names that
    are legitimately available - and that set is what an LLM judge is
    then asked to compare a reply against.

    Constraining the judge to a known list is the point. "Did this
    hallucinate?" is a vague question a model answers unreliably; "does
    this name anything outside these 8 strings?" is a comparison.
    """
    found: set = set()

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in PRODUCT_NAME_KEYS and isinstance(value, str) and value.strip():
                    found.add(value.strip())
                else:
                    walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    for payload in tool_payloads(messages):
        walk(payload)
    return found


def _amount(raw: str) -> float | None:
    try:
        return float(raw.replace(",", ""))
    except ValueError:
        return None


def tool_names(messages: list[dict]) -> dict:
    """tool_call_id -> tool name.

    A tool RESULT message carries only the id it answers; the NAME lives
    in the assistant turn that asked for it. Attribution needs the name,
    so the two are joined here rather than changing what the orchestrator
    appends to the history - that shape is what the providers see, and it
    is not the place to carry something only the UI wants.
    """
    names = {}
    for message in messages or []:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            call_id = call.get("id")
            name = (call.get("function") or {}).get("name")
            if call_id and name:
                names[call_id] = name
    return names


def _evidence_by_tool(messages: list[dict]) -> list:
    """(tool_name, strings, numbers) for each tool result, in call order."""
    names = tool_names(messages)
    out = []
    for message in messages or []:
        if message.get("role") != "tool":
            continue
        try:
            payload = json.loads(message.get("content") or "null")
        except (TypeError, ValueError):
            continue
        strings: set = set()
        numbers: set = set()
        _walk(payload, strings, numbers)
        out.append((names.get(message.get("tool_call_id")), strings, numbers))
    return out


def attribute(reply: str, messages: list[dict]) -> list[dict]:
    """Every claim in the reply that can be traced back to the tool that
    returned it, as character spans a client can mark up.

    THE SAME EXTRACTORS AS check(), USED FORWARDS. Grounding asks "is
    there anything here nobody vouched for"; attribution asks "who
    vouched for this" - one reports the gaps, the other the sources, and
    they must agree, so they read the reply with one pair of patterns
    rather than two that can drift.

    Only prices and order ids. A product NAME cannot be located in prose
    without named entity recognition, so the honest thing is to underline
    what we can actually stand behind and leave the rest alone rather
    than guess at spans and mark the wrong words.

    Spans are character offsets into `reply`, so the client marks them by
    slicing rather than by searching for the text again - a reply that
    says "₹599" twice would otherwise have both marked from one match.
    """
    if not reply:
        return []

    evidence = _evidence_by_tool(messages)
    claims = []

    def source_of(kind: str, raw: str):
        for tool, strings, numbers in evidence:
            if kind == "order_id" and raw in strings:
                return tool
            if kind == "price":
                value = _amount(raw)
                if value is not None and value in numbers:
                    return tool
        return None

    # THE SHORTENED ID IS FOUND BY VALUE, NOT BY PHRASING. Two
    # consecutive live runs wrote "order ending 1378" and "order …1378"
    # for the same data, so any pattern over the prose is a guess about
    # wording that changes between calls. The set of REAL suffixes is
    # known exactly - it is in the tool results - so the digits are
    # searched for directly and the words around them are irrelevant.
    for tool, strings, _ in evidence:
        for value in strings:
            if not ORDER_ID.fullmatch(value):
                continue
            suffix = value[-4:]
            # (?<!\d) / (?!\d) so "1378" does not match inside "21378"
            # or a year like "2026" sitting next to other digits.
            for match in re.finditer(rf"(?<!\d){re.escape(suffix)}(?!\d)", reply):
                claims.append({
                    "start": match.start(), "end": match.end(),
                    "text": match.group(0), "kind": "order_suffix", "tool": tool,
                })

    for pattern, kind in ((ORDER_ID, "order_id"), (MONEY, "price")):
        for match in pattern.finditer(reply):
            # MONEY captures the digits; the span must cover the symbol
            # too, or the mark starts after the rupee sign.
            raw = match.group(0) if kind == "order_id" else match.group(1)
            tool = source_of(kind, raw)
            if tool is None:
                continue
            claims.append({
                "start": match.start(),
                "end": match.end(),
                "text": match.group(0),
                "kind": kind,
                "tool": tool,
            })

    # De-duplicated: a full id and its own last four overlap, and two
    # orders can legitimately share a suffix. First span wins.
    claims.sort(key=lambda c: (c["start"], -(c["end"] - c["start"])))
    deduped, last_end = [], -1
    for claim in claims:
        if claim["start"] >= last_end:
            deduped.append(claim)
            last_end = claim["end"]
    return deduped


def check(reply: str, messages: list[dict]) -> Grounding:
    """Trace every checkable claim in `reply` back to a tool result."""
    report = Grounding()
    if not reply:
        return report

    strings, numbers = observed(messages)
    report.tool_results_seen = len(tool_payloads(messages))

    for order_id in set(ORDER_ID.findall(reply)):
        report.checked_order_ids += 1
        if order_id not in strings:
            report.ungrounded_order_ids.append(order_id)

    for raw in set(MONEY.findall(reply)):
        value = _amount(raw)
        if value is None:
            continue
        report.checked_amounts += 1
        if value not in numbers:
            report.unmatched_amounts.append(raw)

    # AN UNMATCHED SUFFIX IS REPORTED, NOT FAILED, and the restraint is
    # deliberate. "ending 1378" naming no returned order is very probably
    # invented - but four digits is weaker evidence than a whole id, and
    # promoting this to a hard failure would silently re-score every case
    # in the eval baseline. Making it fail should be its own change, with
    # its own measurement, not a side effect of teaching attribution to
    # read the product's vocabulary.
    for raw in set(ORDER_SUFFIX.findall(reply)):
        if not any(
            s.endswith(raw) for s in strings if ORDER_ID.fullmatch(s)
        ):
            report.unmatched_order_suffixes.append(raw)

    for pattern in ACTION_PROMISES:
        match = pattern.search(reply)
        if match:
            report.action_promises.append(match.group(0))

    return report
