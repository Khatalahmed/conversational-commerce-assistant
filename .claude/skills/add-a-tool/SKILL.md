---
name: add-a-tool
description: Add a new tool the marketplace assistant can call, or audit an existing one. Use when adding a capability the model should be able to invoke, when a tool exists in the registry but the assistant never picks it, or when a tool returns data the model cannot use. Covers the seven files one tool touches and the failure each missed step causes.
---

# Adding a tool to the marketplace assistant

One tool touches **seven** places. Miss step 2 and you leak database
fields. Miss step 3 and you blow the token budget. Miss step 6 and the
UI goes silent mid-answer. None of them fail loudly.

Work in this order — each step depends on the one before.

## 1. The repo function — `app/repos/<x>_repo.py`

Read the collection through the allowlist projection and sanitize on the
way out. Both layers, every time:

```python
projection = get_projection(COLLECTION)
docs = await db[COLLECTION].find(query, projection).limit(limit).to_list(limit)
return [sanitize_document(COLLECTION, d) for d in docs]
```

**Anything user-scoped takes `user_id` as its first parameter** and
queries on it. Never trust an id the model supplied — `tool_executor`
injects the verified one. An unparseable id must return empty, never
fall through to an unscoped query that hands one user everyone's data.

If the tool returns a capped list, add a `count_*` function beside it.
See step 4.

## 2. The allowlist — `app/security/field_allowlist.py`

Check every field you read is listed for that collection. Add only what
the answer needs, and say in a comment why anything sensitive is
excluded. **A field not listed is invisible by default** — that's the
design, so a new database column can't leak on its own.

## 3. The trimmer — `app/agent/tool_executor.py`

Raw documents are far larger than the model needs; one measured
`search_products` result was **51,000 tokens**. Add a function that keeps
only what an answer references, and register it in `_TRIMMERS`.

Trimmers do two jobs:

- **Shrink.** Drop image URLs, full variant trees, internal ids.
- **Make usable.** If the result carries raw ObjectIds, resolve them to
  names in ONE batched query (`get_products_by_ids`), not one per row.

**Compute anything the model cannot.** It does not know today's date, so
a bare `expiresAt` gets printed as `2026-07-09T09:15:32` into a phone
chat bubble. Hand it `lapsedDaysAgo` instead. Same lesson as `daysLate`
on orders — both were found in real answers, not in review.

## 4. The summariser — same file, `_SUMMARISERS`

Only if the tool returns a capped list. A window with no total **reads as
the total**: measured on a real account, 67 orders were answered as "you
have 8 orders". Return `showing`, the true total, and a `note` sentence —
the model reads the sentence more reliably than the two numbers.

## 5. The schema and registry — `app/agent/tools.py`

Descriptions are re-sent on **every round**, so carry only what
distinguishes this tool from its neighbours. Sequencing rules ("call X
first") belong in the system prompt, not here — otherwise you pay for
them per tool.

```python
"list_bargains": (bargains_repo.list_bargains, True),  # True = needs user_id
```

**If the tool needs no arguments, say so explicitly.** Twice measured:
`get_recommendations` and `list_bargains` were both ignored in favour of
asking the user questions, because their neighbours all require ids and
the model assumed they did too.

## 6. The progress label — `app/agent/orchestrator.py`

Add to `_TOOL_STATUS`. A gerund in the user's words — "Looking up your
offers", not the tool name. Without it the demo shows a generic
"Looking that up" for several seconds.

## 7. The source label — `app/api/routes/demo_ui.py`

Add to `SOURCES`. A noun phrase — "your offers" — used by the "✓ checked"
footnote and by response attribution when a claim traces to this tool.

## Then

- **Tests.** Repo tests in `test_<x>_repo.py`, including the wrong-user
  case. Trimmer tests in `test_tool_result_trimming.py`. The registry
  tests in `test_tool_registry.py` pick up the new tool automatically.
- **Tool count.** Several docstrings state it. `grep -rn "35 tool" app/`
- **An eval case** in `app/evals/cases.py`, naming the tools that would
  count as looking in the right place. A case the system *cannot*
  satisfy teaches only distrust — retire it rather than leave it red.
- **Run the suite** against the previous baseline. A new tool enlarges
  the selection space, which is a real regression risk for every other
  case:

```bash
python scripts/run_evals.py --label after-x --against evals/<previous>.json
```

Adding a tool changes the cached prefix, so the next run pays one cold
prompt before the ~94% cache rate resumes.
