"""
WHAT:
    A single-page chat UI at /demo, served by this same app.

WHY IT EXISTS:
    /docs is a developer tool. A client watching JSON go into a text box
    and JSON come back sees plumbing, not a product - and spends the
    demo translating rather than reacting. The same assistant behind a
    chat bubble reads as the thing they would ship.

    Deliberately one file with no build step, no framework and no
    external requests: it is served from the app's own origin, so there
    is no CORS to configure and nothing to deploy separately. That rule
    is why there are no web fonts and no icon library here - a demo that
    breaks because a CDN is slow is worse than one that uses system
    fonts.

THREE THINGS IT DOES THAT ARE NOT DECORATION:

    A PHONE FRAME ON DESKTOP. This assistant ships inside the marketplace
    mobile app, and the answers are formatted for a chat bubble on a
    phone - the system prompt says so explicitly, and forbids markdown
    tables and headings for that reason. Shown full-width in a browser
    the output looks oddly terse; shown in a phone-shaped column it
    looks like the product. On an actual phone the frame disappears.

    STARTER QUESTIONS. The demo dataset is small - around a dozen
    products - so an empty text box invites a visitor to invent a query
    that legitimately finds nothing, and "I couldn't find that" reads as
    weakness even when it is the correct answer. The chips ask things
    that work against ANY dataset, because they lean on the signed-in
    user's own records and on general listings rather than naming
    products that may not exist.

    A "CHECKED" LINE UNDER EACH ANSWER. The status events already say
    which lookups ran; keeping them visible afterwards turns the central
    claim of this project - that it reads real data rather than inventing
    it - into something a client can see rather than something they have
    to be told.

WHAT IT DOES NOT DO:
    No credential storage beyond the browser tab, no history persistence
    of its own (the server already owns that, keyed by session id), and
    no styling framework. It is a demo surface, not a product front end.

    Message text is always written with textContent, never innerHTML.
    Model output is not trusted markup, and a product description is
    attacker-controllable - see the prompt-injection note in
    orchestrator.build_system_prompt.
"""

import asyncio
import re
import secrets
from datetime import datetime, timezone

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.config.settings import get_settings
from app.db.connection import get_database
from app.security.auth import create_test_token

router = APIRouter()

USERS_COLLECTION = "users"
# The picker shows the busiest accounts first. More than this is a scroll
# nobody reads - and the real database has 16 buyers in total anyway.
MAX_ACCOUNTS = 25
# The collapsed "never ordered" list. The real database has 270 of them,
# so this is a sample with a count beside it rather than the whole set:
# rendering 270 rows into a phone-sized column helps nobody.
MAX_INACTIVE = 60
# Long enough to outlast a demo without anyone re-minting mid-question.
TOKEN_MINUTES = 240

PAGE = r"""
<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="color-scheme" content="dark">
<title>the marketplace Assistant</title>
<style>
  :root {
    --bg:#080a0f;
    --frame:#0e1118;
    --surface:#141824;
    --surface-2:#1a1f2d;
    --line:#232a3a;
    --line-soft:#1b2130;
    --text:#e9edf6;
    --muted:#8b94ab;
    --faint:#67708a;
    --accent:#00d99a;
    --accent-dim:#00a878;
    --blue:#3b82f6;
    --danger:#ff6b7a;
  }
  * { box-sizing:border-box; -webkit-tap-highlight-color:transparent; }
  /* THE hidden ATTRIBUTE ONLY SETS display:none AT THE LOWEST
     SPECIFICITY, so any rule giving an element a display of its own
     silently beats it. Restated with !important, which is what the
     attribute is supposed to mean. */
  [hidden] { display:none !important; }

  body {
    margin:0; background:var(--bg); color:var(--text);
    font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",sans-serif;
    -webkit-font-smoothing:antialiased;
    display:flex; align-items:center; justify-content:center;
    min-height:100dvh; padding:24px;
  }
  body::before {
    content:""; position:fixed; inset:0; pointer-events:none;
    background:
      radial-gradient(60ch 40ch at 20% 0%, rgba(0,217,154,.10), transparent 60%),
      radial-gradient(50ch 40ch at 90% 100%, rgba(59,130,246,.10), transparent 60%);
  }

  /* ---- the phone ---- */
  .device {
    position:relative; z-index:1;
    width:100%; max-width:430px; height:min(880px, calc(100dvh - 48px));
    background:var(--frame); border:1px solid var(--line); border-radius:38px;
    box-shadow:
      0 0 0 1px rgba(255,255,255,.03) inset,
      0 40px 80px -20px rgba(0,0,0,.8),
      0 0 60px -20px rgba(0,217,154,.15);
    display:flex; flex-direction:column; overflow:hidden;
  }

  /* ---- header ---- */
  .bar {
    display:flex; align-items:center; gap:10px;
    padding:16px 16px 13px;
    border-bottom:1px solid var(--line-soft);
    background:linear-gradient(180deg, rgba(255,255,255,.025), transparent);
    flex:none;
  }
  .back {
    background:transparent; border:0; color:var(--muted); cursor:pointer;
    font-size:19px; line-height:1; padding:5px 7px; border-radius:9px;
    transition:.15s; flex:none;
  }
  .back:hover { color:var(--text); background:var(--surface); }
  .mark {
    width:32px; height:32px; border-radius:10px; flex:none;
    background:linear-gradient(135deg, var(--accent), var(--blue));
    display:grid; place-items:center;
    font-weight:800; font-size:15px; color:#06120e; letter-spacing:-.5px;
  }
  .who { display:flex; flex-direction:column; gap:1px; min-width:0; }
  .who b { font-size:14.5px; font-weight:650; letter-spacing:-.2px;
           white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .who span { font-size:11px; color:var(--muted); display:flex; align-items:center;
              gap:5px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .live { width:6px; height:6px; border-radius:50%; background:var(--muted); flex:none; }
  .live.on { background:var(--accent); box-shadow:0 0 0 3px rgba(0,217,154,.15); }
  .bar-actions { margin-left:auto; display:flex; gap:6px; flex:none; }
  .icon-btn {
    background:transparent; border:1px solid var(--line); color:var(--muted);
    border-radius:9px; padding:6px 10px; font-size:11.5px; cursor:pointer;
    transition:.15s; white-space:nowrap;
  }
  .icon-btn:hover { color:var(--text); border-color:#33405a; background:var(--surface); }

  /* ---- screens ---- */
  .screen { flex:1; min-height:0; display:flex; flex-direction:column; }
  .scroll {
    flex:1; min-height:0; overflow-y:auto; overscroll-behavior:contain;
    scrollbar-width:thin; scrollbar-color:#2a3348 transparent;
  }
  .scroll::-webkit-scrollbar { width:7px; }
  .scroll::-webkit-scrollbar-thumb { background:#2a3348; border-radius:4px; }
  .pad { padding:20px 18px; display:flex; flex-direction:column; gap:14px; }

  h2.title { margin:0; font-size:18px; font-weight:650; letter-spacing:-.3px; }
  p.lede { margin:0; color:var(--muted); font-size:12.5px; line-height:1.6; }
  h3.sect {
    margin:0; font-size:10.5px; font-weight:600; letter-spacing:.6px;
    text-transform:uppercase; color:var(--faint);
  }

  /* ---- data scale panel ---- */
  .tiles { display:grid; grid-template-columns:repeat(4, 1fr); gap:6px; }
  .tile {
    background:var(--surface); border:1px solid var(--line-soft);
    border-radius:10px; padding:8px 6px; text-align:center;
  }
  .tile b { display:block; font-size:16px; font-weight:680; letter-spacing:-.4px;
            font-variant-numeric:tabular-nums; line-height:1.15; }
  .tile span { display:block; font-size:9px; color:var(--muted); margin-top:2px; }

  /* Part-to-whole. EMPHASIS, not categorical: the accent carries buyers,
     the rest is de-emphasis gray. Both segments direct-labelled with a
     2px gap - the pair sits in the CVD 6-8 band, legal only with
     secondary encoding. Colours validated against this surface. */
  .split-bar { display:flex; height:9px; border-radius:5px; overflow:hidden; gap:2px; }
  .split-bar i { display:block; height:100%; }
  .split-bar .buyers { background:#00a878; border-radius:5px 0 0 5px; }
  .split-bar .rest   { background:#8b94ab; border-radius:0 5px 5px 0; flex:1; }
  .split-key { display:flex; gap:14px; font-size:11px; color:var(--muted); }
  .split-key span { display:flex; align-items:center; gap:5px; }
  .split-key i { width:8px; height:8px; border-radius:2px; flex:none; }
  .split-key b { color:var(--text); font-weight:600; font-variant-numeric:tabular-nums; }

  /* ---- rows (accounts and threads share this) ---- */
  .rows { display:flex; flex-direction:column; gap:7px; }
  .row {
    display:flex; align-items:center; gap:11px; text-align:left; width:100%;
    background:var(--surface); border:1px solid var(--line); color:var(--text);
    border-radius:13px; padding:11px 13px; cursor:pointer; transition:.15s;
    font:inherit;
  }
  .row:hover:not(:disabled) {
    border-color:var(--accent-dim); background:var(--surface-2);
    transform:translateY(-1px);
  }
  .row:disabled { opacity:.5; cursor:default; }
  .row .av {
    width:34px; height:34px; border-radius:11px; flex:none;
    background:linear-gradient(135deg, var(--accent), var(--blue));
    display:grid; place-items:center; color:#06120e;
    font-weight:750; font-size:14px; text-transform:uppercase;
  }
  .row .av.quiet { background:var(--surface-2); color:var(--muted);
                   border:1px solid var(--line); font-size:15px; }
  .row .nm { display:flex; flex-direction:column; gap:2px; min-width:0; flex:1; }
  .row .nm b { font-size:13.5px; font-weight:600; letter-spacing:-.1px;
               white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .row .nm span { font-size:11px; color:var(--muted);
                  white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .row .go { color:var(--faint); font-size:15px; flex:none; }
  .row .kill {
    background:transparent; border:0; color:var(--faint); cursor:pointer;
    font-size:15px; padding:4px 6px; border-radius:7px; flex:none; line-height:1;
  }
  .row .kill:hover { color:var(--danger); background:rgba(255,107,122,.1); }
  .row.thin { padding:8px 11px; border-radius:11px; }
  .row.thin .av { width:26px; height:26px; border-radius:9px; font-size:12px; }
  .row.thin .nm b { font-size:13px; font-weight:550; }

  /* ---- collapsed "never ordered" ---- */
  .more summary {
    cursor:pointer; color:var(--muted); font-size:12px; padding:6px 2px;
    list-style:none; display:flex; align-items:center; gap:6px;
  }
  .more summary::-webkit-details-marker { display:none; }
  .more summary::before { content:"\25B8"; font-size:10px; transition:transform .15s; }
  .more[open] summary::before { transform:rotate(90deg); }
  .more summary:hover { color:var(--text); }
  .more-note { margin:2px 0 8px 16px; color:var(--faint); font-size:11px; line-height:1.5; }
  .capped { max-height:210px; overflow-y:auto; padding-right:4px;
            scrollbar-width:thin; scrollbar-color:#2a3348 transparent; }
  .capped::-webkit-scrollbar { width:6px; }
  .capped::-webkit-scrollbar-thumb { background:#2a3348; border-radius:3px; }

  /* ---- token fallback ---- */
  .field { display:flex; flex-direction:column; gap:7px; }
  .field label { font-size:11px; color:var(--muted); font-weight:550; letter-spacing:.2px; }
  .field input {
    background:#0a0d13; color:var(--text); border:1px solid var(--line);
    border-radius:11px; padding:12px 13px; width:100%;
    font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12px;
    outline:none; transition:.15s;
  }
  .field input:focus { border-color:var(--accent-dim); box-shadow:0 0 0 3px rgba(0,217,154,.1); }
  .primary {
    background:linear-gradient(135deg, var(--accent), var(--accent-dim));
    color:#04120d; border:0; border-radius:11px; padding:12px 16px;
    font-size:14px; font-weight:650; cursor:pointer; transition:.15s;
  }
  .primary:hover { filter:brightness(1.08); }
  .primary:disabled { opacity:.45; cursor:default; filter:none; }
  .alt {
    background:none; border:0; color:var(--muted); font-size:11.5px; cursor:pointer;
    padding:2px; text-decoration:underline; align-self:flex-start;
  }
  .alt:hover { color:var(--text); }
  .err-line { color:var(--danger); font-size:12px; min-height:15px; }

  /* ---- conversation ---- */
  #log { display:flex; flex-direction:column; gap:9px; padding:18px 16px 8px; }
  .msg {
    max-width:85%; padding:11px 14px; border-radius:18px;
    white-space:pre-wrap; overflow-wrap:anywhere; font-size:14.5px; line-height:1.55;
    animation:rise .22s ease-out;
  }
  @keyframes rise { from { opacity:0; transform:translateY(6px); } to { opacity:1; transform:none; } }
  .user {
    align-self:flex-end; color:#fff; border-bottom-right-radius:6px;
    background:linear-gradient(135deg, #4c8dff, #2563eb);
    box-shadow:0 4px 14px -6px rgba(37,99,235,.7);
  }
  .bot { align-self:flex-start; background:var(--surface); border:1px solid var(--line);
         border-bottom-left-radius:6px; }
  .err { align-self:flex-start; background:rgba(255,107,122,.09);
         border:1px solid rgba(255,107,122,.3); color:#ffc4cb; font-size:13.5px; }

  .welcome { align-self:stretch; padding:8px 4px 2px; animation:rise .3s ease-out; }
  .welcome h3 { margin:0 0 6px; font-size:15px; font-weight:650; letter-spacing:-.2px; }
  .welcome p { margin:0 0 13px; color:var(--muted); font-size:12.5px; line-height:1.6; }
  .chips { display:flex; flex-wrap:wrap; gap:7px; }
  .chip {
    background:var(--surface); border:1px solid var(--line); color:var(--text);
    border-radius:999px; padding:8px 13px; font-size:12.5px; cursor:pointer;
    transition:.15s; text-align:left;
  }
  .chip:hover { border-color:var(--accent-dim); background:var(--surface-2); transform:translateY(-1px); }
  .chip.cat { border-color:#2a3a4f; }

  .typing { display:flex; gap:4px; align-items:center; padding:13px 15px; }
  .typing i { width:6px; height:6px; border-radius:50%; background:var(--muted);
              display:block; animation:blink 1.3s infinite; }
  .typing i:nth-child(2){ animation-delay:.18s }
  .typing i:nth-child(3){ animation-delay:.36s }
  @keyframes blink { 0%,65%,100%{opacity:.22; transform:scale(.85)} 32%{opacity:1; transform:scale(1)} }

  .status { align-self:flex-start; display:flex; align-items:center; gap:8px;
            color:var(--muted); font-size:12.5px; padding:3px 6px; animation:rise .2s ease-out; }
  .pulse { width:7px; height:7px; border-radius:50%; background:var(--accent);
           animation:pulse 1.2s infinite; flex:none; }
  @keyframes pulse { 0%,100%{opacity:.25; transform:scale(.8)} 50%{opacity:1; transform:scale(1)} }
  .writing::after { content:""; display:inline-block; width:2px; height:15px;
                    margin-left:3px; background:var(--accent); vertical-align:-3px;
                    animation:pulse .9s infinite; }

  .meta { align-self:flex-start; color:var(--faint); font-size:11px;
          padding:0 6px 2px; display:flex; flex-wrap:wrap; gap:4px 8px; align-items:center; }
  .meta .checked { color:var(--accent-dim); }
  .meta .copy {
    background:none; border:0; color:var(--faint); cursor:pointer; font-size:11px;
    padding:1px 4px; border-radius:5px;
  }
  .meta .copy:hover { color:var(--text); background:var(--surface); }

  /* Product rows. Tapping expands in place rather than navigating:
     the storefront has no product pages, so a link would go nowhere. In the
     app these open the product screen from product_id. */
  .cards { align-self:flex-start; display:flex; flex-direction:column; gap:6px;
           max-width:85%; padding:2px 0; }
  .card { text-align:left; font:inherit; font-size:13.5px; color:var(--text);
          background:var(--surface); border:1px solid var(--line);
          border-radius:12px; padding:9px 12px; cursor:pointer;
          animation:rise .22s ease-out; }
  .card:hover { border-color:var(--accent-dim); }
  .card .row { display:flex; gap:10px; align-items:center; }
  .card .thumb { width:46px; height:46px; flex:none; border-radius:8px;
                 object-fit:cover; background:var(--surface-2);
                 border:1px solid var(--line-soft); }
  .card .body { min-width:0; }
  .card .nm { font-weight:600; overflow-wrap:anywhere; }
  .card .pr { display:flex; align-items:baseline; gap:7px; margin-top:3px;
              color:var(--muted); font-size:12.5px; }
  .card .pr b { color:var(--text); font-size:13.5px; }
  .card .pr s { color:var(--faint); }
  .card .pr .off { color:var(--accent); font-size:11.5px; }
  .card .detail { display:none; margin-top:6px; padding-top:6px;
                  border-top:1px solid var(--line); color:var(--faint);
                  font-size:11.5px; }
  .card.open .detail { display:block; }

  /* Attributed claims. A dotted underline rather than a highlight: every
     number in a good answer is attributed, and highlighting all of them
     turns the bubble into a colouring book. <mark> defaults to a yellow
     background, so it is cleared explicitly. */
  .msg mark.src { background:transparent; color:inherit; cursor:pointer;
                  border-bottom:1px dashed var(--accent-dim); }
  .msg mark.src:hover { border-bottom-color:var(--accent);
                        background:rgba(0,217,154,.08); }
  .srcnote { align-self:flex-start; color:var(--faint); font-size:11.5px;
             padding:1px 6px 2px; animation:rise .2s ease-out; }
  .srcnote b { color:var(--accent-dim); font-weight:600; }

  .jump {
    position:absolute; right:16px; bottom:78px; z-index:3;
    background:var(--surface-2); border:1px solid var(--line); color:var(--text);
    border-radius:999px; padding:6px 12px; font-size:11.5px; cursor:pointer;
    box-shadow:0 6px 18px -8px rgba(0,0,0,.9);
  }

  /* ---- composer ---- */
  .composer {
    display:flex; gap:9px; padding:12px 14px 16px; flex:none;
    border-top:1px solid var(--line-soft);
    background:linear-gradient(0deg, rgba(255,255,255,.02), transparent);
  }
  .composer input {
    flex:1; min-width:0; background:var(--surface); color:var(--text);
    border:1px solid var(--line); border-radius:13px; padding:13px 15px;
    font-size:14.5px; outline:none; transition:.15s;
  }
  .composer input:focus { border-color:#33405a; background:var(--surface-2); }
  .send {
    width:44px; flex:none; border:0; border-radius:13px; cursor:pointer;
    background:linear-gradient(135deg, var(--accent), var(--accent-dim));
    color:#04120d; font-size:17px; font-weight:700; transition:.15s;
  }
  .send:hover:not(:disabled) { filter:brightness(1.08); }
  .send:disabled { opacity:.35; cursor:default; }
  .send.stop { background:var(--surface-2); border:1px solid var(--line); color:var(--danger); }

  .empty { text-align:center; color:var(--faint); font-size:12.5px; padding:26px 10px; line-height:1.7; }

  @media (max-width:520px) {
    body { padding:0; align-items:stretch; }
    .device { max-width:none; height:100dvh; border:0; border-radius:0;
              box-shadow:none; background:var(--bg); }
    .bar { padding-top:max(16px, env(safe-area-inset-top)); }
    .composer { padding-bottom:max(16px, env(safe-area-inset-bottom)); }
  }
  @media (prefers-reduced-motion:reduce) { * { animation:none !important; transition:none !important; } }
</style>

<div class="device">
  <div class="bar">
    <button class="back" id="backBtn" onclick="goBack()" hidden aria-label="Back">&#8249;</button>
    <div class="mark" id="mark">Z</div>
    <div class="who">
      <b id="barTitle">the marketplace Assistant</b>
      <span><i class="live" id="live"></i><span id="barSub">not connected</span></span>
    </div>
    <div class="bar-actions">
      <button class="icon-btn" id="newBtn" onclick="newThread()" hidden>New</button>
    </div>
  </div>

  <!-- 1. accounts -->
  <div class="screen" id="screenAccounts">
    <div class="scroll"><div class="pad">
      <h2 class="title">Connect to your account</h2>
      <p class="lede" id="gateHint">Pick an account. A signed access token is generated for
         that exact user behind the scenes &mdash; the assistant then answers only
         from their own data.</p>
      <div class="field" id="codeField" hidden>
        <label for="code">ACCESS CODE</label>
        <input id="code" type="password" placeholder="shared code" autocomplete="off"
               onkeydown="if(event.key==='Enter')submitCode()">
        <button class="primary" style="margin-top:9px" onclick="submitCode()">Unlock</button>
      </div>
      <div id="scope" hidden></div>
      <div class="rows" id="accounts"></div>
      <div class="field" id="tokenField" hidden>
        <label for="token">ACCESS TOKEN</label>
        <input id="token" placeholder="eyJhbGciOiJIUzI1NiIs..." autocomplete="off"
               spellcheck="false" onkeydown="if(event.key==='Enter')startWithToken()">
      </div>
      <div class="err-line" id="gateErr"></div>
      <button class="alt" id="altBtn" onclick="showTokenEntry()">Paste an access token instead</button>
      <button class="primary" id="gateBtn" onclick="startWithToken()" hidden>Start chatting</button>
    </div></div>
  </div>

  <!-- 2. conversations -->
  <div class="screen" id="screenThreads" hidden>
    <div class="scroll"><div class="pad">
      <h2 class="title">Conversations</h2>
      <p class="lede" id="threadsLede"></p>
      <div class="rows" id="threadRows"></div>
      <div class="empty" id="threadEmpty" hidden>
        No conversations yet.<br>Start one and it will be saved here.
      </div>
    </div></div>
    <div class="composer">
      <button class="primary" style="flex:1" onclick="newThread()">New conversation</button>
    </div>
  </div>

  <!-- 3. chat -->
  <div class="screen" id="screenChat" hidden style="position:relative">
    <div class="scroll" id="logScroll"><div id="log"></div></div>
    <button class="jump" id="jumpBtn" onclick="stick(true)" hidden>Jump to latest &darr;</button>
    <div class="composer">
      <input id="q" placeholder="Ask about an order, a product, bargaining..."
             onkeydown="if(event.key==='Enter')send()">
      <button class="send" id="send" onclick="onSendClick()" aria-label="Send">&uarr;</button>
    </div>
  </div>
</div>

<script>
let token = "", account = "", thread = null, busy = false, controller = null;

// THE ACCOUNT HALF. These lean on the signed-in user's own records, so
// they work against any dataset and cannot miss.
const STARTERS = [
  "Where is my order?",
  "What's in my cart?",
  "What's trending right now?",
];
// THE CATALOGUE HALF, filled from /demo/stats at page load with real
// product names. Empty if that call fails - the chips are a
// convenience, not a dependency.
let CATALOGUE_STARTERS = [];

// WHAT THE "CHECKED" LINE CALLS EACH SOURCE. The status events carry a
// gerund ("Looking up your orders"), which is right while it happens and
// wrong afterwards. These are the noun forms, grouped so three catalogue
// queries read as one source.
const SOURCES = {
  get_order_status:"your orders", get_order_history:"your orders",
  get_order_detail:"your orders", get_delivery_estimate:"your orders",
  get_tracking:"your orders", get_invoice:"your orders",
  check_cancellation_eligibility:"your orders",
  search_products:"the catalogue", search_products_by_name:"the catalogue",
  search_products_semantically:"the catalogue", find_similar_products:"the catalogue",
  get_product_detail:"the catalogue", get_variant_stock:"the catalogue",
  get_trending_products:"the catalogue", get_recommendations:"the catalogue",
  check_bargain_eligibility:"bargaining rules", suggest_offer_amount:"bargaining rules",
  list_bargains:"your offers",
  get_bargain_status:"your offers", get_counter_offer:"your offers",
  get_live_now:"live sessions", get_session_products:"live sessions",
  get_session_recap:"live sessions",
  get_trending_bits:"Bits", search_by_hashtag:"Bits", get_tagged_products:"Bits",
  get_product_reviews:"reviews", get_seller_info:"seller profiles",
  get_seller_trust_info:"seller profiles",
  get_cart:"your cart", check_coupon_validity:"coupons",
  get_saved_items:"your saved items", get_unread_notifications:"your notifications",
  get_default_address:"your address", get_followers_or_following:"your followers",
};

const $ = (id) => document.getElementById(id);

/* ---------- access code ----------
   Only asked for when the SERVER says so - a 401 from /demo/accounts.
   Unset server-side and this never appears, so laptop demos are
   unchanged. Held in sessionStorage so a refresh does not re-ask, and
   dropped when the tab closes; it is a shared door code, not a
   credential worth persisting. */
let accessCode = "";
try { accessCode = sessionStorage.getItem("demo.code") || ""; } catch (e) {}

function codeHeaders() {
  return accessCode ? {"X-Demo-Code": accessCode} : {};
}

function askForCode(message) {
  $("codeField").hidden = false;
  $("scope").hidden = true;
  $("accounts").innerHTML = "";
  $("altBtn").hidden = true;
  $("gateHint").textContent = message
    || "This demo is protected. Enter the access code you were given.";
  $("code").focus();
}

async function submitCode() {
  const value = $("code").value.trim();
  if (!value) { $("gateErr").textContent = "Enter the code to continue."; return; }
  accessCode = value;
  try { sessionStorage.setItem("demo.code", value); } catch (e) {}
  $("gateErr").textContent = "";
  $("codeField").hidden = true;
  $("gateHint").textContent = "Checking…";
  await loadScope();
  await loadAccounts();
}

/* ---------- screens ---------- */
// Three screens in one frame, and a back button that always means "up
// one level". A chat app that cannot go back is a dead end - which is
// what this was before threads existed.
let screen = "accounts";

function showScreen(name) {
  screen = name;
  ["Accounts", "Threads", "Chat"].forEach(s => {
    $("screen" + s).hidden = (s.toLowerCase() !== name);
  });
  $("backBtn").hidden = (name === "accounts");
  $("newBtn").hidden  = (name !== "chat");

  if (name === "accounts") {
    $("barTitle").textContent = "the marketplace Assistant";
    $("barSub").textContent = account ? "signed out" : "not connected";
  } else if (name === "threads") {
    $("barTitle").textContent = account;
    $("barSub").textContent = "conversations";
    renderThreads();
  } else {
    $("barTitle").textContent = thread ? thread.title : account;
    $("barSub").textContent = account;
    setTimeout(() => stick(true), 0);
  }
}

function goBack() {
  // Leaving a chat mid-answer must not leave the request running.
  if (screen === "chat") { abort(); showScreen("threads"); }
  else if (screen === "threads") signOut();
}

function signOut() {
  token = ""; account = ""; thread = null;
  $("live").classList.remove("on");
  showScreen("accounts");
}

/* ---------- thread storage ----------
   The SERVER owns what the model remembers, keyed by session id. This
   owns what the screen shows, keyed by the same id - so resuming a
   thread gives the model its context back and the reader their
   transcript, without a new endpoint to fetch history.

   localStorage can throw outright (private windows, blocked site data),
   so every access is wrapped: losing saved threads degrades the demo,
   it must never break it. */
function key() { return "demo.threads." + account; }

function loadThreads() {
  try { return JSON.parse(localStorage.getItem(key())) || []; }
  catch (e) { return []; }
}
function saveThreads(list) {
  try { localStorage.setItem(key(), JSON.stringify(list)); } catch (e) {}
}
function upsertThread() {
  if (!thread) return;
  const list = loadThreads().filter(t => t.id !== thread.id);
  list.unshift(thread);
  saveThreads(list.slice(0, 30));
}

function newThread() {
  thread = {
    // The session id IS the thread id, so the server's memory and this
    // transcript stay in step by construction.
    id: "demo-" + Math.random().toString(36).slice(2, 8),
    title: "New conversation",
    updatedAt: Date.now(),
    messages: [],
  };
  renderChat();
  showScreen("chat");
  $("q").focus();
}

function openThread(id) {
  const found = loadThreads().find(t => t.id === id);
  if (!found) return;
  thread = found;
  renderChat();
  showScreen("chat");
}

function deleteThread(id, ev) {
  ev.stopPropagation();
  saveThreads(loadThreads().filter(t => t.id !== id));
  if (thread && thread.id === id) thread = null;
  renderThreads();
}

function renderThreads() {
  const list = loadThreads();
  const box = $("threadRows");
  box.innerHTML = "";
  $("threadEmpty").hidden = list.length > 0;
  $("threadsLede").textContent = list.length
    ? "Signed in as " + account + ". Pick up where you left off, or start a new one."
    : "Signed in as " + account + ".";

  list.forEach(t => {
    const row = document.createElement("button");
    row.className = "row";
    const av = document.createElement("div");
    av.className = "av quiet";
    av.textContent = "\u{1F4AC}";
    const nm = document.createElement("div");
    nm.className = "nm";
    const b = document.createElement("b");
    b.textContent = t.title;
    const s = document.createElement("span");
    const turns = t.messages.filter(m => m.role === "user").length;
    s.textContent = turns + (turns === 1 ? " question" : " questions") + " · " + ago(t.updatedAt);
    nm.append(b, s);
    const kill = document.createElement("button");
    kill.className = "kill";
    kill.textContent = "×";
    kill.title = "Delete conversation";
    kill.onclick = (e) => deleteThread(t.id, e);
    row.append(av, nm, kill);
    row.onclick = () => openThread(t.id);
    box.appendChild(row);
  });
}

function ago(ts) {
  const s = Math.max(0, Math.round((Date.now() - ts) / 1000));
  if (s < 60) return "just now";
  const m = Math.round(s / 60);
  if (m < 60) return m + "m ago";
  const h = Math.round(m / 60);
  if (h < 24) return h + "h ago";
  return Math.round(h / 24) + "d ago";
}

/* ---------- rendering ---------- */
function renderChat() {
  const log = $("log");
  log.innerHTML = "";
  if (!thread || !thread.messages.length) { welcome(); return; }
  thread.messages.forEach(m => {
    if (m.role === "user") add(m.text, "msg user");
    else if (m.role === "error") add(m.text, "msg err");
    else {
      const bubble = add(m.text, "msg bot");
      if (m.attribution && m.attribution.length) {
        renderAttributed(bubble, m.text, m.attribution);
      }
      if (m.products) productCards(m.products);
      if (m.meta) metaLine(m.meta.timing, m.meta.checked);
    }
  });
  // DEFERRED A FRAME, because every caller renders BEFORE showScreen().
  // While the chat screen is still hidden #logScroll has no height, so
  // assigning scrollTop does nothing at all - and the scroll silently
  // did not happen. Measured: reopening a saved conversation always
  // landed at the top, with the newest answer and its cards below the
  // fold, which is the same symptom the streaming fix above addresses
  // by a different route.
  //
  // Rendering first and scrolling next frame keeps that order (no flash
  // of the previous conversation) while letting the scroll run once the
  // element actually has a height.
  requestAnimationFrame(() => stick(true));
}

function welcome() {
  const w = document.createElement("div");
  w.className = "welcome";
  const h = document.createElement("h3");
  h.textContent = "Ask me about your the marketplace account";
  const p = document.createElement("p");
  p.textContent = "Orders and delivery, the catalogue, bargaining, live sessions and Bits. I read your real data — and I'll say so when I can't find something.";
  const chips = document.createElement("div");
  chips.className = "chips";
  // Catalogue questions first: the strongest thing the assistant does,
  // and the ones nobody could have guessed unaided.
  CATALOGUE_STARTERS.forEach(s => chips.appendChild(chip(s, true)));
  STARTERS.forEach(s => chips.appendChild(chip(s, false)));
  w.append(h, p, chips);
  $("log").appendChild(w);
}

function chip(text, isCatalogue) {
  const b = document.createElement("button");
  b.className = "chip" + (isCatalogue ? " cat" : "");
  b.textContent = text;
  b.onclick = () => { $("q").value = text; send(); };
  return b;
}

function add(text, cls) {
  const d = document.createElement("div");
  d.className = cls;
  // textContent, never innerHTML: model output is not trusted markup,
  // and product descriptions are written by sellers.
  d.textContent = text;
  $("log").appendChild(d);
  stick(true);
  return d;
}

function rupees(v) {
  if (v === null || v === undefined) return "—";
  return "₹" + Number(v).toLocaleString("en-IN");
}

/* THE PRODUCTS THE ANSWER DREW ON, drawn by us rather than written by
   the model. Every field here came out of a tool result and travelled
   beside the prose - the model never saw a URL and never composed one,
   so a card cannot point at a product the answer was not about.
   textContent throughout, same as add(): these names are seller-written. */
function productCards(products) {
  if (!products || !products.length) return null;
  const wrap = document.createElement("div");
  wrap.className = "cards";

  products.forEach(p => {
    const card = document.createElement("button");
    card.className = "card";
    card.type = "button";

    const row = document.createElement("div");
    row.className = "row";

    // Four of the 143 products carry no image, and a CDN link can always
    // fail. Either way the row keeps its name and price rather than
    // holding a broken-image icon.
    if (p.image) {
      const img = document.createElement("img");
      img.className = "thumb";
      img.src = p.image;
      img.alt = "";
      img.loading = "lazy";
      img.onerror = () => img.remove();
      row.appendChild(img);
    }

    const body = document.createElement("div");
    body.className = "body";

    const nm = document.createElement("div");
    nm.className = "nm";
    nm.textContent = p.name || "Unnamed product";

    const pr = document.createElement("div");
    pr.className = "pr";
    const cut = p.discountedPrice !== null && p.discountedPrice !== undefined
             && p.price !== null && p.price !== undefined
             && p.discountedPrice < p.price;
    const now = document.createElement("b");
    const effective = (p.discountedPrice === null || p.discountedPrice === undefined)
      ? p.price : p.discountedPrice;
    now.textContent = rupees(effective);
    pr.appendChild(now);
    if (cut) {
      const was = document.createElement("s");
      was.textContent = rupees(p.price);
      const off = document.createElement("span");
      off.className = "off";
      off.textContent = "-" + Math.round((1 - p.discountedPrice / p.price) * 100) + "%";
      pr.append(was, off);
    }

    const detail = document.createElement("div");
    detail.className = "detail";
    detail.textContent = "Opens this product in the marketplace app.";

    body.append(nm, pr);
    row.appendChild(body);
    card.append(row, detail);
    card.onclick = () => card.classList.toggle("open");
    wrap.appendChild(card);
  });

  $("log").appendChild(wrap);
  stick();
  return wrap;
}

/* WHERE EACH NUMBER CAME FROM.

   The server sends character offsets, and the text is SLICED by them
   rather than searched for: a reply saying "₹599" twice would otherwise
   mark the first occurrence twice and the second never.

   Built from text nodes and <mark> elements - never innerHTML. The reply
   is model output and the claim text came out of seller-written data, so
   the same rule that governs add() governs this. */
function renderAttributed(bubble, text, claims) {
  bubble.textContent = "";
  let at = 0;
  (claims || []).forEach(c => {
    // Defensive: a stale or overlapping span would garble the reply, and
    // showing it wrong is worse than showing it unmarked.
    if (typeof c.start !== "number" || c.start < at || c.end > text.length) return;
    if (c.start > at) bubble.appendChild(document.createTextNode(text.slice(at, c.start)));

    const label = SOURCES[c.tool] || "a lookup";
    const mark = document.createElement("mark");
    mark.className = "src";
    mark.textContent = text.slice(c.start, c.end);
    mark.title = "read from " + label;
    mark.onclick = () => showSource(bubble, mark.textContent, label);
    bubble.appendChild(mark);
    at = c.end;
  });
  if (at < text.length) bubble.appendChild(document.createTextNode(text.slice(at)));
}

function showSource(bubble, text, label) {
  let note = bubble.nextElementSibling;
  if (!note || !note.classList.contains("srcnote")) {
    note = document.createElement("div");
    note.className = "srcnote";
    bubble.parentNode.insertBefore(note, bubble.nextSibling);
  }
  note.textContent = "";
  const strong = document.createElement("b");
  strong.textContent = text;
  note.append(strong, document.createTextNode(" — read from " + label));
  stick();
}

function metaLine(timing, checked, copyText) {
  const m = document.createElement("div");
  m.className = "meta";
  const t = document.createElement("span");
  t.textContent = timing;
  m.appendChild(t);
  if (checked) {
    const c = document.createElement("span");
    c.className = "checked";
    c.textContent = "✓ checked " + checked;
    m.appendChild(c);
  }
  if (copyText) {
    const btn = document.createElement("button");
    btn.className = "copy";
    btn.textContent = "copy";
    btn.onclick = () => {
      navigator.clipboard.writeText(copyText).then(() => {
        btn.textContent = "copied";
        setTimeout(() => { btn.textContent = "copy"; }, 1400);
      }).catch(() => { btn.textContent = "press ⌘C"; });
    };
    m.appendChild(btn);
  }
  $("log").appendChild(m);
  stick();
  return m;
}

/* Follow the conversation only if the reader is already near the bottom,
   so scrolling up to re-read does not fight every arriving token. */
function atBottom() {
  const el = $("logScroll");
  return el.scrollHeight - el.scrollTop - el.clientHeight < 90;
}
function stick(force) {
  const el = $("logScroll");
  if (force || atBottom()) el.scrollTop = el.scrollHeight;
  $("jumpBtn").hidden = atBottom();
}
document.addEventListener("DOMContentLoaded", () => {
  $("logScroll").addEventListener("scroll", () => { $("jumpBtn").hidden = atBottom(); });
});

/* ---------- sign in ---------- */
async function loadAccounts() {
  const box = $("accounts");
  try {
    const r = await fetch("/demo/accounts", {headers: codeHeaders()});
    if (r.status === 401) {
      // The server wants the shared code. Handled separately from every
      // other failure because the remedy differs: this one the visitor
      // can fix, by typing it.
      askForCode(accessCode ? "That code was not accepted." : null);
      return;
    }
    if (!r.ok) {
      showTokenEntry();
      const body = await r.json().catch(() => ({}));
      $("gateHint").textContent = body.detail
        || "Account picking is unavailable here - paste an access token to continue.";
      return;
    }
    const {accounts, inactive} = await r.json();
    if (!accounts.length) {
      showTokenEntry();
      $("gateHint").textContent = "No accounts with orders in this database.";
      return;
    }
    accounts.forEach(a => box.appendChild(accountRow(a)));
    if (inactive && inactive.total) box.appendChild(inactiveSection(inactive));
  } catch (e) {
    showTokenEntry();
    $("gateHint").textContent = "Couldn't reach the server. Is it running?";
  }
}

function accountRow(a) {
  const b = document.createElement("button");
  b.className = "row";
  const facts = [a.orders + (a.orders === 1 ? " order" : " orders")];
  if (a.cart) facts.push("cart");
  if (a.city) facts.push(a.city);

  const av = document.createElement("div");
  av.className = "av";
  av.textContent = (a.username || "?").trim().charAt(0);
  const nm = document.createElement("div");
  nm.className = "nm";
  const who = document.createElement("b");
  who.textContent = a.username;
  const sub = document.createElement("span");
  sub.textContent = facts.join(" · ");
  nm.append(who, sub);
  const go = document.createElement("div");
  go.className = "go";
  go.textContent = "→";

  b.append(av, nm, go);
  b.onclick = () => signIn(a.username, b);
  return b;
}

function inactiveSection(inactive) {
  const d = document.createElement("details");
  d.className = "more";
  const sum = document.createElement("summary");
  sum.textContent = inactive.total + " accounts have never ordered";
  d.appendChild(sum);
  const note = document.createElement("p");
  note.className = "more-note";
  note.textContent = (inactive.shown.length < inactive.total
    ? "Showing " + inactive.shown.length + ". " : "")
    + "Signing in as one shows how the assistant answers with nothing to report.";
  d.appendChild(note);
  const list = document.createElement("div");
  list.className = "rows capped";
  inactive.shown.forEach(a => {
    const b = document.createElement("button");
    b.className = "row thin";
    const av = document.createElement("div");
    av.className = "av quiet";
    av.textContent = (a.username || "?").trim().charAt(0);
    const nm = document.createElement("div");
    nm.className = "nm";
    const who = document.createElement("b");
    who.textContent = a.username;
    nm.appendChild(who);
    if (a.cart) {
      const s = document.createElement("span");
      s.textContent = "has a cart";
      nm.appendChild(s);
    }
    b.append(av, nm);
    b.onclick = () => signIn(a.username, b);
    list.appendChild(b);
  });
  d.appendChild(list);
  return d;
}

function showTokenEntry() {
  $("tokenField").hidden = false;
  $("gateBtn").hidden = false;
  $("altBtn").hidden = true;
  $("token").focus();
}

async function signIn(username, row) {
  $("gateErr").textContent = "";
  document.querySelectorAll(".row").forEach(c => c.disabled = true);
  try {
    // The server mints the token. The page never sees a signing secret
    // and never decides who anyone is.
    const r = await fetch("/demo/token", {
      method: "POST",
      headers: Object.assign({"Content-Type": "application/json"}, codeHeaders()),
      body: JSON.stringify({username}),
    });
    const body = await r.json().catch(() => ({}));
    if (!r.ok) {
      $("gateErr").textContent = body.detail || ("Sign-in failed (HTTP " + r.status + ")");
      return;
    }
    token = body.token;
    account = body.username || username;
    // Per-account questions, naming this person own orders. Falls back
    // to the shared catalogue picks server-side, so this is never empty.
    if (Array.isArray(body.suggestions) && body.suggestions.length) {
      CATALOGUE_STARTERS = body.suggestions;
    }
    $("live").classList.add("on");
    enter();
  } catch (e) {
    $("gateErr").textContent = "Couldn't reach the server. Is it running?";
  } finally {
    document.querySelectorAll(".row").forEach(c => c.disabled = false);
  }
}

// The real-data path: generate_test_token.py --real-data, pasted here.
function startWithToken() {
  const t = $("token").value.trim().replace(/^Bearer\s+/i, "");
  if (!t) { $("gateErr").textContent = "Paste a token to continue."; return; }
  token = t;
  account = "token";
  $("live").classList.add("on");
  enter();
}

// Straight into the last conversation if there is one - a returning
// demo should not make you click through a list you have already seen.
function enter() {
  const list = loadThreads();
  if (list.length) { thread = list[0]; renderChat(); showScreen("chat"); $("q").focus(); }
  else newThread();
}

/* ---------- sending ---------- */
function onSendClick() { busy ? abort() : send(); }

function abort() {
  if (controller) { try { controller.abort(); } catch (e) {} }
}

function setBusy(state) {
  busy = state;
  const b = $("send");
  b.classList.toggle("stop", state);
  b.innerHTML = state ? "&#9632;" : "&uarr;";
  b.setAttribute("aria-label", state ? "Stop" : "Send");
  if (!state) $("q").focus();
}

async function send() {
  if (busy) return;
  const box = $("q");
  const text = box.value.trim();
  if (!text) return;
  if (!thread) newThread();

  const w = $("log").querySelector(".welcome");
  if (w) w.remove();
  box.value = "";
  setBusy(true);
  add(text, "msg user");
  thread.messages.push({role: "user", text});
  if (thread.messages.filter(m => m.role === "user").length === 1) {
    // Title from the first question, the way every chat app does it.
    thread.title = text.length > 42 ? text.slice(0, 42).trim() + "…" : text;
    $("barTitle").textContent = thread.title;
  }

  const wait = add("", "msg bot typing");
  wait.innerHTML = "<i></i><i></i><i></i>";
  let status = null, bubble = null, firstToken = null;
  const checked = [];
  let products = [];
  let attribution = [];
  const t0 = performance.now();

  function showStatus(label) {
    if (!status) {
      status = document.createElement("div");
      status.className = "status";
      status.innerHTML = '<span class="pulse"></span><span class="label"></span>';
      $("log").appendChild(status);
    }
    status.querySelector(".label").textContent = label;
    stick();
  }
  function clearScaffolding() {
    if (wait) wait.remove();
    if (status) { status.remove(); status = null; }
  }
  function appendToken(t) {
    if (!bubble) { clearScaffolding(); bubble = add("", "msg bot writing"); firstToken = performance.now() - t0; }
    // MEASURED BEFORE THE TEXT GROWS, and that is the whole bug.
    //
    // This used to append and then call stick(), which asks atBottom()
    // AFTER the height has already jumped. Tokens do not arrive one
    // character at a time - they arrive in chunks, and the first chunk
    // taller than atBottom()'s 90px slack makes the answer look like
    // something the reader had scrolled away from. stick() then declines
    // to scroll, correctly by its own rule, and never recovers for the
    // rest of the turn.
    //
    // Measured on a real answer: scrollTop stayed at 0 from the first
    // chunk to the last, so the eight product cards appended below
    // landed 800px out of view and read as missing entirely.
    const following = atBottom();
    bubble.textContent += t;
    if (following) stick(true);
  }
  function finish(reply) {
    // ASKED BEFORE ANYTHING MOVES, and that order is the whole fix.
    //
    // finish() replaces the streamed text with the authoritative reply,
    // which changes the bubble's height and can push the view off the
    // bottom. productCards() then appends BELOW that and calls stick(),
    // which - correctly, by its own rule - sees atBottom() is false and
    // declines to scroll. The cards land out of sight every time the
    // answer is long enough to matter, and a reader who does not think
    // to scroll concludes there are no cards at all. Measured: eight
    // cards rendered, none visible.
    //
    // So the question is whether the reader was FOLLOWING, and it has to
    // be asked before the height changes rather than after.
    const following = atBottom();
    clearScaffolding();
    // `done` carries the authoritative reply. Usually identical to what
    // the tokens drew - but the fallbacks (rate limited, provider down)
    // never arrive as tokens at all, and this is what puts them on
    // screen without the client needing to know they are special.
    if (!bubble) bubble = add("", "msg bot");
    // The tokens drew plain text as they arrived; `done` carries the
    // authoritative reply, which is the first point the spans can be
    // trusted to line up with what is on screen.
    if (reply) {
      if (attribution.length) renderAttributed(bubble, reply, attribution);
      else bubble.textContent = reply;
    }
    bubble.classList.remove("writing");
    // Between the answer and its meta line, so the "checked" footnote
    // still reads as the last word on the turn.
    productCards(products);

    const ms = performance.now() - t0;
    const timing = firstToken
      ? (firstToken / 1000).toFixed(1) + "s to first word · " + (ms / 1000).toFixed(1) + "s total"
      : (ms / 1000).toFixed(1) + "s";
    const src = checked.join(" · ");
    metaLine(timing, src, bubble.textContent);
    thread.messages.push({role: "bot", text: bubble.textContent, products, attribution, meta: {timing, checked: src}});
    thread.updatedAt = Date.now();
    upsertThread();

    // Only for a reader who was already following. Someone who scrolled
    // up to re-read mid-answer is not dragged back down - that is the
    // rule stick() exists to protect, and this does not weaken it.
    if (following) stick(true);
  }
  function fail(message) {
    clearScaffolding();
    add(message, "msg err");
    thread.messages.push({role: "error", text: message});
    thread.updatedAt = Date.now();
    upsertThread();
  }

  controller = new AbortController();
  try {
    const r = await fetch("/chat/stream", {
      method: "POST",
      headers: {"Content-Type": "application/json", "Authorization": "Bearer " + token},
      body: JSON.stringify({message: text, session_id: thread.id}),
      signal: controller.signal,
    });

    if (r.status === 401) { fail("That session has expired — go back and sign in again."); return; }
    if (r.status === 429) { fail("Too many questions at once. Give it a few seconds and try again."); return; }
    if (!r.ok)            { fail("The server returned HTTP " + r.status + "."); return; }

    const reader = r.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream: true});

      // SSE frames are separated by a blank line. A partial frame stays
      // buffered until the rest arrives - chunk boundaries do not
      // respect message boundaries.
      let split;
      while ((split = buffer.indexOf("\n\n")) !== -1) {
        const frame = buffer.slice(0, split);
        buffer = buffer.slice(split + 2);
        let name = "", data = "";
        for (const line of frame.split("\n")) {
          if (line.startsWith("event:")) name = line.slice(6).trim();
          else if (line.startsWith("data:")) data += line.slice(5).trim();
        }
        if (!data) continue;
        const payload = JSON.parse(data);
        if (name === "status") {
          const source = SOURCES[payload.tool];
          if (source && !checked.includes(source)) checked.push(source);
          showStatus(payload.label);
        }
        else if (name === "token") appendToken(payload.text);
        else if (name === "products") products = payload.products || [];
        else if (name === "attribution") attribution = payload.claims || [];
        else if (name === "done")  finish(payload.reply);
        else if (name === "error") { fail(payload.message); return; }
      }
    }
  } catch (e) {
    // An abort is the user's own doing, not a failure - keep whatever
    // arrived and say plainly that it was stopped.
    if (e && e.name === "AbortError") {
      clearScaffolding();
      if (bubble) {
        bubble.classList.remove("writing");
        thread.messages.push({role: "bot", text: bubble.textContent, meta: {timing: "stopped", checked: checked.join(" · ")}});
        metaLine("stopped", checked.join(" · "), bubble.textContent);
      } else {
        add("Stopped.", "msg err");
      }
      thread.updatedAt = Date.now();
      upsertThread();
    } else {
      fail("Couldn't reach the server. Is it still running?");
    }
  } finally {
    controller = null;
    setBusy(false);
  }
}

/* ---------- boot ---------- */
async function loadScope() {
  try {
    const s = await fetch("/demo/stats", {headers: codeHeaders()}).then(r => r.ok ? r.json() : null);
    if (!s) return;
    if (Array.isArray(s.suggestions)) CATALOGUE_STARTERS = s.suggestions;
    const box = $("scope");
    box.appendChild(tileGroup("Catalogue", s.catalogue));
    box.appendChild(tileGroup("Activity", s.activity));
    if (s.accounts && s.accounts.total) box.appendChild(accountSplit(s.accounts));
    box.hidden = false;
  } catch (e) { /* the demo opens regardless */ }
}

function tileGroup(title, items) {
  const wrap = document.createElement("div");
  wrap.style.marginBottom = "12px";
  const h = document.createElement("h3");
  h.className = "sect";
  h.style.marginBottom = "7px";
  h.textContent = title;
  const grid = document.createElement("div");
  grid.className = "tiles";
  items.forEach(it => {
    const t = document.createElement("div");
    t.className = "tile";
    const v = document.createElement("b");
    v.textContent = it.value.toLocaleString();
    const l = document.createElement("span");
    l.textContent = it.label;
    t.append(v, l);
    grid.appendChild(t);
  });
  wrap.append(h, grid);
  return wrap;
}

// The one thing here drawn rather than printed, because it is a
// part-to-whole and the whole is the point.
function accountSplit(a) {
  const wrap = document.createElement("div");
  const h = document.createElement("h3");
  h.className = "sect";
  h.style.marginBottom = "7px";
  h.textContent = a.total.toLocaleString() + " accounts";
  const bar = document.createElement("div");
  bar.className = "split-bar";
  const buyers = document.createElement("i");
  buyers.className = "buyers";
  buyers.style.width = Math.max(2, (a.buyers / a.total) * 100) + "%";
  const rest = document.createElement("i");
  rest.className = "rest";
  bar.append(buyers, rest);
  // DIRECT LABELS, not a bare legend: the two fills sit in the CVD 6-8
  // separation band, where colour alone is not enough.
  const key = document.createElement("div");
  key.className = "split-key";
  key.style.marginTop = "7px";
  [["#00a878", a.buyers, "have ordered"],
   ["#8b94ab", a.never_ordered, "never have"]].forEach(([c, n, label]) => {
    const s = document.createElement("span");
    const sw = document.createElement("i");
    sw.style.background = c;
    const b = document.createElement("b");
    b.textContent = n.toLocaleString();
    s.append(sw, b, document.createTextNode(" " + label));
    key.appendChild(s);
  });
  wrap.append(h, bar, key);
  return wrap;
}

loadScope();
loadAccounts();

// A quiet liveness check, so the header dot tells the truth rather than
// being decorative. Failure is silent: the dot stays grey.
fetch("/health").then(r => r.json()).then(h => {
  if (h.status === "ok" && token) $("live").classList.add("on");
}).catch(() => {});
</script>
"""


def _check_access(code: str | None) -> None:
    """Refuses the demo data endpoints unless the shared code matches.

    APPLIED TO THE DATA, NOT THE PAGE. /demo itself stays public and
    serves an HTML shell that knows nothing - there is no point hiding
    a page whose only content arrives from the three endpoints below.
    Gating those means an unauthorised visitor sees a code box and
    literally nothing else: no account names, no counts, no catalogue.

    UNSET MEANS NO GATE. Local demos and development are unchanged; the
    code only exists once the service is somewhere reachable.

    compare_digest rather than ==, so a wrong code takes the same time
    as a right one and cannot be guessed character by character.
    """
    expected = get_settings().demo_access_code
    if not expected:
        return
    if not code or not secrets.compare_digest(code, expected):
        raise HTTPException(
            status_code=401,
            detail="This demo needs an access code.",
            headers={"X-Demo-Code-Required": "1"},
        )


@router.get("/demo", response_class=HTMLResponse, include_in_schema=False)
async def demo_ui() -> str:
    return PAGE


@router.get("/demo/accounts", include_in_schema=False)
async def demo_accounts(x_demo_code: str | None = Header(default=None)):
    """The accounts the picker offers, with enough context to choose.

    WHY A PICKER RATHER THAN A TEXT BOX. A demo that opens by asking a
    stranger to type a name they cannot know is a demo that opens with
    the visitor stuck. Worse, a typo produces "no account called ..."
    as the very first thing the product ever says to them.

    The counts are not decoration either. "6 orders - cart - Pune" says,
    before a single question is asked, that this is a real account with
    real history behind it - which is the thing the whole project is
    trying to demonstrate and otherwise has to be asserted out loud.

    ORDERED BY HOW MUCH THERE IS TO SHOW, and accounts with nothing are
    dropped: a card reading "0 orders" invites exactly the click that
    makes the assistant look empty.

    NO DATABASE GUARD, AND THAT IS A NARROW, DELIBERATE DECISION.
    An earlier version refused to run against real customer data,
    reasoning that a list of real buyers with one-click sign-in is a
    customer directory. That reasoning holds completely for a DEPLOYED
    instance - and does not hold for the machine running the demo.
    Whoever opens this page locally already has .env, which carries the
    database credentials and the JWT signing secret; they can read every
    one of these accounts with or without this endpoint. Refusing bought
    no protection from them and only made the demo worse.

    SO THE PROTECTION MOVED RATHER THAN VANISHED. It is now entirely
    DEMO_UI_ENABLED, which removes this endpoint, the token minter and
    the page together. That setting used to be about tidiness - an
    unnecessary door - and is now the only thing between a public URL
    and a customer directory. main.py logs a warning at every startup
    where this combination is live, and docs/deployment.md says so.

    ONE AGGREGATION, NOT A QUERY PER USER. The first version counted
    orders in a loop, which is fine for four demo accounts and 286
    round trips against the real database. Grouping in Mongo returns the
    same answer in about 30ms.
    """
    _check_access(x_demo_code)
    db = get_database()

    by_buyer = await db["orders"].aggregate([
        {"$group": {"_id": "$buyerId", "orders": {"$sum": 1}}},
        {"$sort": {"orders": -1}},
        {"$limit": MAX_ACCOUNTS},
    ]).to_list(length=MAX_ACCOUNTS)
    if not by_buyer:
        return {"accounts": []}

    ids = [row["_id"] for row in by_buyer]
    users = {
        u["_id"]: u
        for u in await db[USERS_COLLECTION]
        .find({"_id": {"$in": ids}}, {"username": 1})
        .to_list(length=len(ids))
    }
    cities = {
        a["user"]: a.get("city")
        for a in await db["addresses"]
        .find({"user": {"$in": ids}}, {"user": 1, "city": 1})
        .to_list(length=None)
    }
    carts = {
        c["user"]
        for c in await db["carts"]
        .find({"user": {"$in": ids}}, {"user": 1})
        .to_list(length=None)
    }

    accounts = []
    for row in by_buyer:
        user = users.get(row["_id"])
        # Unpickable without a username - sign-in looks users up by it.
        if not user or not user.get("username"):
            continue
        accounts.append({
            "username": user["username"],
            "orders": row["orders"],
            "cart": 1 if row["_id"] in carts else 0,
            "city": cities.get(row["_id"]),
        })

    return {"accounts": accounts, "inactive": await _accounts_with_no_orders(db, ids)}


async def _accounts_with_no_orders(db, buyer_ids: list) -> dict:
    """The accounts that have never ordered, for the collapsed section.

    WHY SHOW THEM AT ALL, given a card reading "0 orders" is exactly the
    click that makes the assistant look empty. Two reasons, and they are
    both about honesty:

    The picker otherwise implies the database holds sixteen people. It
    holds 286, and the other 270 have simply never bought anything -
    which is a fact about the business, not a gap in the demo. Hiding
    them entirely invites "is that all?", which is precisely the
    question that prompted this.

    And an empty account is worth demonstrating: asked about orders, the
    assistant says so plainly rather than inventing any. That is the
    read-only, answer-from-data behaviour the whole project claims, and
    it is easier to show than to assert.

    CARTS FIRST. Fourteen of them have a cart, so those are the only
    ones with anything at all to answer about - they belong at the top
    where someone demoing will actually find them.

    NOT LABELLED "sellers", though every one carries sellerStatus: so do
    all sixteen buyers, so the label would distinguish nothing.
    """
    query = {"_id": {"$nin": buyer_ids}, "username": {"$exists": True, "$ne": None}}
    total = await db[USERS_COLLECTION].count_documents(query)
    if not total:
        return {"total": 0, "shown": []}

    users = await db[USERS_COLLECTION].find(
        query, {"username": 1}
    ).limit(MAX_INACTIVE).to_list(length=MAX_INACTIVE)

    ids = [u["_id"] for u in users]
    with_cart = {
        c["user"]
        for c in await db["carts"].find({"user": {"$in": ids}}, {"user": 1}).to_list(
            length=None
        )
    }

    shown = [
        {"username": u["username"], "cart": 1 if u["_id"] in with_cart else 0}
        for u in users
    ]
    # Carts first, then alphabetical - a stable order, and the useful
    # ones where they can be seen.
    shown.sort(key=lambda a: (-a["cart"], str(a["username"]).lower()))
    return {"total": total, "shown": shown}


@router.get("/demo/stats", include_in_schema=False)
async def demo_stats(x_demo_code: str | None = Header(default=None)):
    """What the assistant can actually see, counted.

    WHY IT IS ON THE OPENING SCREEN. The demo answers questions one at a
    time, which makes the data behind it invisible - a client watching
    six replies has no way to tell whether there are twenty records back
    there or twenty thousand. Stating the scale before the first
    question turns "it answered" into "it answered out of 143 products
    and 134 orders".

    It also pre-empts the obvious doubt. The account picker lists
    sixteen people; the database holds 286, and the other 270 have never
    ordered. Shown, that is a fact about the business. Unshown, it is
    the first thing someone asks about.

    COUNTED IN PARALLEL. Nine counts run concurrently against Atlas
    rather than one after another - the page waits for the slowest, not
    for the sum.
    """
    _check_access(x_demo_code)
    db = get_database()

    async def count(collection, query=None):
        return await db[collection].count_documents(query or {})

    (
        products, orders, users, sessions, bits, reviews, coupons, bargains, carts,
    ) = await asyncio.gather(
        count("products"), count("orders"), count(USERS_COLLECTION),
        count("livesessions"), count("bits"), count("reviews"),
        count("coupons"), count("bargains"), count("carts"),
    )

    buyers = len(await db["orders"].distinct("buyerId"))

    return {
        "suggestions": await _catalogue_questions(db),
        # The catalogue side - what any question about products reaches.
        "catalogue": [
            {"label": "products", "value": products},
            {"label": "live sessions", "value": sessions},
            {"label": "Bits", "value": bits},
            {"label": "coupons", "value": coupons},
        ],
        # The customer side.
        "activity": [
            {"label": "orders", "value": orders},
            {"label": "carts", "value": carts},
            {"label": "bargains", "value": bargains},
            {"label": "reviews", "value": reviews},
        ],
        # Part-to-whole, and the only thing here drawn rather than
        # printed - see the accounts bar in the page.
        "accounts": {"total": users, "buyers": buyers, "never_ordered": users - buyers},
    }


async def _catalogue_questions(db) -> list[dict]:
    """Starter questions naming products that actually exist.

    WHY THESE ARE BUILT FROM THE DATABASE. The first version of the chips
    deliberately avoided naming any product, because the demo dataset
    held twelve and a chip that misses makes "I couldn't find that" the
    product's opening line. Against 143 real products that caution costs
    more than it saves: search, similarity, stock and bargaining are the
    strongest things the assistant does, and a visitor who cannot see
    the catalogue has no way to reach any of them - they would have to
    guess a product name.

    So the names come from the catalogue at page load, and a chip cannot
    reference something absent.

    WHAT MAKES A PRODUCT PICKABLE. A name between 8 and 38 characters,
    which excludes both "black" (there are several, and it reads as a
    colour rather than a thing) and the long marketing titles that wrap
    onto three lines in a chip. Variants, so the stock and size
    questions have something to answer with. Not sold, or the answer is
    about something nobody can buy.

    ONE PER CATEGORY, so three chips do not all ask about shirts.

    ORDERED, NOT RANDOM. The same page gives the same three every time -
    a demo you have rehearsed should not reshuffle itself while a client
    is watching.
    """
    candidates = await db["products"].aggregate([
        {"$match": {
            "isSold": False,
            "variants.0": {"$exists": True},
            "$expr": {"$and": [
                {"$gte": [{"$strLenCP": "$name"}, 8]},
                {"$lte": [{"$strLenCP": "$name"}, 38]},
            ]},
        }},
        {"$sort": {"category": 1, "name": 1}},
        {"$project": {"name": 1, "category": 1}},
        {"$limit": 120},
    ]).to_list(length=120)

    picked, seen = [], set()
    for product in candidates:
        category = product.get("category")
        if category in seen:
            continue
        name = (product.get("name") or "").strip()
        if not name:
            continue
        seen.add(category)
        picked.append(name)
        if len(picked) == 3:
            break

    # Each phrasing exercises a different tool path: name lookup, vector
    # similarity, and the bargaining rules. Every product carries
    # bargainSettings, so the third never comes back "not allowed".
    phrasings = [
        "Tell me about the {}",
        "Anything similar to the {}?",
        "Can I bargain on the {}?",
    ]
    return [phrasing.format(name) for name, phrasing in zip(picked, phrasings)]


def _usable_name(value) -> str | None:
    """A product name a chip can carry.

    The same 8-38 character rule the catalogue picks use, for the same
    reasons: several real products are called "black", which reads as a
    colour rather than a thing, and the long marketing titles wrap onto
    three lines in a chip.
    """
    name = (value or "").strip()
    return name if 8 <= len(name) <= 38 else None


async def _questions_for_user(db, user_id, fallback: list[str]) -> list[str]:
    """Starter questions spread ACROSS FEATURE AREAS, grounded per account.

    WHY BREADTH IS THE POINT. The first version asked about orders three
    times, so the assistant read as an order tracker in the first three
    seconds - when it also does bargaining, semantic search, live
    sessions, Bits, coupons and reviews across 35 tools. The opening row
    is the only place a visitor learns what to ask for, so it should
    show the range rather than the same area repeatedly.

    EVERY CHIP IS BACKED BY A REAL RECORD, and the checks below exist
    because three of them would otherwise be broken on this database:

      coupons  16 codes are active, but several are "11", "123" and
               "1211". "Is 11 still valid?" reads as a bug, so codes
               must contain a letter and no spaces.
      reviews  9 products have a review and only FIVE of those products
               still exist - the rest reference deleted rows. Asking
               about a review whose product is gone answers "not found".
      hashtags stored inconsistently, some with a leading # and some
               without, so they are normalised before use.

    Anything unbacked is simply skipped: a shorter row of chips that all
    work beats a longer one where two apologise.
    """
    # -- this account's own items, newest first -----------------------
    orders = await db["orders"].find(
        {"buyerId": user_id}, {"items.name": 1}
    ).sort("createdAt", -1).limit(6).to_list(length=6)

    mine, seen = [], set()
    for order in orders:
        for item in order.get("items") or []:
            name = _usable_name(item.get("name"))
            if name and name.lower() not in seen:
                seen.add(name.lower())
                mine.append(name)

    cart = await db["carts"].find_one({"user": user_id}, {"items.product": 1})
    cart_items = (cart or {}).get("items") or []
    cart_name = None
    for item in cart_items:
        product = await db["products"].find_one(
            {"_id": item.get("product")}, {"name": 1}
        )
        candidate = _usable_name((product or {}).get("name"))
        if candidate:
            cart_name = candidate
            break

    # A product to bargain over or compare against. Theirs if possible,
    # since "can I bargain on the thing I own" is the better story.
    catalogue = [q.split("the ", 1)[-1].rstrip("?") for q in fallback]
    pool = mine + [c for c in catalogue if c]

    questions: list[str] = []
    # Every product named in a chip goes in here, so no two chips ask
    # about the same thing. Without it an account whose history is one
    # product got "Where is my digital clock?" followed immediately by
    # "Anything similar to the digital clock?" - which reads as a bug
    # rather than as breadth.
    used: set[str] = set()

    def take(*candidates):
        for name in candidates:
            if name and name.lower() not in used:
                used.add(name.lower())
                return name
        return None

    # 1. ORDERS - the area they arrived expecting.
    ordered = take(mine[0] if mine else None)
    if ordered:
        questions.append("Where is my " + ordered + "?")

    # 2. BARGAINING - the signature feature, and every product carries
    #    bargainSettings, so this can never come back "not allowed".
    bargain_on = take(cart_name, *mine, *pool)
    if bargain_on:
        questions.append("Can I bargain on the " + bargain_on + "?")

    # 3. DISCOVERY - vector similarity, the thing a keyword box cannot do.
    similar_to = take(*mine, *pool)
    if similar_to:
        questions.append("Anything similar to the " + similar_to + "?")

    # 4. COUPONS - and this one prefers a coupon that is valid TODAY.
    #
    #    isActive is not the same as usable: measured, 16 codes are
    #    active and exactly ONE is inside its date window. Picking on
    #    isActive alone opened the demo with "no, that is expired" -
    #    a correct answer, and a flat one.
    #
    #    The fallback is deliberate rather than a safety net: the valid
    #    coupon expires soon, and when it does this quietly returns to
    #    demonstrating the expiry path instead of showing nothing.
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    readable = {"$regex": r"^[A-Za-z][A-Za-z0-9 ]{3,13}$"}
    coupon = await db["coupons"].find_one(
        {"isActive": True, "code": readable,
         "startDate": {"$lte": now}, "endDate": {"$gte": now}},
        {"code": 1},
    ) or await db["coupons"].find_one(
        {"isActive": True, "code": readable}, {"code": 1}
    )
    if coupon and coupon.get("code"):
        # "Is S QUARE still valid?" made the model stop and ask whether
        # that was a coupon, a seller or a product - a fair question,
        # since the only currently-valid code on this database contains
        # a space. Naming the type removes the ambiguity for every code,
        # spaced or not.
        questions.append(
            'Is the coupon "' + str(coupon["code"]).strip() + '" still valid?'
        )

    # 5. BITS - a real hashtag, normalised: some are stored with a
    #    leading # and some without.
    bit = await db["bits"].find_one(
        {"hashtags.0": {"$exists": True}}, {"hashtags": 1}
    )
    for raw in (bit or {}).get("hashtags") or []:
        tag = str(raw or "").lstrip("#").strip()
        if 4 <= len(tag) <= 18 and tag.replace(" ", "").isalnum():
            questions.append("Show me Bits about #" + tag)
            break

    # 6. CART - only when there is something in it.
    if cart_items:
        questions.append("What's in my cart?")

    return questions[:6]


class DemoSignIn(BaseModel):
    username: str = Field(..., min_length=1, max_length=100)


@router.post("/demo/token", include_in_schema=False)
async def demo_token(request: DemoSignIn, x_demo_code: str | None = Header(default=None)):
    """Mints a token for a named demo account.

    THIS IS AN AUTHENTICATION BYPASS, AND THAT IS THE POINT. It takes a
    username and hands back a signed token for that user with no
    password, no proof, nothing. Pasting a JWT is a terrible way to open
    a demo - the person watching sees a wall of base64 and learns
    nothing about the product - so the demo page asks for a name
    instead.

    WHICH IS ONLY ACCEPTABLE BECAUSE OF DEMO_UI_ENABLED. Anyone who can
    reach this endpoint can read any listed customer's orders, delivery
    city and cart by naming them. On the machine running the demo that
    grants nothing new - .env is right there, with the database
    credentials and the signing secret in it. Reachable from anywhere
    else, it is a customer directory with one-click impersonation.

    There is no database check here any more, deliberately - see
    demo_accounts() for the full reasoning. The single protection is
    that this endpoint, the account listing and the page all live on the
    demo_ui router, so DEMO_UI_ENABLED=false removes all three together
    and a deployment cannot leave the token minter behind. That setting
    is now load-bearing rather than tidy; main.py warns at startup when
    it is on against real data.
    """
    _check_access(x_demo_code)
    db = get_database()
    name = request.username.strip()
    user = await db[USERS_COLLECTION].find_one(
        # Exact match, case-insensitive, escaped - a username is not a
        # search box and must never be a regex the caller controls.
        {"username": {"$regex": f"^{re.escape(name)}$", "$options": "i"}},
        {"username": 1},
    )
    if user is None:
        known = [
            u.get("username")
            for u in await db[USERS_COLLECTION]
            .find({"username": {"$exists": True}}, {"username": 1})
            .limit(10)
            .to_list(length=10)
        ]
        raise HTTPException(
            status_code=404,
            detail=f"No account called '{name}'. Try one of: "
                   + ", ".join(n for n in known if n),
        )

    return {
        "token": create_test_token(
            user_id=str(user["_id"]), expires_in_minutes=TOKEN_MINUTES
        ),
        "username": user.get("username"),
        # Built here rather than at page load, because only now do we
        # know whose account this is - and the whole point of the chips
        # is that they name this person's own orders.
        "suggestions": await _questions_for_user(
            db, user["_id"], await _catalogue_questions(db)
        ),
    }
