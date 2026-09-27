"""The model's roles: hypothesis, critic, adversarial reviewer.

Code writing is deliberately NOT automated: LLM-written strategy code is never
imported or executed by the bot (a process holding exchange keys should not run
generated code). Hypotheses land in research/hypotheses/ for a human to implement
and register; from there the research loop and the gates are fully automatic.

Requires ANTHROPIC_API_KEY (or an `ant auth login` profile).
"""
from __future__ import annotations

import inspect
import json

import anthropic

MODEL = "claude-opus-5"

CRITIC_PROMPT = """Review this backtest for the following errors. For each one,
state PRESENT or ABSENT and quote the line.

1. Look-ahead: is the signal shifted before becoming a position?
2. Survivorship: does the asset list include delisted tickers?
3. Repainting: does any indicator use future data (centered
   moving averages, zigzag, unshifted resample)?
4. Costs: are fees AND slippage applied on turnover?
5. Fill assumption: does it assume execution at a price that
   was never actually available?
6. Parameter fitting: how many parameters, and were they
   chosen by looking at the whole dataset?
7. Sample: does the test period contain both a bull and a
   bear regime?
8. Data alignment: are all series on the same timezone and
   bar-close convention?

Do not summarize. Quote lines."""

ADVERSARIAL_PROMPT = """You are a quant risk manager whose job is to reject this
strategy. Find every reason it would fail in live trading
that a backtest cannot show. Be specific and hostile.
Assume the author is fooling themselves."""

HYPOTHESIS_PROMPT = """Propose a testable trading hypothesis for {symbol} on the {timeframe}
timeframe. State the ECONOMIC MECHANISM — who is on the
other side and why they lose. If you cannot name the
counterparty, the idea is a pattern, not an edge.
Then write the exact entry, exit, and invalidation rules."""

CRITIC_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "number": {"type": "integer"},
                    "name": {"type": "string"},
                    "status": {"type": "string", "enum": ["PRESENT", "ABSENT", "UNCLEAR"]},
                    "quoted_lines": {"type": "array", "items": {"type": "string"}},
                    "explanation": {"type": "string"},
                },
                "required": ["number", "name", "status", "quoted_lines", "explanation"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["items"],
    "additionalProperties": False,
}

# Items that block the gate if the model finds them PRESENT. 2 and 7 are covered
# by data checks the model cannot see from source alone.
BLOCKING_ITEMS = {1, 3, 4, 5, 6, 8}


def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


def _call(prompt: str, content: str, schema: dict | None = None, effort: str = "high") -> str:
    output_config: dict = {"effort": effort}
    if schema is not None:
        output_config["format"] = {"type": "json_schema", "schema": schema}
    resp = _client().beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        output_config=output_config,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": f"{prompt}\n\n<code>\n{content}\n</code>"}],
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"model declined: {getattr(resp.stop_details, 'category', None)}")
    if resp.stop_reason == "max_tokens":
        raise RuntimeError("model response truncated (max_tokens)")
    return next(b.text for b in resp.content if b.type == "text")


def strategy_source(strategy) -> str:
    from . import engine, sizing, walkforward
    from .strategies import base
    parts = [inspect.getsource(m) for m in (engine, sizing, base, walkforward)]
    parts.append(inspect.getsource(type(strategy)))
    return "\n\n# ---- next file ----\n\n".join(parts)


def llm_critic(strategy) -> dict:
    data = json.loads(_call(CRITIC_PROMPT, strategy_source(strategy), schema=CRITIC_SCHEMA))
    blocking = [i for i in data["items"] if i["number"] in BLOCKING_ITEMS and i["status"] == "PRESENT"]
    return {"model": MODEL, "items": data["items"],
            "blocking": [f'{i["number"]}. {i["name"]}' for i in blocking],
            "passed": not blocking}


def adversarial_review(strategy, report_summary: dict) -> str:
    ctx = (f"Mechanism claimed: {strategy.mechanism}\n\nValidation summary:\n"
           f"{json.dumps(report_summary, indent=2, default=str)}\n\n{strategy_source(strategy)}")
    return _call(ADVERSARIAL_PROMPT, ctx)


def hypothesize(symbol: str, timeframe: str) -> str:
    return _call(HYPOTHESIS_PROMPT.format(symbol=symbol, timeframe=timeframe), "(no code yet)")


# ---------------- hypothesis generation for forward-test rounds ----------------

SPEC_PROMPT = """You are the hypothesis role in a disciplined crypto research loop. Propose ONE
new long-only strategy for {symbol} on {timeframe} bars, built ONLY from the menu below.
State the economic mechanism: who is on the other side and why they lose. Prefer a
mechanism that the results so far have NOT already tested. Do not chase the best
recent day: single-day results are mostly noise, and every idea you add raises the
bar for all of them.

Menu (family -> parameter: allowed values):
{families}

Optional filters (at most 2):
{filters}

Already tried (do not repeat): {taken}

Recent round results (return per 24h round, most recent last):
{history}
"""

SPEC_SCHEMA = {
    "type": "object",
    "properties": {
        "family": {"type": "string", "enum": []},           # filled in at call time
        "params": {"type": "array", "items": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "value": {"type": "number"}},
            "required": ["name", "value"], "additionalProperties": False}},
        "filters": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": []},
                "values": {"type": "array", "items": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "value": {"type": "number"}},
                    "required": ["name", "value"], "additionalProperties": False}}},
            "required": ["type", "values"], "additionalProperties": False}},
        "mechanism": {"type": "string"},
    },
    "required": ["family", "params", "filters", "mechanism"],
    "additionalProperties": False,
}


def propose_spec(history: list[dict], taken: list[str], symbol: str = "BTC/USDT",
                 timeframe: str = "15m") -> dict:
    """Ask Claude for one new spec. The reply can only select menu items, and it is
    validated against the menu again before use; nothing is ever executed."""
    import copy

    from .strategies.generated import FAMILIES, FILTERS, validate_spec
    schema = copy.deepcopy(SPEC_SCHEMA)
    schema["properties"]["family"]["enum"] = sorted(FAMILIES)
    schema["properties"]["filters"]["items"]["properties"]["type"]["enum"] = sorted(FILTERS)
    hist = [{"round": h.get("round"),
             "results": [(r["strategy"], r["round_return_pct"]) for r in h.get("results", [])]}
            for h in history[-10:]]
    prompt = SPEC_PROMPT.format(symbol=symbol, timeframe=timeframe,
                                families=json.dumps(FAMILIES), filters=json.dumps(FILTERS),
                                taken=", ".join(taken[-50:]) or "none", history=json.dumps(hist))
    raw = json.loads(_call(prompt, "(menu only; no code)", schema=schema, effort="medium"))
    spec = {
        "family": raw["family"],
        "params": {p["name"]: p["value"] for p in raw["params"]},
        "filters": [{"type": f["type"], **{v["name"]: v["value"] for v in f["values"]}}
                    for f in raw["filters"]],
        "mechanism": raw["mechanism"],
    }
    return validate_spec(spec)
