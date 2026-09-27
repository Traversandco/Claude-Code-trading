# quantstack

An autonomous crypto trading bot built on the stack from *"The $200,000 Quant Stack
You Can Now Build in an Evening"*, plus the discipline half: **it will not place an
order unless the strategy has passed three hard gates, and it keeps checking.**

> Generation is free now. Validation is the job.

**Expect it to say no.** On pure noise the gates reject every strategy (0 of 12 seeds
in testing, including a noise run with a 1.16 Sharpe and 14/20 positive folds). They
also reject most *genuine* but modest edges, because a real 1.9-Sharpe strategy shows
anywhere from 0.7 to 2.6 over four years of daily data. That's the design: the trader
who rejects 199 of 200 hypotheses is doing the job correctly.

Nothing here is financial advice. Run it on paper, then testnet, then small.

---

## How the article maps to the code

| Article | Module | What it does |
|---|---|---|
| 1. Backtest engine | `engine.py` | `signal.shift(1)`, log returns, fees + slippage on turnover |
| 2. Metrics | `metrics.py` | Sharpe, ann. return, max DD, longest DD, Calmar |
| 3. Critic (8 errors) | `leakage.py` + `llm.py` | Each item is an **executable test**; Claude optionally reviews the source too |
| 4. Multiple testing | `stats.py` + `trials.py` | Deflated Sharpe with an append-only **trial ledger** that counts for you |
| 5. Walk-forward | `walkforward.py` | Fit on train, trade test, roll; stitched OOS equity curve |
| 6. Sizing | `sizing.py` | Article's `position_size`, applied via an ATR stop, identical in backtest and live |
| 7. Prompts | `llm.py` | Hypothesis, critic, adversarial review (Claude, `claude-opus-5`) |
| 8. Production | `health.py` + `runner.py` | Health check, circuit breakers, kill switch, auto re-validation |
| Three gates | `gates.py` | Writes a report hashed against the code; the runner trades only against it |

### The three gates, as enforced

1. **No leakage**: every critic item passes (details below). With `llm_critic: true`,
   Claude must also find no look-ahead, repainting, cost, fill, fitting or alignment
   problem in the source.
2. **Deflated Sharpe > 0.95** on the stitched walk-forward OOS returns, with
   `n_trials` taken from the ledger. The ledger counts every grid point of every
   strategy ever run on that market, so trying more things raises the bar for all of them.
3. **Walk-forward survives**: ≥60% of active folds positive, the worst fold no worse
   than a zero-skill strategy's expected worst fold, and OOS Sharpe ≥ 0.3.

The critic's items, all executable:

| # | Check | How |
|---|---|---|
| 1 | Look-ahead | Scramble all bars after *t*; the signal and exposure at or before *t* must not change. Also verifies the engine shifts. |
| 2 | Survivorship | N/A for a single instrument (flagged for universe selection) |
| 3 | Repainting | Signal computed on history truncated at *t* must equal the full-history signal at *t* |
| 4 | Costs | Fees and slippage are > 0 and applied on turnover |
| 5 | Fill | Slippage must cover the median close→next-open gap in *your* data |
| 6 | Fitting | Grid size bounded; parameters are chosen on train windows only |
| 7 | Regimes | Sample has ≥15% bull **and** ≥15% bear bars (200-MA) |
| 8 | Alignment | UTC, monotonic, unique, regular spacing, consistent OHLC |
| 9 | Sanity | OOS Sharpe > 2 fails: leakage until proven otherwise |

The tests show items 1 and 3 catching a centered moving average, a full-sample
z-score, and a `shift(-1)`.

### Where this deviates from the article's code, and why

- **Deflated Sharpe.** The article compares an *annualized* Sharpe against a
  unit-variance noise benchmark and multiplies by √n_obs, which is dimensionally
  inconsistent. Here everything is per-period with the Bailey & López de Prado
  variance term. The article-compatible `deflated_sharpe()` signature is kept and
  converts its input.
- **Walk-forward.** With `test_days=60`, the article's `metrics()` returns
  `insufficient_data` for every fold (it needs 100 obs), and then `df['sharpe']`
  raises a `KeyError`. Its folds also computed indicators from inside the test window
  with no warm-up. Here, folds use a lower `min_obs` and warm up on prior (known)
  bars. Test exposures are stitched so costs at fold boundaries are real. Folds spent
  flat are not judged, because their "Sharpe" is one exit fee divided by a near-zero std.
- **Worst fold.** A fixed floor (e.g. −1) is statistically unreachable: a 60-bar daily
  fold Sharpe has a standard error of ~2.5. The floor is the expected worst fold of a
  zero-skill strategy.
- **Health check.** "Live Sharpe < 50% of backtest" on 30 bars fires on healthy
  strategies about every other month. The shortfall must now also be statistically
  significant. The drawdown alert is unchanged, and account-level circuit breakers sit
  on top.

---

## Quickstart

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest                          # 60 tests, offline
quantstack demo                 # full pipeline on synthetic noise vs. a planted edge
```

Then against a real market:

```bash
cp config.example.yaml config.yaml    # set exchange, symbol, timeframe, strategy, fees
quantstack fetch                      # download closed bars
quantstack validate                   # three gates → reports/<strategy>_<symbol>_<tf>.json
quantstack validate --llm             # + Claude's 8-point critic (needs ANTHROPIC_API_KEY)
quantstack review                     # hostile risk-manager review of the report
quantstack run                        # autonomous loop (mode from config)
```

### Going live: paper → testnet → live

1. `mode: paper` needs no keys. It trades against live prices with simulated fills
   (fees + slippage), and account state lives in `state/paper_account.json`.
2. `mode: testnet` uses the exchange sandbox via ccxt. Put testnet keys in `.env`
   (see `.env.example`).
   On Bybit, prefer `mode: demo`: demo trading uses real market prices, while
   Bybit's testnet has thin, unrealistic order books. Create the demo API key from
   inside "Demo Trading" on your normal Bybit account.
3. `mode: live` is real money. Use a **dedicated sub-account**, because the bot treats
   the whole base-asset balance as its position. Keys should be trade-only, have **no
   withdrawal permission**, and be IP-whitelisted. Start with a small balance and keep
   `max_order_notional` low.

Run it under systemd with `deploy/quantstack.service`. It restarts on crashes but
stays down after a risk halt (exit 3) or a refusal to start (exit 2).

---

## Forward testing: learning from live markets without fooling yourself

Backtests can be overfit; data that did not exist when the rules were written
cannot. `quantstack forward` runs every candidate side by side, each on its own
**simulated** account at live prices with your real fees and slippage.

```bash
quantstack forward        # runs until stopped; every candidate, every bar
quantstack scoreboard     # returns, Sharpe, drawdown, trades, fees, verdict
```

- **Paper only, always.** Unvalidated strategies never reach the exchange, even with
  `mode: live`. Separate accounts are required anyway: one exchange demo account has
  one balance, so strategies sharing it would corrupt each other's records.
- **Pre-registered.** Each candidate is recorded with a date and a code hash. Edit a
  strategy and its record is retired and a new one starts from zero. Nothing is
  deleted, and every candidate ever registered counts toward the deflated Sharpe.
- **Honest about time.** Until `min_bars`, the verdict is TOO EARLY, and the
  scoreboard shows roughly how many bars the current Sharpe needs before it means
  anything. On daily bars a true Sharpe of 1 needs years, not weeks.
- **It informs; it does not unlock.** A PROMISING candidate still has to pass
  `quantstack validate` before `quantstack run` will trade it.

## What the bot does every bar

```
wakes hourly (or every bar, if shorter); a bar is only ever processed once
 ├─ halted? → do nothing until a human runs `quantstack resume`
 ├─ KILL file? → flatten, halt
 ├─ data stale (> 2 bars old)? → skip, never trade on old data
 ├─ validation report: passed, same symbol/tf/strategy, same code hash, < 30 days old?
 │    └─ expired → re-validate on current data automatically; fail → flatten, halt
 ├─ circuit breakers: daily loss > 3%, drawdown from peak > 15% → flatten, halt
 ├─ health check vs. validated OOS metrics → HALT on decay or DD > 1.5× backtest
 ├─ every test_bars: refit params on the last train_bars (walk-forward, continued live)
 ├─ exposure = signal × ATR risk sizing, capped at 20% of equity
 └─ rebalance if the change exceeds 2% of equity; order size capped per bar
```

The same bar is never processed twice. Only closed candles are used, because trading
on the forming candle is the live version of look-ahead. Five consecutive errors → halt.

```bash
quantstack status     # halted? why? cleared to trade? last equity
quantstack kill       # flatten and halt on the next check
quantstack resume     # your decision, not the bot's
```

Logs are JSON lines in `logs/bot.jsonl`.

---

## Adding a strategy

Every strategy states a mechanism: *who is on the other side and why they lose.*

```python
# quantstack/strategies/my_idea.py
class MyIdea(Strategy):
    name = "my_idea"
    mechanism = "Who pays you, and why they keep paying."
    param_grid = {"lookback": [20, 50]}          # every combo is a trial. Keep it small.

    def signal(self, bars, lookback=20):        # causal: bars up to and including t only
        ...
        return series_in_minus1_to_1
```

Register it in `strategies/__init__.py`, set `strategy: my_idea`, and run `quantstack validate`.

`quantstack hypothesize` asks Claude for a hypothesis with a named counterparty and
saves it to `research/hypotheses/`. **The bot never imports or executes LLM-written
code.** A process holding exchange keys should not run generated code, so a human
turns a hypothesis into a `Strategy`. From there, testing, gating, deployment and
monitoring are autonomous.

### Bundled strategies

| Strategy | Data | Mechanism (who pays you) |
|---|---|---|
| `ts_momentum` | price | Late and forced traders chasing and liquidating into moves |
| `rsi_reversion` | price | Forced sellers paying for immediacy after sharp drops |
| `funding_crowding` | price + perp funding | Over-levered longs liquidated at crowded tops; crowded shorts squeezed |

`funding_crowding` was written and its 4-point grid fixed *before* any real data was
looked at: trend-following long, flat when 7-day average funding exceeds 3x or 5x the
0.01%/8h baseline, long when funding is negative. It fetches the linear perp's funding
history (`BTC/USDT` -> `BTC/USDT:USDT`) automatically; only spot is traded.

## Limitations

- Spot, long-only, one symbol per bot process. Shorting needs a perp/margin adapter
  (`allow_short: true` is rejected until one exists).
- The ATR stop is used to *size* positions. Exits come from the strategy signal and the
  account circuit breakers, not from resting stop orders on the exchange.
- The fill model assumes the signal bar's close plus slippage. Critic item 5 checks
  that slippage covers the observed gap, but thin books or large orders need more.
- The two bundled strategies are examples, not recommendations. Neither is claimed to
  pass on real BTC data.
