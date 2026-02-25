# Spot ETH/USDC Bot (MVP)

Local SPOT bot for tests on small amounts.
Current scope: Stage 1 (read-only), no real orders.

## What Stage 1 does
- fetches candles from Binance Spot (via ccxt)
- computes EMA and RSI indicators
- evaluates BUY/NO BUY signal
- appends result to `data/signals_log.csv` and `data/signals_log.jsonl`

## Project structure
- `src/bot_readonly.py` main script (read-only)
- `data/signals_log.csv` history of signal snapshots (compact)
- `data/signals_log.jsonl` history with per-field descriptions (readable)
- `.env` local strategy config
- `.env.example` template config
- `run.ps1` one-command runner

## Quick start (PowerShell)
```powershell
.\run.ps1 -Install
.\run.ps1
```

## Configuration (.env)
```env
EXCHANGE=binance
BINANCE_API_BASES=https://api.binance.com,https://api1.binance.com,https://api2.binance.com,https://api3.binance.com
EXECUTION_MODE=live
DEMO_MODE=1
API_KEY=
API_SECRET=
SYMBOL=ETH/USDC
TIMEFRAME=5m
LOCAL_TZ=Europe/Warsaw
CANDLE_LIMIT=200
EMA_FAST_PERIOD=20
EMA_SLOW_PERIOD=50
EMA_TREND_PERIOD=200
RSI_PERIOD=14
RSI_MIN=40
RSI_MAX=55
RSI_SELL_HIGH=70
RSI_SELL_LOW=35
ENABLE_RSI_SELL=1
FORCE_BUY=0
DISABLE_TREND_FILTER=0
DISABLE_PULLBACK_FILTER=0
RSI_ONLY_SELL=0
MIN_POSITION_ETH=0.002
TP_PCT=1.2
SL_PCT=0.7
USE_FULL_BALANCE=0
QUOTE_PER_TRADE=5
MIN_USDC_TO_TRADE=5
RUN_LOOP=0
SLEEP_SECONDS=300
```

## Notes
- Script uses last closed candle for signal calculation.
- Duplicate candle entries are skipped in CSV log.
- This is analysis mode only. No trading orders are sent.
- Set RUN_LOOP=1 to run continuously. SLEEP_SECONDS overrides auto sleep based on timeframe.
- Live position state is stored in `data/paper_portfolio.json`.
- Trades are logged to `data/trades_log.csv`.
- Set DEMO_MODE=1 to use Binance sandbox (requires API_KEY/API_SECRET).

## Strategy 2.0 Engine (30m + 4h)
- New engine: `src/engine.py`.
- Default TF is `30m`, HTF trend filter is `4h` with no-lookahead (HTF features shifted by one HTF candle).
- Market regime with hysteresis:
  - `trend_bull` if EMA50_4h > EMA200_4h
  - `trend_bear` if EMA50_4h < EMA200_4h
  - `trend_flat` near crossover
  - switch requires `REGIME_SWITCH_CONFIRM` candles and respects `REGIME_MIN_HOLD`
- Entry (long-only, SPOT): compression + breakout
  - ATR(14) < `COMP_RATIO` * SMA(ATR(14), 50)
  - close(signal) > HH20_prev
  - HTF bull filter required
- Exit:
  - ATR-based `SL_MULT` / `TP_MULT`
  - Breakeven after `BREAKEVEN_AFTER_R`
  - `TIE_POLICY=SL_FIRST|TP_FIRST` when TP and SL hit inside the same live candle
- SELL is evaluated before BUY.
- BUY/SELL both use min notional checks (`market.limits.cost.min`) with buffer (`NOTIONAL_BUFFER`) to prevent NOTIONAL errors.

## Minimal paper backtest
Use historical candles fetched by ccxt and simulate fills with fees/slippage:

```powershell
.\.venv\Scripts\python.exe src\engine.py
```

Example `.env` values:
- `EXECUTION_MODE=paper`
- `TIMEFRAME=30m`
- `HTF_TIMEFRAME=4h`
- `CANDLE_LIMIT=800`
- `STAKE_PCT=1.0`
- `FEE_RATE=0.001`
- `SLIPPAGE_RATE=0.0002`
