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
