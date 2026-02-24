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
SYMBOL=ETH/USDC
TIMEFRAME=15m
CANDLE_LIMIT=200
EMA_FAST_PERIOD=20
EMA_SLOW_PERIOD=50
RSI_PERIOD=14
RSI_MIN=45
RSI_MAX=65
```

## Notes
- Script uses last closed candle for signal calculation.
- Duplicate candle entries are skipped in CSV log.
- This is analysis mode only. No trading orders are sent.
