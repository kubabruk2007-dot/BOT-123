# PLAN (krok po kroku)

## Etap 1 — Read-only (bez API)
1. środowisko Python + venv
2. pobieranie świec z Binance (ccxt)
3. wskaźniki: EMA20/EMA50, RSI14
4. sygnał BUY/NO BUY
5. log do CSV w /data

## Etap 2 — Real micro (z API)
1. API key: tylko SPOT trading, withdrawals OFF
2. odczyt salda + walidacja minimalnej kwoty
3. order BUY za stałą kwotę USDC
4. logika SELL: TP/SL (np. ATR albo procent)
5. zabezpieczenia:
   - max 1 pozycja naraz
   - max transakcji dziennie
   - kill switch po stracie dziennej

## Etap 3 — Ulepszenia
- reżimy rynku (trend/range)
- więcej par
- lepsza egzekucja (limit orders)
- statystyki i raporty
