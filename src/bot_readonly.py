import csv
import json
import os
import time
from zoneinfo import ZoneInfo
from pathlib import Path

import ccxt
import pandas as pd

from strategy_rules import DEFAULT_STRATEGY, StrategyConfig, should_buy_signal, should_sell_signal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
LOG_FILE = DATA_DIR / "signals_log.csv"
JSONL_LOG_FILE = DATA_DIR / "signals_log.jsonl"
PORTFOLIO_FILE = DATA_DIR / "paper_portfolio.json"
TRADES_LOG_FILE = DATA_DIR / "trades_log.csv"
ENV_FILE = PROJECT_ROOT / ".env"

DATA_DIR.mkdir(parents=True, exist_ok=True)

LOG_COLUMNS = [
    "ts",
    "symbol",
    "timeframe",
    "price",
    "ema_fast_period",
    "ema_slow_period",
    "ema_trend_period",
    "rsi_period",
    "rsi_min",
    "rsi_max",
    "tp_pct",
    "sl_pct",
    "ema_fast",
    "ema_slow",
    "ema_trend",
    "rsi",
    "buy_signal",
    "position_open",
]

LOG_SCHEMA = {
    "ts": "Timestamp of the closed candle used for the signal (UTC).",
    "symbol": "Trading pair symbol, e.g. ETH/USDC.",
    "timeframe": "Candle timeframe, e.g. 15m.",
    "price": "Close price of the closed candle.",
    "ema_fast_period": "Period for the fast EMA.",
    "ema_slow_period": "Period for the slow EMA.",
    "ema_trend_period": "Period for the trend EMA.",
    "rsi_period": "Period for RSI.",
    "rsi_min": "Lower RSI bound for BUY signal.",
    "rsi_max": "Upper RSI bound for BUY signal.",
    "tp_pct": "Take-profit percent from entry price.",
    "sl_pct": "Stop-loss percent from entry price.",
    "ema_fast": "Computed fast EMA value for the closed candle.",
    "ema_slow": "Computed slow EMA value for the closed candle.",
    "ema_trend": "Computed trend EMA value for the closed candle.",
    "rsi": "Computed RSI value for the closed candle.",
    "buy_signal": "Boolean BUY/NO BUY decision for the closed candle.",
    "position_open": "Whether a paper position is currently open.",
}

TRADE_COLUMNS = [
    "ts",
    "action",
    "symbol",
    "price",
    "base_qty",
    "quote_qty",
    "reason",
    "entry_ts",
    "entry_price",
    "pnl_usdc",
    "pnl_pct",
    "balance_usdc",
    "balance_eth",
]


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def get_env_str(name: str, default: str) -> str:
    value = os.getenv(name, default)
    return value.strip() if isinstance(value, str) else default


def get_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"WARN invalid int for {name}={raw!r}, fallback to {default}")
        return default


def get_env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"WARN invalid float for {name}={raw!r}, fallback to {default}")
        return default


def get_api_credentials() -> tuple[str | None, str | None]:
    api_key = os.getenv("API_KEY")
    api_secret = os.getenv("API_SECRET")
    if api_key:
        api_key = api_key.strip()
    if api_secret:
        api_secret = api_secret.strip()
    return api_key or None, api_secret or None


def get_balance(exchange: ccxt.Exchange, currency: str) -> float:
    try:
        balance = exchange.fetch_balance()
    except Exception as exc:
        print(f"WARN fetch_balance failed: {exc}")
        return 0.0

    if isinstance(balance, dict):
        if currency in balance and isinstance(balance[currency], dict):
            return float(balance[currency].get("free", 0.0) or 0.0)
        free_map = balance.get("free")
        if isinstance(free_map, dict):
            return float(free_map.get(currency, 0.0) or 0.0)
    return 0.0


def amount_to_precision(exchange: ccxt.Exchange, symbol: str, amount: float) -> float:
    try:
        return float(exchange.amount_to_precision(symbol, amount))
    except Exception:
        return float(amount)


def place_market_buy(
    exchange: ccxt.Exchange,
    symbol: str,
    quote_amount: float,
    fallback_price: float,
) -> dict:
    if quote_amount <= 0:
        raise ValueError("quote_amount must be > 0")

    try:
        return exchange.create_order(
            symbol, "market", "buy", None, None, {"quoteOrderQty": quote_amount}
        )
    except Exception:
        price = fallback_price
        try:
            ticker = exchange.fetch_ticker(symbol)
            price = float(ticker.get("last") or price)
        except Exception:
            pass

        base_amount = quote_amount / price if price > 0 else 0.0
        base_amount = amount_to_precision(exchange, symbol, base_amount)
        return exchange.create_order(symbol, "market", "buy", base_amount)


def place_market_sell(
    exchange: ccxt.Exchange,
    symbol: str,
    base_amount: float,
) -> dict:
    if base_amount <= 0:
        raise ValueError("base_amount must be > 0")
    base_amount = amount_to_precision(exchange, symbol, base_amount)
    return exchange.create_order(symbol, "market", "sell", base_amount)


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, pd.NA)
    out = 100 - (100 / (1 + rs))
    out = out.where(avg_loss != 0, 100)
    out = out.where(~((avg_loss == 0) & (avg_gain == 0)), 50)
    return out


def ensure_log_schema(log_file: Path) -> bool:
    if not log_file.exists():
        return True

    try:
        first_line = log_file.read_text(encoding="utf-8").splitlines()[0].strip()
    except Exception as exc:
        print(f"WARN cannot read log header: {exc}")
        return False

    expected = ",".join(LOG_COLUMNS)
    if first_line == expected:
        return False

    backup = log_file.with_name(
        f"{log_file.stem}_legacy_{pd.Timestamp.utcnow():%Y%m%d_%H%M%S}{log_file.suffix}"
    )
    log_file.rename(backup)
    print(f"WARN log schema mismatch, moved old log to: {backup}")
    return True


def ensure_trades_schema(log_file: Path) -> bool:
    if not log_file.exists():
        return True

    try:
        first_line = log_file.read_text(encoding="utf-8").splitlines()[0].strip()
    except Exception as exc:
        print(f"WARN cannot read trades log header: {exc}")
        return False

    expected = ",".join(TRADE_COLUMNS)
    if first_line == expected:
        return False

    backup = log_file.with_name(
        f"{log_file.stem}_legacy_{pd.Timestamp.utcnow():%Y%m%d_%H%M%S}{log_file.suffix}"
    )
    log_file.rename(backup)
    print(f"WARN trades log schema mismatch, moved old log to: {backup}")
    return True


def load_portfolio() -> dict:
    if PORTFOLIO_FILE.exists():
        try:
            return json.loads(PORTFOLIO_FILE.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"WARN cannot read portfolio, resetting: {exc}")

    return {
        "base": "ETH",
        "quote": "USDC",
        "position": {
            "is_open": False,
            "entry_price": None,
            "entry_ts": None,
            "entry_qty": None,
            "pending_sell": False,
            "pending_reason": None,
        },
    }


def save_portfolio(state: dict) -> None:
    try:
        PORTFOLIO_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"WARN cannot save portfolio: {exc}")


def log_trade(entry: dict) -> None:
    header = ensure_trades_schema(TRADES_LOG_FILE)
    try:
        pd.DataFrame([entry]).to_csv(
            TRADES_LOG_FILE, mode="a", index=False, header=header, columns=TRADE_COLUMNS
        )
    except Exception as exc:
        print(f"WARN cannot write trades log: {exc}")


def is_duplicate_candle(log_file: Path, ts: pd.Timestamp, symbol: str, timeframe: str) -> bool:
    if not log_file.exists():
        return False

    try:
        with log_file.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception as exc:
        print(f"WARN duplicate check failed: {exc}")
        return False

    if not rows:
        return False

    last = rows[-1]
    last_ts = pd.to_datetime(last.get("ts"), errors="coerce")
    last_symbol = str(last.get("symbol", ""))
    last_timeframe = str(last.get("timeframe", ""))

    if pd.isna(last_ts):
        return False

    return (last_ts == ts) and (last_symbol == symbol) and (last_timeframe == timeframe)


def make_exchange(
    exchange_id: str,
    base_url: str | None,
    api_key: str | None = None,
    api_secret: str | None = None,
    demo_mode: bool = False,
) -> ccxt.Exchange:
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })

    if base_url and exchange_id == "binance":
        base_url = base_url.rstrip("/")
        urls = exchange.urls.copy()
        api_urls = urls.get("api", {})
        if isinstance(api_urls, dict):
            api_urls = api_urls.copy()
            api_urls["public"] = f"{base_url}/api/v3"
            api_urls["private"] = f"{base_url}/api/v3"
            urls["api"] = api_urls
        else:
            urls["api"] = f"{base_url}/api/v3"
        exchange.urls = urls

    if api_key and api_secret:
        exchange.apiKey = api_key
        exchange.secret = api_secret
        if demo_mode and hasattr(exchange, "set_sandbox_mode"):
            exchange.set_sandbox_mode(True)

    return exchange


def main() -> None:
    load_env_file(ENV_FILE)

    exchange_id = get_env_str("EXCHANGE", "binance").lower()
    execution_mode = get_env_str("EXECUTION_MODE", "paper").lower()
    demo_mode = get_env_int("DEMO_MODE", 1) == 1
    symbol = get_env_str("SYMBOL", "ETH/USDC")
    timeframe = get_env_str("TIMEFRAME", "15m")
    local_tz = get_env_str("LOCAL_TZ", "Europe/Warsaw")
    candle_limit = get_env_int("CANDLE_LIMIT", 200)
    run_loop = get_env_int("RUN_LOOP", 0) == 1
    sleep_seconds = get_env_int("SLEEP_SECONDS", 0)

    ema_fast_period = get_env_int("EMA_FAST_PERIOD", DEFAULT_STRATEGY.ema_fast_period)
    ema_slow_period = get_env_int("EMA_SLOW_PERIOD", DEFAULT_STRATEGY.ema_slow_period)
    ema_trend_period = get_env_int("EMA_TREND_PERIOD", DEFAULT_STRATEGY.ema_trend_period)
    rsi_period = get_env_int("RSI_PERIOD", DEFAULT_STRATEGY.rsi_period)
    rsi_min = get_env_float("RSI_MIN", DEFAULT_STRATEGY.rsi_min)
    rsi_max = get_env_float("RSI_MAX", DEFAULT_STRATEGY.rsi_max)
    rsi_sell_high = get_env_float("RSI_SELL_HIGH", DEFAULT_STRATEGY.rsi_sell_high)
    rsi_sell_low = get_env_float("RSI_SELL_LOW", DEFAULT_STRATEGY.rsi_sell_low)
    enable_rsi_sell = get_env_int("ENABLE_RSI_SELL", int(DEFAULT_STRATEGY.enable_rsi_sell)) == 1
    force_buy = get_env_int("FORCE_BUY", int(DEFAULT_STRATEGY.force_buy)) == 1
    disable_trend_filter = (
        get_env_int("DISABLE_TREND_FILTER", int(DEFAULT_STRATEGY.disable_trend_filter)) == 1
    )
    disable_pullback_filter = (
        get_env_int("DISABLE_PULLBACK_FILTER", int(DEFAULT_STRATEGY.disable_pullback_filter)) == 1
    )
    rsi_only_sell = get_env_int("RSI_ONLY_SELL", int(DEFAULT_STRATEGY.rsi_only_sell)) == 1
    min_position_eth = get_env_float("MIN_POSITION_ETH", 0.000001)
    tp_pct = get_env_float("TP_PCT", DEFAULT_STRATEGY.tp_pct)
    sl_pct = get_env_float("SL_PCT", DEFAULT_STRATEGY.sl_pct)
    use_full_balance = get_env_int("USE_FULL_BALANCE", 0) == 1
    quote_per_trade = get_env_float("QUOTE_PER_TRADE", 5.0)
    min_usdc_to_trade = get_env_float("MIN_USDC_TO_TRADE", quote_per_trade)
    strategy = StrategyConfig(
        ema_fast_period=ema_fast_period,
        ema_slow_period=ema_slow_period,
        ema_trend_period=ema_trend_period,
        rsi_period=rsi_period,
        rsi_min=rsi_min,
        rsi_max=rsi_max,
        rsi_sell_high=rsi_sell_high,
        rsi_sell_low=rsi_sell_low,
        enable_rsi_sell=enable_rsi_sell,
        force_buy=force_buy,
        disable_trend_filter=disable_trend_filter,
        disable_pullback_filter=disable_pullback_filter,
        rsi_only_sell=rsi_only_sell,
        tp_pct=tp_pct,
        sl_pct=sl_pct,
    )
    portfolio = load_portfolio()
    api_key, api_secret = get_api_credentials()

    if execution_mode == "paper":
        print("ERROR: Paper mode is disabled. Set EXECUTION_MODE=live.")
        return

    if api_key and api_secret:
        print("INFO: API credentials loaded.")
    elif api_key or api_secret:
        print("WARN: API credentials incomplete (need both API_KEY and API_SECRET).")
    else:
        print("ERROR: Missing API credentials for live/demo mode.")
        return

    if candle_limit < 3:
        print("ERROR: CANDLE_LIMIT must be >= 3")
        return
    if ema_fast_period <= 0 or ema_slow_period <= 0 or ema_trend_period <= 0 or rsi_period <= 0:
        print("ERROR: EMA/RSI periods must be > 0")
        return
    if strategy.rsi_min > strategy.rsi_max:
        print("ERROR: RSI_MIN cannot be greater than RSI_MAX")
        return
    if strategy.tp_pct <= 0 or strategy.sl_pct <= 0:
        print("ERROR: TP_PCT and SL_PCT must be > 0")
        return

    if exchange_id == "binance":
        if demo_mode:
            base_env = get_env_str(
                "BINANCE_TESTNET_BASES",
                "https://testnet.binance.vision",
            )
        else:
            base_env = get_env_str(
                "BINANCE_API_BASES",
                "https://api.binance.com,https://api1.binance.com,https://api2.binance.com,https://api3.binance.com",
            )
        bases = [b.strip() for b in base_env.split(",") if b.strip()]
    else:
        bases = [""]

    def infer_sleep_seconds() -> int:
        if sleep_seconds > 0:
            return sleep_seconds
        tf = timeframe.strip().lower()
        if tf.endswith("m") and tf[:-1].isdigit():
            return max(60, int(tf[:-1]) * 60)
        if tf.endswith("h") and tf[:-1].isdigit():
            return max(60, int(tf[:-1]) * 3600)
        if tf.endswith("d") and tf[:-1].isdigit():
            return max(60, int(tf[:-1]) * 86400)
        return 60

    loop_sleep = infer_sleep_seconds()

    try:
        while True:
            ohlcv = None
            last_exc = None
            for base in bases:
                try:
                    exchange = make_exchange(
                        exchange_id,
                        base or None,
                        api_key=api_key,
                        api_secret=api_secret,
                        demo_mode=demo_mode,
                    )
                    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=candle_limit)
                    if base:
                        print(f"INFO: using {exchange_id} base {base}")
                    break
                except Exception as exc:
                    last_exc = exc
                    print(f"WARN fetch_ohlcv failed using base={base or 'default'}: {exc}")

            if ohlcv is None:
                print(f"ERROR fetch_ohlcv: {exchange_id} {last_exc}")
            else:
                df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
                if len(df) < 3:
                    print("ERROR: Not enough candles to compute closed-candle signal.")
                else:
                    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
                    df["ema_fast"] = ema(df["close"], ema_fast_period)
                    df["ema_slow"] = ema(df["close"], ema_slow_period)
                    df["ema_trend"] = ema(df["close"], ema_trend_period)
                    df["rsi"] = rsi(df["close"], rsi_period)

                    live_row = df.iloc[-1]
                    signal_row = df.iloc[-2]
                    prev_row = df.iloc[-3]

                    signal_ts = pd.to_datetime(signal_row["ts"])
                    if signal_ts.tzinfo is None:
                        signal_ts = signal_ts.tz_localize("UTC")
                    try:
                        signal_ts_local = signal_ts.tz_convert(ZoneInfo(local_tz))
                    except Exception:
                        signal_ts_local = signal_ts
                    price = float(signal_row["close"])
                    ema_fast_value = float(signal_row["ema_fast"])
                    ema_slow_value = float(signal_row["ema_slow"])
                    ema_trend_value = float(signal_row["ema_trend"])
                    rsi_now = float(signal_row["rsi"])
                    rsi_prev = float(prev_row["rsi"])
                    signal_ts_iso = signal_ts.isoformat()

                    buy_signal = should_buy_signal(
                        ema_fast_value=ema_fast_value,
                        ema_slow_value=ema_slow_value,
                        ema_trend_value=ema_trend_value,
                        rsi_now=rsi_now,
                        rsi_prev=rsi_prev,
                        config=strategy,
                    )

                    bal_usdc = get_balance(exchange, "USDC")
                    bal_eth = get_balance(exchange, "ETH")
                    print(f"BALANCE | USDC={bal_usdc:.2f} | ETH={bal_eth:.6f}")
                    print(
                        f"{symbol} | tf={timeframe} | candle={signal_ts_local} | "
                        f"price={price:.2f} | "
                        f"ema_fast({ema_fast_period})={ema_fast_value:.2f} "
                        f"ema_slow({ema_slow_period})={ema_slow_value:.2f} "
                        f"ema_trend({ema_trend_period})={ema_trend_value:.2f} | "
                        f"rsi({rsi_period})={rsi_now:.1f}"
                    )
                    print("BUY SIGNAL =", buy_signal)

                    position = portfolio.get("position", {})
                    position_open = bool(position.get("is_open"))
                    is_new_signal = portfolio.get("last_signal_ts") != signal_ts_iso
                    eth_balance = get_balance(exchange, "ETH")
                    if (not position_open) and eth_balance > min_position_eth:
                        position_open = True
                        portfolio["position"] = {
                            "is_open": True,
                            "entry_price": price,
                            "entry_ts": signal_ts_iso,
                            "entry_qty": eth_balance,
                            "pending_sell": False,
                            "pending_reason": None,
                        }
                        save_portfolio(portfolio)

                    sell_signal = False
                    sell_reason = None
                    sell_price = None

                    if position_open:
                        entry_price = float(position.get("entry_price") or 0)
                        entry_ts = position.get("entry_ts")
                        entry_qty = float(position.get("entry_qty") or 0)
                        live_high = float(live_row["high"])
                        live_low = float(live_row["low"])
                        sell_signal, sell_reason, sell_price = should_sell_signal(
                            position=position,
                            live_high=live_high,
                            live_low=live_low,
                            price=price,
                            ema_slow_value=ema_slow_value,
                            ema_trend_value=ema_trend_value,
                            rsi_now=rsi_now,
                            is_new_signal=is_new_signal,
                            config=strategy,
                        )

                        if sell_reason and entry_price > 0 and entry_qty > 0:
                            available_eth = get_balance(exchange, "ETH")

                            if available_eth + 1e-12 < entry_qty:
                                print("WARN: SELL skipped (insufficient ETH balance).")
                                portfolio["position"]["pending_sell"] = False
                                portfolio["position"]["pending_reason"] = None
                                save_portfolio(portfolio)
                            else:
                                portfolio["position"]["pending_sell"] = False
                                portfolio["position"]["pending_reason"] = None

                                try:
                                    order = place_market_sell(exchange, symbol, entry_qty)
                                    filled_qty = float(order.get("filled") or entry_qty)
                                    avg_price = float(order.get("average") or sell_price)
                                    quote_qty = float(order.get("cost") or (filled_qty * avg_price))
                                    pnl_usdc = quote_qty - (filled_qty * entry_price)
                                    pnl_pct = (pnl_usdc / (filled_qty * entry_price)) * 100 if filled_qty > 0 else 0.0
                                except Exception as exc:
                                    print(f"ERROR live SELL failed: {exc}")
                                    portfolio["position"]["pending_sell"] = True
                                    portfolio["position"]["pending_reason"] = sell_reason
                                    save_portfolio(portfolio)
                                    quote_qty = 0.0
                                    pnl_usdc = 0.0
                                    pnl_pct = 0.0
                                    avg_price = sell_price
                                    filled_qty = 0.0

                                if filled_qty > 0:
                                    portfolio["position"] = {
                                        "is_open": False,
                                        "entry_price": None,
                                        "entry_ts": None,
                                        "entry_qty": None,
                                    }

                                log_trade({
                                    "ts": pd.Timestamp.utcnow().isoformat(),
                                    "action": "SELL",
                                    "symbol": symbol,
                                    "price": avg_price,
                                    "base_qty": filled_qty,
                                    "quote_qty": quote_qty,
                                    "reason": sell_reason,
                                    "entry_ts": entry_ts,
                                    "entry_price": entry_price,
                                    "pnl_usdc": pnl_usdc,
                                    "pnl_pct": pnl_pct,
                                    "balance_usdc": get_balance(exchange, "USDC"),
                                    "balance_eth": get_balance(exchange, "ETH"),
                                })
                                save_portfolio(portfolio)
                                position_open = filled_qty <= 0

                    print("SELL SIGNAL =", sell_signal, f"(reason={sell_reason})")

                    if (not position_open) and is_new_signal and buy_signal:
                        usdc_balance = get_balance(exchange, "USDC")

                        if use_full_balance:
                            quote_to_spend = usdc_balance
                        else:
                            quote_to_spend = min(usdc_balance, max(0.0, quote_per_trade))

                        if usdc_balance < min_usdc_to_trade or quote_to_spend <= 0:
                            print("WARN: BUY skipped (insufficient USDC balance).")
                        else:
                            try:
                                order = place_market_buy(
                                    exchange, symbol, quote_to_spend, fallback_price=price
                                )
                                entry_qty_used = float(order.get("filled") or 0.0)
                                entry_price_used = float(order.get("average") or price)
                            except Exception as exc:
                                print(f"ERROR live BUY failed: {exc}")
                                entry_qty_used = 0.0
                                entry_price_used = price

                            if entry_qty_used > 0:
                                portfolio["position"] = {
                                    "is_open": True,
                                    "entry_price": entry_price_used,
                                    "entry_ts": signal_ts_iso,
                                    "entry_qty": entry_qty_used,
                                }

                            log_trade({
                                "ts": pd.Timestamp.utcnow().isoformat(),
                                "action": "BUY",
                                "symbol": symbol,
                                "price": entry_price_used,
                                "base_qty": entry_qty_used,
                                "quote_qty": quote_to_spend if entry_qty_used > 0 else 0.0,
                                "reason": "signal_buy",
                                "entry_ts": signal_ts_iso,
                                "entry_price": entry_price_used,
                                "pnl_usdc": 0.0,
                                "pnl_pct": 0.0,
                                "balance_usdc": get_balance(exchange, "USDC"),
                                "balance_eth": get_balance(exchange, "ETH"),
                            })
                            save_portfolio(portfolio)
                            position_open = entry_qty_used > 0

                    if is_new_signal:
                        portfolio["last_signal_ts"] = signal_ts_iso
                        save_portfolio(portfolio)

                    header = ensure_log_schema(LOG_FILE)

                    if is_duplicate_candle(LOG_FILE, signal_ts, symbol, timeframe):
                        print("Skip save: duplicate candle already present in log.")
                    else:
                        out = pd.DataFrame([
                            {
                                "ts": signal_ts,
                                "symbol": symbol,
                                "timeframe": timeframe,
                                "price": price,
                                "ema_fast_period": ema_fast_period,
                                "ema_slow_period": ema_slow_period,
                                "ema_trend_period": ema_trend_period,
                                "rsi_period": rsi_period,
                                "rsi_min": rsi_min,
                                "rsi_max": rsi_max,
                                "tp_pct": tp_pct,
                                "sl_pct": sl_pct,
                                "ema_fast": ema_fast_value,
                                "ema_slow": ema_slow_value,
                                "ema_trend": ema_trend_value,
                                "rsi": rsi_now,
                                "buy_signal": buy_signal,
                                "position_open": position_open,
                            }
                        ])

                        try:
                            out.to_csv(LOG_FILE, mode="a", index=False, header=header, columns=LOG_COLUMNS)
                        except Exception as exc:
                            print(f"ERROR write_csv: {exc}")
                        else:
                            jsonl_data = out.iloc[0].to_dict()
                            if isinstance(jsonl_data.get("ts"), pd.Timestamp):
                                jsonl_data["ts"] = jsonl_data["ts"].isoformat()

                            jsonl_payload = {
                                "schema": LOG_SCHEMA,
                                "data": jsonl_data,
                            }
                            try:
                                with JSONL_LOG_FILE.open("a", encoding="utf-8") as f:
                                    f.write(json.dumps(jsonl_payload, ensure_ascii=True) + "\n")
                            except Exception as exc:
                                print(f"ERROR write_jsonl: {exc}")
                            else:
                                print(f"Saved: {LOG_FILE}")
                                print(f"Saved: {JSONL_LOG_FILE}")

            if not run_loop:
                break

            print(f"INFO: sleeping {loop_sleep}s...")
            time.sleep(loop_sleep)
    except KeyboardInterrupt:
        print("INFO: stopped by user.")


if __name__ == "__main__":
    main()
