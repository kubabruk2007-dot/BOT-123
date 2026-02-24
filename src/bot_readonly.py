import csv
import json
import os
from pathlib import Path

import ccxt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
LOG_FILE = DATA_DIR / "signals_log.csv"
JSONL_LOG_FILE = DATA_DIR / "signals_log.jsonl"
ENV_FILE = PROJECT_ROOT / ".env"

DATA_DIR.mkdir(parents=True, exist_ok=True)

LOG_COLUMNS = [
    "ts",
    "symbol",
    "timeframe",
    "price",
    "ema_fast_period",
    "ema_slow_period",
    "rsi_period",
    "rsi_min",
    "rsi_max",
    "ema_fast",
    "ema_slow",
    "rsi",
    "buy_signal",
]

LOG_SCHEMA = {
    "ts": "Timestamp of the closed candle used for the signal (UTC).",
    "symbol": "Trading pair symbol, e.g. ETH/USDC.",
    "timeframe": "Candle timeframe, e.g. 15m.",
    "price": "Close price of the closed candle.",
    "ema_fast_period": "Period for the fast EMA.",
    "ema_slow_period": "Period for the slow EMA.",
    "rsi_period": "Period for RSI.",
    "rsi_min": "Lower RSI bound for BUY signal.",
    "rsi_max": "Upper RSI bound for BUY signal.",
    "ema_fast": "Computed fast EMA value for the closed candle.",
    "ema_slow": "Computed slow EMA value for the closed candle.",
    "rsi": "Computed RSI value for the closed candle.",
    "buy_signal": "Boolean BUY/NO BUY decision for the closed candle.",
}


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


def make_exchange(exchange_id: str, base_url: str | None) -> ccxt.Exchange:
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

    return exchange


def main() -> None:
    load_env_file(ENV_FILE)

    exchange_id = get_env_str("EXCHANGE", "binance").lower()
    symbol = get_env_str("SYMBOL", "ETH/USDC")
    timeframe = get_env_str("TIMEFRAME", "15m")
    candle_limit = get_env_int("CANDLE_LIMIT", 200)

    ema_fast_period = get_env_int("EMA_FAST_PERIOD", 20)
    ema_slow_period = get_env_int("EMA_SLOW_PERIOD", 50)
    rsi_period = get_env_int("RSI_PERIOD", 14)
    rsi_min = get_env_float("RSI_MIN", 45.0)
    rsi_max = get_env_float("RSI_MAX", 65.0)

    if candle_limit < 3:
        print("ERROR: CANDLE_LIMIT must be >= 3")
        return
    if ema_fast_period <= 0 or ema_slow_period <= 0 or rsi_period <= 0:
        print("ERROR: EMA/RSI periods must be > 0")
        return
    if rsi_min > rsi_max:
        print("ERROR: RSI_MIN cannot be greater than RSI_MAX")
        return

    if exchange_id == "binance":
        base_env = get_env_str(
            "BINANCE_API_BASES",
            "https://api.binance.com,https://api1.binance.com,https://api2.binance.com,https://api3.binance.com",
        )
        bases = [b.strip() for b in base_env.split(",") if b.strip()]
    else:
        bases = [""]

    ohlcv = None
    last_exc = None
    for base in bases:
        try:
            exchange = make_exchange(exchange_id, base or None)
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=candle_limit)
            if base:
                print(f"INFO: using {exchange_id} base {base}")
            break
        except Exception as exc:
            last_exc = exc
            print(f"WARN fetch_ohlcv failed using base={base or 'default'}: {exc}")

    if ohlcv is None:
        print(f"ERROR fetch_ohlcv: {exchange_id} {last_exc}")
        return

    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    if len(df) < 3:
        print("ERROR: Not enough candles to compute closed-candle signal.")
        return

    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df["ema_fast"] = ema(df["close"], ema_fast_period)
    df["ema_slow"] = ema(df["close"], ema_slow_period)
    df["rsi"] = rsi(df["close"], rsi_period)

    signal_row = df.iloc[-2]
    prev_row = df.iloc[-3]

    signal_ts = pd.to_datetime(signal_row["ts"])
    price = float(signal_row["close"])
    ema_fast_value = float(signal_row["ema_fast"])
    ema_slow_value = float(signal_row["ema_slow"])
    rsi_now = float(signal_row["rsi"])
    rsi_prev = float(prev_row["rsi"])

    buy_signal = (
        (ema_fast_value > ema_slow_value)
        and (rsi_now > rsi_prev)
        and (rsi_min <= rsi_now <= rsi_max)
    )

    print(
        f"{symbol} | tf={timeframe} | candle={signal_ts} | price={price:.2f} | "
        f"ema_fast({ema_fast_period})={ema_fast_value:.2f} "
        f"ema_slow({ema_slow_period})={ema_slow_value:.2f} | "
        f"rsi({rsi_period})={rsi_now:.1f}"
    )
    print("BUY SIGNAL =", buy_signal)

    header = ensure_log_schema(LOG_FILE)

    if is_duplicate_candle(LOG_FILE, signal_ts, symbol, timeframe):
        print("Skip save: duplicate candle already present in log.")
        return

    out = pd.DataFrame([
        {
            "ts": signal_ts,
            "symbol": symbol,
            "timeframe": timeframe,
            "price": price,
            "ema_fast_period": ema_fast_period,
            "ema_slow_period": ema_slow_period,
            "rsi_period": rsi_period,
            "rsi_min": rsi_min,
            "rsi_max": rsi_max,
            "ema_fast": ema_fast_value,
            "ema_slow": ema_slow_value,
            "rsi": rsi_now,
            "buy_signal": buy_signal,
        }
    ])

    try:
        out.to_csv(LOG_FILE, mode="a", index=False, header=header, columns=LOG_COLUMNS)
    except Exception as exc:
        print(f"ERROR write_csv: {exc}")
        return

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
        return

    print(f"Saved: {LOG_FILE}")
    print(f"Saved: {JSONL_LOG_FILE}")


if __name__ == "__main__":
    main()
