from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import ccxt
import pandas as pd

from indicators import adx, atr, ema, highest_high, sma
from strategy_rules import (
    DEFAULT_STRATEGY,
    REGIME_FLAT,
    StrategyConfig,
    build_position_after_entry,
    detect_regime_from_htf,
    should_buy_signal,
    should_sell_signal,
    update_regime_hysteresis,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PORTFOLIO_FILE = DATA_DIR / "paper_portfolio.json"
TRADES_LOG_FILE = DATA_DIR / "trades_log.csv"
ENV_FILE = PROJECT_ROOT / ".env"
DATA_DIR.mkdir(parents=True, exist_ok=True)

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
    "regime",
    "mode",
]


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


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
        f"{log_file.stem}_legacy_{pd.Timestamp.now('UTC'):%Y%m%d_%H%M%S}{log_file.suffix}"
    )
    log_file.rename(backup)
    print(f"WARN trades log schema mismatch, moved old log to: {backup}")
    return True


def log_trade(entry: dict[str, Any]) -> None:
    header = ensure_trades_schema(TRADES_LOG_FILE)
    try:
        pd.DataFrame([entry]).to_csv(
            TRADES_LOG_FILE,
            mode="a",
            index=False,
            header=header,
            columns=TRADE_COLUMNS,
        )
    except Exception as exc:
        print(f"WARN cannot write trades log: {exc}")


def load_portfolio(initial_usdc: float) -> dict[str, Any]:
    if PORTFOLIO_FILE.exists():
        try:
            state = json.loads(PORTFOLIO_FILE.read_text(encoding="utf-8"))
        except Exception:
            state = {}
    else:
        state = {}

    balances = state.get("balances", {})
    if not isinstance(balances, dict):
        balances = {}

    return {
        "base": "ETH",
        "quote": "USDC",
        "balances": {
            "USDC": float(balances.get("USDC", initial_usdc) or initial_usdc),
            "ETH": float(balances.get("ETH", 0.0) or 0.0),
        },
        "position": state.get("position")
        or {
            "is_open": False,
            "entry_price": None,
            "entry_ts": None,
            "entry_qty": None,
            "pending_sell": False,
            "pending_reason": None,
        },
        "regime_state": state.get("regime_state")
        or {
            "active": REGIME_FLAT,
            "candidate": REGIME_FLAT,
            "candidate_count": 0,
            "hold_count": 0,
        },
        "last_signal_ts": state.get("last_signal_ts"),
    }


def save_portfolio(state: dict[str, Any]) -> None:
    try:
        PORTFOLIO_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"WARN cannot save portfolio: {exc}")


def timeframe_to_minutes(tf: str) -> int:
    tf = tf.strip().lower()
    if tf.endswith("m") and tf[:-1].isdigit():
        return int(tf[:-1])
    if tf.endswith("h") and tf[:-1].isdigit():
        return int(tf[:-1]) * 60
    if tf.endswith("d") and tf[:-1].isdigit():
        return int(tf[:-1]) * 24 * 60
    return 1


def make_exchange(
    exchange_id: str,
    base_url: str | None,
    api_key: str | None = None,
    api_secret: str | None = None,
    demo_mode: bool = False,
) -> ccxt.Exchange:
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({"enableRateLimit": True, "options": {"defaultType": "spot"}})

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

def fetch_ohlcv_data(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    htf_timeframe: str,
    candle_limit: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ltf_raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=candle_limit)
    if not ltf_raw:
        raise RuntimeError("No LTF candles returned.")

    ltf_minutes = max(1, timeframe_to_minutes(timeframe))
    htf_minutes = max(1, timeframe_to_minutes(htf_timeframe))
    htf_limit = max(260, int(math.ceil(candle_limit * ltf_minutes / htf_minutes)) + 50)
    htf_raw = exchange.fetch_ohlcv(symbol, timeframe=htf_timeframe, limit=htf_limit)
    if not htf_raw:
        raise RuntimeError("No HTF candles returned.")

    cols = ["ts", "open", "high", "low", "close", "volume"]
    df_ltf = pd.DataFrame(ltf_raw, columns=cols)
    df_htf = pd.DataFrame(htf_raw, columns=cols)
    df_ltf["ts"] = pd.to_datetime(df_ltf["ts"], unit="ms", utc=True)
    df_htf["ts"] = pd.to_datetime(df_htf["ts"], unit="ms", utc=True)
    return df_ltf, df_htf


def build_feature_frame(
    df_ltf: pd.DataFrame,
    df_htf: pd.DataFrame,
    strategy: StrategyConfig,
) -> pd.DataFrame:
    ltf = df_ltf.copy().sort_values("ts")
    htf = df_htf.copy().sort_values("ts")

    ltf["atr14"] = atr(ltf, 14)
    ltf["atr_sma50"] = sma(ltf["atr14"], 50)
    ltf["hh20_prev"] = highest_high(ltf, 20)
    ltf["adx14"] = adx(ltf, strategy.adx_period)

    htf["ema50_htf"] = ema(htf["close"], 50)
    htf["ema200_htf"] = ema(htf["close"], 200)
    htf["raw_regime_now"] = [
        detect_regime_from_htf(
            float(e50) if pd.notna(e50) else None,
            float(e200) if pd.notna(e200) else None,
            strategy.htf_flat_band_pct,
        )
        for e50, e200 in zip(htf["ema50_htf"], htf["ema200_htf"], strict=False)
    ]

    htf_features = htf[["ts", "ema50_htf", "ema200_htf", "raw_regime_now"]].copy()
    htf_features["ema50_htf"] = htf_features["ema50_htf"].shift(1)
    htf_features["ema200_htf"] = htf_features["ema200_htf"].shift(1)
    htf_features["raw_regime"] = htf_features["raw_regime_now"].shift(1)
    htf_features = htf_features.drop(columns=["raw_regime_now"])

    merged = pd.merge_asof(
        ltf.sort_values("ts"),
        htf_features.sort_values("ts"),
        on="ts",
        direction="backward",
    )
    merged["raw_regime"] = merged["raw_regime"].fillna(REGIME_FLAT)
    return merged


def get_balance(exchange: ccxt.Exchange, currency: str) -> float:
    try:
        balance = exchange.fetch_balance()
    except Exception as exc:
        print(f"WARN fetch_balance failed: {exc}")
        return 0.0
    if isinstance(balance, dict):
        cur = balance.get(currency)
        if isinstance(cur, dict):
            return float(cur.get("free", 0.0) or 0.0)
        free_map = balance.get("free")
        if isinstance(free_map, dict):
            return float(free_map.get(currency, 0.0) or 0.0)
    return 0.0


def amount_to_precision(exchange: ccxt.Exchange, symbol: str, amount: float) -> float:
    try:
        return float(exchange.amount_to_precision(symbol, amount))
    except Exception:
        return float(amount)


def get_market_min_notional(exchange: ccxt.Exchange, symbol: str) -> float:
    try:
        market = exchange.market(symbol)
    except Exception:
        return 0.0
    limits = market.get("limits") if isinstance(market, dict) else None
    if isinstance(limits, dict):
        cost = limits.get("cost")
        if isinstance(cost, dict) and cost.get("min") is not None:
            try:
                return float(cost["min"])
            except Exception:
                return 0.0
    return 0.0


def place_market_buy(exchange: ccxt.Exchange, symbol: str, quote_amount: float, fallback_price: float) -> dict[str, Any]:
    if quote_amount <= 0:
        raise ValueError("quote_amount must be > 0")
    try:
        return exchange.create_order(symbol, "market", "buy", None, None, {"quoteOrderQty": quote_amount})
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


def place_market_sell(exchange: ccxt.Exchange, symbol: str, base_amount: float) -> dict[str, Any]:
    if base_amount <= 0:
        raise ValueError("base_amount must be > 0")
    base_amount = amount_to_precision(exchange, symbol, base_amount)
    return exchange.create_order(symbol, "market", "sell", base_amount)


def notional_ok(notional: float, min_notional: float) -> bool:
    if notional <= 0:
        return False
    if min_notional <= 0:
        return True
    return notional >= min_notional


def close_position_state() -> dict[str, Any]:
    return {
        "is_open": False,
        "entry_price": None,
        "entry_ts": None,
        "entry_qty": None,
        "pending_sell": False,
        "pending_reason": None,
    }


def apply_paper_fill(price: float, side: str, slippage_rate: float) -> float:
    if side == "buy":
        return price * (1 + slippage_rate)
    return price * (1 - slippage_rate)


def run_paper_backtest(
    df: pd.DataFrame,
    symbol: str,
    strategy: StrategyConfig,
    portfolio: dict[str, Any],
    stake_pct: float,
    quote_per_trade: float,
    min_usdc_to_trade: float,
    buy_min_notional: float,
    sell_min_notional: float,
    min_position_eth: float,
    fee_rate: float,
    slippage_rate: float,
    close_eot: bool,
) -> None:
    if len(df) < 3:
        print("ERROR: Not enough candles for paper backtest.")
        return

    mode = "paper"
    for i in range(2, len(df)):
        signal_row = df.iloc[i - 1]
        live_row = df.iloc[i]
        signal_ts_iso = pd.Timestamp(signal_row["ts"]).isoformat()
        price = float(signal_row["close"])

        raw_regime = str(signal_row.get("raw_regime", REGIME_FLAT))
        portfolio["regime_state"] = update_regime_hysteresis(
            raw_regime=raw_regime,
            regime_state=portfolio.get("regime_state"),
            switch_confirm=strategy.regime_switch_confirm,
            min_hold=strategy.regime_min_hold,
        )
        active_regime = str(portfolio["regime_state"].get("active", REGIME_FLAT))

        position = dict(portfolio.get("position") or {})
        if bool(position.get("is_open")):
            sell_signal, sell_reason, sell_target, updated_position = should_sell_signal(
                position=position,
                live_high=float(live_row["high"]),
                live_low=float(live_row["low"]),
                config=strategy,
            )
            position = updated_position
            if sell_signal and sell_reason and sell_target:
                available_eth = float(portfolio["balances"]["ETH"])
                if available_eth < min_position_eth:
                    print(
                        f"INFO: PAPER SELL ignored (ETH dust {available_eth:.6f} < {min_position_eth:.6f})."
                    )
                    position = close_position_state()
                    portfolio["position"] = position
                    continue
                qty = available_eth
                expected_notional = qty * float(sell_target)
                if notional_ok(expected_notional, sell_min_notional):
                    exit_price = apply_paper_fill(float(sell_target), "sell", slippage_rate)
                    quote_net = qty * exit_price * (1 - fee_rate)
                    entry_quote_spent = float(position.get("entry_quote_spent") or (qty * float(position.get("entry_price") or 0)))

                    portfolio["balances"]["ETH"] = max(0.0, float(portfolio["balances"]["ETH"]) - qty)
                    portfolio["balances"]["USDC"] = float(portfolio["balances"]["USDC"]) + quote_net

                    pnl_usdc = quote_net - entry_quote_spent
                    pnl_pct = (pnl_usdc / entry_quote_spent) * 100 if entry_quote_spent > 0 else 0.0
                    log_trade(
                        {
                            "ts": pd.Timestamp.now("UTC").isoformat(),
                            "action": "SELL",
                            "symbol": symbol,
                            "price": exit_price,
                            "base_qty": qty,
                            "quote_qty": quote_net,
                            "reason": sell_reason,
                            "entry_ts": position.get("entry_ts"),
                            "entry_price": position.get("entry_price"),
                            "pnl_usdc": pnl_usdc,
                            "pnl_pct": pnl_pct,
                            "balance_usdc": portfolio["balances"]["USDC"],
                            "balance_eth": portfolio["balances"]["ETH"],
                            "regime": active_regime,
                            "mode": mode,
                        }
                    )
                    position = close_position_state()
                else:
                    print("WARN: PAPER SELL skipped (notional below min). Clearing dust position state.")
                    position = close_position_state()

        if not bool(position.get("is_open")):
            buy_ok, buy_reason = should_buy_signal(
                signal_row=signal_row,
                active_regime=active_regime,
                config=strategy,
            )
            atr_entry = signal_row.get("atr14")
            if buy_ok and pd.notna(atr_entry) and float(atr_entry) > 0:
                usdc_balance = float(portfolio["balances"]["USDC"])
                quote_to_spend = min(usdc_balance, usdc_balance * stake_pct)
                if quote_per_trade > 0:
                    quote_to_spend = min(quote_to_spend, quote_per_trade)
                if quote_to_spend >= min_usdc_to_trade and notional_ok(quote_to_spend, buy_min_notional):
                    buy_fill = apply_paper_fill(price, "buy", slippage_rate)
                    qty = (quote_to_spend * (1 - fee_rate)) / buy_fill if buy_fill > 0 else 0.0
                    if qty > 0:
                        portfolio["balances"]["USDC"] = usdc_balance - quote_to_spend
                        portfolio["balances"]["ETH"] = float(portfolio["balances"]["ETH"]) + qty
                        position = build_position_after_entry(
                            entry_price=buy_fill,
                            entry_ts=signal_ts_iso,
                            entry_qty=qty,
                            atr_entry=float(atr_entry),
                            config=strategy,
                        )
                        position["entry_quote_spent"] = quote_to_spend
                        log_trade(
                            {
                                "ts": pd.Timestamp.now("UTC").isoformat(),
                                "action": "BUY",
                                "symbol": symbol,
                                "price": buy_fill,
                                "base_qty": qty,
                                "quote_qty": quote_to_spend,
                                "reason": buy_reason,
                                "entry_ts": signal_ts_iso,
                                "entry_price": buy_fill,
                                "pnl_usdc": 0.0,
                                "pnl_pct": 0.0,
                                "balance_usdc": portfolio["balances"]["USDC"],
                                "balance_eth": portfolio["balances"]["ETH"],
                                "regime": active_regime,
                                "mode": mode,
                            }
                        )

        portfolio["position"] = position
        portfolio["last_signal_ts"] = signal_ts_iso

    if close_eot and bool((portfolio.get("position") or {}).get("is_open")):
        last = df.iloc[-1]
        pos = dict(portfolio["position"])
        qty = float(pos.get("entry_qty") or 0.0)
        eot_price = float(last["close"])
        expected_notional = qty * eot_price
        if notional_ok(expected_notional, sell_min_notional):
            sell_fill = apply_paper_fill(eot_price, "sell", slippage_rate)
            quote_net = qty * sell_fill * (1 - fee_rate)
            entry_quote_spent = float(pos.get("entry_quote_spent") or (qty * float(pos.get("entry_price") or 0)))
            pnl_usdc = quote_net - entry_quote_spent
            pnl_pct = (pnl_usdc / entry_quote_spent) * 100 if entry_quote_spent > 0 else 0.0
            portfolio["balances"]["ETH"] = max(0.0, float(portfolio["balances"]["ETH"]) - qty)
            portfolio["balances"]["USDC"] = float(portfolio["balances"]["USDC"]) + quote_net
            log_trade(
                {
                    "ts": pd.Timestamp.now("UTC").isoformat(),
                    "action": "SELL",
                    "symbol": symbol,
                    "price": sell_fill,
                    "base_qty": qty,
                    "quote_qty": quote_net,
                    "reason": "eot_close",
                    "entry_ts": pos.get("entry_ts"),
                    "entry_price": pos.get("entry_price"),
                    "pnl_usdc": pnl_usdc,
                    "pnl_pct": pnl_pct,
                    "balance_usdc": portfolio["balances"]["USDC"],
                    "balance_eth": portfolio["balances"]["ETH"],
                    "regime": str((portfolio.get("regime_state") or {}).get("active", REGIME_FLAT)),
                    "mode": "paper",
                }
            )
        portfolio["position"] = close_position_state()

    save_portfolio(portfolio)
    print(
        f"PAPER DONE | USDC={portfolio['balances']['USDC']:.2f} "
        f"| ETH={portfolio['balances']['ETH']:.6f}"
    )

def main() -> None:
    load_env_file(ENV_FILE)

    exchange_id = get_env_str("EXCHANGE", "binance").lower()
    execution_mode = get_env_str("EXECUTION_MODE", "paper").lower()
    demo_mode = get_env_int("DEMO_MODE", 1) == 1
    symbol = get_env_str("SYMBOL", "ETH/USDC")
    timeframe = get_env_str("TIMEFRAME", "30m")
    htf_timeframe = get_env_str("HTF_TIMEFRAME", "4h")
    local_tz = get_env_str("LOCAL_TZ", "Europe/Warsaw")
    candle_limit = get_env_int("CANDLE_LIMIT", 800)
    run_loop = get_env_int("RUN_LOOP", 0) == 1
    sleep_seconds = get_env_int("SLEEP_SECONDS", 0)

    strategy = StrategyConfig(
        htf_flat_band_pct=get_env_float("HTF_FLAT_BAND_PCT", DEFAULT_STRATEGY.htf_flat_band_pct),
        comp_ratio=get_env_float("COMP_RATIO", DEFAULT_STRATEGY.comp_ratio),
        chop_comp_ratio_multiplier=get_env_float(
            "CHOP_COMP_RATIO_MULTIPLIER",
            DEFAULT_STRATEGY.chop_comp_ratio_multiplier,
        ),
        sl_mult=get_env_float("SL_MULT", DEFAULT_STRATEGY.sl_mult),
        tp_mult=get_env_float("TP_MULT", DEFAULT_STRATEGY.tp_mult),
        breakeven_after_r=get_env_float("BREAKEVEN_AFTER_R", DEFAULT_STRATEGY.breakeven_after_r),
        tie_policy=get_env_str("TIE_POLICY", DEFAULT_STRATEGY.tie_policy).upper(),
        regime_switch_confirm=get_env_int("REGIME_SWITCH_CONFIRM", DEFAULT_STRATEGY.regime_switch_confirm),
        regime_min_hold=get_env_int("REGIME_MIN_HOLD", DEFAULT_STRATEGY.regime_min_hold),
        adx_period=get_env_int("ADX_PERIOD", DEFAULT_STRATEGY.adx_period),
        use_adx_filter=get_env_int("USE_ADX_FILTER", int(DEFAULT_STRATEGY.use_adx_filter)) == 1,
        adx_min=get_env_float("ADX_MIN", DEFAULT_STRATEGY.adx_min),
    )

    stake_pct = max(0.0, min(1.0, get_env_float("STAKE_PCT", 1.0)))
    quote_per_trade = get_env_float("QUOTE_PER_TRADE", 5.0)
    min_usdc_to_trade = get_env_float("MIN_USDC_TO_TRADE", 5.0)
    min_position_eth = get_env_float("MIN_POSITION_ETH", 0.002)
    notional_buffer = get_env_float("NOTIONAL_BUFFER", 1.08)
    fee_rate = max(0.0, get_env_float("FEE_RATE", 0.001))
    slippage_rate = max(0.0, get_env_float("SLIPPAGE_RATE", 0.0002))
    enable_maker_entry = get_env_int("ENABLE_MAKER_ENTRY", 0) == 1
    enable_oco = get_env_int("ENABLE_OCO", 0) == 1
    paper_start_usdc = get_env_float("PAPER_START_USDC", 1000.0)
    close_eot = get_env_int("CLOSE_EOT", 1) == 1

    if candle_limit < 260:
        print(f"WARN: CANDLE_LIMIT={candle_limit} too low for warm-up. Using 260.")
        candle_limit = 260
    if strategy.tie_policy not in {"SL_FIRST", "TP_FIRST"}:
        print("ERROR: TIE_POLICY must be SL_FIRST or TP_FIRST")
        return

    api_key, api_secret = get_api_credentials()
    if execution_mode == "live" and (not api_key or not api_secret):
        print("ERROR: Missing API credentials for live mode.")
        return

    if exchange_id == "binance":
        if demo_mode:
            base_env = get_env_str("BINANCE_TESTNET_BASES", "https://testnet.binance.vision")
        else:
            base_env = get_env_str(
                "BINANCE_API_BASES",
                "https://api.binance.com,https://api1.binance.com,https://api2.binance.com,https://api3.binance.com",
            )
        bases = [b.strip() for b in base_env.split(",") if b.strip()]
    else:
        bases = [""]

    if enable_maker_entry or enable_oco:
        print("INFO: ENABLE_MAKER_ENTRY/ENABLE_OCO are future flags (TODO). Current execution uses MARKET.")

    portfolio = load_portfolio(initial_usdc=paper_start_usdc)
    loop_sleep = sleep_seconds if sleep_seconds > 0 else max(60, timeframe_to_minutes(timeframe) * 60)

    def build_exchange_and_data() -> tuple[ccxt.Exchange, pd.DataFrame]:
        last_exc: Exception | None = None
        for base in bases:
            try:
                exchange = make_exchange(
                    exchange_id=exchange_id,
                    base_url=base or None,
                    api_key=api_key if execution_mode == "live" else None,
                    api_secret=api_secret if execution_mode == "live" else None,
                    demo_mode=demo_mode,
                )
                exchange.load_markets()
                df_ltf, df_htf = fetch_ohlcv_data(
                    exchange=exchange,
                    symbol=symbol,
                    timeframe=timeframe,
                    htf_timeframe=htf_timeframe,
                    candle_limit=candle_limit,
                )
                if base:
                    print(f"INFO: using {exchange_id} base {base}")
                return exchange, build_feature_frame(df_ltf, df_htf, strategy)
            except Exception as exc:
                last_exc = exc
                print(f"WARN fetch failed using base={base or 'default'}: {exc}")
        raise RuntimeError(f"Cannot fetch market data: {last_exc}")

    if execution_mode == "paper":
        exchange, df = build_exchange_and_data()
        market_min_cost = get_market_min_notional(exchange, symbol)
        buy_min_notional = max(min_usdc_to_trade, market_min_cost * notional_buffer)
        sell_min_notional = market_min_cost * notional_buffer
        run_paper_backtest(
            df=df,
            symbol=symbol,
            strategy=strategy,
            portfolio=portfolio,
            stake_pct=stake_pct,
            quote_per_trade=quote_per_trade,
            min_usdc_to_trade=min_usdc_to_trade,
            buy_min_notional=buy_min_notional,
            sell_min_notional=sell_min_notional,
            min_position_eth=min_position_eth,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            close_eot=close_eot,
        )
        return

    try:
        while True:
            exchange, df = build_exchange_and_data()
            if len(df) < 3:
                print("ERROR: Not enough candles.")
                if not run_loop:
                    break
                time.sleep(loop_sleep)
                continue

            market_min_cost = get_market_min_notional(exchange, symbol)
            buy_min_notional = max(min_usdc_to_trade, market_min_cost * notional_buffer)
            sell_min_notional = market_min_cost * notional_buffer

            signal_row = df.iloc[-2]
            live_row = df.iloc[-1]
            signal_ts = pd.Timestamp(signal_row["ts"])
            signal_ts_iso = signal_ts.isoformat()
            signal_local = signal_ts.tz_convert(ZoneInfo(local_tz))
            price = float(signal_row["close"])

            is_new_signal = portfolio.get("last_signal_ts") != signal_ts_iso
            if is_new_signal:
                raw_regime = str(signal_row.get("raw_regime", REGIME_FLAT))
                portfolio["regime_state"] = update_regime_hysteresis(
                    raw_regime=raw_regime,
                    regime_state=portfolio.get("regime_state"),
                    switch_confirm=strategy.regime_switch_confirm,
                    min_hold=strategy.regime_min_hold,
                )
                portfolio["last_signal_ts"] = signal_ts_iso

            active_regime = str((portfolio.get("regime_state") or {}).get("active", REGIME_FLAT))
            bal_usdc = get_balance(exchange, "USDC")
            bal_eth = get_balance(exchange, "ETH")

            print(
                f"{symbol} | tf={timeframe} | candle={signal_local} | close={price:.2f} "
                f"| regime={active_regime} raw={signal_row.get('raw_regime')}"
            )
            print(f"BALANCE | USDC={bal_usdc:.2f} | ETH={bal_eth:.6f}")
            print(
                f"INFO | min_position_eth={min_position_eth:.6f} | "
                f"buy_min_notional={buy_min_notional:.2f} | sell_min_notional={sell_min_notional:.2f}"
            )

            position = dict(portfolio.get("position") or {})
            position_open = bool(position.get("is_open"))
            if (not position_open) and bal_eth > 0 and bal_eth < min_position_eth:
                print(
                    f"INFO: ETH dust ignored ({bal_eth:.6f} < MIN_POSITION_ETH {min_position_eth:.6f})."
                )
            elif (not position_open) and bal_eth >= min_position_eth:
                print(
                    "WARN: ETH balance >= MIN_POSITION_ETH but no tracked open position in portfolio. "
                    "Bot will not auto-sell this balance."
                )

            if position_open:
                sell_signal, sell_reason, sell_target, updated_position = should_sell_signal(
                    position=position,
                    live_high=float(live_row["high"]),
                    live_low=float(live_row["low"]),
                    config=strategy,
                )
                position = updated_position
                print(f"SELL SIGNAL = {sell_signal} (reason={sell_reason})")
                if sell_signal and sell_reason and sell_target:
                    available_eth = get_balance(exchange, "ETH")
                    if available_eth < min_position_eth:
                        print(
                            f"INFO: SELL ignored (ETH dust {available_eth:.6f} < MIN_POSITION_ETH {min_position_eth:.6f})."
                        )
                        position = close_position_state()
                        portfolio["position"] = position
                        save_portfolio(portfolio)
                        continue
                    qty = available_eth
                    expected_notional = qty * float(sell_target)
                    if notional_ok(expected_notional, sell_min_notional):
                        try:
                            order = place_market_sell(exchange, symbol, qty)
                            filled_qty = float(order.get("filled") or qty)
                            avg_price = float(order.get("average") or sell_target)
                            quote_qty = float(order.get("cost") or (filled_qty * avg_price))
                            entry_quote_spent = float(
                                position.get("entry_quote_spent")
                                or (filled_qty * float(position.get("entry_price") or 0.0))
                            )
                            pnl_usdc = quote_qty - entry_quote_spent
                            pnl_pct = (pnl_usdc / entry_quote_spent) * 100 if entry_quote_spent > 0 else 0.0
                            log_trade(
                                {
                                    "ts": pd.Timestamp.now("UTC").isoformat(),
                                    "action": "SELL",
                                    "symbol": symbol,
                                    "price": avg_price,
                                    "base_qty": filled_qty,
                                    "quote_qty": quote_qty,
                                    "reason": sell_reason,
                                    "entry_ts": position.get("entry_ts"),
                                    "entry_price": position.get("entry_price"),
                                    "pnl_usdc": pnl_usdc,
                                    "pnl_pct": pnl_pct,
                                    "balance_usdc": get_balance(exchange, "USDC"),
                                    "balance_eth": get_balance(exchange, "ETH"),
                                    "regime": active_regime,
                                    "mode": "live",
                                }
                            )
                            print(
                                f"SELL EXECUTED | qty={filled_qty:.6f} | price={avg_price:.2f} | "
                                f"quote={quote_qty:.2f} | pnl={pnl_usdc:.4f} USDC ({pnl_pct:.2f}%)"
                            )
                            position = close_position_state()
                        except Exception as exc:
                            print(f"ERROR live SELL failed: {exc}")
                    else:
                        print(
                            f"WARN: SELL skipped (notional below min). "
                            f"expected_notional={expected_notional:.4f} < min={sell_min_notional:.4f}. "
                            "Clearing dust position state."
                        )
                        position = close_position_state()
            else:
                print("SELL SIGNAL = False (no open tracked position)")

            if (not bool(position.get("is_open"))) and is_new_signal:
                buy_ok, buy_reason = should_buy_signal(
                    signal_row=signal_row,
                    active_regime=active_regime,
                    config=strategy,
                )
                print(f"BUY SIGNAL = {buy_ok} (reason={buy_reason})")
                atr_entry = signal_row.get("atr14")
                if buy_ok and pd.notna(atr_entry) and float(atr_entry) > 0:
                    usdc_balance = get_balance(exchange, "USDC")
                    quote_to_spend = min(usdc_balance, usdc_balance * stake_pct)
                    if quote_per_trade > 0:
                        quote_to_spend = min(quote_to_spend, quote_per_trade)
                    if quote_to_spend < min_usdc_to_trade:
                        print("WARN: BUY skipped (USDC below MIN_USDC_TO_TRADE).")
                    elif notional_ok(quote_to_spend, buy_min_notional):
                        try:
                            order = place_market_buy(exchange, symbol, quote_to_spend, fallback_price=price)
                            entry_qty = float(order.get("filled") or 0.0)
                            entry_price = float(order.get("average") or price)
                            entry_quote_spent = float(order.get("cost") or quote_to_spend)
                            if entry_qty > 0:
                                position = build_position_after_entry(
                                    entry_price=entry_price,
                                    entry_ts=signal_ts_iso,
                                    entry_qty=entry_qty,
                                    atr_entry=float(atr_entry),
                                    config=strategy,
                                )
                                position["entry_quote_spent"] = entry_quote_spent
                                log_trade(
                                    {
                                        "ts": pd.Timestamp.now("UTC").isoformat(),
                                        "action": "BUY",
                                        "symbol": symbol,
                                        "price": entry_price,
                                        "base_qty": entry_qty,
                                        "quote_qty": entry_quote_spent,
                                        "reason": buy_reason,
                                        "entry_ts": signal_ts_iso,
                                        "entry_price": entry_price,
                                        "pnl_usdc": 0.0,
                                        "pnl_pct": 0.0,
                                        "balance_usdc": get_balance(exchange, "USDC"),
                                        "balance_eth": get_balance(exchange, "ETH"),
                                        "regime": active_regime,
                                        "mode": "live",
                                    }
                                )
                                print(
                                    f"BUY EXECUTED | qty={entry_qty:.6f} | price={entry_price:.2f} | "
                                    f"quote={entry_quote_spent:.2f} | atr_entry={float(atr_entry):.4f}"
                                )
                        except Exception as exc:
                            print(f"ERROR live BUY failed: {exc}")
                    else:
                        print(
                            f"WARN: BUY skipped (notional below min). "
                            f"quote_to_spend={quote_to_spend:.4f} < min={buy_min_notional:.4f}."
                        )
                elif buy_ok:
                    print("WARN: BUY skipped (ATR warm-up / invalid atr_entry).")
            elif not is_new_signal:
                print("BUY SIGNAL = SKIPPED (same signal candle as previous cycle)")

            portfolio["position"] = position
            save_portfolio(portfolio)

            if not run_loop:
                break
            print(f"INFO: sleeping {loop_sleep}s...")
            time.sleep(loop_sleep)
    except KeyboardInterrupt:
        print("INFO: stopped by user.")


if __name__ == "__main__":
    main()
