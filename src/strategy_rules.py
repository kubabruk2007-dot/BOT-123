from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


REGIME_BULL = "trend_bull"
REGIME_BEAR = "trend_bear"
REGIME_FLAT = "trend_flat"
VALID_TIE_POLICIES = {"SL_FIRST", "TP_FIRST"}


@dataclass(frozen=True)
class StrategyConfig:
    htf_flat_band_pct: float = 0.001
    comp_ratio: float = 0.9
    chop_comp_ratio_multiplier: float = 0.95
    sl_mult: float = 1.5
    tp_mult: float = 2.4
    breakeven_after_r: float = 1.0
    tie_policy: str = "SL_FIRST"
    regime_switch_confirm: int = 3
    regime_min_hold: int = 6
    adx_period: int = 14
    use_adx_filter: bool = False
    adx_min: float = 18.0


DEFAULT_STRATEGY = StrategyConfig()


def detect_regime_from_htf(
    ema50_htf: float | None,
    ema200_htf: float | None,
    flat_band_pct: float,
) -> str:
    if ema50_htf is None or ema200_htf is None or ema200_htf == 0:
        return REGIME_FLAT

    spread = (ema50_htf - ema200_htf) / abs(ema200_htf)
    if spread > flat_band_pct:
        return REGIME_BULL
    if spread < -flat_band_pct:
        return REGIME_BEAR
    return REGIME_FLAT


def update_regime_hysteresis(
    raw_regime: str,
    regime_state: dict[str, Any] | None,
    switch_confirm: int,
    min_hold: int,
) -> dict[str, Any]:
    state = dict(regime_state or {})
    active = str(state.get("active") or raw_regime)
    candidate = str(state.get("candidate") or active)
    candidate_count = int(state.get("candidate_count") or 0)
    hold_count = int(state.get("hold_count") or 0)

    if raw_regime == active:
        hold_count += 1
        candidate = active
        candidate_count = 0
    elif hold_count < min_hold:
        hold_count += 1
    else:
        hold_count += 1
        if raw_regime == candidate:
            candidate_count += 1
        else:
            candidate = raw_regime
            candidate_count = 1
        if candidate_count >= switch_confirm:
            active = candidate
            hold_count = 0
            candidate = active
            candidate_count = 0

    return {
        "active": active,
        "candidate": candidate,
        "candidate_count": candidate_count,
        "hold_count": hold_count,
    }


def should_buy_signal(
    signal_row: pd.Series,
    active_regime: str,
    config: StrategyConfig,
) -> tuple[bool, str]:
    if active_regime == REGIME_BEAR:
        return False, "regime_bear_no_trade"

    htf_regime = str(signal_row.get("raw_regime", REGIME_FLAT))
    if htf_regime != REGIME_BULL:
        return False, "htf_not_bull"

    atr_now = signal_row.get("atr14")
    atr_sma = signal_row.get("atr_sma50")
    hh20_prev = signal_row.get("hh20_prev")
    close_price = signal_row.get("close")
    adx_val = signal_row.get("adx14")

    needed = [atr_now, atr_sma, hh20_prev, close_price]
    if any(pd.isna(v) for v in needed):
        return False, "warmup"

    ratio = config.comp_ratio
    if active_regime == REGIME_FLAT:
        ratio *= config.chop_comp_ratio_multiplier

    compression_ok = float(atr_now) < float(ratio) * float(atr_sma)
    if not compression_ok:
        return False, "no_compression"

    if config.use_adx_filter:
        if pd.isna(adx_val) or float(adx_val) < config.adx_min:
            return False, "adx_too_low"

    breakout_ok = float(close_price) > float(hh20_prev)
    if not breakout_ok:
        return False, "no_breakout"

    return True, "entry_breakout"


def build_position_after_entry(
    entry_price: float,
    entry_ts: str,
    entry_qty: float,
    atr_entry: float,
    config: StrategyConfig,
) -> dict[str, Any]:
    sl_price = entry_price - (config.sl_mult * atr_entry)
    tp_price = entry_price + (config.tp_mult * atr_entry)
    risk_r = max(0.0, entry_price - sl_price)
    return {
        "is_open": True,
        "entry_price": entry_price,
        "entry_ts": entry_ts,
        "entry_qty": entry_qty,
        "atr_entry": atr_entry,
        "sl_price": sl_price,
        "tp_price": tp_price,
        "risk_r": risk_r,
        "breakeven_done": False,
        "pending_sell": False,
        "pending_reason": None,
    }


def should_sell_signal(
    position: dict[str, Any],
    live_high: float,
    live_low: float,
    config: StrategyConfig,
) -> tuple[bool, str | None, float | None, dict[str, Any]]:
    if not position or not bool(position.get("is_open")):
        return False, None, None, position

    tie_policy = str(config.tie_policy).upper()
    if tie_policy not in VALID_TIE_POLICIES:
        tie_policy = "SL_FIRST"

    updated = dict(position)
    entry_price = float(updated.get("entry_price") or 0.0)
    sl_price = float(updated.get("sl_price") or 0.0)
    tp_price = float(updated.get("tp_price") or 0.0)
    risk_r = float(updated.get("risk_r") or max(0.0, entry_price - sl_price))
    breakeven_done = bool(updated.get("breakeven_done"))

    if entry_price <= 0 or sl_price <= 0 or tp_price <= 0:
        return False, None, None, updated

    if (not breakeven_done) and risk_r > 0:
        be_trigger = entry_price + (config.breakeven_after_r * risk_r)
        if live_high >= be_trigger:
            updated["sl_price"] = max(sl_price, entry_price)
            updated["breakeven_done"] = True
            sl_price = float(updated["sl_price"])

    sl_hit = live_low <= sl_price
    tp_hit = live_high >= tp_price

    if sl_hit and tp_hit:
        if tie_policy == "TP_FIRST":
            return True, "tp_hit", tp_price, updated
        return True, "sl_hit", sl_price, updated
    if sl_hit:
        return True, "sl_hit", sl_price, updated
    if tp_hit:
        return True, "tp_hit", tp_price, updated

    return False, None, None, updated
