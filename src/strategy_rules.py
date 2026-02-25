from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StrategyConfig:
    ema_fast_period: int = 20  # Okres szybkiej EMA (reakcja na krotkie ruchy ceny).
    ema_slow_period: int = 50  # Okres wolnej EMA (kierunek srednioterminowy).
    ema_trend_period: int = 200  # Okres EMA trendowej (filtr trendu glownego).
    rsi_period: int = 14  # Okres RSI.
    rsi_min: float = 40.0  # Dolny prog RSI dla wejscia BUY.
    rsi_max: float = 55.0  # Gorny prog RSI dla wejscia BUY.
    rsi_sell_high: float = 70.0  # RSI powyzej tej wartosci moze wywolywac SELL.
    rsi_sell_low: float = 35.0  # RSI ponizej tej wartosci moze wywolywac SELL.
    enable_rsi_sell: bool = True  # Czy RSI ma brac udzial w warunku SELL.
    force_buy: bool = False  # Wymusza BUY niezaleznie od pozostalych warunkow.
    disable_trend_filter: bool = False  # Wylacza filtr trendu (ema_slow > ema_trend).
    disable_pullback_filter: bool = False  # Wylacza filtr pullbacku (ema_fast < ema_slow).
    rsi_only_sell: bool = False  # Gdy True, SELL tylko po ekstremach RSI (na nowym sygnale).
    tp_pct: float = 1.2  # Take profit w procentach od ceny wejscia.
    sl_pct: float = 0.7  # Stop loss w procentach od ceny wejscia.


DEFAULT_STRATEGY = StrategyConfig()


def should_buy_signal(
    ema_fast_value: float,
    ema_slow_value: float,
    ema_trend_value: float,
    rsi_now: float,
    rsi_prev: float,
    config: StrategyConfig,
) -> bool:
    # ema_* i rsi_* to wartosci policzone dla swiecy sygnalowej (zamknietej).
    # Zwraca True, gdy warunki BUY sa spelnione.
    trend_ok = (ema_slow_value > ema_trend_value) or config.disable_trend_filter
    pullback_ok = (ema_fast_value < ema_slow_value) or config.disable_pullback_filter

    signal = (
        trend_ok
        and pullback_ok
        and (rsi_now > rsi_prev)
        and (config.rsi_min <= rsi_now <= config.rsi_max)
    )
    return True if config.force_buy else signal


def should_sell_signal(
    position: dict[str, Any],
    live_high: float,
    live_low: float,
    price: float,
    ema_slow_value: float,
    ema_trend_value: float,
    rsi_now: float,
    is_new_signal: bool,
    config: StrategyConfig,
) -> tuple[bool, str | None, float | None]:
    # position: stan otwartej pozycji (entry_price, pending_sell, pending_reason).
    # live_high/live_low: biezace high/low aktualnie formowanej swiecy.
    # price: cena swiecy sygnalowej (lub fallback do wykonania).
    # Zwraca: (czy_sprzedac, powod_sprzedazy, cena_docelowa_sprzedazy).
    entry_price = float(position.get("entry_price") or 0)
    pending_sell = bool(position.get("pending_sell"))
    pending_reason = position.get("pending_reason")

    if entry_price <= 0:
        return False, None, None

    tp_price = entry_price * (1 + config.tp_pct / 100)
    sl_price = entry_price * (1 - config.sl_pct / 100)

    if pending_sell:
        reason = pending_reason or "retry_sell"
        return True, reason, price

    if config.rsi_only_sell and is_new_signal:
        if rsi_now > config.rsi_sell_high or rsi_now < config.rsi_sell_low:
            return True, "rsi_extreme", price
        return False, None, None

    if live_low <= sl_price:
        return True, "sl_hit", sl_price

    if live_high >= tp_price:
        return True, "tp_hit", tp_price

    if is_new_signal:
        if ema_slow_value < ema_trend_value:
            return True, "trend_loss", price
        if config.enable_rsi_sell and (rsi_now > config.rsi_sell_high or rsi_now < config.rsi_sell_low):
            return True, "rsi_extreme", price

    return False, None, None
