from .base import Strategy
from .rsi_reversion import RsiReversion
from .ts_momentum import TimeSeriesMomentum

REGISTRY: dict[str, type[Strategy]] = {
    TimeSeriesMomentum.name: TimeSeriesMomentum,
    RsiReversion.name: RsiReversion,
}


def get_strategy(name: str) -> Strategy:
    try:
        return REGISTRY[name]()
    except KeyError:
        raise KeyError(f"unknown strategy {name!r}; registered: {sorted(REGISTRY)}") from None


__all__ = ["Strategy", "REGISTRY", "get_strategy"]
