from .base import Strategy
from .funding_crowding import FundingCrowding
from .hlhb import Hlhb
from .ml_direction import MlDirection
from .rsi_reversion import RsiReversion
from .ts_momentum import TimeSeriesMomentum

REGISTRY: dict[str, type[Strategy]] = {
    TimeSeriesMomentum.name: TimeSeriesMomentum,
    RsiReversion.name: RsiReversion,
    FundingCrowding.name: FundingCrowding,
    Hlhb.name: Hlhb,
    MlDirection.name: MlDirection,
}


def get_strategy(name: str) -> Strategy:
    try:
        return REGISTRY[name]()
    except KeyError:
        raise KeyError(f"unknown strategy {name!r}; registered: {sorted(REGISTRY)}") from None


__all__ = ["Strategy", "REGISTRY", "get_strategy"]
