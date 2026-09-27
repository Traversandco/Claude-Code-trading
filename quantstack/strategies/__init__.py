from .base import Strategy
from .batch1 import BullDipBuy, DonchianBreakout, FundingSqueeze, VolSqueezeBreakout
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
    DonchianBreakout.name: DonchianBreakout,
    VolSqueezeBreakout.name: VolSqueezeBreakout,
    FundingSqueeze.name: FundingSqueeze,
    BullDipBuy.name: BullDipBuy,
}


def get_strategy(name: str) -> Strategy:
    try:
        return REGISTRY[name]()
    except KeyError:
        raise KeyError(f"unknown strategy {name!r}; registered: {sorted(REGISTRY)}") from None


__all__ = ["Strategy", "REGISTRY", "get_strategy"]
