import warnings

import pytest

from quantstack.config import BotConfig
from quantstack.data import synthetic_bars

warnings.filterwarnings("ignore", category=RuntimeWarning)


@pytest.fixture
def cfg(tmp_path):
    return BotConfig(state_dir=str(tmp_path / "state"), report_dir=str(tmp_path / "reports"),
                     log_dir=str(tmp_path / "logs"), data_dir=str(tmp_path / "data"))


@pytest.fixture(scope="session")
def noise_bars():
    return synthetic_bars(1500, seed=1, trend_strength=0.0)


@pytest.fixture(scope="session")
def edge_bars():
    # A planted persistent-drift edge that clears all three gates with margin, at
    # fold-activity thresholds of 5%, 10% and 25% alike (not on a hairline).
    return synthetic_bars(1500, seed=2, trend_strength=0.5)
