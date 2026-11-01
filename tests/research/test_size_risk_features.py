import math

import numpy as np
import pandas as pd
import pytest

from apps.research.features.registry import Group, feature_names
from apps.research.features.size_risk import (
    canonical_industry,
    industry_risk_features,
    size_liquidity_features,
)
from tests.research.feature_panels import at, listing, rising

NAN = math.nan
ROOT = math.sqrt(252)


def market_moves(n: int) -> np.ndarray:
    return np.array([0.01 * math.sin(k * 0.7) + 0.004 * math.cos(k * 1.9) for k in range(n)])


def tracking(moves: np.ndarray, beta: float, noise: float = 0.0) -> list[float]:
    """Closes whose daily return is beta × the market move plus a fixed wiggle."""
    steps = beta * moves + noise * np.array([math.sin(k * 2.3) for k in range(len(moves))])
    steps[0] = 0.0
    return [float(close) for close in 100.0 * np.cumprod(1 + steps)]


def ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Slope and annualized residual std with two degrees of freedom used."""
    slope, intercept = np.polyfit(x, y, 1)
    residuals = y - (slope * x + intercept)
    return float(slope), float(np.sqrt((residuals**2).sum() / (len(x) - 2)) * ROOT)


def test_columns_follow_the_registry() -> None:
    panel = listing("2330", rising(30))

    assert list(size_liquidity_features(panel).columns) == [
        "stock_id",
        "date",
        *feature_names(Group.SIZE_LIQUIDITY),
    ]
    assert list(industry_risk_features(panel).columns) == [
        "stock_id",
        "date",
        *feature_names(Group.INDUSTRY_RISK),
    ]


def test_market_cap_uses_the_raw_close_carried_over_halts() -> None:
    closes = [50.0, 52.0, NAN]
    panel = listing("2330", closes, adj_close=[25.0, 26.0, NAN], shares=[1e6, 2e6, 2e6])

    features = size_liquidity_features(panel)

    assert at(features, "2330", 1, "size_log_mcap") == pytest.approx(math.log(52.0 * 2e6))
    assert at(features, "2330", 2, "size_log_mcap") == pytest.approx(math.log(52.0 * 2e6))
    assert math.isnan(
        at(size_liquidity_features(panel.assign(shares=NAN)), "2330", 1, "size_log_mcap")
    )


def test_dollar_volume_median_is_the_universe_median_turnover() -> None:
    panel = listing("2330", rising(5), turnover_med_20d=[NAN, NAN, 1.0, 2.0, 3.0])

    values = size_liquidity_features(panel)["liq_dollar_vol_med_20d"].tolist()

    assert values[2:] == [1.0, 2.0, 3.0]


def test_amihud_skips_days_without_turnover() -> None:
    closes = [100.0, 110.0, 99.0, NAN, 108.9, 108.9]
    turnover = [1e6, 2e6, 3e6, NAN, 0.0, 4e6]
    features = size_liquidity_features(listing("2330", closes, turnover=turnover))

    expected = np.mean([0.1 / 2e6, 0.1 / 3e6, 0.0 / 4e6])
    assert at(features, "2330", 5, "liq_amihud_20d") == pytest.approx(expected)
    assert math.isnan(at(features, "2330", 0, "liq_amihud_20d"))


def test_amihud_averages_the_last_twenty_rows_and_skips_days_without_a_close() -> None:
    closes = [100.0 * 1.01**k for k in range(25)]
    turnover = [1e6] * 25
    turnover[4], turnover[5] = 1e3, 1e5
    panel = listing("2330", closes, turnover=turnover).assign(
        close=lambda frame: frame["close"].mask(frame.index == 7)
    )
    steps = pd.Series(closes) / pd.Series(closes).shift(1) - 1

    features = size_liquidity_features(panel)

    kept = [k for k in range(5, 25) if k != 7]
    expected = np.mean([steps[k] / turnover[k] for k in kept])
    assert at(features, "2330", 24, "liq_amihud_20d") == pytest.approx(expected)


def test_beta_and_idiosyncratic_volatility_match_least_squares() -> None:
    moves = market_moves(320)
    closes = tracking(moves, 1.4, noise=0.006)
    closes[100] = NAN
    market = pd.Series(moves).mask(pd.Series(range(320)).isin([150, 151]))
    panel = listing("2330", closes, market_ret=market)
    returns = pd.Series(closes) / pd.Series(closes).shift(1) - 1

    features = industry_risk_features(panel)

    window = market.iloc[51:301]
    kept = returns.iloc[51:301].notna() & window.notna()
    slope, _ = ols(window[kept].to_numpy(), returns.iloc[51:301][kept].to_numpy())
    _, idio = ols(moves[241:301], returns.iloc[241:301].to_numpy())
    assert at(features, "2330", 300, "risk_beta_250d") == pytest.approx(slope)
    assert at(features, "2330", 300, "risk_idio_vol_60d") == pytest.approx(idio)


def test_beta_needs_sixty_pairs_and_idiosyncratic_volatility_twenty() -> None:
    moves = market_moves(80)
    features = industry_risk_features(listing("2330", tracking(moves, 0.8), market_ret=moves))

    assert math.isnan(at(features, "2330", 59, "risk_beta_250d"))
    assert at(features, "2330", 60, "risk_beta_250d") == pytest.approx(0.8)
    assert math.isnan(at(features, "2330", 19, "risk_idio_vol_60d"))
    assert at(features, "2330", 20, "risk_idio_vol_60d") == pytest.approx(0.0, abs=1e-6)


def test_beta_is_missing_without_market_returns() -> None:
    moves = market_moves(80)
    panel = listing("2330", tracking(moves, 0.8), market_ret=NAN)

    assert industry_risk_features(panel)["risk_beta_250d"].isna().all()


def test_industry_names_of_both_exchanges_are_unified() -> None:
    names = pd.Series(
        ["其他電子類", "其他電子業", " 金融業", "金融保險", "觀光事業", "", None, "半導體業"]
    )

    assert canonical_industry(names).tolist() == [
        "其他電子業",
        "其他電子業",
        "金融保險",
        "金融保險",
        "觀光餐旅",
        None,
        None,
        "半導體業",
    ]


def test_sector_return_averages_universe_stocks_of_the_same_industry() -> None:
    def stock(
        stock_id: str, growth: float, industry: str, market: str = "TWSE", in_universe: bool = True
    ) -> pd.DataFrame:
        closes = [100.0 * (1 + growth) ** k for k in range(25)]
        return listing(stock_id, closes, industry=industry, market=market, in_universe=in_universe)

    panel = pd.concat(
        [
            stock("2330", 0.01, industry="其他電子業"),
            stock("6488", 0.02, industry="其他電子類", market="TPEx"),
            stock("2317", 0.05, industry="其他電子業", in_universe=False),
            stock("1101", 0.03, industry=""),
        ],
        ignore_index=True,
    )

    features = industry_risk_features(panel)

    sector = ((1.01**20 - 1) + (1.02**20 - 1)) / 2
    assert at(features, "2330", 24, "ind_sector_ret_20d") == pytest.approx(sector)
    assert at(features, "2317", 24, "ind_sector_ret_20d") == pytest.approx(sector)
    assert at(features, "2317", 24, "ind_rel_ret_20d") == pytest.approx(1.05**20 - 1 - sector)
    assert math.isnan(at(features, "1101", 24, "ind_sector_ret_20d"))
    assert math.isnan(at(features, "2330", 19, "ind_sector_ret_20d"))


def test_unsorted_panel_is_refused() -> None:
    panel = listing("2330", rising(10)).iloc[::-1]
    with pytest.raises(ValueError, match="sorted"):
        size_liquidity_features(panel)
    with pytest.raises(ValueError, match="sorted"):
        industry_risk_features(panel)
