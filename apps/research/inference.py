"""One day's outputs of a fitted LightGBM run, read back from its model file (SPEC §7)."""

from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from apps.research.features.store import read_partitions
from apps.research.features.transforms import STOCK_LEVEL, MarketBins, winsorize
from apps.research.gbdt import Bagged
from apps.research.models import REST_FEATURE, ModelObjective, ModelRun
from apps.research.rank_model import MODEL_TYPE, rank_inputs
from apps.research.target_models import MARKET_FEATURES, sorted_quantiles

ROW_COLUMNS = ["stock_id", "date", "in_universe"]
TOP_CONTRIBUTIONS = 5


def day_rows(day: date, directory: Path, features: Sequence[str]) -> pd.DataFrame:
    """Every stock's raw features on day, from the day's yearly partition."""
    wanted = [*ROW_COLUMNS, *dict.fromkeys(features)]
    rows = read_partitions(day, day, directory, wanted)
    if rows.empty:
        raise ValueError(f"no feature rows on {day}; run build_features for it")
    return rows.reset_index(drop=True)


def _members(model: Any) -> tuple[lgb.Booster, ...]:
    if isinstance(model, Bagged):
        return model.boosters
    if isinstance(model, lgb.Booster):
        return (model,)
    raise ValueError(f"expected LightGBM boosters, not {type(model).__name__}")


def _boosters(run: ModelRun, model: Any) -> list[Any]:
    if run.objective == ModelObjective.RANK:
        return [model]
    if run.objective == ModelObjective.QUANTILE:
        return list(model["boosters"].values())
    return [model["booster"]]


def _check(run: ModelRun, model: Any) -> None:
    if run.model_type != MODEL_TYPE:
        raise ValueError(f"inference serves {MODEL_TYPE} runs, not {run.model_type}")
    features = list(run.feature_list)
    for booster in _boosters(run, model):
        for member in _members(booster):
            if member.feature_name() != features:
                raise ValueError(
                    f"{run.run_id} was fit on {member.feature_name()}, not its feature_list"
                )


def model_inputs(run: ModelRun, model: Any, rows: pd.DataFrame) -> pd.DataFrame:
    """The run's features as its fit saw them: z-scored for rank, binned market values otherwise."""
    _check(run, model)
    features = list(run.feature_list)
    if run.objective == ModelObjective.RANK:
        return rank_inputs(rows, features)[features].astype("float32")
    stock = [name for name in features if name in STOCK_LEVEL]
    foreign = [name for name in features if name not in stock and name not in MARKET_FEATURES]
    if foreign:
        raise ValueError(f"{run.run_id} reads features no raw-value model takes: {foreign}")
    return MarketBins(model["market_bins"]).apply(winsorize(rows, stock))[features]


def run_values(run: ModelRun, model: Any, rows: pd.DataFrame) -> pd.DataFrame:
    """The prediction columns the run fills, one row per row of rows."""
    inputs = model_inputs(run, model, rows)
    objective = run.objective
    if objective == ModelObjective.RANK:
        values = {"score": np.asarray(model.predict(inputs))}
    elif objective == ModelObjective.REGRESSION:
        values = {"ret_pred": np.asarray(model["booster"].predict(inputs))}
    elif objective == ModelObjective.QUANTILE:
        raw = {name: np.asarray(fit.predict(inputs)) for name, fit in model["boosters"].items()}
        quantiles = sorted_quantiles(raw)
        values = {**quantiles, "ret_pred": quantiles["q50"]}
    elif objective == ModelObjective.VOLATILITY:
        values = {"vol_pred": np.exp(np.asarray(model["booster"].predict(inputs)))}
    else:
        probs = np.asarray(model["booster"].predict(inputs))
        calibrator = model["calibrator"]
        values = {"prob_up_5pct": probs if calibrator is None else calibrator.predict(probs)}
    return pd.DataFrame(values, index=rows.index)


def top_contributions(
    members: Sequence[lgb.Booster], inputs: pd.DataFrame, stock_ids: pd.Series
) -> pd.DataFrame:
    """Per stock, the bag-mean pred_contrib of the largest features, then the rest summed."""
    features = np.asarray(inputs.columns, dtype=object)
    mean = np.mean([np.asarray(m.predict(inputs, pred_contrib=True)) for m in members], axis=0)
    shares = mean[:, :-1]
    order = np.argsort(-np.abs(shares), axis=1, kind="stable")
    shown = min(TOP_CONTRIBUTIONS, len(features))
    hidden = len(features) - shown
    names = features[order[:, :shown]]
    values = np.take_along_axis(shares, order[:, :shown], axis=1)
    counts = np.ones_like(values, dtype=np.int64)
    if hidden:
        rest = np.take_along_axis(shares, order[:, shown:], axis=1).sum(axis=1, keepdims=True)
        names = np.hstack([names, np.full((len(names), 1), REST_FEATURE, dtype=object)])
        values = np.hstack([values, rest])
        counts = np.hstack([counts, np.full((len(counts), 1), hidden)])
    width = names.shape[1]
    return pd.DataFrame(
        {
            "stock_id": np.repeat(stock_ids.to_numpy(), width),
            "feature": names.ravel(),
            "position": np.tile(np.arange(1, width + 1), len(inputs)),
            "contrib": values.ravel(),
            "n_features": counts.ravel(),
        }
    )


def run_contributions(run: ModelRun, model: Any, rows: pd.DataFrame) -> pd.DataFrame:
    """The rank run's top contributions per stock on the rows' day."""
    if run.objective != ModelObjective.RANK:
        raise ValueError(f"contributions come from rank runs, not {run.objective}")
    inputs = model_inputs(run, model, rows)
    return top_contributions(_members(model), inputs, rows["stock_id"])
