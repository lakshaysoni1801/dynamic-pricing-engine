"""Train the demand model and evaluate pricing policies against ground truth.

The evaluation is deliberately not circular. A pricing policy chooses prices using the
XGBoost demand MODEL, but the profit those prices actually earn is computed from the
market's TRUE latent demand function. This penalizes model error the way a real
deployment would, and prevents the optimizer from scoring well simply by exploiting its
own model's mistakes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import settings
from .data import Market, generate_market
from .demand_model import DemandModel
from .optimizer import optimize_prices


@dataclass
class Split:
    market: Market
    train_idx: np.ndarray
    test_idx: np.ndarray


def make_split(cfg=settings) -> Split:
    market = generate_market(cfg)
    n = market.X.shape[0]
    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(n)
    n_test = int(n * cfg.test_frac)
    test_idx = np.sort(perm[:n_test])
    train_idx = np.sort(perm[n_test:])
    return Split(market, train_idx, test_idx)


def train_model(split: Split, cfg=settings) -> DemandModel:
    m = split.market
    model = DemandModel(cfg)
    model.fit(m.X[split.train_idx], m.price[split.train_idx], m.demand[split.train_idx])
    return model


@dataclass
class PolicyResult:
    profit_per_unit_context: float
    revenue_per_unit_context: float
    mean_price: float


def _score_policy(market: Market, idx: np.ndarray, price: np.ndarray) -> PolicyResult:
    cost = market.X[idx, 0]
    true_dem = market.true_demand(price, idx)
    profit = (price - cost) * true_dem
    revenue = price * true_dem
    return PolicyResult(
        profit_per_unit_context=float(profit.mean()),
        revenue_per_unit_context=float(revenue.mean()),
        mean_price=float(price.mean()),
    )


def evaluate(split: Split, model: DemandModel, cfg=settings) -> dict:
    m = split.market
    idx = split.test_idx
    cost = m.X[idx, 0]

    # Baseline: the historical pricing policy actually used in the data.
    historical = _score_policy(m, idx, m.price[idx])

    # Optimized policy: model chooses prices; scored against TRUE demand.
    decision = optimize_prices(model, m.X[idx], cost, objective="profit", cfg=cfg)
    optimized = _score_policy(m, idx, decision.price)

    # Oracle ceiling: choose prices using the true demand function directly.
    lo, hi = cost * cfg.price_min_mult, cost * cfg.price_max_mult
    fracs = np.linspace(0, 1, cfg.price_grid_steps).reshape(-1, 1)
    grid = lo.reshape(1, -1) + fracs * (hi - lo).reshape(1, -1)
    true_profit = np.array([(g - cost) * m.true_demand(g, idx) for g in grid])
    oracle_price = grid[np.argmax(true_profit, axis=0), np.arange(len(idx))]
    oracle = _score_policy(m, idx, oracle_price)

    def uplift_pct(a: float, b: float) -> float:
        """Percentage uplift of a over b, rounded only for display."""
        return round((a / b - 1.0) * 100.0, 2)

    def raw_uplift(a: float, b: float) -> float:
        """Unrounded fractional uplift, used internally for ratio calculations so we
        never divide two numbers that have already been rounded for display."""
        return a / b - 1.0

    historical_profit = historical.profit_per_unit_context
    optimized_uplift = raw_uplift(optimized.profit_per_unit_context, historical_profit)
    oracle_uplift = raw_uplift(oracle.profit_per_unit_context, historical_profit)

    # Fraction of the oracle's uplift captured by the model-based policy, computed from
    # the raw (unrounded) uplift values rather than from the display-rounded percentages,
    # so the reported fraction reflects the actual ratio rather than an artifact of
    # rounding two numbers to 2 decimal places before dividing them.
    captured_fraction = round(optimized_uplift / oracle_uplift, 3) if oracle_uplift > 1e-9 else 0.0

    # Demand-model accuracy on held-out historical prices (sanity of the model itself).
    pred_hist = model.predict(m.X[idx], m.price[idx])
    true_hist = m.true_demand(m.price[idx], idx)
    mape = float(np.mean(np.abs(pred_hist - true_hist) / np.clip(true_hist, 1e-6, None)) * 100)

    return {
        "profit_uplift_pct": uplift_pct(optimized.profit_per_unit_context, historical_profit),
        "revenue_change_pct": uplift_pct(optimized.revenue_per_unit_context,
                                         historical.revenue_per_unit_context),
        "oracle_profit_uplift_pct": uplift_pct(oracle.profit_per_unit_context, historical_profit),
        "captured_fraction_of_oracle": captured_fraction,
        "demand_model_mape_pct": round(mape, 2),
        "mean_price_historical": round(historical.mean_price, 2),
        "mean_price_optimized": round(optimized.mean_price, 2),
        "n_test": len(idx),
    }
