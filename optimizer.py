"""Revenue / profit optimizer.

Given a fitted demand model and a batch of contexts, search a price grid per row and
select the price that maximizes expected contribution profit, (price - cost) * demand,
subject to price bounds expressed as multiples of unit cost.

Unconstrained *revenue* maximization is degenerate under elastic demand: with elasticity
above 1 the revenue-maximizing price collapses to the lower bound. Contribution profit has
a proper interior optimum because margin is zero at price = cost, so profit is the
economically meaningful objective for retail dynamic pricing. The objective is selectable
for transparency, but the default and the headline metric are profit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .config import settings
from .demand_model import DemandModel


@dataclass
class PricingDecision:
    price: np.ndarray            # chosen price per row
    expected_demand: np.ndarray  # model-predicted demand at the chosen price
    expected_profit: np.ndarray  # model-predicted profit at the chosen price


def optimize_prices(
    model: DemandModel,
    context_X: np.ndarray,
    cost: np.ndarray,
    objective: Literal["profit", "revenue"] = "profit",
    cfg=settings,
) -> PricingDecision:
    n = context_X.shape[0]
    lo = cost * cfg.price_min_mult
    hi = cost * cfg.price_max_mult
    # (steps, n) grid of candidate prices, per-row bounds
    fracs = np.linspace(0.0, 1.0, cfg.price_grid_steps).reshape(-1, 1)
    grid = lo.reshape(1, -1) + fracs * (hi - lo).reshape(1, -1)

    best_val = np.full(n, -np.inf)
    best_price = grid[0].copy()
    best_demand = np.zeros(n)
    for s in range(grid.shape[0]):
        price_row = grid[s]
        demand = model.predict(context_X, price_row)
        if objective == "revenue":
            val = price_row * demand
        else:
            val = (price_row - cost) * demand
        better = val > best_val
        best_val = np.where(better, val, best_val)
        best_price = np.where(better, price_row, best_price)
        best_demand = np.where(better, demand, best_demand)

    if objective == "revenue":
        best_profit = (best_price - cost) * best_demand
    else:
        best_profit = best_val
    return PricingDecision(
        price=best_price, expected_demand=best_demand, expected_profit=best_profit
    )
