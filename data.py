"""Synthetic pricing market with a KNOWN latent demand function.

This module models a market where each observation is a product in a context (segment,
competitor price, season, promotion, cost) sold at some historical price, yielding a
noisy observed demand.

The crucial design point for an honest pricing project: the *true* demand function is
known here because the data is synthetic, but it is used ONLY to (a) generate observed
training samples with noise and (b) score pricing policies during evaluation. The demand
MODEL that the optimizer relies on never sees the true function, only noisy samples. This
lets us evaluate an optimized pricing policy against ground truth rather than against the
model's own predictions, which would be circular and inflate the result.

Latent demand has several realistic, nonlinear features:
  * segment-specific price elasticity (some customers are far more price sensitive);
  * a competitor-price reference effect with a kink: demand falls faster once our price
    rises above the competitor's;
  * multiplicative seasonality and a promotion uplift;
  * saturating base demand.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import settings

FEATURE_COLUMNS = [
    "cost",
    "competitor_price",
    "segment",         # 0,1,2 -> categorical, encoded as int
    "season",          # continuous in [0,1], yearly phase
    "on_promo",        # 0/1
    "base_popularity",  # latent-ish popularity proxy, observable
]


@dataclass
class Market:
    X: np.ndarray            # (n, n_features) observable context (excludes price)
    price: np.ndarray        # (n,) historical price charged
    demand: np.ndarray       # (n,) observed noisy demand at that price
    feature_columns: list[str]
    _params: dict            # latent parameters, for the true demand function

    def true_demand(self, price: np.ndarray, idx: np.ndarray | None = None) -> np.ndarray:
        """Ground-truth expected demand at an arbitrary price for the given rows.

        `idx` selects rows (defaults to all). `price` must broadcast to those rows.
        """
        p = self._params
        if idx is None:
            idx = np.arange(self.X.shape[0])
        cost = self.X[idx, 0]
        comp = self.X[idx, 1]
        segment = self.X[idx, 2].astype(int)
        season = self.X[idx, 3]
        promo = self.X[idx, 4]
        base_pop = self.X[idx, 5]

        elasticity = p["segment_elasticity"][segment]
        base = p["base_scale"] * base_pop

        # reference effect: price relative to competitor, with a kink above parity
        rel = price / np.clip(comp, 1e-6, None)
        over = np.maximum(rel - 1.0, 0.0)
        ref_effect = -p["under_slope"] * (rel - 1.0) - p["over_kink"] * over

        # own-price elasticity in log space (nonlinear in raw price)
        ref_price = cost * p["ref_markup"]
        own_effect = -elasticity * np.log(np.clip(price / ref_price, 1e-6, None))

        seasonal = 1.0 + p["season_amp"] * np.sin(2 * np.pi * season)
        promo_mult = 1.0 + p["promo_uplift"] * promo

        log_units = np.log(np.clip(base, 1e-6, None)) + own_effect + ref_effect
        units = np.exp(log_units) * seasonal * promo_mult
        return np.clip(units, 0.0, None)


def generate_market(cfg=settings) -> Market:
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_samples

    cost = rng.uniform(5, 40, n)
    segment = rng.integers(0, 3, n)
    season = rng.uniform(0, 1, n)
    on_promo = (rng.random(n) < 0.2).astype(float)
    base_popularity = rng.gamma(shape=3.0, scale=1.0, size=n)  # positive skew
    # competitor prices cluster around a markup on cost
    competitor_price = cost * rng.uniform(1.4, 2.4, n) * (1 - 0.15 * on_promo)

    params = {
        "segment_elasticity": np.array([1.2, 1.8, 2.6]),  # segment 2 is very price-sensitive
        "base_scale": 30.0,
        "under_slope": 0.5,
        "over_kink": 2.5,          # demand falls sharply once above competitor price
        "ref_markup": 1.8,
        "season_amp": 0.25,
        "promo_uplift": 0.35,
    }

    X = np.column_stack([cost, competitor_price, segment, season, on_promo, base_popularity])
    market = Market(X=X, price=np.zeros(n), demand=np.zeros(n),
                    feature_columns=FEATURE_COLUMNS, _params=params)

    # historical pricing policy: a blend of a noisy cost-plus rule and competitor
    # matching, with realistic price exploration (promotions, regional differences,
    # experiments) layered on top. The baseline is competitive on average and NOT a
    # strawman, but it ignores segment-level elasticity, which is the headroom a
    # model-based optimizer captures. The exploration noise also gives the demand model
    # enough price variation to estimate the price-response curve, mirroring firms that
    # vary prices enough to learn from.
    markup = rng.normal(1.9, 0.35, n).clip(1.1, 3.2)
    cost_plus_price = cost * markup
    competitor_match = competitor_price * rng.uniform(0.9, 1.05, n)
    center = 0.55 * cost_plus_price + 0.45 * competitor_match
    exploration = rng.lognormal(mean=0.0, sigma=0.12, size=n)
    historical_price = center * exploration
    market.price = historical_price

    true = market.true_demand(historical_price)
    noise = rng.lognormal(mean=0.0, sigma=0.15, size=n)  # multiplicative observation noise
    market.demand = true * noise
    return market
