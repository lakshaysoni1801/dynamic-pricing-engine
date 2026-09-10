import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pricing_engine.config import settings
from pricing_engine.data import FEATURE_COLUMNS, generate_market
from pricing_engine.demand_model import DemandModel, build_design_matrix
from pricing_engine.optimizer import optimize_prices


def _small():
    from dataclasses import replace

    return replace(settings, n_samples=3000, n_estimators=60)


def test_market_is_deterministic():
    a = generate_market(_small())
    b = generate_market(_small())
    assert np.array_equal(a.X, b.X)
    assert np.array_equal(a.demand, b.demand)


def test_true_demand_decreases_with_price():
    m = generate_market(_small())
    idx = np.arange(200)
    cost = m.X[idx, 0]
    low = m.true_demand(cost * 1.2, idx)
    high = m.true_demand(cost * 2.8, idx)
    # higher price should not increase demand on average (downward-sloping)
    assert high.mean() < low.mean()


def test_design_matrix_puts_price_first():
    m = generate_market(_small())
    design = build_design_matrix(m.X[:5], m.price[:5])
    assert design.shape == (5, len(FEATURE_COLUMNS) + 1)
    assert np.allclose(design[:, 0], m.price[:5])


def test_demand_model_learns_signal():
    cfg = _small()
    m = generate_market(cfg)
    model = DemandModel(cfg).fit(m.X, m.price, m.demand)
    pred = model.predict(m.X, m.price)
    # correlation with observed demand should be strongly positive
    corr = np.corrcoef(pred, m.demand)[0, 1]
    assert corr > 0.8
    assert (pred >= 0).all()


def test_optimizer_respects_price_bounds():
    cfg = _small()
    m = generate_market(cfg)
    model = DemandModel(cfg).fit(m.X, m.price, m.demand)
    idx = np.arange(300)
    cost = m.X[idx, 0]
    decision = optimize_prices(model, m.X[idx], cost, objective="profit", cfg=cfg)
    assert (decision.price >= cost * cfg.price_min_mult - 1e-6).all()
    assert (decision.price <= cost * cfg.price_max_mult + 1e-6).all()
    assert (decision.expected_profit >= 0).all()
