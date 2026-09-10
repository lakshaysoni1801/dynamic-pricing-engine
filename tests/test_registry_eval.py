import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pricing_engine.config import settings
from pricing_engine.data import generate_market
from pricing_engine.demand_model import DemandModel
from pricing_engine.registry import ModelRegistry
from pricing_engine.train import evaluate, make_split, train_model


def _small():
    return replace(settings, n_samples=3000, n_estimators=60)


def test_registry_register_promote_resolve(tmp_path):
    reg = ModelRegistry(str(tmp_path / "reg"))
    cfg = _small()
    m = generate_market(cfg)
    model = DemandModel(cfg).fit(m.X, m.price, m.demand)

    v1 = reg.register(model, metrics={"mape": 1.0}, params={"n": 60})
    v2 = reg.register(model, metrics={"mape": 0.9}, params={"n": 60})
    assert v1.version == 1 and v2.version == 2
    assert reg.resolve("production") is None  # nothing promoted yet

    reg.promote(2, "production")
    assert reg.resolve("production") == 2
    loaded = reg.load(alias="production")
    assert isinstance(loaded, DemandModel)
    assert len(reg.list_versions()) == 2


def test_registry_rejects_unknown_promotion(tmp_path):
    reg = ModelRegistry(str(tmp_path / "reg"))
    try:
        reg.promote(99)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_evaluation_uplift_is_bounded_by_oracle():
    # Use enough data that the model is meaningfully trained; a tiny config can produce a
    # policy that underperforms the baseline, which is honest but not what we assert here.
    cfg = replace(settings, n_samples=8000, n_estimators=150)
    split = make_split(cfg)
    model = train_model(split, cfg)
    res = evaluate(split, model, cfg)
    # the model-based policy can never beat the oracle that knows true demand
    assert res["profit_uplift_pct"] <= res["oracle_profit_uplift_pct"] + 0.05
    assert res["captured_fraction_of_oracle"] <= 1.0 + 1e-6
    assert res["demand_model_mape_pct"] >= 0
