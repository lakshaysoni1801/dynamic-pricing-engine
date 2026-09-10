import sys
import warnings
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

warnings.filterwarnings("ignore")

from fastapi.testclient import TestClient

from pricing_engine.config import settings
from pricing_engine.data import generate_market
from pricing_engine.demand_model import DemandModel
from pricing_engine.registry import ModelRegistry
from pricing_engine.service import create_app

CTX = {
    "cost": 20.0,
    "competitor_price": 45.0,
    "segment": 2,
    "season": 0.3,
    "on_promo": 0,
    "base_popularity": 3.0,
}


def _registry_with_model(tmp_path):
    cfg = replace(settings, n_samples=3000, n_estimators=60)
    reg = ModelRegistry(str(tmp_path / "reg"))
    m = generate_market(cfg)
    model = DemandModel(cfg).fit(m.X, m.price, m.demand)
    mv = reg.register(model, metrics={}, params={})
    reg.promote(mv.version)
    return reg


def test_health_503_without_model(tmp_path):
    reg = ModelRegistry(str(tmp_path / "empty"))
    client = TestClient(create_app(reg))
    assert client.get("/health").json()["model_version"] is None
    assert client.post("/predict-demand", json={"context": CTX, "price": 40.0}).status_code == 503


def test_predict_and_optimize(tmp_path):
    client = TestClient(create_app(_registry_with_model(tmp_path)))
    assert client.get("/health").json()["model_version"] == 1

    r = client.post("/predict-demand", json={"context": CTX, "price": 40.0})
    assert r.status_code == 200
    assert r.json()["expected_demand"] >= 0

    r = client.post("/optimize-price", json={"context": CTX, "objective": "profit"})
    body = r.json()
    assert r.status_code == 200
    cost = CTX["cost"]
    assert cost * settings.price_min_mult - 1e-6 <= body["recommended_price"]
    assert body["recommended_price"] <= cost * settings.price_max_mult + 1e-6


def test_invalid_objective_rejected(tmp_path):
    client = TestClient(create_app(_registry_with_model(tmp_path)))
    r = client.post("/optimize-price", json={"context": CTX, "objective": "bogus"})
    assert r.status_code == 422


def test_input_validation_rejects_bad_context(tmp_path):
    client = TestClient(create_app(_registry_with_model(tmp_path)))
    bad = dict(CTX, cost=-5.0)  # negative cost violates the schema
    r = client.post("/predict-demand", json={"context": bad, "price": 40.0})
    assert r.status_code == 422
