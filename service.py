"""FastAPI serving layer.

Exposes the production demand model and the optimizer over HTTP:

  GET  /health                 -> liveness + which model version is serving
  POST /predict-demand         -> expected demand at a given price and context
  POST /optimize-price         -> revenue/profit-maximizing price for a context

The app loads the model promoted to the "production" alias in the registry at startup.
If no model is registered yet, the endpoints return 503 until one is promoted, so the
service degrades safely rather than serving a random model.
"""
from __future__ import annotations

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .data import FEATURE_COLUMNS
from .optimizer import optimize_prices
from .registry import ModelRegistry


class Context(BaseModel):
    cost: float = Field(gt=0)
    competitor_price: float = Field(gt=0)
    segment: int = Field(ge=0, le=2)
    season: float = Field(ge=0, le=1)
    on_promo: int = Field(ge=0, le=1)
    base_popularity: float = Field(gt=0)

    def to_row(self) -> np.ndarray:
        # Build the row by looking values up by name against FEATURE_COLUMNS (the single
        # source of truth for column order, defined in data.py) rather than hardcoding a
        # second, independent field order here. Previously this method listed
        # self.cost, self.competitor_price, ... by hand, which meant reordering or
        # extending FEATURE_COLUMNS in data.py would NOT break anything loudly here -
        # the model would just silently receive columns in the wrong position, since
        # XGBoost has no notion of feature names, only column position. Deriving the row
        # from FEATURE_COLUMNS makes that class of bug structurally impossible instead of
        # relying on the two lists being kept in sync by convention.
        values = self.model_dump()
        try:
            ordered = [values[col] for col in FEATURE_COLUMNS]
        except KeyError as e:
            # Fails loudly at request time instead of silently mispricing if Context's
            # fields and FEATURE_COLUMNS ever diverge (e.g. a new feature added to one
            # but not the other).
            raise RuntimeError(
                f"Context is missing field '{e.args[0]}' required by FEATURE_COLUMNS; "
                "Context and data.FEATURE_COLUMNS have drifted out of sync."
            ) from e
        return np.array([ordered], dtype=float)


class PredictDemandRequest(BaseModel):
    context: Context
    price: float = Field(gt=0)


class OptimizeRequest(BaseModel):
    context: Context
    objective: str = "profit"


def create_app(registry: ModelRegistry | None = None) -> FastAPI:
    app = FastAPI(title="Pricing Engine", version="0.1.0")
    reg = registry or ModelRegistry()
    state: dict = {"model": None, "version": None}

    @app.get("/health")
    def health() -> dict:
        _ensure_loaded()
        return {"status": "ok", "model_version": state["version"]}

    def _ensure_loaded() -> None:
        version = reg.resolve("production")
        if version is not None and version != state["version"]:
            state["model"] = reg.load(version=version)
            state["version"] = version

    def _require_model():
        if state["model"] is None:
            _ensure_loaded()
        if state["model"] is None:
            raise HTTPException(status_code=503, detail="no production model promoted")
        return state["model"]

    @app.post("/predict-demand")
    def predict_demand(req: PredictDemandRequest) -> dict:
        model = _require_model()
        demand = float(model.predict(req.context.to_row(), np.array([req.price]))[0])
        return {"price": req.price, "expected_demand": round(demand, 3)}

    @app.post("/optimize-price")
    def optimize_price(req: OptimizeRequest) -> dict:
        model = _require_model()
        if req.objective not in ("profit", "revenue"):
            raise HTTPException(status_code=422, detail="objective must be profit or revenue")
        row = req.context.to_row()
        decision = optimize_prices(model, row, np.array([req.context.cost]),
                                   objective=req.objective)
        return {
            "recommended_price": round(float(decision.price[0]), 2),
            "expected_demand": round(float(decision.expected_demand[0]), 3),
            "expected_profit": round(float(decision.expected_profit[0]), 3),
            "objective": req.objective,
            "model_version": state["version"],
        }

    return app


app = create_app()
