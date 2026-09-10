"""XGBoost demand model: predicts expected units sold given price and context.

The model treats price as one input feature alongside the observable context. It is
trained only on noisy observed samples; it never sees the latent true demand function.
The optimizer then queries this model across candidate prices to choose a price.
"""
from __future__ import annotations

import numpy as np
import xgboost as xgb

from .config import settings
from .data import FEATURE_COLUMNS

# The model's input columns: price first, then the observable context features.
MODEL_COLUMNS = ["price", *FEATURE_COLUMNS]


def build_design_matrix(context_X: np.ndarray, price: np.ndarray) -> np.ndarray:
    """Assemble the [price, *context] design matrix the model expects."""
    price = np.asarray(price, dtype=float).reshape(-1, 1)
    return np.hstack([price, context_X])


class DemandModel:
    def __init__(self, cfg=settings):
        self.cfg = cfg
        self.model = xgb.XGBRegressor(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            objective="reg:squarederror",
            random_state=cfg.seed,
            n_jobs=0,
        )
        self.columns = MODEL_COLUMNS

    def fit(self, context_X: np.ndarray, price: np.ndarray, demand: np.ndarray) -> DemandModel:
        # Train in log space; demand is positive and heavy-tailed, so log-targets stabilize
        # the loss and keep predictions positive after inverse-transform.
        design = build_design_matrix(context_X, price)
        self.model.fit(design, np.log1p(demand))
        return self

    def predict(self, context_X: np.ndarray, price: np.ndarray) -> np.ndarray:
        design = build_design_matrix(context_X, price)
        return np.expm1(self.model.predict(design)).clip(0.0)

    def feature_importance(self) -> dict[str, float]:
        imp = self.model.feature_importances_
        return {c: float(round(v, 4)) for c, v in zip(self.columns, imp)}
