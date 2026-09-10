"""Central configuration, overridable via environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


@dataclass(frozen=True)
class Settings:
    n_samples: int = _int("PE_N_SAMPLES", 20000)
    seed: int = _int("PE_SEED", 11)
    test_frac: float = _float("PE_TEST_FRAC", 0.2)

    # price search bounds for the optimizer, as multiples of unit cost
    price_min_mult: float = _float("PE_PRICE_MIN_MULT", 1.05)
    price_max_mult: float = _float("PE_PRICE_MAX_MULT", 3.0)
    price_grid_steps: int = _int("PE_PRICE_GRID_STEPS", 60)

    # XGBoost
    n_estimators: int = _int("PE_N_ESTIMATORS", 400)
    max_depth: int = _int("PE_MAX_DEPTH", 6)
    learning_rate: float = _float("PE_LEARNING_RATE", 0.05)
    subsample: float = _float("PE_SUBSAMPLE", 0.8)
    colsample_bytree: float = _float("PE_COLSAMPLE", 0.8)

    # where the model registry lives
    registry_dir: str = os.environ.get("PE_REGISTRY_DIR", "artifacts/registry")


settings = Settings()
