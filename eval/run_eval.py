"""Train the demand model, evaluate the pricing policy against ground truth, register
the model in the local registry, and promote it to production.

    python eval/run_eval.py [--json results.json] [--no-register]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pricing_engine.config import settings
from pricing_engine.registry import ModelRegistry
from pricing_engine.train import evaluate, make_split, train_model


def run(register: bool = True) -> dict:
    t0 = time.perf_counter()
    split = make_split()
    model = train_model(split)
    res = evaluate(split, model)
    res["train_seconds"] = round(time.perf_counter() - t0, 1)
    res["feature_importance"] = model.feature_importance()

    if register:
        reg = ModelRegistry()
        params = {
            "n_estimators": settings.n_estimators,
            "max_depth": settings.max_depth,
            "learning_rate": settings.learning_rate,
        }
        metrics = {k: res[k] for k in ("profit_uplift_pct", "demand_model_mape_pct")}
        mv = reg.register(model, metrics=metrics, params=params)
        reg.promote(mv.version, "production")
        res["registered_version"] = mv.version
    return res


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--json")
    p.add_argument("--no-register", action="store_true")
    a = p.parse_args(argv)

    r = run(register=not a.no_register)
    print("=" * 60)
    print("Dynamic Pricing Engine — evaluation (vs TRUE demand)")
    print("=" * 60)
    print(f"demand model MAPE (held-out) : {r['demand_model_mape_pct']:.2f}%")
    print(f"mean price historical        : {r['mean_price_historical']}")
    print(f"mean price optimized         : {r['mean_price_optimized']}")
    print("-" * 60)
    print(f"PROFIT UPLIFT vs historical  : {r['profit_uplift_pct']:.1f}%")
    print(f"  oracle ceiling (perfect)   : {r['oracle_profit_uplift_pct']:.1f}%")
    print(f"  captured fraction of oracle: {r['captured_fraction_of_oracle']*100:.0f}%")
    print(f"revenue change (side effect) : {r['revenue_change_pct']:.1f}%")
    print("-" * 60)
    print(f"(n_test={r['n_test']}, trained in {r['train_seconds']}s"
          + (f", registered v{r['registered_version']}" if "registered_version" in r else "")
          + ")")
    print("=" * 60)
    if a.json:
        Path(a.json).write_text(json.dumps(r, indent=2))
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
