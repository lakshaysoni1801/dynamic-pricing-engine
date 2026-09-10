"""Dynamic pricing and revenue optimization engine: an XGBoost demand model plus a
profit-maximizing price optimizer, with a model registry and a serving API."""
from .config import settings
from .data import generate_market
from .demand_model import DemandModel
from .optimizer import PricingDecision, optimize_prices
from .registry import ModelRegistry

__all__ = [
    "DemandModel",
    "ModelRegistry",
    "PricingDecision",
    "generate_market",
    "optimize_prices",
    "settings",
]
__version__ = "0.1.0"
