# ADR 001: Two-stage design (demand model + optimizer), profit as the objective

## Status
Accepted

## Context
"Dynamic pricing" can mean a model that directly outputs a price, or a model that
predicts demand as a function of price which an optimizer then searches. The direct
approach is simpler but opaque: it cannot answer "what would happen at a different
price," which is the whole point of pricing. It also has no natural objective; it just
imitates historical prices.

## Decision
We use two stages:

1. An **XGBoost demand model** predicts expected units as a function of price and context.
2. An **optimizer** searches a price grid per context and selects the price that
   maximizes expected **contribution profit**, `(price - cost) * demand`.

**Why profit, not revenue.** Unconstrained revenue maximization is degenerate under
elastic demand: when elasticity exceeds 1, revenue `price * demand(price)` is decreasing
in price, so the revenue-maximizing price collapses to the lower bound. Contribution
profit has an interior optimum because margin is zero at `price = cost`. Profit is the
objective real retail pricing systems use, so it is the default and the headline metric.
Revenue is still reported as a secondary number and is selectable in the optimizer.

## Consequences
The demand model is inspectable (we can plot the demand curve and read off elasticity),
the optimizer is a transparent grid search with explicit bounds, and the objective is
economically meaningful. The cost is two components to maintain instead of one, and a
dependence on the demand model being well-calibrated in the price region the optimizer
explores, which the evaluation is designed to test honestly (see ADR 002).
