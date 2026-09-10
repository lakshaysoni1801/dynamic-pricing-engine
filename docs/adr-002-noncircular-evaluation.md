# ADR 002: Non-circular offline evaluation against a known demand function

## Status
Accepted

## Context
The hardest part of a pricing project is not building the optimizer, it is evaluating it
honestly. The tempting shortcut is to score the optimized prices using the same demand
model the optimizer used to choose them. That is circular: the optimizer picks whatever
prices the model is most optimistic about, so scoring with that model guarantees a large,
meaningless "uplift." This is the single most common way pricing results are inflated.

## Decision
Because the data is synthetic, the true latent demand function is known. We use it as an
oracle for scoring, never for training:

* The demand **model** is trained only on noisy observed samples.
* The **optimizer** chooses prices using that model.
* The **profit those prices actually earn** is computed from the true demand function,
  so model error is penalized exactly as it would be in a real deployment.

We also report two reference points that make the headline number interpretable:

* an **oracle policy** that chooses prices using the true demand function directly (the
  ceiling any model-based policy could reach), and
* the **fraction of the oracle's uplift that the model-based policy captures**.

The historical baseline is deliberately competitive (a blend of cost-plus and competitor
matching with realistic price variation), not a strawman, so the measured uplift reflects
the value of elasticity-aware optimization rather than beating an artificially bad policy.

## Consequences and honesty notes
On the default configuration the model-based policy delivers roughly a 22 percent profit
uplift and captures about 89 percent of a 26 percent oracle ceiling, with the demand model
at about 8 percent MAPE. These numbers are what the code produces and are reproducible
from the seed.

The magnitude of any pricing uplift depends on how good the historical policy was and how
much price variation exists to learn from. We surface the oracle ceiling and the captured
fraction precisely because they are more robust indicators of the method's quality than the
headline percentage, which will move if the market regime changes. A tiny training set can
even produce a policy that *loses* money versus the baseline; that behavior is preserved
rather than hidden, because it is the truth about model-based pricing with insufficient
data.
