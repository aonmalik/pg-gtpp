# Integrated event-time benchmark

This benchmark evaluates the physics offset and GBT residual using the time-varying nominal physics hazard integrated over each observed event/censoring interval. The learned residual ratio is computed from the observed history and held constant during the forecast interval. Residual shrinkage is selected by integrated validation NLL. Training exposure uses observed event time for failures and full lookahead for censored rows.

This removes the 30-s reference-time piecewise-constant compensator approximation from the primary likelihood comparison.
