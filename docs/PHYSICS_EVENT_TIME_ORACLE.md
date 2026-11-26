# Physics event-time oracle diagnostic

`run_physics_event_time_oracle.py` checks whether the nominal physics model has the correct **event-time likelihood** under the simulator's actual time-varying generator.

The benchmark NLL currently treats the hazard at the reference time as constant over the forecast horizon:

\[
L = t_e\sum_c \lambda_c(t_0)-\log\lambda_{c^*}(t_e)
\]

for failures, with the full horizon used for censoring. The simulator, however, uses the nominal physics hazard as a function of time during adaptive Ogata thinning. The diagnostic therefore recomputes

\[
L = \int_{t_0}^{t_e}\sum_c\lambda_c(t)dt-\log\lambda_{c^*}(t_e)
\]

for failures, and the integrated hazard to the censoring horizon for censored edges.

At mismatch strength 0 this is the appropriate generating-process likelihood check. The trajectory interpolation follows the same bracket-start position/velocity rule used by the event generator.

A large gap between the oracle and reference-time NLL means the piecewise-constant benchmark approximation is materially different from the simulator's generating likelihood. A small gap means the benchmark approximation is a good representation of the generator over the chosen horizon.
