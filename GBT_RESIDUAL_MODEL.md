# Physics-Offset GBT Residual Hazard Model

The non-neural residual benchmark fits four cause-specific histogram gradient-boosting regressors with the nominal physics hazard as an explicit offset:

\[
\lambda_c(x)=\lambda_{phys,c}(x)\exp(f_c(x)).
\]

For a candidate shrinkage scale `s`, validation uses

\[
\lambda_c^{(s)}=\lambda_{phys,c}\exp(s f_c(x)).
\]

## Validation objective

Residual scale selection is based on the **continuous-time competing-risk negative log-likelihood**, not Brier score.

For an at-risk edge censored at exposure `tau`:

\[
\ell = \tau\sum_c\lambda_c.
\]

For an edge whose first failure occurs at time `t` from cause `c*`:

\[
\ell=t\sum_c\lambda_c-\log\lambda_{c*}.
\]

The implementation is `continuous_time_competing_risk_nll()` in `src/simulator/model/gbt_residual.py`.

Observed failures use `edge_event_time_delta`; censored observations use the full lookahead because `edge_event_time_delta` is populated accordingly by the window builder.

The benchmark reports the selected validation NLL and the full scale-search table. Brier, ECE, AUROC, AUPRC and C-index remain secondary evaluation metrics.

At `scale=0`, the GBT model reduces exactly to the physics-only hazard, which provides a useful nested sanity check.
