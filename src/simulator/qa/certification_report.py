"""
qa/certification_report.py

Combines the time-rescaling theorem test, identifiability audit, and
leakage test into a single Pass/Warning/Fail dataset certification,
matching the decision rule from the QA design document (Phase 8).

Design decisions
-----------------
1. **Decision rule, taken directly from the design document.**
   - FAIL: the leakage test fails, or the time-rescaling test rejects at a
     strict significance level (p < 0.01).
   - WARNING: no FAIL condition, but the time-rescaling test rejects at the
     conventional 0.05 level, or the identifiability audit flags a
     correlation risk (> 0.7, per ``IdentifiabilityAuditResult``), or the
     ambiguous-event fraction exceeds a configurable tolerance (default
     0.3 — a QA-framework-recommended, not literature-derived, tolerance;
     see the Phase 4 delivery notes for why this number is a starting
     point, not a validated threshold).
   - PASS: none of the above.
2. **This module does not run the checks itself** — it accepts already-
   computed results from ``time_rescaling_test``, ``identifiability_audit``,
   and ``leakage_tests`` and only implements the combination/decision logic.
   This keeps each check independently runnable and testable (as the QA
   design document requires — "each of which should be individually
   callable from a certification-report generator"), with this module
   purely as the aggregator.

Data structures
----------------
``CertificationVerdict`` — enum-like string values PASS/WARNING/FAIL.
``CertificationReport`` — the three underlying results, the verdict, and
the specific reasons that produced it.

Storage implications
---------------------
The report is small (a handful of scalars plus the three sub-results) and
is meant to be serialized alongside a dataset build, per the design
document — this module provides ``to_dict()`` for exactly that, but does
not itself write to disk (consistent with every other QA module: pure
computation from already-generated artifacts).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .identifiability_audit import IdentifiabilityAuditResult
from .leakage_tests import LeakageTestResult
from .time_rescaling_test import TimeRescalingTestResult


class CertificationVerdict(str, Enum):
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"


@dataclass
class CertificationReport:
    """A single mission's QA certification, combining all three checks."""

    verdict: CertificationVerdict
    reasons: list[str]
    time_rescaling: TimeRescalingTestResult
    identifiability: IdentifiabilityAuditResult
    leakage: LeakageTestResult

    def summary(self) -> str:
        lines = [
            f"=== Mission Certification: {self.verdict.value.upper()} ===",
            "",
            self.leakage.summary(),
            "",
            self.time_rescaling.summary(),
            "",
            self.identifiability.summary(),
        ]
        if self.reasons:
            lines += ["", "Reasons:"] + [f"  - {r}" for r in self.reasons]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "reasons": self.reasons,
            "leakage": {
                "passed": self.leakage.passed,
                "truncate_at_s": self.leakage.truncate_at_s,
                "n_events_in_window": self.leakage.n_events_in_window,
                "n_truncated_events": self.leakage.n_truncated_events,
            },
            "time_rescaling": {
                "n_edges_tested": self.time_rescaling.n_edges_tested,
                "n_edges_rejecting": self.time_rescaling.n_edges_rejecting,
                "fraction_rejecting": self.time_rescaling.fraction_rejecting,
                "binomial_test_p_value": self.time_rescaling.binomial_test_p_value,
                "pooled_ks_p_value_diagnostic_only": self.time_rescaling.pooled_ks_p_value,
            },
            "identifiability": {
                "n_events": self.identifiability.n_events,
                "ambiguous_fraction": self.identifiability.ambiguous_fraction,
                "max_off_diagonal_correlation": self.identifiability.max_off_diagonal_correlation,
                "degenerate_causes": list(self.identifiability.degenerate_causes),
            },
        }


def certify_mission(
    time_rescaling: TimeRescalingTestResult,
    identifiability: IdentifiabilityAuditResult,
    leakage: LeakageTestResult,
    ambiguous_fraction_tolerance: float = 0.3,
    fail_significance: float = 0.01,
    warning_significance: float = 0.05,
) -> CertificationReport:
    """Combine the three QA checks into a single Pass/Warning/Fail
    certification, per this module's decision rule (see docstring, point 1).
    """
    reasons: list[str] = []
    verdict = CertificationVerdict.PASS

    if not leakage.passed:
        verdict = CertificationVerdict.FAIL
        reasons.append("Leakage test FAILED — event generation used information beyond its current bracket.")

    if time_rescaling.n_edges_tested >= 1:
        if time_rescaling.binomial_test_p_value < fail_significance:
            verdict = CertificationVerdict.FAIL
            reasons.append(
                f"Time-rescaling test strongly rejects: {time_rescaling.n_edges_rejecting}/"
                f"{time_rescaling.n_edges_tested} edges individually reject "
                f"(binomial excess-rejection p={time_rescaling.binomial_test_p_value:.4g} < {fail_significance}) "
                f"— the hazard model is likely misspecified, not just numerically approximate."
            )
        elif time_rescaling.binomial_test_p_value < warning_significance:
            if verdict != CertificationVerdict.FAIL:
                verdict = CertificationVerdict.WARNING
            reasons.append(
                f"Time-rescaling test rejects at the conventional level "
                f"(binomial excess-rejection p={time_rescaling.binomial_test_p_value:.4g} < {warning_significance}) "
                f"but not strongly — worth investigating before treating this as settled."
            )
    else:
        reasons.append(
            "Time-rescaling test skipped — insufficient events for a meaningful goodness-of-fit test."
        )

    if identifiability.flags_identifiability_risk():
        if verdict != CertificationVerdict.FAIL:
            verdict = CertificationVerdict.WARNING
        reasons.append(
            f"Identifiability audit flags a correlation risk "
            f"(max off-diagonal correlation = {identifiability.max_off_diagonal_correlation:.3f} > 0.7) "
            f"— the disjoint-covariate assumption behind the competing-risks identifiability "
            f"proposition may not hold cleanly for this dataset."
        )

    if identifiability.degenerate_causes:
        if verdict != CertificationVerdict.FAIL:
            verdict = CertificationVerdict.WARNING
        reasons.append(
            f"Cause(s) {', '.join(identifiability.degenerate_causes)} showed zero variance across all "
            f"events — they never meaningfully contributed to any failure in this mission. Any paper "
            f"claim about this cause's attribution cannot be supported by this dataset as configured."
        )

    if identifiability.n_events > 0 and identifiability.ambiguous_fraction > ambiguous_fraction_tolerance:
        if verdict != CertificationVerdict.FAIL:
            verdict = CertificationVerdict.WARNING
        reasons.append(
            f"Ambiguous-event fraction ({identifiability.ambiguous_fraction:.3f}) exceeds the "
            f"tolerance ({ambiguous_fraction_tolerance}) — a substantial share of cause labels are "
            f"statistically fragile; attribution claims should be scoped accordingly."
        )

    if not reasons:
        reasons.append("All checks within acceptance bounds.")

    return CertificationReport(
        verdict=verdict, reasons=reasons,
        time_rescaling=time_rescaling, identifiability=identifiability, leakage=leakage,
    )
