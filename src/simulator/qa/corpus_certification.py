"""
qa/corpus_certification.py

Aggregates per-mission ``CertificationReport``s (from ``certification_report.py``)
into a single corpus-level QA summary — the missing piece from Phase 4, which
only ever certified one mission at a time. This is what the original QA
design document's Phase 8 ("Dataset Certification Report") actually wanted:
Pass/Warning/Fail counts and aggregate diagnostics across a whole generated
dataset, not just a single hand-picked mission.

Design decisions
-----------------
1. **This module does not generate or certify missions itself** — exactly
   the same separation of concerns as ``certification_report.py``: it
   accepts a list of already-computed ``(mission_id, CertificationReport)``
   pairs and pure-aggregates them. The generation loop (which does need to
   run the simulator and all three QA checks per mission) lives in
   ``scripts/generate_and_certify_batch.py``, kept separate so the
   aggregation logic itself stays trivially unit-testable without needing
   to actually simulate anything.
2. **Corpus-level verdict, not just a count.** A corpus is only as
   trustworthy as its worst missions — a single FAIL anywhere means the
   corpus cannot be certified as a whole, mirroring the per-mission
   decision rule's own "any FAIL condition dominates" structure. A corpus
   FAIL propagates from any mission FAIL; a corpus WARNING triggers if the
   *rate* of per-mission WARNINGs exceeds a tolerance (missions are allowed
   to individually warn sometimes — e.g. one unlucky high-ambiguity
   mission — without that alone invalidating the whole dataset), matching
   how the per-edge time-rescaling test itself already treats "some
   individual rejections are expected by chance" as the correct baseline
   rather than zero-tolerance.
3. **Aggregate diagnostics are reported as distributions (mean + max), not
   just a single pooled number** — pooling the raw ambiguous-fraction or
   correlation values across missions would repeat the exact methodological
   mistake Phase 4 found and fixed in the time-rescaling test (treating
   dependent-ish, mission-level summary statistics as if they were one
   large i.i.d. sample). Reporting the distribution across missions (mean,
   max, and which mission was worst) avoids re-introducing that error while
   still giving an actionable, corpus-wide picture.

Data structures
----------------
``CorpusCertificationResult`` — the Pass/Warning/Fail counts, the
identifying mission for the worst case on each diagnostic, and the
distributional summaries.

Storage implications
---------------------
None — this module only aggregates already-computed, in-memory results.
"""

from __future__ import annotations

from dataclasses import dataclass

from .certification_report import CertificationReport, CertificationVerdict


@dataclass
class CorpusCertificationResult:
    """Corpus-level aggregation of per-mission certification reports."""

    n_missions: int
    n_pass: int
    n_warning: int
    n_fail: int
    failing_mission_ids: list[str]
    mean_ambiguous_fraction: float
    max_ambiguous_fraction: float
    worst_ambiguous_mission_id: str | None
    mean_max_correlation: float
    max_max_correlation: float
    worst_correlation_mission_id: str | None
    total_events: int
    verdict: CertificationVerdict
    reasons: list[str]

    def pass_rate(self) -> float:
        return self.n_pass / self.n_missions if self.n_missions > 0 else float("nan")

    def summary(self) -> str:
        lines = [
            f"=== Corpus Certification: {self.verdict.value.upper()} ===",
            f"Missions: {self.n_missions} total — {self.n_pass} PASS, "
            f"{self.n_warning} WARNING, {self.n_fail} FAIL "
            f"(pass rate {self.pass_rate():.1%})",
            f"Total events across corpus: {self.total_events}",
            f"Ambiguous-event fraction: mean={self.mean_ambiguous_fraction:.3f}, "
            f"max={self.max_ambiguous_fraction:.3f} (mission {self.worst_ambiguous_mission_id})",
            f"Max off-diagonal correlation: mean={self.mean_max_correlation:.3f}, "
            f"max={self.max_max_correlation:.3f} (mission {self.worst_correlation_mission_id})",
        ]
        if self.failing_mission_ids:
            lines.append(f"FAILING missions: {', '.join(self.failing_mission_ids)}")
        if self.reasons:
            lines += ["", "Reasons:"] + [f"  - {r}" for r in self.reasons]
        return "\n".join(lines)


def certify_corpus(
    mission_reports: list[tuple[str, CertificationReport]],
    event_counts: dict[str, int],
    warning_rate_tolerance: float = 0.2,
) -> CorpusCertificationResult:
    """Aggregate per-mission certification reports into a corpus-level
    verdict.

    Decision rule: any per-mission FAIL makes the corpus FAIL. Otherwise,
    if the fraction of missions with a per-mission WARNING exceeds
    ``warning_rate_tolerance`` (default 20%), the corpus is WARNING.
    Otherwise PASS.
    """
    if not mission_reports:
        raise ValueError("certify_corpus requires at least one mission report")

    n_missions = len(mission_reports)
    n_pass = sum(1 for _, r in mission_reports if r.verdict == CertificationVerdict.PASS)
    n_warning = sum(1 for _, r in mission_reports if r.verdict == CertificationVerdict.WARNING)
    n_fail = sum(1 for _, r in mission_reports if r.verdict == CertificationVerdict.FAIL)
    failing_mission_ids = [mid for mid, r in mission_reports if r.verdict == CertificationVerdict.FAIL]

    ambiguous_fractions = [
        (mid, r.identifiability.ambiguous_fraction)
        for mid, r in mission_reports
        if r.identifiability.n_events > 0
    ]
    correlations = [
        (mid, r.identifiability.max_off_diagonal_correlation)
        for mid, r in mission_reports
        if r.identifiability.n_events > 0
    ]

    if ambiguous_fractions:
        mean_amb = sum(v for _, v in ambiguous_fractions) / len(ambiguous_fractions)
        worst_amb_mid, max_amb = max(ambiguous_fractions, key=lambda kv: kv[1])
    else:
        mean_amb, max_amb, worst_amb_mid = float("nan"), float("nan"), None

    if correlations:
        mean_corr = sum(v for _, v in correlations) / len(correlations)
        worst_corr_mid, max_corr = max(correlations, key=lambda kv: kv[1])
    else:
        mean_corr, max_corr, worst_corr_mid = float("nan"), float("nan"), None

    total_events = sum(event_counts.get(mid, 0) for mid, _ in mission_reports)

    reasons: list[str] = []
    verdict = CertificationVerdict.PASS

    if n_fail > 0:
        verdict = CertificationVerdict.FAIL
        reasons.append(
            f"{n_fail}/{n_missions} mission(s) individually FAILED certification "
            f"({', '.join(failing_mission_ids)}) — a corpus cannot be certified while any "
            f"mission fails outright."
        )

    warning_rate = n_warning / n_missions
    if warning_rate > warning_rate_tolerance and verdict != CertificationVerdict.FAIL:
        verdict = CertificationVerdict.WARNING
        reasons.append(
            f"{n_warning}/{n_missions} missions ({warning_rate:.1%}) individually warned, "
            f"exceeding the {warning_rate_tolerance:.0%} tolerance — this is a corpus-wide "
            f"calibration signal, not just isolated unlucky missions."
        )
    elif n_warning > 0:
        reasons.append(
            f"{n_warning}/{n_missions} missions ({warning_rate:.1%}) individually warned, "
            f"within the {warning_rate_tolerance:.0%} tolerance — treated as expected "
            f"per-mission variation, not a corpus-wide problem."
        )

    if not reasons:
        reasons.append("All missions within acceptance bounds.")

    return CorpusCertificationResult(
        n_missions=n_missions, n_pass=n_pass, n_warning=n_warning, n_fail=n_fail,
        failing_mission_ids=failing_mission_ids,
        mean_ambiguous_fraction=mean_amb, max_ambiguous_fraction=max_amb,
        worst_ambiguous_mission_id=worst_amb_mid,
        mean_max_correlation=mean_corr, max_max_correlation=max_corr,
        worst_correlation_mission_id=worst_corr_mid,
        total_events=total_events, verdict=verdict, reasons=reasons,
    )
