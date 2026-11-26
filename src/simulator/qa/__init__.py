"""
PG-GTPP simulator QA and verification framework (Phase 4).

Implements, as runnable code, the three highest-priority checks identified
in the QA design document:

- ``time_rescaling_test`` — the time-rescaling theorem goodness-of-fit test
  for the competing-risks/recovery point process (the strongest available
  statistical check of whether the generative hazard model is correctly
  specified).
- ``identifiability_audit`` — the ambiguous-event-fraction metric and the
  pairwise cause-covariate correlation matrix, the practical, measurable
  stand-in for the competing-risks identifiability concern.
- ``leakage_tests`` — the shifted-window regeneration test, a mechanical
  (not statistical) proof that event generation never depends on
  trajectory samples beyond the current bracket.

``certification_report`` ties these into a single Pass/Warning/Fail
certification, matching the QA design document's decision rule, for one
mission at a time. ``corpus_certification`` aggregates many missions'
certification reports into a single corpus-level Pass/Warning/Fail summary
— the "Dataset Certification Report" the QA design document's Phase 8
originally called for.

Everything here operates on already-generated mission artifacts (a
trajectory array, an event log, and the physics primitives that produced
them) — nothing in this package generates new missions.
"""

from __future__ import annotations
