"""
PG-GTPP model foundation (Phase 5).

Implements the three-component architecture from the PG-GTPP v2 research
design: closed-form physics base intensities (reused unchanged from Phase 3's
``physics_intensity.py`` — no new physics code here), a single small
heterogeneous graph transformer encoder, and a per-cause evidential-style
trust-gated fusion of the physics and neural terms. A mark decoder (PDR,
throughput, delay) with distribution-appropriate heads completes the model.

No training loop, no loss computation, no checkpointing lives here — this
package is model architecture only. See ``dataset/`` for the tensor
pipeline that turns simulator artifacts into this model's input format.
"""

from __future__ import annotations
