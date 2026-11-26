"""
training/train.py

The training loop: iterates windows sampled across a corpus of generated
missions, computes the combined TPP+mark loss, steps the optimizer, and
checkpoints. No mission generation happens here -- this module trains
against already-generated, in-memory mission artifacts.

Design decisions
-----------------
1. In-memory mission cache, not a disk-backed streaming loader. Phase 6's
   scope is "does the training loop work and does loss go down on a real
   corpus" -- the storage-efficient streaming MissionDataStore loader
   (Phase 1) is the natural production data source, but wiring a full
   streaming dataset/dataloader here would mix a data-engineering concern
   into this phase's training-loop concern. MissionCache is a simple,
   explicit in-memory list of mission artifacts (small corpora only) and
   is deliberately not presented as a corpus-scale training solution.
2. Random window sampling, not exhaustive enumeration -- each training
   step samples a batch of (mission, reference_t_idx) pairs uniformly at
   random, matching "streaming window generation" (materialize windows
   transiently, never enumerate/store them).
3. Checkpointing saves model state, optimizer state, and step count in one
   torch.save call, so training can resume exactly.
4. No validation split, no early stopping, no LR schedule -- intentionally
   deferred; Phase 6's scope is a working, loss-decreasing training loop,
   not a tuned training recipe.

Data structures
----------------
MissionCache -- holds one or more missions' full artifacts in memory.
TrainConfig -- plain dataclass of training hyperparameters.
train -- the loop itself.

Storage implications
---------------------
Checkpoints are the only persisted artifact, written only where the caller
specifies.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import torch

from ..dataset.window_builder import WindowBatch, build_window, collate_windows
from ..event_generator import EventGenerator
from ..mission_config import MissionConfig
from ..mission_generator import MissionGenerator
from ..mission_state import NodeType
from ..physics import AcousticChannel, CurrentField, WaveField
from ..physics_intensity import HazardEvaluator
from ..random_manager import RandomManager
from ..trajectory_generator import TrajectoryGenerator
from .loss import TPPLossOutput, compute_loss
from ..model.pg_gtpp_model import PGGTPPModel


@dataclass
class MissionArtifacts:
    mission_id: str
    array: np.ndarray
    node_ids: list[str]
    node_types: dict[str, NodeType]
    timestamps: np.ndarray
    events: list
    evaluator: HazardEvaluator
    candidate_range_factor: float


@dataclass
class MissionCache:
    """An in-memory collection of generated mission artifacts, used as the
    training loop's data source (see module docstring, point 1)."""

    missions: list[MissionArtifacts] = field(default_factory=list)

    @property
    def mission_ids(self) -> list[str]:
        """Mirrors StreamingMissionSource.mission_ids, so code that only
        needs to know which missions are present (not their full
        artifacts) can treat either data source identically."""
        return [m.mission_id for m in self.missions]

    @classmethod
    def generate(
        cls,
        config: MissionConfig,
        master_seed: int,
        n_missions: int,
        truth_hazard_multipliers: tuple[float, float, float, float] | None = None,
        unmodeled_state_strength: float = 0.0,
        unmodeled_state_mode: str = "cause_specific",
    ) -> "MissionCache":
        """Generate missions, optionally separating *truth* hazards from the
        model-visible physics hazards.

        ``truth_hazard_multipliers`` is used only by controlled robustness
        experiments: trajectories and the model-visible ``HazardEvaluator``
        remain unchanged, while the event generator uses a copied
        ``HazardConfig`` with cause-specific multipliers.  This creates a
        controlled physics-model misspecification without changing the
        observed covariates or the mission split.
        """
        if unmodeled_state_strength < 0.0:
            raise ValueError("unmodeled_state_strength must be >= 0")
        if truth_hazard_multipliers is not None:
            if len(truth_hazard_multipliers) != 4 or any(float(x) < 0.0 for x in truth_hazard_multipliers):
                raise ValueError(
                    "truth_hazard_multipliers must contain four non-negative values "
                    "(geo,wave,doppler,current)"
                )
            base_h = config.hazard
            truth_hazard = replace(
                base_h,
                kappa_geo=base_h.kappa_geo * float(truth_hazard_multipliers[0]),
                kappa_wave=base_h.kappa_wave * float(truth_hazard_multipliers[1]),
                kappa_doppler=base_h.kappa_doppler * float(truth_hazard_multipliers[2]),
                kappa_current=base_h.kappa_current * float(truth_hazard_multipliers[3]),
            )
        else:
            truth_hazard = config.hazard
        rm = RandomManager(master_seed=master_seed)
        cache = cls()
        for i in range(n_missions):
            mission_id = f"train_mission_{i:04d}"
            state = MissionGenerator(config, rm).generate(mission_id)
            array, node_ids = TrajectoryGenerator(config, rm).propagate(state)
            wf = WaveField.sample(config.wave, rm.get_numpy_rng(mission_id, "wave"))
            cf = CurrentField.sample(config.current, rm.get_numpy_rng(mission_id, "current"))
            ch = AcousticChannel(config.acoustic)
            node_types = {n.node_id: n.node_type for n in state.nodes}
            timestamps = np.arange(array.shape[0], dtype=np.float64) / config.envelope.node_sample_rate_hz
            truth_config = replace(config, hazard=truth_hazard)
            events = EventGenerator(truth_config, rm, ch, wf, cf).generate(
                mission_id, array, node_ids, node_types, timestamps,
                unmodeled_state_strength=unmodeled_state_strength,
                unmodeled_state_mode=unmodeled_state_mode,
            )
            evaluator = HazardEvaluator(ch, wf, cf, config.hazard)
            cache.missions.append(
                MissionArtifacts(
                    mission_id=mission_id, array=array, node_ids=node_ids, node_types=node_types,
                    timestamps=timestamps, events=events, evaluator=evaluator,
                    candidate_range_factor=config.graph.candidate_range_factor,
                )
            )
        return cache

    @staticmethod
    def _valid_reference_indices(mission: MissionArtifacts, lookahead_s: float) -> list[int]:
        """Indices with a complete forecast horizon inside the mission."""
        if lookahead_s <= 0:
            raise ValueError("lookahead_s must be > 0")
        end = float(mission.timestamps[-1])
        return [i for i, t in enumerate(mission.timestamps) if float(t) + lookahead_s <= end + 1e-9]

    def sample_batch(
        self, rng: np.random.Generator, batch_size: int, window_length: int, lookahead_s: float
    ) -> WindowBatch:
        """Sample batch_size random (mission, reference_t_idx) pairs and
        build a collated batch from them."""
        samples = []
        for _ in range(batch_size):
            mission = self.missions[rng.integers(0, len(self.missions))]
            valid = self._valid_reference_indices(mission, lookahead_s)
            if not valid:
                raise ValueError(f"mission {mission.mission_id} is shorter than lookahead_s={lookahead_s}")
            t_idx = int(valid[rng.integers(0, len(valid))])
            samples.append(
                build_window(
                    mission.mission_id, mission.array, mission.node_ids, mission.node_types,
                    mission.timestamps, mission.events, mission.evaluator, mission.candidate_range_factor,
                    reference_t_idx=t_idx, window_length=window_length, lookahead_s=lookahead_s,
                )
            )
        return collate_windows(samples)

    def split(self, val_fraction: float, seed: int) -> tuple["MissionCache", "MissionCache"]:
        """Partition this cache into disjoint train/val caches **at the
        mission level**, not the window level.

        Splitting at the window level would leak information: windows from
        the same mission share one realized wave/current field and one
        realized event sequence, so a model could partly "memorize" a
        mission's specific hazard pattern from windows seen in training and
        get credit for it on held-out windows of the *same* mission. This
        mirrors the exact leakage concern the QA framework's shifted-window
        regeneration test was built to catch at the simulator level (Phase
        4) -- here it is the analogous concern at the train/val-split level.
        """
        if not (0.0 < val_fraction < 1.0):
            raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
        if len(self.missions) < 2:
            raise ValueError(f"split() requires at least 2 missions, got {len(self.missions)}")

        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(self.missions))
        n_val = max(1, int(round(len(self.missions) * val_fraction)))
        val_indices = set(indices[:n_val].tolist())

        train_cache = MissionCache(missions=[m for i, m in enumerate(self.missions) if i not in val_indices])
        val_cache = MissionCache(missions=[m for i, m in enumerate(self.missions) if i in val_indices])
        return train_cache, val_cache

    def all_windows(
        self, window_length: int, lookahead_s: float, stride: int = 1
    ) -> WindowBatch:
        """Build a batch from EVERY (mission, reference_t_idx) pair across
        the cache (subsampled by ``stride`` along the time axis) -- used for
        deterministic, reproducible validation-set evaluation, unlike
        ``sample_batch``'s random sampling (appropriate for training, not
        for a metric you want to be exactly comparable across runs).
        """
        samples = []
        for mission in self.missions:
            valid = self._valid_reference_indices(mission, lookahead_s)
            for t_idx in valid[::stride]:
                samples.append(
                    build_window(
                        mission.mission_id, mission.array, mission.node_ids, mission.node_types,
                        mission.timestamps, mission.events, mission.evaluator, mission.candidate_range_factor,
                        reference_t_idx=t_idx, window_length=window_length, lookahead_s=lookahead_s,
                    )
                )
        return collate_windows(samples)


@dataclass
class TrainConfig:
    n_steps: int = 100
    batch_size: int = 8
    window_length: int = 8
    lookahead_s: float = 30.0
    learning_rate: float = 1e-3
    mark_loss_weight: float = 1.0
    log_every: int = 10
    checkpoint_every: int | None = None
    checkpoint_dir: Path | None = None
    seed: int = 0
    lr_schedule: str = "none"  # "none" or "cosine" -- see _build_scheduler
    val_every: int | None = None  # if set (with val_cache passed to train()), evaluate every N steps
    val_stride: int = 5  # subsampling stride for MissionCache.all_windows on the validation set
    restore_best_val: bool = False  # restore checkpoint with best validation AUC at train end


@dataclass
class TrainingHistory:
    steps: list[int] = field(default_factory=list)
    total_losses: list[float] = field(default_factory=list)
    tpp_losses: list[float] = field(default_factory=list)
    mark_losses: list[float] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)
    val_steps: list[int] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    val_event_aucs: list[float] = field(default_factory=list)
    val_calibration_errors: list[float] = field(default_factory=list)


def _build_scheduler(
    optimizer: torch.optim.Optimizer, schedule: str, n_steps: int
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """A single, explicitly-named schedule option ("cosine"), not an
    open-ended set -- matching the project's repeated "avoid unnecessary
    complexity" principle. "none" (the default) keeps the learning rate
    fixed, exactly Phase 6's original behavior, so existing training runs
    are unaffected unless a schedule is explicitly requested.
    """
    if schedule == "none":
        return None
    if schedule == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(n_steps, 1))
    raise ValueError(f"Unknown lr_schedule {schedule!r}; expected 'none' or 'cosine'")


def save_checkpoint(path: Path, model: PGGTPPModel, optimizer: torch.optim.Optimizer, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "step": step}, path
    )


def load_checkpoint(path: Path, model: PGGTPPModel, optimizer: torch.optim.Optimizer) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint["step"]


def train(
    model: PGGTPPModel, cache: "MissionCache | object", config: TrainConfig, val_cache: "MissionCache | object | None" = None
) -> TrainingHistory:
    """Run the training loop, sampling random windows from cache for
    config.n_steps optimizer steps.

    ``cache`` (and ``val_cache``) may be a ``MissionCache`` (in-memory) or
    a ``streaming_dataset.StreamingMissionSource`` (disk-backed) -- both
    expose the same ``mission_ids``/``sample_batch``/``all_windows``
    interface (duck-typed, not enforced via a shared base class, to avoid
    an import cycle between ``train.py`` and ``streaming_dataset.py``,
    which itself imports ``MissionArtifacts`` from this module), so this
    function works unchanged against either.

    If val_cache is provided together with config.val_every, a full,
    deterministic validation pass (all_windows, not random sampling)
    runs every config.val_every steps and is recorded in the returned
    history. val_cache should never overlap with cache's missions (see
    MissionCache.split's/StreamingMissionSource.split's leakage-avoidance
    rationale).
    """
    if not cache.mission_ids:
        raise ValueError("cache has no missions -- generate or point to missions before training")
    if config.val_every and val_cache is None:
        raise ValueError("config.val_every is set but no val_cache was provided")

    rng = np.random.default_rng(config.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = _build_scheduler(optimizer, config.lr_schedule, config.n_steps)
    history = TrainingHistory()
    best_val_auc = float("-inf")
    best_model_state = None

    model.train()
    t_start = time.time()
    for step in range(1, config.n_steps + 1):
        batch = cache.sample_batch(rng, config.batch_size, config.window_length, config.lookahead_s)
        if batch.edge_index.shape[1] == 0:
            continue

        optimizer.zero_grad()
        output = model(batch.node_features, batch.node_type_ids, batch.node_mask, batch.edge_index, batch.edge_physics)
        loss_out: TPPLossOutput = compute_loss(output, batch, mark_loss_weight=config.mark_loss_weight)
        # Physics-only models intentionally have no differentiable learned
        # intensity path.  Their loss is therefore a detached diagnostic
        # quantity, so attempting backward() raises an autograd error.
        # Treat such models as fixed evaluators and skip optimization while
        # keeping the normal training/validation bookkeeping intact.
        if loss_out.total_loss.requires_grad:
            loss_out.total_loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        history.steps.append(step)
        history.total_losses.append(loss_out.total_loss.item())
        history.tpp_losses.append(loss_out.tpp_loss.item())
        history.mark_losses.append(loss_out.mark_loss.item())
        history.learning_rates.append(optimizer.param_groups[0]["lr"])

        if config.log_every and step % config.log_every == 0:
            elapsed = time.time() - t_start
            print(
                f"  step {step}/{config.n_steps}  total={loss_out.total_loss.item():.4f}  "
                f"tpp={loss_out.tpp_loss.item():.4f}  mark={loss_out.mark_loss.item():.4f}  "
                f"lr={optimizer.param_groups[0]['lr']:.2e}  "
                f"n_events={loss_out.n_events}/{loss_out.n_at_risk}  ({elapsed:.1f}s elapsed)"
            )

        if config.val_every and val_cache is not None and step % config.val_every == 0:
            from .eval import evaluate  # local import: eval.py imports loss.py, avoid an import cycle at module load time

            val_batch = val_cache.all_windows(config.window_length, config.lookahead_s, stride=config.val_stride)
            metrics = evaluate(model, val_batch, mark_loss_weight=config.mark_loss_weight)
            model.train()  # evaluate() leaves the model in eval() mode; resume training mode
            history.val_steps.append(step)
            history.val_losses.append(metrics.val_loss)
            history.val_event_aucs.append(metrics.event_auc)
            history.val_calibration_errors.append(metrics.calibration_error)
            if config.restore_best_val and np.isfinite(metrics.event_auc) and metrics.event_auc > best_val_auc:
                best_val_auc = float(metrics.event_auc)
                best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if config.log_every:
                print(
                    f"    [val @ step {step}] loss={metrics.val_loss:.4f}  auc={metrics.event_auc:.3f}  "
                    f"ece={metrics.calibration_error:.3f}  brier={metrics.brier_score:.4f}  n_events={metrics.n_events}/{metrics.n_at_risk}"
                )

        if config.checkpoint_every and config.checkpoint_dir and step % config.checkpoint_every == 0:
            save_checkpoint(config.checkpoint_dir / f"checkpoint_step{step}.pt", model, optimizer, step)

    if config.restore_best_val and best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.train()
    return history
