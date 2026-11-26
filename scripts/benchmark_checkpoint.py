"""Crash-safe, resumable persistence helpers for PG-GTPP fair benchmark."""
from __future__ import annotations
import json, os, re, time
from pathlib import Path
from typing import Any

def atomic_json_dump(obj: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}-{time.time_ns()}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, allow_nan=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def load_json(path: Path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default

def save_run_result(out_dir: Path, model_name: str, seed: int, result: dict, config=None, validation=None):
    path = Path(out_dir) / "runs" / str(model_name) / f"seed_{int(seed)}.json"
    atomic_json_dump({
        "model": str(model_name),
        "seed": int(seed),
        "status": "completed",
        "saved_at_unix": time.time(),
        "config": config or {},
        "result": result,
        "validation": validation,
    }, path)
    return path

def save_run_failure(out_dir: Path, model_name: str, seed: int, exc: BaseException, config=None):
    path = Path(out_dir) / "runs" / str(model_name) / f"seed_{int(seed)}.failed.json"
    atomic_json_dump({
        "model": str(model_name),
        "seed": int(seed),
        "status": "failed",
        "saved_at_unix": time.time(),
        "config": config or {},
        "error_type": type(exc).__name__,
        "error": str(exc),
    }, path)
    return path

def completed_runs(out_dir: Path):
    root = Path(out_dir) / "runs"
    found = {}
    if not root.exists():
        return found
    for p in root.glob("*/seed_*.json"):
        if p.name.endswith(".failed.json"):
            continue
        m = re.fullmatch(r"seed_(\d+)\.json", p.name)
        if not m:
            continue
        rec = load_json(p)
        if isinstance(rec, dict) and rec.get("status") == "completed":
            model = str(rec.get("model", p.parent.name))
            seed = int(rec.get("seed", m.group(1)))
            found.setdefault(model, {})[seed] = {"result": rec.get("result", {}), "validation": rec.get("validation")}
    return found

def write_progress(out_dir: Path, models, seeds, completed, config=None, status=None):
    total = len(models) * len(seeds)
    done = sum(len(v) for v in completed.values())
    atomic_json_dump({
        "status": status or ("completed" if done == total else "running"),
        "updated_at_unix": time.time(),
        "completed_runs": done,
        "total_runs": total,
        "models": list(models),
        "seeds": [int(s) for s in seeds],
        "completed": {str(m): sorted(int(s) for s in d) for m, d in completed.items()},
        "config": config or {},
    }, Path(out_dir) / "progress.json")
