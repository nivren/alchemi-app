# SPDX-License-Identifier: Apache-2.0
"""Configuration-driven entrypoint for relaxation and backend comparison."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import tomllib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _load_config(path):
    path = Path(path).expanduser().resolve()
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    config["_config_path"] = str(path)
    return config


def _configured_environment(config):
    env = os.environ.copy()
    for name, value in config.get("environment", {}).items():
        env[str(name)] = str(value)
    source_path = str(PROJECT_ROOT / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = source_path if not existing else f"{source_path}{os.pathsep}{existing}"
    return env


def _status_counts(records):
    counts = {}
    for record in records:
        status = record.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _batch_limits(batch):
    """Return the one explicitly selected active-batch capacity."""
    mode = batch.get("mode", "atoms")
    if mode == "atoms":
        if "batch_size" in batch or "max_batch_size" in batch:
            raise ValueError("batch.mode='atoms' cannot set a structure-count limit")
        if "max_atoms" not in batch:
            raise ValueError("batch.mode='atoms' requires max_atoms")
        max_atoms, max_batch_size = int(batch["max_atoms"]), None
    elif mode == "bsize":
        if "max_atoms" in batch or "max_batch_size" in batch:
            raise ValueError("batch.mode='bsize' cannot set an atom-count limit")
        if "batch_size" not in batch:
            raise ValueError("batch.mode='bsize' requires batch_size")
        max_atoms, max_batch_size = None, int(batch["batch_size"])
    else:
        raise ValueError("batch.mode must be 'atoms' or 'bsize'")
    limit = max_atoms if max_atoms is not None else max_batch_size
    if limit <= 0:
        raise ValueError("the active batch capacity must be positive")
    return mode, max_atoms, max_batch_size


def _load_backend_result(directory):
    timing_path = directory / "timing.json"
    results_path = directory / "results.jsonl"
    if not timing_path.is_file() or not results_path.is_file():
        return None, []
    try:
        timing = json.loads(timing_path.read_text())
        records = [
            json.loads(line)
            for line in results_path.read_text().splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError):
        return None, []
    return timing, records


def _backend_comparison(left, right):
    left_by_id = {record["source_index"]: record for record in left}
    right_by_id = {record["source_index"]: record for record in right}
    shared = sorted(left_by_id.keys() & right_by_id.keys())
    if not shared:
        return {"shared_structures": 0}

    def max_delta(key):
        deltas = []
        for source_id in shared:
            a, b = left_by_id[source_id].get(key), right_by_id[source_id].get(key)
            if a is not None and b is not None:
                deltas.append(abs(float(a) - float(b)))
        return max(deltas) if deltas else None

    return {
        "shared_structures": len(shared),
        "same_input_hashes_for_all": all(
            left_by_id[i].get("source_sha256") == right_by_id[i].get("source_sha256")
            for i in shared
        ),
        "same_status_for_all": all(
            left_by_id[i].get("status") == right_by_id[i].get("status")
            for i in shared
        ),
        "status_mismatches": sum(
            left_by_id[i].get("status") != right_by_id[i].get("status")
            for i in shared
        ),
        "max_abs_step_delta": max(
            abs(int(left_by_id[i].get("steps", 0)) - int(right_by_id[i].get("steps", 0)))
            for i in shared
        ),
        "same_step_count_for_all": all(
            left_by_id[i].get("steps") == right_by_id[i].get("steps")
            for i in shared
        ),
        "max_abs_energy_delta_eV": max_delta("energy_eV"),
        "max_abs_final_fmax_delta_eV_A": max_delta("fmax_eV_A"),
        "max_abs_stress_delta_GPa": max_delta("max_abs_stress_GPa"),
    }


def _validate_batch_config(config):
    run = config["run"]
    inputs = config["inputs"]
    batch = config["batch"]
    _, max_atoms, _ = _batch_limits(batch)
    optimizer = config["optimizer"]
    neighbors = config["neighbors"]
    backends = run.get("inference_backends", ["e3nn", "fasteq"])
    if (
        not isinstance(backends, list)
        or not backends
        or len(backends) != len(set(backends))
        or not set(backends) <= {"e3nn", "fasteq"}
    ):
        raise ValueError(
            "batch_compare requires a nonempty unique inference_backends subset of "
            "['e3nn', 'fasteq']"
        )
    if optimizer.get("name") != "fire":
        raise ValueError("the reproduced MPA-0 benchmark is locked to FIRE")
    neighbor_backend = neighbors.get("backend", "torch_reference")
    if neighbor_backend not in {"torch_reference", "hip"}:
        raise ValueError(
            "neighbors.backend must be 'torch_reference' or 'hip'"
        )
    if neighbor_backend == "hip" and float(neighbors.get("skin_A", 0.0)) != 0.0:
        raise ValueError("the explicit HIP neighbor backend requires skin_A = 0")
    if float(neighbors.get("cutoff_A", 6.0)) != 6.0:
        raise ValueError("the locked MPA-0 checkpoint uses a 6 A neighbor cutoff")
    if run.get("dtype", "float64") != "float64":
        raise ValueError("the reproduced batch comparison requires float64")
    selection_path = _resolve_path(inputs["selection"])
    if not selection_path.is_file():
        raise FileNotFoundError(selection_path)
    if not _resolve_path(inputs["checkpoint"]).is_file():
        raise FileNotFoundError(_resolve_path(inputs["checkpoint"]))
    selection = json.loads(selection_path.read_text())
    selected = selection.get("batch", selection.get("batch64"))
    if selected is None:
        raise ValueError("selection manifest has no batch or batch64 entries")
    if max_atoms is not None and any(
        int(item["natoms"]) > max_atoms for item in selected
    ):
        raise ValueError("an input structure exceeds the active max_atoms capacity")


def _run_batch_comparison(config):
    _validate_batch_config(config)
    run = config["run"]
    inputs = config["inputs"]
    batch = config["batch"]
    _, max_atoms, max_batch_size = _batch_limits(batch)
    optimizer = config["optimizer"]
    convergence = config["convergence"]
    neighbors = config["neighbors"]
    output_root = _resolve_path(run["output_dir"])
    output_root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config["_config_path"], output_root / "parameters.toml")
    log_dir = output_root / "logs"
    log_dir.mkdir()
    env = _configured_environment(config)
    backend_runs = {}

    for backend in run.get("inference_backends", ["e3nn", "fasteq"]):
        backend_dir = output_root / backend
        log_path = log_dir / f"{backend}.log"
        command = [
            sys.executable,
            "-m",
            "alchemi_app.benchmark_mpa0_batch",
            "--selection",
            str(_resolve_path(inputs["selection"])),
            "--checkpoint",
            str(_resolve_path(inputs["checkpoint"])),
            "--output",
            str(backend_dir),
            "--device",
            str(run.get("device", "cuda:0")),
            "--inference-backend",
            backend,
            "--neighbor-backend",
            str(neighbors.get("backend", "torch_reference")),
        ]
        values = {
            "refill-frequency": batch["refill_frequency"],
            "max-updates-per-structure": batch["max_updates_per_structure"],
            "wall-seconds": run["wall_seconds"],
            "components": None,
            "force-tolerance-eV-A": convergence["force_tolerance_eV_A"],
            "force-overflow-eV-A": convergence["force_overflow_eV_A"],
            "dt": optimizer["dt"],
            "dt-max": optimizer["dt_max"],
            "dt-min": optimizer["dt_min"],
            "maxstep": optimizer["maxstep"],
            "n-min": optimizer["n_min"],
            "f-inc": optimizer["f_inc"],
            "f-dec": optimizer["f_dec"],
            "alpha-start": optimizer["alpha_start"],
            "f-alpha": optimizer["f_alpha"],
            "skin-A": neighbors["skin_A"],
            "max-neighbors-per-atom": neighbors["max_neighbors_per_atom"],
        }
        if max_atoms is not None:
            values["max-atoms"] = max_atoms
        else:
            values["max-batch-size"] = max_batch_size
        for option, value in values.items():
            if option == "components":
                if run.get("components", False):
                    command.append("--components")
            else:
                command.extend((f"--{option}", str(value)))

        started = time.perf_counter()
        timed_out = False
        try:
            with log_path.open("w") as log:
                completed = subprocess.run(
                    command,
                    cwd=PROJECT_ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=float(run.get("process_timeout_seconds", 1920)),
                    check=False,
                )
            returncode = completed.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = None
        wall_seconds = time.perf_counter() - started
        timing, records = _load_backend_result(backend_dir)
        backend_runs[backend] = {
            "command": command,
            "returncode": returncode,
            "timed_out": timed_out,
            "process_wall_seconds": wall_seconds,
            "log": str(log_path),
            "timing": timing,
            "record_count": len(records),
            "status_counts": _status_counts(records),
            "records": records,
        }

    e3nn, fasteq = backend_runs.get("e3nn", {}), backend_runs.get("fasteq", {})
    comparison = {}
    if e3nn.get("timing") and fasteq.get("timing"):
        comparison = {
            "process_wall_speedup_e3nn_over_fasteq": (
                e3nn["process_wall_seconds"] / fasteq["process_wall_seconds"]
                if fasteq["process_wall_seconds"] > 0
                else None
            ),
            "application_speedup_e3nn_over_fasteq": (
                e3nn["timing"]["application_seconds"]
                / fasteq["timing"]["application_seconds"]
                if fasteq["timing"]["application_seconds"] > 0
                else None
            ),
            "dynamics_speedup_e3nn_over_fasteq": (
                e3nn["timing"]["dynamics_seconds"]
                / fasteq["timing"]["dynamics_seconds"]
                if fasteq["timing"]["dynamics_seconds"] > 0
                else None
            ),
            "relaxation_result_comparison": _backend_comparison(
                e3nn["records"], fasteq["records"]
            ),
        }
    complete = all(
        backend_run.get("returncode") == 0
        and (backend_run.get("timing") or {}).get("status") == "complete"
        and backend_run.get("record_count")
        == (backend_run.get("timing") or {}).get("structures")
        for backend_run in backend_runs.values()
    )
    summary = {
        "status": "complete" if complete else "incomplete",
        "config": str(Path(config["_config_path"])),
        "configuration_snapshot": "parameters.toml",
        "backend_runs": {
            backend: {key: value for key, value in record.items() if key != "records"}
            for backend, record in backend_runs.items()
        },
        "comparison": comparison,
    }
    (output_root / "comparison.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(summary, allow_nan=False), flush=True)
    if summary["status"] != "complete":
        raise RuntimeError(f"one or more backend runs failed; see {output_root}")
    return summary


def _run_single_relaxation(config):
    run = config["run"]
    inputs = config["inputs"]
    settings = config["relaxation"]
    _, max_atoms, max_batch_size = _batch_limits(config["batch"])
    env = _configured_environment(config)
    os.environ.update(
        {name: value for name, value in env.items() if os.environ.get(name) != value}
    )
    import torch

    from .relax_mace_mpa0 import relax

    input_dir = _resolve_path(inputs["input_dir"])
    paths = sorted(input_dir.glob("*.cif"))
    limit = settings.get("limit")
    if limit is not None:
        paths = paths[: int(limit)]
    backend = settings.get("inference_backend", "e3nn")
    if backend not in ("e3nn", "fasteq"):
        raise ValueError("relaxation.inference_backend must be e3nn or fasteq")
    output_dir = _resolve_path(run["output_dir"])
    records = relax(
        paths,
        _resolve_path(inputs["checkpoint"]),
        output_dir,
        optimizer=settings.get("optimizer", "fire2"),
        device=run.get("device", "cpu"),
        dtype=getattr(torch, settings.get("dtype", "float64")),
        max_batch_size=max_batch_size,
        max_atoms=max_atoms,
        max_steps=int(settings.get("max_steps", 500)),
        max_wall_seconds=float(settings.get("max_wall_seconds", 120)),
        dt=float(settings.get("dt", 0.02)),
        fmax=float(settings.get("fmax_eV_A", 0.01)),
        stress_gpa=float(settings.get("stress_gpa", 0.1)),
        skin=float(settings.get("skin_A", 0.0)),
        fasteq=backend == "fasteq",
    )
    shutil.copy2(config["_config_path"], output_dir / "parameters.toml")
    print(json.dumps({"total": len(records), "status_counts": _status_counts(records)}))
    return records


def run_config(config):
    mode = config["run"].get("mode", "relax")
    if mode == "batch_compare":
        return _run_batch_comparison(config)
    if mode == "relax":
        return _run_single_relaxation(config)
    raise ValueError("run.mode must be 'relax' or 'batch_compare'")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run_config(_load_config(args.config))


if __name__ == "__main__":
    main()
