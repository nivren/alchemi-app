# SPDX-License-Identifier: Apache-2.0
"""CIF relaxation through native FusedStage, scalar FIRE/FIRE2 and inflight refill.

Run with the project environment and PYTHONPATH. This eager reference example
retains inputs and graduated results in host memory; it is not a restart or
large-directory streaming service. Wall limits are checked between steps.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.io import read, write

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import FIRE2VariableCell, FIREVariableCell, FusedStage
from nvalchemi.dynamics.base import ConvergenceHook, DynamicsStage
from nvalchemi.dynamics.sampler import SizeAwareSampler
from nvalchemi.dynamics.sinks import HostMemory
from nvalchemi.hooks import NeighborListHook
from nvalchemi.models.mace import MACEWrapper
from nvalchemi.neighbors import compute_neighbors

MODEL_SHA256 = "75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638"
GPA_TO_EV_A3 = 1 / 160.21766208
_LOADED_FASTEQ_LIBRARIES = set()


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _configure_fasteq_runtime():
    """Select the local HIP code path and load its registered Torch operators."""
    names = (
        "FASTEQ_BACKEND",
        "FASTEQ_CODEGEN_BACKEND",
        "FASTEQ_INFERENCE",
    )
    previous = {name: os.environ.get(name) for name in names}
    os.environ.update(
        FASTEQ_BACKEND="hip",
        FASTEQ_CODEGEN_BACKEND="native",
        FASTEQ_INFERENCE="1",
    )
    try:
        spec = importlib.util.find_spec("fasteq.hip._hip")
        if spec is None or spec.origin is None:
            raise RuntimeError(
                "FastEq HIP operators are missing; install the local newFastEq package"
            )
        import fasteq

        if fasteq._BACKEND != "hip":
            raise RuntimeError(
                f"FastEq selected {fasteq._BACKEND!r}; the HCU path requires 'hip'"
            )
        library = str(Path(spec.origin).resolve())
        if library not in _LOADED_FASTEQ_LIBRARIES:
            torch.ops.load_library(library)
            _LOADED_FASTEQ_LIBRARIES.add(library)
        # Fail here with a clear loader error instead of deep in MACE inference.
        torch.ops.stc_fwd.forward
        torch.ops.stc_bwd.backward
    except Exception:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        raise
    return previous


def _restore_fasteq_runtime(previous):
    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def _load_fasteq_mace(checkpoint, device, dtype):
    """Convert MACE through the installed local cuEquivariance Torch frontend."""
    if device.type != "cuda" or not torch.version.hip:
        raise RuntimeError("--fasteq requires a Hygon HIP build and a cuda device")

    from mace.cli.convert_e3nn_cueq import run as convert_e3nn_to_cueq

    model = torch.load(checkpoint, weights_only=False, map_location=device)
    model.to(dtype=dtype)
    # On Hygon, PyTorch exposes the HIP device through its cuda namespace.
    with torch.cuda.device(device):
        model = convert_e3nn_to_cueq(
            model, return_model=True, device="cuda"
        )
    fast_layers = [
        layer for layer in model.modules() if hasattr(layer, "fast_inference")
    ]
    if not fast_layers or not all(layer.fast_inference for layer in fast_layers):
        raise RuntimeError(
            "the converted MACE model did not activate the local FastEq inference path"
        )
    wrapper = MACEWrapper(model.to(device))
    wrapper.eval()
    return wrapper


class CIFDataset:
    """Keep source indices separate from the sampler's admission-order IDs."""

    def __init__(self, structures, source_indices, dtype, device):
        self.structures = structures
        self.source_indices = source_indices
        self.dtype, self.device = dtype, device

    def __len__(self):
        return len(self.structures)

    def get_metadata(self, i):
        return len(self.structures[i]), 0

    def __getitem__(self, i):
        data = AtomicData.from_atoms(self.structures[i], dtype=self.dtype).to(
            self.device
        )
        data.add_node_property("velocities", torch.zeros_like(data.positions))
        data.forces = torch.zeros_like(data.positions)
        data.energy = torch.zeros(1, 1, dtype=self.dtype, device=self.device)
        data.stress = torch.zeros(1, 3, 3, dtype=self.dtype, device=self.device)
        for name, value in (
            ("source_index", self.source_indices[i]),
            ("relax_steps", 0),
            ("exit_reason", 0),
        ):
            data.add_system_property(
                name, torch.tensor([[value]], dtype=torch.long, device=self.device)
            )
        return data, {}


def validate_atoms(atoms, supported, max_atoms):
    if not len(atoms):
        raise ValueError("empty structure")
    if max_atoms is not None and len(atoms) > max_atoms:
        raise ValueError("atom count exceeds --max-atoms")
    if not atoms.pbc.all():
        raise ValueError("three-dimensional periodicity required")
    if (
        not np.isfinite(atoms.positions).all()
        or not np.isfinite(atoms.cell.array).all()
    ):
        raise ValueError("nonfinite coordinates or cell")
    if abs(np.linalg.det(atoms.cell.array)) <= 1e-10:
        raise ValueError("singular cell")
    unknown = set(map(int, atoms.numbers)) - supported
    if unknown:
        raise ValueError(f"unsupported atomic numbers: {sorted(unknown)}")
    for occupation in atoms.info.get("occupancy", {}).values():
        if len(occupation) != 1 or any(
            abs(float(value) - 1) > 1e-8 for value in occupation.values()
        ):
            raise ValueError("partial or mixed CIF occupancy is unsupported")


class StepCounter:
    stage = DynamicsStage.BEFORE_PRE_UPDATE
    frequency = 1

    def __call__(self, ctx, stage):
        ctx.batch.relax_steps[ctx.batch.status.view(-1) == 0] += 1


class StepBudget:
    stage = DynamicsStage.AFTER_STEP
    frequency = 1

    def __init__(self, max_steps):
        self.max_steps = max_steps

    def __call__(self, ctx, stage):
        b = ctx.batch
        converged = (b.status.view(-1) == 1) & (b.exit_reason.view(-1) == 0)
        b.exit_reason.view(-1)[converged] = 1
        exhausted = (b.status.view(-1) == 0) & (
            b.relax_steps.view(-1) >= self.max_steps
        )
        b.status.view(-1)[exhausted] = 1
        b.exit_reason.view(-1)[exhausted] = 2


class WallDeadline:
    stage = DynamicsStage.BEFORE_STEP
    frequency = 1

    def __init__(self, deadline):
        self.deadline = deadline

    def __call__(self, ctx, stage):
        if time.monotonic() >= self.deadline:
            raise TimeoutError("relaxation wall budget exhausted between steps")


def joint_convergence(fmax, stress):
    return ConvergenceHook(
        criteria=[
            {
                "key": "forces",
                "threshold": fmax,
                "reduce_op": "norm",
                "reduce_dims": -1,
            },
            {
                "key": "stress",
                "threshold": stress,
                "custom_op": lambda x: x.abs().amax((-1, -2)) <= stress,
            },
        ],
        source_status=0,
        target_status=1,
    )


def relax(
    paths,
    checkpoint,
    output,
    *,
    optimizer="fire2",
    device="cpu",
    dtype=torch.float64,
    max_batch_size=None,
    max_atoms=None,
    max_steps=500,
    max_wall_seconds=120,
    dt=0.02,
    fmax=0.01,
    stress_gpa=0.1,
    skin=0.0,
    fasteq=False,
):
    """Return one auditable result per source, using bounded native inflight work."""
    if not paths:
        raise ValueError("no CIF inputs")
    if optimizer not in ("fire", "fire2") or dtype not in (
        torch.float32,
        torch.float64,
    ):
        raise ValueError("unsupported optimizer or dtype")
    capacities = [value for value in (max_batch_size, max_atoms) if value is not None]
    if len(capacities) != 1:
        raise ValueError("set exactly one of max_atoms or max_batch_size")
    if (
        min([*capacities, max_steps]) <= 0
        or any(
            not math.isfinite(x) or x <= 0
            for x in (max_wall_seconds, dt, fmax, stress_gpa)
        )
        or not math.isfinite(skin)
        or skin < 0
    ):
        raise ValueError(
            "positive finite budgets, dt and tolerances required; skin >= 0"
        )
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested device is unavailable")
    checkpoint = Path(checkpoint).expanduser().resolve()
    model_hash = sha256(checkpoint)
    if model_hash != MODEL_SHA256:
        raise ValueError("checkpoint does not match the locked MACE-MPA-0 SHA256")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)  # MPA-0's integer-Z power depends on this.
    started = time.monotonic()
    fasteq_environment = None
    try:
        if fasteq:
            fasteq_environment = _configure_fasteq_runtime()
            model = _load_fasteq_mace(checkpoint, device, dtype)
        else:
            model = MACEWrapper.from_checkpoint(
                checkpoint,
                device=device,
                dtype=dtype,
                enable_cueq=False,
                compile_model=False,
            )
        model.set_config("active_outputs", {"energy", "forces", "stress"})
        model.eval()
        supported = set(map(int, model.model.atomic_numbers.detach().cpu().tolist()))
        records, structures, indices = [], [], []
        for index, path in enumerate(paths):
            path = Path(path).expanduser().resolve()
            record = {
                "source_index": index,
                "source": str(path),
                "status": "not_started",
                "steps": 0,
            }
            try:
                record["input_sha256"] = sha256(path)
                atoms = read(path, format="cif")
                validate_atoms(atoms, supported, max_atoms)
                structures.append(atoms)
                indices.append(index)
                record["natoms"] = len(atoms)
            except Exception as error:
                record.update(
                    status="invalid_input", error=f"{type(error).__name__}: {error}"
                )
            records.append(record)

        if structures:
            dataset = CIFDataset(structures, indices, dtype, device)
            sampler = SizeAwareSampler(
                dataset, max_atoms=max_atoms, max_batch_size=max_batch_size
            )
            sink = HostMemory(capacity=len(structures))
            hook = joint_convergence(fmax, stress_gpa * GPA_TO_EV_A3)
            cls = FIRE2VariableCell if optimizer == "fire2" else FIREVariableCell
            dynamics = cls(
                model=model,
                dt=dt,
                backend="torch_reference",
                hooks=[StepCounter()],
                convergence_hook=hook,
                device_type=device.type,
            )
            stage = FusedStage(
                sub_stages=[(0, dynamics)],
                sampler=sampler,
                sinks=[sink],
                refill_frequency=1,
                device_type=device.type,
            )
            # FusedStage installs migration hooks on its sub-stages. Classify
            # their convergence first, then apply the application's step budget.
            stage.register_hook(StepBudget(max_steps))
            stage.register_hook(
                NeighborListHook(
                    model.model_config.neighbor_config,
                    skin=skin,
                    backend="torch_reference",
                    stage=DynamicsStage.BEFORE_COMPUTE,
                )
            )
            stage.register_hook(WallDeadline(started + max_wall_seconds))
            failure, failure_message = None, None
            try:
                stage.run(n_steps=max_steps * len(structures))
            except Exception as error:
                failure = (
                    "wall_timeout"
                    if isinstance(error, TimeoutError)
                    else "calculation_error"
                )
                failure_message = f"{type(error).__name__}: {error}"
            completed = sink.read().to_data_list() if len(sink) else []
            active = stage.active_batch
            pending = [] if active is None else active.to("cpu").to_data_list()
            for data in completed + pending:
                source = int(data.source_index.item())
                record = records[source]
                record.update(
                    steps=int(data.relax_steps.item()),
                    system_id=int(data.system_id.item()),
                )
                reason = int(data.exit_reason.item())
                record["status"] = {1: "converged", 2: "max_steps"}.get(
                    reason, failure or "max_steps"
                )
                if reason == 0 and failure_message:
                    record["error"] = failure_message
                try:
                    atoms = Atoms(
                        numbers=data.atomic_numbers.detach().cpu().numpy().reshape(-1),
                        positions=data.positions.detach().cpu().numpy(),
                        cell=data.cell[0].detach().cpu().numpy(),
                        pbc=True,
                    )
                    target = output / f"{source:06d}_{Path(record['source']).stem}.cif"
                    write(target, atoms, format="cif")
                    record["final_cif"] = str(target.resolve())
                    record["final_cif_sha256"] = sha256(target)
                    # CIF normalizes the cell orientation. Evaluate the actual
                    # exported geometry so component-wise stress uses that frame.
                    exported = AtomicData.from_atoms(
                        read(target, format="cif"), dtype=dtype
                    )
                    final = Batch.from_data_list([exported]).to(device)
                    compute_neighbors(
                        final,
                        config=model.model_config.neighbor_config,
                        backend="torch_reference",
                    )
                    outputs = stage.compute(final)
                    measured_force = float(outputs["forces"].norm(dim=-1).max())
                    measured_stress = float(outputs["stress"].abs().max())
                    energy = float(outputs["energy"].item())
                    volume = float(torch.linalg.det(final.cell).abs().item())
                    if (
                        not all(
                            math.isfinite(x)
                            for x in (energy, volume, measured_force, measured_stress)
                        )
                        or volume <= 0
                    ):
                        raise ValueError("nonfinite final E/F/stress or invalid volume")
                    passed = (
                        measured_force <= fmax
                        and measured_stress <= stress_gpa * GPA_TO_EV_A3
                    )
                    record.update(
                        energy_eV=energy,
                        volume_A3=volume,
                        fmax_eV_A=measured_force,
                        max_abs_stress_eV_A3=measured_stress,
                        max_abs_stress_GPa=measured_stress / GPA_TO_EV_A3,
                        final_thresholds_passed=passed,
                    )
                    if record["status"] == "converged" and not passed:
                        record["status"] = "final_check_failed"
                except Exception as error:
                    record.update(
                        status="calculation_error",
                        error=f"final evaluation: {type(error).__name__}: {error}",
                    )

        parameter_names = (
            (
                "delaystep",
                "dtgrow",
                "dtshrink",
                "alphashrink",
                "alpha0",
                "tmax",
                "tmin",
                "maxstep",
            )
            if optimizer == "fire2"
            else ("maxstep", "n_min", "f_dec", "f_inc", "alpha_start", "f_alpha")
        )
        parameters = (
            {name: getattr(dynamics, name) for name in parameter_names}
            if structures
            else {}
        )
        if optimizer == "fire":
            parameters.update(dt_max=dt * 10, dt_min=dt * 0.02)
        parameters["dt"] = dt
        metadata = {
            "model_sha256": model_hash,
            "optimizer": optimizer,
            "batch_mode": "atoms" if max_atoms is not None else "bsize",
            "backend": "torch_reference",
            "neighbor_backend": "torch_reference",
            "inference_backend": "rocEquivarience+newFastEq" if fasteq else "e3nn",
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "cpu",
            "torch_version": str(torch.__version__),
            "hip_version": torch.version.hip,
            "dtype": str(dtype),
            "optimizer_parameters": parameters,
            "fmax_tolerance_eV_A": fmax,
            "stress_tolerance_GPa": stress_gpa,
            "skin_A": skin,
            "max_steps_per_system": max_steps,
            "max_wall_seconds": max_wall_seconds,
            "max_atoms": max_atoms,
            "max_batch_size": max_batch_size,
            "elapsed_seconds": time.monotonic() - started,
        }
        with (output / "results.jsonl").open("w") as stream:
            for record in records:
                record.update(metadata)
                stream.write(json.dumps(record, allow_nan=False, sort_keys=True) + "\n")
        return records
    finally:
        torch.set_default_dtype(previous_dtype)
        if fasteq_environment is not None:
            _restore_fasteq_runtime(fasteq_environment)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--optimizer", choices=("fire", "fire2"), default="fire2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--fasteq",
        action="store_true",
        help="use the local rocEquivarience/newFastEq HIP inference path",
    )
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-mode", choices=("atoms", "bsize"), default="atoms")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-atoms", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--max-wall-seconds", type=float, default=120)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--fmax", type=float, default=0.01)
    parser.add_argument("--stress-gpa", type=float, default=0.1)
    parser.add_argument("--skin", type=float, default=0.0)
    args = vars(parser.parse_args())
    batch_mode = args.pop("batch_mode")
    batch_size = args.pop("batch_size")
    max_atoms = args.pop("max_atoms")
    if batch_mode == "atoms":
        if batch_size is not None:
            parser.error("--batch-size is only valid with --batch-mode bsize")
        args["max_batch_size"] = None
        args["max_atoms"] = 256 if max_atoms is None else max_atoms
    else:
        if max_atoms is not None:
            parser.error("--max-atoms is only valid with --batch-mode atoms")
        if batch_size is None:
            parser.error("--batch-mode bsize requires --batch-size")
        args["max_batch_size"] = batch_size
        args["max_atoms"] = None
    directory = args.pop("input_dir")
    if not directory.is_dir():
        parser.error("--input-dir must be an existing directory")
    paths = sorted(directory.glob("*.cif"))
    limit = args.pop("limit")
    if limit is not None:
        if limit <= 0:
            parser.error("--limit must be positive")
        paths = paths[:limit]
    args["output"] = args.pop("output_dir")
    args["dtype"] = getattr(torch, args["dtype"])
    records = relax(paths, **args)
    print(
        json.dumps(
            {
                "total": len(records),
                "statuses": {
                    status: sum(r["status"] == status for r in records)
                    for status in sorted({r["status"] for r in records})
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
