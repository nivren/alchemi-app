# SPDX-License-Identifier: Apache-2.0
"""Exercise the thin CIF entry with an analytic model, separately from MPA-0."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.io import read, write

from alchemi_app import relax_mace_mpa0 as module
from nvalchemi.models.base import NeighborConfig

from .test_support import CellHarmonic


@pytest.fixture
def entry(monkeypatch):
    model = CellHarmonic()
    model.model = SimpleNamespace(atomic_numbers=torch.tensor([3, 11]))
    model.model_config.neighbor_config = NeighborConfig(cutoff=1.0)
    model.eval = lambda: None
    model.set_config = lambda *a: None
    monkeypatch.setattr(module.MACEWrapper, "from_checkpoint", lambda *a, **k: model)
    original = module.sha256
    monkeypatch.setattr(
        module,
        "sha256",
        lambda p: module.MODEL_SHA256 if Path(p).name == "model" else original(p),
    )
    return module, model


def inputs(tmp_path):
    paths = []
    for i, n in enumerate((2, 3, 2)):
        atoms = Atoms(
            numbers=[3 if i != 1 else 11] * n,
            positions=np.arange(n * 3).reshape(n, 3) * 0.09,
            cell=np.diag([2.2 + i * 0.03, 2.1, 2.3]),
            pbc=True,
        )
        path = tmp_path / f"sample_{i}.cif"
        write(path, atoms)
        paths.append(path)
    return paths


@pytest.mark.parametrize("optimizer", ["fire", "fire2"])
@pytest.mark.parametrize("max_steps, expected", [(600, "converged"), (2, "max_steps")])
def test_cif_outputs_convergence_budgets_refill_and_source_ids(
    entry, tmp_path, optimizer, max_steps, expected
):
    module, _ = entry
    paths = inputs(tmp_path)
    invalid = tmp_path / "bad.cif"
    invalid.write_text("not a CIF")
    records = module.relax(
        paths + [invalid],
        tmp_path / "model",
        tmp_path / "result",
        optimizer=optimizer,
        max_atoms=6,
        max_steps=max_steps,
        max_wall_seconds=60,
        dt=0.04,
    )
    assert [r["status"] for r in records] == [expected] * 3 + ["invalid_input"]
    assert sorted(r["system_id"] for r in records[:3]) == [0, 1, 2]
    assert [r["source_index"] for r in records] == [0, 1, 2, 3]
    for record in records[:3]:
        atoms = read(record["final_cif"])
        exact_energy = 0.5 * ((atoms.cell.array - np.eye(3) * 2) ** 2).sum()
        assert record["energy_eV"] == pytest.approx(exact_energy, abs=1e-10)
        assert record["volume_A3"] == pytest.approx(atoms.get_volume())
        assert record["final_thresholds_passed"] == (expected == "converged")
        assert record["steps"] <= max_steps
    saved = [
        json.loads(line)
        for line in (tmp_path / "result/results.jsonl").read_text().splitlines()
    ]
    assert saved == records
    assert torch.get_default_dtype() == torch.float32


def test_wall_timeout_is_distinct_from_success_and_pending_inputs(entry, tmp_path):
    module, _ = entry
    records = module.relax(
        inputs(tmp_path),
        tmp_path / "model",
        tmp_path / "timeout",
        max_atoms=6,
        max_wall_seconds=1e-9,
    )
    assert sorted(r["status"] for r in records) == [
        "not_started",
        "wall_timeout",
        "wall_timeout",
    ]
    assert all(r["steps"] == 0 for r in records)
    assert all("final_cif" in r for r in records if r["status"] == "wall_timeout")


def test_calculation_failure_does_not_mark_active_or_pending_success(
    entry, tmp_path, monkeypatch
):
    module, model = entry
    original = type(model).__call__
    calls = 0

    def fail_once(self, batch):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("analytic fixture inference failure")
        return original(self, batch)

    monkeypatch.setattr(type(model), "__call__", fail_once)
    records = module.relax(
        inputs(tmp_path), tmp_path / "model", tmp_path / "failure", max_atoms=6
    )
    assert sorted(r["status"] for r in records) == [
        "calculation_error",
        "calculation_error",
        "not_started",
    ]
    assert all(
        "inference failure" in r["error"]
        for r in records
        if r["status"] == "calculation_error"
    )


def test_refill_priming_failure_retains_new_arrival_for_export(
    entry, tmp_path, monkeypatch
):
    module, model = entry
    original = type(model).__call__
    calls = 0

    def fail_arrival_once(self, batch):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("arrival priming failure")
        return original(self, batch)

    monkeypatch.setattr(type(model), "__call__", fail_arrival_once)
    records = module.relax(
        inputs(tmp_path),
        tmp_path / "model",
        tmp_path / "arrival_failure",
        max_atoms=6,
        stress_gpa=100,
    )
    assert [r["status"] for r in records] == [
        "converged",
        "converged",
        "calculation_error",
    ]
    assert [r["steps"] for r in records] == [1, 1, 0]
    assert all("final_cif" in r for r in records)
    assert "arrival priming failure" in records[2]["error"]


@pytest.mark.parametrize("occupancy", [{"Li": 0.5}, {"Li": 0.5, "Na": 0.5}])
def test_partial_or_mixed_occupancy_rejected(entry, occupancy):
    module, _ = entry
    atoms = Atoms("Li", positions=[[0, 0, 0]], cell=np.eye(3) * 3, pbc=True)
    atoms.info["occupancy"] = {"0": occupancy}
    with pytest.raises(ValueError, match="occupancy"):
        module.validate_atoms(atoms, {3, 11}, 10)
