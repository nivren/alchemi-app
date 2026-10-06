# alchemi-app: MACE-MPA-0 CIF batch relaxation

This is a downstream application project. It composes the public MACE wrapper,
native variable-cell FIRE/FIRE2, `FusedStage`, `SizeAwareSampler`, hooks, and
sinks from the sibling `alchemi-hygon` toolkit checkout. Application-specific
CIF validation, run budgets, result status, and CIF/JSONL output live here;
the toolkit repository remains the upstream migration and compatibility
tracking project.

The application was extracted from toolkit integration work LP-083. The
separation does not add a new optimizer or framework API. Its workflow still
depends on the toolkit migration that is available in the sibling checkout;
the validated source baseline at extraction is commit `0e354f4`.

The MACE-MPA-0 checkpoint and structure data are user-provided inputs and are
not stored in this repository. The expected checkpoint SHA256 is embedded in
the entrypoint and must match the supplied model.

## Development environment

On the current Hygon development host, keep this project next to
`alchemi-hygon`, activate the toolkit environment, and expose the framework,
ops, and application source trees:

```bash
source ../alchemi-hygon/scripts/activate_hygon_env.sh project
export PYTHONPATH="../alchemi-hygon/packages/framework:../alchemi-hygon/packages/ops:src${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest -q
python -m alchemi_app.relax_mace_mpa0 --help
```

The `pyproject.toml` pins the current toolkit package versions and configures
`uv` to use the sibling migration checkout. Keep the Hygon PyTorch installation
from the toolkit environment; do not resolve a generic CUDA PyTorch build for
this application.

### Optional local FastEq stack on Hygon

The `hcu-fasteq` extra records the local `rocEquivarience` frontend and
`newFastEq` operator sources. Install them additively into the existing Hygon
environment so its Hygon Torch and toolkit packages stay in place. Do not run
`uv sync` against the shared toolkit environment: exact sync can remove
packages that are not declared by this application, and the current project
configuration resolves generic PyTorch from the package index.

From this project directory:

```bash
source ../alchemi-hygon/scripts/activate_hygon_env.sh project
export UV_CACHE_DIR=/tmp/alchemi-app-uv-cache
export PYTHONPATH="../alchemi-hygon/packages/framework:../alchemi-hygon/packages/ops:src${PYTHONPATH:+:$PYTHONPATH}"

uv pip install --python "$HYGON_PYTHON_ENV/bin/python" "cuequivariance==0.8.0"
uv pip install --python "$HYGON_PYTHON_ENV/bin/python" \
  --no-deps --editable ../rocEquivarience

FASTEQ_BACKEND=hip FASTEQ_HIP_ARCH=gfx936 \
  uv pip install --python "$HYGON_PYTHON_ENV/bin/python" \
  --no-deps --no-build-isolation --editable ../newFastEq
```

The FastEq build uses the active Hygon Torch and the host's `hipcc`, CMake, and
Ninja. The app's `--fasteq` option selects `FASTEQ_BACKEND=hip`,
`FASTEQ_CODEGEN_BACKEND=native`, and `FASTEQ_INFERENCE=1` before converting
the MACE model. It loads the compiled `_hip` library with
`torch.ops.load_library`; it is not a Python module to import directly. This
changes the MACE inference path only; FIRE and neighbor computation remain on
their explicitly selected backends.

## Run a bounded relaxation

Provide a directory of periodic CIF files and a local checkpoint matching the
embedded SHA256. Use a fresh output directory for each run:

```bash
HIP_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1 python -m alchemi_app.relax_mace_mpa0 \
  --input-dir /path/to/selected-cifs \
  --output-dir /path/to/new-output \
  --checkpoint /path/to/mace-mpa-0-medium.model \
  --optimizer fire --device cuda:0 --dtype float64 \
  --batch-mode atoms --max-atoms 64 --max-steps 500 \
  --max-wall-seconds 180 --dt 0.02 --fmax 0.01 \
  --stress-gpa 0.1 --skin 0
```

To use the local HCU inference stack, add `--fasteq` to the command above.
This option requires a Hygon HIP PyTorch build and `--device cuda[:index]`.
The output metadata records `inference_backend=rocEquivarience+newFastEq`;
`backend` and `neighbor_backend` continue to identify the FIRE and neighbor
implementations separately.

This is a bounded eager reference application. It does not promise full
directory throughput, restart/streaming behavior, or production performance.
The optional local FastEq stack is an inference integration path; see the
toolkit reports for the CPU and targeted HCU evidence and their limits.

## Configuration-driven relaxation and backend comparison

Use `alchemi_app.run_relaxation` as the project entrypoint. It reads a TOML
parameter file; `run.mode = "relax"` runs one CIF relaxation, while
`run.mode = "batch_compare"` launches the configured `e3nn` and/or `fasteq`
benchmark runs in separate processes with the same inputs and settings.
`run.inference_backends` defaults to both backends; set it to one backend to run
the same batch relaxation probe for only that backend.
For `run.mode = "batch_compare"`, `neighbors.backend` selects the neighbor
implementation independently of the MACE inference and FIRE execution paths.
The explicit `hip` option uses the
registered periodic HIP cell-list implementation (with Torch geometry), needs
an HCU HIP device and `skin_A = 0`, and fails explicitly if the request is not
supported. The benchmark requests `method="cell_list"` for HIP explicitly.
The selected implementation ID is written to `timing.json`.
`configs/mace_relax_template.toml` shows the single-relaxation settings.
Copy and edit that template, then run it with the same `--config` command below.

The 4,000-atom comparison is configured in
`configs/mpa0_batch4000_compare.toml`. It preserves the previous 114-CIF batch,
variable-cell FIRE settings, native atomic/cell-force criterion, neighbor
settings, and output records. Its `comparison.json` reports full process wall
time, application/dynamics time, speedup ratios, convergence/status agreement,
and final-result deltas. The FastEq first-use compilation time is included.
Each backend has its own result directory and log under the configured output
directory; `parameters.toml` records the exact settings used.

Batch capacity uses one mode at a time. `batch.mode = "atoms"` sets only
`max_atoms`; `batch.mode = "bsize"` sets only `batch_size`. The application
uses `SizeAwareSampler` with in-flight refill: when structures finish, their
slots can be filled from the remaining input set. It does not precompute
fixed groups and assign those groups to worker processes. The sampler can
also apply its automatic GPU-memory estimate as an additional safety bound.
The full input selection may exceed `max_atoms` or `batch_size`; those values
limit the active batch, while each individual structure must fit the selected
capacity.

```bash
source ../alchemi-hygon/scripts/activate_hygon_env.sh project
export PYTHONPATH="../alchemi-hygon/packages/framework:../alchemi-hygon/packages/ops:src${PYTHONPATH:+:$PYTHONPATH}"
python -m alchemi_app.run_relaxation \
  --config configs/mpa0_batch4000_compare.toml
```

The output directory must not already exist. Paths in TOML are relative to the
project root unless absolute. The selection manifest and CIFs remain external
input data; benchmark code, configuration, logs, and results belong to this
application project.
