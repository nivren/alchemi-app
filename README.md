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

## Run a bounded relaxation

Provide a directory of periodic CIF files and a local checkpoint matching the
embedded SHA256. Use a fresh output directory for each run:

```bash
HIP_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1 python -m alchemi_app.relax_mace_mpa0 \
  --input-dir /path/to/selected-cifs \
  --output-dir /path/to/new-output \
  --checkpoint /path/to/mace-mpa-0-medium.model \
  --optimizer fire --device cuda:0 --dtype float64 \
  --max-batch-size 4 --max-atoms 64 --max-steps 500 \
  --max-wall-seconds 180 --dt 0.02 --fmax 0.01 \
  --stress-gpa 0.1 --skin 0
```

This is a bounded eager reference application. It does not promise full
directory throughput, restart/streaming behavior, CuEq or HCU-specific
equivariant acceleration, or production performance. See the toolkit reports
for the CPU and targeted HCU evidence and their limits.
