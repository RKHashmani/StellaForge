# CLAUDE.md

Project-level instructions for AI assistants working in the driftless-star repository.
This is the single source of truth for all coding standards.

---

## Part 1: driftless-star Project

### Project Description

driftless-star is an open-source pipeline for stellarator design, connecting five physics stages (equilibrium, Boozer transform, neoclassical transport, turbulence, and profile evolution) into a single reproducible workflow. The current goal is a working forward pass: from boundary Fourier coefficients and profile guesses through to transport-consistent profiles and fusion-power metrics. The pipeline is designed to be closed-loop, so output profiles can eventually feed back to Stage 1 for iterative optimization.

driftless-star is a **recipe repo**: it contains environment definitions, container builds, and orchestration logic, but the upstream solver codes are installed as dependencies. The companion `stellarator_workflow/` submodule contains TeX manuscripts defining the physics equations and I/O contracts. See `docs/guide.md` for the full pipeline design and contributor workflow.

### The 5 Pipeline Stages

| Stage | Name | Primary Code | Alternatives | Spec |
|-------|------|-------------|--------------|------|
| 1 | Equilibrium | `vmec_jax`, `DESC` | `VMEC++` | `docs/stage1-equilibrium/spec.md` |
| 2 | Boozer Transform | `booz_xform_jax` | `BOOZ_XFORM` | `docs/stage2-boozer/spec.md` |
| 3 | Neoclassical | `NEO_JAX`, `sfincs_jax` | `NEO`, `SFINCS` | `docs/stage3-neoclassical/spec.md` |
| 4 | Turbulence | `GKX` | `GX`, `GENE` | `docs/stage4-turbulence/spec.md` |
| 5 | Transport | `NEOPAX` | `Trinity3D` | `docs/stage5-transport/spec.md` |

Forward-pass chain: `vmec_jax` -> `booz_xform_jax` -> `sfincs_jax` -> `GKX` -> `NEOPAX`

**Key notes:**
- `NEO_JAX` is **not** in the forward-pass chain: it computes epsilon_eff as a screening/optimization diagnostic, runs in parallel with `sfincs_jax`, and is not consumed by Stage 5.
- Stages 3 and 4 run in parallel after Stage 2.

### Naming Conventions

- Stage directories: `stage{N}-{name}` (e.g., `stage1-equilibrium`)
- Run inputs and outputs (top-level): each run is a folder under `inputs/` holding its `config.yaml` and stage inputs; the committed `inputs/quick_run/` baseline is tracked so a fresh clone is immediately runnable. Generated artifacts land under `outputs/<run>/stageN_<name>/` (e.g. `outputs/quick_run/stage1_equilibrium/`). `outputs/` and ad-hoc `inputs/<run>/` folders are gitignored (only `inputs/quick_run/` is tracked).
- Container images: `ghcr.io/driftless-star/driftless-star:stage-{N}-{code}-cpu` / `stage-{N}-{code}-gpu` (e.g., `stage-1-vmec-cpu`) (on GHCR)
- W&B projects: `driftless-star-stage{N}-{name}`
- Test files: mirror the source structure in a top-level `tests/`

### Inter-Stage Contracts

Currently, most inter-stage communication is **file-based** using standard physics file formats:
- **NetCDF** (`.nc`): equilibrium (`wout_*.nc`), Boozer (`boozmn_*.nc`), turbulence outputs
- **HDF5** (`.h5`): neoclassical outputs (`sfincs_jax_flux_profiles.h5`), `NEOPAX` profiles

Snakemake rules define which files connect which stages. Each stage's `spec.md` is the authoritative source for required/optional fields in its output files. Where alternative implementations use different file formats or field names, a wrapper or adapter layer will be needed to translate between them.

**Key points from the TeX manuscripts:**

1. **Screening-only outputs vs. transport state variables.** `NEO_JAX`'s epsilon_eff is central to ranking candidate geometries but is NOT advanced by a transport solver. It should not be wired as a transport input.
2. **Dual-role outputs.** Heat/particle flux from `GKX` and neoclassical flux from `SFINCS` are simultaneously optimization objectives (to minimize) AND direct numerical inputs for transport profile evolution.
3. **Turbulence coupling.** `NEOPAX` has turbulence-coupling utilities, but the `GKX` -> `NEOPAX` path (Stage 4 -> Stage 5) is not yet the default.

### Working with This Codebase

- Read `docs/guide.md` first for the big picture and contributor workflow.
- Read `docs/mvp-pipeline.md` for MVP I/O reference, Pixi install/run commands.
- Each stage has its own spec in `docs/stage{N}-{name}/spec.md`.
- See `docs/container-images.md` for Docker/Apptainer build and run examples.
- See `docs/potential_issues.md` for known cross-stage compatibility issues.
- The `stellarator_workflow/` submodule is read-only reference material:
  - `stellarator_workflow.tex` -- governing equations and code details
  - `stellarator_io_reference.tex` -- I/O contracts and handoff specifications
- When modifying a stage, consult its spec doc for I/O contracts.

### Phase-Specific Rules

- **Phase 1 (Document & Run):** Install and run the primary code, document the API and convergence behavior, write example scripts, set up W&B tracking. No cross-stage Python imports: all inter-stage communication is through files (e.g. NetCDF/HDF5). This is necessary to maintain swappability. Do not restructure upstream code.
- **Phase 2 (Containerize & Test):** Build container environments, write unit/regression/integration tests. Container changes must pass integration tests before merge. See `docs/guide.md#container-architecture` for the split-Pixi-workspace + `stages/Dockerfile` approach.
- **Phase 3 (Integrate):** Snakemake rules should support config-driven implementation selection.

---

## Part 2: Workflow Standards

### Planning

- Always use plan mode before changes that touch critical or complex logic (e.g., core algorithms, data pipelines, model architectures, mathematical computations).
- When something goes sideways during implementation, stop and re-plan -- don't keep pushing down a broken path.
- During planning, ask as many clarifying questions as needed to fully understand the relevant parts of the codebase, the current system, and how the proposed change fits in. Do not proceed with a plan until ambiguities are resolved.

### Review & Validation

- Act as a critical reviewer: challenge the user's reasoning, flag edge cases, and identify potential issues before implementation begins.
- After implementation, prove the changes work -- diff behavior between the working branch and `main`, run tests, and demonstrate correctness rather than just asserting it.
- Do not create a PR or propose merging until you can show evidence that the changes are correct and complete.

### Testing

- Every new feature or behavior change must include corresponding tests that verify it works as expected. Write tests alongside (or before) the implementation, not as an afterthought.
- Run the project's test suite after making changes to verify nothing is broken.

---

## Part 3: Scientific Programming Standards

### Reproducibility

- Always seed all RNG sources together so results are deterministic (e.g., `random.seed()`, `np.random.seed()`, `jax.random.PRNGKey()`, and any framework-specific seeds).
- Disable non-deterministic backend behavior for reproducible runs; only enable stochastic optimizations when explicitly trading reproducibility for speed.
- Save full configuration alongside every output so any result can be reproduced (e.g., save args/config to JSON next to output files).
- Default random seed to a deterministic value so runs are reproducible without explicit `--seed`.
- Never use truly random seeds unless explicitly requested by the user.

### Code Quality & Style

- Use the project's configured linter/formatter (e.g., ruff, black). Follow PEP 8 with 120-char line width for Python.
- Add type hints to all new/modified function signatures (parameters + return type).
- Use descriptive names; single-letter math variables (A, Sigma, z) are acceptable when they match paper notation -- add a comment referencing the equation/section.
- Keep functions under 60 lines and ~5 parameters. Extract helpers for longer functions.
- Eliminate code duplication: if the same logic appears twice, factor it into a shared function.
- Never control behavior by commenting/uncommenting code -- use flags or config parameters.
- Prefer f-strings for string formatting in Python.

### Testing

- Write unit tests for all mathematical/numerical functions using the project's test framework (e.g., pytest).
- Test mathematical invariants (e.g., non-negativity constraints, matrix properties like positive semi-definiteness, correct output dimensions).
- For numerical functions, test against known analytical solutions where available.
- Use regression tests: save known-good outputs and compare with explicit tolerances (e.g., `np.testing.assert_allclose`).

### Documentation

- Favor simplicity. State information once in the most appropriate location and reference it elsewhere rather than repeating it. If something is already documented in another file, link to it instead of duplicating.
- Add docstrings to all new/modified public functions using NumPy-style format (Parameters, Returns, Notes).
- For new docstrings, do not reformat existing docstrings in a different style unless modifying the function.
- Include mathematical context in docstrings: reference paper equations/sections where applicable.
- Every module should have a module-level docstring explaining its role in the system.
- Config/argument files should be self-documenting via comments.

### Logging & Error Handling

- Use Python's `logging` module with `logger = logging.getLogger(__name__)` for all output in production code paths.
- `print()` is acceptable only in debug blocks and CLI entry points (`if __name__ == "__main__":`).
- Use typed exceptions with descriptive messages for input validation (e.g., `raise ValueError(...)`, `raise TypeError(...)`). Never bare `assert` for user-facing checks.
- Never silently swallow exceptions. Always log or re-raise caught exceptions.
- Validate data shapes/dimensions at function boundaries when they are non-obvious.

### Numerical Best Practices

- Use library-provided linear algebra functions over manual implementations (e.g., `jax.numpy.linalg`, `numpy.linalg`, `scipy.linalg`).
- Validate mathematical properties of inputs before use (e.g., positive-definiteness of covariance matrices via Cholesky).
- Propagate device and dtype from input data -- never hardcode device or precision (e.g., avoid `device='cuda'` or `dtype=jnp.float32`).
- Use higher precision for computations where numerical accuracy matters (e.g., float64 for information-theoretic or covariance calculations).
- Guard against numerical edge cases: division by zero, log(0), overflow in exp(), and similar.

### Project Organization

- All packages must have proper `__init__.py` files.
- Use proper package imports; avoid path-manipulation hacks (e.g., `sys.path.append()`).
- Scripts should be runnable from the project root, not require `cd` into a subdirectory.
- Keep distinct concerns (data processing, model architecture, training, visualization) as independent modules.

### Version Control

- Make small, focused commits with descriptive messages.
- Never commit large binary files (model checkpoints, datasets) or secrets. Use `.gitignore` appropriately.

### Data Management

- Save raw data separately from processed results.
- Document output formats: array shapes, value ranges, and units in docstrings or project docs.
- Use structured, self-describing formats with descriptive field names (e.g., `.npz` with named arrays, HDF5 with labeled datasets).

### Configuration Management

- All runtime behavior must be controllable via CLI arguments or config files -- never require editing source code.

---

## Part 4: Software Engineering Standards

### Dependency Management

- Pin all dependencies with version bounds (e.g., `>=1.0,<2`); avoid unpinned or `*` versions.
- Commit lockfiles for reproducible installs (e.g., `pixi.lock`, `uv.lock`, `poetry.lock`).
- Keep dependency specs consistent across all environment/packaging files when adding or updating packages.
- Separate dev-only dependencies (testing, linting tools) from core dependencies using optional dependency groups.

### Security

- Never hardcode secrets, API keys, or credentials -- use environment variables or config files.
- Never use `eval()`, `exec()`, or unsafe deserialization on untrusted data.
- Use safe loading modes when available (e.g., `torch.load(..., weights_only=True)`); document the reason when unsafe loading is necessary.
- Validate and sanitize all external inputs (CLI args, file paths, JSON) before use.
- Prefer safe subprocess invocation (e.g., `subprocess.run()` with list args); never use `shell=True` with user-provided input.

### CI/CD & Automation

- Configure CI to run linting, type checking, and tests on pull requests.
- Use pre-commit hooks for formatting and linting before commit.
- Automate repetitive workflows via scripts, not manual multi-step commands.

### Docker & Containerization

- Pin base image versions to specific tags, not `latest`.
- Add `.dockerignore` to exclude unnecessary files from the build context.
- Keep Dockerfiles minimal -- install only production dependencies.

**driftless-star container architecture** (see `docs/guide.md#container-architecture` for full details):
- The repo has two Pixi workspaces with decoupled responsibilities:
  - Root `pixi.toml` defines the orchestration environments: `pipeline` (`snakemake-minimal`, `graphviz`, `pytest`, plus the HTCondor executor plugin on linux-64), `test` (adds the numerical libraries the test suite needs), and `htcondor-runtime` (`snakemake-minimal` only). For a local run Snakemake is never containerized -- it runs on the execution node directly, because nesting containers is fragile and not widely supported. Cluster execution is the one exception: a remote HTCondor job has no access to the submit host's `.pixi/`, so it runs Snakemake from the `htcondor-runtime` image built by `executors/htcondor/apptainer.def`, whose Snakemake version must stay pinned to `pipeline`'s (enforced by `tests/orchestration/test_pixi_envs.py`).
  - `stages/pixi.toml` defines the per-stage physics environments. These are only ever consumed by the container builder, so they are isolated from the orchestration env.
- A single templated `stages/Dockerfile` builds all stage images using `ghcr.io/prefix-dev/pixi:noble` as the base. The docker build context is `stages/`, and build arguments (`ENVIRONMENT`, `CUDA_VERSION`) select the target stage and GPU support.
- Container images are published to GHCR (see the Naming Conventions section for the tag pattern); CI builds all stage variants from `stages/Dockerfile` using a GitHub Actions matrix.
- Source-built upstream packages are pinned to exact git commit SHAs in `stages/pixi.toml`.

### Performance & Memory Management

- Disable gradient computation for all inference and evaluation code paths (e.g., `jax.jit` with no grad, `torch.no_grad()`).
- Use mixed precision when memory is a concern.
- Release memory between intensive phases.
- Profile before optimizing -- use profiling tools to identify actual bottlenecks before making performance changes.
