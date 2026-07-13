# driftless-star Guide

## Project Overview

driftless-star implements the stellarator design workflow described in the companion `stellarator_workflow/` submodule as a containerized, orchestrated software pipeline. The physics and I/O contracts are defined in two TeX manuscripts:
- `stellarator_workflow.tex` -- governing equations and code-by-code details
- `stellarator_io_reference.tex` -- input/output contracts and handoff specifications

**Goal:** A working forward pass -- a single traversal of the pipeline from boundary Fourier coefficients and profile guesses through to transport-consistent profiles and fusion-power metrics (P_fus, Q). An initial **closed loop** is now implemented on top of this: Stage 5's transport solution is fit back into the Stage 1 pressure profile and the forward pass re-run, iterating toward a transport-consistent equilibrium via the external `ouroboros` driver (see [Closing the Loop](mvp-pipeline.md#closing-the-loop)). The re-run rebuilds the equilibrium, so the updated geometry flows on to Stages 3/4. Two extensions remain future work: feeding back the **current** profile (only pressure is fit today), and using Stage 5's transport-evolved n(r)/T(r) as the Stage 3/4 kinetic profiles, which still use `profiles_source: analytical`.

**JAX-first strategy:** The pipeline prioritizes JAX-native implementations for differentiability and tight integration: `vmec_jax` -> `booz_xform_jax` -> `sfincs_jax` -> `SPECTRAX-GK` -> `NEOPAX`. Other codes (`VMEC++`, `BOOZ_XFORM`, `NEO_JAX`, `NEO`, `SFINCS`, `GX`, `GENE`, `Trinity3D`) are swappable alternatives.

## Pipeline Architecture

### Stage Summary

> [!NOTE]
> The input/output artifacts shown reflect the JAX-primary implementations. Alternative codes may use different file formats or field names; see each stage's `spec.md` for details.

| Stage | Physics | JAX Primary | Alternatives | Input Artifacts | Output Artifacts |
|-------|---------|-------------|--------------|-----------------|------------------|
| 1. Equilibrium | Ideal-MHD force balance | `vmec_jax`, `DESC` | `VMEC++` | INDATA/JSON boundary coefficients, pressure/iota/current coefficients, PHIEDGE | `wout_*.nc` (NetCDF) |
| 2. Boozer Transform | Coordinate transform to Boozer angles | `booz_xform_jax` | `BOOZ_XFORM` | `wout_*.nc` | `boozmn_*.nc` (NetCDF) |
| 3. Neoclassical | Effective ripple, drift-kinetic transport | `NEO_JAX`, `sfincs_jax` | `NEO`, `SFINCS` | `NEO_JAX`: `boozmn_*.nc`; `SFINCS`: `wout_*.nc` + input file | `neo_out.*`, `sfincs_jax_flux_profiles.h5` |
| 4. Turbulence | Delta-f gyrokinetic equation | `SPECTRAX-GK` | `GX`, `GENE` | Geometry + species profiles/gradients | gamma, omega, heat/particle flux (NetCDF/CSV) |
| 5. Transport | 1D conservation laws for n_s, p_s | `NEOPAX` | `Trinity3D` | geometry + fluxes | n(r), T(r), E_r(r), P_fus, Q (HDF5/NetCDF) |

### Pipeline DAG

> [!TODO]
> Define Minimum Viable Pipeline (MVP) DAG.

### Interface Contracts

Currently, most inter-stage communication is **file-based** using standard physics file formats:
- **NetCDF** (`.nc`): equilibrium (`wout_*.nc`), Boozer (`boozmn_*.nc`), turbulence outputs
- **HDF5** (`.h5`): neoclassical outputs (`sfincs_jax_flux_profiles.h5`), `NEOPAX` profiles

Snakemake rules define which files connect which stages. Each stage's `spec.md` is the authoritative source for required/optional fields in its output files. Where alternative implementations use different file formats or field names, a wrapper or adapter layer will be needed to translate between them.

**Output directory convention:** Each stage writes to `outputs/<run>/stageN_<name>/` on a shared volume mount.

**Key points** from the TeX manuscripts:

1. **Screening-only outputs vs. transport state variables.** `NEO_JAX`'s epsilon_eff is central to ranking candidate geometries but is NOT advanced by a transport solver. It should not be wired as a transport input.
2. **Dual-role outputs.** Heat/particle flux from `SPECTRAX-GK` and neoclassical flux from `SFINCS` are simultaneously optimization objectives (to minimize) AND direct numerical inputs for transport profile evolution.
3. **Turbulence coupling.** `NEOPAX` has turbulence-coupling utilities, but the `SPECTRAX-GK` -> `NEOPAX` path (Stage 4 -> Stage 5) is not yet the default.

### Swappability Patterns

The pipeline should eventually support config-driven implementation swapping. Possible levels of swappability:

- **Single-stage swap.** Change `inputs/<run>/config.yaml` to select a different implementation for one stage. The output file format should match what downstream stages expect. Example: swap Stage 4 from `SPECTRAX-GK` to `GX`.

- **Multi-stage swap.** A single combined Snakemake rule replaces multiple individual stage rules. It should produce all output files that downstream stages expect. Example: `DESC` can perform both equilibrium solving and Boozer transformation internally, replacing Stages 1 and 2 with a single rule.

- **End-to-end swap.** The entire pipeline DAG is replaced by a single rule. Useful for an all-in-one differentiable pipeline or a completely different solver chain.

## Getting Started

1. Clone the repository and initialize submodules:
   ```bash
   git clone https://github.com/driftless-star/driftless-star.git
   cd driftless-star
   git submodule update --init --recursive
   ```

2. Read this guide for the overall architecture and workflow.

3. Find the relevant stage specification: `docs/stage{N}-{name}/spec.md`

4. Read the project coding standards: `CLAUDE.md`

5. Review the reference TeX manuscripts in `stellarator_workflow/` for physics context:
   - `stellarator_workflow.tex` -- governing equations and code details
   - `stellarator_io_reference.tex` -- I/O contracts and handoff specifications

6. Review `docs/mvp-pipeline.md` for the MVP I/O reference and Pixi install/run commands.

### Contributing Changes

1. Fork the repository and branch from `main` (e.g., `feat/stage1-newsoftware`, `fix/update-naming-schema`)
2. Work through the relevant phase below
3. Open a PR from the fork when deliverables are ready and request a review
4. After review and merge, the corresponding progress item in the [README](../README.md#progress) gets checked off

## Phase 1: Document & Run

Each stage has one owner responsible for their stage end-to-end. Each stage's `spec.md` has pre-populated sections (extracted from the TeX manuscripts) and clearly marked "TO BE COMPLETED" sections to fill in.

Work through these steps in order. Each step should result in updates to the stage's `spec.md`.

### Step 1: Install and Run the Primary Code

- Install the primary code using the Pixi environment for the stage (see `stages/pixi.toml` and `docs/mvp-pipeline.md` for install/run commands)
- If the code is not yet in a Pixi environment, ensure it can be installed via conda-forge or PyPI, add it to `stages/pixi.toml`, and regenerate the lock file
- Get it running locally on a reference case
- Document any installation issues, version requirements, or platform-specific notes
- Fill in the "Installation & Platform" section of the relevant spec

### Step 2: Run a Reference Case and Capture I/O

- Run the code on a standard test case (e.g., a known stellarator configuration)
- Capture the actual input and output files
- Inspect the output files to understand their structure. For example:
  - For NetCDF: `ncdump -h output_file.nc` (shows all variables, dimensions, attributes)
  - For HDF5: `h5dump -H output_file.h5` (shows structure without data)
- Document any change in output file format from the MVP.

### Step 3: Document the API

- Document the main entry points, their parameters, and configuration options
- Provide programmatic usage examples in Python/JAX
- Fill in the "API Documentation" section of the relevant spec

### Step 4: Document Convergence & Validity

- Identify input configurations that are known to converge (cite specific stellarator configs if possible)
- Identify configurations that fail or diverge, optionally identify why.
- Document convergence criteria and tolerances
- Note known numerical edge cases
- Fill in the "Convergence & Validity" section of the relevant spec

### Step 5: Write Example Scripts

- Provide scripts that demonstrate how to run the code standalone (CLI and Python)
- Include sample input data or instructions for obtaining it
- Document common debugging workflows
- Fill in the "Scripts & Workflows" section of the relevant spec

### Step 6: Set Up W&B Tracking

> [!TODO]
> Set up W&B tracking once stages are operational.

Conventions:
- **Project name:** `driftless-star-stage{N}-{name}` (e.g., `driftless-star-stage1-equilibrium`)
- **Run naming:** `{code}_{config}_{timestamp}` (e.g., `vmec_jax_qa_2026-04-01T12:00`)
- **Metrics to log:** Stage-specific convergence metrics, runtime, key physics outputs (see the spec for guidance)
- Create a dashboard with the most important panels for the stage
- Fill in the "W&B Tracking" section of the relevant spec

### Step 7: Create Claude Skills

> [!TODO]
> Create Claude skills once stages are operational.

Two types of skills per stage:

**Development skill** (helps developers work on the stage): installation, debugging, output interpretation, physics context.

**Operational skill** (helps operators run the stage in the pipeline): container builds, test suite, output validation, W&B metrics.

Place skills in the stage's docs directory (e.g., `docs/stage1-equilibrium/skills/`). Fill in the "Claude Skills" section of the relevant spec.

## Phase 2: Containerize & Test

After Phase 1 is complete for a stage, move to containerization and testing.
### Container Architecture

driftless-star is a **recipe repo**: it contains everything needed to build and run the containerized pipeline, but does not contain the upstream solver code itself.

**Two decoupled Pixi workspaces.** The repo splits dependency management along the orchestration / physics boundary:
- **Root `pixi.toml`** -- a single `pipeline` environment (`snakemake-minimal`, `graphviz`, `pytest`). Installed directly on the execution node; Snakemake is never containerized because nested containers are fragile and not widely supported on shared compute.
- **`stages/pixi.toml`** -- per-stage physics environments (e.g., `stage-1-vmec`, `stage-1-vmec-gpu`) that fully specify each stack. These are only consumed by the container builder, so they are entirely isolated from the orchestration env.

Each workspace has its own lockfile (`pixi.lock` / `stages/pixi.lock`).

**The same boundary at the code level.** Two kinds of stage code live in the repo and should not be confused, despite the similar names:
- **`stages/stage{N}-{name}/*.py`** -- physics scripts. Self-contained scripts that import the upstream solver and do the numerical work (e.g., radial flux scans, the Boozer transform, pressure post-processing). They run *inside* the stage containers, and `stages/` is the Docker build context.
- **`src/stage{N}_helper.py`** -- orchestration helpers. Small modules the `Snakefile` imports on the execution node (in the root `pipeline` env, never containerized) and that contain no physics. Stages 3 and 4 compose each stage's `docker run ...` command from its `config.yaml` block; Stage 5 writes a path-resolved copy of the NEOPAX config.

So `src/stage3_helper.py` *builds* the command, while `stages/stage3-neoclassical/sfincs_jax_radial_scan.py` is what that command *runs* inside the container.

**Templated container images.** A single shared `stages/Dockerfile` and `stages/apptainer.def` use build arguments to select the target environment at build time:
- `ENVIRONMENT` -- the Pixi environment name (e.g., `stage-1-vmec`, `stage-2-booz-jax-gpu`). Must be passed explicitly when building locally:
   - `docker build --build-arg ENVIRONMENT=stage-1-vmec stages/`
   - `cd stages && apptainer build --build-arg ENVIRONMENT="stage-1-vmec" stage-1-vmec.sif apptainer.def`
- `CUDA_VERSION` -- set for GPU builds (e.g., `12`), left empty for CPU builds

The Dockerfile uses a multi-stage build on a `ghcr.io/prefix-dev/pixi:noble` base image. See `stages/Dockerfile` for implementation details.

**Container images** are published to GHCR at `ghcr.io/driftless-star/driftless-star`. For MVP, the tags follow the pattern `stage-{N}-{code}-cpu` / `stage-{N}-{code}-gpu` (e.g., `stage-1-vmec-cpu`). Apptainer container images are prefixed with `apptainer-`. CI builds all stage variants from the container image definition files using a GitHub Actions matrix. See `.github/workflows/containers.yml` and `.github/actions/build-docker/action.yml` for the CI setup.

**Adding or updating a stage dependency:**
1. Update `stages/pixi.toml` (add/change the dependency or git rev)
2. Run `pixi lock --manifest-path stages/pixi.toml` (or `pixi install ...` to also install locally)
3. Commit both `stages/pixi.toml` and `stages/pixi.lock`
4. CI rebuilds affected container images on merge

The stages workspace requires Pixi 0.71 or newer. GPU environments resolve on
the named `linux-64-cuda` platform, which declares CUDA 12 independently of the
machine generating the lockfile; CPU environments continue to use the standard
platform names.

Updating the orchestration env follows the same pattern against the root `pixi.toml` / `pixi.lock`.

### Verify Container I/O

- Build and run the container locally
- Verify it can read input files from a mounted volume
- Verify it writes output files to the expected location on the shared volume
- Verify the output directory follows the naming convention: `outputs/<run>/stageN_<name>/`

### Writing Tests

**Unit tests.** Test mathematical invariants specific to the stage. Examples:
- Stage 1: force-balance residual decreases monotonically during convergence
- Stage 2: (|B|_VMEC - |B|_Boozer) / |B|_VMEC < eps; Boozer transform preserves iota
- Stage 3 (`NEO`): epsilon_eff is non-negative, bounded
- Stage 3 (`SFINCS`): transport matrix has expected symmetry properties; full flux mode produces physically reasonable fluxes
- Stage 4: growth rates are real-valued, fluxes are non-negative in steady state
- Stage 5: profiles satisfy conservation (total particle/energy content)

Place tests in `tests/stage{N}-{name}/`.

**Regression tests.** Save known-good output files from a reference case. Write tests that compare new outputs against these baselines using explicit tolerances: `np.testing.assert_allclose(actual, expected, rtol=1e-6)`.

**Integration tests.** Verify that the stage's output is valid input for its downstream consumers. For example:
- Stage 1: verify `wout_*.nc` can be read by `booz_xform_jax` (Stage 2), `sfincs_jax` (Stage 3), and `SPECTRAX-GK` (Stage 4)
- Stage 2: verify `boozmn_*.nc` can be read by `NEO_JAX` (Stage 3)
- Stage 3: verify `sfincs_jax_flux_profiles.h5` (from `sfincs_jax`) can be read by `NEOPAX` (Stage 5)
- Stage 4: verify flux CSV output can be consumed by `NEOPAX` (Stage 5)
- Stage 5: verify end-to-end output (`profiles.h5`: n(r), T(r), E_r(r), P_fus, Q) is produced correctly
- Stage 5 → Stage 1: verify updated profiles from Stage 5 can be fed back as input to Stage 1 to close the optimization loop

Consult plasma experts for additional details.
### Acceptance Criteria

A stage is "done" for Phase 2 when:
- All unit tests pass
- All regression tests pass within tolerances
- Integration tests with adjacent stages pass
- Container builds and runs successfully
- W&B tracking is functional
- All "TO BE COMPLETED" sections in the spec are filled in

## Phase 3: Integrate

When a stage completes Phase 2 (containerized, tested, and producing valid output), the workflow engineer integrates it into the Snakemake pipeline. This is an ongoing process and stages are integrated as they become ready.

### Adding a Stage to the Snakemake DAG

> [!TODO]
> Define the process for adding a stage to the Snakemake DAG.

**Design points to keep in mind:**
- The forward-pass Stage 3 (`sfincs_jax`) reads only the Stage 1 wout, so it runs in parallel with Stage 2; Stage 4 (`SPECTRAX-GK`) also needs the Stage 2 boozmn, so it follows Stage 2.
- `NEO_JAX` reads the Stage 2 boozmn and runs alongside `sfincs_jax` as a screening diagnostic, but it has **no Snakemake rule** and is not part of the forward pass. Its epsilon_eff is a screening metric only. It should not be wired as a dependency for Stage 5.
- Stages 3 and 4 each fan out one Snakemake job per flux surface via a three-rule `prepare` (checkpoint) / `run_one` / `collect` layout; see [Per-surface fan-out](mvp-pipeline.md#per-surface-fan-out-stages-3-and-4).

### Config-Driven Selection

> [!TODO]
> Define the config schema and rule-selection logic.

The pipeline run configuration lives in each run folder (`inputs/<run>/config.yaml`, passed via `--configfile`) and should control which implementation is used per stage, the stage-specific parameters, and resource requirements.

### Integration Testing

When integrating a new stage or implementation:

- **Stage-boundary tests:** Feed known-good output from stage N to stage N+1 and verify physically reasonable results.
- **End-to-end test:** When all stages are integrated, run the full forward pass from a known boundary configuration and compare final outputs (P_fus, Q, profiles) against baselines with explicit tolerances.
- **Swappability tests:** After swapping an implementation, verify the pipeline still produces valid output.

### Pipeline-Level W&B

> [!TODO]
> Set up pipeline-level W&B aggregation.

- **Project:** `driftless-star-pipeline`
- Aggregate per-stage metrics, track implementation selections, total runtime, and final physics outputs (P_fus, Q, profiles).

## How to Document I/O

> [!TODO]
> A standard format for documenting I/O across stages is to be defined.

Output files should be inspected to verify that shapes, types (float32 vs float64), units, and coordinate conventions match between the output and the next stage's expected input.

## I/O Validation

Each stage validates its inputs and outputs in two steps:

1. **Schema validation.** Verify that files match the spec -- all required fields present, correct types, shapes, and file format (NetCDF/HDF5/TOML). The spec tables were initially derived from upstream documentation, not direct inspection of code output. If actual files differ from the spec, update the spec to match the actual input/output data.

2. **Physics validation.** Check that values are physically sensible and compatible with downstream stages: finiteness, non-negativity where expected, conservation laws, cross-stage consistency (e.g., iota in `boozmn_*.nc` matches `wout_*.nc`), convergence criteria met, etc.. The goal is to catch bad data at stage boundaries before it silently propagates through the pipeline.

Each stage's `spec.md` lists the specific schema and physics checks under its Input Validation and Output Validation sections. Validation methods should be implemented as testable functions so they can run as part of the test suite (see [Writing Tests](#writing-tests)).

## Known Risks

**1. Source-build fragility.** Some upstream codes have no release versions and must be built from source. Pin to a tested git commit SHA in `stages/pixi.toml`. Test builds in CI and maintain fallback known-good revisions.

## Coding Conventions

Key points for stage work:

- No cross-stage Python imports during Phase 1: all inter-stage communication is through files (e.g. NetCDF/HDF5). This is necessary to maintain swappability.
- Follow PEP 8 with 120-char line width
- Add type hints to all function signatures
- Use NumPy-style docstrings with equation references
- Use `logging` module, not `print()` -- logging supports severity levels, can be redirected to files, and is silenceable in production without modifying code
- Seed all RNGs for reproducibility
- Save configuration alongside outputs
- Guard against numerical edge cases -- division by zero, log(0), overflow in exp()
- Propagate device and dtype from input data -- don't hardcode `device='cuda'` or `dtype=jnp.float32`
