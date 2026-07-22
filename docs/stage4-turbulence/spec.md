# Stage 4: Turbulence

## Overview

Stage 4 solves the gyrokinetic equations to compute turbulent transport. The primary outputs -- heat and particle fluxes -- are both optimization objectives (to minimize) AND direct transport inputs for Stage 5.

**JAX-first priority:** `SPECTRAX-GK` is the primary code (JAX-native, differentiable). `GX` and `GENE` are traditional alternatives added later.

**Position in pipeline:** Receives geometry from Stage 1/2. Runs in parallel with Stage 3 (Neoclassical). Outputs feed Stage 5 (Transport).

**Important coordination point:** The coupling between `SPECTRAX-GK` output and `NEOPAX` (Stage 5) is less mature than the `GX`-`Trinity3D` coupling. `NEOPAX` has turbulence-coupling utilities but the public examples focus on the neoclassical reduced model. The Stage 4 and 5 owners must coordinate on this interface.

Reference: `stellarator_workflow.tex`, Section 4.7; `stellarator_io_reference.tex`, Sections 3.9-3.10.

---

## Codes

### SPECTRAX-GK (Primary JAX)
- **Repository:** https://github.com/uwplasma/SPECTRAX-GK
- **Language:** Python/JAX
- **Role:** JAX-native gyrokinetic solver for differentiable turbulence calculations

### GX (Alternative)
- **Repository:** https://bitbucket.org/gyrokinetics/gx
- **Language:** Fortran/CUDA
- **Role:** GPU-native gyrokinetic code, mature coupling with `Trinity3D`

### GENE / GENE-3D (Alternative)
- **Website:** https://genecode.org
- **Language:** Fortran
- **Role:** High-fidelity grid-based Eulerian gyrokinetic code

### Installation & Platform

**`SPECTRAX-GK`:** Install via the Pixi environment. From the `stages`/ directory:

```
pixi install --environment stage-4-spectrax
```

See `docs/mvp-pipeline.md` for run commands and I/O details.

> [!TODO]
> Document installation instructions and platform notes for `GX` and `GENE`.

---

## Input Specification

Reference: `stellarator_io_reference.tex`, Sections 3.9-3.10.

### SPECTRAX-GK Inputs

| Field | Type | Description | Source |
|-------|------|-------------|--------|
| TOML config | file | Grid, geometry, physics toggles, time integration | User-specified |
| Geometry | analytic or `*.eik.nc` | Magnetic geometry (can be VMEC-derived) | Stage 1/2 |
| Species profiles | in config | Density, temperature, gradients per species | User-specified |
| Collisionality | in config | Collision parameters | User-specified |
| Beta | in config | Electromagnetic parameter | User-specified |

Required input fields:
- `species`: Defines the physical properties of the active plasma species.
     - Default: ion with `charge=1.0, mass=1.0, density=1.0, temperature=1.0, tprim=2.49, fprim=0.8, nu=0.0, kinetic=True`. (The code seems to have a default. But does not seem to run with not specifying it.)
- `geometry`: Specifies the magnetic equilibrium and flux surface geometry.
- `physics`: Sets the global physical assumptions for the plasma.
- `run`: Configures the execution mode.

Optional input fields:
- `grid`: Defines the resolution of the simulated phase-space.

     - Defaults: `Nx=48, Ny=48, Nz=64, Lx=62.8, Ly=62.8, boundary="periodic", jtwist=None, non_twist=False, kxfac=1.0, z_min=-pi, z_max=pi, y0=None, ntheta=None, nperiod=None, zp=None`

- `time`: Specifies time configurations.
     - Defaults: `t_max=100.0, dt=0.1, method="rk2", sample_stride=1, diagnostics_stride=1, diagnostics=True, save_state=False, checkpoint=False, implicit_restart=20, implicit_preconditioner=None, implicit_solve_method="batched", use_diffrax=True, diffrax_solver="Dopri8", diffrax_adaptive=False, diffrax_rtol=1e-5, diffrax_atol=1e-7, diffrax_max_steps=4096, progress_bar=False, fixed_dt=True, dt_min=1e-7, dt_max=None, cfl=0.9, cfl_fac=None, collision_split=False, collision_scheme="implicit", gx_real_fft=True, nonlinear_dealias=True, laguerre_nonlinear_mode="grid"`

- `init`: Controls how the initial perturbation is built.
     - Defaults: `init_field="density", init_amp=1e-5, init_single=True, random_seed=22, gaussian_init=False, gaussian_width=0.5, gaussian_envelope_constant=1.0, gaussian_envelope_sine=0.0, kpar_init=0.0, init_file=None, init_file_scale=1.0, init_file_mode="replace", init_electrons_only=False`
- `collisions`: Configures the collision operator. Controls collision, hypercollision, and end-damping parameters.
     - Defaults: `nu_hermite=1.0, nu_laguerre=2.0, nu_hyper=0.0, p_hyper=4.0, nu_hyper_l=0.0, nu_hyper_m=1.0, nu_hyper_lm=0.0, p_hyper_l=6.0, p_hyper_m=None, p_hyper_lm=6.0, D_hyper=0.0, p_hyper_kperp=2.0, hypercollisions_const=0.0, hypercollisions_kz=1.0, damp_ends_amp=0.1, damp_ends_widthfrac=0.125, damp_ends_scale_by_dt=False`. Note `p_hyper_m=None` is not a hard numeric default; the runtime follows the GX fallback min(20, Nm/2) when it is omitted.
- `normalization`: Sets the reference units used to non-dimensionalize the equations. 
     - Defaults: `contract="cyclone", rho_star=None, omega_d_scale=None, omega_star_scale=None, diagnostic_norm="gx", flux_scale=1.0, wphi_scale=1.0`
- `terms`: Controls which RHS terms are enabled, as multiplicative weights.
     - Defaults: `streaming=1.0, mirror=1.0, curvature=1.0, gradb=1.0, diamagnetic=1.0, , collisions=1.0, hypercollisions=1.0, hyperdiffusion=0.0, end_damping=1.0, apar=1.0, bpar=1.0, nonlinear=0.0`
- `experts`: Advanced special-purpose controls.
     - Defaults: `fixed_mode=False, iky_fixed=None, ikx_fixed=None, dealias_kz=False`

> [!NOTE]
> **Radial-scan bridge defaults and reproducibility.** The defaults listed above are the raw SPECTRAX-GK config defaults. The `spectrax_gk_radial_scan.py` bridge that the Snakemake forward pass runs generates a runtime TOML per flux surface rather than using the template verbatim, and its `prepare` shaping defaults differ: `nx = 12`, `ny = 12`, `ntheta = 30` (theta points for the generated VMEC geometry), `t_max = 10.0`, `sample_stride = 50`, `diagnostics_stride = 1` (these are also the values set in the `stage4.spectrax_gk` block of the quick-run config). The bridge does **not** emit `random_seed` into the generated TOMLs, so the SPECTRAX-GK default seed is left implicit and runs are not explicitly re-seeded through this path (a known reproducibility gap). The scan's `response_mode` defaults to `none`; in `fd_gradients` mode, unset `dkap_density` / `dkap_temperature` / `perturb_rel_step` each default to `0.5`.

### `GX` Inputs (Alternative)

| Field | Type | Description | Source |
|-------|------|-------------|--------|
| `run_name.in` | Input file | Geometry, species, domain, time stepping, diagnostics, resolution | User-specified |
| VMEC geometry | via geometry module | Field-line geometry from wout | Stage 1 |
| `omega=true` | flag | Enable growth-rate diagnostics | Config |
| `fluxes=true` | flag | Enable flux diagnostics | Config |

### `GENE` Inputs (Alternative)

Installation-dependent. Key physics contract: geometry from VMEC/Boozer, species profiles/gradients, collisionality, electromagnetic parameters, numerical grid settings.

### Input Validation

> SPECTRAX-GK: Scripts check input fields and requirements located at tests/stage4-turbulence/SPECTRAX-GK/test_io.py.

---

## Output Specification

Reference: `stellarator_io_reference.tex`, Sections 3.9-3.10.

### SPECTRAX-GK Outputs

| Field | Type | Description | Used As | Normalization | Units | 
|-------|------|-------------|---------|---------------|-------|
| `t` | 1D array (time) | Simulation time | Time axis / Independent variable | Dimensionless normalized with $R_0/v_{th}$, where $v_{th} = \sqrt{T/m}$, $T$ is the temperature, $m$ is mass, and $R_0$ is the radius.
| `dt` | 1D array (time) | Time step size | Diagnostic | Same as $t$ | 
| `gamma` | 1D array (time) | Growth rate time trace | Objective / screening / Convergence check | Normalized with $v_{th}/R_0$ |
| `omega` | 1D array (time) | Frequency time trace | Diagnostic / Convergence check | Same as gamma |
| `Wg` | 1D array (time) | Free energy (g) trace | Diagnostic | Normalization specified in toml file |
| `Wphi` | 1D array (time) | Free energy (phi) trace | Diagnostic | Normalization specified in toml file |
| `Wapar` | 1D array (time) | Free energy (A_parallel) trace | Diagnostic | Normalization specified in toml file |
| `energy` | 1D array (time) | Free energy trace | Diagnostic | Normalization specified in toml file |
| `heat_flux` | 1D array (time) | Heat flux time trace | **Transport input** | Normalization specified in toml file |
| `particle_flux` | 1D array (time) | Particle flux time trace | **Transport input** | Normalization specified in toml file |
| `heat_flux_s0` | 1D array (time) | Species-resolved particle flux time trace | Diagnostic | Normalization specified in toml file |
| `particle_flux_s0` | 1D array (time) | Species-resolved particle flux time trace |  Diagnostic | Normalization specified in toml file |

Output in CSV files, along with a json file that records the info for only last time step. These are the per-surface diagnostics; in the Snakemake forward pass they are written under each surface's run directory and reduced into the aggregated files below.

#### Aggregated forward-pass outputs (radial scan)

The Snakemake forward pass runs the SPECTRAX-GK radial scan as a per-surface fan-out (`stage4_prepare` checkpoint, one `stage4_run_one` job per flux surface, then `stage4_collect`; see [Per-surface fan-out](../mvp-pipeline.md#per-surface-fan-out-stages-3-and-4)). The `collect` step reduces the per-surface diagnostics into two HDF5 files.

**Forward-chain handoff:** `neopax_fluxes.h5` (HDF5), consumed by `NEOPAX` (Stage 5). Its schema is the exact subset checked by the in-repo contract validator `src/io_contracts.py` (`validate_neopax_fluxes`):

| Dataset | Shape | Meaning |
|---------|-------|---------|
| `rho` | `(n_radii,)` | Normalized radial coordinate (sorted ascending) |
| `r` | `(n_radii,)` | Physical minor-radius coordinate `r = a * rho` |
| `Gamma` | `(n_species, n_radii)` | Turbulent particle flux in NEOPAX units |
| `Q` | `(n_species, n_radii)` | Turbulent heat flux in NEOPAX units |
| `Upar` | `(n_species, n_radii)` | Parallel flow; **always written as zeros** (this path supplies no parallel-flow channel) |
| `meta/species_names` | `(n_species,)` | UTF-8 species labels (dataset inside the `meta` group) |

Required `meta` group attributes (checked by the contract): `particle_flux_units = "m^-2 s^-1"`, `heat_flux_units = "eV m^-2 s^-1"`, and `minor_radius_m`. The `meta` group also carries `rho_star_source`, `radial_flux_coordinate`, `conversion`, `reference_species_name`, and `manifest` provenance attributes. The scan drops the magnetic-axis (`rho = 0`) surface, so when the manifest records a dropped axis the `collect` step re-inserts a zero-flux `rho = 0` column before writing, restoring a full profile for Stage 5.

**Optional perturbed-flux datasets (finite-difference response mode).** When the manifest contains perturbed runs (Stage 4 config `response_mode: fd_gradients`), `collect` additionally writes the datasets below, keyed on a perturbation axis of length `n_perturb` (one entry per `(kind, species)` label pair). The base `Gamma` / `Q` / `rho` / `r` datasets are still computed from base runs only. These are additions outside the `validate_neopax_fluxes` required subset (the validator checks only the required fields and ignores extra datasets), and Stage 5 does not consume them yet.

| Dataset | Shape | Meaning |
|---------|-------|---------|
| `Gamma_perturbed` | `(n_perturb, n_species, n_radii)` | Turbulent particle flux from the perturbed-gradient runs; same NEOPAX-unit conversion and units as `Gamma` |
| `Q_perturbed` | `(n_perturb, n_species, n_radii)` | Turbulent heat flux from the perturbed-gradient runs; same units as `Q` |
| `perturb_delta` | `(n_perturb, n_radii)` | Signed gradient increment applied for that (label, surface) |
| `perturb_present` | `(n_perturb, n_radii)`, bool | True where a perturbed run exists; surfaces without one (including a re-inserted magnetic-axis column) hold zeros and False |
| `perturb_kind` | `(n_perturb,)` | String label (`density_gradient` / `temperature_gradient`) keying the first axis of the arrays above |
| `perturb_species` | `(n_perturb,)` | Species label keying the first axis of the arrays above |

**Companion aggregate:** `flux_summary.h5` (HDF5) -- the richer scan summary the plots and audits build on, holding `rho`, `r`, `rho_index`, `torflux`, `Er`, `heat_flux_total`, `particle_flux_total`, the averaging-window bounds, a `species` group (`names`, `heat_flux`, `particle_flux`), and a `meta` group. It is not read by Stage 5.

**Per-surface run tree** (under `outputs/<run>/stage4_turbulence/`): `manifest.json` at the stage directory records the surfaces, geometry, and profile provenance the `collect` step reduces over; `runs.csv` and `normalization_audit.csv` summarize the planned runs; `runs/rho_*/` holds one directory per surface (basenames like `rho_012_r0p4898`), each with the generated runtime `input.toml`, the local geometry cache `*.eik.nc`, `run.diagnostics.csv` (the per-surface time traces `collect` reads), and `run.summary.json`. In `fd_gradients` response mode each base surface additionally gets perturbed sibling directories `rho_*_fd_n_<species>` (density-gradient channel) and `rho_*_fd_t_<species>` (temperature-gradient channel), one per configured channel-species pair, with the same per-directory contents as a base run. Every run record in `manifest.json` and every row in `runs.csv` carries `response_group` (the base surface ordinal), `response_role` (`base` or `perturbed`), `perturb_kind` (`density_gradient` / `temperature_gradient`), `perturb_species`, and `perturb_delta` (the signed gradient increment); the manifest header additionally records `response_mode`, `perturb_density_species`, `perturb_temperature_species`, `dkap_density`, `dkap_temperature`, and `perturb_rel_step`.

The natural downstream contract is the same as `GX`: turbulent heat and particle flux (steady-state values).

### `GX` Outputs (Alternative)

| File | Description |
|------|-------------|
| `run_name.out.nc` | Linear run output |
| `run_name.nc` | Nonlinear run output |
| `run_name.big.nc` | Saved field diagnostics |
| `run_name.restart.nc` | Restart data |

Key NetCDF groups: `Grids`, `Geometry`, `Diagnostics`, `Inputs`

| Field | Location | Description | Used As |
|-------|----------|-------------|---------|
| `ParticleFlux_st` | `Diagnostics/` | Particle flux (species, time) | **Transport input** (`Trinity3D`) |
| `HeatFlux_st` | `Diagnostics/` | Heat flux (species, time) | **Transport input** (`Trinity3D`) |
| `pflux` | `Fluxes/` | Particle flux (alternative location) | Transport input |
| `qflux` | `Fluxes/` | Heat flux (alternative location) | Transport input |
| `ParticleFlux_zst` | `Diagnostics/` | Zeta-resolved particle flux (stellarator) | Transport input |
| `HeatFlux_zst` | `Diagnostics/` | Zeta-resolved heat flux (stellarator) | Transport input |
| `omega_v_time` | `Special/` | Linear growth rate vs time | Screening |

`GX` spectral representation: Hermite-Laguerre velocity-space basis:

$$h_s = \sum_{\ell,m,k_x,k_y} \hat{h}_{s,\ell,m}(z,t)\, e^{i(k_x x + k_y y)} H_m\left(\frac{v_\parallel}{v_{ts}}\right) L_\ell\left(\frac{v_\perp^2}{v_{ts}^2}\right) F_{Ms}$$

### `GENE` Outputs (Alternative)

Installation-dependent filenames. Key outputs: linear growth rates, real frequencies, eigenfunctions, nonlinear species heat/particle fluxes, spectra, time histories.

### Subset Handed to Next Stage

For transport coupling, the critical handoff is the **turbulent flux vector** (steady-state heat and particle flux per species). For screening, only linear gamma and omega may be retained.

`Trinity3D` obtains flux Jacobians by rerunning `GX` on perturbed gradients and finite-differencing.

### Outputs Used as Objectives

- Linear gamma, omega: rapid screening
- Nonlinear heat flux, particle flux: high-fidelity design objectives
- Heat flux is BOTH an objective AND a transport input (dual-role output)

### Output Validation

> [!TODO]
> See [I/O Validation section](../guide.md#io-validation).

---

## Governing Equations

Generic delta-f gyrokinetic equation:

$$\frac{\partial h_s}{\partial t} + v_\parallel \mathbf{b}\cdot\nabla h_s + \mathbf{v}_{Ds}\cdot\nabla h_s + \mathbf{v}_E\cdot\nabla h_s - C[h_s] = -\frac{Z_s e F_{Ms}}{T_s}\frac{\partial\langle\chi\rangle}{\partial t} - \mathbf{v}_\chi\cdot\nabla F_{Ms}$$

Closed by quasineutrality and (for electromagnetic calculations) appropriate field equations.

Reference: `stellarator_workflow.tex`, Section 4.7.

---

## Convergence & Validity

> SPECTRAX-GK: Scripts checking convergence and stable flux located at tests/stage4-turbulence/SPECTRAX-GK/test_convergence_and_flux.py.
---

## API Documentation

> [!TODO]
> Document key entry-point functions, programmatic usage, JAX differentiation, and configuration effects.

---

## Scripts & Workflows

**`SPECTRAX-GK` (via Pixi):** From the `stages`/ directory:

```
pixi run stage-4-spectrax
```

which executes something morally equivalent to:

```
python -m spectraxgk.cli run --config inputs/quick_run/HSX_vacuum_ns201_quickrun.toml --out outputs/quick_run/stage4_turbulence/hsx_run
```

> [!NOTE]
> The pixi `stage-4-spectrax` task runs the all-in-one radial scan, which fans one `python -m spectraxgk.cli run` subprocess out per flux surface; the command above is the underlying per-surface invocation.

**Input:** `outputs/quick_run/stage1_equilibrium/wout_HSX_vacuum_ns201_quickrun.nc` + `inputs/quick_run/HSX_vacuum_ns201_quickrun.toml`
**Output:** `outputs/quick_run/stage4_turbulence/neopax_fluxes.h5` (+ `flux_summary.h5`, `manifest.json`, `runs.csv`)

> [!NOTE]
> The TOML's `vmec_file` points into `outputs/quick_run/stage1_equilibrium/`. Populate this directory by running `pixi run stage-1-vmec` first. The VMEC geometry path also requires `booz_xform_jax` at runtime (lazy dependency).

See `docs/mvp-pipeline.md` for full I/O details.

> [!TODO]
> Add standalone run scripts and workflows for `GX` and `GENE`.

---

## W&B Tracking

**Project:** `driftless-star-stage4-turbulence`

> [!TODO]
> Set up W&B tracking.

---

## Container Specification (Phase 2)

**`SPECTRAX-GK`:** Built from the single templated `stages/Dockerfile` using build arguments:

```
docker build --file stages/Dockerfile --build-arg ENVIRONMENT=stage-4-spectrax stages/        # CPU
docker build --file stages/Dockerfile --build-arg ENVIRONMENT=stage-4-spectrax-gpu --build-arg CUDA_VERSION=12 stages/  # GPU
```

Published to GHCR as `ghcr.io/driftless-star/driftless-star:stage-4-spectrax-cpu` and `stage-4-spectrax-gpu`. CI builds via `.github/workflows/containers.yml`.

See [guide](../guide.md#container-architecture) for full architecture details.

> [!TODO]
> Define container specifications for `GX` and `GENE`.

---

## Tests (Phase 2)
- SPECTRAX-GK:
    - Unit tests: Tests IO, convergence and stable flux.
    - Regression tests: [TODO]
    - Integration tests: [TODO]


---

## Claude Skills

> [!TODO]
> Create dev, operational, and cross-stage Claude skills for SPECTRAX-GK and GX workflows.
> See [guide](../guide.md#step-7-create-claude-skills) for skill types.
