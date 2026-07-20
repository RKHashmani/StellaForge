# Stage 5: Transport & Power Balance

## Overview

Stage 5 evolves radial density and temperature profiles using neoclassical and
turbulent fluxes, and computes whole-device fusion-power metrics.  This is the
final stage of the forward pass, producing transport-consistent profiles and the
headline numbers ($P_\text{fus}$, $Q$).

**JAX-first priority:** `NEOPAX` is the primary code (JAX-native, uses diffrax ODE
solver).  `Trinity3D` is the traditional alternative with mature `GX`+`SFINCS`
coupling.

**Position in pipeline:** Receives neoclassical transport data from Stage 3 (`sfincs_jax`),
turbulent fluxes from Stage 4 (`SPECTRAX-GK`), and geometry from Stage 1/2.
Produces the forward-pass output: updated $n(r)$, $T(r)$, $E_r(r)$,
$P_\text{fus}$, $Q$.

**Important note on `NEOPAX` turbulence coupling:** `NEOPAX` has
turbulence-coupling utilities, but coupling `SPECTRAX-GK` turbulent flux into
`NEOPAX` is a coordination point with the Stage 4 owner.  The alternative path
(`GX` -> `Trinity3D`) has mature, tested turbulence coupling.

**Outer-loop handoff (future, not forward pass):** Updated pressure and current
profiles feed back to Stage 1 for the next iteration.

> [!NOTE]
> This loop closure is not part of the initial forward-pass goal.

Reference: `stellarator_workflow.tex`, Sections 4.8--4.9;
`stellarator_io_reference.tex`, Sections 3.11--3.12.

---

## Codes

### NEOPAX (Primary JAX)

- **Repository:** <https://github.com/uwplasma/NEOPAX>
- **Language:** Python / JAX
- **Role:** Reduced neoclassical transport and profile evolution from
  neoclassical flux inputs (Stage 3).  Uses diffrax for JAX-native ODE integration.

### Trinity3D (Alternative)

- **Repository:** <https://bitbucket.org/gyrokinetics/t3d>
- **Documentation:** (see `Trinity3D` docs)
- **Language:** Python
- **Role:** Global transport solver coupling `GX` turbulence and `SFINCS`
  neoclassical fluxes.  Implicit linearized time advance.

### Installation & Platform

**`NEOPAX`:** Install via the Pixi environment. From the `stages`/ directory:

```
pixi install --environment stage-5-neopax
```


**Trinity3D:** Install via the Pixi environment. From the `stages`/ directory:

```
pixi install --environment stage-5-t3d
```

See `docs/mvp-pipeline.md` for run commands and I/O details.

> [!TODO]
> Document installation instructions and platform notes for `Trinity3D`.

---

## Input Specification

Reference: `stellarator_io_reference.tex`, Sections 3.11-3.12.

### NEOPAX Inputs

| Field | Type | Description | Source |
|-------|------|-------------|--------|
| `wout_*.nc` | NetCDF | VMEC equilibrium geometry | Stage 1 |
| `boozmn_*.nc` | NetCDF | Boozer-coordinate equilibrium | Stage 2 |

**Optional profile initialization:** NTSS-like HDF5 files with arrays: `r`,
`Er`, `Te`, `ne`, `Pressure`, `I_bs`, and related transport quantities.

### Trinity3D Inputs (Alternative)

| Field | Type | Description | Source |
|-------|------|-------------|--------|
| TOML config (`*.in`) | file | Groups: `[grid]`, `[time]`, `[[model]]`, `[[species]]`, `[geometry]`, `[physics]`, `[log]` | User-specified |
| `wout_*.nc` | via `[geometry]` | VMEC geometry | Stage 1 |
| `gx_template` | via `[[model]]` | `GX` input template for turbulence model | Stage 4 (`GX`) |
| `gx_outputs` | via `[[model]]` | `GX` flux outputs location | Stage 4 (`GX`) |
| `SFINCS` fluxes | via `[[model]]` | Neoclassical fluxes | Stage 3 (`SFINCS`) |

### Input Validation

> [!TODO]
> See [I/O Validation section](../guide.md#io-validation).

---

## Output Specification

Reference: `stellarator_io_reference.tex`, Sections 3.11-3.12.

### NEOPAX Outputs

**Core API returns (in-memory):**

| Field | Type | Description | Used As |
|-------|------|-------------|---------|
| `Lij` | 2D array | Thermal transport matrix | Transport analysis |
| `Gamma` | array (per species) | Particle flux | Objective / transport |
| `Q` | array (per species) | Heat flux | Objective / transport |
| `Upar` | array (per species) | Parallel flow | Diagnostic |

**HDF5 outputs (minimal):**

| Field | Type | Description |
|-------|------|-------------|
| `rho` | 1D array | Radial coordinate |
| `Er` | 1D array | Ambipolar radial electric field |
| `Jboots` | 1D array | Bootstrap current density |

**HDF5 outputs (full NTSS-style):**

| Field | Type | Description | Used As |
|-------|------|-------------|---------|
| `Pressure` | 1D array | Total pressure profile | **Outer-loop handoff** |
| `FluxNeo` | 1D array | Neoclassical particle flux | Diagnostic |
| `FluxQe` | 1D array | Electron heat flux | Diagnostic |
| `FluxQi` | 1D array | Ion heat flux | Diagnostic |
| `AmbiFlux` | 1D array | Ambipolar flux | Diagnostic |
| `J_bs` | 1D array | Bootstrap current density | **Outer-loop handoff** |
| `I_bs` | scalar | Total bootstrap current | Objective |
| `I_tor` | scalar | Total toroidal current | Objective |
| `beta` | 1D array | Local beta | Objective |
| `Te` | 1D array | Electron temperature profile | **Forward-pass output** |
| `TD` | 1D array | Deuterium temperature | Forward-pass output |
| `Tt` | 1D array | Tritium temperature | Forward-pass output |
| `ne` | 1D array | Electron density profile | **Forward-pass output** |
| `nD` | 1D array | Deuterium density | Forward-pass output |
| `nHe` | 1D array | Helium density (ash) | Forward-pass output |

### Trinity3D Outputs (Alternative)

**NetCDF / ADIOS2 / numpy-log outputs:**

Geometry group:

| Field | Description |
|-------|-------------|
| `B0` | Reference magnetic field |
| `Btor` | Toroidal field |
| `a_minor` | Minor radius |
| `R_major` | Major radius |
| `area` | Flux surface area |
| `grho` | Geometric factor |

Profile histories:

| Field | Description |
|-------|-------------|
| `n_e` | Electron density vs ($\rho$, time) |
| `T_e` | Electron temperature vs ($\rho$, time) |
| `n_H` | Hydrogen density vs ($\rho$, time) |
| `T_H` | Hydrogen temperature vs ($\rho$, time) |

Flux histories:

| Field | Description |
|-------|-------------|
| `pflux_*` | Particle flux (total + per-model) |
| `qflux_*` | Heat flux (total + per-model) |

Gradients and sources:

| Field | Description |
|-------|-------------|
| `aLn_*` | Density gradient scale lengths |
| `aLT_*` | Temperature gradient scale lengths |
| `Sn_*` | Particle sources |
| `Sp_*` | Power sources |

Device metrics:

| Field | Description | Used As |
|-------|-------------|---------|
| `beta_vol` | Volume-averaged beta | Objective |
| `Paux_MW` | Auxiliary power (MW) | Objective |
| `Palpha_int_MW` | Alpha heating power (MW) | Diagnostic |
| `Pfus_MW` | Fusion power (MW) | **Key forward-pass output** |
| `Qfus` | Fusion gain $Q = P_\text{fus} / P_\text{aux}$ | **Key forward-pass output** |

### Subset: Forward-Pass Output

The essential forward-pass outputs are: $n_s(r)$, $T_s(r)$, $E_r(r)$, and
whole-device metrics $P_\text{fus}$ and $Q$.

### Subset: Outer-Loop Handoff (Future)

Updated pressure $p(r) = \sum_s n_s(r)\, T_s(r)$ and bootstrap-current
profiles, to be fed back to Stage 1.  NOT part of the initial forward-pass
scope.

### Outputs Used as Objectives

Ambipolar $E_r$, bootstrap current, toroidal current, neoclassical fluxes,
$P_\text{fus}$, $Q$, $\beta$, transport-consistent profiles.

### Output Validation

> [!TODO]
> See [I/O Validation section](../guide.md#io-validation).

---

## Governing Equations

### NEOPAX: Reduced Neoclassical Transport

Thermal transport coefficients assembled into $L_{ij}$ matrix.  Particle flux,
heat flux, and parallel flow:

$$\Gamma_a = -n_a\bigl(L_{11}A_1 + L_{12}A_2 + L_{13}A_3\bigr)$$

$$Q_a = -T_a\, n_a\bigl(L_{21}A_1 + L_{22}A_2 + L_{23}A_3\bigr)$$

$$U_{\parallel a} = -n_a\bigl(L_{31}A_1 + L_{32}A_2 + L_{33}A_3\bigr)$$

Ambipolar $E_r$ from charge-flux imbalance:

$$S_{E_r} \propto -\Gamma_e + \Gamma_D + \Gamma_T$$

Profile evolution integrated with diffrax (JAX ODE solver).

### Trinity3D: 1D Conservation Laws

$$\frac{\partial n_s}{\partial\tau} + \mathcal{G}(\rho)\frac{\partial F_{n,s}}{\partial\rho} = S_{n,s}$$

$$\frac{\partial p_s}{\partial\tau} + \mathcal{G}(\rho)\frac{\partial F_{p,s}}{\partial\rho} = \frac{2}{3}S_{p,s}$$

Implicit linearized time advance:

$$(d_1 I + \alpha\Psi)\,y^{m+1} = -d_0 y^m - d_{-1}y^{m-1} + \alpha\Psi y^m - \alpha\bigl[G(F^+ - F^-) - S\bigr] - (1-\alpha)\bigl[G(F^+_m - F^-_m) - S_m\bigr]$$

Flux Jacobians obtained by finite-differencing perturbed `GX` runs.

### Fusion Power (both codes)

$$P_\text{fus} = \int dV\; n_D\, n_T\, \langle\sigma v\rangle_{DT}\, E_{DT}$$

with $E_{DT} \sim 17.6$ MeV and Bosch--Hale thermal reactivity.

Reference: `stellarator_workflow.tex`, Sections 4.8--4.9.

---

## Convergence & Validity

> [!TODO]
> Document convergence criteria for NEOPAX (diffrax tolerances, steady-state) and Trinity3D (implicit time-step, flux Jacobians), plus known failure modes.

---

## API Documentation

> [!TODO]
> Document Python APIs for NEOPAX and Trinity3D, including the driftless-star adapter interface and example call sequences.

---

## Scripts & Workflows

> [!TODO]
> A reference run script for `NEOPAX` will be added once the Stage 3 → Stage 5 handoff via `sfincs_jax` is designed.

See `docs/mvp-pipeline.md` for full I/O details.

> [!TODO]
> Add standalone run scripts and workflows for `Trinity3D`.

---

## W&B Tracking

**Project:** `driftless-star-stage5-transport`

> [!TODO]
> Set up W&B tracking.

---

## Container Specification (Phase 2)

**`NEOPAX`:** Built from the single templated `stages/Dockerfile` using build arguments:

```
docker build --file stages/Dockerfile --build-arg ENVIRONMENT=stage-5-neopax stages/        # CPU
docker build --file stages/Dockerfile --build-arg ENVIRONMENT=stage-5-neopax-gpu --build-arg CUDA_VERSION=12 stages/  # GPU
```

Published to GHCR as `ghcr.io/driftless-star/driftless-star:stage-5-neopax-cpu` and `stage-5-neopax-gpu`. CI builds via `.github/workflows/containers.yml`.

See [guide](../guide.md#container-architecture) for full architecture details.

> [!TODO]
> Define container specifications for `Trinity3D`.

---

## Tests (Phase 2)

> [!TODO]
> Define unit, integration, regression, and performance tests for Stage 5 transport computations.
> See [guide](../guide.md#writing-tests) for examples.

---

## Claude Skills

> [!TODO]
> Define Claude Code skills for running NEOPAX, comparing results, and diagnosing failure modes.
> See [guide](../guide.md#step-7-create-claude-skills) for skill types.
