# Stage 1: Equilibrium

## Overview

Stage 1 solves the three-dimensional ideal-MHD equilibrium problem, producing the magnetic field geometry and flux-surface profiles that all downstream stages depend on. This is the entry point of the forward-pass pipeline.

**Physics:** Given a plasma boundary shape and profile guesses (pressure, rotational transform or current), find the 3D magnetic equilibrium satisfying force balance: $\nabla p = \mathbf{J} \times \mathbf{B}$, $\nabla \cdot \mathbf{B} = 0$.

**Position in pipeline:** This stage has no upstream dependencies. Its output (`wout_*.nc`) is consumed by Stage 2 (Boozer Transform) and also directly by some turbulence and transport codes.

Reference: `stellarator_workflow/stellarator_workflow.tex`, Section 4.1 (`VMEC++` and `vmec_jax`) and Section 4.2 (`DESC`).

---

## Codes

### vmec_jax (Primary JAX)
- **Repository:** https://github.com/uwplasma/vmec_jax
- **Language:** Python/JAX
- **Role:** JAX-native implementation providing differentiable equilibrium solving with wout-compatible output

### VMEC++ (C++ Alternative)
- **Repository:** https://github.com/proximafusion/vmecpp
- **Documentation:** https://proximafusion.github.io/vmecpp/
- **Language:** C++ with Python bindings
- **Role:** From-scratch C++ reimplementation of `VMEC`. Solves fixed- and free-boundary ideal-MHD equilibria. Preserves the standard `wout` downstream contract.

### DESC (Differentiable Alternative)
- **Repository:** https://github.com/PlasmaControl/DESC
- **Language:** Python/JAX
- **Role:** Differentiable pseudo-spectral equilibrium and optimization suite. Can replace `VMEC++` as the equilibrium engine and also perform some downstream computations (Boozer transform, geometry objectives) internally.

### Installation & Platform

**`vmec_jax`:** Install via the Pixi environment. From the `stages`/ directory:

```
pixi install --environment stage-1-vmec
```

**`desc-opt`:** Install via the Pixi environment. From the `stages`/ directory:

```
pixi install --environment stage-1-desc
```

See `docs/mvp-pipeline.md` for run commands and I/O details.

> [!TODO]
> Document installation instructions and platform notes for `VMEC++` and `DESC`.

---

## Input Specification

`vmec_jax` reads VMEC-style Fortran namelist files with an `&INDATA ... /` block through `vmec_jax.namelist.read_indata`. The parser preserves VMEC naming and indexed Fourier coefficients: scalar/list assignments live in `InData.scalars`, while indexed assignments such as `RBC(0,1)` live in `InData.indexed`.

The radial coordinate for profiles is `s = toroidal_flux / edge_toroidal_flux`, dimensionless and normalized to `[0, 1]`. Profile coefficient arrays use ascending powers of `s` for `power_series` types: `AM[0] + AM[1] s + ...`.

### Required Core Fields

| Field | Type | Required | Normalization / Convention | Units | Description |
|-------|------|----------|----------------------------|-------|-------------|
| `NFP` | int scalar | Yes | Field periods; must be positive | dimensionless | Number of toroidal field periods. |
| `MPOL` | int scalar | Yes | VMEC poloidal resolution; must be positive; the main `vmec_jax` mode table uses `m = 0..MPOL-1` | dimensionless | Poloidal mode-count/limit used to build the main Fourier basis. |
| `NTOR` | int scalar | Yes | VMEC toroidal resolution; must be non-negative | dimensionless | Highest toroidal mode index in field-period-normalized mode number `n`; `NTOR=0` is axisymmetric. |
| `NS_ARRAY` or `NS` | int or list[int] | Yes | Radial grid count(s); finest grid is the last `NS_ARRAY` entry | dimensionless | Number of radial mesh points. `NS_ARRAY` enables multigrid continuation; `NS` is a fallback. Values should be at least 3. |
| `PHIEDGE` | float scalar | Yes | VMEC edge toroidal flux; sign participates in handedness through `signgs` | Weber (Wb) in physical convention | Total toroidal flux at the plasma edge. `vmec_jax` converts to `phipf`/`phips` profiles using VMEC conventions. |
| `RBC(n,m)` | indexed float coefficients | Yes | Boundary Fourier coefficient for `R = sum RBC(n,m) cos(m theta - n NFP zeta) + ...` | meters | Stellarator-symmetric cosine coefficients of the fixed plasma boundary. `RBC(0,0)` is the major-radius offset and should normally be present. |
| `ZBS(n,m)` | indexed float coefficients | Yes for stellarator-symmetric inputs | Boundary Fourier coefficient for `Z = sum ZBS(n,m) sin(m theta - n NFP zeta) + ...` | meters | Stellarator-symmetric sine coefficients of the fixed plasma boundary. |

### Optional Boundary And Symmetry Fields

| Field | Type | Required | Normalization / Convention | Units | Description |
|-------|------|----------|----------------------------|-------|-------------|
| `LASYM` | bool scalar | No; defaults false | VMEC logical | dimensionless | Enables non-stellarator-symmetric Fourier channels. |
| `RBS(n,m)` | indexed float coefficients | Required only when used with `LASYM=T` | Sine channel for `R` | meters | Asymmetric boundary coefficients. Ignored by symmetric inputs. |
| `ZBC(n,m)` | indexed float coefficients | Required only when used with `LASYM=T` | Cosine channel for `Z` | meters | Asymmetric boundary coefficients. Ignored by symmetric inputs. |
| `LCONM1` | bool scalar | No; defaults true | VMEC `m=1` boundary constraint flag | dimensionless | Controls the VMEC m=1 constraint applied during boundary handling. |
| `RAXIS_CC`, `ZAXIS_CS`, `RAXIS_CS`, `ZAXIS_CC` | list[float] | No | Axis Fourier coefficients indexed by toroidal mode | meters | Optional magnetic-axis initial guess. Missing axis data are inferred from the boundary unless restart/override logic supplies them. |

### Profile Fields

| Field | Type | Required | Normalization / Convention | Units | Description |
|-------|------|----------|----------------------------|-------|-------------|
| `PMASS_TYPE` | string | No; defaults `power_series` | Supported: `power_series`, `two_power`, `cubic_spline`, `akima_spline`, `line_segment` | dimensionless | Selects the pressure-profile representation. |
| `AM` | list[float] or scalar | No; defaults zero pressure | Ascending `s` powers for `power_series`; `two_power` uses VMEC two-power parameters | Pa before scaling | Pressure profile coefficients. Physical input pressure is `PRES_SCALE * profile(clip(abs(s * BLOAT), 1))` in Pa. |
| `AM_AUX_S`, `AM_AUX_F` | list[float], list[float] | Required when `PMASS_TYPE` is spline/line segment | `AM_AUX_S` strictly increasing in normalized `s`, normally within `[0, 1]` | `AM_AUX_F` in Pa before scaling | Tabulated pressure knots and values. |
| `PRES_SCALE` | float scalar | No; defaults `1.0` | Multiplicative scale on pressure profile | dimensionless | Scales `AM` or `AM_AUX_F`. `vmec_jax` stores solver pressure internally as `mu0 * Pa`. |
| `BLOAT` | float scalar | No; defaults `1.0` | Evaluates profiles at `clip(abs(s * BLOAT), 1)` | dimensionless | VMEC radial-profile stretching factor. |
| `SPRES_PED` | float scalar | No; defaults `1.0` | Pedestal clamp location in normalized `s` | dimensionless | If less than 1, pressure outside this `s` is clamped to the pedestal value. |
| `NCURR` | int scalar | No; defaults `0` | `0` means iota-prescribed; `1` means current-prescribed | dimensionless | Selects whether `AI` or `AC` is the active equilibrium profile. |
| `PIOTA_TYPE` | string | No; defaults `power_series` | Supported: `power_series`, `cubic_spline`, `akima_spline`, `line_segment` | dimensionless | Selects the rotational-transform profile representation. |
| `AI` | list[float] or scalar | Required for nonzero iota-prescribed profiles; otherwise defaults zero | Ascending `s` powers for `power_series` | dimensionless | Rotational-transform profile coefficients for `NCURR=0`. If `LRFP=T`, input is interpreted as inverse transform and converted to iota. |
| `AI_AUX_S`, `AI_AUX_F` | list[float], list[float] | Required when `PIOTA_TYPE` is spline/line segment | `AI_AUX_S` strictly increasing in normalized `s` | dimensionless | Tabulated rotational-transform knots and values. |
| `PCURR_TYPE` | string | No; defaults `power_series` | Supported: `power_series`, `power_series_i`, `two_power`, `cubic_spline_i`, `cubic_spline_ip`, `akima_spline_i`, `akima_spline_ip`, `line_segment_i`, `line_segment_ip` | dimensionless | Selects the current-profile representation and whether coefficients describe enclosed current (`*_i`) or its radial derivative (`*_ip`). |
| `AC` | list[float] or scalar | Required for nonzero current-prescribed profiles; otherwise defaults zero | For `power_series`, coefficients parameterize `I'(s)` and are integrated to `I(s)`; `power_series_i` parameterizes enclosed `I(s)` directly | amperes in physical VMEC convention | Toroidal-current profile coefficients for `NCURR=1`. |
| `AC_AUX_S`, `AC_AUX_F` | list[float], list[float] | Required when `PCURR_TYPE` is spline/line segment | `AC_AUX_S` strictly increasing in normalized `s` | amperes or amperes per normalized flux, depending on `*_i` vs `*_ip` | Tabulated current-profile knots and values. |
| `LRFP` | bool scalar | No; defaults false | VMEC logical | dimensionless | When true, iota-like inputs are inverted as in RFP-style VMEC handling. |

### Solver And Runtime Controls

| Field | Type | Required | Normalization / Convention | Units | Description |
|-------|------|----------|----------------------------|-------|-------------|
| `NITER` | int scalar | No; solver default if absent | Positive iteration budget | iterations | Single-grid maximum iteration count. |
| `NITER_ARRAY` | list[int] | No | Length should be 1 or match `NS_ARRAY` | iterations per stage | Multigrid iteration budgets. |
| `FTOL_ARRAY` | list[float] | No | Length should be 1 or match `NS_ARRAY`; positive | VMEC residual tolerance | Multigrid convergence tolerances. |
| `DELT` | float scalar | No; code default if absent | Solver step-size control | dimensionless | Initial pseudo-time/gradient step size used by fixed-boundary drivers. |
| `GAMMA` | float scalar | No; defaults `0.0` | Ideal-MHD adiabatic index parameter | dimensionless | Used by pressure/energy paths. |
| `LFREEB` | bool scalar | No; defaults false | Effective only when `MGRID_FILE` is not `NONE` | dimensionless | Enables free-boundary mode. |
| `MGRID_FILE` | string | Required for active free-boundary runs | Path is resolved relative to input file if not absolute | path | External magnetic grid file. `LFREEB=T` with `MGRID_FILE='NONE'` is treated as fixed-boundary. |
| `EXTCUR` or `EXTCUR(i)` | list[float] or indexed floats | Required by some free-boundary mgrid files | Coil-current vector in VMEC order | amperes | External coil-current amplitudes. |
| `NVACSKIP` | int scalar | No; defaults to `NFP` when `<=0` | Free-boundary vacuum update cadence | iterations | Number of plasma iterations between vacuum-field updates. |

### Input Formats
- **INDATA files:** Fortran-style text `input.NAME` format (`vmec_jax` and `VMEC++`)
- **Python objects:** `vmec_jax.namelist.InData`, `VMECConfig`, `VMECState`, and `WoutData` in-memory API objects
- **Hot restart:** Previous `wout_*.nc` or reconstructed `VMECState` as an initial guess where supported
- **JSON:** Not a native `vmec_jax` input format; use VMEC++ or convert to `&INDATA`

### Input Validation

The test helper `test_io_validation.py` contains plain validation checks for dictionary-like input payloads. `validate_input_payload(payload)` checks the documented contract without changing the permissive VMEC namelist parser. It reports missing required fields, invalid scalar ranges, missing boundary coefficient families, and inconsistent `NS_ARRAY`/`NITER_ARRAY`/`FTOL_ARRAY` lengths.

---

## Output Specification

The primary output is a VMEC-compatible `wout_*.nc` NetCDF file or the equivalent in-memory `vmec_jax.io.wout_files.schema.WoutData`. Unless stated otherwise, arrays are stored with VMEC names and shapes. The radial dimension is `radius = ns`; main Fourier fields use `mn_mode = mnmax`; derived Nyquist fields use `mn_mode_nyq = mnmax_nyq`.

### Required Metadata

| Field | Type / Shape | Required | Normalization / Convention | Units | Description |
|-------|--------------|----------|----------------------------|-------|-------------|
| `ns` | int scalar | Yes | Radial mesh count | dimensionless | Number of radial points. |
| `mpol`, `ntor`, `nfp` | int scalars | Yes | Same convention as input | dimensionless | Resolution and field-period metadata. |
| `mnmax`, `mnmax_nyq` | int scalars | Yes | Number of main and Nyquist Fourier modes | dimensionless | Mode-table lengths. |
| `mpol_nyq`, `ntor_nyq` | int scalars | Yes in `vmec_jax` output | Nyquist resolution | dimensionless | Maximum Nyquist mode indices. |
| `lasym__logical__` / `lasym` | int/bool scalar | Yes | VMEC logical stored as 0/1 in NetCDF | dimensionless | Whether asymmetric Fourier channels are active. |
| `signgs` | int scalar | Yes | VMEC coordinate handedness sign | dimensionless | Sign used in flux/Jacobian conventions. |
| `xm`, `xn` | arrays `(mnmax,)` | Yes | `xm=m`; `xn=n*NFP` | dimensionless | Main Fourier mode table. |
| `xm_nyq`, `xn_nyq` | arrays `(mnmax_nyq,)` | Yes | Nyquist `m`; Nyquist `n*NFP` | dimensionless | Nyquist Fourier mode table. |
| `ier_flag` | int scalar | Yes in `vmec_jax` output | `0` means converged by VMEC convention | dimensionless | Solver status flag. |
| `vmec_jax_converged__logical__`, `vmec_jax_status` | bool/string | Yes in `vmec_jax` output | Extra status metadata | dimensionless / text | Explicit convergence status for differentiable/partial diagnostic outputs. |

### Geometry Scalars

| Field | Type | Description | Used As |
|-------|------|-------------|---------|
| `aspect` | scalar float | `Rmajor_p / Aminor_p`, dimensionless. Required in `vmec_jax` output. | Objective |
| `Aminor_p` | scalar float | Effective minor radius from LCFS cross-sectional area, meters. Required. | Geometry |
| `Rmajor_p` | scalar float | Effective major radius, meters. Required. | Geometry |
| `volume_p` | scalar float | Plasma volume, cubic meters in physical convention. Required. | Objective |
| `betatotal`, `betapol`, `betator`, `betaxis` | scalar floats | VMEC beta diagnostics, dimensionless. Present in `vmec_jax` output; zero if not computed. | Objective / diagnostics |
| `wb` | scalar float | Magnetic energy integral in VMEC normalization. Required. | Diagnostics |
| `wp` | scalar float | Pressure energy integral in VMEC normalization. Present in `vmec_jax` output. | Diagnostics |
| `fsqr`, `fsqz`, `fsql` | scalar floats | VMEC force residual diagnostics for radial, vertical, and lambda/constraint equations. Required. | QA signal |
| `ctor` | scalar float | Total toroidal current diagnostic. Present in `vmec_jax` output. | Diagnostics |

### Radial Profiles

| Field | Type / Shape | Required | Normalization / Convention | Units | Description |
|-------|--------------|----------|----------------------------|-------|-------------|
| `phi` | float `(ns,)` | Yes | Toroidal flux profile; axis entry is zero | Wb in physical convention | Integrated toroidal flux. |
| `phipf` | float `(ns,)` | Yes | `d phi / ds` on full mesh | Wb | Toroidal-flux derivative. |
| `phips` | float `(ns,)` | Yes | VMEC half-mesh toroidal-flux derivative storage | Wb | Half-mesh flux derivative, axis slot included. |
| `chipf` | float `(ns,)` | Yes | `d chi / ds` on full mesh | Wb | Poloidal-flux derivative. |
| `iotas` | float `(ns,)` | Yes | Rotational transform on VMEC half mesh, axis slot included | dimensionless | Half-mesh iota profile. |
| `iotaf` | float `(ns,)` | Yes | Rotational transform on full mesh | dimensionless | Full-mesh iota profile. |
| `pres` | float `(ns,)` | Yes | Half-mesh pressure with axis slot included; on disk VMEC stores Pa, `WoutData.pres` stores `mu0 * Pa` | Pa on disk; `mu0*Pa` in memory | Pressure profile. |
| `presf` | float `(ns,)` | Yes | Full-mesh pressure; same unit conversion as `pres` | Pa on disk; `mu0*Pa` in memory | Full-mesh pressure profile. |
| `vp` | float `(ns,)` | Yes | Volume derivative on half mesh, normalized by `(2*pi)^2` in VMEC convention | m^3 in physical convention after VMEC scaling | Radial volume derivative. |
| `buco`, `bvco` | float `(ns,)` | Yes in `vmec_jax` output | Flux-surface covariant magnetic-field functions | tesla-meter in physical convention | VMEC `B_u`/`B_v` profile diagnostics, related to Boozer `I`/`G` conventions. |
| `jcuru`, `jcurv` | float `(ns,)` | Yes in `vmec_jax` output | Flux-surface current diagnostics | A/m^2-like VMEC current normalization | Poloidal/toroidal current-density profiles. |
| `equif`, `fsqt` | float `(ns,)`, float `(nstore_seq,)` | Optional diagnostics | VMEC force-balance traces | VMEC residual units | Flux-surface force balance and iteration trace. |
| `DMerc`, `DShear`, `DWell`, `DCurr`, `DGeod` | float `(ns,)` | Present in `vmec_jax` output | VMEC Mercier terms | dimensionless VMEC stability normalization | Mercier stability components. |
| `D_R`, `HGlasser`, `GlasserCorrection`, `GlasserShearValid` | float `(ns,)` | Present in `vmec_jax` output | vmec_jax Glasser/resistive-interchange diagnostics | dimensionless | Additional stability diagnostics. |
| `jdotb`, `bdotb`, `bdotgradv` | float `(ns,)` | Present in `vmec_jax` output | Flux-surface averaged field/current diagnostics | VMEC physical normalization | Diagnostics used by stability and plotting routines. |

`q_factor` is not a required `vmec_jax` `wout` field; downstream tools should compute it as `1 / iota` with appropriate zero-iota handling.

### Spectral Geometry And Field Coefficients

| Field | Type / Shape | Required | Normalization / Convention | Units | Description |
|-------|--------------|----------|----------------------------|-------|-------------|
| `rmnc`, `rmns` | float `(ns, mnmax)` | Yes | Main Fourier basis; cosine and sine channels for `R` | meters | Radial stack of `R` Fourier coefficients. `rmns` is zero for symmetric equilibria. |
| `zmnc`, `zmns` | float `(ns, mnmax)` | Yes | Main Fourier basis; cosine and sine channels for `Z` | meters | Radial stack of `Z` Fourier coefficients. `zmnc` is zero for symmetric equilibria. |
| `lmnc`, `lmns` | float `(ns, mnmax)` | Yes | VMEC stream-function `lambda` Fourier coefficients | dimensionless angle-like quantity | Radial stack of lambda coefficients. `lmnc` is zero for symmetric equilibria. |
| `gmnc`, `gmns` | float `(ns, mnmax_nyq)` | Yes | Nyquist Fourier basis | m^3 in VMEC coordinate normalization | Jacobian/sqrt(g) coefficients. |
| `bmnc`, `bmns` | float `(ns, mnmax_nyq)` | Yes in `vmec_jax` output | Nyquist Fourier basis | tesla | Magnetic-field magnitude coefficients. |
| `bsupumnc`, `bsupumns`, `bsupvmnc`, `bsupvmns` | float `(ns, mnmax_nyq)` | Yes | Nyquist Fourier basis | contravariant VMEC field normalization | Contravariant magnetic-field coefficients. |
| `bsubumnc`, `bsubumns`, `bsubvmnc`, `bsubvmns`, `bsubsmns`, `bsubsmnc` | float `(ns, mnmax_nyq)` | Yes in `vmec_jax` output | Nyquist Fourier basis | covariant VMEC field normalization | Covariant magnetic-field coefficients, including radial component. |
| `raxis_cc`, `zaxis_cs`, `raxis_cs`, `zaxis_cc` | float `(ntor+1,)` | Yes in `vmec_jax` output | Axis Fourier coefficients by toroidal mode | meters | Magnetic-axis representation. |

`currumnc` and `currvmnc` are not currently required fields in the `vmec_jax` `WoutData` schema. Use radial current diagnostics (`jcuru`, `jcurv`) and field/current derived quantities unless a downstream VMEC2000 compatibility path explicitly supplies current spectra.

#### Python API Objects (`vmec_jax` / `VMEC++`)

| Object | Description |
|--------|-------------|
| `InData` | Parsed `&INDATA` object with `scalars`, `indexed`, and `source_path`. |
| `VMECConfig` | Resolved discretization, Fourier-grid, symmetry, and free-boundary configuration. |
| `VMECState` | JAX PyTree of spectral state arrays `Rcos`, `Rsin`, `Zcos`, `Zsin`, `Lcos`, `Lsin`, each shaped `(ns, K)`. |
| `FixedBoundaryRun` | Driver result bundle containing `cfg`, `indata`, `static`, solved `state`, solver `result`, flux/profiles, and `signgs`. |
| `WoutData` / `wout` | Full VMEC-style output data structure matching the `wout_*.nc` contract above. |
| `jxbout`, `mercier`, Glasser diagnostics | Diagnostic groups represented by radial arrays/scalars in `WoutData` and helper APIs rather than a separate required file for the normal `vmec_jax` path. |

### Subset Handed to Next Stage

Stage 2 (`BOOZ_XFORM` / `booz_xform_jax`) needs the **full** equilibrium spectrum and profiles in `wout_*.nc`. `GX`, `Trinity3D`, and `NEOPAX` geometry readers also consume wout-level data for field-line geometry, rotational transform, and surface metrics.

### Outputs Used as Objectives

- Aspect ratio, volume, beta, target iota(s): direct design objectives
- Mercier criterion: stability objective
- Residuals `fsqr`, `fsqz`, `fsql`: QA convergence signals, not physics design objectives

### Output Validation

The test helper `test_io_validation.py` also contains `validate_wout_payload(payload)`, which checks that required metadata are present, main and Nyquist mode tables have the declared lengths, spectral arrays have shapes `(ns, mnmax)` or `(ns, mnmax_nyq)`, radial profiles have shape `(ns,)`, optional axis arrays have shape `(ntor+1,)`, and required scalar diagnostics exist.

---

## Governing Equations

The equilibrium satisfies ideal-MHD force balance:

$$\nabla p = \mathbf{J} \times \mathbf{B}, \quad \nabla \cdot \mathbf{B} = 0, \quad \mathbf{J} = \frac{1}{\mu_0} \nabla \times \mathbf{B}$$

`VMEC++` finds the stationary point of the energy functional (Hirshman & Whitson 1983):

$$W = \frac{1}{(2\pi)^2} \int \left( \frac{B^2}{2} + \frac{p}{\gamma - 1} \right) dV$$

In `VMEC++` flux coordinates with the stream function lambda:

$$u = \theta + \lambda(s, \theta, \zeta), \quad \frac{du}{d\zeta} = \iota(s)$$

The contravariant field components are:

$$B^\zeta = \frac{\Phi'(s) + \text{lamscale} \cdot \partial_\theta \lambda}{\text{signgs} \cdot \sqrt{g} \cdot 2\pi}$$

$$B^\theta = \frac{\chi'(s) - \text{lamscale} \cdot \partial_\zeta \lambda}{\text{signgs} \cdot \sqrt{g} \cdot 2\pi}$$

`DESC` solves the same physics in a pseudo-spectral formulation:

$$\mathbf{B} = \frac{\partial_\rho \psi}{2\pi\sqrt{g}} \left[ \left(\iota - \frac{\partial\lambda}{\partial\zeta}\right) \mathbf{e}_\theta + \left(1 + \frac{\partial\lambda}{\partial\theta}\right) \mathbf{e}_\zeta \right]$$

Reference: `stellarator_workflow.tex`, Sections 4.1-4.2.

---

## Convergence & Validity

> [!TODO]
> Document convergence behavior, known failure modes, and recommended tolerances.

---

## API Documentation

> [!TODO]
> Document key entry points, configuration parameters, and usage examples.

---

## Scripts & Workflows

**`vmec_jax` (via Pixi):** From the `stages`/ directory:

```
pixi run stage-1-vmec
```

**Input:** `inputs/quick_run/vmec_input.HSX_vacuum_ns201_quickrun`
**Output:** `outputs/quick_run/stage1_equilibrium/wout_HSX_vacuum_ns201_quickrun.nc`

See `docs/mvp-pipeline.md` for full I/O details.

> [!TODO]
> Add standalone run scripts and debugging workflows for `VMEC++` and `DESC`.

---

## W&B Tracking

**Project:** `driftless-star-stage1-equilibrium`

> [!TODO]
> Set up W&B tracking.

---

## Container Specification (Phase 2)

**`vmec_jax`:** Built from the single templated `stages/Dockerfile` using a build process morally equivalent to:

```
docker build --file stages/Dockerfile --build-arg ENVIRONMENT=stage-1-vmec --platform linux/amd64 --tag ghcr.io/driftless-star/driftless-star:stage-1-vmec-cpu stages/  # CPU
docker build --file stages/Dockerfile --build-arg CUDA_VERSION=12 --build-arg ENVIRONMENT=stage-1-vmec-gpu --platform linux/amd64 --tag ghcr.io/driftless-star/driftless-star:stage-1-vmec-gpu stages/  # GPU
```

Published to GHCR as `ghcr.io/driftless-star/driftless-star:stage-1-vmec-cpu` and `stage-1-vmec-gpu`. CI builds via `.github/workflows/containers.yml`.

See [guide](../guide.md#container-architecture) for full architecture details.

> [!TODO]
> Define container specifications for `VMEC++` and `DESC`.

---

## Tests (Phase 2)

> [!TODO]
> Write unit, regression, and integration tests. See [guide](../guide.md#writing-tests) for examples.

---

## Claude Skills

> [!TODO]
> Create development and operational Claude skills. See [guide](../guide.md#step-7-create-claude-skills) for skill types.
