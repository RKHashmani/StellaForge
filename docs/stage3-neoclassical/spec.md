# Stage 3: Neoclassical Transport

## Overview

Stage 3 computes neoclassical transport properties from the Boozer-coordinate equilibrium. It has two codes:

1. **`NEO` / `NEO_JAX`** -- Computes effective ripple (epsilon_eff), a screening/optimization diagnostic. **NOT a transport state variable** -- does not feed into profile evolution. Runs in parallel with the transport code.
2. **`SFINCS` / `sfincs_jax`** -- Solves the full drift-kinetic equation for neoclassical particle flux, heat flux, bootstrap current, and ambipolar E_r. Feeds `NEOPAX` (Stage 5).

**Position in pipeline:** `NEO_JAX` receives `boozmn_*.nc` from Stage 2 (Boozer). `sfincs_jax` receives `wout_*.nc` directly from Stage 1 (Equilibrium). Stage 3 runs in parallel with Stage 4 (Turbulence).

**Reference:** `stellarator_workflow.tex`, Sections 4.4-4.5; `stellarator_io_reference.tex`, Sections 3.4-3.5.

---

## Sub-Stage 3a: `NEO` / `NEO_JAX` (Effective Ripple)

### Codes

**NEO_JAX (Primary JAX):** <https://github.com/uwplasma/NEO_JAX>

**`NEO` (Legacy, part of `STELLOPT`):** <https://github.com/PrincetonUniversity/STELLOPT>

### Input Specification

Reference: `stellarator_io_reference.tex`, Section 3.4.

| Field | Type | Description | Source |
|-------|------|-------------|--------|
| `boozmn_*.nc` | NetCDF file | Boozer-coordinate equilibrium | Stage 2 |
| `neo_in.*` / `neo_param.*` | Control file | Surface list, angular resolution (theta_n, phi_n), Fourier cutoffs, MC controls, accuracy targets, current calculation switch (CALC_CUR) | User-specified |

#### Input Validation

> [!TODO]
> See [I/O Validation section](../guide.md#io-validation).

### Output Specification

Reference: `stellarator_io_reference.tex`, Section 3.4.

**Primary output:** `neo_out.*` and `neolog.*`

| Field | Type | Description | Used As |
|-------|------|-------------|---------|
| `epstot` | 1D array (per surface) | epsilon_eff^{3/2} (total effective ripple) | **Screening objective only** |
| `epspar` | 1D array | Parallel epsilon | Diagnostic |
| `reff` | 1D array | Effective radius | Diagnostic |
| `iota` | 1D array | Rotational transform | Cross-check |
| `b_ref` | scalar | Reference magnetic field | Normalization |
| `r_ref` | scalar | Reference radius | Normalization |
| `ctrone` | 1D array | Contribution from one class | Diagnostic |
| `ctrtot` | 1D array | Total contribution | Diagnostic |
| `bareph` | 1D array | Parallel epsilon (bar) | Diagnostic |
| `barept` | 1D array | Perpendicular epsilon (bar) | Diagnostic |
| `yps` | 1D array | Normalized toroidal flux | Coordinate |

**`NEO_JAX` result objects:** `epsilon_effective`, `epsilon_effective_by_class`

Optional outputs (if CALC_CUR=1): `neo_cur.*`, `current.dat`, `conver.dat`, `diagnostic.dat`, `diagnostic_add.dat`, `diagnostic_bigint.dat`

**Role:** Screening/optimization diagnostic. Usually NO direct transport consumer. epsilon_eff is NOT what `Trinity3D` or `NEOPAX` advances in time.

#### Output Validation

> [!TODO]
> See [I/O Validation section](../guide.md#io-validation).

### Governing Equations

Field-line integrals:

$$y_2 = \int d\phi\, B^{-2}, \quad y_3 = \int d\phi\, |\nabla\psi| B^{-2}, \quad y_4 = \int d\phi\, K_G B^{-3}$$

Trapped-particle integrals:

$$I_f = \int d\phi\, \sqrt{1 - \frac{B}{B_0 \eta}}\, B^{-2}$$

$$H_f = \int d\phi\, \sqrt{1 - \frac{B}{B_0 \eta}} \left(\frac{4}{B/B_0} - \frac{1}{\eta}\right) \frac{K_G}{\sqrt{\eta}} B^{-2}$$

Class-resolved effective ripple:

$$\epsilon_{\text{eff}}^{3/2}(m) = C_\epsilon \frac{y_2}{y_3^2} \text{BigInt}(m), \quad C_\epsilon = \frac{\pi R_0^2 \Delta\eta}{8\sqrt{2}}$$

Total `epstot` is the sum over classes.

**Reference:** `stellarator_workflow.tex`, Section 4.4.

---

## Sub-Stage 3b: `SFINCS` / `sfincs_jax` (Full Neoclassical)

### Codes

**sfincs_jax (Primary JAX):** <https://github.com/uwplasma/sfincs_jax>

**SFINCS (Legacy):** <https://github.com/landreman/sfincs>

### Input Specification

Reference: `stellarator_io_reference.tex`, Section 3.5.

| Field | Type | Required | Description | Source |
|-------|------|----------|-------------|--------|
| `input.namelist` | Fortran namelist text | **Yes** | Primary run configuration (all namelist groups). | User / workflow |
| `wout_*.nc` or `.bc` equilibrium | NetCDF / Boozer file | **Yes** (directly or via `equilibriumFile`) | Magnetic geometry input. In `sfincs_jax`, CLI `--wout-path` / `--equilibrium-file` overrides the namelist path and is written into embedded `input.namelist` in output. | Stage 1 or user |

**Required input fields** :

**Geometry & Equilibrium:**
- `geometryScheme`: geometry model/file mode. Common values: `1` (analytic tokamak-like model), `4` (analytic W7-X-like reduced model), `5` (VMEC from `wout_*.nc` or compatible VMEC equilibrium file), `11/12` (Boozer `.bc` file, with / without stellarator symmetry).
- `equilibriumFile` (or CLI override `--wout-path` / `--equilibrium-file`): path to the equilibrium file required by the selected geometry mode. For `geometryScheme=5`, this is a VMEC `wout` file. For `geometryScheme=11/12`, this is a Boozer `.bc` file.
- `inputRadialCoordinate`: selects which radial coordinate is used to identify the flux surface in the geometry/equilibrium input. This is used for file-backed geometry modes such as `geometryScheme=5` and `11/12`, and common choices are `0/1/2/3` for `psiHat` / `psiN` / `rHat` / `rN`.
- `inputRadialCoordinateForGradients`: selects which radial coordinate is used for profile and electrostatic-potential gradients. Common choices are `0/1/2/3` for gradients with respect to `psiHat` / `psiN` / `rHat` / `rN`.

**Species Definition:**
The definitions below are for default values of 'nu_n', 'Delta' and 'alpha'
- `Zs`: array species charges normalized to proton charge.
- `mHats`: array of mass ratios ($m_s / m_{\text{ref}}$, typically deuterium = 2).
- `nHats`: array of species density values in $10^20 m^{3}$.
- `THats`: array of species temperature values in $keV$.

**Density & Temperature Gradients:**
- `dnHatdpsiHat`, `dnHatdpsiN`, `dnHatdrHat`, or `dnHatdrN`: density gradient profile, with the active variable selected by `inputRadialCoordinateForGradients`.
- `dTHatdpsiHat`, `dTHatdpsiN`, `dTHatdrHat`, or `dTHatdrN`: temperature gradient profile, with the active variable selected by `inputRadialCoordinateForGradients`.

**Electric Field or Potential Gradient:**
- One of: `dPhiHatdpsiHat`, `dPhiHatdpsiN`, `dPhiHatdrHat`, `dPhiHatdrN`, or `Er` (radial electric field), with the active gradient coordinate selected by `inputRadialCoordinateForGradients`.

**Physics Controls:**
- `RHSMode`: right-hand-side / solve mode. `1` = standard drift-kinetic solve for fluxes and flows, `2` = solve multiple right-hand sides to assemble the transport matrix, `3` = monoenergetic transport-coefficient mode.
- `collisionOperator`: collision model. Common values are `0` (full linearized Fokker-Planck operator) and `1` (pitch-angle-scattering / Lorentz operator without the momentum-conserving field term).
- `constraintScheme`: constraint choice used to remove null-space / gauge freedom in the kinetic solve. Common values are `-1` (automatic, generally recommended), `0` (no constraints), `1` (enforce flux-surface-averaged density and pressure constraints), and `2` (enforce the $L=0$ constraint at each energy grid point).
- `Delta`: normalized gyroradius scale, roughly $\rho_*$ at the chosen reference parameters.
- `alpha`: electrostatic normalization factor, $e \bar{\Phi} / \bar{T}$, used to nondimensionalize `Er`, `Phi`, and `dPhiHatd*` quantities.
- `nu_n`: normalized collisionality used in the standard kinetic solve (`RHSMode=1`, and commonly also transport-matrix `RHSMode=2`). In `RHSMode=3`, the monoenergetic collisionality is instead specified through `nuPrime`.

**Resolution Controls:**
- `Ntheta`: poloidal grid points.
- `Nzeta`: toroidal grid points.
- `Nxi`: pitch-angle grid points.
- `Nx`: energy grid points.
- Solver tolerance: `solverTolerance` (commonly around $10^{-6}$ to $10^{-10}$ depending on the case and solver path). In this `sfincs_jax` checkout, this is the main named tolerance parameter exposed in the input namelist for the linear solve.

> [!NOTE]
> **Radial-scan bridge namelist overrides.** The `sfincs_jax_radial_scan.py` bridge that the Snakemake forward pass runs does not use the template namelist verbatim; it patches a fresh copy per flux surface. It sets `RHSMode = 1`, `inputRadialCoordinate = 3` (`rN`), and `rN_wish` to the surface's rho, and it **drops** `inputRadialCoordinateForGradients` from the template so `sfincs_jax` infers the gradient coordinates on its own (species gradients supplied as `dNHatdrNs`/`dTHatdrNs` select mode 3, and the potential gradient supplied as `Er` selects mode 4). Unless overridden on the command line, it also forces the reduced quickrun smoke-test resolution `Ntheta = 5, Nzeta = 11, Nxi = 12, NL = 3, Nx = 4, solverTolerance = 1e-6`, which overrides the raw `sfincs_jax` namelist defaults listed in the field table below.

**Optional input fields** :

**Phi1 / Electrostatic Effects:**
- `includePhi1`: `.true./.false.` — supported in this `sfincs_jax` checkout; enables the Phi1 / quasineutrality / lambda block.
- `includePhi1InKineticEquation`: supported, but only as the current parity-first / frozen-linearization implementation of Phi1 coupling in the kinetic equation.
- `includePhi1InCollisionOperator`: supported for the Fokker-Planck (`collisionOperator = 0`) path, and requires `includePhi1 = .true.` plus `includePhi1InKineticEquation = .true.`.
- `readExternalPhi1`: recognized in the input surface, but not currently supported end-to-end in this `sfincs_jax` checkout.




**Distribution Function Export:**
- `export_full_f`: `.true./.false.` — write full (Maxwellian + perturbation) distribution to output.
- `export_delta_f`: `.true./.false.` — write perturbed part of distribution separately.
- `export_f_theta_option`, `export_f_zeta_option`, `export_f_xi_option`, `export_f_x_option`: select how the export grid is chosen for each coordinate.
- `export_f_theta`, `export_f_zeta`, `export_f_xi`, `export_f_x`: explicit coordinate values used when the corresponding export option selects custom sampling.



**All Input Fields**:

| Field | Type | Default | Required | Condition | Meaning | Units | 
|---|---|---|---|---|-----------------|----|
| RHSMode | integer | 1 | No (defaulted) | Always | Solve mode: `1` = standard drift-kinetic solve, `2` = transport-matrix assembly from multiple right-hand sides, `3` = monoenergetic transport-coefficient mode |
| outputFileName | string | ``sfincsOutput.h5'' | No (defaulted) | Always | Name which will be used for the HDF5 output file |
| saveMatlabOutput | Boolean | .false. | No (defaulted) | Always | If this switch is set to true, Matlab m-files are created which store the system matrix, right-hand side, and solution vector |
| MatlabOutputFilename | string | ``sfincsMatrices'' | Conditional | Only when saveMatlabOutput == .true.. | Start of the filenames which will be used for Matlab output. |
| saveMatricesAndVectorsInBinary | Boolean | .false. | No (defaulted) | Always | If this switch is set to true, the matrix, right-hand-side, and solution vector of the linear system will be saved in PETSc's binary format |
| binaryOutputFilename | string | ``sfincsBinary'' | Conditional | Only when saveMatricesAndVectorsInBinary == .true.. | Start of the filenames which will be used for binary output of the system matrices, right-hand-side vectors, and solution vectors |
| solveSystem | Boolean | .true. | No (defaulted) | Always | If this parameter is false, the system of equations will not actually be solved |
| ambipolarSolve | Boolean | .false. | Conditional | Primarily when `RHSMode == 1` | Legacy / upstream namelist-driven ambipolar root solve for `Er`; in this checkout, the more practical public workflow is usually the CLI `scan-er` + `ambipolar-solve` path |
| NEr\_ambipolarSolve | integer | 20 | Conditional | When ambipolarSolve == .true.. | Maximum number of solves to allow while finding the ambipolar Er. |
| Er\_search\_tolerance\_dx | real | 1.d-8 | Conditional | When ambipolarSolve == .true. and ambipolarSolveOption/=2. | Tolerance used for ambipolar solve |
| Er\_search\_tolerance\_f | real | 1.d-10 | Conditional | When ambipolarSolve == .true.. | Tolerance used for ambipolar solve |
| ambipolarSolveOption | integer | 1 | Conditional | When ambipolarSolve == .true. | Indicates which root solving algorithm to use for ambipolar solve |
| Er\_min | real | -100 | Conditional | When ambipolarSolve == .true. and ambipolarSolveOption /= 3. | Minimum value of Er used to bracket the ambipolar root. |
| Er\_max | real | 100 | Conditional | When ambipolarSolve == .true. and ambipolarSolveOption /= 3. | Maximum value of Er used to bracket the ambipolar root. |
| geometryScheme | integer | 1 | No (defaulted) | Always | How the magnetic geometry is specified. In this `sfincs_jax` checkout, the implemented modes are `1`, `2`, `4`, `5`, `11`, and `12` |
| inputRadialCoordinate | integer | 3 | Conditional | When the selected geometry needs a flux-surface choice | Which radial coordinate is used to select the target flux surface (`psiHat`, `psiN`, `rHat`, or `rN`) |
| inputRadialCoordinateForGradients | integer | 4 | Conditional | Whenever profile / electric-field gradients are specified | Which radial coordinate is used for input gradients. `0/1/2/3/4` correspond to `psiHat`, `psiN`, `rHat`, `rN`, and `Er`-based input |
| B0OverBBar | real | 1.0 | Conditional | Only when geometryScheme == 1 | Magnitude of (0,0) Boozer harmonic of B field |
| GHat | real | 3.7481 | Conditional | Only when geometryScheme == 1 | Poloidal current outside flux surface |
| IHat | real | 0.0 | Conditional | Only when geometryScheme == 1 | Toroidal current inside flux surface |
| iota | real | 0.4542 | Conditional | Only when geometryScheme == 1 | Rotational transform |
| psiAHat | real | 0.15596 | Conditional | Only when geometryScheme == 1 | Normalized toroidal flux at LCFS |
| aHat | real | 0.5585 | Conditional | Only when geometryScheme == 1 | Effective minor radius at LCFS |
| equilibriumFile | string | ``'' | Conditional | Only when geometryScheme == 5, 11, or 12 | Filename for magnetic equilibrium (vmec wout or .bc) |
| VMECRadialOption | integer | 1 | Conditional | Only when geometryScheme == 5, 11 or 12 | Controls interpolation vs nearest surface lookup |
| rippleScale | real | 1.0 | Conditional | Only when geometryScheme == 5 | Scales VMEC geometry components |
| Zs | 1D array of reals | 1.0 | No (defaulted) | Always | Charges of each species (proton units) |
| mHats | 1D array of reals | 1.0 | No (defaulted) | Always | Masses of each species (reference mass units) |
| nHats | 1D array of reals | 1.0 | Conditional | Whenever RHSMode == 1 | Densities of each species |
| THats | 1D array of reals | 1.0 | Conditional | Whenever RHSMode == 1 | Temperatures of each species |
| dnHatdpsiHats | 1D array of reals | 0.0 | Conditional | Whenever RHSMode == 1 and inputRadialCoordinateForGradients == 0 | Radial density gradients w.r.t psiHat |
| dTHatdpsiHats | 1D array of reals | 0.0 | Conditional | Whenever RHSMode == 1 and inputRadialCoordinateForGradients == 0 | Radial temperature gradients w.r.t psiHat |
| dnHatdpsiNs | 1D array of reals | 0.0 | Conditional | Whenever RHSMode == 1 and inputRadialCoordinateForGradients == 1 | Radial density gradients w.r.t psiN |
| dTHatdpsiNs | 1D array of reals | 0.0 | Conditional | Whenever RHSMode == 1 and inputRadialCoordinateForGradients == 1 | Radial temperature gradients w.r.t psiN |
| dnHatdrHats | 1D array of reals | 0.0 | Conditional | Whenever RHSMode == 1 and inputRadialCoordinateForGradients == 2 | Radial density gradients w.r.t rHat |
| dTHatdrHats | 1D array of reals | 0.0 | Conditional | Whenever RHSMode == 1 and inputRadialCoordinateForGradients == 2 | Radial temperature gradients w.r.t rHat |
| dnHatdrNs | 1D array of reals | 0.0 | Conditional | Whenever RHSMode == 1 and inputRadialCoordinateForGradients == 3 | Radial density gradients w.r.t rN |
| dTHatdrNs | 1D array of reals | 0.0 | Conditional | Whenever RHSMode == 1 and inputRadialCoordinateForGradients == 3 | Radial temperature gradients w.r.t rN |
| withAdiabatic | Boolean | .false. | Conditional | When RHSMode == 1 and includePhi1 == .true. | Add adiabatic species to quasineutrality |
| adiabaticZ | real | -1.0 | Conditional | When includePhi1 == .true. and withAdiabatic == .true. | Charge of adiabatic species |
| adiabaticMHat | real | 5.44617e-4 | Conditional | When includePhi1 == .true. and withAdiabatic == .true. | Mass of adiabatic species |
| adiabaticNHat | real | 1.0 | Conditional | When includePhi1 == .true. and withAdiabatic == .true. | Density of adiabatic species |
| adiabaticTHat | real | 1.0 | Conditional | When includePhi1 == .true. and withAdiabatic == .true. | Temperature of adiabatic species |
| withNBIspec | Boolean | .false. | Conditional | When `RHSMode == 1`, `includePhi1 == .true.`, `readExternalPhi1 == .false.`, and typically `quasineutralityOption == 1` | Add an NBI species to quasineutrality only, not to the kinetic equation |
| NBIspecZ | real | 1.0 | Conditional | When withNBIspec== .true. | Charge of NBI species |
| NBIspecNHat | real | 0.0 | Conditional | When withNBIspec== .true. | Density of NBI species |
| Delta | real | 4.5694e-3 | Conditional | Standard kinetic / transport workflows | Normalized gyroradius scale, roughly the reference `rho_*` used in the SFINCS normalization |
| alpha | real | 1.0 | Conditional | Standard kinetic / transport workflows | Electrostatic normalization factor, approximately `e * PhiBar / TBar`; it sets the normalization used for `Phi`, `Er`, and `dPhiHatd*` quantities |
| nuPrime | real | 1.0 | Conditional | Only when RHSMode == 3 | Dimensionless collisionality for monoenergetic coeffs |
| EStar | real | 0.0 | Conditional | Only when RHSMode == 3 | Normalized radial E field for monoenergetic coeffs |
| EParallelHat | real | 0.0 | Conditional | Standard kinetic solves | Inductive / applied parallel electric field in SFINCS normalization |
| dPhiHatdpsiHat | real | 0.0 | Conditional | When inputRadialCoordinateForGradients == 0 | Electrostatic potential gradient w.r.t psiHat |
| dPhiHatdpsiN | real | 0.0 | Conditional | When inputRadialCoordinateForGradients == 1 | Electrostatic potential gradient w.r.t psiN |
| dPhiHatdrHat | real | 0.0 | Conditional | When inputRadialCoordinateForGradients == 2 | Electrostatic potential gradient w.r.t rHat |
| dPhiHatdrN | real | 0.0 | Conditional | When inputRadialCoordinateForGradients == 3 | Electrostatic potential gradient w.r.t rN |
| Er | real | 0.0 | Conditional | When inputRadialCoordinateForGradients == 4 | Radial electric field |
| collisionOperator | integer | 0 | No (defaulted) | Always | Choice of collision operator: `0` = full linearized Fokker-Planck, `1` = pitch-angle-scattering / Lorentz model |
| constraintScheme | integer | -1 | No (defaulted) | Always | Constraint control for null space and conservation |
| includeXDotTerm | Boolean | .true. | Conditional | When radial E field is nonzero | Include speed-change term from E_r |
| includeElectricFieldTermInXiDot | Boolean | .true. | Conditional | When radial E field is nonzero | Include pitch-angle-change term from E_r |
| useDKESExBDrift | Boolean | .false. | Conditional | When radial electric-field terms are active | Use the DKES-style `E x B` drift formula rather than the full-trajectory form |
| includePhi1 | Boolean | .false. | Conditional | Whenever RHSMode == 1 | Include first-order potential Phi1 |
| readExternalPhi1 | Boolean | .false. | Conditional | When includePhi1 == .true. | Recognized input switch for reading `Phi1Hat` from an external file, but not currently supported end-to-end in this `sfincs_jax` checkout |
| externalPhi1Filename | string | ``externalPhi1.h5'' | Conditional | When readExternalPhi1 == .true. | Filename for the external `Phi1Hat` input in that same recognized-but-not-fully-supported path |
| includePhi1InKineticEquation | Boolean | .true. | Conditional | When includePhi1 == .true. | Couple Phi1 into kinetic equation |
| includePhi1InCollisionOperator | Boolean | .false. | Conditional | When includePhi1 == .true. | Include Phi1 in collision operator |
| quasineutralityOption | integer | 1 | Conditional | When includePhi1 == .true. and readExternalPhi1 == .false. | Choice of quasineutrality equation (1 or 2) |
| includeTemperatureEquilibrationTerm | Boolean | .false. | Conditional | Whenever RHSMode == 1 | Include temperature equilibration term |
| magneticDriftScheme | integer | 0 | Conditional | Whenever RHSMode == 1 | Control poloidal/toroidal magnetic drifts |
| EParallelHatSpec | 1D array of reals | 0.0 | Conditional | When used in kinetic solves | Species-dependent parallel forcing / drive term; this is not one of the most commonly used public `sfincs_jax` input paths |
| Ntheta | integer | 15 | No (defaulted) | Always | Poloidal grid points |
| Nzeta | integer | 15 | No (defaulted) | Always | Toroidal grid points per period |
| Nxi | integer | 16 | No (defaulted) | Always | Pitch-angle grid (Legendre polynomials) |
| Nx | integer | 5 | No (defaulted) | Always | Energy grid points |
| solverTolerance | real | 1e-6 | Conditional | When useIterativeLinearSolver == .true. | Krylov solver convergence tolerance |
| NL | integer | 4 | Conditional | When collisionOperator == 0 | Legendre polynomials for Rosenbluth potentials |
| NxPotentialsPerVth | real | 40.0 | Conditional | When collisionOperator == 0 and xGridScheme ≠ 5 | Rosenbluth grid points (obsolete) |
| xMax | real | 5.0 | Conditional | When collisionOperator == 0 and xGridScheme ≠ 5 | Rosenbluth max speed (obsolete) |
| forceOddNthetaAndNzeta | Boolean | .true. | No (defaulted) | Always | Force odd Ntheta and Nzeta |
| thetaDerivativeScheme | integer | 2 | No (defaulted) | Always | Poloidal discretization (0=spectral, ...) |
| zetaDerivativeScheme | integer | 2 | No (defaulted) | Always | Toroidal discretization (0=spectral, ...) |
| ExBDerivativeSchemeTheta | integer | 0 | Conditional | When radial electric-field terms are active | `E x B` drift upwinding / derivative scheme in `theta` |
| ExBDerivativeSchemeZeta | integer | 0 | Conditional | When radial electric-field terms are active | `E x B` drift upwinding / derivative scheme in `zeta` |
| magneticDriftDerivativeScheme | integer | 3 | Conditional | When `magneticDriftScheme != 0` | Magnetic-drift derivative / upwinding scheme |
| xGridScheme | integer | 5 | Conditional | When RHSMode == 1 or 2 | Speed discretization scheme |
| xPotentialsGridScheme | integer | 2 | Conditional | When RHSMode == 1 or 2 and xGridScheme == 5 | Rosenbluth potential grid scheme |
| xDotDerivativeScheme | integer | 0 | Conditional | When includeXDotTerm == .true. | Collisionless differentiation matrix |
| useIterativeLinearSolver | Boolean | .true. | No (defaulted) | Always | Use iterative (vs direct) solver |
| whichParallelSolverToFactorPreconditioner | integer | 1 | No (defaulted) | Always | Solver for preconditioner factorization |
| PETSCPreallocationStrategy | integer | 1 | No (defaulted) | Always | Memory allocation strategy for matrix |
| reusePreconditioner | Boolean | .true. | Conditional | When includePhi1 == .true. | Reuse preconditioner across iterations |
| psiHat_wish | real | -1 | Conditional | When inputRadialCoordinate == 0 | Requested flux surface (psiHat) |
| psiN_wish | real | 0.25 | Conditional | When inputRadialCoordinate == 1 | Requested flux surface (psiN) |
| rHat_wish | real | -1 | Conditional | When inputRadialCoordinate == 2 | Requested flux surface (rHat) |
| rN_wish | real | 0.5 | Conditional | When inputRadialCoordinate == 3 | Requested flux surface (rN) |
| epsilon_t | real | -0.07053 | Conditional | Only when geometryScheme == 1 | Toroidal variation in B |
| epsilon_h | real | 0.05067 | Conditional | Only when geometryScheme == 1 | Helical variation in B |
| epsilon_antisymm | real | 0.0 | Conditional | Only when geometryScheme == 1 | Stellarator-antisymmetric variation in B |
| helicity_l | integer | 2 | Conditional | When geometryScheme == 1 or 5 with rippleScale ≠ 1 | Poloidal mode of helical variation |
| helicity_n | integer | 10 | Conditional | When geometryScheme == 1 or 5 with rippleScale ≠ 1 | Toroidal mode of helical variation |
| helicity_antisymm_l | integer | 1 | Conditional | Only when geometryScheme == 1 | Poloidal mode of antisymmetric variation |
| helicity_antisymm_n | integer | 0 | Conditional | Only when geometryScheme == 1 | Toroidal mode of antisymmetric variation |
| VMEC_Nyquist_option | integer | 1 | Conditional | Only when geometryScheme == 5 | VMEC mode number handling |
| min_Bmn_to_load | real | 0.0 | Conditional | When geometryScheme == 5, 11, or 12 | Filter cutoff for B field harmonics |
| nu_n | real | 8.330e-3 | Conditional | Standard kinetic / transport workflows (`RHSMode = 1`, and commonly `2`) | Normalized collisionality. In `RHSMode = 3`, `nu_n` is effectively overridden by `nuPrime` |
| xGrid_k | integer | 0 | Conditional | When `RHSMode == 1 or 2` and `xGridScheme` is one of `{1,2,5,6}` | Orthogonal-polynomial weight exponent for the speed grid |
| Nxi_for_x_option | integer | 1 | No (defaulted) | Always | How Nxi depends on speed |
| preconditioner_species | integer | 1 | Conditional | When `useIterativeLinearSolver == .true.` and `Nspecies >= 2` | Species coupling retained in the preconditioner |
| preconditioner_x | integer | 1 | Conditional | When `useIterativeLinearSolver == .true.` and `RHSMode` is one of `{1,2}` | Speed-grid coupling retained in the preconditioner |
| preconditioner_x_min_L | integer | 0 | Conditional | When `preconditioner_x != 0` | Legendre-mode threshold for preconditioner simplification |
| preconditioner_theta | integer | 0 | Conditional | When useIterativeLinearSolver == .true. | Theta coupling in preconditioner |
| preconditioner_theta_min_L | integer | 0 | Conditional | When `preconditioner_theta != 0` | Legendre-mode threshold for preconditioner simplification |
| preconditioner_zeta | integer | 0 | Conditional | When useIterativeLinearSolver == .true. | Zeta coupling in preconditioner |
| preconditioner_zeta_min_L | integer | 0 | Conditional | When `preconditioner_zeta != 0` | Legendre-mode threshold for preconditioner simplification |
| preconditioner_xi | integer | 1 | Conditional | When useIterativeLinearSolver == .true. | Pitch-angle coupling (0=full, 1=tridiag) |
| preconditioner_magnetic_drifts_max_L | integer | 2 | Conditional | When useIterativeLinearSolver == .true. | Legendre mode cutoff for magnetic drift terms |
| export_full_f | Boolean | .false. | No (defaulted) | Always | Export full distribution function |
| export_delta_f | Boolean | .false. | No (defaulted) | Always | Export perturbed distribution function |
| export_f_theta_option | integer | 2 | Conditional | When export_full_f or export_delta_f == .true. | Theta grid control for export |
| export_f_zeta_option | integer | 2 | Conditional | When export_full_f or export_delta_f == .true. | Zeta grid control for export |
| export_f_theta | 1D array of reals | 0.0 | Conditional | When `export_f_theta_option` selects explicit / custom sampling | Theta values for distribution export |
| export_f_zeta | 1D array of reals | 0.0 | Conditional | When `export_f_zeta_option` selects explicit / custom sampling | Zeta values for distribution export |
| export_f_xi_option | integer | 1 | Conditional | When export_full_f or export_delta_f == .true. | Xi discretization for export |
| export_f_xi | 1D array of reals | 0.0 | Conditional | When export_f_xi_option == 1 | Xi values for distribution export |
| export_f_x_option | integer | 0 | Conditional | When export_full_f or export_delta_f == .true. | Speed grid control for export |
| export_f_x | 1D array of reals | 1.0 | Conditional | When `export_f_x_option` selects explicit / custom sampling | Speed values for distribution export |

#### Input Validation

> [!TODO]
> See [I/O Validation section](../guide.md#io-validation).

### Output Specification


The Snakemake forward pass runs the `sfincs_jax` radial scan as a per-surface fan-out (`stage3_prepare` checkpoint, one `stage3_run_one` job per flux surface, then `stage3_collect`; see [Per-surface fan-out](../mvp-pipeline.md#per-surface-fan-out-stages-3-and-4)). It produces one aggregated handoff file plus a per-surface run tree.

**Forward-chain handoff:** `sfincs_jax_flux_profiles.h5` (HDF5) -- flux profiles versus radius, consumed by `NEOPAX` (Stage 5). The `collect` step assembles it from every per-surface `result.json`. Its schema is the exact subset checked by the in-repo contract validator `src/io_contracts.py` (`validate_sfincs_flux`):

| Dataset | Shape | Meaning |
|---------|-------|---------|
| `rho` | `(n_radii,)` | Normalized radial coordinate of each aggregated surface |
| `r` | `(n_radii,)` | Radial coordinate written from `sfincs_jax`'s `rHat` (alias of `rHat`) |
| `rHat` | `(n_radii,)` | `sfincs_jax` `rHat` radial coordinate |
| `Gamma` | `(n_species, n_radii)` | Neoclassical particle flux in NEOPAX units |
| `Q` | `(n_species, n_radii)` | Neoclassical heat flux in NEOPAX units |
| `Upar` | `(n_species, n_radii)` | Parallel-flow observable |
| `species_names` | `(n_species,)` | UTF-8 species labels (root dataset) |

Root attributes:
- `axis_zero_padded` (bool, **required** by the contract): the scan drops the magnetic-axis (`rho = 0`) surface, so when no aggregated surface sits at `rho = 0` the `collect` step prepends a zero-flux `rho = 0` column and sets this flag `true`.
- Provenance echoes read back from the scan `manifest.json` rather than measured live: `backend`, `max_parallel`, `worker_sharding`, `profiles_source`, `source_transport_solution`, `source_sfincs_template`, `time_index`, `time_value`, `include_phi1`. `profiles_source` is one of `analytical` (kinetic profiles built from the analytical parameters in the `common_input.toml` `[profiles]` block, on `analytical_n_radii` points), `transport_h5` (read from a NEOPAX `transport_solution.h5`), or `prescribed` (read from SI profile arrays written into that same `[profiles]` block by the closed loop's Stage 5 post-processing; see [Closing the Loop](../mvp-pipeline.md#closing-the-loop) for the array contract). The latter two take their radial grid from the profile data rather than from `analytical_n_radii`.
- Unit / convention notes: `Upar_note`, `normalization_note`, `radius_note` (how `Gamma`/`Q`/`Upar` are converted to NEOPAX units and which coordinate `r`/`rHat` carries).

**Per-surface run tree** (under `outputs/<run>/stage3_neoclassical/`): `manifest.json` at the stage directory records the surfaces and provenance the `collect` step reduces over; `runs/rho_*/` holds one directory per surface (basenames like `rho_012_r0p4898`), each with `input.namelist` (patched per surface), `payload.json` (worker inputs), `sfincsOutput.h5` (native per-surface solver output), and `result.json` (the extracted fluxes `collect` reads).

**Native SFINCS output (per surface):** `sfincsOutput.h5` (HDF5) -- the native solver file. The `sfincs_jax` radial scan writes one per flux surface (under `runs/rho_*/`) and aggregates them into the handoff above; the standalone `SFINCS` (Fortran) binary writes it directly. Its fields:

| Field | Availability | Meaning | Primary Use | Normalization | Units |
|-------|--------------|---------|-------------|---------------|----|
| `particleFlux_vm_rN` | `RHSMode=1` solve outputs | Neoclassical particle flux in vm normalization (`rN` coordinate) | **Transport input** | `vm` flux normalization, reported on `rN` radial coordinate |
| `heatFlux_vm_rN` | `RHSMode=1` solve outputs | Neoclassical heat flux in vm normalization (`rN` coordinate) | **Transport input** | `vm` flux normalization, reported on `rN` radial coordinate |
| `particleFlux_vd_rN`, `heatFlux_vd_rN` | when `includePhi1=.true.` diagnostics are written | Total (magnetic + `E×B`) flux variants with `Phi1` effects | Transport input (Phi1-on workflows) | `vd` (drift + `E×B`) flux normalization, reported on `rN` |
| `FSABjHat` | solved runs | Flux-surface-averaged parallel current (bootstrap diagnostic) | Equilibrium/diagnostic coupling | `Hat` quantity (SFINCS normalized current) and flux-surface-averaged (`FSAB`) |
| `FSABFlow` | solved runs | Flux-surface-averaged parallel flow by species | Diagnostic | SFINCS normalized flow; flux-surface-averaged (`FSA/FSAB`) |
| `Phi1Hat` | when `includePhi1=.true.` | First-order electrostatic potential on `(theta,zeta)` grid | Diagnostic / analysis | `Hat` potential normalization ($\Phi_1$ normalized by $T_{\mathrm{ref}}/e$) |
| `transportMatrix` | `RHSMode=2/3` with `--compute-transport-matrix` transport-matrix workflow | Transport matrix assembled across `whichRHS` solves | Analysis / reduced-model fitting | Mixed normalized transport coefficients in SFINCS internal normalization |

Normalization notes for output names:
- Suffix `_vm` = magnetic-drift transport normalization; `_vd` = drift + `E\times B` transport normalization.
- Radial suffixes: `_psiHat`, `_psiN`, `_rHat`, `_rN` indicate the radial coordinate used for the reported flux.
- `Hat` denotes SFINCS normalized quantities; `FSA`/`FSAB` denotes flux-surface average.

Common required metadata outputs:

**Grids & Geometry:**
- `theta`, `zeta`: poloidal and toroidal angles on the computational grid.
- `x`: normalized kinetic energy (velocity space grid).
- `BHat`: magnetic field strength (normalized).
- `DHat`: geometric factor (Jacobian-related).
- `VPrimeHat`: derivative of volume w.r.t. flux coordinate.
- `FSABHat2`: flux-surface-averaged $B^2$ quantity.

**Run Metadata:**
- `Nspecies`: number of species in calculation.
- `Ntheta`, `Nzeta`, `Nxi`, `Nx`: resolution settings echoed to output.
- `RHSMode`: solve mode used.
- `NIterations`: number of nonlinear iterations taken.
- `elapsed time (s)`: elapsed wall-clock time written in the transport-matrix workflow.

**Species & Profile Information:**
- `Zs`, `mHats`, `nHats`, `THats`: species charges, mass ratios, densities, temperatures (echoed from input).
- `dnHatdpsiHat`, `dnHatdpsiN`, `dnHatdrHat`, `dnHatdrN`: density-gradient values written in all supported radial-coordinate forms.
- `dTHatdpsiHat`, `dTHatdpsiN`, `dTHatdrHat`, `dTHatdrN`: temperature-gradient values written in all supported radial-coordinate forms.
- `psiHat`, `psiN`, `rHat`, `rN`: scalar flux-surface location values written in all supported radial-coordinate forms.
- `iota`: rotational transform at the chosen flux surface.

Common conditional physics outputs:

**Radial Flux Families** (coordinate variants: `_psiHat`, `_psiN`, `_rHat`, `_rN`):
- `particleFlux*`: neoclassical particle flux per species.
- `heatFlux*`: neoclassical heat flux per species.
- `momentumFlux*`: neoclassical parallel momentum flux.
- `FSABFlow` and `FSABFlow_vs_x`: flux-surface-averaged flows by species.

**Flux-Surface Diagnostics:**
- `FSABjHat`: flux-surface-averaged parallel current (bootstrap diagnostic).
- `FSABFlow`: flux-surface-averaged parallel flow.
- `jHat`, `flow`: local (non-flux-surface-averaged) parallel current and flow.
- `densityPerturbation`, `pressurePerturbation`: moment diagnostics derived from the perturbed distribution.
- `NTV`: neoclassical toroidal viscosity (when Phi1 included).

**Classical Transport** (when included):
- `classicalParticleFluxNoPhi1_*`, `classicalHeatFluxNoPhi1_*`: static classical-flux diagnostics without Phi1 contributions.
- `classicalParticleFlux_*`, `classicalHeatFlux_*`: per-iteration classical-flux diagnostics in the reported radial-coordinate variants.

**Distribution Function Exports** (only when `export_full_f` or `export_delta_f` enabled):
- `delta_f`: perturbation part of distribution function on grid (theta, zeta, xi, x).
- `full_f`: total distribution (Maxwellian + perturbation).
- `export_f_theta`, `export_f_zeta`, `export_f_xi`, `export_f_x`: parameter specification for export grid resolution.

**Solver Diagnostics** (sfincs_jax-specific):
- `linearSolver*`: residual norms, iteration counts, convergence flags, and related solve metadata written for `RHSMode=1` output solves.
- `transportMatrix`: full matrix when `RHSMode=2/3` is run with `--compute-transport-matrix`.
- `QN_*` (conditional, env var `SFINCS_JAX_WRITE_QN_DIAGNOSTICS`): debug terms from quasineutrality solve.

**Handoff to `Trinity3D`:** The `Trinity3D` adapter reads:
- When `includePhi1=.false.` (standard neoclassical): `particleFlux_vm_rN`, `heatFlux_vm_rN`.
- When `includePhi1=.true.` (with Phi1 effects): `particleFlux_vd_rN`, `heatFlux_vd_rN` (includes both magnetic drift and E×B contributions).

#### Output Validation

> [!TODO]
> See [I/O Validation section](../guide.md#io-validation).

### Governing Equations

First-order drift-kinetic equation:

$$(v_\parallel \mathbf{b} + \frac{d\Phi_0}{dr}\frac{\mathbf{B}\times\nabla r}{B^2})\cdot\nabla f_{s1} + [\text{mirror/drift terms}]\frac{\partial f_{s1}}{\partial\xi} - (\mathbf{v}_{ms}\cdot\nabla r)\frac{Z_s e}{2T_s x_s}\frac{d\Phi_0}{dr}\frac{\partial f_{s1}}{\partial x_s}$$

$$+ (\mathbf{v}_{ms}\cdot\nabla r)\left[\frac{1}{n_s}\frac{dn_s}{dr} + \frac{Z_s e}{T_s}\frac{d\Phi_0}{dr} + (x_s^2 - \frac{3}{2})\frac{1}{T_s}\frac{dT_s}{dr}\right]f_{sM} = C_s[f_{s1}] + S_s$$

Collision operator: $C_s[f_s] = \sum_b C_{sb}^l[f_s, f_b]$ with Lorentz, energy-diffusion, and field-particle components.

When Phi1 is included, coupled to quasineutrality.

**Reference:** `stellarator_workflow.tex`, Section 4.5.

---

## Installation & Platform

**`sfincs`:** Install via the Pixi environment. From the `stages`/ directory:

```
pixi install --environment stage-3-sfincs-fortran
```

**`sfincs_jax`:** Install via the Pixi environment. From the `stages`/ directory:

```
pixi install --environment stage-3-sfincs
```

**`neo-jax`:** Install via the Pixi environment. From the `stages`/ directory:

```
pixi install --environment stage-3-neo-jax
```

See `docs/mvp-pipeline.md` for run commands and I/O details.

> [!TODO]
> Document installation instructions and platform notes for `NEO_JAX` and `NEO`.

---

## Convergence & Validity

> [!TODO]
> Document convergence behavior, known failure modes, and recommended tolerances for all three sub-stages.

---

## API Documentation

> [!TODO]
> Document key entry points, configuration parameters, and usage examples for all three sub-stages.

---

## Scripts & Workflows

**`sfincs_jax` (via Pixi):** From the `stages`/ directory:

```
pixi run stage-3-sfincs
```

> [!NOTE]
> The pixi task and the Snakemake `stage3_prepare` checkpoint both pass the wout path to `sfincs_jax` via `--wout-path`. Populate `outputs/quick_run/stage1_equilibrium/` by running `pixi run stage-1-vmec` first. The namelist's `equilibriumFile` field is retained as a fallback for the `sfincs_fortran` backend and for direct `sfincs_jax` invocations that omit `--wout-path`.

**Input:** `outputs/quick_run/stage1_equilibrium/wout_HSX_vacuum_ns201_quickrun.nc` + `inputs/quick_run/sfincs_input.HSX_vacuum_ns201_quickrun`
**Output:** `outputs/quick_run/stage3_neoclassical/sfincs_jax_flux_profiles.h5`

See `docs/mvp-pipeline.md` for full I/O details.

**`SFINCS` (Fortran, via Pixi):** From the `stages`/ directory:

```
pixi run stage-3-sfincs-fortran
```

Alternative implementation to `sfincs_jax`. Consumes the same input namelist and writes the native `sfincsOutput.h5`, a different file from the `sfincs_jax_flux_profiles.h5` forward-chain handoff and not wired into the Snakemake forward pass; the task stages the namelist as `input.namelist` in the output directory before invocation because the Fortran binary reads that filename from its working directory.

**Input:** same as `sfincs_jax` above.
**Output:** `outputs/quick_run/stage3_neoclassical/sfincsOutput.h5` (the native SFINCS file, separate from the `sfincs_jax` handoff).

See `docs/mvp-pipeline.md` for full I/O details.

> [!TODO]
> Add standalone run scripts and workflows for `NEO_JAX`, `NEO`, and `SFINCS`.

---

## W&B Tracking

**Project:** `driftless-star-stage3-neoclassical`

> [!TODO]
> Set up W&B tracking for all three sub-stages.

---

## Container Specification (Phase 2)

**`sfincs_jax`:** Built from the single templated `stages/Dockerfile` using build arguments:

```
docker build --file stages/Dockerfile --build-arg ENVIRONMENT=stage-3-sfincs stages/        # CPU
docker build --file stages/Dockerfile --build-arg ENVIRONMENT=stage-3-sfincs-gpu --build-arg CUDA_VERSION=12 stages/  # GPU
```

Published to GHCR as `ghcr.io/driftless-star/driftless-star:stage-3-sfincs-cpu` and `stage-3-sfincs-gpu`. CI builds via `.github/workflows/containers.yml`.

See [guide](../guide.md#container-architecture) for full architecture details.

> [!TODO]
> Define container specifications for `NEO_JAX`, `NEO`, and `SFINCS`.

---

## Tests (Phase 2)

See [guide](../guide.md#writing-tests) for examples.

> [!TODO]
> Write unit, regression, and integration tests for all three sub-stages.

---

## Claude Skills

See [guide](../guide.md#step-7-create-claude-skills) for skill types.

> [!TODO]
> Create development and operational Claude skills for all three sub-stages.
