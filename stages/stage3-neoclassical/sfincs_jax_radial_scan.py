#!/usr/bin/env python3
"""Run a radial sfincs_jax flux scan from NEOPAX-style profile inputs.

This script is a first-step bridge between NEOPAX profile definitions/outputs and
sfincs_jax. It:

1. reads a NEOPAX ``transport_solution.h5`` file, analytical profile parameters
   from the NEOPAX TOML, or prescribed profile arrays from that TOML,
2. extracts one saved time slice, defaulting to the final one, when using
   ``transport_h5`` profiles,
3. builds one local sfincs_jax run per selected radial point,
4. launches those runs in parallel on CPUs or pinned GPUs,
5. collects particle flux, heat flux, and parallel-flow diagnostics,
6. writes an HDF5 profile file with datasets ``r``, ``Gamma``, ``Q``, and
   ``Upar`` that can be read by NEOPAX's ``FluxesRFileTransportModel``.
7. optionally writes PNG summary plots for ``Gamma``, ``Q``, and ``Upar``.

Default sfincs_jax resolution overrides used by this bridge (quickrun smoke test):
- ``Ntheta = 5``
- ``Nzeta = 11``
- ``Nxi = 12``
- ``NL = 3``
- ``Nx = 4``
- ``solverTolerance = 1e-6``

Important note on normalization:
The written ``Gamma`` and ``Q`` values are taken from sfincs_jax's
SFINCS-style output fields, preferring the ``*_rHat`` variants when available.
This makes the bridge practical for workflow prototyping, but exact
normalization against NEOPAX's native flux units should still be verified
carefully before treating the result as a strict physical replacement.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import h5py
import numpy as np
from scipy.constants import elementary_charge, proton_mass

NEOPAX_DENSITY_REFERENCE_M3 = 1.0e20
NEOPAX_TEMPERATURE_REFERENCE_EV = 1.0e3
SFINCS_REFERENCE_R_M = 1.0
SFINCS_REFERENCE_MASS_KG = proton_mass
SFINCS_REFERENCE_T_J = NEOPAX_TEMPERATURE_REFERENCE_EV * elementary_charge
SFINCS_REFERENCE_V_MS = float(np.sqrt(2.0 * SFINCS_REFERENCE_T_J / SFINCS_REFERENCE_MASS_KG))
SFINCS_GAMMA_TO_NEOPAX = NEOPAX_DENSITY_REFERENCE_M3 * SFINCS_REFERENCE_V_MS / SFINCS_REFERENCE_R_M
SFINCS_Q_TO_NEOPAX = (
    NEOPAX_DENSITY_REFERENCE_M3
    * SFINCS_REFERENCE_MASS_KG
    * SFINCS_REFERENCE_V_MS**3
    / SFINCS_REFERENCE_R_M
    / elementary_charge
)

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[import-not-found,no-redef]


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
STAGES_DIR = SCRIPT_DIR.parent

DEFAULT_COMMON_CONFIG = STAGES_DIR.parent / "inputs" / "quick_run" / "common_input.toml"
DEFAULT_SFINCS_TEMPLATE = STAGES_DIR.parent / "inputs" / "quick_run" / "sfincs_input.HSX_vacuum_ns201_quickrun"
DEFAULT_OUTPUT_DIR = STAGES_DIR.parent / "outputs" / "quick_run" / "stage3_neoclassical"


@dataclass(frozen=True)
class SpeciesMeta:
    name: str
    charge: float
    mass_mp: float


@dataclass(frozen=True)
class TransportSnapshot:
    rho: np.ndarray
    density: np.ndarray
    temperature: np.ndarray
    er: np.ndarray
    time_value: float | None


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        return tomllib.load(fh)


def _infer_neopax_root(config_path: Path) -> Path:
    parent = config_path.resolve().parent
    if parent.parent.name == "examples":
        return parent.parent.parent
    return parent


def _resolve_relative(base: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    expanded = os.path.expandvars(os.path.expanduser(value))
    path = Path(expanded)
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def _default_transport_solution_path(config_path: Path, cfg: dict[str, Any]) -> Path:
    output_cfg = cfg.get("transport_output", {})
    output_dir = _resolve_relative(_infer_neopax_root(config_path), output_cfg.get("transport_output_dir"))
    if output_dir is None:
        raise ValueError("Could not resolve [transport_output].transport_output_dir from the NEOPAX config.")
    return output_dir / "transport_solution.h5"


def _parse_species_from_config(cfg: dict[str, Any]) -> list[SpeciesMeta]:
    species_cfg = cfg.get("species", {})
    names = list(species_cfg.get("names", []))
    charges = list(species_cfg.get("charge_qp", []))
    masses = list(species_cfg.get("mass_mp", []))
    if not names or len(names) != len(charges) or len(names) != len(masses):
        raise ValueError("NEOPAX [species] must define matching names, charge_qp, and mass_mp arrays.")
    return [
        SpeciesMeta(name=str(name), charge=float(charge), mass_mp=float(mass))
        for name, charge, mass in zip(names, charges, masses)
    ]


def _load_transport_snapshot(h5_path: Path, *, time_index: int) -> TransportSnapshot:
    with h5py.File(h5_path, "r") as f:
        keys = set(f.keys())
        if "rho" not in keys or "density" not in keys or "Er" not in keys:
            raise KeyError(f"{h5_path} is missing required NEOPAX transport datasets.")

        rho = np.asarray(f["rho"][()], dtype=np.float64)
        density_all = np.asarray(f["density"][()], dtype=np.float64)
        er_all = np.asarray(f["Er"][()], dtype=np.float64)

        if "temperature" in keys:
            temperature_all = np.asarray(f["temperature"][()], dtype=np.float64)
        elif "pressure" in keys:
            pressure_all = np.asarray(f["pressure"][()], dtype=np.float64)
            density_safe = np.maximum(density_all, 1.0e-30)
            temperature_all = pressure_all / density_safe
        else:
            raise KeyError(f"{h5_path} must contain either temperature or pressure.")

        ts = np.asarray(f["ts"][()], dtype=np.float64) if "ts" in keys else None

    if density_all.ndim == 2:
        idx = 0
        density = density_all
        temperature = temperature_all
        er = er_all
    else:
        idx = int(time_index)
        if idx < 0:
            idx = density_all.shape[0] + idx
        if idx < 0 or idx >= density_all.shape[0]:
            raise IndexError(f"time index {time_index} out of range for {h5_path}")
        density = np.asarray(density_all[idx], dtype=np.float64)
        temperature = np.asarray(temperature_all[idx], dtype=np.float64)
        er = np.asarray(er_all[idx], dtype=np.float64)

    time_value = None if ts is None else float(ts[idx])
    return TransportSnapshot(rho=rho, density=density, temperature=temperature, er=er, time_value=time_value)


def _match_species_factors(raw: Any, n_species: int, *, default: float) -> np.ndarray:
    if raw is None:
        return np.full((n_species,), float(default), dtype=np.float64)
    if isinstance(raw, (int, float)):
        return np.full((n_species,), float(raw), dtype=np.float64)
    arr = np.asarray(list(raw), dtype=np.float64)
    if arr.size == 1:
        return np.full((n_species,), float(arr[0]), dtype=np.float64)
    if arr.size != n_species:
        raise ValueError(f"Expected {n_species} profile factors, got {arr.size}")
    return arr


def _build_standard_analytical_snapshot(
    cfg: dict[str, Any],
    *,
    n_species: int,
    n_radial: int,
) -> TransportSnapshot:
    profile_cfg = cfg.get("profiles", {})
    rho_edge = float(cfg.get("geometry", {}).get("rho_edge", 1.0))
    rho = np.linspace(0.0, rho_edge, int(n_radial), dtype=np.float64)
    x = rho / max(float(rho[-1]) if rho.size else 1.0, 1.0e-30)

    n0 = _match_species_factors(profile_cfg.get("n0", profile_cfg.get("ni0", profile_cfg.get("ne0", 4.21))), n_species, default=4.21)
    n_edge = _match_species_factors(profile_cfg.get("n_edge", profile_cfg.get("nib", profile_cfg.get("neb", 0.6))), n_species, default=0.6)
    t0 = _match_species_factors(profile_cfg.get("T0", profile_cfg.get("ti0", profile_cfg.get("te0", 17.8))), n_species, default=17.8)
    t_edge = _match_species_factors(profile_cfg.get("T_edge", profile_cfg.get("tib", profile_cfg.get("teb", 0.7))), n_species, default=0.7)
    density_shape_power = _match_species_factors(profile_cfg.get("density_shape_power", 2.0), n_species, default=2.0)
    density_shape_alpha = _match_species_factors(profile_cfg.get("density_shape_alpha", 1.0), n_species, default=1.0)
    temperature_shape_power = _match_species_factors(profile_cfg.get("temperature_shape_power", 2.0), n_species, default=2.0)
    temperature_shape_alpha = _match_species_factors(profile_cfg.get("temperature_shape_alpha", 1.0), n_species, default=1.0)

    c_density = profile_cfg.get("c_density")
    if c_density is None and n_species == 3:
        deuterium_ratio = float(profile_cfg.get("deuterium_ratio", 0.5))
        tritium_ratio = float(profile_cfg.get("tritium_ratio", 0.5))
        c_density = [1.0, deuterium_ratio, tritium_ratio]

    density_species_scale = _match_species_factors(c_density, n_species, default=1.0)
    temperature_species_scale = _match_species_factors(profile_cfg.get("c_temperature"), n_species, default=1.0)
    density_global_scale = _match_species_factors(profile_cfg.get("n_scale", 1.0), n_species, default=1.0)
    temperature_global_scale = _match_species_factors(profile_cfg.get("T_scale", 1.0), n_species, default=1.0)

    density = np.zeros((n_species, rho.size), dtype=np.float64)
    temperature = np.zeros((n_species, rho.size), dtype=np.float64)
    for i in range(n_species):
        density_shape = (1.0 - x**density_shape_power[i]) ** density_shape_alpha[i]
        temperature_shape = (1.0 - x**temperature_shape_power[i]) ** temperature_shape_alpha[i]
        density_base = n_edge[i] + (n0[i] - n_edge[i]) * density_shape
        temperature_base = t_edge[i] + (t0[i] - t_edge[i]) * temperature_shape
        density[i, :] = density_species_scale[i] * density_global_scale[i] * density_base
        temperature[i, :] = temperature_species_scale[i] * temperature_global_scale[i] * temperature_base

    er0_scale = float(profile_cfg.get("er0_scale", 100.0))
    er0_peak_rho = float(profile_cfg.get("er0_peak_rho", 0.8))
    er = er0_scale * x * (er0_peak_rho - x)

    return TransportSnapshot(
        rho=np.asarray(rho, dtype=np.float64),
        density=np.asarray(density, dtype=np.float64),
        temperature=np.asarray(temperature, dtype=np.float64),
        er=np.asarray(er, dtype=np.float64),
        time_value=None,
    )


def _require_profile_array(profile_cfg: dict[str, Any], key: str, *, ndim: int) -> np.ndarray:
    """Read a required ``[profiles]`` array as float64 and check its dimensionality."""
    if key not in profile_cfg:
        raise ValueError(f"[profiles].{key} is required when --profiles-source=prescribed")
    array = np.asarray(profile_cfg[key], dtype=np.float64)
    if array.ndim != ndim:
        raise ValueError(f"[profiles].{key} must be {ndim}-D, got shape {array.shape}")
    return array


def _build_prescribed_snapshot(cfg: dict[str, Any], *, n_species: int) -> TransportSnapshot:
    """Build a snapshot from the prescribed profile arrays stored in the config ``[profiles]`` section.

    The prescribed arrays are SI (density m^-3, temperature eV), so both are divided by the
    NEOPAX reference values to reach the snapshot units of 1e20 m^-3 and keV. Er is kV/m in
    both and passes through unchanged. The arrays carry no radial coordinate, so the grid is
    rebuilt from the same ``[geometry]`` block the transport solver builds its own grid from.

    Parameters
    ----------
    cfg : dict[str, Any]
        Parsed common-input TOML.
    n_species : int
        Number of species the run expects, taken from ``[species].names``.

    Returns
    -------
    TransportSnapshot
        Profiles on the reconstructed radial grid, in snapshot units.
    """
    profile_cfg = cfg.get("profiles", {})
    model = str(profile_cfg.get("model", "")).strip().lower()
    if model != "prescribed":
        raise ValueError(f"[profiles].model is {model!r}; --profiles-source=prescribed requires 'prescribed'")
    density = _require_profile_array(profile_cfg, "density", ndim=2)
    temperature = _require_profile_array(profile_cfg, "temperature", ndim=2)
    er = _require_profile_array(profile_cfg, "Er", ndim=1)
    if temperature.shape != density.shape:
        raise ValueError(
            f"[profiles].temperature has shape {temperature.shape}, expected the [profiles].density "
            f"shape {density.shape}"
        )
    if density.shape[0] != n_species:
        raise ValueError(
            f"[profiles].density holds {density.shape[0]} species rows, expected {n_species} to match "
            "the number of entries in [species].names"
        )
    n_radial = int(density.shape[1])
    if er.size != n_radial:
        raise ValueError(f"[profiles].Er holds {er.size} points, expected {n_radial} to match [profiles].density")
    geometry_cfg = cfg.get("geometry", {})
    if "n_radial" not in geometry_cfg:
        raise ValueError("[geometry].n_radial is required when --profiles-source=prescribed")
    if n_radial != int(geometry_cfg["n_radial"]):
        raise ValueError(
            f"[profiles] arrays hold {n_radial} radial points, expected {int(geometry_cfg['n_radial'])} "
            "to match [geometry].n_radial"
        )
    if n_radial < 3:
        raise ValueError(f"[profiles] arrays hold {n_radial} radial points, expected at least 3 for radial gradients")
    rho_edge = float(geometry_cfg.get("rho_edge", 1.0))
    return TransportSnapshot(
        rho=np.linspace(0.0, rho_edge, n_radial, dtype=np.float64),
        density=density / NEOPAX_DENSITY_REFERENCE_M3,
        temperature=temperature / NEOPAX_TEMPERATURE_REFERENCE_EV,
        er=er,
        time_value=None,
    )


def _parse_index_list(text: str | None) -> list[int] | None:
    if text is None or not text.strip():
        return None
    out: list[int] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if chunk:
            out.append(int(chunk))
    return out


def _choose_radius_indices(
    rho: np.ndarray,
    *,
    explicit: list[int] | None,
    rho_min: float | None,
    rho_max: float | None,
    num_radii: int | None,
) -> list[int]:
    if explicit is not None:
        idxs = sorted(set(int(v) for v in explicit))
        for idx in idxs:
            if idx < 0 or idx >= rho.size:
                raise IndexError(f"rho index {idx} out of range [0, {rho.size - 1}]")
        return idxs

    # By default, skip the magnetic axis point. A local sfincs_jax solve at
    # rho=0 is usually not the most useful first-pass transport postprocessing
    # target, and users can still include it explicitly via --rho-indices.
    mask = np.ones_like(rho, dtype=bool)
    mask &= ~np.isclose(rho, 0.0)
    if rho_min is not None:
        mask &= rho >= float(rho_min)
    if rho_max is not None:
        mask &= rho <= float(rho_max)
    candidates = np.where(mask)[0]
    if candidates.size == 0:
        raise ValueError("No radii satisfy the requested rho filter.")
    if num_radii is None or int(num_radii) <= 0 or int(num_radii) >= candidates.size:
        return [int(v) for v in candidates]
    picks = np.linspace(0, candidates.size - 1, int(num_radii))
    return sorted(set(int(candidates[int(round(p))]) for p in picks))


def _gradient_profile(values: np.ndarray, rho: np.ndarray) -> np.ndarray:
    return np.gradient(np.asarray(values, dtype=np.float64), np.asarray(rho, dtype=np.float64), edge_order=2)


def _bool_literal(value: bool) -> str:
    return ".true." if bool(value) else ".false."


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return _bool_literal(value)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.16g}"


def _format_value(value: Any) -> str:
    if isinstance(value, (list, tuple, np.ndarray)):
        return " ".join(_format_scalar(v) for v in np.asarray(value).ravel())
    return _format_scalar(value)


def _patch_group_value(*, text: str, group: str, key: str, value: Any) -> str:
    import re

    start = re.search(rf"(?im)^\s*&{re.escape(group)}\s*$", text)
    if start is None:
        raise ValueError(f"Missing namelist group &{group}")
    end = re.search(r"(?m)^\s*/\s*$", text[start.end() :])
    if end is None:
        raise ValueError(f"Missing '/' terminator for &{group}")
    end_pos = start.end() + end.start()
    group_txt = text[start.end() : end_pos]

    pat = re.compile(rf"(?im)^[ \t]*{re.escape(key)}[ \t]*=[ \t]*([^!\n\r]+)[ \t]*$")
    line = f"  {key} = {_format_value(value)}"
    match = pat.search(group_txt)
    if match is not None:
        group_txt2 = group_txt.replace(match.group(0), line)
    else:
        if not group_txt.endswith("\n"):
            group_txt = group_txt + "\n"
        group_txt2 = group_txt + line + "\n"
    return text[: start.end()] + group_txt2 + text[end_pos:]


def _drop_group_key(*, text: str, group: str, key: str) -> str:
    import re

    start = re.search(rf"(?im)^\s*&{re.escape(group)}\s*$", text)
    if start is None:
        raise ValueError(f"Missing namelist group &{group}")
    end = re.search(r"(?m)^\s*/\s*$", text[start.end() :])
    if end is None:
        raise ValueError(f"Missing '/' terminator for &{group}")
    end_pos = start.end() + end.start()
    group_txt = text[start.end() : end_pos]

    pat = re.compile(rf"(?im)^[ \t]*{re.escape(key)}[ \t]*=[^!\n\r]*(?:![^\n\r]*)?[\r]?\n?")
    group_txt2 = pat.sub("", group_txt)
    return text[: start.end()] + group_txt2 + text[end_pos:]


def _prepare_input_text(
    *,
    template_text: str,
    species: list[SpeciesMeta],
    snapshot: TransportSnapshot,
    radius_index: int,
    include_phi1: bool | None,
    resolution_overrides: dict[str, int | None],
    solver_tolerance: float | None,
) -> str:
    rho = snapshot.rho
    density = snapshot.density
    temperature = snapshot.temperature
    er = snapshot.er

    dndr = np.vstack([_gradient_profile(density[i], rho) for i in range(density.shape[0])])
    dtdr = np.vstack([_gradient_profile(temperature[i], rho) for i in range(temperature.shape[0])])

    text = template_text
    text = _patch_group_value(text=text, group="general", key="RHSMode", value=1)
    text = _patch_group_value(text=text, group="geometryParameters", key="inputRadialCoordinate", value=3)
    # Let sfincs_jax infer gradient coordinates separately:
    # species from dNHatdrNs/dTHatdrNs -> mode 3, Phi from Er -> mode 4.
    text = _drop_group_key(text=text, group="geometryParameters", key="inputRadialCoordinateForGradients")
    text = _patch_group_value(text=text, group="geometryParameters", key="rN_wish", value=float(rho[radius_index]))

    text = _patch_group_value(
        text=text,
        group="speciesParameters",
        key="Zs",
        value=[sp.charge for sp in species],
    )
    text = _patch_group_value(
        text=text,
        group="speciesParameters",
        key="mHats",
        value=[sp.mass_mp for sp in species],
    )
    text = _patch_group_value(
        text=text,
        group="speciesParameters",
        key="nHats",
        value=density[:, radius_index],
    )
    text = _patch_group_value(
        text=text,
        group="speciesParameters",
        key="THats",
        value=temperature[:, radius_index],
    )
    text = _patch_group_value(
        text=text,
        group="speciesParameters",
        key="dNHatdrNs",
        value=dndr[:, radius_index],
    )
    text = _patch_group_value(
        text=text,
        group="speciesParameters",
        key="dTHatdrNs",
        value=dtdr[:, radius_index],
    )
    text = _patch_group_value(
        text=text,
        group="physicsParameters",
        key="Er",
        value=float(er[radius_index]),
    )

    if include_phi1 is not None:
        text = _patch_group_value(
            text=text,
            group="physicsParameters",
            key="includePhi1",
            value=bool(include_phi1),
        )

    for key, value in resolution_overrides.items():
        if value is not None:
            text = _patch_group_value(text=text, group="resolutionParameters", key=key, value=int(value))

    if solver_tolerance is not None:
        text = _patch_group_value(
            text=text,
            group="resolutionParameters",
            key="solverTolerance",
            value=float(solver_tolerance),
        )

    return text


def _infer_wout_path(config_path: Path, cfg: dict[str, Any], explicit: str | None) -> Path | None:
    if explicit:
        # CLI override: standard path semantics (absolute or relative to CWD),
        # not relative to the NEOPAX root.
        expanded = os.path.expandvars(os.path.expanduser(explicit))
        return Path(expanded).resolve()
    geometry_cfg = cfg.get("geometry", {})
    return _resolve_relative(_infer_neopax_root(config_path), geometry_cfg.get("vmec_file"))


def _last_species_vector(arr: np.ndarray, n_species: int) -> np.ndarray:
    data = np.asarray(arr, dtype=np.float64)
    if data.ndim == 1:
        if data.shape[0] != n_species:
            raise ValueError(f"Expected {n_species} species values, got shape {data.shape}")
        return data
    if data.ndim == 2:
        if data.shape[0] == n_species:
            return data[:, -1]
        if data.shape[1] == n_species:
            return data[-1, :]
    raise ValueError(f"Unsupported diagnostic shape {data.shape} for n_species={n_species}")


def _last_scalar(arr: np.ndarray) -> float:
    data = np.asarray(arr, dtype=np.float64)
    if data.ndim == 0:
        return float(data)
    return float(np.ravel(data)[-1])


def _extract_flux_triplet(
    results: dict[str, Any],
    n_species: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float | None, dict[str, str]]:
    gamma_key = None
    q_key = None
    for key in ("particleFlux_vm_rHat", "particleFlux_vm_rN", "particleFlux_vm_psiHat"):
        if key in results:
            gamma_key = key
            break
    for key in ("heatFlux_vm_rHat", "heatFlux_vm_rN", "heatFlux_vm_psiHat"):
        if key in results:
            q_key = key
            break
    if gamma_key is None or q_key is None or "FSABFlow" not in results:
        raise KeyError("sfincs_jax output is missing required flux/flow diagnostics.")

    gamma_hat = _last_species_vector(np.asarray(results[gamma_key]), n_species)
    q_hat = _last_species_vector(np.asarray(results[q_key]), n_species)
    gamma = SFINCS_GAMMA_TO_NEOPAX * gamma_hat
    q = SFINCS_Q_TO_NEOPAX * q_hat

    fsab_flow = _last_species_vector(np.asarray(results["FSABFlow"]), n_species)
    b0_over_bbar = _last_scalar(np.asarray(results.get("B0OverBBar", 1.0), dtype=np.float64))
    # NTX fixed-field parallel-flow audit bridge:
    # NEOPAX's physical Upar closure matches the SFINCS hat-normalized FSABFlow
    # after restoring the historical factor 2 * B0OverBBar / sqrt(pi).
    flow_bridge = 2.0 * float(b0_over_bbar) / float(np.sqrt(np.pi))
    upar = flow_bridge * fsab_flow
    meta = {
        "Gamma_key": gamma_key,
        "Q_key": q_key,
        "Upar_key": "FSABFlow",
        "Gamma_scale_to_neopax": f"{SFINCS_GAMMA_TO_NEOPAX:.16g}",
        "Q_scale_to_neopax": f"{SFINCS_Q_TO_NEOPAX:.16g}",
        "Upar_bridge": "2*B0OverBBar/sqrt(pi)",
        "B0OverBBar": f"{b0_over_bbar:.16g}",
    }
    r_hat = None
    if "rHat" in results:
        r_hat = _last_scalar(np.asarray(results["rHat"], dtype=np.float64))
        meta["rHat"] = f"{r_hat:.16g}"
    return gamma, q, upar, r_hat, meta


def _build_worker_env(args: argparse.Namespace, *, gpu_id: str | None) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if args.dense_fp_max is not None:
        env["SFINCS_JAX_RHSMODE1_DENSE_FP_MAX"] = str(int(args.dense_fp_max))
    if str(args.backend).lower() == "gpu":
        # On NVIDIA-backed JAX installs, using the generic "gpu" alias can
        # still let JAX probe other accelerator backends such as ROCm. Force
        # CUDA explicitly for these worker processes.
        env["JAX_PLATFORMS"] = "cuda"
        env["JAX_PLATFORM_NAME"] = "cuda"
        if gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        # Drop accelerator-selection variables inherited from the parent shell
        # that can redirect JAX toward a different backend family.
        env.pop("PJRT_DEVICE", None)
        env.pop("JAX_BACKEND_TARGET", None)
        env.pop("ROCM_VISIBLE_DEVICES", None)
        env.pop("HIP_VISIBLE_DEVICES", None)
        env["SFINCS_JAX_SHARD"] = "0"
        env["SFINCS_JAX_AUTO_SHARD"] = "0"
        env["SFINCS_JAX_MATVEC_SHARD_AXIS"] = "off"
    else:
        env["JAX_PLATFORMS"] = "cpu"
        env["JAX_PLATFORM_NAME"] = "cpu"
        env["CUDA_VISIBLE_DEVICES"] = ""
        # Some recent JAX/XLA builds reject the legacy CPU-thread flags that older
        # shells or sfincs_jax opt-in settings may add. Keep other XLA flags, but
        # scrub the unsupported CPU-thread knobs for per-worker CPU launches.
        xla_flags = env.get("XLA_FLAGS", "")
        filtered_xla_flags = " ".join(
            token
            for token in xla_flags.split()
            if not token.startswith("--xla_cpu_parallelism_threads=")
            and not token.startswith("--xla_cpu_multi_thread_eigen_num_threads=")
            and not token.startswith("--xla_cpu_multi_thread_eigen=")
        ).strip()
        if filtered_xla_flags:
            env["XLA_FLAGS"] = filtered_xla_flags
        else:
            env.pop("XLA_FLAGS", None)
        cores = int(args.cores_per_run)
        threads = max(1, cores)
        if cores > 0:
            env["SFINCS_JAX_CORES"] = str(cores)
        # Pin native thread pools so N parallel workers do not each try to use
        # the whole machine. This matters a lot for medium/heavy CPU scans.
        env["OMP_NUM_THREADS"] = str(threads)
        env["OPENBLAS_NUM_THREADS"] = str(threads)
        env["MKL_NUM_THREADS"] = str(threads)
        env["VECLIB_MAXIMUM_THREADS"] = str(threads)
        env["NUMEXPR_NUM_THREADS"] = str(threads)
        env.pop("SFINCS_JAX_XLA_THREADS", None)
        worker_sharding = str(getattr(args, "worker_sharding", "off")).strip().lower()
        if worker_sharding != "off" and cores > 1:
            env["SFINCS_JAX_CPU_DEVICES"] = str(cores)
            env["SFINCS_JAX_SHARD"] = "1"
            if worker_sharding == "auto":
                env["SFINCS_JAX_AUTO_SHARD"] = "1"
                env["SFINCS_JAX_MATVEC_SHARD_AXIS"] = "auto"
            else:
                env["SFINCS_JAX_AUTO_SHARD"] = "0"
                env["SFINCS_JAX_MATVEC_SHARD_AXIS"] = worker_sharding
        else:
            # Default to process-level parallelism across radii rather than
            # per-worker multi-device sharding.
            env.pop("SFINCS_JAX_CPU_DEVICES", None)
            env["SFINCS_JAX_SHARD"] = "0"
            env["SFINCS_JAX_AUTO_SHARD"] = "0"
            env["SFINCS_JAX_MATVEC_SHARD_AXIS"] = "off"
    return env


def _run_single_worker_from_payload(payload_path: Path) -> int:
    with payload_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    input_path = Path(payload["input_path"])
    output_path = Path(payload["output_path"])
    result_json = Path(payload["result_json"])
    wout_path = payload.get("wout_path")
    n_species = int(payload["n_species"])
    benchmark_repeats = max(0, int(payload.get("benchmark_repeats", 0)))
    benchmark_warmup = max(0, int(payload.get("benchmark_warmup", 0)))

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from sfincs_jax.io import read_sfincs_h5, write_sfincs_jax_output_h5

    solve_count = 1 if benchmark_repeats <= 0 else benchmark_warmup + benchmark_repeats
    elapsed_s: list[float] = []
    for _ in range(solve_count):
        t0 = time.perf_counter()
        write_sfincs_jax_output_h5(
            input_namelist=input_path,
            output_path=output_path,
            wout_path=None if wout_path in (None, "") else Path(wout_path),
            compute_transport_matrix=False,
            compute_solution=True,
            overwrite=True,
            verbose=bool(payload.get("verbose", False)),
        )
        elapsed_s.append(time.perf_counter() - t0)
    results = read_sfincs_h5(output_path)
    gamma, q, upar, r_hat, meta = _extract_flux_triplet(results, n_species)
    summary = {
        "radius_index": int(payload["radius_index"]),
        "rho": float(payload["rho"]),
        "rHat": None if r_hat is None else float(r_hat),
        "Gamma": gamma.tolist(),
        "Q": q.tolist(),
        "Upar": upar.tolist(),
        "meta": meta,
    }
    if benchmark_repeats > 0:
        warm_runs = elapsed_s[benchmark_warmup:]
        summary["benchmark"] = {
            "warmup_runs": int(benchmark_warmup),
            "repeats": int(benchmark_repeats),
            "all_runs_s": [float(v) for v in elapsed_s],
            "cold_run_s": float(elapsed_s[0]),
            "warm_runs_s": [float(v) for v in warm_runs],
            "warm_mean_s": float(np.mean(warm_runs)) if warm_runs else float("nan"),
            "warm_min_s": float(np.min(warm_runs)) if warm_runs else float("nan"),
        }
    with result_json.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return 0


def _existing_result_is_usable(
    result_json: Path,
    *,
    require_benchmark: bool,
) -> bool:
    if not result_json.exists():
        return False
    try:
        summary = json.loads(result_json.read_text(encoding="utf-8"))
    except Exception:
        return False
    required = ("rho", "rHat", "Gamma", "Q", "Upar")
    if any(key not in summary for key in required):
        return False
    if require_benchmark and "benchmark" not in summary:
        return False
    return True


def _write_summary_plots(
    *,
    output_dir: Path,
    rho: np.ndarray,
    gamma: np.ndarray,
    q: np.ndarray,
    upar: np.ndarray,
    species: list[SpeciesMeta],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "Plotting was requested but matplotlib is not installed."
        ) from exc

    traces = (
        ("Gamma", np.asarray(gamma, dtype=np.float64), output_dir / "Gamma_vs_rho.png"),
        ("Q", np.asarray(q, dtype=np.float64), output_dir / "Q_vs_rho.png"),
        ("Upar", np.asarray(upar, dtype=np.float64), output_dir / "Upar_vs_rho.png"),
    )

    for label, values, path in traces:
        fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
        for i, sp in enumerate(species):
            ax.plot(rho, values[i], marker="o", linewidth=1.5, markersize=4.0, label=sp.name)
        ax.set_xlabel("rho")
        ax.set_ylabel(label)
        ax.set_title(f"{label} from sfincs_jax radial scan")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.savefig(path, dpi=180)
        plt.close(fig)


def _suppress_benign_worker_stderr(stderr: str, args: argparse.Namespace) -> str:
    if str(args.backend).lower() != "cpu":
        return stderr
    text = (stderr or "").strip()
    if not text:
        return ""
    benign_markers = (
        "Jax plugin configuration error",
        "jax_plugins.xla_cuda12.initialize()",
        "cuda_device_count()",
        "operation cuInit(0) failed: CUDA_ERROR_NO_DEVICE",
    )
    if all(marker in text for marker in benign_markers):
        return ""
    return stderr


def _launch_one_subprocess(
    *,
    script_path: Path,
    payload_path: Path,
    env: dict[str, str],
    stream_output: bool,
) -> subprocess.Popen[str]:
    cmd = [sys.executable, str(script_path), "--worker-payload", str(payload_path)]
    kwargs: dict[str, Any] = {
        "env": env,
        "text": True,
    }
    if not stream_output:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(
        cmd,
        **kwargs,
    )


def _terminate_worker_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            proc.terminate()
        else:
            os.killpg(proc.pid, 15)
    except ProcessLookupError:
        return
    except Exception:
        try:
            proc.terminate()
        except Exception:
            return


def _kill_worker_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            proc.kill()
        else:
            os.killpg(proc.pid, 9)
    except ProcessLookupError:
        return
    except Exception:
        try:
            proc.kill()
        except Exception:
            return


def _cleanup_worker_processes(active: list[dict[str, Any]]) -> None:
    for item in active:
        _terminate_worker_process(item["proc"])
    deadline = time.time() + 5.0
    while time.time() < deadline:
        remaining = [item for item in active if item["proc"].poll() is None]
        if not remaining:
            return
        time.sleep(0.1)
    for item in active:
        _kill_worker_process(item["proc"])
    for item in active:
        proc = item["proc"]
        try:
            proc.communicate(timeout=1.0)
        except Exception:
            pass


def _collect_worker_result(
    proc: subprocess.Popen[str],
    *,
    stream_output: bool,
) -> tuple[int, str, str]:
    if stream_output:
        code = proc.wait()
        return int(code), "", ""
    stdout, stderr = proc.communicate()
    return int(proc.returncode), stdout or "", stderr or ""


def _run_tasks_in_parallel(
    *,
    task_payloads: list[Path],
    args: argparse.Namespace,
    gpu_ids: list[str],
) -> None:
    max_parallel = max(1, int(args.max_parallel))
    script_path = Path(__file__).resolve()
    total = len(task_payloads)
    completed = 0

    def _spawn(payload_path: Path, slot: int) -> dict[str, Any]:
        gpu_id = None
        if str(args.backend).lower() == "gpu":
            gpu_id = gpu_ids[slot % len(gpu_ids)]
        env = _build_worker_env(args, gpu_id=gpu_id)
        stream_output = bool(args.verbose_workers) and max_parallel == 1
        proc = _launch_one_subprocess(
            script_path=script_path,
            payload_path=payload_path,
            env=env,
            stream_output=stream_output,
        )
        return {
            "proc": proc,
            "payload_path": payload_path,
            "gpu_id": gpu_id,
            "stream_output": stream_output,
        }

    active: list[dict[str, Any]] = []
    pending = list(task_payloads)
    slot = 0
    try:
        while pending or active:
            while pending and len(active) < max_parallel:
                payload_path = pending.pop(0)
                worker = _spawn(payload_path, slot)
                active.append(worker)
                rho_value = None
                try:
                    payload = json.loads(Path(payload_path).read_text(encoding="utf-8"))
                    rho_value = float(payload.get("rho"))
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    rho_value = None
                rho_note = f" rho={rho_value:.4f}" if rho_value is not None else ""
                gpu_note = f" gpu={worker['gpu_id']}" if worker["gpu_id"] is not None else ""
                print(
                    f"[sfincs-scan] launched {completed + len(active)}/{total}:{rho_note}{gpu_note}",
                    flush=True,
                )
                slot += 1
            time.sleep(0.1)
            finished: list[dict[str, Any]] = []
            for item in active:
                if item["proc"].poll() is not None:
                    finished.append(item)
            if not finished:
                continue
            active = [item for item in active if item not in finished]
            for item in finished:
                proc = item["proc"]
                payload_path = item["payload_path"]
                gpu_id = item["gpu_id"]
                stream_output = item["stream_output"]
                code, stdout, stderr = _collect_worker_result(proc, stream_output=stream_output)
                label = str(payload_path) if payload_path is not None else "<payload>"
                if code != 0:
                    _cleanup_worker_processes(active)
                    msg = [
                        f"sfincs_jax worker failed for {label}",
                    ]
                    if gpu_id is not None:
                        msg.append(f"gpu={gpu_id}")
                    if code < 0 or code in (137, 143):
                        msg.append(
                            f"worker exited via signal (returncode={code}) "
                            "with no Python traceback. This signature usually "
                            "means the container ran out of memory and the "
                            "kernel killed the worker. Try one of:"
                        )
                        msg.append(
                            f"  - lower --max-parallel "
                            f"(current run: {int(args.max_parallel)})"
                        )
                        msg.append(
                            "  - increase the RAM available to the workers "
                            "(either on the host or in the container runtime)"
                        )
                    if stdout.strip():
                        msg.append("stdout:\n" + stdout.strip())
                    if stderr.strip():
                        msg.append("stderr:\n" + stderr.strip())
                    raise RuntimeError("\n".join(msg))
                if bool(args.verbose_workers) and not stream_output:
                    if stdout.strip():
                        print(f"[sfincs-worker stdout] {label}\n{stdout.strip()}", flush=True)
                    stderr_to_print = _suppress_benign_worker_stderr(stderr, args)
                    if stderr_to_print.strip():
                        print(f"[sfincs-worker stderr] {label}\n{stderr_to_print.strip()}", flush=True)
                completed += 1
                rho_value = None
                if payload_path is not None:
                    try:
                        payload = json.loads(Path(payload_path).read_text(encoding="utf-8"))
                        rho_value = float(payload.get("rho"))
                    except (OSError, ValueError, TypeError, json.JSONDecodeError):
                        rho_value = None
                rho_note = f" rho={rho_value:.4f}" if rho_value is not None else ""
                gpu_note = f" gpu={gpu_id}" if gpu_id is not None else ""
                print(f"[sfincs-scan] completed {completed}/{total}:{rho_note}{gpu_note}", flush=True)
    except KeyboardInterrupt:
        _cleanup_worker_processes(active)
        raise
    except Exception:
        _cleanup_worker_processes(active)
        raise


def _prepare(args: argparse.Namespace) -> tuple[dict[str, Any], list[Path]]:
    """Build per-surface sfincs_jax inputs and the scan manifest.

    Loads profiles, selects flux surfaces, and writes an ``input.namelist`` plus ``payload.json`` under
    each ``output_dir/runs/<run_subdir>`` directory, then records ``output_dir/manifest.json`` describing
    the scan. Reuse detection is preserved: a surface whose namelist is byte-identical to the existing
    one and whose ``result.json`` is usable is treated as already complete and is not returned as
    pending.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments for the prepare phase.

    Returns
    -------
    tuple[dict[str, Any], list[Path]]
        The manifest dict (also written to ``manifest.json``) and the ``payload.json`` paths of surfaces
        that still need to run. Pending-ness is in-memory only and is never stored in the manifest. Each
        ``runs`` entry records only paths relative to ``output_dir`` so a host scheduler can rebuild real
        paths under its own view of the output directory.
    """
    config_path = Path(args.common_config).resolve()
    cfg = _load_toml(config_path)
    species = _parse_species_from_config(cfg)
    profiles_source = str(args.profiles_source).lower()
    if profiles_source == "analytical":
        transport_solution = None
        snapshot = _build_standard_analytical_snapshot(
            cfg,
            n_species=len(species),
            n_radial=int(args.analytical_n_radii),
        )
    elif profiles_source == "prescribed":
        transport_solution = None
        snapshot = _build_prescribed_snapshot(cfg, n_species=len(species))
    else:
        transport_solution = (
            Path(args.neopax_result).resolve()
            if args.neopax_result is not None
            else _default_transport_solution_path(config_path, cfg)
        )
        if not transport_solution.exists():
            raise FileNotFoundError(
                f"Could not find NEOPAX transport result at {transport_solution}. "
                "If needed, enable [transport_output].transport_write_hdf5 = true and rerun NEOPAX."
            )
        snapshot = _load_transport_snapshot(transport_solution, time_index=int(args.time_index))
    explicit_indices = _parse_index_list(args.rho_indices)
    radius_indices = _choose_radius_indices(
        snapshot.rho,
        explicit=explicit_indices,
        rho_min=args.rho_min,
        rho_max=args.rho_max,
        num_radii=args.num_radii,
    )

    template_path = Path(args.sfincs_template).resolve()
    template_text = template_path.read_text(encoding="utf-8")

    output_dir = Path(args.output_dir).resolve()
    run_dir = output_dir / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)

    wout_path = _infer_wout_path(config_path, cfg, args.wout_path)
    resolution_overrides = {
        "Ntheta": args.ntheta,
        "Nzeta": args.nzeta,
        "Nxi": args.nxi,
        "NL": args.nl,
        "Nx": args.nx,
    }

    pending_task_payloads: list[Path] = []
    manifest_runs: list[dict[str, Any]] = []
    for position, radius_index in enumerate(radius_indices):
        rho_value = float(snapshot.rho[radius_index])
        run_name = f"rho_{radius_index:03d}_r{rho_value:.4f}".replace(".", "p")
        surface_dir = run_dir / run_name
        surface_dir.mkdir(parents=True, exist_ok=True)

        input_text = _prepare_input_text(
            template_text=template_text,
            species=species,
            snapshot=snapshot,
            radius_index=radius_index,
            include_phi1=args.include_phi1,
            resolution_overrides=resolution_overrides,
            solver_tolerance=args.solver_tolerance,
        )
        input_path = surface_dir / "input.namelist"
        result_json = surface_dir / "result.json"
        existing_input_text = None
        if input_path.exists():
            try:
                existing_input_text = input_path.read_text(encoding="utf-8")
            except Exception:
                existing_input_text = None
        can_reuse = (
            existing_input_text == input_text
            and _existing_result_is_usable(
                result_json,
                require_benchmark=bool(int(args.benchmark_repeats) > 0),
            )
        )
        input_path.write_text(input_text, encoding="utf-8")

        payload = {
            "radius_index": int(radius_index),
            "rho": rho_value,
            "input_path": str(input_path),
            "output_path": str(surface_dir / "sfincsOutput.h5"),
            "result_json": str(result_json),
            "wout_path": None if wout_path is None else str(wout_path),
            "n_species": len(species),
            "verbose": bool(args.verbose_workers),
            "benchmark_repeats": int(args.benchmark_repeats),
            "benchmark_warmup": int(args.benchmark_warmup),
        }
        payload_path = surface_dir / "payload.json"
        with payload_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        if not can_reuse:
            pending_task_payloads.append(payload_path)
        manifest_runs.append(
            {
                "index": position,
                "radius_index": int(radius_index),
                "rho": rho_value,
                "run_subdir": run_name,
                "payload": "payload.json",
                "result_json": "result.json",
            }
        )

    manifest = {
        "schema_version": 1,
        "profiles_source": str(args.profiles_source),
        "source_transport_solution": None if transport_solution is None else str(transport_solution),
        "source_sfincs_template": str(template_path),
        "time_index": int(args.time_index),
        "time_value": None if snapshot.time_value is None else float(snapshot.time_value),
        "include_phi1": args.include_phi1,
        "backend": str(args.backend).lower(),
        "max_parallel": int(args.max_parallel),
        "worker_sharding": str(args.worker_sharding).lower(),
        "benchmark_repeats": int(args.benchmark_repeats),
        "benchmark_warmup": int(args.benchmark_warmup),
        "species_meta": [
            {"name": sp.name, "charge": sp.charge, "mass_mp": sp.mass_mp} for sp in species
        ],
        "wout_path": None if wout_path is None else str(wout_path),
        "runs": manifest_runs,
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)

    return manifest, pending_task_payloads


def _collect(args: argparse.Namespace, manifest: dict[str, Any]) -> int:
    """Reduce per-surface worker results into the Stage 3 flux-profile HDF5.

    Reads each surface's ``result.json`` (located relative to ``output_dir`` from the manifest run
    entries), stacks the flux arrays, applies the magnetic-axis zero-padding, and writes
    ``sfincs_jax_flux_profiles.h5``. Provenance attributes are taken from the manifest rather than live
    arguments so this phase can run independently of the process that prepared the scan.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments; only ``output_dir`` and ``plot`` are consumed.
    manifest : dict[str, Any]
        The scan manifest produced by :func:`_prepare`.

    Returns
    -------
    int
        Process exit code (0 on success).
    """
    output_dir = Path(args.output_dir).resolve()
    run_dir = output_dir / "runs"
    species = [
        SpeciesMeta(name=meta["name"], charge=meta["charge"], mass_mp=meta["mass_mp"])
        for meta in manifest["species_meta"]
    ]

    rho_out = []
    rhat_out = []
    gamma_out = []
    q_out = []
    upar_out = []
    raw_meta = {"Gamma_key": None, "Q_key": None, "Upar_key": None}
    benchmark_rows: list[dict[str, Any]] = []
    for run in sorted(manifest["runs"], key=lambda r: r["rho"]):
        result_json = run_dir / run["run_subdir"] / run["result_json"]
        if not result_json.exists():
            raise FileNotFoundError(
                f"Missing sfincs_jax worker result for run {run['run_subdir']} at {result_json}. "
                "Run the surface before collecting."
            )
        summary = json.loads(result_json.read_text(encoding="utf-8"))
        rho_out.append(float(summary["rho"]))
        rhat_value = summary.get("rHat")
        if rhat_value is None:
            raise KeyError(f"Worker summary {result_json} is missing rHat.")
        rhat_out.append(float(rhat_value))
        gamma_out.append(np.asarray(summary["Gamma"], dtype=np.float64))
        q_out.append(np.asarray(summary["Q"], dtype=np.float64))
        upar_out.append(np.asarray(summary["Upar"], dtype=np.float64))
        raw_meta = dict(summary.get("meta", raw_meta))
        if "benchmark" in summary:
            benchmark_rows.append(
                {
                    "rho": float(summary["rho"]),
                    "radius_index": int(summary["radius_index"]),
                    **dict(summary["benchmark"]),
                }
            )

    rho_arr = np.asarray(rho_out, dtype=np.float64)
    rhat_arr = np.asarray(rhat_out, dtype=np.float64)
    gamma_arr = np.stack(gamma_out, axis=1)
    q_arr = np.stack(q_out, axis=1)
    upar_arr = np.stack(upar_out, axis=1)
    axis_padded = False
    if rho_arr.size > 0 and not np.any(np.isclose(rho_arr, 0.0)):
        zero_flux = np.zeros((len(species), 1), dtype=np.float64)
        rho_arr = np.concatenate([np.asarray([0.0], dtype=np.float64), rho_arr])
        rhat_arr = np.concatenate([np.asarray([0.0], dtype=np.float64), rhat_arr])
        gamma_arr = np.concatenate([zero_flux, gamma_arr], axis=1)
        q_arr = np.concatenate([zero_flux, q_arr], axis=1)
        upar_arr = np.concatenate([zero_flux, upar_arr], axis=1)
        axis_padded = True

    time_value = manifest["time_value"]
    include_phi1 = manifest["include_phi1"]
    out_h5 = output_dir / "sfincs_jax_flux_profiles.h5"
    with h5py.File(out_h5, "w") as f:
        f.create_dataset("r", data=rhat_arr)
        f.create_dataset("rHat", data=rhat_arr)
        f.create_dataset("rho", data=rho_arr)
        f.create_dataset("Gamma", data=gamma_arr)
        f.create_dataset("Q", data=q_arr)
        f.create_dataset("Upar", data=upar_arr)
        f.create_dataset("species_names", data=np.asarray([sp.name.encode("utf-8") for sp in species]))
        f.attrs["profiles_source"] = str(manifest["profiles_source"])
        f.attrs["source_transport_solution"] = (
            "" if manifest["source_transport_solution"] is None else str(manifest["source_transport_solution"])
        )
        f.attrs["source_sfincs_template"] = str(manifest["source_sfincs_template"])
        f.attrs["time_index"] = int(manifest["time_index"])
        f.attrs["time_value"] = np.nan if time_value is None else float(time_value)
        f.attrs["backend"] = str(manifest["backend"])
        f.attrs["max_parallel"] = int(manifest["max_parallel"])
        f.attrs["worker_sharding"] = str(manifest["worker_sharding"])
        f.attrs["include_phi1"] = bool(include_phi1) if include_phi1 is not None else -1
        f.attrs["axis_zero_padded"] = bool(axis_padded)
        for key, value in raw_meta.items():
            if value is not None:
                f.attrs[f"raw_{key}"] = str(value)
        f.attrs["Upar_note"] = (
            "Upar is derived from sfincs_jax FSABFlow using the NTX fixed-field "
            "parallel-flow bridge factor 2*B0OverBBar/sqrt(pi)."
        )
        f.attrs["normalization_note"] = (
            "Gamma is converted from sfincs_jax particleFlux_vm_* using nbar*vbar/Rbar "
            "with nbar=1e20 m^-3, Tbar=1 keV, mbar=mp, Rbar=1 m. "
            "Q is converted from heatFlux_vm_* using (nbar*mbar*vbar^3/Rbar)/e so the written "
            "values match NEOPAX's eV-based physical heat-flux convention. "
            "Upar uses the NTX archive-backed observable bridge rather than a raw FSABFlow alias."
        )
        f.attrs["radius_note"] = (
            "The saved coordinate r/rHat uses sfincs_jax's rHat output. "
            "rho is also saved separately from the source NEOPAX transport file."
        )
        if benchmark_rows:
            f.attrs["benchmark_note"] = (
                "Timing benchmark mode was enabled. cold_run_s includes first-run JAX startup/compile effects; "
                "warm_* fields are repeated same-process timings after the configured warmup runs."
            )
            f.create_dataset(
                "benchmark_rho",
                data=np.asarray([row["rho"] for row in benchmark_rows], dtype=np.float64),
            )
            f.create_dataset(
                "benchmark_cold_run_s",
                data=np.asarray([row["cold_run_s"] for row in benchmark_rows], dtype=np.float64),
            )
            f.create_dataset(
                "benchmark_warm_mean_s",
                data=np.asarray([row["warm_mean_s"] for row in benchmark_rows], dtype=np.float64),
            )

    if bool(args.plot):
        _write_summary_plots(
            output_dir=output_dir,
            rho=rho_arr,
            gamma=gamma_arr,
            q=q_arr,
            upar=upar_arr,
            species=species,
        )

    if benchmark_rows:
        print("[sfincs-scan] benchmark summary (same-worker repeated solves):", flush=True)
        for row in benchmark_rows:
            print(
                "[sfincs-scan] "
                f"rho={float(row['rho']):.4f} cold={float(row['cold_run_s']):.3f}s "
                f"warm_mean={float(row['warm_mean_s']):.3f}s "
                f"warm_min={float(row['warm_min_s']):.3f}s "
                f"repeats={int(row['repeats'])}",
                flush=True,
            )

    print(f"wrote {out_h5}", flush=True)
    return 0


def cmd_main(args: argparse.Namespace) -> int:
    manifest, pending_task_payloads = _prepare(args)

    backend = str(args.backend).lower()
    gpu_ids = [token.strip() for token in str(args.gpu_ids).split(",") if token.strip()]
    if backend == "gpu" and not gpu_ids:
        gpu_ids = ["0"]

    runs = manifest["runs"]
    total_runs = len(runs)
    reused_runs = total_runs - len(pending_task_payloads)
    rho_values = [run["rho"] for run in runs]
    rho_min_val = float(min(rho_values))
    rho_max_val = float(max(rho_values))
    backend_note = f"backend={backend}"
    parallel_note = f"max_parallel={int(args.max_parallel)}"
    if backend == "gpu":
        placement_note = f"gpu_ids={','.join(gpu_ids)}"
    else:
        placement_note = (
            f"cores_per_run={int(args.cores_per_run)} "
            f"worker_sharding={str(args.worker_sharding).lower()}"
        )
    print(
        "[sfincs-scan] "
        f"selected {total_runs} radii over rho in [{rho_min_val:.4f}, {rho_max_val:.4f}] "
        f"({backend_note}, {parallel_note}, {placement_note})"
    , flush=True)
    if reused_runs:
        print(
            "[sfincs-scan] "
            f"reusing {reused_runs}/{total_runs} existing completed runs; "
            f"launching {len(pending_task_payloads)} new workers.",
            flush=True,
        )

    if pending_task_payloads:
        _run_tasks_in_parallel(
            task_payloads=pending_task_payloads,
            args=args,
            gpu_ids=gpu_ids,
        )

    return _collect(args, manifest)


def cmd_prepare(args: argparse.Namespace) -> int:
    _prepare(args)
    return 0


def cmd_run_one(args: argparse.Namespace) -> int:
    payload_path = Path(args.payload).resolve()
    gpu_id = None
    if str(args.backend).lower() == "gpu":
        gpu_tokens = [token.strip() for token in str(args.gpu_ids).split(",") if token.strip()]
        gpu_id = gpu_tokens[0] if gpu_tokens else "0"
    env = _build_worker_env(args, gpu_id=gpu_id)
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker-payload", str(payload_path)],
        env=env,
    )
    return proc.returncode


def cmd_collect(args: argparse.Namespace) -> int:
    manifest_path = Path(args.output_dir).resolve() / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No manifest.json found at {manifest_path}. Run the prepare phase before collecting."
        )
    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    return _collect(args, manifest)


def _add_output_dir(p: argparse.ArgumentParser) -> None:
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory for runs and collected fluxes.")


def _add_io_and_shaping(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--common-config",
        default=str(DEFAULT_COMMON_CONFIG),
        help="Path to the shared common_input TOML.",
    )
    p.add_argument(
        "--profiles-source",
        choices=("transport_h5", "analytical", "prescribed"),
        default="analytical",
        help=(
            "Choose whether profiles come from transport_solution.h5, from the TOML analytical profile "
            "block, or from prescribed profile arrays in the TOML [profiles] block."
        ),
    )
    p.add_argument(
        "--neopax-result",
        default=None,
        help="Optional explicit path to transport_solution.h5. Used only for profiles-source=transport_h5.",
    )
    p.add_argument(
        "--analytical-n-radii",
        type=int,
        default=51,
        help="Number of rho grid points to reconstruct for profiles-source=analytical.",
    )
    p.add_argument(
        "--sfincs-template",
        default=str(DEFAULT_SFINCS_TEMPLATE),
        help="Template sfincs_jax input.namelist.",
    )
    _add_output_dir(p)
    p.add_argument("--time-index", type=int, default=-1, help="Time index in transport_solution.h5. Default: final.")
    p.add_argument("--rho-indices", default=None, help="Comma-separated explicit rho indices.")
    p.add_argument("--rho-min", type=float, default=None, help="Minimum rho to include.")
    p.add_argument("--rho-max", type=float, default=None, help="Maximum rho to include.")
    p.add_argument("--num-radii", type=int, default=None, help="Number of radii to sample inside the rho filter.")
    p.add_argument("--wout-path", default=None, help="Optional VMEC equilibrium override for sfincs_jax.")
    p.add_argument("--include-phi1", dest="include_phi1", action="store_true", help="Force includePhi1 = true.")
    p.add_argument("--no-include-phi1", dest="include_phi1", action="store_false", help="Force includePhi1 = false.")
    p.set_defaults(include_phi1=None)
    p.add_argument("--ntheta", type=int, default=5, help="Override Ntheta. Default: 5.")
    p.add_argument("--nzeta", type=int, default=11, help="Override Nzeta. Default: 11.")
    p.add_argument("--nxi", type=int, default=12, help="Override Nxi. Default: 12.")
    p.add_argument("--nl", type=int, default=3, help="Override NL. Default: 3.")
    p.add_argument("--nx", type=int, default=4, help="Override Nx. Default: 4.")
    p.add_argument("--solver-tolerance", type=float, default=1.0e-6, help="Override solverTolerance. Default: 1e-6.")


def _add_dense_fp_max(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--dense-fp-max",
        type=int,
        default=None,
        help="Set SFINCS_JAX_RHSMODE1_DENSE_FP_MAX for worker processes.",
    )


def _add_backend(p: argparse.ArgumentParser) -> None:
    p.add_argument("--backend", choices=("cpu", "gpu"), default="cpu", help="Parallel execution backend.")


def _add_gpu_ids(p: argparse.ArgumentParser) -> None:
    p.add_argument("--gpu-ids", default="0", help="Comma-separated GPU ids for backend=gpu.")


def _add_max_parallel(p: argparse.ArgumentParser) -> None:
    p.add_argument("--max-parallel", type=int, default=8, help="Maximum concurrent sfincs_jax runs.")


def _add_cores_per_run(p: argparse.ArgumentParser) -> None:
    p.add_argument("--cores-per-run", type=int, default=1, help="CPU cores per run for backend=cpu.")


def _add_worker_sharding(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--worker-sharding",
        choices=("off", "auto", "theta", "zeta", "x", "flat"),
        default="off",
        help=(
            "Per-worker sfincs_jax sharding mode for backend=cpu when cores-per-run > 1. "
            "Default: off."
        ),
    )


def _add_benchmark(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--benchmark-repeats",
        type=int,
        default=0,
        help="Repeat each selected surface this many extra times inside one worker and report warm timings.",
    )
    p.add_argument(
        "--benchmark-warmup",
        type=int,
        default=1,
        help="Number of same-worker warmup solves to discard before benchmark repeats.",
    )


def _add_plot(p: argparse.ArgumentParser) -> None:
    p.add_argument("--plot", dest="plot", action="store_true", help="Write PNG plots of Gamma, Q, and Upar versus rho.")
    p.add_argument("--no-plot", dest="plot", action="store_false", help="Skip PNG plots.")
    p.set_defaults(plot=True)


def _add_verbose_workers(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--verbose-workers",
        dest="verbose_workers",
        action="store_true",
        help="Allow verbose sfincs_jax worker logging.",
    )
    p.add_argument(
        "--no-verbose-workers",
        dest="verbose_workers",
        action="store_false",
        help="Silence sfincs_jax worker logging.",
    )
    p.set_defaults(verbose_workers=True)


def _add_worker_payload(p: argparse.ArgumentParser) -> None:
    p.add_argument("--worker-payload", default=None, help=argparse.SUPPRESS)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            __doc__
            + "\n\n"
            + "Unless overridden on the command line, this script forces the following "
            + "sfincs_jax resolution settings (quickrun smoke test): "
            + "Ntheta=5, Nzeta=11, Nxi=12, NL=3, Nx=4, solverTolerance=1e-6."
        )
    )
    _add_io_and_shaping(p)
    _add_dense_fp_max(p)
    _add_backend(p)
    _add_gpu_ids(p)
    _add_max_parallel(p)
    _add_cores_per_run(p)
    _add_worker_sharding(p)
    _add_benchmark(p)
    _add_plot(p)
    _add_verbose_workers(p)
    _add_worker_payload(p)
    p.set_defaults(func=cmd_main)

    sub = p.add_subparsers(dest="command", required=False)

    prepare = sub.add_parser("prepare", help="Write per-surface sfincs_jax inputs and the scan manifest.")
    _add_io_and_shaping(prepare)
    _add_backend(prepare)
    _add_max_parallel(prepare)
    _add_worker_sharding(prepare)
    _add_benchmark(prepare)
    _add_verbose_workers(prepare)
    prepare.set_defaults(func=cmd_prepare)

    run_one = sub.add_parser("run-one", help=argparse.SUPPRESS)
    run_one.add_argument("--payload", required=True, help="Path to a per-surface payload.json to execute.")
    _add_dense_fp_max(run_one)
    _add_backend(run_one)
    _add_gpu_ids(run_one)
    _add_cores_per_run(run_one)
    _add_worker_sharding(run_one)
    run_one.set_defaults(func=cmd_run_one)

    collect = sub.add_parser("collect", help="Reduce per-surface results into the flux-profile HDF5.")
    _add_output_dir(collect)
    _add_plot(collect)
    collect.set_defaults(func=cmd_collect)

    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.worker_payload:
        return _run_single_worker_from_payload(Path(args.worker_payload).resolve())
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
