"""Run runtime-resolution convergence and flux stability checks.

The script performs two checks:

1. Resolution convergence: rerun the same runtime case on a small sequence of
   increasing resolutions and verify that late-time diagnostics do not shift
   much between refinement levels.
2. Flux stability: verify that the late-time heat/particle flux is with small
   relative variance

If a reference .nc file is provided, the script also compares the runs.

run with 
python run_physics_validation_checks.py --config path/to/runtime.toml --reference-nc path/to/reference.nc --outdir path/to/output_dir

Outputs a json file of summaries, including convergence test settings, diagnostic 
statistics, and pass/fail results for two of the tests.
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from gkx.io import load_runtime_from_toml
from gkx.runtime import RuntimeNonlinearResult, run_runtime_nonlinear
from gkx.runtime_artifacts import write_runtime_nonlinear_artifacts


def _parse_scales(value: str) -> list[float]:
    scales: list[float] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        scale = float(part)
        if scale <= 0.0:
            raise argparse.ArgumentTypeError("resolution scales must be > 0")
        scales.append(scale)
    if not scales:
        raise argparse.ArgumentTypeError("at least one resolution scale is required")
    return scales


def _scale_int(value: int, scale: float) -> int:
    return max(1, int(math.ceil(float(value) * float(scale))))


def _window_stats(series: np.ndarray, *, tail_fraction: float) -> dict[str, float]:
    data = np.asarray(series, dtype=float).reshape(-1)
    finite = np.isfinite(data)
    data = data[finite]
    if data.size == 0:
        raise ValueError("series contains no finite samples")
    start = int(max(0, math.floor((1.0 - tail_fraction) * data.size)))
    tail = data[start:]
    if tail.size == 0:
        tail = data
    mean = float(np.mean(tail))
    std = float(np.std(tail))
    rel_std = float(std / max(abs(mean), 1.0e-30))
    x = np.arange(tail.size, dtype=float)
    if tail.size >= 2:
        slope = float(np.polyfit(x, tail, 1)[0])
    else:
        slope = 0.0
    rel_slope = float(abs(slope) / max(abs(mean), 1.0e-30))
    return {
        "mean": mean,
        "std": std,
        "rel_std": rel_std,
        "slope": slope,
        "rel_slope": rel_slope,
        "n_tail": float(tail.size),
    }


def _diag_summary(result: RuntimeNonlinearResult, *, tail_fraction: float) -> dict[str, dict[str, float]]:
    if result.diagnostics is None:
        raise RuntimeError("runtime run did not return diagnostics")
    diag = result.diagnostics
    return {
        "gamma": _window_stats(np.asarray(diag.gamma_t), tail_fraction=tail_fraction),
        "omega": _window_stats(np.asarray(diag.omega_t), tail_fraction=tail_fraction),
        "Wg": _window_stats(np.asarray(diag.Wg_t), tail_fraction=tail_fraction),
        "Wphi": _window_stats(np.asarray(diag.Wphi_t), tail_fraction=tail_fraction),
        "Wapar": _window_stats(np.asarray(diag.Wapar_t), tail_fraction=tail_fraction),
        "energy": _window_stats(np.asarray(diag.energy_t), tail_fraction=tail_fraction),
        "heat_flux": _window_stats(np.asarray(diag.heat_flux_t), tail_fraction=tail_fraction),
        "particle_flux": _window_stats(np.asarray(diag.particle_flux_t), tail_fraction=tail_fraction),
    }


def _reference_last_scalars(path: Path) -> dict[str, float]:
    """Load the final scalar diagnostics from a GX-style .nc file."""

    try:
        from netCDF4 import Dataset
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "reference-nc comparison requested, but the optional 'netCDF4' dependency is not installed"
        ) from exc

    scalar_names = {
        "Wg": ("Wg_kyst", "Wg_t"),
        "Wphi": ("Wphi_kyst", "Wphi_t"),
        "Wapar": ("Wapar_kyst", "Wapar_t"),
        "heat_flux": ("HeatFlux_kyst", "heat_flux_t"),
        "particle_flux": ("ParticleFlux_kyst", "particle_flux_t"),
    }

    out: dict[str, float] = {}
    with Dataset(path, "r") as root:
        group = root.groups["Diagnostics"] if "Diagnostics" in root.groups else root
        for key, candidates in scalar_names.items():
            for name in candidates:
                if name in group.variables:
                    arr = np.asarray(group.variables[name][-1], dtype=float)
                    out[key] = float(np.sum(arr))
                    break
            else:
                raise KeyError(f"{path} does not contain any known diagnostics variable for '{key}'")
    return out


def _relative_error(test: float, ref: float) -> float:
    return float(abs(test - ref) / max(abs(ref), 1.0e-30))


def _compare_scalars(test: dict[str, float], ref: dict[str, float]) -> dict[str, dict[str, float]]:
    common = test.keys() & ref.keys()
    return {
        key: {
            "abs": float(abs(test[key] - ref[key])),
            "rel": _relative_error(test[key], ref[key]),
        }
        for key in sorted(common)
    }


def _run_case(
    cfg,
    *,
    scale: float,
    base_nl: int,
    base_nm: int,
    ky_target: float,
    kx_target: float | None,
    dt: float | None,
    steps: int | None,
    sample_stride: int | None,
    diagnostics_stride: int | None,
    laguerre_mode: str | None,
    diagnostics: bool | None,
    return_state: bool,
    show_progress: bool,
) -> RuntimeNonlinearResult:
    grid = replace(
        cfg.grid,
        Nx=_scale_int(cfg.grid.Nx, scale),
        Ny=_scale_int(cfg.grid.Ny, scale),
        Nz=_scale_int(cfg.grid.Nz, scale),
    )
    cfg_use = replace(cfg, grid=grid)
    nl_use = _scale_int(base_nl, scale)
    nm_use = _scale_int(base_nm, scale)
    return run_runtime_nonlinear(
        cfg_use,
        ky_target=ky_target,
        kx_target=kx_target,
        Nl=nl_use,
        Nm=nm_use,
        dt=dt,
        steps=steps,
        sample_stride=sample_stride,
        diagnostics_stride=diagnostics_stride,
        laguerre_mode=laguerre_mode,
        diagnostics=diagnostics,
        return_state=return_state,
        show_progress=show_progress,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True, help="Runtime TOML file.")
    p.add_argument(
        "--reference-nc",
        type=Path,
        default=None,
        help="Optional GX-style .nc file used as a reference diagnostics target.",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("tools_out") / "runtime_validation_checks",
        help="Directory where summaries and diagnostics CSV files are written.",
    )
    p.add_argument(
        "--scales",
        type=_parse_scales,
        default=_parse_scales("1.0,1.5,2.0"),
        help="Comma-separated resolution scales for the convergence sweep.",
    )
    p.add_argument("--ky-target", type=float, default=0.3, help="Target ky for the nonlinear runtime run.")
    p.add_argument("--kx-target", type=float, default=None, help="Optional target kx for the nonlinear runtime run.")
    p.add_argument("--dt", type=float, default=None, help="Override the runtime timestep.")
    p.add_argument("--steps", type=int, default=None, help="Override the number of steps.")
    p.add_argument("--sample-stride", type=int, default=None, help="Override diagnostics sample stride.")
    p.add_argument("--diagnostics-stride", type=int, default=None, help="Override diagnostics stride.")
    p.add_argument(
        "--laguerre-mode",
        type=str,
        default=None,
        choices=("grid", "spectral"),
        help="Optional nonlinear Laguerre handling override.",
    )
    p.add_argument(
        "--tail-fraction",
        type=float,
        default=0.2,
        help="Fraction of the final samples used for plateau statistics.",
    )
    p.add_argument(
        "--convergence-rtol",
        type=float,
        default=1.0e-2,
        help="Maximum relative change allowed between consecutive resolution levels.",
    )
    p.add_argument(
        "--flux-rel-std-max",
        type=float,
        default=0.1,
        help="Maximum relative standard deviation allowed in the final flux window.",
    )
    p.add_argument(
        "--flux-rel-slope-max",
        type=float,
        default=0.1,
        help="Maximum relative late-window slope allowed for the flux plateau check.",
    )
    p.add_argument(
        "--reference-rtol",
        type=float,
        default=1.0e-2,
        help="Maximum relative error allowed against the reference .nc diagnostics.",
    )
    p.add_argument("--no-diagnostics", action="store_true", help="Disable runtime diagnostics output.")
    p.add_argument("--return-state", action="store_true", help="Also return and persist the final state.")
    p.add_argument("--progress", action="store_true", help="Show solver progress.")
    return p


def main() -> int:
    args = build_parser().parse_args()
    cfg, raw = load_runtime_from_toml(args.config)

    run_cfg = raw.get("run", {}) if isinstance(raw.get("run", {}), dict) else {}
    base_nl = int(run_cfg.get("Nl", 4))
    base_nm = int(run_cfg.get("Nm", 8))
    diagnostics_on = None if args.no_diagnostics else True

    outdir = args.outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    try:
        with tempfile.TemporaryDirectory(prefix="runtime_validation_", dir=str(outdir)) as tmpdir:
            tmp_root = Path(tmpdir)
            previous_metrics: dict[str, float] | None = None
            convergence_ok = True
            flux_ok = True

            for scale in args.scales:
                run_dir = tmp_root / f"scale_{scale:.3f}".replace(".", "p")
                run_dir.mkdir(parents=True, exist_ok=True)
                result = _run_case(
                    cfg,
                    scale=scale,
                    base_nl=base_nl,
                    base_nm=base_nm,
                    ky_target=float(args.ky_target),
                    kx_target=args.kx_target,
                    dt=args.dt,
                    steps=args.steps,
                    sample_stride=args.sample_stride,
                    diagnostics_stride=args.diagnostics_stride,
                    laguerre_mode=args.laguerre_mode,
                    diagnostics=diagnostics_on,
                    return_state=bool(args.return_state),
                    show_progress=bool(args.progress),
                )
                write_runtime_nonlinear_artifacts(run_dir / "run.csv", result)
                diag_summary = _diag_summary(result, tail_fraction=float(args.tail_fraction))
                scalar_metrics = {name: stats["mean"] for name, stats in diag_summary.items()}

                if previous_metrics is not None:
                    step_metrics = _compare_scalars(scalar_metrics, previous_metrics)
                    if any(vals["rel"] > float(args.convergence_rtol) for vals in step_metrics.values()):
                        convergence_ok = False
                previous_metrics = scalar_metrics

                flux_plateau = {
                    "heat_flux": {
                        "rel_std": diag_summary["heat_flux"]["rel_std"],
                        "rel_slope": diag_summary["heat_flux"]["rel_slope"],
                    },
                    "particle_flux": {
                        "rel_std": diag_summary["particle_flux"]["rel_std"],
                        "rel_slope": diag_summary["particle_flux"]["rel_slope"],
                    },
                }
                if (
                    flux_plateau["heat_flux"]["rel_std"] > float(args.flux_rel_std_max)
                    or flux_plateau["heat_flux"]["rel_slope"] > float(args.flux_rel_slope_max)
                    or flux_plateau["particle_flux"]["rel_std"] > float(args.flux_rel_std_max)
                    or flux_plateau["particle_flux"]["rel_slope"] > float(args.flux_rel_slope_max)
                ):
                    flux_ok = False

                results.append(
                    {
                        "scale": float(scale),
                        "Nl": _scale_int(base_nl, scale),
                        "Nm": _scale_int(base_nm, scale),
                        "grid": {
                            "Nx": _scale_int(cfg.grid.Nx, scale),
                            "Ny": _scale_int(cfg.grid.Ny, scale),
                            "Nz": _scale_int(cfg.grid.Nz, scale),
                        },
                        "selected_mode": {
                            "ky": float(result.ky_selected) if result.ky_selected is not None else None,
                            "kx": float(result.kx_selected) if result.kx_selected is not None else None,
                        },
                        "metrics": diag_summary,
                        "flux_plateau": flux_plateau,
                    }
                )

            reference_metrics: dict[str, float] = {}
            reference_errors: dict[str, dict[str, float]] = {}
            reference_ok = True
            if args.reference_nc is not None:
                reference_metrics = _reference_last_scalars(args.reference_nc)
                finest_metrics = results[-1]["metrics"]
                finest_scalar_metrics = {name: stats["mean"] for name, stats in finest_metrics.items()}
                reference_errors = _compare_scalars(finest_scalar_metrics, reference_metrics)
                reference_ok = all(vals["rel"] <= float(args.reference_rtol) for vals in reference_errors.values())

            summary = {
                "config": str(args.config.resolve()),
                "reference_nc": None if args.reference_nc is None else str(args.reference_nc.resolve()),
                "scales": list(map(float, args.scales)),
                "base_Nl": base_nl,
                "base_Nm": base_nm,
                "tail_fraction": float(args.tail_fraction),
                "convergence_rtol": float(args.convergence_rtol),
                "flux_rel_std_max": float(args.flux_rel_std_max),
                "flux_rel_slope_max": float(args.flux_rel_slope_max),
                "reference_rtol": float(args.reference_rtol),
                "runs": results,
                "reference_metrics": reference_metrics,
                "reference_errors": reference_errors,
                "checks": {
                    "convergence_ok": bool(convergence_ok),
                    "flux_ok": bool(flux_ok),
                    "reference_ok": bool(reference_ok),
                    "overall_ok": bool(convergence_ok and flux_ok and reference_ok),
                },
            }

            summary_path = outdir / "summary.json"
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            print(f"saved {summary_path}")
            print(json.dumps(summary["checks"], indent=2, sort_keys=True))
            if not summary["checks"]["overall_ok"]:
                raise SystemExit(1)
    except KeyboardInterrupt:
        raise SystemExit(130)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())