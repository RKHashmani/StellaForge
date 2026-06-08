"""Log Stage 4 SPECTRAX-GK outputs to Weights & Biases.

This script targets an already-generated Stage 4 output directory such as
`stages/stage4-turbulence/output/HSX_vacuum_ns201_quickrun/`.

It uploads the full output directory as a WandB artifact and logs a compact
set of Stage 4 physics panels:

- final heat / particle flux versus rho,
- growth rate and frequency versus rho,
- per-radius time traces for heat flux, particle flux, gamma, and omega,
- a summary table of the key final diagnostics,
- the existing PNG summary plots when present.

The output directory name is configurable via `--run-name`, which maps to the
directory name under the output root.

While Stage 4 is running, this concurrent version checks the progress log for
new output at a configurable interval and logs only newly appended progress
metrics. After Stage 4 finishes, it logs the completed output artifacts and
summary panels.

Note: This is a trial script to explore the utility of WandB for Stage 4 result tracking.
Unsure about how actually useful it will be.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[3] / "stages" / "stage4-turbulence" / "output"
DEFAULT_PROJECT = "stellaforge-stage4-turbulence"

STEP_LINE_RE = re.compile(
    r"\[spectrax-gk\]\s+step=(?P<step>\d+)/(\d+)\s+progress=\s*(?P<progress>[0-9.]+)%\s+"
    r"t=(?P<t>[-+0-9.eE]+).*?gamma=(?P<gamma>[-+0-9.eE]+)\s+omega=(?P<omega>[-+0-9.eE]+)"
    r".*?Wphi=(?P<Wphi>[-+0-9.eE]+)\s+Wg=(?P<Wg>[-+0-9.eE]+)"
)
CHUNK_DONE_RE = re.compile(
    r"runtime: completed nonlinear chunk (?P<chunk>\d+):\s+t=(?P<t>[-+0-9.eE]+)"
    r"/\d+\s+progress=\s*(?P<progress>[0-9.]+)%.*?chunk_wall=(?P<chunk_wall>\S+)"
)
FINAL_SUMMARY_RE = re.compile(
    r"^nonlinear:\s+t=(?P<t>[-+0-9.eE]+)\s+ky_sel=(?P<ky>[-+0-9.eE]+)\s+kx_sel=(?P<kx>[-+0-9.eE]+)\s+"
    r"dt_mean=(?P<dt>[-+0-9.eE]+)\s+Wg=(?P<Wg>[-+0-9.eE]+)\s+Wphi=(?P<Wphi>[-+0-9.eE]+)\s+Wapar=(?P<Wapar>[-+0-9.eE]+)"
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows: list[dict[str, Any]] = []
        for row in reader:
            parsed: dict[str, Any] = {}
            for key, value in row.items():
                if value is None or value == "":
                    parsed[key] = value
                    continue
                try:
                    parsed[key] = float(value)
                except ValueError:
                    parsed[key] = value
            rows.append(parsed)
    return rows


def _load_last_row(path: Path) -> dict[str, Any]:
    rows = _load_csv_rows(path)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows[-1]


def _find_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return Path(args.output_dir).expanduser().resolve()
    if args.run_name is None:
        raise ValueError("Either --run-name or --output-dir must be provided")
    return (Path(args.output_root).expanduser().resolve() / args.run_name).resolve()


def _default_progress_log_path(output_dir: Path) -> Path:
    return output_dir / f"{output_dir.name}.log"


def _resolve_progress_log(args: argparse.Namespace, output_dir: Path) -> Path:
    if args.progress_log is not None:
        return Path(args.progress_log).expanduser().resolve()
    return _default_progress_log_path(output_dir)


def _import_wandb() -> Any:
    try:
        import wandb
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "wandb is not installed."
        ) from exc
    return wandb


def _parse_tags(value: str | None) -> list[str]:
    if value is None or not value.strip():
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _read_manifest(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing Stage 4 manifest: {path}")
    return _load_json(path)


def _load_runs_csv(output_dir: Path) -> list[dict[str, Any]]:
    """Read `runs.csv` from an output directory if present and return rows.

    This preserves numeric parsing via _load_csv_rows and returns an empty
    list when the file is missing.
    """
    path = output_dir / "runs.csv"
    if not path.exists():
        return []
    return _load_csv_rows(path)


def _merge_runs_csv_into_manifest(manifest: dict[str, Any], csv_rows: list[dict[str, Any]]) -> None:
    """Merge fields from runs.csv rows into manifest['runs'] entries.

    Matching strategy (best-effort): prefer exact `output_prefix` match, then
    fallback to matching by `run_dir` suffix against the manifest `output_prefix`.
    """
    if not csv_rows:
        return
    runs = manifest.get("runs", [])
    for csv_row in csv_rows:
        matched = False
        csv_prefix = csv_row.get("output_prefix")
        csv_rundir = csv_row.get("run_dir")
        for run_spec in runs:
            rp = run_spec.get("output_prefix")
            if csv_prefix and rp == csv_prefix:
                run_spec.update(csv_row)
                matched = True
                break
            if csv_rundir and rp and str(rp).endswith(str(csv_rundir)):
                run_spec.update(csv_row)
                matched = True
                break
        if not matched:
            runs.append(csv_row.copy())
    manifest["runs"] = runs


def _make_runs_detailed_table(wandb: Any, manifest: dict[str, Any]) -> Any:
    """Create a detailed per-run table using all available keys from manifest['runs'].

    This ensures fields coming from `runs.csv` are exposed as individual
    columns for easy inspection in WandB.
    """
    runs = manifest.get("runs", [])
    if not runs:
        return None

    common_keys = [
        "index",
        "rho_index",
        "rho",
        "torflux",
        "Er",
        "run_dir",
        "output_prefix",
        "config_path",
    ]
    keys = []
    for k in common_keys:
        if any(k in r for r in runs) and k not in keys:
            keys.append(k)

    for r in runs:
        for k in r.keys():
            if k not in keys:
                keys.append(k)

    table = wandb.Table(columns=keys)
    for r in runs:
        row = []
        for k in keys:
            v = r.get(k)
            
            if isinstance(v, (int, float)):
                row.append(v)
            else:
                try:
                    row.append(_safe_float(v))
                except Exception:
                    row.append(v)
        table.add_data(*row)
    return table


def _final_row_from_run(run_spec: dict[str, Any]) -> dict[str, Any]:
    summary_path = Path(f"{run_spec['output_prefix']}.summary.json")
    diag_path = Path(f"{run_spec['output_prefix']}.diagnostics.csv")
    final: dict[str, Any] = {}
    if summary_path.exists():
        final.update(_load_json(summary_path))
    if diag_path.exists():
        final.update(_load_last_row(diag_path))
    _canonicalize_final_metrics(final)
    return final


def _canonicalize_final_metrics(final: dict[str, Any]) -> None:
    """Populate canonical `*_last` keys from the raw diagnostics names.

    Stage 4 writes the time-trace CSV with raw column names such as `heat_flux`
    and `particle_flux`, while the summary JSON uses `heat_flux_last` and
    related keys. This helper makes the downstream WandB panels robust to either
    source.
    """

    aliases = {
        "t_last": ("t",),
        "gamma_last": ("gamma",),
        "omega_last": ("omega",),
        "Wg_last": ("Wg",),
        "Wphi_last": ("Wphi",),
        "Wapar_last": ("Wapar",),
        "energy_last": ("energy",),
        "heat_flux_last": ("heat_flux",),
        "particle_flux_last": ("particle_flux",),
    }
    for canonical, candidates in aliases.items():
        if canonical in final:
            continue
        for candidate in candidates:
            if candidate in final:
                final[canonical] = final[candidate]
                break


def _series_from_run(run_spec: dict[str, Any]) -> list[dict[str, Any]]:
    diag_path = Path(f"{run_spec['output_prefix']}.diagnostics.csv")
    if not diag_path.exists():
        return []
    return _load_csv_rows(diag_path)


def _safe_float(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _is_finite_number(value: Any) -> bool:
    try:
        return float(value) == float(value) and abs(float(value)) != float("inf")
    except (TypeError, ValueError):
        return False


def _float_or_nan(value: str | None) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def _load_progress_chunks(log_path: Path) -> list[dict[str, Any]]:
    if not log_path.exists():
        return []

    chunks: list[dict[str, Any]] = []
    current: dict[str, Any] = {"chunk_index": 1, "samples": []}
    current_chunk = 1
    with log_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            sample_match = STEP_LINE_RE.search(line)
            if sample_match:
                sample = {
                    "chunk_index": current_chunk,
                    "step": int(sample_match.group("step")),
                    "progress": _float_or_nan(sample_match.group("progress")),
                    "t": _float_or_nan(sample_match.group("t")),
                    "gamma": _float_or_nan(sample_match.group("gamma")),
                    "omega": _float_or_nan(sample_match.group("omega")),
                    "Wphi": _float_or_nan(sample_match.group("Wphi")),
                    "Wg": _float_or_nan(sample_match.group("Wg")),
                }
                current.setdefault("samples", []).append(sample)
                continue

            chunk_match = CHUNK_DONE_RE.search(line)
            if chunk_match:
                current["chunk_index"] = current_chunk
                current["chunk_complete_t"] = _float_or_nan(chunk_match.group("t"))
                current["chunk_progress"] = _float_or_nan(chunk_match.group("progress"))
                current["chunk_wall"] = chunk_match.group("chunk_wall")
                chunks.append(current)
                current_chunk = int(chunk_match.group("chunk")) + 1
                current = {"chunk_index": current_chunk, "samples": []}
                continue

            final_match = FINAL_SUMMARY_RE.search(line)
            if final_match:
                current["final_summary"] = {
                    "t": _float_or_nan(final_match.group("t")),
                    "ky_sel": _float_or_nan(final_match.group("ky")),
                    "kx_sel": _float_or_nan(final_match.group("kx")),
                    "dt_mean": _float_or_nan(final_match.group("dt")),
                    "Wg": _float_or_nan(final_match.group("Wg")),
                    "Wphi": _float_or_nan(final_match.group("Wphi")),
                    "Wapar": _float_or_nan(final_match.group("Wapar")),
                }

    if current.get("samples") or current.get("final_summary"):
        chunks.append(current)
    return chunks


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", default=None, help="Stage 4 output directory name under the output root.")
    p.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory that contains Stage 4 outputs.",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Optional explicit Stage 4 output directory. Overrides --run-name and --output-root.",
    )
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--entity", default=None)
    p.add_argument("--group", default="stage4-turbulence")
    p.add_argument("--job-type", default="stage4-wandb-log")
    p.add_argument("--name", default=None, help="WandB run name. Defaults to the output directory name.")
    p.add_argument("--mode", default="online", choices=("online", "offline", "dryrun", "disabled"))
    p.add_argument("--tags", default="stage4,turbulence,spectraxgk")
    p.add_argument("--notes", default=None)
    p.add_argument(
        "--include-full-output-dir",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Upload the full Stage 4 output directory as a WandB artifact.",
    )
    p.add_argument(
        "--include-run-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log per-radius diagnostics tables and time-trace panels.",
    )
    p.add_argument(
        "--progress-log",
        default=None,
        help="Optional explicit path to the Stage 4 progress log. Defaults to <output-dir>/<run-name>.log.",
    )
    p.add_argument(
        "--split-progress-log-runs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create separate WandB runs for each 1-100 step chunk found in the progress log.",
    )
    p.add_argument(
        "--max-trace-runs",
        type=int,
        default=8,
        help="Maximum number of per-radius time traces to log.",
    )
    p.add_argument(
        "--poll-interval-minutes",
        type=float,
        default=10.0,
        help="Minutes between checks for newly appended progress-log output (default: 10).",
    )
    p.add_argument(
        "--stop-file",
        default=None,
        help="Optional file whose creation signals that Stage 4 has finished. The file is removed when observed.",
    )
    return p


def _make_summary_table(wandb: Any, manifest: dict[str, Any]) -> Any:
    columns = [
        "rho",
        "rho_index",
        "torflux",
        "Er",
        "rho_star_physical",
        "t_last",
        "gamma_last",
        "omega_last",
        "heat_flux_last",
        "particle_flux_last",
        "Wg_last",
        "Wphi_last",
        "Wapar_last",
    ]
    table = wandb.Table(columns=columns)
    for run_spec in manifest.get("runs", []):
        final = _final_row_from_run(run_spec)
        if not _is_finite_number(final.get("heat_flux_last")) and not _is_finite_number(final.get("particle_flux_last")):
            continue
        table.add_data(
            _safe_float(run_spec.get("rho")),
            int(run_spec.get("rho_index", -1)),
            _safe_float(run_spec.get("torflux")),
            _safe_float(run_spec.get("Er")),
            _safe_float(run_spec.get("rho_star_physical")),
            _safe_float(final.get("t_last")),
            _safe_float(final.get("gamma_last")),
            _safe_float(final.get("omega_last")),
            _safe_float(final.get("heat_flux_last")),
            _safe_float(final.get("particle_flux_last")),
            _safe_float(final.get("Wg_last")),
            _safe_float(final.get("Wphi_last")),
            _safe_float(final.get("Wapar_last")),
        )
    return table


def _make_final_flux_table(wandb: Any, manifest: dict[str, Any]) -> Any:
    columns = ["rho", "gamma_last", "omega_last", "heat_flux_last", "particle_flux_last"]
    species_names = list(manifest.get("runtime_species_names", []))
    columns.extend([f"heat_flux_{name}" for name in species_names])
    columns.extend([f"particle_flux_{name}" for name in species_names])
    table = wandb.Table(columns=columns)
    for run_spec in manifest.get("runs", []):
        final = _final_row_from_run(run_spec)
        if not _is_finite_number(final.get("heat_flux_last")) and not _is_finite_number(final.get("particle_flux_last")):
            continue
        row: list[Any] = [
            _safe_float(run_spec.get("rho")),
            _safe_float(final.get("gamma_last")),
            _safe_float(final.get("omega_last")),
            _safe_float(final.get("heat_flux_last")),
            _safe_float(final.get("particle_flux_last")),
        ]
        for idx, _name in enumerate(species_names):
            row.append(_safe_float(final.get(f"heat_flux_s{idx}")))
        for idx, _name in enumerate(species_names):
            row.append(_safe_float(final.get(f"particle_flux_s{idx}")))
        table.add_data(*row)
    return table


def _log_summary_panels(wandb: Any, manifest: dict[str, Any]) -> None:
    summary_table = _make_summary_table(wandb, manifest)
    flux_table = _make_final_flux_table(wandb, manifest)

    if summary_table.data:
        wandb.log(
            {
                "stage4/summary_table": summary_table,
                "stage4/final_flux_table": flux_table,
                "stage4/final/gamma_vs_rho": wandb.plot.line(summary_table, "rho", "gamma_last", title="Gamma vs rho"),
                "stage4/final/omega_vs_rho": wandb.plot.line(summary_table, "rho", "omega_last", title="Omega vs rho"),
                "stage4/final/heat_flux_vs_rho": wandb.plot.line(
                    summary_table, "rho", "heat_flux_last", title="Heat flux vs rho"
                ),
                "stage4/final/particle_flux_vs_rho": wandb.plot.line(
                    summary_table, "rho", "particle_flux_last", title="Particle flux vs rho"
                ),
            }
        )


def _log_time_trace_panels(wandb: Any, manifest: dict[str, Any], *, max_trace_runs: int) -> None:
    trace_keys = ["t", "gamma", "omega", "heat_flux", "particle_flux", "Wg", "Wphi", "Wapar", "energy"]
    for run_spec in manifest.get("runs", [])[: max(0, max_trace_runs)]:
        rho = _safe_float(run_spec.get("rho"))
        series = _series_from_run(run_spec)
        if not series:
            continue
        columns = [key for key in trace_keys if any(key in row for row in series)]
        if not columns:
            continue
        table = wandb.Table(columns=columns)
        for row in series:
            table.add_data(*[_safe_float(row.get(key)) for key in columns])
        prefix = f"stage4/time_trace/rho_{rho:.4f}".replace(".", "p")
        plots: dict[str, Any] = {f"{prefix}/table": table}
        for metric in ("gamma", "omega", "heat_flux", "particle_flux"):
            if metric in columns:
                plots[f"{prefix}/{metric}_vs_t"] = wandb.plot.line(
                    table, "t", metric, title=f"{metric} vs t (rho={rho:.4f})"
                )
        wandb.log(plots)


def _log_progress_chunk_run(
    *,
    wandb: Any,
    output_dir: Path,
    base_run_name: str,
    chunk: dict[str, Any],
    manifest: dict[str, Any],
    project: str,
    entity: str | None,
    mode: str,
) -> None:
    samples = list(chunk.get("samples", []))
    if not samples:
        return

    columns = ["step", "progress", "t", "gamma", "omega", "Wphi", "Wg"]
    table = wandb.Table(columns=columns)
    for sample in samples:
        table.add_data(
            int(sample.get("step", 0)),
            _safe_float(sample.get("progress")),
            _safe_float(sample.get("t")),
            _safe_float(sample.get("gamma")),
            _safe_float(sample.get("omega")),
            _safe_float(sample.get("Wphi")),
            _safe_float(sample.get("Wg")),
        )

    chunk_index = int(chunk.get("chunk_index", 1))
    run_name = f"{base_run_name}-chunk-{chunk_index:02d}"
    summary = dict(chunk.get("final_summary", {}))
    last_sample = samples[-1]
    summary.setdefault("t", last_sample.get("t"))
    summary.setdefault("gamma", last_sample.get("gamma"))
    summary.setdefault("omega", last_sample.get("omega"))
    summary.setdefault("Wphi", last_sample.get("Wphi"))
    summary.setdefault("Wg", last_sample.get("Wg"))

    with wandb.init(
        project=project,
        entity=entity,
        group=f"stage4-progress/{output_dir.name}",
        job_type="stage4-progress-chunk",
        name=run_name,
        config={
            "output_dir": str(output_dir),
            "chunk_index": chunk_index,
            "n_samples": len(samples),
            "base_run_name": base_run_name,
            "manifest_runs": len(manifest.get("runs", [])),
        },
        mode=mode,
        reinit=True,
    ):
        wandb.run.summary["chunk_index"] = chunk_index
        wandb.run.summary["n_samples"] = len(samples)
        for key, value in summary.items():
            wandb.run.summary[f"final/{key}"] = value

        wandb.log(
            {
                f"stage4/chunk_{chunk_index:02d}/step_trace": table,
                f"stage4/chunk_{chunk_index:02d}/gamma_vs_step": wandb.plot.line(
                    table, "step", "gamma", title=f"gamma vs step (chunk {chunk_index:02d})"
                ),
                f"stage4/chunk_{chunk_index:02d}/omega_vs_step": wandb.plot.line(
                    table, "step", "omega", title=f"omega vs step (chunk {chunk_index:02d})"
                ),
                f"stage4/chunk_{chunk_index:02d}/Wphi_vs_step": wandb.plot.line(
                    table, "step", "Wphi", title=f"Wphi vs step (chunk {chunk_index:02d})"
                ),
                f"stage4/chunk_{chunk_index:02d}/Wg_vs_step": wandb.plot.line(
                    table, "step", "Wg", title=f"Wg vs step (chunk {chunk_index:02d})"
                ),
                f"stage4/chunk_{chunk_index:02d}/step_progress": wandb.plot.line(
                    table, "step", "progress", title=f"progress vs step (chunk {chunk_index:02d})"
                ),
            }
        )


def _log_existing_pngs(wandb: Any, output_dir: Path) -> None:
    images = {}
    for png_name in ("Gamma_vs_rho.png", "Q_vs_rho.png"):
        path = output_dir / png_name
        if path.exists():
            images[f"stage4/images/{path.stem}"] = wandb.Image(str(path), caption=path.name)
    if images:
        wandb.log(images)


def _log_artifact(wandb: Any, output_dir: Path, manifest: dict[str, Any]) -> None:
    artifact = wandb.Artifact(
        name=f"stage4-output-{output_dir.name}",
        type="stage4-output",
        description="Stage 4 SPECTRAX-GK output directory",
        metadata={
            "output_dir": str(output_dir),
            "electron_model": str(manifest.get("electron_model", "")),
            "n_runs": len(manifest.get("runs", [])),
            "project": DEFAULT_PROJECT,
        },
    )
    artifact.add_dir(str(output_dir))
    wandb.log_artifact(artifact)


def _read_new_lines(log_path: Path, offset: int) -> tuple[list[str], int]:
    """Read complete newly appended lines from a growing log."""
    if not log_path.exists():
        return [], 0 if offset else offset

    size = log_path.stat().st_size
    if size < offset:
        offset = 0
    with log_path.open("rb") as fh:
        fh.seek(offset)
        appended = fh.read()
    complete_end = appended.rfind(b"\n") + 1
    if complete_end == 0:
        return [], offset
    complete = appended[:complete_end].decode("utf-8", errors="replace")
    return complete.splitlines(keepends=True), offset + complete_end


def _log_new_progress_lines(wandb: Any, lines: list[str], *, chunk_index: int) -> int:
    """Log progress metrics parsed from newly appended Stage 4 log lines."""
    for line in lines:
        sample_match = STEP_LINE_RE.search(line)
        if sample_match:
            wandb.log(
                {
                    "stage4/live/chunk_index": chunk_index,
                    "stage4/live/step": int(sample_match.group("step")),
                    "stage4/live/progress": _float_or_nan(sample_match.group("progress")),
                    "stage4/live/t": _float_or_nan(sample_match.group("t")),
                    "stage4/live/gamma": _float_or_nan(sample_match.group("gamma")),
                    "stage4/live/omega": _float_or_nan(sample_match.group("omega")),
                    "stage4/live/Wphi": _float_or_nan(sample_match.group("Wphi")),
                    "stage4/live/Wg": _float_or_nan(sample_match.group("Wg")),
                }
            )
            continue

        chunk_match = CHUNK_DONE_RE.search(line)
        if chunk_match:
            wandb.log(
                {
                    "stage4/live/completed_chunk": int(chunk_match.group("chunk")),
                    "stage4/live/chunk_progress": _float_or_nan(chunk_match.group("progress")),
                    "stage4/live/chunk_t": _float_or_nan(chunk_match.group("t")),
                }
            )
            chunk_index = int(chunk_match.group("chunk")) + 1
            continue

        final_match = FINAL_SUMMARY_RE.search(line)
        if final_match:
            wandb.log(
                {
                    "stage4/live/final/t": _float_or_nan(final_match.group("t")),
                    "stage4/live/final/ky_sel": _float_or_nan(final_match.group("ky")),
                    "stage4/live/final/kx_sel": _float_or_nan(final_match.group("kx")),
                    "stage4/live/final/dt_mean": _float_or_nan(final_match.group("dt")),
                    "stage4/live/final/Wg": _float_or_nan(final_match.group("Wg")),
                    "stage4/live/final/Wphi": _float_or_nan(final_match.group("Wphi")),
                    "stage4/live/final/Wapar": _float_or_nan(final_match.group("Wapar")),
                }
            )
    return chunk_index


def _wait_for_next_check(interval_seconds: float, stop_file: Path | None) -> bool:
    """Wait for the next poll, returning early when the stop file appears."""
    deadline = time.monotonic() + interval_seconds
    while time.monotonic() < deadline:
        if stop_file is not None and stop_file.exists():
            return True
        time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
    return stop_file is not None and stop_file.exists()


def _log_completed_outputs(
    wandb: Any,
    output_dir: Path,
    *,
    include_full_output_dir: bool,
    include_run_diagnostics: bool,
    max_trace_runs: int,
) -> dict[str, Any] | None:
    """Log completed Stage 4 files when a manifest is available."""
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        return None

    manifest = _read_manifest(output_dir)
    csv_rows = _load_runs_csv(output_dir)
    if csv_rows:
        _merge_runs_csv_into_manifest(manifest, csv_rows)
        cols = list(dict(csv_rows[0]).keys())
        runs_tbl = wandb.Table(columns=cols)
        for row in csv_rows:
            runs_tbl.add_data(*[row.get(column) for column in cols])
        wandb.log({"stage4/runs_csv": runs_tbl})

    detailed = _make_runs_detailed_table(wandb, manifest)
    if detailed is not None:
        wandb.log({"stage4/runs_detailed": detailed})
    _log_summary_panels(wandb, manifest)
    if include_run_diagnostics:
        _log_time_trace_panels(wandb, manifest, max_trace_runs=max_trace_runs)
    _log_existing_pngs(wandb, output_dir)
    if include_full_output_dir:
        _log_artifact(wandb, output_dir, manifest)
    return manifest


def main() -> int:
    args = build_parser().parse_args()
    output_dir = _find_output_dir(args)
    if args.poll_interval_minutes <= 0:
        raise SystemExit("--poll-interval-minutes must be greater than zero")

    wandb = _import_wandb()
    run_name = args.name or output_dir.name
    progress_log = _resolve_progress_log(args, output_dir)
    stop_file = Path(args.stop_file).expanduser().resolve() if args.stop_file else None
    interval_seconds = float(args.poll_interval_minutes) * 60.0
    config = {
        "output_dir": str(output_dir),
        "output_root": str(Path(args.output_root).expanduser().resolve()),
        "run_name": run_name,
        "progress_log": str(progress_log),
        "poll_interval_minutes": float(args.poll_interval_minutes),
    }

    with wandb.init(
        project=args.project,
        entity=args.entity,
        group=args.group,
        job_type=args.job_type,
        name=run_name,
        notes=args.notes,
        tags=_parse_tags(args.tags),
        config=config,
        mode=args.mode,
    ):
        wandb.run.summary["output_dir"] = str(output_dir)
        wandb.run.summary["progress_log"] = str(progress_log)
        offset = 0
        chunk_index = 1
        while True:
            lines, offset = _read_new_lines(progress_log, offset)
            if lines:
                chunk_index = _log_new_progress_lines(wandb, lines, chunk_index=chunk_index)
                wandb.run.summary["last_progress_log_offset"] = offset
            if stop_file is not None and stop_file.exists():
                stop_file.unlink(missing_ok=True)
                break
            _wait_for_next_check(interval_seconds, stop_file)

        manifest = _log_completed_outputs(
            wandb,
            output_dir,
            include_full_output_dir=bool(args.include_full_output_dir),
            include_run_diagnostics=bool(args.include_run_diagnostics),
            max_trace_runs=int(args.max_trace_runs),
        )
        if manifest is not None:
            wandb.run.summary["manifest"] = str(output_dir / "manifest.json")
            wandb.run.summary["n_runs"] = len(manifest.get("runs", []))

    chunks = _load_progress_chunks(progress_log)
    if bool(args.split_progress_log_runs) and chunks and manifest is not None:
        for chunk in chunks:
            _log_progress_chunk_run(
                wandb=wandb,
                output_dir=output_dir,
                base_run_name=run_name,
                chunk=chunk,
                manifest=manifest,
                project=args.project,
                entity=args.entity,
                mode=args.mode,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
