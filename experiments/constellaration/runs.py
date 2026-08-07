"""Generate and launch driftless-star runs from the ConStellaration dataset.

The generator uses Hugging Face's Dataset Viewer REST API so rows are fetched in
small pages instead of downloading the full dataset.  Each selected
``plasma_config_id`` becomes a self-contained pipeline input directory whose
resolution and profile templates come from ``inputs/quick_run``.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib import error, parse, request

import yaml

from src.utils.config_edit import apply_assignments

logger = logging.getLogger(__name__)

DEFAULT_DATASET = "proxima-fusion/constellaration"
DEFAULT_OUTPUT_ROOT = Path("/staging/groups/driftless_star/constellaration_runs")
DATASET_API = "https://datasets-server.huggingface.co"
PAGE_SIZE = 100
DEFAULT_BATCH_SIZE = 100
DEFAULT_PARALLEL_RUNS = 10
DEFAULT_VMEC_NITER = 10_000
SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9._-]+$")
TRANSIENT_HTTP_CODES = {429, 500, 502, 503, 504}


@dataclass(frozen=True)
class GenerationSpec:
    """Dataset coordinates and physics setting recorded for generated runs."""

    dataset: str
    config_name: str
    split: str
    phiedge: float


def _logical_absolute(path: Path) -> Path:
    """Make a path absolute without resolving shared-filesystem symlinks.

    HTCondor must see the logical ``/staging/...`` spelling. ``Path.resolve``
    turns it into the backing ``/mnt/htc-cephfs/...`` path on submit hosts,
    which the scheduler then mistakes for a file-transfer source.
    """
    path = path.expanduser()
    return path if path.is_absolute() else Path.cwd() / path


def _now() -> str:
    """Return a stable UTC timestamp for the run manifest."""
    return datetime.now(UTC).isoformat()


def _request_json(endpoint: str, parameters: dict[str, str | int], token: str | None) -> dict[str, Any]:
    """Fetch one JSON response from the Hugging Face Dataset Viewer API."""
    url = f"{DATASET_API}/{endpoint}?{parse.urlencode(parameters)}"
    headers = {"User-Agent": "driftless-star-constellaration-generator/1"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    for attempt in range(6):
        try:
            with request.urlopen(request.Request(url, headers=headers), timeout=60) as response:
                return json.load(response)
        except error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:500]
            if exc.code not in TRANSIENT_HTTP_CODES or attempt == 5:
                raise RuntimeError(f"Hugging Face dataset API returned HTTP {exc.code}: {detail}") from exc
            delay = min(2**attempt, 10)
            logger.warning("Dataset API HTTP %d; retrying in %d seconds (%s)", exc.code, delay, detail)
        except error.URLError as exc:
            if attempt == 5:
                raise RuntimeError(f"Could not reach the Hugging Face dataset API: {exc.reason}") from exc
            delay = min(2**attempt, 10)
            logger.warning("Dataset API connection failed; retrying in %d seconds (%s)", delay, exc.reason)
        time.sleep(delay)
    raise RuntimeError("Unreachable: dataset API retry loop exhausted")


def _validate_run_id(run_id: str) -> str:
    """Validate an ID before using it as a run name or directory component."""
    if not SAFE_RUN_ID.fullmatch(run_id) or run_id in {".", ".."}:
        raise ValueError(
            f"Unsafe plasma_config_id {run_id!r}; IDs may contain only letters, digits, '.', '_', and '-'."
        )
    return run_id


def _unwrap_api_row(wrapper: dict[str, Any]) -> dict[str, Any]:
    """Return a dataset row and retain its Dataset Viewer row index."""
    truncated = set(wrapper.get("truncated_cells") or [])
    required = {"boundary.r_cos", "boundary.z_sin", "plasma_config_id"}
    if truncated & required:
        raise ValueError(f"Dataset API truncated required cells: {sorted(truncated & required)}")
    row = dict(wrapper["row"])
    row["_dataset_row_idx"] = wrapper.get("row_idx")
    return row


def fetch_dataset_rows(
    *,
    dataset: str,
    config_name: str,
    split: str,
    offset: int,
    limit: int,
    ids: Sequence[str] = (),
    token: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch selected dataset rows without downloading the dataset.

    Parameters
    ----------
    dataset, config_name, split : str
        Hugging Face dataset coordinates.
    offset, limit : int
        Contiguous row slice used when ``ids`` is empty.
    ids : sequence of str, optional
        Exact ``plasma_config_id`` values. Each ID is queried through the
        Dataset Viewer ``/filter`` endpoint.
    token : str, optional
        Hugging Face token for private/gated datasets. The default public
        dataset does not require one.

    Returns
    -------
    list of dict
        Rows containing boundary Fourier coefficients and provenance fields.
    """
    if offset < 0:
        raise ValueError(f"offset must be >= 0, got {offset}")
    if limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit}")

    if ids:
        rows: list[dict[str, Any]] = []
        for requested_id in dict.fromkeys(_validate_run_id(value) for value in ids):
            payload = _request_json(
                "filter",
                {
                    "dataset": dataset,
                    "config": config_name,
                    "split": split,
                    "where": f'"plasma_config_id"=\'{requested_id}\'',
                    "offset": 0,
                    "length": 2,
                },
                token,
            )
            matches = payload.get("rows") or []
            if len(matches) != 1:
                raise ValueError(
                    f"Expected exactly one row for plasma_config_id {requested_id!r}, found {len(matches)}"
                )
            rows.append(_unwrap_api_row(matches[0]))
        return rows

    rows = []
    while len(rows) < limit:
        length = min(PAGE_SIZE, limit - len(rows))
        payload = _request_json(
            "rows",
            {
                "dataset": dataset,
                "config": config_name,
                "split": split,
                "offset": offset + len(rows),
                "length": length,
            },
            token,
        )
        page = payload.get("rows") or []
        rows.extend(_unwrap_api_row(row) for row in page)
        if len(page) < length:
            break
    if len(rows) != limit:
        raise ValueError(f"Requested {limit} rows at offset {offset}, but the dataset returned {len(rows)}")
    return rows


def fetch_dataset_row_count(
    *, dataset: str, config_name: str, split: str, token: str | None = None
) -> int:
    """Return the Dataset Viewer row count for one config and split."""
    payload = _request_json(
        "rows",
        {
            "dataset": dataset,
            "config": config_name,
            "split": split,
            "offset": 0,
            "length": 1,
        },
        token,
    )
    total = payload.get("num_rows_total")
    if not isinstance(total, int) or total < 0:
        raise RuntimeError(f"Dataset Viewer returned an invalid num_rows_total: {total!r}")
    return total


def _coefficient_matrix(row: dict[str, Any], key: str) -> list[list[float]]:
    """Validate one rectangular, finite Fourier coefficient matrix."""
    raw = row.get(key)
    if not isinstance(raw, list) or not raw or not all(isinstance(values, list) and values for values in raw):
        raise ValueError(f"{key} must be a non-empty two-dimensional list")
    width = len(raw[0])
    if width % 2 != 1 or any(len(values) != width for values in raw):
        raise ValueError(f"{key} must be rectangular with an odd toroidal dimension")
    matrix = [[float(value) for value in values] for values in raw]
    if not all(math.isfinite(value) for values in matrix for value in values):
        raise ValueError(f"{key} contains a non-finite coefficient")
    return matrix


def build_vmec_input(row: dict[str, Any], *, phiedge: float) -> str:
    """Convert one ConStellaration boundary to a quick-run VMEC INDATA namelist.

    Parameters
    ----------
    row : dict
        Dataset row with ``boundary.r_cos``, ``boundary.z_sin``, field-period,
        and stellarator-symmetry fields.
    phiedge : float
        Edge toroidal flux in Webers, copied from the shipped quick run by
        default.

    Returns
    -------
    str
        VMEC namelist using the dataset's native ``(m, n)`` Fourier ordering.
    """
    if not row.get("boundary.is_stellarator_symmetric", False):
        raise ValueError("The driftless-star quick pipeline currently requires a stellarator-symmetric boundary")
    if not math.isfinite(phiedge) or phiedge == 0:
        raise ValueError(f"phiedge must be finite and nonzero, got {phiedge}")

    r_cos = _coefficient_matrix(row, "boundary.r_cos")
    z_sin = _coefficient_matrix(row, "boundary.z_sin")
    if (len(r_cos), len(r_cos[0])) != (len(z_sin), len(z_sin[0])):
        raise ValueError("boundary.r_cos and boundary.z_sin must have the same shape")
    nfp = int(row["boundary.n_field_periods"])
    if nfp < 1:
        raise ValueError(f"boundary.n_field_periods must be >= 1, got {nfp}")

    mpol = len(r_cos)
    ntor = (len(r_cos[0]) - 1) // 2
    lines = [
        "! Generated from proxima-fusion/constellaration by driftless-star.",
        "&INDATA",
        "  MGRID_FILE = 'NONE'",
        "  LFREEB = F",
        "  LASYM = F",
        "  LOLDOUT = T",
        "  LWOUTTXT = T",
        f"  NFP = {nfp}",
        "  NCURR = 1",
        f"  MPOL = {mpol}",
        f"  NTOR = {ntor}",
        "  NS_ARRAY = 11",
        f"  NITER_ARRAY = {DEFAULT_VMEC_NITER}",
        "  FTOL_ARRAY = 1.0e-4",
        "  NSTEP = 200",
        "  NVACSKIP = 6",
        "  GAMMA = 0.0",
        f"  PHIEDGE = {phiedge:.16g}",
        "  BLOAT = 1.0",
        "  CURTOR = 0.0",
        "  AM = 0.0",
        "  AC = 0.0",
        "  RAXIS_CC = 0.0",
        "  ZAXIS_CS = 0.0",
        "",
    ]
    for m, (r_values, z_values) in enumerate(zip(r_cos, z_sin, strict=True)):
        for n in range(-ntor, ntor + 1):
            index = n + ntor
            lines.append(
                f"  RBC({n:3d},{m:3d}) = {r_values[index]: .16e}, "
                f"ZBS({n:3d},{m:3d}) = {z_values[index]: .16e}"
            )
    lines.extend(["/", ""])
    return "\n".join(lines)


def _template_paths(template_dir: Path) -> dict[str, Path]:
    """Resolve and validate the four shipped quick-run templates."""
    config_path = template_dir / "config.yaml"
    config = yaml.safe_load(config_path.read_text())
    run_name = config["run_name"]
    files = config["filenames"]
    paths = {
        "config": config_path,
        "s3": template_dir / files["s3_config"].format(run_name=run_name),
        "s4": template_dir / files["s4_config"].format(run_name=run_name),
        "s5": template_dir / files["s5_config"].format(run_name=run_name),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Quick-run template files do not exist: {missing}")
    return paths


def _render_template_inputs(
    config: dict[str, Any], templates: dict[str, Path], run_id: str, output_dir: Path
) -> dict[str, str]:
    """Render Stage 3/4/5 templates with paths for one generated run."""
    paths = config["filenames"]
    s1_output = output_dir / "stage1_equilibrium" / paths["s1_output"].format(run_name=run_id)
    s2_output = output_dir / "stage2_boozer" / paths["s2_output"].format(run_name=run_id)
    s3_output = output_dir / "stage3_neoclassical" / paths["s3_output"].format(run_name=run_id)
    s4_output = output_dir / "stage4_turbulence" / paths["s4_output"].format(run_name=run_id)
    s5_output_dir = output_dir / "stage5_transport"
    return {
        paths["s3_config"].format(run_name=run_id): apply_assignments(
            templates["s3"].read_text(), {"equilibriumFile": json.dumps(str(s1_output))}
        ),
        paths["s4_config"].format(run_name=run_id): apply_assignments(
            templates["s4"].read_text(),
            {
                "vmec_file": json.dumps(str(s1_output)),
                "geometry_file": json.dumps(str(output_dir / "stage4_turbulence" / f"wout_{run_id}.eik.nc")),
            },
        ),
        paths["s5_config"].format(run_name=run_id): apply_assignments(
            templates["s5"].read_text(),
            {
                "vmec_file": json.dumps(str(s1_output)),
                "boozer_file": json.dumps(str(s2_output)),
                "neoclassical_file": json.dumps(str(s3_output)),
                "turbulence_file": json.dumps(str(s4_output)),
                "transport_output_dir": json.dumps(f"{s5_output_dir}/"),
            },
        ),
    }


def materialize_run(
    row: dict[str, Any],
    *,
    output_root: Path,
    template_dir: Path,
    spec: GenerationSpec,
    overwrite: bool,
) -> Path:
    """Write one self-contained run folder and return its config path."""
    run_id = _validate_run_id(str(row["plasma_config_id"]))
    input_dir = output_root / "inputs" / run_id
    output_dir = output_root / "outputs" / run_id
    config_path = input_dir / "config.yaml"
    if config_path.exists() and not overwrite:
        raise FileExistsError(f"Run {run_id!r} already exists at {input_dir}; pass --overwrite to replace its inputs")

    templates = _template_paths(template_dir)
    config = yaml.safe_load(templates["config"].read_text())
    config["run_name"] = run_id
    config["input_dir"] = str(input_dir)
    config["output_dir"] = str(output_dir)
    config["dataset_source"] = {
        "repository": spec.dataset,
        "config": spec.config_name,
        "split": spec.split,
        "row_index": row.get("_dataset_row_idx"),
        "plasma_config_id": run_id,
    }

    paths = config["filenames"]
    s1_name = paths["s1_input"].format(run_name=run_id)
    rendered_templates = _render_template_inputs(config, templates, run_id, output_dir)
    provenance = {
        "dataset": spec.dataset,
        "config": spec.config_name,
        "split": spec.split,
        "row_index": row.get("_dataset_row_idx"),
        "plasma_config_id": run_id,
        "row": {key: value for key, value in row.items() if not key.startswith("_dataset_")},
    }

    input_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "# Generated ConStellaration quick run. Runtime paths are absolute so this folder can live outside the repo.\n"
        + yaml.safe_dump(config, sort_keys=False)
    )
    (input_dir / s1_name).write_text(build_vmec_input(row, phiedge=spec.phiedge))
    for filename, text in rendered_templates.items():
        (input_dir / filename).write_text(text)
    (input_dir / "dataset_row.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return config_path


def generate_runs(args: argparse.Namespace) -> list[Path]:
    """Fetch rows selected by CLI arguments and materialize their run folders."""
    repo_root = Path(__file__).resolve().parents[2]
    output_root = _logical_absolute(args.output_root)
    template_dir = args.template_dir
    if not template_dir.is_absolute():
        template_dir = repo_root / template_dir
    ids = list(args.id)
    if args.ids_file:
        ids.extend(
            line.strip()
            for line in args.ids_file.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    rows = fetch_dataset_rows(
        dataset=args.dataset,
        config_name=args.dataset_config,
        split=args.split,
        offset=args.offset,
        limit=args.limit,
        ids=ids,
        token=os.environ.get("HF_TOKEN"),
    )
    spec = GenerationSpec(args.dataset, args.dataset_config, args.split, args.phiedge)
    configs = [
        materialize_run(
            row,
            output_root=output_root,
            template_dir=template_dir,
            spec=spec,
            overwrite=args.overwrite,
        )
        for row in rows
    ]
    for config_path in configs:
        logger.info("Generated %s", config_path)
    return configs


def discover_configs(output_root: Path, ids: Sequence[str] = ()) -> list[Path]:
    """Discover generated configs, optionally restricting the exact run IDs."""
    inputs = _logical_absolute(output_root) / "inputs"
    if ids:
        configs = [inputs / _validate_run_id(run_id) / "config.yaml" for run_id in dict.fromkeys(ids)]
    else:
        configs = sorted(inputs.glob("*/config.yaml"))
    missing = [str(path) for path in configs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Generated run configs do not exist: {missing}")
    if not configs:
        raise FileNotFoundError(f"No generated run configs found under {inputs}")
    return configs


def _htcondor_log_paths(config_path: Path, repo_root: Path) -> tuple[Path, Path]:
    """Return the live home jobdir and its staging archive directory."""
    run_id = config_path.parent.name
    output_root = config_path.parent.parent.parent
    live_jobdir = repo_root / "jobs" / "constellaration" / run_id
    archived_jobdir = output_root / "outputs" / run_id / "htcondor"
    return live_jobdir, archived_jobdir


def _collect_htcondor_logs(config_path: Path, repo_root: Path) -> Path | None:
    """Move one inactive controller's home logs into its staging output tree."""
    live_jobdir, archived_jobdir = _htcondor_log_paths(config_path, repo_root)
    if not live_jobdir.exists():
        return None
    if live_jobdir.is_symlink():
        raise RuntimeError(f"Refusing to collect symlinked HTCondor job directory: {live_jobdir}")

    archived_jobdir.mkdir(parents=True, exist_ok=True)
    attempt = 1
    destination = archived_jobdir / f"attempt_{attempt}"
    while destination.exists():
        attempt += 1
        destination = archived_jobdir / f"attempt_{attempt}"
    shutil.move(str(live_jobdir), str(destination))
    logger.info("Collected HTCondor controller logs in %s", destination)
    return destination


def _launch_command(
    config_path: Path,
    args: argparse.Namespace,
    repo_root: Path | None = None,
) -> list[str]:
    """Build the forward-pass or closed-loop command for one config."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]
    loop_iters = args.loop_iters
    profile = getattr(args, "profile", None)
    container_runtime = getattr(args, "container_runtime", None)
    gpu_ids = getattr(args, "gpu_ids", None)
    jobs_per_gpu = getattr(args, "jobs_per_gpu", None)
    nolock = getattr(args, "nolock", False)
    is_htcondor = bool(profile and "htcondor" in str(profile).lower())
    htcondor_jobdir = None
    if is_htcondor:
        # CHTC requires the live event log to be under /home, not /staging.
        # Each run still receives a distinct directory; _launch_config moves
        # it into the staging output tree after the controller exits.
        htcondor_jobdir, _ = _htcondor_log_paths(config_path, repo_root)

    if loop_iters:
        command = [
            sys.executable,
            "-m",
            "src.ouroboros",
            "--config",
            str(config_path),
            "--max-iters",
            str(loop_iters),
            "--cores",
            str(args.cores),
        ]
        if profile:
            command.extend(["--profile", profile])
        if htcondor_jobdir:
            command.extend(["--htcondor-jobdir", str(htcondor_jobdir)])
        if container_runtime:
            command.extend(["--container-runtime", container_runtime])
        if gpu_ids is not None:
            command.extend(["--gpu-ids", str(gpu_ids)])
        if jobs_per_gpu is not None:
            command.extend(["--jobs-per-gpu", str(jobs_per_gpu)])
        if nolock:
            command.append("--nolock")
        return command

    snakemake = (
        [sys.executable, "-m", "src.snakemake_htcondor"]
        if is_htcondor
        else ["snakemake"]
    )
    command = [*snakemake]
    if args.dry_run:
        command.append("--dry-run")
    if nolock:
        command.append("--nolock")
    if profile:
        command.extend(["--profile", profile])
    if htcondor_jobdir:
        command.extend(["--htcondor-jobdir", str(htcondor_jobdir)])
    command.extend(["--cores", str(args.cores), "--configfile", str(config_path)])
    overrides = []
    if container_runtime:
        overrides.append(f"container_runtime={container_runtime}")
    if gpu_ids is not None:
        overrides.append(f"gpu_ids={gpu_ids}")
    if jobs_per_gpu is not None:
        overrides.append(f"jobs_per_gpu={jobs_per_gpu}")
    if overrides:
        command.extend(["--config", *overrides])
    return command


def _launch_config(config_path: Path, args: argparse.Namespace, repo_root: Path) -> int:
    """Launch one config and return its process exit code."""
    is_htcondor = bool(getattr(args, "profile", None) and "htcondor" in str(args.profile).lower())
    launch_env = None
    if is_htcondor:
        # Preserve logs left by an interrupted earlier controller and ensure
        # the new plugin instance starts from a fresh unified event log.
        _collect_htcondor_logs(config_path, repo_root)
        launch_env = os.environ.copy()
        launch_env["DRIFTLESS_STAR_RUN_ID"] = _validate_run_id(config_path.parent.name)
    try:
        result = subprocess.run(
            _launch_command(config_path, args, repo_root),
            cwd=repo_root,
            check=False,
            env=launch_env,
        )
        return result.returncode
    finally:
        if is_htcondor:
            _collect_htcondor_logs(config_path, repo_root)


def launch_runs(args: argparse.Namespace) -> int:
    """Launch every selected generated config sequentially through Snakemake."""
    if args.cores < 1:
        raise ValueError(f"cores must be >= 1, got {args.cores}")
    if args.loop_iters < 0:
        raise ValueError(f"loop-iters must be >= 0, got {args.loop_iters}")
    if args.dry_run and args.loop_iters:
        raise ValueError("--dry-run cannot be combined with --loop-iters; use a forward dry-run first")
    repo_root = Path(__file__).resolve().parents[2]
    configs = discover_configs(args.output_root, args.id)
    failures: list[Path] = []
    for index, config_path in enumerate(configs, start=1):
        run_id = config_path.parent.name
        logger.info("Launching run %d/%d: %s", index, len(configs), run_id)
        returncode = _launch_config(config_path, args, repo_root)
        if returncode:
            failures.append(config_path)
            logger.error("Run %s failed with exit code %d", run_id, returncode)
            if not args.keep_going:
                break
    if failures:
        logger.error("%d run(s) failed: %s", len(failures), ", ".join(path.parent.name for path in failures))
        return 1
    logger.info("Completed %d run(s)", len(configs))
    return 0


@contextmanager
def _batch_lock(output_root: Path):
    """Prevent two batch drivers from mutating the same manifest and folders."""
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".batch.lock"
    with lock_path.open("a+") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another batch driver is already using {output_root}") from exc
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Atomically persist the batch manifest."""
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _load_manifest(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Load a manifest or initialize one for this dataset selection."""
    coordinates = {
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "split": args.split,
    }
    if not path.exists():
        return {
            "schema_version": 1,
            **coordinates,
            "next_offset": 0,
            "created_at": _now(),
            "updated_at": _now(),
            "runs": {},
            "batches": [],
        }
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported manifest schema in {path}")
    mismatches = [
        key for key, expected in coordinates.items() if manifest.get(key) != expected
    ]
    if mismatches:
        details = ", ".join(
            f"{key}: manifest={manifest.get(key)!r}, requested={coordinates[key]!r}"
            for key in mismatches
        )
        raise ValueError(f"Dataset selection does not match the existing manifest ({details})")
    return manifest


def _save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = _now()
    _write_manifest(path, manifest)


def _requested_ids(args: argparse.Namespace) -> list[str]:
    """Collect and deduplicate IDs supplied directly and through a text file."""
    ids = list(args.id)
    if args.ids_file:
        ids.extend(
            line.strip()
            for line in args.ids_file.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    return list(dict.fromkeys(_validate_run_id(run_id) for run_id in ids))


def _create_batch(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    manifest_path: Path,
    output_root: Path,
    count_override: int | None = None,
) -> dict[str, Any]:
    """Fetch, materialize, and register one new batch."""
    ids = _requested_ids(args)
    requested_count = len(ids) if ids else (count_override if count_override is not None else args.batch_size)
    if requested_count > DEFAULT_BATCH_SIZE:
        raise ValueError(
            f"A batch may contain at most {DEFAULT_BATCH_SIZE} configs to protect the staging file quota"
        )
    offset = args.offset if args.offset is not None else int(manifest["next_offset"])
    rows = fetch_dataset_rows(
        dataset=args.dataset,
        config_name=args.dataset_config,
        split=args.split,
        offset=offset,
        limit=requested_count,
        ids=ids,
        token=os.environ.get("HF_TOKEN"),
    )
    run_ids = [_validate_run_id(str(row["plasma_config_id"])) for row in rows]
    duplicates = [run_id for run_id in run_ids if run_id in manifest["runs"]]
    if duplicates:
        raise ValueError(f"IDs already recorded in the manifest: {', '.join(duplicates)}")

    repo_root = Path(__file__).resolve().parents[2]
    template_dir = args.template_dir
    if not template_dir.is_absolute():
        template_dir = repo_root / template_dir
    spec = GenerationSpec(args.dataset, args.dataset_config, args.split, args.phiedge)
    config_paths = [
        materialize_run(
            row,
            output_root=output_root,
            template_dir=template_dir,
            spec=spec,
            overwrite=args.overwrite,
        )
        for row in rows
    ]

    number = max((int(batch["number"]) for batch in manifest["batches"]), default=0) + 1
    batch = {
        "number": number,
        "name": f"run{number}",
        "offset": offset if not ids else None,
        "count": len(run_ids),
        "ids": run_ids,
        "status": "generated",
        "archive": f"run{number}.tar",
        "created_at": _now(),
        "updated_at": _now(),
    }
    manifest["batches"].append(batch)
    for row, config_path in zip(rows, config_paths, strict=True):
        run_id = str(row["plasma_config_id"])
        manifest["runs"][run_id] = {
            "batch": number,
            "dataset_row_index": row.get("_dataset_row_idx"),
            "config": str(config_path.relative_to(output_root)),
            "status": "pending",
            "has_run": False,
            "attempts": 0,
            "last_exit_code": None,
            "updated_at": _now(),
        }
    if not ids:
        manifest["next_offset"] = offset + len(rows)
    _save_manifest(manifest_path, manifest)
    logger.info("Generated batch %s with %d configs", batch["name"], len(run_ids))
    return batch


def _active_batch(manifest: dict[str, Any]) -> dict[str, Any] | None:
    """Return the oldest batch that has not yet been safely archived."""
    return next((batch for batch in manifest["batches"] if batch["status"] != "archived"), None)


def _verify_archive(archive_path: Path, run_ids: Sequence[str]) -> None:
    """Check that an archive can be read and contains every run config."""
    with tarfile.open(archive_path, "r") as archive:
        members = set(archive.getnames())
    missing = [f"inputs/{run_id}/config.yaml" for run_id in run_ids
               if f"inputs/{run_id}/config.yaml" not in members]
    if missing:
        raise RuntimeError(f"Archive {archive_path} is missing required members: {missing}")


def _archive_batch(
    output_root: Path,
    batch: dict[str, Any],
    manifest: dict[str, Any],
    manifest_path: Path,
) -> Path:
    """Archive a successful batch, verify it, then remove its loose run trees."""
    archive_path = output_root / batch["archive"]
    temporary = output_root / f".{batch['archive']}.tmp"
    run_ids = batch["ids"]
    batch["status"] = "archiving"
    batch["updated_at"] = _now()
    _save_manifest(manifest_path, manifest)

    if not archive_path.exists():
        if temporary.exists():
            temporary.unlink()
        with tarfile.open(temporary, "w") as archive:
            for run_id in run_ids:
                _validate_run_id(run_id)
                for directory in ("inputs", "outputs"):
                    source = output_root / directory / run_id
                    if source.exists():
                        if source.is_symlink():
                            raise RuntimeError(f"Refusing to archive symlinked run directory: {source}")
                        archive.add(source, arcname=f"{directory}/{run_id}", recursive=True)
        _verify_archive(temporary, run_ids)
        os.replace(temporary, archive_path)
    _verify_archive(archive_path, run_ids)

    # The verified tar is now the durable copy. Removing these exact per-ID
    # directories is what brings the staging inode/file count back down.
    for run_id in run_ids:
        for directory in ("inputs", "outputs"):
            source = output_root / directory / run_id
            if source.exists():
                if source.is_symlink():
                    raise RuntimeError(f"Refusing to remove symlinked run directory: {source}")
                shutil.rmtree(source)
        run = manifest["runs"][run_id]
        run["status"] = "archived"
        run["archive"] = archive_path.name
        run["archive_config"] = f"inputs/{run_id}/config.yaml"
        run["updated_at"] = _now()
    batch["status"] = "archived"
    batch["updated_at"] = _now()
    batch["archived_at"] = _now()
    _save_manifest(manifest_path, manifest)
    logger.info("Archived %s and removed its verified loose input/output trees", archive_path)
    return archive_path


def _process_one_batch(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    manifest_path: Path,
    output_root: Path,
    repo_root: Path,
    *,
    count_override: int | None = None,
) -> int:
    """Resume or create, run, and archive exactly one data batch."""
    batch = _active_batch(manifest)
    if batch is None:
        batch = _create_batch(
            args,
            manifest,
            manifest_path,
            output_root,
            count_override=count_override,
        )
    else:
        logger.info("Resuming unfinished batch %s", batch["name"])

    batch["status"] = "running"
    batch["max_parallel"] = args.max_parallel
    batch["updated_at"] = _now()
    _save_manifest(manifest_path, manifest)

    # Every configuration owns a distinct absolute input/output tree, so
    # parallel Snakemake controllers cannot target the same artifacts. They
    # do share the repository as cwd, however, which requires --nolock to
    # avoid Snakemake's coarse working-directory lock.
    args.nolock = args.max_parallel > 1
    pending_ids = [
        run_id for run_id in batch["ids"]
        if manifest["runs"][run_id]["status"] not in {"succeeded", "archived"}
    ]
    pending = iter(pending_ids)
    futures: dict[Future[int], str] = {}
    failures: list[str] = []
    stop_launching = False

    def submit_one(executor: ThreadPoolExecutor, run_id: str) -> None:
        run = manifest["runs"][run_id]
        config_path = output_root / run["config"]
        if not config_path.is_file():
            raise FileNotFoundError(f"Pending config does not exist: {config_path}")
        run["status"] = "running"
        if not args.dry_run:
            run["has_run"] = True
            run["attempts"] += 1
        run["updated_at"] = _now()
        _save_manifest(manifest_path, manifest)
        logger.info(
            "Launching %s config %d/%d: %s (%d controller(s) active)",
            batch["name"],
            batch["ids"].index(run_id) + 1,
            batch["count"],
            run_id,
            len(futures) + 1,
        )
        futures[executor.submit(_launch_config, config_path, args, repo_root)] = run_id

    try:
        with ThreadPoolExecutor(max_workers=args.max_parallel, thread_name_prefix="constellaration") as executor:
            for _ in range(min(args.max_parallel, len(pending_ids))):
                submit_one(executor, next(pending))

            while futures:
                completed, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in completed:
                    run_id = futures.pop(future)
                    run = manifest["runs"][run_id]
                    try:
                        returncode = future.result()
                        error_message = None
                    except BaseException as exc:
                        returncode = 1
                        error_message = f"{type(exc).__name__}: {exc}"
                        logger.exception("Controller for run %s raised an exception", run_id)
                    if args.dry_run:
                        run["status"] = "pending"
                    else:
                        run["status"] = "succeeded" if returncode == 0 else "failed"
                    run["last_exit_code"] = returncode
                    if error_message:
                        run["last_error"] = error_message
                    else:
                        run.pop("last_error", None)
                    run["updated_at"] = _now()
                    _save_manifest(manifest_path, manifest)
                    if returncode:
                        failures.append(run_id)
                        logger.error("Run %s failed with exit code %d", run_id, returncode)
                        if not args.keep_going:
                            stop_launching = True

                while not stop_launching and len(futures) < args.max_parallel:
                    try:
                        run_id = next(pending)
                    except StopIteration:
                        break
                    submit_one(executor, run_id)
    except BaseException:
        for run_id in futures.values():
            run = manifest["runs"][run_id]
            if run["status"] == "running":
                run["status"] = "interrupted"
                run["updated_at"] = _now()
        batch["status"] = "interrupted"
        batch["updated_at"] = _now()
        _save_manifest(manifest_path, manifest)
        raise

    if args.dry_run:
        batch["status"] = "generated"
        batch["updated_at"] = _now()
        _save_manifest(manifest_path, manifest)
        return 1 if failures else 0
    incomplete = [
        run_id for run_id in batch["ids"]
        if manifest["runs"][run_id]["status"] != "succeeded"
    ]
    if incomplete:
        batch["status"] = "failed"
        batch["updated_at"] = _now()
        _save_manifest(manifest_path, manifest)
        logger.error("Batch %s remains unarchived; retry these IDs: %s", batch["name"], ", ".join(incomplete))
        return 1
    batch["status"] = "completed"
    batch["updated_at"] = _now()
    _save_manifest(manifest_path, manifest)
    _archive_batch(output_root, batch, manifest, manifest_path)
    return 0


def batch_runs(args: argparse.Namespace) -> int:
    """Generate, launch, record, and archive one batch or the full dataset."""
    if not 1 <= args.batch_size <= DEFAULT_BATCH_SIZE:
        raise ValueError(f"batch-size must be between 1 and {DEFAULT_BATCH_SIZE}")
    if args.cores < 1:
        raise ValueError(f"cores must be >= 1, got {args.cores}")
    if not 1 <= args.max_parallel <= DEFAULT_BATCH_SIZE:
        raise ValueError(f"max-parallel must be between 1 and {DEFAULT_BATCH_SIZE}")
    if args.loop_iters < 0:
        raise ValueError(f"loop-iters must be >= 0, got {args.loop_iters}")
    if args.dry_run and args.loop_iters:
        raise ValueError("--dry-run cannot be combined with --loop-iters; pass --loop-iters 0")
    all_mode = getattr(args, "all", False)
    if all_mode and args.dry_run:
        raise ValueError("--all cannot be combined with --dry-run")
    if all_mode and (args.id or args.ids_file):
        raise ValueError("--all cannot be combined with --id or --ids-file")
    if all_mode and args.offset is not None:
        raise ValueError("--all cannot be combined with --offset; it resumes from manifest.next_offset")

    output_root = _logical_absolute(args.output_root)
    manifest_path = output_root / "manifest.json"
    repo_root = Path(__file__).resolve().parents[2]
    with _batch_lock(output_root):
        manifest = _load_manifest(manifest_path, args)
        total_rows = None
        if all_mode:
            total_rows = fetch_dataset_row_count(
                dataset=args.dataset,
                config_name=args.dataset_config,
                split=args.split,
                token=os.environ.get("HF_TOKEN"),
            )
            manifest["dataset_total_rows"] = total_rows
            manifest["dataset_exhausted"] = False
            manifest.pop("dataset_completed_at", None)
            _save_manifest(manifest_path, manifest)
            logger.info("Continuous mode: %d total dataset rows", total_rows)

        while True:
            count_override = None
            if all_mode and _active_batch(manifest) is None:
                remaining = total_rows - int(manifest["next_offset"])
                if remaining <= 0:
                    manifest["dataset_exhausted"] = True
                    manifest["dataset_completed_at"] = _now()
                    _save_manifest(manifest_path, manifest)
                    logger.info("Continuous mode complete: all %d dataset rows are archived", total_rows)
                    return 0
                count_override = min(args.batch_size, remaining)

            result = _process_one_batch(
                args,
                manifest,
                manifest_path,
                output_root,
                repo_root,
                count_override=count_override,
            )
            if result or not all_mode:
                return result


def _add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"Hugging Face dataset (default: {DEFAULT_DATASET}).",
    )
    parser.add_argument("--dataset-config", default="default", help="Dataset subset/config name (default: default).")
    parser.add_argument("--split", default="train", help="Dataset split (default: train).")


def _add_launch_arguments(parser: argparse.ArgumentParser, *, loop_default: int) -> None:
    """Add options shared by the plain launcher and resumable batch driver."""
    parser.add_argument("--cores", type=int, default=4, help="Snakemake cores per run (default: 4).")
    parser.add_argument("--profile", help="Optional Snakemake profile, such as the repository HTCondor profile.")
    parser.add_argument("--dry-run", action="store_true", help="Plan forward runs without executing containers.")
    parser.add_argument(
        "--loop-iters",
        type=int,
        default=loop_default,
        help=f"Use the closed-loop driver for this many iterations; 0 runs one forward pass (default: {loop_default}).",
    )
    parser.add_argument("--container-runtime", choices=("docker", "apptainer"), help="Container runtime override.")
    parser.add_argument("--gpu-ids", help="GPU pool override: all or a comma-separated list.")
    parser.add_argument("--jobs-per-gpu", type=int, help="Maximum concurrent jobs per GPU.")
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue launching later configs after a run fails.",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for generation and launch operations."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate quick-run inputs from dataset rows.")
    _add_dataset_arguments(generate)
    generate.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Run root (default: {DEFAULT_OUTPUT_ROOT}).",
    )
    generate.add_argument(
        "--template-dir",
        type=Path,
        default=Path("inputs/quick_run"),
        help="Quick-run template directory relative to the repo root.",
    )
    generate.add_argument(
        "--offset",
        type=int,
        default=0,
        help="First dataset row when no --id is supplied (default: 0).",
    )
    generate.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Number of consecutive rows when no --id is supplied (default: 1).",
    )
    generate.add_argument(
        "--id",
        action="append",
        default=[],
        help="Exact plasma_config_id to generate; repeat for multiple runs.",
    )
    generate.add_argument("--ids-file", type=Path, help="Text file containing one plasma_config_id per line.")
    generate.add_argument(
        "--phiedge",
        type=float,
        default=2.67716881180502,
        help="VMEC edge toroidal flux in Webers (default: shipped quick-run value).",
    )
    generate.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace generated input files for IDs that already exist.",
    )
    generate.set_defaults(handler=generate_runs)

    launch = subparsers.add_parser("launch", help="Launch generated configs sequentially.")
    launch.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Run root (default: {DEFAULT_OUTPUT_ROOT}).",
    )
    launch.add_argument(
        "--id",
        action="append",
        default=[],
        help="Launch only this generated ID; repeat for multiple runs.",
    )
    _add_launch_arguments(launch, loop_default=0)
    launch.set_defaults(handler=launch_runs)

    batch = subparsers.add_parser(
        "batch",
        help="Generate, launch, record, and archive a resumable batch of at most 100 configs.",
    )
    _add_dataset_arguments(batch)
    batch.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Run root (default: {DEFAULT_OUTPUT_ROOT}).",
    )
    batch.add_argument(
        "--template-dir",
        type=Path,
        default=Path("inputs/quick_run"),
        help="Quick-run template directory relative to the repo root.",
    )
    batch.add_argument(
        "--offset",
        type=int,
        default=None,
        help="First dataset row for a new batch; default resumes at manifest.next_offset.",
    )
    batch.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Configs in each batch (default and maximum: {DEFAULT_BATCH_SIZE}).",
    )
    batch.add_argument(
        "--max-parallel",
        type=int,
        default=DEFAULT_PARALLEL_RUNS,
        help=f"Maximum active config controllers (default: {DEFAULT_PARALLEL_RUNS}; maximum: {DEFAULT_BATCH_SIZE}).",
    )
    batch.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="Repeat batches until every row in the selected Hugging Face dataset split is archived.",
    )
    batch.add_argument("--id", action="append", default=[], help="Exact ID; repeat to make an explicit batch.")
    batch.add_argument("--ids-file", type=Path, help="Text file containing one plasma_config_id per line.")
    batch.add_argument(
        "--phiedge",
        type=float,
        default=2.67716881180502,
        help="VMEC edge toroidal flux in Webers (default: shipped quick-run value).",
    )
    batch.add_argument("--overwrite", action="store_true", help="Replace pre-existing generated input files.")
    _add_launch_arguments(batch, loop_default=3)
    batch.set_defaults(handler=batch_runs)

    return parser


def main(argv: Iterable[str] | None = None) -> int:
    """Run the ConStellaration generation/launch CLI."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    result = args.handler(args)
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
