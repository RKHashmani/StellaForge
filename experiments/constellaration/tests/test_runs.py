"""Tests for generating and launching ConStellaration dataset runs."""

from __future__ import annotations

import argparse
import io
import json
import threading
from pathlib import Path
from urllib import error

import pytest
import yaml

from experiments.constellaration import runs


def _row(run_id: str = "Dtest123") -> dict:
    """Return a small valid stellarator-symmetric dataset row."""
    return {
        "plasma_config_id": run_id,
        "_dataset_row_idx": 7,
        "boundary.is_stellarator_symmetric": True,
        "boundary.n_field_periods": 3,
        "boundary.r_cos": [[0.0, 1.0, 0.1], [0.2, 0.3, 0.4]],
        "boundary.z_sin": [[0.0, 0.0, 0.1], [-0.2, 0.0, 0.2]],
    }


def _spec(phiedge: float = 2.5) -> runs.GenerationSpec:
    return runs.GenerationSpec(runs.DEFAULT_DATASET, "default", "train", phiedge)


def test_build_vmec_input_maps_dataset_modes() -> None:
    text = runs.build_vmec_input(_row(), phiedge=2.5)

    assert "NFP = 3" in text
    assert "MPOL = 2" in text
    assert "NTOR = 1" in text
    assert "NITER_ARRAY = 10000" in text
    assert "PHIEDGE = 2.5" in text
    assert "RBC( -1,  1) =  2.0000000000000001e-01" in text
    assert "ZBS(  1,  1) =  2.0000000000000001e-01" in text


def test_build_vmec_input_rejects_asymmetric_boundary() -> None:
    row = _row()
    row["boundary.is_stellarator_symmetric"] = False
    with pytest.raises(ValueError, match="stellarator-symmetric"):
        runs.build_vmec_input(row, phiedge=1.0)


def test_fetch_dataset_rows_pages_without_datasets_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict]] = []

    def fake_request(endpoint: str, parameters: dict, token: str | None) -> dict:
        calls.append((endpoint, parameters))
        start = int(parameters["offset"])
        length = int(parameters["length"])
        return {
            "rows": [
                {"row_idx": index, "row": _row(f"id{index}"), "truncated_cells": []}
                for index in range(start, start + length)
            ]
        }

    monkeypatch.setattr(runs, "_request_json", fake_request)
    rows = runs.fetch_dataset_rows(
        dataset=runs.DEFAULT_DATASET,
        config_name="default",
        split="train",
        offset=5,
        limit=101,
    )

    assert [row["plasma_config_id"] for row in rows[:2]] == ["id5", "id6"]
    assert rows[-1]["plasma_config_id"] == "id105"
    assert [(call[0], call[1]["offset"], call[1]["length"]) for call in calls] == [
        ("rows", 5, 100),
        ("rows", 105, 1),
    ]


def test_fetch_dataset_rows_filters_exact_id(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake_request(endpoint: str, parameters: dict, token: str | None) -> dict:
        captured.update(endpoint=endpoint, parameters=parameters, token=token)
        return {"rows": [{"row_idx": 9, "row": _row("Dabc"), "truncated_cells": []}]}

    monkeypatch.setattr(runs, "_request_json", fake_request)
    result = runs.fetch_dataset_rows(
        dataset=runs.DEFAULT_DATASET,
        config_name="default",
        split="train",
        offset=0,
        limit=1,
        ids=["Dabc"],
        token="secret",
    )

    assert result[0]["_dataset_row_idx"] == 9
    assert captured["endpoint"] == "filter"
    assert captured["parameters"]["where"] == '"plasma_config_id"=\'Dabc\''
    assert captured["token"] == "secret"


def test_fetch_dataset_row_count_uses_rows_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runs,
        "_request_json",
        lambda endpoint, parameters, token: {"num_rows_total": 237, "rows": []},
    )

    assert runs.fetch_dataset_row_count(
        dataset=runs.DEFAULT_DATASET,
        config_name="default",
        split="train",
    ) == 237


def test_request_json_retries_transient_dataset_index_response(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    def fake_urlopen(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise error.HTTPError(
                "https://example.invalid",
                500,
                "Internal Server Error",
                {},
                io.BytesIO(b'{"error":"the dataset index is loading"}'),
            )
        return Response(b'{"rows": []}')

    monkeypatch.setattr(runs.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(runs.time, "sleep", lambda seconds: None)

    assert runs._request_json("filter", {"dataset": "test"}, None) == {"rows": []}
    assert calls == 2


def test_materialize_run_writes_self_contained_quick_run(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    config_path = runs.materialize_run(
        _row(),
        output_root=tmp_path,
        template_dir=repo_root / "inputs/quick_run",
        spec=_spec(),
        overwrite=False,
    )

    config = yaml.safe_load(config_path.read_text())
    input_dir = tmp_path / "inputs/Dtest123"
    assert config["run_name"] == "Dtest123"
    assert config["input_dir"] == str(input_dir)
    assert config["output_dir"] == str(tmp_path / "outputs/Dtest123")
    assert config["dataset_source"]["row_index"] == 7
    assert (input_dir / "vmec_input.Dtest123").is_file()
    assert (input_dir / "sfincs_input.Dtest123").is_file()
    assert (input_dir / "Dtest123.toml").is_file()
    assert (input_dir / "common_input.toml").is_file()
    provenance = json.loads((input_dir / "dataset_row.json").read_text())
    assert provenance["plasma_config_id"] == "Dtest123"
    assert provenance["row"]["boundary.n_field_periods"] == 3
    assert str(tmp_path / "outputs/Dtest123/stage1_equilibrium/wout_Dtest123.nc") in (
        input_dir / "Dtest123.toml"
    ).read_text()


def test_materialize_run_refuses_to_overwrite(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    kwargs = {
        "output_root": tmp_path,
        "template_dir": repo_root / "inputs/quick_run",
        "spec": _spec(phiedge=1.0),
        "overwrite": False,
    }
    runs.materialize_run(_row(), **kwargs)
    with pytest.raises(FileExistsError, match="--overwrite"):
        runs.materialize_run(_row(), **kwargs)


def test_logical_absolute_does_not_resolve_staging_symlink(tmp_path: Path) -> None:
    physical = tmp_path / "physical"
    physical.mkdir()
    logical = tmp_path / "logical"
    logical.symlink_to(physical, target_is_directory=True)

    assert runs._logical_absolute(logical) == logical


def test_launch_runs_builds_forward_commands(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    configs = [tmp_path / "inputs/id1/config.yaml", tmp_path / "inputs/id2/config.yaml"]
    for config in configs:
        config.parent.mkdir(parents=True)
        config.write_text("run_name: test\n")
    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs):
        commands.append(command)
        return argparse.Namespace(returncode=0)

    monkeypatch.setattr(runs.subprocess, "run", fake_run)
    args = argparse.Namespace(
        output_root=tmp_path,
        id=[],
        cores=6,
        profile=None,
        dry_run=True,
        loop_iters=0,
        keep_going=False,
    )

    assert runs.launch_runs(args) == 0
    assert commands == [
        ["snakemake", "--dry-run", "--cores", "6", "--configfile", str(configs[0])],
        ["snakemake", "--dry-run", "--cores", "6", "--configfile", str(configs[1])],
    ]


def test_launch_runs_stops_after_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for run_id in ("id1", "id2"):
        config = tmp_path / "inputs" / run_id / "config.yaml"
        config.parent.mkdir(parents=True)
        config.write_text("run_name: test\n")
    calls = 0

    def fake_run(command: list[str], **kwargs):
        nonlocal calls
        calls += 1
        return argparse.Namespace(returncode=2)

    monkeypatch.setattr(runs.subprocess, "run", fake_run)
    args = argparse.Namespace(
        output_root=tmp_path,
        id=[],
        cores=1,
        profile=None,
        dry_run=False,
        loop_iters=0,
        keep_going=False,
    )

    assert runs.launch_runs(args) == 1
    assert calls == 1


def test_closed_loop_launch_forwards_htcondor_runtime_overrides(tmp_path: Path) -> None:
    config = tmp_path / "inputs/id1/config.yaml"
    args = argparse.Namespace(
        loop_iters=3,
        cores=8,
        profile="executors/htcondor/profiles/htcondor-gpu",
        dry_run=False,
        container_runtime="apptainer",
        gpu_ids="all",
        jobs_per_gpu=2,
        nolock=True,
    )

    repo_root = tmp_path / "repo"
    command = runs._launch_command(config, args, repo_root)

    assert command[:4] == [runs.sys.executable, "-m", "src.ouroboros", "--config"]
    assert command[command.index("--max-iters") + 1] == "3"
    assert command[command.index("--profile") + 1] == args.profile
    assert command[command.index("--container-runtime") + 1] == "apptainer"
    assert command[command.index("--gpu-ids") + 1] == "all"
    assert command[command.index("--jobs-per-gpu") + 1] == "2"
    assert command[command.index("--htcondor-jobdir") + 1] == str(repo_root / "jobs/constellaration/id1")
    assert "--nolock" in command


def test_forward_htcondor_launch_uses_compat_cli_and_isolated_jobdir(tmp_path: Path) -> None:
    config = tmp_path / "inputs/id1/config.yaml"
    args = argparse.Namespace(
        loop_iters=0,
        cores=8,
        profile="executors/htcondor/profiles/htcondor-gpu",
        dry_run=False,
        container_runtime="apptainer",
        gpu_ids="all",
        jobs_per_gpu=None,
        nolock=True,
    )

    repo_root = tmp_path / "repo"
    command = runs._launch_command(config, args, repo_root)

    assert command[:3] == [runs.sys.executable, "-m", "src.snakemake_htcondor"]
    assert command[command.index("--htcondor-jobdir") + 1] == str(repo_root / "jobs/constellaration/id1")


def test_launch_config_collects_home_htcondor_logs_into_run_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = tmp_path / "runs/inputs/id1/config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("run_name: id1\n")
    repo_root = tmp_path / "repo"
    live_jobdir = repo_root / "jobs/constellaration/id1"

    def fake_run(command: list[str], **kwargs):
        assert command[command.index("--htcondor-jobdir") + 1] == str(live_jobdir)
        assert kwargs["env"]["DRIFTLESS_STAR_RUN_ID"] == "id1"
        live_jobdir.mkdir(parents=True)
        (live_jobdir / "snakemake-rules.log").write_text("event log")
        return argparse.Namespace(returncode=0)

    monkeypatch.setattr(runs.subprocess, "run", fake_run)
    args = argparse.Namespace(
        loop_iters=0,
        cores=8,
        profile="executors/htcondor/profiles/htcondor-gpu",
        dry_run=False,
        container_runtime="apptainer",
        gpu_ids="all",
        jobs_per_gpu=None,
        nolock=True,
    )

    assert runs._launch_config(config, args, repo_root) == 0
    collected = tmp_path / "runs/outputs/id1/htcondor/attempt_1/snakemake-rules.log"
    assert collected.read_text() == "event log"
    assert not live_jobdir.exists()


def test_launch_runs_rejects_dry_run_with_loop(tmp_path: Path) -> None:
    args = argparse.Namespace(
        output_root=tmp_path,
        id=[],
        cores=1,
        profile=None,
        dry_run=True,
        loop_iters=2,
        keep_going=False,
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        runs.launch_runs(args)


def test_batch_runs_records_archives_and_removes_loose_trees(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rows = [_row("id1"), _row("id2")]
    rows[0]["_dataset_row_idx"] = 10
    rows[1]["_dataset_row_idx"] = 11
    monkeypatch.setattr(runs, "fetch_dataset_rows", lambda **kwargs: rows)
    launched: list[str] = []
    launch_barrier = threading.Barrier(2)

    def fake_launch(config_path: Path, args: argparse.Namespace, repo_root: Path) -> int:
        launch_barrier.wait(timeout=2)
        launched.append(config_path.parent.name)
        return 0

    monkeypatch.setattr(runs, "_launch_config", fake_launch)
    args = argparse.Namespace(
        dataset=runs.DEFAULT_DATASET,
        dataset_config="default",
        split="train",
        output_root=tmp_path,
        template_dir=Path("inputs/quick_run"),
        offset=None,
        batch_size=2,
        max_parallel=2,
        id=[],
        ids_file=None,
        phiedge=2.5,
        overwrite=False,
        cores=8,
        profile="executors/htcondor/profiles/htcondor-gpu",
        dry_run=False,
        loop_iters=3,
        container_runtime="apptainer",
        gpu_ids="all",
        jobs_per_gpu=None,
        keep_going=False,
    )

    assert runs.batch_runs(args) == 0
    assert sorted(launched) == ["id1", "id2"]
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["next_offset"] == 2
    assert manifest["batches"][0]["status"] == "archived"
    assert manifest["batches"][0]["archive"] == "run1.tar"
    assert manifest["batches"][0]["max_parallel"] == 2
    assert manifest["runs"]["id1"]["status"] == "archived"
    assert manifest["runs"]["id1"]["has_run"] is True
    assert manifest["runs"]["id1"]["attempts"] == 1
    assert (tmp_path / "run1.tar").is_file()
    assert not (tmp_path / "inputs/id1").exists()
    assert not (tmp_path / "inputs/id2").exists()
    runs._verify_archive(tmp_path / "run1.tar", ["id1", "id2"])


def test_all_mode_repeats_batches_and_archives_final_partial_batch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fetches: list[tuple[int, int]] = []

    def fake_fetch(**kwargs):
        offset = kwargs["offset"]
        limit = kwargs["limit"]
        fetches.append((offset, limit))
        rows = []
        for index in range(offset, offset + limit):
            row = _row(f"id{index}")
            row["_dataset_row_idx"] = index
            rows.append(row)
        return rows

    monkeypatch.setattr(runs, "fetch_dataset_row_count", lambda **kwargs: 5)
    monkeypatch.setattr(runs, "fetch_dataset_rows", fake_fetch)
    monkeypatch.setattr(runs, "_launch_config", lambda config_path, args, repo_root: 0)
    args = argparse.Namespace(
        dataset=runs.DEFAULT_DATASET,
        dataset_config="default",
        split="train",
        output_root=tmp_path,
        template_dir=Path("inputs/quick_run"),
        offset=None,
        batch_size=2,
        max_parallel=2,
        all=True,
        id=[],
        ids_file=None,
        phiedge=2.5,
        overwrite=False,
        cores=8,
        profile="executors/htcondor/profiles/htcondor-gpu",
        dry_run=False,
        loop_iters=3,
        container_runtime="apptainer",
        gpu_ids="all",
        jobs_per_gpu=None,
        keep_going=True,
    )

    assert runs.batch_runs(args) == 0
    assert fetches == [(0, 2), (2, 2), (4, 1)]
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["dataset_total_rows"] == 5
    assert manifest["dataset_exhausted"] is True
    assert manifest["next_offset"] == 5
    assert [batch["count"] for batch in manifest["batches"]] == [2, 2, 1]
    assert [batch["archive"] for batch in manifest["batches"]] == ["run1.tar", "run2.tar", "run3.tar"]
    assert all((tmp_path / f"run{number}.tar").is_file() for number in (1, 2, 3))
