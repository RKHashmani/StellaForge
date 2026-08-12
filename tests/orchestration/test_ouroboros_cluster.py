"""Tests for the cluster-dispatch flags of ``src.ouroboros``, the closed-loop driver.

``--profile`` sends the loop's forward passes to a cluster executor instead of the local machine, and
``--container-runtime`` picks the tool that runs each stage's container, since an execute node has no
Docker daemon and an HTCondor run needs ``apptainer``. Neither flag is useful without the other.

What makes them worth testing is that they fail quietly. A ``--profile`` the driver forgets to emit
raises no error at all -- the loop simply keeps running locally, and nobody notices until they wonder
why the cluster queue is empty. So these tests assemble the command line the driver would run, without
running it, and check that both flags are present, correctly positioned, and repeated on every
iteration, and that an unknown runtime is rejected on the submit host rather than inside the first
cluster job.

Only ``container_runtime=...`` has a real placement requirement: it belongs in the trailing
``--config`` group, which beats every config file, so a cluster run overrides the committed ``docker``
default instead of inheriting it. ``--profile`` is written ahead of both config groups for readability
only -- each group's value list stops at the next token starting with ``-``, so the pinned Snakemake
reads ``--profile`` the same way in either position (verified in both orders).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import src.ouroboros as ouroboros
from tests.orchestration.test_ouroboros import _write_config
from tests.orchestration.test_ouroboros_gpu import _captured_argv

PROFILE = "executors/htcondor/profiles/htcondor-gpu"


def test_profile_is_emitted_ahead_of_both_config_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    cmd = _captured_argv(monkeypatch, profile=PROFILE)

    assert cmd == [
        "snakemake", "out/converge_status.json", "--cores", "4",
        "--profile", PROFILE,
        "--configfile", "cfg.yaml",
        "--config", "input_dir=in", "output_dir=out",
    ]


# From iteration 2 the driver appends its loop overrides file to the --configfile group, which is the case
# where a second value could push --profile out of position.
def test_profile_survives_an_extra_configfile(monkeypatch: pytest.MonkeyPatch) -> None:
    cmd = _captured_argv(monkeypatch, profile=PROFILE, extra_configfiles=[Path("loop_overrides.yaml")])

    assert cmd[:6] == ["snakemake", "out/converge_status.json", "--cores", "4", "--profile", PROFILE]
    assert cmd[6:9] == ["--configfile", "cfg.yaml", "loop_overrides.yaml"]


# Both flags describe where the whole invocation runs, not one pass of it, so every iteration has to
# receive them. An iteration that lost either would quietly fall back to a local docker run partway
# through the loop.
def test_both_flags_reach_every_iteration(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    monkeypatch.setattr(ouroboros, "_seed_iteration_inputs", lambda **kw: None)

    forwarded: list[tuple] = []

    def fake_run(*, target, extra_config=None, profile=None, **kw):
        forwarded.append((profile, extra_config))
        signal = Path(target)
        signal.parent.mkdir(parents=True, exist_ok=True)
        signal.write_text(json.dumps({"status": "continue"}))  # neither halt nor converged -> keep looping

    monkeypatch.setattr(ouroboros, "run_forward_pass", fake_run)
    monkeypatch.setattr("sys.argv", ["ouroboros", "--config", str(config_path), "--max-iters", "2",
                                     "--profile", PROFILE, "--container-runtime", "apptainer"])
    ouroboros.main()

    assert forwarded == [(PROFILE, ["container_runtime=apptainer"])] * 2


# The choices list makes a typo fail at parse time on the submit host, rather than as an unclear
# Snakefile ValueError inside the first cluster job, minutes after the queue accepted it.
def test_unknown_container_runtime_is_rejected_at_parse(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    monkeypatch.setattr("sys.argv", ["ouroboros", "--config", str(config_path),
                                     "--container-runtime", "singularity"])

    with pytest.raises(SystemExit):
        ouroboros.main()
