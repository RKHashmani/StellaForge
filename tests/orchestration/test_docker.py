"""Tests for ``src.utils.docker``.

``resolve_docker_user`` decides whether every stage's container launch carries
``--user "$(id -u):$(id -g)"``. Getting this wrong does not degrade a run, it stops one
outright. Passing the flag to a rootless daemon runs the container as an unprivileged
subuid that cannot read the bind-mounted repo, and the stage reports the resulting
``EACCES`` as a misleading "file not found" from its own argument parser. Omitting it on
a rootful daemon leaves every output file root-owned in the user's working tree.

The probe is therefore pinned in both directions, and so is its failure behavior. A dry
run planning commands on a machine with no Docker is a supported workflow, so an
unreachable daemon must fall back to the rootful flag rather than raise, which also keeps
the planned text identical to what it was before detection existed.
"""

from __future__ import annotations

import json
import re
import subprocess

import pytest

from src.utils import resolve_docker_user
from src.utils.docker import HOST_USER_FLAG, _probe_rootless, daemon_is_rootless

ROOTLESS_OPTIONS = ["name=seccomp,profile=builtin", "name=rootless"]
ROOTFUL_OPTIONS = ["name=seccomp,profile=builtin", "name=cgroupns"]


# The probe is cached for the lifetime of a process, so a test that leaves a result behind would decide the outcome of
# whichever test ran next.
@pytest.fixture(autouse=True)
def _clear_probe_cache() -> None:
    daemon_is_rootless.cache_clear()
    yield
    daemon_is_rootless.cache_clear()


def _fake_docker_info(monkeypatch: pytest.MonkeyPatch, stdout: str = "", returncode: int = 0) -> None:
    """Stand in for the `docker info` call with a fixed result."""

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(args=["docker"], returncode=returncode, stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)


def _raising_docker_info(monkeypatch: pytest.MonkeyPatch, exc: Exception) -> None:
    """Stand in for the `docker info` call with a failure to invoke it at all."""

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess:
        raise exc

    monkeypatch.setattr(subprocess, "run", fake_run)


# An explicit mode must not consult the daemon at all, so that a user who knows their host can pin the behavior and a
# machine with no Docker can still plan commands. The probe is replaced with one that fails the test if it is reached.
@pytest.mark.parametrize(("mode", "expected"), [("host", HOST_USER_FLAG), ("root", "")])
def test_explicit_mode_is_used_without_probing_the_daemon(
    monkeypatch: pytest.MonkeyPatch, mode: str, expected: str
) -> None:
    _raising_docker_info(monkeypatch, AssertionError("the daemon must not be probed for an explicit mode"))
    assert resolve_docker_user({"docker_user": mode}) == expected


# Case is normalized the way gpu_ids already normalizes "ALL", so a config written in either case resolves the same.
@pytest.mark.parametrize("mode", ["HOST", " host ", "Root", " ROOT "])
def test_explicit_mode_ignores_case_and_surrounding_space(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    _raising_docker_info(monkeypatch, AssertionError("the daemon must not be probed for an explicit mode"))
    expected = HOST_USER_FLAG if mode.strip().lower() == "host" else ""
    assert resolve_docker_user({"docker_user": mode}) == expected


# The two daemons this exists to tell apart. A rootless daemon already maps the invoking user onto the container's
# root, so the flag must disappear; a rootful one needs it to keep outputs out of root ownership. An absent key must
# behave exactly like "auto", because no committed run config carries the key and every one of them must still work.
@pytest.mark.parametrize("config", [{}, {"docker_user": "auto"}])
@pytest.mark.parametrize(
    ("options", "expected"), [(ROOTLESS_OPTIONS, ""), (ROOTFUL_OPTIONS, HOST_USER_FLAG)]
)
def test_auto_follows_the_daemon(
    monkeypatch: pytest.MonkeyPatch, config: dict, options: list[str], expected: str
) -> None:
    _fake_docker_info(monkeypatch, stdout=json.dumps(options))
    assert resolve_docker_user(config) == expected


# Every way the probe can fail must resolve to the rootful flag rather than raise. That is the behavior the Snakefile
# had before detection existed, so a machine with no Docker plans exactly the command it always did, and a wedged or
# unreadable daemon costs a suboptimal flag instead of a broken parse.
@pytest.mark.parametrize(
    ("name", "apply"),
    [
        ("docker not installed", lambda mp: _raising_docker_info(mp, FileNotFoundError("docker"))),
        ("daemon unreachable", lambda mp: _fake_docker_info(mp, stdout="", returncode=1)),
        ("probe timed out", lambda mp: _raising_docker_info(mp, subprocess.TimeoutExpired("docker", 10.0))),
        ("output not json", lambda mp: _fake_docker_info(mp, stdout="not json at all")),
        ("options not a list", lambda mp: _fake_docker_info(mp, stdout="null")),
        ("empty output", lambda mp: _fake_docker_info(mp, stdout="")),
    ],
)
def test_unreadable_daemon_falls_back_to_the_rootful_flag(
    monkeypatch: pytest.MonkeyPatch, name: str, apply: object
) -> None:
    apply(monkeypatch)
    assert _probe_rootless() is False
    assert resolve_docker_user({"docker_user": "auto"}) == HOST_USER_FLAG


# The Snakefile resolves this at parse time and the closed loop re-invokes snakemake once per iteration, so the probe
# must cost one daemon round trip per process rather than one per resolve.
def test_daemon_is_probed_at_most_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def counting_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess:
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(
            args=["docker"], returncode=0, stdout=json.dumps(ROOTLESS_OPTIONS), stderr=""
        )

    monkeypatch.setattr(subprocess, "run", counting_run)
    assert [resolve_docker_user({}) for _ in range(5)] == [""] * 5
    assert calls == 1


# A typo must name the offending value, because the alternative is a silently rootful run whose only symptom is
# root-owned outputs, or a silently rootless one that cannot read its inputs.
@pytest.mark.parametrize("value", ["hosts", "rootless", "", "none", True, None, 0, ["host"]])
def test_unknown_mode_is_rejected_naming_the_value(value: object) -> None:
    with pytest.raises(ValueError, match=re.escape(f"config['docker_user'] is {value!r}")):
        resolve_docker_user({"docker_user": value})
