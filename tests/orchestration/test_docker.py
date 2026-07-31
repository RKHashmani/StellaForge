"""Tests for ``src.utils.docker``.

``resolve_docker_user`` decides which ``--user`` flag every stage's container launch carries.
Getting this wrong does not degrade a run, it stops one outright. Passing the host uid to a
rootless runtime runs the container as an unprivileged subuid that cannot read the bind-mounted
repo, and the stage reports the resulting ``EACCES`` as a misleading "file not found" from its own
argument parser. Passing container root to a rootful daemon leaves every output file root-owned in
the user's working tree.

Both runtime schemas are pinned, since the repo supports a rootless podman aliased to ``docker``
and the two report the mapping in different places. So is the failure behavior. A dry run planning
commands on a machine with no runtime is a supported workflow, so an unreachable runtime must fall
back to the rootful flag rather than raise.
"""

from __future__ import annotations

import json
import re
import subprocess

import pytest

from src.utils import resolve_docker_user
from src.utils.docker import HOST_USER_FLAG, ROOT_USER_FLAG, _probe_daemon

# Docker reports the user mapping in a SecurityOptions list, Podman under host.security.rootless.
DOCKER_ROOTLESS = {"SecurityOptions": ["name=seccomp,profile=builtin", "name=rootless"]}
DOCKER_ROOTFUL = {"SecurityOptions": ["name=seccomp,profile=builtin", "name=cgroupns"]}
DOCKER_USERNS = {"SecurityOptions": ["name=seccomp,profile=builtin", "name=userns"]}
PODMAN_ROOTLESS = {"host": {"security": {"rootless": True}}}
PODMAN_ROOTFUL = {"host": {"security": {"rootless": False}}}


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


# An explicit mode must not consult the runtime at all, so that a user who knows their host can pin the behavior and a
# machine with no Docker can still plan commands. The probe is replaced with one that fails the test if it is reached.
@pytest.mark.parametrize(("mode", "expected"), [("host", HOST_USER_FLAG), ("root", ROOT_USER_FLAG)])
def test_explicit_mode_is_used_without_probing_the_runtime(
    monkeypatch: pytest.MonkeyPatch, mode: str, expected: str
) -> None:
    _raising_docker_info(monkeypatch, AssertionError("the runtime must not be probed for an explicit mode"))
    assert resolve_docker_user({"docker_user": mode}) == expected


# Case is normalized the way gpu_ids already normalizes "ALL", so a config written in either case resolves the same.
@pytest.mark.parametrize("mode", ["HOST", " host ", "Root", " ROOT "])
def test_explicit_mode_ignores_case_and_surrounding_space(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    _raising_docker_info(monkeypatch, AssertionError("the runtime must not be probed for an explicit mode"))
    expected = HOST_USER_FLAG if mode.strip().lower() == "host" else ROOT_USER_FLAG
    assert resolve_docker_user({"docker_user": mode}) == expected


# The four runtimes this exists to tell apart. A rootless runtime already maps the invoking user onto container root,
# so it needs uid 0; a rootful one needs the host uid to keep outputs out of root ownership. Docker and Podman report
# the same fact in different places, and the repo supports podman aliased to docker. An absent key must behave exactly
# like "auto", because no committed run config carries the key and every one of them must still work.
@pytest.mark.parametrize("config", [{}, {"docker_user": "auto"}])
@pytest.mark.parametrize(
    ("info", "expected"),
    [
        (DOCKER_ROOTLESS, ROOT_USER_FLAG),
        (DOCKER_ROOTFUL, HOST_USER_FLAG),
        (PODMAN_ROOTLESS, ROOT_USER_FLAG),
        (PODMAN_ROOTFUL, HOST_USER_FLAG),
    ],
)
def test_auto_follows_the_runtime(monkeypatch: pytest.MonkeyPatch, config: dict, info: dict, expected: str) -> None:
    _fake_docker_info(monkeypatch, stdout=json.dumps(info))
    assert resolve_docker_user(config) == expected


# A rootful daemon with user-namespace remapping maps container uids onto subordinate host uids, so neither container
# root nor the invoking uid owns the bind mount and no automatic choice is correct. This asserts "auto" refuses at
# parse time and names the escape hatch rather than planning a run that writes files nobody can read.
def test_auto_rejects_a_userns_remapped_daemon(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_docker_info(monkeypatch, stdout=json.dumps(DOCKER_USERNS))
    with pytest.raises(ValueError, match="subordinate host uids"):
        resolve_docker_user({})


# Every way the probe can fail must resolve to the rootful flag rather than raise, one case per branch. That is the
# behavior the Snakefile had before detection existed, so a machine with no runtime plans exactly the command it
# always did, and a wedged or unreadable runtime costs a suboptimal flag instead of a broken parse.
@pytest.mark.parametrize(
    ("name", "apply"),
    [
        ("runtime not installed", lambda mp: _raising_docker_info(mp, FileNotFoundError("docker"))),
        ("runtime unreachable", lambda mp: _fake_docker_info(mp, stdout="", returncode=1)),
        ("output not json", lambda mp: _fake_docker_info(mp, stdout="not json at all")),
        ("document is not an object", lambda mp: _fake_docker_info(mp, stdout="null")),
        ("neither schema present", lambda mp: _fake_docker_info(mp, stdout=json.dumps({"ServerVersion": "27.0"}))),
    ],
)
def test_unreadable_runtime_falls_back_to_the_host_flag(
    monkeypatch: pytest.MonkeyPatch, name: str, apply: object
) -> None:
    apply(monkeypatch)
    assert _probe_daemon() == "unknown"
    assert resolve_docker_user({"docker_user": "auto"}) == HOST_USER_FLAG


# A typo must name the offending value, because the alternative is a silently rootful run whose only symptom is
# root-owned outputs, or a silently rootless one that cannot read its inputs.
@pytest.mark.parametrize("value", ["hosts", "rootless", "", "none", True, None, 0, ["host"]])
def test_unknown_mode_is_rejected_naming_the_value(value: object) -> None:
    with pytest.raises(ValueError, match=re.escape(f"config['docker_user'] is {value!r}")):
        resolve_docker_user({"docker_user": value})
