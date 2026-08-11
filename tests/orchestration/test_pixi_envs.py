"""Tests for the root Pixi workspace's environment wiring, read out of ``pixi.lock``.

These tests read the lockfile directly, because they check what actually got installed rather than the
behaviour of any Python they could import and call, which is why nothing else in the suite catches
them. Three things are checked:

1. every environment that contains Snakemake contains the same version of it;
2. the HTCondor executor plugin is installed only where jobs are submitted, never in the execute-node
   container image;
3. the implicit ``default`` environment stays empty.

Each test's own comment says what breaks if it fails. The rest of this docstring explains the
``pixi.toml`` pins these tests enforce, since the manifest carries no comments of its own.

Why the Snakemake versions have to match. On a cluster run the HTCondor executor never sends Snakemake
to the execute node. It builds each job's command line using the *submit* host's Snakemake and expects
the node's Snakemake to understand it. The submit host uses the ``pipeline`` environment; the node uses
``htcondor-runtime``, which ``executors/htcondor/apptainer.def`` bakes into the container image.
``test`` is compared as well, because the dry runs under ``tests/e2e/`` call a real ``snakemake -n``,
and planning the pipeline with a different Snakemake than production uses cannot catch an
incompatibility.

Why an exact pin rather than a shared version range. Pixi re-solves only the environments whose specs
changed, so two environments can end up on different versions even from identical specs -- ``test``
once sat on 9.23.1 while ``pipeline`` sat on 9.20.0. And these tests can read the lockfile but not the
container image, which is built separately and copied to the cluster ahead of time: under a range, a
routine relock would move every environment together and pass here while the image already on the
cluster kept the old version. With an exact pin the version changes only when someone edits that line,
which is the signal to rebuild the image. It has to stay at 9.16 or above, the release that added
``--runtime-source-cache-path``, which the ``tests/e2e/`` dry runs pass.

Why the executor plugin is declared as it is. ``feature.htcondor`` lists it under
``pypi-dependencies`` because bioconda's newest build is 0.2.1 and conda-forge has none, so 0.3.0 is
available no other way. It is scoped to ``target.linux-64`` because 0.3.0 needs ``htcondor>=24.5.1``,
which ships Linux wheels only, and an unscoped declaration fails the lock on the workspace's
``osx-arm64`` platform. The feature composes into ``pipeline`` because Snakemake discovers executors
through installed entry points.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCKFILE = REPO_ROOT / "pixi.lock"

# Every environment that holds a Snakemake, and so has to agree with the others on its version.
SNAKEMAKE_ENVIRONMENTS = ("pipeline", "htcondor-runtime", "test")

# Version field of a conda URL's trailing <name>-<version>-<build>.conda, matched on the exact package name.
_CONDA_PKG = re.compile(r"/(?P<name>[a-z0-9._-]+?)-(?P<version>[0-9][^-]*)-[^-]+\.conda$")


@pytest.fixture(scope="module")
def lock() -> dict:
    """Parse ``pixi.lock`` once for the module.

    Uses libyaml through ``CSafeLoader`` when the build provides it, because the lockfile is a few
    hundred KB and the pure-Python loader would otherwise take most of the test's runtime.
    """
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    return yaml.load(LOCKFILE.read_text(), Loader=loader)


def _package_version(lock: dict, environment: str, package: str, platform: str = "linux-64") -> str:
    """Return the version of ``package`` locked for ``environment`` on ``platform``.

    Parameters
    ----------
    lock : dict
        Parsed ``pixi.lock`` contents.
    environment : str
        Pixi environment name, e.g. ``"pipeline"``.
    package : str
        Exact conda package name, e.g. ``"snakemake-minimal"``.
    platform : str, optional
        Platform subdir to read. Defaults to ``linux-64``, the only platform all these environments
        share, since ``htcondor-runtime`` is deliberately Linux-only.

    Returns
    -------
    str
        The locked version string.

    Raises
    ------
    AssertionError
        If the environment, the platform, or the package is absent from the lockfile.
    """
    environments = lock["environments"]
    assert environment in environments, f"no '{environment}' environment in pixi.lock"
    packages = environments[environment].get("packages", {})
    assert platform in packages, f"'{environment}' locks no {platform} packages"

    for entry in packages[platform]:
        url = entry.get("conda") if isinstance(entry, dict) else None
        if not url:
            continue
        match = _CONDA_PKG.search(url)
        if match and match.group("name") == package:
            return match.group("version")

    raise AssertionError(f"'{package}' not found in the {platform} packages of '{environment}'")


def test_every_environment_resolves_the_same_snakemake(lock: dict) -> None:
    versions = {env: _package_version(lock, env, "snakemake-minimal")
                for env in SNAKEMAKE_ENVIRONMENTS}

    assert len(set(versions.values())) == 1, (
        "Snakemake versions disagree across environments: "
        + ", ".join(f"{env}={version}" for env, version in versions.items()) + ".\n"
        "The HTCondor executor formats each remote command line with the SUBMIT host's flag\n"
        "vocabulary, so htcondor-runtime.sif's Snakemake has to parse it, and the tests/e2e/ dry\n"
        "runs are only meaningful if they plan DAGs with the same Snakemake production uses.\n"
        "Fix: set the snakemake-minimal pin to one exact version on both feature.pipeline and\n"
        "feature.htcondor-runtime in pixi.toml, relock, and rebuild htcondor-runtime.sif."
    )


# The executor turns each remote job into a plain `snakemake --mode remote ...` with no --executor flag, so
# the node never loads the plugin, and shipping it would pull htcondor2 and everything under it into the SIF
# for nothing.
def test_executor_plugin_is_submit_side_only(lock: dict) -> None:
    def has_plugin(environment: str) -> bool:
        packages = lock["environments"][environment].get("packages", {}).get("linux-64", [])
        return any("snakemake_executor_plugin_htcondor" in str(entry.get("pypi", ""))
                   for entry in packages if isinstance(entry, dict))

    assert has_plugin("pipeline"), "the submit host's env must carry the HTCondor executor plugin"
    assert not has_plugin("htcondor-runtime"), "the execute-node image must not carry the plugin"


# Every real environment sets `no-default-feature = true` and both repo tasks resolve through
# `default-environment`, so nothing installs `default` and anything reaching it is declared in the wrong
# place -- as happened when a root-level `[dependencies] snakemake` added a third solved environment to
# the lock.
def test_default_environment_stays_empty(lock: dict) -> None:
    assert lock["environments"]["default"].get("packages") == {}, (
        "the default environment gained packages; a dependency was declared at the root of "
        "pixi.toml instead of on a feature"
    )