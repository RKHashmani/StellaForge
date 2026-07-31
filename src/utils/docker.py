"""Container user resolution

``resolve_docker_user`` decides which ``--user`` flag every stage's ``docker run`` carries. A
rootful daemon needs ``--user "$(id -u):$(id -g)"`` so a container's writes into the bind-mounted
repo land host-owned rather than root-owned. A rootless runtime already maps the invoking host
user onto container uid 0, so it needs ``--user 0:0`` instead, and passing the host uid there
leaves the container as an unprivileged subuid that cannot read the bind mount at all. The stage
then reports the resulting ``EACCES`` as whatever its argument parser makes of an unreadable path,
typically a misleading "file not found".

Which runtime is in use is a property of the execution host, not of a run, so the default
``"auto"`` probes it instead of asking committed run configs to carry a host detail. One
``docker info`` document serves both the Docker and Podman schemas, since a rootless podman
aliased to ``docker`` is a supported runtime. The Snakefile resolves this once at parse time, so
the answer describes the host that plans the jobs rather than any host that later runs one.
"""

from __future__ import annotations

import json
import logging
import subprocess

logger = logging.getLogger(__name__)

# Both carry their own trailing space, since they sit between the device flag and -e HOME=/tmp in the Snakefile prefix.
HOST_USER_FLAG = '--user "$(id -u):$(id -g)" '
ROOT_USER_FLAG = "--user 0:0 "

# `docker info` spellings for a rootless daemon and for rootful user-namespace remapping.
_ROOTLESS_MARKER = "name=rootless"
_USERNS_MARKER = "name=userns"

# Local daemon round trip, ~65 ms. The timeout only guards a wedged daemon, where failing open to the rootful default
# is better than hanging every snakemake invocation.
_PROBE_TIMEOUT_S = 10.0

_MODES = ("auto", "host", "root")


def _probe_daemon() -> str:
    """Ask the container runtime how it maps the invoking user onto container uids.

    Returns
    -------
    str
        ``"rootless"`` when the runtime maps the caller onto container uid 0, ``"userns"`` for a rootful Docker daemon
        with user-namespace remapping, ``"rootful"`` for a plain rootful daemon, and ``"unknown"`` when the runtime
        cannot be reached or its answer cannot be read.

    Notes
    -----
    Docker reports the mapping in a ``SecurityOptions`` list and Podman under ``host.security.rootless``, so one
    document answers for both. Failure is deliberately not fatal and resolves to the rootful default, because the
    Snakefile calls this at parse time and a dry run planning commands on a machine with no runtime at all is
    supported.
    """
    try:
        completed = subprocess.run(
            ["docker", "info", "--format", "{{json .}}"],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("Could not run `docker info` to detect the container runtime (%s); assuming rootful.", exc)
        return "unknown"

    if completed.returncode != 0:
        logger.debug(
            "`docker info` exited %d while detecting the container runtime (%s); assuming rootful.",
            completed.returncode,
            completed.stderr.strip(),
        )
        return "unknown"

    try:
        info = json.loads(completed.stdout)
    except ValueError as exc:
        logger.debug("Could not parse the `docker info` document (%s); assuming rootful.", exc)
        return "unknown"

    if not isinstance(info, dict):
        logger.debug("`docker info` reported %r rather than a document; assuming rootful.", info)
        return "unknown"

    options = info.get("SecurityOptions")
    if isinstance(options, list):
        if _ROOTLESS_MARKER in options:
            return "rootless"
        return "userns" if _USERNS_MARKER in options else "rootful"

    host = info.get("host")
    security = host.get("security") if isinstance(host, dict) else None
    rootless = security.get("rootless") if isinstance(security, dict) else None
    if isinstance(rootless, bool):
        return "rootless" if rootless else "rootful"

    logger.debug("`docker info` carried no recognised user mapping; assuming rootful.")
    return "unknown"


def resolve_docker_user(config: dict) -> str:
    """Turn a run config's ``docker_user`` key into the ``--user`` fragment of the docker run prefix.

    Parameters
    ----------
    config : dict
        Parsed run config. ``docker_user`` is ``"auto"`` (the default) to match the execution host's runtime,
        ``"host"`` to always run the container as the invoking user, or ``"root"`` to always run it as container root.

    Returns
    -------
    str
        Either ``'--user "$(id -u):$(id -g)" '`` or ``'--user 0:0 '``, ready to be concatenated into the prefix.

    Raises
    ------
    ValueError
        If ``docker_user`` is not one of ``"auto"``, ``"host"`` or ``"root"``, or if ``"auto"`` finds a rootful daemon
        with user-namespace remapping, where neither mode gives the invoking user ownership of the bind mount.

    Notes
    -----
    ``"root"`` names container uid 0 explicitly rather than omitting the flag, so the resolved user stays the same if
    a stage image ever declares a non-root ``USER``. Under a rootless runtime that uid is the invoking host user.
    """
    value = config.get("docker_user", "auto")
    if not isinstance(value, str) or value.strip().lower() not in _MODES:
        raise ValueError(
            f"config['docker_user'] is {value!r}; write one of {', '.join(repr(m) for m in _MODES)}. "
            '"auto" matches the execution host\'s runtime, "host" always runs the container as the invoking user, '
            'and "root" always runs it as container root.'
        )

    mode = value.strip().lower()
    if mode == "auto":
        daemon = _probe_daemon()
        if daemon == "userns":
            raise ValueError(
                "The Docker daemon remaps container users onto subordinate host uids, so neither container root nor "
                'the invoking uid owns the bind-mounted repo. Set docker_user to "host" or "root" to pin one and '
                "accept the file ownership that follows."
            )
        mode = "root" if daemon == "rootless" else "host"
    return HOST_USER_FLAG if mode == "host" else ROOT_USER_FLAG
