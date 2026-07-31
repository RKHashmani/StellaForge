"""Container user resolution

``resolve_docker_user`` decides whether every stage's ``docker run`` carries
``--user "$(id -u):$(id -g)"``. The flag exists so that a container's writes into the
bind-mounted repo land host-owned rather than root-owned, which is what a *rootful*
daemon needs. A *rootless* daemon already maps the invoking host user onto uid 0 inside
its own user namespace, so it gives that ownership for free, and passing the flag there
is actively harmful, since the container becomes an unprivileged subuid that cannot read
the bind mount at all. The stage then reports the resulting ``EACCES`` as whatever its
argument parser makes of an unreadable path, typically a misleading "file not found".

Which daemon is in use is a property of the execution host, not of a run, so the default
``"auto"`` detects it instead of asking committed run configs to carry a host detail. The
Snakefile resolves this once at parse time.
"""

from __future__ import annotations

import json
import logging
import subprocess
from functools import lru_cache

logger = logging.getLogger(__name__)

# Placed between the device flag and -e HOME=/tmp in the Snakefile's prefix, so it carries its own trailing space and
# drops out cleanly when empty.
HOST_USER_FLAG = '--user "$(id -u):$(id -g)" '

# `docker info` spelling for a rootless daemon.
_ROOTLESS_MARKER = "name=rootless"

# Local daemon round trip, ~65 ms. The timeout only guards a wedged daemon, where failing open to the rootful default
# is better than hanging every snakemake invocation.
_PROBE_TIMEOUT_S = 10.0

_MODES = ("auto", "host", "root")


def _probe_rootless() -> bool:
    """Ask the Docker daemon whether it is running rootless.

    Returns
    -------
    bool
        True only when the daemon reports the rootless security option. Any failure to reach or read the daemon
        returns False, which resolves ``"auto"`` to the rootful default and so preserves the historical command.

    Notes
    -----
    Failure is deliberately not fatal. The Snakefile calls this at parse time, and a dry run planning commands on a
    machine with no Docker at all is a supported workflow.
    """
    try:
        completed = subprocess.run(
            ["docker", "info", "-f", "{{json .SecurityOptions}}"],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("Could not run `docker info` to detect a rootless daemon (%s); assuming rootful.", exc)
        return False

    if completed.returncode != 0:
        logger.debug(
            "`docker info` exited %d while detecting a rootless daemon (%s); assuming rootful.",
            completed.returncode,
            completed.stderr.strip(),
        )
        return False

    try:
        options = json.loads(completed.stdout)
    except ValueError as exc:
        logger.debug("Could not parse `docker info` security options (%s); assuming rootful.", exc)
        return False

    if not isinstance(options, list):
        logger.debug("`docker info` reported no security options list (%r); assuming rootful.", options)
        return False

    return _ROOTLESS_MARKER in options


@lru_cache(maxsize=1)
def daemon_is_rootless() -> bool:
    """Return whether the Docker daemon is rootless, probing it at most once per process.

    Returns
    -------
    bool
        True when the daemon reports the rootless security option, False otherwise or when it cannot be reached.
    """
    return _probe_rootless()


def resolve_docker_user(config: dict) -> str:
    """Turn a run config's ``docker_user`` key into the ``--user`` fragment of the docker run prefix.

    Parameters
    ----------
    config : dict
        Parsed run config. ``docker_user`` is ``"auto"`` (the default) to match the execution host's daemon,
        ``"host"`` to always run the container as the invoking user, or ``"root"`` to always leave the image's own
        user in place.

    Returns
    -------
    str
        Either ``'--user "$(id -u):$(id -g)" '`` or the empty string, ready to be concatenated into the prefix.

    Raises
    ------
    ValueError
        If ``docker_user`` is not one of ``"auto"``, ``"host"`` or ``"root"``.

    Notes
    -----
    ``"root"`` omits the flag rather than passing ``--user 0:0`` because ``stages/Dockerfile`` declares no ``USER``,
    so the image's default user is already root and the two spellings are equivalent.
    """
    value = config.get("docker_user", "auto")
    if not isinstance(value, str) or value.strip().lower() not in _MODES:
        raise ValueError(
            f"config['docker_user'] is {value!r}; write one of {', '.join(repr(m) for m in _MODES)}. "
            '"auto" matches the execution host\'s daemon, "host" always runs the container as the invoking user, '
            'and "root" always leaves the image\'s own user in place.'
        )

    mode = value.strip().lower()
    if mode == "auto":
        mode = "root" if daemon_is_rootless() else "host"
    return HOST_USER_FLAG if mode == "host" else ""
