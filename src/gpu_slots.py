"""GPU slot allocator wrapping one pipeline job's container launch.

Pins each concurrent job to one free slot so that no two jobs claim the same share of a device. Takes
an exclusive ``flock`` on one lock file per slot, substitutes the acquired id into the wrapped
command's ``@GPU_ID@`` tokens, runs the command as a plain argv list with the lock held, and exits
with its status. Progress lines go to stderr. Run it from the repository root as
``python -m src.gpu_slots --gpu-ids 4,5 --jobs-per-gpu 2 --lock-dir DIR -- COMMAND...``. The Multi-GPU
scheduling section of ``docs/mvp-pipeline.md`` covers how the Snakefile drives this and what the run
config's keys mean.

``--gpu-ids all`` resolves at job start to the entries of ``CUDA_VISIBLE_DEVICES`` when that variable
is set, taken verbatim as integer indices or GPU UUID names, and to every index ``nvidia-smi`` reports
otherwise. MIG device names are unsupported, since a slot hands a job one whole GPU.

The filesystem holding the lock directory must grant locks, and one that refuses stops the run. A
SIGKILLed wrapper orphans its docker client child, so a container can briefly outlive the freed slot;
Snakemake cancels jobs by signalling the process group, which reaches the docker client too.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import random
import subprocess
import sys
import time
from pathlib import Path

from .utils import parse_gpu_pool

GPU_ID_TOKEN = "@GPU_ID@"


def _pool_from_visible_devices(raw: str) -> tuple[str, ...]:
    """Read a set ``CUDA_VISIBLE_DEVICES`` value as the pool it allows this job.

    Parameters
    ----------
    raw : str
        The variable's value, whose comma-separated entries are integer indices or GPU UUID names such as
        ``"GPU-c64146c9-1234-5678-9abc-def012345678"``.

    Returns
    -------
    tuple[str, ...]
        The entries as written with surrounding whitespace removed, in the order the variable lists them.

    Raises
    ------
    RuntimeError
        If the variable holds no entry, has an empty entry, or repeats one.

    Notes
    -----
    A UUID name has no numeric form to normalize to, so entries are kept verbatim and compared as written. Two
    spellings of one device, such as an index and that device's UUID, therefore pass as distinct entries.
    """
    if not raw.strip():
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES is set but empty, so --gpu-ids all resolves to no device at all. Unset it to use "
            "every GPU nvidia-smi reports on this host, or set it to the ids this run may take, such as \"4,5\"."
        )

    ids = [entry.strip() for entry in raw.split(",")]
    if not all(ids):
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES has an empty entry in {raw!r}, so --gpu-ids all cannot tell which devices this "
            "run may take. Write its entries as \"4,5\" or as GPU UUID names, with no doubled or trailing comma."
        )

    # A repeated entry hands the same device two job slots, silently oversubscribing it while the run still looks
    # evenly spread across the pool.
    duplicates = sorted({gpu_id for gpu_id in ids if ids.count(gpu_id) > 1})
    if duplicates:
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES names device(s) {duplicates} more than once in {raw!r}, which would give one "
            "device twice its share of the run. List each device once."
        )
    return tuple(ids)


def _pool_from_nvidia_smi() -> tuple[str, ...]:
    """Enumerate the execution host's GPU indices with ``nvidia-smi``.

    Returns
    -------
    tuple[str, ...]
        Every index nvidia-smi reports, normalized by ``parse_gpu_pool``, in the order of the query's output.

    Raises
    ------
    RuntimeError
        If nvidia-smi is not on PATH, hangs, exits nonzero, reports no GPU, or reports lines that are not an id list.
    """
    query = ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"]
    try:
        completed = subprocess.run(query, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "nvidia-smi gave no answer within 30 seconds while enumerating this host's GPUs for --gpu-ids all, "
            "which usually means a wedged driver. Repair the driver installation, or name the ids this run may "
            "take in CUDA_VISIBLE_DEVICES or in the run config's gpu_ids to skip enumeration."
        ) from exc
    except FileNotFoundError as exc:
        raise RuntimeError(
            "--gpu-ids all needs nvidia-smi to enumerate this host's GPUs, but no nvidia-smi is on PATH. Set "
            "CUDA_VISIBLE_DEVICES to the ids this run may take, or set the run config's gpu_ids to an explicit id "
            "list such as \"4,5,6,7\", or set it to null to run the cpu images."
        ) from exc

    if completed.returncode != 0:
        # The stderr of a driver mismatch runs to several lines, so it is collapsed to one bounded fragment that
        # still names the failure.
        reported = " ".join(completed.stderr.split())[:200] or "nothing on stderr"
        raise RuntimeError(
            f"nvidia-smi exited with status {completed.returncode} while enumerating this host's GPUs for --gpu-ids "
            f"all, reporting {reported!r}. Repair the driver installation, or name the ids this run may take in "
            "CUDA_VISIBLE_DEVICES or in the run config's gpu_ids."
        )

    indices = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not indices:
        raise RuntimeError(
            "nvidia-smi reported no GPU on this host, so --gpu-ids all resolves to no device at all. Set the run "
            "config's gpu_ids to null to run the cpu images instead."
        )
    try:
        return parse_gpu_pool(",".join(indices))
    except ValueError as exc:
        raise RuntimeError(
            f"nvidia-smi reported {indices} as this host's GPU indices, which is not a usable id list. Name the ids "
            "this run may take in CUDA_VISIBLE_DEVICES or in the run config's gpu_ids to skip enumeration."
        ) from exc


def resolve_wrapper_pool(value: str) -> tuple[str, ...]:
    """Turn the ``--gpu-ids`` flag into the ids this job may be pinned to.

    Parameters
    ----------
    value : str
        The flag as written, either ``"all"`` to resolve the execution host's pool now or a comma-separated id list
        to take as given.

    Returns
    -------
    tuple[str, ...]
        The ids concurrent jobs are pinned to, in the order their slots are tried.

    Raises
    ------
    RuntimeError
        If ``"all"`` cannot be resolved on this host, which is an environment failure rather than a usage error.
    ValueError
        If an explicit id list is unusable.
    """
    if value.strip().lower() == "all":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is None:
            return _pool_from_nvidia_smi()
        return _pool_from_visible_devices(visible)
    return parse_gpu_pool(value)


def split_wrapper_argv(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split arguments at the first literal ``--`` into wrapper flags and the wrapped command.

    Parameters
    ----------
    argv : list[str]
        Arguments after the program name, holding this wrapper's own flags, a ``--`` separator, and the command to
        run under the acquired slot.

    Returns
    -------
    tuple[list[str], list[str]]
        The wrapper's own flags and the wrapped command's argv.

    Raises
    ------
    ValueError
        If no ``--`` separator is present, or if nothing follows it.

    Notes
    -----
    The split is manual rather than delegated to the argument parser, so a flag spelled the same way on both sides
    always stays on the side it was written. Only the first separator is consumed, leaving any later ``--`` to the
    wrapped command.
    """
    if "--" not in argv:
        raise ValueError(
            "gpu_slots needs a '--' separator between its own flags and the command to run, as in "
            "'--gpu-ids 4,5 --jobs-per-gpu 2 --lock-dir .snakemake/gpu_slots -- docker run ...'."
        )
    separator = argv.index("--")
    command = argv[separator + 1 :]
    if not command:
        raise ValueError(
            "gpu_slots was given nothing after its '--' separator; write the command to run under the acquired slot "
            "after it, as in '-- docker run ...'."
        )
    return argv[:separator], command


def slot_names(pool: tuple[str, ...], jobs_per_gpu: int) -> list[tuple[str, str]]:
    """List every GPU slot as a ``(gpu_id, lock filename)`` pair, in the order slots are tried.

    Parameters
    ----------
    pool : tuple[str, ...]
        Normalized GPU ids that concurrent jobs are pinned to.
    jobs_per_gpu : int
        How many concurrent jobs may share one pooled id.

    Returns
    -------
    list[tuple[str, str]]
        One pair per slot, ordered by slot index first and id second. Every id's first slot therefore comes before
        any id's second slot, which spreads concurrent jobs across the whole pool before doubling up on a device.
    """
    return [(gpu_id, f"gpu{gpu_id}_slot{index}.lock") for index in range(jobs_per_gpu) for gpu_id in pool]


def acquire_slot(
    lock_dir: Path, pool: tuple[str, ...], jobs_per_gpu: int, poll_interval: float
) -> tuple[str, str, int]:
    """Block until one GPU slot is free, then take and hold its lock.

    Parameters
    ----------
    lock_dir : Path
        Existing directory holding one lock file per slot.
    pool : tuple[str, ...]
        Normalized GPU ids that concurrent jobs are pinned to.
    jobs_per_gpu : int
        How many concurrent jobs may share one pooled id.
    poll_interval : float
        Seconds to wait before rescanning the slots after a full pass found none free.

    Returns
    -------
    tuple[str, str, int]
        The acquired GPU id, the name of the lock file taken, and the open file descriptor holding the lock. The
        caller keeps the descriptor open for as long as the job needs the device and closes it to free the slot.

    Notes
    -----
    A wait is reported once rather than every pass, so a job queued behind a long-running one leaves a single line in
    its log instead of a message per poll.
    """
    candidates = slot_names(pool, jobs_per_gpu)
    reported_wait = False
    while True:
        for gpu_id, lock_name in candidates:
            fd = os.open(lock_dir / lock_name, os.O_CREAT | os.O_RDWR, 0o644)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                # Only a slot held by another job is contention. Any other error means this filesystem hands out no
                # locks at all, which must not be retried as if a slot were merely busy.
                os.close(fd)
                continue
            except OSError:
                os.close(fd)
                raise
            return gpu_id, lock_name, fd

        if not reported_wait:
            print(
                f"[gpu_slots] waiting for a free GPU slot (pool={','.join(pool)}, jobs_per_gpu={jobs_per_gpu})",
                file=sys.stderr,
                flush=True,
            )
            reported_wait = True
        # Jobs released together would otherwise rescan the slots in lockstep and keep contending for the same lock
        # file, so each waiter draws its own unseeded delay to fall out of step with the others.
        time.sleep(poll_interval + random.uniform(0.0, poll_interval / 4.0))


def _build_parser() -> argparse.ArgumentParser:
    """Build the parser for the flags written before the ``--`` separator.

    Returns
    -------
    argparse.ArgumentParser
        Parser for the wrapper's own flags, which never sees the wrapped command's arguments.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.gpu_slots",
        description="Run a command pinned to one free GPU slot from a pool, holding the slot for the command's life.",
    )
    parser.add_argument(
        "--gpu-ids",
        required=True,
        help='Comma-separated GPU ids to share out, such as "4,5,6,7", or "all" for the ids this host offers.',
    )
    parser.add_argument("--jobs-per-gpu", required=True, type=int, help="Concurrent jobs allowed to share one id.")
    parser.add_argument("--lock-dir", required=True, type=Path, help="Directory holding one lock file per slot.")
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="Seconds between rescans of the slots while every one of them is taken (default: %(default)s).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Acquire a GPU slot, run the wrapped command pinned to it, and report the command's exit code.

    Parameters
    ----------
    argv : list[str] | None
        Arguments after the program name. ``None`` reads them from ``sys.argv``.

    Returns
    -------
    int
        The wrapped command's exit code, 1 if ``--gpu-ids all`` could not be resolved on this host or the lock
        directory's filesystem hands out no locks, 127 if the command's executable was not found, or ``128 + N`` if
        signal N killed it, matching how a shell reports the same outcomes.

    Raises
    ------
    ValueError
        If the ``--`` separator or the command after it is missing, if ``--gpu-ids`` is not a usable id list, or if
        ``--jobs-per-gpu`` is below 1.
    """
    wrapper_args, command = split_wrapper_argv(sys.argv[1:] if argv is None else argv)
    args = _build_parser().parse_args(wrapper_args)

    # An unresolvable "all" pool says nothing about how the run was written, so it is reported as a plain message
    # and a failing status rather than as a traceback the user would have to read past.
    try:
        pool = resolve_wrapper_pool(args.gpu_ids)
    except RuntimeError as exc:
        print(f"[gpu_slots] {exc}", file=sys.stderr, flush=True)
        return 1

    if args.jobs_per_gpu < 1:
        raise ValueError(f"--jobs-per-gpu must be at least 1, got {args.jobs_per_gpu}.")

    args.lock_dir.mkdir(parents=True, exist_ok=True)
    try:
        gpu_id, lock_name, fd = acquire_slot(args.lock_dir, pool, args.jobs_per_gpu, args.poll_interval)
    except OSError as exc:
        print(
            f"[gpu_slots] cannot lock a slot under {args.lock_dir} ({exc.strerror}), so no job can be pinned to a "
            "device. Some network filesystems, such as Lustre mounted without flock support, answer every lock this "
            "way. Run the pipeline from a working copy on a filesystem that grants locks, such as local disk.",
            file=sys.stderr,
            flush=True,
        )
        return 1
    print(f"[gpu_slots] acquired GPU {gpu_id} ({lock_name})", file=sys.stderr, flush=True)

    pinned_command = [arg.replace(GPU_ID_TOKEN, gpu_id) for arg in command]
    try:
        completed = subprocess.run(pinned_command)
    except FileNotFoundError:
        print(f"[gpu_slots] cannot run {pinned_command[0]!r}; no such executable on PATH.", file=sys.stderr, flush=True)
        return 127
    finally:
        # The slot stays taken until the command has exited, since closing the descriptor releases the lock.
        os.close(fd)

    if completed.returncode < 0:
        killing_signal = -completed.returncode
        return 128 + killing_signal
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main())
