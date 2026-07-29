"""Tests for ``src.gpu_slots`` (the GPU slot allocator wrapping each container launch).

The allocator is the only thing keeping two concurrently scheduled jobs off the same
share of a device, and it enforces that through an exclusive ``flock`` per slot rather
than through bookkeeping the pipeline can inspect. The argument splitting, pool
resolution and slot naming are checked in-process, while everything about mutual
exclusion is checked by launching real wrappers as subprocesses, because a lock taken
inside one interpreter would be re-entrant and would prove nothing.

Resolving ``--gpu-ids all`` reads the host the job landed on, so every test that touches
it sets ``CUDA_VISIBLE_DEVICES`` and ``PATH`` itself, pointing ``PATH`` at a directory
holding only a fake ``nvidia-smi``. The answers are therefore the same on a workstation
with GPUs and on CI with none.

The wrapped payloads are plain Python scripts that record the id they were pinned to
and the wall-clock window they held it for. Comparing those windows is what pins the
exclusion guarantee, and every recorded window sits strictly inside its wrapper's
lock-held window, so overlapping records would mean overlapping locks.

Every subprocess wait carries a timeout, so a regression that deadlocks the allocator
fails the suite instead of hanging it.
"""

from __future__ import annotations

import contextlib
import errno
import fcntl
import os
import re
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from src.gpu_slots import GPU_ID_TOKEN, acquire_slot, main, resolve_wrapper_pool, slot_names, split_wrapper_argv

REPO_ROOT = Path(__file__).resolve().parents[2]
# Generous enough that a loaded machine never trips it, short enough that a deadlocked allocator fails quickly.
WRAPPER_TIMEOUT = 60.0
# Long enough that several waves of jobs are forced to queue behind each other, short enough to keep the suite quick.
HOLD_SECONDS = 0.5

# Records the id substituted into its argv and the window it held the slot for. time.monotonic() counts from boot on
# Linux and macOS, so the values are comparable between the workers.
RECORD_INTERVAL_PAYLOAD = """
import sys
import time
from pathlib import Path

gpu_id, record = sys.argv[1], sys.argv[2]
start = time.monotonic()
time.sleep({hold})
Path(record).write_text(f"{{gpu_id}} {{start}} {{time.monotonic()}}\\n")
""".format(hold=HOLD_SECONDS)

# Holds its slot until killed, publishing its pid so the test can reap it after the wrapper above it is SIGKILLed.
HOLD_UNTIL_KILLED_PAYLOAD = """
import os
import sys
import time
from pathlib import Path

Path(sys.argv[1]).write_text(str(os.getpid()))
time.sleep(600)
"""


def _wrapper_argv(
    lock_dir: Path, gpu_ids: str, jobs_per_gpu: int, command: list[str], poll_interval: float | None = None
) -> list[str]:
    """Build the argv that runs one command under the allocator.

    Parameters
    ----------
    lock_dir : Path
        Directory the allocator creates its per-slot lock files in.
    gpu_ids : str
        Pool passed to ``--gpu-ids``, either a comma-separated id list or ``"all"``.
    jobs_per_gpu : int
        Concurrent jobs allowed to share one id.
    command : list[str]
        Command to run under the acquired slot, placed after the ``--`` separator.
    poll_interval : float, optional
        Seconds between rescans while every slot is taken. ``None`` leaves the allocator's own default.

    Returns
    -------
    list[str]
        Full argv, invoking the allocator as a module so it runs exactly as the Snakefile spells it.
    """
    argv = [
        sys.executable, "-m", "src.gpu_slots",
        "--gpu-ids", gpu_ids,
        "--jobs-per-gpu", str(jobs_per_gpu),
        "--lock-dir", str(lock_dir),
    ]
    if poll_interval is not None:
        argv += ["--poll-interval", str(poll_interval)]
    return [*argv, "--", *command]


def _nvidia_smi_shim(tmp_path: Path, body: str) -> Path:
    """Write a fake ``nvidia-smi`` and return the directory to use as the whole PATH.

    Parameters
    ----------
    tmp_path : Path
        Directory the shim's bin directory is created under.
    body : str
        Shell body standing in for the real query, printing what the tool would print and exiting with the status the
        test is reproducing.

    Returns
    -------
    Path
        Directory holding the shim. Tests set PATH to exactly this directory, so a real nvidia-smi installed on the
        host cannot be reached and the answer does not depend on the machine the suite runs on.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    shim = bin_dir / "nvidia-smi"
    shim.write_text(f"#!/bin/sh\n{body}\n")
    shim.chmod(0o755)
    return bin_dir


def _read_records(records: list[Path]) -> list[tuple[str, float, float]]:
    """Parse the ``<gpu id> <start> <end>`` line each worker wrote.

    Parameters
    ----------
    records : list[Path]
        One record file per worker.

    Returns
    -------
    list[tuple[str, float, float]]
        The id each worker was pinned to and the window it held its slot for.
    """
    parsed = []
    for record in records:
        gpu_id, start, end = record.read_text().split()
        parsed.append((gpu_id, float(start), float(end)))
    return parsed


def _max_overlap(windows: list[tuple[float, float]]) -> int:
    """Count the largest number of windows that were open at the same instant.

    Parameters
    ----------
    windows : list[tuple[float, float]]
        Start and end times, in any order.

    Returns
    -------
    int
        The peak simultaneous count. Windows that merely touch at an endpoint do not count as overlapping, because a
        released slot is immediately available to the next waiter.
    """
    events = [(start, 1) for start, _ in windows] + [(end, -1) for _, end in windows]
    events.sort()
    peak = 0
    open_now = 0
    for _, delta in events:
        open_now += delta
        peak = max(peak, open_now)
    return peak


def _run_workers(tmp_path: Path, gpu_ids: str, jobs_per_gpu: int, workers: int) -> list[tuple[str, float, float]]:
    """Launch ``workers`` wrappers at once over one pool and return what each one recorded.

    Parameters
    ----------
    tmp_path : Path
        Directory holding the payload script, the lock files, and the per-worker records.
    gpu_ids : str
        Comma-separated pool every wrapper shares.
    jobs_per_gpu : int
        Concurrent jobs allowed to share one id.
    workers : int
        Number of wrappers to launch, normally more than the pool has slots so some must queue.

    Returns
    -------
    list[tuple[str, float, float]]
        One ``(gpu id, start, end)`` triple per worker.
    """
    payload = tmp_path / "record_interval.py"
    payload.write_text(RECORD_INTERVAL_PAYLOAD)
    records = [tmp_path / f"worker_{index}.txt" for index in range(workers)]
    # A short poll interval keeps a freed slot from sitting idle for most of the test's runtime.
    procs = [
        subprocess.Popen(
            _wrapper_argv(tmp_path / "locks", gpu_ids, jobs_per_gpu,
                          [sys.executable, str(payload), GPU_ID_TOKEN, str(record)], poll_interval=0.05),
            cwd=REPO_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        for record in records
    ]
    try:
        for proc in procs:
            assert proc.wait(timeout=WRAPPER_TIMEOUT) == 0
    finally:
        for proc in procs:
            proc.kill()
    return _read_records(records)


def _wait_for_text(log: Path, needle: str, timeout: float = WRAPPER_TIMEOUT) -> None:
    """Block until ``needle`` appears in ``log``, failing the test if it never does.

    Parameters
    ----------
    log : Path
        File the wrapper's stderr is redirected to.
    needle : str
        Text to wait for. The allocator flushes each of its messages, so a match means the event has happened.
    timeout : float
        Seconds to wait before failing.

    Raises
    ------
    AssertionError
        If the text has not appeared within ``timeout``.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if needle in log.read_text():
            return
        time.sleep(0.02)
    raise AssertionError(f"{needle!r} never appeared in {log}:\n{log.read_text()}")


def _flock_refusing_every_lock(limit: int = 10) -> Callable[[int, int], None]:
    """Build a stand-in for ``fcntl.flock`` that reports a filesystem granting no locks at all.

    Parameters
    ----------
    limit : int
        Attempts to allow before the stand-in fails the test outright. The allocator has to let the first refusal
        out, so a retry means a failure that was not contention was mistaken for a busy slot. Bounding the attempts
        turns that regression into a prompt failure instead of the endless wait it would be in a real run.

    Returns
    -------
    Callable[[int, int], None]
        Replacement for ``fcntl.flock`` that always raises ``ENOTSUP``, which is how a network mount without lock
        support answers.
    """
    attempts = {"n": 0}

    def refuse(fd: int, operation: int) -> None:
        attempts["n"] += 1
        if attempts["n"] > limit:
            raise AssertionError("the allocator retried a lock failure that was not contention")
        raise OSError(errno.ENOTSUP, "Operation not supported")

    return refuse


def _kill_recorded_pid(pid_file: Path) -> None:
    """Kill the process whose pid was written to ``pid_file``, if it is still running.

    Parameters
    ----------
    pid_file : Path
        File a payload wrote its own pid into. A missing file means the payload never got that far.
    """
    with contextlib.suppress(FileNotFoundError, ValueError, ProcessLookupError, PermissionError):
        os.kill(int(pid_file.read_text()), signal.SIGKILL)


# The wrapper's own flags and the command it runs are both plain argv, so the split has to be positional. This pins the
# happy path and the two shapes that leave the allocator with nothing to run.
def test_split_wrapper_argv_splits_at_the_separator() -> None:
    assert split_wrapper_argv(["--gpu-ids", "4,5", "--", "docker", "run"]) == (["--gpu-ids", "4,5"], ["docker", "run"])


# Only the first separator belongs to the allocator. A wrapped command that takes its own '--' must receive it intact,
# which is what lets a pipeline job pass flags through to the program inside its container.
def test_split_wrapper_argv_keeps_a_later_separator_with_the_command() -> None:
    wrapper, command = split_wrapper_argv(["--lock-dir", "d", "--", "docker", "run", "img", "--", "--flag"])
    assert wrapper == ["--lock-dir", "d"]
    assert command == ["docker", "run", "img", "--", "--flag"]


# Without a separator the allocator cannot tell its own flags from the command, and with nothing after one there is no
# command to hold the slot for. Both are caller mistakes that must fail loudly rather than acquire a slot for nothing.
@pytest.mark.parametrize("argv", [["--gpu-ids", "4,5", "docker", "run"], ["--gpu-ids", "4,5", "--"]])
def test_split_wrapper_argv_rejects_a_missing_command(argv: list[str]) -> None:
    with pytest.raises(ValueError, match="'--'"):
        split_wrapper_argv(argv)


# A set CUDA_VISIBLE_DEVICES is how a shared host hands one user their share of it, and exporting it is the supported
# way to hold "all" to that share. Its entries are taken verbatim because they may be GPU UUID names with no numeric
# form to normalize to, and because the docker client accepts either spelling as written. The written order is kept,
# since it is the order slots are tried.
@pytest.mark.parametrize(
    ("visible", "expected"),
    [
        ("4, 5", ("4", "5")),
        ("1,0", ("1", "0")),
        ("07", ("07",)),
        ("GPU-abc,GPU-def", ("GPU-abc", "GPU-def")),
    ],
)
def test_all_reads_a_set_visible_devices_verbatim(
    monkeypatch: pytest.MonkeyPatch, visible: str, expected: tuple[str, ...]
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    assert resolve_wrapper_pool("all") == expected


# The Snakefile always writes the flag as "all", but a hand-run wrapper may capitalize it or leave a stray space around
# it. Either would otherwise be read as an id list and rejected as unparseable.
@pytest.mark.parametrize("spelling", ["all", "ALL", " all "])
def test_all_is_recognized_however_it_is_written(monkeypatch: pytest.MonkeyPatch, spelling: str) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")
    assert resolve_wrapper_pool(spelling) == ("4", "5")


# A variable that names no usable pool must stop the job rather than quietly resolve to something else, since falling
# back to every GPU of the host would take devices this run was never allowed. The duplicate case is the one that would
# otherwise hand a single device two slots while the run still looks evenly spread across the pool.
@pytest.mark.parametrize(
    ("visible", "expected"),
    [
        ("", "CUDA_VISIBLE_DEVICES is set but empty"),
        ("   ", "CUDA_VISIBLE_DEVICES is set but empty"),
        ("4,,5", "has an empty entry in '4,,5'"),
        ("4,", "has an empty entry"),
        ("4,4", "names device(s) ['4'] more than once"),
    ],
)
def test_all_rejects_a_visible_devices_value_that_names_no_usable_pool(
    monkeypatch: pytest.MonkeyPatch, visible: str, expected: str
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    with pytest.raises(RuntimeError, match=re.escape(expected)):
        resolve_wrapper_pool("all")


# With the variable unset, "all" means every GPU the host reports, which is read from nvidia-smi at job start rather
# than at parse time so one config can be run on hosts with different device counts. The reported indices go through the
# same normalization an explicit pool does, so a padded index and a plain one name one slot and not two.
def test_all_enumerates_the_host_when_visible_devices_is_unset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("PATH", str(_nvidia_smi_shim(tmp_path, "echo 0\necho 01")))

    assert resolve_wrapper_pool("all") == ("0", "1")


# Enumeration failures are environment failures rather than mistakes in how the run was written, so each one has to be
# reported as such instead of resolving to a pool the job would then run on. The cases are a driver that reports no
# device, a query that answers with prose, and a driver mismatch that exits nonzero.
@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("exit 0", "reported no GPU on this host"),
        ("echo 'No devices were found'", "not a usable id list"),
        ("echo 'Driver/library version mismatch' >&2\nexit 9", "exited with status 9"),
    ],
)
def test_all_reports_an_unusable_nvidia_smi_as_an_environment_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, body: str, expected: str
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("PATH", str(_nvidia_smi_shim(tmp_path, body)))

    with pytest.raises(RuntimeError, match=re.escape(expected)):
        resolve_wrapper_pool("all")


# A host with no nvidia-smi at all is the case a user hits by asking for GPUs on a machine that has none, so the message
# has to name the missing tool rather than surface as a bare file-not-found from deep inside the allocator.
def test_all_reports_a_missing_nvidia_smi(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    empty_bin = tmp_path / "empty_bin"
    empty_bin.mkdir()
    monkeypatch.setenv("PATH", str(empty_bin))

    with pytest.raises(RuntimeError, match="nvidia-smi"):
        resolve_wrapper_pool("all")


# Ids named in the run config are the run's own allocation, so they must be used as written even on a host whose
# CUDA_VISIBLE_DEVICES says otherwise. Reading the variable here would silently move the run onto devices it was not
# given, and would make one config resolve to different pools on different login shells.
def test_an_explicit_pool_ignores_visible_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "9")
    assert resolve_wrapper_pool("0,1") == ("0", "1")


# Slots are tried in the returned order, so every id's first slot must come before any id's second. That ordering is
# what spreads concurrent jobs across the whole pool before doubling up on a device that is already busy.
def test_slot_names_spread_across_the_pool_before_doubling_up() -> None:
    assert slot_names(("4", "5"), 2) == [
        ("4", "gpu4_slot0.lock"),
        ("5", "gpu5_slot0.lock"),
        ("4", "gpu4_slot1.lock"),
        ("5", "gpu5_slot1.lock"),
    ]


# The whole point of the wrapper is that the command it runs sees a real id in place of the token and that the command's
# exit status reaches Snakemake unchanged. This runs one wrapper end to end and checks the substituted id came from the
# pool, that a non-zero payload status is passed through rather than swallowed, and that the acquired device is reported
# on stderr where the rule's tee pipe records it.
def test_wrapper_substitutes_the_token_and_passes_the_exit_code_through(tmp_path: Path) -> None:
    command = [sys.executable, "-c", "import sys; print(sys.argv[1]); raise SystemExit(3)", GPU_ID_TOKEN]
    result = subprocess.run(
        _wrapper_argv(tmp_path / "locks", "4,5", 1, command),
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=WRAPPER_TIMEOUT,
    )
    assert result.stdout.strip() in ("4", "5"), result
    assert result.returncode == 3, result
    assert "acquired GPU" in result.stderr, result


# A missing executable and a killed command are reported the way a shell reports them, so a job that dies inside its
# container is not mistaken for a clean run. The signal case also exercises the release path, since the slot must be
# freed even though the command never returned normally.
@pytest.mark.parametrize(
    ("command", "expected"),
    [
        (["driftless-star-no-such-executable"], 127),
        ([sys.executable, "-c", "import os, signal; os.kill(os.getpid(), signal.SIGTERM)"], 143),
    ],
)
def test_wrapper_reports_shell_style_failure_codes(tmp_path: Path, command: list[str], expected: int) -> None:
    result = subprocess.run(
        _wrapper_argv(tmp_path / "locks", "4", 1, command),
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=WRAPPER_TIMEOUT,
    )
    assert result.returncode == expected, result


# The Snakefile plans "all" as the word itself, so the resolution, the slot files it implies and the substituted id all
# have to work through the real command line and not only through the resolver. The wrapper is given a two-id share of
# the host in its own environment, and the id its command prints must be one of those two rather than the word.
def test_all_reaches_the_wrapped_command_as_a_real_id(tmp_path: Path) -> None:
    command = [sys.executable, "-c", "import sys; print(sys.argv[1]); raise SystemExit(3)", GPU_ID_TOKEN]
    result = subprocess.run(
        _wrapper_argv(tmp_path / "locks", "all", 1, command),
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=WRAPPER_TIMEOUT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "0,1"},
    )
    assert result.stdout.strip() in ("0", "1"), result
    assert result.returncode == 3, result
    assert "acquired GPU" in result.stderr, result


# A host that cannot be resolved says nothing about how the run was written, so the wrapper reports it as a plain line
# on the stderr the job's tee pipe records and fails, instead of raising a traceback the user has to read past. The
# command must not run at all, since running it unpinned would put it on whichever devices the container defaults to.
def test_an_unresolvable_all_pool_fails_before_running_the_command(tmp_path: Path) -> None:
    marker = tmp_path / "payload_ran.txt"
    command = [sys.executable, "-c", "import sys; from pathlib import Path; Path(sys.argv[1]).write_text('ran')",
               str(marker)]
    result = subprocess.run(
        _wrapper_argv(tmp_path / "locks", "all", 1, command),
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=WRAPPER_TIMEOUT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    assert result.returncode == 1, result
    assert "[gpu_slots]" in result.stderr, result
    assert "Traceback" not in result.stderr, result
    assert not marker.exists(), result


# Six jobs over a two-id pool with one slot each is the case the allocator exists for. Every job must run, both devices
# must be used rather than one being starved, and no two jobs may ever have held the same id at the same time, which is
# what pins the exclusion the pipeline relies on to keep two containers off one device.
def test_one_slot_per_id_serializes_jobs_on_each_device(tmp_path: Path) -> None:
    parsed = _run_workers(tmp_path, gpu_ids="0,1", jobs_per_gpu=1, workers=6)

    assert len(parsed) == 6
    assert {gpu_id for gpu_id, _, _ in parsed} == {"0", "1"}
    for gpu_id in ("0", "1"):
        windows = [(start, end) for held, start, end in parsed if held == gpu_id]
        assert _max_overlap(windows) <= 1, parsed
    assert _max_overlap([(start, end) for _, start, end in parsed]) <= 2, parsed


# Raising jobs_per_gpu is how a user packs several small jobs onto one device, so the ceiling must rise exactly with it
# and no further. Two ids at two jobs each may run four jobs at once, but never a fifth and never three on one device.
def test_jobs_per_gpu_raises_the_ceiling_by_exactly_that_factor(tmp_path: Path) -> None:
    parsed = _run_workers(tmp_path, gpu_ids="0,1", jobs_per_gpu=2, workers=6)

    assert len(parsed) == 6
    for gpu_id in ("0", "1"):
        windows = [(start, end) for held, start, end in parsed if held == gpu_id]
        assert _max_overlap(windows) <= 2, parsed
    assert _max_overlap([(start, end) for _, start, end in parsed]) <= 4, parsed


# A cancelled or crashed job must not strand its device for the rest of the run. The kernel drops a flock when the
# holding process dies, so this SIGKILLs a wrapper that is holding the only slot of a one-id pool and requires the job
# queued behind it to acquire and finish. The wrapper is killed rather than asked to exit, which is the case no
# user-space cleanup could handle. Its payload is orphaned by that kill and is reaped through the pid it published,
# since the payload never inherits the lock and so cannot keep the slot taken.
def test_a_killed_wrapper_frees_its_slot_for_the_next_job(tmp_path: Path) -> None:
    payload = tmp_path / "hold_until_killed.py"
    payload.write_text(HOLD_UNTIL_KILLED_PAYLOAD)
    pid_file = tmp_path / "holder.pid"
    holder_log = tmp_path / "holder.err"
    waiter_log = tmp_path / "waiter.err"
    lock_dir = tmp_path / "locks"

    with holder_log.open("wb") as holder_stderr, waiter_log.open("wb") as waiter_stderr:
        holder = subprocess.Popen(
            _wrapper_argv(lock_dir, "0", 1, [sys.executable, str(payload), str(pid_file)], poll_interval=0.05),
            cwd=REPO_ROOT, stdout=subprocess.DEVNULL, stderr=holder_stderr,
        )
        waiter = None
        try:
            _wait_for_text(holder_log, "acquired GPU")
            waiter = subprocess.Popen(
                _wrapper_argv(lock_dir, "0", 1, [sys.executable, "-c", "pass"], poll_interval=0.05),
                cwd=REPO_ROOT, stdout=subprocess.DEVNULL, stderr=waiter_stderr,
            )
            # The queued job must report the wait before the holder dies, proving the slot really was taken and that
            # the acquisition below is a release rather than a race the waiter won on its first pass.
            _wait_for_text(waiter_log, "waiting for a free GPU slot")
            holder.kill()
            holder.wait(timeout=WRAPPER_TIMEOUT)
            assert waiter.wait(timeout=WRAPPER_TIMEOUT) == 0, waiter_log.read_text()
        finally:
            holder.kill()
            if waiter is not None:
                waiter.kill()
            _kill_recorded_pid(pid_file)

    assert "acquired GPU 0" in waiter_log.read_text()


# Only a slot another job already holds is contention. A filesystem that grants no locks at all refuses every attempt
# with a different error, and counting that as contention would leave the job polling for a slot it can never take,
# which is an endless wait rather than a failure anyone can see. The scan therefore has to let such an error out on
# the first attempt.
def test_a_filesystem_granting_no_locks_stops_the_scan(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(fcntl, "flock", _flock_refusing_every_lock())
    lock_dir = tmp_path / "locks"
    lock_dir.mkdir()

    with pytest.raises(OSError, match="Operation not supported"):
        acquire_slot(lock_dir, ("4", "5"), 1, poll_interval=0.01)


# The wrapper reports that filesystem the way it reports an unresolvable pool, as one [gpu_slots] line naming the lock
# directory and a failing status rather than a traceback. The command must not run, since running it unpinned would
# put it on whichever devices the container defaults to, which is exactly what the allocator exists to prevent.
def test_a_filesystem_granting_no_locks_fails_before_running_the_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(fcntl, "flock", _flock_refusing_every_lock())
    marker = tmp_path / "payload_ran.txt"
    command = [sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).write_text('ran')"]
    argv = ["--gpu-ids", "4", "--jobs-per-gpu", "1", "--lock-dir", str(tmp_path / "locks"), "--", *command]

    assert main(argv) == 1
    assert "[gpu_slots]" in capsys.readouterr().err
    assert not marker.exists()
