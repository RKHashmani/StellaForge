"""Dry-run tests of the GPU wiring the Snakefile plans into every container launch.

``snakemake -n -p`` resolves the GPU pool config and formats every stage's shell
command without starting a container, so the three modes can be told apart from the
planned text alone on a machine with no GPU and no Docker.

What is pinned here is the boundary between the modes. A cpu run must plan no device
flag at all, while both GPU modes must route every device-taking container launch
through the allocator, because one unwrapped launch would take a device the allocator
believes is free. Steps that only rewrite a file are planned with neither a device nor
the allocator and are counted out, since the guarantee is that no launch reaches a
device unallocated rather than that every launch takes one. The two GPU modes differ
only in where the pool comes from. An explicit list is planned as ids, and "all" is
planned as the word itself, since the host's own ids are resolved by the allocator when
the job starts. The malformed cases pin that the pool is resolved at parse time, so a
typo costs nothing rather than failing hours into a run.

The ``_dry_run`` helper is shared with the DAG tests, which documents the isolation it
applies to keep every write under tmp.
"""

from __future__ import annotations

from pathlib import Path

from tests.e2e.test_dry_run_dag import _dry_run

# Exactly how the Snakefile spells the allocator in front of a wrapped launch, for a pool named as ids and for one left
# to the execution host. The lock directory is a repo-relative path because every job already runs from the repo root,
# so all of a run's jobs share one set of slot locks. The "all" spelling carries a jobs_per_gpu of 1 because the run
# that uses it overrides only gpu_ids and the committed config sets that key to 1.
SLOT_PREFIX = "python -m src.gpu_slots --gpu-ids 4,5 --jobs-per-gpu 2 --lock-dir .snakemake/gpu_slots -- "
ALL_SLOT_PREFIX = "python -m src.gpu_slots --gpu-ids all --jobs-per-gpu 1 --lock-dir .snakemake/gpu_slots -- "

# Scripts that only rewrite a file, so they are planned without a device and without the allocator and are counted
# out of the every-launch-is-wrapped totals below. The guarantee is that no launch reaches a device unallocated,
# not that every launch takes one.
DEVICE_FREE_SCRIPTS = ("relabel_neopax_flux_radius.py",)


def _launch_counts(output: str) -> tuple[int, int, int]:
    """Return (launches taking a device, allocator invocations, launches planned without a device)."""
    device_free = sum(output.count(script) for script in DEVICE_FREE_SCRIPTS)
    return output.count("docker run") - device_free, output.count("src.gpu_slots"), device_free


# The committed config runs on cpu, so the default plan must carry no device flag and no allocator, and must select the
# cpu image variant for every stage. Passing gpu_ids=null on the command line is the same request written the other way,
# and snakemake's config parser keeps it as the plain text "null", so this pins that both spellings plan the same run.
def test_null_pool_plans_cpu_images_with_no_device_flag(tmp_path: Path) -> None:
    result = _dry_run(tmp_path, targets=[], config_overrides=["gpu_ids=null"], printshellcmds=True)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "--gpus" not in output, output
    assert "src.gpu_slots" not in output, output
    assert "stage-1-vmec-cpu" in output, output


# Naming the host instead of a pool still pins every job to one device, so an "all" run must be wrapped exactly like an
# explicit pool, with the word carried through to the allocator and the id token left for it to substitute. The plan
# must never hand a container the whole machine, which is what the retired --gpus all flag did and what would let two
# jobs share a device the allocator believes it gave to one of them.
def test_all_plans_wrapped_containers_pinned_to_one_device(tmp_path: Path) -> None:
    result = _dry_run(tmp_path, targets=[], config_overrides=["gpu_ids=all"], printshellcmds=True)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert ALL_SLOT_PREFIX in output, output
    assert "--gpus device=@GPU_ID@" in output, output
    assert "stage-1-vmec-gpu" in output, output
    assert "--gpus all" not in output, output
    # device_taking == allocated == the device-flag count leaves the device-free launches provably
    # holding no device, since the three totals account for every planned launch.
    device_taking, allocated, _ = _launch_counts(output)
    assert device_taking > 0, output
    assert device_taking == allocated, output
    assert output.count("--gpus device=@GPU_ID@") == allocated, output


# With an explicit pool the guarantee is that no container can reach a device without going through the allocator, so
# the count of planned launches and the count of allocator invocations must match exactly. A single unwrapped launch
# would see a device the allocator has already handed to another job. The token is left unsubstituted in the plan
# because the id is only known once a slot is acquired at run time.
def test_pinned_pool_wraps_every_planned_container(tmp_path: Path) -> None:
    result = _dry_run(tmp_path, targets=[], config_overrides=["gpu_ids=4,5", "jobs_per_gpu=2"], printshellcmds=True)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert SLOT_PREFIX in output, output
    assert "--gpus device=@GPU_ID@" in output, output
    assert "stage-1-vmec-gpu" in output, output
    device_taking, allocated, _ = _launch_counts(output)
    assert device_taking > 0, output
    assert device_taking == allocated, output
    assert output.count("--gpus device=@GPU_ID@") == allocated, output


# A mistyped id must be caught while the DAG is being built, before any job runs, and the message must name the entry
# that could not be read so a long pool with one bad id is actionable.
def test_malformed_pool_fails_at_parse_naming_the_entry(tmp_path: Path) -> None:
    result = _dry_run(tmp_path, targets=[], config_overrides=["gpu_ids=bogus"])
    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert "bogus" in output, output


# Raising jobs_per_gpu without naming ids describes an oversubscription of nothing, which is almost always a forgotten
# gpu_ids rather than a deliberate choice. The message must name both keys so the user knows which one to add.
def test_jobs_per_gpu_without_a_pool_fails_at_parse(tmp_path: Path) -> None:
    result = _dry_run(tmp_path, targets=[], config_overrides=["jobs_per_gpu=2"])
    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert "jobs_per_gpu" in output, output
    assert "gpu_ids" in output, output
