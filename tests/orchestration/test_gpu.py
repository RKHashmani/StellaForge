"""Tests for ``src.utils.gpu``.

``resolve_gpu_settings`` turns a run config's ``gpu_ids`` and ``jobs_per_gpu`` keys
into the container image variant every stage runs in, the pool of device ids
concurrent jobs are pinned to, and how many jobs may share one id. A pool misread
here would run the whole pipeline on the wrong image variant, crowd every job onto
one device, or pin jobs to a device that is not there, so these tests pin the exact
settings each accepted spelling resolves to and the rejection of the values a typo
produces.

Both spellings of every mode are covered because a config file and a snakemake
``--config gpu_ids=...`` override deliver the same mode differently. A bare integer
arrives as an ``int`` while ``null`` and every id list arrive as plain text.
"""

from __future__ import annotations

import re

import pytest

from src.utils import GpuSettings, parse_gpu_pool, resolve_gpu_settings

CPU = GpuSettings(device="cpu", pool=(), jobs_per_gpu=1)


# Every config a user is allowed to write, paired with the settings it must resolve to. The three modes are cpu (no
# gpu_ids), the execution host's own devices enumerated at job start ("all"), and an explicit pool named here. Both gpu
# modes pin each job to one id, so "all" resolves to an empty pool only because its ids are not knowable until the job
# runs on the host. The whitespace, leading-zero and quoted-integer cases pin the normalization that makes " 04 , 5" and
# "4,5" the same pool, so a lock file name cannot depend on how the ids were spelled.
@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({}, CPU),
        ({"gpu_ids": None}, CPU),
        ({"gpu_ids": "null"}, CPU),
        ({"gpu_ids": "None"}, CPU),
        ({"gpu_ids": "all"}, GpuSettings("gpu", (), 1)),
        ({"gpu_ids": "ALL"}, GpuSettings("gpu", (), 1)),
        ({"gpu_ids": "4,5,6,7"}, GpuSettings("gpu", ("4", "5", "6", "7"), 1)),
        ({"gpu_ids": " 4 , 5 "}, GpuSettings("gpu", ("4", "5"), 1)),
        ({"gpu_ids": "07,8"}, GpuSettings("gpu", ("7", "8"), 1)),
        ({"gpu_ids": 4}, GpuSettings("gpu", ("4",), 1)),
        ({"gpu_ids": 0}, GpuSettings("gpu", ("0",), 1)),
        ({"gpu_ids": "0"}, GpuSettings("gpu", ("0",), 1)),
        ({"gpu_ids": "4,5", "jobs_per_gpu": 2}, GpuSettings("gpu", ("4", "5"), 2)),
        ({"gpu_ids": "all", "jobs_per_gpu": 2}, GpuSettings("gpu", (), 2)),
    ],
)
def test_accepted_configs_resolve_to_expected_settings(config: dict, expected: GpuSettings) -> None:
    assert resolve_gpu_settings(config) == expected


# Each rejected config is paired with a fragment of the message it must raise, so a rewording that drops the offending
# key or value from the message fails here. The device cases catch a config written before the key was replaced, which
# would otherwise plan a cpu run while the user believes they asked for GPUs. The gpu_ids cases catch a mistyped list,
# a negative id, a repeat that would silently double up on one device, and the types a YAML list or a bare true would
# deliver. The jobs_per_gpu cases catch values that are not whole numbers of at least 1, and the last three catch
# raising the count for a cpu run, which has no device to share out at all and is almost always a forgotten gpu_ids.
@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({"device": "cpu"}, "gpu_ids"),
        ({"device": "gpu"}, "gpu_ids"),
        ({"gpu_ids": ""}, re.escape("config['gpu_ids'] is empty")),
        ({"gpu_ids": "all,1"}, re.escape("entry 'all' is not an integer")),
        ({"gpu_ids": "a,b"}, re.escape("entry 'a' is not an integer")),
        ({"gpu_ids": "-1"}, re.escape("entry '-1' is negative")),
        ({"gpu_ids": -1}, re.escape("config['gpu_ids'] is negative (-1)")),
        ({"gpu_ids": "4,,5"}, re.escape("has an empty entry in '4,,5'")),
        ({"gpu_ids": "4,4"}, re.escape("names id(s) ['4'] more than once")),
        ({"gpu_ids": "04,4"}, re.escape("names id(s) ['4'] more than once")),
        ({"gpu_ids": True}, re.escape("got True")),
        ({"gpu_ids": 4.5}, re.escape("got float")),
        ({"gpu_ids": [4, 5]}, re.escape("got list")),
        ({"gpu_ids": "4,5", "jobs_per_gpu": 0}, re.escape("config['jobs_per_gpu'] must be at least 1, got 0")),
        ({"gpu_ids": "4,5", "jobs_per_gpu": -1}, re.escape("config['jobs_per_gpu'] must be at least 1, got -1")),
        ({"gpu_ids": "4,5", "jobs_per_gpu": "2"}, re.escape("whole number of at least 1, got '2'")),
        ({"gpu_ids": "4,5", "jobs_per_gpu": 1.5}, re.escape("whole number of at least 1, got 1.5")),
        ({"gpu_ids": "4,5", "jobs_per_gpu": True}, re.escape("whole number of at least 1, got True")),
        ({"jobs_per_gpu": 2}, re.escape("config['jobs_per_gpu'] is 2 but config['gpu_ids'] asks for a cpu run")),
        ({"gpu_ids": None, "jobs_per_gpu": 2}, re.escape("is 2 but config['gpu_ids'] asks for a cpu run")),
        ({"gpu_ids": "null", "jobs_per_gpu": 3}, re.escape("is 3 but config['gpu_ids'] asks for a cpu run")),
    ],
)
def test_rejected_configs_raise_naming_the_offending_key(config: dict, expected: str) -> None:
    with pytest.raises(ValueError, match=expected):
        resolve_gpu_settings(config)


# The pool order is the order slots are tried, so it must survive parsing unsorted. A user who lists the ids they were
# allocated in an unusual order still gets exactly those ids.
def test_parse_gpu_pool_preserves_the_written_order() -> None:
    assert parse_gpu_pool("7,0,3") == ("7", "0", "3")


# The parser is also reached directly by the slot allocator's --gpu-ids flag, where the user sees only its message. Each
# case must name the entry that is wrong rather than only the whole list, so a long pool with one typo is actionable.
@pytest.mark.parametrize(
    ("value", "offender"),
    [
        ("4,x,5", "'x'"),
        ("4,-2", "'-2'"),
        ("4,5,4", "['4']"),
    ],
)
def test_parse_gpu_pool_errors_name_the_offending_entry(value: str, offender: str) -> None:
    with pytest.raises(ValueError, match=re.escape(offender)):
        parse_gpu_pool(value)
