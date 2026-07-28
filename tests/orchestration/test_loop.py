"""Tests for ``src.utils.loop.resolve_rerun_flags``.

``resolve_rerun_flags`` decides which stages a loop iteration recomputes and which
reuse their iteration 1 output. A flag misread as frozen would silently
serve stale physics for the rest of the run, so these tests pin the defaulting
behavior, the exact set of frozen combinations the dependency structure permits,
and the rejection of malformed config shapes.
"""

from __future__ import annotations

import pytest

from src.utils import resolve_rerun_flags

ALL_STAGES = ("stage1", "stage2", "stage3", "stage4", "stage5")
ALL_TRUE = dict.fromkeys(ALL_STAGES, True)

# Every frozen set the input dependencies allow. A stage may only be frozen when all stages feeding it are frozen too,
# which rules out combinations such as freezing Stage 4 while Stage 2 reruns.
VALID_FROZEN_SETS = [
    (),
    ("stage1",),
    ("stage1", "stage2"),
    ("stage1", "stage3"),
    ("stage1", "stage2", "stage3"),
    ("stage1", "stage2", "stage4"),
    ("stage1", "stage2", "stage3", "stage4"),
    ("stage1", "stage2", "stage3", "stage4", "stage5"),
]


def _frozen(*stages: str) -> dict:
    """Build a run config whose ``loop.rerun`` block freezes exactly the named stages.

    Parameters
    ----------
    *stages : str
        Stage names to mark false. All other stages are left out so they take the default.

    Returns
    -------
    dict
        A config carrying only the ``loop.rerun`` block.
    """
    return {"loop": {"rerun": dict.fromkeys(stages, False)}}


# A run config that never mentions the loop at all must still yield a full flag set, because the Snakefile and the
# driver index the result by stage name unconditionally. These cases cover every way the block can be missing or empty,
# including the bare `loop:` and `rerun:` YAML lines that parse to None rather than to an empty mapping.
@pytest.mark.parametrize(
    "config",
    [
        {},
        {"loop": None},
        {"loop": {}},
        {"loop": {"iterations": 3}},
        {"loop": {"rerun": None}},
        {"loop": {"rerun": {}}},
    ],
)
def test_missing_loop_block_defaults_to_all_rerunning(config: dict) -> None:
    assert resolve_rerun_flags(config) == ALL_TRUE


# The returned dict must always carry exactly the five stage keys in pipeline order, whatever the config supplied. Order
# matters because callers iterate the result to report and to walk stages upstream to downstream.
@pytest.mark.parametrize("config", [{}, _frozen("stage1"), _frozen("stage1", "stage2", "stage3", "stage4", "stage5")])
def test_result_has_all_stage_keys_in_order(config: dict) -> None:
    assert tuple(resolve_rerun_flags(config)) == ALL_STAGES


# Naming only some stages is the common case, since a user typically freezes the expensive upstream stages and lets the
# rest rerun. Stages left out of the block must default to rerunning rather than inheriting a neighbour's flag.
def test_partial_keys_default_the_rest_to_rerunning() -> None:
    flags = resolve_rerun_flags({"loop": {"rerun": {"stage1": False, "stage2": False}}})
    assert flags == {"stage1": False, "stage2": False, "stage3": True, "stage4": True, "stage5": True}


# Explicitly writing true for a stage must be honoured the same as omitting it, so a config that spells out every flag
# behaves identically to one that relies on the defaults.
def test_explicit_true_matches_the_default() -> None:
    explicit = {"loop": {"rerun": {"stage1": False, "stage2": True, "stage3": True, "stage4": True, "stage5": True}}}
    assert resolve_rerun_flags(explicit) == resolve_rerun_flags(_frozen("stage1"))


# The eight combinations below are every frozen set consistent with the stage input dependencies. Each must be accepted
# and returned faithfully, since these are exactly the configurations a user is allowed to write.
@pytest.mark.parametrize("frozen", VALID_FROZEN_SETS)
def test_valid_frozen_sets_are_accepted(frozen: tuple[str, ...]) -> None:
    flags = resolve_rerun_flags(_frozen(*frozen))
    assert {stage for stage, rerun in flags.items() if not rerun} == set(frozen)


# Freezing a stage whose input stages still rerun would pair a cached output with regenerated upstream data, which is
# the stale-physics failure this validation exists to prevent. Each case names a stage frozen while something it reads
# from reruns, and the error must name that frozen stage so the user knows which flag to change.
@pytest.mark.parametrize(
    ("frozen", "offender"),
    [
        (("stage2",), "stage2"),
        (("stage3",), "stage3"),
        (("stage4",), "stage4"),
        (("stage5",), "stage5"),
        (("stage1", "stage4"), "stage4"),
        (("stage2", "stage3", "stage4", "stage5"), "stage2"),
        (("stage1", "stage2", "stage3", "stage5"), "stage5"),
        (("stage1", "stage3", "stage4"), "stage4"),
    ],
)
def test_frozen_stage_with_rerunning_input_is_rejected(frozen: tuple[str, ...], offender: str) -> None:
    with pytest.raises(ValueError, match=offender):
        resolve_rerun_flags(_frozen(*frozen))


# A malformed block must fail loudly at config load rather than being coerced into something plausible. The string and
# list cases catch a mistyped YAML structure; the unknown key catches a typo such as a sixth stage that would otherwise
# be ignored; the "false" and 0 cases catch values that are truthy or falsy in Python but are not booleans, which would
# freeze or rerun a stage against the user's intent if accepted.
@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({"loop": "yes"}, "config\\['loop'\\]"),
        ({"loop": {"rerun": ["stage1"]}}, "config\\['loop'\\]\\['rerun'\\]"),
        ({"loop": {"rerun": {"stage6": True}}}, "stage6"),
        ({"loop": {"rerun": {"stage1": "false"}}}, "stage1"),
        ({"loop": {"rerun": {"stage1": 0}}}, "stage1"),
        ({"loop": {"rerun": {"stage2": 1}}}, "stage2"),
    ],
)
def test_malformed_config_is_rejected(config: dict, expected: str) -> None:
    with pytest.raises(ValueError, match=expected):
        resolve_rerun_flags(config)
