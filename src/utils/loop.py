"""Closed-loop rerun flag resolution

``resolve_rerun_flags`` turns a run config's ``loop.rerun`` block into a complete
per-stage flag dict. A stage flagged false is frozen, meaning later loop iterations
reuse its iteration 1 output. The Snakefile and the closed-loop driver
(``src/ouroboros.py``) both import it.
"""

from __future__ import annotations

LOOP_STAGES: tuple[str, ...] = ("stage1", "stage2", "stage3", "stage4", "stage5")

# This map lists the stage outputs that each stage reads directly. Snakefile rule inputs define it.
# Stages 3 and 4 also read profiles from ``common_input``. These profiles change in each iteration.
# A frozen stage therefore keeps fluxes calculated from an earlier profile state.
_STAGE_INPUTS: dict[str, tuple[str, ...]] = {
    "stage2": ("stage1",),
    "stage3": ("stage1", "stage2"),
    "stage4": ("stage1", "stage2"),
    "stage5": ("stage1", "stage2", "stage3", "stage4"),
}


def _read_optional_mapping(container: dict, key: str, label: str, expected: str) -> dict:
    """Return one nested mapping from a config, treating an absent or null value as empty.

    Parameters
    ----------
    container : dict
        Mapping to read the key from.
    key : str
        Key to look up.
    label : str
        Config path shown in the error message, for example ``config['loop']['rerun']``.
    expected : str
        Description of the mapping's contents shown in the error message, for example ``stage name to true/false``.

    Returns
    -------
    dict
        The nested mapping, or an empty dict when the key is absent or its value is ``None``. A bare ``loop:`` line in
        YAML parses to ``None``, which is treated the same as omitting the block.

    Raises
    ------
    ValueError
        If the value is present and is not a mapping.
    """
    value = container.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping of {expected}, got {type(value).__name__}.")
    return value


def _check_frozen_stage_inputs(flags: dict[str, bool]) -> None:
    """Raise if any frozen stage reads the output of a stage that still reruns.

    Parameters
    ----------
    flags : dict[str, bool]
        Rerun flag for every stage in ``LOOP_STAGES``.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If a stage is frozen while at least one stage it reads from reruns.
    """
    for stage, input_stages in _STAGE_INPUTS.items():
        if flags[stage]:
            continue
        rerunning = [name for name in input_stages if flags[name]]
        if rerunning:
            raise ValueError(
                f"config['loop']['rerun']['{stage}'] is false but {rerunning} still rerun, so the frozen {stage} "
                f"output would be paired with regenerated upstream stage outputs and go stale. Freeze {rerunning} "
                f"as well, or let {stage} rerun."
            )


def resolve_rerun_flags(config: dict) -> dict[str, bool]:
    """Return one rerun flag for each pipeline stage.

    Parameters
    ----------
    config : dict
        Parsed run config. The optional ``loop.rerun`` mapping contains one boolean per stage.
        ``false`` reuses the stage output from iteration 1. Omitted stages rerun.

    Returns
    -------
    dict[str, bool]
        One entry per stage in ``LOOP_STAGES``, in that order.

    Raises
    ------
    ValueError
        If a loop setting is invalid or a frozen stage reads output from a stage that reruns.

    Notes
    -----
    A frozen stage requires frozen input stages. The valid frozen sets are:
    ``{}``, ``{stage1}``, ``{stage1, stage2}``, ``{stage1, stage2, stage3}``,
    ``{stage1, stage2, stage4}``, ``{stage1, stage2, stage3, stage4}`` and all five stages.
    """
    loop = _read_optional_mapping(config, "loop", "config['loop']", "loop settings")
    rerun = _read_optional_mapping(loop, "rerun", "config['loop']['rerun']", "stage name to true/false")

    unknown = [key for key in rerun if key not in LOOP_STAGES]
    if unknown:
        raise ValueError(
            f"config['loop']['rerun'] has unknown stage key(s) {sorted(unknown)}; allowed keys are {list(LOOP_STAGES)}."
        )

    flags: dict[str, bool] = {}
    for stage in LOOP_STAGES:
        value = rerun.get(stage, True)
        # Reject ints and strings that YAML did not parse as booleans, since 0/1 and "false" would pass a
        # truthiness check.
        if not isinstance(value, bool):
            raise ValueError(f"config['loop']['rerun']['{stage}'] must be true or false, got {value!r}.")
        flags[stage] = value

    _check_frozen_stage_inputs(flags)
    return flags
