"""GPU pool resolution

``resolve_gpu_settings`` turns a run config's top-level ``gpu_ids`` and ``jobs_per_gpu``
keys into the container image variant to run, the pool of GPU ids concurrent jobs are
pinned to, and how many jobs may share one id. The Snakefile validates at parse time and
the closed-loop driver (``src/ouroboros.py``) at startup, so an unusable pool is reported
before any job starts.
"""

from __future__ import annotations

from typing import NamedTuple


class GpuSettings(NamedTuple):
    """Resolved GPU pool settings for one run.

    Attributes
    ----------
    device : str
        ``"cpu"`` or ``"gpu"``, selecting the ``-cpu`` or ``-gpu`` container image variant every stage runs in.
    pool : tuple[str, ...]
        Normalized GPU ids that concurrent jobs are pinned to, in config order. Empty when running on cpu and when
        ``"all"`` leaves the ids to be resolved on the execution host at job start.
    jobs_per_gpu : int
        How many concurrent jobs may share one pooled id. Always 1 unless a gpu run raises it.
    """

    device: str
    pool: tuple[str, ...]
    jobs_per_gpu: int


def parse_gpu_pool(value: str) -> tuple[str, ...]:
    """Split a comma-separated GPU id list into normalized ids.

    Parameters
    ----------
    value : str
        Comma-separated GPU ids, for example ``"4,5,6,7"``. Whitespace around an id is ignored and leading zeros are
        dropped, so ``" 04 , 5"`` normalizes to ``("4", "5")``.

    Returns
    -------
    tuple[str, ...]
        The normalized ids, in the order they were written.

    Raises
    ------
    ValueError
        If an entry is empty, is not an integer, or is negative, or if two entries name the same device once
        normalized.
    """
    ids: list[str] = []
    for entry in value.split(","):
        token = entry.strip()
        if not token:
            raise ValueError(f"config['gpu_ids'] has an empty entry in {value!r}; write ids as \"4,5,6,7\".")
        try:
            gpu_id = int(token)
        except ValueError as exc:
            raise ValueError(
                f"config['gpu_ids'] entry {token!r} is not an integer; write ids as \"4,5,6,7\"."
            ) from exc
        if gpu_id < 0:
            raise ValueError(f"config['gpu_ids'] entry {token!r} is negative; GPU ids start at 0.")
        ids.append(str(gpu_id))

    # A repeated id hands the same device two job slots, silently oversubscribing it while the run still looks
    # evenly spread across the pool.
    duplicates = sorted({gpu_id for gpu_id in ids if ids.count(gpu_id) > 1})
    if duplicates:
        raise ValueError(f"config['gpu_ids'] names id(s) {duplicates} more than once in {value!r}; list each id once.")
    return tuple(ids)


def _resolve_pool(raw: object) -> tuple[str, tuple[str, ...]]:
    """Turn a raw ``gpu_ids`` value into the container image variant and the pinned id pool.

    Parameters
    ----------
    raw : object
        Value of the run config's ``gpu_ids`` key, or ``None`` when the key is absent.

    Returns
    -------
    tuple[str, tuple[str, ...]]
        The image variant (``"cpu"`` or ``"gpu"``) and the normalized ids jobs are pinned to. The pool is empty for a
        cpu run and for an ``"all"`` run, whose ids are resolved on the execution host at job start.

    Raises
    ------
    ValueError
        If the value is a boolean, a negative integer, an empty or unrecognized string, or any other type.
    """
    if raw is None:
        return "cpu", ()
    # ``True`` would otherwise reach the integer branch and pin every job to id 1, so booleans are rejected first.
    if isinstance(raw, bool):
        raise ValueError(f"config['gpu_ids'] must be null, \"all\", or a comma-separated id list, got {raw!r}.")
    if isinstance(raw, int):
        if raw < 0:
            raise ValueError(f"config['gpu_ids'] is negative ({raw}); GPU ids start at 0.")
        return "gpu", (str(raw),)
    if isinstance(raw, str):
        text = raw.strip()
        lowered = text.lower()
        if lowered in ("null", "none"):
            return "cpu", ()
        if lowered == "all":
            return "gpu", ()
        if not text:
            raise ValueError(
                "config['gpu_ids'] is empty; use null for cpu, \"all\" for the execution host's own GPUs, or a "
                "comma-separated id list such as \"4,5,6,7\"."
            )
        return "gpu", parse_gpu_pool(text)
    raise ValueError(
        f"config['gpu_ids'] must be null, \"all\", or a comma-separated id list, got {type(raw).__name__}. "
        "Quote a list of ids as a single string, such as \"4,5,6,7\"."
    )


def _resolve_jobs_per_gpu(raw: object, device: str) -> int:
    """Check a raw ``jobs_per_gpu`` value against the resolved image variant.

    Parameters
    ----------
    raw : object
        Value of the run config's ``jobs_per_gpu`` key, or 1 when the key is absent.
    device : str
        Image variant resolved from ``gpu_ids``, either ``"cpu"`` or ``"gpu"``.

    Returns
    -------
    int
        The number of concurrent jobs allowed to share one pooled id.

    Raises
    ------
    ValueError
        If the value is not a whole number of at least 1, or if it is raised above 1 for a cpu run.
    """
    # A quoted "2" and a true/false value are rejected rather than coerced, so a typo cannot quietly change how many
    # jobs land on each device.
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f"config['jobs_per_gpu'] must be a whole number of at least 1, got {raw!r}.")
    if raw < 1:
        raise ValueError(f"config['jobs_per_gpu'] must be at least 1, got {raw}.")
    if raw != 1 and device == "cpu":
        raise ValueError(
            f"config['jobs_per_gpu'] is {raw} but config['gpu_ids'] asks for a cpu run, which has no devices to "
            "share out. Raise jobs_per_gpu only alongside gpu_ids set to \"all\" or to an id list such as \"4,5,6,7\"."
        )
    return raw


def resolve_gpu_settings(config: dict) -> GpuSettings:
    """Turn a run config's ``gpu_ids`` and ``jobs_per_gpu`` keys into resolved GPU pool settings.

    Parameters
    ----------
    config : dict
        Parsed run config. ``gpu_ids`` is null to run the cpu images, ``"all"`` to run the gpu images with each
        concurrent job pinned to one free GPU of the execution host, or a comma-separated id list such as
        ``"4,5,6,7"`` to run the gpu images with each concurrent job pinned to one free id from that pool. An
        ``"all"`` pool is resolved at job start and covers every GPU of the host, or only the ids
        ``CUDA_VISIBLE_DEVICES`` lists when that variable is set. ``jobs_per_gpu`` defaults to 1 and may only be
        raised for a gpu run.

    Returns
    -------
    GpuSettings
        The container image variant, the pinned id pool, and the per-id job slot count.

    Raises
    ------
    ValueError
        If the config still carries a ``device`` key, if ``gpu_ids`` has an unusable type or value, or if
        ``jobs_per_gpu`` is not a whole number of at least 1 or is raised above 1 for a cpu run.

    Notes
    -----
    Values passed on the snakemake command line as ``--config gpu_ids=...`` arrive already parsed. A bare integer
    comes through as an ``int``, while ``null`` and any id list arrive as plain text, because the fallback parser
    keeps every scalar it does not recognize as a string. Both spellings of each mode are therefore accepted.
    """
    if "device" in config:
        raise ValueError(
            "config['device'] is not read anymore; set the top-level gpu_ids key instead, where null runs the cpu "
            "images, \"all\" runs the gpu images with each concurrent job pinned to one free GPU of the execution "
            "host, and a comma-separated id list such as \"4,5,6,7\" runs the gpu images with each concurrent job "
            "pinned to one free id from that pool."
        )

    device, pool = _resolve_pool(config.get("gpu_ids"))
    jobs_per_gpu = _resolve_jobs_per_gpu(config.get("jobs_per_gpu", 1), device)
    return GpuSettings(device=device, pool=pool, jobs_per_gpu=jobs_per_gpu)
