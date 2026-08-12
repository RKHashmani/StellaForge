"""Validate the closed-loop ``converge_status.json`` signal.

The loop driver validates this signal after each iteration. The ``pipeline`` environment has no
NumPy. Therefore, this module uses only the standard library. The file contracts in ``io_contracts``
use the same :class:`ContractError` and ``_raise_if`` helper.
"""

from __future__ import annotations

from typing import Any


class ContractError(Exception):
    """Report a file or signal that violates its consumed-field contract."""


def _raise_if(problems: list[str], subject: str) -> None:
    """Raise :class:`ContractError` if ``problems`` is not empty."""
    if problems:
        raise ContractError(f"{subject}:\n  " + "\n  ".join(problems))


# One forward pass can report these closed-loop outcomes:
#   ``continue``: No stop condition occurred. Run another pass.
#   ``converged``: The transport pressure profile reached steady state.
#   ``horizon``: The transport clock reached the configured ``t_final``.
#   ``halted``: The run cannot continue. Use different initial conditions.
LOOP_STATUSES = ("continue", "converged", "horizon", "halted")


def _check_signal(signal: Any) -> list[str]:
    """Check the closed-loop signal dictionary.

    Examples
    --------
    >>> _check_signal({"status": "converged"})
    []
    >>> _check_signal({})
    ["missing required key 'status'"]
    >>> _check_signal({"status": "done"})
    ["key 'status' must be one of continue, converged, horizon, halted, got 'done'"]
    """
    if not isinstance(signal, dict):
        return [f"signal must be a dict, got {type(signal).__name__}"]
    if "status" not in signal:
        return ["missing required key 'status'"]
    if signal["status"] not in LOOP_STATUSES:
        return [f"key 'status' must be one of {', '.join(LOOP_STATUSES)}, got {signal['status']!r}"]
    return []


def validate_signal(signal: dict) -> None:
    """Validate the closed-loop status signal.

    Parameters
    ----------
    signal : dict
        Parsed ``converge_status.json`` payload. This value is a dictionary, not a file path.

    Raises
    ------
    ContractError
        If ``status`` is missing or is not one of ``LOOP_STATUSES``.

    Examples
    --------
    >>> validate_signal({"status": "continue"}) is None
    True
    """
    _raise_if(_check_signal(signal), "signal dict violates the converge_status contract")
