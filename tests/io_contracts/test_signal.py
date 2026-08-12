"""Test the closed-loop signal contract.

The validator accepts a parsed ``converge_status.json`` dictionary. It has no file layer. Tests call
the checker directly and use ``validate_signal`` to check the public raise path.
"""

from __future__ import annotations

import pytest

from src.signal_contract import LOOP_STATUSES, ContractError, _check_signal, validate_signal


@pytest.mark.parametrize("status", LOOP_STATUSES)
def test_every_declared_status_passes(status: str) -> None:
    assert validate_signal({"status": status}) is None


# The driver has no branch for an unknown status. Without validation, it would keep the loop
# running. The exact list also checks the message and excludes extra problems.
def test_check_signal_unknown_status() -> None:
    assert _check_signal({"status": "done"}) == [
        "key 'status' must be one of continue, converged, horizon, halted, got 'done'"
    ]


# An older signal can contain only convergence and halt booleans. It has no status and must fail.
def test_check_signal_rejects_the_two_boolean_shape() -> None:
    assert _check_signal({"converged": True, "halt": False}) == ["missing required key 'status'"]


# A list has the wrong top-level type. The checker rejects it before checking keys.
def test_check_signal_not_a_dict() -> None:
    assert _check_signal(["status"]) == ["signal must be a dict, got list"]


def test_validate_signal_raises_on_missing_key() -> None:
    with pytest.raises(ContractError, match="status"):
        validate_signal({})
