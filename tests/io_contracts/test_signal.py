"""Tests for the closed-loop signal contract in ``src/io_contracts.py``.

The signal is a parsed ``converge_status.json`` dict, so the validator has no file
layer: ``validate_signal`` is the pure checker plus the raise. Invalid cases are
exercised on the checker directly; the raise path and all-problems reporting are shown
through the public ``validate_signal``.
"""

from __future__ import annotations

import pytest

from src.io_contracts import ContractError, _check_signal, validate_signal


def test_valid_signal_passes() -> None:
    assert validate_signal({"converged": True, "halt": False}) is None


def test_check_signal_missing_key() -> None:
    assert _check_signal({"converged": True}) == ["missing required key 'halt'"]


def test_check_signal_wrong_type() -> None:
    assert _check_signal({"converged": 1, "halt": False}) == ["key 'converged' must be bool, got int"]


def test_check_signal_not_a_dict() -> None:
    assert _check_signal(["converged", "halt"]) == ["signal must be a dict, got list"]


def test_validate_signal_raises_on_missing_key() -> None:
    with pytest.raises(ContractError, match="halt"):
        validate_signal({"converged": True})


def test_validate_signal_reports_all_problems() -> None:
    with pytest.raises(ContractError) as exc:
        validate_signal({})
    message = str(exc.value)
    assert "converged" in message and "halt" in message
