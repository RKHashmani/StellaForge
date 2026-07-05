"""Tests for the Boozer ``boozmn`` NetCDF contract in ``src/io_contracts.py``.

The Stage-3-consumed subset is ``bmnc_b`` (indexed ``[surface, mode]``), the 1D Boozer
profiles, the ``ixm_b``/``ixn_b`` mode-number arrays, ``jlist``, and ``nfp_b``.
Shape-mismatch cases run on the inner checker; the valid and missing-``bmnc_b`` files
exercise the NetCDF read plus the raise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.io_contracts import ContractError, _check_boozmn, validate_boozmn
from tests.helpers.synthetic import write_boozmn


def _valid(n_surf: int = 3, mnboz: int = 4) -> dict[str, np.ndarray | None]:
    data: dict[str, np.ndarray | None] = {"bmnc_b": np.ones((n_surf, mnboz))}
    data.update({name: np.linspace(0.0, 1.0, n_surf) for name in ("iota_b", "buco_b", "bvco_b", "phi_b", "phip_b")})
    data.update({"ixm_b": np.arange(mnboz), "ixn_b": np.zeros(mnboz, dtype=int)})
    data.update({"jlist": np.arange(2, 2 + n_surf), "nfp_b": np.array(1)})
    return data


def test_valid_inner_passes() -> None:
    assert _check_boozmn(_valid()) == []


def test_missing_bmnc_flagged() -> None:
    data = _valid()
    data["bmnc_b"] = None
    assert "missing required field 'bmnc_b'" in _check_boozmn(data)


def test_mode_axis_mismatch_flagged() -> None:
    data = _valid(n_surf=3, mnboz=4)
    data["bmnc_b"] = np.ones((3, 2))  # mode axis 2 != mnboz 4 (from ixm_b)
    assert any("bmnc_b" in problem and "mode axis" in problem for problem in _check_boozmn(data))


def test_mode_array_length_mismatch_flagged() -> None:
    data = _valid(n_surf=3, mnboz=4)
    data["ixm_b"] = np.arange(2)  # sets mnboz to 2, so ixn_b length 4 is inconsistent
    assert any("ixn_b" in problem for problem in _check_boozmn(data))


def test_missing_nfp_b_flagged() -> None:
    data = _valid()
    data["nfp_b"] = None
    assert "missing required field 'nfp_b'" in _check_boozmn(data)


def test_valid_file_passes(tmp_path: Path) -> None:
    assert validate_boozmn(write_boozmn(tmp_path / "b.nc")) is None


def test_file_missing_bmnc_raises(tmp_path: Path) -> None:
    written = write_boozmn(tmp_path / "b.nc", omit=("bmnc_b",))
    with pytest.raises(ContractError, match="bmnc_b"):
        validate_boozmn(written)
