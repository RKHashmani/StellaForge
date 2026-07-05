"""Tests for the VMEC ``wout`` NetCDF contract in ``src/io_contracts.py``.

The Stage-2-consumed subset is the 2D geometry coefficients, the 1D profiles, and the
``xm``/``xn``/``nfp`` mode information the coefficients are indexed by. Shape-mismatch
cases run on the inner checker (NetCDF cannot store an inconsistent file); the valid and
missing-``xm`` files exercise the NetCDF read plus the raise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.io_contracts import ContractError, _check_wout, validate_wout
from tests.helpers.synthetic import write_wout


def _valid(ns: int = 4, mnmax: int = 3) -> dict[str, np.ndarray | None]:
    two_d = np.ones((ns, mnmax))
    one_d = np.linspace(0.0, 1.0, ns)
    data: dict[str, np.ndarray | None] = {name: two_d.copy() for name in ("rmnc", "zmns", "lmns", "bmnc", "bsubumnc", "bsubvmnc")}
    data.update({name: one_d.copy() for name in ("iotas", "phi", "phipf")})
    data.update({"xm": np.arange(mnmax, dtype=float), "xn": np.zeros(mnmax), "nfp": np.array(1)})
    return data


def test_valid_inner_passes() -> None:
    assert _check_wout(_valid()) == []


def test_missing_rmnc_flagged() -> None:
    data = _valid()
    data["rmnc"] = None
    assert "missing required field 'rmnc'" in _check_wout(data)


def test_mode_axis_mismatch_flagged() -> None:
    data = _valid(ns=4, mnmax=3)
    data["bmnc"] = np.ones((4, 2))  # mode axis 2 != mnmax 3 (from xm)
    assert any("bmnc" in problem and "mode axis" in problem for problem in _check_wout(data))


def test_missing_xm_flagged() -> None:
    data = _valid()
    data["xm"] = None
    assert "missing required field 'xm'" in _check_wout(data)


def test_missing_nfp_flagged() -> None:
    data = _valid()
    data["nfp"] = None
    assert "missing required field 'nfp'" in _check_wout(data)


def test_valid_file_passes(tmp_path: Path) -> None:
    assert validate_wout(write_wout(tmp_path / "wout.nc")) is None


def test_file_missing_xm_raises(tmp_path: Path) -> None:
    written = write_wout(tmp_path / "wout.nc", omit=("xm",))
    with pytest.raises(ContractError, match="xm"):
        validate_wout(written)
