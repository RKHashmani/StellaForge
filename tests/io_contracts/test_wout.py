"""Tests for the VMEC ``wout`` NetCDF contract in ``src/io_contracts.py``.

The Stage-2-consumed subset is two Fourier-coefficient families: the shape coefficients on
the ``mn_mode`` grid indexed by ``xm``/``xn``, and the field-strength coefficients on the
denser Nyquist grid indexed by ``xm_nyq``/``xn_nyq``. It also covers the 1D profiles,
``nfp``, and the minor-radius scalars (``Aminor_p``, or ``volume_p`` with ``Rmajor_p``) the
Stage 4 reader needs. Shape-mismatch cases run on the inner checker (NetCDF cannot store an
inconsistent file); the valid and missing-``xm`` files exercise the NetCDF read plus the raise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.io_contracts import ContractError, _check_wout, validate_wout
from tests.helpers.synthetic import write_wout


def _valid(ns: int = 4, mnmax: int = 3, mnmax_nyq: int = 5) -> dict[str, np.ndarray | None]:
    two_d = np.ones((ns, mnmax))
    two_d_nyq = np.ones((ns, mnmax_nyq))
    one_d = np.linspace(0.0, 1.0, ns)
    data: dict[str, np.ndarray | None] = {name: two_d.copy() for name in ("rmnc", "zmns", "lmns")}
    data.update({name: two_d_nyq.copy() for name in ("bmnc", "bsubumnc", "bsubvmnc")})
    data.update({name: one_d.copy() for name in ("iotas", "phi", "phipf")})
    data.update({"xm": np.arange(mnmax, dtype=float), "xn": np.zeros(mnmax)})
    data.update({"xm_nyq": np.arange(mnmax_nyq, dtype=float), "xn_nyq": np.zeros(mnmax_nyq)})
    data.update({"nfp": np.array(1), "Aminor_p": np.array(0.3)})
    return data


def test_valid_inner_passes() -> None:
    assert _check_wout(_valid()) == []


def test_missing_rmnc_flagged() -> None:
    data = _valid()
    data["rmnc"] = None
    assert "missing required field 'rmnc'" in _check_wout(data)


def test_nyq_mode_axis_mismatch_flagged() -> None:
    data = _valid(ns=4, mnmax=3, mnmax_nyq=5)
    data["bmnc"] = np.ones((4, 3))  # mode axis 3 != mnmax_nyq 5 (from xm_nyq)
    assert any("bmnc" in problem and "mode axis" in problem for problem in _check_wout(data))


def test_missing_xm_flagged() -> None:
    data = _valid()
    data["xm"] = None
    assert "missing required field 'xm'" in _check_wout(data)


def test_missing_xm_nyq_flagged() -> None:
    data = _valid()
    data["xm_nyq"] = None
    assert "missing required field 'xm_nyq'" in _check_wout(data)


def test_missing_nfp_flagged() -> None:
    data = _valid()
    data["nfp"] = None
    assert "missing required field 'nfp'" in _check_wout(data)


def test_missing_minor_radius_flagged() -> None:
    data = _valid()
    data["Aminor_p"] = None  # volume_p / Rmajor_p are absent from the valid dict too
    assert (
        "missing required minor-radius scalars: 'Aminor_p' or both 'volume_p' and 'Rmajor_p'"
        in _check_wout(data)
    )


def test_valid_file_passes(tmp_path: Path) -> None:
    assert validate_wout(write_wout(tmp_path / "wout.nc")) is None


def test_file_missing_xm_raises(tmp_path: Path) -> None:
    written = write_wout(tmp_path / "wout.nc", omit=("xm",))
    with pytest.raises(ContractError, match="xm"):
        validate_wout(written)
