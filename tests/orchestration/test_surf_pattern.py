"""Tests for the Snakefile's shared per-surface wildcard constraint.

``SURF_PATTERN`` is the regex both Stage 3 and Stage 4 pin their ``{surf}``
wildcard to, so it alone decides which per-surface run-directory basenames
Snakemake will schedule. The Snakefile is not importable Python, so the literal
is pulled straight from its text and exercised with ``re.fullmatch``: the
baseline ``rho_<idx>_r<radius>`` names and the fd_gradients perturbed siblings
``_fd_{n,t}_<species>`` must match, while a bad channel letter, a missing
species, or a species carrying punctuation must not.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_SNAKEFILE_TEXT = (Path(__file__).resolve().parents[2] / "Snakefile").read_text()
_match = re.search(r'^SURF_PATTERN = r"(.+)"$', _SNAKEFILE_TEXT, re.M)
assert _match is not None, "SURF_PATTERN literal not found in Snakefile"
SURF_PATTERN = _match.group(1)


# Names Snakemake must schedule: the baseline rho_<idx>_r<radius> directories and their
# fd_gradients perturbed siblings across both channels (_fd_n_ density, _fd_t_ temperature)
# and single- or multi-character species names.
@pytest.mark.parametrize(
    "name",
    [
        "rho_003_r0p2500",
        "rho_012_r0p4898",
        "rho_003_r0p2500_fd_n_ion",
        "rho_003_r0p2500_fd_t_ion",
        "rho_003_r0p2500_fd_n_D",
    ],
)
def test_surf_pattern_accepts(name: str) -> None:
    assert re.fullmatch(SURF_PATTERN, name) is not None


# Names the constraint must reject: a bare index with no radius, a channel letter outside
# n/t, an empty species after the trailing underscore, and a species carrying punctuation
# that the run-directory sanitizer never produces.
@pytest.mark.parametrize(
    "name",
    [
        "rho_003",
        "rho_003_r0p2500_fd_x_D",
        "rho_003_r0p2500_fd_n_",
        "rho_003_r0p2500_fd_n_he-3",
    ],
)
def test_surf_pattern_rejects(name: str) -> None:
    assert re.fullmatch(SURF_PATTERN, name) is None
