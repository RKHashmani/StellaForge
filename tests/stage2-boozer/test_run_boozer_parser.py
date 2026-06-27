"""Tests for the Stage 2 ``run_boozer`` CLI parser.

``run_boozer.py`` runs as a standalone script and imports ``booz_xform_jax``
lazily inside the transform, so its argument parser loads with no solver present.
The Snakefile's ``rule stage2_boozer`` invokes the script with ``--wout`` and
``--output``, so both must stay required.
"""

from __future__ import annotations

import pytest

from tests.helpers.stage_import import load_stage_module

run_boozer = load_stage_module("stages/stage2-boozer/run_boozer.py")


def test_parser_accepts_wout_and_output() -> None:
    args = run_boozer.build_parser().parse_args(["--wout", "w.nc", "--output", "b.nc"])
    assert str(args.wout) == "w.nc"
    assert str(args.output) == "b.nc"


def test_parser_requires_both_flags() -> None:
    parser = run_boozer.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])                  # neither flag
    with pytest.raises(SystemExit):
        parser.parse_args(["--wout", "w.nc"])  # missing --output
