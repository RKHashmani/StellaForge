"""Tests for ``src.utils.config_edit.apply_assignments``.

``apply_assignments`` rewrites ``key = value`` lines in upstream config text.
Stage 5's ``prepare_neopax_config`` depends on it to point NEOPAX at the right
artifacts, so the exact rewrite semantics are pinned here: the ``key =`` prefix is
preserved, no quoting is added, only column-0 keys match, and a key is never
mistaken for a longer key sharing its prefix.
"""

from __future__ import annotations

from src.utils import apply_assignments


def test_replaces_value_only() -> None:
    assert apply_assignments("a = 1\nb = 2\n", {"a": "9"}) == "a = 9\nb = 2\n"


def test_preserves_prefix_spacing() -> None:
    assert apply_assignments("x=1", {"x": "2"}) == "x=2"
    assert apply_assignments("x  =  1", {"x": "2"}) == "x  =  2"


def test_absent_key_unchanged() -> None:
    assert apply_assignments("a = 1\n", {"c": "3"}) == "a = 1\n"


def test_multiple_keys() -> None:
    text = "a = 1\nb = 2\nc = 3\n"
    assert apply_assignments(text, {"a": "10", "c": "30"}) == "a = 10\nb = 2\nc = 30\n"


def test_inserts_caller_value_verbatim() -> None:
    # The real prepare_neopax_config case: a caller-quoted relative path is inserted
    # exactly as given (no quoting added, quotes and slashes preserved).
    out = apply_assignments("vmec_file = OLD", {"vmec_file": '"../stage1_equilibrium/wout.nc"'})
    assert out == 'vmec_file = "../stage1_equilibrium/wout.nc"'


def test_only_column_zero_keys_match() -> None:
    # Indented keys are left untouched (TOML top-level keys sit at column 0).
    assert apply_assignments("  a = 1\n", {"a": "9"}) == "  a = 1\n"


def test_prefix_collision_safety() -> None:
    # Rewriting "transport_output" must not touch "transport_output_dir".
    text = 'transport_output_dir = "old"\n'
    assert apply_assignments(text, {"transport_output": "x"}) == text
