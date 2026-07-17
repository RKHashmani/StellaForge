"""Shared fixtures for the driftless-star test suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def config() -> dict:
    """The committed quick_run config (``inputs/quick_run/config.yaml``)."""
    import yaml

    return yaml.safe_load((REPO_ROOT / "inputs/quick_run/config.yaml").read_text())


@pytest.fixture
def paths(config: dict) -> dict:
    """All pipeline paths resolved from the quick_run config."""
    from src.utils import resolve_pipeline_paths

    return resolve_pipeline_paths(config)
