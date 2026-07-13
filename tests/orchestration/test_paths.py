"""Tests for ``src.utils.paths.resolve_pipeline_paths``.

``resolve_pipeline_paths`` is the single source of truth the Snakefile and the
closed-loop driver both use to turn one run's config into concrete file paths.
A change that silently alters a derived path would mis-wire the stages, so these
tests pin the path contract against the committed quick_run config and against a
minimal synthetic config that exercises the ``{run_name}`` templating in isolation.
"""

from __future__ import annotations

from src.utils import resolve_pipeline_paths

EXPECTED_KEYS = {
    "input_dir",
    "output_dir",
    "s1_input",
    "s3_config",
    "s4_config",
    "s5_config",
    "s1_output",
    "s2_output",
    "s3_output",
    "s4_output",
    "s5_output",
    "s5_signal",
    "stage1_dir",
    "stage2_dir",
    "stage3_dir",
    "stage4_dir",
    "stage5_dir",
    "stage5_post_dir",
    "stage3_manifest",
    "stage4_manifest",
    "s5_resolved_config",
    "s1_feedback",
}


# `resolve_pipeline_paths` turns one run's config into a dict of every concrete file path the pipeline uses. The `paths`
# argument is a fixture (defined in conftest.py) holding that dict for the committed quick_run config. This asserts the
# dict has exactly the expected set of keys and pins each stage's input/output path to its precise string, so an
# accidental path change is caught.
def test_quick_run_paths_match_contract(paths: dict) -> None:
    assert set(paths) == EXPECTED_KEYS
    assert paths["s1_input"] == "inputs/quick_run/vmec_input.HSX_vacuum_ns201_quickrun"
    assert paths["s1_output"] == "outputs/quick_run/stage1_equilibrium/wout_HSX_vacuum_ns201_quickrun.nc"
    assert paths["s2_output"] == "outputs/quick_run/stage2_boozer/boozmn_HSX_vacuum_ns201_quickrun.nc"
    assert paths["s3_output"] == "outputs/quick_run/stage3_neoclassical/sfincs_jax_flux_profiles.h5"
    assert paths["s4_output"] == "outputs/quick_run/stage4_turbulence/neopax_fluxes.h5"
    assert paths["s3_config"] == "inputs/quick_run/sfincs_input.HSX_vacuum_ns201_quickrun"
    assert paths["s4_config"] == "inputs/quick_run/HSX_vacuum_ns201_quickrun.toml"
    assert paths["s5_config"] == "inputs/quick_run/common_input.toml"
    assert paths["s5_output"] == "outputs/quick_run/stage5_transport/transport_solution.h5"
    assert paths["s5_signal"] == "outputs/quick_run/stage5_post_processing/converge_status.json"


# Pins the paths that are derived for generated artifacts rather than user inputs: the Stage 3/4 per-surface scan
# manifests, the path-resolved NEOPAX config copy, and the evolved Stage 1 boundary the closed loop feeds back. All must
# land under the output tree, not the input tree.
def test_quick_run_generated_paths(paths: dict) -> None:
    assert paths["stage3_manifest"] == "outputs/quick_run/stage3_neoclassical/manifest.json"
    assert paths["stage4_manifest"] == "outputs/quick_run/stage4_turbulence/manifest.json"
    assert paths["s5_resolved_config"] == "outputs/quick_run/stage5_transport/common_input_updated.toml"
    # The evolved Stage 1 boundary the loop feeds back lives under the post-processing dir.
    assert paths["s1_feedback"] == "outputs/quick_run/stage5_post_processing/vmec_input.HSX_vacuum_ns201_quickrun"


def test_no_unsubstituted_run_name(paths: dict) -> None:
    assert all("{run_name}" not in value for value in paths.values())


# Callers (like the closed-loop driver) can override the input/output base directories. The `config` fixture is the
# quick_run config; here it is resolved with explicit sandbox dirs, and the test asserts those override the config's own
# dirs and that every derived path is rebuilt under them. This is what lets each loop iteration write into its own
# folder.
def test_dir_overrides_take_precedence(config: dict) -> None:
    p = resolve_pipeline_paths(config, input_dir="/sandbox/in", output_dir="/sandbox/out")
    assert p["input_dir"] == "/sandbox/in"
    assert p["output_dir"] == "/sandbox/out"
    assert p["s1_input"].startswith("/sandbox/in/")
    assert p["s1_output"].startswith("/sandbox/out/stage1_equilibrium/")
    assert p["s5_resolved_config"].startswith("/sandbox/out/stage5_transport/")


# Instead of the real quick_run config, this builds a tiny handmade config with a `run_name` of "demo" and simple
# template strings, then checks the `{run_name}` substitution and the input-vs-output directory routing in isolation.
# Testing the templating on a minimal input makes the expected results easy to read and independent of the committed
# config.
def test_templating_in_isolation() -> None:
    cfg = {
        "run_name": "demo",
        "input_dir": "IN",
        "output_dir": "OUT",
        "filenames": {
            "s1_input": "vmec_input.{run_name}",
            "s1_output": "wout_{run_name}.nc",
            "s2_output": "boozmn_{run_name}.nc",
            "s3_config": "sfincs_input.{run_name}",
            "s3_output": "flux.h5",
            "s4_config": "{run_name}.toml",
            "s4_output": "neopax_fluxes.h5",
            "s5_config": "common_input.toml",
            "s5_output": "transport_solution.h5",
            "s5_signal": "converge_status.json",
        },
    }
    p = resolve_pipeline_paths(cfg)
    assert p["s1_input"] == "IN/vmec_input.demo"
    assert p["s1_output"] == "OUT/stage1_equilibrium/wout_demo.nc"
    assert p["s4_config"] == "IN/demo.toml"
    assert p["s5_signal"] == "OUT/stage5_post_processing/converge_status.json"
