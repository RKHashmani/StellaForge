"""Launch Snakemake with compatibility fixes for the HTCondor executor.

This module is intentionally used only by submit-side HTCondor controllers.
The remote jobs still use the ordinary Snakemake executable installed in the
HTCondor runtime container.
"""

from __future__ import annotations

from functools import wraps
import os
import re
from typing import Any

RUN_ID_ENV = "DRIFTLESS_STAR_RUN_ID"


def _run_batch_name(run_id: str, plugin_batch_name: str) -> str:
    """Turn ``stage1_vmec-2`` into ``<run_id>_stage1_vmec``."""
    rule_name = re.sub(r"-\d+$", "", plugin_batch_name)
    return f"{run_id}_{rule_name}"


def patch_htcondor_batch_names(htcondor_module: Any, run_id: str) -> bool:
    """Prefix actual HTCondor batch names without modifying rule names."""
    current_submit = htcondor_module.Submit
    if getattr(current_submit, "_driftless_star_batch_name_fix", False):
        return False

    def submit(description: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(description, dict) and description.get("batch_name"):
            # Changing job.name itself would also change Snakemake's remote
            # target and make the execute-side invocation seek a nonexistent
            # rule, so only change the final HTCondor submit description.
            description["batch_name"] = _run_batch_name(
                run_id, str(description["batch_name"])
            )
        return current_submit(description, *args, **kwargs)

    submit._driftless_star_batch_name_fix = True
    htcondor_module.Submit = submit
    return True


def patch_htcondor_executor(executor_class: type[Any] | None = None) -> bool:
    """Install the executor 0.3.0 ``_event_logs`` initialization guard.

    Returns ``True`` when the patch was installed and ``False`` when it was
    already present.
    """
    if executor_class is None:
        from snakemake_executor_plugin_htcondor import Executor

        executor_class = Executor

    if getattr(executor_class, "_driftless_star_event_logs_fix", False):
        return False

    original = getattr(executor_class, "_drain_unified_log", None)
    if original is None:
        raise RuntimeError(
            "The driftless-star compatibility patch was written against "
            "snakemake-executor-plugin-htcondor 0.3.0, and the installed plugin no "
            "longer has _drain_unified_log. Re-check the patch against the new "
            "plugin version."
        )

    @wraps(original)
    def drain_unified_log(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not hasattr(self, "_event_logs"):
            self._event_logs = {}
        return original(self, *args, **kwargs)

    executor_class._drain_unified_log = drain_unified_log
    executor_class._driftless_star_event_logs_fix = True
    return True


def main() -> None:
    """Patch the installed plugin and enter Snakemake's normal CLI."""
    import snakemake_executor_plugin_htcondor as plugin

    patch_htcondor_executor(plugin.Executor)
    if run_id := os.environ.get(RUN_ID_ENV):
        patch_htcondor_batch_names(plugin.htcondor, run_id)
    from snakemake.cli import main as snakemake_main

    snakemake_main()


if __name__ == "__main__":
    main()
