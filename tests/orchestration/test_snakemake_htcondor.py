"""Tests for the submit-side HTCondor executor compatibility launcher."""

from __future__ import annotations

from src.snakemake_htcondor import (
    _run_batch_name,
    patch_htcondor_batch_names,
    patch_htcondor_executor,
)


def test_event_logs_are_initialized_before_unified_log_is_drained() -> None:
    class FakeExecutor:
        def _drain_unified_log(self):
            self._event_logs.setdefault(123, []).append("submitted")

    assert patch_htcondor_executor(FakeExecutor) is True
    executor = FakeExecutor()
    executor._drain_unified_log()

    assert executor._event_logs == {123: ["submitted"]}
    assert patch_htcondor_executor(FakeExecutor) is False


def test_batch_name_contains_config_id_and_rule_without_jobid() -> None:
    assert (
        _run_batch_name("DGDvAUqji95R8kRxZmucCg6", "stage1_vmec-2")
        == "DGDvAUqji95R8kRxZmucCg6_stage1_vmec"
    )


def test_submit_wrapper_rewrites_actual_htcondor_batch_name() -> None:
    submitted: list[dict] = []

    class FakeHTCondor:
        @staticmethod
        def Submit(description):
            submitted.append(dict(description))
            return "submission"

    assert patch_htcondor_batch_names(FakeHTCondor, "config123") is True
    description = {"batch_name": "stage2_boozer-7", "request_cpus": "1"}

    assert FakeHTCondor.Submit(description) == "submission"
    assert submitted == [{"batch_name": "config123_stage2_boozer", "request_cpus": "1"}]
    assert patch_htcondor_batch_names(FakeHTCondor, "config123") is False
