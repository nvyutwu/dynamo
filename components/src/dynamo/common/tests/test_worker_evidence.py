# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from dynamo.vllm import worker_evidence

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


class FakeProcess:
    def __init__(self):
        self.pid = 41
        self.terminate_calls = 0
        self.wait_calls = []

    def poll(self):
        return None

    def terminate(self):
        self.terminate_calls += 1

    def wait(self, timeout):
        self.wait_calls.append(timeout)


@pytest.fixture(autouse=True)
def reset_collector(monkeypatch):
    monkeypatch.setattr(worker_evidence, "_collector_process", None)
    monkeypatch.setattr(worker_evidence, "_collector_owner_pid", None)


def test_collector_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv(worker_evidence.ENABLED_ENV, raising=False)
    called = False

    def popen(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(worker_evidence.subprocess, "Popen", popen)

    assert worker_evidence.start_worker_evidence_collector() is None
    assert called is False


def test_collector_launch_is_opt_in_non_shell_and_idempotent(monkeypatch):
    monkeypatch.setenv(worker_evidence.ENABLED_ENV, "true")
    process = FakeProcess()
    calls = []

    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return process

    monkeypatch.setattr(worker_evidence.subprocess, "Popen", popen)

    first = worker_evidence.start_worker_evidence_collector()
    second = worker_evidence.start_worker_evidence_collector()

    assert first is process
    assert second is process
    assert len(calls) == 1
    command, options = calls[0]
    assert command == [
        worker_evidence.sys.executable,
        "-u",
        "-m",
        "dynamo.vllm.worker_evidence",
    ]
    assert options["shell"] is False
    assert options["stdin"] is worker_evidence.subprocess.DEVNULL
    assert options["start_new_session"] is False


def test_collector_cleanup_terminates_and_waits(monkeypatch):
    process = FakeProcess()
    monkeypatch.setattr(worker_evidence, "_collector_process", process)
    monkeypatch.setattr(
        worker_evidence, "_collector_owner_pid", worker_evidence.os.getpid()
    )

    worker_evidence.stop_worker_evidence_collector()

    assert process.terminate_calls == 1
    assert process.wait_calls == [5]
    assert worker_evidence._collector_process is None


def test_event_record_is_physical_inventory_without_tokens_or_cumulative_scores():
    event = type("BlockStored", (), {})()
    event.block_hashes = [101]
    event.parent_block_hash = None
    event.token_ids = [1] * 12288
    event.block_size = 12288
    event.lora_name = None
    event.extra_keys = None
    event.medium = "CPU"
    event.group_idx = 0
    event.locality = "LOCAL"
    event.ownership = "offloading_connector"
    event.kv_cache_spec_kind = "AttentionSpec"
    event.kv_cache_spec_sliding_window = None

    row = worker_evidence.event_record(
        event,
        {},
        ("pod", 0, 0),
        lambda tokens, _block_size: [sum(tokens)],
        lambda parent, local: parent + local,
    )

    assert row["evidence_semantics"] == "physical_tier_inventory"
    assert row["medium"] == "CPU"
    assert row["ownership"] == "offloading_connector"
    assert "token_ids" not in row
    assert "block_hashes" not in row
    assert "hbm_blocks" not in row
    assert "cpu_ram_cumulative_blocks" not in row
    assert "cpu_ram_only_blocks" not in row


def test_metric_snapshot_semantics_are_worker_counters_not_routing_fields():
    parsed = worker_evidence.parse_metrics(
        'vllm:kv_offload_load_bytes_total{model_name="kimi",engine="0",prompt="secret"} 64\n'
    )

    assert parsed["evidence_semantics"] == "worker_offload_metrics"
    assert parsed["samples"][0]["value"] == 64.0
    assert parsed["samples"][0]["labels"] == {"engine": "0", "model_name": "kimi"}
    assert "hbm_blocks" not in parsed
    assert "cpu_ram_cumulative_blocks" not in parsed


def test_unsupported_mapping_still_preserves_external_physical_identity():
    event = type("BlockStored", (), {})()
    event.block_hashes = [2**65 + 7]
    event.parent_block_hash = 2**64 + 9
    event.token_ids = [1] * 256
    event.block_size = 256
    event.lora_name = None
    event.extra_keys = None
    event.medium = "GPU"
    event.group_idx = 0
    event.locality = "LOCAL"
    event.kv_cache_spec_kind = "AttentionSpec"
    event.kv_cache_spec_sliding_window = None

    row = worker_evidence.event_record(
        event, {}, ("pod", 0, 0), lambda *_: [], lambda *_: 0
    )

    assert row["hash_mapping_status"] == "block_size_not_12288"
    assert row["dynamo_sequence_hashes"] is None
    assert row["external_block_hashes"] == ["7"]
    assert row["parent_external_block_hash"] == "9"
    assert row["block_count"] == 1
    assert "token_ids" not in row
    assert "extra_keys" not in row
    assert "lora_name" not in row


def test_remove_preserves_external_hash_count_when_mapping_is_unknown():
    event = type("BlockRemoved", (), {})()
    event.block_hashes = [11, 12]
    event.medium = "CPU"
    event.group_idx = 0
    event.locality = "LOCAL"

    row = worker_evidence.event_record(
        event, {}, ("pod", 0, 0), lambda *_: [], lambda *_: 0
    )

    assert row["hash_mapping_status"] == "remove_mapping_unavailable"
    assert row["external_block_hashes"] == ["11", "12"]
    assert row["parent_external_block_hash"] is None
    assert row["block_count"] == 2


def test_non_main_group_remains_physical_evidence_without_routing_hash_mapping():
    event = type("BlockStored", (), {})()
    event.block_hashes = [101]
    event.parent_block_hash = None
    event.token_ids = [1] * 12288
    event.block_size = 12288
    event.lora_name = None
    event.extra_keys = None
    event.medium = "CPU"
    event.group_idx = 1
    event.locality = "LOCAL"
    event.ownership = None
    event.kv_cache_spec_kind = "AttentionSpec"
    event.kv_cache_spec_sliding_window = None

    row = worker_evidence.event_record(
        event, {}, ("pod", 0, 1), lambda *_: [99], lambda *_: 0
    )

    assert row["external_block_hashes"] == ["101"]
    assert row["dynamo_sequence_hashes"] is None
    assert row["hash_mapping_status"] == "routing_mapping_group_not_main"


def test_identity_map_evicts_oldest_mapping_and_reports_unknown_later():
    identity_map = worker_evidence.IdentityMap(max_entries=1)
    identity_map.remember(("pod", 0, 0, "11"), 101)
    identity_map.remember(("pod", 0, 0, "12"), 102)
    event = type("BlockRemoved", (), {})()
    event.block_hashes = [11]
    event.medium = "CPU"
    event.group_idx = 0
    event.locality = "LOCAL"

    row = worker_evidence.event_record(
        event, identity_map, ("pod", 0, 0), lambda *_: [], lambda *_: 0
    )

    assert identity_map.evictions_total == 1
    assert row["hash_mapping_status"] == "remove_mapping_unavailable"
    assert row["hash_mapping_entries"] == 1
    assert row["hash_mapping_evictions_total"] == 1


def test_invalid_enable_value_fails_before_spawning(monkeypatch):
    monkeypatch.setenv(worker_evidence.ENABLED_ENV, "sometimes")
    with pytest.raises(ValueError, match="must be a boolean"):
        worker_evidence.start_worker_evidence_collector()


def test_all_blocks_cleared_is_an_explicit_physical_inventory_reset():
    event = type("AllBlocksCleared", (), {})()
    row = worker_evidence.event_record(
        event, {}, ("pod", 0, None), lambda *_: [], lambda *_: 0
    )

    assert row == {
        "event_type": "AllBlocksCleared",
        "evidence_semantics": "physical_tier_inventory",
        "scope_reset": True,
    }


def test_fatal_collector_emits_one_terminal_record_and_exits(monkeypatch, capsys):
    def fail():
        raise RuntimeError("initialization failed")

    monkeypatch.setattr(worker_evidence, "run", fail)
    with pytest.raises(SystemExit) as error:
        worker_evidence.main()

    assert error.value.code == 1
    output = capsys.readouterr().out
    assert output.count('"record_type":"collector_fatal"') == 1
    assert "disabled_after_initialization_failure" in output


def test_vllm_entrypoint_starts_collector_after_restore_guard(monkeypatch):
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "dynamo.common.snapshot.restore_context",
        SimpleNamespace(maybe_run_restore_standby_mode=lambda: calls.append("restore")),
    )
    monkeypatch.setitem(
        sys.modules,
        "dynamo.vllm.worker_evidence",
        SimpleNamespace(
            start_worker_evidence_collector=lambda: calls.append("evidence")
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "dynamo.vllm.main",
        SimpleNamespace(main=lambda: calls.append("main")),
    )
    entrypoint = Path(worker_evidence.__file__).with_name("__main__.py")

    runpy.run_path(str(entrypoint), run_name="__main__")

    assert calls == ["restore", "evidence", "main"]


def test_collector_cleanup_kills_child_that_does_not_exit(monkeypatch):
    process = FakeProcess()
    waits = 0
    kill_calls = 0

    def wait(timeout):
        nonlocal waits
        waits += 1
        if waits == 1:
            raise subprocess.TimeoutExpired("collector", timeout)

    def kill():
        nonlocal kill_calls
        kill_calls += 1

    process.wait = wait
    process.kill = kill
    monkeypatch.setattr(worker_evidence, "_collector_process", process)
    monkeypatch.setattr(
        worker_evidence, "_collector_owner_pid", worker_evidence.os.getpid()
    )

    worker_evidence.stop_worker_evidence_collector()

    assert process.terminate_calls == 1
    assert kill_calls == 1
    assert waits == 2


def test_forked_process_cleanup_does_not_terminate_parent_collector(monkeypatch):
    process = FakeProcess()
    monkeypatch.setattr(worker_evidence, "_collector_process", process)
    monkeypatch.setattr(
        worker_evidence, "_collector_owner_pid", worker_evidence.os.getpid() - 1
    )

    worker_evidence.stop_worker_evidence_collector()

    assert process.terminate_calls == 0
    assert worker_evidence._collector_process is process
