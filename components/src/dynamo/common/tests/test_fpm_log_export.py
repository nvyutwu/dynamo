# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import importlib.util
import json
from pathlib import Path

import pytest


def load_exporter():
    path = Path(__file__).parents[1] / "fpm_log_export.py"
    spec = importlib.util.spec_from_file_location("fpm_log_export", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preserves_coordinates_identity_and_units():
    module = load_exporter()
    metrics = {
        "worker_id": "worker-a",
        "dp_rank": 0,
        "counter_id": 19,
        "wall_time": 0.0125,
        "scheduled_requests": {
            "sum_prefill_tokens": 1536,
            "num_decode_requests": 16,
            "sum_decode_kv_tokens": 786432,
        },
    }
    record = json.loads(module.encode_record(metrics, dropped=3, observed_ns=123))
    assert record["fpm"] == metrics
    assert record["wall_time_unit"] == "seconds"
    assert record["timing_scope"] == "scheduler_wall_time"
    assert record["publisher_queue_dropped_total"] == 3
    assert record["observed_at_unix_ns"] == 123


def test_heartbeat_is_retained_for_gap_audit():
    module = load_exporter()
    record = json.loads(
        module.encode_record({"wall_time": 0.0}, dropped=1, observed_ns=9)
    )
    assert record["fpm"]["wall_time"] == 0
    assert record["publisher_queue_dropped_total"] == 1


def test_nonfinite_duration_is_rejected():
    module = load_exporter()
    with pytest.raises(ValueError):
        module.encode_record({"wall_time": float("nan")}, dropped=0, observed_ns=0)
