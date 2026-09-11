# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in FPM log envelope; encoding belongs in the publisher thread."""

import json


def encode_record(metrics: dict, *, dropped: int, observed_ns: int) -> str:
    """Preserve native coordinates and units without claiming CUDA duration."""
    return json.dumps(
        {
            "event": "dynamo_fpm_record",
            "schema_version": 1,
            "observed_at_unix_ns": observed_ns,
            "wall_time_unit": "seconds",
            "timing_scope": "scheduler_wall_time",
            "publisher_queue_dropped_total": dropped,
            "fpm": metrics,
        },
        separators=(",", ":"),
        allow_nan=False,
    )
