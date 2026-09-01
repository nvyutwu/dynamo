# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from dynamo.common.constants import (
    ROUTER_HINT_INVENTORY_EPOCH_RUNTIME_KEY,
    ROUTER_HINT_RUNTIME_CAPABILITY_KEY,
    ROUTER_HINT_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY,
    ROUTER_HINT_WORKER_TYPE_RUNTIME_KEY,
)
from dynamo.llm import WorkerType
from dynamo.vllm.router_hints import enable_router_hint_support
from dynamo.vllm.router_hints import _ROUTER_HINT_INVENTORY_EPOCH


def engine_args(tiers):
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={"secondary_tiers": tiers}
        )
    )


def test_enable_router_hint_support_maps_explicit_ports_to_global_dp_ranks():
    runtime_config = MagicMock()
    tier = {
        "router_capabilities": ["router_hint"],
        "control_advertise_host": "worker-a",
        "control_ports": [24000, "24001"],
    }

    enable_router_hint_support(
        runtime_config, engine_args([tier]), WorkerType.Prefill, (4, 2)
    )

    runtime_config.set_engine_specific.assert_any_call(
        ROUTER_HINT_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY,
        json.dumps({"4": "tcp://worker-a:24000", "5": "tcp://worker-a:24001"}),
    )
    runtime_config.set_engine_specific.assert_any_call(
        ROUTER_HINT_WORKER_TYPE_RUNTIME_KEY, json.dumps("prefill")
    )
    runtime_config.set_engine_specific.assert_any_call(
        ROUTER_HINT_RUNTIME_CAPABILITY_KEY, json.dumps(True)
    )
    runtime_config.set_engine_specific.assert_any_call(
        ROUTER_HINT_INVENTORY_EPOCH_RUNTIME_KEY,
        json.dumps(_ROUTER_HINT_INVENTORY_EPOCH),
    )
    assert tier["inventory_epoch"] == _ROUTER_HINT_INVENTORY_EPOCH


def test_enable_router_hint_support_normalizes_aggregated_and_ipv6():
    runtime_config = MagicMock()
    tier = {
        "router_capabilities": ["router_hint"],
        "control_advertise_host": "2001:db8::1",
        "control_ports": [23280],
    }

    enable_router_hint_support(
        runtime_config, engine_args([tier]), WorkerType.Aggregated
    )

    runtime_config.set_engine_specific.assert_any_call(
        ROUTER_HINT_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY,
        json.dumps({"0": "tcp://[2001:db8::1]:23280"}),
    )
    runtime_config.set_engine_specific.assert_any_call(
        ROUTER_HINT_WORKER_TYPE_RUNTIME_KEY, json.dumps("aggregated")
    )


@pytest.mark.parametrize(
    "tier,dp_range,error",
    [
        (
            {
                "router_capabilities": ["router_hint"],
                "control_advertise_host": "worker-a",
                "control_port": 23280,
            },
            (0, 1),
            "control_ports to be a list",
        ),
        (
            {
                "router_capabilities": ["router_hint"],
                "control_advertise_host": "worker-a",
                "control_ports": [23280],
            },
            (4, 2),
            "exactly 2 entries",
        ),
        (
            {
                "router_capabilities": ["router_hint"],
                "control_advertise_host": "0.0.0.0",
                "control_ports": [23280],
            },
            (0, 1),
            "advertisable source control endpoints",
        ),
    ],
)
def test_enable_router_hint_support_rejects_incomplete_endpoint_maps(
    tier, dp_range, error
):
    runtime_config = MagicMock()
    with pytest.raises(ValueError, match=error):
        enable_router_hint_support(
            runtime_config, engine_args([tier]), WorkerType.Prefill, dp_range
        )
    runtime_config.set_engine_specific.assert_not_called()


def test_enable_router_hint_support_skips_without_capability_or_supported_role():
    runtime_config = MagicMock()
    enable_router_hint_support(
        runtime_config,
        engine_args([{"control_advertise_host": "worker-a", "control_ports": [1]}]),
        WorkerType.Prefill,
    )
    enable_router_hint_support(
        runtime_config,
        engine_args(
            [
                {
                    "router_capabilities": ["router_hint"],
                    "control_advertise_host": "worker-a",
                    "control_ports": [1],
                }
            ]
        ),
        WorkerType.Encode,
    )
    runtime_config.set_engine_specific.assert_not_called()
