# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ipaddress
import json
import logging
import secrets
from collections.abc import Mapping, MutableMapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.engine.arg_utils import AsyncEngineArgs

from dynamo.common.constants import (
    ROUTER_HINT_INVENTORY_EPOCH_RUNTIME_KEY,
    ROUTER_HINT_RUNTIME_CAPABILITY_KEY,
    ROUTER_HINT_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY,
    ROUTER_HINT_WORKER_TYPE_RUNTIME_KEY,
)
from dynamo.llm import ModelRuntimeConfig, WorkerType


logger = logging.getLogger(__name__)

# One random generation per backend process. A restarted process advertises a new
# value even when it reuses the same stable worker identity or control endpoint.
_ROUTER_HINT_INVENTORY_EPOCH = secrets.randbits(64)


def _secondary_tiers(engine_args: AsyncEngineArgs) -> list[Mapping[str, Any]]:
    kv_config = getattr(engine_args, "kv_transfer_config", None)
    extra_config = getattr(kv_config, "kv_connector_extra_config", None)
    if not isinstance(extra_config, Mapping):
        return []
    secondary_tiers = extra_config.get("secondary_tiers")
    if not isinstance(secondary_tiers, list):
        return []
    return [tier for tier in secondary_tiers if isinstance(tier, Mapping)]


def _supports_router_hint(tier: Mapping[str, Any]) -> bool:
    capabilities = tier.get("router_capabilities")
    return isinstance(capabilities, list) and (
        ROUTER_HINT_RUNTIME_CAPABILITY_KEY in capabilities
    )


def _router_hint_tiers(engine_args: AsyncEngineArgs) -> list[Mapping[str, Any]]:
    return [
        tier for tier in _secondary_tiers(engine_args) if _supports_router_hint(tier)
    ]


def _router_hint_source_host(host: str | None) -> str | None:
    if not host:
        return None
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return host
    if address.is_unspecified:
        return None
    if address.version == 6:
        return f"[{address.compressed}]"
    return address.compressed


def _router_hint_source_port(configured_port: object) -> int | None:
    if isinstance(configured_port, bool) or not isinstance(configured_port, (int, str)):
        return None
    try:
        control_port = int(configured_port)
    except ValueError:
        return None
    return control_port if 0 < control_port <= 65535 else None


def _router_hint_source_control_endpoints(
    tier: Mapping[str, Any], dp_range: tuple[int, int]
) -> dict[str, str] | None:
    dp_start, dp_size = dp_range
    if dp_start < 0 or dp_size <= 0:
        return None
    control_ports = tier.get("control_ports")
    if not isinstance(control_ports, list):
        raise ValueError("router_hint support requires control_ports to be a list")
    if len(control_ports) != dp_size:
        raise ValueError(
            "router_hint support requires control_ports to contain exactly "
            f"{dp_size} entries for the worker-local DP ranks; got {len(control_ports)}"
        )
    configured_host = tier.get("control_advertise_host")
    host = _router_hint_source_host(
        configured_host if isinstance(configured_host, str) else None
    )
    if host is None:
        return None

    endpoints: dict[str, str] = {}
    for local_dp_rank, global_dp_rank in enumerate(range(dp_start, dp_start + dp_size)):
        control_port = _router_hint_source_port(control_ports[local_dp_rank])
        if control_port is None:
            return None
        endpoints[str(global_dp_rank)] = f"tcp://{host}:{control_port}"
    return endpoints


def _router_hint_worker_type(worker_type: WorkerType) -> str | None:
    role = getattr(worker_type, "value", None)
    if not isinstance(role, str):
        role = str(worker_type)
    if role == "agg":
        role = "aggregated"
    return role if role in {"aggregated", "prefill", "decode"} else None


def configure_router_hint_inventory_epoch(engine_args: AsyncEngineArgs) -> bool:
    """Inject the source generation before vLLM constructs secondary tiers."""
    router_hint_tiers = _router_hint_tiers(engine_args)
    if not router_hint_tiers:
        return False
    if len(router_hint_tiers) > 1:
        raise ValueError(
            "router_hint support requires exactly one router-hint-capable secondary tier"
        )
    tier = router_hint_tiers[0]
    if not isinstance(tier, MutableMapping):
        raise ValueError("router_hint support requires a mutable tier configuration")
    tier["inventory_epoch"] = _ROUTER_HINT_INVENTORY_EPOCH
    return True


def enable_router_hint_support(
    runtime_config: ModelRuntimeConfig,
    engine_args: AsyncEngineArgs,
    worker_type: WorkerType,
    dp_range: tuple[int, int] = (0, 1),
) -> None:
    router_hint_worker_type = _router_hint_worker_type(worker_type)
    if router_hint_worker_type is None:
        return

    if not configure_router_hint_inventory_epoch(engine_args):
        return
    router_hint_tiers = _router_hint_tiers(engine_args)

    # A tier whose control endpoint is not advertisable (the common case is a
    # wildcard `control_advertise_host` such as 0.0.0.0 or ::) cannot serve as a
    # remote source. That is not a reason to fail registration: the router
    # requires an endpoint only for the source side, so the worker still
    # advertises the capability, its worker type, and its inventory epoch and
    # remains usable as a hint *target*. A malformed `control_ports` list is a
    # different matter and still raises out of the helper below.
    endpoints = _router_hint_source_control_endpoints(router_hint_tiers[0], dp_range)
    if endpoints is None:
        logger.warning(
            "router_hint: no advertisable source control endpoint for DP ranks "
            "%s..%s (check control_advertise_host and control_ports); this worker "
            "will consume router hints but will not be offered as a KVCR source",
            dp_range[0],
            dp_range[0] + dp_range[1] - 1,
        )
    else:
        runtime_config.set_engine_specific(
            ROUTER_HINT_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY, json.dumps(endpoints)
        )
    runtime_config.set_engine_specific(
        ROUTER_HINT_INVENTORY_EPOCH_RUNTIME_KEY,
        json.dumps(_ROUTER_HINT_INVENTORY_EPOCH),
    )
    runtime_config.set_engine_specific(
        ROUTER_HINT_WORKER_TYPE_RUNTIME_KEY, json.dumps(router_hint_worker_type)
    )
    runtime_config.set_engine_specific(
        ROUTER_HINT_RUNTIME_CAPABILITY_KEY, json.dumps(True)
    )
