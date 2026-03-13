# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import os

import uvloop

from dynamo.common.utils.graceful_shutdown import install_signal_handlers
from dynamo.common.utils.otel_instrumentation import init_dynamo_otel_metrics
from dynamo.common.utils.runtime import create_runtime
from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.trtllm.args import parse_args
from dynamo.trtllm.workers import init_worker

configure_dynamo_logging()
shutdown_endpoints: list = []


async def worker():
    config = parse_args()

    shutdown_event = asyncio.Event()
    runtime, loop = create_runtime(
        discovery_backend=config.discovery_backend,
        request_plane=config.request_plane,
        event_plane=config.event_plane,
        use_kv_events=config.use_kv_events,
    )

    install_signal_handlers(loop, runtime, shutdown_endpoints, shutdown_event)

    # Initialize OTEL metrics bridge for worker metrics (scrapes DYN_SYSTEM_PORT /metrics)
    # Only active if both OTEL_EXPORTER_OTLP_METRICS_ENDPOINT and DYN_SYSTEM_PORT are set
    _system_port = os.getenv("DYN_SYSTEM_PORT", "-1")
    if _system_port not in ("-1", "0", ""):
        init_dynamo_otel_metrics(http_port=int(_system_port))

    logging.info(f"Initializing the worker with config: {config}")
    await init_worker(runtime, config, shutdown_event, shutdown_endpoints)


def main():
    uvloop.run(worker())


if __name__ == "__main__":
    main()
