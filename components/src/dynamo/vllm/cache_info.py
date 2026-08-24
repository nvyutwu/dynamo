# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any

from vllm.config import VllmConfig
from vllm.v1.engine.async_llm import AsyncLLM

logger = logging.getLogger(__name__)

DYNAMO_KV_EVENT_BLOCK_SIZE_KEY = "dynamo_kv_event_block_size"
MAIN_ATTENTION_KV_CACHE_KINDS = {
    "full_attention",
    "mla_attention",
    "sink_full_attention",
}


def _decode_context_parallel_size(vllm_config: VllmConfig) -> int:
    """Return vLLM's decode-context-parallel world size, or 1 when unavailable.

    vLLM scales attention-group blocks by this factor before publishing KV events, while
    ``get_kv_cache_group_metadata`` reports the unscaled spec value. Older configs and test stubs
    may not expose the field, so preserve their pre-DCP behavior instead of failing startup.
    """
    parallel_config = getattr(vllm_config, "parallel_config", None)
    try:
        dcp_world_size = int(
            getattr(parallel_config, "decode_context_parallel_size", 1) or 1
        )
    except (TypeError, ValueError):
        return 1
    return dcp_world_size if dcp_world_size > 1 else 1


def get_configured_kv_event_block_size(vllm_config: VllmConfig) -> int:
    """Return the configured KV event block size, falling back to vLLM's cache block size."""
    additional_config = vllm_config.additional_config or {}
    cached = additional_config.get(DYNAMO_KV_EVENT_BLOCK_SIZE_KEY)
    if cached is not None:
        return cached
    return vllm_config.cache_config.block_size * _decode_context_parallel_size(
        vllm_config
    )


def select_main_attention_block_size(
    group_metadata: list[dict[str, Any]],
    fallback_block_size: int,
) -> int:
    """Select the main-attention KV block size from engine cache-group metadata."""
    if not group_metadata:
        return fallback_block_size

    for group in group_metadata:
        if group.get("kind") in MAIN_ATTENTION_KV_CACHE_KINDS:
            return group.get("block_size", fallback_block_size)

    return fallback_block_size


async def configure_kv_event_block_size(
    engine: AsyncLLM,
    vllm_config: VllmConfig,
) -> int:
    """Fetch engine cache-group metadata and cache the KV event block size on vLLM config."""
    fallback_block_size = vllm_config.cache_config.block_size
    try:
        group_metadata = await engine.engine_core.call_utility_async(
            "get_kv_cache_group_metadata"
        )
    except Exception as e:
        logger.warning(
            "Failed to fetch KV cache group metadata; falling back to "
            "vLLM cache_config.block_size: %s",
            e,
        )
        kv_event_block_size = fallback_block_size
    else:
        kv_event_block_size = select_main_attention_block_size(
            group_metadata,
            fallback_block_size,
        )

    dcp_world_size = _decode_context_parallel_size(vllm_config)
    engine_block_size = kv_event_block_size
    kv_event_block_size = engine_block_size * dcp_world_size
    logger.info(
        "KV-event block size %d = main-attention block %d x "
        "decode_context_parallel_size %d",
        kv_event_block_size,
        engine_block_size,
        dcp_world_size,
    )

    if vllm_config.additional_config is None:
        vllm_config.additional_config = {}
    vllm_config.additional_config[DYNAMO_KV_EVENT_BLOCK_SIZE_KEY] = kv_event_block_size
    return kv_event_block_size
