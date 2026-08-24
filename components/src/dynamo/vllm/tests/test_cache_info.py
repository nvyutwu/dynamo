# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from dynamo.vllm.cache_info import (
    configure_kv_event_block_size,
    get_configured_kv_event_block_size,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_1,
    pytest.mark.xpu_1,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("dcp_size", "expected"), [(1, 1536), (2, 3072)])
async def test_kv_event_block_size_tracks_dcp(dcp_size: int, expected: int) -> None:
    config = SimpleNamespace(
        additional_config=None,
        cache_config=SimpleNamespace(block_size=64),
        parallel_config=SimpleNamespace(decode_context_parallel_size=dcp_size),
    )
    engine = SimpleNamespace(
        engine_core=SimpleNamespace(
            call_utility_async=AsyncMock(
                return_value=[{"kind": "mla_attention", "block_size": 1536}]
            )
        )
    )

    assert await configure_kv_event_block_size(engine, config) == expected
    assert get_configured_kv_event_block_size(config) == expected


def test_uncached_fallback_tracks_dcp() -> None:
    config = SimpleNamespace(
        additional_config=None,
        cache_config=SimpleNamespace(block_size=64),
        parallel_config=SimpleNamespace(decode_context_parallel_size=2),
    )

    assert get_configured_kv_event_block_size(config) == 128
