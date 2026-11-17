# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KV-cache group/manager metadata reads across vLLM's #61452 API change.

Legacy groups (``kv_cache_spec``) must classify exactly as before; #61452 groups
(``kv_cache_specs`` + ``manager_config``) must classify by manager policy.
Fake groups cover both shapes on any vLLM; real vLLM objects cover the
installed one.
"""

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core import single_type_kv_cache_manager as stm
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import MambaSpec

from dynamo.vllm import instrumented_scheduler as instrumented_scheduler_module
from dynamo.vllm import kv_cache_compat as compat

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def _mamba_spec():
    kwargs = (
        {"mamba_cache_mode": "align"}
        if "mamba_cache_mode" in MambaSpec.__dataclass_fields__
        else {}
    )
    return MambaSpec(block_size=16, shapes=(), dtypes=(), **kwargs)


def _policy_group(manager_cls, specs, **params):
    """A #61452-shaped group (no kv_cache_spec attribute)."""
    return SimpleNamespace(
        kv_cache_specs=dict(specs),
        manager_config=SimpleNamespace(
            manager_cls=manager_cls, params=SimpleNamespace(**params)
        ),
        block_size=16,
        dcp_sharded=True,
    )


# -- #61452 shape (fakes; runs on every vLLM) --------------------------------


def test_policy_group_layout_kinds():
    full = _policy_group(stm.FullAttentionManager, {"a": object()})
    window = _policy_group(stm.SlidingWindowManager, {"w": object()}, sliding_window=8)
    no_window = _policy_group(
        stm.SlidingWindowManager, {"w": object()}, sliding_window=0
    )
    state = _policy_group(stm.MambaManager, {"m": _mamba_spec()})
    other = _policy_group(stm.CrossAttentionManager, {"x": object()})

    assert compat.group_layout_kind(full) == compat.LAYOUT_FULL
    assert compat.group_layout_kind(window) == compat.LAYOUT_WINDOW
    assert compat.group_layout_kind(no_window) == compat.LAYOUT_OTHER
    assert compat.group_layout_kind(state) == compat.LAYOUT_STATE
    assert compat.group_layout_kind(other) == compat.LAYOUT_OTHER
    assert compat.group_is_state(state)
    assert not compat.group_is_state(full)
    assert isinstance(compat.group_spec(state), MambaSpec)
    assert compat.group_policy_fields(window) == {
        "manager": "SlidingWindowManager",
        "sliding_window": 8,
        "block_size": 16,
        "dcp_sharded": True,
    }


def test_policy_manager_reads_params_not_spec():
    mamba = stm.MambaManager.__new__(stm.MambaManager)
    mamba.manager_config = SimpleNamespace(
        params=SimpleNamespace(num_prefill_checkpoint_blocks=2)
    )
    full = stm.FullAttentionManager.__new__(stm.FullAttentionManager)
    full.manager_config = SimpleNamespace(params=SimpleNamespace())

    assert compat.manager_is_state(mamba)
    assert compat.manager_is_state(mamba, strict=True)
    assert not compat.manager_is_state(full)
    assert compat.manager_num_prefill_checkpoint_blocks(mamba) == 2
    assert compat.manager_num_prefill_checkpoint_blocks(full) == 0
    assert not compat.manager_is_circular_table(full)


# -- legacy shape (fakes; runs on every vLLM) ---------------------------------


def test_legacy_shape_keeps_spec_predicates():
    named_mamba = SimpleNamespace(kv_cache_spec=type("MambaSpec", (), {})())
    real_mamba = SimpleNamespace(kv_cache_spec=_mamba_spec())
    ring = SimpleNamespace(
        kv_cache_spec=SimpleNamespace(
            participates_in_prefix_caching=False, num_prefill_checkpoint_blocks=3
        )
    )

    # Name-based state predicate (live state) vs isinstance (random state),
    # exactly as InstrumentedScheduler always distinguished them.
    assert compat.manager_is_state(named_mamba)
    assert not compat.manager_is_state(named_mamba, strict=True)
    assert compat.manager_is_state(real_mamba, strict=True)
    assert compat.manager_is_circular_table(ring)
    assert compat.manager_num_prefill_checkpoint_blocks(ring) == 3
    assert compat.manager_num_prefill_checkpoint_blocks(named_mamba) == 0
    assert compat.group_spec(named_mamba) is named_mamba.kv_cache_spec
    assert compat.group_spec(SimpleNamespace(kv_cache_spec=None)) is None
    assert compat.group_is_state(real_mamba)
    assert not compat.group_is_state(named_mamba)  # isinstance, as before
    assert compat.group_policy_fields(real_mamba) == {}
    assert compat.group_layout_kind(named_mamba) == compat.LAYOUT_STATE


@pytest.mark.skipif(not compat.LEGACY_KV_CACHE_SPECS, reason="needs vLLM before #61452")
def test_legacy_spec_classes_classify_as_before():
    from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

    kw = dict(block_size=16, num_kv_heads=1, head_size=8, dtype=torch.bfloat16)
    full = SimpleNamespace(kv_cache_spec=FullAttentionSpec(**kw))
    window = SimpleNamespace(kv_cache_spec=SlidingWindowSpec(**kw, sliding_window=4))
    assert compat.group_layout_kind(full) == compat.LAYOUT_FULL
    assert compat.group_layout_kind(window) == compat.LAYOUT_WINDOW


# -- real objects of the installed vLLM ---------------------------------------


@pytest.mark.skipif(compat.LEGACY_KV_CACHE_SPECS, reason="needs vLLM with #61452")
def test_real_kv_cache_groups_classify_by_manager_policy():
    from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheGroup

    spec = AttentionSpec(
        block_size=16, num_kv_heads=1, head_size=8, dtype=torch.bfloat16
    )
    full = KVCacheGroup.from_layers({"a": stm.FullAttentionManager.layer_config(spec)})
    window = KVCacheGroup.from_layers(
        {"w": stm.SlidingWindowManager.layer_config(spec, sliding_window=8)}
    )
    state = KVCacheGroup.from_layers(
        {
            "m": stm.MambaManager.layer_config(
                _mamba_spec(), mamba_cache_mode="align", num_prefill_checkpoint_blocks=1
            )
        }
    )

    assert not hasattr(full, "kv_cache_spec")
    assert compat.group_layout_kind(full) == compat.LAYOUT_FULL
    assert compat.group_layout_kind(window) == compat.LAYOUT_WINDOW
    assert compat.group_layout_kind(state) == compat.LAYOUT_STATE
    assert compat.group_is_state(state)
    assert compat.group_spec(full) is spec
    assert compat.group_policy_fields(state)["mamba_cache_mode"] == "align"


def test_scheduler_outputs_build_on_installed_vllm():
    """Benchmark-built outputs must construct whichever fields vLLM has."""
    stub = instrumented_scheduler_module.InstrumentedScheduler.__new__(
        instrumented_scheduler_module.InstrumentedScheduler
    )
    stub.kv_cache_manager = SimpleNamespace(num_kv_cache_groups=2)
    kwarg = stub._legacy_common_prefix_blocks_kwarg()
    output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
        **kwarg,
    )
    if "num_common_prefix_blocks" in SchedulerOutput.__dataclass_fields__:
        assert output.num_common_prefix_blocks == [0, 0]
    else:
        assert kwarg == {}
