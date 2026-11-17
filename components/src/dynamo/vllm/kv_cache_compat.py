# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Version-tolerant reads of vLLM KV-cache group and manager metadata.

vLLM #61452 moved the KV-cache allocation policy out of ``KVCacheSpec``:

* before: ``KVCacheGroupSpec(layer_names, kv_cache_spec)``, where the spec class
  (``FullAttentionSpec``, ``SlidingWindowSpec``, ``MambaSpec``, ...) carried the
  policy, and each ``SingleTypeKVCacheManager`` kept ``.kv_cache_spec``;
* after: ``KVCacheGroup(kv_cache_specs, manager_config, block_size,
  dcp_sharded, ...)``, where ``manager_config.manager_cls`` (and its
  ``params``) carries the policy and the specs only describe tensor layout.
  ``FullAttentionSpec``/``SlidingWindowSpec`` no longer exist and managers have
  no ``.kv_cache_spec``.

Everything here keeps the pre-#61452 behaviour byte-for-byte when the legacy
classes are importable (``LEGACY_KV_CACHE_SPECS``) and maps the same questions
onto the manager policy otherwise. Only metadata is read; nothing is mutated.
"""

from __future__ import annotations

from typing import Any

from vllm.v1.kv_cache_interface import MambaSpec

try:  # vLLM before #61452
    from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec
except ImportError:  # vLLM with #61452: policy lives in ManagerConfig
    FullAttentionSpec = None  # type: ignore[assignment,misc]
    SlidingWindowSpec = None  # type: ignore[assignment,misc]

LEGACY_KV_CACHE_SPECS: bool = FullAttentionSpec is not None

# Layout kinds used by the benchmark warm-up gate.
LAYOUT_FULL = "full"
LAYOUT_WINDOW = "window"
LAYOUT_STATE = "state"
LAYOUT_OTHER = "other"


def _manager_classes() -> dict[str, Any]:
    """vLLM manager classes by name (missing ones are absent)."""
    from vllm.v1.core import single_type_kv_cache_manager as stm

    names = (
        "FullAttentionManager",
        "SlidingWindowManager",
        "MambaManager",
        "CircularBufferManager",
    )
    return {name: getattr(stm, name) for name in names if hasattr(stm, name)}


def is_state_spec(spec: Any) -> bool:
    """Recurrent (Mamba/KDA) state spec; the predicate Dynamo always used."""
    return isinstance(spec, MambaSpec) or "Mamba" in type(spec).__name__


def group_distinct_specs(group: Any) -> tuple[Any, ...]:
    """The distinct per-layer specs of a KV-cache group (either API)."""
    if hasattr(group, "kv_cache_spec"):
        spec = group.kv_cache_spec
        return () if spec is None else (spec,)
    distinct = getattr(group, "distinct_specs", None)
    if distinct is not None:
        return tuple(distinct)
    specs = getattr(group, "kv_cache_specs", None) or {}
    return tuple(dict.fromkeys(specs.values()))


def group_spec(group: Any) -> Any:
    """The group's representative spec.

    Legacy groups: ``group.kv_cache_spec`` exactly (``None`` if absent).
    #61452 groups: the first distinct per-layer spec (all layers of a group
    share one manager policy, so its class names the kind of cache held).
    """
    if hasattr(group, "kv_cache_spec") or not hasattr(group, "kv_cache_specs"):
        return getattr(group, "kv_cache_spec", None)
    specs = group_distinct_specs(group)
    return specs[0] if specs else None


def _manager_cls(group_or_manager: Any) -> Any:
    config = getattr(group_or_manager, "manager_config", None)
    return getattr(config, "manager_cls", None)


def _issubclass(cls: Any, name: str) -> bool:
    target = _manager_classes().get(name)
    return isinstance(cls, type) and target is not None and issubclass(cls, target)


def group_is_state(group: Any) -> bool:
    """Whether a KV-cache group holds recurrent (Mamba/KDA) state.

    Legacy: ``isinstance(group.kv_cache_spec, MambaSpec)`` exactly.
    """
    if hasattr(group, "kv_cache_spec"):
        return isinstance(group.kv_cache_spec, MambaSpec)
    if _issubclass(_manager_cls(group), "MambaManager"):
        return True
    specs = group_distinct_specs(group)
    return bool(specs) and all(isinstance(spec, MambaSpec) for spec in specs)


def group_layout_kind(group: Any) -> str:
    """Classify a group as full attention, finite window, recurrent state or other.

    Legacy (spec classes): ``SlidingWindowSpec`` with ``sliding_window > 0`` is
    a finite window, a Mamba spec is state, ``FullAttentionSpec`` (MLA, RSWA and
    sink subclasses included) is full. #61452: the same on the manager policy
    (``SlidingWindowManager`` with ``params.sliding_window > 0``,
    ``MambaManager``, ``FullAttentionManager`` and subclasses).
    """
    if hasattr(group, "kv_cache_spec"):
        spec = group.kv_cache_spec
        if (
            SlidingWindowSpec is not None
            and isinstance(spec, SlidingWindowSpec)
            and spec.sliding_window > 0
        ):
            return LAYOUT_WINDOW
        if is_state_spec(spec):
            return LAYOUT_STATE
        if FullAttentionSpec is not None and isinstance(spec, FullAttentionSpec):
            return LAYOUT_FULL
        return LAYOUT_OTHER

    cls = _manager_cls(group)
    if _issubclass(cls, "SlidingWindowManager"):
        params = getattr(getattr(group, "manager_config", None), "params", None)
        window = getattr(params, "sliding_window", 0) or 0
        return LAYOUT_WINDOW if window > 0 else LAYOUT_OTHER
    if _issubclass(cls, "MambaManager"):
        return LAYOUT_STATE
    if _issubclass(cls, "FullAttentionManager"):
        return LAYOUT_FULL
    specs = group_distinct_specs(group)
    if specs and all(is_state_spec(spec) for spec in specs):
        return LAYOUT_STATE
    return LAYOUT_OTHER


def manager_spec(manager: Any) -> Any:
    """``manager.kv_cache_spec`` where it exists (pre-#61452), else ``None``."""
    return getattr(manager, "kv_cache_spec", None)


def manager_is_state(manager: Any, *, strict: bool = False) -> bool:
    """Whether a single-type manager serves recurrent state.

    Legacy: ``isinstance(spec, MambaSpec)`` when ``strict``, else
    ``is_state_spec(spec)``, on ``manager.kv_cache_spec`` exactly as before.
    #61452: ``isinstance(manager, MambaManager)``.
    """
    if hasattr(manager, "kv_cache_spec"):
        spec = manager.kv_cache_spec
        return isinstance(spec, MambaSpec) if strict else is_state_spec(spec)
    mamba = _manager_classes().get("MambaManager")
    return mamba is not None and isinstance(manager, mamba)


def manager_num_prefill_checkpoint_blocks(manager: Any) -> int:
    """Mamba 'align' prefill-checkpoint blocks of a manager (0 if none).

    Legacy: ``manager.kv_cache_spec.num_prefill_checkpoint_blocks`` (default 0),
    as before. #61452: ``manager.manager_config.params`` holds it.
    """
    if hasattr(manager, "kv_cache_spec"):
        return getattr(manager.kv_cache_spec, "num_prefill_checkpoint_blocks", 0)
    params = getattr(getattr(manager, "manager_config", None), "params", None)
    return getattr(params, "num_prefill_checkpoint_blocks", 0)


def manager_is_circular_table(manager: Any) -> bool:
    """A fixed-length ring table excluded from prefix caching.

    Legacy: the spec's ``participates_in_prefix_caching`` is ``False``, as
    before. #61452: ``CircularBufferManager`` (the ring-buffer policy).
    """
    if hasattr(manager, "kv_cache_spec"):
        spec = manager.kv_cache_spec
        return getattr(spec, "participates_in_prefix_caching", True) is False
    ring = _manager_classes().get("CircularBufferManager")
    return ring is not None and isinstance(manager, ring)


def group_policy_fields(group: Any) -> dict[str, Any]:
    """Manager policy of a #61452 group for provenance records ({} on legacy)."""
    if hasattr(group, "kv_cache_spec"):
        return {}
    config = getattr(group, "manager_config", None)
    if config is None:
        return {}
    out: dict[str, Any] = {"manager": getattr(config.manager_cls, "__name__", None)}
    params = getattr(config, "params", None)
    for name in ("sliding_window", "attention_chunk_size", "mamba_cache_mode"):
        if hasattr(params, name):
            out[name] = getattr(params, name)
    for name in ("block_size", "dcp_sharded"):
        if hasattr(group, name):
            out[name] = getattr(group, name)
    return out
