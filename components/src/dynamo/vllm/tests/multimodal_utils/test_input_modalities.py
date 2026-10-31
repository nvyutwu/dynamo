# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from dynamo.vllm.multimodal_utils.input_modalities import (
    INPUT_MODALITY_LIMITS_RUNTIME_KEY,
    publish_input_modality_limits,
)

pytestmark = [pytest.mark.unit, pytest.mark.vllm, pytest.mark.gpu_0, pytest.mark.pre_merge]


class _RuntimeConfig:
    def __init__(self):
        self.values = {}

    def set_engine_specific(self, key, value):
        self.values[key] = value


def _registry(limits):
    info = SimpleNamespace(allowed_mm_limits=limits)
    return SimpleNamespace(get_processing_info=lambda model_config: info)


def _publish(model_config, limits=None):
    runtime_config = _RuntimeConfig()
    publish_input_modality_limits(
        runtime_config,
        SimpleNamespace(model_config=model_config),
        registry=_registry(limits or {}),
    )
    raw = runtime_config.values.get(INPUT_MODALITY_LIMITS_RUNTIME_KEY)
    return None if raw is None else json.loads(raw)


def _model(**kwargs):
    kwargs.setdefault("multimodal_config", object())
    return SimpleNamespace(**kwargs)


@pytest.mark.parametrize(
    ("model_config", "expected"),
    [
        (_model(is_multimodal_model=True), {"image": 10}),
        (_model(is_multimodal_model=lambda: True), {"image": 10}),
        (_model(is_multimodal_model=False), {}),
        (SimpleNamespace(), None),
        (_model(is_multimodal_model=Mock()), None),
    ],
    ids=["property-true", "method-true", "text-only", "missing", "non-bool"],
)
def test_publishes_limits_only_for_a_boolean_capability(model_config, expected):
    assert _publish(model_config, {"image": 10}) == expected


def test_custom_internal_modality_publishes_nothing():
    assert _publish(_model(is_multimodal_model=True), {"vision_chunk": 1}) is None


@pytest.mark.parametrize("limit", [-1, True, "2"])
def test_invalid_limit_raises(limit):
    with pytest.raises(ValueError):
        _publish(_model(is_multimodal_model=True), {"image": limit})
