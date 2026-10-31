# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish effective engine input limits rather than inferring from a model alias."""

import json
import logging

logger = logging.getLogger(__name__)

INPUT_MODALITY_LIMITS_RUNTIME_KEY = "input_modality_limits"


def publish_input_modality_limits(runtime_config, vllm_config, *, registry=None):
    model_config = vllm_config.model_config
    multimodal = getattr(model_config, "is_multimodal_model", None)
    if callable(multimodal):
        # Tolerate a method form, as _is_multimodal_model does (vLLM uses a property).
        multimodal = multimodal()
    if not isinstance(multimodal, bool):
        # Older/config-less adapters (and test doubles) do not declare a capability.
        # Missing or non-boolean is unknown: publish nothing.
        if multimodal is not None:
            logger.debug(
                "is_multimodal_model is %s, not bool; input modality limits not published",
                type(multimodal).__name__,
            )
        return
    limits = {}
    if multimodal:
        mm_config = model_config.multimodal_config
        if mm_config is None:
            raise ValueError(
                "Multimodal engine is missing its multimodal configuration"
            )
        if registry is None:
            from vllm.multimodal import MULTIMODAL_REGISTRY

            registry = MULTIMODAL_REGISTRY
        info = registry.get_processing_info(model_config)
        # A custom internal modality (for example vision_chunk) does not
        # establish its OpenAI input mapping. Preserve unknown behavior rather
        # than accidentally declaring supported image/video inputs disabled.
        allowed_limits = info.allowed_mm_limits
        if not set(allowed_limits).issubset({"image", "audio", "video"}):
            return
        # Use vLLM's actual processor policy, including supported maxima and
        # --limit-mm-per-prompt/language-model-only settings.
        for modality, configured_limit in allowed_limits.items():
            if (
                isinstance(configured_limit, bool)
                or not isinstance(configured_limit, int)
                or configured_limit < 0
            ):
                raise ValueError("Invalid engine input modality limit")
            limits[modality] = configured_limit
    # An explicit empty map means text-only. Omitted modalities are unsupported.
    runtime_config.set_engine_specific(
        INPUT_MODALITY_LIMITS_RUNTIME_KEY, json.dumps(limits, sort_keys=True)
    )
