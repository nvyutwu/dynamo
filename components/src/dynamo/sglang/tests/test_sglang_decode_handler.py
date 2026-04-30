# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from dynamo.sglang.request_handlers.handler_base import BaseWorkerHandler
from dynamo.sglang.request_handlers.llm.decode_handler import (
    DecodeWorkerHandler,
    _extract_media_urls,
)
from dynamo.sglang.request_handlers.multimodal.worker_handler import StreamProcessor

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
]


def test_extract_media_urls_supports_string_and_wire_items():
    mm_data = {
        "video_url": [
            "file:///tmp/test.mp4",
            {"Url": "https://example.com/test.mp4"},
            {"ignored": "value"},
        ]
    }

    assert _extract_media_urls(mm_data, "video_url") == [
        "file:///tmp/test.mp4",
        "https://example.com/test.mp4",
    ]


def test_extract_media_urls_returns_none_for_missing_or_invalid_items():
    assert _extract_media_urls({}, "image_url") is None
    assert (
        _extract_media_urls({"image_url": [{"ignored": "value"}]}, "image_url") is None
    )


def _new_decode_handler(*, use_sglang_tokenizer: bool = False):
    handler = DecodeWorkerHandler.__new__(DecodeWorkerHandler)
    handler.use_sglang_tokenizer = use_sglang_tokenizer
    handler.config = SimpleNamespace(
        server_args=SimpleNamespace(served_model_name="test-model")
    )

    @asynccontextmanager
    async def no_cancellation_monitor(*args, **kwargs):
        yield None

    handler._cancellation_monitor = no_cancellation_monitor
    return handler


async def _stream(items):
    for item in items:
        yield item


class _Context:
    def is_stopped(self):
        return False


def test_build_sampling_params_passes_n_for_token_requests():
    handler = _new_decode_handler(use_sglang_tokenizer=False)

    sampling_params = handler._build_sampling_params(
        {
            "sampling_options": {"temperature": 0.2, "top_p": 0.9, "n": 3},
            "stop_conditions": {"max_tokens": 8},
        }
    )

    assert sampling_params["n"] == 3
    assert sampling_params["temperature"] == 0.2
    assert sampling_params["max_new_tokens"] == 8


def test_build_sampling_params_passes_n_for_sglang_tokenizer_requests():
    handler = _new_decode_handler(use_sglang_tokenizer=True)

    sampling_params = handler._build_sampling_params(
        {"temperature": 0.2, "top_p": 0.9, "n": 2, "max_tokens": 8}
    )

    assert sampling_params["n"] == 2
    assert sampling_params["temperature"] == 0.2
    assert sampling_params["max_new_tokens"] == 8


@pytest.mark.asyncio
async def test_process_token_stream_tracks_logprobs_per_choice_index():
    handler = _new_decode_handler()

    chunks = await _collect(
        handler._process_token_stream(
            _stream(
                [
                    {
                        "index": 0,
                        "output_ids": [101],
                        "meta_info": {
                            "id": "request-1",
                            "finish_reason": None,
                            "output_token_logprobs": [(-0.1, 101, "a")],
                        },
                    },
                    {
                        "index": 1,
                        "output_ids": [201],
                        "meta_info": {
                            "id": "request-1",
                            "finish_reason": None,
                            "output_token_logprobs": [(-0.2, 201, "b")],
                        },
                    },
                    {
                        "index": 0,
                        "output_ids": [102],
                        "meta_info": {
                            "id": "request-1",
                            "finish_reason": None,
                            "output_token_logprobs": [
                                (-0.1, 101, "a"),
                                (-0.3, 102, "c"),
                            ],
                        },
                    },
                ]
            ),
            _Context(),
        )
    )

    assert [chunk["index"] for chunk in chunks] == [0, 1, 0]
    assert [chunk["token_ids"] for chunk in chunks] == [[101], [201], [102]]
    assert [chunk["log_probs"] for chunk in chunks] == [[-0.1], [-0.2], [-0.3]]


@pytest.mark.asyncio
async def test_process_text_stream_tracks_delta_per_choice_index():
    handler = _new_decode_handler()

    chunks = await _collect(
        handler._process_text_stream(
            _stream(
                [
                    {
                        "index": 0,
                        "text": "He",
                        "meta_info": {"id": "request-1", "finish_reason": None},
                    },
                    {
                        "index": 1,
                        "text": "Go",
                        "meta_info": {"id": "request-1", "finish_reason": None},
                    },
                    {
                        "index": 0,
                        "text": "Hello",
                        "meta_info": {"id": "request-1", "finish_reason": None},
                    },
                    {
                        "index": 1,
                        "text": "Good",
                        "meta_info": {"id": "request-1", "finish_reason": None},
                    },
                ]
            ),
            _Context(),
        )
    )

    choices = [chunk["choices"][0] for chunk in chunks]
    assert [choice["index"] for choice in choices] == [0, 1, 0, 1]
    assert [choice["delta"]["content"] for choice in choices] == [
        "He",
        "Go",
        "llo",
        "od",
    ]


@pytest.mark.asyncio
async def test_multimodal_stream_keeps_reading_after_one_choice_finishes():
    chunks = await _collect(
        StreamProcessor.process_sglang_stream(
            _stream(
                [
                    {
                        "index": 0,
                        "output_ids": [101],
                        "text": "a",
                        "meta_info": {"finish_reason": None},
                    },
                    {
                        "index": 1,
                        "output_ids": [201],
                        "text": "b",
                        "meta_info": {"finish_reason": None},
                    },
                    {
                        "index": 0,
                        "output_ids": [],
                        "text": "a",
                        "meta_info": {"finish_reason": {"type": "stop"}},
                    },
                    {
                        "index": 1,
                        "output_ids": [],
                        "text": "b",
                        "meta_info": {"finish_reason": {"type": "stop"}},
                    },
                ]
            )
        )
    )

    outputs = [json.loads(chunk) for chunk in chunks]

    assert [output["index"] for output in outputs] == [0, 1, 0, 1]
    assert [output["finished"] for output in outputs] == [False, False, True, True]
    assert [output.get("finish_reason") for output in outputs] == [
        None,
        None,
        "stop",
        "stop",
    ]


async def _collect(stream):
    return [item async for item in stream]


def test_guided_decoding_json_maps_to_sglang_json_schema():
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }

    result = BaseWorkerHandler._get_guided_decoding_params({"json": schema})

    assert set(result) == {"json_schema"}
    assert json.loads(result["json_schema"]) == schema


def test_guided_decoding_structural_tag_maps_to_sglang_structural_tag():
    structural_tag = {
        "type": "structural_tag",
        "structures": [
            {
                "begin": "<tool_call>",
                "schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
                "end": "</tool_call>",
            }
        ],
        "triggers": ["<tool_call>"],
    }

    result = BaseWorkerHandler._get_guided_decoding_params(
        {"structural_tag": structural_tag}
    )

    assert set(result) == {"structural_tag"}
    assert json.loads(result["structural_tag"]) == structural_tag


def test_guided_decoding_regex_maps_to_sglang_regex():
    pattern = r"\d{4}-\d{2}-\d{2}"
    result = BaseWorkerHandler._get_guided_decoding_params({"regex": pattern})
    assert result == {"regex": pattern}


def test_guided_decoding_grammar_maps_to_sglang_ebnf():
    grammar = 'root ::= "yes" | "no"'
    result = BaseWorkerHandler._get_guided_decoding_params({"grammar": grammar})
    assert result == {"ebnf": grammar}


def test_guided_decoding_choice_maps_to_regex_alternation():
    result = BaseWorkerHandler._get_guided_decoding_params(
        {"choice": ["positive", "negative", "neutral"]}
    )
    assert result == {"regex": "(?:positive|negative|neutral)"}


def test_guided_decoding_choice_escapes_regex_metacharacters():
    result = BaseWorkerHandler._get_guided_decoding_params(
        {"choice": ["a.b", "c+d"]}
    )
    assert result == {"regex": r"(?:a\.b|c\+d)"}


def test_guided_decoding_priority_json_over_others():
    schema = {"type": "object"}
    result = BaseWorkerHandler._get_guided_decoding_params(
        {
            "json": schema,
            "regex": "ignored",
            "grammar": "root ::= ignored",
            "choice": ["ignored"],
        }
    )
    assert set(result) == {"json_schema"}


def test_guided_decoding_empty_choice_falls_through():
    assert BaseWorkerHandler._get_guided_decoding_params({"choice": []}) == {}


def test_guided_decoding_unsupported_keys_ignored():
    assert (
        BaseWorkerHandler._get_guided_decoding_params(
            {"backend": "xgrammar", "whitespace_pattern": " "}
        )
        == {}
    )


def test_validate_parser_flags_allows_both_reasoning_parsers():
    """`--reasoning-parser` and `--dyn-reasoning-parser` serve different
    purposes: SGLang side gates the grammar mask until after `</think>`,
    Dynamo side splits reasoning_content from content at the OpenAI
    frontend. Both must be allowed to coexist for thinking-on guided JSON.
    """
    from dynamo.sglang.args import _validate_parser_flags

    # Should NOT raise / sys.exit when both reasoning parsers are set.
    _validate_parser_flags("glm45", "glm45", "reasoning-parser")


def test_validate_parser_flags_rejects_both_tool_call_parsers():
    """tool-call parsers DO double-consume the output stream — keep the
    exclusivity guard for tool-call-parser only.
    """
    from dynamo.sglang.args import _validate_parser_flags

    with pytest.raises(SystemExit):
        _validate_parser_flags("glm47", "glm47", "tool-call-parser")


class _FakeServerArgs:
    def __init__(self, reasoning_parser):
        self.reasoning_parser = reasoning_parser


class _FakeDynamoArgs:
    def __init__(self, dyn_reasoning_parser=None):
        self.dyn_reasoning_parser = dyn_reasoning_parser


class _FakeConfig:
    def __init__(self, sglang_parser=None, dynamo_parser=None):
        self.server_args = _FakeServerArgs(sglang_parser)
        self.dynamo_args = _FakeDynamoArgs(dynamo_parser)


class _FakeTokenizer:
    """Decode the suffix tokens directly — `<think>` is a single id 99,
    `</think>` is 100 in this stub. Matches BPE behavior for these markers
    on most modern tokenizers."""

    _MAP = {99: "<think>", 100: "</think>"}

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self._MAP.get(i, "x") for i in ids)


class _FakeEngine:
    class _TM:
        tokenizer = _FakeTokenizer()

    tokenizer_manager = _TM()


class _StubHandler:
    """Just enough surface to call BaseWorkerHandler._resolve_require_reasoning."""

    def __init__(self, sglang_parser=None, dynamo_parser=None):
        self.config = _FakeConfig(sglang_parser, dynamo_parser)
        self.engine = _FakeEngine()

    _resolve_require_reasoning = (
        BaseWorkerHandler._resolve_require_reasoning
    )


def test_resolve_require_reasoning_no_parser_returns_false():
    h = _StubHandler(sglang_parser=None, dynamo_parser=None)
    assert h._resolve_require_reasoning({"input_ids": [1, 2, 99]}) is False


def test_resolve_require_reasoning_prompt_ends_with_think():
    h = _StubHandler(sglang_parser="glm45")
    assert h._resolve_require_reasoning({"prompt": "Hello\n<think>"}) is True


def test_resolve_require_reasoning_prompt_ends_with_close_think():
    h = _StubHandler(sglang_parser="glm45")
    assert h._resolve_require_reasoning({"prompt": "Hello\n</think>"}) is False


def test_resolve_require_reasoning_token_ids_thinking_on():
    h = _StubHandler(sglang_parser="glm45")
    # Last token is `<think>` (id 99 in the stub tokenizer).
    assert h._resolve_require_reasoning({"input_ids": [1, 2, 3, 99]}) is True


def test_resolve_require_reasoning_token_ids_thinking_off():
    h = _StubHandler(sglang_parser="glm45")
    # Last token is `</think>` (id 100).
    assert h._resolve_require_reasoning({"input_ids": [1, 2, 3, 100]}) is False


def test_resolve_require_reasoning_dynamo_parser_only():
    """Even if only --dyn-reasoning-parser is set (no SGLang side), the
    detector should still run — Dynamo may want to parse the response, and
    SGLang's grammar backend will fall back to non-reasoner mode."""
    h = _StubHandler(sglang_parser=None, dynamo_parser="glm45")
    assert h._resolve_require_reasoning({"input_ids": [1, 2, 99]}) is True


def test_resolve_require_reasoning_empty_input_returns_false():
    h = _StubHandler(sglang_parser="glm45")
    assert h._resolve_require_reasoning({"input_ids": []}) is False
    assert h._resolve_require_reasoning({}) is False


# ─── _install_require_reasoning_proxy ────────────────────────────────────────


class _RecorderTokenizerManager:
    """Records the GenerateReqInput each generate_request call receives.

    Mirrors the surface of SGLang's TokenizerManager that
    _install_require_reasoning_proxy touches.
    """

    def __init__(self):
        self.last_obj = None
        self.calls = 0

    def generate_request(self, obj, request):  # pylint: disable=unused-argument
        self.last_obj = obj
        self.calls += 1
        return obj  # not a real generator — fine for the unit test


class _GenerateReqInputStub:
    """Stand-in for SGLang's GenerateReqInput. Default require_reasoning=False."""

    def __init__(self):
        self.require_reasoning = False


class _EngineWithTM:
    def __init__(self, tm):
        self.tokenizer_manager = tm


def test_install_require_reasoning_proxy_sets_flag_when_cv_true():
    from dynamo.sglang.request_handlers.handler_base import (
        _DYN_REQUIRE_REASONING_CV,
        _install_require_reasoning_proxy,
    )

    tm = _RecorderTokenizerManager()
    engine = _EngineWithTM(tm)
    _install_require_reasoning_proxy(engine)

    obj = _GenerateReqInputStub()
    token = _DYN_REQUIRE_REASONING_CV.set(True)
    try:
        tm.generate_request(obj, None)
    finally:
        _DYN_REQUIRE_REASONING_CV.reset(token)

    assert obj.require_reasoning is True
    assert tm.last_obj is obj
    assert tm.calls == 1


def test_install_require_reasoning_proxy_leaves_flag_when_cv_false():
    from dynamo.sglang.request_handlers.handler_base import (
        _DYN_REQUIRE_REASONING_CV,
        _install_require_reasoning_proxy,
    )

    tm = _RecorderTokenizerManager()
    engine = _EngineWithTM(tm)
    _install_require_reasoning_proxy(engine)

    obj = _GenerateReqInputStub()
    # CV defaults to False — call without setting.
    assert _DYN_REQUIRE_REASONING_CV.get() is False
    tm.generate_request(obj, None)
    assert obj.require_reasoning is False


def test_install_require_reasoning_proxy_is_idempotent():
    from dynamo.sglang.request_handlers.handler_base import (
        _install_require_reasoning_proxy,
    )

    tm = _RecorderTokenizerManager()
    engine = _EngineWithTM(tm)
    _install_require_reasoning_proxy(engine)
    first_wrapper = tm.generate_request
    _install_require_reasoning_proxy(engine)
    second_wrapper = tm.generate_request
    assert first_wrapper is second_wrapper, "double-wrapping the tokenizer_manager"
    assert tm._dynamo_require_reasoning_wrapped is True


def test_install_require_reasoning_proxy_handles_none_engine():
    from dynamo.sglang.request_handlers.handler_base import (
        _install_require_reasoning_proxy,
    )

    # Should not raise — guards against engines that haven't initialized
    # tokenizer_manager yet.
    class _NoTM:
        tokenizer_manager = None

    _install_require_reasoning_proxy(_NoTM())
