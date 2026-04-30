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
    def __init__(self, reasoning_parser, model_path=None):
        self.reasoning_parser = reasoning_parser
        self.model_path = model_path
        self.tokenizer_path = None


class _FakeDynamoArgs:
    def __init__(self, dyn_reasoning_parser=None):
        self.dyn_reasoning_parser = dyn_reasoning_parser


class _FakeConfig:
    def __init__(self, sglang_parser=None, dynamo_parser=None, model_path=None):
        self.server_args = _FakeServerArgs(sglang_parser, model_path)
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


class _FakeEngineNoTokenizer:
    """Mirrors SGLang's skip_tokenizer_init=True: tokenizer_manager exists but
    tokenizer_manager.tokenizer is None. This is the production deployment
    shape for Dynamo+SGLang and was the silent-failure case the v3 fix
    addresses.
    """

    class _TM:
        tokenizer = None

    tokenizer_manager = _TM()


class _StubHandler:
    """Just enough surface to call BaseWorkerHandler._resolve_require_reasoning.

    Bypasses BaseWorkerHandler.__init__, so we set ``_reasoning_tokenizer``
    directly the way the real init would have (engine tokenizer if available,
    else fallback).
    """

    def __init__(
        self,
        sglang_parser=None,
        dynamo_parser=None,
        engine=None,
        reasoning_tokenizer=_FakeTokenizer(),
        model_path=None,
    ):
        self.config = _FakeConfig(sglang_parser, dynamo_parser, model_path)
        self.engine = engine if engine is not None else _FakeEngine()
        self._reasoning_tokenizer = reasoning_tokenizer

    _resolve_require_reasoning = (
        BaseWorkerHandler._resolve_require_reasoning
    )
    _has_reasoning_parser = BaseWorkerHandler._has_reasoning_parser
    _acquire_reasoning_tokenizer = BaseWorkerHandler._acquire_reasoning_tokenizer


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


def test_install_require_reasoning_proxy_preserves_function_metadata():
    """functools.wraps lets debuggers/tracers see through the wrapper."""
    from dynamo.sglang.request_handlers.handler_base import (
        _install_require_reasoning_proxy,
    )

    tm = _RecorderTokenizerManager()
    original = tm.generate_request
    engine = _EngineWithTM(tm)
    _install_require_reasoning_proxy(engine)

    wrapped = tm.generate_request
    # __wrapped__ is the canonical breadcrumb left by functools.wraps.
    assert getattr(wrapped, "__wrapped__", None) is original
    assert wrapped.__name__ == original.__name__


# ─── tokenizer fallback when engine.tokenizer_manager.tokenizer is None ─────


def test_resolve_require_reasoning_returns_false_when_tokenizer_missing():
    """When skip_tokenizer_init=True and no fallback tokenizer was loaded,
    token-input requests must still terminate cleanly with False (not raise).
    Pre-fix this branch swallowed an AttributeError on None.decode and
    silently disabled the entire fix.
    """
    h = _StubHandler(
        sglang_parser="glm45",
        engine=_FakeEngineNoTokenizer(),
        reasoning_tokenizer=None,
    )
    # Even though the prompt ends in <think> (token id 99), with no tokenizer
    # we cannot decode the suffix and must return False.
    assert h._resolve_require_reasoning({"input_ids": [1, 2, 3, 99]}) is False
    # The text-input branch does not need a tokenizer and should still work.
    assert h._resolve_require_reasoning({"prompt": "hello\n<think>"}) is True


def test_acquire_reasoning_tokenizer_returns_engine_tokenizer():
    """Happy path: engine has a tokenizer, no fallback load needed."""
    h = _StubHandler(sglang_parser="glm45")  # engine = _FakeEngine() with tokenizer
    assert h._acquire_reasoning_tokenizer() is _FakeEngine.tokenizer_manager.tokenizer


def test_acquire_reasoning_tokenizer_returns_none_without_parser():
    """If no reasoning parser is configured, don't waste time loading
    a tokenizer — every request will return False anyway."""
    h = _StubHandler(sglang_parser=None, dynamo_parser=None)
    assert h._acquire_reasoning_tokenizer() is None


def test_acquire_reasoning_tokenizer_warns_when_no_path(caplog):
    """skip_tokenizer_init=True path with no model_path on server_args:
    must return None and emit a warning so operators can diagnose."""
    import logging as _logging

    h = _StubHandler(
        sglang_parser="glm45",
        engine=_FakeEngineNoTokenizer(),
        model_path=None,
    )
    with caplog.at_level(_logging.WARNING):
        result = h._acquire_reasoning_tokenizer()
    assert result is None
    assert any(
        "no tokenizer is available" in rec.message
        or "no model_path" in rec.message
        or "Reasoning parser is configured" in rec.message
        for rec in caplog.records
    )


# ─── concurrent CV propagation ───────────────────────────────────────────────


def test_cv_propagates_across_concurrent_tasks():
    """Two concurrent asyncio tasks must each see their own CV value when
    the wrapper executes, even when they share a single tokenizer_manager.
    Regression guard: if the wrapper ever moved CV-read out of the synchronous
    call into a thread pool or to a non-task-bound coroutine, this would fail.
    """
    import asyncio

    from dynamo.sglang.request_handlers.handler_base import (
        _DYN_REQUIRE_REASONING_CV,
        _install_require_reasoning_proxy,
    )

    tm = _RecorderTokenizerManager()
    seen_flags: list[tuple[str, bool]] = []

    original = tm.generate_request

    def _record_call(obj, request):
        result = original(obj, request)
        seen_flags.append((request, obj.require_reasoning))
        return result

    tm.generate_request = _record_call
    engine = _EngineWithTM(tm)
    _install_require_reasoning_proxy(engine)

    async def one_request(label: str, expect: bool):
        token = _DYN_REQUIRE_REASONING_CV.set(expect)
        try:
            await asyncio.sleep(0)  # yield so tasks interleave
            obj = _GenerateReqInputStub()
            tm.generate_request(obj, label)
            await asyncio.sleep(0)
            assert obj.require_reasoning is expect
        finally:
            _DYN_REQUIRE_REASONING_CV.reset(token)

    async def main():
        await asyncio.gather(
            one_request("task-true", True),
            one_request("task-false", False),
        )

    asyncio.run(main())

    # Each task saw its own CV value — no cross-contamination.
    by_label = dict(seen_flags)
    assert by_label["task-true"] is True
    assert by_label["task-false"] is False


def test_cv_resets_on_cancellation_between_set_and_await():
    """Mirrors decode_handler's try/finally invariant: even if the awaited
    engine.async_generate is cancelled, the contextvar must be reset on the
    way out so the next request on the same task sees the default.
    """
    import asyncio

    from dynamo.sglang.request_handlers.handler_base import (
        _DYN_REQUIRE_REASONING_CV,
    )

    async def main():
        token = _DYN_REQUIRE_REASONING_CV.set(True)
        try:
            # Simulate engine.async_generate being awaited then cancelled.
            task = asyncio.create_task(asyncio.sleep(10))
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        finally:
            _DYN_REQUIRE_REASONING_CV.reset(token)
        # After reset, default applies.
        assert _DYN_REQUIRE_REASONING_CV.get() is False

    asyncio.run(main())


# ─── end-to-end: fake engine.async_generate routes through the wrapper ──────


def test_end_to_end_engine_async_generate_routes_through_wrapper():
    """Smokes the full chain that decode_handler / prefill_handler relies on:

    handler sets CV → handler awaits engine.async_generate → engine internally
    calls tokenizer_manager.generate_request (wrapped) with a GenerateReqInput
    constructed during the call → wrapper reads CV → wrapper mutates obj →
    handler's finally resets CV → engine returns the async generator.

    The mutation must persist on `obj` after the CV is reset.
    """
    import asyncio

    from dynamo.sglang.request_handlers.handler_base import (
        _DYN_REQUIRE_REASONING_CV,
        _install_require_reasoning_proxy,
    )

    tm = _RecorderTokenizerManager()

    class _FakeAsyncGenerator:
        async def __aiter__(self):
            yield {"meta_info": {"id": "x", "finish_reason": None}, "output_ids": [1]}

    class _FakeEngineRoutes:
        def __init__(self, tm):
            self.tokenizer_manager = tm

        async def async_generate(self, **kwargs):
            # Mirrors SGLang's Engine.async_generate: build a GenerateReqInput,
            # call tokenizer_manager.generate_request synchronously inside the
            # await (before returning the async generator).
            obj = _GenerateReqInputStub()
            self.tokenizer_manager.generate_request(obj, None)
            # Snapshot so the test can assert on it post-await.
            self.last_obj = obj
            return _FakeAsyncGenerator()

    engine = _FakeEngineRoutes(tm)
    _install_require_reasoning_proxy(engine)

    async def run_request(expect: bool):
        token = _DYN_REQUIRE_REASONING_CV.set(expect)
        try:
            stream = await engine.async_generate()
        finally:
            _DYN_REQUIRE_REASONING_CV.reset(token)
        # CV is back to default by now — but mutation on obj persists.
        assert engine.last_obj.require_reasoning is expect
        # The returned async generator is still usable.
        async for _ in stream:
            pass

    asyncio.run(run_request(True))
    asyncio.run(run_request(False))


# ─── prefill handler call site is wrapped ────────────────────────────────────


def test_prefill_handler_imports_cv_and_wraps_async_generate():
    """Source-level guard that the prefill handler also drives the CV.
    Without this wrap, disagg requests enter SGLang prefill with
    require_reasoning=False even when the prompt is mid-thinking, which
    can defeat the gate when grammar state is shared with decode.
    """
    import inspect

    from dynamo.sglang.request_handlers.llm import prefill_handler

    src = inspect.getsource(prefill_handler)
    assert "_DYN_REQUIRE_REASONING_CV" in src, (
        "prefill_handler must import _DYN_REQUIRE_REASONING_CV"
    )
    assert "_DYN_REQUIRE_REASONING_CV.set(" in src and (
        "_DYN_REQUIRE_REASONING_CV.reset(" in src
    ), "prefill handler must wrap engine.async_generate with CV try/finally"
    assert "self._resolve_require_reasoning(" in src, (
        "prefill handler must derive the CV value via _resolve_require_reasoning"
    )
