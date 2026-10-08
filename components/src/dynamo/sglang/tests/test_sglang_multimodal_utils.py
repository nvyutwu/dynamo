# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from dynamo.llm.exceptions import InvalidArgument
from dynamo.sglang.request_handlers.llm.mm_disagg_utils import (
    build_disagg_mm_kwargs,
    extract_media_urls,
    parse_mm_limits,
    raise_if_mm_limit_exceeded,
    raise_if_unextracted_multimodal,
)
from dynamo.sglang.request_handlers.multimodal.worker_handler import StreamProcessor

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.multimodal,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
]


def test_extract_media_urls_supports_string_and_wire_items():
    mm_data = {
        "video_url": [
            "file:///tmp/test.mp4",
            {"Url": "https://example.com/test.mp4"},
        ]
    }

    assert extract_media_urls(mm_data, "video_url") == [
        "file:///tmp/test.mp4",
        "https://example.com/test.mp4",
    ]


def test_build_disagg_mm_kwargs_includes_audio_urls():
    request = {
        "multi_modal_data": {
            "image_url": [{"Url": "https://example.com/image.png"}],
            "audio_url": [{"Url": "https://example.com/audio.wav"}],
            "video_url": [{"Url": "https://example.com/video.mp4"}],
        }
    }

    assert build_disagg_mm_kwargs(request) == {
        "image_data": ["https://example.com/image.png"],
        "audio_data": ["https://example.com/audio.wav"],
        "video_data": ["https://example.com/video.mp4"],
    }


def test_extract_media_urls_returns_none_for_missing_modality():
    assert extract_media_urls({}, "image_url") is None
    assert extract_media_urls(None, "image_url") is None
    assert extract_media_urls({"image_url": []}, "image_url") is None


def test_extract_media_urls_rejects_malformed_payloads():
    with pytest.raises(ValueError, match="must be a list"):
        extract_media_urls({"image_url": "https://example.com/a.png"}, "image_url")

    with pytest.raises(ValueError, match="Frontend-decoded"):
        extract_media_urls({"image_url": [{"Decoded": "..."}]}, "image_url")

    with pytest.raises(ValueError, match="Unsupported"):
        extract_media_urls({"image_url": [{"ignored": "value"}]}, "image_url")

    with pytest.raises(ValueError, match="Unsupported"):
        extract_media_urls(
            {"image_url": [{"Url": "https://example.com/a.png", "Decoded": {}}]},
            "image_url",
        )

    with pytest.raises(ValueError, match="must be a list"):
        extract_media_urls({"image_url": ""}, "image_url")


class TestMultimodalGuard:
    """Tests for multimodal guard when frontend extraction is missing."""

    @staticmethod
    def _image_message():
        return {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/a.jpg"},
                },
                {"type": "text", "text": "describe image"},
            ],
        }

    @pytest.mark.parametrize(
        "request_factory",
        [
            lambda msg: {"token_ids": [1, 2, 3], "messages": [msg]},
            lambda msg: {"token_ids": [1, 2, 3], "extra_args": {"messages": [msg]}},
        ],
        ids=["top_level_messages", "extra_args_messages"],
    )
    def test_raises_for_image_url(self, request_factory):
        with pytest.raises(RuntimeError, match="multi_modal_data"):
            raise_if_unextracted_multimodal(request_factory(self._image_message()))

    def test_raises_for_audio_url(self):
        request = {
            "token_ids": [1, 2, 3],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio_url",
                            "audio_url": {"url": "https://example.com/audio.wav"},
                        }
                    ],
                }
            ],
        }

        with pytest.raises(RuntimeError, match="audio_url"):
            raise_if_unextracted_multimodal(request)

    def test_text_only_request_bypasses_guard(self):
        raise_if_unextracted_multimodal({"token_ids": [10, 20, 30]})


async def _stream(items):
    for item in items:
        yield item


@pytest.mark.asyncio
async def test_multimodal_stream_keeps_reading_after_one_choice_finishes():
    chunks = [
        chunk
        async for chunk in StreamProcessor.process_sglang_stream(
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
    ]

    outputs = [json.loads(chunk) for chunk in chunks]

    assert [output["index"] for output in outputs] == [0, 1, 0, 1]
    assert [output["finished"] for output in outputs] == [False, False, True, True]
    assert [output.get("finish_reason") for output in outputs] == [
        None,
        None,
        "stop",
        "stop",
    ]


def test_parse_mm_limits_accepts_vllm_form():
    assert parse_mm_limits(None) == {}
    assert parse_mm_limits("") == {}
    assert parse_mm_limits('{"image": 10}') == {"image": 10}
    assert parse_mm_limits('{"image": 0, "video": 1, "audio": 2}') == {
        "image": 0,
        "video": 1,
        "audio": 2,
    }


@pytest.mark.parametrize(
    "value",
    [
        "10",
        "[10]",
        '{"picture": 1}',
        '{"image": -1}',
        '{"image": 1.5}',
        '{"image": true}',
        "{",
    ],
)
def test_parse_mm_limits_rejects_malformed_values(value):
    with pytest.raises(ValueError, match="--limit-mm-per-prompt"):
        parse_mm_limits(value)


def _image_request(count):
    return {
        "token_ids": [1, 2, 3],
        "multi_modal_data": {
            "image_url": [{"Url": f"https://example.com/{i}.png"} for i in range(count)]
        },
    }


def test_mm_limit_allows_up_to_the_limit():
    raise_if_mm_limit_exceeded(_image_request(10), {"image": 10})
    raise_if_mm_limit_exceeded(_image_request(11), {})
    raise_if_mm_limit_exceeded({"token_ids": [1]}, {"image": 0})
    # Unlisted modalities are unlimited.
    raise_if_mm_limit_exceeded(_image_request(11), {"video": 0})


def test_mm_limit_rejects_one_over_with_vllm_message():
    with pytest.raises(InvalidArgument) as exc:
        raise_if_mm_limit_exceeded(_image_request(11), {"image": 10})
    assert str(exc.value) == "At most 10 image(s) may be provided in one prompt."

    with pytest.raises(InvalidArgument, match="At most 0 image"):
        raise_if_mm_limit_exceeded(_image_request(1), {"image": 0})
