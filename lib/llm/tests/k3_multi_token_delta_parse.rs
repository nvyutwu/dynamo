// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Kimi K3 tool-call parsing must be independent of how the backend's token
//! stream is chunked.
//!
//! Regression test for a production failure where a special token split across
//! stream chunks left the K3 reasoning parser with a dangling closer, so the
//! response came back as `reasoning_content = "...<|close|>call<|sep|>"`,
//! `content = ""`, `tool_calls = null` while plain chat on the same endpoint was
//! clean. The test replays candidate raw K3 outputs through the real frontend
//! pipeline (reasoning parser -> tool jail) at every chunk width from 1 to 32
//! bytes and asserts the reasoning text survives intact and the tool section is
//! parsed into exactly one tool call.

use dynamo_llm::preprocessor::OpenAIPreprocessor;
use dynamo_llm::protocols::openai::chat_completions::NvCreateChatCompletionStreamResponse;
use dynamo_parsers::tool_calling::ToolDefinition;
use dynamo_protocols::types::{
    ChatChoiceStream, ChatCompletionMessageContent, ChatCompletionStreamResponseDelta, Role,
};
use dynamo_runtime::protocols::annotated::Annotated;
use futures::{StreamExt, stream};

fn get_text(content: &ChatCompletionMessageContent) -> &str {
    match content {
        ChatCompletionMessageContent::Text(text) => text.as_str(),
        ChatCompletionMessageContent::Parts(_) => "",
    }
}

fn chunk(content: &str) -> Annotated<NvCreateChatCompletionStreamResponse> {
    #[allow(deprecated)]
    let choice = ChatChoiceStream {
        index: 0,
        delta: ChatCompletionStreamResponseDelta {
            role: Some(Role::Assistant),
            content: Some(ChatCompletionMessageContent::Text(content.to_string())),
            tool_calls: None,
            function_call: None,
            refusal: None,
            reasoning_content: None,
        },
        finish_reason: None,
        logprobs: None,
    };

    let response = NvCreateChatCompletionStreamResponse {
        inner: dynamo_protocols::types::CreateChatCompletionStreamResponse {
            id: "test-id".to_string(),
            choices: vec![choice],
            created: 1234567890,
            model: "kimi-k3".to_string(),
            system_fingerprint: None,
            object: "chat.completion.chunk".to_string(),
            usage: None,
            service_tier: None,
        },
        nvext: None,
        llm_metrics: None,
    };

    Annotated {
        id: Some("test-id".to_string()),
        data: Some(response),
        event: None,
        comment: None,
        error: None,
    }
}

fn weather_tool() -> Vec<ToolDefinition> {
    vec![ToolDefinition {
        name: "get_weather".to_string(),
        parameters: Some(serde_json::json!({
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        })),
        strict: None,
    }]
}

/// Split `raw` into `n`-character pieces to emulate a multi-token DSpark delta.
fn split_every(raw: &str, n: usize) -> Vec<String> {
    let chars: Vec<char> = raw.chars().collect();
    chars
        .chunks(n)
        .map(|c| c.iter().collect::<String>())
        .collect()
}

async fn run(raw_chunks: &[String], prompt_injected_reasoning: bool) -> (String, String, usize) {
    let input = stream::iter(raw_chunks.iter().map(|c| chunk(c)).collect::<Vec<_>>());
    let reasoned = OpenAIPreprocessor::parse_reasoning_content_from_stream(
        input,
        "kimi_k3".to_string(),
        prompt_injected_reasoning,
    );
    let jailed = OpenAIPreprocessor::apply_tool_calling_jail(
        Some("kimi_k3".to_string()),
        None,
        Some(weather_tool()),
        false,
        false,
        reasoned,
    );

    let mut jailed = std::pin::pin!(jailed);
    let (mut reasoning, mut content, mut tool_calls) = (String::new(), String::new(), 0usize);
    while let Some(out) = jailed.next().await {
        if let Some(data) = out.data {
            for ch in &data.inner.choices {
                if let Some(r) = &ch.delta.reasoning_content {
                    reasoning.push_str(r);
                }
                if let Some(c) = &ch.delta.content {
                    content.push_str(get_text(c));
                }
                if let Some(tc) = &ch.delta.tool_calls {
                    tool_calls += tc.len();
                }
            }
        }
    }
    (reasoning, content, tool_calls)
}

#[tokio::test]
async fn k3_multi_token_deltas_preserve_tool_section() {
    let think =
        "The user is asking for current weather in Paris. Need use get_weather tool. Call it.";
    let call = "<|open|>tools<|sep|><|open|>call tool=\"get_weather\" index=\"1\"<|sep|><|open|>argument key=\"city\" type=\"string\"<|sep|>Paris<|close|>argument<|sep|><|close|>call<|sep|><|close|>tools<|sep|><|close|>message<|sep|><|end_of_msg|>";

    let cands: Vec<(&str, String)> = vec![
        // A: model never closed think, never opened the call — only an orphan closer.
        ("orphan-close-only", format!("{think}<|close|>call<|sep|>")),
        // B: canonical, well-formed.
        ("well-formed", format!("{think}<|close|>think<|sep|>{call}")),
        // C: think close missing, tools block otherwise intact.
        ("no-think-close", format!("{think}{call}")),
        // D: canonical plain-chat response (the probe that is clean on staging).
        (
            "plain-response",
            format!(
                "{think}<|close|>think<|sep|><|open|>response<|sep|>Tokyo<|close|>response<|sep|><|close|>message<|sep|><|end_of_msg|>"
            ),
        ),
    ];

    let mut bad = Vec::new();
    for (name, raw) in &cands {
        for granularity in 1..=32usize {
            let pieces = split_every(raw, granularity);
            let (r, c, tc) = run(&pieces, true).await;
            let expect_tool = *name != "orphan-close-only" && *name != "plain-response";
            let ok = r == think && (!expect_tool || tc == 1);
            if !ok {
                bad.push(format!(
                    "[{name:18}] width={granularity:<3} R={r:?} C={c:?} tool_calls={tc}"
                ));
            }
        }
    }
    for line in &bad {
        println!("FAIL {line}");
    }
    assert!(
        bad.is_empty(),
        "{} chunk widths mis-parsed (see FAIL lines above)",
        bad.len()
    );
}
