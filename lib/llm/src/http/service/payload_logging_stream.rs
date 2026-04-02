// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stream wrappers that accumulate response chunks for payload logging.
//!
//! Each wrapper type observes items as they flow through the stream, accumulates
//! content/reasoning/tool-call deltas, and emits a single `openai.response` log record
//! when the stream is exhausted or dropped (client disconnect).
//!
//! # Design
//!
//! Wrappers implement [`futures::Stream`] by delegating to the inner stream's `poll_next`:
//! - On `Poll::Ready(Some(item))`: observe the chunk, then pass it through unchanged.
//! - On `Poll::Ready(None)`: emit the log record once, then return `None`.
//! - On `Drop`: emit the log record once (handles client-disconnect cancellation).
//!
//! The `emitted: bool` flag prevents double-emission when the stream returns `None` and
//! is subsequently dropped.

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use futures::Stream;

use crate::protocols::openai::chat_completions::NvCreateChatCompletionStreamResponse;
use crate::protocols::openai::completions::NvCreateCompletionResponse;
use crate::types::Annotated;

use super::payload_logger::PayloadLogger;

// ─── Chat Completions ────────────────────────────────────────────────────────

/// Per-choice accumulated state from streaming chat completion chunks.
#[derive(Default)]
struct ChatChoiceAccum {
    content: String,
    reasoning_content: String,
    /// tool_call_index → (id, name, accumulated_arguments)
    tool_calls: HashMap<u32, (Option<String>, Option<String>, String)>,
    finish_reason: Option<String>,
}

/// Accumulated chat completion response state, built chunk-by-chunk.
struct ChatPayloadAccumulator {
    id: String,
    model: String,
    choices: Vec<ChatChoiceAccum>,
    /// prompt_tokens, completion_tokens, total_tokens
    usage: Option<(u32, u32, u32)>,
}

impl ChatPayloadAccumulator {
    fn new() -> Self {
        Self {
            id: String::new(),
            model: String::new(),
            choices: Vec::new(),
            usage: None,
        }
    }

    /// Observe one streaming chunk, accumulating its deltas.
    fn observe(&mut self, chunk: &NvCreateChatCompletionStreamResponse) {
        let inner = &chunk.inner;
        if self.id.is_empty() {
            self.id = inner.id.clone();
        }
        if self.model.is_empty() {
            self.model = inner.model.clone();
        }
        // Accumulate usage (final chunk carries it when stream_options.include_usage=true)
        if let Some(u) = &inner.usage {
            self.usage = Some((u.prompt_tokens, u.completion_tokens, u.total_tokens));
        }
        for choice in &inner.choices {
            let idx = choice.index as usize;
            if idx >= self.choices.len() {
                self.choices.resize_with(idx + 1, ChatChoiceAccum::default);
            }
            let accum = &mut self.choices[idx];

            // Accumulate text content
            if let Some(content) = &choice.delta.content {
                use dynamo_protocols::types::ChatCompletionMessageContent;
                match content {
                    ChatCompletionMessageContent::Text(text) => {
                        accum.content.push_str(text);
                    }
                    ChatCompletionMessageContent::Parts(_) => {
                        // Multimodal parts: skip for text-only logging
                    }
                }
            }

            // Accumulate reasoning content
            if let Some(reasoning) = &choice.delta.reasoning_content {
                accum.reasoning_content.push_str(reasoning);
            }

            // Accumulate tool calls by index
            if let Some(tool_chunks) = &choice.delta.tool_calls {
                for tc in tool_chunks {
                    let entry = accum.tool_calls.entry(tc.index).or_default();
                    if entry.0.is_none() {
                        entry.0.clone_from(&tc.id);
                    }
                    if let Some(func) = &tc.function {
                        if entry.1.is_none() {
                            entry.1.clone_from(&func.name);
                        }
                        if let Some(args) = &func.arguments {
                            entry.2.push_str(args);
                        }
                    }
                }
            }

            // Capture finish reason using serde for correct snake_case serialization
            // (e.g., FinishReason::ToolCalls → "tool_calls", not "ToolCalls")
            if let Some(fr) = &choice.finish_reason {
                accum.finish_reason = serde_json::to_value(fr)
                    .ok()
                    .and_then(|v| v.as_str().map(str::to_owned));
            }
        }
    }

    /// Serialize the accumulated state to a JSON string suitable for `openai.response`.
    fn into_payload_json(self, streaming: bool) -> String {
        let choices: Vec<serde_json::Value> = self
            .choices
            .into_iter()
            .enumerate()
            .map(|(idx, c)| {
                let mut tool_calls: Vec<serde_json::Value> = c
                    .tool_calls
                    .into_iter()
                    .map(|(tc_idx, (id, name, args))| {
                        serde_json::json!({
                            "index": tc_idx,
                            "id": id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": args
                            }
                        })
                    })
                    .collect();
                // Sort by index for deterministic output
                tool_calls.sort_by_key(|v| v["index"].as_u64().unwrap_or(0));

                let mut message = serde_json::json!({
                    "role": "assistant",
                    "content": c.content,
                });
                if !c.reasoning_content.is_empty() {
                    message["reasoning_content"] = serde_json::Value::String(c.reasoning_content);
                }
                if !tool_calls.is_empty() {
                    message["tool_calls"] = serde_json::Value::Array(tool_calls);
                }
                serde_json::json!({
                    "index": idx,
                    "message": message,
                    "finish_reason": c.finish_reason,
                })
            })
            .collect();

        let mut payload = serde_json::json!({
            "id": self.id,
            "model": self.model,
            "choices": choices,
            "stream": streaming,
        });
        if let Some((pt, ct, tt)) = self.usage {
            payload["usage"] = serde_json::json!({
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "total_tokens": tt,
            });
        }
        serde_json::to_string(&payload).unwrap_or_else(|_| "{}".to_string())
    }
}

/// Stream wrapper for `/v1/chat/completions` streaming responses.
///
/// Wraps `Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>>`, accumulates
/// deltas, and emits one `openai.response` log when the stream completes or is dropped.
pub struct ChatPayloadLoggingStream<S> {
    inner: Pin<Box<S>>,
    accum: ChatPayloadAccumulator,
    logger: Arc<PayloadLogger>,
    rid: String,
    endpoint: &'static str,
    emitted: bool,
}

impl<S> ChatPayloadLoggingStream<S>
where
    S: Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>>,
{
    /// Wrap `stream` with payload logging using the given `rid` and `endpoint`.
    pub fn new(
        stream: S,
        logger: Arc<PayloadLogger>,
        rid: String,
        endpoint: &'static str,
    ) -> Self {
        Self {
            inner: Box::pin(stream),
            accum: ChatPayloadAccumulator::new(),
            logger,
            rid,
            endpoint,
            emitted: false,
        }
    }

    fn emit_log_once(&mut self, streaming: bool) {
        if self.emitted {
            return;
        }
        self.emitted = true;
        // Take the accumulator by replacing with a new empty one
        let accum = std::mem::replace(&mut self.accum, ChatPayloadAccumulator::new());
        let payload = accum.into_payload_json(streaming);
        self.logger.log_response(&self.rid, self.endpoint, &payload);
    }
}

impl<S> Stream for ChatPayloadLoggingStream<S>
where
    S: Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>>,
{
    type Item = Annotated<NvCreateChatCompletionStreamResponse>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(item)) => {
                // Skip accumulation entirely when logging is disabled — single branch-predicted
                // bool check, zero string allocation on the hot path.
                if self.logger.is_enabled() {
                    if let Some(data) = &item.data {
                        self.accum.observe(data);
                    }
                }
                Poll::Ready(Some(item))
            }
            Poll::Ready(None) => {
                self.emit_log_once(true);
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

impl<S> Drop for ChatPayloadLoggingStream<S> {
    fn drop(&mut self) {
        self.emit_log_once(true);
    }
}

// ─── Completions ─────────────────────────────────────────────────────────────

/// Accumulated completions response state, built chunk-by-chunk.
struct CompletionPayloadAccumulator {
    id: String,
    model: String,
    /// Per-choice text (completions API supports batch prompts)
    texts: Vec<String>,
    finish_reasons: Vec<Option<String>>,
    usage: Option<(u32, u32, u32)>,
}

impl CompletionPayloadAccumulator {
    fn new() -> Self {
        Self {
            id: String::new(),
            model: String::new(),
            texts: Vec::new(),
            finish_reasons: Vec::new(),
            usage: None,
        }
    }

    fn observe(&mut self, chunk: &NvCreateCompletionResponse) {
        let inner = &chunk.inner;
        if self.id.is_empty() {
            self.id = inner.id.clone();
        }
        if self.model.is_empty() {
            self.model = inner.model.clone();
        }
        if let Some(u) = &inner.usage {
            self.usage = Some((u.prompt_tokens, u.completion_tokens, u.total_tokens));
        }
        for choice in &inner.choices {
            let idx = choice.index as usize;
            if idx >= self.texts.len() {
                self.texts.resize(idx + 1, String::new());
                self.finish_reasons.resize(idx + 1, None);
            }
            self.texts[idx].push_str(&choice.text);
            if let Some(fr) = &choice.finish_reason {
                self.finish_reasons[idx] = serde_json::to_value(fr)
                    .ok()
                    .and_then(|v| v.as_str().map(str::to_owned));
            }
        }
    }

    fn into_payload_json(self, streaming: bool) -> String {
        let choices: Vec<serde_json::Value> = self
            .texts
            .into_iter()
            .zip(self.finish_reasons)
            .enumerate()
            .map(|(idx, (text, fr))| {
                serde_json::json!({
                    "index": idx,
                    "text": text,
                    "finish_reason": fr,
                })
            })
            .collect();
        let mut payload = serde_json::json!({
            "id": self.id,
            "model": self.model,
            "choices": choices,
            "stream": streaming,
        });
        if let Some((pt, ct, tt)) = self.usage {
            payload["usage"] = serde_json::json!({
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "total_tokens": tt,
            });
        }
        serde_json::to_string(&payload).unwrap_or_else(|_| "{}".to_string())
    }
}

/// Stream wrapper for `/v1/completions` streaming responses.
pub struct CompletionPayloadLoggingStream<S> {
    inner: Pin<Box<S>>,
    accum: CompletionPayloadAccumulator,
    logger: Arc<PayloadLogger>,
    rid: String,
    emitted: bool,
}

impl<S> CompletionPayloadLoggingStream<S>
where
    S: Stream<Item = Annotated<NvCreateCompletionResponse>>,
{
    pub fn new(stream: S, logger: Arc<PayloadLogger>, rid: String) -> Self {
        Self {
            inner: Box::pin(stream),
            accum: CompletionPayloadAccumulator::new(),
            logger,
            rid,
            emitted: false,
        }
    }

    fn emit_log_once(&mut self, streaming: bool) {
        if self.emitted {
            return;
        }
        self.emitted = true;
        let accum = std::mem::replace(&mut self.accum, CompletionPayloadAccumulator::new());
        let payload = accum.into_payload_json(streaming);
        self.logger
            .log_response(&self.rid, "OpenAIServingCompletions", &payload);
    }
}

impl<S> Stream for CompletionPayloadLoggingStream<S>
where
    S: Stream<Item = Annotated<NvCreateCompletionResponse>>,
{
    type Item = Annotated<NvCreateCompletionResponse>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(item)) => {
                if self.logger.is_enabled() {
                    if let Some(data) = &item.data {
                        self.accum.observe(data);
                    }
                }
                Poll::Ready(Some(item))
            }
            Poll::Ready(None) => {
                self.emit_log_once(true);
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

impl<S> Drop for CompletionPayloadLoggingStream<S> {
    fn drop(&mut self) {
        self.emit_log_once(true);
    }
}
