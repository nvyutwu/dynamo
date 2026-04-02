// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prompt/response payload logging for the Dynamo FE HTTP service.
//!
//! When `DYN_LOG_PAYLOADS=1`, every API request and assembled response are emitted as
//! structured `tracing::info!` events to the `dynamo.payload` target. The existing
//! `OpenTelemetryTracingBridge` (configured in `dynamo_runtime::logging::init()`) forwards
//! these to the OTEL logs backend when `OTEL_EXPORT_ENABLED=1`.
//!
//! Two records are emitted per request:
//! - **`openai.request`** — emitted on request arrival, contains request headers + payload.
//! - **`openai.response`** — emitted after the last token is sent (or on client disconnect),
//!   contains the assembled response payload.
//!
//! Format matches vLLM's payload logging for Loki/Grafana compatibility.
//!
//! # Example Loki query
//! ```logql
//! {service_name="frontend"} | json | rid = "chatcmpl-abc123"
//! ```

use axum::http::HeaderMap;
use dynamo_runtime::config::env_is_truthy;
use std::sync::Arc;

/// Environment variable to enable payload logging. Set to `1`, `true`, `on`, or `yes`.
pub const DYN_LOG_PAYLOADS: &str = "DYN_LOG_PAYLOADS";

/// Payload logger. Emits `openai.request` and `openai.response` log records to the
/// `dynamo.payload` tracing target when enabled.
///
/// All methods are non-blocking: they enqueue an event in the tracing subscriber's
/// internal channel and return immediately. The `BatchLogRecordProcessor` in the
/// OTEL bridge handles async export to the configured OTLP endpoint.
pub struct PayloadLogger {
    enabled: bool,
}

impl PayloadLogger {
    /// Create a new `PayloadLogger`. Reads `DYN_LOG_PAYLOADS` from the environment.
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            enabled: env_is_truthy(DYN_LOG_PAYLOADS),
        })
    }

    /// Returns `true` if payload logging is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Emit an `openai.request` log record.
    ///
    /// - `rid`: request ID (e.g., `"chatcmpl-<uuid>"`)
    /// - `endpoint`: one of `"OpenAIServingChat"`, `"OpenAIServingCompletions"`,
    ///   `"OpenAIServingResponses"`
    /// - `payload`: JSON-serialized request body
    /// - `headers`: JSON-serialized request headers (sensitive headers must be pre-filtered)
    pub fn log_request(
        &self,
        rid: &str,
        endpoint: &'static str,
        payload: &str,
        headers: &str,
    ) {
        if !self.enabled {
            return;
        }
        tracing::info!(
            target: "dynamo.payload",
            rid,
            endpoint,
            payload,
            headers,
            "openai.request"
        );
    }

    /// Emit an `openai.response` log record.
    ///
    /// For streaming responses this is called once after the stream is exhausted (or on
    /// client disconnect), not per chunk.
    pub fn log_response(&self, rid: &str, endpoint: &'static str, payload: &str) {
        if !self.enabled {
            return;
        }
        tracing::info!(
            target: "dynamo.payload",
            rid,
            endpoint,
            payload,
            "openai.response"
        );
    }
}

impl Default for PayloadLogger {
    fn default() -> Self {
        Self {
            enabled: env_is_truthy(DYN_LOG_PAYLOADS),
        }
    }
}

/// Serialize a `HeaderMap` to a compact JSON string, filtering sensitive headers.
///
/// Excluded headers: `authorization`, `cookie`, `set-cookie`.
/// Non-UTF-8 header values are silently dropped.
pub fn headers_to_json(headers: &HeaderMap) -> String {
    let map: std::collections::BTreeMap<&str, &str> = headers
        .iter()
        .filter_map(|(name, value)| {
            let name = name.as_str();
            if matches!(name, "authorization" | "cookie" | "set-cookie") {
                return None;
            }
            value.to_str().ok().map(|v| (name, v))
        })
        .collect();
    serde_json::to_string(&map).unwrap_or_else(|_| "{}".to_string())
}
