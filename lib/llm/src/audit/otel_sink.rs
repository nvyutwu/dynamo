// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! OTLP exporter sink for the audit bus.
//!
//! Emits exactly one OTLP `LogRecord` per `AuditRecord`. The exporter is
//! constructed once at sink init (not per emit). Network I/O happens on the
//! SDK's internal batch processor; `emit()` is non-blocking enqueue.
//!
//! Transport follows `OTEL_EXPORTER_OTLP_LOGS_PROTOCOL` with
//! `OTEL_EXPORTER_OTLP_PROTOCOL` as fallback. Supported values are
//! `http/protobuf` (default) and `grpc`.

use std::sync::Arc;

use anyhow::{Context as _, Result};
use async_trait::async_trait;
use dynamo_runtime::config::environment_names::logging::otlp as env_otlp;
use opentelemetry::logs::{AnyValue, LogRecord, Logger, LoggerProvider, Severity};
use opentelemetry_otlp::{Protocol, WithExportConfig};
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::logs::{SdkLogger, SdkLoggerProvider};
use serde_json::json;

use super::config::AuditPolicy;
use super::handle::{AuditEventType, AuditRecord};
use super::sink::AuditSink;

/// Bounded dedicated rayon pool size for off-runtime audit-record
/// serialization. Two threads is enough at our measured publish rate
/// (~2 records per chat completion) while small enough to never become
/// a CPU hog when the host is saturated. Configurable via
/// `DYN_AUDIT_OTEL_SERDE_THREADS` if a future deployment needs to tune
/// this (e.g. heavier audit payloads, more sinks).
const DEFAULT_OTEL_SERDE_THREADS: usize = 2;
const ENV_OTEL_SERDE_THREADS: &str = "DYN_AUDIT_OTEL_SERDE_THREADS";

const DEFAULT_OTLP_HTTP_LOGS_ENDPOINT: &str = "http://localhost:4318/v1/logs";
const DEFAULT_OTLP_GRPC_ENDPOINT: &str = "http://localhost:4317";

/// Static body string for every emitted record. Backends should filter by
/// the `endpoint` attribute, not by body.
const AUDIT_LOG_BODY: &str = "dynamo.audit";

/// Logical endpoint label so phase 2 (completions / responses) can be
/// distinguished without changing the body.
const AUDIT_ENDPOINT_CHAT_COMPLETION: &str = "openai.chat_completion";

/// Instrumentation scope name on the emitted `LogRecord`.
const AUDIT_INSTRUMENTATION_SCOPE: &str = "dynamo.audit";

/// Default service name when `OTEL_SERVICE_NAME` is unset.
const DEFAULT_SERVICE_NAME: &str = "dynamo";

pub struct OtelSink {
    /// Held so the SDK's batch processor flushes when the sink is dropped on
    /// audit-bus shutdown. The field is never read directly — its job is to
    /// keep the provider alive for the sink's lifetime. TODO(phase D): wire
    /// an explicit `force_flush` hook on the worker cancellation path so
    /// records aren't lost if the runtime is torn down before Drop runs.
    #[allow(dead_code)]
    provider: SdkLoggerProvider,
    logger: SdkLogger,
    max_payload_bytes: usize,
    /// Bounded dedicated rayon pool for serializing `AuditRecord` to JSON
    /// off the tokio runtime. Without this, the heavy serde walk over the
    /// full `Arc<NvCreateChatCompletionRequest>` (typically ~30 KB JSON) runs
    /// on a tokio worker shared with HTTP request futures, and cooperative
    /// scheduling lets it starve the request future for the duration of the
    /// walk. Moving the walk to dedicated OS threads removes that contention.
    serde_pool: Arc<rayon::ThreadPool>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OtlpLogsProtocol {
    HttpProtobuf,
    Grpc,
}

impl OtlpLogsProtocol {
    fn from_env() -> Self {
        let raw = std::env::var(env_otlp::OTEL_EXPORTER_OTLP_LOGS_PROTOCOL)
            .or_else(|_| std::env::var(env_otlp::OTEL_EXPORTER_OTLP_PROTOCOL))
            .unwrap_or_else(|_| "http/protobuf".to_string());
        match raw.trim().to_ascii_lowercase().as_str() {
            "http/protobuf" | "http/proto" | "http" => Self::HttpProtobuf,
            "grpc" => Self::Grpc,
            other => {
                tracing::warn!(
                    protocol = other,
                    "audit otel: unsupported OTLP logs protocol; defaulting to http/protobuf"
                );
                Self::HttpProtobuf
            }
        }
    }

    fn default_endpoint(self) -> &'static str {
        match self {
            Self::HttpProtobuf => DEFAULT_OTLP_HTTP_LOGS_ENDPOINT,
            Self::Grpc => DEFAULT_OTLP_GRPC_ENDPOINT,
        }
    }
}

impl OtelSink {
    pub fn new(
        provider: SdkLoggerProvider,
        max_payload_bytes: usize,
        serde_pool: Arc<rayon::ThreadPool>,
    ) -> Self {
        let logger = provider.logger(AUDIT_INSTRUMENTATION_SCOPE);
        Self {
            provider,
            logger,
            max_payload_bytes,
            serde_pool,
        }
    }

    /// Build the bounded rayon pool for off-runtime serialization. The thread
    /// count defaults to `DEFAULT_OTEL_SERDE_THREADS` and is overridable via
    /// `DYN_AUDIT_OTEL_SERDE_THREADS`. Returns an error (rather than panicking)
    /// if `ThreadPoolBuilder::build` fails — e.g. OS-level thread spawn limits.
    fn build_serde_pool() -> Result<Arc<rayon::ThreadPool>> {
        let num_threads = std::env::var(ENV_OTEL_SERDE_THREADS)
            .ok()
            .and_then(|raw| raw.parse::<usize>().ok())
            .filter(|n| *n > 0)
            .unwrap_or(DEFAULT_OTEL_SERDE_THREADS);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .thread_name(|i| format!("otel-audit-serde-{i}"))
            .build()
            .with_context(|| {
                format!("building OTEL audit serde rayon pool ({num_threads} threads)")
            })?;
        Ok(Arc::new(pool))
    }

    pub async fn from_policy(policy: &AuditPolicy) -> Result<Self> {
        let protocol = OtlpLogsProtocol::from_env();
        let endpoint = std::env::var(env_otlp::OTEL_EXPORTER_OTLP_LOGS_ENDPOINT)
            .or_else(|_| std::env::var(env_otlp::OTEL_EXPORTER_OTLP_TRACES_ENDPOINT))
            .unwrap_or_else(|_| protocol.default_endpoint().to_string());

        let exporter = match protocol {
            OtlpLogsProtocol::HttpProtobuf => opentelemetry_otlp::LogExporter::builder()
                .with_http()
                .with_protocol(Protocol::HttpBinary)
                .with_endpoint(endpoint.clone())
                .build(),
            OtlpLogsProtocol::Grpc => opentelemetry_otlp::LogExporter::builder()
                .with_tonic()
                .with_endpoint(endpoint.clone())
                .build(),
        }
        .with_context(|| {
            format!(
                "building OTLP audit log exporter for endpoint {endpoint} using {protocol:?}"
            )
        })?;

        let service_name = std::env::var(env_otlp::OTEL_SERVICE_NAME)
            .unwrap_or_else(|_| DEFAULT_SERVICE_NAME.to_string());
        let resource = Resource::builder_empty()
            .with_service_name(service_name)
            .build();

        let provider = SdkLoggerProvider::builder()
            .with_batch_exporter(exporter)
            .with_resource(resource)
            .build();

        let serde_pool = Self::build_serde_pool()?;

        Ok(Self::new(provider, policy.otel_max_payload_bytes, serde_pool))
    }

    /// Serialize an `AuditRecord` into the `payload` attribute string.
    ///
    /// Pure-CPU and the bulk of `OtelSink::emit`'s cost. Designed to be called
    /// from `self.serde_pool` so it executes off the tokio runtime — see
    /// `emit`.
    fn payload_for_limit(
        rec: &AuditRecord,
        max_payload_bytes: usize,
    ) -> Option<(String, bool, Option<String>)> {
        let payload = match serde_json::to_string(rec) {
            Ok(s) => s,
            Err(err) => {
                tracing::warn!(target: "dynamo_llm::audit", "audit otel: serialize failed: {err}");
                return None;
            }
        };
        if payload.len() <= max_payload_bytes {
            return Some((payload, true, None));
        }

        marker_payload(
            rec,
            format!(
                "otel_payload_too_large:max_bytes={}:actual_bytes={}",
                max_payload_bytes,
                payload.len()
            ),
        )
    }
}

fn event_type_attr(event_type: AuditEventType) -> &'static str {
    match event_type {
        AuditEventType::Request => "request",
        AuditEventType::Response => "response",
    }
}

fn marker_payload(rec: &AuditRecord, reason: String) -> Option<(String, bool, Option<String>)> {
    tracing::warn!(
        target: "dynamo_llm::audit",
        request_id = %rec.request_id,
        audit_drop_reason = %reason,
        "audit otel: emitting incomplete marker"
    );

    let payload = json!({
        "schema_version": rec.schema_version,
        "event_type": event_type_attr(rec.event_type),
        "request_id": &rec.request_id,
        "requested_streaming": rec.requested_streaming,
        "model": &rec.model,
        "audit_complete": false,
        "audit_drop_reason": reason,
    });

    match serde_json::to_string(&payload) {
        Ok(s) => Some((s, false, Some(reason))),
        Err(err) => {
            tracing::warn!(target: "dynamo_llm::audit", "audit otel: marker serialize failed: {err}");
            None
        }
    }
}

#[async_trait]
impl AuditSink for OtelSink {
    fn name(&self) -> &'static str {
        "otel"
    }

    async fn emit(&self, rec: &AuditRecord) {
        // v8.2 OTEL serde offload: move the `serde_json::to_string` walk over
        // the full `Arc<NvCreateChatCompletionRequest>` from the tokio sink
        // task onto a dedicated bounded rayon pool so it cannot contend with
        // HTTP request futures via the shared tokio runtime. The
        // `logger.emit(record)` call below stays on the tokio task — it's
        // just an enqueue to the SDK BatchLogProcessor and is cheap.
        let max = self.max_payload_bytes;
        let rec_for_serde = rec.clone();
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.serde_pool.spawn(move || {
            let start = std::time::Instant::now();
            let result = Self::payload_for_limit(&rec_for_serde, max);
            let elapsed_us = start.elapsed().as_micros() as u64;
            let payload_len = result.as_ref().map(|(p, _, _)| p.len()).unwrap_or(0);
            tracing::debug!(
                target: "dynamo.audit.otel.serde",
                request_id = %rec_for_serde.request_id,
                event_type = event_type_attr(rec_for_serde.event_type),
                elapsed_us,
                payload_len,
                "OTEL audit payload serialized off-runtime"
            );
            let _ = tx.send(result);
        });

        let payload_result = match rx.await {
            Ok(result) => result,
            Err(err) => {
                tracing::warn!(
                    target: "dynamo_llm::audit",
                    request_id = %rec.request_id,
                    error = %err,
                    "audit otel: rayon serde worker dropped sender (likely panic in payload_for_limit); skipping record"
                );
                return;
            }
        };
        let Some((payload, audit_complete, audit_drop_reason)) = payload_result else {
            return;
        };

        let mut record = self.logger.create_log_record();
        record.set_severity_number(Severity::Info);
        record.set_severity_text("INFO".into());
        record.set_body(AnyValue::String(AUDIT_LOG_BODY.into()));
        record.add_attribute("request_id", AnyValue::String(rec.request_id.clone().into()));
        record.add_attribute(
            "event_type",
            AnyValue::String(event_type_attr(rec.event_type).into()),
        );
        record.add_attribute(
            "endpoint",
            AnyValue::String(AUDIT_ENDPOINT_CHAT_COMPLETION.into()),
        );
        record.add_attribute("model", AnyValue::String(rec.model.clone().into()));
        record.add_attribute("streaming", AnyValue::Boolean(rec.requested_streaming));
        record.add_attribute("audit_complete", AnyValue::Boolean(audit_complete));
        if let Some(reason) = audit_drop_reason {
            record.add_attribute("audit_drop_reason", AnyValue::String(reason.into()));
        }
        record.add_attribute("payload", AnyValue::String(payload.into()));
        self.logger.emit(record);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
    use serial_test::serial;

    fn sample_record() -> AuditRecord {
        AuditRecord {
            schema_version: 1,
            event_type: AuditEventType::Request,
            request_id: "req-otel-1".to_string(),
            requested_streaming: true,
            model: "test-model".to_string(),
            request: None,
            response: None,
        }
    }

    /// Sample record with a full request payload that exercises every wire
    /// type the serializer has to encode — strings, ints, bools, **floats**
    /// (the sampling params: temperature/top_p/frequency_penalty/presence_penalty,
    /// see `lib/protocols/src/types/chat.rs`), arrays of objects (messages),
    /// and nested objects (tools / nvext). The point of this record is to
    /// cover the round-trip path that production actually uses via
    /// `OtelSink::payload_for_limit`.
    fn sample_record_with_request() -> AuditRecord {
        let request_json = serde_json::json!({
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Reply with a single word."},
            ],
            "stream": true,
            "store": true,
            "temperature": 0.7,
            "top_p": 0.95,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.25,
            "max_tokens": 64,
            "n": 1,
            "seed": 42,
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }],
        });
        let request: NvCreateChatCompletionRequest =
            serde_json::from_value(request_json).expect("construct test request");
        AuditRecord {
            schema_version: 1,
            event_type: AuditEventType::Request,
            request_id: "req-otel-with-floats".to_string(),
            requested_streaming: true,
            model: "test-model".to_string(),
            request: Some(Arc::new(request)),
            response: None,
        }
    }

    /// Exercises the production serialization path: `payload_for_limit`
    /// → string → `serde_json::from_str` (what downstream consumers use to
    /// parse the `payload` attribute). Validates semantic round-trip on a
    /// record that contains floats (sampling params) + nested arrays/objects
    /// (messages, tools) — i.e. the same wire shape as a real chat-completion
    /// request.
    #[test]
    fn payload_for_limit_round_trips_a_full_request() {
        let rec = sample_record_with_request();
        let (payload, complete, drop_reason) =
            OtelSink::payload_for_limit(&rec, usize::MAX).expect("payload serializes");
        assert!(complete);
        assert!(drop_reason.is_none());

        let decoded: AuditRecord = serde_json::from_str(&payload)
            .expect("payload string decodes back to AuditRecord");
        assert_eq!(decoded.request_id, rec.request_id);
        assert_eq!(decoded.requested_streaming, rec.requested_streaming);
        assert_eq!(decoded.model, rec.model);
        assert_eq!(decoded.event_type, rec.event_type);

        // Round-trip the record through the JSON Value form to compare
        // structurally — sidesteps any field-ordering differences and proves
        // semantic equivalence (which is the only contract downstream
        // consumers rely on).
        let rec_value = serde_json::to_value(&rec).expect("rec serializes via serde_json");
        let decoded_value =
            serde_json::to_value(&decoded).expect("decoded serializes via serde_json");
        assert_eq!(rec_value, decoded_value);
    }

    #[test]
    fn payload_over_limit_emits_incomplete_marker() {
        let rec = sample_record();
        let (payload, audit_complete, audit_drop_reason) =
            OtelSink::payload_for_limit(&rec, 1).unwrap();

        assert!(!audit_complete);
        assert!(audit_drop_reason.unwrap().starts_with("otel_payload_too_large:"));
        let decoded: serde_json::Value = serde_json::from_str(&payload).unwrap();

        assert_eq!(decoded["audit_complete"], false);
        assert!(
            decoded["audit_drop_reason"]
                .as_str()
                .unwrap()
                .starts_with("otel_payload_too_large:")
        );
        assert!(decoded.get("request").is_none());
        assert!(decoded.get("response").is_none());
    }

    #[test]
    #[serial]
    fn protocol_env_defaults_to_http_protobuf() {
        temp_env::with_vars(
            [
                (env_otlp::OTEL_EXPORTER_OTLP_LOGS_PROTOCOL, None::<&str>),
                (env_otlp::OTEL_EXPORTER_OTLP_PROTOCOL, None::<&str>),
            ],
            || assert_eq!(OtlpLogsProtocol::from_env(), OtlpLogsProtocol::HttpProtobuf),
        );
    }

    #[test]
    #[serial]
    fn logs_protocol_takes_precedence_over_global() {
        temp_env::with_vars(
            [
                (env_otlp::OTEL_EXPORTER_OTLP_LOGS_PROTOCOL, Some("grpc")),
                (env_otlp::OTEL_EXPORTER_OTLP_PROTOCOL, Some("http/protobuf")),
            ],
            || assert_eq!(OtlpLogsProtocol::from_env(), OtlpLogsProtocol::Grpc),
        );
    }
}

