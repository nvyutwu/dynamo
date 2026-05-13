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
    pub fn new(provider: SdkLoggerProvider, max_payload_bytes: usize) -> Self {
        let logger = provider.logger(AUDIT_INSTRUMENTATION_SCOPE);
        let num_threads = std::env::var(ENV_OTEL_SERDE_THREADS)
            .ok()
            .and_then(|raw| raw.parse::<usize>().ok())
            .filter(|n| *n > 0)
            .unwrap_or(DEFAULT_OTEL_SERDE_THREADS);
        let serde_pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .thread_name(|i| format!("otel-audit-serde-{i}"))
                .build()
                .expect("OTEL audit serde rayon pool"),
        );
        Self {
            provider,
            logger,
            max_payload_bytes,
            serde_pool,
        }
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

        Ok(Self::new(provider, policy.otel_max_payload_bytes))
    }

    /// Serialize an `AuditRecord` into the `payload` attribute string.
    ///
    /// Pure-CPU and the bulk of `OtelSink::emit`'s cost. `sonic-rs` is used
    /// instead of `serde_json::to_string` here (~2-3× faster on dense JSON
    /// like our `messages`/`tools` payloads). Designed to be called from
    /// `self.serde_pool` so it executes off the tokio runtime — see `emit`.
    fn payload_for_limit(
        rec: &AuditRecord,
        max_payload_bytes: usize,
    ) -> Option<(String, bool, Option<String>)> {
        let payload = match sonic_rs::to_string(rec) {
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
        // v8.2 OTEL serde offload (A2+B): move the `to_string` walk over the
        // full `Arc<NvCreateChatCompletionRequest>` from the tokio sink task
        // onto a dedicated rayon pool so it cannot contend with HTTP request
        // futures via the shared tokio runtime. The `logger.emit(record)`
        // call below stays on the tokio task — it's just an enqueue to the
        // SDK BatchLogProcessor and is cheap.
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

        let Some((payload, audit_complete, audit_drop_reason)) = rx.await.unwrap_or(None) else {
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

    /// The `payload` attribute is a JSON-encoded string of the full
    /// `AuditRecord`. Validates that the encoding round-trips so that
    /// downstream consumers can parse the attribute back into the record.
    #[test]
    fn payload_attribute_round_trips_through_json() {
        let rec = sample_record();
        let encoded = serde_json::to_string(&rec).expect("AuditRecord serializes");
        let decoded: AuditRecord =
            serde_json::from_str(&encoded).expect("payload string decodes back to AuditRecord");
        assert_eq!(decoded.request_id, rec.request_id);
        assert_eq!(decoded.requested_streaming, rec.requested_streaming);
        assert_eq!(decoded.model, rec.model);
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

