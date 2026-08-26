// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::http::HeaderMap;
use dynamo_protocols::types::{
    ChatCompletionRequestMessage, ChatCompletionRequestUserMessageContent,
    ChatCompletionRequestUserMessageContentPart,
};

use crate::protocols::openai::chat_completions::{
    NvCreateChatCompletionRequest, NvCreateChatCompletionResponse,
};

const INLINE_MEDIA_REDACTION_TAG: &str = "redacted-inline-media";

fn inline_media_data_uri_parts(value: &str) -> Option<(&str, &str)> {
    let scheme = value.get(..5)?;
    if !scheme.eq_ignore_ascii_case("data:") {
        return None;
    }
    let data_uri = value.get(5..)?;
    let (metadata, encoded_body) = data_uri.split_once(',')?;
    let mime_type = metadata.split(';').next()?.trim().to_ascii_lowercase();
    ["image/", "video/", "audio/"]
        .iter()
        .any(|prefix| mime_type.starts_with(prefix))
        .then_some((metadata, encoded_body))
}

/// Replace an inline image/video/audio data URI with a compact,
/// non-reversible marker suitable for request-payload logs.
///
/// The hash covers the encoded body in the parsed request. Keeping the original
/// data-URI metadata preserves the MIME type (and any parameters), while the
/// body itself never reaches a request-trace sink.
fn redact_inline_media_data_uri(value: &str) -> Option<String> {
    let (metadata, encoded_body) = inline_media_data_uri_parts(value)?;
    let encoded_hash = blake3::hash(encoded_body.as_bytes()).to_hex();
    Some(format!(
        "data:{metadata},{INLINE_MEDIA_REDACTION_TAG};encoded_blake3={encoded_hash};encoded_bytes={}",
        encoded_body.len()
    ))
}

fn redact_inline_media_data_uris(value: &mut serde_json::Value) -> usize {
    match value {
        serde_json::Value::Array(values) => values
            .iter_mut()
            .map(redact_inline_media_data_uris)
            .sum(),
        serde_json::Value::Object(values) => values
            .values_mut()
            .map(redact_inline_media_data_uris)
            .sum(),
        serde_json::Value::String(value) => {
            let Some(redacted) = redact_inline_media_data_uri(value) else {
                return 0;
            };
            *value = redacted;
            1
        }
        _ => 0,
    }
}

fn request_contains_inline_media_data_uri(req: &NvCreateChatCompletionRequest) -> bool {
    req.inner.messages.iter().any(|message| {
        let ChatCompletionRequestMessage::User(user_message) = message else {
            return false;
        };
        let ChatCompletionRequestUserMessageContent::Array(parts) = &user_message.content else {
            return false;
        };
        parts.iter().any(|part| {
            let url = match part {
                ChatCompletionRequestUserMessageContentPart::ImageUrl(part) => part
                    .image_url
                    .as_ref()
                    .map(|image_url| image_url.url.as_str()),
                ChatCompletionRequestUserMessageContentPart::VideoUrl(part) => part
                    .video_url
                    .as_ref()
                    .map(|video_url| video_url.url.as_str()),
                ChatCompletionRequestUserMessageContentPart::AudioUrl(part) => part
                    .audio_url
                    .as_ref()
                    .map(|audio_url| audio_url.url.as_str()),
                _ => None,
            };
            url.and_then(inline_media_data_uri_parts).is_some()
        })
    })
}

fn sanitized_request_snapshot(
    req: &NvCreateChatCompletionRequest,
) -> Result<NvCreateChatCompletionRequest, String> {
    if !request_contains_inline_media_data_uri(req) {
        return Ok(req.clone());
    }

    // Use the request's serialized shape so every supported message/content
    // variant is redacted without duplicating a mutable typed traversal. The
    // read-only fast path above keeps text-only payload capture clone-only.
    let mut value = serde_json::to_value(req).map_err(|error| error.to_string())?;
    let redacted_count = redact_inline_media_data_uris(&mut value);
    if redacted_count == 0 {
        return Err("inline media was detected but no data URI was redacted".to_string());
    }
    serde_json::from_value(value).map_err(|error| error.to_string())
}

/// Context key for the allowlisted headers captured at the HTTP layer.
pub const HTTP_HEADERS_CONTEXT_KEY: &str = "request_trace.http.request.headers";

/// True when payload records are being captured and a header allowlist is set.
pub(crate) fn http_header_capture_active() -> bool {
    if !super::config::capture_enabled() {
        return false;
    }
    let policy = super::policy();
    policy.emit_request_payload_records() && !policy.http_header_capture_list.is_empty()
}

/// Collect the allowlisted request headers (case-insensitive, comma-joined on
/// repeats). `None` unless payload capture is active and the allowlist is non-empty.
pub fn capture_http_headers(headers: &HeaderMap) -> Option<BTreeMap<String, String>> {
    if !http_header_capture_active() {
        return None;
    }
    capture_http_headers_with_list(headers, &super::policy().http_header_capture_list)
}

fn capture_http_headers_with_list(
    headers: &HeaderMap,
    capture_list: &[String],
) -> Option<BTreeMap<String, String>> {
    if capture_list.is_empty() {
        return None;
    }
    let mut out = BTreeMap::new();
    for name in capture_list {
        let joined = headers
            .get_all(name.as_str())
            .iter()
            .filter_map(|value| value.to_str().ok())
            .filter(|value| !value.is_empty())
            .collect::<Vec<_>>()
            .join(", ");
        if !joined.is_empty() {
            out.insert(name.clone(), joined);
        }
    }
    (!out.is_empty()).then_some(out)
}

pub struct RequestPayloadHandle {
    requested_streaming: bool,
    request_id: String,
    model: String,
    event_time: SystemTime,
    request: Arc<NvCreateChatCompletionRequest>,
    http_request_headers: Option<Arc<BTreeMap<String, String>>>,
}

impl RequestPayloadHandle {
    pub fn streaming(&self) -> bool {
        self.requested_streaming
    }

    pub fn request_id(&self) -> &str {
        &self.request_id
    }

    /// Publish one request trace payload record. Consumes the handle to enforce
    /// exactly one payload record per request. `response` is `None` on client
    /// cancel / gateway timeout / aggregation failure; the record still carries
    /// the request so those cases remain inspectable.
    pub fn emit(self, response: Option<Arc<NvCreateChatCompletionResponse>>) {
        super::record::emit_request_payload(
            super::RequestTracePayload {
                request_id: self.request_id,
                endpoint: "openai.chat_completion".to_string(),
                model: self.model,
                request: Some(self.request),
                response,
                http_request_headers: self.http_request_headers,
                payload_complete: true,
                payload_drop_reason: None,
            },
            unix_time_ms(self.event_time),
        );
    }
}

pub fn create_handle(
    req: &NvCreateChatCompletionRequest,
    request_id: &str,
    http_request_headers: Option<Arc<BTreeMap<String, String>>>,
) -> Option<RequestPayloadHandle> {
    let policy = super::policy();
    // `capture_enabled()` is `policy.enabled && CAPTURE_ACTIVE`: it additionally
    // requires request trace initialization, so a stale payload handle cannot be
    // created before/after the request trace lifecycle.
    create_handle_with_config(
        req,
        request_id,
        super::config::capture_enabled(),
        policy.emit_request_payload_records(),
        http_request_headers,
    )
}

fn unix_time_ms(time: SystemTime) -> u64 {
    time.duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis().min(u128::from(u64::MAX)) as u64)
        .unwrap_or(0)
}

fn create_handle_with_config(
    req: &NvCreateChatCompletionRequest,
    request_id: &str,
    enabled: bool,
    emit_request_payload: bool,
    http_request_headers: Option<Arc<BTreeMap<String, String>>>,
) -> Option<RequestPayloadHandle> {
    if !enabled || !emit_request_payload {
        return None;
    }
    let requested_streaming = req.inner.stream.unwrap_or(false);
    let model = req.inner.model.clone();
    let request = match sanitized_request_snapshot(req) {
        Ok(request) => request,
        Err(error) => {
            // Redaction is a privacy boundary. Fail closed rather than falling
            // back to the original request, which may contain inline media.
            tracing::warn!(
                request_id,
                %error,
                "request payload skipped because inline-media sanitization failed"
            );
            return None;
        }
    };

    Some(RequestPayloadHandle {
        requested_streaming,
        request_id: request_id.to_string(),
        model,
        // Snapshot the inbound request before the preprocessor overrides
        // stream/usage. Inline media bodies are replaced with compact metadata
        // markers before the snapshot can reach any request-trace sink.
        event_time: SystemTime::now(),
        request: Arc::new(request),
        http_request_headers,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Arc;

    fn create_test_request(model: &str, store: bool) -> NvCreateChatCompletionRequest {
        let json = serde_json::json!({
            "model": model,
            "messages": [{"role": "user", "content": "test"}],
            "store": store
        });
        serde_json::from_value(json).expect("Failed to create test request")
    }

    fn create_test_response(content: &str) -> NvCreateChatCompletionResponse {
        let json = serde_json::json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop"
            }]
        });
        serde_json::from_value(json).expect("Failed to create test response")
    }

    #[test]
    fn request_payload_records_emit_even_when_store_is_false() {
        let request = create_test_request("test-model", false);
        let handle = create_handle_with_config(&request, "test-id", true, true, None);

        assert!(
            handle.is_some(),
            "request_payload records should create a handle even with store=false"
        );
    }

    #[test]
    fn request_payload_records_disabled_skips_store_true_payloads() {
        let request = create_test_request("test-model", true);
        let handle = create_handle_with_config(&request, "test-id", true, false, None);

        assert!(
            handle.is_none(),
            "request_payload records disabled should skip payloads even with store=true"
        );
    }

    #[test]
    fn request_payload_redacts_inline_media_and_preserves_urls() {
        let image_body = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB";
        let video_body = "AAAAHGZ0eXBtcDQyAAAAAG1wNDJpc29t";
        let image_data_uri = format!("data:image/png;base64,{image_body}");
        let video_data_uri = format!("data:video/mp4;charset=binary;base64,{video_body}");
        let request: NvCreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe these inputs"},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_uri}
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/image.png"}
                    },
                    {
                        "type": "video_url",
                        "video_url": {"url": video_data_uri}
                    }
                ]
            }]
        }))
        .unwrap();

        let handle = create_handle_with_config(&request, "test-id", true, true, None).unwrap();
        let serialized = serde_json::to_string(handle.request.as_ref()).unwrap();

        assert!(!serialized.contains(image_body));
        assert!(!serialized.contains(video_body));
        assert!(serialized.contains("https://example.com/image.png"));
        assert_eq!(serialized.matches(INLINE_MEDIA_REDACTION_TAG).count(), 2);
        assert!(serialized.contains("data:image/png;base64,redacted-inline-media;"));
        assert!(
            serialized.contains("data:video/mp4;charset=binary;base64,redacted-inline-media;")
        );
        assert!(serialized.contains(&format!("encoded_bytes={}", image_body.len())));
        assert!(serialized.contains(&format!(
            "encoded_blake3={}",
            blake3::hash(image_body.as_bytes()).to_hex()
        )));
    }

    #[test]
    fn inline_media_redaction_covers_media_data_uris_only() {
        assert!(redact_inline_media_data_uri("https://example.com/image.png").is_none());
        assert!(redact_inline_media_data_uri("data:text/plain;base64,aGVsbG8=").is_none());
        assert!(redact_inline_media_data_uri("data:image/svg+xml,<svg/>").is_some());
        assert!(redact_inline_media_data_uri("DATA:IMAGE/PNG;BASE64,AAAA").is_some());
    }

    #[test]
    fn capture_http_headers_records_only_allowlisted() {
        let capture_list = vec!["x-request-id".to_string(), "nvcf-function-id".to_string()];

        let mut headers = HeaderMap::new();
        headers.insert("x-request-id", "abc-123".parse().unwrap());
        headers.insert("NVCF-Function-Id", "fn-9".parse().unwrap());
        headers.insert("authorization", "Bearer secret".parse().unwrap());

        let captured = capture_http_headers_with_list(&headers, &capture_list)
            .expect("allowlisted headers are captured");
        assert_eq!(
            captured.get("x-request-id").map(String::as_str),
            Some("abc-123")
        );
        assert_eq!(
            captured.get("nvcf-function-id").map(String::as_str),
            Some("fn-9")
        );
        assert!(
            !captured.contains_key("authorization"),
            "non-allowlisted header must never be captured"
        );
    }

    #[test]
    fn capture_http_headers_empty_list_captures_nothing() {
        let mut headers = HeaderMap::new();
        headers.insert("x-request-id", "abc-123".parse().unwrap());

        assert!(
            capture_http_headers_with_list(&headers, &[]).is_none(),
            "empty allowlist must capture nothing"
        );
    }

    #[test]
    fn capture_http_headers_joins_repeated_headers() {
        let capture_list = vec!["x-tag".to_string()];

        let mut headers = HeaderMap::new();
        headers.append("x-tag", "a".parse().unwrap());
        headers.append("x-tag", "b".parse().unwrap());

        let captured = capture_http_headers_with_list(&headers, &capture_list)
            .expect("repeated header is captured");
        assert_eq!(captured.get("x-tag").map(String::as_str), Some("a, b"));
    }

    #[test]
    fn capture_http_headers_omits_repeated_empty_values() {
        let capture_list = vec!["x-tag".to_string()];

        let mut headers = HeaderMap::new();
        headers.append("x-tag", "".parse().unwrap());
        headers.append("x-tag", "".parse().unwrap());

        assert!(
            capture_http_headers_with_list(&headers, &capture_list).is_none(),
            "repeated empty values must be omitted, not joined into \", \""
        );
    }

    #[test]
    fn capture_http_headers_skips_empty_values_when_joining() {
        let capture_list = vec!["x-tag".to_string()];

        let mut headers = HeaderMap::new();
        headers.append("x-tag", "".parse().unwrap());
        headers.append("x-tag", "tenant-a".parse().unwrap());

        let captured = capture_http_headers_with_list(&headers, &capture_list)
            .expect("non-empty value is captured");
        assert_eq!(captured.get("x-tag").map(String::as_str), Some("tenant-a"));
    }

    /// Test-only constructor. `create_handle` gates on env vars + a cached
    /// `OnceLock` policy, which is too brittle for a focused bus-roundtrip test.
    impl RequestPayloadHandle {
        pub(crate) fn for_test(request_id: &str, model: &str, streaming: bool) -> Self {
            Self {
                requested_streaming: streaming,
                request_id: request_id.to_string(),
                model: model.to_string(),
                event_time: SystemTime::now(),
                request: Arc::new(create_test_request(model, true)),
                http_request_headers: None,
            }
        }
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn emit_publishes_one_combined_record_with_and_without_response() {
        // Exercises the contract: `emit` publishes exactly one request trace
        // payload record. With a response present the record carries both
        // request and response; with `None` it carries the request only.
        crate::request_trace::init_bus_for_test(8);
        let mut rx = crate::request_trace::subscribe();

        RequestPayloadHandle::for_test("payload-test-req-ok", "test-model", true)
            .emit(Some(Arc::new(create_test_response("hello"))));
        RequestPayloadHandle::for_test("payload-test-req-cancel", "test-model", true).emit(None);

        let mut records = HashMap::new();
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while records.len() < 2 {
                let record = rx.recv().await.expect("record receives ok");
                if record.event_type != crate::request_trace::RequestTraceEventType::RequestPayload
                {
                    continue;
                }
                let Some(payload) = record.payload.as_ref() else {
                    continue;
                };
                if matches!(
                    payload.request_id.as_str(),
                    "payload-test-req-ok" | "payload-test-req-cancel"
                ) {
                    records.insert(payload.request_id.clone(), record);
                }
            }
        })
        .await
        .expect("expected request payload records before timeout");

        let first = records
            .remove("payload-test-req-ok")
            .expect("payload-test-req-ok record");
        let second = records
            .remove("payload-test-req-cancel")
            .expect("payload-test-req-cancel record");

        assert_eq!(
            first.event_type,
            crate::request_trace::RequestTraceEventType::RequestPayload
        );
        let first_payload = first.payload.as_ref().expect("first payload");
        assert_eq!(first_payload.request_id, "payload-test-req-ok");
        assert!(first_payload.request.is_some());
        assert!(first_payload.response.is_some());

        assert_eq!(
            second.event_type,
            crate::request_trace::RequestTraceEventType::RequestPayload
        );
        let second_payload = second.payload.as_ref().expect("second payload");
        assert_eq!(second_payload.request_id, "payload-test-req-cancel");
        assert!(second_payload.request.is_some());
        assert!(second_payload.response.is_none());
    }
}
