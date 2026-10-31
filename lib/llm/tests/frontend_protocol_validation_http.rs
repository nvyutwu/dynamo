// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP regressions for validation performed by protocol adapters.

use std::sync::Arc;
use std::time::{Duration, Instant};

use dynamo_llm::{
    discovery::UNKNOWN_METRIC_MODEL,
    http::service::metrics::{Endpoint, ErrorType, RequestType, Status},
    protocols::{Annotated, openai::chat_completions::NvCreateChatCompletionStreamResponse},
};
use dynamo_runtime::{
    config::environment_names::llm::{
        DYN_DISABLE_FRONTEND_NVEXT, DYN_DISABLE_OPENAI_STRICT_SCHEMA_VALIDATION,
        DYN_ENABLE_ANTHROPIC_API, DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS,
        DYN_HTTP_PRE_COMMIT_ERROR_PEEK_MS,
    },
    error::{DynamoError, ErrorType as DynamoErrorType},
};
use serde_json::{Value, json};
use serial_test::serial;

#[allow(dead_code)]
#[path = "common/http_harness.rs"]
mod http_harness;
#[path = "common/ports.rs"]
mod ports;
#[allow(dead_code)]
#[path = "common/scripted_chat_engine.rs"]
mod scripted_chat_engine;

use http_harness::{HarnessService, MODEL, load_agent_fixture};
use scripted_chat_engine::{Script, ScriptedChatEngine};

// Strict function-schema validation is opt-in on the rainhollow branch
// (DYN_STRICT_FUNCTION_SCHEMA); these tests exercise the strict rules.
const BASE_ENV: [(&str, Option<&str>); 5] = [
    (DYN_ENABLE_ANTHROPIC_API, Some("1")),
    (DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS, Some("0")),
    (DYN_HTTP_PRE_COMMIT_ERROR_PEEK_MS, None),
    (DYN_DISABLE_OPENAI_STRICT_SCHEMA_VALIDATION, None),
    ("DYN_STRICT_FUNCTION_SCHEMA", Some("1")),
];

fn nvext_disabled_env() -> Vec<(&'static str, Option<&'static str>)> {
    let mut env = BASE_ENV.to_vec();
    env.push((DYN_DISABLE_FRONTEND_NVEXT, Some("1")));
    env
}

const HTTP_TIMEOUT: Duration = Duration::from_secs(10);

async fn bounded<F: std::future::Future>(future: F) -> F::Output {
    tokio::time::timeout(HTTP_TIMEOUT, future)
        .await
        .expect("HTTP operation exceeded HTTP_TIMEOUT")
}

async fn post_json(svc: &HarnessService, path: &str, body: Value) -> reqwest::Response {
    let request = svc
        .client
        .post(format!("{}{path}", svc.base_url))
        .json(&body);
    bounded(request.send()).await.unwrap()
}

#[derive(Clone, Copy)]
enum ExpectedError {
    Validation,
    NotImplemented,
    UnsupportedContent,
    DeadlineExceeded,
}

impl ExpectedError {
    fn status(self) -> reqwest::StatusCode {
        match self {
            Self::Validation => reqwest::StatusCode::BAD_REQUEST,
            Self::NotImplemented => reqwest::StatusCode::BAD_REQUEST,
            Self::UnsupportedContent => reqwest::StatusCode::BAD_REQUEST,
            Self::DeadlineExceeded => reqwest::StatusCode::TOO_MANY_REQUESTS,
        }
    }

    fn anthropic_type(self) -> &'static str {
        match self {
            Self::Validation => "invalid_request_error",
            Self::NotImplemented => "invalid_request_error",
            Self::UnsupportedContent => "invalid_request_error",
            Self::DeadlineExceeded => "rate_limit_error",
        }
    }
}

async fn assert_openai_error(response: reqwest::Response, expected: ExpectedError, message: &str) {
    let status = expected.status();
    assert_eq!(response.status(), status);
    let body: Value = response.json().await.unwrap();
    assert_eq!(body["code"].as_u64(), Some(u64::from(status.as_u16())));
    assert!(
        body["message"].as_str().is_some_and(|actual| actual
            .to_ascii_lowercase()
            .contains(&message.to_ascii_lowercase())),
        "unexpected OpenAI error body: {body}"
    );
}

async fn assert_anthropic_error(
    response: reqwest::Response,
    expected: ExpectedError,
    message: &str,
) {
    assert_eq!(response.status(), expected.status());
    let body: Value = response.json().await.unwrap();
    assert_eq!(body["type"], "error");
    assert_eq!(body["error"]["type"], expected.anthropic_type());
    assert!(
        body["error"]["message"]
            .as_str()
            .is_some_and(|actual| actual.contains(message)),
        "unexpected Anthropic error body: {body}"
    );
}

async fn assert_anthropic_status(
    response: reqwest::Response,
    status: reqwest::StatusCode,
    error_type: &str,
    message: &str,
) {
    assert_eq!(response.status(), status);
    let body: Value = response.json().await.unwrap();
    assert_eq!(body["type"], "error");
    assert_eq!(body["error"]["type"], error_type);
    assert!(
        body["error"]["message"]
            .as_str()
            .is_some_and(|actual| actual.contains(message)),
        "unexpected Anthropic error body: {body}"
    );
}

#[tokio::test]
#[serial]
async fn invalid_anthropic_cache_salt_is_rejected_when_nvext_is_disabled() {
    temp_env::async_with_vars(nvext_disabled_env(), async {
        let svc = HarnessService::start(Vec::new()).await;
        for streaming in [false, true] {
            let response = post_json(
                &svc,
                "/v1/messages",
                json!({
                    "model": MODEL,
                    "max_tokens": 16,
                    "stream": streaming,
                    "messages": [{"role": "user", "content": "ping"}],
                    "nvext": {
                        "cache_salt": 42,
                        "agent_hints": {"priority": 5}
                    }
                }),
            )
            .await;

            assert_anthropic_error(
                response,
                ExpectedError::Validation,
                "invalid nvext.cache_salt: expected a string or null",
            )
            .await;
            let request_type = if streaming {
                RequestType::Stream
            } else {
                RequestType::Unary
            };
            assert_error_metrics(
                &svc,
                &Endpoint::AnthropicMessages,
                &request_type,
                &[(ErrorType::Validation, 1), (ErrorType::Internal, 0)],
            );
        }
        assert!(svc.engine.take_requests().await.is_empty());
        svc.shutdown().await;
    })
    .await;
}

fn assert_error_metrics(
    svc: &HarnessService,
    endpoint: &Endpoint,
    request_type: &RequestType,
    expected: &[(ErrorType, u64)],
) {
    assert_error_metrics_for_model(svc, MODEL, endpoint, request_type, expected);
}

fn assert_error_metrics_for_model(
    svc: &HarnessService,
    model: &str,
    endpoint: &Endpoint,
    request_type: &RequestType,
    expected: &[(ErrorType, u64)],
) {
    for (error_type, expected) in expected {
        assert_eq!(
            svc.metrics.get_request_counter(
                model,
                endpoint,
                request_type,
                &Status::Error,
                error_type,
            ),
            *expected,
            "unexpected {error_type:?} count for {endpoint}/{request_type}"
        );
    }
}

/// A router deadline rejection reaches the Anthropic surface as HTTP 429
/// `rate_limit_error` with the `Cancelled` metric label, the same contract as
/// the OpenAI surface, rather than falling through to a 500 `api_error`.
#[tokio::test]
#[serial]
async fn anthropic_deadline_exceeded_maps_to_rate_limit_error_with_cancelled_metric() {
    temp_env::async_with_vars(BASE_ENV, async {
        let error = DynamoError::builder()
            .error_type(DynamoErrorType::DeadlineExceeded)
            .reason(
                dynamo_runtime::error::ErrorReason::new("router.queue_deadline_exceeded").unwrap(),
            )
            .message("internal deadline detail")
            .build();
        let svc = HarnessService::start_with_generate_error(error).await;

        let response = post_json(
            &svc,
            "/v1/messages",
            json!({
                "model": MODEL,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "ping"}]
            }),
        )
        .await;

        assert_anthropic_error(
            response,
            ExpectedError::DeadlineExceeded,
            "request deadline exceeded",
        )
        .await;
        assert_error_metrics(
            &svc,
            &Endpoint::AnthropicMessages,
            &RequestType::Unary,
            &[
                (ErrorType::Cancelled, 1),
                (ErrorType::Overload, 0),
                (ErrorType::Internal, 0),
            ],
        );
        svc.shutdown().await;
    })
    .await;
}

fn tool_name_requests(name: &str) -> [(&'static str, Value, bool); 3] {
    [
        (
            "/v1/messages",
            json!({
                "model": MODEL,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "ping"}],
                "tools": [{
                    "name": name,
                    "input_schema": {"type": "object", "properties": {}}
                }]
            }),
            true,
        ),
        (
            "/v1/responses",
            json!({
                "model": MODEL,
                "input": "ping",
                "tools": [{
                    "type": "function",
                    "name": name,
                    "parameters": {"type": "object", "properties": {}}
                }]
            }),
            false,
        ),
        (
            "/v1/chat/completions",
            json!({
                "model": MODEL,
                "messages": [{"role": "user", "content": "ping"}],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": name,
                        "parameters": {"type": "object", "properties": {}}
                    }
                }]
            }),
            false,
        ),
    ]
}

fn strict_tool_requests(function: Value) -> [(&'static str, Value); 3] {
    let mut flat = function.clone();
    flat["type"] = json!("function");
    [
        (
            "/v1/chat/completions",
            json!({
                "model": MODEL, "messages": [{"role": "user", "content": "ping"}],
                "tools": [{"type": "function", "function": function}]
            }),
        ),
        (
            "/v1/responses",
            json!({
                "model": MODEL, "input": "ping", "tools": [flat]
            }),
        ),
        (
            "/v1/responses",
            json!({
                "model": MODEL, "input": "ping", "tools": [{
                    "type": "namespace", "name": "utilities", "description": "Utility functions",
                    "tools": [flat]
                }]
            }),
        ),
    ]
}

#[tokio::test]
#[serial]
async fn strict_tool_schema_rejected_before_dispatch() {
    temp_env::async_with_vars(BASE_ENV, async {
        let svc = HarnessService::start(Vec::new()).await;
        for (path, body) in strict_tool_requests(json!({
            "name": "search", "strict": true, "parameters": {
                "type": "object", "properties": {"query": {"type": "string", "minLength": 1}}
            }
        })) {
            for stream in [false, true] {
                for choice in [
                    json!("auto"),
                    json!("none"),
                    json!({
                        "type": "allowed_tools", "mode": "auto",
                        "tools": [{"type": "function", "name": "retained"}]
                    }),
                ] {
                    if choice.is_object() && path != "/v1/responses" {
                        continue;
                    }
                    let mut body = body.clone();
                    body["stream"] = json!(stream);
                    if choice.is_object() {
                        body["tools"].as_array_mut().unwrap().push(json!({
                            "type": "function", "name": "retained"
                        }));
                    }
                    body["tool_choice"] = choice;
                    let response = post_json(&svc, path, body).await;
                    assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);
                    let error: Value = bounded(response.json()).await.unwrap();
                    assert_eq!(error["code"], 400);
                    assert_eq!(error["type"], "Bad Request");
                    let message = error["message"].as_str().unwrap();
                    let prefix = if path == "/v1/responses" {
                        "Failed to convert responses request: "
                    } else {
                        "Validation: "
                    };
                    assert!(message.starts_with(prefix), "{message}");
                    assert!(
                        message.contains("Invalid schema for function 'search': In context="),
                        "{message}"
                    );
                    assert!(message.contains("additionalProperties"), "{message}");
                    assert!(svc.engine.take_requests().await.is_empty());
                }
            }
        }
        for request_type in [RequestType::Unary, RequestType::Stream] {
            assert_error_metrics(
                &svc,
                &Endpoint::Responses,
                &request_type,
                &[(ErrorType::Validation, 6), (ErrorType::Internal, 0)],
            );
        }
        svc.shutdown().await;
    })
    .await;
}

// The process-wide startup policy needs a fresh process for opt-in cases.
async fn run_compatibility_subprocess(test_name: &str) -> bool {
    const CHILD: &str = "DYNAMO_STRICT_SCHEMA_TEST_CHILD";
    if std::env::var(CHILD).as_deref() == Ok(test_name) {
        return false;
    }
    let output = tokio::time::timeout(
        Duration::from_secs(45),
        tokio::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", test_name, "--nocapture"])
            .env(CHILD, test_name)
            .env(DYN_DISABLE_OPENAI_STRICT_SCHEMA_VALIDATION, "1")
            .kill_on_drop(true)
            .output(),
    )
    .await
    .expect("compatibility subprocess must finish")
    .unwrap();
    assert!(
        output.status.success(),
        "compatibility subprocess failed: {}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("1 passed; 0 failed"),
        "subprocess must execute the selected test: {}",
        String::from_utf8_lossy(&output.stdout),
    );
    true
}

#[tokio::test]
#[serial]
async fn strict_tool_schema_compatibility_preserves_submitted_functions_across_adapters() {
    let parameters = json!({
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "value": {"oneOf": [{"type": "string"}, {"type": "integer"}]},
            "options": {
                "type": "object",
                "properties": {"limit": {"type": "integer", "default": 5}}
            }
        },
        "required": ["query"],
        "additionalProperties": false
    });
    if run_compatibility_subprocess(
        "strict_tool_schema_compatibility_preserves_submitted_functions_across_adapters",
    )
    .await
    {
        temp_env::async_with_vars(BASE_ENV, async {
            let svc = HarnessService::start(Vec::new()).await;
            temp_env::async_with_vars(
                [(DYN_DISABLE_OPENAI_STRICT_SCHEMA_VALIDATION, Some("1"))],
                async {
                    let response = post_json(
                        &svc,
                        "/v1/messages",
                        json!({
                            "model": MODEL, "max_tokens": 64,
                            "messages": [{"role": "user", "content": "ping"}],
                            "tools": [{"name": "search", "strict": true, "input_schema": parameters}]
                        }),
                    )
                    .await;
                    assert_anthropic_error(response, ExpectedError::Validation, "must include property").await;
                    assert!(svc.engine.take_requests().await.is_empty());
                },
            )
            .await;
            svc.shutdown().await;
        })
        .await;
        return;
    }

    let function = json!({"name": "search", "strict": true, "parameters": parameters});
    let mut requests = strict_tool_requests(function).to_vec();
    requests.push((
        "/v1/messages",
        json!({
            "model": MODEL, "max_tokens": 64,
            "messages": [{"role": "user", "content": "ping"}],
            "tools": [{"name": "search", "strict": true, "input_schema": parameters}]
        }),
    ));

    let script = load_agent_fixture("text.sse").await.unwrap();
    let mut env = BASE_ENV;
    env.iter_mut()
        .find(|(name, _)| *name == DYN_DISABLE_OPENAI_STRICT_SCHEMA_VALIDATION)
        .unwrap()
        .1 = Some("1");
    temp_env::async_with_vars(env, async {
        let svc = HarnessService::start(vec![script; requests.len()]).await;
        temp_env::async_with_vars(
            [(DYN_DISABLE_OPENAI_STRICT_SCHEMA_VALIDATION, None::<&str>)],
            async {
                for (path, body) in requests {
                    let response = post_json(&svc, path, body).await;
                    assert_eq!(response.status(), reqwest::StatusCode::OK);
                    bounded(response.bytes()).await.unwrap();
                    let requests = svc.engine.take_requests().await;
                    assert_eq!(requests.len(), 1);
                    let tools = requests[0].inner.tools.as_ref().unwrap();
                    assert_eq!(tools.len(), 1);
                    assert_eq!(tools[0].function.name, "search");
                    assert_eq!(tools[0].function.strict, Some(true));
                    assert_eq!(tools[0].function.parameters.as_ref(), Some(&parameters));
                }
            },
        )
        .await;
        assert_eq!(svc.engine.remaining_scripts().await, 0);
        svc.shutdown().await;
    })
    .await;
}

#[tokio::test]
#[serial]
async fn strict_tool_schema_compatibility_retains_basic_validation() {
    if run_compatibility_subprocess("strict_tool_schema_compatibility_retains_basic_validation")
        .await
    {
        return;
    }
    let mut env = BASE_ENV;
    env.iter_mut()
        .find(|(name, _)| *name == DYN_DISABLE_OPENAI_STRICT_SCHEMA_VALIDATION)
        .unwrap()
        .1 = Some("1");
    temp_env::async_with_vars(env, async {
        let svc = HarnessService::start(Vec::new()).await;
        for (function, message) in [
            (
                json!({"name": "invalid name", "strict": true, "parameters": {}}),
                "invalid name",
            ),
            (
                json!({"name": "search", "strict": true, "parameters": []}),
                "must be a JSON Schema object",
            ),
        ] {
            for (path, body) in strict_tool_requests(function) {
                let response = post_json(&svc, path, body).await;
                assert_openai_error(response, ExpectedError::Validation, message).await;
            }
        }
        assert!(svc.engine.take_requests().await.is_empty());
        svc.shutdown().await;
    })
    .await;
}

#[tokio::test]
#[serial]
async fn strict_tool_schema_diagnostics_are_bounded() {
    temp_env::async_with_vars(BASE_ENV, async {
        let svc = HarnessService::start(Vec::new()).await;
        let name = "x".repeat(1 << 20);
        for (path, body) in strict_tool_requests(json!({
            "name": "search", "strict": true, "parameters": {
                "type": "object", "additionalProperties": false, "properties": {name: true}
            }
        })) {
            for stream in [false, true] {
                let mut body = body.clone();
                body["stream"] = json!(stream);
                let response = post_json(&svc, path, body).await;
                assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);
                let bytes = bounded(response.bytes()).await.unwrap();
                assert!(bytes.len() <= 4096, "error body is {} bytes", bytes.len());
                let error: Value = serde_json::from_slice(&bytes).unwrap();
                assert_eq!(error["code"], 400);
                let message = error["message"].as_str().unwrap();
                assert!(message.contains("In context=#/required"));
                assert!(message.contains("must include property"));
                assert!(message.contains("..."));
                assert!(svc.engine.take_requests().await.is_empty());
            }
        }
        svc.shutdown().await;
    })
    .await;
}

#[tokio::test]
#[serial]
async fn strict_tool_schema_reference_chain_completes() {
    temp_env::async_with_vars(BASE_ENV, async {
        let count = 16_000;
        let mut definitions = serde_json::Map::new();
        for i in 0..count {
            definitions.insert(
                format!("n{i:05}"),
                json!({"type": "object", "$ref": format!("#/$defs/n{:05}", i + 1)}),
            );
        }
        definitions.insert(
            format!("n{count:05}"),
            json!({"type": "object", "additionalProperties": false}),
        );
        let parameters = json!({"$ref": "#/$defs/n00000", "$defs": definitions});
        let script = load_agent_fixture("text.sse").await.unwrap();
        let svc = HarnessService::start(vec![script]).await;
        let started = Instant::now();
        let response = post_json(
            &svc,
            "/v1/responses",
            json!({
                "model": MODEL, "input": "ping", "tools": [{
                    "type": "function", "name": "search", "strict": true, "parameters": parameters
                }]
            }),
        )
        .await;
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        bounded(response.bytes()).await.unwrap();
        assert!(
            started.elapsed() < HTTP_TIMEOUT,
            "reference-chain request exceeded HTTP_TIMEOUT"
        );
        let requests = svc.engine.take_requests().await;
        assert_eq!(requests.len(), 1);
        assert!(
            requests[0].inner.tools.as_ref().unwrap()[0]
                .function
                .parameters
                .as_ref()
                == Some(&parameters)
        );
        svc.shutdown().await;
    })
    .await;
}

#[tokio::test]
#[serial]
async fn strict_tool_schema_acceptance_preserves_submitted_functions() {
    temp_env::async_with_vars(BASE_ENV, async {
        let valid = json!({"type": "object", "properties": {"query": {"type": "string", "minLength": 1}}, "required": ["query"], "additionalProperties": false});
        let loose = json!({"type": "object", "properties": {"query": {"type": "string"}}});
        let mut functions = vec![json!({"name": "search", "strict": true, "parameters": valid})];
        for strict in [None, Some(json!(false))] {
            let mut function = json!({"name": "search", "parameters": loose});
            if let Some(strict) = strict { function["strict"] = strict; }
            functions.push(function);
        }
        functions.push(json!({"name": "search", "strict": true}));
        let script = load_agent_fixture("text.sse").await.unwrap();
        let svc = HarnessService::start(vec![script; functions.len() * 6]).await;
        for function in functions {
            for (path, body) in strict_tool_requests(function.clone()) {
                for stream in [false, true] {
                    let mut body = body.clone();
                    body["stream"] = json!(stream);
                    let response = post_json(&svc, path, body).await;
                    assert_eq!(response.status(), reqwest::StatusCode::OK);
                    bounded(response.bytes()).await.unwrap();
                    let requests = svc.engine.take_requests().await;
                    assert_eq!(requests.len(), 1);
                    let tools = requests[0].inner.tools.as_ref().unwrap();
                    assert_eq!(tools.len(), 1);
                    let actual = &tools[0].function;
                    assert_eq!(actual.name, "search");
                    assert_eq!(actual.strict, function["strict"].as_bool());
                    assert_eq!(actual.parameters.as_ref(), function.get("parameters"));
                }
            }
        }
        assert_eq!(svc.engine.remaining_scripts().await, 0);
        svc.shutdown().await;
    }).await;
}

fn dynamo_error(error_type: DynamoErrorType, message: &str) -> anyhow::Error {
    anyhow::Error::new(
        DynamoError::builder()
            .error_type(error_type)
            .message(message)
            .build(),
    )
}

fn backend_error_script(status: reqwest::StatusCode, message: &str) -> Script {
    vec![Annotated::<NvCreateChatCompletionStreamResponse> {
        data: None,
        id: None,
        event: Some("error".to_string()),
        comment: Some(vec![
            json!({
                "message": message,
                "code": status.as_u16(),
            })
            .to_string(),
        ]),
        error: None,
    }]
}

#[tokio::test]
#[serial]
async fn responses_conversion_distinguishes_invalid_from_unsupported() {
    temp_env::async_with_vars(BASE_ENV, async {
        let svc = HarnessService::start(Vec::new()).await;

        for (stream, content, expected, message) in [
            (
                true,
                json!({"type": "input_image", "file_id": "file_123"}),
                ExpectedError::UnsupportedContent,
                "image input by file_id",
            ),
            (
                false,
                json!({
                    "type": "input_file",
                    "file_url": "https://example.com/report.pdf"
                }),
                ExpectedError::UnsupportedContent,
                "file input content",
            ),
            (
                false,
                json!({"type": "input_image"}),
                ExpectedError::Validation,
                "requires file_id or image_url",
            ),
            (
                true,
                json!({"type": "input_file"}),
                ExpectedError::Validation,
                "requires exactly one of file_data, file_id, or file_url",
            ),
        ] {
            let response = post_json(
                &svc,
                "/v1/responses",
                json!({
                    "model": MODEL,
                    "stream": stream,
                    "input": [{"role": "user", "content": [content]}]
                }),
            )
            .await;
            assert_openai_error(response, expected, message).await;
        }

        for request_type in [RequestType::Unary, RequestType::Stream] {
            assert_error_metrics(
                &svc,
                &Endpoint::Responses,
                &request_type,
                &[
                    (ErrorType::NotImplemented, 1),
                    (ErrorType::Validation, 1),
                    (ErrorType::Internal, 0),
                ],
            );
        }

        assert!(svc.engine.take_requests().await.is_empty());
        svc.shutdown().await;
    })
    .await;
}

#[tokio::test]
#[serial]
async fn anthropic_tools_reject_unsupported_and_malformed_definitions() {
    temp_env::async_with_vars(BASE_ENV, async {
        let svc = HarnessService::start(Vec::new()).await;

        for (stream, tool_choice) in [
            (false, None),
            (true, Some(json!({"type": "tool", "name": "web_search"}))),
        ] {
            let mut body = json!({
                "model": MODEL,
                "max_tokens": 16,
                "stream": stream,
                "messages": [{"role": "user", "content": "ping"}],
                "tools": [{
                    "type": "web_search_20260209",
                    "name": "web_search"
                }]
            });
            if let Some(tool_choice) = tool_choice {
                body["tool_choice"] = tool_choice;
            }
            let response = post_json(&svc, "/v1/messages", body).await;
            assert_anthropic_error(
                response,
                ExpectedError::NotImplemented,
                "server tool type \"web_search_20260209\" is not supported",
            )
            .await;
        }

        let response = post_json(
            &svc,
            "/v1/messages",
            json!({
                    "model": MODEL,
                    "max_tokens": 16,
                    "stream": false,
                    "messages": [{"role": "user", "content": "ping"}],
                    "tools": [{"name": "get_weather"}]
            }),
        )
        .await;
        assert_anthropic_error(
            response,
            ExpectedError::Validation,
            "tools[0].input_schema: field required",
        )
        .await;

        for request_type in [RequestType::Unary, RequestType::Stream] {
            let validation = match &request_type {
                RequestType::Unary => 1,
                RequestType::Stream => 0,
            };
            assert_error_metrics(
                &svc,
                &Endpoint::AnthropicMessages,
                &request_type,
                &[
                    (ErrorType::NotImplemented, 1),
                    (ErrorType::Validation, validation),
                    (ErrorType::Internal, 0),
                ],
            );
        }

        assert!(svc.engine.take_requests().await.is_empty());
        svc.shutdown().await;
    })
    .await;
}

#[tokio::test]
#[serial]
async fn anthropic_named_tool_choice_requires_a_name_before_dispatch() {
    temp_env::async_with_vars(BASE_ENV, async {
        let script = load_agent_fixture("fragmented-tool.sse").await.unwrap();
        let svc = HarnessService::start([script]).await;
        let mut body = json!({
            "model": MODEL,
            "max_tokens": 64,
            "stream": true,
            "messages": [{"role": "user", "content": "List the current directory."}],
            "tools": [
                {"name": "list_directory", "input_schema": {
                    "type": "object", "properties": {"path": {"type": "string"}}
                }},
                {"name": "get_weather", "input_schema": {
                    "type": "object", "properties": {"city": {"type": "string"}}
                }}
            ],
            "tool_choice": {"type": "tool"}
        });

        let response = post_json(&svc, "/v1/messages", body.clone()).await;
        assert_anthropic_error(response, ExpectedError::Validation, "tool_choice.name").await;
        assert!(svc.engine.take_requests().await.is_empty());

        body["tool_choice"]["name"] = json!("list_directory");
        let response = post_json(&svc, "/v1/messages", body).await;
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        bounded(response.bytes()).await.unwrap();
        let requests = svc.engine.take_requests().await;
        assert_eq!(requests.len(), 1);
        assert_eq!(
            serde_json::to_value(&requests[0].inner.tool_choice).unwrap(),
            json!({"type": "function", "function": {"name": "list_directory"}})
        );
        svc.shutdown().await;
    })
    .await;
}

#[tokio::test]
#[serial]
async fn anthropic_backend_error_events_record_status_classification() {
    temp_env::async_with_vars(BASE_ENV, async {
        let overload_status = reqwest::StatusCode::from_u16(529).unwrap();
        let cancelled_status = reqwest::StatusCode::from_u16(499).unwrap();
        let engine = Arc::new(ScriptedChatEngine::new(
            [
                backend_error_script(cancelled_status, "backend cancellation context"),
                backend_error_script(reqwest::StatusCode::SERVICE_UNAVAILABLE, "worker offline"),
                backend_error_script(reqwest::StatusCode::TOO_MANY_REQUESTS, "rate limited"),
                backend_error_script(overload_status, "pool overloaded"),
                backend_error_script(reqwest::StatusCode::NOT_FOUND, "missing backend model"),
                backend_error_script(reqwest::StatusCode::BAD_REQUEST, "bad backend request"),
                backend_error_script(reqwest::StatusCode::INTERNAL_SERVER_ERROR, "backend panic"),
            ]
            .into_iter()
            .map(Ok),
        ));
        let svc = HarnessService::start_with_engine(engine).await;

        for (status, error_type, message) in [
            (cancelled_status, "api_error", "Request cancelled"),
            (
                reqwest::StatusCode::SERVICE_UNAVAILABLE,
                "overloaded_error",
                "Internal server error",
            ),
            (
                reqwest::StatusCode::TOO_MANY_REQUESTS,
                "rate_limit_error",
                "Too Many Requests",
            ),
            (overload_status, "overloaded_error", "Internal server error"),
            (
                reqwest::StatusCode::NOT_FOUND,
                "not_found_error",
                "Not Found",
            ),
            (
                reqwest::StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "Bad Request",
            ),
            (
                reqwest::StatusCode::INTERNAL_SERVER_ERROR,
                "api_error",
                "Internal server error",
            ),
        ] {
            let response = post_json(
                &svc,
                "/v1/messages",
                json!({
                    "model": MODEL,
                    "max_tokens": 16,
                    "stream": false,
                    "messages": [{"role": "user", "content": "ping"}]
                }),
            )
            .await;
            assert_anthropic_status(response, status, error_type, message).await;
        }

        assert_error_metrics(
            &svc,
            &Endpoint::AnthropicMessages,
            &RequestType::Unary,
            &[
                (ErrorType::Cancelled, 1),
                (ErrorType::Unavailable, 1),
                (ErrorType::Overload, 2),
                (ErrorType::NotFound, 1),
                (ErrorType::Validation, 1),
                (ErrorType::Internal, 1),
            ],
        );

        assert_eq!(svc.engine.take_requests().await.len(), 7);
        svc.shutdown().await;
    })
    .await;
}

#[tokio::test]
#[serial]
async fn anthropic_handler_errors_record_classification_with_response() {
    temp_env::async_with_vars(BASE_ENV, async {
        let mut failures = Vec::new();
        for _ in 0..2 {
            failures.extend([
                dynamo_error(DynamoErrorType::ResourceExhausted, "too busy"),
                dynamo_error(DynamoErrorType::Unavailable, "no worker"),
                dynamo_error(DynamoErrorType::Cancelled, "client went away"),
                dynamo_error(DynamoErrorType::InvalidArgument, "bad backend input"),
                anyhow::anyhow!("generation failed"),
            ]);
        }
        let engine = Arc::new(ScriptedChatEngine::new(
            failures.into_iter().map(Err::<Script, _>),
        ));
        let svc = HarnessService::start_with_engine(engine).await;

        for (stream, request_type) in [(false, RequestType::Unary), (true, RequestType::Stream)] {
            for (status, error_type, message) in [
                (
                    reqwest::StatusCode::from_u16(529).unwrap(),
                    "overloaded_error",
                    "Service temporarily overloaded",
                ),
                (
                    reqwest::StatusCode::SERVICE_UNAVAILABLE,
                    "overloaded_error",
                    "Service temporarily unavailable",
                ),
                (
                    reqwest::StatusCode::from_u16(499).unwrap(),
                    "api_error",
                    "Request cancelled",
                ),
                (
                    reqwest::StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                    "Invalid request",
                ),
                (
                    reqwest::StatusCode::INTERNAL_SERVER_ERROR,
                    "api_error",
                    "Internal server error",
                ),
            ] {
                let response = post_json(
                    &svc,
                    "/v1/messages",
                    json!({
                        "model": MODEL,
                        "max_tokens": 16,
                        "stream": stream,
                        "messages": [{"role": "user", "content": "ping"}]
                    }),
                )
                .await;
                assert_anthropic_status(response, status, error_type, message).await;
            }

            let response = post_json(
                &svc,
                "/v1/messages",
                json!({
                    "model": "missing-model",
                    "max_tokens": 16,
                    "stream": stream,
                    "messages": [{"role": "user", "content": "ping"}]
                }),
            )
            .await;
            assert_anthropic_status(
                response,
                reqwest::StatusCode::NOT_FOUND,
                "not_found_error",
                "Model 'missing-model' not found",
            )
            .await;

            assert_error_metrics(
                &svc,
                &Endpoint::AnthropicMessages,
                &request_type,
                &[
                    (ErrorType::Overload, 1),
                    (ErrorType::Unavailable, 1),
                    (ErrorType::Cancelled, 1),
                    (ErrorType::Validation, 1),
                    (ErrorType::Internal, 1),
                ],
            );
            assert_error_metrics_for_model(
                &svc,
                UNKNOWN_METRIC_MODEL,
                &Endpoint::AnthropicMessages,
                &request_type,
                &[(ErrorType::NotFound, 1)],
            );
            assert_error_metrics(
                &svc,
                &Endpoint::AnthropicMessages,
                &request_type,
                &[(ErrorType::NotFound, 0)],
            );
        }

        assert_eq!(svc.engine.take_requests().await.len(), 10);
        svc.shutdown().await;
    })
    .await;
}

/// Invalid keys must produce HTTP 400 before generation, including before HTTP
/// 200 streaming headers can be committed for a streaming request.
#[tokio::test]
#[serial]
async fn completion_endpoints_reject_invalid_logit_bias_keys() {
    temp_env::async_with_vars(BASE_ENV, async {
        let svc = HarnessService::start(Vec::new()).await;

        for stream in [false, true] {
            for (path, body) in [
                (
                    "/v1/chat/completions",
                    json!({
                        "model": MODEL,
                        "messages": [{"role": "user", "content": "ping"}],
                        "stream": stream,
                        "logit_bias": {"not-a-token": 1.0}
                    }),
                ),
                (
                    "/v1/completions",
                    json!({
                        "model": MODEL,
                        "prompt": "ping",
                        "stream": stream,
                        "logit_bias": {"not-a-token": 1.0}
                    }),
                ),
            ] {
                let response = post_json(&svc, path, body).await;
                assert_openai_error(
                    response,
                    ExpectedError::Validation,
                    "Logit bias key must be a non-negative 32-bit integer token id",
                )
                .await;
            }
        }

        assert!(svc.engine.take_requests().await.is_empty());
        svc.shutdown().await;
    })
    .await;
}

// `reqwest::send` completes when response headers arrive. A 400 for `stream: true`
// proves converted-request validation ran before the HTTP 200 SSE response was committed.
#[tokio::test]
#[serial]
async fn converted_validation_errors_are_returned_before_streaming_headers() {
    temp_env::async_with_vars(BASE_ENV, async {
        let svc = HarnessService::start(Vec::new()).await;

        for (stream, field) in [
            (false, json!({"top_p": 2.0})),
            (true, json!({"temperature": 3.0})),
        ] {
            let mut body = json!({"model": MODEL, "input": "ping", "stream": stream});
            body.as_object_mut()
                .unwrap()
                .extend(field.as_object().unwrap().clone());
            let response = post_json(&svc, "/v1/responses", body).await;
            assert_openai_error(response, ExpectedError::Validation, "must be").await;
        }

        for (stream, field) in [
            (false, json!({"temperature": 3.0})),
            (true, json!({"top_p": 2.0})),
        ] {
            let mut body = json!({
                "model": MODEL,
                "max_tokens": 16,
                "stream": stream,
                "messages": [{"role": "user", "content": "ping"}]
            });
            body.as_object_mut()
                .unwrap()
                .extend(field.as_object().unwrap().clone());
            let response = post_json(&svc, "/v1/messages", body).await;
            assert_anthropic_error(response, ExpectedError::Validation, "must be").await;
        }

        for endpoint in [Endpoint::Responses, Endpoint::AnthropicMessages] {
            for request_type in [RequestType::Unary, RequestType::Stream] {
                assert_error_metrics(
                    &svc,
                    &endpoint,
                    &request_type,
                    &[(ErrorType::Validation, 1), (ErrorType::Internal, 0)],
                );
            }
        }

        assert!(svc.engine.take_requests().await.is_empty());
        svc.shutdown().await;
    })
    .await;
}

#[tokio::test]
#[serial]
async fn responses_auto_tool_choice_without_tools_reaches_backend() {
    temp_env::async_with_vars(BASE_ENV, async {
        let script = load_agent_fixture("text.sse").await.unwrap();
        let svc = HarnessService::start(vec![script; 4]).await;

        for tools in [Value::Null, json!([])] {
            for stream in [false, true] {
                let mut body = json!({
                    "model": MODEL,
                    "input": "ping",
                    "tool_choice": "auto",
                    "stream": stream
                });
                if !tools.is_null() {
                    body["tools"] = tools.clone();
                }
                let response = post_json(&svc, "/v1/responses", body).await;
                assert_eq!(response.status(), reqwest::StatusCode::OK);
                bounded(response.bytes()).await.unwrap();

                let requests = svc.engine.take_requests().await;
                assert_eq!(requests.len(), 1);
                assert!(requests[0].inner.tools.is_none());
                assert!(requests[0].inner.tool_choice.is_none());
            }
        }

        assert_eq!(svc.engine.remaining_scripts().await, 0);
        svc.shutdown().await;
    })
    .await;
}

#[tokio::test]
#[serial]
async fn responses_reject_empty_input_and_required_tool_choice_without_tools() {
    temp_env::async_with_vars(BASE_ENV, async {
        let svc = HarnessService::start(Vec::new()).await;

        for (body, message) in [
            (
                json!({"model": MODEL, "input": [], "max_tokens": 10}),
                "messages",
            ),
            (
                json!({
                    "model": MODEL,
                    "input": "ping",
                    "tools": [],
                    "tool_choice": "required"
                }),
                "tool_choice is \"required\"",
            ),
        ] {
            let response = post_json(&svc, "/v1/responses", body).await;
            assert_openai_error(response, ExpectedError::Validation, message).await;
        }

        assert!(svc.engine.take_requests().await.is_empty());
        svc.shutdown().await;
    })
    .await;
}

#[tokio::test]
#[serial]
async fn chat_completions_reject_empty_tools() {
    temp_env::async_with_vars(BASE_ENV, async {
        let svc = HarnessService::start(Vec::new()).await;

        let body = json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 10,
            "tools": []
        });
        let response = post_json(&svc, "/v1/chat/completions", body).await;
        assert_openai_error(
            response,
            ExpectedError::Validation,
            "tools array cannot be empty; omit the field entirely to disable tool use",
        )
        .await;

        assert!(svc.engine.take_requests().await.is_empty());
        svc.shutdown().await;
    })
    .await;
}

#[tokio::test]
#[serial]
async fn anthropic_content_validation_applies_to_messages_and_count_tokens() {
    temp_env::async_with_vars(BASE_ENV, async {
        let svc = HarnessService::start(Vec::new()).await;

        for (path, body, expected, message) in [
            (
                "/v1/messages",
                json!({
                    "model": MODEL,
                    "max_tokens": 10,
                    "stream": true,
                    "messages": [{"role": "user", "content": ["hello"]}]
                }),
                ExpectedError::Validation,
                "content blocks must be objects",
            ),
            (
                "/v1/messages",
                json!({
                    "model": MODEL,
                    "max_tokens": 10,
                    "messages": [{"role": "user", "content": []}]
                }),
                ExpectedError::Validation,
                "must contain at least one content block",
            ),
            (
                "/v1/messages/count_tokens",
                json!({
                    "model": MODEL,
                    "messages": [{"role": "user", "content": ["hello"]}]
                }),
                ExpectedError::Validation,
                "content blocks must be objects",
            ),
            (
                "/v1/messages",
                json!({
                    "model": MODEL,
                    "max_tokens": 16,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "future_block_type", "value": 1},
                            {"type": "text", "text": "ping"}
                        ]
                    }]
                }),
                ExpectedError::UnsupportedContent,
                "content block type \"future_block_type\"",
            ),
        ] {
            let response = post_json(&svc, path, body).await;
            assert_anthropic_error(response, expected, message).await;
        }

        for request_type in [RequestType::Unary, RequestType::Stream] {
            let not_implemented = match &request_type {
                RequestType::Unary => 1,
                RequestType::Stream => 0,
            };
            assert_error_metrics(
                &svc,
                &Endpoint::AnthropicMessages,
                &request_type,
                &[
                    (ErrorType::Validation, 1),
                    (ErrorType::NotImplemented, not_implemented),
                    (ErrorType::Internal, 0),
                ],
            );
        }

        assert!(svc.engine.take_requests().await.is_empty());
        svc.shutdown().await;
    })
    .await;
}

#[tokio::test]
#[serial]
async fn tool_name_limit_is_shared_across_protocols() {
    temp_env::async_with_vars(BASE_ENV, async {
        let valid_script = load_agent_fixture("text.sse").await.unwrap();
        let svc =
            HarnessService::start([valid_script.clone(), valid_script.clone(), valid_script]).await;

        let max_length_tool_name = "a".repeat(128);
        for (path, body, _) in tool_name_requests(&max_length_tool_name) {
            let response = post_json(&svc, path, body).await;
            assert_eq!(response.status(), reqwest::StatusCode::OK);
        }

        let too_long_tool_name = "a".repeat(129);
        for (path, body, anthropic) in tool_name_requests(&too_long_tool_name) {
            let response = post_json(&svc, path, body).await;
            if anthropic {
                assert_anthropic_error(response, ExpectedError::Validation, "128 character limit")
                    .await;
            } else {
                assert_openai_error(response, ExpectedError::Validation, "128 character limit")
                    .await;
            }
        }

        assert_eq!(svc.engine.take_requests().await.len(), 3);
        svc.shutdown().await;
    })
    .await;
}

fn run_immutable_sampling_http_child(enabled: bool) {
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "immutable_sampling_http_child",
            "--ignored",
            "--nocapture",
        ])
        .env(
            "DYN_KIMI_K3_IMMUTABLE_PARAMS",
            if enabled { "1" } else { "0" },
        )
        .output()
        .expect("start isolated immutable-sampling HTTP test");
    assert!(
        output.status.success(),
        "HTTP policy test (enabled={enabled}) failed:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

#[test]
fn immutable_sampling_http_enabled_returns_actionable_400_before_dispatch() {
    run_immutable_sampling_http_child(true);
}

#[test]
fn immutable_sampling_http_disabled_preserves_existing_parameter_ranges() {
    run_immutable_sampling_http_child(false);
}

fn immutable_sampling_requests(stream: bool) -> [(&'static str, Value); 3] {
    [
        (
            "/v1/chat/completions",
            json!({
                "model": MODEL, "messages": [{"role": "user", "content": "ping"}],
                "stream": stream, "temperature": 0.6
            }),
        ),
        (
            "/v1/responses",
            json!({
                "model": MODEL, "input": "ping", "stream": stream, "temperature": 0.6
            }),
        ),
        (
            "/v1/completions",
            json!({
                "model": MODEL, "prompt": "ping", "stream": stream, "temperature": 0.6
            }),
        ),
    ]
}

// Each subprocess owns the LazyLock gate from its first read. Other HTTP tests
// keep their original environment, including when the suite runs in parallel.
#[tokio::test]
#[ignore = "run by the two isolated-process HTTP policy tests"]
async fn immutable_sampling_http_child() {
    let enabled = std::env::var("DYN_KIMI_K3_IMMUTABLE_PARAMS").unwrap() == "1";
    tokio::time::timeout(Duration::from_secs(60), async {
        temp_env::async_with_vars(BASE_ENV, async {
            let script = load_agent_fixture("text.sse").await.unwrap();
            let svc = HarnessService::start(vec![script; 64]).await;
            let rejected = [
                ("top_p", json!(0.8), "fixed at 0.95"),
                ("presence_penalty", json!(0.5), "fixed at 0"),
                ("frequency_penalty", json!(-0.5), "fixed at 0"),
                ("n", json!(2), "fixed at 1"),
                ("temperature", json!(1.1), "between 0 and 1 (inclusive)"),
                ("temperature", json!(2.0), "between 0 and 1 (inclusive)"),
            ];
            for stream in [false, true] {
                for (path, body) in immutable_sampling_requests(stream) {
                    for (parameter, value, correction) in &rejected {
                        let mut body = body.clone();
                        body[*parameter] = value.clone();
                        if path == "/v1/completions" && !enabled {
                            // This harness registers only a chat engine. Verify the
                            // completion adapter directly for accepted requests.
                            let request: dynamo_llm::protocols::openai::completions::NvCreateCompletionRequest =
                                serde_json::from_value(body).unwrap();
                            dynamo_llm::engines::ValidateRequest::validate(&request).unwrap();
                            continue;
                        }
                        let response = post_json(&svc, path, body).await;
                        if enabled {
                            assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);
                            assert!(response.headers()["content-type"].to_str().unwrap().starts_with("application/json"));
                            let error: Value = bounded(response.json()).await.unwrap();
                            assert_eq!(error["code"], 400);
                            let message = error["message"].as_str().unwrap();
                            assert!(message.starts_with("Validation: "), "{error}");
                            assert!(message.contains(&format!("`{parameter}`")), "{error}");
                            assert!(message.contains(correction), "{error}");
                            assert!(
                                message.contains(&format!("received {}", value.as_f64().unwrap())),
                                "{error}"
                            );
                            if *parameter != "temperature" {
                                assert!(message.contains(&format!("Omit `{parameter}` or set it to")), "{error}");
                            }
                            assert!(svc.engine.take_requests().await.is_empty());
                        } else {
                            assert_eq!(response.status(), reqwest::StatusCode::OK);
                            bounded(response.bytes()).await.unwrap();
                            assert_eq!(svc.engine.take_requests().await.len(), 1);
                        }
                    }
                }
            }
            // Omission and every representative temperature in the allowed range
            // must pass through to generation, including streaming requests.
            for stream in [false, true] {
                for (path, body) in immutable_sampling_requests(stream).into_iter().take(2) {
                    for temperature in [None, Some(0.0), Some(0.2), Some(0.6), Some(1.0)] {
                        let mut body = body.clone();
                        body.as_object_mut().unwrap().remove("temperature");
                        if let Some(temperature) = temperature {
                            body["temperature"] = json!(temperature);
                            body["top_p"] = json!(0.95);
                            body["presence_penalty"] = json!(0);
                            body["frequency_penalty"] = json!(0);
                            body["n"] = json!(1);
                        }
                        let response = post_json(&svc, path, body).await;
                        assert_eq!(response.status(), reqwest::StatusCode::OK);
                        bounded(response.bytes()).await.unwrap();
                        let requests = svc.engine.take_requests().await;
                        assert_eq!(requests.len(), 1);
                        assert_eq!(requests[0].inner.temperature, temperature.map(|t| t as f32));
                    }
                }
            }
            svc.shutdown().await;
        }).await;
    }).await.expect("immutable-sampling HTTP checks exceeded sixty seconds");
}

// Vendor-format `thinking` objects: `type` defaults to `enabled`, and a nested
// `effort` reaches the renderer ahead of the top-level `reasoning_effort`.
#[tokio::test]
#[serial]
async fn vendor_thinking_object_defaults_and_effort_reach_engine() {
    temp_env::async_with_vars(BASE_ENV, async {
        let cases = [
            (json!({"thinking": {"keep": "all"}}), None),
            (
                json!({"thinking": {"type": "enabled", "keep": "all", "effort": "low"}, "reasoning_effort": "max"}),
                Some("low"),
            ),
            (json!({"reasoning_effort": "high"}), Some("high")),
        ];
        let script = load_agent_fixture("text.sse").await.unwrap();
        let svc = HarnessService::start(vec![script; cases.len() * 2]).await;
        for (extra, effort) in &cases {
            for stream in [false, true] {
                let mut body = json!({
                    "model": MODEL, "messages": [{"role": "user", "content": "ping"}],
                    "stream": stream
                });
                for (key, value) in extra.as_object().unwrap() {
                    body[key] = value.clone();
                }
                let response = post_json(&svc, "/v1/chat/completions", body).await;
                let status = response.status();
                let text = bounded(response.text()).await.unwrap();
                assert_eq!(status, reqwest::StatusCode::OK, "{extra} stream={stream}: {text}");
                let requests = svc.engine.take_requests().await;
                assert_eq!(requests.len(), 1);
                let args = requests[0].chat_template_args.as_ref().unwrap();
                assert_eq!(args.get("thinking"), Some(&json!(true)), "{extra}");
                assert_eq!(
                    args.get("reasoning_effort").and_then(Value::as_str),
                    *effort,
                    "{extra} stream={stream}"
                );
            }
        }
        for stream in [false, true] {
            let body = json!({
                "model": MODEL, "messages": [{"role": "user", "content": "ping"}],
                "stream": stream, "thinking": {"type": "enabled", "effort": "bogus"}
            });
            let response = post_json(&svc, "/v1/chat/completions", body).await;
            assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);
            let error: Value = bounded(response.json()).await.unwrap();
            let message = error["message"].as_str().unwrap();
            assert!(message.contains("`thinking.effort`"), "{error}");
            assert!(svc.engine.take_requests().await.is_empty());
        }
        svc.shutdown().await;
    })
    .await;
}

#[tokio::test]
#[serial]
async fn vendor_thinking_regression_malformed_fields_return400_before_dispatch() {
    tokio::time::timeout(Duration::from_secs(30), async {
        temp_env::async_with_vars(BASE_ENV, async {
            let script = load_agent_fixture("text.sse").await.unwrap();
            let svc = HarnessService::start(vec![script; 24]).await;
            for stream in [false, true] {
                for (thinking, field) in [
                    (json!({"type": null}), "thinking.type"),
                    (json!({"type": true}), "thinking.type"),
                    (json!({"effort": null}), "thinking.effort"),
                    (json!({"effort": false}), "thinking.effort"),
                    (json!({"effort": "none"}), "thinking.effort"),
                    (json!({"effort": "xhigh"}), "thinking.effort"),
                ] {
                    let response = post_json(
                        &svc,
                        "/v1/chat/completions",
                        json!({
                            "model": MODEL, "messages": [{"role": "user", "content": "ping"}],
                            "stream": stream, "thinking": thinking
                        }),
                    )
                    .await;
                    assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);
                    let error: Value = bounded(response.json()).await.unwrap();
                    assert!(
                        error["message"].as_str().unwrap().contains(field),
                        "{error}"
                    );
                    assert!(svc.engine.take_requests().await.is_empty());
                }
            }
            svc.shutdown().await;
        })
        .await;
    })
    .await
    .expect("bounded malformed thinking HTTP checks");
}

#[tokio::test]
#[serial]
async fn vendor_thinking_regression_precedence_reaches_worker() {
    tokio::time::timeout(Duration::from_secs(30), async {
        temp_env::async_with_vars(BASE_ENV, async {
            let script = load_agent_fixture("text.sse").await.unwrap();
            let svc = HarnessService::start(vec![script; 4]).await;
            for stream in [false, true] {
                let response = post_json(
                    &svc,
                    "/v1/chat/completions",
                    json!({
                        "model": MODEL, "messages": [{"role": "user", "content": "ping"}],
                        "stream": stream, "thinking": {"effort": "low"},
                        "reasoning_effort": "max",
                        "chat_template_kwargs": {"thinking_effort": "high"}
                    }),
                )
                .await;
                assert_eq!(response.status(), reqwest::StatusCode::OK);
                bounded(response.bytes()).await.unwrap();
                let requests = svc.engine.take_requests().await;
                assert_eq!(requests.len(), 1);
                let args = requests[0].chat_template_args.as_ref().unwrap();
                assert_eq!(args["thinking_effort"], json!("low"));
                assert_eq!(args["reasoning_effort"], json!("low"));
                assert!(requests[0].thinking.is_none());
            }
            svc.shutdown().await;
        })
        .await;
    })
    .await
    .expect("bounded thinking precedence HTTP checks");
}
