// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{fmt::Display, sync::LazyLock};

use dynamo_runtime::config::{
    env_is_truthy, environment_names::llm::DYN_IGNORE_OPENAI_FE_UNSUPPORTED_FIELDS,
};

use super::tools::{ToolChoiceError, validate_openai_tool_choice};

//
// Hyperparameter Contraints
//

/// Minimum allowed value for OpenAI's `temperature` sampling option
pub const MIN_TEMPERATURE: f32 = 0.0;
/// Maximum allowed value for OpenAI's `temperature` sampling option
pub const MAX_TEMPERATURE: f32 = 2.0;
/// Allowed range of values for OpenAI's `temperature`` sampling option
pub const TEMPERATURE_RANGE: (f32, f32) = (MIN_TEMPERATURE, MAX_TEMPERATURE);

/// Minimum allowed value for OpenAI's `top_p` sampling option
pub const MIN_TOP_P: f32 = 0.0;
/// Maximum allowed value for OpenAI's `top_p` sampling option
pub const MAX_TOP_P: f32 = 1.0;

/// Minimum allowed value for `min_p`
pub const MIN_MIN_P: f32 = 0.0;
/// Maximum allowed value for `min_p`
pub const MAX_MIN_P: f32 = 1.0;
/// Allowed range of values for `min_p`
pub const MIN_P_RANGE: (f32, f32) = (MIN_MIN_P, MAX_MIN_P);

/// Minimum allowed value for OpenAI's `frequency_penalty` sampling option
pub const MIN_FREQUENCY_PENALTY: f32 = -2.0;
/// Maximum allowed value for OpenAI's `frequency_penalty` sampling option
pub const MAX_FREQUENCY_PENALTY: f32 = 2.0;
/// Allowed range of values for OpenAI's `frequency_penalty` sampling option
pub const FREQUENCY_PENALTY_RANGE: (f32, f32) = (MIN_FREQUENCY_PENALTY, MAX_FREQUENCY_PENALTY);

/// Minimum allowed value for OpenAI's `presence_penalty` sampling option
pub const MIN_PRESENCE_PENALTY: f32 = -2.0;
/// Maximum allowed value for OpenAI's `presence_penalty` sampling option
pub const MAX_PRESENCE_PENALTY: f32 = 2.0;
/// Allowed range of values for OpenAI's `presence_penalty` sampling option
pub const PRESENCE_PENALTY_RANGE: (f32, f32) = (MIN_PRESENCE_PENALTY, MAX_PRESENCE_PENALTY);

/// Minimum allowed value for `length_penalty`
pub const MIN_LENGTH_PENALTY: f32 = -2.0;
/// Maximum allowed value for `length_penalty`
pub const MAX_LENGTH_PENALTY: f32 = 2.0;
/// Allowed range of values for `length_penalty`
pub const LENGTH_PENALTY_RANGE: (f32, f32) = (MIN_LENGTH_PENALTY, MAX_LENGTH_PENALTY);

/// Maximum allowed value for `top_logprobs`
pub const MIN_TOP_LOGPROBS: u8 = 0;
/// Maximum allowed value for `top_logprobs`
pub const MAX_TOP_LOGPROBS: u8 = 20;

/// Minimum allowed value for `logprobs` in completion requests
pub const MIN_LOGPROBS: u8 = 0;
/// Maximum allowed value for `logprobs` in completion requests
pub const MAX_LOGPROBS: u8 = 5;

/// Minimum allowed value for `n` (number of choices)
pub const MIN_N: u8 = 1;
/// Maximum allowed value for `n` (number of choices)
pub const MAX_N: u8 = 128;
/// Allowed range of values for `n` (number of choices)
pub const N_RANGE: (u8, u8) = (MIN_N, MAX_N);

/// Maximum allowed total number of choices (batch_size × n)
pub const MAX_TOTAL_CHOICES: usize = 128;

/// Minimum allowed value for OpenAI's `logit_bias` values
pub const MIN_LOGIT_BIAS: f32 = -100.0;
/// Maximum allowed value for OpenAI's `logit_bias` values
pub const MAX_LOGIT_BIAS: f32 = 100.0;

/// Minimum allowed value for `best_of`
pub const MIN_BEST_OF: u8 = 0;
/// Maximum allowed value for `best_of`
pub const MAX_BEST_OF: u8 = 20;
/// Allowed range of values for `best_of`
pub const BEST_OF_RANGE: (u8, u8) = (MIN_BEST_OF, MAX_BEST_OF);

/// Maximum allowed number of stop sequences.
pub const MAX_STOP_SEQUENCES: usize = 32;
/// Maximum allowed number of tools.
pub const MAX_TOOLS: usize = 1536;
// Metadata validation constants removed - we are no longer restricting the metadata field char limits
/// Both `/v1/messages` and `/v1/responses` define a 128-character tool-name limit.
pub const MAX_FUNCTION_NAME_LENGTH: usize = 128;
/// Moonshot's limit for Kimi K3 dynamic tool names declared on system messages
/// (`messages[].tools`), as enforced by the production K3 frontend.
pub const MAX_DYNAMIC_TOOL_NAME_LENGTH: usize = 96;
/// Minimum allowed value for `repetition_penalty`
pub const MIN_REPETITION_PENALTY: f32 = 0.0;
/// Maximum allowed value for `repetition_penalty`
pub const MAX_REPETITION_PENALTY: f32 = 2.0;

//
// Shared Fields
//

/// Extra-body fields accepted for backend-specific handling.
pub const PASSTHROUGH_EXTRA_FIELDS: &[&str] = &[
    "cache_salt",
    "stop_token_ids",
    "detokenize",
    "allowed_token_ids",
    "bad_words_token_ids",
    "logprob_token_ids",
];

static IGNORE_OPENAI_FE_UNSUPPORTED_FIELDS: LazyLock<bool> =
    LazyLock::new(|| env_is_truthy(DYN_IGNORE_OPENAI_FE_UNSUPPORTED_FIELDS));

/// True when this frontend serves Kimi K3 and should accept malformed
/// `tool_calls[].function.arguments` on prior assistant messages. Those args
/// are history context that the K3 chat path never re-parses (Moonshot accepts
/// them), but the Kimi-Vendor-Verifier `prompt_tokens` case
/// `k3_tool_bad_arguments` is a pure token-count test that expects HTTP 200.
/// When truthy, the JSON-object-string check on assistant tool-call arguments
/// is skipped. Set on the K3 deployment only; the Dynamo frontend is deployed
/// per-model, so leaving it unset keeps MiniMax-M3 (which needs the guard) and
/// every other model byte-identical to today's behavior.
static KIMI_K3_LENIENT_TOOL_ARGS: LazyLock<bool> =
    LazyLock::new(|| env_is_truthy("DYN_KIMI_K3_LENIENT_TOOL_ARGS"));

/// True when this frontend serves Kimi K3 and should enforce Moonshot's
/// immutable sampling-parameter contract: `temperature` in `[0, 1]`, `top_p`
/// exactly `0.95`, `presence_penalty` and `frequency_penalty` exactly `0`, and
/// `n` exactly `1`. Omitted parameters are always accepted. Any other value is
/// rejected with HTTP 400, matching the vendor API (Kimi-Vendor-Verifier
/// `tests/params`). Set on the K3 deployment only; unset leaves every other
/// model's accepted parameter ranges unchanged.
static KIMI_K3_IMMUTABLE_PARAMS: LazyLock<bool> =
    LazyLock::new(|| env_is_truthy("DYN_KIMI_K3_IMMUTABLE_PARAMS"));

/// True when this frontend serves Kimi K3 and should reject requests for token
/// logprobs with HTTP 400. K3 logprobs do not line up with the returned text:
/// they include reasoning and template tokens, and multi-token speculative
/// steps (DSpark) repeat a window. The K3 reference providers (Moonshot,
/// Baseten) do not offer logprobs either. Set on the K3 deployment only; unset
/// leaves every other model unchanged.
static KIMI_K3_DISABLE_LOGPROBS: LazyLock<bool> =
    LazyLock::new(|| env_is_truthy("DYN_KIMI_K3_DISABLE_LOGPROBS"));

/// Validates that no unsupported fields are present in the request.
///
/// Fields in `PASSTHROUGH_EXTRA_FIELDS` are validated by downstream handlers.
/// Other fields may be ignored and dropped when
/// `DYN_IGNORE_OPENAI_FE_UNSUPPORTED_FIELDS` is truthy.
pub fn validate_no_unsupported_fields(
    unsupported_fields: &std::collections::HashMap<String, serde_json::Value>,
) -> Result<(), anyhow::Error> {
    validate_no_unsupported_fields_with_ignore(
        unsupported_fields,
        *IGNORE_OPENAI_FE_UNSUPPORTED_FIELDS,
    )
}

fn validate_no_unsupported_fields_with_ignore(
    unsupported_fields: &std::collections::HashMap<String, serde_json::Value>,
    ignore_unsupported_fields: bool,
) -> Result<(), anyhow::Error> {
    let unknown: Vec<_> = unsupported_fields
        .keys()
        .filter(|k| !PASSTHROUGH_EXTRA_FIELDS.contains(&k.as_str()))
        .map(|s| format!("`{}`", s))
        .collect();
    if !unknown.is_empty() && !ignore_unsupported_fields {
        anyhow::bail!("Unsupported parameter(s): {}", unknown.join(", "));
    }
    if let Some(value) = unsupported_fields.get("cache_salt")
        && !value.is_string()
    {
        anyhow::bail!("`cache_salt` must be a string");
    }
    if let Some(value) = unsupported_fields.get("stop_token_ids") {
        serde_json::from_value::<Vec<crate::types::TokenIdType>>(value.clone())
            .map_err(|_| anyhow::anyhow!("`stop_token_ids` must be an array of token IDs"))?;
    }
    if let Some(value) = unsupported_fields.get("detokenize")
        && !value.is_boolean()
    {
        anyhow::bail!("`detokenize` must be a boolean");
    }
    if let Some(value) = unsupported_fields.get("allowed_token_ids") {
        serde_json::from_value::<Vec<crate::types::TokenIdType>>(value.clone())
            .map_err(|_| anyhow::anyhow!("`allowed_token_ids` must be an array of token IDs"))?;
    }
    if let Some(value) = unsupported_fields.get("bad_words_token_ids") {
        serde_json::from_value::<Vec<Vec<crate::types::TokenIdType>>>(value.clone()).map_err(
            |_| anyhow::anyhow!("`bad_words_token_ids` must be an array of token ID arrays"),
        )?;
    }
    if let Some(value) = unsupported_fields.get("logprob_token_ids") {
        serde_json::from_value::<Vec<crate::types::TokenIdType>>(value.clone())
            .map_err(|_| anyhow::anyhow!("`logprob_token_ids` must be an array of token IDs"))?;
    }
    Ok(())
}

/// Validates response_format for chat completions.
///
/// Dynamo currently supports translating:
/// - `{"type":"json_object"}` -> guided decoding JSON object schema
/// - `{"type":"json_schema","json_schema":{"schema": ...}}` -> guided decoding JSON schema
///
/// `{"type":"text"}` is accepted and means no structured constraint.
pub fn validate_response_format(
    response_format: &Option<dynamo_protocols::types::ResponseFormat>,
) -> Result<(), anyhow::Error> {
    use dynamo_protocols::types::ResponseFormat;

    let Some(fmt) = response_format else {
        return Ok(());
    };

    match fmt {
        ResponseFormat::Text => Ok(()),
        ResponseFormat::JsonObject => Ok(()),
        ResponseFormat::JsonSchema { json_schema } => {
            // Validate name field format
            if json_schema.name.is_empty() {
                anyhow::bail!("`response_format.json_schema.name` cannot be empty");
            }

            // Validate schema presence. `schema` is a non-optional
            // `serde_json::Value`, so an explicit `null` is the only way it
            // can still arrive empty.
            if json_schema.schema.is_null() {
                anyhow::bail!(
                    "`response_format.json_schema.schema` is required when `response_format.type` is `json_schema`"
                );
            }

            // Schema must be a JSON object — numbers, strings, arrays, and
            // booleans are not valid JSON Schema documents.
            if !json_schema.schema.is_object() {
                anyhow::bail!(
                    "`response_format.json_schema.schema` must be a JSON object, got {}",
                    match &json_schema.schema {
                        serde_json::Value::Array(_) => "array",
                        serde_json::Value::String(_) => "string",
                        serde_json::Value::Number(_) => "number",
                        serde_json::Value::Bool(_) => "boolean",
                        _ => "non-object",
                    }
                );
            }
            Ok(())
        }
    }
}

/// Validates the temperature parameter
pub fn validate_temperature(temperature: Option<f32>) -> Result<(), anyhow::Error> {
    if let Some(temp) = temperature
        && !(MIN_TEMPERATURE..=MAX_TEMPERATURE).contains(&temp)
    {
        anyhow::bail!(
            "Temperature must be between {} and {}, got {}",
            MIN_TEMPERATURE,
            MAX_TEMPERATURE,
            temp
        );
    }
    Ok(())
}

/// Validates the top_p parameter
pub fn validate_top_p(top_p: Option<f32>) -> Result<(), anyhow::Error> {
    if let Some(p) = top_p
        && !(p.is_finite() && p > MIN_TOP_P && p <= MAX_TOP_P)
    {
        anyhow::bail!(
            "Top_p must be between {} and {}, got {}",
            MIN_TOP_P,
            MAX_TOP_P,
            p
        );
    }
    Ok(())
}

// Validate top_k
pub fn validate_top_k(top_k: Option<i32>) -> Result<(), anyhow::Error> {
    match top_k {
        None => Ok(()),
        Some(k) if k >= -1 => Ok(()),
        _ => anyhow::bail!("Top_k must be null or greater than or equal to -1"),
    }
}

/// Validates mutual exclusion of temperature and top_p
pub fn validate_temperature_top_p_exclusion(
    temperature: Option<f32>,
    top_p: Option<f32>,
) -> Result<(), anyhow::Error> {
    match (temperature, top_p) {
        (Some(t), Some(p)) if t != 1.0 && p != 1.0 => {
            anyhow::bail!("Only one of temperature or top_p should be set (not both)");
        }
        _ => Ok(()),
    }
}

/// Validates frequency penalty parameter
pub fn validate_frequency_penalty(frequency_penalty: Option<f32>) -> Result<(), anyhow::Error> {
    if let Some(penalty) = frequency_penalty
        && !(MIN_FREQUENCY_PENALTY..=MAX_FREQUENCY_PENALTY).contains(&penalty)
    {
        anyhow::bail!(
            "Frequency penalty must be between {} and {}, got {}",
            MIN_FREQUENCY_PENALTY,
            MAX_FREQUENCY_PENALTY,
            penalty
        );
    }
    Ok(())
}

/// Validates presence penalty parameter
pub fn validate_presence_penalty(presence_penalty: Option<f32>) -> Result<(), anyhow::Error> {
    if let Some(penalty) = presence_penalty
        && !(MIN_PRESENCE_PENALTY..=MAX_PRESENCE_PENALTY).contains(&penalty)
    {
        anyhow::bail!(
            "Presence penalty must be between {} and {}, got {}",
            MIN_PRESENCE_PENALTY,
            MAX_PRESENCE_PENALTY,
            penalty
        );
    }
    Ok(())
}

pub fn validate_repetition_penalty(repetition_penalty: Option<f32>) -> Result<(), anyhow::Error> {
    // It should be greater than 0.0 and less than equal to 2.0
    if let Some(penalty) = repetition_penalty
        && (penalty <= MIN_REPETITION_PENALTY || penalty > MAX_REPETITION_PENALTY)
    {
        anyhow::bail!(
            "Repetition penalty must be between {} and {}, got {}",
            MIN_REPETITION_PENALTY,
            MAX_REPETITION_PENALTY,
            penalty
        );
    }
    Ok(())
}

/// Validates min_p parameter
pub fn validate_min_p(min_p: Option<f32>) -> Result<(), anyhow::Error> {
    if let Some(p) = min_p
        && !(MIN_MIN_P..=MAX_MIN_P).contains(&p)
    {
        anyhow::bail!(
            "Min_p must be between {} and {}, got {}",
            MIN_MIN_P,
            MAX_MIN_P,
            p
        );
    }
    Ok(())
}

/// Validates logit bias map
pub fn validate_logit_bias(
    logit_bias: &Option<std::collections::HashMap<String, serde_json::Value>>,
) -> Result<(), anyhow::Error> {
    let logit_bias = match logit_bias {
        Some(val) => val,
        None => return Ok(()),
    };

    for (token, bias_value) in logit_bias {
        let bias = bias_value.as_f64().ok_or_else(|| {
            anyhow::anyhow!(
                "Logit bias value for token '{}' must be a number, got {:?}",
                token,
                bias_value
            )
        })? as f32;

        if !(MIN_LOGIT_BIAS..=MAX_LOGIT_BIAS).contains(&bias) {
            anyhow::bail!(
                "Logit bias for token '{}' must be between {} and {}, got {}",
                token,
                MIN_LOGIT_BIAS,
                MAX_LOGIT_BIAS,
                bias
            );
        }
    }
    Ok(())
}

/// Validates n parameter (number of choices)
pub fn validate_n(n: Option<u8>) -> Result<(), anyhow::Error> {
    if let Some(value) = n
        && !(MIN_N..=MAX_N).contains(&value)
    {
        anyhow::bail!("n must be between {} and {}, got {}", MIN_N, MAX_N, value);
    }
    Ok(())
}

/// Validates total choices (batch_size × n) doesn't exceed maximum
pub fn validate_total_choices(batch_size: usize, n: u8) -> Result<(), anyhow::Error> {
    let total_choices = batch_size * (n as usize);
    if total_choices > MAX_TOTAL_CHOICES {
        anyhow::bail!(
            "Total choices (batch_size × n = {} × {} = {}) exceeds maximum of {}",
            batch_size,
            n,
            total_choices,
            MAX_TOTAL_CHOICES
        );
    }
    Ok(())
}

/// Validates n and temperature interaction
/// When n > 1, temperature must be > 0 to ensure diverse outputs
pub fn validate_n_with_temperature(
    n: Option<u8>,
    temperature: Option<f32>,
) -> Result<(), anyhow::Error> {
    if let Some(n_value) = n
        && n_value > 1
    {
        let temp = temperature.unwrap_or(1.0);
        if temp == 0.0 {
            anyhow::bail!(
                "When n > 1, temperature must be greater than 0 to ensure diverse outputs. Got n={}, temperature={}",
                n_value,
                temp
            );
        }
    }
    Ok(())
}

/// Validates model parameter
pub fn validate_model(model: &str) -> Result<(), anyhow::Error> {
    if model.trim().is_empty() {
        anyhow::bail!("Model cannot be empty");
    }
    Ok(())
}

/// Validates user parameter
pub fn validate_user(user: Option<&str>) -> Result<(), anyhow::Error> {
    if let Some(user_id) = user
        && user_id.trim().is_empty()
    {
        anyhow::bail!("User ID cannot be empty");
    }
    Ok(())
}

/// Validates stop sequences
pub fn validate_stop(stop: &Option<dynamo_protocols::types::Stop>) -> Result<(), anyhow::Error> {
    if let Some(stop_value) = stop {
        match stop_value {
            dynamo_protocols::types::Stop::String(s) => {
                if s.is_empty() {
                    anyhow::bail!("Stop sequence cannot be empty");
                }
            }
            dynamo_protocols::types::Stop::StringArray(sequences) => {
                if sequences.is_empty() {
                    anyhow::bail!("Stop sequences array cannot be empty");
                }
                if sequences.len() > MAX_STOP_SEQUENCES {
                    anyhow::bail!(
                        "Maximum of {} stop sequences allowed, got {}",
                        MAX_STOP_SEQUENCES,
                        sequences.len()
                    );
                }
                for (i, sequence) in sequences.iter().enumerate() {
                    if sequence.is_empty() {
                        anyhow::bail!("Stop sequence at index {} cannot be empty", i);
                    }
                }
            }
            dynamo_protocols::types::Stop::TokenIdArray(token_ids) => {
                if token_ids.is_empty() {
                    anyhow::bail!("Stop token IDs array cannot be empty");
                }
                if token_ids.len() > MAX_STOP_SEQUENCES {
                    anyhow::bail!(
                        "Maximum of {} stop token IDs allowed, got {}",
                        MAX_STOP_SEQUENCES,
                        token_ids.len()
                    );
                }
            }
        }
    }
    Ok(())
}

//
// Chat Completion Specific
//

/// Validates messages array
pub fn validate_messages(
    messages: &[dynamo_protocols::types::ChatCompletionRequestMessage],
) -> Result<(), anyhow::Error> {
    validate_messages_with_lenient_tool_args(messages, *KIMI_K3_LENIENT_TOOL_ARGS)
}

/// Inner form of [`validate_messages`] with the K3 lenient-tool-args gate passed
/// explicitly so tests can exercise both branches without touching the
/// process-global `KIMI_K3_LENIENT_TOOL_ARGS` `LazyLock`.
fn validate_messages_with_lenient_tool_args(
    messages: &[dynamo_protocols::types::ChatCompletionRequestMessage],
    lenient_tool_args: bool,
) -> Result<(), anyhow::Error> {
    if messages.is_empty() {
        anyhow::bail!("Messages array cannot be empty");
    }
    // Prior assistant tool-call messages in the request must carry arguments
    // as a JSON object string; reject bad non-empty shapes before chat-template rendering.
    // This was caught in MiniMax-M3 multi-turn tool-call tests.
    // Skipped when `DYN_KIMI_K3_LENIENT_TOOL_ARGS` is truthy: on the K3 path
    // these args are history context that is never re-parsed, so malformed
    // arguments must be accepted (Kimi-Vendor-Verifier `k3_tool_bad_arguments`).
    for (message_index, message) in messages.iter().enumerate() {
        if let dynamo_protocols::types::ChatCompletionRequestMessage::Tool(tool) = message
            && tool.tool_call_id.trim().is_empty()
        {
            anyhow::bail!("`messages[{message_index}].tool_call_id` cannot be empty");
        }
        if !lenient_tool_args
            && let dynamo_protocols::types::ChatCompletionRequestMessage::Assistant(assistant) =
                message
            && let Some(tool_calls) = &assistant.tool_calls
        {
            for (tool_call_index, tool_call) in tool_calls.iter().enumerate() {
                validate_json_object_string(
                    &tool_call.function.arguments,
                    format!(
                        "`messages[{message_index}].tool_calls[{tool_call_index}].function.arguments`"
                    ),
                )?;
            }
        }
    }
    Ok(())
}

fn validate_json_object_string(value: &str, field: String) -> Result<(), anyhow::Error> {
    if value.trim().is_empty() {
        return Ok(());
    }
    let parsed: serde_json::Value = serde_json::from_str(value)
        .map_err(|error| anyhow::anyhow!("{field} must be a valid JSON object string: {error}"))?;
    if !parsed.is_object() {
        anyhow::bail!("{field} must be a valid JSON object string");
    }
    Ok(())
}

/// Validates top_logprobs parameter
pub fn validate_top_logprobs(top_logprobs: Option<u8>) -> Result<(), anyhow::Error> {
    if let Some(value) = top_logprobs
        && !(0..=20).contains(&value)
    {
        anyhow::bail!(
            "Top_logprobs must be between 0 and {}, got {}",
            MAX_TOP_LOGPROBS,
            value
        );
    }
    Ok(())
}

/// Validates tools array
pub fn validate_tools(
    tools: &Option<&[dynamo_protocols::types::ChatCompletionTool]>,
) -> Result<(), anyhow::Error> {
    let tools = match tools {
        Some(val) => val,
        None => return Ok(()),
    };

    if tools.len() > MAX_TOOLS {
        anyhow::bail!(
            "Maximum of {} tools are supported, got {}",
            MAX_TOOLS,
            tools.len()
        );
    }

    for (i, tool) in tools.iter().enumerate() {
        if tool.function.name.len() > MAX_FUNCTION_NAME_LENGTH {
            anyhow::bail!(
                "Function name at index {} exceeds {} character limit, got {} characters",
                i,
                MAX_FUNCTION_NAME_LENGTH,
                tool.function.name.len()
            );
        }
        if tool.function.name.trim().is_empty() {
            anyhow::bail!("Function name at index {} cannot be empty", i);
        }
        if !tool
            .function
            .name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
        {
            anyhow::bail!(
                "Function at index {} has an invalid name: \"{}\". \
                 Only a-z, A-Z, 0-9, underscores, and dashes are allowed.",
                i,
                tool.function.name,
            );
        }
        if let Some(parameters) = &tool.function.parameters
            && !parameters.is_object()
        {
            anyhow::bail!(
                "Function parameters at index {} for \"{}\" must be a JSON Schema object",
                i,
                tool.function.name,
            );
        }
    }
    Ok(())
}

/// Moonshot's dynamic-tool name rule: an ASCII letter or underscore, then
/// ASCII letters, digits, underscores, or dashes. Stricter than the top-level
/// tool rule in [`validate_tools`], which also allows a leading digit.
fn is_valid_dynamic_tool_name(name: &str) -> bool {
    let mut bytes = name.bytes();
    matches!(bytes.next(), Some(b) if b.is_ascii_alphabetic() || b == b'_')
        && bytes.all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

/// Validates Kimi K3 dynamic tools declared on system messages
/// (`messages[].tools`).
///
/// `dynamo-protocols` already rejects `tools` on any other role and a
/// non-array value; the K3 renderer rejects entries that are not function
/// tools and a system message carrying both `content` and `tools`. This
/// enforces the name policy the vendor API applies to dynamic tools, which
/// neither of those layers covers.
pub fn validate_dynamic_system_tools<'a>(
    tools: impl Iterator<Item = &'a serde_json::Value>,
) -> Result<(), anyhow::Error> {
    for (i, tool) in tools.enumerate() {
        let Some(name) = dynamo_protocols::types::dynamic_tool_name(tool) else {
            anyhow::bail!("Dynamic tool at index {i} needs a string `function.name`");
        };
        if name.len() > MAX_DYNAMIC_TOOL_NAME_LENGTH {
            anyhow::bail!(
                "Dynamic tool name at index {i} exceeds {MAX_DYNAMIC_TOOL_NAME_LENGTH} character \
                 limit, got {} characters",
                name.len()
            );
        }
        if !is_valid_dynamic_tool_name(name) {
            anyhow::bail!(
                "Dynamic tool at index {i} has an invalid name: \"{name}\". Names start with a \
                 letter or underscore and contain only a-z, A-Z, 0-9, underscores, and dashes."
            );
        }
    }
    Ok(())
}

/// Validates that forced tool_choice requests refer to available tools.
pub fn validate_tool_choice(
    tool_choice: &Option<dynamo_protocols::types::ChatCompletionToolChoiceOption>,
    tools: Option<&[dynamo_protocols::types::ChatCompletionTool]>,
) -> Result<(), anyhow::Error> {
    use dynamo_protocols::types::ChatCompletionToolChoiceOption;

    match validate_openai_tool_choice(tool_choice.as_ref(), tools) {
        Ok(()) => Ok(()),
        Err(ToolChoiceError::EmptyTools) => {
            anyhow::bail!("tool_choice is \"required\" but tools is empty")
        }
        Err(ToolChoiceError::MissingTools) => match tool_choice {
            Some(ChatCompletionToolChoiceOption::Required) => {
                anyhow::bail!("tool_choice is \"required\" but tools is empty")
            }
            Some(ChatCompletionToolChoiceOption::Named(named)) => anyhow::bail!(
                "tool named \"{}\" in tool_choice is not present in tools",
                named.function.name
            ),
            _ => Err(ToolChoiceError::MissingTools.into()),
        },
        Err(ToolChoiceError::ToolNotFound(name)) => {
            anyhow::bail!("tool named \"{name}\" in tool_choice is not present in tools")
        }
        Err(error) => Err(error.into()),
    }
}

/// Validates reasoning effort parameter
pub fn validate_reasoning_effort(
    _reasoning_effort: &Option<dynamo_protocols::types::ReasoningEffort>,
) -> Result<(), anyhow::Error> {
    // TODO ADD HERE
    // ReasoningEffort is an enum, so if it exists, it's valid by definition
    // This function is here for completeness and future validation needs
    Ok(())
}

/// Validates service tier parameter
pub fn validate_service_tier(
    _service_tier: &Option<dynamo_protocols::types::ServiceTier>,
) -> Result<(), anyhow::Error> {
    // TODO ADD HERE
    // ServiceTier is an enum, so if it exists, it's valid by definition
    // This function is here for completeness and future validation needs
    Ok(())
}

//
// Completion Specific
//

/// Validates prompt
pub fn validate_prompt(prompt: &dynamo_protocols::types::Prompt) -> Result<(), anyhow::Error> {
    match prompt {
        dynamo_protocols::types::Prompt::String(s) => {
            if s.is_empty() {
                anyhow::bail!("Prompt string cannot be empty");
            }
        }
        dynamo_protocols::types::Prompt::StringArray(arr) => {
            if arr.is_empty() {
                anyhow::bail!("Prompt string array cannot be empty");
            }
            for (i, s) in arr.iter().enumerate() {
                if s.is_empty() {
                    anyhow::bail!("Prompt string at index {} cannot be empty", i);
                }
            }
        }
        dynamo_protocols::types::Prompt::IntegerArray(arr) => {
            if arr.is_empty() {
                anyhow::bail!("Prompt integer array cannot be empty");
            }
        }
        dynamo_protocols::types::Prompt::ArrayOfIntegerArray(arr) => {
            if arr.is_empty() {
                anyhow::bail!("Prompt array of integer arrays cannot be empty");
            }
            for (i, inner_arr) in arr.iter().enumerate() {
                if inner_arr.is_empty() {
                    anyhow::bail!("Prompt integer array at index {} cannot be empty", i);
                }
            }
        }
    }
    Ok(())
}

/// Validates prompt and prompt_embeds fields together.
///
/// This function consolidates all prompt-related validation:
/// - Ensures at least one of prompt or prompt_embeds is provided
/// - If prompt_embeds is provided, validates its format (base64, size limits)
/// - If prompt_embeds is NOT provided, validates that prompt is non-empty
///
/// Format for prompt_embeds: PyTorch tensor serialized with torch.save() and base64-encoded
pub fn validate_prompt_or_embeds(
    prompt: Option<&dynamo_protocols::types::Prompt>,
    prompt_embeds: Option<&str>,
) -> Result<(), anyhow::Error> {
    // Check that at least one is provided
    if prompt.is_none() && prompt_embeds.is_none() {
        anyhow::bail!("At least one of 'prompt' or 'prompt_embeds' must be provided");
    }

    // If prompt_embeds is provided, validate it
    if let Some(embeds) = prompt_embeds {
        validate_prompt_embeds_format(embeds)?;
    } else if let Some(p) = prompt {
        // Only validate prompt content if prompt_embeds is NOT provided
        // When embeddings are present, prompt can be empty/placeholder
        validate_prompt(p)?;
    }

    Ok(())
}

/// Validates prompt_embeds format (internal helper)
/// Format: PyTorch tensor serialized with torch.save() and base64-encoded
fn validate_prompt_embeds_format(embeds: &str) -> Result<(), anyhow::Error> {
    use base64::{Engine as _, engine::general_purpose};

    // Validate base64 encoding first
    let decoded = general_purpose::STANDARD
        .decode(embeds)
        .map_err(|_| anyhow::anyhow!("prompt_embeds must be valid base64-encoded data"))?;

    // Check minimum size on decoded bytes (100 bytes)
    const MIN_SIZE: usize = 100;
    if decoded.len() < MIN_SIZE {
        anyhow::bail!(
            "prompt_embeds decoded data must be at least {MIN_SIZE} bytes, got {} bytes",
            decoded.len()
        );
    }

    // Check maximum size on decoded bytes (10MB)
    const MAX_SIZE: usize = 10 * 1024 * 1024;
    if decoded.len() > MAX_SIZE {
        anyhow::bail!(
            "prompt_embeds decoded data exceeds maximum size of 10MB, got {} bytes",
            decoded.len()
        );
    }

    Ok(())
}

/// Validates prompt_embeds field (public wrapper for standalone validation)
/// Format: PyTorch tensor serialized with torch.save() and base64-encoded
pub fn validate_prompt_embeds(prompt_embeds: Option<&str>) -> Result<(), anyhow::Error> {
    if let Some(embeds) = prompt_embeds {
        validate_prompt_embeds_format(embeds)?;
    }
    Ok(())
}

/// Validates logprobs parameter (for completion requests)
pub fn validate_logprobs(logprobs: Option<u8>) -> Result<(), anyhow::Error> {
    if let Some(value) = logprobs
        && !(MIN_LOGPROBS..=MAX_LOGPROBS).contains(&value)
    {
        anyhow::bail!(
            "Logprobs must be between 0 and {}, got {}",
            MAX_LOGPROBS,
            value
        );
    }
    Ok(())
}

/// Validates best_of parameter
pub fn validate_best_of(best_of: Option<u8>, n: Option<u8>) -> Result<(), anyhow::Error> {
    if let Some(best_of_value) = best_of {
        if !(MIN_BEST_OF..=MAX_BEST_OF).contains(&best_of_value) {
            anyhow::bail!(
                "Best_of must be between 0 and {}, got {}",
                MAX_BEST_OF,
                best_of_value
            );
        }

        if let Some(n_value) = n
            && best_of_value < n_value
        {
            anyhow::bail!(
                "Best_of must be greater than or equal to n, got best_of={} and n={}",
                best_of_value,
                n_value
            );
        }
    }
    Ok(())
}

/// Validates suffix parameter
pub fn validate_suffix(suffix: Option<&str>) -> Result<(), anyhow::Error> {
    if let Some(suffix_str) = suffix {
        // Suffix can be empty, but if it's very long it might cause issues
        if suffix_str.len() > 10000 {
            anyhow::bail!("Suffix is too long, maximum 10000 characters");
        }
    }
    Ok(())
}

const MAX_OUTPUT_TOKENS: u32 = 1_048_576;

/// Environment variable that sets the deployment's maximum output length. It is
/// advertised as `max_output_tokens` on `/v1/models` and enforced on every
/// generation API. (The default for requests that omit `max_tokens` is the
/// engine's, e.g. vLLM `--override-generation-config '{"max_new_tokens": N}'`.)
pub const MAX_OUTPUT_TOKENS_ENV: &str = "DYN_MAX_OUTPUT_TOKENS";

fn parse_max_output_tokens(value: Option<&str>) -> Option<u32> {
    value
        .and_then(|v| v.trim().parse::<u32>().ok())
        .filter(|&v| v > 0)
        .map(|v| v.min(MAX_OUTPUT_TOKENS))
}

/// The configured maximum output length (`DYN_MAX_OUTPUT_TOKENS`), if set.
pub fn configured_max_output_tokens() -> Option<u32> {
    static CONFIGURED: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    *CONFIGURED.get_or_init(|| {
        parse_max_output_tokens(std::env::var(MAX_OUTPUT_TOKENS_ENV).ok().as_deref())
    })
}

/// Largest accepted `max_tokens` / `max_completion_tokens` / `max_output_tokens`.
pub fn max_output_tokens_limit() -> u32 {
    configured_max_output_tokens().unwrap_or(MAX_OUTPUT_TOKENS)
}

/// Validates max_tokens parameter
pub fn validate_max_tokens(max_tokens: Option<u32>) -> Result<(), anyhow::Error> {
    if let Some(tokens) = max_tokens
        && tokens == 0
    {
        anyhow::bail!("Max tokens must be greater than 0, got {}", tokens);
    }
    let limit = max_output_tokens_limit();
    if let Some(tokens) = max_tokens
        && tokens > limit
    {
        anyhow::bail!("Max tokens must not exceed {}, got {}", limit, tokens);
    }
    Ok(())
}

/// Validates max_completion_tokens parameter
pub fn validate_max_completion_tokens(
    max_completion_tokens: Option<u32>,
) -> Result<(), anyhow::Error> {
    if let Some(tokens) = max_completion_tokens
        && tokens == 0
    {
        anyhow::bail!(
            "Max completion tokens must be greater than 0, got {}",
            tokens
        );
    }
    let limit = max_output_tokens_limit();
    if let Some(tokens) = max_completion_tokens
        && tokens > limit
    {
        anyhow::bail!(
            "Max completion tokens must not exceed {}, got {}",
            limit,
            tokens
        );
    }
    Ok(())
}

//
// Helpers
//

pub fn validate_range<T>(value: Option<T>, range: &(T, T)) -> anyhow::Result<Option<T>>
where
    T: PartialOrd + Display,
{
    if value.is_none() {
        return Ok(None);
    }
    let value = value.unwrap();
    if value < range.0 || value > range.1 {
        anyhow::bail!("Value {} is out of range [{}, {}]", value, range.0, range.1);
    }
    Ok(Some(value))
}

/// A nested `chat_template` bypasses Dynamo's top-level rejection and is
/// promoted into the rendered template, so block it for every chat processor.
pub fn validate_chat_template_args(
    chat_template_args: Option<&std::collections::HashMap<String, serde_json::Value>>,
) -> Result<(), anyhow::Error> {
    if let Some(args) = chat_template_args
        && args.contains_key("chat_template")
    {
        anyhow::bail!("`chat_template` is not supported inside `chat_template_args`");
    }
    Ok(())
}

/// vLLM `ChatCompletionRequest` (`mode="before"`): both flags true on the raw
/// payload is an error. Omitted `add_generation_prompt` finalizes to true (vLLM
/// 0.27.1 and the Python frontend), so `continue_final_message=true` requires
/// an explicit `add_generation_prompt=false`. Generic HuggingFace continuation
/// is not assistant-only; last-message role is checked at truncation time.
pub fn validate_continue_final_message(
    add_generation_prompt: Option<bool>,
    continue_final_message: Option<bool>,
) -> Result<(), anyhow::Error> {
    if continue_final_message != Some(true) {
        return Ok(());
    }
    if add_generation_prompt.unwrap_or(true) {
        anyhow::bail!(
            "Cannot set both `continue_final_message` and `add_generation_prompt` to True."
        );
    }
    Ok(())
}

/// Chat-template generation controls are meaningless on `/v1/completions`.
/// Reject them so they are not silently ignored after landing on `CommonExt`.
pub fn validate_chat_only_generation_flags(
    add_generation_prompt: Option<bool>,
    continue_final_message: Option<bool>,
) -> Result<(), anyhow::Error> {
    if add_generation_prompt.is_some() || continue_final_message.is_some() {
        anyhow::bail!(
            "`add_generation_prompt` and `continue_final_message` are only supported on /v1/chat/completions"
        );
    }
    Ok(())
}

/// Reject a request for token logprobs when `DYN_KIMI_K3_DISABLE_LOGPROBS` is
/// truthy. See [`KIMI_K3_DISABLE_LOGPROBS`]. `requested_by` names the field that
/// asked for them; `None` when the request did not (`false`, `0` or absent).
pub fn validate_kimi_k3_no_logprobs(requested_by: Option<&str>) -> Result<(), anyhow::Error> {
    validate_kimi_k3_no_logprobs_with_gate(requested_by, *KIMI_K3_DISABLE_LOGPROBS)
}

fn validate_kimi_k3_no_logprobs_with_gate(
    requested_by: Option<&str>,
    enforce: bool,
) -> Result<(), anyhow::Error> {
    if enforce && let Some(field) = requested_by {
        anyhow::bail!(
            "`{field}` is not supported for this model: token logprobs are not available"
        );
    }
    Ok(())
}

/// Environment variable listing the media kinds a deployment rejects with HTTP 400
/// (comma-separated, any of `image`, `video`, `audio`). Kimi K3 serves text and
/// images only. A video or audio part would otherwise reach a worker, which fetches
/// and decodes it before vLLM's per-prompt limit check: the client gets a 500, and a
/// malformed H.264 stream can crash the worker process. Unset accepts every kind.
pub const REJECT_MEDIA_INPUTS_ENV: &str = "DYN_REJECT_MEDIA_INPUTS";

/// Media kinds a deployment turns away; see [`REJECT_MEDIA_INPUTS_ENV`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RejectedMediaInputs {
    pub image: bool,
    pub video: bool,
    pub audio: bool,
}

impl RejectedMediaInputs {
    fn parse(value: Option<&str>) -> Self {
        let mut rejected = Self::default();
        for kind in value.unwrap_or_default().split(',').map(str::trim) {
            match kind.to_ascii_lowercase().as_str() {
                "" => {}
                "image" => rejected.image = true,
                "video" => rejected.video = true,
                "audio" => rejected.audio = true,
                other => tracing::warn!(
                    "ignoring {other:?} in {REJECT_MEDIA_INPUTS_ENV}; expected image, video or audio"
                ),
            }
        }
        rejected
    }

    fn rejects(&self, kind: &str) -> bool {
        match kind {
            "image" => self.image,
            "video" => self.video,
            "audio" => self.audio,
            _ => false,
        }
    }
}

static REJECTED_MEDIA_INPUTS: LazyLock<RejectedMediaInputs> = LazyLock::new(|| {
    RejectedMediaInputs::parse(std::env::var(REJECT_MEDIA_INPUTS_ENV).ok().as_deref())
});

/// `(kind, content part type)` of a user content part that carries media.
fn user_media_part(
    part: &dynamo_protocols::types::ChatCompletionRequestUserMessageContentPart,
) -> Option<(&'static str, &'static str)> {
    use dynamo_protocols::types::ChatCompletionRequestUserMessageContentPart as Part;
    match part {
        Part::Text(_) => None,
        Part::ImageUrl(_) => Some(("image", "image_url")),
        Part::VideoUrl(_) => Some(("video", "video_url")),
        Part::AudioUrl(_) => Some(("audio", "audio_url")),
        Part::InputAudio(_) => Some(("audio", "input_audio")),
    }
}

/// `(kind, content part type)` of a tool content part that carries media.
fn tool_media_part(
    part: &dynamo_protocols::types::ChatCompletionRequestToolMessageContentPart,
) -> Option<(&'static str, &'static str)> {
    use dynamo_protocols::types::ChatCompletionRequestToolMessageContentPart as Part;
    match part {
        Part::Text(_) => None,
        Part::ImageUrl(_) => Some(("image", "image_url")),
        Part::VideoUrl(_) => Some(("video", "video_url")),
        Part::AudioUrl(_) => Some(("audio", "audio_url")),
    }
}

/// Reject media input (and audio output) the deployment does not serve, before the
/// request is routed to a worker. See [`REJECT_MEDIA_INPUTS_ENV`].
pub fn validate_media_inputs(
    messages: &[dynamo_protocols::types::ChatCompletionRequestMessage],
    audio_output_requested: bool,
) -> Result<(), anyhow::Error> {
    validate_media_inputs_with(messages, audio_output_requested, *REJECTED_MEDIA_INPUTS)
}

/// Inner form of [`validate_media_inputs`] with the rejected kinds passed explicitly
/// so tests do not depend on the process-global `LazyLock`.
fn validate_media_inputs_with(
    messages: &[dynamo_protocols::types::ChatCompletionRequestMessage],
    audio_output_requested: bool,
    rejected: RejectedMediaInputs,
) -> Result<(), anyhow::Error> {
    use dynamo_protocols::types::{
        ChatCompletionRequestMessage as Message, ChatCompletionRequestToolMessageContent as Tool,
        ChatCompletionRequestUserMessageContent as User,
    };
    if rejected == RejectedMediaInputs::default() {
        return Ok(());
    }
    if rejected.audio && audio_output_requested {
        anyhow::bail!("audio output is not supported for this model");
    }
    for (index, message) in messages.iter().enumerate() {
        let media = match message {
            Message::User(user) => match &user.content {
                User::Array(parts) => parts
                    .iter()
                    .filter_map(user_media_part)
                    .find(|(kind, _)| rejected.rejects(kind)),
                User::Text(_) => None,
            },
            Message::Tool(tool) => match &tool.content {
                Tool::Array(parts) => parts
                    .iter()
                    .filter_map(tool_media_part)
                    .find(|(kind, _)| rejected.rejects(kind)),
                Tool::Text(_) => None,
            },
            _ => None,
        };
        if let Some((kind, part)) = media {
            anyhow::bail!(
                "{kind} input is not supported for this model: remove the `{part}` content part \
                 from messages[{index}]"
            );
        }
    }
    Ok(())
}

/// Enforce Moonshot's immutable sampling-parameter contract for Kimi K3 when
/// `DYN_KIMI_K3_IMMUTABLE_PARAMS` is truthy. See [`KIMI_K3_IMMUTABLE_PARAMS`].
pub fn validate_kimi_k3_immutable_params(
    temperature: Option<f32>,
    top_p: Option<f32>,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    n: Option<u8>,
) -> Result<(), anyhow::Error> {
    validate_kimi_k3_immutable_params_with_gate(
        temperature,
        top_p,
        presence_penalty,
        frequency_penalty,
        n,
        *KIMI_K3_IMMUTABLE_PARAMS,
    )
}

/// Inner form of [`validate_kimi_k3_immutable_params`] with the gate passed
/// explicitly so tests can exercise both branches without touching the
/// process-global `LazyLock`.
fn validate_kimi_k3_immutable_params_with_gate(
    temperature: Option<f32>,
    top_p: Option<f32>,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    n: Option<u8>,
    enforce: bool,
) -> Result<(), anyhow::Error> {
    if !enforce {
        return Ok(());
    }
    if let Some(t) = temperature
        && !(0.0..=1.0).contains(&t)
    {
        anyhow::bail!(
            "`temperature` must be between 0.0 and 1.0 for Kimi K3; values outside that range are not supported (got {t})"
        );
    }
    if let Some(p) = top_p
        && p != 0.95
    {
        anyhow::bail!(
            "`top_p` is fixed at 0.95 for Kimi K3; overriding it is not supported (got {p})"
        );
    }
    if let Some(v) = presence_penalty
        && v != 0.0
    {
        anyhow::bail!(
            "`presence_penalty` is fixed at 0 for Kimi K3; overriding it is not supported (got {v})"
        );
    }
    if let Some(v) = frequency_penalty
        && v != 0.0
    {
        anyhow::bail!(
            "`frequency_penalty` is fixed at 0 for Kimi K3; overriding it is not supported (got {v})"
        );
    }
    if let Some(v) = n
        && v != 1
    {
        anyhow::bail!("`n` is fixed at 1 for Kimi K3; overriding it is not supported (got {v})");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use serde_json::json;

    use super::*;

    #[test]
    fn max_output_tokens_env_parsing() {
        assert_eq!(parse_max_output_tokens(Some("131072")), Some(131072));
        assert_eq!(parse_max_output_tokens(Some(" 131072 ")), Some(131072));
        assert_eq!(parse_max_output_tokens(Some("0")), None);
        assert_eq!(parse_max_output_tokens(Some("abc")), None);
        assert_eq!(parse_max_output_tokens(None), None);
        // never above the protocol ceiling
        assert_eq!(
            parse_max_output_tokens(Some("99999999")),
            Some(MAX_OUTPUT_TOKENS)
        );
    }

    #[test]
    fn kimi_k3_no_logprobs_gate() {
        let err = validate_kimi_k3_no_logprobs_with_gate(Some("top_logprobs"), true).unwrap_err();
        assert!(
            err.to_string()
                .contains("`top_logprobs` is not supported for this model"),
            "{err}"
        );
        // not requested, or the gate is off
        assert!(validate_kimi_k3_no_logprobs_with_gate(None, true).is_ok());
        assert!(validate_kimi_k3_no_logprobs_with_gate(Some("logprobs"), false).is_ok());
    }

    #[test]
    fn reject_media_inputs_env_parsing() {
        let video_audio = RejectedMediaInputs {
            image: false,
            video: true,
            audio: true,
        };
        assert_eq!(
            RejectedMediaInputs::parse(Some(" video, Audio ")),
            video_audio
        );
        assert_eq!(
            RejectedMediaInputs::parse(Some("video,,bogus,audio")),
            video_audio
        );
        assert_eq!(
            RejectedMediaInputs::parse(Some("")),
            RejectedMediaInputs::default()
        );
        assert_eq!(
            RejectedMediaInputs::parse(None),
            RejectedMediaInputs::default()
        );
    }

    fn media_messages(
        content: serde_json::Value,
    ) -> Vec<dynamo_protocols::types::ChatCompletionRequestMessage> {
        serde_json::from_value(json!([
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": content}
        ]))
        .unwrap()
    }

    #[test]
    fn reject_media_inputs_turns_away_video_and_audio() {
        let video_audio = RejectedMediaInputs::parse(Some("video,audio"));
        for (part, kind) in [
            (
                json!({"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}}),
                "video input",
            ),
            (
                json!({"type": "audio_url", "audio_url": {"url": "https://example.com/a.wav"}}),
                "audio input",
            ),
            (
                json!({"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}}),
                "audio input",
            ),
        ] {
            let messages = media_messages(json!([{"type": "text", "text": "describe"}, part]));
            let err = validate_media_inputs_with(&messages, false, video_audio).unwrap_err();
            let message = err.to_string();
            assert!(message.contains(kind), "{message}");
            assert!(message.contains("messages[1]"), "{message}");
        }

        // images and plain text pass; nothing is rejected when the gate is unset
        let image = media_messages(json!([
            {"type": "text", "text": "describe"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
        ]));
        assert!(validate_media_inputs_with(&image, false, video_audio).is_ok());
        assert!(
            validate_media_inputs_with(&media_messages(json!("hi")), false, video_audio).is_ok()
        );
        let video = media_messages(json!([
            {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}}
        ]));
        assert!(validate_media_inputs_with(&video, false, RejectedMediaInputs::default()).is_ok());
    }

    #[test]
    fn reject_media_inputs_covers_tool_results_and_audio_output() {
        let video_audio = RejectedMediaInputs::parse(Some("video,audio"));
        let messages: Vec<dynamo_protocols::types::ChatCompletionRequestMessage> =
            serde_json::from_value(json!([
                {"role": "user", "content": "run the tool"},
                {"role": "assistant", "content": null, "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {"name": "record", "arguments": "{}"}
                }]},
                {"role": "tool", "tool_call_id": "call_1", "content": [
                    {"type": "audio_url", "audio_url": {"url": "https://example.com/a.wav"}}
                ]}
            ]))
            .unwrap();
        let err = validate_media_inputs_with(&messages, false, video_audio).unwrap_err();
        assert!(err.to_string().contains("messages[2]"), "{err}");

        let text = media_messages(json!("hi"));
        let err = validate_media_inputs_with(&text, true, video_audio).unwrap_err();
        assert!(err.to_string().contains("audio output"), "{err}");
        assert!(
            validate_media_inputs_with(&text, true, RejectedMediaInputs::parse(Some("video")))
                .is_ok()
        );
    }

    fn unknown_fields() -> HashMap<String, serde_json::Value> {
        HashMap::from([("experimental_field".to_string(), json!("value"))])
    }

    #[test]
    fn validate_chat_template_args_rejects_nested_chat_template() {
        let args = HashMap::from([(
            "chat_template".to_string(),
            json!("{% for _ in range(10**9) %}x{% endfor %}"),
        )]);
        let err = validate_chat_template_args(Some(&args)).unwrap_err();
        assert!(err.to_string().contains("chat_template"));
    }

    #[test]
    fn validate_chat_template_args_accepts_other_keys() {
        let args = HashMap::from([("enable_thinking".to_string(), json!(false))]);
        validate_chat_template_args(Some(&args)).unwrap();
        validate_chat_template_args(None).unwrap();
    }

    #[test]
    fn validate_response_format_rejects_null_json_schema() {
        let response_format = serde_json::from_value(json!({
            "type": "json_schema",
            "json_schema": {
                "name": "test_schema",
                "schema": null
            }
        }))
        .unwrap();

        let err = validate_response_format(&Some(response_format)).unwrap_err();
        assert!(err.to_string().contains("schema` is required"));
    }

    #[test]
    fn validate_no_unsupported_fields_accepts_logprob_token_ids() {
        let fields = HashMap::from([("logprob_token_ids".to_string(), json!([14, 15]))]);
        validate_no_unsupported_fields_with_ignore(&fields, false).unwrap();
    }

    #[test]
    fn validate_no_unsupported_fields_rejects_malformed_logprob_token_ids() {
        for bad in [json!(["notanint"]), json!(7), json!([[1, 2]]), json!([-1])] {
            let fields = HashMap::from([("logprob_token_ids".to_string(), bad)]);
            let err = validate_no_unsupported_fields_with_ignore(&fields, false).unwrap_err();
            assert!(err.to_string().contains("must be an array of token IDs"));
        }
    }

    #[test]
    fn validate_no_unsupported_fields_rejects_unknown_fields_by_default() {
        let err = validate_no_unsupported_fields_with_ignore(&unknown_fields(), false).unwrap_err();
        assert!(err.to_string().contains("Unsupported parameter(s)"));
    }

    #[test]
    fn validate_no_unsupported_fields_ignores_unknown_fields_when_configured() {
        validate_no_unsupported_fields_with_ignore(&unknown_fields(), true).unwrap();
    }

    #[test]
    fn validate_no_unsupported_fields_still_validates_passthrough_fields_when_ignoring_unknowns() {
        let unsupported_fields = HashMap::from([
            ("experimental_field".to_string(), json!("value")),
            ("stop_token_ids".to_string(), json!("bad")),
        ]);

        let err =
            validate_no_unsupported_fields_with_ignore(&unsupported_fields, true).unwrap_err();
        assert!(err.to_string().contains("stop_token_ids"));
    }

    #[test]
    fn validate_top_p_rejects_zero() {
        let err = validate_top_p(Some(0.0)).unwrap_err();
        assert!(err.to_string().contains("Top_p"));
    }

    #[test]
    fn validate_top_p_accepts_valid_values() {
        validate_top_p(Some(0.1)).unwrap();
        validate_top_p(Some(1.0)).unwrap();
        validate_top_p(None).unwrap();
    }

    #[test]
    fn validate_response_format_rejects_non_object_schema() {
        let fmt = serde_json::from_value(json!({
            "type": "json_schema",
            "json_schema": { "name": "test", "schema": 42 }
        }))
        .unwrap();
        let err = validate_response_format(&Some(fmt)).unwrap_err();
        assert!(err.to_string().contains("must be a JSON object"));
    }

    #[test]
    fn validate_response_format_accepts_valid_object_schema() {
        let fmt = serde_json::from_value(json!({
            "type": "json_schema",
            "json_schema": {
                "name": "test",
                "schema": { "type": "object", "properties": {} }
            }
        }))
        .unwrap();
        validate_response_format(&Some(fmt)).unwrap();
    }

    /// Assistant message carrying a single tool call with the given `arguments`
    /// string (which may be malformed JSON).
    fn assistant_with_tool_args(
        arguments: &str,
    ) -> dynamo_protocols::types::ChatCompletionRequestMessage {
        dynamo_protocols::types::ChatCompletionRequestMessage::Assistant(
            dynamo_protocols::types::ChatCompletionRequestAssistantMessage {
                tool_calls: Some(vec![
                    dynamo_protocols::types::ChatCompletionMessageToolCall {
                        id: "call_1".to_string(),
                        r#type: dynamo_protocols::types::FunctionType::Function,
                        function: dynamo_protocols::types::FunctionCall {
                            name: "get_weather".to_string(),
                            arguments: arguments.to_string(),
                        },
                    },
                ]),
                ..Default::default()
            },
        )
    }

    #[test]
    fn validate_messages_rejects_malformed_tool_args_by_default() {
        // Truncated JSON object — the MiniMax-M3 guard must still reject this.
        let messages = vec![assistant_with_tool_args(r#"{"location":"x"#)];
        let err = validate_messages_with_lenient_tool_args(&messages, false).unwrap_err();
        assert!(
            err.to_string()
                .contains("must be a valid JSON object string"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn validate_messages_accepts_malformed_tool_args_when_lenient() {
        // With the K3 lenient gate on, the same malformed history args pass.
        let messages = vec![assistant_with_tool_args(r#"{"location":"x"#)];
        validate_messages_with_lenient_tool_args(&messages, true).unwrap();
    }

    #[test]
    fn validate_messages_still_checks_tool_call_id_when_lenient() {
        // The lenient gate only relaxes tool-arg JSON parsing; the empty
        // tool_call_id check on `tool` messages is unaffected.
        let messages = vec![dynamo_protocols::types::ChatCompletionRequestMessage::Tool(
            dynamo_protocols::types::ChatCompletionRequestToolMessage {
                tool_call_id: "  ".to_string(),
                ..Default::default()
            },
        )];
        let err = validate_messages_with_lenient_tool_args(&messages, true).unwrap_err();
        assert!(
            err.to_string().contains("tool_call_id"),
            "unexpected error: {err}"
        );
    }

    fn immutable(
        temperature: Option<f32>,
        top_p: Option<f32>,
        presence_penalty: Option<f32>,
        frequency_penalty: Option<f32>,
        n: Option<u8>,
        enforce: bool,
    ) -> Result<(), anyhow::Error> {
        validate_kimi_k3_immutable_params_with_gate(
            temperature,
            top_p,
            presence_penalty,
            frequency_penalty,
            n,
            enforce,
        )
    }

    #[test]
    fn kimi_k3_immutable_params_accepts_vendor_defaults() {
        // Mirrors Kimi-Vendor-Verifier tests/params IMMUTABLE_PARAMS defaults.
        immutable(None, None, None, None, None, true).unwrap();
        for temperature in [0.0, 0.6, 1.0] {
            immutable(Some(temperature), None, None, None, None, true).unwrap();
        }
        immutable(None, Some(0.95), None, None, None, true).unwrap();
        immutable(None, None, Some(0.0), None, None, true).unwrap();
        immutable(None, None, None, Some(0.0), None, true).unwrap();
        immutable(None, None, None, None, Some(1), true).unwrap();
    }

    #[test]
    fn kimi_k3_immutable_params_rejects_vendor_wrong_values() {
        // Mirrors Kimi-Vendor-Verifier tests/params IMMUTABLE_PARAMS wrong_value.
        for temperature in [1.1, 2.0, -0.1] {
            let err = immutable(Some(temperature), None, None, None, None, true).unwrap_err();
            assert!(err.to_string().contains("`temperature`"), "{err}");
        }
        let err = immutable(None, Some(0.8), None, None, None, true).unwrap_err();
        assert!(err.to_string().contains("`top_p`"), "{err}");
        // Clients and test harnesses key on the model name and "not supported".
        assert_eq!(
            err.to_string(),
            "`top_p` is fixed at 0.95 for Kimi K3; overriding it is not supported (got 0.8)"
        );
        let err = immutable(None, None, Some(0.5), None, None, true).unwrap_err();
        assert!(err.to_string().contains("`presence_penalty`"), "{err}");
        let err = immutable(None, None, None, Some(0.5), None, true).unwrap_err();
        assert!(err.to_string().contains("`frequency_penalty`"), "{err}");
        let err = immutable(None, None, None, None, Some(2), true).unwrap_err();
        assert!(err.to_string().contains("`n`"), "{err}");
    }

    #[test]
    fn kimi_k3_immutable_params_is_a_no_op_when_disabled() {
        immutable(Some(2.0), Some(0.8), Some(0.5), Some(0.5), Some(2), false).unwrap();
    }
}
