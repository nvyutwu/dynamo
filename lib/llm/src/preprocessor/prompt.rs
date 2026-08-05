// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prompt formatting (lib/llm side).
//!
//! The reusable chat-template / prompt-formatting engine lives in the
//! standalone, runtime-free [`dynamo_renderer`] crate. This module holds only the
//! lib/llm-local glue that can't live there:
//!   * implements [`OAIChatLikeRequest`] for Dynamo's `Nv*` request wrappers,
//!   * keeps media-IO config off the rendering trait via [`MediaRequestExt`]
//!     (so `dynamo_renderer` need not depend on the media module),
//!   * adapts a [`ModelDeploymentCard`] into a [`PromptFormatter`]
//!     ([`prompt_formatter_from_mdc`]).
//!
//! Everything else imports from `dynamo_renderer` directly.

use std::sync::LazyLock;

use anyhow::{Context, Result};
use minijinja::value::Value;

use dynamo_renderer::{
    ChatTemplate, ChatTemplateValue, ContextMixins, OAIChatLikeRequest, PromptFormatter,
    PromptInput, TextInput, TokenInput, deepseek_formatter_for, kimi_k3_formatter_for,
    may_be_fix_tool_schema,
};
use dynamo_runtime::config::env_is_truthy;

/// Kimi K3 vendor-parity opt-in: when truthy, strip an empty/null tool
/// `description` from the tool declaration. The upstream `may_be_fix_tool_schema`
/// backfills `"description": ""` for tools declared without one (other chat
/// templates concatenate it unconditionally and would fail on a null), but
/// Moonshot's K3 reference omits an absent description — emitting `""` adds
/// tokens and breaks `usage.prompt_tokens` parity. The flag is the K3 signal
/// (this runs in the shared `tools()` path), so it is off by default and every
/// other model is byte-identical to today.
static STRIP_EMPTY_TOOL_DESCRIPTION: LazyLock<bool> =
    LazyLock::new(|| env_is_truthy("DYN_KIMI_K3_STRIP_EMPTY_TOOL_DESCRIPTION"));

use crate::model_card::{ModelDeploymentCard, PromptFormatterArtifact};
use crate::preprocessor::media::MediaDecoder;
use crate::protocols::openai::{
    chat_completions::NvCreateChatCompletionRequest, completions::NvCreateCompletionRequest,
};

/// lib/llm-local extension carrying multimodal media-IO config. Kept off
/// [`OAIChatLikeRequest`] so `dynamo_renderer` stays free of the media module;
/// the multimodal preprocessing path bounds on `OAIChatLikeRequest + MediaRequestExt`.
pub trait MediaRequestExt {
    fn media_io_kwargs(&self) -> Option<&MediaDecoder>;
}

/// Post-`may_be_fix_tool_schema` policy: when `strip` is true, drop empty/null
/// tool descriptions (K3 vendor parity); otherwise return the fixed tools Value
/// untouched (the default for every model). `may_be_fix_tool_schema` yields a
/// minijinja Value, so this round-trips through serde_json to edit the
/// post-backfill data. Pure (flag passed in) so it is unit-testable without the
/// process-global env read.
fn apply_k3_tool_description_policy(fixed: Value, strip: bool) -> Value {
    if !strip {
        return fixed;
    }
    let Ok(mut fixed_json) = serde_json::to_value(&fixed) else {
        return fixed;
    };
    strip_empty_tool_descriptions(&mut fixed_json);
    Value::from_serialize(&fixed_json)
}

/// Remove a tool `description` that is empty (`""`) or null from each tool in a
/// tools array, in place. Real, non-empty descriptions are kept verbatim and no
/// other field is touched, so upstream key ordering is preserved. Handles both
/// the Chat Completions shape (`description` under `function`) and the Responses
/// tool shape (top-level `description`). See [`STRIP_EMPTY_TOOL_DESCRIPTION`].
fn strip_empty_tool_descriptions(tools: &mut serde_json::Value) {
    let Some(array) = tools.as_array_mut() else {
        return;
    };
    for tool in array {
        if let Some(function) = tool
            .get_mut("function")
            .and_then(serde_json::Value::as_object_mut)
        {
            remove_empty_description(function);
        }
        if let Some(object) = tool.as_object_mut() {
            remove_empty_description(object);
        }
    }
}

fn remove_empty_description(object: &mut serde_json::Map<String, serde_json::Value>) {
    let is_empty = object
        .get("description")
        .is_some_and(|value| value.is_null() || value.as_str() == Some(""));
    if is_empty {
        object.remove("description");
    }
}

impl OAIChatLikeRequest for NvCreateChatCompletionRequest {
    fn model(&self) -> String {
        self.inner.model.clone()
    }

    fn messages(&self) -> Value {
        let messages_json = serde_json::to_value(&self.inner.messages).unwrap();
        Value::from_serialize(&messages_json)
    }

    fn typed_messages(&self) -> Option<&[dynamo_protocols::types::ChatCompletionRequestMessage]> {
        Some(self.inner.messages.as_slice())
    }

    fn tools(&self) -> Option<Value> {
        if self.inner.tools.is_none() {
            return None;
        }
        // Try to fix the tool schema if it is missing type and properties. This
        // also backfills `description: ""` for tools declared without one.
        let fixed = may_be_fix_tool_schema(serde_json::to_value(&self.inner.tools).unwrap())?;
        Some(apply_k3_tool_description_policy(
            fixed,
            *STRIP_EMPTY_TOOL_DESCRIPTION,
        ))
    }

    fn tool_choice(&self) -> Option<Value> {
        if self.inner.tool_choice.is_none() {
            None
        } else {
            Some(Value::from_serialize(&self.inner.tool_choice))
        }
    }

    fn response_format(&self) -> Option<Value> {
        self.inner
            .response_format
            .as_ref()
            .map(Value::from_serialize)
    }

    fn should_add_generation_prompt(&self) -> bool {
        // Using vLLM default behavior
        true
    }

    fn extract_text(&self) -> Option<TextInput> {
        Some(TextInput::Single(String::new()))
    }

    fn chat_template_args(&self) -> Option<&std::collections::HashMap<String, serde_json::Value>> {
        self.chat_template_args.as_ref()
    }

    fn mm_processor_kwargs(&self) -> Option<&serde_json::Value> {
        self.inner.mm_processor_kwargs.as_ref()
    }
}

impl MediaRequestExt for NvCreateChatCompletionRequest {
    fn media_io_kwargs(&self) -> Option<&MediaDecoder> {
        self.media_io_kwargs.as_ref()
    }
}

impl OAIChatLikeRequest for NvCreateCompletionRequest {
    fn model(&self) -> String {
        self.inner.model.clone()
    }
    fn messages(&self) -> minijinja::value::Value {
        let message = dynamo_protocols::types::ChatCompletionRequestMessage::User(
            dynamo_protocols::types::ChatCompletionRequestUserMessage {
                content: dynamo_protocols::types::ChatCompletionRequestUserMessageContent::Text(
                    crate::protocols::openai::completions::prompt_to_string(&self.inner.prompt),
                ),
                name: None,
            },
        );

        minijinja::value::Value::from_serialize(vec![message])
    }

    fn should_add_generation_prompt(&self) -> bool {
        true
    }

    fn prompt_input_type(&self) -> PromptInput {
        match &self.inner.prompt {
            dynamo_protocols::types::Prompt::IntegerArray(_) => {
                PromptInput::Tokens(TokenInput::Single(vec![]))
            }
            dynamo_protocols::types::Prompt::ArrayOfIntegerArray(_) => {
                PromptInput::Tokens(TokenInput::Batch(vec![]))
            }
            dynamo_protocols::types::Prompt::String(_) => {
                PromptInput::Text(TextInput::Single(String::new()))
            }
            dynamo_protocols::types::Prompt::StringArray(_) => {
                PromptInput::Text(TextInput::Batch(vec![]))
            }
        }
    }

    fn extract_tokens(&self) -> Option<TokenInput> {
        match &self.inner.prompt {
            dynamo_protocols::types::Prompt::IntegerArray(tokens) => {
                Some(TokenInput::Single(tokens.clone()))
            }
            dynamo_protocols::types::Prompt::ArrayOfIntegerArray(arrays) => {
                Some(TokenInput::Batch(arrays.clone()))
            }
            _ => None,
        }
    }

    fn extract_text(&self) -> Option<TextInput> {
        match &self.inner.prompt {
            dynamo_protocols::types::Prompt::String(text) => {
                Some(TextInput::Single(text.to_string()))
            }
            dynamo_protocols::types::Prompt::StringArray(texts) => {
                Some(TextInput::Batch(texts.to_vec()))
            }
            _ => None,
        }
    }
}

impl MediaRequestExt for NvCreateCompletionRequest {
    fn media_io_kwargs(&self) -> Option<&MediaDecoder> {
        None
    }
}

/// Build a [`PromptFormatter`] from a [`ModelDeploymentCard`].
///
/// Model families whose HF repos ship no Jinja `chat_template` get a native
/// Rust formatter; everything else loads the
/// HF `tokenizer_config.json` template (and any separate chat-template file)
/// and builds via [`PromptFormatter::from_parts`].
pub fn prompt_formatter_from_mdc(mdc: &ModelDeploymentCard) -> Result<PromptFormatter> {
    // Prefer the authoritative `model_type` from config.json — it's set by the
    // model author and survives any `--served-model-name` rename. An empty
    // `model_type` carries no signal — normalize to `None` so the display-name
    // fallback still runs.
    let model_type_lower = mdc
        .model_info
        .as_ref()
        .and_then(|info| info.get_model_info().ok())
        .map(|info| info.model_type().to_lowercase())
        .filter(|s| !s.is_empty());
    let display_name_lower = mdc.display_name.to_lowercase();

    if let Some(formatter) = kimi_k3_formatter_for(
        &model_type_lower,
        &display_name_lower,
        mdc.runtime_config.exclude_tools_when_tool_choice_none,
    ) {
        return Ok(formatter);
    }

    if let Some(formatter) = deepseek_formatter_for(&model_type_lower, &display_name_lower) {
        return Ok(formatter);
    }

    match mdc
        .prompt_formatter
        .as_ref()
        .ok_or(anyhow::anyhow!("MDC does not contain a prompt formatter"))?
    {
        PromptFormatterArtifact::HfTokenizerConfigJson(checked_file) => {
            let Some(file) = checked_file.path() else {
                anyhow::bail!(
                    "HfTokenizerConfigJson for {} is a URL, cannot load",
                    mdc.display_name
                );
            };
            let contents = std::fs::read_to_string(file).with_context(|| {
                format!(
                    "prompt_formatter_from_mdc fs:read_to_string '{}'",
                    file.display()
                )
            })?;
            let mut config: ChatTemplate = serde_json::from_str(&contents).inspect_err(|err| {
                crate::log_json_err(&file.display().to_string(), &contents, err)
            })?;

            // Some HF models (e.g. Llama-4-Maverick) store the chat template in a
            // separate file, or it may be a custom template provided via CLI flag.
            match mdc.chat_template_file.as_ref() {
                Some(PromptFormatterArtifact::HfChatTemplateJinja {
                    file: checked_file, ..
                }) => {
                    let Some(path) = checked_file.path() else {
                        anyhow::bail!(
                            "HfChatTemplateJinja for {} is a URL, cannot load",
                            mdc.display_name
                        );
                    };
                    let chat_template = std::fs::read_to_string(path)
                        .with_context(|| format!("fs:read_to_string '{}'", path.display()))?;
                    config.chat_template = Some(ChatTemplateValue(either::Left(chat_template)));
                }
                Some(PromptFormatterArtifact::HfChatTemplateJson {
                    file: checked_file, ..
                }) => {
                    let Some(path) = checked_file.path() else {
                        anyhow::bail!(
                            "HfChatTemplateJson for {} is a URL, cannot load",
                            mdc.display_name
                        );
                    };
                    let raw = std::fs::read_to_string(path)
                        .with_context(|| format!("fs:read_to_string '{}'", path.display()))?;
                    let wrapper: serde_json::Value = serde_json::from_str(&raw)
                        .with_context(|| format!("Failed to parse '{}' as JSON", path.display()))?;
                    let field = wrapper.get("chat_template").ok_or_else(|| {
                        anyhow::anyhow!(
                            "'{}' does not contain a 'chat_template' field",
                            path.display()
                        )
                    })?;
                    let value = serde_json::from_value::<ChatTemplateValue>(field.clone())
                        .with_context(|| {
                            format!(
                                "Failed to deserialize 'chat_template' in '{}'",
                                path.display()
                            )
                        })?;
                    config.chat_template = Some(value);
                }
                _ => {}
            }
            PromptFormatter::from_parts(
                config,
                mdc.prompt_context
                    .clone()
                    .map_or(ContextMixins::default(), |x| ContextMixins::new(&x)),
                mdc.runtime_config.exclude_tools_when_tool_choice_none,
            )
        }
        PromptFormatterArtifact::HfChatTemplateJinja { .. }
        | PromptFormatterArtifact::HfChatTemplateJson { .. } => Err(anyhow::anyhow!(
            "prompt_formatter should not have type HfChatTemplate*"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn as_json(value: Value) -> serde_json::Value {
        serde_json::to_value(&value).unwrap()
    }

    /// Shape as produced by `may_be_fix_tool_schema`: a tool declared without a
    /// description has `"description": ""` backfilled.
    fn backfilled_tools() -> serde_json::Value {
        json!([
            {
                "type": "function",
                "function": {
                    "name": "no_desc",
                    "description": "",
                    "parameters": {"type": "object", "properties": {}}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "null_desc",
                    "description": null,
                    "parameters": {"type": "object", "properties": {}}
                }
            }
        ])
    }

    // (a) flag on -> empty/null descriptions dropped; name/parameters/type kept.
    #[test]
    fn policy_on_strips_empty_and_null_descriptions_keeping_other_fields() {
        let fixed = Value::from_serialize(&backfilled_tools());
        let out = as_json(apply_k3_tool_description_policy(fixed, true));
        let arr = out.as_array().unwrap();
        for tool in arr {
            let function = tool.get("function").unwrap();
            assert!(
                function.get("description").is_none(),
                "empty/null description must be dropped: {tool}"
            );
            assert!(function.get("name").is_some(), "name retained");
            assert!(function.get("parameters").is_some(), "parameters retained");
            assert_eq!(tool.get("type").unwrap(), "function");
        }
        assert_eq!(arr[0]["function"]["name"], "no_desc");
        assert_eq!(arr[1]["function"]["name"], "null_desc");
    }

    // (b) flag on -> a real description is kept verbatim.
    #[test]
    fn policy_on_keeps_real_description() {
        let tools = json!([{
            "type": "function",
            "function": {
                "name": "weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {}}
            }
        }]);
        let out = as_json(apply_k3_tool_description_policy(
            Value::from_serialize(&tools),
            true,
        ));
        assert_eq!(out[0]["function"]["description"], "Get weather");
    }

    // (c) flag off -> byte-identical to today; the backfilled empty description
    // is preserved (no stripping).
    #[test]
    fn policy_off_preserves_empty_description() {
        let out = as_json(apply_k3_tool_description_policy(
            Value::from_serialize(&backfilled_tools()),
            false,
        ));
        assert_eq!(out[0]["function"]["description"], "");
    }

    // Direct strip: null description dropped, top-level (Responses shape) covered.
    #[test]
    fn strip_drops_null_and_top_level_description() {
        let mut tools = json!([
            {"type": "function", "function": {"name": "a", "description": null, "parameters": {}}},
            {"type": "function", "name": "b", "description": "", "parameters": {}}
        ]);
        strip_empty_tool_descriptions(&mut tools);
        assert!(tools[0]["function"].get("description").is_none());
        assert_eq!(tools[0]["function"]["name"], "a");
        assert!(tools[1].get("description").is_none(), "top-level empty description dropped");
        assert_eq!(tools[1]["name"], "b");
    }
}
