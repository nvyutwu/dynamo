// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Backend-published input limits. Missing metadata preserves legacy behavior;
//! a present map is authoritative, including omitted/zero modalities.

use dynamo_protocols::types::{
    ChatCompletionRequestMessage, ChatCompletionRequestToolMessageContent,
    ChatCompletionRequestToolMessageContentPart, ChatCompletionRequestUserMessageContent,
    ChatCompletionRequestUserMessageContentPart,
};
use std::collections::BTreeMap;

pub(crate) const INPUT_MODALITY_LIMITS_RUNTIME_KEY: &str = "input_modality_limits";
pub(crate) type InputModalityLimits = BTreeMap<String, u64>;

pub(crate) fn validate_input_modalities(
    messages: &[ChatCompletionRequestMessage],
    limits: Option<&InputModalityLimits>,
) -> Result<(), String> {
    let Some(limits) = limits else {
        return Ok(());
    };
    let mut counts = BTreeMap::<&str, u64>::new();
    for message in messages {
        match message {
            ChatCompletionRequestMessage::User(user) => {
                if let ChatCompletionRequestUserMessageContent::Array(parts) = &user.content {
                    for part in parts {
                        let modality = match part {
                            ChatCompletionRequestUserMessageContentPart::Text(_) => continue,
                            ChatCompletionRequestUserMessageContentPart::ImageUrl(_) => "image",
                            ChatCompletionRequestUserMessageContentPart::VideoUrl(_) => "video",
                            ChatCompletionRequestUserMessageContentPart::AudioUrl(_)
                            | ChatCompletionRequestUserMessageContentPart::InputAudio(_) => "audio",
                        };
                        *counts.entry(modality).or_default() += 1;
                    }
                }
            }
            ChatCompletionRequestMessage::Tool(tool) => {
                if let ChatCompletionRequestToolMessageContent::Array(parts) = &tool.content {
                    for part in parts {
                        let modality = match part {
                            ChatCompletionRequestToolMessageContentPart::Text(_) => continue,
                            ChatCompletionRequestToolMessageContentPart::ImageUrl(_) => "image",
                            ChatCompletionRequestToolMessageContentPart::VideoUrl(_) => "video",
                            ChatCompletionRequestToolMessageContentPart::AudioUrl(_) => "audio",
                        };
                        *counts.entry(modality).or_default() += 1;
                    }
                }
            }
            _ => {}
        }
    }
    for (modality, count) in counts {
        let limit = limits.get(modality).copied().unwrap_or(0);
        if limit == 0 {
            return Err(format!(
                "This deployment does not support {modality} inputs; the backend input modality limit is 0."
            ));
        }
        if count > limit {
            return Err(format!(
                "This deployment accepts at most {limit} {modality} inputs per prompt; received {count}."
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    fn user(parts: Value) -> Vec<ChatCompletionRequestMessage> {
        serde_json::from_value(json!([{"role":"user","content":parts}])).unwrap()
    }
    fn images_only() -> InputModalityLimits {
        BTreeMap::from([("image".into(), 2)])
    }
    #[test]
    fn text_and_supported_images_pass() {
        let messages = user(
            json!([{"type":"text","text":"describe"},{"type":"image_url","image_url":{"url":"https://example.invalid/image"}}]),
        );
        assert!(validate_input_modalities(&messages, Some(&images_only())).is_ok());
    }
    #[test]
    fn absent_legacy_contract_is_unknown_not_text_only() {
        let messages =
            user(json!([{"type":"video_url","video_url":{"url":"https://example.invalid/video"}}]));
        assert!(validate_input_modalities(&messages, None).is_ok());
    }
    #[test]
    fn missing_and_zero_modality_reject_url_and_inline_audio() {
        for part in [
            json!({"type":"audio_url","audio_url":{"url":"https://example.invalid/audio"}}),
            json!({"type":"input_audio","input_audio":{"data":"AAAA","format":"wav"}}),
        ] {
            let messages = user(json!([part]));
            assert!(
                validate_input_modalities(&messages, Some(&images_only()))
                    .unwrap_err()
                    .contains("does not support audio")
            );
            let explicit_zero = BTreeMap::from([("audio".into(), 0)]);
            assert!(validate_input_modalities(&messages, Some(&explicit_zero)).is_err());
        }
    }
    #[test]
    fn video_is_rejected_before_url_or_uuid_resolution() {
        for part in [
            json!({"type":"video_url","video_url":{"url":"https://example.invalid/video"}}),
            json!({"type":"video_url","uuid":"cached-video"}),
        ] {
            assert!(
                validate_input_modalities(&user(json!([part])), Some(&images_only()))
                    .unwrap_err()
                    .contains("does not support video")
            );
        }
    }
    #[test]
    fn present_empty_contract_rejects_images() {
        let messages =
            user(json!([{"type":"image_url","image_url":{"url":"https://example.invalid/image"}}]));
        assert!(
            validate_input_modalities(&messages, Some(&BTreeMap::new()))
                .unwrap_err()
                .contains("does not support image")
        );
    }
    #[test]
    fn count_limit_applies_across_conversation() {
        let mut messages =
            user(json!([{"type":"image_url","uuid":"one"},{"type":"image_url","uuid":"two"}]));
        messages.extend(user(json!([{"type":"image_url","uuid":"three"}])));
        assert!(
            validate_input_modalities(&messages, Some(&images_only()))
                .unwrap_err()
                .contains("received 3")
        );
    }
    #[test]
    fn tool_output_media_is_checked() {
        let messages: Vec<ChatCompletionRequestMessage> = serde_json::from_value(json!([{"role":"tool","tool_call_id":"call1","content":[{"type":"audio_url","audio_url":{"url":"https://example.invalid/audio"}}]}])).unwrap();
        assert!(
            validate_input_modalities(&messages, Some(&images_only()))
                .unwrap_err()
                .contains("audio")
        );
    }
    #[test]
    fn supported_audio_and_video_stay_supported() {
        let limits = BTreeMap::from([("audio".into(), 2), ("video".into(), 1)]);
        let messages = user(
            json!([{"type":"input_audio","input_audio":{"data":"AAAA","format":"wav"}},{"type":"video_url","uuid":"video"}]),
        );
        assert!(validate_input_modalities(&messages, Some(&limits)).is_ok());
    }
    #[test]
    fn malformed_published_contract_is_not_silently_unknown() {
        for value in [
            json!(null),
            json!([]),
            json!({"image":-1}),
            json!({"image":"2"}),
            json!({"image":1.5}),
        ] {
            assert!(serde_json::from_value::<InputModalityLimits>(value).is_err());
        }
    }
}
