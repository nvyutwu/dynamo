// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Backend
//!
//! An [`Backend`] is the final stage of the pipeline. It represents the execution of the LLM
//! on some processing hardware.
//!
//! At minimum, the Backend is split into two components, the [`Backend`] itself and a downstream [`ExecutionContext`].
//!
//! The [`ExecutionContext`] can be thought of as the core driver of the forward pass, whereas the [`Backend`] is the
//! manager of all resources and concurrent tasks surrounding the LLM execution context / forward pass.
//!
//! For almost every known scenario, detokenization and initial post processing must happen in the Backend.
//! Further post-processing can happen in the response stream. One example is the jailing mechanism for partial
//! hidden stop condition matches, which can be handled in the response stream rather than the backend.

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    time::Instant,
};

use anyhow::Result;
use futures::stream::{self, StreamExt};

use crate::model_card::ModelDeploymentCard;
use dynamo_runtime::dynamo_nvtx_range;
use dynamo_runtime::{
    pipeline::{
        AsyncEngineContextProvider, ManyOut, Operator, ResponseStream, ServerStreamingEngine,
        SingleIn, async_trait,
    },
    protocols::annotated::Annotated,
};

use crate::protocols::{
    TokenIdType,
    common::{
        StopConditions,
        llm_backend::{
            BackendOutput, EmbeddingsEngineOutput, FinishReason, LLMEngineOutput,
            PreprocessedRequest, TopLogprobs,
        },
        preprocessor::PreprocessedEmbeddingRequest,
        timing::RequestTracker,
    },
};
use crate::tokenizers::{DecodeStream, Tokenizer};
use dynamo_protocols::types::StopReason;

/// Represents the output stream from the execution engine
pub type ExecutionOutputStream = Annotated<LLMEngineOutput>;

/// Context for executing LLM inference, engine consumes backend input and produces execution output stream
pub type ExecutionContext = ServerStreamingEngine<PreprocessedRequest, ExecutionOutputStream>;

/// Backend handles resource management and orchestrates LLM execution
#[allow(dead_code)]
pub struct Backend {
    pub tokenizer: Option<Tokenizer>, // Handles token encoding/decoding
    validate_engine_decode: bool,     // Enable validation of engine decoding
}

/// Internal state for managing token decoding and stream processing
/// Supports n>1 (multiple choices) by maintaining per-choice decoders.
#[allow(dead_code)]
struct DecoderUnfoldState {
    stream: ManyOut<ExecutionOutputStream>,
    decoders: HashMap<u32, Decoder>,
    finished_choices: HashSet<u32>,
    validate_engine_decode: bool,
    /// Set to true when all expected choices are finished locally, causing the stream to end
    finished: bool,
    /// Tokenizer used to decode top-logprob token_ids when the backend omits text
    /// (e.g. SGLang with --skip-tokenizer-init forcibly drops it).
    tokenizer: Tokenizer,
    skip_special_tokens: bool,
}

fn fill_missing_top_logprob_text(
    tokenizer: &Tokenizer,
    top_logprobs: &mut TopLogprobs,
    skip_special_tokens: bool,
) {
    for position in top_logprobs.iter_mut() {
        for entry in position.iter_mut() {
            if entry.token.is_none()
                && let Ok(decoded) = tokenizer.decode(&[entry.token_id], skip_special_tokens)
            {
                let token: String = decoded.into();
                if entry.bytes.is_none() && !token.is_empty() {
                    entry.bytes = Some(token.as_bytes().to_vec());
                }
                entry.token = Some(token);
            }
        }
    }
}

struct DecoderParams {
    prompt_token_ids: Vec<TokenIdType>,
    stop_conditions: StopConditions,
    skip_special_tokens: bool,
    include_stop_str_in_output: bool,
    tracker: Option<Arc<RequestTracker>>,
    n: u32,
    /// The rendered prompt opened a reasoning section (the preprocessor forwards
    /// `extra_args.reasoning_ended = false`), so generation starts mid-reasoning.
    starts_in_reasoning: bool,
}

impl DecoderParams {
    fn from_request(request: &PreprocessedRequest) -> Self {
        Self {
            prompt_token_ids: request.token_ids.clone(),
            stop_conditions: request.stop_conditions.clone(),
            // Default to true to match upstream framework behavior:
            //   vLLM/sgLang/TRT-LLM: SamplingParams.skip_special_tokens defaults True
            // Without this, models that occasionally emit a special token id
            // (e.g. DeepSeek-V4 producing token id 0 = `<｜begin▁of▁sentence｜>`
            // mid-output) leak the token's text into `content` / `reasoning_content`.
            skip_special_tokens: request.output_options.skip_special_tokens.unwrap_or(true),
            include_stop_str_in_output: request
                .sampling_options
                .include_stop_str_in_output
                .unwrap_or(false),
            tracker: request.tracker.clone(),
            n: request.sampling_options.n.unwrap_or(1) as u32,
            starts_in_reasoning: request
                .extra_args
                .as_ref()
                .and_then(|args| args.get("reasoning_ended"))
                .and_then(|v| v.as_bool())
                == Some(false),
        }
    }
}

impl Backend {
    pub fn from_tokenizer(tokenizer: Tokenizer) -> Arc<Self> {
        Arc::new(Self {
            tokenizer: Some(tokenizer),
            validate_engine_decode: false,
        })
    }

    pub fn from_mdc(mdc: &ModelDeploymentCard) -> Arc<Self> {
        match mdc.tokenizer() {
            Ok(tokenizer) => Self::from_tokenizer(tokenizer),
            Err(err) => {
                tracing::warn!(%err, "error loading tokenizer from ModelDeploymentCard");
                Arc::new(Self {
                    tokenizer: None,
                    validate_engine_decode: false,
                })
            }
        }
    }

    fn decoder(
        &self,
        stream: ManyOut<ExecutionOutputStream>,
        params: DecoderParams,
    ) -> anyhow::Result<DecoderUnfoldState> {
        let Some(tokenizer) = self.tokenizer.as_ref() else {
            anyhow::bail!("Backend built from blank ModelDeploymentCard, no tokenizer");
        };

        // Pre-create one decoder per expected choice so interleaved n>1
        // responses do not share tokenizer state.
        let n = params.n.max(1);
        let mut decoders = HashMap::with_capacity(n as usize);
        for idx in 0..n {
            let decoder = Decoder::new(
                tokenizer.decode_stream(&params.prompt_token_ids, params.skip_special_tokens),
                params.stop_conditions.clone(),
                params.include_stop_str_in_output,
                params.tracker.clone(),
            )
            .with_reasoning_start(params.starts_in_reasoning);
            decoders.insert(idx, decoder);
        }

        Ok(DecoderUnfoldState {
            stream,
            decoders,
            finished_choices: HashSet::new(),
            validate_engine_decode: self.validate_engine_decode,
            finished: false,
            tokenizer: tokenizer.clone(),
            skip_special_tokens: params.skip_special_tokens,
        })
    }
}

#[async_trait]
impl
    Operator<
        SingleIn<PreprocessedRequest>,
        ManyOut<Annotated<BackendOutput>>,
        SingleIn<PreprocessedRequest>,
        ManyOut<Annotated<LLMEngineOutput>>,
    > for Backend
{
    async fn generate(
        &self,
        request: SingleIn<PreprocessedRequest>,
        next: ServerStreamingEngine<PreprocessedRequest, Annotated<LLMEngineOutput>>,
    ) -> Result<ManyOut<Annotated<BackendOutput>>> {
        let decoder_params = DecoderParams::from_request(&request);

        let next_stream = next.generate(request).await?;

        let context = next_stream.context();
        let state = self.decoder(next_stream, decoder_params)?;

        let processed_stream = stream::unfold(state, |mut state| async move {
            // If we've already detected a local stop condition, end the stream
            if state.finished {
                return None;
            }

            match state.stream.next().await {
                Some(output) => {
                    // move to state.process_output
                    // handle any error conditions / unwraps here

                    // events are pass thru
                    if output.is_event() || output.data.is_none() {
                        return Some((output, state));
                    }

                    // Top-logprob text is independent of the selected token text. An engine may
                    // decode the selected token while still omitting candidate text, so repair the
                    // candidates before the decoded-text fast path below.
                    //
                    // Per-entry decode is O(positions * top_k) per delta. Bounded in
                    // practice (streaming: 1 * top_k <= 20) and dwarfed by serialization
                    // on the same path, so we ship the simple version. Revisit if a
                    // streaming flamegraph with top_logprobs=20 puts this above ~1%:
                    // the cheapest win is a shared LRU on the Tokenizer keyed by
                    // (token_id, skip_special_tokens) — top-k entries repeat heavily
                    // across positions and requests. Do NOT batch as a single
                    // decode(&[ids..]) call: BPE merge / leading-space rules differ
                    // between single-token and sequence decode and will corrupt strings.
                    let mut output = output;
                    if let Some(top_logprobs) = output
                        .data
                        .as_mut()
                        .and_then(|data| data.top_logprobs.as_mut())
                    {
                        fill_missing_top_logprob_text(
                            &state.tokenizer,
                            top_logprobs,
                            state.skip_special_tokens,
                        );
                    }

                    // if we have a data field without an event, then we might need to update the data
                    if let Some(data) = &output.data
                        && data.text.is_some()
                        && !state.validate_engine_decode
                    {
                        // Text already decoded; track finish for this choice
                        if data.finish_reason.is_some() {
                            let choice_idx = data.index.unwrap_or(0);
                            state.finished_choices.insert(choice_idx);
                        }
                        return Some((output, state));
                    }

                    let data = output.data.as_ref().unwrap();
                    let choice_idx = data.index.unwrap_or(0);

                    let Some(decoder) = state.decoders.get_mut(&choice_idx) else {
                        tracing::error!(
                            "engine emitted choice index {choice_idx}, but only {} choices were requested",
                            state.decoders.len()
                        );
                        let mut output = output;
                        if let Some(data) = &mut output.data {
                            data.finish_reason = Some(FinishReason::Error(format!(
                                "invalid choice index {choice_idx}"
                            )));
                        }
                        return Some((output, state));
                    };

                    let mut result = match decoder.process_token_ids(&data.token_ids) {
                        Ok(result) => result,
                        Err(e) => {
                            tracing::error!("Failed to process token_ids for choice {choice_idx}: {e}");
                            state.finished_choices.insert(choice_idx);
                            if state.finished_choices.len() >= state.decoders.len() {
                                state.stream.context().stop_generating();
                                state.finished = true;
                            }
                            let mut output = output;
                            if let Some(data) = &mut output.data {
                                data.finish_reason =
                                    Some(FinishReason::Error(format!("decode error: {e}")));
                            }
                            return Some((output, state));
                        }
                    };

                    if result.stop_trigger.is_none() && data.finish_reason.is_some()
                        && let Some(pending) = decoder.flush_pending()
                    {
                        result.text.get_or_insert_with(String::new).push_str(&pending);
                        if let Some(token) = result.tokens.last_mut() {
                            token.get_or_insert_with(String::new).push_str(&pending);
                        }
                    }

                    // NOTE: the `finish_reason` is computed from the generated `token_ids` alone.
                    // The `data` field can have a `finish_reason` set, coming from the underlying
                    // LLM inference `Engine`, and empty `token_ids`. See comment below for more details.
                    //
                    // stop_reason is only set for user-provided stop sequences, not for system
                    // EOS tokens (HiddenStopTokenDetected). This matches OpenAI API behavior where
                    // stop_reason is only present when a user-specified stop sequence is matched.
                    let (finish_reason, stop_reason) = match &result.stop_trigger {
                        Some(StopTrigger::MaxTokensLimit) => (Some(FinishReason::Length), None),
                        Some(StopTrigger::HiddenStopTokenDetected(_)) => {
                            // System EOS token - no stop_reason (user didn't request this stop)
                            (Some(FinishReason::Stop), None)
                        }
                        Some(StopTrigger::UserStopTokenDetected(token_id)) => {
                            // User-provided token stop (hidden from output)
                            (
                                Some(FinishReason::Stop),
                                Some(StopReason::Int((*token_id).into())),
                            )
                        }
                        Some(StopTrigger::VisibleStopTokenDetected(token_id)) => {
                            // Token stop included in output.
                            (
                                Some(FinishReason::Stop),
                                Some(StopReason::Int((*token_id).into())),
                            )
                        }
                        Some(StopTrigger::HiddenStopSequenceDetected(seq)) => {
                            // User-provided stop sequence (hidden from output)
                            (
                                Some(FinishReason::Stop),
                                Some(StopReason::String(seq.clone())),
                            )
                        }
                        Some(StopTrigger::VisibleStopSequenceDetected(seq)) => {
                            // User-provided stop sequence (included in output)
                            (
                                Some(FinishReason::Stop),
                                Some(StopReason::String(seq.clone())),
                            )
                        }
                        None => (None, None),
                    };

                    // If we detected a local stop condition, mark this choice as finished.
                    // Once all expected choices are finished, stop the upstream generator.
                    if finish_reason.is_some() && data.finish_reason.is_none() {
                        state.finished_choices.insert(choice_idx);
                        if state.finished_choices.len() >= state.decoders.len() {
                            state.stream.context().stop_generating();
                            state.finished = true;
                        }
                    }

                    let text = result.text;
                    let tokens = result.tokens;

                    if state.validate_engine_decode {
                        if data.finish_reason != finish_reason {
                            tracing::warn!(
                                "finish reason mismatch: expected {:?}, got {:?}",
                                data.finish_reason,
                                finish_reason
                            );
                        }

                        if data.text.is_some() && data.text != text {
                            tracing::warn!(
                                "text mismatch: expected {:?}, got {:?}",
                                data.text,
                                text
                            );
                        }
                    }

                    // update output in-place
                    let mut data = output.data.take().unwrap();

                    // NOTE: If `finish_reason.is_some()`, then one of the stop conditions was triggered
                    // by the token generation. We should update the `data.finish_reason` in that case.
                    // However, if `finish_reason.is_none()`, it is possible that we are in the case where
                    // `data.token_ids` is empty, and `data.finish_reason` is already correctly set.
                    // In that case, `process_token_ids` above will rewrite `finish_reason` to `None`,
                    // which we don't want to propagate to `data.finish_reason`.
                    if finish_reason.is_some() {
                        data.finish_reason = finish_reason;
                        data.stop_reason = stop_reason.or(data.stop_reason);
                    }
                    data.text = text;
                    data.tokens = Some(tokens);

                    output.data = Some(data);

                    Some((output, state))
                }

                None => None,
            }
        })
        .fuse();

        // convert stream of processed Annotated<LLMEngineOutput> to Annotated<BackendOutput>
        //let mdcsum = self.mdcsum.clone();
        let stream = processed_stream.map(move |output| {
            output.map_data(|data| {
                Ok(BackendOutput {
                    token_ids: data.token_ids,
                    tokens: data.tokens.unwrap_or_default(),
                    text: data.text,
                    cum_log_probs: data.cum_log_probs,
                    log_probs: data.log_probs,
                    top_logprobs: data.top_logprobs,
                    finish_reason: data.finish_reason,
                    stop_reason: data.stop_reason,
                    //mdcsum: mdcsum.clone(),
                    index: data.index,
                    completion_usage: data.completion_usage,
                    disaggregated_params: data.disaggregated_params,
                    encoder_result: data.encoder_result,
                    worker_trace_link: data.worker_trace_link,
                    engine_data: data.engine_data,
                    routing_data: data.routing_data,
                })
            })
        });

        Ok(ResponseStream::new(Box::pin(stream), context))
    }
}

#[async_trait]
impl
    Operator<
        SingleIn<PreprocessedEmbeddingRequest>,
        ManyOut<Annotated<EmbeddingsEngineOutput>>,
        SingleIn<PreprocessedEmbeddingRequest>,
        ManyOut<Annotated<EmbeddingsEngineOutput>>,
    > for Backend
{
    async fn generate(
        &self,
        request: SingleIn<PreprocessedEmbeddingRequest>,
        next: ServerStreamingEngine<
            PreprocessedEmbeddingRequest,
            Annotated<EmbeddingsEngineOutput>,
        >,
    ) -> Result<ManyOut<Annotated<EmbeddingsEngineOutput>>> {
        // For embeddings, we mostly pass through since no detokenization is needed
        // But we could add validation, logging, or other post-processing here
        let response_stream = next.generate(request).await?;

        // Could add embedding-specific post-processing here:
        // - Validation of embedding dimensions
        // - Normalization if requested
        // - Usage statistics validation

        Ok(response_stream)
    }
}

/// The [`Decoder`] object could be a member of either the internal LLM engine or part of the
/// postprocessor. If in the postprocessor, should be minimally in the same process or at very minimum
/// on the same physical machine connected by an IPC.
#[allow(dead_code)]
pub struct Decoder {
    decode_stream: DecodeStream,
    tracker: Option<Arc<RequestTracker>>,

    // do not trigger stop conditions until at least this many tokens have been generated
    min_tokens: u32,

    // single tokens that if found in the response will trigger a stop condition after the
    // minimum number of tokens have been generated
    hidden_stop_ids: HashSet<TokenIdType>,

    // single tokens that if found in the response will trigger a stop condition and be returned
    visible_stop_ids: HashSet<TokenIdType>,

    // user-provided token stop IDs, kept separate from system/EOS stop IDs so
    // stop_reason can report user-triggered token stops without reporting EOS.
    user_stop_ids: HashSet<TokenIdType>,

    // text sequences that if found in the response will trigger a stop condition after the
    // minimum number of tokens have been generated (excluded from output)
    hidden_stop_sequences: Vec<String>,

    // text sequences that if found in the response will trigger a stop condition after the
    // minimum number of tokens have been generated (included in output)
    visible_stop_sequences: Vec<String>,

    // number of generated tokens
    generated_tokens: u32,

    // content jailed by partial hidden stop matches
    jail: String,

    // maximum number of bytes for the largest stop sequence
    jail_max_bytes: usize,

    // the number of bytes currently jailed
    jailed_bytes: usize,

    // generation is inside a reasoning section: stop sequences apply to the answer only,
    // so string matching starts after the reasoning end marker
    in_reasoning: bool,

    // recent reasoning text, kept to find an end marker split across tokens
    reasoning_tail: String,
}

/// Reasoning end markers of the parsers whose prompts can open a reasoning section
/// (Kimi K3, MiniMax M3, and the generic `<think>` form).
const REASONING_END_MARKERS: [&str; 3] = ["<|close|>think<|sep|>", "</mm:think>", "</think>"];

#[allow(dead_code)]
#[derive(Debug)]
pub enum StopTrigger {
    MaxTokensLimit,
    HiddenStopTokenDetected(TokenIdType),
    UserStopTokenDetected(TokenIdType),
    VisibleStopTokenDetected(TokenIdType),
    HiddenStopSequenceDetected(String),
    VisibleStopSequenceDetected(String),
}

pub struct StepResult {
    pub token: Option<String>,
    pub stop_trigger: Option<StopTrigger>,
}

impl StepResult {
    fn ok(token: Option<String>) -> Self {
        Self {
            token,
            stop_trigger: None,
        }
    }

    fn with_stop_trigger(token: Option<String>, stop_trigger: StopTrigger) -> Self {
        Self {
            token,
            stop_trigger: Some(stop_trigger),
        }
    }
}

/// Result of processing a sequence of tokens
pub struct SeqResult {
    pub tokens: Vec<Option<String>>,       // Individual decoded tokens
    pub text: Option<String>,              // Combined decoded text
    pub stop_trigger: Option<StopTrigger>, // Reason for stopping generation, if any
}

#[allow(dead_code)]
impl Decoder {
    pub fn new(
        decode_stream: DecodeStream,
        stop_condition: StopConditions,
        include_stop_str_in_output: bool,
        tracker: Option<Arc<RequestTracker>>,
    ) -> Self {
        let user_stop_ids: HashSet<TokenIdType> = stop_condition
            .stop_token_ids
            .unwrap_or_default()
            .iter()
            .copied()
            .collect();
        let system_stop_ids: HashSet<TokenIdType> = stop_condition
            .stop_token_ids_hidden
            .unwrap_or_default()
            .iter()
            .copied()
            .collect();
        let visible_stop_ids: HashSet<TokenIdType> = stop_condition
            .stop_token_ids_visible
            .unwrap_or_default()
            .iter()
            .copied()
            .collect();
        let hidden_stop_ids = user_stop_ids.union(&system_stop_ids).copied().collect();

        // Categorize stop sequences based on include_stop_str_in_output:
        // - When true: user-provided stop sequences go to visible (included in output)
        // - When false: user-provided stop sequences go to hidden (excluded from output)
        let (hidden_stop_sequences, visible_stop_sequences) = if include_stop_str_in_output {
            (Vec::new(), stop_condition.stop.unwrap_or_default())
        } else {
            (stop_condition.stop.unwrap_or_default(), Vec::new())
        };

        // Calculate jail_max_bytes considering both hidden and visible stop sequences
        let jail_max_bytes = hidden_stop_sequences
            .iter()
            .chain(visible_stop_sequences.iter())
            .map(|x| x.len())
            .max()
            .unwrap_or(0);

        Self {
            decode_stream,
            tracker,
            hidden_stop_ids,
            visible_stop_ids,
            user_stop_ids,
            hidden_stop_sequences,
            visible_stop_sequences,
            min_tokens: stop_condition.min_tokens.unwrap_or(0),
            generated_tokens: 0,
            jail: String::new(),
            jail_max_bytes,
            jailed_bytes: 0,
            in_reasoning: false,
            reasoning_tail: String::new(),
        }
    }

    /// Generation starts inside a reasoning section (the prompt opened it). Stop sequences
    /// are then matched only after the reasoning end marker, so a stop string that occurs
    /// while the model thinks cannot end the request before it answers.
    pub fn with_reasoning_start(mut self, starts_in_reasoning: bool) -> Self {
        self.in_reasoning = starts_in_reasoning && self.jail_max_bytes > 0;
        self
    }

    /// Byte offset just past the earliest reasoning end marker in `text`, if any.
    fn reasoning_end(text: &str) -> Option<usize> {
        REASONING_END_MARKERS
            .iter()
            .filter_map(|marker| text.find(marker).map(|pos| pos + marker.len()))
            .min()
    }

    /// Match stops before releasing text. The jail retains search history plus any
    /// suffix that could still become a hidden stop string on a later token.
    fn match_stop_sequences(&mut self, new_text: &str) -> StepResult {
        let pending_start = self.jail.len() - self.jailed_bytes;
        self.jail.push_str(new_text);

        let hidden_match = self
            .hidden_stop_sequences
            .iter()
            .filter_map(|seq| {
                galil_seiferas::gs_find(self.jail.as_bytes(), seq.as_bytes())
                    .map(|offset| (offset, seq))
            })
            .min_by_key(|(offset, _)| *offset);
        if let Some((offset, seq)) = hidden_match {
            let emitted = self.jail[pending_start..offset.max(pending_start)].to_string();
            let trigger = StopTrigger::HiddenStopSequenceDetected(seq.clone());
            self.jail.clear();
            self.jailed_bytes = 0;
            return StepResult::with_stop_trigger(Some(emitted), trigger);
        }

        let visible_match = self
            .visible_stop_sequences
            .iter()
            .filter_map(|seq| {
                galil_seiferas::gs_find(self.jail.as_bytes(), seq.as_bytes())
                    .map(|offset| (offset, seq))
            })
            .min_by_key(|(offset, _)| *offset);
        if let Some((offset, seq)) = visible_match {
            let stop_end = offset + seq.len();
            let emitted = self.jail[pending_start..stop_end.max(pending_start)].to_string();
            let trigger = StopTrigger::VisibleStopSequenceDetected(seq.clone());
            self.jail.clear();
            self.jailed_bytes = 0;
            return StepResult::with_stop_trigger(Some(emitted), trigger);
        }

        // Only hidden stops need emission buffering. UTF-8 character boundaries
        // ensure a withheld prefix never divides a code point.
        let held = self
            .hidden_stop_sequences
            .iter()
            .map(|seq| {
                seq.char_indices()
                    .rev()
                    .map(|(len, _)| len)
                    .find(|&len| len > 0 && self.jail.ends_with(&seq[..len]))
                    .unwrap_or(0)
            })
            .max()
            .unwrap_or(0);
        let emit_end = self.jail.len() - held;
        let emitted = self.jail[pending_start..emit_end].to_string();
        self.jailed_bytes = held;
        Self::maybe_drain_to_max_bytes(&mut self.jail, self.jail_max_bytes);
        StepResult::ok(Some(emitted))
    }

    /// An unfinished stop prefix is ordinary text when generation ends without
    /// a string match. Drain it once at EOS or an engine-supplied finish.
    fn flush_pending(&mut self) -> Option<String> {
        let pending = self.jailed_string();
        self.jailed_bytes = 0;
        self.jail.clear();
        pending
    }

    /// Minimum amount of work to determine if a given generated/decoded sequence should be stopped
    /// This method can be called by the inner most loop of the LLM engine or minimally in the same
    /// process as the LLM engine.
    ///
    /// In the future, this method may kick off async cpu/tokio tasks and or async cuda tasks to
    /// handle logits post-processing and/or other tasks.
    pub fn step(&mut self, token_id: TokenIdType) -> Result<StepResult> {
        // increment the generated tokens
        self.generated_tokens += 1;

        // decode the token
        let detokenize_start = self.tracker.as_ref().map(|_| Instant::now());
        let token = {
            let _nvtx = dynamo_nvtx_range!("detokenize");
            self.decode_stream.step(token_id)?
        };
        if let (Some(start), Some(tracker)) = (detokenize_start, &self.tracker) {
            tracker.record_detokenize_latency(start.elapsed());
        }

        // stop conditions to not apply until the minimum number of tokens have been generated
        if self.generated_tokens < self.min_tokens {
            return Ok(StepResult::ok(token));
        }

        // Check token stops. Visible token IDs are included in output.
        if self.visible_stop_ids.contains(&token_id) {
            let token = match (self.flush_pending(), token) {
                (Some(mut pending), Some(token)) => {
                    pending.push_str(&token);
                    Some(pending)
                }
                (pending, token) => pending.or(token),
            };
            return Ok(StepResult::with_stop_trigger(
                token,
                StopTrigger::VisibleStopTokenDetected(token_id),
            ));
        }

        // Check token stops. User-provided token IDs take precedence over
        // system/EOS IDs so stop_reason only reports stops the caller requested.
        if self.hidden_stop_ids.contains(&token_id) {
            let trigger = if self.user_stop_ids.contains(&token_id) {
                StopTrigger::UserStopTokenDetected(token_id)
            } else {
                StopTrigger::HiddenStopTokenDetected(token_id)
            };
            return Ok(StepResult::with_stop_trigger(self.flush_pending(), trigger));
        }

        // While the model is still reasoning, stop sequences do not apply; start matching
        // on the text after the reasoning end marker.
        if self.in_reasoning
            && let Some(token) = &token
        {
            let tail_start = self.reasoning_tail.len();
            self.reasoning_tail.push_str(token);
            let Some(end) = Self::reasoning_end(&self.reasoning_tail) else {
                let keep = REASONING_END_MARKERS
                    .iter()
                    .map(|m| m.len())
                    .max()
                    .unwrap_or(0);
                Self::maybe_drain_to_max_bytes(&mut self.reasoning_tail, keep);
                return Ok(StepResult::ok(Some(token.clone())));
            };
            self.in_reasoning = false;
            // the answer part of this token (after the marker), if the marker ended in it
            let answer_start = end.saturating_sub(tail_start).min(token.len());
            let answer = self.reasoning_tail[end..].to_string();
            self.reasoning_tail.clear();
            let mut result = self.match_stop_sequences(&answer);
            let mut out = token[..answer_start].to_string();
            if let Some(emitted) = result.token.take() {
                out.push_str(&emitted);
            }
            result.token = Some(out);
            return Ok(result);
        }

        // check stop sequences - the jail will always hold at least the largest stop sequence
        // if jail_max_bytes is 0, then there are no stop sequences
        if self.jail_max_bytes > 0
            && let Some(token) = &token
        {
            return Ok(self.match_stop_sequences(token));
        }

        Ok(StepResult::ok(token))
    }

    pub fn process_token_ids(&mut self, token_ids: &[TokenIdType]) -> Result<SeqResult> {
        let mut text: Option<String> = None;
        let mut tokens = Vec::with_capacity(token_ids.len());

        for token_id in token_ids {
            let StepResult {
                token,
                stop_trigger,
            } = self.step(*token_id)?;

            // Always include token text (for visible stops, the stop string is already in the token)
            if let Some(token) = &token {
                text.get_or_insert_with(|| String::with_capacity(token_ids.len()))
                    .push_str(token);
            }
            tokens.push(token);

            if let Some(stop_trigger) = stop_trigger {
                return Ok(SeqResult {
                    tokens,
                    text,
                    stop_trigger: Some(stop_trigger),
                });
            }
        }

        Ok(SeqResult {
            tokens,
            text,
            stop_trigger: None,
        })
    }

    fn jailed_string(&self) -> Option<String> {
        if self.jailed_bytes > 0 {
            // get the last jailed_bytes from the jail
            Some(self.jail[self.jail.len() - self.jailed_bytes..].to_string())
        } else {
            None
        }
    }

    fn maybe_drain_to_max_bytes(s: &mut String, max_bytes: usize) {
        if s.len() > max_bytes {
            let mut drain_len = s.len() - max_bytes;
            while !s.is_char_boundary(drain_len) {
                drain_len -= 1;
            }
            s.drain(0..drain_len);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::common::llm_backend::TopLogprob;
    use crate::protocols::common::{OutputOptions, SamplingOptions};
    use crate::protocols::openai::{
        DeltaGeneratorExt, chat_completions::DeltaGenerator, delta_common::DeltaGeneratorOptions,
    };
    use crate::tokenizers::traits;
    use dynamo_runtime::pipeline::{AsyncEngine, Error, ResponseStream};
    use futures::StreamExt;
    use std::sync::Arc;

    #[test]
    fn test_char_boundary_drain() {
        let mut s = String::from("helloñworld"); // 12 bytes total ñ is 2 bytes
        let max_bytes = 6; // 12 - 6 = 6 which is inside ñ
        assert!(!s.is_char_boundary(s.len() - max_bytes)); // initially we are not on a char boundary
        Decoder::maybe_drain_to_max_bytes(&mut s, max_bytes);
        assert!(s.is_char_boundary(0)); // front of jail string on valid char boundary
        assert_eq!(s, "ñworld");
    }

    /// A mock tokenizer that always returns Err from decode().
    /// Used to test the error propagation path in Decoder::process_token_ids().
    struct FailingDecoder;

    impl traits::Encoder for FailingDecoder {
        fn encode(&self, _input: &str) -> anyhow::Result<crate::tokenizers::Encoding> {
            Ok(crate::tokenizers::Encoding::Sp(vec![]))
        }
        fn encode_batch(
            &self,
            _inputs: &[&str],
        ) -> anyhow::Result<Vec<crate::tokenizers::Encoding>> {
            Ok(vec![])
        }
    }

    impl traits::Decoder for FailingDecoder {
        fn decode(
            &self,
            _token_ids: &[TokenIdType],
            _skip_special_tokens: bool,
        ) -> anyhow::Result<traits::DecodeResult> {
            Err(anyhow::anyhow!(
                "Unable to decode into a valid UTF-8 string: incomplete utf-8 byte sequence from index 6"
            ))
        }
    }

    impl traits::Tokenizer for FailingDecoder {}

    struct CandidateDecoder;

    impl traits::Encoder for CandidateDecoder {
        fn encode(&self, _input: &str) -> anyhow::Result<crate::tokenizers::Encoding> {
            Ok(crate::tokenizers::Encoding::Sp(vec![]))
        }

        fn encode_batch(
            &self,
            _inputs: &[&str],
        ) -> anyhow::Result<Vec<crate::tokenizers::Encoding>> {
            Ok(vec![])
        }
    }

    impl traits::Decoder for CandidateDecoder {
        fn decode(
            &self,
            token_ids: &[TokenIdType],
            _skip_special_tokens: bool,
        ) -> anyhow::Result<traits::DecodeResult> {
            let token = match token_ids {
                [] => "",
                [101] => "Okay",
                [102] => " Okay",
                [103] => "<|user|>",
                [101, 103] => "Okay<|user|>",
                _ => anyhow::bail!("unexpected token IDs: {token_ids:?}"),
            };
            Ok(traits::DecodeResult::Complete(token.to_string()))
        }
    }

    impl traits::Tokenizer for CandidateDecoder {}

    /// Test tokenizer: each id decodes to a fixed piece; a sequence decodes to the concatenation.
    struct PieceDecoder(&'static [&'static str]);

    impl traits::Encoder for PieceDecoder {
        fn encode(&self, _input: &str) -> anyhow::Result<crate::tokenizers::Encoding> {
            Ok(crate::tokenizers::Encoding::Sp(vec![]))
        }
        fn encode_batch(
            &self,
            _inputs: &[&str],
        ) -> anyhow::Result<Vec<crate::tokenizers::Encoding>> {
            Ok(vec![])
        }
    }

    impl traits::Decoder for PieceDecoder {
        fn decode(
            &self,
            token_ids: &[TokenIdType],
            _skip_special_tokens: bool,
        ) -> anyhow::Result<traits::DecodeResult> {
            Ok(traits::DecodeResult::Complete(
                token_ids.iter().map(|id| self.0[*id as usize]).collect(),
            ))
        }
    }

    impl traits::Tokenizer for PieceDecoder {}

    /// Run `ids` through a decoder with hidden stop sequence "7"; returns the emitted text
    /// and whether a stop sequence fired.
    fn run_stop_7(
        pieces: &'static [&'static str],
        ids: &[TokenIdType],
        in_reasoning: bool,
    ) -> (String, bool) {
        let tokenizer: Arc<dyn traits::Tokenizer> = Arc::new(PieceDecoder(pieces));
        let decode_stream = crate::tokenizers::DecodeStream::new(tokenizer, &[], false);
        let stop_conditions = StopConditions {
            stop: Some(vec!["7".to_string()]),
            ..Default::default()
        };
        let mut decoder = Decoder::new(decode_stream, stop_conditions, false, None)
            .with_reasoning_start(in_reasoning);
        let mut out = String::new();
        for id in ids {
            let step = decoder.step(*id).unwrap();
            if let Some(t) = step.token {
                out.push_str(&t);
            }
            if step.stop_trigger.is_some() {
                return (out, true);
            }
        }
        (out, false)
    }

    fn stop_prefix_decoder(
        pieces: &'static [&'static str],
        stops: &[&str],
        include_stop: bool,
    ) -> Decoder {
        let tokenizer: Arc<dyn traits::Tokenizer> = Arc::new(PieceDecoder(pieces));
        Decoder::new(
            crate::tokenizers::DecodeStream::new(tokenizer, &[], false),
            StopConditions {
                stop: Some(stops.iter().map(|s| s.to_string()).collect()),
                stop_token_ids_hidden: Some(vec![99]),
                ..Default::default()
            },
            include_stop,
            None,
        )
    }

    #[test]
    fn stop_prefix_is_withheld_until_full_match() {
        let mut decoder =
            stop_prefix_decoder(&["alpha", " STOP", "_HERE", " beta"], &["STOP_HERE"], false);
        assert_eq!(decoder.step(0).unwrap().token.as_deref(), Some("alpha"));
        assert_eq!(decoder.step(1).unwrap().token.as_deref(), Some(" "));
        let last = decoder.step(2).unwrap();
        assert_eq!(last.token.as_deref().unwrap_or(""), "");
        assert!(
            matches!(last.stop_trigger, Some(StopTrigger::HiddenStopSequenceDetected(ref s)) if s == "STOP_HERE")
        );
    }

    #[test]
    fn responses_stop_wire_request_uses_shared_prefix_decoder() {
        use crate::protocols::common::StopConditionsProvider;
        use crate::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
        use crate::protocols::openai::responses::NvCreateResponse;

        for stream in [false, true] {
            for stop in [
                serde_json::json!("STOP_HERE"),
                serde_json::json!(["STOP_HERE"]),
            ] {
                let response: NvCreateResponse = serde_json::from_value(serde_json::json!({
                    "model": "test-model", "input": "alpha STOP_HERE beta", "stream": stream, "stop": stop
                })).unwrap();
                let chat: NvCreateChatCompletionRequest = response.try_into().unwrap();
                let conditions = chat.extract_stop_conditions().unwrap();
                let tokenizer: Arc<dyn traits::Tokenizer> =
                    Arc::new(PieceDecoder(&["alpha", " STOP", "_HERE", " beta"]));
                let mut decoder = Decoder::new(
                    crate::tokenizers::DecodeStream::new(tokenizer, &[], false),
                    conditions,
                    false,
                    None,
                );
                assert_eq!(decoder.step(0).unwrap().token.as_deref(), Some("alpha"));
                assert_eq!(decoder.step(1).unwrap().token.as_deref(), Some(" "));
                let terminal = decoder.step(2).unwrap();
                assert_eq!(terminal.token.as_deref().unwrap_or(""), "");
                assert!(matches!(terminal.stop_trigger,
                    Some(StopTrigger::HiddenStopSequenceDetected(ref s)) if s == "STOP_HERE"));
            }
        }
    }

    #[test]
    fn stop_prefix_handles_one_character_per_token() {
        let mut decoder = stop_prefix_decoder(
            &["alpha ", "S", "T", "O", "P", "_", "H", "E", "R", "E"],
            &["STOP_HERE"],
            false,
        );
        let result = decoder
            .process_token_ids(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            .unwrap();
        assert_eq!(result.text.as_deref(), Some("alpha "));
        assert!(result.stop_trigger.is_some());
    }

    #[test]
    fn stop_prefix_false_alarm_releases_buffered_text() {
        let mut decoder =
            stop_prefix_decoder(&["alpha STOP", "_OTHER beta"], &["STOP_HERE"], false);
        assert_eq!(decoder.step(0).unwrap().token.as_deref(), Some("alpha "));
        let next = decoder.step(1).unwrap();
        assert_eq!(next.token.as_deref(), Some("STOP_OTHER beta"));
        assert!(next.stop_trigger.is_none());
    }

    #[test]
    fn stop_prefix_unicode_and_overlapping_stops() {
        let mut decoder =
            stop_prefix_decoder(&["hello ", "é", "终", "止 tail"], &["é终止", "终止"], false);
        let result = decoder.process_token_ids(&[0, 1, 2, 3]).unwrap();
        assert_eq!(result.text.as_deref(), Some("hello "));
        assert!(
            matches!(result.stop_trigger, Some(StopTrigger::HiddenStopSequenceDetected(ref s)) if s == "é终止")
        );
    }

    #[test]
    fn stop_prefix_selects_earliest_match_in_a_chunk() {
        let mut decoder =
            stop_prefix_decoder(&["alpha STOP_HERE beta"], &["beta", "STOP_HERE"], false);
        let result = decoder.step(0).unwrap();
        assert_eq!(result.token.as_deref(), Some("alpha "));
        assert!(
            matches!(result.stop_trigger, Some(StopTrigger::HiddenStopSequenceDetected(ref s)) if s == "STOP_HERE")
        );
    }

    #[test]
    fn stop_prefix_visible_stop_is_returned() {
        let mut decoder =
            stop_prefix_decoder(&["alpha", " STOP", "_HERE beta"], &["STOP_HERE"], true);
        let result = decoder.process_token_ids(&[0, 1, 2]).unwrap();
        assert_eq!(result.text.as_deref(), Some("alpha STOP_HERE"));
        assert!(matches!(
            result.stop_trigger,
            Some(StopTrigger::VisibleStopSequenceDetected(_))
        ));
    }

    #[test]
    fn stop_prefix_after_reasoning_end_is_withheld() {
        let pieces = &[
            "STOP_HERE in reasoning",
            "<|close|>think<|sep|>alpha STOP",
            "_HERE beta",
        ];
        let mut decoder =
            stop_prefix_decoder(pieces, &["STOP_HERE"], false).with_reasoning_start(true);
        let result = decoder.process_token_ids(&[0, 1, 2]).unwrap();
        assert_eq!(
            result.text.as_deref(),
            Some("STOP_HERE in reasoning<|close|>think<|sep|>alpha ")
        );
        assert!(result.stop_trigger.is_some());
    }

    #[test]
    fn stop_prefix_unmatched_suffix_is_flushed_at_eos() {
        let mut decoder = stop_prefix_decoder(&["alpha STOP", "<eos>"], &["STOP_HERE"], false);
        decoder.hidden_stop_ids.insert(1);
        assert_eq!(decoder.step(0).unwrap().token.as_deref(), Some("alpha "));
        let terminal = decoder.step(1).unwrap();
        assert_eq!(terminal.token.as_deref(), Some("STOP"));
        assert!(matches!(
            terminal.stop_trigger,
            Some(StopTrigger::HiddenStopTokenDetected(1))
        ));
        assert!(decoder.flush_pending().is_none());
    }

    #[test]
    fn stop_prefix_visible_token_stop_flushes_pending_text() {
        let mut decoder = stop_prefix_decoder(&["alpha STOP", "!"], &["STOP_HERE"], false);
        decoder.visible_stop_ids.insert(1);
        assert_eq!(decoder.step(0).unwrap().token.as_deref(), Some("alpha "));
        let terminal = decoder.step(1).unwrap();
        assert_eq!(terminal.token.as_deref(), Some("STOP!"));
        assert!(matches!(
            terminal.stop_trigger,
            Some(StopTrigger::VisibleStopTokenDetected(1))
        ));
    }

    #[test]
    fn stop_prefix_choices_keep_independent_pending_text() {
        let pieces = &["alpha STOP", "second STOP", "_HERE", "_OTHER"];
        let mut first = stop_prefix_decoder(pieces, &["STOP_HERE"], false);
        let mut second = stop_prefix_decoder(pieces, &["STOP_HERE"], false);
        assert_eq!(first.step(0).unwrap().token.as_deref(), Some("alpha "));
        assert_eq!(second.step(1).unwrap().token.as_deref(), Some("second "));
        assert_eq!(first.step(2).unwrap().token.as_deref(), Some(""));
        assert_eq!(second.step(3).unwrap().token.as_deref(), Some("STOP_OTHER"));
    }

    struct SyntheticStopPrefixEndEngine {
        terminal_has_tokens: bool,
        finish_reason: FinishReason,
    }

    #[async_trait]
    impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
        for SyntheticStopPrefixEndEngine
    {
        async fn generate(
            &self,
            request: SingleIn<PreprocessedRequest>,
        ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
            let mut outputs = Vec::new();
            if !self.terminal_has_tokens {
                outputs.push(Annotated::from_data(LLMEngineOutput {
                    token_ids: vec![0],
                    index: Some(0),
                    ..Default::default()
                }));
            }
            outputs.push(Annotated::from_data(LLMEngineOutput {
                token_ids: if self.terminal_has_tokens {
                    vec![0]
                } else {
                    vec![]
                },
                finish_reason: Some(self.finish_reason.clone()),
                index: Some(0),
                ..Default::default()
            }));
            Ok(ResponseStream::new(
                Box::pin(futures::stream::iter(outputs)),
                request.context(),
            ))
        }
    }

    #[tokio::test]
    async fn stop_prefix_engine_finish_flushes_with_and_without_token_ids() {
        for terminal_has_tokens in [false, true] {
            for finish_reason in [FinishReason::Stop, FinishReason::Length] {
                let tokenizer: Arc<dyn traits::Tokenizer> = Arc::new(PieceDecoder(&["alpha STOP"]));
                let backend = Backend::from_tokenizer(Tokenizer::from(tokenizer));
                let request = PreprocessedRequest::builder()
                    .model("test-model".to_string())
                    .token_ids(vec![])
                    .stop_conditions(StopConditions {
                        stop: Some(vec!["STOP_HERE".to_string()]),
                        ..Default::default()
                    })
                    .sampling_options(SamplingOptions::default())
                    .output_options(OutputOptions::default())
                    .build()
                    .unwrap();
                let engine: ServerStreamingEngine<PreprocessedRequest, Annotated<LLMEngineOutput>> =
                    Arc::new(SyntheticStopPrefixEndEngine {
                        terminal_has_tokens,
                        finish_reason: finish_reason.clone(),
                    });
                let mut stream =
                    Operator::generate(backend.as_ref(), SingleIn::new(request), engine)
                        .await
                        .unwrap();
                let mut text = String::new();
                let mut tokens = Vec::new();
                let mut finish = None;
                while let Some(output) = stream.next().await {
                    let data = output.data.unwrap();
                    text.push_str(data.text.as_deref().unwrap_or(""));
                    tokens.extend(data.token_ids);
                    if data.finish_reason.is_some() {
                        finish = data.finish_reason;
                    }
                }
                assert_eq!(
                    text, "alpha STOP",
                    "terminal_has_tokens={terminal_has_tokens}"
                );
                assert_eq!(tokens, vec![0], "buffer flush must not invent token IDs");
                assert_eq!(finish, Some(finish_reason));
            }
        }
    }

    const K3_PIECES: &[&str] = &[
        "Count: 1 2 3",                // 0
        " 4 5 6 7",                    // 1
        "<|close|>think<|sep|>",       // 2
        "1 2 3",                       // 3
        " 6 7 8",                      // 4
        "<|close|>th",                 // 5
        "ink<|sep|>6 7",               // 6
        "x<|close|>think<|sep|>A 7 B", // 7
    ];

    #[test]
    fn stop_sequence_does_not_fire_inside_reasoning() {
        // "7" in the reasoning is ignored; the stop fires in the answer.
        let (out, stopped) = run_stop_7(K3_PIECES, &[0, 1, 2, 3, 4], true);
        assert!(stopped);
        assert_eq!(out, "Count: 1 2 3 4 5 6 7<|close|>think<|sep|>1 2 3 6 ");
    }

    #[test]
    fn stop_sequence_fires_in_reasoning_without_the_flag() {
        // previous behaviour, kept when the prompt did not open a reasoning section
        let (out, stopped) = run_stop_7(K3_PIECES, &[0, 1, 2, 3, 4], false);
        assert!(stopped);
        assert_eq!(out, "Count: 1 2 3 4 5 6 ");
    }

    #[test]
    fn reasoning_end_marker_split_across_tokens() {
        let (out, stopped) = run_stop_7(K3_PIECES, &[0, 5, 6], true);
        assert!(stopped);
        assert_eq!(out, "Count: 1 2 3<|close|>think<|sep|>6 ");
    }

    #[test]
    fn stop_in_the_same_token_as_the_reasoning_end() {
        let (out, stopped) = run_stop_7(K3_PIECES, &[7], true);
        assert!(stopped);
        assert_eq!(out, "x<|close|>think<|sep|>A ");
    }

    #[test]
    fn reasoning_never_ends_no_stop() {
        let (out, stopped) = run_stop_7(K3_PIECES, &[0, 1], true);
        assert!(!stopped);
        assert_eq!(out, "Count: 1 2 3 4 5 6 7");
    }

    struct SyntheticSglangEngine {
        engine_decodes_text: bool,
    }

    struct SyntheticSglangStopEngine;

    #[async_trait]
    impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
        for SyntheticSglangStopEngine
    {
        async fn generate(
            &self,
            request: SingleIn<PreprocessedRequest>,
        ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
            let output = LLMEngineOutput {
                // SGLang's output_ids_through_stop() includes the matched stop
                // position. The frontend must hide its text without mutating
                // this raw token sequence.
                token_ids: vec![101, 103],
                finish_reason: Some(FinishReason::Stop),
                index: Some(0),
                ..Default::default()
            };

            Ok(ResponseStream::new(
                Box::pin(futures::stream::once(async move {
                    Annotated::from_data(output)
                })),
                request.context(),
            ))
        }
    }

    #[async_trait]
    impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
        for SyntheticSglangEngine
    {
        async fn generate(
            &self,
            request: SingleIn<PreprocessedRequest>,
        ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
            let output = LLMEngineOutput {
                token_ids: vec![101],
                tokens: self
                    .engine_decodes_text
                    .then(|| vec![Some("Okay".to_string())]),
                text: self.engine_decodes_text.then(|| "Okay".to_string()),
                log_probs: Some(vec![-0.125]),
                top_logprobs: Some(vec![vec![
                    TopLogprob {
                        rank: 1,
                        token_id: 101,
                        token: None,
                        logprob: -0.125,
                        bytes: None,
                    },
                    TopLogprob {
                        rank: 2,
                        token_id: 102,
                        token: None,
                        logprob: -1.5,
                        bytes: None,
                    },
                ]]),
                index: Some(0),
                ..Default::default()
            };

            Ok(ResponseStream::new(
                Box::pin(futures::stream::once(async move {
                    Annotated::from_data(output)
                })),
                request.context(),
            ))
        }
    }

    #[test]
    fn test_fill_missing_top_logprob_text_without_worker_tokenizer() {
        let tokenizer: Arc<dyn traits::Tokenizer> = Arc::new(CandidateDecoder);
        let tokenizer = Tokenizer::from(tokenizer);
        let mut top_logprobs = vec![vec![
            TopLogprob {
                rank: 1,
                token_id: 101,
                token: None,
                logprob: -0.1,
                bytes: None,
            },
            TopLogprob {
                rank: 2,
                token_id: 102,
                token: None,
                logprob: -0.2,
                bytes: None,
            },
        ]];

        fill_missing_top_logprob_text(&tokenizer, &mut top_logprobs, true);

        assert_eq!(top_logprobs[0][0].token.as_deref(), Some("Okay"));
        assert_eq!(top_logprobs[0][0].bytes, Some(b"Okay".to_vec()));
        assert_eq!(top_logprobs[0][1].token.as_deref(), Some(" Okay"));
        assert_eq!(top_logprobs[0][1].bytes, Some(b" Okay".to_vec()));
        assert_eq!(top_logprobs[0][0].logprob, -0.1);
        assert_eq!(top_logprobs[0][1].logprob, -0.2);
    }

    async fn assert_sglang_top_logprobs_are_decoded_in_openai_response(engine_decodes_text: bool) {
        let tokenizer: Arc<dyn traits::Tokenizer> = Arc::new(CandidateDecoder);
        let backend = Backend::from_tokenizer(Tokenizer::from(tokenizer));
        let request = PreprocessedRequest::builder()
            .model("test-model".to_string())
            .token_ids(vec![])
            .stop_conditions(StopConditions::default())
            .sampling_options(SamplingOptions::default())
            .output_options(OutputOptions {
                logprobs: Some(2),
                ..Default::default()
            })
            .build()
            .expect("valid preprocessed request");
        let engine: ServerStreamingEngine<PreprocessedRequest, Annotated<LLMEngineOutput>> =
            Arc::new(SyntheticSglangEngine {
                engine_decodes_text,
            });

        let mut stream = Operator::generate(backend.as_ref(), SingleIn::new(request), engine)
            .await
            .expect("backend generation succeeds");
        let output = stream
            .next()
            .await
            .expect("backend emits a response")
            .data
            .expect("response contains backend output");

        let options = DeltaGeneratorOptions::new(None, None, true, None);
        let mut generator = DeltaGenerator::new(
            "test-model".to_string(),
            options,
            "test-request".to_string(),
        );
        let response = generator
            .choice_from_postprocessor(output)
            .expect("OpenAI response conversion succeeds");
        let content = response.inner.choices[0]
            .logprobs
            .as_ref()
            .expect("client-visible logprobs")
            .content
            .as_ref()
            .expect("client-visible logprob content");
        let candidates = &content[0].top_logprobs;

        assert_eq!(candidates[0].token, "Okay");
        assert_eq!(candidates[0].bytes, Some(b"Okay".to_vec()));
        assert_eq!(candidates[0].logprob, -0.125);
        assert_eq!(candidates[1].token, " Okay");
        assert_eq!(candidates[1].bytes, Some(b" Okay".to_vec()));
        assert_eq!(candidates[1].logprob, -1.5);
    }

    #[tokio::test]
    async fn test_sglang_top_logprobs_are_decoded_in_openai_response() {
        assert_sglang_top_logprobs_are_decoded_in_openai_response(false).await;
    }

    #[tokio::test]
    async fn test_sglang_top_logprobs_are_decoded_before_engine_text_fast_path() {
        assert_sglang_top_logprobs_are_decoded_in_openai_response(true).await;
    }

    #[tokio::test]
    async fn test_sglang_hidden_stop_text_is_suppressed_without_mutating_tito_ids() {
        use crate::protocols::common::extensions::NvExt;

        let tokenizer: Arc<dyn traits::Tokenizer> = Arc::new(CandidateDecoder);
        let backend = Backend::from_tokenizer(Tokenizer::from(tokenizer));
        let request = PreprocessedRequest::builder()
            .model("test-model".to_string())
            .token_ids(vec![])
            .stop_conditions(StopConditions {
                stop_token_ids_hidden: Some(vec![103]),
                ..Default::default()
            })
            .sampling_options(SamplingOptions::default())
            .output_options(OutputOptions::default())
            .build()
            .expect("valid preprocessed request");
        let engine: ServerStreamingEngine<PreprocessedRequest, Annotated<LLMEngineOutput>> =
            Arc::new(SyntheticSglangStopEngine);

        let mut stream = Operator::generate(backend.as_ref(), SingleIn::new(request), engine)
            .await
            .expect("backend generation succeeds");
        let output = stream
            .next()
            .await
            .expect("backend emits a response")
            .data
            .expect("response contains backend output");

        assert_eq!(output.text.as_deref(), Some("Okay"));
        assert_eq!(output.tokens, vec![Some("Okay".to_string()), None]);
        assert_eq!(output.token_ids, vec![101, 103]);
        assert_eq!(output.finish_reason, Some(FinishReason::Stop));

        let nvext = NvExt::builder()
            .extra_fields(vec!["completion_token_ids".to_string()])
            .build()
            .expect("valid nvext selection");
        let options = DeltaGeneratorOptions::new(None, None, false, Some(&nvext));
        let mut generator = DeltaGenerator::new(
            "test-model".to_string(),
            options,
            "test-request".to_string(),
        );
        let response = generator
            .choice_from_postprocessor(output)
            .expect("OpenAI response conversion succeeds");

        assert_eq!(
            response
                .nvext
                .as_ref()
                .and_then(|fields| fields.get("completion_token_ids")),
            Some(&serde_json::json!([101, 103]))
        );
        assert_eq!(generator.get_usage().completion_tokens, 2);
    }

    /// When the tokenizer's decode() returns Err, Decoder::process_token_ids()
    /// should propagate the error. In the backend unfold closure, this error
    /// gets caught and converted to FinishReason::Error.
    #[test]
    fn test_decoder_process_token_ids_propagates_decode_error() {
        let tokenizer: Arc<dyn traits::Tokenizer> = Arc::new(FailingDecoder);
        let decode_stream = crate::tokenizers::DecodeStream::new(tokenizer, &[], false);
        let stop_conditions = StopConditions::default();

        let mut decoder = Decoder::new(decode_stream, stop_conditions, false, None);

        let result = decoder.process_token_ids(&[42]);
        assert!(
            result.is_err(),
            "process_token_ids should propagate decode errors"
        );

        let err_msg = result.err().unwrap().to_string();
        assert!(
            err_msg.contains("incomplete utf-8 byte sequence"),
            "error should contain the original decode error message, got: {err_msg}"
        );
    }

    /// Verify that the error message format matches what the backend unfold
    /// closure would wrap into FinishReason::Error.
    #[test]
    fn test_decoder_error_message_format_for_finish_reason() {
        let tokenizer: Arc<dyn traits::Tokenizer> = Arc::new(FailingDecoder);
        let decode_stream = crate::tokenizers::DecodeStream::new(tokenizer, &[], false);
        let stop_conditions = StopConditions::default();

        let mut decoder = Decoder::new(decode_stream, stop_conditions, false, None);

        let result = decoder.process_token_ids(&[42]);
        let err = result.err().expect("should be Err");

        // This is what the backend unfold closure does:
        let finish_reason = FinishReason::Error(format!("decode error: {err}"));
        match &finish_reason {
            FinishReason::Error(msg) => {
                assert!(
                    msg.starts_with("decode error:"),
                    "FinishReason::Error should have 'decode error:' prefix, got: {msg}"
                );
                assert!(
                    msg.contains("incomplete utf-8 byte sequence"),
                    "FinishReason::Error should contain original error, got: {msg}"
                );
            }
            other => panic!("Expected FinishReason::Error, got: {:?}", other),
        }
    }
}
