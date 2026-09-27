// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::pin::Pin;
use std::sync::Arc;

use dynamo_runtime::engine::{AsyncEngineContext, AsyncEngineContextProvider};
use dynamo_runtime::pipeline::Context;
use dynamo_runtime::protocols::annotated::Annotated;
use futures::{Stream, StreamExt};

use crate::protocols::common::preprocessor::PreprocessedRequest;
use crate::protocols::common::timing::RequestTracker;
use crate::protocols::openai::{
    chat_completions::NvCreateChatCompletionStreamResponse, completions::NvCreateCompletionResponse,
};
use crate::request_trace::{
    AgentContextTraceState, RequestReplayMetrics, SharedFinishReasonMetadata,
    SharedOutputSequenceHashCapture,
};

struct RequestTraceRequestEndState {
    request_tracker: Arc<RequestTracker>,
    /// `None` when the request cannot be replayed as one token stream (multimodal inputs): the
    /// record is still emitted, without its `replay` section.
    replay_metrics: Option<Arc<RequestReplayMetrics>>,
    output_sequence_hash_capture: Option<SharedOutputSequenceHashCapture>,
}

pub(crate) struct RequestEndTraceState {
    agent: Option<AgentContextTraceState>,
    request: Option<RequestTraceRequestEndState>,
    request_id: String,
    request_context: Arc<dyn AsyncEngineContext>,
}

fn request_trace_rejection(common_request: &PreprocessedRequest) -> Option<&'static str> {
    if common_request.prompt_embeds.is_some() {
        return Some("prompt embeddings are not supported");
    }
    if common_request.sampling_options.n.unwrap_or(1) > 1 {
        return Some("multiple output choices are not supported");
    }
    if common_request.sampling_options.best_of.unwrap_or(1) > 1 {
        return Some("best_of greater than one is not supported");
    }
    None
}

/// Requests whose frontend token ids do not describe what the engine prefilled cannot be replayed
/// as one token stream. They are traced without the `replay` section instead of being dropped, so
/// their timings, worker, routing decision and cache-loss stages stay observable.
fn replay_unsupported_reason(common_request: &PreprocessedRequest) -> Option<&'static str> {
    if common_request.multi_modal_data.is_some() {
        return Some("multimodal inputs cannot be replayed as one token stream");
    }
    None
}

fn shared_replay_metrics(
    token_ids: &[crate::protocols::TokenIdType],
    trace_block_size: usize,
) -> Option<Arc<RequestReplayMetrics>> {
    if trace_block_size == 0 {
        return None;
    }
    super::replay_metrics(token_ids, trace_block_size).map(Arc::new)
}

pub(crate) fn build_request_end_trace_state(
    common_request: &PreprocessedRequest,
    tracker: &Option<Arc<RequestTracker>>,
    context: &Context<()>,
    trace_block_size: usize,
) -> Option<RequestEndTraceState> {
    build_request_end_trace_state_for_policy(
        common_request,
        tracker,
        context,
        trace_block_size,
        super::policy().emit_request_end_records(),
    )
}

fn build_request_end_trace_state_for_policy(
    common_request: &PreprocessedRequest,
    tracker: &Option<Arc<RequestTracker>>,
    context: &Context<()>,
    trace_block_size: usize,
    request_trace_enabled: bool,
) -> Option<RequestEndTraceState> {
    let has_agent_context = common_request.agent_context.is_some();

    if !request_trace_enabled {
        return None;
    }

    let request_id = context.id();
    if let Some(reason) = request_trace_rejection(common_request) {
        tracing::warn!(
            %request_id,
            reason,
            "request trace skipped because the request cannot be represented as one replay request"
        );
        return None;
    }

    let request_tracker = match tracker {
        Some(tracker) => tracker.clone(),
        None => {
            tracing::warn!(
                %request_id,
                "request trace skipped because the request tracker is unavailable"
            );
            return None;
        }
    };

    let replay_metrics = match replay_unsupported_reason(common_request) {
        Some(reason) => {
            tracing::debug!(
                %request_id,
                reason,
                "request trace emitted without replay hashes"
            );
            None
        }
        None => match shared_replay_metrics(&common_request.token_ids, trace_block_size) {
            Some(metrics) => Some(metrics),
            None => {
                tracing::warn!(
                    %request_id,
                    "request trace skipped because the KV cache block size is unavailable"
                );
                return None;
            }
        },
    };

    let agent = has_agent_context
        .then(|| super::build_agent_context_trace_state(common_request, tracker, context))
        .flatten();

    let output_sequence_hash_capture = replay_metrics.as_ref().map(|replay_metrics| {
        super::output_sequence_hash_capture(&common_request.token_ids, replay_metrics)
    });
    let request = RequestTraceRequestEndState {
        request_tracker,
        output_sequence_hash_capture,
        replay_metrics,
    };

    Some(RequestEndTraceState {
        agent,
        request: Some(request),
        request_id: request_id.to_string(),
        request_context: context.context(),
    })
}

impl RequestEndTraceState {
    fn emit(&mut self) {
        let Some(RequestTraceRequestEndState {
            request_tracker,
            replay_metrics,
            output_sequence_hash_capture,
        }) = self.request.take()
        else {
            return;
        };
        let replay_metrics = replay_metrics.map(|replay_metrics| {
            let mut replay_metrics = super::into_owned_replay_metrics(replay_metrics);
            if let Some(capture) = &output_sequence_hash_capture {
                replay_metrics.output_sequence_hashes = capture.lock().unwrap().sequence_hashes();
            }
            replay_metrics
        });
        if let Some(agent_state) = self.agent.take() {
            let (agent_context, mut metrics) =
                super::request_metrics_from_agent_state(agent_state, self.request_id.clone());
            metrics.replay = replay_metrics;
            super::record::emit_agent_request_end(agent_context, metrics);
        } else {
            super::record::emit_request_end(
                self.request_id.clone(),
                &request_tracker,
                replay_metrics,
            );
        }
    }
}

impl Drop for RequestEndTraceState {
    fn drop(&mut self) {
        if self.request_context.is_killed() {
            self.emit();
        }
    }
}

pub(crate) fn output_sequence_hash_capture_handle(
    trace_state: &Option<RequestEndTraceState>,
) -> Option<SharedOutputSequenceHashCapture> {
    trace_state
        .as_ref()
        .and_then(|state| state.request.as_ref()?.output_sequence_hash_capture.clone())
}

pub(crate) fn finish_reason_metadata_handle(
    trace_state: &Option<RequestEndTraceState>,
) -> Option<SharedFinishReasonMetadata> {
    trace_state
        .as_ref()
        .and_then(|state| state.agent.as_ref())
        .map(|state| state.finish_reason_metadata.clone())
}

fn wrap_request_end_stream<Resp>(
    stream: Pin<Box<dyn Stream<Item = Annotated<Resp>> + Send>>,
    trace_state: Option<RequestEndTraceState>,
) -> Pin<Box<dyn Stream<Item = Annotated<Resp>> + Send>>
where
    Resp: Send + 'static,
{
    let Some(mut trace_state) = trace_state else {
        return stream;
    };

    let (stream, done) = crate::telemetry::stream::notify_on_completion(stream);
    tokio::spawn(async move {
        done.await;
        trace_state.emit();
    });
    stream
}

pub(crate) fn wrap_chat_request_end_stream(
    stream: Pin<Box<dyn Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send>>,
    trace_state: Option<RequestEndTraceState>,
) -> Pin<Box<dyn Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send>> {
    let Some(finish_reason_metadata) = finish_reason_metadata_handle(&trace_state) else {
        return wrap_request_end_stream(stream, trace_state);
    };

    let stream = stream.map(move |response| {
        super::record_chat_finish_reason_metadata(&finish_reason_metadata, &response);
        response
    });
    wrap_request_end_stream(Box::pin(stream), trace_state)
}

pub(crate) fn wrap_completion_request_end_stream(
    stream: Pin<Box<dyn Stream<Item = Annotated<NvCreateCompletionResponse>> + Send>>,
    trace_state: Option<RequestEndTraceState>,
) -> Pin<Box<dyn Stream<Item = Annotated<NvCreateCompletionResponse>> + Send>> {
    let Some(finish_reason_metadata) = finish_reason_metadata_handle(&trace_state) else {
        return wrap_request_end_stream(stream, trace_state);
    };

    let stream = stream.map(move |response| {
        super::record_completion_finish_reason_metadata(&finish_reason_metadata, &response);
        response
    });
    wrap_request_end_stream(Box::pin(stream), trace_state)
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };
    use std::task::{Context as TaskContext, Poll};
    use std::time::{Duration, Instant};

    use super::*;
    use crate::protocols::common::extensions::{AgentCompaction, AgentContext, InputTrigger};
    use crate::protocols::common::{OutputOptions, SamplingOptions, StopConditions};
    use crate::request_trace::BUS;
    use crate::request_trace::RequestTraceEventSource;

    struct TrackerDropStream {
        tracker: Arc<RequestTracker>,
        dropped: Arc<AtomicBool>,
    }

    impl Stream for TrackerDropStream {
        type Item = Annotated<NvCreateCompletionResponse>;

        fn poll_next(self: Pin<&mut Self>, _cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
            Poll::Pending
        }
    }

    impl Drop for TrackerDropStream {
        fn drop(&mut self) {
            self.tracker.record_osl(9);
            self.tracker.record_finish();
            self.dropped.store(true, Ordering::Release);
        }
    }

    fn preprocessed_request(sampling_options: SamplingOptions) -> PreprocessedRequest {
        PreprocessedRequest::builder()
            .model("test-model".to_string())
            .token_ids(vec![1, 2, 3])
            .stop_conditions(StopConditions::default())
            .sampling_options(sampling_options)
            .output_options(OutputOptions::default())
            .eos_token_ids(vec![])
            .annotations(vec![])
            .build()
            .unwrap()
    }

    fn request_end_state(
        request_id: &str,
        tracker: Arc<RequestTracker>,
    ) -> (RequestEndTraceState, Arc<dyn AsyncEngineContext>) {
        let context = Context::new(()).context();
        (
            RequestEndTraceState {
                agent: None,
                request: Some(RequestTraceRequestEndState {
                    request_tracker: tracker,
                    replay_metrics: Some(Arc::new(RequestReplayMetrics {
                        trace_block_size: 2,
                        input_length: 2,
                        input_sequence_hashes: vec![11],
                        output_sequence_hashes: Vec::new(),
                    })),
                    output_sequence_hash_capture: None,
                }),
                request_id: request_id.to_string(),
                request_context: context.clone(),
            },
            context,
        )
    }

    fn drain_request_records(
        receiver: &mut tokio::sync::broadcast::Receiver<crate::request_trace::RequestTraceRecord>,
        request_id: &str,
    ) -> Vec<crate::request_trace::RequestTraceRecord> {
        let mut records = Vec::new();
        loop {
            match receiver.try_recv() {
                Ok(record)
                    if record
                        .request
                        .as_ref()
                        .is_some_and(|request| request.request_id == request_id) =>
                {
                    records.push(record);
                }
                Ok(_) | Err(tokio::sync::broadcast::error::TryRecvError::Lagged(_)) => {}
                Err(tokio::sync::broadcast::error::TryRecvError::Empty)
                | Err(tokio::sync::broadcast::error::TryRecvError::Closed) => break,
            }
        }
        records
    }

    #[test]
    fn rejects_unsupported_request_shapes() {
        let mut multi_choice = preprocessed_request(SamplingOptions {
            n: Some(2),
            ..Default::default()
        });
        assert_eq!(
            request_trace_rejection(&multi_choice),
            Some("multiple output choices are not supported")
        );

        multi_choice.sampling_options.n = Some(1);
        multi_choice.sampling_options.best_of = Some(2);
        assert_eq!(
            request_trace_rejection(&multi_choice),
            Some("best_of greater than one is not supported")
        );

        multi_choice.sampling_options.best_of = Some(1);
        multi_choice.prompt_embeds = Some("embedding".to_string());
        assert_eq!(
            request_trace_rejection(&multi_choice),
            Some("prompt embeddings are not supported")
        );

        multi_choice.prompt_embeds = None;
        multi_choice.multi_modal_data = Some(Default::default());
        assert_eq!(
            request_trace_rejection(&multi_choice),
            None,
            "multimodal requests are traced; only their replay section is withheld"
        );
        assert_eq!(
            replay_unsupported_reason(&multi_choice),
            Some("multimodal inputs cannot be replayed as one token stream")
        );
        assert_eq!(
            replay_unsupported_reason(&preprocessed_request(SamplingOptions::default())),
            None
        );
    }

    #[test]
    fn multimodal_request_is_traced_without_replay() {
        BUS.init(16);
        let mut receiver = BUS.subscribe();

        let mut request = preprocessed_request(SamplingOptions::default());
        request.multi_modal_data = Some(Default::default());
        let tracker = Arc::new(RequestTracker::new());
        tracker.record_isl(3, Some(0));
        tracker.record_osl(2);
        tracker.record_finish();
        let context = Context::new(());
        let request_id = context.id().to_string();

        let mut state =
            build_request_end_trace_state_for_policy(&request, &Some(tracker), &context, 2, true)
                .expect("multimodal requests are traced");
        let request_state = state.request.as_ref().expect("request state");
        assert!(request_state.replay_metrics.is_none());
        assert!(request_state.output_sequence_hash_capture.is_none());

        state.emit();
        let records = drain_request_records(&mut receiver, &request_id);
        assert_eq!(records.len(), 1);
        let traced = records[0].request.as_ref().expect("request payload");
        assert!(traced.replay.is_none());
        assert_eq!(traced.input_tokens, Some(3));
        assert_eq!(traced.output_tokens, Some(2));
        let json = serde_json::to_value(&records[0]).expect("serializable record");
        assert!(
            json["request"].get("replay").is_none(),
            "the replay section is omitted, not serialized as null"
        );
        assert_eq!(json["event_type"], "request_end");

        // Text-only requests are unchanged: they keep their replay hashes.
        let text_state = build_request_end_trace_state_for_policy(
            &preprocessed_request(SamplingOptions::default()),
            &Some(Arc::new(RequestTracker::new())),
            &Context::new(()),
            2,
            true,
        )
        .expect("text requests are traced");
        let text_request = text_state.request.as_ref().expect("request state");
        assert!(text_request.replay_metrics.is_some());
        assert!(text_request.output_sequence_hash_capture.is_some());
    }

    #[test]
    fn replay_hashing_requires_block_size() {
        assert!(shared_replay_metrics(&[1, 2, 3], 0).is_none());

        let replay = shared_replay_metrics(&[1, 2, 3], 2).unwrap();
        assert_eq!(replay.input_sequence_hashes.len(), 2);
    }

    #[test]
    fn long_isl_hashing_reports_mode_costs_without_threshold() {
        let token_ids = (0..131_072_u32).collect::<Vec<_>>();

        let started = Instant::now();
        let request_only = shared_replay_metrics(&token_ids, 64).unwrap();
        let request_elapsed = started.elapsed();

        let started = Instant::now();
        let repeated = shared_replay_metrics(&token_ids, 64).unwrap();
        let repeated_elapsed = started.elapsed();

        eprintln!(
            "long-ISL replay hashing: request_only={request_elapsed:?}, repeated={repeated_elapsed:?}"
        );
        assert_eq!(request_only.input_sequence_hashes.len(), 2_048);
        assert_eq!(
            request_only.input_sequence_hashes,
            repeated.input_sequence_hashes
        );
    }

    #[test]
    fn cancellation_before_response_stream_emits_request_end() {
        BUS.init(16);
        let mut receiver = BUS.subscribe();
        let tracker = Arc::new(RequestTracker::new());
        tracker.record_isl(2, None);
        let (state, context) = request_end_state("req-pre-stream-cancel", tracker);

        context.kill();
        drop(state);

        let records = drain_request_records(&mut receiver, "req-pre-stream-cancel");
        assert_eq!(records.len(), 1);
        let record = &records[0];
        let request = record.request.as_ref().expect("request payload");
        assert_eq!(request.input_tokens, Some(2));
        assert_eq!(request.output_tokens, Some(0));
    }

    #[test]
    fn early_error_without_cancellation_does_not_emit_request_end() {
        BUS.init(16);
        let mut receiver = BUS.subscribe();
        let (state, _context) =
            request_end_state("req-pre-stream-error", Arc::new(RequestTracker::new()));

        drop(state);

        assert!(drain_request_records(&mut receiver, "req-pre-stream-error").is_empty());
    }

    #[test]
    fn explicit_emit_followed_by_drop_emits_once() {
        BUS.init(16);
        let mut receiver = BUS.subscribe();
        let (mut state, context) =
            request_end_state("req-emit-once", Arc::new(RequestTracker::new()));

        state.emit();
        context.kill();
        drop(state);

        assert_eq!(
            drain_request_records(&mut receiver, "req-emit-once").len(),
            1
        );
    }

    #[tokio::test]
    async fn cancellation_after_response_stream_reads_tracker_after_inner_drop() {
        BUS.init(16);
        let mut receiver = BUS.subscribe();
        let tracker = Arc::new(RequestTracker::new());
        let dropped = Arc::new(AtomicBool::new(false));
        let (state, _context) = request_end_state("req-drop", tracker.clone());
        let stream = TrackerDropStream {
            tracker,
            dropped: dropped.clone(),
        };

        let wrapped = wrap_request_end_stream(Box::pin(stream), Some(state));
        drop(wrapped);

        let record = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let record = receiver.recv().await.unwrap();
                if record
                    .request
                    .as_ref()
                    .is_some_and(|request| request.request_id == "req-drop")
                {
                    break record;
                }
            }
        })
        .await
        .unwrap();
        let request = record.request.as_ref().expect("request payload");
        assert!(dropped.load(Ordering::Acquire));
        assert_eq!(request.output_tokens, Some(9));
    }

    #[tokio::test]
    async fn agent_context_emits_enriched_request_trace_row() {
        BUS.init(16);
        let mut receiver = BUS.subscribe();
        let tracker = Arc::new(RequestTracker::new());
        let dropped = Arc::new(AtomicBool::new(false));
        let mut request = preprocessed_request(SamplingOptions::default());
        request.agent_context = Some(AgentContext {
            session_id: "root".to_string(),
            parent_session_id: None,
            session_final: None,
            compaction: Some(AgentCompaction {
                trigger: Some("manual".to_string()),
                reason: Some("user_requested".to_string()),
                implementation: Some("responses_compact".to_string()),
                phase: Some("standalone_turn".to_string()),
                strategy: Some("memento".to_string()),
            }),
            input_trigger: Some(InputTrigger::ToolResult),
        });
        let mut context = Context::new(());
        context.insert(
            crate::request_trace::X_REQUEST_ID_CONTEXT_KEY,
            "llm-call-1".to_string(),
        );
        let expected_request_id = context.id().to_string();
        let state = build_request_end_trace_state_for_policy(
            &request,
            &Some(tracker.clone()),
            &context,
            2,
            true,
        )
        .unwrap();
        let stream = TrackerDropStream {
            tracker,
            dropped: dropped.clone(),
        };

        let wrapped = wrap_request_end_stream(Box::pin(stream), Some(state));
        drop(wrapped);

        let record = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let record = receiver.recv().await.unwrap();
                if record
                    .request
                    .as_ref()
                    .is_some_and(|request| request.request_id == expected_request_id)
                {
                    break record;
                }
            }
        })
        .await
        .unwrap();
        assert!(dropped.load(Ordering::Acquire));
        assert_eq!(record.event_source, Some(RequestTraceEventSource::Dynamo));
        let agent_context = record.agent_context.as_ref().expect("agent context");
        assert_eq!(agent_context.session_id, "root");
        assert_eq!(agent_context.input_trigger, Some(InputTrigger::ToolResult));
        assert_eq!(
            agent_context
                .compaction
                .as_ref()
                .and_then(|compaction| compaction.strategy.as_deref()),
            Some("memento")
        );
        let request = record.request.as_ref().expect("request payload");
        assert_eq!(request.model.as_deref(), Some("test-model"));
        assert_eq!(request.x_request_id.as_deref(), Some("llm-call-1"));
        assert_eq!(request.output_tokens, Some(9));
        assert_eq!(
            request
                .replay
                .as_ref()
                .expect("replay metrics")
                .input_length,
            3
        );
    }

    #[test]
    fn agent_context_does_not_bypass_request_trace_eligibility() {
        let mut request = preprocessed_request(SamplingOptions {
            best_of: Some(2),
            ..Default::default()
        });
        request.agent_context = Some(AgentContext {
            session_id: "root".to_string(),
            parent_session_id: None,
            session_final: None,
            compaction: None,
            input_trigger: None,
        });
        let tracker = Some(Arc::new(RequestTracker::new()));
        let context = Context::new(());

        let state = build_request_end_trace_state_for_policy(&request, &tracker, &context, 2, true);

        assert!(state.is_none());
    }
}
