// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use crate::protocols::common::extensions::AgentContext;
use crate::protocols::common::timing::RequestTracker;
use crate::request_trace::RequestReplayMetrics;
use crate::request_trace::RequestTraceEventSource;
use crate::request_trace::RequestTracePayload;

use super::RequestTraceEventType;
use super::RequestTraceMetrics;
use super::RequestTraceRecord;
use super::RequestTraceSchema;
use super::RequestTraceWorkerInfo;
use super::publish;

fn unix_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis().min(u128::from(u64::MAX)) as u64)
        .unwrap_or(0)
}

fn sanitize_finite(value: Option<f64>) -> Option<f64> {
    value.filter(|value| value.is_finite())
}

fn sanitize_request(request: &mut RequestTraceMetrics) {
    request.prefill_wait_time_ms = sanitize_finite(request.prefill_wait_time_ms);
    request.prefill_time_ms = sanitize_finite(request.prefill_time_ms);
    request.ttft_ms = sanitize_finite(request.ttft_ms);
    request.total_time_ms = sanitize_finite(request.total_time_ms);
    request.avg_itl_ms = sanitize_finite(request.avg_itl_ms);
    request.kv_hit_rate = sanitize_finite(request.kv_hit_rate);
    request.kv_transfer_estimated_latency_ms =
        sanitize_finite(request.kv_transfer_estimated_latency_ms);
}

fn event_time_unix_ms_from_request(request: &RequestTraceMetrics) -> u64 {
    request
        .request_received_ms
        .map_or_else(unix_time_ms, |received_ms| {
            request
                .total_time_ms
                .map(|ms| received_ms.saturating_add(ms.max(0.0).round() as u64))
                .unwrap_or(received_ms)
        })
}

pub(crate) fn emit_request_end(
    request_id: String,
    tracker: &RequestTracker,
    replay: RequestReplayMetrics,
) {
    let timing = tracker.get_timing_info();
    let event_time_unix_ms = timing.total_time_ms.map_or_else(unix_time_ms, |elapsed| {
        timing
            .request_received_ms
            .saturating_add(elapsed.max(0.0).round() as u64)
    });
    let worker = tracker
        .get_worker_info()
        .map(|worker| RequestTraceWorkerInfo {
            prefill_worker_id: worker.prefill_worker_id,
            prefill_dp_rank: worker.prefill_dp_rank,
            decode_worker_id: worker.decode_worker_id,
            decode_dp_rank: worker.decode_dp_rank,
        });

    let mut request = RequestTraceMetrics {
        request_id,
        x_request_id: None,
        model: None,
        input_tokens: tracker.isl_tokens().map(|v| v as u64),
        output_tokens: Some(tracker.osl_tokens()),
        cached_tokens: tracker.cached_tokens().map(|v| v as u64),
        backend_actual_cached_tokens: tracker.backend_cached_tokens().map(|v| v as u64),
        request_received_ms: Some(timing.request_received_ms),
        prefill_wait_time_ms: timing.prefill_wait_time_ms,
        prefill_time_ms: timing.prefill_time_ms,
        ttft_ms: timing.ttft_ms,
        total_time_ms: timing.total_time_ms,
        avg_itl_ms: tracker.avg_itl_ms(),
        kv_hit_rate: timing.kv_hit_rate,
        kv_transfer_estimated_latency_ms: timing.kv_transfer_estimated_latency_ms,
        queue_depth: timing.router_queue_depth.map(|v| v as u64),
        worker,
        replay: Some(replay),
        finish_reason_metadata: None,
        routing_decision: tracker.routing_decision_trace(),
    };
    sanitize_request(&mut request);

    publish(RequestTraceRecord {
        schema: RequestTraceSchema::V1,
        event_type: RequestTraceEventType::RequestEnd,
        event_time_unix_ms,
        event_source: None,
        agent_context: None,
        request: Some(request),
        tool: None,
        payload: None,
    });
}

pub(crate) fn emit_agent_request_end(
    agent_context: AgentContext,
    mut request: RequestTraceMetrics,
) {
    sanitize_request(&mut request);
    publish(RequestTraceRecord {
        schema: RequestTraceSchema::V1,
        event_type: RequestTraceEventType::RequestEnd,
        event_time_unix_ms: event_time_unix_ms_from_request(&request),
        event_source: Some(RequestTraceEventSource::Dynamo),
        agent_context: Some(agent_context),
        request: Some(request),
        tool: None,
        payload: None,
    });
}

pub(crate) fn emit_request_payload(payload: RequestTracePayload, event_time_unix_ms: u64) {
    publish(RequestTraceRecord {
        schema: RequestTraceSchema::V1,
        event_type: RequestTraceEventType::RequestPayload,
        event_time_unix_ms,
        event_source: Some(RequestTraceEventSource::Dynamo),
        agent_context: None,
        request: None,
        tool: None,
        payload: Some(payload),
    });
}

pub(crate) fn publish_tool_record(record: RequestTraceRecord) {
    if !super::config::capture_enabled() || !super::config::policy().emit_tool_records() {
        return;
    }

    if let Err(error) = validate_tool_record(&record) {
        tracing::warn!(
            %error,
            event_type = ?record.event_type,
            "dropping invalid request trace tool record"
        );
        return;
    }

    publish(record);
}

pub(crate) fn validate_tool_record(record: &RequestTraceRecord) -> anyhow::Result<()> {
    if record.schema != RequestTraceSchema::V1 {
        anyhow::bail!("unsupported request trace schema: {:?}", record.schema);
    }
    if record.event_source != Some(RequestTraceEventSource::Harness) {
        anyhow::bail!(
            "request trace tool records must be harness-originated, got {:?}",
            record.event_source
        );
    }
    if !record.event_type.is_tool_event() {
        anyhow::bail!("expected tool event, got {:?}", record.event_type);
    }
    if record.agent_context.is_none() {
        anyhow::bail!("missing agent_context");
    }
    let Some(tool) = record.tool.as_ref() else {
        anyhow::bail!("missing tool payload");
    };
    if tool.duration_ms.is_some_and(|value| !value.is_finite()) {
        anyhow::bail!("tool duration_ms must be finite");
    }
    if record.request.is_some() {
        anyhow::bail!("tool event must not include request metrics");
    }
    if record.payload.is_some() {
        anyhow::bail!("tool event must not include request payload");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::common::timing::{RoutingDecisionCandidate, RoutingDecisionTrace};
    use crate::request_trace::BUS;
    use crate::request_trace::RequestTraceToolEvent;

    #[tokio::test]
    async fn emits_tracker_timing_lengths_and_hashes() {
        BUS.init(16);
        let mut rx = BUS.subscribe();
        let tracker = RequestTracker::new();
        tracker.record_isl(8, Some(4));
        tracker.record_backend_cached_tokens(Some(4));
        tracker.record_kv_hit(2.0, 4);
        tracker.record_osl(7);
        tracker.record_finish();

        emit_request_end(
            "req-1".to_string(),
            &tracker,
            RequestReplayMetrics {
                trace_block_size: 2,
                input_length: 3,
                input_sequence_hashes: vec![11, 22],
            },
        );

        let record = loop {
            let record = rx.recv().await.unwrap();
            if record
                .request
                .as_ref()
                .is_some_and(|request| request.request_id == "req-1")
            {
                break record;
            }
        };
        let request = record.request.as_ref().expect("request payload");
        assert_eq!(request.request_id, "req-1");
        assert_eq!(request.input_tokens, Some(8));
        assert_eq!(request.output_tokens, Some(7));
        assert_eq!(request.cached_tokens, Some(4));
        assert_eq!(request.backend_actual_cached_tokens, Some(4));
        assert_eq!(request.kv_hit_rate, Some(0.5));
        assert_eq!(
            request.request_received_ms,
            Some(tracker.request_received_epoch_ms())
        );
        assert!(
            record.event_time_unix_ms
                >= request
                    .request_received_ms
                    .expect("request received timestamp")
        );
        assert_eq!(
            request
                .replay
                .as_ref()
                .expect("replay metrics")
                .input_length,
            3
        );
    }

    async fn emitted_cached_tokens(
        request_id: &str,
        predicted_cached_tokens: usize,
        backend_cached_tokens: Option<usize>,
    ) -> Option<u64> {
        BUS.init(16);
        let mut rx = BUS.subscribe();
        let tracker = RequestTracker::new();
        tracker.record_isl(24_576, Some(predicted_cached_tokens));
        tracker.record_backend_cached_tokens(backend_cached_tokens);

        emit_request_end(
            request_id.to_string(),
            &tracker,
            RequestReplayMetrics {
                trace_block_size: 12_288,
                input_length: 24_576,
                input_sequence_hashes: vec![11, 22],
            },
        );

        let request = loop {
            let record = rx.recv().await.unwrap();
            if record
                .request
                .as_ref()
                .is_some_and(|request| request.request_id == request_id)
            {
                break record.request.unwrap();
            }
        };
        assert_eq!(tracker.cached_tokens(), Some(predicted_cached_tokens));
        assert_eq!(request.cached_tokens, Some(predicted_cached_tokens as u64));
        request.backend_actual_cached_tokens
    }

    #[tokio::test]
    async fn request_end_uses_backend_actual_instead_of_router_prediction() {
        assert_eq!(
            emitted_cached_tokens("actual-partial", 24_576, Some(12_288)).await,
            Some(12_288)
        );
    }

    #[tokio::test]
    async fn request_end_preserves_backend_actual_zero() {
        assert_eq!(
            emitted_cached_tokens("actual-zero", 24_576, Some(0)).await,
            Some(0)
        );
    }

    #[tokio::test]
    async fn request_end_keeps_missing_backend_actual_unknown() {
        assert_eq!(
            emitted_cached_tokens("actual-missing", 24_576, None).await,
            None
        );
    }

    fn routing_trace_with_raw(
        raw: dynamo_kv_router::scheduling::RawCacheCoverage,
    ) -> RoutingDecisionTrace {
        let selected = raw.selected.unwrap_or(dynamo_kv_router::scheduling::RawCacheCandidate {
            worker_id: 7,
            dp_rank: 0,
            hbm_prefix_tokens: 0,
            cpu_extension_tokens: 0,
            total_tokens: 0,
        });
        RoutingDecisionTrace {
            score_decision: None,
            decision_explanation: None,
            frontend_instance: Some("frontend-test".into()),
            schema: "dynamo.router.decision.v44.v2".into(),
            candidate_scope: "selected_and_best_eligible_cache_holder".into(),
            block_size: 12_288,
            input_tokens: 24_704,
            selected: RoutingDecisionCandidate {
                worker_id: selected.worker_id,
                dp_rank: selected.dp_rank,
                effective_overlap_blocks: 2.0,
                cached_tokens: 24_576,
                hbm_blocks: 1,
                cpu_ram_cumulative_blocks: 2,
                cpu_ram_only_blocks: 1,
                disk_cumulative_blocks: 2,
            },
            eligible_oracle: None,
            raw_cache_coverage: Some(raw),
        }
    }

    #[tokio::test]
    async fn request_end_preserves_complete_raw_coverage_and_backend_actual_separately() {
        use dynamo_kv_router::scheduling::{
            RawCacheCandidate, RawCacheCoverage, RawCacheObservation,
        };

        BUS.init(16);
        let mut rx = BUS.subscribe();
        let tracker = RequestTracker::new();
        tracker.record_isl(24_704, Some(24_576));
        tracker.record_backend_cached_tokens(Some(24_704));
        let selected = RawCacheCandidate {
            worker_id: u64::MAX,
            dp_rank: u32::MAX,
            hbm_prefix_tokens: 12_288,
            cpu_extension_tokens: 12_288,
            total_tokens: 24_576,
        };
        tracker.record_routing_decision_trace(routing_trace_with_raw(RawCacheCoverage {
            schema: "dynamo.router.raw_cache_coverage.v1".into(),
            basis: "router_index".into(),
            observation: RawCacheObservation::Complete,
            input_tokens: 24_704,
            resident: Some(RawCacheCandidate {
                total_tokens: 24_704,
                hbm_prefix_tokens: 24_704,
                cpu_extension_tokens: 0,
                ..selected
            }),
            eligible: Some(selected),
            selected: Some(selected),
            overload_gap_tokens: Some(128),
            selection_gap_tokens: Some(0),
        }));

        emit_request_end(
            "raw-complete".into(),
            &tracker,
            RequestReplayMetrics {
                trace_block_size: 12_288,
                input_length: 24_704,
                input_sequence_hashes: vec![11, 22, 33],
            },
        );
        let record = loop {
            let record = rx.recv().await.unwrap();
            if record.request.as_ref().is_some_and(|r| r.request_id == "raw-complete") {
                break record;
            }
        };
        let value = serde_json::to_value(record).unwrap();
        assert_eq!(value["request"]["backend_actual_cached_tokens"], 24_704);
        let raw = &value["request"]["routing_decision"]["raw_cache_coverage"];
        assert_eq!(raw["schema"], "dynamo.router.raw_cache_coverage.v1");
        assert_eq!(raw["input_tokens"], 24_704);
        assert_eq!(raw["selected"]["worker_id"], u64::MAX);
        assert_eq!(raw["selected"]["cpu_extension_tokens"], 12_288);
        assert_eq!(raw["overload_gap_tokens"], 128);

        let missing_tracker = RequestTracker::new();
        missing_tracker.record_isl(24_704, Some(0));
        missing_tracker.record_routing_decision_trace(routing_trace_with_raw(RawCacheCoverage {
            schema: "dynamo.router.raw_cache_coverage.v1".into(),
            basis: "router_index".into(),
            observation: RawCacheObservation::MissingIndex,
            input_tokens: 24_704,
            resident: None,
            eligible: None,
            selected: None,
            overload_gap_tokens: None,
            selection_gap_tokens: None,
        }));
        emit_request_end(
            "raw-missing".into(),
            &missing_tracker,
            RequestReplayMetrics {
                trace_block_size: 12_288,
                input_length: 24_704,
                input_sequence_hashes: vec![11, 22, 33],
            },
        );
        let missing_record = loop {
            let record = rx.recv().await.unwrap();
            if record.request.as_ref().is_some_and(|r| r.request_id == "raw-missing") {
                break record;
            }
        };
        let missing_value = serde_json::to_value(missing_record).unwrap();
        let missing_raw =
            &missing_value["request"]["routing_decision"]["raw_cache_coverage"];
        assert_eq!(missing_raw["observation"], "missing_index");
        assert!(missing_raw["resident"].is_null());
        assert!(missing_raw["overload_gap_tokens"].is_null());
    }

    #[test]
    fn rejects_non_finite_tool_duration() {
        let mut record = RequestTraceRecord {
            schema: RequestTraceSchema::V1,
            event_type: RequestTraceEventType::ToolEnd,
            event_time_unix_ms: 2_000,
            event_source: Some(RequestTraceEventSource::Harness),
            agent_context: Some(AgentContext {
                session_id: "root".to_string(),
                parent_session_id: None,
                session_final: None,
                compaction: None,
                input_trigger: None,
            }),
            request: None,
            tool: Some(RequestTraceToolEvent {
                tool_call_id: "tool-1".to_string(),
                tool_class: "web_search".to_string(),
                started_at_unix_ms: None,
                ended_at_unix_ms: None,
                duration_ms: Some(f64::NAN),
                status: None,
                output_bytes: None,
                output_tokens: None,
                tool_name_hash: None,
                error_type: None,
            }),
            payload: None,
        };

        let err = validate_tool_record(&record).expect_err("NaN duration should fail validation");
        assert!(err.to_string().contains("tool duration_ms must be finite"));

        record.tool.as_mut().expect("tool payload").duration_ms = Some(f64::INFINITY);
        let err =
            validate_tool_record(&record).expect_err("infinite duration should fail validation");
        assert!(err.to_string().contains("tool duration_ms must be finite"));
    }
}
