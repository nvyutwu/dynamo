// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Opt-in routing-decision trace: the candidate table behind one worker selection.
//!
//! Disabled by default. When `DYN_ROUTER_DECISION_TRACE_ENABLED` is unset the selector reads one
//! cached flag per selection and builds nothing, so normal serving has no extra work or trace
//! volume.

use std::collections::HashMap;
use std::sync::LazyLock;

use dynamo_truthy::env_is_truthy;

use super::{
    MaterializedSelectionInput, ScoredWorkerCandidate, WorkerInputView, WorkerInputs, WorkerPicker,
};
use crate::protocols::{
    RoutingDecisionCandidate, RoutingDecisionParameter, RoutingDecisionTrace, WorkerConfigLike,
    WorkerId, WorkerWithDpRank,
};
use crate::scheduling::types::SchedulingRequest;

const DECISION_TRACE_ENABLED_ENV: &str = "DYN_ROUTER_DECISION_TRACE_ENABLED";
const DECISION_TRACE_SAMPLE_RATE_ENV: &str = "DYN_ROUTER_DECISION_TRACE_SAMPLE_RATE";

static ROUTER_DECISION_TRACE_ENABLED: LazyLock<bool> =
    LazyLock::new(|| env_is_truthy(DECISION_TRACE_ENABLED_ENV));

pub(crate) fn router_decision_trace_enabled() -> bool {
    *ROUTER_DECISION_TRACE_ENABLED
}

pub(crate) fn router_decision_trace_sample_rate() -> f64 {
    static SAMPLE_RATE: LazyLock<f64> = LazyLock::new(|| {
        let Some(value) = std::env::var_os(DECISION_TRACE_SAMPLE_RATE_ENV) else {
            return 1.0;
        };
        let value = value.to_string_lossy();
        match value.parse::<f64>() {
            Ok(rate) if rate.is_finite() && (0.0..=1.0).contains(&rate) => rate,
            _ => {
                tracing::warn!(
                    value = %value,
                    "Ignoring invalid DYN_ROUTER_DECISION_TRACE_SAMPLE_RATE; expected 0 through 1"
                );
                0.0
            }
        }
    });
    *SAMPLE_RATE
}

/// Deterministic per-request sampling, so every router hop of one request agrees.
pub(super) fn sampled_request(request_id: &str, sample_rate: f64) -> bool {
    if sample_rate <= 0.0 {
        return false;
    }
    if sample_rate >= 1.0 {
        return true;
    }
    let sample = xxhash_rust::xxh3::xxh3_64(request_id.as_bytes()) as f64 / u64::MAX as f64;
    sample < sample_rate
}

/// Build the trace for one composed-policy selection. Called only after a pick, only when
/// the flag is on, and never for a pinned selection (the host, not the policy, chose it).
///
/// Every candidate's cache and load inputs are read from the request snapshot the policy
/// scored on, whatever inputs the picker itself declared. `total_cost_blocks` is the policy's
/// summed scorer cost for that row. The picker explains itself through
/// [`WorkerPicker::explain_pick`]: policy id, branch taken, resolved parameters and,
/// optionally, its own per-row overlap ranking and oracle row.
#[allow(clippy::too_many_arguments)]
pub(super) fn decision_trace_for_selection<C: WorkerConfigLike>(
    worker_type: &'static str,
    input: &MaterializedSelectionInput<'_>,
    workers: &HashMap<WorkerId, C>,
    request: &SchedulingRequest,
    picker: &dyn WorkerPicker,
    view: WorkerInputView<'_>,
    row: usize,
    sample_rate: f64,
) -> Option<RoutingDecisionTrace> {
    let request_id = request.mode.request_id()?;
    if !sampled_request(request_id, sample_rate) {
        return None;
    }
    let scored = view.candidates();
    let selected = scored.get(row)?.worker;
    let explanation = picker.explain_pick(&input.context, view, row);
    let ranked_overlap = explanation
        .as_ref()
        .filter(|explanation| explanation.row_overlap.len() == scored.len())
        .map(|explanation| explanation.row_overlap.as_slice());
    let mut candidates: Vec<RoutingDecisionCandidate> = scored
        .iter()
        .enumerate()
        .map(|(index, candidate)| {
            let mut traced = decision_candidate(input, workers, request, candidate, index == row);
            if let Some(ranked) = ranked_overlap {
                traced.effective_overlap_blocks = ranked[index];
            }
            traced
        })
        .collect();
    let selected_overlap = candidates[row].effective_overlap_blocks;
    let max_row = explanation
        .as_ref()
        .and_then(|explanation| explanation.max_overlap_row)
        .filter(|max_row| *max_row < candidates.len())
        .or_else(|| {
            (0..candidates.len()).max_by(|a, b| {
                candidates[*a]
                    .effective_overlap_blocks
                    .total_cmp(&candidates[*b].effective_overlap_blocks)
            })
        })?;
    let max_overlap_worker = scored[max_row].worker;
    let max_overlap_blocks = candidates[max_row].effective_overlap_blocks;
    candidates[max_row].max_overlap = true;
    candidates.sort_unstable_by_key(|candidate| (candidate.worker_id, candidate.dp_rank));
    let (policy, selection_reason, parameters) = match explanation {
        Some(explanation) => (
            explanation.policy,
            explanation.reason,
            explanation.parameters,
        ),
        None => (
            "custom".to_string(),
            "custom_picker".to_string(),
            Vec::new(),
        ),
    };
    let parameter = |names: &[&str]| {
        parameters
            .iter()
            .find(|(name, _)| names.contains(&name.as_str()))
            .map(|(_, value)| *value)
    };
    let weight = |names: &[&str]| parameter(names).unwrap_or(0.0);
    // A policy that reports the temperature it applied wins over the request override,
    // which composed policies may ignore.
    let router_temperature = parameter(&["router_temperature"])
        .or(input.context.router_temperature_override)
        .unwrap_or(0.0);
    Some(RoutingDecisionTrace {
        schema: "dynamo.router.decision.v1".to_string(),
        worker_type: worker_type.to_owned(),
        policy,
        selection_reason,
        candidate_scope: "eligible_workers_only".to_string(),
        block_size: input.context.block_size,
        request_blocks: input.context.request_blocks,
        track_prefill_tokens: input.context.track_prefill_tokens,
        selected_worker_id: selected.worker_id,
        selected_dp_rank: selected.dp_rank,
        max_overlap_worker_id: max_overlap_worker.worker_id,
        max_overlap_dp_rank: max_overlap_worker.dp_rank,
        avoidable_prefill_token_equivalents: (max_overlap_blocks - selected_overlap)
            * input.context.block_size as f64,
        overlap_score_credit: weight(&["overlap_score_credit"]),
        overlap_score_credit_decay: weight(&["overlap_score_credit_decay"]),
        prefill_load_scale: weight(&["prefill_load_scale"]),
        host_cache_hit_weight: weight(&["host_cache_hit_weight", "host_cache_weight"]),
        disk_cache_hit_weight: weight(&["disk_cache_hit_weight"]),
        shared_cache_multiplier: weight(&["shared_cache_multiplier"]),
        decode_active_request_weight: weight(&["decode_active_request_weight"]),
        router_temperature,
        policy_parameters: parameters
            .into_iter()
            .map(|(name, value)| RoutingDecisionParameter { name, value })
            .collect(),
        candidates,
    })
}

fn decision_candidate<C: WorkerConfigLike>(
    input: &MaterializedSelectionInput<'_>,
    workers: &HashMap<WorkerId, C>,
    request: &SchedulingRequest,
    candidate: &ScoredWorkerCandidate,
    selected: bool,
) -> RoutingDecisionCandidate {
    let worker: WorkerWithDpRank = candidate.worker;
    let row = input.row(worker, None, WorkerInputs::CACHE | WorkerInputs::LOAD);
    let shared_beyond_device_blocks = input.cache_snapshot.shared_hits.map_or(0, |hits| {
        hits.hits_beyond(row.cache.device_overlap_blocks.round().max(0.0) as u32)
    });
    let preferred_taint_multiplier = workers.get(&worker.worker_id).and_then(|config| {
        request
            .routing_constraints
            .preferred_taint_multiplier(config.taints())
    });
    RoutingDecisionCandidate {
        worker_id: worker.worker_id,
        dp_rank: worker.dp_rank,
        eligible: true,
        selected,
        max_overlap: false,
        total_cost_blocks: candidate.cost,
        effective_overlap_blocks: row.cache.effective_overlap_blocks,
        device_overlap_blocks: row.cache.device_overlap_blocks,
        host_overlap_blocks: row.cache.host_overlap_blocks,
        disk_overlap_blocks: row.cache.disk_overlap_blocks,
        shared_beyond_device_blocks,
        raw_prefill_blocks: 0.0,
        active_prefill_tokens: row.load.active_prefill_tokens,
        prefill_cost_blocks: 0.0,
        decode_cost_blocks: row.load.decode_cost_blocks,
        active_requests: row.load.active_requests,
        active_request_cost_blocks: 0.0,
        overlap_credit_blocks: 0.0,
        overlap_credit_decay: 1.0,
        effective_overlap_score_credit: 0.0,
        adjusted_prefill_blocks: 0.0,
        base_score_blocks: candidate.cost,
        preferred_taint_multiplier,
        decode_overlap_formula: false,
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::base_request;
    use super::*;
    use crate::KvRouterConfig;
    use crate::plugins::worker_selection::{WorkerSelectionContext, WorkerSelectionPolicyError};
    use crate::scheduling::selector::{
        WorkerSelectionInput, WorkerSelectionPolicy, WorkerSelector,
    };
    use crate::test_utils::SimpleWorkerConfig;

    fn view(rows: &[ScoredWorkerCandidate]) -> WorkerInputView<'_> {
        WorkerInputView {
            candidates: rows,
            cache: None,
            load: None,
        }
    }

    fn scored(worker_id: u64, cost: f64) -> ScoredWorkerCandidate {
        ScoredWorkerCandidate {
            worker: WorkerWithDpRank::from_worker_id(worker_id),
            cost,
            preferred_taint_multiplier: None,
        }
    }

    /// Worker 1 holds the longest prefix (4 device + 2 host blocks); worker 2 was picked.
    fn request_with_overlap() -> SchedulingRequest {
        let mut request = base_request(128);
        let warm = WorkerWithDpRank::from_worker_id(1);
        let cold = WorkerWithDpRank::from_worker_id(2);
        request.overlap.tier_overlap_blocks.device.insert(warm, 4);
        request
            .overlap
            .tier_overlap_blocks
            .host_pinned
            .insert(warm, 2);
        request.overlap.effective_overlap_blocks.insert(warm, 5.0);
        request.overlap.effective_overlap_blocks.insert(cold, 1.0);
        request.overlap.tier_overlap_blocks.device.insert(cold, 1);
        request.worker_loads.insert(
            warm,
            crate::sequences::WorkerLoadProjection {
                active_requests: 3,
                active_prefill_tokens: 64,
                ..Default::default()
            },
        );
        request.worker_loads.insert(
            cold,
            crate::sequences::WorkerLoadProjection {
                active_requests: 1,
                ..Default::default()
            },
        );
        request
    }

    fn workers() -> HashMap<WorkerId, SimpleWorkerConfig> {
        HashMap::from([
            (1, SimpleWorkerConfig::default()),
            (2, SimpleWorkerConfig::default()),
        ])
    }

    #[test]
    fn sampling_is_deterministic_and_bounded() {
        assert!(!sampled_request("req", 0.0));
        assert!(sampled_request("req", 1.0));
        let first = sampled_request("req-sampled", 0.5);
        assert_eq!(first, sampled_request("req-sampled", 0.5));
    }

    #[test]
    fn trace_reports_inputs_selected_and_max_overlap_rows() {
        let request = request_with_overlap();
        let workers = workers();
        let input = MaterializedSelectionInput::new(&request, 16);
        // Rows deliberately out of worker order: the trace sorts by worker id.
        let rows = [scored(2, 1.5), scored(1, 7.0)];
        let trace = decision_trace_for_selection(
            "aggregated",
            &input,
            &workers,
            &request,
            &FirstRow,
            view(&rows),
            0,
            1.0,
        )
        .expect("full sampling with a request id keeps the trace");

        assert_eq!(trace.selected_worker_id, 2);
        assert_eq!(trace.max_overlap_worker_id, 1);
        assert_eq!(trace.request_blocks, 8);
        // (5 - 1) effective blocks the router chose not to reuse, in tokens.
        assert_eq!(trace.avoidable_prefill_token_equivalents, 64.0);
        assert_eq!(
            trace
                .candidates
                .iter()
                .map(|candidate| candidate.worker_id)
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
        let warm = &trace.candidates[0];
        let cold = &trace.candidates[1];
        assert!(warm.max_overlap && !warm.selected);
        assert!(cold.selected && !cold.max_overlap);
        assert_eq!(
            (warm.device_overlap_blocks, warm.host_overlap_blocks),
            (4.0, 2.0)
        );
        assert_eq!(warm.effective_overlap_blocks, 5.0);
        assert_eq!((warm.active_requests, warm.active_prefill_tokens), (3, 64));
        assert_eq!(warm.total_cost_blocks, 7.0);
        assert_eq!(cold.total_cost_blocks, 1.5);
    }

    #[test]
    fn trace_needs_a_request_id_and_a_sample() {
        let mut request = request_with_overlap();
        let workers = workers();
        let rows = [scored(1, 1.0), scored(2, 2.0)];
        {
            let input = MaterializedSelectionInput::new(&request, 16);
            assert!(
                decision_trace_for_selection(
                    "aggregated",
                    &input,
                    &workers,
                    &request,
                    &FirstRow,
                    view(&rows),
                    0,
                    0.0,
                )
                .is_none()
            );
        }
        request.mode = crate::scheduling::ScheduleMode::QueryOnly { request_id: None };
        let input = MaterializedSelectionInput::new(&request, 16);
        assert!(
            decision_trace_for_selection(
                "aggregated",
                &input,
                &workers,
                &request,
                &FirstRow,
                view(&rows),
                0,
                1.0,
            )
            .is_none()
        );
    }

    struct FirstRow;

    impl WorkerPicker for FirstRow {
        fn pick(
            &mut self,
            _context: &WorkerSelectionContext<'_>,
            _input: WorkerInputView<'_>,
        ) -> Result<usize, WorkerSelectionPolicyError> {
            Ok(0)
        }
    }

    struct Explaining;

    impl WorkerPicker for Explaining {
        fn pick(
            &mut self,
            _context: &WorkerSelectionContext<'_>,
            _input: WorkerInputView<'_>,
        ) -> Result<usize, WorkerSelectionPolicyError> {
            Ok(0)
        }

        fn explain_pick(
            &self,
            _context: &WorkerSelectionContext<'_>,
            _input: WorkerInputView<'_>,
            row: usize,
        ) -> Option<crate::plugins::worker_selection::PickExplanation> {
            Some(crate::plugins::worker_selection::PickExplanation {
                policy: "test-policy".to_string(),
                reason: format!("picked_row_{row}"),
                // The policy ranks worker 2 (row 0) above worker 1 (row 1).
                max_overlap_row: Some(0),
                row_overlap: vec![6.5, 4.0],
                parameters: vec![
                    ("host_cache_weight".to_string(), 0.75),
                    ("cache_ratio".to_string(), 0.8125),
                ],
            })
        }
    }

    #[test]
    fn picker_explanation_sets_policy_ranking_and_parameters() {
        let request = request_with_overlap();
        let workers = workers();
        let input = MaterializedSelectionInput::new(&request, 16);
        let rows = [scored(2, 0.0), scored(1, 0.0)];
        let trace = decision_trace_for_selection(
            "aggregated",
            &input,
            &workers,
            &request,
            &Explaining,
            view(&rows),
            0,
            1.0,
        )
        .unwrap();
        assert_eq!(trace.policy, "test-policy");
        assert_eq!(trace.selection_reason, "picked_row_0");
        assert_eq!(trace.max_overlap_worker_id, 2);
        assert_eq!(trace.avoidable_prefill_token_equivalents, 0.0);
        assert_eq!(trace.host_cache_hit_weight, 0.75);
        assert_eq!(trace.policy_parameters.len(), 2);
        let selected = trace.candidates.iter().find(|c| c.worker_id == 2).unwrap();
        assert!(selected.selected && selected.max_overlap);
        assert_eq!(selected.effective_overlap_blocks, 6.5);
        // Raw tier inputs are still the request snapshot's.
        assert_eq!(selected.device_overlap_blocks, 1.0);
    }

    #[test]
    fn unexplained_pick_omits_policy_parameters_from_json() {
        let request = request_with_overlap();
        let workers = workers();
        let input = MaterializedSelectionInput::new(&request, 16);
        let rows = [scored(1, 0.0), scored(2, 0.0)];
        let trace = decision_trace_for_selection(
            "aggregated",
            &input,
            &workers,
            &request,
            &FirstRow,
            view(&rows),
            0,
            1.0,
        )
        .unwrap();
        assert_eq!(
            (trace.policy.as_str(), trace.selection_reason.as_str()),
            ("custom", "custom_picker")
        );
        let value = serde_json::to_value(&trace).unwrap();
        assert!(value.get("policy_parameters").is_none());
    }

    /// The flag is a process-wide `LazyLock` and this crate's unit tests never set it, so a
    /// real selection must carry no trace and build nothing.
    #[test]
    fn selection_without_the_flag_carries_no_trace() {
        assert!(!router_decision_trace_enabled());
        let request = request_with_overlap();
        let workers = workers();
        let policy = WorkerSelectionPolicy::new(
            KvRouterConfig::default(),
            "aggregated",
            Vec::new(),
            Box::new(FirstRow),
        );
        let result = policy
            .select_worker(WorkerSelectionInput::configured(
                &workers,
                &request,
                request.eligibility(),
                16,
            ))
            .unwrap();
        assert!(result.decision_trace.is_none());
    }
}
