// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Two-tier worker-selection cost function.
//!
//! Dynamo's built-in selector folds cache overlap and load into one additive cost. This policy
//! instead ranks on two tiers, taking the first that applies. For each eligible worker it reads
//! device-KV overlap and active-request count, then:
//!
//! 1. Load tier: if active-request spread exceeds `balance_abs_threshold` and the largest count
//!    exceeds `balance_rel_threshold` times the smallest, select the least-loaded worker.
//! 2. Cache tier: otherwise, if the largest device-KV overlap is strictly greater than
//!    `cache_threshold` of the request's block count, select the least-loaded worker holding that
//!    maximum overlap.
//! 3. Otherwise, select the least-loaded worker.
//!
//! Both load gates must hold to take step 1, so load displaces cache affinity only when the
//! imbalance is both large in absolute terms and disproportionate.
//!
//! The thresholds and selection order are the exact implementation ported from
//! `experimental/sgl-router`'s `cache_aware_zmq` policy, using Dynamo's authoritative device-KV
//! overlap instead of that router's own approximate cache history. The thresholds are exposed as
//! instance parameters defaulting to that router's values, so an instance with no `parameters`
//! mapping reproduces it exactly.
//!
//! Ties between equally ranked workers resolve on candidate row order, which the host leaves
//! unspecified. This matches the ported implementation; note that Dynamo's built-in selector
//! instead samples uniformly among ties.

use std::sync::Arc;

use dynamo_kv_router::services::selection::{
    WorkerSelectionPolicyFactory, WorkerSelectionPolicyParameters,
    WorkerSelectionPolicyProviderError, WorkerSelectionPolicyRegistry,
    WorkerSelectionPolicyRegistryError,
};
use dynamo_kv_router::{
    KvRouterConfig, WorkerCacheInput, WorkerInputView, WorkerInputs, WorkerLoadInput, WorkerPicker,
    WorkerSelectionContext, WorkerSelectionPolicy, WorkerSelectionPolicyError,
};

/// Policy type selected by `worker_selection.instances[].type`.
pub const POLICY_TYPE: &str = "dynamo-two-tier-cost-fn";

/// Keep these equal to `experimental/sgl-router`'s `cache_aware_zmq` defaults, so an instance
/// with no `parameters` mapping reproduces that policy exactly.
const DEFAULT_CACHE_THRESHOLD: f64 = 0.5;
const DEFAULT_BALANCE_ABS_THRESHOLD: usize = 32;
const DEFAULT_BALANCE_REL_THRESHOLD: f64 = 1.1;

/// Tunables for [`POLICY_TYPE`], named after their `sgl-router` counterparts.
///
/// Every field is optional and keeps the upstream default when omitted. Unknown keys are rejected
/// at startup rather than ignored, so a misremembered name fails loudly.
#[derive(Debug, Clone, Copy, serde::Deserialize)]
#[serde(deny_unknown_fields, default)]
struct Parameters {
    /// Fraction of the request's blocks that must be device-resident on the best worker before the
    /// cache tier applies. Compared strictly.
    cache_threshold: f64,
    /// Minimum active-request spread before the load tier applies.
    balance_abs_threshold: usize,
    /// Minimum ratio of largest to smallest active-request count before the load tier applies.
    balance_rel_threshold: f64,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            cache_threshold: DEFAULT_CACHE_THRESHOLD,
            balance_abs_threshold: DEFAULT_BALANCE_ABS_THRESHOLD,
            balance_rel_threshold: DEFAULT_BALANCE_REL_THRESHOLD,
        }
    }
}

impl Parameters {
    fn validate(&self) -> Result<(), WorkerSelectionPolicyProviderError> {
        if !self.cache_threshold.is_finite() || !(0.0..=1.0).contains(&self.cache_threshold) {
            return Err(WorkerSelectionPolicyProviderError::new(
                "cache_threshold must be a finite number between 0.0 and 1.0",
            ));
        }
        if !self.balance_rel_threshold.is_finite() || self.balance_rel_threshold < 1.0 {
            return Err(WorkerSelectionPolicyProviderError::new(
                "balance_rel_threshold must be a finite number greater than or equal to 1.0",
            ));
        }
        Ok(())
    }
}

fn least_loaded(load: &[WorkerLoadInput], rows: impl Iterator<Item = usize>) -> Option<usize> {
    rows.min_by_key(|&row| load[row].active_requests())
}

fn select_row(
    parameters: &Parameters,
    cache: &[WorkerCacheInput],
    load: &[WorkerLoadInput],
    request_blocks: u64,
) -> Option<usize> {
    select_row_with_evidence(parameters, cache, load, request_blocks, false).map(|(row, _)| row)
}

fn select_row_with_evidence(
    parameters: &Parameters,
    cache: &[WorkerCacheInput],
    load: &[WorkerLoadInput],
    request_blocks: u64,
    capture: bool,
) -> Option<(
    usize,
    Option<dynamo_kv_router::protocols::RoutingTwoTierRule>,
)> {
    if cache.is_empty() || cache.len() != load.len() {
        return None;
    }
    let min_load = load.iter().map(|item| item.active_requests()).min()?;
    let max_load = load.iter().map(|item| item.active_requests()).max()?;
    let absolute_load_gate = max_load.saturating_sub(min_load) > parameters.balance_abs_threshold;
    let relative_load_gate =
        (max_load as f64) > parameters.balance_rel_threshold * (min_load as f64);
    // Preserve the ordinary load-tier early return's work when tracing is off.
    let max_overlap = if capture || !(absolute_load_gate && relative_load_gate) {
        cache
            .iter()
            .map(|item| item.device_overlap_blocks())
            .max_by(f64::total_cmp)?
    } else {
        0.0
    };
    let cache_ratio = if request_blocks == 0 {
        0.0
    } else {
        max_overlap / request_blocks as f64
    };
    let cache_gate = cache_ratio > parameters.cache_threshold;
    let (row, rule) = if absolute_load_gate && relative_load_gate {
        (least_loaded(load, 0..load.len())?, "load_imbalance")
    } else if cache_gate {
        (
            least_loaded(
                load,
                cache.iter().enumerate().filter_map(|(row, item)| {
                    (item.device_overlap_blocks() == max_overlap).then_some(row)
                }),
            )?,
            "device_cache_affinity",
        )
    } else {
        (least_loaded(load, 0..load.len())?, "least_loaded_fallback")
    };
    let evidence = capture.then(|| dynamo_kv_router::protocols::RoutingTwoTierRule {
        cache_threshold: parameters.cache_threshold,
        balance_abs_threshold: parameters.balance_abs_threshold,
        balance_rel_threshold: parameters.balance_rel_threshold,
        request_blocks,
        min_active_requests: min_load,
        max_active_requests: max_load,
        max_device_overlap_blocks: max_overlap,
        cache_ratio,
        absolute_load_gate,
        relative_load_gate,
        cache_gate,
        selected_rule: rule.into(),
        selected_row: row,
        selected_device_overlap_blocks: cache[row].device_overlap_blocks(),
        selected_active_requests: load[row].active_requests(),
        tie_strategy: "first_candidate_row".into(),
        tie_count: (0..load.len())
            .filter(|&other| {
                load[other].active_requests() == load[row].active_requests()
                    && (rule != "device_cache_affinity"
                        || cache[other].device_overlap_blocks() == max_overlap)
            })
            .count(),
        candidate_count: load.len(),
        load_observed_candidate_count: None,
        selected_load_observed: None,
    });
    Some((row, evidence))
}

struct TwoTierCostFnPicker {
    parameters: Parameters,
}

impl WorkerPicker for TwoTierCostFnPicker {
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::CACHE | WorkerInputs::LOAD
    }

    fn pick(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        let cache = input
            .cache()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("cache input unavailable"))?;
        let load = input
            .load()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("load input unavailable"))?;
        select_row(&self.parameters, cache, load, context.request_blocks())
            .ok_or_else(|| WorkerSelectionPolicyError::failed("no eligible worker"))
    }
    fn pick_with_explanation(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<
        (
            usize,
            Option<dynamo_kv_router::protocols::RoutingDecisionExplanation>,
        ),
        WorkerSelectionPolicyError,
    > {
        let cache = input
            .cache()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("cache input unavailable"))?;
        let load = input
            .load()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("load input unavailable"))?;
        let (row, evidence) = select_row_with_evidence(
            &self.parameters,
            cache,
            load,
            context.request_blocks(),
            true,
        )
        .ok_or_else(|| WorkerSelectionPolicyError::failed("no eligible worker"))?;
        let mut explanation = dynamo_kv_router::protocols::RoutingDecisionExplanation::unavailable(
            POLICY_TYPE,
            input.candidates()[row].worker(),
            "",
        );
        explanation.kind = "two_tier_rule".into();
        explanation.reason = None;
        explanation.two_tier_rule = evidence;
        Ok((row, Some(explanation)))
    }
}

fn provider(
    parameters: &WorkerSelectionPolicyParameters,
) -> Result<WorkerSelectionPolicyFactory, WorkerSelectionPolicyProviderError> {
    let parameters: Parameters = parameters.deserialize()?;
    parameters.validate()?;

    Ok(Arc::new(
        move |config: &KvRouterConfig, worker_type, _partition| {
            WorkerSelectionPolicy::new(
                config.clone(),
                worker_type.as_str(),
                Vec::new(),
                Box::new(TwoTierCostFnPicker { parameters }),
            )
        },
    ))
}

pub fn register(
    registry: &mut WorkerSelectionPolicyRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    registry.register(POLICY_TYPE, Arc::new(provider))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use dynamo_kv_router::protocols::{RoutingConstraints, WorkerConfigLike, WorkerWithDpRank};
    use dynamo_kv_router::scheduling::{OverlapSignals, ScheduleMode};
    use dynamo_kv_router::{
        SchedulingRequest, WorkerLoadProjection, WorkerSelectionInput, WorkerSelector,
    };

    use super::*;

    const BLOCK_SIZE: u32 = 16;
    /// Ten blocks, so five overlapping blocks sit exactly on the 0.5 threshold.
    const TEN_BLOCKS: usize = 160;
    const A: u64 = 29;
    const B: u64 = 41;

    struct TestWorker;

    impl WorkerConfigLike for TestWorker {
        fn data_parallel_start_rank(&self) -> u32 {
            0
        }
        fn data_parallel_size(&self) -> u32 {
            1
        }
        fn max_num_batched_tokens(&self) -> Option<u64> {
            None
        }
        fn total_kv_blocks(&self) -> Option<u64> {
            Some(1024)
        }
    }

    fn worker(id: u64) -> WorkerWithDpRank {
        WorkerWithDpRank::from_worker_id(id)
    }

    /// Select among workers given as `(worker_id, device_overlap_blocks, active_requests)`.
    fn select(workers: [(u64, usize, usize); 2]) -> WorkerWithDpRank {
        select_with(Parameters::default(), workers)
    }

    fn select_with(parameters: Parameters, workers: [(u64, usize, usize); 2]) -> WorkerWithDpRank {
        select_result(parameters, workers).worker
    }

    fn select_result(
        parameters: Parameters,
        workers: [(u64, usize, usize); 2],
    ) -> dynamo_kv_router::protocols::WorkerSelectionResult {
        select_configured(parameters, workers, |_| {})
    }

    fn select_configured(
        parameters: Parameters,
        workers: [(u64, usize, usize); 2],
        configure: impl FnOnce(&mut SchedulingRequest),
    ) -> dynamo_kv_router::protocols::WorkerSelectionResult {
        select_with_policy(
            workers,
            configure,
            WorkerSelectionPolicy::new(
                KvRouterConfig::default(),
                "test",
                Vec::new(),
                Box::new(TwoTierCostFnPicker { parameters }),
            ),
        )
    }

    fn select_with_policy(
        workers: [(u64, usize, usize); 2],
        configure: impl FnOnce(&mut SchedulingRequest),
        policy: WorkerSelectionPolicy,
    ) -> dynamo_kv_router::protocols::WorkerSelectionResult {
        let mut request = SchedulingRequest {
            mode: ScheduleMode::QueryOnly { request_id: None },
            token_seq: None,
            isl_tokens: TEN_BLOCKS,
            lora_name: None,
            expected_output_tokens: None,
            affinity_target: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: RoutingConstraints::default(),
            router_config_override: None,
            track_prefill_tokens: true,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_context: None,
            overlap: OverlapSignals::default(),
            kv_transfer_candidates: None,
            retain_kv_transfer_chain: false,
            shared_cache_hits: None,
            worker_loads: Default::default(),
            resp_tx: None,
        };
        for (id, overlap_blocks, active_requests) in workers {
            request
                .overlap
                .tier_overlap_blocks
                .device
                .insert(worker(id), overlap_blocks);
            request.worker_loads.insert(
                worker(id),
                WorkerLoadProjection {
                    active_requests,
                    ..Default::default()
                },
            );
        }
        configure(&mut request);
        let configs = HashMap::from(workers.map(|(id, _, _)| (id, TestWorker)));
        policy
            .select_worker(WorkerSelectionInput::configured(
                &configs,
                &request,
                request.eligibility(),
                BLOCK_SIZE,
            ))
            .unwrap()
    }

    #[test]
    fn configured_policy_identity_and_resolved_parameters_follow_registry_activation() {
        let mut identities = Vec::new();
        for (threshold, expected) in [(0.2, B), (0.4, A)] {
            let file = tempfile::NamedTempFile::new().unwrap();
            std::fs::write(file.path(), format!("worker_selection:\n  aggregated: tuned\n  instances:\n    - name: tuned\n      type: dynamo-two-tier-cost-fn\n      parameters:\n        cache_threshold: {threshold}\n        balance_abs_threshold: 7\n        balance_rel_threshold: 1.5\n")).unwrap();
            let config = KvRouterConfig {
                router_policy_config: Some(file.path().display().to_string()),
                ..Default::default()
            };
            let mut registry = WorkerSelectionPolicyRegistry::default();
            register(&mut registry).unwrap();
            let factory = registry
                .resolve_for_worker_type(&config, dynamo_kv_router::WorkerType::Aggregated)
                .unwrap()
                .unwrap();
            let policy = factory(
                &config,
                dynamo_kv_router::WorkerType::Aggregated,
                dynamo_kv_router::RoutingPartitionRef::new("synthetic-model", "default"),
            );
            let result = select_with_policy([(A, 0, 0), (B, 3, 4)], |_| {}, policy);
            assert_eq!(result.worker, worker(expected));
            if dynamo_kv_router::protocols::routing_decision_trace_enabled() {
                let explanation = result.decision_explanation.unwrap();
                assert_eq!(explanation.policy_instance.as_deref(), Some("tuned"));
                assert_eq!(explanation.policy_type, POLICY_TYPE);
                let identity = explanation.configuration_identity.as_ref().unwrap();
                assert!(identity.starts_with("blake3:"));
                identities.push(identity.clone());
                let evidence = explanation.two_tier_rule.as_ref().unwrap();
                assert_eq!(evidence.cache_threshold, threshold);
                assert_eq!(evidence.balance_abs_threshold, 7);
                assert_eq!(evidence.balance_rel_threshold, 1.5);
                assert_eq!(evidence.candidate_count, 2);
                assert_eq!(evidence.load_observed_candidate_count, Some(2));
                println!(
                    "DECISION_EXPLANATION_FIXTURE={}",
                    serde_json::to_string(&explanation).unwrap()
                );
            }
        }
        if !identities.is_empty() {
            assert_ne!(identities[0], identities[1]);
        }
    }

    #[test]
    fn decision_explanation_records_strict_gates_and_first_row_ties() {
        // These are real host selections, including input materialization.
        for (rows, rule) in [
            ([(A, 0, 0), (B, 10, 32)], "device_cache_affinity"),
            ([(A, 0, 0), (B, 10, 33)], "load_imbalance"),
            ([(A, 0, 640), (B, 10, 704)], "device_cache_affinity"),
            ([(A, 0, 640), (B, 10, 705)], "load_imbalance"),
            ([(A, 0, 0), (B, 5, 4)], "least_loaded_fallback"),
        ] {
            let result = select_result(Parameters::default(), rows);
            if !dynamo_kv_router::protocols::routing_decision_trace_enabled() {
                assert!(result.decision_explanation.is_none());
                assert!(result.score_decision.is_none());
                continue;
            }
            let explanation = result.decision_explanation.expect("decision explanation");
            assert_eq!(explanation.kind, "two_tier_rule");
            assert_eq!(explanation.policy_type, POLICY_TYPE);
            assert_eq!(explanation.selected_worker, result.worker);
            let json = serde_json::to_string(&explanation).unwrap();
            let decoded: dynamo_kv_router::protocols::RoutingDecisionExplanation =
                serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.selected_worker, result.worker);
            println!("DECISION_EXPLANATION_FIXTURE={json}");
            let evidence = explanation.two_tier_rule.unwrap();
            assert_eq!(evidence.selected_rule, rule);
            assert_eq!(evidence.request_blocks, 10);
            assert_eq!(evidence.cache_threshold, 0.5);
            assert_eq!(evidence.balance_abs_threshold, 32);
            assert_eq!(evidence.balance_rel_threshold, 1.1);
            assert_eq!(evidence.tie_strategy, "first_candidate_row");
            assert!(result.score_decision.is_none());
        }
        let result = select_result(Parameters::default(), [(A, 6, 3), (B, 6, 3)]);
        if !dynamo_kv_router::protocols::routing_decision_trace_enabled() {
            assert!(result.decision_explanation.is_none());
            return;
        }
        let evidence = result.decision_explanation.unwrap().two_tier_rule.unwrap();
        assert_eq!(evidence.tie_count, 2);
        assert_eq!(evidence.selected_row, 0);
    }

    #[test]
    fn decision_explanation_cpu_overlap_does_not_change_policy_and_pin_is_real_input() {
        let rows = [(A, 0, 0), (B, 6, 4)];
        let selected = select_configured(Parameters::default(), rows, |request| {
            request
                .overlap
                .tier_overlap_blocks
                .host_pinned
                .insert(worker(A), 10);
        });
        assert_eq!(selected.worker, worker(B));
        let pinned = select_configured(Parameters::default(), rows, |request| {
            request.pinned_worker = Some(worker(A));
        });
        assert_eq!(pinned.worker, worker(A));
        if dynamo_kv_router::protocols::routing_decision_trace_enabled() {
            let evidence = selected
                .decision_explanation
                .unwrap()
                .two_tier_rule
                .unwrap();
            assert_eq!(evidence.selected_rule, "device_cache_affinity");
            assert_eq!(evidence.max_device_overlap_blocks, 6.0);
            let evidence = pinned.decision_explanation.unwrap().two_tier_rule.unwrap();
            assert_eq!(evidence.selected_rule, "least_loaded_fallback");
            assert_eq!(evidence.min_active_requests, 0);
            assert_eq!(evidence.max_active_requests, 0);
        }
    }

    #[test]
    fn zero_request_blocks_and_empty_inputs_have_explicit_policy_behavior() {
        assert_eq!(select_row(&Parameters::default(), &[], &[], 0), None);
        assert_eq!(
            select_row(
                &Parameters::default(),
                &[WorkerCacheInput::default()],
                &[],
                1
            ),
            None
        );
        assert_eq!(
            select_row(
                &Parameters::default(),
                &[WorkerCacheInput::default()],
                &[WorkerLoadInput::default()],
                0
            ),
            Some(0)
        );
    }

    #[test]
    fn cache_tier_outranks_a_less_loaded_worker() {
        // Six of ten blocks is 0.6, above the 0.5 threshold, so B wins despite carrying more load.
        assert_eq!(select([(A, 0, 0), (B, 6, 4)]), worker(B));
    }

    #[test]
    fn cache_tier_threshold_is_strict() {
        // Five of ten blocks is exactly 0.5, so the comparison fails and load decides.
        assert_eq!(select([(A, 0, 0), (B, 5, 4)]), worker(A));
    }

    #[test]
    fn parameters_override_the_upstream_defaults() {
        // Three of ten blocks is 0.3: below the 0.5 default, above a tuned 0.2 threshold.
        let workers = [(A, 0, 0), (B, 3, 4)];
        assert_eq!(select(workers), worker(A));

        let tuned = Parameters {
            cache_threshold: 0.2,
            ..Parameters::default()
        };
        assert_eq!(select_with(tuned, workers), worker(B));
    }

    #[test]
    fn rejects_out_of_range_parameters() {
        let cache = |v| {
            Parameters {
                cache_threshold: v,
                ..Default::default()
            }
            .validate()
        };
        let ratio = |v| {
            Parameters {
                balance_rel_threshold: v,
                ..Default::default()
            }
            .validate()
        };

        assert!(cache(-0.1).is_err() && cache(1.1).is_err() && cache(f64::NAN).is_err());
        assert!(ratio(0.9).is_err() && ratio(f64::NAN).is_err());
        assert!(Parameters::default().validate().is_ok());
    }

    #[test]
    fn load_tier_needs_both_gates() {
        // Spread 40 > 32 and 40 > 1.1 * 0: the load tier fires and ignores B's full overlap.
        assert_eq!(select([(A, 0, 0), (B, 10, 40)]), worker(A));
        // Spread 64 > 32, but 704 is not > 1.1 * 640, so the cache tier still decides. This pair
        // straddles the ratio boundary: 705 would clear it and take the load tier.
        assert_eq!(select([(A, 0, 640), (B, 10, 704)]), worker(B));
        assert_eq!(select([(A, 0, 640), (B, 10, 705)]), worker(A));
    }
}
