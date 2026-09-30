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
//! 2. Cache tier: otherwise, if the largest *effective* KV overlap is strictly greater than
//!    `cache_threshold` of the request's block count, select the least-loaded worker holding that
//!    maximum overlap. Effective overlap is device-resident blocks plus host-pinned (CPU offload)
//!    blocks scaled by `host_cache_weight`, so a worker holding the prefix in CPU can win the cache
//!    tier over one holding nothing, while still losing to an equal device-resident hit.
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
//! `host_cache_weight` defaults to [`KvRouterConfig::host_cache_hit_weight`], which
//! `DYN_ROUTER_HOST_CACHE_HIT_WEIGHT` sets (0.75 by default) and which Dynamo's built-in selector
//! already applies to the same quantity — so the two tiers agree on what a CPU hit is worth unless
//! an instance deliberately overrides it. Set the weight to 0.0 to restore device-only ranking.
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
    KvRouterConfig, PickExplanation, WorkerCacheInput, WorkerInputView, WorkerInputs,
    WorkerLoadInput, WorkerPicker, WorkerSelectionContext, WorkerSelectionPolicy,
    WorkerSelectionPolicyError,
};

/// Policy type selected by `worker_selection.instances[].type`.
pub const POLICY_TYPE: &str = "dynamo-two-tier-cost-fn";

/// Keep these equal to `experimental/sgl-router`'s `cache_aware_zmq` defaults, so an instance
/// with no `parameters` mapping reproduces that policy exactly.
const DEFAULT_CACHE_THRESHOLD: f64 = 0.5;
const DEFAULT_BALANCE_ABS_THRESHOLD: usize = 32;
const DEFAULT_BALANCE_REL_THRESHOLD: f64 = 1.1;
const DEFAULT_RESPECT_SOFT_AFFINITY: bool = false;
const DEFAULT_SOFT_AFFINITY_LOAD_GATE: bool = false;

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
    /// Weight applied to host-pinned (CPU offload) overlap when ranking cache affinity.
    ///
    /// `None` — inherit `KvRouterConfig::host_cache_hit_weight`, i.e. whatever
    /// `DYN_ROUTER_HOST_CACHE_HIT_WEIGHT` is set to. Set explicitly only when this policy must
    /// value CPU residency differently from the built-in selector.
    host_cache_weight: Option<f64>,
    /// Whether to retain an eligible soft-affinity target before considering other workers.
    respect_soft_affinity: bool,
    /// With `respect_soft_affinity`, release the target when its load fails the balance gate
    /// against the least-loaded eligible row, so the load tier can place the request elsewhere.
    soft_affinity_load_gate: bool,
    /// Absolute active-request gap the load gate uses. `None` — inherit `balance_abs_threshold`.
    soft_affinity_gate_abs: Option<usize>,
    /// Load ratio the load gate uses. `None` — inherit `balance_rel_threshold`.
    soft_affinity_gate_rel: Option<f64>,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            cache_threshold: DEFAULT_CACHE_THRESHOLD,
            balance_abs_threshold: DEFAULT_BALANCE_ABS_THRESHOLD,
            balance_rel_threshold: DEFAULT_BALANCE_REL_THRESHOLD,
            host_cache_weight: None,
            respect_soft_affinity: DEFAULT_RESPECT_SOFT_AFFINITY,
            soft_affinity_load_gate: DEFAULT_SOFT_AFFINITY_LOAD_GATE,
            soft_affinity_gate_abs: None,
            soft_affinity_gate_rel: None,
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
        if let Some(rel) = self.soft_affinity_gate_rel
            && (!rel.is_finite() || rel < 1.0)
        {
            return Err(WorkerSelectionPolicyProviderError::new(
                "soft_affinity_gate_rel must be a finite number greater than or equal to 1.0",
            ));
        }
        if let Some(weight) = self.host_cache_weight
            && (!weight.is_finite() || weight < 0.0)
        {
            return Err(WorkerSelectionPolicyProviderError::new(
                "host_cache_weight must be a finite number greater than or equal to 0.0",
            ));
        }
        Ok(())
    }

    fn soft_affinity_gate_abs(&self) -> usize {
        self.soft_affinity_gate_abs
            .unwrap_or(self.balance_abs_threshold)
    }

    fn soft_affinity_gate_rel(&self) -> f64 {
        self.soft_affinity_gate_rel
            .unwrap_or(self.balance_rel_threshold)
    }
}

/// Both gates: `high` exceeds `low` by more than `abs` and by more than `rel` times.
fn exceeds(abs: usize, rel: f64, high: usize, low: usize) -> bool {
    high.saturating_sub(low) > abs && (high as f64) > rel * (low as f64)
}

/// The load tier's balance gates.
fn imbalanced(parameters: &Parameters, high: usize, low: usize) -> bool {
    exceeds(
        parameters.balance_abs_threshold,
        parameters.balance_rel_threshold,
        high,
        low,
    )
}

fn least_loaded(load: &[WorkerLoadInput], rows: impl Iterator<Item = usize>) -> Option<usize> {
    rows.min_by_key(|&row| load[row].active_requests())
}

/// Blocks this worker can reuse, counting CPU-offloaded blocks at `host_cache_weight`.
///
/// Device and host overlap are disjoint prefix measurements from the same indexer, so they add
/// rather than max. A weight of 0.0 reproduces the device-only ranking this policy shipped with.
fn effective_overlap(cache: &WorkerCacheInput, host_cache_weight: f64) -> f64 {
    cache.device_overlap_blocks() + host_cache_weight * cache.host_overlap_blocks()
}

/// Branch names reported in the routing-decision trace. Stable strings; dashboards key on them.
const REASON_LOAD_IMBALANCE: &str = "load_imbalance_least_loaded";
const REASON_CACHE_TIER: &str = "cache_tier_least_loaded_among_max_overlap";
const REASON_NO_CACHE_WINNER: &str = "least_loaded_no_cache_winner";
const REASON_SOFT_AFFINITY: &str = "soft_affinity_target";
const REASON_SOFT_AFFINITY_RELEASED: &str = "soft_affinity_released_load_gate";

/// What happened to the request's soft-affinity target, reported as `soft_affinity_outcome` in
/// the routing-decision trace so pin keep / release / drop rates can be counted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AffinityOutcome {
    /// No soft target, or `respect_soft_affinity` is off.
    None = 0,
    /// The target was eligible and kept.
    Kept = 1,
    /// The target was eligible but the load gate released it.
    ReleasedByGate = 2,
    /// The target was not among the eligible candidates (busy-threshold filtered or gone).
    NotEligible = 3,
}

/// One two-tier decision, with the quantities that determined it.
struct Decision {
    row: usize,
    reason: &'static str,
    /// Row holding the largest two-tier effective overlap (device + host_cache_weight * host).
    max_overlap_row: usize,
    max_overlap: f64,
    cache_ratio: f64,
    min_load: usize,
    max_load: usize,
    /// Two-tier effective overlap per input row, in blocks.
    row_overlap: Vec<f64>,
    affinity: AffinityOutcome,
    /// Active requests on the soft target's chosen row, when it was eligible.
    affinity_target_load: Option<usize>,
}

/// The two-tier decision restricted to rows where `include` holds; `|_| true` is the policy.
fn decide(
    parameters: &Parameters,
    cache: &[WorkerCacheInput],
    load: &[WorkerLoadInput],
    request_blocks: u64,
    host_cache_weight: f64,
    include: impl Fn(usize) -> bool + Copy,
) -> Option<Decision> {
    if cache.is_empty() || cache.len() != load.len() {
        return None;
    }

    let row_overlap: Vec<f64> = cache
        .iter()
        .map(|item| effective_overlap(item, host_cache_weight))
        .collect();
    let rows = || (0..load.len()).filter(move |&row| include(row));
    let max_overlap_row = rows().max_by(|a, b| row_overlap[*a].total_cmp(&row_overlap[*b]))?;
    let max_overlap = row_overlap[max_overlap_row];
    let cache_ratio = if request_blocks == 0 {
        0.0
    } else {
        max_overlap / request_blocks as f64
    };
    let min_load = rows().map(|row| load[row].active_requests()).min()?;
    let max_load = rows().map(|row| load[row].active_requests()).max()?;
    let decision = |row, reason| Decision {
        row,
        reason,
        max_overlap_row,
        max_overlap,
        cache_ratio,
        min_load,
        max_load,
        row_overlap: row_overlap.clone(),
        affinity: AffinityOutcome::None,
        affinity_target_load: None,
    };

    if imbalanced(parameters, max_load, min_load) {
        return least_loaded(load, rows()).map(|row| decision(row, REASON_LOAD_IMBALANCE));
    }
    if cache_ratio > parameters.cache_threshold {
        return least_loaded(load, rows().filter(|&row| row_overlap[row] == max_overlap))
            .map(|row| decision(row, REASON_CACHE_TIER));
    }
    least_loaded(load, rows()).map(|row| decision(row, REASON_NO_CACHE_WINNER))
}

struct TwoTierCostFnPicker {
    parameters: Parameters,
    /// Resolved once at construction: the instance override when given, else the router config's
    /// `host_cache_hit_weight`.
    host_cache_weight: f64,
}

impl TwoTierCostFnPicker {
    /// The policy's decision, shared by `pick` and `explain_pick` so the trace names the branch
    /// that chose the row. With `respect_soft_affinity`, an eligible soft target is retained by
    /// running the two-tier decision over the target's rows only; the load gate, when enabled,
    /// releases it once its load exceeds the least-loaded eligible row's by more than
    /// `soft_affinity_gate_abs` and `soft_affinity_gate_rel` (default: the load tier's thresholds). The oracle fields
    /// (`max_overlap_row`, cache ratio, load range) always describe the full candidate set.
    fn decide_for(
        &self,
        context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Option<Decision> {
        let (cache, load) = (input.cache()?, input.load()?);
        let blocks = context.request_blocks();
        let global = decide(
            &self.parameters,
            cache,
            load,
            blocks,
            self.host_cache_weight,
            |_| true,
        )?;
        if self.parameters.respect_soft_affinity
            && let Some(target) = context.affinity_target()
        {
            let candidates = input.candidates();
            let is_target = |row: usize| {
                let worker = candidates[row].worker();
                worker.worker_id == target.worker_id
                    && target.dp_rank.is_none_or(|rank| worker.dp_rank == rank)
            };
            let Some(kept) = decide(
                &self.parameters,
                cache,
                load,
                blocks,
                self.host_cache_weight,
                is_target,
            )
            .map(|d| d.row) else {
                return Some(Decision {
                    affinity: AffinityOutcome::NotEligible,
                    ..global
                });
            };
            let target_load = load[kept].active_requests();
            let released = self.parameters.soft_affinity_load_gate
                && exceeds(
                    self.parameters.soft_affinity_gate_abs(),
                    self.parameters.soft_affinity_gate_rel(),
                    target_load,
                    global.min_load,
                );
            return Some(if released {
                Decision {
                    reason: REASON_SOFT_AFFINITY_RELEASED,
                    affinity: AffinityOutcome::ReleasedByGate,
                    affinity_target_load: Some(target_load),
                    ..global
                }
            } else {
                Decision {
                    row: kept,
                    reason: REASON_SOFT_AFFINITY,
                    affinity: AffinityOutcome::Kept,
                    affinity_target_load: Some(target_load),
                    ..global
                }
            });
        }
        Some(global)
    }
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
        input
            .cache()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("cache input unavailable"))?;
        input
            .load()
            .ok_or_else(|| WorkerSelectionPolicyError::failed("load input unavailable"))?;
        self.decide_for(context, input)
            .map(|decision| decision.row)
            .ok_or_else(|| WorkerSelectionPolicyError::failed("no eligible worker"))
    }

    /// The routing-decision trace for this policy: which branch fired, the thresholds in force,
    /// the two-tier overlap ranking per candidate and the row it treats as the cache oracle.
    fn explain_pick(
        &self,
        context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
        row: usize,
    ) -> Option<PickExplanation> {
        let decision = self.decide_for(context, input)?;
        // `pick` and `explain_pick` see the same input, so the branch must agree with the row
        // that was actually chosen. If it does not, say so rather than report a fiction.
        let reason = if decision.row == row {
            decision.reason.to_string()
        } else {
            format!("{}_row_mismatch", decision.reason)
        };
        Some(PickExplanation {
            policy: POLICY_TYPE.to_string(),
            reason,
            max_overlap_row: Some(decision.max_overlap_row),
            row_overlap: decision.row_overlap,
            parameters: vec![
                ("cache_threshold".into(), self.parameters.cache_threshold),
                (
                    "balance_abs_threshold".into(),
                    self.parameters.balance_abs_threshold as f64,
                ),
                (
                    "balance_rel_threshold".into(),
                    self.parameters.balance_rel_threshold,
                ),
                ("host_cache_weight".into(), self.host_cache_weight),
                ("cache_ratio".into(), decision.cache_ratio),
                ("max_effective_overlap_blocks".into(), decision.max_overlap),
                ("min_active_requests".into(), decision.min_load as f64),
                ("max_active_requests".into(), decision.max_load as f64),
                (
                    "soft_affinity_gate_abs".into(),
                    self.parameters.soft_affinity_gate_abs() as f64,
                ),
                (
                    "soft_affinity_gate_rel".into(),
                    self.parameters.soft_affinity_gate_rel(),
                ),
                (
                    "soft_affinity_outcome".into(),
                    decision.affinity as u8 as f64,
                ),
                (
                    "soft_affinity_target_active_requests".into(),
                    decision.affinity_target_load.map_or(-1.0, |l| l as f64),
                ),
            ],
        })
    }
}

fn provider(
    parameters: &WorkerSelectionPolicyParameters,
) -> Result<WorkerSelectionPolicyFactory, WorkerSelectionPolicyProviderError> {
    let parameters: Parameters = parameters.deserialize()?;
    parameters.validate()?;

    // Announce the RESOLVED parameters, not the file contents: every field is optional and
    // silently keeps an upstream default when omitted, so the YAML says what was asked for and
    // this says what is actually in force.
    //
    // Why this line exists: nothing else in the router names the active worker-selection policy.
    // `Router policy class configured policy_class="default"` (queue.rs) is the QUEUEING profile
    // and reads "default" even when this policy is driving every routing decision. That has now
    // been misread as "two-tier did not load" by two separate investigations, once inside a
    // published A/B report. Grep for the messages below instead -- they are emitted only when
    // this policy is genuinely constructed.
    tracing::info!(
        policy_type = POLICY_TYPE,
        cache_threshold = parameters.cache_threshold,
        balance_abs_threshold = parameters.balance_abs_threshold,
        balance_rel_threshold = parameters.balance_rel_threshold,
        host_cache_weight = ?parameters.host_cache_weight,
        "Two-tier worker-selection policy enabled"
    );

    Ok(Arc::new(
        move |config: &KvRouterConfig, worker_type, _partition| {
            let host_cache_weight = parameters
                .host_cache_weight
                .unwrap_or(config.host_cache_hit_weight);
            // Logged per role, and with the SOURCE of the weight, because the cache tier ranks on
            //     device_blocks + host_cache_weight * host_blocks
            // and that weight has two possible origins: this policy's YAML `parameters`, or
            // DYN_ROUTER_HOST_CACHE_HIT_WEIGHT via KvRouterConfig. The YAML silently wins. It is
            // also the variable an A/B most often changes, so an experiment that sets the env var
            // while the YAML pins the field would otherwise compare two identical arms with no
            // indication anything was ignored.
            tracing::info!(
                policy_type = POLICY_TYPE,
                worker_type = worker_type.as_str(),
                host_cache_weight,
                host_cache_weight_source = if parameters.host_cache_weight.is_some() {
                    "policy_yaml"
                } else {
                    "DYN_ROUTER_HOST_CACHE_HIT_WEIGHT"
                },
                disk_cache_hit_weight = config.disk_cache_hit_weight,
                overlap_score_credit = config.overlap_score_credit,
                prefill_load_scale = config.prefill_load_scale,
                "Two-tier worker-selection policy instantiated"
            );
            WorkerSelectionPolicy::new(
                config.clone(),
                worker_type.as_str(),
                Vec::new(),
                Box::new(TwoTierCostFnPicker {
                    parameters,
                    host_cache_weight,
                }),
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

    use dynamo_kv_router::protocols::{
        RoutingConstraints, WorkerAffinityTarget, WorkerConfigLike, WorkerWithDpRank,
    };
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
        select_tiers(
            parameters,
            workers.map(|(id, device_blocks, active)| (id, device_blocks, 0, active)),
        )
    }

    /// Select among workers given as
    /// `(worker_id, device_blocks, host_pinned_blocks, active_requests)`.
    fn select_tiers(
        parameters: Parameters,
        workers: [(u64, usize, usize, usize); 2],
    ) -> WorkerWithDpRank {
        select_affine(parameters, workers, None)
    }

    /// `select_tiers` with a soft-affinity target on the request.
    fn select_affine(
        parameters: Parameters,
        workers: [(u64, usize, usize, usize); 2],
        affinity_target: Option<WorkerAffinityTarget>,
    ) -> WorkerWithDpRank {
        let mut request = SchedulingRequest {
            mode: ScheduleMode::QueryOnly { request_id: None },
            token_seq: None,
            isl_tokens: TEN_BLOCKS,
            lora_name: None,
            expected_output_tokens: None,
            affinity_target,
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
        for (id, device_blocks, host_blocks, active_requests) in workers {
            request
                .overlap
                .tier_overlap_blocks
                .device
                .insert(worker(id), device_blocks);
            request
                .overlap
                .tier_overlap_blocks
                .host_pinned
                .insert(worker(id), host_blocks);
            request.worker_loads.insert(
                worker(id),
                WorkerLoadProjection {
                    active_requests,
                    ..Default::default()
                },
            );
        }
        let configs = HashMap::from(workers.map(|(id, _, _, _)| (id, TestWorker)));
        // Resolve exactly as `provider` does, so the tests exercise the real defaulting path
        // rather than a hand-picked weight.
        let config = KvRouterConfig::default();
        let host_cache_weight = parameters
            .host_cache_weight
            .unwrap_or(config.host_cache_hit_weight);
        WorkerSelectionPolicy::new(
            config,
            "test",
            Vec::new(),
            Box::new(TwoTierCostFnPicker {
                parameters,
                host_cache_weight,
            }),
        )
        .select_worker(WorkerSelectionInput::configured(
            &configs,
            &request,
            request.eligibility(),
            BLOCK_SIZE,
        ))
        .unwrap()
        .worker
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
    fn host_overlap_can_win_the_cache_tier() {
        // B holds nothing on device but eight of ten blocks in CPU offload. At the inherited
        // weight of 0.75 that is an effective 6.0 blocks, a ratio of 0.6 above the 0.5 threshold,
        // so B wins despite carrying more load. Before this change B scored 0 and A took it.
        assert_eq!(
            select_tiers(Parameters::default(), [(A, 0, 0, 0), (B, 0, 8, 4)]),
            worker(B)
        );
    }

    #[test]
    fn host_cache_weight_zero_restores_device_only_ranking() {
        // The escape hatch: weight 0.0 reproduces the behaviour this policy shipped with, so a
        // deployment that does not want CPU residency to influence routing can turn it off
        // without reverting the code.
        let device_only = Parameters {
            host_cache_weight: Some(0.0),
            ..Parameters::default()
        };
        assert_eq!(
            select_tiers(device_only, [(A, 0, 0, 0), (B, 0, 8, 4)]),
            worker(A)
        );
    }

    #[test]
    fn device_blocks_outrank_the_same_count_of_host_blocks() {
        // Equal block counts, different tiers: A's six device blocks score 6.0 against B's six
        // host blocks at 4.5, so A wins even though B is idle and A carries four requests. A CPU
        // hit must never be treated as interchangeable with a device hit.
        assert_eq!(
            select_tiers(Parameters::default(), [(A, 6, 0, 4), (B, 0, 6, 0)]),
            worker(A)
        );
    }

    #[test]
    fn rejects_negative_host_cache_weight() {
        let weight = |v| {
            Parameters {
                host_cache_weight: Some(v),
                ..Default::default()
            }
            .validate()
        };
        assert!(weight(-0.1).is_err());
        assert!(weight(f64::NAN).is_err());
        assert!(weight(0.0).is_ok());
        assert!(weight(1.0).is_ok());
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
        let gate_ratio = |v| {
            Parameters {
                soft_affinity_gate_rel: Some(v),
                ..Default::default()
            }
            .validate()
        };
        assert!(gate_ratio(0.9).is_err() && gate_ratio(f64::NAN).is_err());
        assert!(gate_ratio(1.0).is_ok());
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

    /// `(worker_id, device_blocks, active_requests)` with a soft target on A.
    fn select_soft(parameters: Parameters, workers: [(u64, usize, usize); 2]) -> WorkerWithDpRank {
        select_affine(
            parameters,
            workers.map(|(id, device_blocks, active)| (id, device_blocks, 0, active)),
            Some(WorkerAffinityTarget::new(A, None)),
        )
    }

    fn respecting() -> Parameters {
        Parameters {
            respect_soft_affinity: true,
            ..Parameters::default()
        }
    }

    fn gated() -> Parameters {
        Parameters {
            soft_affinity_load_gate: true,
            ..respecting()
        }
    }

    #[test]
    fn soft_target_is_ignored_by_default_and_retained_when_respected() {
        let workers = [(A, 0, 0), (B, 6, 4)];
        assert_eq!(select_soft(Parameters::default(), workers), worker(B));
        assert_eq!(select_soft(respecting(), workers), worker(A));
        // An absent target falls back to the policy.
        assert_eq!(
            select_affine(
                respecting(),
                [(A, 0, 0, 0), (B, 6, 0, 4)],
                Some(WorkerAffinityTarget::new(999, None)),
            ),
            worker(B)
        );
    }

    /// Upstream #15139 semantics: with one row per worker the in-target load gate compares the
    /// target with itself, so an overloaded target is kept however imbalanced the pool is.
    #[test]
    fn respected_target_ignores_cross_worker_load_without_the_gate() {
        assert_eq!(select(workers_40_vs_2()), worker(B));
        assert_eq!(select_soft(respecting(), workers_40_vs_2()), worker(A));
        assert_eq!(
            select_soft(respecting(), [(A, 0, 400), (B, 0, 0)]),
            worker(A)
        );
    }

    fn workers_40_vs_2() -> [(u64, usize, usize); 2] {
        [(A, 0, 40), (B, 0, 2)]
    }

    #[test]
    fn load_gate_releases_only_an_imbalanced_target() {
        assert_eq!(select_soft(gated(), workers_40_vs_2()), worker(B));
        // Spread 28 is within 32: kept, although plain two-tier would pick B.
        assert_eq!(select_soft(gated(), [(A, 0, 30), (B, 0, 2)]), worker(A));
        // Spread 64 but ratio 1.1 not exceeded: kept.
        assert_eq!(select_soft(gated(), [(A, 0, 704), (B, 0, 640)]), worker(A));
        // A cache-hot peer at equal load does not dislodge the target.
        assert_eq!(select_soft(gated(), [(A, 0, 5), (B, 9, 5)]), worker(A));
        // Host-tier overlap counts toward the released decision as usual.
        assert_eq!(
            select_affine(
                gated(),
                [(A, 0, 0, 40), (B, 0, 8, 2)],
                Some(WorkerAffinityTarget::new(A, None)),
            ),
            worker(B)
        );
    }

    #[test]
    fn load_gate_takes_its_own_thresholds_without_moving_the_load_tier() {
        let own_gate = Parameters {
            soft_affinity_gate_abs: Some(6),
            soft_affinity_gate_rel: Some(1.5),
            ..gated()
        };
        // Gap 7 > 6 and 9 > 1.5 * 2: released; the shared 32 / 1.1 gate keeps it.
        assert_eq!(select_soft(gated(), [(A, 0, 9), (B, 0, 2)]), worker(A));
        assert_eq!(select_soft(own_gate, [(A, 0, 9), (B, 0, 2)]), worker(B));
        // Gap 6 is not > 6: kept.
        assert_eq!(select_soft(own_gate, [(A, 0, 8), (B, 0, 2)]), worker(A));
        // Gap 10 but 30 is not > 1.5 * 20: kept.
        assert_eq!(select_soft(own_gate, [(A, 0, 30), (B, 0, 20)]), worker(A));
        // Without a soft target the load tier still uses 32 / 1.1: B's cache wins at a gap of 7.
        assert_eq!(select_with(own_gate, [(A, 0, 2), (B, 10, 9)]), worker(B));
    }
}
