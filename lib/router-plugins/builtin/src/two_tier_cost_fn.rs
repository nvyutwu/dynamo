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
//!    `cache_threshold` of the request's block count (greater than or equal with
//!    `cache_threshold_inclusive`), select the least-loaded worker holding that
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
//! Every tunable also has an environment override, named `DYN_ROUTER_TWO_TIER_` plus the field name in
//! upper case (`DYN_ROUTER_TWO_TIER_CACHE_THRESHOLD`, `..._CACHE_THRESHOLD_INCLUSIVE`,
//! `..._BALANCE_ABS_THRESHOLD`, `..._BALANCE_REL_THRESHOLD`, `..._HOST_CACHE_WEIGHT`).
//! An override wins over the instance's `parameters` and the default, applies to every instance of
//! this policy, and is validated the same way. It exists because deployments bake the policy YAML
//! into the image, so the environment is the only knob that can change without a rebuild.
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

const ENV_CACHE_THRESHOLD: &str = "DYN_ROUTER_TWO_TIER_CACHE_THRESHOLD";
const ENV_CACHE_THRESHOLD_INCLUSIVE: &str = "DYN_ROUTER_TWO_TIER_CACHE_THRESHOLD_INCLUSIVE";
const ENV_BALANCE_ABS_THRESHOLD: &str = "DYN_ROUTER_TWO_TIER_BALANCE_ABS_THRESHOLD";
const ENV_BALANCE_REL_THRESHOLD: &str = "DYN_ROUTER_TWO_TIER_BALANCE_REL_THRESHOLD";
const ENV_HOST_CACHE_WEIGHT: &str = "DYN_ROUTER_TWO_TIER_HOST_CACHE_WEIGHT";

/// Tunables for [`POLICY_TYPE`], named after their `sgl-router` counterparts.
///
/// Every field is optional and keeps the upstream default when omitted. Unknown keys are rejected
/// at startup rather than ignored, so a misremembered name fails loudly.
#[derive(Debug, Clone, Copy, serde::Deserialize)]
#[serde(deny_unknown_fields, default)]
struct Parameters {
    /// Fraction of the request's blocks that must be device-resident on the best worker before the
    /// cache tier applies. Compared strictly unless `cache_threshold_inclusive` is set.
    cache_threshold: f64,
    /// Compare the cache ratio with `>=` instead of the upstream `>`.
    cache_threshold_inclusive: bool,
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
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            cache_threshold: DEFAULT_CACHE_THRESHOLD,
            cache_threshold_inclusive: false,
            balance_abs_threshold: DEFAULT_BALANCE_ABS_THRESHOLD,
            balance_rel_threshold: DEFAULT_BALANCE_REL_THRESHOLD,
            host_cache_weight: None,
        }
    }
}

/// Read one override. Unset or blank means "no override"; anything else must parse.
fn env_override<T: std::str::FromStr>(
    lookup: &impl Fn(&str) -> Option<String>,
    name: &str,
) -> Result<Option<T>, WorkerSelectionPolicyProviderError> {
    let Some(raw) = lookup(name) else {
        return Ok(None);
    };
    let value = raw.trim();
    if value.is_empty() {
        return Ok(None);
    }
    value.parse().map(Some).map_err(|_| {
        WorkerSelectionPolicyProviderError::new(format!("{name}={raw:?} is not a valid value"))
    })
}

/// Like [`env_override`] for flags: accepts `1`/`0`, `true`/`false`, `yes`/`no`, `on`/`off`.
fn env_flag(
    lookup: &impl Fn(&str) -> Option<String>,
    name: &str,
) -> Result<Option<bool>, WorkerSelectionPolicyProviderError> {
    let Some(raw) = lookup(name) else {
        return Ok(None);
    };
    match raw.trim().to_ascii_lowercase().as_str() {
        "" => Ok(None),
        "1" | "true" | "yes" | "on" => Ok(Some(true)),
        "0" | "false" | "no" | "off" => Ok(Some(false)),
        _ => Err(WorkerSelectionPolicyProviderError::new(format!(
            "{name}={raw:?} is not a boolean (use 1/0, true/false, yes/no or on/off)"
        ))),
    }
}

impl Parameters {
    /// Apply the `DYN_ROUTER_TWO_TIER_*` overrides on top of the YAML (or default) values and
    /// return the names of the variables that were applied, for the startup log. Validation runs
    /// afterwards on the merged values, so an override is held to the same bounds as YAML.
    fn apply_env_overrides(
        &mut self,
        lookup: impl Fn(&str) -> Option<String>,
    ) -> Result<Vec<&'static str>, WorkerSelectionPolicyProviderError> {
        let mut applied = Vec::new();
        if let Some(value) = env_override(&lookup, ENV_CACHE_THRESHOLD)? {
            self.cache_threshold = value;
            applied.push(ENV_CACHE_THRESHOLD);
        }
        if let Some(value) = env_flag(&lookup, ENV_CACHE_THRESHOLD_INCLUSIVE)? {
            self.cache_threshold_inclusive = value;
            applied.push(ENV_CACHE_THRESHOLD_INCLUSIVE);
        }
        if let Some(value) = env_override(&lookup, ENV_BALANCE_ABS_THRESHOLD)? {
            self.balance_abs_threshold = value;
            applied.push(ENV_BALANCE_ABS_THRESHOLD);
        }
        if let Some(value) = env_override(&lookup, ENV_BALANCE_REL_THRESHOLD)? {
            self.balance_rel_threshold = value;
            applied.push(ENV_BALANCE_REL_THRESHOLD);
        }
        if let Some(value) = env_override(&lookup, ENV_HOST_CACHE_WEIGHT)? {
            self.host_cache_weight = Some(value);
            applied.push(ENV_HOST_CACHE_WEIGHT);
        }
        Ok(applied)
    }

    /// Whether `cache_ratio` clears the cache-tier gate.
    fn cache_tier_fires(&self, cache_ratio: f64) -> bool {
        if self.cache_threshold_inclusive {
            cache_ratio >= self.cache_threshold
        } else {
            cache_ratio > self.cache_threshold
        }
    }

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
        if let Some(weight) = self.host_cache_weight
            && (!weight.is_finite() || weight < 0.0)
        {
            return Err(WorkerSelectionPolicyProviderError::new(
                "host_cache_weight must be a finite number greater than or equal to 0.0",
            ));
        }
        Ok(())
    }
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
}

fn decide(
    parameters: &Parameters,
    cache: &[WorkerCacheInput],
    load: &[WorkerLoadInput],
    request_blocks: u64,
    host_cache_weight: f64,
) -> Option<Decision> {
    if cache.is_empty() || cache.len() != load.len() {
        return None;
    }

    let row_overlap: Vec<f64> = cache
        .iter()
        .map(|item| effective_overlap(item, host_cache_weight))
        .collect();
    let max_overlap_row =
        (0..row_overlap.len()).max_by(|a, b| row_overlap[*a].total_cmp(&row_overlap[*b]))?;
    let max_overlap = row_overlap[max_overlap_row];
    let cache_ratio = if request_blocks == 0 {
        0.0
    } else {
        max_overlap / request_blocks as f64
    };
    let min_load = load.iter().map(|item| item.active_requests()).min()?;
    let max_load = load.iter().map(|item| item.active_requests()).max()?;
    let decision = |row, reason| Decision {
        row,
        reason,
        max_overlap_row,
        max_overlap,
        cache_ratio,
        min_load,
        max_load,
        row_overlap: row_overlap.clone(),
    };

    if max_load.saturating_sub(min_load) > parameters.balance_abs_threshold
        && (max_load as f64) > parameters.balance_rel_threshold * (min_load as f64)
    {
        return least_loaded(load, 0..load.len()).map(|row| decision(row, REASON_LOAD_IMBALANCE));
    }
    if parameters.cache_tier_fires(cache_ratio) {
        return least_loaded(
            load,
            row_overlap
                .iter()
                .enumerate()
                .filter_map(|(row, overlap)| (*overlap == max_overlap).then_some(row)),
        )
        .map(|row| decision(row, REASON_CACHE_TIER));
    }
    least_loaded(load, 0..load.len()).map(|row| decision(row, REASON_NO_CACHE_WINNER))
}

fn select_row(
    parameters: &Parameters,
    cache: &[WorkerCacheInput],
    load: &[WorkerLoadInput],
    request_blocks: u64,
    host_cache_weight: f64,
) -> Option<usize> {
    decide(parameters, cache, load, request_blocks, host_cache_weight).map(|d| d.row)
}

struct TwoTierCostFnPicker {
    parameters: Parameters,
    /// Resolved once at construction: the instance override when given, else the router config's
    /// `host_cache_hit_weight`.
    host_cache_weight: f64,
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
        select_row(
            &self.parameters,
            cache,
            load,
            context.request_blocks(),
            self.host_cache_weight,
        )
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
        let decision = decide(
            &self.parameters,
            input.cache()?,
            input.load()?,
            context.request_blocks(),
            self.host_cache_weight,
        )?;
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
                    "cache_threshold_inclusive".into(),
                    f64::from(u8::from(self.parameters.cache_threshold_inclusive)),
                ),
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
            ],
        })
    }
}

fn provider(
    parameters: &WorkerSelectionPolicyParameters,
) -> Result<WorkerSelectionPolicyFactory, WorkerSelectionPolicyProviderError> {
    let mut parameters: Parameters = parameters.deserialize()?;
    let env_overrides = parameters.apply_env_overrides(|name| {
        std::env::var_os(name).map(|value| value.to_string_lossy().into_owned())
    })?;
    parameters.validate()?;
    let host_cache_weight_from_env = env_overrides.contains(&ENV_HOST_CACHE_WEIGHT);

    // Announce the RESOLVED parameters, not the file contents: every field is optional and
    // silently keeps an upstream default when omitted, so the YAML says what was asked for and
    // this says what is actually in force. `env_overrides` names the DYN_ROUTER_TWO_TIER_*
    // variables that replaced a YAML or default value.
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
        cache_threshold_inclusive = parameters.cache_threshold_inclusive,
        balance_abs_threshold = parameters.balance_abs_threshold,
        balance_rel_threshold = parameters.balance_rel_threshold,
        host_cache_weight = ?parameters.host_cache_weight,
        env_overrides = ?env_overrides,
        "Two-tier worker-selection policy enabled"
    );

    Ok(Arc::new(
        move |config: &KvRouterConfig, worker_type, _partition| {
            let host_cache_weight = parameters
                .host_cache_weight
                .unwrap_or(config.host_cache_hit_weight);
            // Logged per role, and with the SOURCE of the weight, because the cache tier ranks on
            //     device_blocks + host_cache_weight * host_blocks
            // and that weight has three possible origins, in precedence order:
            // DYN_ROUTER_TWO_TIER_HOST_CACHE_WEIGHT, this policy's YAML `parameters`, or
            // DYN_ROUTER_HOST_CACHE_HIT_WEIGHT via KvRouterConfig. It is also the variable an A/B
            // most often changes, so an experiment that sets the router-wide env var while the YAML
            // pins the field would otherwise compare two identical arms with no indication anything
            // was ignored.
            tracing::info!(
                policy_type = POLICY_TYPE,
                worker_type = worker_type.as_str(),
                host_cache_weight,
                host_cache_weight_source = if host_cache_weight_from_env {
                    ENV_HOST_CACHE_WEIGHT
                } else if parameters.host_cache_weight.is_some() {
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
        assert!(Parameters::default().validate().is_ok());
    }

    /// An environment lookup backed by fixed pairs, so tests never touch the process environment.
    fn env(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let map: HashMap<String, String> = pairs
            .iter()
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect();
        move |name| map.get(name).cloned()
    }

    #[test]
    fn env_overrides_replace_yaml_and_defaults() {
        // YAML set two fields; every override wins over YAML and default alike.
        let mut parameters = Parameters {
            cache_threshold: 0.6,
            balance_abs_threshold: 8,
            ..Parameters::default()
        };
        let applied = parameters
            .apply_env_overrides(env(&[
                (ENV_CACHE_THRESHOLD, "0.4"),
                (ENV_CACHE_THRESHOLD_INCLUSIVE, "true"),
                (ENV_BALANCE_ABS_THRESHOLD, " 16 "),
                (ENV_BALANCE_REL_THRESHOLD, "1.5"),
                (ENV_HOST_CACHE_WEIGHT, "1.0"),
            ]))
            .unwrap();
        assert_eq!(
            applied,
            [
                ENV_CACHE_THRESHOLD,
                ENV_CACHE_THRESHOLD_INCLUSIVE,
                ENV_BALANCE_ABS_THRESHOLD,
                ENV_BALANCE_REL_THRESHOLD,
                ENV_HOST_CACHE_WEIGHT,
            ]
        );
        assert_eq!(parameters.cache_threshold, 0.4);
        assert!(parameters.cache_threshold_inclusive);
        assert_eq!(parameters.balance_abs_threshold, 16);
        assert_eq!(parameters.balance_rel_threshold, 1.5);
        assert_eq!(parameters.host_cache_weight, Some(1.0));
        assert!(parameters.validate().is_ok());
    }

    #[test]
    fn unset_or_blank_env_keeps_yaml_values() {
        let mut parameters = Parameters {
            cache_threshold: 0.3,
            host_cache_weight: Some(0.5),
            ..Parameters::default()
        };
        let applied = parameters
            .apply_env_overrides(env(&[
                (ENV_CACHE_THRESHOLD, "  "),
                (ENV_CACHE_THRESHOLD_INCLUSIVE, ""),
            ]))
            .unwrap();
        assert!(applied.is_empty());
        assert_eq!(parameters.cache_threshold, 0.3);
        assert!(!parameters.cache_threshold_inclusive);
        assert_eq!(parameters.host_cache_weight, Some(0.5));
    }

    #[test]
    fn malformed_env_is_rejected_by_name() {
        for (name, value) in [
            (ENV_CACHE_THRESHOLD, "0.4x"),
            (ENV_CACHE_THRESHOLD_INCLUSIVE, "maybe"),
            (ENV_BALANCE_ABS_THRESHOLD, "-1"),
            (ENV_BALANCE_ABS_THRESHOLD, "1.5"),
            (ENV_BALANCE_REL_THRESHOLD, "fast"),
            (ENV_HOST_CACHE_WEIGHT, "heavy"),
        ] {
            let error = Parameters::default()
                .apply_env_overrides(env(&[(name, value)]))
                .unwrap_err();
            assert!(error.to_string().contains(name), "{name}={value}: {error}");
        }
    }

    #[test]
    fn env_overrides_are_validated_like_yaml() {
        // These parse, but are out of bounds: the merged parameters must still fail validation.
        for (name, value) in [
            (ENV_CACHE_THRESHOLD, "1.5"),
            (ENV_CACHE_THRESHOLD, "NaN"),
            (ENV_BALANCE_REL_THRESHOLD, "0.5"),
            (ENV_HOST_CACHE_WEIGHT, "-1"),
        ] {
            let mut parameters = Parameters::default();
            parameters
                .apply_env_overrides(env(&[(name, value)]))
                .unwrap();
            assert!(parameters.validate().is_err(), "{name}={value}");
        }
    }

    #[test]
    fn inclusive_threshold_admits_the_boundary() {
        // Five of ten blocks is exactly 0.5: load decides by default, the cache tier when inclusive.
        let workers = [(A, 0, 0), (B, 5, 4)];
        assert_eq!(select(workers), worker(A));
        let inclusive = Parameters {
            cache_threshold_inclusive: true,
            ..Parameters::default()
        };
        assert_eq!(select_with(inclusive, workers), worker(B));
    }

    #[test]
    fn lower_threshold_from_env_routes_to_the_cache_holder() {
        // Five of ten blocks: below the strict 0.5 default, above an overridden 0.4.
        let workers = [(A, 0, 0), (B, 5, 4)];
        let mut parameters = Parameters::default();
        parameters
            .apply_env_overrides(env(&[(ENV_CACHE_THRESHOLD, "0.4")]))
            .unwrap();
        assert_eq!(select_with(parameters, workers), worker(B));
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
