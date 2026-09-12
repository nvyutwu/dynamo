// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{borrow::Cow, collections::HashMap};

use serde::{Deserialize, Serialize};

use super::{RawIndexState, RoutingEligibility, SchedulingRequest};
use crate::protocols::{DpRank, WorkerConfigLike, WorkerId, WorkerWithDpRank};

pub const RAW_CACHE_COVERAGE_SCHEMA: &str = "dynamo.router.raw_cache_coverage.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RawCacheObservation {
    Complete,
    MissingIndex,
    StaleIndex,
    InvalidCandidate,
    InvariantFailure,
}

impl RawCacheObservation {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Complete => "complete",
            Self::MissingIndex => "missing_index",
            Self::StaleIndex => "stale_index",
            Self::InvalidCandidate => "invalid_candidate",
            Self::InvariantFailure => "invariant_failure",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RawCacheCandidate {
    pub worker_id: WorkerId,
    pub dp_rank: DpRank,
    pub hbm_prefix_tokens: u64,
    pub cpu_extension_tokens: u64,
    pub total_tokens: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RawCacheCoverage {
    pub schema: Cow<'static, str>,
    pub basis: Cow<'static, str>,
    pub observation: RawCacheObservation,
    pub input_tokens: u64,
    pub resident: Option<RawCacheCandidate>,
    pub eligible: Option<RawCacheCandidate>,
    pub selected: Option<RawCacheCandidate>,
    pub overload_gap_tokens: Option<u64>,
    pub selection_gap_tokens: Option<u64>,
}

impl RawCacheCoverage {
    fn incomplete(observation: RawCacheObservation, input_tokens: usize) -> Self {
        Self {
            schema: RAW_CACHE_COVERAGE_SCHEMA.into(),
            basis: "router_index".into(),
            observation,
            input_tokens: input_tokens as u64,
            resident: None,
            eligible: None,
            selected: None,
            overload_gap_tokens: None,
            selection_gap_tokens: None,
        }
    }
}

fn candidate(
    request: &SchedulingRequest,
    worker: WorkerWithDpRank,
    block_size: u32,
) -> Result<RawCacheCandidate, ()> {
    if block_size == 0 {
        return Err(());
    }
    let prompt = request.isl_tokens as u64;
    let block_size = u64::from(block_size);
    let hbm = request
        .overlap
        .tier_overlap_blocks
        .device
        .get(&worker)
        .copied()
        .unwrap_or(0) as u64;
    // HostPinned is a marginal extension beyond the device prefix.
    let cpu = request
        .overlap
        .tier_overlap_blocks
        .host_pinned
        .get(&worker)
        .copied()
        .unwrap_or(0) as u64;
    let available_blocks = hbm.checked_add(cpu).ok_or(())?;
    if available_blocks > (request.isl_tokens as u64).div_ceil(block_size) {
        return Err(());
    }
    let hbm_prefix_tokens = hbm.saturating_mul(block_size).min(prompt);
    let total_tokens = hbm
        .saturating_add(cpu)
        .saturating_mul(block_size)
        .min(prompt);
    Ok(RawCacheCandidate {
        worker_id: worker.worker_id,
        dp_rank: worker.dp_rank,
        hbm_prefix_tokens,
        cpu_extension_tokens: total_tokens.saturating_sub(hbm_prefix_tokens),
        total_tokens,
    })
}

fn max_candidate<C: WorkerConfigLike>(
    workers: &HashMap<WorkerId, C>,
    request: &SchedulingRequest,
    eligibility: RoutingEligibility<'_>,
    block_size: u32,
) -> Result<Option<RawCacheCandidate>, ()> {
    if let Some(worker) = eligibility.pinned_worker() {
        return Ok(eligibility
            .validate_worker_rank(workers, worker)
            .ok()
            .map(|_| candidate(request, worker, block_size))
            .transpose()?);
    }
    let mut best = None;
    for (&worker_id, config) in workers {
        let start = config.data_parallel_start_rank();
        let end = start.saturating_add(config.data_parallel_size());
        for dp_rank in start..end {
            let worker = WorkerWithDpRank::new(worker_id, dp_rank);
            if eligibility.validate_worker_rank(workers, worker).is_err() {
                continue;
            }
            let current = candidate(request, worker, block_size)?;
            if best.is_none_or(|prior: RawCacheCandidate| {
                (current.total_tokens, current.worker_id, current.dp_rank)
                    > (prior.total_tokens, prior.worker_id, prior.dp_rank)
            }) {
                best = Some(current);
            }
        }
    }
    Ok(best)
}

#[doc(hidden)]
pub fn compute_raw_cache_coverage<C: WorkerConfigLike>(
    workers: &HashMap<WorkerId, C>,
    request: &SchedulingRequest,
    resident_eligibility: RoutingEligibility<'_>,
    eligible_eligibility: RoutingEligibility<'_>,
    selected_worker: WorkerWithDpRank,
    block_size: u32,
) -> RawCacheCoverage {
    if request.overlap.raw_index_state == RawIndexState::Stale {
        return RawCacheCoverage::incomplete(RawCacheObservation::StaleIndex, request.isl_tokens);
    }
    if request.overlap.raw_index_state != RawIndexState::Observed || request.isl_tokens == 0 {
        return RawCacheCoverage::incomplete(RawCacheObservation::MissingIndex, request.isl_tokens);
    }
    if eligible_eligibility
        .pinned_worker()
        .is_some_and(|pinned| pinned != selected_worker)
        || eligible_eligibility
            .validate_worker_rank(workers, selected_worker)
            .is_err()
    {
        return RawCacheCoverage::incomplete(
            RawCacheObservation::InvalidCandidate,
            request.isl_tokens,
        );
    }
    let resident = match max_candidate(workers, request, resident_eligibility, block_size) {
        Ok(Some(candidate)) => candidate,
        Ok(None) => {
            return RawCacheCoverage::incomplete(
                RawCacheObservation::InvalidCandidate,
                request.isl_tokens,
            );
        }
        Err(()) => {
            return RawCacheCoverage::incomplete(
                RawCacheObservation::InvariantFailure,
                request.isl_tokens,
            );
        }
    };
    let eligible = match max_candidate(workers, request, eligible_eligibility, block_size) {
        Ok(Some(candidate)) => candidate,
        Ok(None) => {
            return RawCacheCoverage::incomplete(
                RawCacheObservation::InvalidCandidate,
                request.isl_tokens,
            );
        }
        Err(()) => {
            return RawCacheCoverage::incomplete(
                RawCacheObservation::InvariantFailure,
                request.isl_tokens,
            );
        }
    };
    let Ok(selected) = candidate(request, selected_worker, block_size) else {
        return RawCacheCoverage::incomplete(
            RawCacheObservation::InvariantFailure,
            request.isl_tokens,
        );
    };
    if selected.total_tokens > eligible.total_tokens
        || eligible.total_tokens > resident.total_tokens
        || resident.total_tokens > request.isl_tokens as u64
    {
        return RawCacheCoverage::incomplete(
            RawCacheObservation::InvariantFailure,
            request.isl_tokens,
        );
    }
    RawCacheCoverage {
        schema: RAW_CACHE_COVERAGE_SCHEMA.into(),
        basis: "router_index".into(),
        observation: RawCacheObservation::Complete,
        input_tokens: request.isl_tokens as u64,
        resident: Some(resident),
        eligible: Some(eligible),
        selected: Some(selected),
        overload_gap_tokens: Some(resident.total_tokens - eligible.total_tokens),
        selection_gap_tokens: Some(eligible.total_tokens - selected.total_tokens),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use rustc_hash::FxHashMap;

    use super::*;
    use crate::protocols::RoutingConstraints;
    use crate::scheduling::{OverlapSignals, ScheduleMode, TierOverlapBlocks};
    use crate::sequences::WorkerLoadProjection;
    use crate::test_utils::SimpleWorkerConfig;

    fn request(input_tokens: usize, observed: bool) -> SchedulingRequest {
        SchedulingRequest {
            mode: ScheduleMode::QueryOnly { request_id: None },
            token_seq: None,
            isl_tokens: input_tokens,
            lora_name: None,
            expected_output_tokens: None,
            affinity_target: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: RoutingConstraints::default(),
            router_config_override: None,
            track_prefill_tokens: false,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_context: None,
            overlap: OverlapSignals {
                raw_index_state: if observed {
                    RawIndexState::Observed
                } else {
                    RawIndexState::Missing
                },
                tier_overlap_blocks: TierOverlapBlocks::default(),
                effective_overlap_blocks: HashMap::new(),
                effective_cached_tokens: HashMap::new(),
            },
            kv_transfer_candidates: None,
            retain_kv_transfer_chain: false,
            shared_cache_hits: None,
            worker_loads: FxHashMap::<WorkerWithDpRank, WorkerLoadProjection>::default(),
            resp_tx: None,
        }
    }

    fn workers() -> HashMap<WorkerId, SimpleWorkerConfig> {
        HashMap::from([
            (1, SimpleWorkerConfig::default()),
            (2, SimpleWorkerConfig::default()),
            (3, SimpleWorkerConfig::default()),
        ])
    }

    #[test]
    fn raw_coverage_uses_unweighted_rank_specific_same_candidate_tiers() {
        let workers = workers();
        let a = WorkerWithDpRank::new(1, 0);
        let b = WorkerWithDpRank::new(2, 0);
        let c = WorkerWithDpRank::new(3, 0);
        let mut request = request(24_576, true);
        request
            .overlap
            .tier_overlap_blocks
            .device
            .extend([(a, 2), (b, 1), (c, 1)]);
        request
            .overlap
            .tier_overlap_blocks
            .host_pinned
            .extend([(a, 0), (b, 1), (c, 0)]);
        // Deliberately contradictory effective scores prove the raw helper ignores weights.
        request
            .overlap
            .effective_cached_tokens
            .extend([(a, 1), (b, 99), (c, 100)]);
        let overloaded = HashSet::from([1]);
        let eligible = request.eligibility_with_overloaded(Some(&overloaded));
        let coverage = compute_raw_cache_coverage(
            &workers,
            &request,
            request.eligibility(),
            eligible,
            c,
            12_288,
        );

        assert_eq!(coverage.observation, RawCacheObservation::Complete);
        assert_eq!(coverage.resident.unwrap().worker_id, 2); // stable tie chooses larger identity
        assert_eq!(coverage.resident.unwrap().hbm_prefix_tokens, 12_288);
        assert_eq!(coverage.resident.unwrap().cpu_extension_tokens, 12_288);
        assert_eq!(coverage.eligible.unwrap().worker_id, 2);
        assert_eq!(coverage.selected.unwrap().total_tokens, 12_288);
        assert_eq!(coverage.overload_gap_tokens, Some(0));
        assert_eq!(coverage.selection_gap_tokens, Some(12_288));

        request.overlap.effective_cached_tokens = HashMap::from([(a, 1_000), (b, 0), (c, 7)]);
        request.overlap.effective_overlap_blocks = HashMap::from([(a, 0.1), (b, 99.0), (c, 4.0)]);
        let after_weighted_inputs_change = compute_raw_cache_coverage(
            &workers,
            &request,
            request.eligibility(),
            request.eligibility_with_overloaded(Some(&overloaded)),
            c,
            12_288,
        );
        assert_eq!(after_weighted_inputs_change, coverage);
    }

    #[test]
    fn raw_coverage_separates_resident_overload_and_selection_boundaries() {
        let workers = workers();
        let a = WorkerWithDpRank::new(1, 0);
        let b = WorkerWithDpRank::new(2, 0);
        let c = WorkerWithDpRank::new(3, 0);
        let mut request = request(24, true);
        request
            .overlap
            .tier_overlap_blocks
            .device
            .extend([(a, 3), (b, 2), (c, 1)]);
        let overloaded = HashSet::from([1]);
        let coverage = compute_raw_cache_coverage(
            &workers,
            &request,
            request.eligibility(),
            request.eligibility_with_overloaded(Some(&overloaded)),
            c,
            8,
        );

        assert_eq!(coverage.resident.unwrap().worker_id, 1);
        assert_eq!(coverage.eligible.unwrap().worker_id, 2);
        assert_eq!(coverage.selected.unwrap().worker_id, 3);
        assert_eq!(coverage.overload_gap_tokens, Some(8));
        assert_eq!(coverage.selection_gap_tokens, Some(8));
    }

    #[test]
    fn raw_coverage_distinguishes_missing_index_from_observed_zero_and_caps_tail() {
        let workers = workers();
        let selected = WorkerWithDpRank::new(1, 0);
        let missing = request(24_704, false);
        assert_eq!(
            compute_raw_cache_coverage(
                &workers,
                &missing,
                missing.eligibility(),
                missing.eligibility(),
                selected,
                12_288
            )
            .observation,
            RawCacheObservation::MissingIndex
        );

        let mut zero = request(24_704, true);
        let coverage = compute_raw_cache_coverage(
            &workers,
            &zero,
            zero.eligibility(),
            zero.eligibility(),
            selected,
            12_288,
        );
        assert_eq!(coverage.observation, RawCacheObservation::Complete);
        assert_eq!(coverage.selected.unwrap().total_tokens, 0);
        zero.overlap.tier_overlap_blocks.device.insert(selected, 3);
        let capped = compute_raw_cache_coverage(
            &workers,
            &zero,
            zero.eligibility(),
            zero.eligibility(),
            selected,
            12_288,
        );
        assert_eq!(capped.selected.unwrap().total_tokens, 24_704);

        zero.overlap.tier_overlap_blocks.device.insert(selected, 4);
        assert_eq!(
            compute_raw_cache_coverage(
                &workers,
                &zero,
                zero.eligibility(),
                zero.eligibility(),
                selected,
                12_288,
            )
            .observation,
            RawCacheObservation::InvariantFailure
        );
    }

    #[test]
    fn raw_coverage_respects_pin_and_rejects_invalid_selected_rank() {
        let mut workers = workers();
        workers.get_mut(&1).unwrap().data_parallel_size = 2;
        let rank0 = WorkerWithDpRank::new(1, 0);
        let rank1 = WorkerWithDpRank::new(1, 1);
        let invalid = WorkerWithDpRank::new(1, 2);
        let mut request = request(16, true);
        request.pinned_worker = Some(rank0);
        request
            .overlap
            .tier_overlap_blocks
            .device
            .extend([(rank0, 1), (rank1, 0)]);
        request
            .overlap
            .tier_overlap_blocks
            .host_pinned
            .insert(rank1, 8);
        let coverage = compute_raw_cache_coverage(
            &workers,
            &request,
            request.eligibility(),
            request.eligibility(),
            rank0,
            4,
        );
        assert_eq!(coverage.selected.unwrap().total_tokens, 4);
        assert_eq!(coverage.resident.unwrap().dp_rank, 0);
        assert_eq!(coverage.eligible.unwrap().dp_rank, 0);

        assert_eq!(
            compute_raw_cache_coverage(
                &workers,
                &request,
                request.eligibility(),
                request.eligibility(),
                rank1,
                4,
            )
            .observation,
            RawCacheObservation::InvalidCandidate
        );

        assert_eq!(
            compute_raw_cache_coverage(
                &workers,
                &request,
                request.eligibility(),
                request.eligibility(),
                invalid,
                4
            )
            .observation,
            RawCacheObservation::InvalidCandidate
        );
    }

    #[test]
    fn raw_coverage_respects_allowed_and_available_sets_and_stale_state() {
        let workers = workers();
        let a = WorkerWithDpRank::new(1, 0);
        let b = WorkerWithDpRank::new(2, 0);
        let mut request = request(16, true);
        request.allowed_worker_ids = Some(HashSet::from([1, 2]));
        request
            .overlap
            .tier_overlap_blocks
            .device
            .extend([(a, 2), (b, 1)]);
        let available = HashSet::from([2]);
        let eligibility = request
            .eligibility()
            .with_available_workers(Some(&available));
        let coverage =
            compute_raw_cache_coverage(&workers, &request, eligibility, eligibility, b, 8);
        assert_eq!(coverage.observation, RawCacheObservation::Complete);
        assert_eq!(coverage.resident.unwrap().worker_id, 2);

        let _ = eligibility;
        request.overlap.raw_index_state = RawIndexState::Stale;
        let eligibility = request
            .eligibility()
            .with_available_workers(Some(&available));
        let stale = compute_raw_cache_coverage(&workers, &request, eligibility, eligibility, b, 8);
        assert_eq!(stale.observation, RawCacheObservation::StaleIndex);
        assert!(stale.resident.is_none());
    }
}
