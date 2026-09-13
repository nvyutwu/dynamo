// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{borrow::Cow, collections::HashMap};

use serde::{Deserialize, Serialize};

use super::{RawIndexState, RoutingEligibility, SchedulingRequest};
use crate::protocols::{
    DpRank, WorkerConfigLike, WorkerId, WorkerWithDpRank, complete_block_count,
};

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

/// Diagnostic maxima over valid entries only, not full-domain R/E/S. Retained
/// when malformed index entries or funnel inconsistencies prevent complete coverage.
/// Bounded to three candidates regardless of worker count; no hash inventory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RawCacheCandidateDiagnostics {
    pub rejected_candidates: u64,
    pub best_valid_resident: Option<RawCacheCandidate>,
    pub best_valid_eligible: Option<RawCacheCandidate>,
    pub valid_selected: Option<RawCacheCandidate>,
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
    /// Additive diagnostics never authorize complete-token metric accounting.
    /// Boxed so complete observations do not retain three extra candidates.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidate_diagnostics: Option<Box<RawCacheCandidateDiagnostics>>,
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
            candidate_diagnostics: None,
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
    // Router hashes represent complete blocks only. Reject a partial-tail
    // block instead of masking an invalid hashes-only request with a clamp.
    let complete_blocks = complete_block_count(request.isl_tokens, block_size, false) as u64;
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
    if available_blocks > complete_blocks {
        return Err(());
    }
    // The complete-block bound guarantees these products fit in the prompt.
    let hbm_prefix_tokens = hbm * block_size;
    let total_tokens = available_blocks * block_size;
    Ok(RawCacheCandidate {
        worker_id: worker.worker_id,
        dp_rank: worker.dp_rank,
        hbm_prefix_tokens,
        cpu_extension_tokens: cpu * block_size,
        total_tokens,
    })
}

fn candidate_maximum(best: &mut Option<RawCacheCandidate>, current: RawCacheCandidate) {
    if best.is_none_or(|prior| {
        (current.total_tokens, current.worker_id, current.dp_rank)
            > (prior.total_tokens, prior.worker_id, prior.dp_rank)
    }) {
        *best = Some(current);
    }
}

fn scan_candidates<C: WorkerConfigLike>(
    workers: &HashMap<WorkerId, C>,
    request: &SchedulingRequest,
    resident_eligibility: RoutingEligibility<'_>,
    eligible_eligibility: RoutingEligibility<'_>,
    selected_worker: WorkerWithDpRank,
    block_size: u32,
) -> RawCacheCandidateDiagnostics {
    let mut scan = RawCacheCandidateDiagnostics {
        rejected_candidates: 0,
        best_valid_resident: None,
        best_valid_eligible: None,
        valid_selected: None,
    };
    let mut visit = |worker| {
        let allowed = |eligibility: RoutingEligibility<'_>| {
            eligibility.pinned_worker().is_none_or(|pin| pin == worker)
                && eligibility.validate_worker_rank(workers, worker).is_ok()
        };
        let resident = allowed(resident_eligibility);
        let eligible = allowed(eligible_eligibility);
        if !resident && !eligible {
            return;
        }
        // Convert once for the union of W and V, including the selected rank.
        // A malformed entry is counted once even when it belongs to both sets.
        let Ok(current) = candidate(request, worker, block_size) else {
            scan.rejected_candidates += 1;
            return;
        };
        if resident {
            candidate_maximum(&mut scan.best_valid_resident, current);
        }
        if eligible {
            candidate_maximum(&mut scan.best_valid_eligible, current);
        }
        if worker == selected_worker {
            scan.valid_selected = Some(current);
        }
    };
    // Preserve the constant-work explicit-pin path rather than scanning a fleet
    // to find a candidate whose identity is already fixed in both domains.
    if resident_eligibility.pinned_worker() == Some(selected_worker)
        && eligible_eligibility.pinned_worker() == Some(selected_worker)
    {
        visit(selected_worker);
        return scan;
    }
    for (&worker_id, config) in workers {
        let start = config.data_parallel_start_rank();
        let end = start.saturating_add(config.data_parallel_size());
        for dp_rank in start..end {
            visit(WorkerWithDpRank::new(worker_id, dp_rank));
        }
    }
    scan
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
    let scan = scan_candidates(
        workers,
        request,
        resident_eligibility,
        eligible_eligibility,
        selected_worker,
        block_size,
    );
    if scan.rejected_candidates > 0 {
        let mut coverage =
            RawCacheCoverage::incomplete(RawCacheObservation::InvariantFailure, request.isl_tokens);
        coverage.candidate_diagnostics = Some(Box::new(scan));
        return coverage;
    }
    let (Some(resident), Some(eligible), Some(selected)) = (
        scan.best_valid_resident,
        scan.best_valid_eligible,
        scan.valid_selected,
    ) else {
        return RawCacheCoverage::incomplete(
            RawCacheObservation::InvalidCandidate,
            request.isl_tokens,
        );
    };
    if selected.total_tokens > eligible.total_tokens
        || eligible.total_tokens > resident.total_tokens
        || resident.total_tokens > request.isl_tokens as u64
    {
        let mut coverage =
            RawCacheCoverage::incomplete(RawCacheObservation::InvariantFailure, request.isl_tokens);
        coverage.candidate_diagnostics = Some(Box::new(scan));
        return coverage;
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
        candidate_diagnostics: None,
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
    fn raw_funnel_invariant_failure_retains_candidate_diagnostics() {
        let workers = workers();
        let a = WorkerWithDpRank::new(1, 0);
        let b = WorkerWithDpRank::new(2, 0);
        let mut resident_request = request(16, true);
        resident_request.allowed_worker_ids = Some(HashSet::from([1]));
        let mut request = request(16, true);
        request
            .overlap
            .tier_overlap_blocks
            .device
            .extend([(a, 1), (b, 2)]);
        // Deliberately inconsistent domains: R=8 < E=S=16. No individual
        // candidate is malformed, but the canonical funnel is still invalid.
        let coverage = compute_raw_cache_coverage(
            &workers,
            &request,
            resident_request.eligibility(),
            request.eligibility(),
            b,
            8,
        );
        assert_eq!(coverage.observation, RawCacheObservation::InvariantFailure);
        assert!(coverage.resident.is_none());
        assert!(coverage.eligible.is_none());
        assert!(coverage.selected.is_none());
        assert!(coverage.overload_gap_tokens.is_none());
        assert!(coverage.selection_gap_tokens.is_none());
        let diagnostics = coverage
            .candidate_diagnostics
            .as_ref()
            .expect("funnel diagnostics");
        assert_eq!(diagnostics.rejected_candidates, 0);
        assert_eq!(diagnostics.best_valid_resident.unwrap().total_tokens, 8);
        assert_eq!(diagnostics.best_valid_eligible.unwrap().total_tokens, 16);
        assert_eq!(diagnostics.valid_selected.unwrap().worker_id, b.worker_id);
        let decoded: RawCacheCoverage =
            serde_json::from_value(serde_json::to_value(&coverage).unwrap()).unwrap();
        assert_eq!(decoded, coverage);
    }

    #[test]
    fn malformed_candidate_preserves_bounded_valid_diagnostics_without_complete_maxima() {
        let workers = workers();
        let a = WorkerWithDpRank::new(1, 0);
        let b = WorkerWithDpRank::new(2, 0);
        let bad = WorkerWithDpRank::new(3, 0);
        let mut request = request(16, true);
        request
            .overlap
            .tier_overlap_blocks
            .device
            .extend([(a, 2), (b, 1), (bad, 3)]);
        let coverage = compute_raw_cache_coverage(
            &workers,
            &request,
            request.eligibility(),
            request.eligibility(),
            b,
            8,
        );
        assert_eq!(coverage.observation, RawCacheObservation::InvariantFailure);
        // The full-domain maximum is unknown. Existing readers must not mistake
        // a maximum over valid entries for complete R/E/S evidence.
        assert!(coverage.resident.is_none());
        assert!(coverage.eligible.is_none());
        assert!(coverage.selected.is_none());
        assert!(coverage.overload_gap_tokens.is_none());
        assert!(coverage.selection_gap_tokens.is_none());
        let wire = serde_json::to_value(&coverage).unwrap();
        let decoded: RawCacheCoverage = serde_json::from_value(wire.clone()).unwrap();
        assert_eq!(decoded, coverage);
        let diagnostics = &wire["candidate_diagnostics"];
        assert_eq!(diagnostics["rejected_candidates"], 1);
        assert_eq!(diagnostics["best_valid_resident"]["worker_id"], 1);
        assert_eq!(diagnostics["best_valid_eligible"]["total_tokens"], 16);
        assert_eq!(diagnostics["valid_selected"]["worker_id"], 2);
        assert_eq!(diagnostics["valid_selected"]["total_tokens"], 8);
        // The malformed entry is counted once despite belonging to W and V.
        let malformed_selected = compute_raw_cache_coverage(
            &workers,
            &request,
            request.eligibility(),
            request.eligibility(),
            bad,
            8,
        );
        let wire = serde_json::to_value(&malformed_selected).unwrap();
        assert_eq!(wire["candidate_diagnostics"]["rejected_candidates"], 1);
        assert!(wire["candidate_diagnostics"]["valid_selected"].is_null());
    }

    #[test]
    fn malformed_candidate_outside_both_domains_does_not_poison_coverage() {
        let workers = workers();
        let a = WorkerWithDpRank::new(1, 0);
        let bad = WorkerWithDpRank::new(3, 0);
        let mut request = request(16, true);
        request.allowed_worker_ids = Some(HashSet::from([1]));
        request
            .overlap
            .tier_overlap_blocks
            .device
            .extend([(a, 2), (bad, 3)]);
        let coverage = compute_raw_cache_coverage(
            &workers,
            &request,
            request.eligibility(),
            request.eligibility(),
            a,
            8,
        );
        assert_eq!(coverage.observation, RawCacheObservation::Complete);
        assert_eq!(coverage.resident.unwrap().total_tokens, 16);
        let legacy_shape = serde_json::to_value(&coverage).unwrap();
        assert!(legacy_shape.get("candidate_diagnostics").is_none());
        let decoded: RawCacheCoverage = serde_json::from_value(legacy_shape).unwrap();
        assert_eq!(decoded, coverage);
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
    fn raw_coverage_distinguishes_missing_index_from_observed_zero_and_rejects_partial_block() {
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
        zero.overlap.tier_overlap_blocks.device.insert(selected, 2);
        let complete = compute_raw_cache_coverage(
            &workers,
            &zero,
            zero.eligibility(),
            zero.eligibility(),
            selected,
            12_288,
        );
        assert_eq!(complete.selected.unwrap().total_tokens, 24_576);

        zero.overlap.tier_overlap_blocks.device.insert(selected, 3);
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
    fn raw_coverage_rejects_partial_blocks_in_both_tiers() {
        let workers = workers();
        let selected = WorkerWithDpRank::new(1, 0);
        for prompt in [1, 12_287, 12_289, 24_704] {
            let complete_blocks = prompt / 12_288;
            for (gpu, cpu) in [
                (complete_blocks + 1, 0),
                (complete_blocks, 1),
                (0, complete_blocks + 1),
            ] {
                let mut request = request(prompt, true);
                request
                    .overlap
                    .tier_overlap_blocks
                    .device
                    .insert(selected, gpu);
                request
                    .overlap
                    .tier_overlap_blocks
                    .host_pinned
                    .insert(selected, cpu);
                let coverage = compute_raw_cache_coverage(
                    &workers,
                    &request,
                    request.eligibility(),
                    request.eligibility(),
                    selected,
                    12_288,
                );
                assert_eq!(coverage.observation, RawCacheObservation::InvariantFailure);
                assert!(coverage.resident.is_none());
                assert!(coverage.selected.is_none());
                assert!(coverage.candidate_diagnostics.unwrap().rejected_candidates > 0);
            }
        }
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
