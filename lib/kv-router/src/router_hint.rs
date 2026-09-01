// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Router-generated hints that are attached to selected backend requests.

use serde::{Deserialize, Serialize};

use crate::protocols::{ExternalSequenceBlockHash, WorkerWithDpRank};

/// Key for router-generated backend hints inside KV transfer params.
pub const ROUTER_HINT_EXTRA_ARGS_KEY: &str = "router_hint";

/// Worker runtime_data key. Boolean true means the worker can consume router_hint extra args.
pub const ROUTER_HINT_RUNTIME_CAPABILITY_KEY: &str = "router_hint";

/// Worker runtime_data key for matching router-hint sources to targets by backend role.
pub const ROUTER_HINT_WORKER_TYPE_RUNTIME_KEY: &str = "router_hint_worker_type";

/// Worker runtime_data key for per-global-DP-rank advertised KVCC control endpoints.
pub const ROUTER_HINT_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY: &str =
    "router_hint_source_control_endpoints";

/// Worker runtime_data key for the source process's inventory generation.
pub const ROUTER_HINT_INVENTORY_EPOCH_RUNTIME_KEY: &str = "router_hint_inventory_epoch";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RouterHint {
    pub source_control_endpoint: String,
    /// Source process generation advertised with the inventory used for this decision.
    pub source_inventory_epoch: u64,
    /// First request block the destination should consider for remediation.
    pub start_block: u32,
    /// Number of consecutive remediation blocks represented by this hint.
    pub hinted_blocks: u32,
    /// Root-aligned source-side KV block hashes. `block_hashes[i]`
    /// corresponds to request block `i`; `start_block..start_block+hinted_blocks`
    /// is the intended suffix.
    pub block_hashes: Vec<ExternalSequenceBlockHash>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouterHintRootCandidates {
    pub block_hashes: Vec<ExternalSequenceBlockHash>,
    pub owner_prefix_blocks: Vec<(WorkerWithDpRank, usize)>,
}

impl RouterHintRootCandidates {
    pub fn best_source<F>(
        &self,
        prefix_blocks_to_beat: usize,
        mut is_eligible_source: F,
    ) -> Option<(WorkerWithDpRank, Vec<ExternalSequenceBlockHash>)>
    where
        F: FnMut(WorkerWithDpRank) -> bool,
    {
        let (source, prefix_blocks) = self
            .owner_prefix_blocks
            .iter()
            .copied()
            .filter(|(worker, blocks)| {
                *blocks > prefix_blocks_to_beat && is_eligible_source(*worker)
            })
            .max_by(|(left_worker, left_blocks), (right_worker, right_blocks)| {
                left_blocks
                    .cmp(right_blocks)
                    .then_with(|| right_worker.cmp(left_worker))
            })?;

        Some((source, self.block_hashes.get(..prefix_blocks)?.to_vec()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn best_source_selects_longest_eligible_prefix() {
        let worker_a = WorkerWithDpRank::new(7, 0);
        let worker_b = WorkerWithDpRank::new(8, 0);
        let excluded = WorkerWithDpRank::new(9, 0);
        let candidates = RouterHintRootCandidates {
            block_hashes: vec![
                ExternalSequenceBlockHash(101),
                ExternalSequenceBlockHash(102),
                ExternalSequenceBlockHash(103),
            ],
            owner_prefix_blocks: vec![(worker_b, 2), (excluded, 3), (worker_a, 3)],
        };

        let selected = candidates.best_source(0, |worker| worker != excluded);

        assert_eq!(
            selected,
            Some((
                worker_a,
                vec![
                    ExternalSequenceBlockHash(101),
                    ExternalSequenceBlockHash(102),
                    ExternalSequenceBlockHash(103),
                ],
            ))
        );
    }

    #[test]
    fn best_source_fails_closed_on_invalid_prefix_length() {
        let candidates = RouterHintRootCandidates {
            block_hashes: vec![ExternalSequenceBlockHash(101)],
            owner_prefix_blocks: vec![(WorkerWithDpRank::new(7, 0), 2)],
        };

        assert!(candidates.best_source(0, |_| true).is_none());
    }
}
