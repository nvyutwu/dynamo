// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::collections::hash_map::Entry;

use dynamo_kv_router::protocols::{
    ExternalSequenceBlockHash, KvCacheRemoveData, KvCacheStoreData, ResidencyDomain, StorageTier,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum EventDedupPolicy {
    RefCounted,
    SetLike,
}

/// Policy-driven deduplication filter for publisher KV cache events.
///
/// vLLM can emit multiple store/remove events for the same block hash.
/// Refcounts are tracked **per DP rank** because identical block hashes
/// on different ranks represent independent blocks.
///
/// `RefCounted` Stores increment a refcount and Removes pass only when it
/// reaches zero. `SetLike` events bypass bookkeeping when their producer
/// guarantees at most one logical residency per owner, tier, and block hash.
/// Clears reset refcounts for the emitting rank across storage tiers.
pub(super) struct EventDedupFilter {
    /// Per-(dp_rank, storage_tier, residency_domain) refcounts.
    per_rank_tier:
        HashMap<(u32, StorageTier, ResidencyDomain), HashMap<ExternalSequenceBlockHash, usize>>,
}

impl EventDedupFilter {
    pub(super) fn new() -> Self {
        Self {
            per_rank_tier: HashMap::new(),
        }
    }

    /// Track a store event. Increments refcount for each block hash on the
    /// given (DP rank, storage tier). Stores always pass through — this only
    /// updates bookkeeping.
    pub(super) fn track_store_in_domain(
        &mut self,
        dp_rank: u32,
        storage_tier: StorageTier,
        residency_domain: ResidencyDomain,
        policy: EventDedupPolicy,
        data: &KvCacheStoreData,
    ) {
        if policy == EventDedupPolicy::SetLike {
            return;
        }
        let refcounts = self
            .per_rank_tier
            .entry((dp_rank, storage_tier, residency_domain))
            .or_default();
        for block in &data.blocks {
            *refcounts.entry(block.block_hash).or_insert(0) += 1;
        }
    }

    /// Filter a remove event. Retains only block hashes whose refcount on the
    /// given (DP rank, storage tier) decrements to 0 (removing them from the
    /// map). Returns `None` if no hashes survive filtering.
    pub(super) fn filter_remove_in_domain(
        &mut self,
        dp_rank: u32,
        storage_tier: StorageTier,
        residency_domain: ResidencyDomain,
        policy: EventDedupPolicy,
        mut data: KvCacheRemoveData,
    ) -> Option<KvCacheRemoveData> {
        if policy == EventDedupPolicy::SetLike {
            return (!data.block_hashes.is_empty()).then_some(data);
        }
        let refcounts = self
            .per_rank_tier
            .entry((dp_rank, storage_tier, residency_domain))
            .or_default();
        data.block_hashes.retain(|hash| {
            match refcounts.entry(*hash) {
                Entry::Occupied(mut entry) => {
                    *entry.get_mut() -= 1;
                    if *entry.get() == 0 {
                        entry.remove();
                        true // refcount hit 0 -> pass through
                    } else {
                        false // still has references -> filter out
                    }
                }
                Entry::Vacant(_) => {
                    true // not tracked -> pass through defensively
                }
            }
        });
        if data.block_hashes.is_empty() {
            None
        } else {
            Some(data)
        }
    }

    /// Clear refcounts for one DP rank and residency domain across storage tiers.
    pub(super) fn clear_rank_domain(
        &mut self,
        dp_rank: u32,
        domain: ResidencyDomain,
        policy: EventDedupPolicy,
    ) {
        if policy == EventDedupPolicy::SetLike {
            return;
        }
        self.per_rank_tier
            .retain(|(tracked_dp_rank, _, tracked_domain), _| {
                *tracked_dp_rank != dp_rank || *tracked_domain != domain
            });
    }

    /// Forget refcounts for one physical tier only.
    ///
    /// Dedup state is keyed by `(dp_rank, tier, domain)`, so a tier-scoped clear
    /// must drop exactly one key. Dropping the whole rank/domain would make the
    /// next store on a surviving tier look new and republish blocks the frontend
    /// already indexes.
    pub(super) fn clear_rank_tier_domain(
        &mut self,
        dp_rank: u32,
        storage_tier: StorageTier,
        domain: ResidencyDomain,
        policy: EventDedupPolicy,
    ) {
        if policy == EventDedupPolicy::SetLike {
            return;
        }
        self.per_rank_tier.remove(&(dp_rank, storage_tier, domain));
    }

    #[cfg(test)]
    pub(super) fn track_store(
        &mut self,
        dp_rank: u32,
        storage_tier: StorageTier,
        data: &KvCacheStoreData,
    ) {
        self.track_store_in_domain(
            dp_rank,
            storage_tier,
            ResidencyDomain::Worker,
            EventDedupPolicy::RefCounted,
            data,
        );
    }

    #[cfg(test)]
    pub(super) fn filter_remove(
        &mut self,
        dp_rank: u32,
        storage_tier: StorageTier,
        data: KvCacheRemoveData,
    ) -> Option<KvCacheRemoveData> {
        self.filter_remove_in_domain(
            dp_rank,
            storage_tier,
            ResidencyDomain::Worker,
            EventDedupPolicy::RefCounted,
            data,
        )
    }

    #[cfg(test)]
    pub(super) fn clear_rank(&mut self, dp_rank: u32) {
        self.per_rank_tier
            .retain(|(tracked_dp_rank, _, _), _| *tracked_dp_rank != dp_rank);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_kv_router::protocols::{KvCacheStoredBlockData, LocalBlockHash};

    fn store(hashes: &[u64]) -> KvCacheStoreData {
        KvCacheStoreData {
            parent_hash: None,
            start_position: None,
            blocks: hashes
                .iter()
                .map(|&hash| KvCacheStoredBlockData {
                    block_hash: ExternalSequenceBlockHash(hash),
                    tokens_hash: LocalBlockHash(hash),
                    mm_extra_info: None,
                })
                .collect(),
        }
    }

    fn remove(hashes: &[u64]) -> KvCacheRemoveData {
        KvCacheRemoveData {
            block_hashes: hashes
                .iter()
                .map(|&hash| ExternalSequenceBlockHash(hash))
                .collect(),
        }
    }

    /// A refcount of 2 swallows the first remove; a forgotten key passes it
    /// through defensively. That asymmetry is what makes a surviving refcount
    /// observable, so each tier is stored twice and probed with one remove.
    fn seeded_filter() -> EventDedupFilter {
        let mut dedup = EventDedupFilter::new();
        for tier in [StorageTier::Device, StorageTier::HostPinned] {
            for _ in 0..2 {
                dedup.track_store_in_domain(
                    0,
                    tier,
                    ResidencyDomain::Worker,
                    EventDedupPolicy::RefCounted,
                    &store(&[1]),
                );
            }
        }
        dedup
    }

    fn remove_is_swallowed(
        dedup: &mut EventDedupFilter,
        tier: StorageTier,
        domain: ResidencyDomain,
    ) -> bool {
        dedup
            .filter_remove_in_domain(0, tier, domain, EventDedupPolicy::RefCounted, remove(&[1]))
            .is_none()
    }

    #[test]
    fn a_tier_scoped_clear_forgets_only_the_named_tier() {
        let mut dedup = seeded_filter();
        dedup.clear_rank_tier_domain(
            0,
            StorageTier::Device,
            ResidencyDomain::Worker,
            EventDedupPolicy::RefCounted,
        );

        assert!(
            !remove_is_swallowed(&mut dedup, StorageTier::Device, ResidencyDomain::Worker),
            "the cleared tier must have forgotten its refcounts"
        );
        assert!(
            remove_is_swallowed(&mut dedup, StorageTier::HostPinned, ResidencyDomain::Worker),
            "the untouched tier must keep its refcounts; dropping them would make \
             the next store on that tier look new and republish indexed blocks"
        );
    }

    #[test]
    fn an_all_tier_clear_still_forgets_every_tier_of_the_rank() {
        let mut dedup = seeded_filter();
        dedup.clear_rank_domain(0, ResidencyDomain::Worker, EventDedupPolicy::RefCounted);

        for tier in [StorageTier::Device, StorageTier::HostPinned] {
            assert!(!remove_is_swallowed(
                &mut dedup,
                tier,
                ResidencyDomain::Worker
            ));
        }
    }

    #[test]
    fn a_tier_scoped_clear_spares_other_ranks_and_other_domains() {
        let mut dedup = seeded_filter();
        for _ in 0..2 {
            dedup.track_store_in_domain(
                1,
                StorageTier::Device,
                ResidencyDomain::Worker,
                EventDedupPolicy::RefCounted,
                &store(&[1]),
            );
            dedup.track_store_in_domain(
                0,
                StorageTier::Device,
                ResidencyDomain::CacheOwner,
                EventDedupPolicy::RefCounted,
                &store(&[1]),
            );
        }

        dedup.clear_rank_tier_domain(
            0,
            StorageTier::Device,
            ResidencyDomain::Worker,
            EventDedupPolicy::RefCounted,
        );

        assert!(
            dedup
                .filter_remove_in_domain(
                    1,
                    StorageTier::Device,
                    ResidencyDomain::Worker,
                    EventDedupPolicy::RefCounted,
                    remove(&[1]),
                )
                .is_none(),
            "another rank keeps its refcounts"
        );
        assert!(
            remove_is_swallowed(&mut dedup, StorageTier::Device, ResidencyDomain::CacheOwner),
            "another ownership domain on the same tier keeps its refcounts"
        );
    }

    #[test]
    fn a_set_like_producer_keeps_bypassing_the_bookkeeping() {
        let mut dedup = seeded_filter();
        dedup.clear_rank_tier_domain(
            0,
            StorageTier::Device,
            ResidencyDomain::Worker,
            EventDedupPolicy::SetLike,
        );
        assert!(
            remove_is_swallowed(&mut dedup, StorageTier::Device, ResidencyDomain::Worker),
            "a SetLike clear must not touch RefCounted state"
        );
    }
}
