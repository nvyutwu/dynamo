// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::{Duration, Instant};

use dynamo_kv_router::indexer::LocalKvIndexer;
use dynamo_kv_router::protocols::{KvCacheEventData, Placement, PlacementEvent, RouterEvent};

use super::dedup::{EventDedupFilter, EventDedupPolicy};
use super::sinks::emit;

/// Pure accumulator for adjacent compatible placement mutations.
///
/// The caller-provided key carries the exact logical owner/source and physical
/// tier. The coalescer adds DP-rank, operation, and Store-chain boundaries. It
/// deliberately owns no event IDs, timers, indexers, publishers, or error
/// policy, so both publisher lifecycles can share it without sharing commits.
#[derive(Debug)]
pub(super) struct PlacementEventCoalescer<K> {
    pending: Option<(K, PlacementEvent)>,
    max_batch_blocks: usize,
}

impl<K: Eq> PlacementEventCoalescer<K> {
    pub(super) fn new(max_batch_blocks: usize) -> Self {
        Self {
            pending: None,
            max_batch_blocks,
        }
    }

    pub(super) fn has_pending(&self) -> bool {
        self.pending.is_some()
    }

    /// Push one input and return up to two ready outputs in stream order.
    ///
    /// Two outputs are possible when an incompatible or oversized mutation
    /// follows a pending mutation, or when `Cleared` flushes the pending value
    /// and then passes through as its own barrier.
    pub(super) fn push(&mut self, key: K, event: PlacementEvent) -> [Option<PlacementEvent>; 2] {
        if matches!(
            &event.event.data,
            KvCacheEventData::Cleared | KvCacheEventData::TierCleared(_)
        ) {
            return [self.flush(), Some(event)];
        }

        let can_merge = self.pending.as_ref().is_some_and(|(pending_key, pending)| {
            pending_key == &key
                && pending.event.dp_rank == event.event.dp_rank
                && compatible_data(&pending.event.data, &event.event.data)
        });
        if can_merge {
            let pending = &mut self
                .pending
                .as_mut()
                .expect("merge compatibility requires one pending event")
                .1;
            merge_data(&mut pending.event.data, event.event.data);
            if event_block_count(pending) >= self.max_batch_blocks {
                return [self.flush(), None];
            }
            return [None, None];
        }

        let flushed = self.flush();
        if event_block_count(&event) >= self.max_batch_blocks {
            return [flushed, Some(event)];
        }
        self.pending = Some((key, event));
        [flushed, None]
    }

    pub(super) fn flush(&mut self) -> Option<PlacementEvent> {
        self.pending.take().map(|(_, event)| event)
    }
}

fn merge_data(pending: &mut KvCacheEventData, next: KvCacheEventData) {
    match (pending, next) {
        (KvCacheEventData::Stored(pending), KvCacheEventData::Stored(next)) => {
            pending.blocks.extend(next.blocks);
        }
        (KvCacheEventData::Removed(pending), KvCacheEventData::Removed(next)) => {
            pending.block_hashes.extend(next.block_hashes);
        }
        _ => unreachable!("merge compatibility requires matching mutation kinds"),
    }
}

fn compatible_data(pending: &KvCacheEventData, next: &KvCacheEventData) -> bool {
    match (pending, next) {
        (KvCacheEventData::Stored(pending), KvCacheEventData::Stored(next)) => {
            next.parent_hash == pending.blocks.last().map(|block| block.block_hash)
        }
        (KvCacheEventData::Removed(_), KvCacheEventData::Removed(_)) => true,
        _ => false,
    }
}

fn event_block_count(event: &PlacementEvent) -> usize {
    match &event.event.data {
        KvCacheEventData::Stored(data) => data.blocks.len(),
        KvCacheEventData::Removed(data) => data.block_hashes.len(),
        KvCacheEventData::Cleared | KvCacheEventData::TierCleared(_) => 0,
    }
}

/// Accumulator for in-flight KV cache events that will be merged into a single
/// [`RouterEvent`] before being forwarded to the event sink.
#[derive(Debug)]
pub(super) struct BatchingState {
    coalescer: PlacementEventCoalescer<Placement>,
    pub(super) next_publish_id: u64,
    pub(super) last_flush_time: Instant,
}

impl BatchingState {
    pub(super) fn new(max_batch_blocks: usize) -> Self {
        Self {
            coalescer: PlacementEventCoalescer::new(max_batch_blocks),
            next_publish_id: 1,
            last_flush_time: Instant::now(),
        }
    }

    pub(super) fn has_pending(&self) -> bool {
        self.coalescer.has_pending()
    }

    pub(super) fn record_flush_time(&mut self) {
        self.last_flush_time = Instant::now();
    }

    pub(super) fn remaining_timeout(&self, timeout_ms: u64) -> Duration {
        let timeout = Duration::from_millis(timeout_ms);
        let elapsed = self.last_flush_time.elapsed();
        if elapsed >= timeout {
            Duration::ZERO
        } else {
            timeout - elapsed
        }
    }

    pub(super) fn is_timeout_elapsed(&self, timeout_ms: u64) -> bool {
        self.remaining_timeout(timeout_ms) == Duration::ZERO
    }

    pub(super) async fn flush(
        &mut self,
        local_indexer: &Option<Arc<LocalKvIndexer>>,
        worker_id: u64,
        dedup: &mut EventDedupFilter,
        output: &mut Vec<RouterEvent>,
    ) {
        if let Some(event) = self.coalescer.flush() {
            self.emit_ready(event, local_indexer, worker_id, dedup, output)
                .await;
        }
        self.record_flush_time();
    }

    pub(super) async fn push(
        &mut self,
        event: PlacementEvent,
        local_indexer: &Option<Arc<LocalKvIndexer>>,
        worker_id: u64,
        dedup: &mut EventDedupFilter,
        output: &mut Vec<RouterEvent>,
    ) {
        let key = event.placement.clone();
        let ready = self.coalescer.push(key, event);
        let flushed = ready.iter().any(Option::is_some);
        for ready in ready.into_iter().flatten() {
            self.emit_ready(ready, local_indexer, worker_id, dedup, output)
                .await;
        }
        if flushed {
            self.record_flush_time();
        }
    }

    async fn emit_ready(
        &mut self,
        placement_event: PlacementEvent,
        local_indexer: &Option<Arc<LocalKvIndexer>>,
        worker_id: u64,
        dedup: &mut EventDedupFilter,
        output: &mut Vec<RouterEvent>,
    ) {
        let tier = placement_event.placement.tier;
        let domain = placement_event.placement.residency_domain;
        let mut event = placement_event.event;
        event.data = match event.data {
            KvCacheEventData::Removed(data) => {
                let Some(filtered) = dedup.filter_remove_in_domain(
                    event.dp_rank,
                    tier,
                    domain,
                    EventDedupPolicy::RefCounted,
                    data,
                ) else {
                    return;
                };
                KvCacheEventData::Removed(filtered)
            }
            KvCacheEventData::Stored(data) => {
                dedup.track_store_in_domain(
                    event.dp_rank,
                    tier,
                    domain,
                    EventDedupPolicy::RefCounted,
                    &data,
                );
                KvCacheEventData::Stored(data)
            }
            KvCacheEventData::Cleared | KvCacheEventData::TierCleared(_) => {
                unreachable!("Cleared is handled by the publisher's barrier policy")
            }
        };
        event.event_id = self.next_publish_id;
        let _ = emit(local_indexer, worker_id, tier, domain, event, output).await;
        self.next_publish_id = self
            .next_publish_id
            .checked_add(1)
            .expect("KV event publisher outbound cursor exhausted");
    }
}

#[cfg(test)]
mod tests {
    use dynamo_kv_router::protocols::{
        ExternalSequenceBlockHash, KvCacheEvent, KvCacheRemoveData, KvCacheStoreData,
        KvCacheStoredBlockData, LocalBlockHash, Placement, ResidencyDomain, StorageTier,
    };

    use super::*;

    fn event(data: KvCacheEventData) -> PlacementEvent {
        PlacementEvent::new(
            Placement::local_worker(7, 0, StorageTier::HostPinned),
            KvCacheEvent {
                event_id: 0,
                data,
                dp_rank: 0,
            },
        )
    }

    fn stored(parent: Option<u64>, block: u64) -> PlacementEvent {
        event(KvCacheEventData::Stored(KvCacheStoreData {
            parent_hash: parent.map(ExternalSequenceBlockHash),
            start_position: None,
            blocks: vec![KvCacheStoredBlockData {
                block_hash: ExternalSequenceBlockHash(block),
                tokens_hash: LocalBlockHash(block),
                mm_extra_info: None,
            }],
        }))
    }

    fn removed(block: u64) -> PlacementEvent {
        event(KvCacheEventData::Removed(KvCacheRemoveData {
            block_hashes: vec![ExternalSequenceBlockHash(block)],
        }))
    }

    #[test]
    fn coalesces_legacy_mutations_and_keeps_clear_as_a_boundary() {
        let mut coalescer = PlacementEventCoalescer::new(128);
        let mut output = Vec::new();
        for input in [
            stored(None, 1),
            stored(Some(1), 2),
            removed(1),
            removed(2),
            event(KvCacheEventData::Cleared),
        ] {
            let key = input.placement.clone();
            output.extend(coalescer.push(key, input).into_iter().flatten());
        }
        output.extend(coalescer.flush());

        assert_eq!(output.len(), 3);
        assert!(matches!(
            &output[0].event.data,
            KvCacheEventData::Stored(data) if data.blocks.len() == 2
        ));
        assert!(matches!(
            &output[1].event.data,
            KvCacheEventData::Removed(data) if data.block_hashes.len() == 2
        ));
        assert!(matches!(output[2].event.data, KvCacheEventData::Cleared));
    }

    mod partial_tail_review {
        use super::*;
        use std::sync::atomic::AtomicU32;

        use dynamo_kv_router::config::KvRouterConfig;
        use dynamo_kv_router::indexer::{
            LowerTierIndexers, LowerTierQueryOptions, MatchDetails, PartialTailQuery,
            TieredMatchDetails, query_lower_tiers_with_options_and_tail,
        };
        use dynamo_kv_router::protocols::{
            BlockHashOptions, WorkerWithDpRank, compute_block_hash_for_seq,
            compute_seq_hash_for_block,
        };
        use dynamo_kv_router::scheduling::overlap::cache_hit_estimates_from_tiered_matches;
        use dynamo_kv_router::zmq_wire::{
            BlockHashValue, KvCacheSpecKind, RawKvEvent, ZmqEventNormalizer,
            convert_event_with_partial_tail,
        };

        const BLOCK: usize = 12_288;
        const SUB: usize = 128;
        const WORKER: WorkerWithDpRank = WorkerWithDpRank {
            worker_id: 7,
            dp_rank: 0,
        };

        fn hashes(tokens: &[u32]) -> Vec<u64> {
            compute_seq_hash_for_block(&compute_block_hash_for_seq(
                tokens,
                SUB as u32,
                BlockHashOptions::default(),
            ))
        }

        fn raw_store(tokens: &[u32], tail: bool) -> RawKvEvent {
            let end = tokens.len() / SUB;
            let start = if tail {
                end / (BLOCK / SUB) * (BLOCK / SUB)
            } else {
                0
            };
            let all_hashes = hashes(tokens);
            let block_hashes = if tail {
                all_hashes[start..end].to_vec()
            } else {
                vec![all_hashes[BLOCK / SUB - 1]]
            };
            RawKvEvent::BlockStored {
                block_hashes: block_hashes
                    .into_iter()
                    .map(BlockHashValue::Unsigned)
                    .collect(),
                parent_block_hash: start
                    .checked_sub(1)
                    .map(|i| BlockHashValue::Unsigned(all_hashes[i])),
                token_ids: if tail {
                    tokens[start * SUB..end * SUB].to_vec()
                } else {
                    tokens[..BLOCK].to_vec()
                },
                block_size: if tail { SUB } else { BLOCK },
                medium: Some("CPU".into()),
                lora_name: None,
                cache_namespace: None,
                block_mm_infos: None,
                is_eagle: None,
                group_idx: Some(3),
                kv_cache_spec_kind: Some(KvCacheSpecKind::MlaAttention),
                kv_cache_spec_sliding_window: None,
                locality: None,
                ownership: None,
            }
        }

        fn raw_remove(hashes: Vec<u64>, group: u32) -> RawKvEvent {
            RawKvEvent::BlockRemoved {
                block_hashes: hashes.into_iter().map(BlockHashValue::Unsigned).collect(),
                medium: Some("CPU".into()),
                group_idx: Some(group),
                kv_cache_spec_kind: None,
                kv_cache_spec_sliding_window: None,
                locality: None,
                ownership: None,
            }
        }

        struct Replay {
            normalizer: ZmqEventNormalizer,
            batching: BatchingState,
            dedup: EventDedupFilter,
            indexers: LowerTierIndexers,
            publisher_sub: u32,
        }

        impl Replay {
            fn new(publisher_sub: u32) -> Self {
                Self {
                    normalizer: ZmqEventNormalizer::new(BLOCK as u32),
                    batching: BatchingState::new(128),
                    dedup: EventDedupFilter::new(),
                    indexers: LowerTierIndexers::new(1, BLOCK as u32),
                    publisher_sub,
                }
            }

            async fn apply(&mut self, raw: RawKvEvent) {
                let Some(raw) = self.normalizer.preprocess(raw, WORKER) else {
                    return;
                };
                let event = convert_event_with_partial_tail(
                    raw,
                    0,
                    BLOCK as u32,
                    WORKER,
                    &Arc::new(AtomicU32::new(0)),
                    None,
                    None,
                    self.publisher_sub,
                )
                .unwrap();
                let mut output = Vec::new();
                self.batching
                    .push(event, &None, WORKER.worker_id, &mut self.dedup, &mut output)
                    .await;
                self.batching
                    .flush(&None, WORKER.worker_id, &mut self.dedup, &mut output)
                    .await;
                for event in output {
                    let wire = serde_json::to_vec(&event).unwrap();
                    let event: RouterEvent = serde_json::from_slice(&wire).unwrap();
                    let result = self
                        .indexers
                        .get_or_create(event.storage_tier)
                        .apply_event_and_wait(event)
                        .await;
                    assert!(result.is_ok(), "{result:?}");
                }
            }

            async fn store(&mut self, tokens: &[u32]) {
                if tokens.len() >= BLOCK {
                    self.apply(raw_store(tokens, false)).await;
                }
                self.apply(raw_store(tokens, true)).await;
            }

            fn cached(&self, tokens: &[u32]) -> usize {
                let opts = BlockHashOptions::default();
                let sequence = compute_block_hash_for_seq(tokens, BLOCK as u32, opts);
                let tail = PartialTailQuery {
                    tokens,
                    block_size: BLOCK as u32,
                    sub_block_size: SUB as u32,
                    hash_options: opts,
                };
                let tiered = TieredMatchDetails {
                    device: MatchDetails::new(),
                    lower_tier: query_lower_tiers_with_options_and_tail(
                        &self.indexers,
                        &sequence,
                        &MatchDetails::new(),
                        LowerTierQueryOptions::default(),
                        Some(&tail),
                    ),
                };
                let config = KvRouterConfig {
                    host_cache_hit_weight: 1.0,
                    disk_cache_hit_weight: 1.0,
                    ..Default::default()
                };
                cache_hit_estimates_from_tiered_matches(&config, BLOCK as u32, &tiered)
                    .cached_tokens
                    .get(&WORKER)
                    .copied()
                    .unwrap_or(0)
            }
        }

        #[tokio::test]
        async fn single_tail_store_lookup_remove_through_publisher() {
            for len in [5_000, 15_133] {
                let tokens: Vec<u32> = (0..len).collect();
                let mut replay = Replay::new(SUB as u32);
                replay.store(&tokens).await;
                assert_eq!(replay.cached(&tokens), tokens.len() / SUB * SUB);
                let start = tokens.len() / BLOCK * (BLOCK / SUB);
                replay
                    .apply(raw_remove(hashes(&tokens)[start..].to_vec(), 3))
                    .await;
                assert_eq!(replay.cached(&tokens), tokens.len() / BLOCK * BLOCK);
            }
        }

        #[tokio::test]
        async fn shared_tail_hashes_survive_until_final_publisher_reference() {
            for evict_short_first in [true, false] {
                let tokens: Vec<u32> = (0..15_133).collect();
                let short = &tokens[..13_000];
                let mut replay = Replay::new(SUB as u32);
                replay.store(short).await;
                replay.apply(raw_store(&tokens, true)).await;
                let (first, survivor) = if evict_short_first {
                    (short, tokens.as_slice())
                } else {
                    (tokens.as_slice(), short)
                };
                replay
                    .apply(raw_remove(hashes(first)[BLOCK / SUB..].to_vec(), 3))
                    .await;
                assert_eq!(replay.cached(survivor), survivor.len() / SUB * SUB);
                replay
                    .apply(raw_remove(hashes(survivor)[BLOCK / SUB..].to_vec(), 3))
                    .await;
                assert_eq!(replay.cached(&tokens), BLOCK);
            }
        }

        #[tokio::test]
        async fn cached_tail_remains_visible_when_extension_crosses_router_block() {
            let tokens: Vec<u32> = (0..26_000).collect();
            let mut replay = Replay::new(SUB as u32);
            replay.store(&tokens[..15_133]).await;
            assert_eq!(replay.cached(&tokens[..15_333]), 15_104);
            assert_eq!(replay.cached(&tokens), 15_104);
        }

        #[tokio::test]
        async fn cached_root_tail_remains_visible_to_longer_request() {
            let tokens: Vec<u32> = (0..13_000).collect();
            let mut replay = Replay::new(SUB as u32);
            replay.store(&tokens[..5_000]).await;
            assert_eq!(replay.cached(&tokens), 4_992);
        }

        #[tokio::test]
        async fn hybrid_tail_does_not_credit_an_unstored_interior_boundary() {
            let tokens: Vec<u32> = (0..15_133).collect();
            let mut replay = Replay::new(SUB as u32);
            replay.store(&tokens).await;
            let mut divergent = tokens.clone();
            divergent[14_080..].fill(99_999);
            assert_eq!(replay.cached(&divergent), BLOCK);
        }

        #[tokio::test]
        async fn hybrid_tail_loses_credit_when_one_companion_group_is_evicted() {
            let tokens: Vec<u32> = (0..15_133).collect();
            let mut replay = Replay::new(SUB as u32);
            let mut announcement = raw_store(&tokens, true);
            if let RawKvEvent::BlockStored {
                group_idx,
                kv_cache_spec_kind,
                medium,
                ..
            } = &mut announcement
            {
                *group_idx = Some(0);
                *kv_cache_spec_kind = Some(KvCacheSpecKind::Mamba);
                *medium = None;
            }
            assert!(replay.normalizer.preprocess(announcement, WORKER).is_none());
            replay.store(&tokens).await;
            assert_eq!(replay.cached(&tokens), 15_104);
            replay
                .apply(raw_remove(vec![*hashes(&tokens).last().unwrap()], 0))
                .await;
            assert_eq!(replay.cached(&tokens), BLOCK);
        }

        #[tokio::test]
        async fn the_same_tail_in_two_tiers_is_not_counted_twice() {
            let tokens: Vec<u32> = (0..5_000).collect();
            let mut replay = Replay::new(SUB as u32);
            replay.store(&tokens).await;
            let mut disk = raw_store(&tokens, true);
            if let RawKvEvent::BlockStored { medium, .. } = &mut disk {
                *medium = Some("STORAGE".into());
            }
            replay.apply(disk).await;
            assert_eq!(replay.cached(&tokens), 4_992);
        }

        #[tokio::test]
        async fn root_tail_is_not_reused_after_a_complete_prefix_match() {
            let tokens: Vec<u32> = (0..15_133).collect();
            let mut replay = Replay::new(SUB as u32);
            replay.store(&tokens[..5_000]).await;
            replay.apply(raw_store(&tokens, false)).await;
            assert_eq!(replay.cached(&tokens), BLOCK);
        }

        #[tokio::test]
        async fn deeper_full_chunk_does_not_double_count_a_host_root_tail() {
            let tokens: Vec<u32> = (0..15_133).collect();
            let mut replay = Replay::new(SUB as u32);
            replay.store(&tokens[..5_000]).await;
            let mut disk = raw_store(&tokens, false);
            if let RawKvEvent::BlockStored { medium, .. } = &mut disk {
                *medium = Some("STORAGE".into());
            }
            replay.apply(disk).await;
            assert_eq!(replay.cached(&tokens), BLOCK);
        }

        #[tokio::test]
        async fn frontend_tail_lookup_cannot_recover_stores_dropped_by_publisher() {
            let tokens: Vec<u32> = (0..15_133).collect();
            let mut replay = Replay::new(0);
            replay.store(&tokens).await;
            assert_eq!(replay.cached(&tokens), BLOCK);
        }
    }

    #[test]
    fn exact_source_key_prevents_cross_owner_coalescing() {
        let mut coalescer = PlacementEventCoalescer::new(128);
        let first = stored(None, 1);
        assert!(
            coalescer
                .push(ResidencyDomain::Worker, first)
                .into_iter()
                .flatten()
                .next()
                .is_none()
        );
        let second = stored(Some(1), 2);
        let output = coalescer
            .push(ResidencyDomain::CacheOwner, second)
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        assert_eq!(output.len(), 1);
        assert!(matches!(
            &output[0].event.data,
            KvCacheEventData::Stored(data) if data.blocks.len() == 1
        ));
    }
}
