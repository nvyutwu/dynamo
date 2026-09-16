// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-tier registry of [`LowerTierIndexer`] instances and helpers for walking
//! the device → host → disk continuation chain.
//!
//! The primary KV indexer (radix tree) handles device-tier overlap scoring.
//! When a request arrives, we want to extend the per-worker match by walking
//! whichever lower tiers a worker has registered. [`LowerTierIndexers`] holds
//! one [`ThreadPoolIndexer<LowerTierIndexer>`] per non-device [`StorageTier`]
//! and lazily allocates each tier on first event arrival.
//!
//! Both the request-plane indexer (`dynamo-llm`) and the standalone HTTP
//! indexer (this crate's `services::indexer` module) share this implementation
//! so tier semantics stay aligned across the two surfaces.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::indexer::{
    KvIndexerMetrics, LowerTierContinuation, LowerTierIndexer, LowerTierMatchDetails, MatchDetails,
    ThreadPoolIndexer, WireTieredMatchDetails, record_unsupported_residency_event,
};
use crate::kv_hints::{KvTransferCandidateSource, KvTransferCandidates};
use crate::protocols::{
    BlockHashOptions, LocalBlockHash, ResidencyProjection, ResidencyRoutingSnapshot, RouterEvent,
    StorageTier, WorkerWithDpRank, compute_block_hash_for_seq,
};
use arc_swap::ArcSwap;
use rustc_hash::FxHashMap;

/// Holds one per-tier [`ThreadPoolIndexer<LowerTierIndexer>`] for every
/// non-device [`StorageTier`] that has received at least one event.
#[derive(Clone)]
pub struct LowerTierIndexers {
    metrics: Option<Arc<KvIndexerMetrics>>,
    num_threads: usize,
    block_size: u32,
    routing_snapshot: Arc<ArcSwap<ResidencyRoutingSnapshot>>,
    indexers: Arc<RwLock<HashMap<StorageTier, Arc<ThreadPoolIndexer<LowerTierIndexer>>>>>,
}

impl LowerTierIndexers {
    /// Metrics-less constructor for call sites without a `KvIndexerMetrics` handle.
    /// Router production assembly should use [`new_with_metrics`](Self::new_with_metrics)
    /// so lower-tier traffic is included in `kv_cache_events_applied`.
    pub fn new(num_threads: usize, block_size: u32) -> Self {
        Self::new_with_metrics(num_threads, block_size, None)
    }

    /// Same as [`new`](Self::new) but wires `kv_cache_events_applied`
    /// counters into every lazily created per-tier indexer, matching the
    /// observability of the device-tier path.
    pub fn new_with_metrics(
        num_threads: usize,
        block_size: u32,
        metrics: Option<Arc<KvIndexerMetrics>>,
    ) -> Self {
        assert!(
            num_threads > 0,
            "lower-tier indexer threads must be non-zero"
        );
        Self {
            num_threads,
            block_size,
            metrics,
            routing_snapshot: Arc::new(ArcSwap::from_pointee(ResidencyRoutingSnapshot::default())),
            indexers: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Replace the immutable owner-to-worker projection used by future
    /// lookups. Discovery and liveness reconciliation happen outside this
    /// crate; the indexer only consumes the already-resolved snapshot.
    pub fn set_residency_projection(&self, projection: ResidencyProjection) {
        self.set_residency_routing_snapshot(ResidencyRoutingSnapshot::from_projection(projection));
    }

    pub fn set_residency_routing_snapshot(&self, snapshot: ResidencyRoutingSnapshot) {
        self.routing_snapshot.store(Arc::new(snapshot));
    }

    /// Return the per-tier indexer for `storage_tier`, lazily allocating it
    /// the first time a non-device tier is seen.
    pub fn get_or_create(
        &self,
        storage_tier: StorageTier,
    ) -> Arc<ThreadPoolIndexer<LowerTierIndexer>> {
        debug_assert!(!storage_tier.is_gpu());
        if let Some(indexer) = self.indexers.read().unwrap().get(&storage_tier).cloned() {
            return indexer;
        }
        self.indexers
            .write()
            .unwrap()
            .entry(storage_tier)
            .or_insert_with(|| {
                Arc::new(ThreadPoolIndexer::new_with_metrics(
                    LowerTierIndexer::new(),
                    self.num_threads,
                    self.block_size,
                    self.metrics.clone(),
                ))
            })
            .clone()
    }

    /// All currently allocated lower-tier indexers, in unspecified order.
    pub fn all(&self) -> Vec<Arc<ThreadPoolIndexer<LowerTierIndexer>>> {
        self.indexers.read().unwrap().values().cloned().collect()
    }

    /// All currently allocated lower-tier indexers paired with the
    /// [`StorageTier`] each one indexes. Used by callers that need to retag
    /// per-tier dumps (e.g. peer-recovery).
    pub fn entries(&self) -> Vec<(StorageTier, Arc<ThreadPoolIndexer<LowerTierIndexer>>)> {
        self.indexers
            .read()
            .unwrap()
            .iter()
            .map(|(tier, indexer)| (*tier, indexer.clone()))
            .collect()
    }

    fn is_empty(&self) -> bool {
        self.indexers.read().unwrap().is_empty()
    }

    /// Lookup without allocation; returns `None` if the tier is unseen.
    pub fn get(
        &self,
        storage_tier: StorageTier,
    ) -> Option<Arc<ThreadPoolIndexer<LowerTierIndexer>>> {
        self.indexers.read().unwrap().get(&storage_tier).cloned()
    }

    pub fn record_unsupported_residency_event(&self, event: &RouterEvent) {
        record_unsupported_residency_event(self.metrics.as_deref(), event);
    }
}

/// Native tiered-match container: the device-tier match plus a per-tier map
/// of lower-tier hits. Wire-friendly representations live in
/// [`WireTieredMatchDetails`]; conversions in both directions are provided.
#[derive(Debug, Clone, Default)]
pub struct TieredMatchDetails {
    pub device: MatchDetails,
    pub lower_tier: HashMap<StorageTier, LowerTierMatchDetails>,
}

impl TieredMatchDetails {
    pub fn kv_transfer_candidates(&self) -> Option<&KvTransferCandidates> {
        self.lower_tier
            .get(&StorageTier::HostPinned)
            .and_then(|details| details.kv_transfer_candidates.as_ref())
            .or(self.device.kv_transfer_candidates.as_ref())
    }
}

impl From<&TieredMatchDetails> for WireTieredMatchDetails {
    fn from(d: &TieredMatchDetails) -> Self {
        Self {
            device: d.device.overlap_scores.clone().into(),
            lower_tier: d
                .lower_tier
                .iter()
                .map(|(tier, details)| (*tier, details.into()))
                .collect(),
        }
    }
}

impl From<WireTieredMatchDetails> for TieredMatchDetails {
    fn from(w: WireTieredMatchDetails) -> Self {
        // `last_matched_hashes` is only needed server-side to seed the tier walk,
        // so we leave it empty on the inbound side.
        let mut lower_tier = HashMap::with_capacity(w.lower_tier.len());
        for (tier, details) in w.lower_tier {
            if lower_tier.insert(tier, details.into()).is_some() {
                tracing::warn!(
                    ?tier,
                    "Duplicate StorageTier in WireTieredMatchDetails; keeping last entry"
                );
            }
        }
        Self {
            device: MatchDetails {
                overlap_scores: w.device.into(),
                ..Default::default()
            },
            lower_tier,
        }
    }
}

/// The order in which lower tiers are walked when extending a match. Device
/// → HostPinned → Disk → External.
pub fn lower_tier_query_order() -> [StorageTier; 3] {
    [
        StorageTier::HostPinned,
        StorageTier::Disk,
        StorageTier::External,
    ]
}

#[derive(Debug, Clone, Copy, Default)]
pub struct LowerTierQueryOptions {
    pub retain_kv_transfer_chain: bool,
}

/// The request tokens needed to extend a lower-tier walk into the block after
/// the last complete match, at the engine's hash granularity.
///
/// Engines that offload a prompt's partial tail publish it as a chain of
/// `sub_block_size`-token blocks hanging off the last complete block, and the
/// lower-tier index stores that chain with the same edge scheme as complete
/// blocks. The walk therefore only needs the request's sub-block local hashes
/// for the block a worker stopped in, computed lazily per block index.
#[derive(Debug, Clone, Copy)]
pub struct PartialTailQuery<'a> {
    pub tokens: &'a [u32],
    pub block_size: u32,
    pub sub_block_size: u32,
    pub hash_options: BlockHashOptions<'a>,
}

impl PartialTailQuery<'_> {
    pub fn is_enabled(&self) -> bool {
        self.sub_block_size > 0
            && self.block_size.is_multiple_of(self.sub_block_size)
            && self.sub_block_size < self.block_size
            // Multimodal placeholder expansion and the Eagle window are defined
            // per full block; a sub-block split would hash differently from
            // the engine (convert.rs drops those tails for the same reason).
            && self.hash_options.block_mm_infos.is_none()
            && self.hash_options.is_eagle != Some(true)
    }

    /// Local hashes of the complete sub-blocks inside block `block_idx`, in
    /// order. Empty when the block is complete (its full hash is walked by the
    /// caller) or when fewer than one sub-block of tokens follow it.
    pub fn sub_block_hashes(&self, block_idx: usize) -> Vec<LocalBlockHash> {
        let start = block_idx * self.block_size as usize;
        if start >= self.tokens.len() {
            return Vec::new();
        }
        let end = (start + self.block_size as usize).min(self.tokens.len());
        if end - start >= self.block_size as usize {
            return Vec::new();
        }
        compute_block_hash_for_seq(
            &self.tokens[start..end],
            self.sub_block_size,
            self.hash_options,
        )
    }
}

/// Walk every allocated lower tier in [`lower_tier_query_order`] and build a
/// per-tier match map seeded from `device_matches`. Per-worker continuations
/// flow forward: a worker that matched N device blocks starts the host walk
/// at block N (anchored on its last device hash), and so on.
pub fn query_lower_tiers(
    indexers: &LowerTierIndexers,
    sequence: &[LocalBlockHash],
    device_matches: &MatchDetails,
) -> HashMap<StorageTier, LowerTierMatchDetails> {
    query_lower_tiers_with_options(
        indexers,
        sequence,
        device_matches,
        LowerTierQueryOptions::default(),
    )
}

fn merge_kv_transfer_tier_candidates(
    device_candidates: Option<&KvTransferCandidates>,
    tier_matches: &LowerTierMatchDetails,
    routing_snapshot: Arc<ResidencyRoutingSnapshot>,
) -> Option<KvTransferCandidates> {
    let mut block_hashes = device_candidates
        .map(|candidates| candidates.block_hashes.clone())
        .unwrap_or_default();
    let mut owner_prefix_blocks: FxHashMap<KvTransferCandidateSource, usize> = FxHashMap::default();

    if let Some(candidates) = device_candidates {
        owner_prefix_blocks.extend(candidates.owner_prefix_blocks.iter().copied());
    }

    let Some(extensions) = tier_matches.kv_transfer_extensions.as_ref() else {
        return device_candidates.cloned();
    };

    // KV transfer hints intentionally retain one compact root-aligned chain. The
    // lower-tier walk records each matched child hash once at its request-block
    // position and tracks per-owner depths separately, avoiding per-worker hash
    // copies on the lookup hot path. Positional equality assumes
    // ExternalSequenceBlockHash is stable across workers and tiers for the same
    // request-prefix position.
    for (pos, hash) in &extensions.block_hashes {
        if block_hashes.len() < *pos {
            break;
        }

        if *pos < block_hashes.len() {
            if block_hashes[*pos] != *hash {
                break;
            }
        } else {
            block_hashes.push(*hash);
        }
    }

    owner_prefix_blocks.extend(
        extensions
            .owner_prefix_blocks
            .iter()
            .filter(|(_, blocks)| **blocks <= block_hashes.len())
            .map(|(worker, blocks)| (*worker, *blocks)),
    );

    let mut owner_prefix_blocks = owner_prefix_blocks
        .into_iter()
        .filter(|(_, blocks)| *blocks > 0)
        .collect::<Vec<_>>();
    if block_hashes.is_empty() || owner_prefix_blocks.is_empty() {
        return None;
    }
    owner_prefix_blocks.sort_unstable_by_key(|(worker, _)| *worker);

    Some(KvTransferCandidates {
        block_hashes,
        owner_prefix_blocks,
        routing_snapshot: Some(routing_snapshot),
    })
}

pub fn query_lower_tiers_with_options(
    indexers: &LowerTierIndexers,
    sequence: &[LocalBlockHash],
    device_matches: &MatchDetails,
    options: LowerTierQueryOptions,
) -> HashMap<StorageTier, LowerTierMatchDetails> {
    if indexers.is_empty() {
        return HashMap::new();
    }
    let snapshot = indexers.routing_snapshot.load_full();
    query_lower_tiers_with_options_and_snapshot(
        indexers,
        sequence,
        device_matches,
        options,
        snapshot,
    )
}

/// [`query_lower_tiers_with_options`] extended into the request's partial
/// tail when `tail` is given; see [`PartialTailQuery`].
pub fn query_lower_tiers_with_options_and_tail(
    indexers: &LowerTierIndexers,
    sequence: &[LocalBlockHash],
    device_matches: &MatchDetails,
    options: LowerTierQueryOptions,
    tail: Option<&PartialTailQuery<'_>>,
) -> HashMap<StorageTier, LowerTierMatchDetails> {
    if indexers.is_empty() {
        return HashMap::new();
    }
    let snapshot = indexers.routing_snapshot.load_full();
    query_lower_tiers_with_tail(indexers, sequence, device_matches, options, snapshot, tail)
}

pub fn query_lower_tiers_with_options_and_projection(
    indexers: &LowerTierIndexers,
    sequence: &[LocalBlockHash],
    device_matches: &MatchDetails,
    options: LowerTierQueryOptions,
    projection: &ResidencyProjection,
) -> HashMap<StorageTier, LowerTierMatchDetails> {
    query_lower_tiers_with_options_and_snapshot(
        indexers,
        sequence,
        device_matches,
        options,
        Arc::new(ResidencyRoutingSnapshot::from_projection(
            projection.clone(),
        )),
    )
}

pub fn query_lower_tiers_with_options_and_snapshot(
    indexers: &LowerTierIndexers,
    sequence: &[LocalBlockHash],
    device_matches: &MatchDetails,
    options: LowerTierQueryOptions,
    snapshot: Arc<ResidencyRoutingSnapshot>,
) -> HashMap<StorageTier, LowerTierMatchDetails> {
    query_lower_tiers_with_tail(indexers, sequence, device_matches, options, snapshot, None)
}

/// Extend each tier's complete-block match into the request's partial tail.
///
/// Workers are grouped by the block they stopped in; each group walks that
/// block's sub-block hashes from its last matched hash (or from the root when
/// the request is shorter than one block). Tail hits do not move the
/// continuation handed to the next tier: a deeper tier still starts from the
/// complete-block boundary.
fn walk_partial_tail(
    indexer: &LowerTierIndexer,
    tail: &PartialTailQuery<'_>,
    continuations: &FxHashMap<WorkerWithDpRank, LowerTierContinuation>,
    snapshot: &ResidencyRoutingSnapshot,
) -> FxHashMap<WorkerWithDpRank, usize> {
    let mut tail_hits = FxHashMap::default();
    let mut by_block: FxHashMap<usize, FxHashMap<WorkerWithDpRank, LowerTierContinuation>> =
        FxHashMap::default();
    for (worker, continuation) in continuations {
        by_block.entry(continuation.start_pos).or_default().insert(
            *worker,
            LowerTierContinuation {
                start_pos: 0,
                last_matched_hash: continuation.last_matched_hash,
            },
        );
    }

    // A request shorter than one block has no complete-block walk to seed
    // workers from; its tail hangs off the root and is found via root edges.
    if (tail.tokens.len() as u32) < tail.block_size {
        by_block.entry(0).or_default();
    }

    for (block_idx, mut group) in by_block {
        let sub_hashes = tail.sub_block_hashes(block_idx);
        let Some(&first_hash) = sub_hashes.first() else {
            continue;
        };
        if block_idx == 0 {
            for worker in indexer.root_workers(first_hash, snapshot.projection()) {
                group
                    .entry(worker)
                    .or_insert_with(|| LowerTierContinuation::from_root(0));
            }
        }
        let matches = indexer.query_match_details_with_options_and_snapshot(
            &sub_hashes,
            &group,
            false,
            snapshot,
        );
        for (worker, hits) in matches.hits {
            if hits > 0 {
                tail_hits.insert(worker, hits);
            }
        }
    }
    tail_hits
}

pub fn query_lower_tiers_with_tail(
    indexers: &LowerTierIndexers,
    sequence: &[LocalBlockHash],
    device_matches: &MatchDetails,
    options: LowerTierQueryOptions,
    snapshot: Arc<ResidencyRoutingSnapshot>,
    tail: Option<&PartialTailQuery<'_>>,
) -> HashMap<StorageTier, LowerTierMatchDetails> {
    let tail = tail.filter(|tail| tail.is_enabled());
    let projection = snapshot.projection();
    let mut continuations = LowerTierMatchDetails::default().next_continuations;
    for (worker, matched_blocks) in &device_matches.overlap_scores.scores {
        let Some(last_hash) = device_matches.last_matched_hashes.get(worker).copied() else {
            debug_assert!(
                false,
                "device match result missing last matched hash for worker {worker:?}"
            );
            continue;
        };

        continuations.insert(
            *worker,
            LowerTierContinuation::new(*matched_blocks as usize, last_hash),
        );
    }

    let mut lower_tier_matches = HashMap::new();

    for storage_tier in lower_tier_query_order() {
        let Some(indexer) = indexers.get(storage_tier) else {
            continue;
        };

        if let Some(&first_hash) = sequence.first() {
            let root_workers: Vec<_> = indexer.backend().root_workers(first_hash, projection);
            for worker in root_workers.iter() {
                continuations
                    .entry(*worker)
                    .or_insert_with(|| LowerTierContinuation::from_root(0));
            }
        }

        let retain_kv_transfer_chain =
            options.retain_kv_transfer_chain && storage_tier == StorageTier::HostPinned;
        let mut tier_matches = indexer
            .backend()
            .query_match_details_with_options_and_snapshot(
                sequence,
                &continuations,
                retain_kv_transfer_chain,
                &snapshot,
            );
        if retain_kv_transfer_chain {
            tier_matches.kv_transfer_candidates = merge_kv_transfer_tier_candidates(
                device_matches.kv_transfer_candidates.as_ref(),
                &tier_matches,
                snapshot.clone(),
            );
        }
        if let Some(tail) = tail {
            tier_matches.tail_sub_block_size = tail.sub_block_size;
            tier_matches.tail_hits = walk_partial_tail(
                indexer.backend(),
                tail,
                &tier_matches.next_continuations,
                &snapshot,
            );
        }
        let matched_workers = tier_matches.hits.values().filter(|&&hits| hits > 0).count();
        tracing::debug!(
            ?storage_tier,
            queried_workers = continuations.len(),
            matched_workers,
            tail_matched_workers = tier_matches.tail_hits.len(),
            "Queried lower-tier indexer"
        );
        continuations = tier_matches.next_continuations.clone();
        lower_tier_matches.insert(storage_tier, tier_matches);
    }

    lower_tier_matches
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::{
        CacheOwnerId, CacheSemanticsId, DcId, IdentitySource, IndexerDomainId, PoolId,
        RoutingScopeId, StableDpSlotId,
    };
    use crate::indexer::KvIndexerInterface;
    use crate::protocols::{
        ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheStoreData,
        LocalBlockHash, OverlapScores, RouterEvent, WorkerWithDpRank,
    };
    use crate::test_utils::{router_event, stored_blocks_with_sequence_hashes};

    fn local_hashes(values: &[u64]) -> Vec<LocalBlockHash> {
        values.iter().copied().map(LocalBlockHash).collect()
    }

    fn cache_owner_id() -> CacheOwnerId {
        CacheOwnerId::new(
            PoolId::new(
                IndexerDomainId::new(
                    CacheSemanticsId::new([1; 16], IdentitySource::Explicit),
                    RoutingScopeId::new([2; 16], IdentitySource::Explicit),
                ),
                DcId::new(3),
            ),
            StableDpSlotId::new([4; 16], IdentitySource::Explicit),
        )
    }

    fn store_event(
        worker_id: u64,
        dp_rank: u32,
        event_id: u64,
        parent_hash: Option<u64>,
        local_values: &[u64],
        external_hashes: &[u64],
    ) -> crate::protocols::RouterEvent {
        router_event(
            worker_id,
            event_id,
            dp_rank,
            KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: parent_hash.map(ExternalSequenceBlockHash),
                start_position: None,
                blocks: stored_blocks_with_sequence_hashes(
                    &local_hashes(local_values),
                    external_hashes,
                ),
            }),
        )
    }

    #[tokio::test]
    async fn cache_owner_lane_survives_worker_reprojection_and_clears_once() {
        let owner = cache_owner_id();
        let indexers = LowerTierIndexers::new(2, 4);
        let lower = indexers.get_or_create(StorageTier::HostPinned);
        let store =
            |worker_id: u64, event_id: u64, parent_hash: Option<u64>, local: u64, external: u64| {
                RouterEvent::with_cache_owner(
                    worker_id,
                    KvCacheEvent {
                        event_id,
                        dp_rank: 3,
                        data: KvCacheEventData::Stored(KvCacheStoreData {
                            parent_hash: parent_hash.map(ExternalSequenceBlockHash),
                            start_position: None,
                            blocks: stored_blocks_with_sequence_hashes(
                                &[LocalBlockHash(local)],
                                &[external],
                            ),
                        }),
                    },
                    StorageTier::HostPinned,
                    owner,
                )
            };

        lower
            .apply_event_and_wait(store(17, 1, None, 11, 101))
            .await
            .unwrap();
        lower
            .apply_event_and_wait(store(29, 2, Some(101), 12, 102))
            .await
            .unwrap();

        let dumped = lower.dump_events().await.unwrap();
        assert!(dumped.iter().all(|event| event.state_source == Some(owner)));

        let replacement = WorkerWithDpRank::new(29, 3);
        let projection = ResidencyProjection::new([(owner, replacement)]).unwrap();
        assert_eq!(
            lower
                .backend()
                .root_workers(LocalBlockHash(11), &projection),
            vec![replacement]
        );

        lower
            .apply_event_and_wait(RouterEvent::with_cache_owner(
                replacement.worker_id,
                KvCacheEvent {
                    event_id: 3,
                    dp_rank: replacement.dp_rank,
                    data: KvCacheEventData::Cleared,
                },
                StorageTier::HostPinned,
                owner,
            ))
            .await
            .unwrap();
        assert!(
            lower
                .backend()
                .root_workers(LocalBlockHash(11), &projection)
                .is_empty()
        );
    }

    #[test]
    fn query_lower_tiers_returns_empty_when_no_tiers_allocated() {
        let indexers = LowerTierIndexers::new(1, 4);

        // Mismatched device_matches: a score entry with no paired
        // `last_matched_hashes` entry. Would `debug_assert!`-panic in the
        // old body; the early-return must skip the seeding loop entirely.
        let mut overlap_scores = OverlapScores::new();
        overlap_scores
            .scores
            .insert(WorkerWithDpRank::new(99, 0), 3);
        let device_matches = MatchDetails {
            overlap_scores,
            last_matched_hashes: Default::default(),
            kv_transfer_candidates: None,
        };

        let sequence = vec![LocalBlockHash(1), LocalBlockHash(2)];
        let result = query_lower_tiers(&indexers, &sequence, &device_matches);
        assert!(result.is_empty());
    }

    fn tail_query<'a>(tokens: &'a [u32], block_size: u32, sub: u32) -> PartialTailQuery<'a> {
        PartialTailQuery {
            tokens,
            block_size,
            sub_block_size: sub,
            hash_options: BlockHashOptions::default(),
        }
    }

    #[tokio::test]
    async fn partial_tail_walk_extends_complete_block_match_at_sub_block_granularity() {
        // Block 8, sub-block 4. Worker 7 holds blocks [0,8) and [8,16) plus the
        // 4-token tail [16,20) chained off block 1, exactly as the engine
        // publishes a partial-tail offload.
        let tokens: Vec<u32> = (100..122).collect();
        let full = compute_block_hash_for_seq(&tokens, 8, BlockHashOptions::default());
        let sub = compute_block_hash_for_seq(&tokens[16..20], 4, BlockHashOptions::default());
        assert_eq!((full.len(), sub.len()), (2, 1));

        let indexers = LowerTierIndexers::new(1, 8);
        let lower_tier = indexers.get_or_create(StorageTier::HostPinned);
        lower_tier
            .apply_event(store_event(
                7,
                0,
                0,
                None,
                &[full[0].0, full[1].0],
                &[901, 902],
            ))
            .await;
        lower_tier
            .apply_event(store_event(7, 0, 1, Some(902), &[sub[0].0], &[903]))
            .await;
        let _ = lower_tier.dump_events().await.unwrap();

        let device = MatchDetails::new();
        let tail = tail_query(&tokens, 8, 4);
        let result = query_lower_tiers_with_tail(
            &indexers,
            &full,
            &device,
            LowerTierQueryOptions::default(),
            indexers.routing_snapshot.load_full(),
            Some(&tail),
        );
        let host = &result[&StorageTier::HostPinned];
        let worker = WorkerWithDpRank::new(7, 0);
        assert_eq!(host.hits[&worker], 2);
        assert_eq!(host.tail_hits[&worker], 1);
        assert_eq!(host.tail_sub_block_size, 4);

        // Without a tail query the result is unchanged from today.
        let plain = query_lower_tiers(&indexers, &full, &device);
        let host = &plain[&StorageTier::HostPinned];
        assert_eq!(host.hits[&worker], 2);
        assert!(host.tail_hits.is_empty());
    }

    #[tokio::test]
    async fn partial_tail_walk_finds_a_root_tail_for_a_request_shorter_than_one_block() {
        let tokens: Vec<u32> = (200..206).collect();
        let sub = compute_block_hash_for_seq(&tokens[..4], 4, BlockHashOptions::default());
        let indexers = LowerTierIndexers::new(1, 8);
        let lower_tier = indexers.get_or_create(StorageTier::HostPinned);
        lower_tier
            .apply_event(store_event(9, 0, 0, None, &[sub[0].0], &[801]))
            .await;
        let _ = lower_tier.dump_events().await.unwrap();

        // No complete block: the request sequence is empty and no worker is
        // seeded by the complete-block walk; the tail walk seeds from the root.
        let tail = tail_query(&tokens, 8, 4);
        let result = query_lower_tiers_with_tail(
            &indexers,
            &[],
            &MatchDetails::new(),
            LowerTierQueryOptions::default(),
            indexers.routing_snapshot.load_full(),
            Some(&tail),
        );
        let host = &result[&StorageTier::HostPinned];
        let worker = WorkerWithDpRank::new(9, 0);
        assert_eq!(host.hits.get(&worker).copied().unwrap_or(0), 0);
        assert_eq!(host.tail_hits[&worker], 1);
    }

    #[test]
    fn partial_tail_query_is_disabled_for_incompatible_shapes() {
        let tokens: Vec<u32> = (0..20).collect();
        assert!(tail_query(&tokens, 8, 4).is_enabled());
        assert!(!tail_query(&tokens, 8, 0).is_enabled());
        assert!(!tail_query(&tokens, 8, 3).is_enabled());
        assert!(!tail_query(&tokens, 8, 8).is_enabled());
        // A complete block yields no sub-block hashes; the tail of [16,20) does.
        assert!(tail_query(&tokens, 8, 4).sub_block_hashes(0).is_empty());
        assert_eq!(tail_query(&tokens, 8, 4).sub_block_hashes(2).len(), 1);
        assert!(tail_query(&tokens, 8, 4).sub_block_hashes(3).is_empty());
    }

    #[tokio::test]
    async fn query_lower_tiers_extends_kv_transfer_chain_from_device_prefix() {
        let indexers = LowerTierIndexers::new(1, 4);
        let worker = WorkerWithDpRank::new(7, 0);
        let lower_tier = indexers.get_or_create(StorageTier::HostPinned);
        lower_tier
            .apply_event(store_event(7, 0, 0, Some(101), &[12], &[102]))
            .await;
        let _ = lower_tier.dump_events().await.unwrap();

        let mut overlap_scores = OverlapScores::new();
        overlap_scores.scores.insert(worker, 1);
        let mut last_matched_hashes = FxHashMap::default();
        last_matched_hashes.insert(worker, ExternalSequenceBlockHash(101));
        let device_matches = MatchDetails {
            overlap_scores,
            last_matched_hashes,
            kv_transfer_candidates: Some(KvTransferCandidates {
                block_hashes: vec![ExternalSequenceBlockHash(101)],
                owner_prefix_blocks: vec![(worker.into(), 1)],
                routing_snapshot: None,
            }),
        };

        let sequence = local_hashes(&[11, 12, 13]);
        let result = query_lower_tiers_with_options(
            &indexers,
            &sequence,
            &device_matches,
            LowerTierQueryOptions {
                retain_kv_transfer_chain: true,
            },
        );
        let candidates = result
            .get(&StorageTier::HostPinned)
            .and_then(|details| details.kv_transfer_candidates.as_ref())
            .unwrap();

        assert_eq!(
            candidates.block_hashes,
            vec![
                ExternalSequenceBlockHash(101),
                ExternalSequenceBlockHash(102),
            ]
        );
        assert_eq!(candidates.owner_prefix_blocks, vec![(worker.into(), 2)]);
    }

    #[tokio::test]
    async fn query_lower_tiers_keeps_divergent_hint_source_at_shared_prefix() {
        let indexers = LowerTierIndexers::new(1, 4);
        let worker_1 = WorkerWithDpRank::new(7, 0);
        let worker_2 = WorkerWithDpRank::new(8, 0);
        let lower_tier = indexers.get_or_create(StorageTier::HostPinned);
        lower_tier
            .apply_event(store_event(7, 0, 0, Some(101), &[12], &[102]))
            .await;
        lower_tier
            .apply_event(store_event(8, 0, 1, Some(101), &[12], &[202]))
            .await;
        let _ = lower_tier.dump_events().await.unwrap();

        let mut overlap_scores = OverlapScores::new();
        overlap_scores.scores.insert(worker_1, 1);
        overlap_scores.scores.insert(worker_2, 1);
        let mut last_matched_hashes = FxHashMap::default();
        last_matched_hashes.insert(worker_1, ExternalSequenceBlockHash(101));
        last_matched_hashes.insert(worker_2, ExternalSequenceBlockHash(101));
        let device_matches = MatchDetails {
            overlap_scores,
            last_matched_hashes,
            kv_transfer_candidates: Some(KvTransferCandidates {
                block_hashes: vec![ExternalSequenceBlockHash(101)],
                owner_prefix_blocks: vec![(worker_1.into(), 1), (worker_2.into(), 1)],
                routing_snapshot: None,
            }),
        };

        let sequence = local_hashes(&[11, 12, 13]);
        let result = query_lower_tiers_with_options(
            &indexers,
            &sequence,
            &device_matches,
            LowerTierQueryOptions {
                retain_kv_transfer_chain: true,
            },
        );
        let candidates = result
            .get(&StorageTier::HostPinned)
            .and_then(|details| details.kv_transfer_candidates.as_ref())
            .unwrap();

        assert_eq!(
            candidates.block_hashes,
            vec![
                ExternalSequenceBlockHash(101),
                ExternalSequenceBlockHash(102),
            ]
        );
        assert_eq!(
            candidates.owner_prefix_blocks,
            vec![(worker_1.into(), 2), (worker_2.into(), 1)]
        );
    }

    #[tokio::test]
    async fn query_lower_tiers_retains_kv_transfer_chain_when_enabled() {
        let indexers = LowerTierIndexers::new(1, 4);
        let lower_tier = indexers.get_or_create(StorageTier::HostPinned);
        lower_tier
            .apply_event(store_event(7, 0, 0, None, &[11, 12], &[101, 102]))
            .await;
        let _ = lower_tier.dump_events().await.unwrap();

        let sequence = local_hashes(&[11, 12, 13]);
        let result = query_lower_tiers_with_options(
            &indexers,
            &sequence,
            &MatchDetails::default(),
            LowerTierQueryOptions {
                retain_kv_transfer_chain: true,
            },
        );
        let candidates = result
            .get(&StorageTier::HostPinned)
            .and_then(|details| details.kv_transfer_candidates.as_ref())
            .unwrap();

        assert_eq!(
            candidates.block_hashes,
            vec![
                ExternalSequenceBlockHash(101),
                ExternalSequenceBlockHash(102)
            ]
        );
        assert_eq!(
            candidates.owner_prefix_blocks,
            vec![(WorkerWithDpRank::new(7, 0).into(), 2)]
        );
    }
}
