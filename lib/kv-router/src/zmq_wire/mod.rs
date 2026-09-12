// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Wire-format types for vLLM ZMQ KV event streams.
//!
//! These types mirror the Python `msgspec`-defined structures emitted by vLLM
//! engines over ZMQ PUB sockets. They are independent of the dynamo runtime
//! and can be used by any crate that needs to decode the raw ZMQ payloads.

use std::sync::Arc;
use std::sync::atomic::AtomicU32;

use rmp_serde as rmps;
use rustc_hash::FxHashMap;
use rustc_hash::FxHashSet;

use crate::protocols::{DpRank, PlacementEvent, StorageTier, WorkerWithDpRank};

mod convert;
mod deserialize;
mod extra_keys;
mod filter;
#[cfg(test)]
mod tests;
mod types;

pub use convert::{
    StoredBlockOptions, convert_event, create_stored_block_from_parts, create_stored_blocks,
    normalize_mm_placeholder_runs, normalize_mm_token_runs,
};
pub use extra_keys::{
    extra_keys_to_block_mm_infos, extra_keys_to_cache_namespace, mark_mm_hash_for_extra_key,
    parse_mm_hash_from_extra_key,
};
pub use filter::KvCacheSpecKind;
pub use types::{
    BlockHashValue, ExtraKeyItem, KvEventBatch, KvEventOwnership, KvTokenIds, Locality, RawKvEvent,
};

use filter::KvCacheEventMetadata;

pub fn decode_event_batch(payload: &[u8]) -> Result<KvEventBatch, rmps::decode::Error> {
    rmps::from_slice(payload)
}

#[derive(Debug, Clone)]
pub struct ZmqEventNormalizer {
    kv_block_size: u32,
    /// Model's image placeholder token id, when MM-aware routing is active.
    /// Lets `convert_event` normalize vLLM BlockStored events to the canonical
    /// pad_value scheme. `None` for text-only models / non-MM deployments.
    image_token_id: Option<u32>,
    /// Model's video placeholder token id. When an event contains this token,
    /// image and video objects are normalized with modality-aware mapping.
    video_token_id: Option<u32>,
    warning_count: Arc<AtomicU32>,
    group_metadata: FxHashMap<(DpRank, u32), KvCacheGroupMetadata>,
    cache_namespaces: FxHashMap<(WorkerWithDpRank, u64), CacheNamespaceState>,
    cache_namespace_tiers: FxHashMap<(WorkerWithDpRank, u64), FxHashSet<StorageTier>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum CacheNamespaceState {
    Namespaced(Arc<str>),
    Ambiguous,
}

#[derive(Debug, Clone, Copy)]
struct KvCacheGroupMetadata {
    kind: KvCacheSpecKind,
    sliding_window: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZmqEventFilterReason {
    IgnoredEvent,
    NonLocalLocality,
    UnknownMedium,
    AmbiguousScopedClear,
    UnsupportedOwnership,
    UnknownOwnership,
    AmbiguousCacheNamespace,
    NonMainAttentionKind,
    UnknownKind,
    NonMainAttentionGroup,
    UnlearnedGroupIdx,
}

impl ZmqEventFilterReason {
    pub fn as_label(self) -> &'static str {
        match self {
            Self::IgnoredEvent => "ignored_event",
            Self::NonLocalLocality => "non_local_locality",
            Self::UnknownMedium => "unknown_medium",
            Self::AmbiguousScopedClear => "ambiguous_scoped_clear",
            Self::UnsupportedOwnership => "unsupported_ownership",
            Self::UnknownOwnership => "unknown_ownership",
            Self::AmbiguousCacheNamespace => "ambiguous_cache_namespace",
            Self::NonMainAttentionKind => "non_main_attention_kind",
            Self::UnknownKind => "unknown_kind",
            Self::NonMainAttentionGroup => "non_main_attention_group",
            Self::UnlearnedGroupIdx => "unlearned_group_idx",
        }
    }
}

impl ZmqEventNormalizer {
    pub fn new(kv_block_size: u32) -> Self {
        Self {
            kv_block_size,
            image_token_id: None,
            video_token_id: None,
            warning_count: Arc::new(AtomicU32::new(0)),
            group_metadata: FxHashMap::default(),
            cache_namespaces: FxHashMap::default(),
            cache_namespace_tiers: FxHashMap::default(),
        }
    }

    pub fn with_warning_count(kv_block_size: u32, warning_count: Arc<AtomicU32>) -> Self {
        Self {
            kv_block_size,
            image_token_id: None,
            video_token_id: None,
            warning_count,
            group_metadata: FxHashMap::default(),
            cache_namespaces: FxHashMap::default(),
            cache_namespace_tiers: FxHashMap::default(),
        }
    }

    /// Set the model's image placeholder token id so vLLM BlockStored events
    /// get normalized to the canonical pad_value scheme. No-op for text-only
    /// models (leave unset).
    pub fn with_image_token_id(mut self, image_token_id: Option<u32>) -> Self {
        self.image_token_id = image_token_id;
        self
    }

    pub fn with_video_token_id(mut self, video_token_id: Option<u32>) -> Self {
        self.video_token_id = video_token_id;
        self
    }

    pub fn preprocess(&mut self, raw: RawKvEvent, worker: WorkerWithDpRank) -> Option<RawKvEvent> {
        self.preprocess_with_reason(raw, worker).ok()
    }

    pub fn preprocess_with_reason(
        &mut self,
        raw: RawKvEvent,
        worker: WorkerWithDpRank,
    ) -> Result<RawKvEvent, ZmqEventFilterReason> {
        match raw.ownership() {
            Ok(KvEventOwnership::Framework) => {}
            Ok(KvEventOwnership::Kvcr) => {
                return Err(ZmqEventFilterReason::UnsupportedOwnership);
            }
            Err(_) => return Err(ZmqEventFilterReason::UnknownOwnership),
        }
        self.preprocess_residency_with_reason(raw, worker)
    }

    /// Normalize a version-gated state-agent stream which may contain both
    /// framework and vLLM-enriched KVCR transitions.
    pub fn preprocess_residency_with_reason(
        &mut self,
        mut raw: RawKvEvent,
        worker: WorkerWithDpRank,
    ) -> Result<RawKvEvent, ZmqEventFilterReason> {
        if raw.ownership().is_err() {
            return Err(ZmqEventFilterReason::UnknownOwnership);
        }
        if raw.is_ignored() {
            return Err(ZmqEventFilterReason::IgnoredEvent);
        }
        if matches!(
            &raw,
            RawKvEvent::AllBlocksCleared {
                medium: Some(_),
                ..
            }
        ) {
            return Err(ZmqEventFilterReason::AmbiguousScopedClear);
        }

        // Non-local events are dropped by policy (no shared-index consumer yet).
        // Classify them here, before the lower-tier bypass, so the gate covers
        // every tier (including Disk/External). Otherwise the listener would
        // accept the event, burn a next_event_id, and only drop it in
        // conversion, leaving an id gap the event processor mistakes for an
        // engine drop (engines_dropped_events).
        if matches!(raw.locality(), Some(Locality::Remote | Locality::Unknown)) {
            return Err(ZmqEventFilterReason::NonLocalLocality);
        }

        // Classify by medium before touching normalizer state:
        //  - Device / HostPinned (GPU, CPU offload #10368) stay on the normalizer
        //    path so their salted namespaces still propagate.
        //  - Disk / External (STORAGE) are hash-only lower-tier events with no
        //    extra_keys/cache_namespace, so they must not mutate per-group
        //    metadata or the salted-namespace chain and are outside the SW/SSM
        //    group filter's semantics; bypass straight to conversion, which keeps
        //    them (no event id is wasted).
        //  - Unrecognized media (e.g. vLLM 0.26.0 FS/OBJ) fail closed here so the
        //    listener records an intentional filter. Bypassing to conversion,
        //    which drops them, would instead accept the event, burn a
        //    next_event_id, and leave an id gap the event processor mistakes for
        //    an engine drop -- the same trap the locality gate above avoids.
        if let Some(m) = raw.medium() {
            match StorageTier::from_kv_medium(m) {
                Some(StorageTier::Device | StorageTier::HostPinned) => {}
                Some(_) => return Ok(raw),
                None => return Err(ZmqEventFilterReason::UnknownMedium),
            }
        }

        let metadata = raw.metadata();
        if matches!(raw, RawKvEvent::BlockStored { .. }) {
            self.learn_metadata(metadata, worker.dp_rank);
        }
        if let Some(reason) = self.filter_reason(metadata, worker.dp_rank) {
            return Err(reason);
        }
        self.propagate_cache_namespace(&mut raw, worker)?;
        Ok(raw)
    }

    pub fn normalize_preprocessed(
        &self,
        raw: RawKvEvent,
        event_id: u64,
        worker: WorkerWithDpRank,
    ) -> Option<PlacementEvent> {
        convert_event(
            raw,
            event_id,
            self.kv_block_size,
            worker,
            &self.warning_count,
            self.image_token_id,
            self.video_token_id,
        )
    }

    pub fn normalize(
        &mut self,
        raw: RawKvEvent,
        event_id: u64,
        worker: WorkerWithDpRank,
    ) -> Option<PlacementEvent> {
        let raw = self.preprocess(raw, worker)?;
        self.normalize_preprocessed(raw, event_id, worker)
    }

    fn learn_metadata(&mut self, metadata: KvCacheEventMetadata, dp_rank: DpRank) {
        let (Some(group_idx), Some(kind)) = (metadata.group_idx, metadata.kv_cache_spec_kind)
        else {
            return;
        };

        self.group_metadata.insert(
            (dp_rank, group_idx),
            KvCacheGroupMetadata {
                kind,
                sliding_window: metadata.kv_cache_spec_sliding_window,
            },
        );
    }

    fn propagate_cache_namespace(
        &mut self,
        raw: &mut RawKvEvent,
        worker: WorkerWithDpRank,
    ) -> Result<(), ZmqEventFilterReason> {
        match raw {
            RawKvEvent::BlockStored {
                block_hashes,
                parent_block_hash,
                cache_namespace,
                medium,
                ..
            } => {
                let tier = medium
                    .as_deref()
                    .and_then(StorageTier::from_kv_medium)
                    .unwrap_or(StorageTier::Device);
                if cache_namespace.as_deref() == Some("") {
                    *cache_namespace = None;
                }
                let namespace = if let Some(namespace) = cache_namespace.as_deref() {
                    parent_block_hash
                        .as_ref()
                        .and_then(|parent| {
                            self.cache_namespaces.get(&(worker, (*parent).into_u64()))
                        })
                        .and_then(|state| match state {
                            CacheNamespaceState::Namespaced(parent_namespace)
                                if parent_namespace.as_ref() == namespace =>
                            {
                                Some(Arc::clone(parent_namespace))
                            }
                            _ => None,
                        })
                        .or_else(|| Some(Arc::from(namespace)))
                } else if let Some(parent) = parent_block_hash.as_ref() {
                    match self.cache_namespaces.get(&(worker, (*parent).into_u64())) {
                        Some(CacheNamespaceState::Namespaced(namespace)) => {
                            *cache_namespace = Some(namespace.to_string());
                            Some(Arc::clone(namespace))
                        }
                        Some(CacheNamespaceState::Ambiguous) => {
                            return Err(ZmqEventFilterReason::AmbiguousCacheNamespace);
                        }
                        None => {
                            // Deliberately preserve the unsalted interpretation when a
                            // listener joins in the middle of a chain. The vLLM wire
                            // format cannot distinguish an unknown salted parent from a
                            // genuinely unsalted one, and retaining every unsalted block
                            // here would duplicate the index on the event hot path. The
                            // backend still enforces its own cache isolation; this narrow
                            // fail-open case can only pollute the router overlap score.
                            None
                        }
                    }
                } else {
                    None
                };

                if let Some(namespace) = namespace {
                    let state = CacheNamespaceState::Namespaced(namespace);
                    for block_hash in block_hashes.iter() {
                        self.cache_namespaces
                            .entry((worker, (*block_hash).into_u64()))
                            .and_modify(|existing| {
                                if *existing != state {
                                    *existing = CacheNamespaceState::Ambiguous;
                                }
                            })
                            .or_insert_with(|| state.clone());
                    }
                } else {
                    // Do not retain every unsalted block. If this hash already
                    // belongs to a namespace, however, fail closed on future
                    // propagation because the external hash is now ambiguous.
                    for block_hash in block_hashes.iter() {
                        if let Some(existing) = self
                            .cache_namespaces
                            .get_mut(&(worker, (*block_hash).into_u64()))
                        {
                            *existing = CacheNamespaceState::Ambiguous;
                        }
                    }
                }
                for block_hash in block_hashes.iter() {
                    let key = (worker, (*block_hash).into_u64());
                    if self.cache_namespaces.contains_key(&key) {
                        self.cache_namespace_tiers
                            .entry(key)
                            .or_default()
                            .insert(tier);
                    }
                }
            }
            RawKvEvent::BlockRemoved {
                block_hashes,
                medium,
                ..
            } => {
                let tier = medium
                    .as_deref()
                    .and_then(StorageTier::from_kv_medium)
                    .unwrap_or(StorageTier::Device);
                for block_hash in block_hashes.iter() {
                    let key = (worker, (*block_hash).into_u64());
                    if let Some(tiers) = self.cache_namespace_tiers.get_mut(&key) {
                        tiers.remove(&tier);
                        if tiers.is_empty() {
                            self.cache_namespace_tiers.remove(&key);
                        } else {
                            continue;
                        }
                    }
                    if !matches!(
                        self.cache_namespaces.get(&key),
                        Some(CacheNamespaceState::Ambiguous)
                    ) {
                        self.cache_namespaces.remove(&key);
                    }
                }
            }
            RawKvEvent::AllBlocksCleared { medium, .. } => {
                let reset_tier = medium.as_deref().and_then(StorageTier::from_kv_medium);
                if medium.is_none() {
                    self.cache_namespaces
                        .retain(|(known_worker, _), _| *known_worker != worker);
                    self.cache_namespace_tiers
                        .retain(|(known_worker, _), _| *known_worker != worker);
                } else if let Some(reset_tier) = reset_tier {
                    self.clear_namespace_tier(worker, reset_tier);
                }
            }
            RawKvEvent::TierBlocksCleared { medium, .. } => {
                if let Some(reset_tier) = StorageTier::from_kv_medium(medium) {
                    self.clear_namespace_tier(worker, reset_tier);
                }
            }
            RawKvEvent::Ignored => {}
        }
        Ok(())
    }

    fn clear_namespace_tier(&mut self, worker: WorkerWithDpRank, reset_tier: StorageTier) {
        let keys: Vec<_> = self
            .cache_namespace_tiers
            .keys()
            .filter(|(known_worker, _)| *known_worker == worker)
            .copied()
            .collect();
        for key in keys {
            if let Some(tiers) = self.cache_namespace_tiers.get_mut(&key) {
                tiers.remove(&reset_tier);
                if tiers.is_empty() {
                    self.cache_namespace_tiers.remove(&key);
                    self.cache_namespaces.remove(&key);
                }
            }
        }
    }

    fn filter_reason(
        &self,
        metadata: KvCacheEventMetadata,
        dp_rank: DpRank,
    ) -> Option<ZmqEventFilterReason> {
        if let Some(kind) = metadata.kv_cache_spec_kind {
            if kind.is_main_attention() {
                return None;
            }
            if kind == KvCacheSpecKind::Unknown {
                return Some(ZmqEventFilterReason::UnknownKind);
            }
            return Some(ZmqEventFilterReason::NonMainAttentionKind);
        }

        let group_idx = metadata.group_idx?;

        if let Some(metadata) = self.group_metadata.get(&(dp_rank, group_idx)) {
            let _sliding_window = metadata.sliding_window;
            if metadata.kind.is_main_attention() {
                return None;
            }
            if metadata.kind == KvCacheSpecKind::Unknown {
                return Some(ZmqEventFilterReason::UnknownKind);
            }
            return Some(ZmqEventFilterReason::NonMainAttentionGroup);
        }

        if group_idx == 0 {
            None
        } else {
            Some(ZmqEventFilterReason::UnlearnedGroupIdx)
        }
    }
}
