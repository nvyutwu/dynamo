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

use crate::protocols::{DpRank, PlacementEvent, WorkerWithDpRank};

mod convert;
mod deserialize;
mod extra_keys;
mod filter;
#[cfg(test)]
mod tests;
mod types;

pub use convert::{
    StoredBlockOptions, convert_event, create_stored_block_from_parts, create_stored_blocks,
};
pub use extra_keys::{
    extra_keys_to_block_mm_infos, extra_keys_to_cache_namespace, parse_mm_hash_from_extra_key,
};
pub use filter::KvCacheSpecKind;
pub use types::{BlockHashValue, ExtraKeyItem, KvEventBatch, KvTokenIds, RawKvEvent};

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
    /// Finest token granularity the engine hashes at, i.e. vLLM's
    /// `hash_block_size` / `--prefix-match-unit`. The host tier publishes at
    /// exactly this size and device partial blocks are multiples of it, so it
    /// is what makes those events indexable rather than discarded.
    hash_block_size: u32,
    /// The wire schema does not advertise `mamba_cache_mode`; preserve the
    /// historical fail-closed policy unless a known-align deployment opts in.
    mamba_align_indexing_enabled: bool,
    /// Host offload events can be emitted at a finer granularity than the
    /// request router's device cache blocks. This guarded compatibility mode
    /// coalesces complete lower-tier groups back to the device granularity so
    /// their token hashes are queryable by the router.
    lower_tier_aggregate_to_device: bool,
    warning_count: Arc<AtomicU32>,
    group_metadata: FxHashMap<(DpRank, u32), KvCacheGroupMetadata>,
    cache_namespaces: FxHashMap<(WorkerWithDpRank, u64), CacheNamespaceState>,
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
            // Default preserves pre-fix behaviour: only full blocks pass.
            // Callers that know the engine's hash granularity should override
            // via with_hash_block_size so host-tier and partial events index.
            hash_block_size: kv_block_size,
            mamba_align_indexing_enabled: false,
            lower_tier_aggregate_to_device: false,
            image_token_id: None,
            warning_count: Arc::new(AtomicU32::new(0)),
            group_metadata: FxHashMap::default(),
            cache_namespaces: FxHashMap::default(),
        }
    }

    pub fn with_warning_count(kv_block_size: u32, warning_count: Arc<AtomicU32>) -> Self {
        Self {
            kv_block_size,
            hash_block_size: kv_block_size,
            mamba_align_indexing_enabled: false,
            lower_tier_aggregate_to_device: false,
            image_token_id: None,
            warning_count,
            group_metadata: FxHashMap::default(),
            cache_namespaces: FxHashMap::default(),
        }
    }

    /// Read the engine's hash granularity from `DYN_KV_HASH_BLOCK_SIZE`.
    ///
    /// vLLM does not yet advertise `hash_block_size` alongside its KV-event
    /// block size, so this env var is the seam until it does. Unset or invalid
    /// leaves full-blocks-only behaviour, which is what shipped before.
    ///
    /// Set it to the engine's `--prefix-match-unit` (equivalently the
    /// `hash_block_size` resolved by `resolve_kv_cache_block_sizes`). Without
    /// it every host-tier event and every device partial block is discarded,
    /// the HostPinned index stays empty, and no router hint is ever produced.
    pub fn with_hash_block_size_from_env(self) -> Self {
        match std::env::var("DYN_KV_HASH_BLOCK_SIZE") {
            Ok(raw) => match raw.trim().parse::<u32>() {
                Ok(value) if value > 0 => {
                    tracing::info!(
                        hash_block_size = value,
                        kv_block_size = self.kv_block_size,
                        "KV hash block size set from DYN_KV_HASH_BLOCK_SIZE; \
                         host-tier and partial blocks will be indexed"
                    );
                    self.with_hash_block_size(value)
                }
                _ => {
                    tracing::warn!(
                        value = %raw,
                        "DYN_KV_HASH_BLOCK_SIZE is not a positive integer; \
                         falling back to full-blocks-only indexing"
                    );
                    self
                }
            },
            Err(_) => self,
        }
    }

    /// Set the engine's finest hash granularity. Without this the normalizer
    /// only accepts full cache blocks, which silently discards every host-tier
    /// event and every device partial block. Feed it the value the worker
    /// advertises alongside its KV-event block size.
    pub fn with_hash_block_size(mut self, hash_block_size: u32) -> Self {
        if hash_block_size > 0 {
            self.hash_block_size = hash_block_size;
        }
        self
    }

    /// Admit Mamba events only when the deployment explicitly asserts vLLM's
    /// reusable `--mamba-cache-mode align` representation.
    pub fn with_mamba_align_indexing(mut self, enabled: bool) -> Self {
        self.mamba_align_indexing_enabled = enabled;
        self
    }

    /// Read the Mamba-align admission gate from the environment. The default
    /// is fail-closed because this property is not represented on the wire.
    pub fn with_mamba_align_indexing_from_env(self) -> Self {
        match std::env::var("DYN_KV_INDEX_MAMBA_ALIGN") {
            Ok(raw) if matches!(raw.trim(), "1" | "true" | "TRUE" | "yes" | "YES") => {
                tracing::warn!(
                    "Mamba KV events admitted to the router index by \
                     DYN_KV_INDEX_MAMBA_ALIGN; this requires \
                     --mamba-cache-mode align"
                );
                self.with_mamba_align_indexing(true)
            }
            Ok(raw) if matches!(raw.trim(), "0" | "false" | "FALSE" | "no" | "NO") => self,
            Ok(raw) => {
                tracing::warn!(
                    value = %raw,
                    "DYN_KV_INDEX_MAMBA_ALIGN is not a boolean; leaving Mamba \
                     KV events out of the router index"
                );
                self
            }
            Err(_) => self,
        }
    }

    /// Opt into coalescing lower-tier hash-sized entries into device-sized
    /// routing blocks. This is needed only when the engine emits a complete
    /// host chunk at a finer granularity than the router's device cache block.
    /// It defaults off because partial lower-tier residency must not be rounded
    /// up to a whole device block.
    pub fn with_lower_tier_aggregation(mut self, enabled: bool) -> Self {
        self.lower_tier_aggregate_to_device = enabled;
        self
    }

    /// Read the lower-tier aggregation compatibility switch from the
    /// environment. Only an explicit true value enables it.
    pub fn with_lower_tier_aggregation_from_env(self) -> Self {
        match std::env::var("DYN_KV_LOWER_TIER_AGGREGATE_TO_DEVICE") {
            Ok(raw) if matches!(raw.trim(), "1" | "true" | "TRUE" | "yes" | "YES") => {
                tracing::warn!(
                    "Lower-tier KV events will be aggregated to device cache-block granularity"
                );
                self.with_lower_tier_aggregation(true)
            }
            Ok(raw) if matches!(raw.trim(), "0" | "false" | "FALSE" | "no" | "NO") => self,
            Ok(raw) => {
                tracing::warn!(
                    value = %raw,
                    "DYN_KV_LOWER_TIER_AGGREGATE_TO_DEVICE is not a boolean; leaving lower-tier aggregation disabled"
                );
                self
            }
            Err(_) => self,
        }
    }

    /// Set the model's image placeholder token id so vLLM BlockStored events
    /// get normalized to the canonical pad_value scheme. No-op for text-only
    /// models (leave unset).
    pub fn with_image_token_id(mut self, image_token_id: Option<u32>) -> Self {
        self.image_token_id = image_token_id;
        self
    }

    pub fn preprocess(&mut self, raw: RawKvEvent, worker: WorkerWithDpRank) -> Option<RawKvEvent> {
        self.preprocess_with_reason(raw, worker).ok()
    }

    pub fn preprocess_with_reason(
        &mut self,
        mut raw: RawKvEvent,
        worker: WorkerWithDpRank,
    ) -> Result<RawKvEvent, ZmqEventFilterReason> {
        if raw.is_ignored() {
            return Err(ZmqEventFilterReason::IgnoredEvent);
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
        convert::convert_event_with_lower_tier_aggregation(
            raw,
            event_id,
            self.kv_block_size,
            self.hash_block_size,
            worker,
            &self.warning_count,
            self.image_token_id,
            self.lower_tier_aggregate_to_device,
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
                ..
            } => {
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
            }
            RawKvEvent::BlockRemoved { block_hashes, .. } => {
                for block_hash in block_hashes.iter() {
                    let key = (worker, (*block_hash).into_u64());
                    if !matches!(
                        self.cache_namespaces.get(&key),
                        Some(CacheNamespaceState::Ambiguous)
                    ) {
                        self.cache_namespaces.remove(&key);
                    }
                }
            }
            RawKvEvent::AllBlocksCleared => {
                self.cache_namespaces
                    .retain(|(known_worker, _), _| *known_worker != worker);
            }
            RawKvEvent::Ignored => {}
        }
        Ok(())
    }

    fn filter_reason(
        &self,
        metadata: KvCacheEventMetadata,
        dp_rank: DpRank,
    ) -> Option<ZmqEventFilterReason> {
        if let Some(kind) = metadata.kv_cache_spec_kind {
            if kind.is_indexable(self.mamba_align_indexing_enabled) {
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
            if metadata.kind.is_indexable(self.mamba_align_indexing_enabled) {
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
