// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Engine-hash key extraction for the hybrid probe index.
//!
//! The normalizer's main path drops every event the radix tree cannot use: non-main-attention
//! groups by kind, and blocks whose size is not the router block. The hybrid probe index needs
//! exactly those events, so this module reads the raw vLLM event *before* that filter and
//! turns it into `(tier, group, op, keys)` tuples that the frontend applies to
//! `indexer::HybridProbeIndex`.
//!
//! Two producer-contract rules live here:
//!
//! * A chain-shaped store (one hash per hash unit: a CPU partial tail, or an anchored
//!   recurrent row) lists the row's content hashes, but only its terminal hash is a lookup key
//!   in the engine. Crediting the interior hashes is the over-prediction of the tail prototype.
//! * Removals are applied to the keys they name. Today's CPU tail removals list every content
//!   hash of the row; the vLLM change that announces removals by key makes this exact.

use crate::protocols::{DpRank, HybridCacheGroup, StorageTier};

use super::ZmqEventNormalizer;
use super::filter::{KvCacheEventMetadata, KvCacheSpecKind};
use super::types::{BlockHashValue, RawKvEvent};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HybridKeyOp {
    Stored,
    Removed,
}

/// One set mutation for the hybrid probe index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HybridKeyEvent {
    pub tier: StorageTier,
    pub group: HybridCacheGroup,
    pub op: HybridKeyOp,
    /// Engine keys as published (low 64 bits of the digest).
    pub hashes: Vec<u64>,
}

/// Which hybrid group a cache-spec kind belongs to. Sliding-window and other kinds are not
/// part of the two-group reuse rule and are ignored.
pub fn hybrid_group_for_kind(kind: KvCacheSpecKind) -> Option<HybridCacheGroup> {
    if kind.is_main_attention() {
        Some(HybridCacheGroup::FullAttention)
    } else if kind == KvCacheSpecKind::Mamba {
        Some(HybridCacheGroup::Recurrent)
    } else {
        None
    }
}

/// Tier an event lands in. A missing medium is the legacy GPU default.
fn hybrid_tier(medium: Option<&str>) -> Option<StorageTier> {
    match medium {
        None => Some(StorageTier::Device),
        Some(medium) => match StorageTier::from_kv_medium(medium) {
            Some(tier @ (StorageTier::Device | StorageTier::HostPinned)) => Some(tier),
            _ => None,
        },
    }
}

fn to_u64(hashes: &[BlockHashValue]) -> Vec<u64> {
    hashes.iter().map(|hash| hash.into_u64()).collect()
}

impl ZmqEventNormalizer {
    /// Enable hybrid key extraction with the engine's `prefix_match_unit` in tokens.
    pub fn with_hybrid_keys(mut self, hash_unit: Option<u32>) -> Self {
        self.hybrid_hash_unit = hash_unit.filter(|unit| *unit > 0);
        self
    }

    pub fn hybrid_keys_enabled(&self) -> bool {
        self.hybrid_hash_unit.is_some()
    }

    /// Derive hybrid key mutations from a raw event.
    ///
    /// Call this before [`Self::preprocess_with_reason`], which consumes the event and drops
    /// the recurrent group. Returns an empty vector when extraction is disabled, the event
    /// carries no hashes, its tier or group is not indexed, or it is a clear (clears travel on
    /// the normal `Cleared` path with their tier scope).
    pub fn hybrid_keys(&mut self, raw: &RawKvEvent, dp_rank: DpRank) -> Vec<HybridKeyEvent> {
        let Some(unit) = self.hybrid_hash_unit else {
            return Vec::new();
        };
        match raw {
            RawKvEvent::BlockStored {
                block_hashes,
                block_size,
                medium,
                ..
            } => {
                let metadata = raw.metadata();
                self.learn_metadata(metadata, dp_rank);
                if block_hashes.is_empty() {
                    return Vec::new();
                }
                let Some(tier) = hybrid_tier(medium.as_deref()) else {
                    return Vec::new();
                };
                let Some(group) = self.hybrid_group(metadata, dp_rank) else {
                    return Vec::new();
                };
                let block_size = u32::try_from(*block_size).unwrap_or(u32::MAX);
                // Chain-shaped rows list one hash per hash unit; only the boundary hash is a
                // key. Host rows of any size other than a full chunk or a placeholder are
                // chains as well.
                let terminal_only = block_size == unit
                    || (tier == StorageTier::HostPinned
                        && block_size != 0
                        && block_size != self.kv_block_size);
                let hashes = if terminal_only {
                    block_hashes
                        .last()
                        .map(|hash| vec![hash.into_u64()])
                        .unwrap_or_default()
                } else {
                    to_u64(block_hashes)
                };
                vec![HybridKeyEvent {
                    tier,
                    group,
                    op: HybridKeyOp::Stored,
                    hashes,
                }]
            }
            RawKvEvent::BlockRemoved {
                block_hashes,
                medium,
                ..
            } => {
                if block_hashes.is_empty() {
                    return Vec::new();
                }
                let Some(tier) = hybrid_tier(medium.as_deref()) else {
                    return Vec::new();
                };
                let hashes = to_u64(block_hashes);
                match self.hybrid_group(raw.metadata(), dp_rank) {
                    Some(group) => vec![HybridKeyEvent {
                        tier,
                        group,
                        op: HybridKeyOp::Removed,
                        hashes,
                    }],
                    // A removal with no group information names the key in every group; a
                    // key only ever lives in the group that stored it, so this is exact.
                    None => [HybridCacheGroup::FullAttention, HybridCacheGroup::Recurrent]
                        .into_iter()
                        .map(|group| HybridKeyEvent {
                            tier,
                            group,
                            op: HybridKeyOp::Removed,
                            hashes: hashes.clone(),
                        })
                        .collect(),
                }
            }
            RawKvEvent::AllBlocksCleared { .. }
            | RawKvEvent::TierBlocksCleared { .. }
            | RawKvEvent::Ignored => Vec::new(),
        }
    }

    fn hybrid_group(
        &self,
        metadata: KvCacheEventMetadata,
        dp_rank: DpRank,
    ) -> Option<HybridCacheGroup> {
        if let Some(kind) = metadata.kv_cache_spec_kind {
            return hybrid_group_for_kind(kind);
        }
        let group_idx = metadata.group_idx?;
        self.group_metadata
            .get(&(dp_rank, group_idx))
            .and_then(|group| hybrid_group_for_kind(group.kind))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BLOCK: usize = 12_288;
    const UNIT: u32 = 128;

    fn stored(
        hashes: &[u64],
        block_size: usize,
        medium: Option<&str>,
        group_idx: Option<u32>,
        kind: Option<KvCacheSpecKind>,
    ) -> RawKvEvent {
        RawKvEvent::BlockStored {
            block_hashes: hashes
                .iter()
                .map(|h| BlockHashValue::Unsigned(*h))
                .collect(),
            parent_block_hash: None,
            token_ids: Vec::new(),
            block_size,
            medium: medium.map(str::to_owned),
            lora_name: None,
            cache_namespace: None,
            block_mm_infos: None,
            is_eagle: None,
            group_idx,
            kv_cache_spec_kind: kind,
            kv_cache_spec_sliding_window: None,
            locality: None,
            ownership: None,
        }
    }

    fn removed(
        hashes: &[u64],
        medium: Option<&str>,
        group_idx: Option<u32>,
        kind: Option<KvCacheSpecKind>,
    ) -> RawKvEvent {
        RawKvEvent::BlockRemoved {
            block_hashes: hashes
                .iter()
                .map(|h| BlockHashValue::Unsigned(*h))
                .collect(),
            medium: medium.map(str::to_owned),
            group_idx,
            kv_cache_spec_kind: kind,
            kv_cache_spec_sliding_window: None,
            locality: None,
            ownership: None,
        }
    }

    fn normalizer() -> ZmqEventNormalizer {
        ZmqEventNormalizer::new(BLOCK as u32).with_hybrid_keys(Some(UNIT))
    }

    fn event(
        tier: StorageTier,
        group: HybridCacheGroup,
        op: HybridKeyOp,
        hashes: &[u64],
    ) -> HybridKeyEvent {
        HybridKeyEvent {
            tier,
            group,
            op,
            hashes: hashes.to_vec(),
        }
    }

    #[test]
    fn disabled_normalizer_extracts_nothing() {
        let mut normalizer = ZmqEventNormalizer::new(BLOCK as u32);
        assert!(!normalizer.hybrid_keys_enabled());
        let raw = stored(
            &[1],
            BLOCK,
            Some("GPU"),
            Some(1),
            Some(KvCacheSpecKind::MlaAttention),
        );
        assert!(normalizer.hybrid_keys(&raw, 0).is_empty());
    }

    #[test]
    fn gpu_full_blocks_and_partial_tail_keep_every_hash() {
        let mut normalizer = normalizer();
        let full = stored(
            &[11, 12],
            BLOCK,
            Some("GPU"),
            Some(1),
            Some(KvCacheSpecKind::MlaAttention),
        );
        assert_eq!(
            normalizer.hybrid_keys(&full, 0),
            vec![event(
                StorageTier::Device,
                HybridCacheGroup::FullAttention,
                HybridKeyOp::Stored,
                &[11, 12]
            )]
        );
        let tail = stored(
            &[13],
            11_904,
            Some("GPU"),
            Some(1),
            Some(KvCacheSpecKind::MlaAttention),
        );
        assert_eq!(normalizer.hybrid_keys(&tail, 0)[0].hashes, vec![13]);
    }

    #[test]
    fn recurrent_kind_maps_to_the_recurrent_group() {
        let mut normalizer = normalizer();
        let kda = stored(
            &[21, 22],
            1_536,
            Some("GPU"),
            Some(0),
            Some(KvCacheSpecKind::Mamba),
        );
        assert_eq!(
            normalizer.hybrid_keys(&kda, 0),
            vec![event(
                StorageTier::Device,
                HybridCacheGroup::Recurrent,
                HybridKeyOp::Stored,
                &[21, 22]
            )]
        );
        // Empty hash lists (retention-thinned KDA blocks) contribute nothing.
        let empty = stored(
            &[],
            1_536,
            Some("GPU"),
            Some(0),
            Some(KvCacheSpecKind::Mamba),
        );
        assert!(normalizer.hybrid_keys(&empty, 0).is_empty());
    }

    #[test]
    fn chain_shaped_rows_credit_only_the_terminal_hash() {
        let mut normalizer = normalizer();
        // CPU partial tail: block_size == hash unit, one hash per unit from the chunk start.
        let tail = stored(
            &[101, 102, 103],
            UNIT as usize,
            Some("CPU"),
            Some(1),
            Some(KvCacheSpecKind::MlaAttention),
        );
        assert_eq!(
            normalizer.hybrid_keys(&tail, 0),
            vec![event(
                StorageTier::HostPinned,
                HybridCacheGroup::FullAttention,
                HybridKeyOp::Stored,
                &[103]
            )]
        );
        // The same shape on the GPU (anchored recurrent keys) is a chain too.
        let gpu_chain = stored(
            &[201, 202],
            UNIT as usize,
            Some("GPU"),
            Some(0),
            Some(KvCacheSpecKind::Mamba),
        );
        assert_eq!(normalizer.hybrid_keys(&gpu_chain, 0)[0].hashes, vec![202]);
        // A host row of any other non-chunk, non-placeholder size is chain-shaped as well.
        let odd = stored(
            &[301, 302],
            1_536,
            Some("CPU"),
            Some(0),
            Some(KvCacheSpecKind::Mamba),
        );
        assert_eq!(normalizer.hybrid_keys(&odd, 0)[0].hashes, vec![302]);
    }

    #[test]
    fn host_placeholders_and_chunks_keep_their_key() {
        let mut normalizer = normalizer();
        let placeholder = stored(
            &[401],
            0,
            Some("CPU"),
            Some(0),
            Some(KvCacheSpecKind::Mamba),
        );
        assert_eq!(
            normalizer.hybrid_keys(&placeholder, 0),
            vec![event(
                StorageTier::HostPinned,
                HybridCacheGroup::Recurrent,
                HybridKeyOp::Stored,
                &[401]
            )]
        );
        let chunk = stored(
            &[402],
            BLOCK,
            Some("CPU"),
            Some(1),
            Some(KvCacheSpecKind::MlaAttention),
        );
        assert_eq!(normalizer.hybrid_keys(&chunk, 0)[0].hashes, vec![402]);
    }

    #[test]
    fn group_is_learned_from_stores_for_events_without_a_kind() {
        let mut normalizer = normalizer();
        let learn = stored(
            &[1],
            1_536,
            Some("GPU"),
            Some(0),
            Some(KvCacheSpecKind::Mamba),
        );
        normalizer.hybrid_keys(&learn, 3);
        let later = stored(&[2], 1_536, Some("GPU"), Some(0), None);
        assert_eq!(
            normalizer.hybrid_keys(&later, 3)[0].group,
            HybridCacheGroup::Recurrent
        );
        // Another rank has not announced group 0 yet: unknown group, nothing extracted.
        assert!(normalizer.hybrid_keys(&later, 4).is_empty());
        let removal = removed(&[2], Some("GPU"), Some(0), None);
        assert_eq!(
            normalizer.hybrid_keys(&removal, 3),
            vec![event(
                StorageTier::Device,
                HybridCacheGroup::Recurrent,
                HybridKeyOp::Removed,
                &[2]
            )]
        );
    }

    #[test]
    fn removal_without_group_information_names_both_groups() {
        let mut normalizer = normalizer();
        let removal = removed(&[5, 6], Some("CPU"), None, None);
        let events = normalizer.hybrid_keys(&removal, 0);
        assert_eq!(events.len(), 2);
        assert!(events.iter().all(|e| e.tier == StorageTier::HostPinned
            && e.op == HybridKeyOp::Removed
            && e.hashes == vec![5, 6]));
        assert_ne!(events[0].group, events[1].group);
    }

    #[test]
    fn unsupported_tiers_kinds_and_clears_are_ignored() {
        let mut normalizer = normalizer();
        let disk = stored(
            &[1],
            BLOCK,
            Some("DISK"),
            Some(1),
            Some(KvCacheSpecKind::MlaAttention),
        );
        assert!(normalizer.hybrid_keys(&disk, 0).is_empty());
        let unknown_medium = stored(
            &[1],
            BLOCK,
            Some("FS"),
            Some(1),
            Some(KvCacheSpecKind::MlaAttention),
        );
        assert!(normalizer.hybrid_keys(&unknown_medium, 0).is_empty());
        let sliding = stored(
            &[1],
            BLOCK,
            Some("GPU"),
            Some(2),
            Some(KvCacheSpecKind::SlidingWindow),
        );
        assert!(normalizer.hybrid_keys(&sliding, 0).is_empty());
        assert!(
            normalizer
                .hybrid_keys(&RawKvEvent::AllBlocksCleared { ownership: None }, 0)
                .is_empty()
        );
        assert!(
            normalizer
                .hybrid_keys(
                    &RawKvEvent::TierBlocksCleared {
                        medium: "GPU".to_string(),
                        ownership: None
                    },
                    0
                )
                .is_empty()
        );
        // Legacy events without a medium are GPU events.
        let legacy = stored(
            &[9],
            BLOCK,
            None,
            Some(1),
            Some(KvCacheSpecKind::FullAttention),
        );
        assert_eq!(
            normalizer.hybrid_keys(&legacy, 0)[0].tier,
            StorageTier::Device
        );
    }
}
