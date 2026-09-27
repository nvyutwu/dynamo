// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Engine-hash probe index for hybrid-attention engines ("D3").
//!
//! A hybrid model such as Kimi-K3 keeps two KV-cache groups with different reuse rules: the
//! full-attention (MLA) group caches dense blocks, while the recurrent (KDA) group keeps a
//! resume point only where a prefill step ended. The engine reuses a prefix only up to the
//! deepest boundary where *both* groups hold a key, so a radix tree over the full-attention
//! group alone credits blocks the engine cannot use and misses the partial-tail boundaries
//! where the recurrent state actually lives.
//!
//! This index keeps no tree. For every `(worker, tier, group)` it stores the set of engine
//! prefix-chain hashes the worker has announced (one hash per cached boundary). A lookup
//! computes the request's own engine chain hashes (see [`super::EngineChainHasher`]) and probes
//! the sets exactly the way the engine's `find_longest_cache_hit` and the offload connector's
//! `_lookup` walk their boundaries: full-attention coverage first, then the deepest recurrent
//! key at or below it, walking leaf to root because a hash hit at boundary `b` certifies the
//! whole prefix `[0, b)`.
//!
//! Contract with the event producer: a chain-shaped event (one hash per hash unit, such as a
//! CPU partial tail) contributes only its terminal hash, and removals must name the same keys
//! that were stored. Both are enforced in `zmq_wire::hybrid`.
//!
//! Keys are reference counted per `(worker, tier, group)`: vLLM publishes one `BlockStored`
//! per physical copy of a block and one `BlockRemoved` per evicted copy, and two requests of
//! one session scheduled in the same step can hold two copies of the same content. A key
//! stays resident until its last copy is removed, the same rule the worker publisher's
//! `RefCounted` dedup applies to block events.

use parking_lot::RwLock;
use rustc_hash::FxHashMap;

use crate::protocols::{DpRank, HybridCacheGroup, StorageTier, WorkerId, WorkerWithDpRank};

/// Reusable tokens for one request on one worker, split the way the engine reports them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HybridHit {
    /// Tokens served from the worker's device (HBM) cache.
    pub device_tokens: usize,
    /// Tokens restored from the worker's host (CPU offload) tier on top of the device hit.
    pub host_tokens: usize,
}

impl HybridHit {
    pub fn total_tokens(self) -> usize {
        self.device_tokens + self.host_tokens
    }
}

/// Size snapshot of the index.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HybridIndexStats {
    pub workers: usize,
    pub keys: usize,
}

#[derive(Debug, Default)]
struct WorkerKeys {
    /// `[device full, device recurrent, host full, host recurrent]`, see [`slot`]; the value
    /// is the number of resident physical copies announced for the key.
    sets: [FxHashMap<u64, u32>; 4],
}

impl WorkerKeys {
    fn is_empty(&self) -> bool {
        self.sets.iter().all(FxHashMap::is_empty)
    }

    fn key_count(&self) -> usize {
        self.sets.iter().map(FxHashMap::len).sum()
    }

    fn clear_tier(&mut self, tier: StorageTier) {
        for group in [HybridCacheGroup::FullAttention, HybridCacheGroup::Recurrent] {
            if let Some(index) = slot(tier, group) {
                self.sets[index].clear();
            }
        }
    }
}

fn slot(tier: StorageTier, group: HybridCacheGroup) -> Option<usize> {
    let tier_offset = match tier {
        StorageTier::Device => 0,
        StorageTier::HostPinned => 2,
        StorageTier::Disk | StorageTier::External => return None,
    };
    let group_offset = match group {
        HybridCacheGroup::FullAttention => 0,
        HybridCacheGroup::Recurrent => 1,
    };
    Some(tier_offset + group_offset)
}

/// Per-worker sets of announced engine hashes with the engine's reuse rule as the lookup.
pub struct HybridProbeIndex {
    block_size: u32,
    hash_unit: u32,
    workers: RwLock<FxHashMap<WorkerWithDpRank, WorkerKeys>>,
}

impl std::fmt::Debug for HybridProbeIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stats = self.stats();
        f.debug_struct("HybridProbeIndex")
            .field("block_size", &self.block_size)
            .field("hash_unit", &self.hash_unit)
            .field("workers", &stats.workers)
            .field("keys", &stats.keys)
            .finish()
    }
}

impl HybridProbeIndex {
    /// `block_size` is the full-attention group's block in tokens (the router block size,
    /// 12,288 for Kimi-K3 under DCP 8); `hash_unit` is the engine's `prefix_match_unit`.
    pub fn new(block_size: u32, hash_unit: u32) -> Self {
        assert!(hash_unit > 0, "hash_unit must be positive");
        assert!(
            block_size > 0 && block_size.is_multiple_of(hash_unit),
            "block_size {block_size} must be a positive multiple of hash_unit {hash_unit}"
        );
        Self {
            block_size,
            hash_unit,
            workers: RwLock::new(FxHashMap::default()),
        }
    }

    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    pub fn hash_unit(&self) -> u32 {
        self.hash_unit
    }

    /// Record announced keys, one reference per hash occurrence. Returns `false` when `tier`
    /// is not indexed (disk, external).
    pub fn store(
        &self,
        worker: WorkerWithDpRank,
        tier: StorageTier,
        group: HybridCacheGroup,
        hashes: &[u64],
    ) -> bool {
        let Some(index) = slot(tier, group) else {
            return false;
        };
        if hashes.is_empty() {
            return true;
        }
        let mut workers = self.workers.write();
        let keys = workers.entry(worker).or_default();
        for hash in hashes {
            let copies = keys.sets[index].entry(*hash).or_insert(0);
            *copies = copies.saturating_add(1);
        }
        true
    }

    /// Release one reference per hash; a key leaves the index when its last copy is removed.
    /// Removals of unknown keys are ignored. Returns `false` when `tier` is not indexed.
    pub fn remove(
        &self,
        worker: WorkerWithDpRank,
        tier: StorageTier,
        group: HybridCacheGroup,
        hashes: &[u64],
    ) -> bool {
        let Some(index) = slot(tier, group) else {
            return false;
        };
        let mut workers = self.workers.write();
        if let Some(keys) = workers.get_mut(&worker) {
            for hash in hashes {
                if let Some(copies) = keys.sets[index].get_mut(hash) {
                    *copies -= 1;
                    if *copies == 0 {
                        keys.sets[index].remove(hash);
                    }
                }
            }
            if keys.is_empty() {
                workers.remove(&worker);
            }
        }
        true
    }

    /// Drop one tier of a worker rank (`Some(tier)`) or every tier (`None`).
    pub fn clear(&self, worker: WorkerWithDpRank, tier: Option<StorageTier>) {
        let mut workers = self.workers.write();
        match tier {
            None => {
                workers.remove(&worker);
            }
            Some(tier) => {
                if let Some(keys) = workers.get_mut(&worker) {
                    keys.clear_tier(tier);
                    if keys.is_empty() {
                        workers.remove(&worker);
                    }
                }
            }
        }
    }

    /// Drop every rank of a worker.
    pub fn remove_worker(&self, worker_id: WorkerId) {
        self.workers
            .write()
            .retain(|worker, _| worker.worker_id != worker_id);
    }

    pub fn remove_worker_dp_rank(&self, worker_id: WorkerId, dp_rank: DpRank) {
        self.clear(WorkerWithDpRank::new(worker_id, dp_rank), None);
    }

    pub fn stats(&self) -> HybridIndexStats {
        let workers = self.workers.read();
        HybridIndexStats {
            workers: workers.len(),
            keys: workers.values().map(WorkerKeys::key_count).sum(),
        }
    }

    /// Announced keys of one worker for one tier and group, sorted (diagnostics and tests).
    pub fn keys(
        &self,
        worker: WorkerWithDpRank,
        tier: StorageTier,
        group: HybridCacheGroup,
    ) -> Vec<u64> {
        let Some(index) = slot(tier, group) else {
            return Vec::new();
        };
        let workers = self.workers.read();
        let mut keys: Vec<u64> = workers
            .get(&worker)
            .map(|keys| keys.sets[index].keys().copied().collect())
            .unwrap_or_default();
        keys.sort_unstable();
        keys
    }

    /// Predict what every known worker would reuse for a request whose engine chain hashes
    /// are `chain` (element `i` covers the prefix `[0, (i + 1) * hash_unit)`) and whose
    /// prompt has `num_tokens` tokens. Workers with no reusable tokens are omitted.
    pub fn lookup(
        &self,
        chain: &[u64],
        num_tokens: usize,
    ) -> FxHashMap<WorkerWithDpRank, HybridHit> {
        let unit = self.hash_unit as usize;
        let block = self.block_size as usize;
        let workers = self.workers.read();
        let mut hits = FxHashMap::default();
        for (worker, keys) in workers.iter() {
            let probe = |tier: StorageTier, group: HybridCacheGroup, boundary: usize| -> bool {
                let Some(index) = slot(tier, group) else {
                    return false;
                };
                if boundary == 0 || !boundary.is_multiple_of(unit) {
                    return false;
                }
                chain
                    .get(boundary / unit - 1)
                    .is_some_and(|hash| keys.sets[index].contains_key(hash))
            };
            let hit = reconcile(&probe, num_tokens, unit, block);
            if hit.total_tokens() > 0 {
                hits.insert(*worker, hit);
            }
        }
        hits
    }
}

/// The engine's reuse rule over key presence, probed leaf to root.
///
/// `probe(tier, group, boundary)` answers whether the worker holds the key for the request
/// prefix `[0, boundary)` in that tier and group. Mirrors, in order:
///
/// 1. `HybridKVCacheCoordinator.find_longest_cache_hit`: full-attention coverage `L` is the
///    deepest device full-attention key (whole blocks, then the interior hash-unit boundaries
///    of the next block from the top down); the device hit is the deepest device recurrent key
///    at or below `L`.
/// 2. `OffloadingConnectorScheduler._lookup`: from the block-aligned device hit, the complete
///    host hit is the deepest full-attention chunk of the maximal prefix that has a recurrent
///    state at its boundary (the recurrent groups are one-chunk sliding-window lookups); the
///    partial-tail probe anchors on the host full-attention prefix and scans the next block's
///    boundaries from the top down for a boundary both groups hold.
/// 3. `Scheduler.schedule`: a host hit replaces the device partial tail only when it is longer.
pub fn reconcile(
    probe: &dyn Fn(StorageTier, HybridCacheGroup, usize) -> bool,
    num_tokens: usize,
    hash_unit: usize,
    block_size: usize,
) -> HybridHit {
    use HybridCacheGroup::{FullAttention, Recurrent};
    use StorageTier::{Device, HostPinned};

    let (u, b) = (hash_unit, block_size);
    if num_tokens == 0 || u == 0 || b == 0 || b % u != 0 {
        return HybridHit::default();
    }
    // The last token is always computed, so the deepest usable boundary excludes it.
    let max_len = ((num_tokens - 1) / u) * u;

    // Device: full-attention coverage, whole blocks first.
    let mut coverage = 0usize;
    while coverage + b <= max_len && probe(Device, FullAttention, coverage + b) {
        coverage += b;
    }
    // Then the producer's partial-tail key inside the next block, top down.
    let mut boundary = (coverage + b - u).min(max_len);
    while boundary > coverage {
        if probe(Device, FullAttention, boundary) {
            coverage = boundary;
            break;
        }
        boundary -= u;
    }
    // Device hit: deepest recurrent key at or below the coverage, top down.
    let mut local = 0usize;
    let mut boundary = coverage;
    while boundary > 0 {
        if probe(Device, Recurrent, boundary) {
            local = boundary;
            break;
        }
        boundary -= u;
    }
    let partial_tail = local % b;
    let aligned_local = local - partial_tail;

    // Host complete chunks beyond the aligned device hit (`_lookup_complete_chunks`): the
    // full-attention group is a maximal-prefix lookup; each recurrent group is a
    // sliding-window lookup of width one chunk, so the hit ends at the deepest full-attention
    // chunk that has a recurrent state at its boundary. Interior boundaries need no state:
    // K3 stores KDA states only where a prefill step ends, and a divergent request that ended
    // a step at 24,576 made chunks 1-2 restorable although no state ever existed at 12,288
    // (lyrix run 2, E01). The last prompt token is always computed, so a prompt that ends
    // exactly on a chunk boundary cannot reuse its last chunk.
    let num_chunks = (num_tokens - 1) / b;
    let first_chunk = aligned_local / b;
    let mut anchor = first_chunk;
    while anchor < num_chunks && probe(HostPinned, FullAttention, (anchor + 1) * b) {
        anchor += 1;
    }
    let mut complete = first_chunk;
    for chunk in (first_chunk + 1..=anchor).rev() {
        if probe(HostPinned, Recurrent, chunk * b) {
            complete = chunk;
            break;
        }
    }
    let complete_hit = complete * b - aligned_local;
    // The partial-tail probe anchors on the host full-attention prefix alone
    // (`_full_attention_complete_hit`), which may reach one chunk further when the prompt
    // ends exactly on a chunk boundary.
    let mut anchor_fa = anchor;
    let anchor_chunks = num_tokens / b;
    while anchor_fa < anchor_chunks && probe(HostPinned, FullAttention, (anchor_fa + 1) * b) {
        anchor_fa += 1;
    }
    let anchor_hit = complete_hit.max(anchor_fa * b - aligned_local);
    let boundary0 = aligned_local + anchor_hit;
    let max_boundary = ((num_tokens - 1).min(boundary0 + b - 1) / u) * u;
    let mut ext = complete_hit;
    let mut boundary = max_boundary;
    while boundary > boundary0 {
        if probe(HostPinned, FullAttention, boundary) && probe(HostPinned, Recurrent, boundary) {
            ext = boundary - aligned_local;
            break;
        }
        boundary -= u;
    }

    // Scheduler merge: a host hit replaces the device partial tail only when strictly longer.
    if partial_tail > 0 && ext > partial_tail {
        HybridHit {
            device_tokens: aligned_local,
            host_tokens: ext,
        }
    } else if partial_tail > 0 {
        HybridHit {
            device_tokens: local,
            host_tokens: 0,
        }
    } else {
        HybridHit {
            device_tokens: local,
            host_tokens: ext,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const UNIT: usize = 128;
    const BLOCK: usize = 12_288;

    /// A synthetic engine chain: boundaries up to `shared` reproduce the stored prompt's
    /// hashes, later boundaries belong to a different prefix.
    fn chain(num_tokens: usize, shared: usize) -> Vec<u64> {
        (1..=num_tokens / UNIT)
            .map(|i| {
                let boundary = i * UNIT;
                let salt = if boundary <= shared { 1 } else { 0x5eed_0000 };
                boundary as u64 * 0x9e37 + salt
            })
            .collect()
    }

    fn key(boundary: usize) -> u64 {
        boundary as u64 * 0x9e37 + 1
    }

    fn worker() -> WorkerWithDpRank {
        WorkerWithDpRank::new(7, 0)
    }

    /// Worker X after storing a 24,300-token prompt under production retention: MLA keys at
    /// the full block and the partial tail, KDA keys at the last step end and the tail; the
    /// CPU tier holds the chunk, the step-end row and the tail row.
    fn stored_index() -> HybridProbeIndex {
        let index = HybridProbeIndex::new(BLOCK as u32, UNIT as u32);
        let w = worker();
        use HybridCacheGroup::{FullAttention, Recurrent};
        use StorageTier::{Device, HostPinned};
        index.store(w, Device, FullAttention, &[key(12_288), key(24_192)]);
        index.store(w, Device, Recurrent, &[key(23_040), key(24_192)]);
        index.store(
            w,
            HostPinned,
            FullAttention,
            &[key(12_288), key(23_040), key(24_192)],
        );
        index.store(w, HostPinned, Recurrent, &[key(23_040), key(24_192)]);
        index
    }

    fn hit(index: &HybridProbeIndex, num_tokens: usize, shared: usize) -> HybridHit {
        index
            .lookup(&chain(num_tokens, shared), num_tokens)
            .get(&worker())
            .copied()
            .unwrap_or_default()
    }

    #[test]
    fn replay_reuses_up_to_the_tail_key() {
        let index = stored_index();
        assert_eq!(
            hit(&index, 24_300, 24_300),
            HybridHit {
                device_tokens: 24_192,
                host_tokens: 0
            }
        );
    }

    #[test]
    fn extension_reuses_the_stored_tail() {
        let index = stored_index();
        assert_eq!(
            hit(&index, 24_600, 24_300),
            HybridHit {
                device_tokens: 24_192,
                host_tokens: 0
            }
        );
    }

    #[test]
    fn divergence_before_the_recurrent_keys_reuses_nothing() {
        let index = stored_index();
        // MLA coverage stays at 12,288 but no KDA key exists at or below it, and the host
        // rows beyond 12,288 belong to the old prefix.
        assert_eq!(hit(&index, 24_300, 20_000), HybridHit::default());
        assert!(index.lookup(&chain(24_300, 20_000), 24_300).is_empty());
    }

    #[test]
    fn evicted_device_tail_falls_back_to_the_host_rows() {
        let index = stored_index();
        let w = worker();
        index.remove(
            w,
            StorageTier::Device,
            HybridCacheGroup::FullAttention,
            &[key(24_192)],
        );
        index.remove(
            w,
            StorageTier::Device,
            HybridCacheGroup::Recurrent,
            &[key(24_192)],
        );
        // Device: MLA coverage 12,288, no KDA key at or below it -> 0 device tokens; the CPU
        // rows still hold both groups at 24,192, anchored on the 12,288 chunk.
        assert_eq!(
            hit(&index, 24_300, 24_300),
            HybridHit {
                device_tokens: 0,
                host_tokens: 24_192
            }
        );
        index.clear(w, Some(StorageTier::HostPinned));
        assert_eq!(hit(&index, 24_300, 24_300), HybridHit::default());
    }

    #[test]
    fn host_hit_replaces_a_shorter_device_tail_only() {
        let index = HybridProbeIndex::new(BLOCK as u32, UNIT as u32);
        let w = worker();
        use HybridCacheGroup::{FullAttention, Recurrent};
        use StorageTier::{Device, HostPinned};
        // Device holds a short partial tail (12,288 + 1,536); host holds a longer one.
        index.store(w, Device, FullAttention, &[key(12_288), key(13_824)]);
        index.store(w, Device, Recurrent, &[key(13_824)]);
        index.store(w, HostPinned, FullAttention, &[key(12_288), key(23_040)]);
        index.store(w, HostPinned, Recurrent, &[key(23_040)]);
        assert_eq!(
            hit(&index, 24_300, 24_300),
            HybridHit {
                device_tokens: 12_288,
                host_tokens: 10_752
            }
        );
        // An equal-length host tail does not replace the device tail.
        index.remove(w, HostPinned, FullAttention, &[key(23_040)]);
        index.remove(w, HostPinned, Recurrent, &[key(23_040)]);
        index.store(w, HostPinned, FullAttention, &[key(13_824)]);
        index.store(w, HostPinned, Recurrent, &[key(13_824)]);
        assert_eq!(
            hit(&index, 24_300, 24_300),
            HybridHit {
                device_tokens: 13_824,
                host_tokens: 0
            }
        );
    }

    /// K3 lyrix run 2, E01: chunks 1-2 of a 28,200-token fill sat on the CPU as full-attention
    /// rows and the only recurrent state on the CPU was at 24,576 (a divergent request's
    /// prefill step ended there); the engine restored 24,576. The recurrent lookup is a
    /// one-chunk sliding window, so no state is needed at 12,288.
    #[test]
    fn recurrent_state_at_the_last_chunk_alone_completes_the_hit() {
        use HybridCacheGroup::{FullAttention, Recurrent};
        use StorageTier::HostPinned;
        let w = worker();
        let index = HybridProbeIndex::new(BLOCK as u32, UNIT as u32);
        index.store(w, HostPinned, FullAttention, &[key(12_288), key(24_576)]);
        index.store(w, HostPinned, Recurrent, &[key(24_576)]);
        assert_eq!(
            hit(&index, 29_762, 29_762),
            HybridHit {
                device_tokens: 0,
                host_tokens: 24_576
            }
        );
        // A state at chunk 1 only stops the hit there, even with two full-attention chunks.
        let index = HybridProbeIndex::new(BLOCK as u32, UNIT as u32);
        index.store(w, HostPinned, FullAttention, &[key(12_288), key(24_576)]);
        index.store(w, HostPinned, Recurrent, &[key(12_288)]);
        assert_eq!(hit(&index, 29_762, 29_762).host_tokens, 12_288);
        // A prompt that ends exactly on a chunk boundary cannot reuse its last chunk.
        let index = HybridProbeIndex::new(BLOCK as u32, UNIT as u32);
        index.store(w, HostPinned, FullAttention, &[key(12_288), key(24_576)]);
        index.store(w, HostPinned, Recurrent, &[key(12_288), key(24_576)]);
        assert_eq!(hit(&index, 24_576, 24_576).host_tokens, 12_288);
    }

    #[test]
    fn full_attention_alone_is_never_credited() {
        let index = HybridProbeIndex::new(BLOCK as u32, UNIT as u32);
        let w = worker();
        index.store(
            w,
            StorageTier::Device,
            HybridCacheGroup::FullAttention,
            &[key(12_288), key(24_576)],
        );
        assert_eq!(hit(&index, 30_000, 30_000), HybridHit::default());
        // A recurrent key exactly on the block boundary unlocks the whole block.
        index.store(
            w,
            StorageTier::Device,
            HybridCacheGroup::Recurrent,
            &[key(12_288)],
        );
        assert_eq!(
            hit(&index, 30_000, 30_000),
            HybridHit {
                device_tokens: 12_288,
                host_tokens: 0
            }
        );
    }

    #[test]
    fn short_prompts_and_last_token_rule() {
        let index = HybridProbeIndex::new(BLOCK as u32, UNIT as u32);
        let w = worker();
        index.store(
            w,
            StorageTier::Device,
            HybridCacheGroup::FullAttention,
            &[key(256)],
        );
        index.store(
            w,
            StorageTier::Device,
            HybridCacheGroup::Recurrent,
            &[key(256)],
        );
        // 256 tokens: the boundary at 256 would leave nothing to compute, so it is skipped.
        assert_eq!(hit(&index, 256, 256), HybridHit::default());
        assert_eq!(
            hit(&index, 257, 257),
            HybridHit {
                device_tokens: 256,
                host_tokens: 0
            }
        );
        assert_eq!(hit(&index, 0, 0), HybridHit::default());
    }

    #[test]
    fn bookkeeping_clears_and_removes_workers() {
        let index = stored_index();
        let other = WorkerWithDpRank::new(7, 1);
        index.store(
            other,
            StorageTier::Device,
            HybridCacheGroup::Recurrent,
            &[key(128)],
        );
        assert_eq!(
            index.stats(),
            HybridIndexStats {
                workers: 2,
                keys: 10
            }
        );
        assert_eq!(
            index.keys(worker(), StorageTier::Device, HybridCacheGroup::Recurrent),
            vec![key(23_040), key(24_192)]
        );
        assert!(!index.store(
            worker(),
            StorageTier::Disk,
            HybridCacheGroup::FullAttention,
            &[1]
        ));
        index.clear(worker(), Some(StorageTier::Device));
        assert_eq!(
            index.stats(),
            HybridIndexStats {
                workers: 2,
                keys: 6
            }
        );
        index.remove_worker_dp_rank(7, 1);
        assert_eq!(
            index.stats(),
            HybridIndexStats {
                workers: 1,
                keys: 5
            }
        );
        index.remove_worker(7);
        assert_eq!(index.stats(), HybridIndexStats::default());
    }

    #[test]
    fn duplicate_copies_keep_the_key_until_the_last_removal() {
        // Two requests of one session computed the same block in one step: vLLM announces
        // the content twice and evicts the copies one at a time.
        let index = HybridProbeIndex::new(BLOCK as u32, UNIT as u32);
        let w = worker();
        use HybridCacheGroup::FullAttention;
        use StorageTier::Device;
        index.store(w, Device, FullAttention, &[key(12_288)]);
        index.store(w, Device, FullAttention, &[key(12_288)]);
        assert_eq!(
            index.stats(),
            HybridIndexStats {
                workers: 1,
                keys: 1
            }
        );
        index.remove(w, Device, FullAttention, &[key(12_288)]);
        assert_eq!(index.keys(w, Device, FullAttention), vec![key(12_288)]);
        // Removing a key that was never stored is ignored.
        index.remove(w, Device, FullAttention, &[key(24_576)]);
        assert_eq!(index.keys(w, Device, FullAttention), vec![key(12_288)]);
        index.remove(w, Device, FullAttention, &[key(12_288)]);
        assert_eq!(index.stats(), HybridIndexStats::default());
    }

    #[test]
    fn dense_retention_probe_walks_leaf_to_root() {
        // Every 1,536 boundary has a KDA key; the deepest one at or below the MLA coverage wins.
        let index = HybridProbeIndex::new(BLOCK as u32, UNIT as u32);
        let w = worker();
        let kda: Vec<u64> = (1..=15).map(|i| key(i * 1_536)).collect();
        index.store(w, StorageTier::Device, HybridCacheGroup::Recurrent, &kda);
        index.store(
            w,
            StorageTier::Device,
            HybridCacheGroup::FullAttention,
            &[key(12_288), key(20_096)],
        );
        // Coverage = 20,096 (partial tail key); deepest KDA <= 20,096 is 19,968 (13 * 1,536).
        assert_eq!(
            hit(&index, 24_300, 24_300),
            HybridHit {
                device_tokens: 19_968,
                host_tokens: 0
            }
        );
    }

    #[test]
    fn hybrid_key_events_round_trip_and_bypass_block_indexes() {
        use crate::protocols::{
            ExternalSequenceBlockHash, HybridKeysData, KvCacheEvent, KvCacheEventData, RouterEvent,
        };

        let data = KvCacheEventData::HybridKeysStored(HybridKeysData {
            group: HybridCacheGroup::Recurrent,
            hashes: vec![
                ExternalSequenceBlockHash(7),
                ExternalSequenceBlockHash(u64::MAX),
            ],
        });
        let json = serde_json::to_string(&data).unwrap();
        assert!(
            json.contains("hybrid_keys_stored") && json.contains("recurrent"),
            "{json}"
        );
        let back: KvCacheEventData = serde_json::from_str(&json).unwrap();
        assert_eq!(back, data);
        let packed = rmp_serde::to_vec_named(&data).unwrap();
        let back: KvCacheEventData = rmp_serde::from_slice(&packed).unwrap();
        assert_eq!(back, data);

        let event = RouterEvent::with_storage_tier(
            7,
            KvCacheEvent {
                event_id: 1,
                data,
                dp_rank: 0,
            },
            StorageTier::HostPinned,
        );
        assert!(event.is_hybrid_keys());
        assert!(matches!(event.targets_primary(), Ok(false)));
        assert!(matches!(
            event.targets_lower_tier(StorageTier::HostPinned),
            Ok(false)
        ));
        let wire = rmp_serde::to_vec_named(&event).unwrap();
        let back: RouterEvent = rmp_serde::from_slice(&wire).unwrap();
        assert_eq!(back, event);

        // Every block index ignores the variant instead of failing the event.
        let mut tree = crate::RadixTree::new();
        assert!(tree.apply_event(event.clone()).is_ok());
        assert_eq!(tree.current_size(), 0);
    }
}
