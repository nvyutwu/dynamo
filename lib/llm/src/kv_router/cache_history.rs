// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded history of KV identities that have definitely been computed.
//!
//! This backs stage **F1** of the cache-loss funnel ("prompt tokens previously
//! computed"). Every other stage is derivable from state the router or the
//! backend already publishes; F1 is not, because nothing else distinguishes
//! "this context is new" from "this context was computed and then evicted".
//!
//! The identity is the canonical rolling sequence hash, not a bare token-block
//! hash, so equal token blocks under different preceding contexts stay distinct
//! and match Dynamo's own routing identity. No token IDs, prompt text, or
//! request identifiers are retained.
//!
//! Scope caveat: the ledger is **per frontend process**. A deployment running N
//! frontend replicas behind a non-KV-aware load balancer sees roughly 1/N of the
//! history per replica, so F1 is a lower bound there. Read it per replica, or
//! accept the dilution.

use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
};

use parking_lot::Mutex;

use dynamo_kv_router::protocols::{BlockExtraInfo, TokensWithHashes};

/// Enables the F0-F5 funnel. Absent or falsey means no allocation and no F1.
pub const CACHE_LOSS_FUNNEL_ENABLED_ENV: &str = "DYN_CACHE_LOSS_FUNNEL_ENABLED";
pub const HISTORY_BLOCK_CAPACITY_ENV: &str = "DYN_CACHE_LOSS_HISTORY_BLOCKS";
pub const HISTORY_BYTES_ENV: &str = "DYN_CACHE_LOSS_HISTORY_BYTES";

/// Default number of complete sequence-hash records retained by a frontend.
/// Each record represents one full KV block.
pub const DEFAULT_HISTORY_BLOCK_CAPACITY: usize = 5_000_000;
pub const DEFAULT_HISTORY_BYTES: usize = 256 * 1024 * 1024;

/// Conservative planning estimate: an 8-byte FIFO sequence hash plus the
/// amortized hash-map key, refcount, bucket slack, and allocator overhead.
/// Deliberately larger than `size_of::<u64>()`; this is a capacity model, not a
/// promise about a particular allocator build.
pub const ESTIMATED_BYTES_PER_HISTORY_RECORD: usize = 32;

/// FIFO ledger of canonical sequence hashes that a completed request computed.
///
/// Records are retained in arrival order; a refcount keeps a hash present while
/// any retained record still refers to it, so a repeated prefix does not expire
/// early just because an older occurrence aged out.
#[derive(Debug)]
pub struct CacheHistory {
    capacity_blocks: usize,
    capacity_bytes: usize,
    block_tokens: u64,
    records: VecDeque<u64>,
    retained: HashMap<u64, u32>,
}

impl CacheHistory {
    /// Build from the environment, or `None` when cache-loss telemetry is off.
    pub fn from_env(block_tokens: u32) -> Option<Arc<Mutex<Self>>> {
        if !cache_loss_enabled() || block_tokens == 0 {
            return None;
        }
        let requested_blocks = parse_positive_env(HISTORY_BLOCK_CAPACITY_ENV)
            .unwrap_or(DEFAULT_HISTORY_BLOCK_CAPACITY);
        let requested_bytes =
            parse_positive_env(HISTORY_BYTES_ENV).unwrap_or(DEFAULT_HISTORY_BYTES);
        let capacity_bytes = requested_bytes.max(ESTIMATED_BYTES_PER_HISTORY_RECORD);
        let byte_limited_blocks = capacity_bytes / ESTIMATED_BYTES_PER_HISTORY_RECORD;
        let capacity_blocks = requested_blocks.min(byte_limited_blocks.max(1));
        Some(Arc::new(Mutex::new(Self::new_with_budget(
            capacity_blocks,
            block_tokens,
            capacity_bytes,
        ))))
    }

    pub fn new(capacity_blocks: usize, block_tokens: u32) -> Self {
        Self::new_with_budget(
            capacity_blocks,
            block_tokens,
            capacity_blocks.saturating_mul(ESTIMATED_BYTES_PER_HISTORY_RECORD),
        )
    }

    fn new_with_budget(capacity_blocks: usize, block_tokens: u32, capacity_bytes: usize) -> Self {
        assert!(
            capacity_blocks > 0,
            "cache history capacity must be positive"
        );
        assert!(
            block_tokens > 0,
            "cache history block size must be positive"
        );
        Self {
            capacity_blocks,
            capacity_bytes,
            block_tokens: u64::from(block_tokens),
            records: VecDeque::with_capacity(capacity_blocks.min(65_536)),
            retained: HashMap::new(),
        }
    }

    /// Tokens in the longest complete prefix whose identities were computed by a
    /// prior completed request within this process lifetime.
    pub fn previously_computed_tokens(&self, sequence_hashes: &[u64]) -> u64 {
        let blocks = sequence_hashes
            .iter()
            .take_while(|hash| self.retained.contains_key(*hash))
            .count() as u64;
        blocks.saturating_mul(self.block_tokens)
    }

    /// Retain complete canonical sequence hashes, evicting oldest-first.
    pub fn record_completed(&mut self, sequence_hashes: impl IntoIterator<Item = u64>) {
        for hash in sequence_hashes {
            if self.records.len() == self.capacity_blocks {
                let evicted = self.records.pop_front().expect("history was non-empty");
                if let Some(count) = self.retained.get_mut(&evicted) {
                    *count -= 1;
                    if *count == 0 {
                        self.retained.remove(&evicted);
                    }
                } else {
                    debug_assert!(false, "history refcount missing for an evicted record");
                }
            }
            self.records.push_back(hash);
            *self.retained.entry(hash).or_default() += 1;
        }
    }

    pub fn stats(&self) -> CacheHistoryStats {
        CacheHistoryStats {
            capacity_blocks: self.capacity_blocks,
            capacity_bytes: self.capacity_bytes,
            retained_records: self.records.len(),
            retained_unique_hashes: self.retained.len(),
            represented_tokens: (self.records.len() as u64).saturating_mul(self.block_tokens),
            estimated_retained_bytes: self
                .records
                .len()
                .saturating_mul(ESTIMATED_BYTES_PER_HISTORY_RECORD),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CacheHistoryStats {
    pub capacity_blocks: usize,
    pub capacity_bytes: usize,
    pub retained_records: usize,
    pub retained_unique_hashes: usize,
    pub represented_tokens: u64,
    pub estimated_retained_bytes: usize,
}

fn cache_loss_enabled() -> bool {
    std::env::var(CACHE_LOSS_FUNNEL_ENABLED_ENV)
        .ok()
        .is_some_and(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
}

fn parse_positive_env(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .filter(|&value: &usize| value > 0)
}

/// Per-request state: the prompt identity snapshot plus generated tokens, so the
/// request's own computed context can be admitted to the ledger when it ends.
///
/// Generated tokens are kept per output-choice index. At finalization the newest
/// sampled token of each branch is excluded: it was returned to the caller but
/// has not been fed back through the model, so it has no KV entry yet.
pub struct CacheHistoryRequest {
    prompt_tokens: Vec<u32>,
    block_mm_infos: Option<Vec<Option<BlockExtraInfo>>>,
    lora_name: Option<String>,
    cache_namespace: Option<String>,
    block_size: u32,
    is_eagle: bool,
    output_branches: HashMap<u32, Vec<u32>>,
    finalized: bool,
}

impl CacheHistoryRequest {
    pub fn new(
        prompt_tokens: Vec<u32>,
        block_mm_infos: Option<Vec<Option<BlockExtraInfo>>>,
        lora_name: Option<String>,
        cache_namespace: Option<String>,
        block_size: u32,
        is_eagle: bool,
    ) -> Self {
        Self {
            prompt_tokens,
            block_mm_infos,
            lora_name,
            cache_namespace,
            block_size,
            is_eagle,
            output_branches: HashMap::new(),
            finalized: false,
        }
    }

    /// F1 for this request, against the ledger as it stands before the request runs.
    pub fn previously_computed_tokens(&self, history: &CacheHistory) -> u64 {
        history.previously_computed_tokens(&self.sequence_hashes(&self.prompt_tokens))
    }

    pub fn observe_output(&mut self, output_index: u32, token_ids: &[u32]) {
        if !token_ids.is_empty() {
            self.output_branches
                .entry(output_index)
                .or_default()
                .extend_from_slice(token_ids);
        }
    }

    pub fn prompt_hashes(&self) -> Vec<u64> {
        self.sequence_hashes(&self.prompt_tokens)
    }

    /// One hash chain per output branch, over prompt + generated-minus-newest.
    pub fn output_hashes(&self) -> Vec<Vec<u64>> {
        self.output_branches
            .values()
            .filter_map(|output| {
                let computed_output = &output[..output.len().saturating_sub(1)];
                (!computed_output.is_empty()).then(|| {
                    let mut sequence =
                        Vec::with_capacity(self.prompt_tokens.len() + computed_output.len());
                    sequence.extend_from_slice(&self.prompt_tokens);
                    sequence.extend_from_slice(computed_output);
                    self.sequence_hashes(&sequence)
                })
            })
            .collect()
    }

    /// Admit this request's computed context. Idempotent.
    pub fn finalize(&mut self, history: &mut CacheHistory) {
        if self.finalized {
            return;
        }
        self.finalized = true;
        history.record_completed(self.prompt_hashes());
        for hashes in self.output_hashes() {
            history.record_completed(hashes);
        }
    }

    /// Mirrors the identity `KvRouter` uses for routing. Any field that
    /// participates in routing identity must participate here too, or F1 will
    /// credit a prefix that the router would treat as a different context.
    fn sequence_hashes(&self, tokens: &[u32]) -> Vec<u64> {
        let mut tokens_with_hashes =
            TokensWithHashes::new(tokens.to_vec(), self.block_size).with_is_eagle(self.is_eagle);
        if let Some(infos) = &self.block_mm_infos {
            tokens_with_hashes = tokens_with_hashes.with_mm_infos(infos.clone());
        }
        if let Some(lora_name) = &self.lora_name {
            tokens_with_hashes = tokens_with_hashes.with_lora_name(lora_name.clone());
        }
        if let Some(cache_namespace) = &self.cache_namespace {
            tokens_with_hashes = tokens_with_hashes.with_cache_namespace(cache_namespace.clone());
        }
        tokens_with_hashes.get_or_compute_seq_hashes().to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retains_recent_records_and_expires_the_oldest() {
        let mut history = CacheHistory::new(2, 16);
        history.record_completed([10, 20]);
        assert_eq!(history.previously_computed_tokens(&[10, 20]), 32);

        history.record_completed([30]);
        assert_eq!(history.previously_computed_tokens(&[10, 20]), 0);
        assert_eq!(history.previously_computed_tokens(&[20, 30]), 32);
        assert_eq!(history.stats().represented_tokens, 32);
    }

    #[test]
    fn duplicate_records_keep_a_hash_retained_until_all_expire() {
        let mut history = CacheHistory::new(2, 8);
        history.record_completed([7, 7]);
        history.record_completed([9]);

        assert_eq!(history.previously_computed_tokens(&[7]), 8);
        history.record_completed([11]);
        assert_eq!(history.previously_computed_tokens(&[7]), 0);
    }

    #[test]
    fn previously_computed_counts_only_a_complete_leading_prefix() {
        let mut history = CacheHistory::new(8, 4);
        history.record_completed([1, 2, 3]);
        // A gap at the second position stops the walk; suffix hits do not count.
        assert_eq!(history.previously_computed_tokens(&[1, 99, 3]), 4);
        assert_eq!(history.previously_computed_tokens(&[99, 1, 2]), 0);
    }

    #[test]
    fn generated_history_excludes_the_newest_sampled_token() {
        let mut request = CacheHistoryRequest::new(vec![1, 2, 3, 4], None, None, None, 2, false);
        request.observe_output(0, &[5, 6, 7]);
        let mut history = CacheHistory::new(32, 2);
        request.finalize(&mut history);

        // Prompt has two complete blocks; prompt plus the first two generated
        // tokens has three. The final sampled token is deliberately absent.
        assert_eq!(history.stats().retained_records, 5);
    }

    #[test]
    fn finalize_is_idempotent() {
        let mut request = CacheHistoryRequest::new(vec![1, 2, 3, 4], None, None, None, 2, false);
        let mut history = CacheHistory::new(32, 2);
        request.finalize(&mut history);
        let after_first = history.stats().retained_records;
        request.finalize(&mut history);
        assert_eq!(history.stats().retained_records, after_first);
    }

    #[test]
    fn a_completed_request_makes_its_own_prompt_a_later_f1_hit() {
        let mut history = CacheHistory::new(64, 2);
        let mut first = CacheHistoryRequest::new(vec![1, 2, 3, 4], None, None, None, 2, false);
        assert_eq!(first.previously_computed_tokens(&history), 0);
        first.finalize(&mut history);

        let second = CacheHistoryRequest::new(vec![1, 2, 3, 4, 5, 6], None, None, None, 2, false);
        assert_eq!(second.previously_computed_tokens(&history), 4);
    }

    #[test]
    fn default_budget_can_hold_the_requested_five_million_records() {
        let capacity_from_default_budget =
            DEFAULT_HISTORY_BYTES / ESTIMATED_BYTES_PER_HISTORY_RECORD;
        assert!(capacity_from_default_budget >= DEFAULT_HISTORY_BLOCK_CAPACITY);
    }
}
