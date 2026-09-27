// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Replay the CPU simulator's engine stream into the hybrid probe index.
//!
//! The fixture was exported by `export_replay_fixture.py` from a simulator that runs vLLM's
//! real scheduler, hybrid KV cache manager, offload connector scheduler and CPU LRU manager
//! at the Kimi-K3 production geometry (MLA 1,536 x DCP 8 = 12,288, KDA 1,536 in align mode,
//! hash unit 128, prefill budget 8,192). Per batch it holds the raw KV events every worker
//! published, then the requests of that batch with their engine prefix-chain hashes and the
//! tokens the engine itself would reuse on each worker (device and host). The test applies
//! the events through the same extraction the worker publisher uses and checks that every
//! `lookup` reproduces the engine's answer exactly.

use std::collections::HashMap;
use std::path::PathBuf;

use dynamo_kv_router::indexer::{HybridHit, HybridProbeIndex};
use dynamo_kv_router::protocols::{StorageTier, WorkerWithDpRank};
use dynamo_kv_router::zmq_wire::{
    BlockHashValue, HybridKeyOp, KvCacheSpecKind, RawKvEvent, ZmqEventNormalizer,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    block_size: u32,
    hash_unit: u32,
    workers: Vec<String>,
    steps: Vec<Step>,
}

#[derive(Deserialize)]
struct Step {
    #[serde(default)]
    events: HashMap<String, Vec<Event>>,
    requests: Vec<Request>,
}

#[derive(Deserialize)]
struct Event {
    #[serde(rename = "type")]
    kind: String,
    medium: Option<String>,
    #[serde(default)]
    block_hashes: Vec<u64>,
    block_size: Option<usize>,
    group_idx: Option<u32>,
    kv_cache_spec_kind: Option<KvCacheSpecKind>,
}

#[derive(Deserialize)]
struct Request {
    id: String,
    kind: String,
    num_tokens: usize,
    chain: Vec<u64>,
    #[serde(default)]
    expected: HashMap<String, Expected>,
}

#[derive(Deserialize)]
struct Expected {
    device: usize,
    host: usize,
}

impl Event {
    fn into_raw(self) -> Option<RawKvEvent> {
        let block_hashes = self
            .block_hashes
            .into_iter()
            .map(BlockHashValue::Unsigned)
            .collect();
        match self.kind.as_str() {
            "BlockStored" => Some(RawKvEvent::BlockStored {
                block_hashes,
                parent_block_hash: None,
                token_ids: Vec::new(),
                block_size: self.block_size.unwrap_or(0),
                medium: self.medium,
                lora_name: None,
                cache_namespace: None,
                block_mm_infos: None,
                is_eagle: None,
                group_idx: self.group_idx,
                kv_cache_spec_kind: self.kv_cache_spec_kind,
                kv_cache_spec_sliding_window: None,
                locality: None,
                ownership: None,
            }),
            "BlockRemoved" => Some(RawKvEvent::BlockRemoved {
                block_hashes,
                medium: self.medium,
                group_idx: self.group_idx,
                kv_cache_spec_kind: self.kv_cache_spec_kind,
                kv_cache_spec_sliding_window: None,
                locality: None,
                ownership: None,
            }),
            "AllBlocksCleared" => Some(RawKvEvent::AllBlocksCleared { ownership: None }),
            "TierBlocksCleared" => Some(RawKvEvent::TierBlocksCleared {
                medium: self.medium.unwrap_or_else(|| "GPU".to_string()),
                ownership: None,
            }),
            _ => None,
        }
    }
}

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

fn replay(name: &str) {
    let path = fixture_path(name);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    let fixture: Fixture = serde_json::from_str(&text).expect("fixture parses");

    let index = HybridProbeIndex::new(fixture.block_size, fixture.hash_unit);
    let worker_ids: HashMap<&str, WorkerWithDpRank> = fixture
        .workers
        .iter()
        .enumerate()
        .map(|(i, name)| (name.as_str(), WorkerWithDpRank::new(i as u64 + 1, 0)))
        .collect();
    // One normalizer per worker, as on the worker-side publisher.
    let mut normalizers: HashMap<&str, ZmqEventNormalizer> = fixture
        .workers
        .iter()
        .map(|name| {
            (
                name.as_str(),
                ZmqEventNormalizer::new(fixture.block_size)
                    .with_hybrid_keys(Some(fixture.hash_unit)),
            )
        })
        .collect();

    let mut events = 0usize;
    let mut key_events = 0usize;
    let mut pairs = 0usize;
    let mut hits = 0usize;
    let mut mismatches: Vec<String> = Vec::new();

    for (step_idx, step) in fixture.steps.into_iter().enumerate() {
        for (worker_name, worker_events) in step.events {
            let worker = worker_ids[worker_name.as_str()];
            let normalizer = normalizers
                .get_mut(worker_name.as_str())
                .expect("known worker");
            for event in worker_events {
                events += 1;
                let Some(raw) = event.into_raw() else {
                    continue;
                };
                match &raw {
                    RawKvEvent::AllBlocksCleared { .. } => index.clear(worker, None),
                    RawKvEvent::TierBlocksCleared { medium, .. } => {
                        if let Some(tier) = StorageTier::from_kv_medium(medium) {
                            index.clear(worker, Some(tier));
                        }
                    }
                    _ => {}
                }
                for key_event in normalizer.hybrid_keys(&raw, 0) {
                    key_events += 1;
                    match key_event.op {
                        HybridKeyOp::Stored => {
                            index.store(worker, key_event.tier, key_event.group, &key_event.hashes);
                        }
                        HybridKeyOp::Removed => {
                            index.remove(
                                worker,
                                key_event.tier,
                                key_event.group,
                                &key_event.hashes,
                            );
                        }
                    }
                }
            }
        }
        for request in step.requests {
            let predicted = index.lookup(&request.chain, request.num_tokens);
            for (worker_name, worker) in &worker_ids {
                pairs += 1;
                let expected = request
                    .expected
                    .get(*worker_name)
                    .map(|e| HybridHit {
                        device_tokens: e.device,
                        host_tokens: e.host,
                    })
                    .unwrap_or_default();
                let got = predicted.get(worker).copied().unwrap_or_default();
                if expected.total_tokens() > 0 {
                    hits += 1;
                }
                if got != expected {
                    mismatches.push(format!(
                        "step {step_idx} request {} ({}, {} tokens) worker {worker_name}: engine {:?}, probe {:?}",
                        request.id, request.kind, request.num_tokens, expected, got
                    ));
                }
            }
        }
    }

    eprintln!(
        "{name}: {events} raw events -> {key_events} key events; {pairs} request/worker pairs, {hits} with engine reuse, {} mismatches",
        mismatches.len()
    );
    assert!(pairs > 0, "fixture has no requests");
    assert!(hits > 0, "fixture has no reuse to check");
    assert!(
        mismatches.is_empty(),
        "{} of {pairs} pairs differ from the engine:\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );
}

/// Production retention (`prefix_cache_retention_interval=0`), production event shapes plus
/// the vLLM branch's `chain_removals_by_key`: every request/worker pair matches the engine.
#[test]
fn replays_k3_production_retention_stream_exactly() {
    replay("hybrid_replay_k3_prod.json");
}

/// Production retention with the anchored recurrent rows / GPU-anchored KDA keys of the vLLM
/// patch switched on: the chain-shaped rows credit only their terminal hash and every pair
/// still matches the engine.
#[test]
fn replays_k3_production_retention_anchored_stream_exactly() {
    replay("hybrid_replay_k3_prod_anchored.json");
}

/// Dense retention (every KDA block keeps its key) with the GPU-anchored KDA keys: the probe
/// walks leaf to root through the dense key set and matches the engine on every pair. This
/// stream also holds two physical copies of one MLA block (two turns of a session computed in
/// the same step) evicted one at a time, which the index's reference counting must survive.
#[test]
fn replays_k3_dense_retention_stream_exactly() {
    replay("hybrid_replay_k3_dense.json");
}
