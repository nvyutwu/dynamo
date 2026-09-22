// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The opt-in routing-decision trace under the two-tier worker-selection policy.
//!
//! Upstream's trace fires only for the default selector. This binary drives the real two-tier
//! policy, as the baked image YAML resolves it, through `WorkerSelector::select_worker` with
//! `DYN_ROUTER_DECISION_TRACE_ENABLED=1` and checks that every branch of the policy yields a
//! `routing_decision` whose worker ids, overlap ranking, oracle and parameters match the inputs.
//!
//! Separate binary on purpose: the enable flag is read once per process through a `LazyLock`,
//! so it must be set before any selection in this process. Do not merge into the crate's unit
//! tests, which run selections with the flag unset.

use std::collections::HashMap;

use dynamo_kv_router::protocols::{
    RoutingConstraints, RoutingDecisionTrace, WorkerConfigLike, WorkerWithDpRank,
};
use dynamo_kv_router::scheduling::{OverlapSignals, ScheduleMode};
use dynamo_kv_router::services::selection::WorkerSelectionPolicyRegistry;
use dynamo_kv_router::{
    KvRouterConfig, RoutingPartitionRef, SchedulingRequest, WorkerLoadProjection,
    WorkerSelectionInput, WorkerSelector, WorkerType,
};

const BAKED_IMAGE_YAML: &str = r#"
worker_selection:
  aggregated: dynamo-two-tier-cost-fn
  instances:
    - name: dynamo-two-tier-cost-fn
      type: dynamo-two-tier-cost-fn
"#;

const BLOCK_SIZE: u32 = 16;
/// Eight blocks: makes the cache_ratio arithmetic below exact.
const EIGHT_BLOCKS: usize = 128;
const A: u64 = 7;
const B: u64 = 11;

struct TestWorker;

impl WorkerConfigLike for TestWorker {
    fn data_parallel_start_rank(&self) -> u32 {
        0
    }
    fn data_parallel_size(&self) -> u32 {
        1
    }
    fn max_num_batched_tokens(&self) -> Option<u64> {
        None
    }
    fn total_kv_blocks(&self) -> Option<u64> {
        Some(1024)
    }
}

fn worker(id: u64) -> WorkerWithDpRank {
    WorkerWithDpRank::from_worker_id(id)
}

fn enable_trace() {
    // Edition 2024: mutating the process environment is unsafe. This binary is single-purpose
    // and sets it before the first selection, which is the only ordering that matters.
    unsafe {
        std::env::set_var("DYN_ROUTER_DECISION_TRACE_ENABLED", "1");
        std::env::remove_var("DYN_ROUTER_DECISION_TRACE_SAMPLE_RATE");
    }
}

/// `(worker_id, device_blocks, host_pinned_blocks, active_requests)` per worker.
fn select(
    request_id: &str,
    workers: [(u64, usize, usize, usize); 2],
) -> (WorkerWithDpRank, Option<RoutingDecisionTrace>) {
    enable_trace();
    let policy_file = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(policy_file.path(), BAKED_IMAGE_YAML).unwrap();
    let config = KvRouterConfig {
        router_policy_config: Some(policy_file.path().display().to_string()),
        ..Default::default()
    };
    let mut registry = WorkerSelectionPolicyRegistry::default();
    dynamo_custom_policy_builtin::register(&mut registry).unwrap();
    let factory = registry
        .resolve(&config)
        .expect("baked YAML must resolve")
        .expect("baked YAML must produce a factory");
    let mut policy = factory(
        &config,
        WorkerType::Aggregated,
        RoutingPartitionRef::new("model", "default"),
    );

    let mut request = SchedulingRequest {
        mode: ScheduleMode::QueryOnly {
            request_id: Some(request_id.to_string()),
        },
        token_seq: None,
        isl_tokens: EIGHT_BLOCKS,
        lora_name: None,
        expected_output_tokens: None,
        affinity_target: None,
        pinned_worker: None,
        allowed_worker_ids: None,
        routing_constraints: RoutingConstraints::default(),
        router_config_override: None,
        track_prefill_tokens: true,
        priority_jump: 0.0,
        strict_priority: 0,
        policy_class: None,
        session_context: None,
        overlap: OverlapSignals::default(),
        kv_transfer_candidates: None,
        retain_kv_transfer_chain: false,
        shared_cache_hits: None,
        worker_loads: Default::default(),
        resp_tx: None,
    };
    for (id, device_blocks, host_blocks, active_requests) in workers {
        request
            .overlap
            .tier_overlap_blocks
            .device
            .insert(worker(id), device_blocks);
        request
            .overlap
            .tier_overlap_blocks
            .host_pinned
            .insert(worker(id), host_blocks);
        request.worker_loads.insert(
            worker(id),
            WorkerLoadProjection {
                active_requests,
                ..Default::default()
            },
        );
    }
    let configs = HashMap::from(workers.map(|(id, _, _, _)| (id, TestWorker)));
    let result = policy
        .select_worker(WorkerSelectionInput::configured(
            &configs,
            &request,
            request.eligibility(),
            BLOCK_SIZE,
        ))
        .unwrap();
    (result.worker, result.decision_trace)
}

fn param(trace: &RoutingDecisionTrace, name: &str) -> f64 {
    trace
        .policy_parameters
        .iter()
        .find(|p| p.name == name)
        .unwrap_or_else(|| panic!("parameter {name} missing from {trace:?}"))
        .value
}

/// Worker A holds the prefix only in CPU offload (6 of 8 blocks), B holds 4 blocks in HBM and is
/// less loaded. At host_cache_weight 0.75 A's two-tier overlap is 2 + 0.75*6 = 6.5 > B's 4, and
/// 6.5/8 > cache_threshold 0.5, so the cache tier fires and A wins despite the load.
#[test]
fn cache_tier_win_is_traced_with_cpu_overlap_and_oracle() {
    let (selected, trace) = select("req-cache-tier", [(A, 2, 6, 3), (B, 4, 0, 1)]);
    assert_eq!(selected, worker(A));
    let trace = trace.expect("two-tier selection must emit a routing_decision when enabled");

    assert_eq!(trace.policy, "dynamo-two-tier-cost-fn");
    assert_eq!(
        trace.selection_reason,
        "cache_tier_least_loaded_among_max_overlap"
    );
    assert_eq!(trace.selected_worker_id, A);
    assert_eq!(trace.max_overlap_worker_id, A);
    assert_eq!(trace.request_blocks, 8);
    assert_eq!(trace.block_size, BLOCK_SIZE);
    assert_eq!(trace.avoidable_prefill_token_equivalents, 0.0);

    assert_eq!(param(&trace, "host_cache_weight"), 0.75);
    assert_eq!(param(&trace, "cache_threshold"), 0.5);
    assert_eq!(param(&trace, "max_effective_overlap_blocks"), 6.5);
    assert_eq!(param(&trace, "cache_ratio"), 6.5 / 8.0);
    assert_eq!(param(&trace, "min_active_requests"), 1.0);
    assert_eq!(param(&trace, "max_active_requests"), 3.0);

    let a = trace.candidates.iter().find(|c| c.worker_id == A).unwrap();
    let b = trace.candidates.iter().find(|c| c.worker_id == B).unwrap();
    assert!(a.selected && a.max_overlap && !b.selected && !b.max_overlap);
    assert_eq!((a.device_overlap_blocks, a.host_overlap_blocks), (2.0, 6.0));
    assert_eq!((b.device_overlap_blocks, b.host_overlap_blocks), (4.0, 0.0));
    // The trace carries the policy's own ranking, not the raw effective overlap.
    assert_eq!(a.effective_overlap_blocks, 6.5);
    assert_eq!(b.effective_overlap_blocks, 4.0);
    assert_eq!((a.active_requests, b.active_requests), (3, 1));
}

/// Nobody clears the cache threshold (max 2 of 8 blocks), so the least-loaded worker wins and the
/// trace names the oracle it passed over, with the avoidable prefill in token equivalents.
#[test]
fn least_loaded_fallback_is_traced_with_the_passed_over_oracle() {
    let (selected, trace) = select("req-fallback", [(A, 2, 0, 5), (B, 1, 0, 1)]);
    assert_eq!(selected, worker(B));
    let trace = trace.unwrap();
    assert_eq!(trace.selection_reason, "least_loaded_no_cache_winner");
    assert_eq!(trace.selected_worker_id, B);
    assert_eq!(trace.max_overlap_worker_id, A);
    assert_eq!(param(&trace, "cache_ratio"), 2.0 / 8.0);
    // (2 - 1) blocks * 16 tokens the router chose to recompute rather than reuse.
    assert_eq!(trace.avoidable_prefill_token_equivalents, 16.0);
}

/// B holds the whole prefix but is 40 requests deeper than A: the load tier overrides cache
/// affinity and the trace says so.
#[test]
fn load_imbalance_override_is_traced() {
    let (selected, trace) = select("req-imbalance", [(A, 0, 0, 0), (B, 8, 0, 40)]);
    assert_eq!(selected, worker(A));
    let trace = trace.unwrap();
    assert_eq!(trace.selection_reason, "load_imbalance_least_loaded");
    assert_eq!(trace.selected_worker_id, A);
    assert_eq!(trace.max_overlap_worker_id, B);
    assert_eq!(param(&trace, "balance_abs_threshold"), 32.0);
    assert_eq!(param(&trace, "max_active_requests"), 40.0);
    assert_eq!(trace.avoidable_prefill_token_equivalents, 8.0 * 16.0);
}

/// Prints the JSON exactly as it appears under `routing_decision` on a `request_end` record, for
/// review. Run with `--nocapture`.
#[test]
fn sample_trace_json() {
    let (_, trace) = select("req-sample", [(A, 2, 6, 3), (B, 4, 0, 1)]);
    let trace = trace.unwrap();
    let json = serde_json::to_string_pretty(&trace).unwrap();
    println!("SAMPLE_ROUTING_DECISION_BEGIN\n{json}\nSAMPLE_ROUTING_DECISION_END");
    assert!(json.contains("\"policy\": \"dynamo-two-tier-cost-fn\""));
    assert!(json.contains("\"policy_parameters\""));
}
