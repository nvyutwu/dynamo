# dynamo-kv-router

`dynamo-kv-router` provides the core KV-aware routing data structures and scheduling primitives used
by Dynamo to steer requests toward workers with the best cache overlap.

## What This Crate Provides

- `RadixTree` and `ConcurrentRadixTree` for prefix-overlap indexing
- `ThreadPoolIndexer` and `PositionalIndexer` for higher-throughput index backends
- `KvRouterConfig`, `RouterQueuePolicy`, and `LocalScheduler` for request routing
- Protocol and hashing helpers such as `RouterEvent`, `WorkerId`,
  `compute_block_hash_for_seq`, and `compute_seq_hash_for_block`

## Basic Rust Usage

```rust
use dynamo_kv_router::{
    KvRouterConfig, RadixTree, compute_block_hash_for_seq, compute_seq_hash_for_block,
};
use dynamo_kv_router::protocols::BlockHashOptions;

let prompt_tokens = vec![1_u32, 2, 3, 4, 5, 6, 7, 8];
let local_hashes = compute_block_hash_for_seq(&prompt_tokens, 4, BlockHashOptions::default());
let seq_hashes = compute_seq_hash_for_block(&local_hashes);

let router_config = KvRouterConfig::default();
let index = RadixTree::new();
let scores = index.find_matches(local_hashes, false);

assert!(router_config.use_kv_events);
assert_eq!(seq_hashes.len(), 2);
assert!(scores.scores.is_empty());
```

For end-to-end routing, pair the indexers with `LocalScheduler` and the worker/config protocol
types re-exported from the crate root.

## Features

- `metrics`: Prometheus metrics for router internals
- `runtime-protocols`: integration points with `dynamo-runtime`
- `standalone-indexer`: standalone indexer service support
- `bench`: internal benchmarking helpers

## Further Reading

- Router guide: <https://docs.nvidia.com/dynamo/components/router>
- Indexer internals:
  <https://github.com/ai-dynamo/dynamo/blob/main/lib/kv-router/src/indexer/README.md>
- Benchmarking the sharded KV indexer: [INDEXER_BENCH.md](../bench/kv_router/INDEXER_BENCH.md)
- Dynamo repository: <https://github.com/ai-dynamo/dynamo>

### Explaining cache-aware routing decisions

With `DYN_ROUTER_DECISION_TRACE_ENABLED=1` on the frontend, the existing request
trace's `routing_decision` uses schema `dynamo.router.decision.v44.v2`. Its cache
candidate fields are unchanged. The optional `score_decision` contains selected
and best-eligible-cache candidates' score inputs from the same admission view,
before reservation. `frontend_instance` records `HOSTNAME` when available.
Custom selectors that do not supply this explanation leave it absent. Older v1
records remain readable; missing explanations are not zero load.

Each score includes observed-load availability, active prefill tokens, active
decode blocks, additional active blocks for this request, tier overlap, effective
cache credit, and the final preference multiplier. Reconstruct its cost as:

```text
prefill_cost_blocks = prefill_load_scale * (raw_prefill_blocks - overlap_credit_blocks)
decode_cost_blocks = active_decode_blocks + additional_active_blocks
total_cost = (prefill_cost_blocks + decode_cost_blocks) * preference_multiplier
```

`method` distinguishes `minimum_cost` (including random ties), `softmax`, and
`pinned`; temperature is not used for a pinned choice. Effective weights, the
prefill tracking flag, request-override presence, and the load floor used for
credit decay are included. Pinned selection bypasses preference multiplication,
matching the selector. The snapshot recomputes only these two candidates from
immutable request inputs using the same score function; it does not resample or
change the winning worker. Collection is opt-in and retains no fleet-sized map,
request payload, or block hashes. Request identity comes from the enclosing
request trace; no new high-cardinality Prometheus labels are introduced.

For aggregated serving, prefill and decode costs refer to the same worker.
`E > S` plus a lower selected cost establishes a cache/load tradeoff for a
minimum-cost decision (inspect preference weighting as well). Different worker
IDs alone can be cache ties. Existing R/E/S token counters remain aggregate
coverage metrics, not decision-reason counters.

This does not establish physical cache availability or eviction during waiting.
`selection_unix_ms` is the router wall-clock time, not backend scheduling time.
The request tracker retains its first routing snapshot; retries must not be
interpreted as a complete per-attempt history. To attribute `S > actual`, join
backend lookup/acquisition timing and full-prefix versus partial-tail reuse,
then sampled per-tier checkpoint eviction/restoration evidence. A 128-token
partial-tail match and a 12,288-token router block are different granularities.

Validation commands (limit thread counts on constrained CPU hosts):

```bash
DYN_ROUTER_DECISION_TRACE_ENABLED=1 RUST_TEST_THREADS=2 TOKIO_WORKER_THREADS=2 \
  cargo test -p dynamo-kv-router --lib
DYN_ROUTER_DECISION_TRACE_ENABLED=0 cargo test -p dynamo-kv-router --lib decision_scores_explain_cache_sacrifice
cargo test -p dynamo-llm --no-default-features --lib routing_decision
```
