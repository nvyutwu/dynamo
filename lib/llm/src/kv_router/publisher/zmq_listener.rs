// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result};
use futures::StreamExt;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use dynamo_kv_router::protocols::*;
use dynamo_kv_router::zmq_wire::*;

use crate::kv_router::metrics::kv_publisher_metrics;
use crate::utils::zmq::{connect_sub_socket, multipart_message};

pub(super) struct DecodedZmqKvBatch {
    pub(super) source_cursor: u64,
    pub(super) batch: KvEventBatch,
}

/// Decode the transport envelope shared by legacy and residency-aware inputs.
///
/// Callers retain their own malformed-input and protocol-version policies.
pub(super) fn decode_zmq_kv_batch(
    mut frames: crate::utils::zmq::MultipartMessage,
) -> Result<DecodedZmqKvBatch> {
    if frames.len() != 3 {
        anyhow::bail!("expected three ZMQ frames, received {}", frames.len());
    }
    let payload = frames.pop().expect("frame count was validated");
    let sequence = frames.pop().expect("frame count was validated");
    let sequence: [u8; 8] = sequence.try_into().map_err(|sequence: Vec<u8>| {
        anyhow::anyhow!(
            "ZMQ sequence must contain eight bytes, received {}",
            sequence.len()
        )
    })?;
    let batch = decode_event_batch(&payload).context("failed to decode KV event batch")?;
    Ok(DecodedZmqKvBatch {
        source_cursor: u64::from_be_bytes(sequence),
        batch,
    })
}

/// Wrap one hybrid key mutation as a placement event in the worker's tier.
pub(super) fn hybrid_placement_event(
    worker: WorkerWithDpRank,
    event_id: u64,
    key_event: HybridKeyEvent,
) -> PlacementEvent {
    let data = HybridKeysData {
        group: key_event.group,
        hashes: key_event
            .hashes
            .into_iter()
            .map(ExternalSequenceBlockHash)
            .collect(),
    };
    let data = match key_event.op {
        HybridKeyOp::Stored => KvCacheEventData::HybridKeysStored(data),
        HybridKeyOp::Removed => KvCacheEventData::HybridKeysRemoved(data),
    };
    PlacementEvent::new(
        Placement::local_worker(worker.worker_id, worker.dp_rank, key_event.tier),
        KvCacheEvent {
            event_id,
            data,
            dp_rank: worker.dp_rank,
        },
    )
}

/// `DYN_KV_EVENTS_HYBRID_KEYS` enables hybrid key extraction on the worker publisher;
/// `DYN_KV_EVENTS_HYBRID_HASH_UNIT` is the engine's `prefix_match_unit` (default 128).
pub(super) fn hybrid_hash_unit_from_env() -> Option<u32> {
    let enabled = std::env::var("DYN_KV_EVENTS_HYBRID_KEYS")
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false);
    if !enabled {
        return None;
    }
    let unit = std::env::var("DYN_KV_EVENTS_HYBRID_HASH_UNIT")
        .ok()
        .and_then(|value| value.trim().parse::<u32>().ok())
        .unwrap_or(128);
    if unit == 0 {
        tracing::warn!("DYN_KV_EVENTS_HYBRID_HASH_UNIT must be positive; hybrid keys disabled");
        return None;
    }
    Some(unit)
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn start_zmq_listener(
    zmq_endpoint: String,
    zmq_topic: String,
    worker_id: WorkerId,
    tx: mpsc::UnboundedSender<Vec<PlacementEvent>>,
    cancellation_token: CancellationToken,
    kv_block_size: u32,
    next_event_id: Arc<AtomicU64>,
    image_token_id: Option<u32>,
    video_token_id: Option<u32>,
    hybrid_hash_unit: Option<u32>,
) {
    tracing::debug!(
        "KVEventPublisher connecting to ZMQ endpoint {} (topic '{}')",
        zmq_endpoint,
        zmq_topic
    );
    if let Some(unit) = hybrid_hash_unit {
        tracing::info!(
            hash_unit = unit,
            "KVEventPublisher emitting hybrid engine-hash key events alongside block events"
        );
    }

    let mut normalizer = ZmqEventNormalizer::new(kv_block_size)
        .with_image_token_id(image_token_id)
        .with_video_token_id(video_token_id)
        .with_hybrid_keys(hybrid_hash_unit);
    let socket = match connect_sub_socket(&zmq_endpoint, Some(&zmq_topic)).await {
        Ok(socket) => socket,
        Err(error) => {
            tracing::error!(endpoint = %zmq_endpoint, topic = %zmq_topic, error = %error, "ZMQ listener failed to connect");
            return;
        }
    };
    let mut socket = socket;
    let metrics = kv_publisher_metrics();

    if cancellation_token.is_cancelled() {
        return;
    }

    let mut messages_processed = 0u64;

    let exit_reason = 'main: loop {
        tokio::select! {
            biased;

            _ = cancellation_token.cancelled() => {
                tracing::debug!("ZMQ listener received cancellation signal");
                break 'main String::from("cancellation token cancelled");
            }

            msg_result = socket.next() => {
                let frames = match msg_result {
                    Some(Ok(frames)) => multipart_message(frames),
                    Some(Err(error)) => {
                        tracing::error!(endpoint = %zmq_endpoint, error = %error, "ZMQ listener recv failed");
                        break 'main format!("ZMQ recv failed: {error}");
                    }
                    None => break 'main String::from("ZMQ stream ended"),
                };
                let DecodedZmqKvBatch {
                    source_cursor: engine_seq,
                    batch,
                } = match decode_zmq_kv_batch(frames) {
                    Ok(decoded) => decoded,
                    Err(error) => {
                        tracing::warn!(%error, "Failed to decode ZMQ KV batch");
                        continue;
                    }
                };

                tracing::trace!(
                    "ZMQ listener on {} received batch with {} events (engine_seq={}, dp_rank={})",
                    zmq_endpoint,
                    batch.events.len(),
                    engine_seq,
                    batch.data_parallel_rank.unwrap_or(0)
                );

                let dp_rank = batch.data_parallel_rank.unwrap_or(0).cast_unsigned();
                let mut events = Vec::with_capacity(batch.events.len());
                for raw_event in batch.events {
                    let event_type = raw_event.event_type_label();
                    if let Some(metrics) = &metrics {
                        metrics.increment_zmq_event("received", event_type);
                    }
                    let worker = WorkerWithDpRank::new(worker_id, dp_rank);
                    // Hybrid keys are read before the block filter below, which drops the
                    // recurrent group and every non-router-block store the probe index needs.
                    for key_event in normalizer.hybrid_keys(&raw_event, dp_rank) {
                        if let Some(metrics) = &metrics {
                            metrics.increment_zmq_event("hybrid_keys", event_type);
                        }
                        events.push(hybrid_placement_event(
                            worker,
                            next_event_id.fetch_add(1, Ordering::SeqCst),
                            key_event,
                        ));
                    }
                    let raw_event = match normalizer.preprocess_with_reason(raw_event, worker) {
                        Ok(raw_event) => raw_event,
                        Err(reason) => {
                            if let Some(metrics) = &metrics {
                                metrics.increment_zmq_filtered_event(event_type, reason.as_label());
                            }
                            continue;
                        }
                    };
                    if let Some(metrics) = &metrics {
                        metrics.increment_zmq_event("accepted", event_type);
                    }
                    let event_id = next_event_id.fetch_add(1, Ordering::SeqCst);
                    let Some(event) =
                        normalizer.normalize_preprocessed(raw_event, event_id, worker)
                    else {
                        if let Some(metrics) = &metrics {
                            metrics.increment_zmq_conversion_issue(event_type, "conversion_none");
                        }
                        continue;
                    };
                    if matches!(event.event.data, KvCacheEventData::Stored(ref data) if data.blocks.is_empty())
                        && let Some(metrics) = &metrics
                    {
                        metrics.increment_zmq_suspicious_event(event_type, "empty_store_blocks");
                    }
                    events.push(event);
                }
                if !events.is_empty() {
                    let event_count = events.len() as u64;
                    if tx.send(events).is_err() {
                        tracing::warn!("Failed to send message to channel - receiver dropped");
                        break 'main String::from("channel receiver dropped");
                    }
                    messages_processed += event_count;
                }
            }
        }
    };

    tracing::debug!(
        "ZMQ listener exiting, reason: {}, messages processed: {}",
        exit_reason,
        messages_processed
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hybrid_placement_event_carries_tier_group_and_keys() {
        let worker = WorkerWithDpRank::new(3, 1);
        let event = hybrid_placement_event(
            worker,
            42,
            HybridKeyEvent {
                tier: StorageTier::HostPinned,
                group: HybridCacheGroup::Recurrent,
                op: HybridKeyOp::Removed,
                hashes: vec![5, 6],
            },
        );
        assert_eq!(
            event.placement,
            Placement::local_worker(3, 1, StorageTier::HostPinned)
        );
        assert_eq!(event.event.event_id, 42);
        assert_eq!(event.event.dp_rank, 1);
        match &event.event.data {
            KvCacheEventData::HybridKeysRemoved(data) => {
                assert_eq!(data.group, HybridCacheGroup::Recurrent);
                assert_eq!(
                    data.hashes,
                    vec![ExternalSequenceBlockHash(5), ExternalSequenceBlockHash(6)]
                );
            }
            other => panic!("unexpected event data {other:?}"),
        }
        let router_event = event.into_router_event().unwrap();
        assert!(router_event.is_hybrid_keys());
        assert_eq!(router_event.storage_tier, StorageTier::HostPinned);
        assert!(matches!(router_event.targets_primary(), Ok(false)));

        let stored = hybrid_placement_event(
            worker,
            43,
            HybridKeyEvent {
                tier: StorageTier::Device,
                group: HybridCacheGroup::FullAttention,
                op: HybridKeyOp::Stored,
                hashes: vec![9],
            },
        );
        assert!(matches!(
            stored.event.data,
            KvCacheEventData::HybridKeysStored(_)
        ));
        assert!(stored.placement.is_local_gpu());
    }
}
