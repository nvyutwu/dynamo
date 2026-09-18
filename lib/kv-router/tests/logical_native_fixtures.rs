// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cross-framework parity: replay cut sequences that the vLLM producer actually
//! encoded, through the public consumer API, and compare against the engine's
//! own expectations.
//!
//! The Rust unit tests build their own wire structs, so they cannot catch a
//! producer/consumer schema drift. These fixtures can: they are MessagePack
//! bytes emitted by `LogicalCPUProjector` and the native CPU manager, captured
//! by `generate_fixtures.py` in the release repro directory.
//!
//! Synthetic native hashes and token IDs; no tensor transfer is claimed.

use dynamo_kv_router::protocols::{ResidencyOwner, WorkerWithDpRank};
use dynamo_kv_router::zmq_wire::logical::{DecodeLimits, LogicalStream, query_tokens};
use serde::Deserialize;

const FIXTURES: &str = include_str!("data/native-cpu-root-v1.json");

#[derive(Deserialize)]
struct Fixtures {
    contract: String,
    namespace: String,
    repeated_query_token: u32,
    data_parallel_rank: u32,
    cuts: Vec<Cut>,
    checkpoints: Vec<Checkpoint>,
    replay: Replay,
}

#[derive(Deserialize)]
struct Cut {
    #[serde(rename = "type")]
    kind: String,
    cursor: u64,
    /// The publisher's own frame sequence, as it appeared on the socket.
    transport_sequence: u64,
    payload_hex: String,
}

/// What the router replay endpoint actually returned for a request from
/// sequence 0 — the path a late subscriber takes.
#[derive(Deserialize)]
struct Replay {
    cuts: Vec<Cut>,
    queries: Vec<Expectation>,
}

#[derive(Deserialize)]
struct Checkpoint {
    name: String,
    /// Number of leading cuts the engine had published at this point.
    through: usize,
    queries: Vec<Expectation>,
}

#[derive(Deserialize)]
struct Expectation {
    #[serde(rename = "N")]
    prompt_tokens: u32,
    expected: u32,
}

fn payload(cut: &Cut) -> Vec<u8> {
    (0..cut.payload_hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&cut.payload_hex[i..i + 2], 16).expect("hex payload"))
        .collect()
}

fn fixtures() -> Fixtures {
    let parsed: Fixtures = serde_json::from_str(FIXTURES).expect("fixture JSON");
    assert_eq!(parsed.contract, "vllm-logical-cpu-v1");
    parsed
}

fn owner(fixtures: &Fixtures) -> ResidencyOwner {
    ResidencyOwner::worker(WorkerWithDpRank::new(11, fixtures.data_parallel_rank))
}

/// Every checkpoint, replayed from the start of the stream exactly as a
/// subscriber that never missed a cut would see it.
#[test]
fn native_cut_sequences_match_the_engine_expectations() {
    let fixtures = fixtures();
    assert!(
        fixtures.cuts.iter().any(|cut| cut.kind == "LogicalUpdate"),
        "fixtures must exercise the incremental path, not snapshots alone",
    );
    for checkpoint in &fixtures.checkpoints {
        let mut stream = LogicalStream::new(
            owner(&fixtures),
            fixtures.namespace.clone(),
            DecodeLimits::default(),
        );
        for (index, cut) in fixtures.cuts[..checkpoint.through].iter().enumerate() {
            stream
                .ingest(cut.transport_sequence, &payload(cut))
                .unwrap_or_else(|e| panic!("{}: cut {index} rejected: {e}", checkpoint.name));
        }
        let view = stream
            .view()
            .unwrap_or_else(|| panic!("{}: no view after a complete stream", checkpoint.name));
        for expectation in &checkpoint.queries {
            let tokens = vec![fixtures.repeated_query_token; expectation.prompt_tokens as usize];
            let result = query_tokens(view, &tokens);
            assert_eq!(
                result.eligible_cpu_root_tokens,
                Some(expectation.expected),
                "{}: N={} produced {:?} (raw coverage {})",
                checkpoint.name,
                expectation.prompt_tokens,
                result.eligible_cpu_root_tokens,
                result.raw_cpu_tokens,
            );
        }
    }
}

/// A subscriber that joins mid-stream must reach the same answer once it has
/// seen an authoritative snapshot, and must hold no view before that.
#[test]
fn a_late_subscriber_recovers_only_from_an_authoritative_snapshot() {
    let fixtures = fixtures();
    let last = fixtures.checkpoints.last().expect("a checkpoint");
    let final_snapshot = fixtures.cuts[..last.through]
        .iter()
        .rposition(|cut| cut.kind == "LogicalSnapshot")
        .expect("the stream re-bases");

    let mut stream = LogicalStream::new(
        owner(&fixtures),
        fixtures.namespace.clone(),
        DecodeLimits::default(),
    );
    // Updates alone cannot bootstrap: they are dropped, not guessed at.
    for cut in fixtures.cuts[..final_snapshot].iter() {
        if cut.kind == "LogicalUpdate" {
            assert!(
                !stream
                    .ingest(cut.transport_sequence, &payload(cut))
                    .expect("clean drop")
            );
            assert!(stream.view().is_none(), "an update bootstrapped a view");
        }
    }
    for cut in fixtures.cuts[final_snapshot..last.through].iter() {
        stream
            .ingest(cut.transport_sequence, &payload(cut))
            .expect("tail of the stream");
    }
    let view = stream.view().expect("recovered view");
    for expectation in &last.queries {
        let tokens = vec![fixtures.repeated_query_token; expectation.prompt_tokens as usize];
        assert_eq!(
            query_tokens(view, &tokens).eligible_cpu_root_tokens,
            Some(expectation.expected),
            "late subscriber disagreed at N={}",
            expectation.prompt_tokens,
        );
    }
}

/// The router replay endpoint's own answer, replayed on its own. A subscriber
/// that missed everything live must reach the right answer from the replay
/// window alone — which requires that window to still hold a covering snapshot.
#[test]
fn the_replay_window_alone_rebuilds_the_view() {
    let fixtures = fixtures();
    assert!(
        !fixtures.replay.cuts.is_empty(),
        "the replay request returned nothing",
    );
    assert_eq!(
        fixtures.replay.cuts[0].kind, "LogicalSnapshot",
        "a replay window that opens with an update cannot be applied at all",
    );
    let mut stream = LogicalStream::new(
        owner(&fixtures),
        fixtures.namespace.clone(),
        DecodeLimits::default(),
    );
    for cut in &fixtures.replay.cuts {
        stream
            .ingest(cut.transport_sequence, &payload(cut))
            .expect("replayed cut rejected");
    }
    let view = stream.view().expect("no view after a full replay");
    for expectation in &fixtures.replay.queries {
        let tokens = vec![fixtures.repeated_query_token; expectation.prompt_tokens as usize];
        assert_eq!(
            query_tokens(view, &tokens).eligible_cpu_root_tokens,
            Some(expectation.expected),
            "replay-only subscriber disagreed at N={}",
            expectation.prompt_tokens,
        );
    }
}

/// The engine's cursors must already be contiguous within an epoch. If the
/// producer ever skips one, this catches it in the fixtures rather than in a
/// consumer that silently went unknown under load.
#[test]
fn published_cursors_are_contiguous_within_each_epoch() {
    let fixtures = fixtures();
    let mut expected: Option<u64> = None;
    for cut in &fixtures.cuts {
        match expected {
            Some(next) if cut.cursor == next => {}
            _ => assert_eq!(cut.cursor, 0, "a new epoch must restart at cursor 0"),
        }
        expected = Some(cut.cursor + 1);
    }
}
