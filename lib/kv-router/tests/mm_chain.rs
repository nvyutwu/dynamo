// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A real Kimi-K3 image request (lyrix job 3172181, r5 image, request M00-mm-fill): the engine's
//! GPU full-attention row carried the tokens, the extra key
//! `(mark_mm_hash_for_extra_key(mm_hash), 83)` and the unit-chain key at 12,032. The frontend sees
//! the normalised routing sequence and the block-level image span; the probe path must recover the
//! engine tokens, rebuild the extra keys and reproduce that key.

use dynamo_kv_router::indexer::{
    EngineChainHasher, EngineHashAlgo, hybrid_mm_identifier, mm_engine_tokens,
    mm_extra_keys_by_unit,
};
use dynamo_kv_router::protocols::{BlockExtraInfo, BlockMmObjectInfo};
use dynamo_kv_router::zmq_wire::normalize_mm_placeholder_runs;

#[derive(serde::Deserialize)]
struct Fixture {
    tokens: Vec<u32>,
    mm_hash_hex: String,
    identifier: String,
    item_offset: usize,
    item_end: usize,
    image_token_id: u32,
    expected_last_unit_key: u64,
    units: usize,
}

#[test]
fn kimi_k3_image_request_chain_matches_the_engine() {
    let fx: Fixture = serde_json::from_str(include_str!("fixtures/mm_k3_m00_fill.json")).unwrap();
    let mm_hash = u64::from_str_radix(&fx.mm_hash_hex, 16).unwrap();
    assert_eq!(hybrid_mm_identifier(mm_hash), fx.identifier);
    const BLOCK: usize = 12_288;
    const UNIT: u32 = 128;

    // frontend routing sequence: one (partial) router block, pad run normalised
    let (routing, _) =
        normalize_mm_placeholder_runs(&fx.tokens, Some(fx.image_token_id), None, &[mm_hash])
            .unwrap();
    assert_ne!(routing, fx.tokens);
    let infos = vec![Some(BlockExtraInfo {
        mm_objects: vec![BlockMmObjectInfo {
            mm_hash,
            offsets: vec![(fx.item_offset, fx.item_end)],
        }],
    })];

    let engine = mm_engine_tokens(&routing, &infos, BLOCK, Some(fx.image_token_id), None)
        .expect("recoverable");
    assert_eq!(
        engine, fx.tokens,
        "engine tokens recovered from the routing sequence"
    );

    let units = mm_extra_keys_by_unit(
        &infos,
        BLOCK,
        UNIT as usize,
        fx.units,
        &hybrid_mm_identifier,
    );
    assert!(units[0].is_some(), "unit 0 carries the image key");
    assert!(units[1].is_none());

    let hasher = EngineChainHasher::new(EngineHashAlgo::Sha256Cbor, "0");
    let keys = hasher.chain_keys_with_extra(&engine, UNIT, &|i| units.get(i).cloned().flatten());
    assert_eq!(keys.len(), fx.units);
    assert_eq!(
        keys[fx.units - 1],
        fx.expected_last_unit_key,
        "engine's own key at 12,032"
    );
    // without the image key the chain diverges from unit 0 on
    let plain = hasher.chain_keys(&engine, UNIT);
    assert_ne!(plain[fx.units - 1], fx.expected_last_unit_key);
}
