// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frontend reproduction of vLLM's prefix-chain block hash.
//!
//! vLLM (`kv_cache_utils.hash_block_tokens`) hashes every `prefix_match_unit`-token block as
//! `hash_fn((parent_digest, tuple(token_ids), extra_keys))`, seeding the chain with
//! `NONE_HASH = hash_fn(seed)`. The `sha256_cbor` and `xxhash_cbor` algorithms serialize that
//! tuple with canonical CBOR, which is language independent; this module re-implements the
//! encoder for exactly the value shapes involved (byte string, array of unsigned integers,
//! null, text) and applies the same digest. KV events publish the low 64 bits of each digest
//! (`int.from_bytes(digest, "big") & (2**64 - 1)`), which is what [`EngineChainHasher::chain_keys`]
//! returns so the result can be compared directly with published keys.
//!
//! Requests that carry extra keys (LoRA adapter, cache salt, multimodal hashes) are not
//! reproduced here; callers must fall back to the local-hash index for them.
//!
//! The default `sha256` (pickle) algorithm is not reproducible outside CPython and is not
//! supported: deployments that enable the engine-hash probe must start vLLM with
//! `--prefix-caching-hash-algo sha256_cbor` (or `xxhash_cbor` with a shared `PYTHONHASHSEED`).

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Seed vLLM derives `NONE_HASH` from when `PYTHONHASHSEED` is unset and the algorithm is
/// cryptographic (`kv_cache_utils.DEFAULT_NONE_HASH_SEED`).
pub const DEFAULT_NONE_HASH_SEED: &str = "vllm-none-hash";

/// Engine prefix-cache hash algorithms the frontend can reproduce.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EngineHashAlgo {
    /// `--prefix-caching-hash-algo sha256_cbor`: SHA-256 over canonical CBOR (32-byte digests).
    #[default]
    Sha256Cbor,
    /// `--prefix-caching-hash-algo xxhash_cbor`: XXH3-128 over canonical CBOR (16-byte digests).
    /// vLLM randomizes the seed per process unless `PYTHONHASHSEED` is set.
    XxhashCbor,
}

impl EngineHashAlgo {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Sha256Cbor => "sha256_cbor",
            Self::XxhashCbor => "xxhash_cbor",
        }
    }

    fn digest(self, input: &[u8]) -> Vec<u8> {
        match self {
            Self::Sha256Cbor => Sha256::digest(input).to_vec(),
            Self::XxhashCbor => xxhash_rust::xxh3::xxh3_128(input).to_be_bytes().to_vec(),
        }
    }
}

impl fmt::Display for EngineHashAlgo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for EngineHashAlgo {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "sha256_cbor" => Ok(Self::Sha256Cbor),
            "xxhash_cbor" => Ok(Self::XxhashCbor),
            other => Err(format!(
                "unsupported engine hash algorithm '{other}'; expected sha256_cbor or xxhash_cbor \
                 (the pickle-based sha256/xxhash algorithms cannot be reproduced outside CPython)"
            )),
        }
    }
}

/// Computes the engine's prefix-chain hashes for a request's token ids.
#[derive(Debug, Clone)]
pub struct EngineChainHasher {
    algo: EngineHashAlgo,
    none_hash: Vec<u8>,
}

impl EngineChainHasher {
    /// `seed` is the string vLLM hashed into `NONE_HASH`: `PYTHONHASHSEED` when set on the
    /// engine, else [`DEFAULT_NONE_HASH_SEED`] for the cryptographic algorithms.
    pub fn new(algo: EngineHashAlgo, seed: &str) -> Self {
        let mut buf = Vec::with_capacity(seed.len() + 9);
        cbor_text(&mut buf, seed);
        Self {
            algo,
            none_hash: algo.digest(&buf),
        }
    }

    pub fn algo(&self) -> EngineHashAlgo {
        self.algo
    }

    /// The digest every chain starts from (`kv_cache_utils.NONE_HASH`).
    pub fn none_hash(&self) -> &[u8] {
        &self.none_hash
    }

    /// Full digests for every complete `unit`-token block of `tokens`, in prefix order.
    pub fn chain_digests(&self, tokens: &[u32], unit: u32) -> Vec<Vec<u8>> {
        let unit = unit as usize;
        if unit == 0 {
            return Vec::new();
        }
        let mut digests = Vec::with_capacity(tokens.len() / unit);
        let mut parent = self.none_hash.clone();
        let mut buf = Vec::with_capacity(unit * 5 + 48);
        for block in tokens.chunks_exact(unit) {
            buf.clear();
            encode_block(&mut buf, &parent, block);
            parent = self.algo.digest(&buf);
            digests.push(parent.clone());
        }
        digests
    }

    /// Wire keys (low 64 bits of each digest) for every complete `unit`-token boundary of
    /// `tokens`, in prefix order: element `i` names the prefix `[0, (i + 1) * unit)`.
    pub fn chain_keys(&self, tokens: &[u32], unit: u32) -> Vec<u64> {
        let unit = unit as usize;
        if unit == 0 {
            return Vec::new();
        }
        let mut keys = Vec::with_capacity(tokens.len() / unit);
        let mut parent = self.none_hash.clone();
        let mut buf = Vec::with_capacity(unit * 5 + 48);
        for block in tokens.chunks_exact(unit) {
            buf.clear();
            encode_block(&mut buf, &parent, block);
            parent = self.algo.digest(&buf);
            keys.push(wire_key(&parent));
        }
        keys
    }
}

/// The integer vLLM publishes for a digest: `int.from_bytes(digest, "big") & (2**64 - 1)`.
pub fn wire_key(digest: &[u8]) -> u64 {
    let take = digest.len().min(8);
    digest[digest.len() - take..]
        .iter()
        .fold(0u64, |acc, &byte| (acc << 8) | u64::from(byte))
}

/// Canonical CBOR for the Python tuple `(parent: bytes, tuple(token_ids), None)`.
fn encode_block(buf: &mut Vec<u8>, parent: &[u8], tokens: &[u32]) {
    cbor_head(buf, 4, 3);
    cbor_bytes(buf, parent);
    cbor_head(buf, 4, tokens.len() as u64);
    for &token in tokens {
        cbor_head(buf, 0, u64::from(token));
    }
    buf.push(0xf6);
}

fn cbor_head(buf: &mut Vec<u8>, major: u8, value: u64) {
    let mt = major << 5;
    if value < 24 {
        buf.push(mt | value as u8);
    } else if value <= 0xff {
        buf.push(mt | 24);
        buf.push(value as u8);
    } else if value <= 0xffff {
        buf.push(mt | 25);
        buf.extend_from_slice(&(value as u16).to_be_bytes());
    } else if value <= 0xffff_ffff {
        buf.push(mt | 26);
        buf.extend_from_slice(&(value as u32).to_be_bytes());
    } else {
        buf.push(mt | 27);
        buf.extend_from_slice(&value.to_be_bytes());
    }
}

fn cbor_bytes(buf: &mut Vec<u8>, bytes: &[u8]) {
    cbor_head(buf, 2, bytes.len() as u64);
    buf.extend_from_slice(bytes);
}

fn cbor_text(buf: &mut Vec<u8>, text: &str) {
    cbor_head(buf, 3, text.len() as u64);
    buf.extend_from_slice(text.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    /// Token ids spanning the one-, two- and four-byte CBOR integer encodings.
    fn fixture_tokens() -> Vec<u32> {
        (0..300u32).map(|i| (i * 7919 + 13) % 152_064).collect()
    }

    /// Golden values produced by vLLM 29bf597539 itself:
    /// `init_none_hash(sha256_cbor)`, `hash_block_tokens(sha256_cbor, prev, tokens[s:s+128], None)`
    /// and `maybe_convert_block_hash` over `fixture_tokens()`.
    #[test]
    fn sha256_cbor_matches_vllm_none_hash_and_chain() {
        let hasher = EngineChainHasher::new(EngineHashAlgo::Sha256Cbor, DEFAULT_NONE_HASH_SEED);
        assert_eq!(
            hex(hasher.none_hash()),
            "9bd96a485ad84efdafb72ee48a1d7a69bcead0f8f0433173941b276b9581eef0"
        );

        let tokens = fixture_tokens();
        assert_eq!(&tokens[..4], &[13, 7932, 15851, 23770]);
        let digests = hasher.chain_digests(&tokens, 128);
        assert_eq!(
            digests.len(),
            2,
            "only complete 128-token blocks are hashed"
        );
        assert_eq!(
            hex(&digests[0]),
            "3c922374e6fe67f3c64e75b88a3b8bd6185b538f7019674b72aefbf8dd8754f8"
        );
        assert_eq!(
            hex(&digests[1]),
            "2e69fa8c8aebe362593b51b8b212f1aeb45f9bb56c4258c0bf2a94f1e384e2a5"
        );
        assert_eq!(
            hasher.chain_keys(&tokens, 128),
            vec![8_263_819_412_558_533_880, 13_774_986_176_891_249_317]
        );
    }

    #[test]
    fn chain_is_a_prefix_function() {
        let hasher = EngineChainHasher::new(EngineHashAlgo::Sha256Cbor, DEFAULT_NONE_HASH_SEED);
        let tokens = fixture_tokens();
        let full = hasher.chain_keys(&tokens, 128);
        let shorter = hasher.chain_keys(&tokens[..256], 128);
        assert_eq!(shorter, full[..2]);
        let mut diverged = tokens.clone();
        diverged[200] ^= 1;
        let other = hasher.chain_keys(&diverged, 128);
        assert_eq!(
            other[0], full[0],
            "blocks before the change keep their hash"
        );
        assert_ne!(
            other[1], full[1],
            "the block holding the change and everything after differ"
        );
    }

    #[test]
    fn xxhash_cbor_uses_xxh3_128_big_endian() {
        // xxhash.xxh3_128_digest(b"abc").hex() == "06b05ab6733a618578af5f94892f3950"
        assert_eq!(
            hex(&EngineHashAlgo::XxhashCbor.digest(b"abc")),
            "06b05ab6733a618578af5f94892f3950"
        );
        let hasher = EngineChainHasher::new(EngineHashAlgo::XxhashCbor, "seed");
        assert_eq!(hasher.none_hash().len(), 16);
    }

    #[test]
    fn wire_key_takes_low_64_bits_big_endian() {
        let digest: Vec<u8> = (1..=32u8).collect();
        assert_eq!(wire_key(&digest), 0x191a_1b1c_1d1e_1f20);
        assert_eq!(wire_key(&[0x01, 0x02]), 0x0102);
    }

    #[test]
    fn cbor_integer_widths_are_minimal() {
        let mut buf = Vec::new();
        cbor_head(&mut buf, 0, 23);
        cbor_head(&mut buf, 0, 24);
        cbor_head(&mut buf, 0, 256);
        cbor_head(&mut buf, 0, 65_536);
        assert_eq!(
            buf,
            vec![
                0x17, 0x18, 0x18, 0x19, 0x01, 0x00, 0x1a, 0x00, 0x01, 0x00, 0x00
            ]
        );
    }

    #[test]
    fn algo_round_trips_through_str() {
        assert_eq!(
            "sha256_cbor".parse::<EngineHashAlgo>().unwrap(),
            EngineHashAlgo::Sha256Cbor
        );
        assert_eq!(
            "XXHASH_CBOR".parse::<EngineHashAlgo>().unwrap(),
            EngineHashAlgo::XxhashCbor
        );
        assert!("sha256".parse::<EngineHashAlgo>().is_err());
        assert_eq!(EngineHashAlgo::Sha256Cbor.to_string(), "sha256_cbor");
    }
}
