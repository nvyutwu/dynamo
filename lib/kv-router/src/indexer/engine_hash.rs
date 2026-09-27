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

    /// [`Self::chain_digests`] with vLLM's per-block extra keys: `extra(i)` returns the keys of
    /// unit block `i` (tokens `[i*unit, (i+1)*unit)`), e.g. the `(identifier, offset_in_block)`
    /// pairs of the multimodal items overlapping it and the cache salt on block 0.
    pub fn chain_digests_with_extra(
        &self,
        tokens: &[u32],
        unit: u32,
        extra: &dyn Fn(usize) -> Option<Vec<ExtraKey>>,
    ) -> Vec<Vec<u8>> {
        let unit = unit as usize;
        if unit == 0 {
            return Vec::new();
        }
        let mut digests = Vec::with_capacity(tokens.len() / unit);
        let mut parent = self.none_hash.clone();
        let mut buf = Vec::with_capacity(unit * 5 + 48);
        for (i, block) in tokens.chunks_exact(unit).enumerate() {
            buf.clear();
            let keys = extra(i);
            encode_block_with_extra(&mut buf, &parent, block, keys.as_deref());
            parent = self.algo.digest(&buf);
            digests.push(parent.clone());
        }
        digests
    }

    /// [`Self::chain_keys`] with vLLM's per-block extra keys (see
    /// [`Self::chain_digests_with_extra`]).
    pub fn chain_keys_with_extra(
        &self,
        tokens: &[u32],
        unit: u32,
        extra: &dyn Fn(usize) -> Option<Vec<ExtraKey>>,
    ) -> Vec<u64> {
        self.chain_digests_with_extra(tokens, unit, extra)
            .iter()
            .map(|digest| wire_key(digest))
            .collect()
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

/// One element of vLLM's `extra_keys` tuple for a block (`generate_block_hash_extra_keys`):
/// a multimodal item as `(identifier, offset_in_block)` or a plain string (LoRA name, cache
/// salt on the first block).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExtraKey {
    Mm { identifier: String, offset: i64 },
    Text(String),
}

/// Canonical CBOR for the Python tuple `(parent: bytes, tuple(token_ids), extra_keys)` where
/// `extra_keys` is `None` or a tuple of `(str, int)` tuples and strings.
fn encode_block(buf: &mut Vec<u8>, parent: &[u8], tokens: &[u32]) {
    encode_block_with_extra(buf, parent, tokens, None);
}

fn encode_block_with_extra(
    buf: &mut Vec<u8>,
    parent: &[u8],
    tokens: &[u32],
    extra: Option<&[ExtraKey]>,
) {
    cbor_head(buf, 4, 3);
    cbor_bytes(buf, parent);
    cbor_head(buf, 4, tokens.len() as u64);
    for &token in tokens {
        cbor_head(buf, 0, u64::from(token));
    }
    match extra {
        None => buf.push(0xf6),
        Some(keys) if keys.is_empty() => buf.push(0xf6),
        Some(keys) => {
            cbor_head(buf, 4, keys.len() as u64);
            for key in keys {
                match key {
                    ExtraKey::Mm { identifier, offset } => {
                        cbor_head(buf, 4, 2);
                        cbor_text(buf, identifier);
                        cbor_int(buf, *offset);
                    }
                    ExtraKey::Text(text) => cbor_text(buf, text),
                }
            }
        }
    }
}

/// vLLM's per-unit-block multimodal extra keys, rebuilt from the frontend's block-level
/// multimodal info (`BlockExtraInfo`: per router block, each object's placeholder ranges clipped
/// to that block). A placeholder run that crosses a router-block boundary appears in both blocks
/// (`(s, block_size)` then `(0, e)`); consecutive clipped ranges of the same object are one
/// occurrence whose absolute start is the first range's. For unit block `i` the keys are, in
/// start order, `(identifier(mm_hash), start - i*unit)` for every occurrence overlapping the
/// unit — negative when the item began earlier — exactly `generate_block_hash_extra_keys`
/// (`_gen_mm_extra_hash_keys`). Returns one entry per unit block (`None` = no extra keys).
pub fn mm_extra_keys_by_unit(
    block_mm_infos: &[Option<crate::protocols::BlockExtraInfo>],
    block_size: usize,
    unit: usize,
    num_units: usize,
    identifier: &dyn Fn(u64) -> String,
) -> Vec<Option<Vec<ExtraKey>>> {
    // occurrences: (abs_start, abs_end, mm_hash)
    let mut occurrences: Vec<(usize, usize, u64)> = Vec::new();
    for (block_index, info) in block_mm_infos.iter().enumerate() {
        let Some(info) = info else { continue };
        let base = block_index * block_size;
        for object in &info.mm_objects {
            for &(s, e) in &object.offsets {
                let (abs_s, abs_e) = (base + s, base + e);
                // continuation of an occurrence that ended exactly at this block's start
                if s == 0
                    && let Some(last) = occurrences
                        .iter_mut()
                        .rev()
                        .find(|(_, end, hash)| *hash == object.mm_hash && *end == base)
                {
                    last.1 = abs_e;
                    continue;
                }
                occurrences.push((abs_s, abs_e, object.mm_hash));
            }
        }
    }
    occurrences.sort_by_key(|(s, _, _)| *s);
    let mut out = Vec::with_capacity(num_units);
    for i in 0..num_units {
        let (u_start, u_end) = (i * unit, (i + 1) * unit);
        let keys: Vec<ExtraKey> = occurrences
            .iter()
            .filter(|(s, e, _)| *s < u_end && *e > u_start)
            .map(|(s, _, hash)| ExtraKey::Mm {
                identifier: identifier(*hash),
                offset: *s as i64 - u_start as i64,
            })
            .collect();
        out.push((!keys.is_empty()).then_some(keys));
    }
    out
}

/// vLLM's multimodal item identifier (`MultiModalHasher.hash_kwargs("blake3", model_id=…,
/// image=item)`): kwargs are hashed in key order, each as its key then its serialization. An
/// image fetched by vLLM arrives as `MediaWithBytes` without `io_config`, whose serialization
/// re-emits the key before the original encoded bytes, so the stream is
/// `b"image" b"image" <bytes> b"model_id" <model_id>`. Raw `bytes` items (`media_with_bytes =
/// false`) emit the key once.
pub fn mm_identifier_blake3(model_id: &str, image_bytes: &[u8], media_with_bytes: bool) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"image");
    if media_with_bytes {
        hasher.update(b"image");
    }
    hasher.update(image_bytes);
    hasher.update(b"model_id");
    hasher.update(model_id.as_bytes());
    hasher.finalize().to_hex().to_string()
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

/// CBOR integer: major 0 for non-negative values, major 1 (`-1 - n`) for negative ones — vLLM's
/// in-block offset of a multimodal item is negative when the item started in an earlier block.
fn cbor_int(buf: &mut Vec<u8>, value: i64) {
    if value >= 0 {
        cbor_head(buf, 0, value as u64);
    } else {
        cbor_head(buf, 1, (-1 - value) as u64);
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
    /// Goldens from vLLM `hash_block_tokens(sha256_cbor, …)` with `extra_keys`, seed "0",
    /// tokens 1000..1127 (`kv_cache_utils.generate_block_hash_extra_keys` shapes).
    #[test]
    fn extra_keys_reproduce_vllm_block_hashes() {
        let hasher = EngineChainHasher::new(EngineHashAlgo::Sha256Cbor, "0");
        assert_eq!(
            hex(hasher.none_hash()),
            "4e1195df020de59e0d65a33a4279f1183e7ae4e5d980e309f8b55adff2e61c3e"
        );
        let tokens: Vec<u32> = (1000..1128).collect();
        let digest = |extra: Option<Vec<ExtraKey>>| {
            let d = hasher.chain_digests_with_extra(&tokens, 128, &|_| extra.clone());
            hex(&d[0])
        };
        assert_eq!(
            digest(None),
            "199e2f20e73a6e2cc3e0e6fdf521eff7b636122d5e122320c22c380d95e5863e"
        );
        let mm = ExtraKey::Mm {
            identifier: "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".into(),
            offset: 5,
        };
        assert_eq!(
            digest(Some(vec![mm])),
            "3597444c4db23e603150784eeb8467b6895e64d29667e2496941709e6bad8adb"
        );
        assert_eq!(
            digest(Some(vec![ExtraKey::Text("my-salt".into())])),
            "84611d254438937d076c5739d973f17e1febd6a252387f535357a817319908b0"
        );
        assert_eq!(
            digest(Some(vec![
                ExtraKey::Mm {
                    identifier: "aaaa".into(),
                    offset: 3
                },
                ExtraKey::Text("my-salt".into()),
            ])),
            "3f56ea40d074550a58b6ddadba236109d86356a3cc0aeca8a0c70c4be0df5cc0"
        );
        // second block parented on the plain first block, two items in the block
        let two: Vec<u32> = (1000..1128).chain(1000..1128).collect();
        let d = hasher.chain_digests_with_extra(&two, 128, &|i| {
            (i == 1).then(|| {
                vec![
                    ExtraKey::Mm {
                        identifier: "aaaa".into(),
                        offset: 0,
                    },
                    ExtraKey::Mm {
                        identifier: "bbbb".into(),
                        offset: 64,
                    },
                ]
            })
        });
        assert_eq!(
            hex(&d[1]),
            "2ea9f0275be48f4499ccdd95ca2d8cd26c5a995c2a9b08d7aa4ef2a5902569fc"
        );
        assert_eq!(wire_key(&d[1]), 12272012827533076988);
        // an item that began in an earlier block carries a negative in-block offset
        assert_eq!(
            digest(Some(vec![ExtraKey::Mm {
                identifier: "aaaa".into(),
                offset: -3
            }])),
            "c914451f83e397a290b4a1749465eb9a832fc78ef98f8e32dace9514242864f2"
        );
    }

    /// vLLM golden (`generate_block_hash_extra_keys`, this session): items A [100,300), B [300,330),
    /// C [450,500) in a 512-token prompt, unit 128 → unit0 (A,100); unit1 (A,-28); unit2 (A,-156),(B,44);
    /// unit3 (C,66). The frontend's block info uses router block 256 here so A is clipped across
    /// two blocks and B/C share block 1.
    #[test]
    fn mm_extra_keys_by_unit_match_vllm_generate_block_hash_extra_keys() {
        use crate::protocols::{BlockExtraInfo, BlockMmObjectInfo};
        let block = |objects: Vec<(u64, Vec<(usize, usize)>)>| {
            Some(BlockExtraInfo {
                mm_objects: objects
                    .into_iter()
                    .map(|(mm_hash, offsets)| BlockMmObjectInfo { mm_hash, offsets })
                    .collect(),
            })
        };
        let infos = vec![
            block(vec![(0xA, vec![(100, 256)])]),
            block(vec![
                (0xA, vec![(0, 44)]),
                (0xB, vec![(44, 74)]),
                (0xC, vec![(194, 244)]),
            ]),
        ];
        let ident = |h: u64| format!("{h:x}");
        let keys = mm_extra_keys_by_unit(&infos, 256, 128, 4, &ident);
        let mm = |id: &str, offset: i64| ExtraKey::Mm {
            identifier: id.into(),
            offset,
        };
        assert_eq!(keys[0], Some(vec![mm("a", 100)]));
        assert_eq!(keys[1], Some(vec![mm("a", -28)]));
        assert_eq!(keys[2], Some(vec![mm("a", -156), mm("b", 44)]));
        assert_eq!(keys[3], Some(vec![mm("c", 66)]));
        // a text-only block yields None, and the same image twice is two occurrences
        let infos = vec![None, block(vec![(0xA, vec![(0, 10), (20, 30)])])];
        let keys = mm_extra_keys_by_unit(&infos, 128, 128, 2, &ident);
        assert_eq!(keys[0], None);
        assert_eq!(keys[1], Some(vec![mm("a", 0), mm("a", 20)]));
    }

    /// Goldens from vLLM `MultiModalHasher.hash_kwargs("blake3", model_id="model-x", image=…)`
    /// for raw bytes and for `MediaWithBytes(PIL image, bytes)` without io_config.
    #[test]
    fn mm_identifier_matches_vllm_multimodal_hasher() {
        let bytes = b"\x89PNG\r\n\x1a\nfakeimagebytes";
        assert_eq!(
            mm_identifier_blake3("model-x", bytes, false),
            "64c6d9eba7d0afd03255306a918077dd7dcec4e7b74c3845378453c26874901c"
        );
        assert_eq!(
            mm_identifier_blake3("model-x", bytes, true),
            "3c33a6df32c3c9be34ca134b438c73647e304ebd030efb566520d685c6eb4880"
        );
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
