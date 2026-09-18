// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Isolated native-cpu-root v1 codec. Never call the legacy raw-event decoder
//! on this topic or publish these maps on the legacy socket.
//!
//! The engine publishes a `LogicalSnapshot` baseline followed by contiguous
//! `LogicalUpdate` cuts. A snapshot re-bases this consumer unconditionally; an
//! update is applied only when it continues the accepted cut in the same epoch,
//! so a lost message leaves the view unknown until the next authoritative
//! snapshot rather than producing a half-restored index.

use crate::indexer::logical::{
    Binding, Confidence, Contribution, CoverageRole, Edge, Limits, LogicalView, Mutation, Plan,
    Profile, Publisher, Query, QueryResult, Scope, Snapshot,
};
use crate::protocols::{
    BlockHashOptions, ExternalSequenceBlockHash, ResidencyOwner, compute_block_hash_for_seq,
};
use serde::{Deserialize, Deserializer, Serialize, Serializer, de};
use std::collections::{HashMap, HashSet, VecDeque};

pub const LOGICAL_TOPIC: &str = "logical-cache-v1";

#[derive(Debug, Clone, PartialEq, Eq)]
struct NativeHash(Vec<u8>);

impl Serialize for NativeHash {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_bytes(&self.0)
    }
}
impl<'de> Deserialize<'de> for NativeHash {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Bytes;
        impl<'de> de::Visitor<'de> for Bytes {
            type Value = NativeHash;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("nonempty native prefix hash bytes (at most 128 bytes)")
            }
            fn visit_bytes<E: de::Error>(self, bytes: &[u8]) -> Result<Self::Value, E> {
                if bytes.is_empty() || bytes.len() > 128 {
                    return Err(E::custom("invalid native hash size"));
                }
                Ok(NativeHash(bytes.to_vec()))
            }
        }
        deserializer.deserialize_bytes(Bytes)
    }
}

/// One strict schema for both cut shapes.
///
/// A flattened header plus per-shape structs would read better, but serde
/// silently drops `deny_unknown_fields` on any struct containing `flatten`, and
/// an unrecognised producer field must never be accepted as understood. Exactly
/// one shape's keys are present; the other's are absent rather than empty, so a
/// truncated snapshot cannot be read as an update that changed nothing.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireCut {
    #[serde(rename = "type")]
    event_type: String,
    profile: String,
    version: u32,
    namespace: String,
    epoch: String,
    cursor: u64,
    confidence: String,
    reason: String,
    block_size: u32,
    hash_unit: u32,
    max_rows: u64,
    max_plans: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    segments: Option<Vec<WireSegment>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    plans: Option<Vec<WirePlan>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    added_segments: Option<Vec<WireSegment>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    removed_segments: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    added_plans: Option<Vec<WirePlan>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    removed_plans: Option<Vec<String>>,
}

/// Validated scope/geometry fields, shared by both shapes so those checks
/// cannot diverge between the two messages.
#[derive(Debug, Clone)]
struct WireHeader {
    epoch: String,
    cursor: u64,
    confidence: String,
    reason: String,
    block_size: u32,
    hash_unit: u32,
    max_rows: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireSegment {
    id: String,
    kind: String,
    start: u64,
    end: u64,
    parent_hash: Option<NativeHash>,
    end_hash: NativeHash,
    token_ids: Vec<u32>,
    hash_unit: u32,
    hashes: Vec<NativeHash>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WirePlan {
    id: String,
    kind: String,
    end: u64,
    anchor: u64,
    anchor_hash: Option<NativeHash>,
    terminal_hash: NativeHash,
    minimum_prompt_tokens: u64,
    binding: String,
}

#[derive(Debug, Clone, Copy)]
pub struct DecodeLimits {
    pub bytes: usize,
    pub state: Limits,
    pub epochs: usize,
}
impl Default for DecodeLimits {
    fn default() -> Self {
        Self {
            bytes: 16 * 1024 * 1024,
            state: Limits::default(),
            epochs: 256,
        }
    }
}

impl DecodeLimits {
    /// Size the ceilings for an engine that advertises `max_rows` at this
    /// geometry, instead of guessing.
    ///
    /// The defaults are deliberately conservative and are reached by POOL SIZE
    /// ALONE, not by traffic — an engine whose pool exceeds them has every cut
    /// refused, permanently, for that worker. The plan ceiling binds first and
    /// is easy to miss: `max_plans` is `max_rows * (B/h + 1)`, so at the K3
    /// geometry (B=12288, h=128) the multiplier is 97 and the default
    /// `plans: 65_536` caps the pool at 675 rows, well under `bindings: 4096`.
    /// The payload ceiling can bind earlier still, because a snapshot carries
    /// `token_ids` for every live row (B tokens each).
    ///
    /// Callers must still size `bytes` against a measured snapshot; this helper
    /// only makes the state ceilings consistent with the advertised pool.
    pub fn for_pool(max_rows: u64, block_size: u32, hash_unit: u32) -> Option<Self> {
        if hash_unit == 0 || block_size <= hash_unit || !block_size.is_multiple_of(hash_unit) {
            return None;
        }
        let per_row = u64::from(block_size / hash_unit) + 1;
        let plans = max_rows.checked_mul(per_row)?;
        let bindings: usize = max_rows.try_into().ok()?;
        Some(Self {
            state: Limits {
                bindings,
                plans: plans.try_into().ok()?,
                edges: bindings.checked_mul((block_size / hash_unit) as usize)?,
            },
            ..Self::default()
        })
    }
}

#[derive(Debug, thiserror::Error)]
#[error("unsupported or invalid logical CPU cut: {0}")]
pub struct LogicalDecodeError(String);

fn invalid(message: &str) -> LogicalDecodeError {
    LogicalDecodeError(message.into())
}
fn position(value: u64) -> Result<u32, LogicalDecodeError> {
    value
        .try_into()
        .map_err(|_| invalid("token position exceeds u32"))
}
fn role(value: &str) -> Result<CoverageRole, LogicalDecodeError> {
    match value {
        "complete" => Ok(CoverageRole::Prefix),
        "tail" => Ok(CoverageRole::Tail),
        _ => Err(invalid("unknown segment/plan kind")),
    }
}

/// Preserve full native-byte identity at this boundary. The continuation index
/// uses compact hashes, so detect collisions before admitting any materialized
/// state. Native identity hashes are NEVER used as token-local hashes.
///
/// Interning is a pure function of the bytes, so identities stay stable across
/// the cuts of one epoch; the retained table only exists to detect a collision
/// between two distinct native hashes.
#[derive(Default)]
struct Identities(HashMap<u64, Vec<u8>>);
impl Identities {
    fn intern(&mut self, bytes: &[u8]) -> Result<u64, LogicalDecodeError> {
        if bytes.is_empty() || bytes.len() > 1024 {
            return Err(invalid("empty/oversize identity"));
        }
        let id = xxhash_rust::xxh3::xxh3_64(bytes);
        if let Some(old) = self.0.insert(id, bytes.to_vec())
            && old != bytes
        {
            return Err(invalid("identity collision"));
        }
        Ok(id)
    }
    fn hash(&mut self, hash: &NativeHash) -> Result<ExternalSequenceBlockHash, LogicalDecodeError> {
        self.intern(&hash.0).map(ExternalSequenceBlockHash)
    }
    fn optional(
        &mut self,
        hash: &Option<NativeHash>,
    ) -> Result<Option<ExternalSequenceBlockHash>, LogicalDecodeError> {
        hash.as_ref().map(|h| self.hash(h)).transpose()
    }
    fn len(&self) -> usize {
        self.0.len()
    }
}

/// What a decoded cut asks the consumer to do.
enum Cut {
    /// Authoritative replacement: accepted whatever the consumer already holds.
    Snapshot {
        header: WireHeader,
        segments: Vec<WireSegment>,
        plans: Vec<WirePlan>,
    },
    /// Continuation: valid only against the immediately preceding accepted cut.
    Update {
        header: WireHeader,
        added_segments: Vec<WireSegment>,
        removed_segments: Vec<String>,
        added_plans: Vec<WirePlan>,
        removed_plans: Vec<String>,
    },
}

impl Cut {
    fn header(&self) -> &WireHeader {
        match self {
            Cut::Snapshot { header, .. } | Cut::Update { header, .. } => header,
        }
    }
}

fn decode_cut(
    payload: &[u8],
    owner: ResidencyOwner,
    namespace: &str,
    limits: DecodeLimits,
) -> Result<Cut, LogicalDecodeError> {
    if payload.len() > limits.bytes {
        return Err(invalid("payload budget exceeded"));
    }
    let (timestamp, mut cuts, dp_rank): (f64, Vec<WireCut>, i32) =
        rmp_serde::from_slice(payload).map_err(|e| LogicalDecodeError(e.to_string()))?;
    if !timestamp.is_finite() || cuts.len() != 1 || dp_rank < 0 {
        return Err(invalid("invalid logical envelope"));
    }
    // v1 binds a CPU-root view to exactly one engine worker rank. A cache owner
    // has no DP rank to check, so it is out of scope rather than unchecked.
    let ResidencyOwner::Worker(worker) = owner else {
        return Err(invalid("cpu-root profile supports worker owners only"));
    };
    if worker.dp_rank != dp_rank as u32 {
        return Err(invalid("endpoint DP rank mismatch"));
    }

    let wire = cuts.pop().expect("exactly one cut");
    let header = check_header(&wire, namespace, limits)?;
    let snapshot_shape = wire.segments.is_some() || wire.plans.is_some();
    let update_shape = wire.added_segments.is_some()
        || wire.removed_segments.is_some()
        || wire.added_plans.is_some()
        || wire.removed_plans.is_some();
    match wire.event_type.as_str() {
        "LogicalSnapshot" => {
            let (Some(segments), Some(plans), false) = (wire.segments, wire.plans, update_shape)
            else {
                return Err(invalid("snapshot must carry exactly its own fields"));
            };
            if segments.len() as u64 > header.max_rows {
                return Err(invalid("state budget exceeded"));
            }
            Ok(Cut::Snapshot {
                header,
                segments,
                plans,
            })
        }
        "LogicalUpdate" => {
            let (
                Some(added_segments),
                Some(removed_segments),
                Some(added_plans),
                Some(removed_plans),
                false,
            ) = (
                wire.added_segments,
                wire.removed_segments,
                wire.added_plans,
                wire.removed_plans,
                snapshot_shape,
            )
            else {
                return Err(invalid("update must carry exactly its own fields"));
            };
            Ok(Cut::Update {
                header,
                added_segments,
                removed_segments,
                added_plans,
                removed_plans,
            })
        }
        _ => Err(invalid("unsupported logical event type")),
    }
}

fn check_header(
    wire: &WireCut,
    namespace: &str,
    limits: DecodeLimits,
) -> Result<WireHeader, LogicalDecodeError> {
    if wire.profile != "native-cpu-root" || wire.version != 1 {
        return Err(invalid("unsupported profile/version"));
    }
    if wire.namespace != namespace || namespace.is_empty() || namespace.len() > 1024 {
        return Err(invalid("namespace mismatch"));
    }
    if wire.hash_unit == 0
        || wire.block_size <= wire.hash_unit
        || !wire.block_size.is_multiple_of(wire.hash_unit)
    {
        return Err(invalid("invalid geometry"));
    }
    let expected_plans = wire
        .max_rows
        .checked_mul(u64::from(wire.block_size / wire.hash_unit) + 1)
        .ok_or_else(|| invalid("capacity overflow"))?;
    if wire.max_plans != expected_plans {
        return Err(invalid(
            "max_plans is not the declared capacity-derived ceiling",
        ));
    }
    // `max_rows` is the engine's whole CPU pool capacity, so a production-sized
    // pool can exceed a conservative consumer default and be refused here --
    // permanently, for that worker. Name both numbers so an operator can size
    // `DecodeLimits`/`RegistryLimits` instead of guessing at "budget exceeded".
    if wire.max_rows > limits.state.bindings as u64 {
        return Err(LogicalDecodeError(format!(
            "engine advertises max_rows={} but this consumer admits at most {} bindings; \
             raise Limits::bindings to at least the engine's CPU pool capacity",
            wire.max_rows, limits.state.bindings,
        )));
    }
    if wire.max_plans > limits.state.plans as u64 {
        return Err(LogicalDecodeError(format!(
            "engine advertises max_plans={} but this consumer admits at most {}; \
             raise Limits::plans alongside Limits::bindings",
            wire.max_plans, limits.state.plans,
        )));
    }
    if wire.epoch.len() != 32 || !wire.epoch.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(invalid("epoch must be UUID hex"));
    }
    if wire
        .plans
        .as_ref()
        .is_some_and(|plans| plans.len() as u64 > wire.max_plans)
        || wire
            .added_plans
            .as_ref()
            .is_some_and(|plans| plans.len() as u64 > wire.max_plans)
    {
        return Err(invalid("state budget exceeded"));
    }
    Ok(WireHeader {
        epoch: wire.epoch.clone(),
        cursor: wire.cursor,
        confidence: wire.confidence.clone(),
        reason: wire.reason.clone(),
        block_size: wire.block_size,
        hash_unit: wire.hash_unit,
        max_rows: wire.max_rows,
    })
}

fn epoch_bytes(header: &WireHeader) -> Result<[u8; 16], LogicalDecodeError> {
    Ok(u128::from_str_radix(&header.epoch, 16)
        .map_err(|_| invalid("invalid epoch"))?
        .to_be_bytes())
}

/// Per-epoch normalizing state. Native identities and declared segment names
/// live here because an update names bindings introduced by an earlier cut.
struct Normalizer {
    profile: Profile,
    limits: DecodeLimits,
    publisher: Publisher,
    view: LogicalView,
    hashes: Identities,
    ids: Identities,
    segments: HashMap<String, (Binding, NativeHash)>,
    plan_names: HashMap<u64, String>,
    /// Declared plan name -> (interned identity, declared binding name). The
    /// binding name is retained so retiring a row also retires its plan names,
    /// which a later cut may then legitimately reuse.
    plan_bindings: HashMap<String, (u64, String)>,
    /// Reverse of the above: declared binding name -> its plan names. Retiring a
    /// row is then a lookup rather than a scan of the whole plan table.
    plans_by_segment: HashMap<String, HashSet<String>>,
    applied: u64,
}

impl Normalizer {
    fn new(profile: Profile, scope: Scope, limits: DecodeLimits) -> Self {
        let publisher = Publisher::new(profile, scope, limits.state);
        let view = LogicalView::from_snapshot(publisher.snapshot(), limits.state);
        Self {
            profile,
            limits,
            publisher,
            view,
            hashes: Identities::default(),
            ids: Identities::default(),
            segments: HashMap::new(),
            plan_names: HashMap::new(),
            plan_bindings: HashMap::new(),
            plans_by_segment: HashMap::new(),
            applied: 0,
        }
    }

    fn segment(&mut self, segment: &WireSegment) -> Result<Mutation, LogicalDecodeError> {
        let namespace = self.view.scope().namespace.clone();
        let kind = role(&segment.kind)?;
        let start = position(segment.start)?;
        let end = position(segment.end)?;
        let unit = if kind == CoverageRole::Prefix {
            self.profile.block_size
        } else {
            self.profile.hash_unit
        };
        if start >= end
            || segment.hash_unit != unit
            || !start.is_multiple_of(self.profile.block_size)
            || segment.token_ids.len() != (end - start) as usize
            || segment.hashes.len() != ((end - start) / unit) as usize
            || !((end - start).is_multiple_of(unit))
            || segment.hashes.last() != Some(&segment.end_hash)
        {
            return Err(invalid("malformed segment geometry/material"));
        }
        let binding = Binding {
            id: self.ids.intern(segment.id.as_bytes())?,
            role: kind,
            start,
            end,
            parent: self.hashes.optional(&segment.parent_hash)?,
            child: self.hashes.hash(&segment.end_hash)?,
        };
        let locals = compute_block_hash_for_seq(
            &segment.token_ids,
            unit,
            BlockHashOptions {
                cache_namespace: Some(&namespace),
                ..Default::default()
            },
        );
        let mut parent = binding.parent;
        let mut edges = Vec::with_capacity(locals.len());
        for (native, local) in segment.hashes.iter().zip(locals) {
            let child = self.hashes.hash(native)?;
            edges.push(Edge {
                role: kind,
                parent,
                local,
                child,
            });
            parent = Some(child);
        }
        if self
            .segments
            .insert(
                segment.id.clone(),
                (binding.clone(), segment.end_hash.clone()),
            )
            .is_some()
        {
            return Err(invalid("duplicate segment ID"));
        }
        Ok(Mutation::Coverage(Contribution { binding, edges }))
    }

    fn plan(&mut self, plan: &WirePlan) -> Result<Mutation, LogicalDecodeError> {
        let Some((binding, end_hash)) = self.segments.get(&plan.binding).cloned() else {
            return Err(invalid("plan has missing binding"));
        };
        if role(&plan.kind)? != binding.role
            || (binding.role == CoverageRole::Tail && plan.terminal_hash != end_hash)
        {
            return Err(invalid("plan/binding mismatch"));
        }
        let id = self.ids.intern(plan.id.as_bytes())?;
        if self.plan_names.insert(id, plan.id.clone()).is_some() {
            return Err(invalid("duplicate plan ID"));
        }
        self.plan_bindings
            .insert(plan.id.clone(), (id, plan.binding.clone()));
        self.plans_by_segment
            .entry(plan.binding.clone())
            .or_default()
            .insert(plan.id.clone());
        // Validate the terminal identity as bytes even for an in-segment plan;
        // no fabricated fine edges are needed to match its whole anchor.
        self.hashes.hash(&plan.terminal_hash)?;
        Ok(Mutation::Plan(Plan {
            id,
            binding: binding.id,
            end: position(plan.end)?,
            anchor_end: position(plan.anchor)?,
            anchor: self.hashes.optional(&plan.anchor_hash)?,
            minimum_prompt_tokens: position(plan.minimum_prompt_tokens)?,
        }))
    }

    fn drop_segment(&mut self, name: &str) -> Option<Mutation> {
        let (binding, _) = self.segments.remove(name)?;
        // The model retires a binding's plans with it. Drop their names here
        // too, so a later cut may legitimately reuse them without colliding.
        // Indexed, not scanned: a cut retiring many rows against a plan table at
        // its ceiling would otherwise be quadratic in string comparisons.
        if let Some(plans) = self.plans_by_segment.remove(name) {
            for plan in plans {
                if let Some((id, _)) = self.plan_bindings.remove(&plan) {
                    self.plan_names.remove(&id);
                }
            }
        }
        Some(Mutation::RemoveCoverage(binding.id))
    }

    fn drop_plan(&mut self, name: &str) -> Option<Mutation> {
        let (id, segment) = self.plan_bindings.remove(name)?;
        if let Some(plans) = self.plans_by_segment.get_mut(&segment) {
            plans.remove(name);
            if plans.is_empty() {
                self.plans_by_segment.remove(&segment);
            }
        }
        self.plan_names.remove(&id);
        Some(Mutation::RemovePlan(id))
    }

    /// Guard the retained normalizing tables, which outlive a single cut and so
    /// are not covered by the per-message payload ceiling.
    fn within_limits(&self) -> bool {
        self.segments.len() <= self.limits.state.bindings
            && self.plan_names.len() <= self.limits.state.plans
            // The reverse index is keyed by segment and its values partition the
            // plan names, so both are already covered by the two ceilings above;
            // check it anyway so a future divergence is caught here, not in RSS.
            && self.plans_by_segment.len() <= self.limits.state.bindings
            && self.plan_bindings.len() <= self.limits.state.plans
            && self.hashes.len() <= self.limits.state.edges.saturating_mul(2)
            && self.ids.len()
                <= self
                    .limits
                    .state
                    .bindings
                    .saturating_add(self.limits.state.plans)
    }

    fn commit(&mut self, mutations: Vec<Mutation>) -> Result<(), LogicalDecodeError> {
        self.applied += 1;
        let batch = self.publisher.apply(self.applied, mutations);
        self.view.apply(batch);
        if !self.within_limits() {
            return Err(invalid("retained normalizer state budget exceeded"));
        }
        if self.view.confidence() != Confidence::Known {
            return Err(invalid("inconsistent contribution/plan state"));
        }
        Ok(())
    }
}

pub struct NormalizedSnapshot {
    pub snapshot: Snapshot,
    /// Diagnostics retain the engine's declared plan identity, not its numeric
    /// interning key. This table is cut-scoped and bounded by max_plans.
    pub plan_names: HashMap<u64, String>,
}

/// Decode one self-sufficient snapshot. Kept public for fixture-level checks
/// and for consumers that only ever adopt an authoritative replacement.
pub fn decode_snapshot(
    payload: &[u8],
    owner: ResidencyOwner,
    namespace: &str,
    limits: DecodeLimits,
) -> Result<NormalizedSnapshot, LogicalDecodeError> {
    let cut = decode_cut(payload, owner, namespace, limits)?;
    let cursor = cut.header().cursor;
    let Cut::Snapshot {
        header,
        segments,
        plans,
    } = cut
    else {
        return Err(invalid("expected an authoritative snapshot"));
    };
    let mut normalizer = build_snapshot(&header, segments, plans, owner, namespace, limits)?;
    let mut snapshot = normalizer.view.snapshot();
    snapshot.cursor = cursor;
    Ok(NormalizedSnapshot {
        snapshot,
        plan_names: std::mem::take(&mut normalizer.plan_names),
    })
}

fn build_snapshot(
    header: &WireHeader,
    segments: Vec<WireSegment>,
    plans: Vec<WirePlan>,
    owner: ResidencyOwner,
    namespace: &str,
    limits: DecodeLimits,
) -> Result<Normalizer, LogicalDecodeError> {
    let profile = Profile {
        version: 1,
        block_size: header.block_size,
        hash_unit: header.hash_unit,
    };
    let scope = Scope {
        owner,
        namespace: namespace.into(),
        incarnation: epoch_bytes(header)?,
        generation: 0,
    };
    let mut normalizer = Normalizer::new(profile, scope, limits);
    if header.confidence != "known" {
        if header.confidence != "unknown"
            || header.reason.is_empty()
            || !segments.is_empty()
            || !plans.is_empty()
        {
            return Err(invalid("invalid confidence state"));
        }
        // An explicit loss must never be readable as an authoritative empty cache.
        normalizer.view = LogicalView::from_snapshot(
            Snapshot {
                confidence: Confidence::Gap,
                ..normalizer.publisher.snapshot()
            },
            limits.state,
        );
        return Ok(normalizer);
    }
    if !header.reason.is_empty() {
        return Err(invalid("known cut carries a failure reason"));
    }
    let mut mutations = Vec::with_capacity(segments.len() + plans.len());
    for segment in &segments {
        mutations.push(normalizer.segment(segment)?);
    }
    for plan in &plans {
        mutations.push(normalizer.plan(plan)?);
    }
    normalizer.commit(mutations)?;
    Ok(normalizer)
}

fn apply_update(
    normalizer: &mut Normalizer,
    header: &WireHeader,
    added_segments: &[WireSegment],
    removed_segments: &[String],
    added_plans: &[WirePlan],
    removed_plans: &[String],
) -> Result<(), LogicalDecodeError> {
    // The producer downgrades confidence only through a snapshot, so an update
    // that claims a loss is malformed rather than a recoverable state.
    if header.confidence != "known" || !header.reason.is_empty() {
        return Err(invalid("update must carry a known view"));
    }
    if normalizer.profile.block_size != header.block_size
        || normalizer.profile.hash_unit != header.hash_unit
    {
        return Err(invalid("geometry changed without a snapshot"));
    }
    let mut mutations = Vec::with_capacity(
        removed_plans.len() + removed_segments.len() + added_segments.len() + added_plans.len(),
    );
    // Retire before admitting, so a reused identity cannot double-credit.
    for name in removed_plans {
        if let Some(mutation) = normalizer.drop_plan(name) {
            mutations.push(mutation);
        }
    }
    for name in removed_segments {
        if let Some(mutation) = normalizer.drop_segment(name) {
            mutations.push(mutation);
        }
    }
    for segment in added_segments {
        mutations.push(normalizer.segment(segment)?);
    }
    for plan in added_plans {
        mutations.push(normalizer.plan(plan)?);
    }
    if normalizer.segments.len() as u64 > header.max_rows {
        return Err(invalid("state budget exceeded"));
    }
    normalizer.commit(mutations)
}

/// Supported initial query shape: plain token IDs in the explicitly bound
/// model/layout namespace. Callers with LoRA/MM/salt/EAGLE must not use this helper.
pub fn query_tokens(view: &LogicalView, tokens: &[u32]) -> QueryResult {
    if view.confidence() != Confidence::Known {
        return view.query(&Query {
            scope: view.scope(),
            prompt_tokens: tokens.len().try_into().unwrap_or(u32::MAX),
            full: &[],
            fine: &[],
        });
    }
    let profile = view.profile();
    let options = BlockHashOptions {
        cache_namespace: Some(&view.scope().namespace),
        ..Default::default()
    };
    let full = compute_block_hash_for_seq(tokens, profile.block_size, options);
    let fine = compute_block_hash_for_seq(tokens, profile.hash_unit, options);
    view.query(&Query {
        scope: view.scope(),
        prompt_tokens: tokens.len().try_into().unwrap_or(u32::MAX),
        full: &full,
        fine: &fine,
    })
}

/// Ordered endpoint-local lifetime fencing. A snapshot repairs any loss; an
/// update that does not continue the accepted cut retires the view instead of
/// being applied. Malformed input retires only this view.
/// The epoch-history ceiling prevents restart churn growing memory without bound.
pub struct LogicalStream {
    owner: ResidencyOwner,
    namespace: String,
    limits: DecodeLimits,
    transport_cursor: Option<u64>,
    epoch: Option<[u8; 16]>,
    cursor: u64,
    /// Superseded epochs, so a replayed stale cut cannot resurrect a retired
    /// view. Bounded and FIFO: on overflow the OLDEST entry is forgotten rather
    /// than the newest refused. Refusing would turn a full history into a
    /// permanent refusal of every future CPU reset, which is the protocol's own
    /// recovery path — a worse failure than losing the fence on an epoch old
    /// enough to have fallen out of a bounded window.
    retired: VecDeque<[u8; 16]>,
    retired_index: HashSet<[u8; 16]>,
    /// A frame was lost on the wire and the view was retired because of it.
    /// The subscriber must request a replay; nothing on the publication path
    /// will repair this by itself, because the producer publishes only on
    /// change and a sticky unknown state has nothing further to say.
    needs_replay: bool,
    normalizer: Option<Normalizer>,
}

impl LogicalStream {
    pub fn new(owner: ResidencyOwner, namespace: String, limits: DecodeLimits) -> Self {
        Self {
            owner,
            namespace,
            limits,
            transport_cursor: None,
            epoch: None,
            cursor: 0,
            retired: VecDeque::new(),
            retired_index: HashSet::new(),
            needs_replay: false,
            normalizer: None,
        }
    }

    /// True when a wire-level frame loss retired the view. The subscriber owning
    /// the socket must request a replay from sequence 0; the retained window
    /// opens with a covering snapshot, which re-bases this stream.
    pub fn needs_replay(&self) -> bool {
        self.needs_replay
    }

    /// Superseded epochs currently fenced. Exposed for bound assertions.
    pub fn retired_epochs(&self) -> usize {
        self.retired.len()
    }

    /// The producer's own cursor for the last accepted cut, as it appeared on
    /// the wire. This is the value to reconcile against engine-side logs.
    ///
    /// It is deliberately NOT `LogicalView::query(..).cursor`: the view's cursor
    /// is an internal commit counter for this normalizer's baseline, which is
    /// re-seeded at zero on every snapshot and so does not track the epoch.
    pub fn cursor(&self) -> Option<u64> {
        self.normalizer.as_ref().map(|_| self.cursor)
    }

    pub fn view(&self) -> Option<&LogicalView> {
        self.normalizer.as_ref().map(|n| &n.view)
    }

    pub fn plan_names(&self) -> Option<&HashMap<u64, String>> {
        self.normalizer.as_ref().map(|n| &n.plan_names)
    }

    pub fn invalidate(&mut self) {
        self.normalizer = None;
    }

    /// Called by the lifecycle owner when an authenticated producer attachment
    /// is replaced. Transport sequences restart with the socket, so continuity
    /// is not inferred from the wire.
    pub fn reattach(&mut self) {
        self.invalidate();
        // Only the transport sequence genuinely restarts with the socket.
        self.transport_cursor = None;
        // The accepted epoch AND its cursor are deliberately KEPT. Re-attachment
        // is a socket event on this side; it says nothing about the producer,
        // whose epoch changes only on a CPU reset or a restart. Fencing the
        // epoch would retire a still-live producer and reject its very next
        // snapshot; zeroing the cursor would throw away the high-water mark that
        // makes the retained epoch mean anything, letting a replayed old cut
        // rebuild a stale view. A producer that really did restart arrives with
        // a NEW epoch and is fenced through the ordinary `retire` path.
        //
        // The cost is deliberate and fails SAFE: a producer whose view never
        // changes again publishes nothing above the mark, so this consumer stays
        // unknown rather than re-adopting. A producer liveness cut would close
        // that, and is recorded as a protocol change for approval, not invented
        // here. Unknown-and-recoverable beats stale-and-positive.
    }

    /// Record a superseded epoch, forgetting the oldest if the window is full.
    fn fence(&mut self, epoch: [u8; 16]) {
        if self.retired_index.contains(&epoch) {
            return;
        }
        while self.retired.len() >= self.limits.epochs {
            if let Some(oldest) = self.retired.pop_front() {
                self.retired_index.remove(&oldest);
            } else {
                break;
            }
        }
        if self.limits.epochs == 0 {
            return;
        }
        self.retired.push_back(epoch);
        self.retired_index.insert(epoch);
    }

    fn retire(&mut self, epoch: [u8; 16]) -> Result<(), LogicalDecodeError> {
        // Reject only an epoch we actually know to be superseded. A full
        // history must never become a blanket refusal.
        if self.retired_index.contains(&epoch) {
            self.invalidate();
            return Err(invalid("retired epoch"));
        }
        if let Some(old) = self.epoch {
            self.fence(old);
        }
        Ok(())
    }

    pub fn ingest(
        &mut self,
        transport_cursor: u64,
        payload: &[u8],
    ) -> Result<bool, LogicalDecodeError> {
        if let Some(old) = self.transport_cursor {
            if transport_cursor <= old {
                return Ok(false);
            }
            if transport_cursor != old + 1 {
                // A frame was lost. Protocol-cursor contiguity cannot see this:
                // a snapshot is accepted unconditionally, so a DROPPED snapshot
                // is invisible to it — and the one that announces a loss is
                // exactly the cut whose loss leaves stale POSITIVE credit,
                // because a sticky unknown producer publishes nothing further.
                // Retire now and let the socket owner replay.
                self.invalidate();
                self.needs_replay = true;
            }
        }
        self.transport_cursor = Some(transport_cursor);
        let cut = match decode_cut(payload, self.owner, &self.namespace, self.limits) {
            Ok(value) => value,
            Err(error) => {
                self.invalidate();
                return Err(error);
            }
        };
        match cut {
            Cut::Snapshot {
                header,
                segments,
                plans,
            } => {
                let epoch = epoch_bytes(&header)?;
                // The in-epoch high-water mark, checked whether or not a view
                // is currently held. Gating this on `normalizer.is_some()`
                // would mean a RETIRED view could be rebuilt from any replayed
                // old cut of the same epoch — stale credit, which is the one
                // failure this consumer must not have. A replay still recovers:
                // it delivers the retained window in order, ending above the
                // mark. The producer's cursor is monotonic within an epoch and
                // restarts only in `reset()`, which mints a new epoch.
                if self.epoch == Some(epoch) && header.cursor <= self.cursor {
                    return Ok(false);
                }
                if self.epoch != Some(epoch) {
                    self.retire(epoch)?;
                }
                match build_snapshot(
                    &header,
                    segments,
                    plans,
                    self.owner,
                    &self.namespace,
                    self.limits,
                ) {
                    Ok(normalizer) => {
                        self.epoch = Some(epoch);
                        self.cursor = header.cursor;
                        // An authoritative cut is exactly what a replay is for.
                        self.needs_replay = false;
                        self.normalizer = Some(normalizer);
                        Ok(true)
                    }
                    Err(error) => {
                        self.invalidate();
                        Err(error)
                    }
                }
            }
            Cut::Update {
                header,
                added_segments,
                removed_segments,
                added_plans,
                removed_plans,
            } => {
                let epoch = epoch_bytes(&header)?;
                if self.epoch != Some(epoch) || self.normalizer.is_none() {
                    // No accepted baseline for this epoch: an update cannot
                    // create one. Wait for the producer's next snapshot.
                    self.invalidate();
                    return Ok(false);
                }
                if header.cursor <= self.cursor {
                    return Ok(false);
                }
                if header.cursor != self.cursor + 1 {
                    self.invalidate();
                    return Ok(false);
                }
                let mut normalizer = self.normalizer.take().expect("baseline checked above");
                match apply_update(
                    &mut normalizer,
                    &header,
                    &added_segments,
                    &removed_segments,
                    &added_plans,
                    &removed_plans,
                ) {
                    Ok(()) => {
                        self.cursor = header.cursor;
                        self.normalizer = Some(normalizer);
                        Ok(true)
                    }
                    Err(error) => {
                        self.invalidate();
                        Err(error)
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::WorkerWithDpRank;

    fn cut(event_type: &str, cursor: u64) -> WireCut {
        WireCut {
            event_type: event_type.into(),
            profile: "native-cpu-root".into(),
            version: 1,
            namespace: "test".into(),
            epoch: "00000000000000000000000000000001".into(),
            cursor,
            confidence: "known".into(),
            reason: String::new(),
            block_size: 32,
            hash_unit: 4,
            max_rows: 16,
            max_plans: 144,
            segments: None,
            plans: None,
            added_segments: None,
            removed_segments: None,
            added_plans: None,
            removed_plans: None,
        }
    }

    fn segment() -> WireSegment {
        WireSegment {
            id: "full".into(),
            kind: "complete".into(),
            start: 0,
            end: 32,
            parent_hash: None,
            end_hash: NativeHash(vec![32]),
            token_ids: (0..32).collect(),
            hash_unit: 32,
            hashes: vec![NativeHash(vec![32])],
        }
    }

    fn plan() -> WirePlan {
        WirePlan {
            id: "checkpoint".into(),
            kind: "complete".into(),
            end: 28,
            anchor: 32,
            anchor_hash: Some(NativeHash(vec![32])),
            terminal_hash: NativeHash(vec![28]),
            minimum_prompt_tokens: 33,
            binding: "full".into(),
        }
    }

    fn wire() -> WireCut {
        WireCut {
            segments: Some(vec![segment()]),
            plans: Some(vec![plan()]),
            ..cut("LogicalSnapshot", 0)
        }
    }

    fn update(cursor: u64) -> WireCut {
        WireCut {
            added_segments: Some(vec![]),
            removed_segments: Some(vec![]),
            added_plans: Some(vec![]),
            removed_plans: Some(vec![]),
            ..cut("LogicalUpdate", cursor)
        }
    }

    fn bytes(wire: WireCut) -> Vec<u8> {
        rmp_serde::to_vec_named(&(0.0, vec![wire], 0i32)).unwrap()
    }
    fn update_bytes(wire: WireCut) -> Vec<u8> {
        bytes(wire)
    }
    fn owner() -> ResidencyOwner {
        ResidencyOwner::worker(WorkerWithDpRank::new(1, 0))
    }
    fn tokens() -> Vec<u32> {
        (0..33).collect()
    }

    fn eligible(stream: &LogicalStream) -> Option<u32> {
        query_tokens(stream.view().unwrap(), &tokens()).eligible_cpu_root_tokens
    }

    #[test]
    fn isolated_map_codec_normalizes_tokens_not_native_prefix_bytes() {
        let normalized =
            decode_snapshot(&bytes(wire()), owner(), "test", DecodeLimits::default()).unwrap();
        let view = LogicalView::from_snapshot(normalized.snapshot, Limits::default());
        assert_eq!(
            query_tokens(&view, &tokens()).eligible_cpu_root_tokens,
            Some(28)
        );
        assert!(super::super::decode_event_batch(&bytes(wire())).is_err());
    }

    #[test]
    fn hash_only_or_mismatched_namespace_is_unsupported_not_raw_credit() {
        let normalized =
            decode_snapshot(&bytes(wire()), owner(), "test", DecodeLimits::default()).unwrap();
        let view = LogicalView::from_snapshot(normalized.snapshot, Limits::default());
        assert_eq!(
            view.query(&Query {
                scope: view.scope(),
                prompt_tokens: 33,
                full: &[],
                fine: &[]
            })
            .status,
            Confidence::UnsupportedQuery
        );
        assert!(
            decode_snapshot(&bytes(wire()), owner(), "other", DecodeLimits::default()).is_err()
        );
    }

    #[test]
    fn malformed_geometry_binding_and_budget_are_rejected() {
        let mut malformed = wire();
        malformed.segments.as_mut().unwrap()[0].token_ids.pop();
        assert!(
            decode_snapshot(&bytes(malformed), owner(), "test", DecodeLimits::default()).is_err()
        );
        let mut missing = wire();
        missing.plans.as_mut().unwrap()[0].binding = "absent".into();
        assert!(
            decode_snapshot(&bytes(missing), owner(), "test", DecodeLimits::default()).is_err()
        );
        assert!(
            decode_snapshot(
                &bytes(wire()),
                owner(),
                "test",
                DecodeLimits {
                    bytes: 32,
                    ..Default::default()
                }
            )
            .is_err()
        );
    }

    #[test]
    fn an_unrecognised_wire_field_is_rejected_not_ignored() {
        // A future producer field must not be accepted as if this consumer had
        // understood it; `flatten` makes that easy to lose, so it is pinned.
        #[derive(Serialize)]
        struct Extended {
            #[serde(flatten)]
            base: WireCut,
            gpu_assisted_plans: Vec<u32>,
        }
        let payload = rmp_serde::to_vec_named(&(
            0.0,
            vec![Extended {
                base: wire(),
                gpu_assisted_plans: vec![1],
            }],
            0i32,
        ))
        .unwrap();
        assert!(decode_snapshot(&payload, owner(), "test", DecodeLimits::default()).is_err());
    }

    #[test]
    fn cache_owner_scope_is_out_of_profile_rather_than_unchecked() {
        use crate::identity::{
            CacheOwnerId, CacheSemanticsId, DcId, IdentitySource, IndexerDomainId, PoolId,
            RoutingScopeId, StableDpSlotId,
        };
        let cache_owner = CacheOwnerId::new(
            PoolId::new(
                IndexerDomainId::new(
                    CacheSemanticsId::new([1; 16], IdentitySource::Explicit),
                    RoutingScopeId::new([2; 16], IdentitySource::Explicit),
                ),
                DcId::new(3),
            ),
            StableDpSlotId::new([4; 16], IdentitySource::Explicit),
        );
        // A cache owner carries no DP rank to check against the payload, so v1
        // declines the scope instead of accepting an unchecked binding.
        let owner = ResidencyOwner::cache_owner(cache_owner);
        assert!(decode_snapshot(&bytes(wire()), owner, "test", DecodeLimits::default()).is_err());
    }

    #[test]
    fn contiguous_update_extends_the_accepted_snapshot() {
        let mut stream = LogicalStream::new(owner(), "test".into(), DecodeLimits::default());
        assert!(stream.ingest(10, &bytes(wire())).unwrap());
        assert_eq!(eligible(&stream), Some(28));

        // Withdraw only the declaration: the coverage row stays resident.
        let mut retire = update(1);
        retire.removed_plans = Some(vec!["checkpoint".into()]);
        assert!(stream.ingest(11, &update_bytes(retire)).unwrap());
        assert_eq!(eligible(&stream), Some(0));
        assert_eq!(stream.view().unwrap().confidence(), Confidence::Known);

        // Re-declare it on the same still-live binding.
        let mut restore = update(2);
        restore.added_plans = Some(vec![plan()]);
        assert!(stream.ingest(12, &update_bytes(restore)).unwrap());
        assert_eq!(eligible(&stream), Some(28));
    }

    #[test]
    fn a_missed_update_leaves_the_view_unknown_until_the_next_snapshot() {
        let mut stream = LogicalStream::new(owner(), "test".into(), DecodeLimits::default());
        assert!(stream.ingest(10, &bytes(wire())).unwrap());
        // Cursor 2 arrives without cursor 1: applying it would invent a view.
        let mut skipped = update(2);
        skipped.removed_plans = Some(vec!["checkpoint".into()]);
        assert!(!stream.ingest(11, &update_bytes(skipped)).unwrap());
        assert!(stream.view().is_none());
        // Cursor 1 is now stale; only an authoritative cut restores the view.
        assert!(!stream.ingest(12, &update_bytes(update(1))).unwrap());
        assert!(stream.view().is_none());
        let mut repaired = wire();
        repaired.cursor = 3;
        repaired.plans = Some(vec![]);
        assert!(stream.ingest(13, &bytes(repaired)).unwrap());
        assert_eq!(eligible(&stream), Some(0));
    }

    #[test]
    fn an_update_never_bootstraps_a_view_without_its_baseline() {
        let mut stream = LogicalStream::new(owner(), "test".into(), DecodeLimits::default());
        let mut orphan = update(1);
        orphan.added_segments = Some(vec![segment()]);
        orphan.added_plans = Some(vec![plan()]);
        assert!(!stream.ingest(10, &update_bytes(orphan)).unwrap());
        assert!(stream.view().is_none());
    }

    #[test]
    fn update_across_an_epoch_boundary_is_not_continuity() {
        let mut stream = LogicalStream::new(owner(), "test".into(), DecodeLimits::default());
        assert!(stream.ingest(10, &bytes(wire())).unwrap());
        let mut foreign = update(1);
        foreign.epoch = "00000000000000000000000000000002".into();
        foreign.removed_plans = Some(vec!["checkpoint".into()]);
        assert!(!stream.ingest(11, &update_bytes(foreign)).unwrap());
        assert!(stream.view().is_none());
    }

    #[test]
    fn an_update_may_not_downgrade_confidence_or_change_geometry() {
        let mut stream = LogicalStream::new(owner(), "test".into(), DecodeLimits::default());
        assert!(stream.ingest(10, &bytes(wire())).unwrap());
        let mut lossy = update(1);
        lossy.confidence = "unknown".into();
        lossy.reason = "loss".into();
        assert!(stream.ingest(11, &update_bytes(lossy)).is_err());
        assert!(stream.view().is_none());
    }

    #[test]
    fn stream_snapshot_replacement_repairs_loss_and_fences_old_epochs() {
        let mut stream = LogicalStream::new(owner(), "test".into(), DecodeLimits::default());
        assert!(stream.view().is_none());
        assert!(stream.ingest(10, &bytes(wire())).unwrap());
        assert!(!stream.ingest(10, &bytes(wire())).unwrap());
        let mut next = wire();
        next.cursor = 2;
        next.plans = Some(vec![]);
        assert!(stream.ingest(12, &bytes(next)).unwrap());
        assert_eq!(eligible(&stream), Some(0));
        let mut reset = wire();
        reset.epoch = "00000000000000000000000000000002".into();
        reset.segments = Some(vec![]);
        reset.plans = Some(vec![]);
        assert!(stream.ingest(13, &bytes(reset)).unwrap());
        assert!(stream.ingest(14, &bytes(wire())).is_err());
        assert!(stream.view().is_none());
    }

    #[test]
    fn unknown_snapshot_never_restores_positive_credit() {
        let mut stream = LogicalStream::new(owner(), "test".into(), DecodeLimits::default());
        stream.ingest(0, &bytes(wire())).unwrap();
        let mut unknown = wire();
        unknown.cursor = 1;
        unknown.confidence = "unknown".into();
        unknown.reason = "loss".into();
        unknown.segments = Some(vec![]);
        unknown.plans = Some(vec![]);
        stream.ingest(1, &bytes(unknown)).unwrap();
        assert_eq!(eligible(&stream), None);
    }

    #[test]
    fn a_lost_frame_retires_the_view_instead_of_keeping_stale_positive_credit() {
        // The dangerous loss is a dropped SNAPSHOT, and worst of all the one
        // announcing that the engine's view went unknown: the producer is
        // sticky-unknown afterwards and publishes nothing more, so without a
        // wire-level gap check this consumer would answer with positive credit
        // forever. Protocol-cursor contiguity cannot see it — a snapshot is
        // accepted unconditionally.
        let mut stream = LogicalStream::new(owner(), "test".into(), DecodeLimits::default());
        assert!(stream.ingest(7, &bytes(wire())).unwrap());
        assert_eq!(eligible(&stream), Some(28));
        assert!(!stream.needs_replay());

        // Transport sequence 8 never arrives. 9 does.
        let mut later = wire();
        later.cursor = 2;
        later.plans = Some(vec![]);
        // The cut itself is a valid snapshot, so it re-bases — but the gap is
        // still reported, because a dropped UPDATE would have left the view
        // retired with nothing to repair it.
        assert!(stream.ingest(9, &bytes(later)).unwrap());
        assert!(
            !stream.needs_replay(),
            "an authoritative cut answers the gap"
        );
        assert_eq!(eligible(&stream), Some(0));

        // Now lose a frame and have the next one be an update: no baseline, so
        // the view stays retired and the socket owner is told to replay.
        let mut update_after_gap = update(3);
        update_after_gap.removed_plans = Some(vec!["checkpoint".into()]);
        assert!(!stream.ingest(20, &update_bytes(update_after_gap)).unwrap());
        assert!(
            stream.view().is_none(),
            "stale credit survived a lost frame"
        );
        assert!(
            stream.needs_replay(),
            "the socket owner was not told to replay"
        );
    }

    /// Pins the ceiling the defaults actually impose at the K3 geometry, so a
    /// silent capacity cliff cannot drift back in unnoticed. If this number
    /// changes, the go-live sizing note has to change with it.
    #[test]
    fn the_default_ceilings_cap_a_k3_pool_at_675_rows() {
        const B: u32 = 12_288;
        const H: u32 = 128;
        let defaults = DecodeLimits::default();
        let per_row = u64::from(B / H) + 1;
        let cap_from_plans = defaults.state.plans as u64 / per_row;
        assert_eq!(per_row, 97);
        assert_eq!(cap_from_plans, 675);
        assert!(
            cap_from_plans < defaults.state.bindings as u64,
            "the plan ceiling is the one that binds; the sizing note says so",
        );
        // Sizing for the real pool lifts it, consistently across all three.
        let sized = DecodeLimits::for_pool(100_000, B, H).expect("valid geometry");
        assert_eq!(sized.state.bindings, 100_000);
        assert_eq!(sized.state.plans as u64, 100_000 * per_row);
        let mut big = wire();
        big.block_size = B;
        big.hash_unit = H;
        big.max_rows = 100_000;
        big.max_plans = big.max_rows * per_row;
        big.segments = Some(vec![]);
        big.plans = Some(vec![]);
        // Header now passes; only the payload budget remains to be measured.
        assert!(decode_snapshot(&bytes(big), owner(), "test", sized).is_ok());
    }

    #[test]
    fn a_pool_larger_than_the_consumer_ceiling_names_both_numbers() {
        // `max_rows` is the engine's whole CPU pool capacity. A production pool
        // can exceed a conservative consumer default, and the refusal is
        // permanent for that worker, so the error has to be actionable rather
        // than a bare "budget exceeded".
        let mut big = wire();
        big.max_rows = 100_000;
        big.max_plans = big.max_rows * (u64::from(big.block_size / big.hash_unit) + 1);
        let message = match decode_snapshot(&bytes(big), owner(), "test", DecodeLimits::default()) {
            Ok(_) => panic!("an oversized pool must be refused"),
            Err(error) => error.to_string(),
        };
        assert!(message.contains("100000"), "{message}");
        assert!(message.contains("bindings"), "{message}");
    }

    #[test]
    fn a_full_epoch_history_evicts_rather_than_refusing_every_future_reset() {
        // A CPU reset mints a new epoch, and a reset is the protocol's own
        // recovery path. A bounded history must therefore forget its oldest
        // entry, not start refusing every future recovery: a long-lived router
        // would otherwise wedge at "no view" for that worker until it restarts.
        let limits = DecodeLimits {
            epochs: 4,
            ..DecodeLimits::default()
        };
        let mut stream = LogicalStream::new(owner(), "test".into(), limits);
        for generation in 1..=12u128 {
            let mut reset = wire();
            reset.epoch = format!("{generation:032x}");
            assert!(
                stream.ingest(generation as u64, &bytes(reset)).unwrap(),
                "epoch {generation} was refused",
            );
            assert_eq!(eligible(&stream), Some(28), "epoch {generation}");
        }
        // The history stays bounded while doing it.
        assert!(stream.retired_epochs() <= 4, "{}", stream.retired_epochs());
    }

    #[test]
    fn reattaching_to_an_unchanged_producer_accepts_its_next_snapshot() {
        // A producer's epoch changes only on a CPU reset or a restart, so a
        // socket re-attachment on this side usually finds the SAME epoch still
        // running. Fencing it here would reject that producer's very next
        // snapshot and strand the worker with no view at all.
        let mut stream = LogicalStream::new(owner(), "test".into(), DecodeLimits::default());
        assert!(stream.ingest(40, &bytes(wire())).unwrap());
        assert_eq!(eligible(&stream), Some(28));

        stream.reattach();
        assert!(stream.view().is_none());

        // A stale cut from the replaced socket must NOT rebuild the view: the
        // epoch survived re-attachment precisely so its high-water mark still
        // means something.
        assert!(!stream.ingest(0, &bytes(wire())).unwrap());
        assert!(
            stream.view().is_none(),
            "a replayed old cut resurrected a stale view"
        );

        // The producer's next real cut is above the mark and is accepted.
        let mut next = wire();
        next.cursor = 1;
        assert!(
            stream.ingest(1, &bytes(next)).unwrap(),
            "a still-live producer's own epoch was fenced by our socket churn",
        );
        assert_eq!(eligible(&stream), Some(28));
        assert_eq!(stream.retired_epochs(), 0, "nothing was superseded");
    }

    #[test]
    fn reattachment_churn_does_not_grow_the_epoch_history_without_bound() {
        let limits = DecodeLimits {
            epochs: 4,
            ..DecodeLimits::default()
        };
        let mut stream = LogicalStream::new(owner(), "test".into(), limits);
        for generation in 1..=12u128 {
            let mut fresh = wire();
            fresh.epoch = format!("{generation:032x}");
            assert!(stream.ingest(0, &bytes(fresh)).unwrap());
            stream.reattach();
        }
        assert!(stream.retired_epochs() <= 4, "{}", stream.retired_epochs());
    }

    #[test]
    fn reattachment_fences_the_old_producer_and_restarts_transport_sequences() {
        let mut stream = LogicalStream::new(owner(), "test".into(), DecodeLimits::default());
        assert!(stream.ingest(40, &bytes(wire())).unwrap());
        stream.reattach();
        assert!(stream.view().is_none());
        // A restarted producer's sequence 0 is accepted, and its old epoch is
        // fenced so a replayed stale cut cannot resurrect the view.
        let mut fresh = wire();
        fresh.epoch = "0000000000000000000000000000000a".into();
        assert!(stream.ingest(0, &bytes(fresh)).unwrap());
        assert!(stream.ingest(1, &bytes(wire())).is_err());
        assert!(stream.view().is_none());
    }
}
