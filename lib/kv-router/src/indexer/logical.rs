// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Generic CPU-root logical cache model. This is not an engine wire schema.
//! The engine certifies readiness; coverage alone never declares a resume point.

use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};

use super::lower_tier::OwnedContinuationIndex;
use crate::protocols::{ExternalSequenceBlockHash, LocalBlockHash, ResidencyOwner};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Profile {
    pub version: u32,
    pub block_size: u32,
    pub hash_unit: u32,
}

impl Profile {
    fn supported(self) -> bool {
        self.version == 1
            && self.hash_unit > 0
            && self.block_size > self.hash_unit
            && self.block_size.is_multiple_of(self.hash_unit)
    }
}

/// Exactly one owner/rank, matching domain and source lifetime. Never project
/// several owners onto a worker before matching their independent CPU roots.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Scope {
    pub owner: ResidencyOwner,
    pub namespace: String,
    pub incarnation: [u8; 16],
    pub generation: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CoverageRole {
    Prefix,
    Tail,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Edge {
    pub role: CoverageRole,
    pub parent: Option<ExternalSequenceBlockHash>,
    pub local: LocalBlockHash,
    pub child: ExternalSequenceBlockHash,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Binding {
    pub id: u64,
    pub role: CoverageRole,
    pub start: u32,
    pub end: u32,
    pub parent: Option<ExternalSequenceBlockHash>,
    pub child: ExternalSequenceBlockHash,
}

impl Binding {
    fn valid(&self, profile: Profile) -> bool {
        self.start < self.end
            && self.start.is_multiple_of(profile.block_size)
            && (self.start == 0) == self.parent.is_none()
            && match self.role {
                CoverageRole::Prefix => self.end - self.start == profile.block_size,
                CoverageRole::Tail => {
                    self.end - self.start < profile.block_size
                        && self.end.is_multiple_of(profile.hash_unit)
                }
            }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Contribution {
    pub binding: Binding,
    pub edges: Vec<Edge>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Plan {
    pub id: u64,
    pub binding: u64,
    pub end: u32,
    pub anchor_end: u32,
    pub anchor: Option<ExternalSequenceBlockHash>,
    pub minimum_prompt_tokens: u32,
}

impl Plan {
    fn valid(&self, binding: &Binding, profile: Profile) -> bool {
        if self.end == 0
            || !self.end.is_multiple_of(profile.hash_unit)
            || !self.anchor_end.is_multiple_of(profile.block_size)
            || self.minimum_prompt_tokens <= self.end.max(self.anchor_end)
            || (self.anchor_end == 0) != self.anchor.is_none()
        {
            return false;
        }
        match binding.role {
            CoverageRole::Prefix => {
                self.anchor_end == binding.end
                    && self.anchor == Some(binding.child)
                    && self.end > binding.start
                    && self.end <= binding.end
            }
            CoverageRole::Tail => {
                self.anchor_end == binding.start
                    && self.anchor == binding.parent
                    && self.end == binding.end
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum Mutation {
    Coverage(Contribution),
    RemoveCoverage(u64),
    Plan(Plan),
    RemovePlan(u64),
    ClearCpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Confidence {
    Known,
    Gap,
    UnsupportedProfile,
    Invalid,
    Overflow,
    UnsupportedQuery,
    ScopeMismatch,
}

/// Explicit local ceilings, independent of an engine's advertised capacity.
#[derive(Debug, Clone, Copy)]
pub struct Limits {
    pub bindings: usize,
    pub edges: usize,
    pub plans: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            bindings: 4096,
            edges: 262_144,
            plans: 65_536,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Snapshot {
    pub profile: Profile,
    pub scope: Scope,
    pub cursor: u64,
    pub confidence: Confidence,
    pub edges: Vec<Edge>,
    pub bindings: Vec<Binding>,
    pub plans: Vec<Plan>,
}

/// A materialized mutation cut, not an engine event. Semantic tables are retained
/// even when ownership coalescing produces no edge changes.
#[derive(Debug, Clone)]
pub struct MaterializedBatch {
    pub profile: Profile,
    pub scope: Scope,
    pub cursor: u64,
    pub confidence: Confidence,
    pub added_edges: Vec<Edge>,
    pub removed_edges: Vec<Edge>,
    pub bindings: Vec<Binding>,
    pub plans: Vec<Plan>,
}

/// The single contribution-reference owner. This runs before publication, never
/// in a receiving index. IDs are interned by the transport boundary; recurrent
/// hash-only events have no representation in this typed input.
pub struct Publisher {
    profile: Profile,
    scope: Scope,
    limits: Limits,
    cursor: u64,
    confidence: Confidence,
    contributions: FxHashMap<u64, Contribution>,
    contribution_edges: usize,
    references: FxHashMap<Edge, usize>,
    plans: FxHashMap<u64, Plan>,
    by_binding: FxHashMap<u64, FxHashSet<u64>>,
}

impl Publisher {
    pub fn new(profile: Profile, scope: Scope, limits: Limits) -> Self {
        Self {
            profile,
            scope,
            limits,
            cursor: 0,
            confidence: if profile.supported() {
                Confidence::Known
            } else {
                Confidence::UnsupportedProfile
            },
            contributions: FxHashMap::default(),
            contribution_edges: 0,
            references: FxHashMap::default(),
            plans: FxHashMap::default(),
            by_binding: FxHashMap::default(),
        }
    }

    fn clear(&mut self) {
        self.contributions.clear();
        self.contribution_edges = 0;
        self.references.clear();
        self.plans.clear();
        self.by_binding.clear();
    }

    fn fail(&mut self, reason: Confidence) {
        self.clear();
        self.confidence = reason;
    }

    fn remove_plan(&mut self, id: u64) {
        if let Some(plan) = self.plans.remove(&id)
            && let Some(members) = self.by_binding.get_mut(&plan.binding)
        {
            members.remove(&id);
            if members.is_empty() {
                self.by_binding.remove(&plan.binding);
            }
        }
    }

    fn mutation(&mut self, mutation: Mutation) -> Result<(), Confidence> {
        match mutation {
            Mutation::Coverage(contribution) => {
                let binding = &contribution.binding;
                if let Some(old) = self.contributions.get(&binding.id) {
                    return if old == &contribution {
                        Ok(())
                    } else {
                        Err(Confidence::Invalid)
                    };
                }
                if !binding.valid(self.profile) {
                    return Err(Confidence::Invalid);
                }
                let unit = match binding.role {
                    CoverageRole::Prefix => self.profile.block_size,
                    CoverageRole::Tail => self.profile.hash_unit,
                };
                if contribution.edges.len() != ((binding.end - binding.start) / unit) as usize {
                    return Err(Confidence::Invalid);
                }
                let mut parent = binding.parent;
                let mut seen = FxHashSet::default();
                for edge in &contribution.edges {
                    if edge.role != binding.role
                        || edge.parent != parent
                        || !seen.insert(edge.child)
                    {
                        return Err(Confidence::Invalid);
                    }
                    parent = Some(edge.child);
                }
                if parent != Some(binding.child) {
                    return Err(Confidence::Invalid);
                }
                // Charge contributions before coalescing: sharing cannot evade the memory ceiling.
                if self.contributions.len() >= self.limits.bindings
                    || self
                        .contribution_edges
                        .saturating_add(contribution.edges.len())
                        > self.limits.edges
                {
                    return Err(Confidence::Overflow);
                }
                self.contribution_edges += contribution.edges.len();
                for edge in &contribution.edges {
                    *self.references.entry(*edge).or_default() += 1;
                }
                self.contributions.insert(binding.id, contribution);
            }
            Mutation::RemoveCoverage(id) => {
                if let Some(contribution) = self.contributions.remove(&id) {
                    self.contribution_edges -= contribution.edges.len();
                    for edge in contribution.edges {
                        if let Some(count) = self.references.get_mut(&edge) {
                            *count -= 1;
                            if *count == 0 {
                                self.references.remove(&edge);
                            }
                        }
                    }
                    if let Some(plans) = self.by_binding.remove(&id) {
                        for plan in plans {
                            self.plans.remove(&plan);
                        }
                    }
                }
            }
            Mutation::Plan(plan) => {
                if let Some(old) = self.plans.get(&plan.id) {
                    return if old == &plan {
                        Ok(())
                    } else {
                        Err(Confidence::Invalid)
                    };
                }
                let Some(contribution) = self.contributions.get(&plan.binding) else {
                    return Err(Confidence::Invalid);
                };
                if !plan.valid(&contribution.binding, self.profile) {
                    return Err(Confidence::Invalid);
                }
                let cap = match contribution.binding.role {
                    CoverageRole::Prefix => self.profile.block_size / self.profile.hash_unit,
                    CoverageRole::Tail => 1,
                } as usize;
                if self.plans.len() >= self.limits.plans
                    || self
                        .by_binding
                        .get(&plan.binding)
                        .is_some_and(|ids| ids.len() >= cap)
                {
                    return Err(Confidence::Overflow);
                }
                self.by_binding
                    .entry(plan.binding)
                    .or_default()
                    .insert(plan.id);
                self.plans.insert(plan.id, plan);
            }
            Mutation::RemovePlan(id) => self.remove_plan(id),
            Mutation::ClearCpu => self.clear(),
        }
        Ok(())
    }

    pub fn apply(&mut self, cursor: u64, mutations: Vec<Mutation>) -> MaterializedBatch {
        let before: FxHashSet<_> = self.references.keys().copied().collect();
        if cursor > self.cursor {
            if self.cursor.checked_add(1) != Some(cursor) {
                self.fail(Confidence::Gap);
            }
            self.cursor = cursor;
            if self.confidence == Confidence::Known {
                for mutation in mutations {
                    if let Err(reason) = self.mutation(mutation) {
                        self.fail(reason);
                        break;
                    }
                }
            }
        }
        MaterializedBatch {
            profile: self.profile,
            scope: self.scope.clone(),
            cursor: self.cursor,
            confidence: self.confidence,
            added_edges: self
                .references
                .keys()
                .filter(|e| !before.contains(e))
                .copied()
                .collect(),
            removed_edges: before
                .into_iter()
                .filter(|e| !self.references.contains_key(e))
                .collect(),
            bindings: self
                .contributions
                .values()
                .map(|c| c.binding.clone())
                .collect(),
            plans: self.plans.values().cloned().collect(),
        }
    }

    pub fn snapshot(&self) -> Snapshot {
        Snapshot {
            profile: self.profile,
            scope: self.scope.clone(),
            cursor: self.cursor,
            confidence: self.confidence,
            edges: self.references.keys().copied().collect(),
            bindings: self
                .contributions
                .values()
                .map(|c| c.binding.clone())
                .collect(),
            plans: self.plans.values().cloned().collect(),
        }
    }

    pub fn live_counts(&self) -> (usize, usize, usize) {
        (
            self.contributions.len(),
            self.references.len(),
            self.plans.len(),
        )
    }
}

type Anchor = (u32, Option<ExternalSequenceBlockHash>);

/// Materialized view shared by local, frontend and remote consumers. Apply/query
/// require a coherent caller-held lock or an immutable published replacement.
/// No native group semantics and no contribution refcounts live here.
pub struct LogicalView {
    profile: Profile,
    scope: Scope,
    cursor: u64,
    confidence: Confidence,
    limits: Limits,
    prefix: OwnedContinuationIndex,
    tail: OwnedContinuationIndex,
    edges: FxHashSet<Edge>,
    bindings: FxHashMap<u64, Binding>,
    plans: FxHashMap<u64, Plan>,
    by_anchor: FxHashMap<Anchor, Vec<u64>>,
    tails_by_anchor: FxHashMap<Anchor, u32>,
}

pub struct Query<'a> {
    pub scope: &'a Scope,
    pub prompt_tokens: u32,
    pub full: &'a [LocalBlockHash],
    /// Root-aligned fine hashes using the same matching options as stores.
    pub fine: &'a [LocalBlockHash],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueryResult {
    pub status: Confidence,
    pub raw_cpu_tokens: u32,
    pub eligible_cpu_root_tokens: Option<u32>,
    pub selected_plan: Option<u64>,
    /// This view's own commit counter, NOT the producer's wire cursor. A
    /// consumer that re-seeds the view on every snapshot restarts it at zero.
    /// To reconcile against engine-side logs use `LogicalStream::cursor()`.
    pub cursor: u64,
}

impl LogicalView {
    pub fn from_snapshot(snapshot: Snapshot, limits: Limits) -> Self {
        let mut view = Self {
            profile: snapshot.profile,
            scope: snapshot.scope,
            cursor: snapshot.cursor,
            confidence: snapshot.confidence,
            limits,
            prefix: OwnedContinuationIndex::default(),
            tail: OwnedContinuationIndex::default(),
            edges: FxHashSet::default(),
            bindings: FxHashMap::default(),
            plans: FxHashMap::default(),
            by_anchor: FxHashMap::default(),
            tails_by_anchor: FxHashMap::default(),
        };
        if !view.profile.supported() {
            view.fail(Confidence::UnsupportedProfile);
        }
        if view.confidence != Confidence::Known {
            return view;
        }
        if snapshot.edges.len() > limits.edges {
            view.fail(Confidence::Overflow);
            return view;
        }
        for edge in snapshot.edges {
            if !view.insert_edge(edge) {
                view.fail(Confidence::Invalid);
                return view;
            }
        }
        if let Err(reason) = view.replace_tables(snapshot.bindings, snapshot.plans) {
            view.fail(reason);
        }
        view
    }

    fn fail(&mut self, reason: Confidence) {
        self.confidence = reason;
        self.prefix = OwnedContinuationIndex::default();
        self.tail = OwnedContinuationIndex::default();
        self.edges.clear();
        self.bindings.clear();
        self.plans.clear();
        self.by_anchor.clear();
        self.tails_by_anchor.clear();
    }

    fn insert_edge(&mut self, edge: Edge) -> bool {
        let index = match edge.role {
            CoverageRole::Prefix => &mut self.prefix,
            CoverageRole::Tail => &mut self.tail,
        };
        if !index.insert(self.scope.owner, edge.parent, edge.local, edge.child) {
            return false;
        }
        self.edges.insert(edge);
        true
    }

    fn replace_tables(
        &mut self,
        bindings: Vec<Binding>,
        plans: Vec<Plan>,
    ) -> Result<(), Confidence> {
        if bindings.len() > self.limits.bindings || plans.len() > self.limits.plans {
            return Err(Confidence::Overflow);
        }
        self.bindings.clear();
        self.plans.clear();
        self.by_anchor.clear();
        self.tails_by_anchor.clear();
        for binding in bindings {
            if !binding.valid(self.profile) || self.bindings.contains_key(&binding.id) {
                return Err(Confidence::Invalid);
            }
            if binding.role == CoverageRole::Tail {
                let end = self
                    .tails_by_anchor
                    .entry((binding.start, binding.parent))
                    .or_default();
                *end = (*end).max(binding.end);
            }
            self.bindings.insert(binding.id, binding);
        }
        // Snapshots must contain the exact matching material for every live
        // local binding. A partial dump is unknown, not an authoritative miss.
        let reverse: FxHashMap<_, _> = self
            .edges
            .iter()
            .map(|edge| ((edge.role, edge.child), edge))
            .collect();
        let mut covered = FxHashSet::default();
        let mut charged_edges = 0usize;
        for binding in self.bindings.values() {
            let unit = match binding.role {
                CoverageRole::Prefix => self.profile.block_size,
                CoverageRole::Tail => self.profile.hash_unit,
            };
            let count = ((binding.end - binding.start) / unit) as usize;
            charged_edges = charged_edges.saturating_add(count);
            if charged_edges > self.limits.edges {
                return Err(Confidence::Overflow);
            }
            let mut child = Some(binding.child);
            for _ in 0..count {
                let Some(edge) = child.and_then(|hash| reverse.get(&(binding.role, hash))) else {
                    return Err(Confidence::Invalid);
                };
                covered.insert(**edge);
                child = edge.parent;
            }
            if child != binding.parent {
                return Err(Confidence::Invalid);
            }
        }
        if covered.len() != self.edges.len() {
            return Err(Confidence::Invalid);
        }
        let mut per_binding = FxHashMap::<u64, usize>::default();
        for plan in plans {
            let Some(binding) = self.bindings.get(&plan.binding) else {
                return Err(Confidence::Invalid);
            };
            if !plan.valid(binding, self.profile) || self.plans.contains_key(&plan.id) {
                return Err(Confidence::Invalid);
            }
            let count = per_binding.entry(plan.binding).or_default();
            *count += 1;
            let cap = match binding.role {
                CoverageRole::Prefix => self.profile.block_size / self.profile.hash_unit,
                CoverageRole::Tail => 1,
            } as usize;
            if *count > cap {
                return Err(Confidence::Overflow);
            }
            self.by_anchor
                .entry((plan.anchor_end, plan.anchor))
                .or_default()
                .push(plan.id);
            self.plans.insert(plan.id, plan);
        }
        Ok(())
    }

    pub fn apply(&mut self, batch: MaterializedBatch) {
        if batch.scope != self.scope {
            return;
        } // Lifecycle owner fences replacement snapshots.
        if batch.profile != self.profile {
            self.fail(Confidence::UnsupportedProfile);
            return;
        }
        if batch.cursor <= self.cursor {
            return;
        }
        let contiguous = self.cursor.checked_add(1) == Some(batch.cursor);
        self.cursor = batch.cursor;
        if !contiguous {
            self.fail(Confidence::Gap);
            return;
        }
        if batch.confidence != Confidence::Known {
            self.fail(batch.confidence);
            return;
        }
        if self.confidence != Confidence::Known {
            return;
        }
        for edge in batch.removed_edges {
            if self.edges.remove(&edge) {
                let index = match edge.role {
                    CoverageRole::Prefix => &mut self.prefix,
                    CoverageRole::Tail => &mut self.tail,
                };
                index.remove(self.scope.owner, edge.child);
            }
        }
        for edge in batch.added_edges {
            if self.edges.len() >= self.limits.edges {
                self.fail(Confidence::Overflow);
                return;
            }
            if !self.insert_edge(edge) {
                self.fail(Confidence::Invalid);
                return;
            }
        }
        if let Err(reason) = self.replace_tables(batch.bindings, batch.plans) {
            self.fail(reason);
        }
    }

    pub fn plan_count(&self) -> usize {
        self.plans.len()
    }
    pub fn scope(&self) -> &Scope {
        &self.scope
    }
    pub fn profile(&self) -> Profile {
        self.profile
    }
    pub fn confidence(&self) -> Confidence {
        self.confidence
    }

    pub fn snapshot(&self) -> Snapshot {
        Snapshot {
            profile: self.profile,
            scope: self.scope.clone(),
            cursor: self.cursor,
            confidence: self.confidence,
            edges: self.edges.iter().copied().collect(),
            bindings: self.bindings.values().cloned().collect(),
            plans: self.plans.values().cloned().collect(),
        }
    }

    pub fn query(&self, query: &Query<'_>) -> QueryResult {
        let mut result = QueryResult {
            status: self.confidence,
            raw_cpu_tokens: 0,
            eligible_cpu_root_tokens: None,
            selected_plan: None,
            cursor: self.cursor,
        };
        if query.scope != &self.scope {
            result.status = Confidence::ScopeMismatch;
            return result;
        }
        if self.confidence != Confidence::Known {
            return result;
        }
        if query.full.len() != (query.prompt_tokens / self.profile.block_size) as usize
            || query.fine.len() != (query.prompt_tokens / self.profile.hash_unit) as usize
        {
            result.status = Confidence::UnsupportedQuery;
            return result;
        }
        result.eligible_cpu_root_tokens = Some(0);
        let mut anchors = vec![(0, None)];
        let mut parent = None;
        for (position, local) in query.full.iter().enumerate() {
            let Some(child) = self.prefix.child(self.scope.owner, parent, *local) else {
                break;
            };
            let end = (position as u32 + 1) * self.profile.block_size;
            anchors.push((end, Some(child)));
            result.raw_cpu_tokens = end;
            parent = Some(child);
        }
        // Enumerate only plans/tails at matching anchors, including the root and
        // anchors behind the deepest full match. Never scan the whole cache.
        for anchor in anchors {
            let mut tail_matches = FxHashMap::default();
            let mut parent = anchor.1;
            if let Some(end) = self.tails_by_anchor.get(&anchor) {
                for position in anchor.0 / self.profile.hash_unit
                    ..(*end).min(query.prompt_tokens) / self.profile.hash_unit
                {
                    let Some(child) =
                        self.tail
                            .child(self.scope.owner, parent, query.fine[position as usize])
                    else {
                        break;
                    };
                    let end = (position + 1) * self.profile.hash_unit;
                    tail_matches.insert(end, child);
                    result.raw_cpu_tokens = result.raw_cpu_tokens.max(end);
                    parent = Some(child);
                }
            }
            for id in self.by_anchor.get(&anchor).into_iter().flatten() {
                let plan = &self.plans[id];
                if query.prompt_tokens < plan.minimum_prompt_tokens {
                    continue;
                }
                let binding = &self.bindings[&plan.binding];
                if binding.role == CoverageRole::Tail
                    && tail_matches.get(&plan.end) != Some(&binding.child)
                {
                    continue;
                }
                let best = result.eligible_cpu_root_tokens.unwrap_or(0);
                if plan.end > best
                    || (plan.end == best && result.selected_plan.is_none_or(|old| plan.id < old))
                {
                    result.eligible_cpu_root_tokens = Some(plan.end);
                    result.selected_plan = Some(plan.id);
                }
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::WorkerWithDpRank;

    const B: u32 = 12_288;
    const H: u32 = 128;

    fn profile() -> Profile {
        Profile {
            version: 1,
            block_size: B,
            hash_unit: H,
        }
    }

    fn scope() -> Scope {
        Scope {
            owner: ResidencyOwner::worker(WorkerWithDpRank::new(7, 0)),
            namespace: "test".into(),
            incarnation: [1; 16],
            generation: 1,
        }
    }

    fn edge(role: CoverageRole, start: u32, end: u32) -> Edge {
        Edge {
            role,
            parent: (start > 0).then_some(ExternalSequenceBlockHash(start as u64)),
            local: LocalBlockHash(end as u64),
            child: ExternalSequenceBlockHash(end as u64),
        }
    }

    fn coverage(id: u64, role: CoverageRole, start: u32, end: u32) -> Contribution {
        let step = if role == CoverageRole::Prefix { B } else { H };
        Contribution {
            binding: Binding {
                id,
                role,
                start,
                end,
                parent: (start > 0).then_some(ExternalSequenceBlockHash(start as u64)),
                child: ExternalSequenceBlockHash(end as u64),
            },
            edges: (start..end)
                .step_by(step as usize)
                .map(|p| edge(role, p, p + step))
                .collect(),
        }
    }

    fn plan(id: u64, binding: u64, end: u32, anchor: u32, min_n: u32) -> Plan {
        Plan {
            id,
            binding,
            end,
            anchor_end: anchor,
            anchor: (anchor > 0).then_some(ExternalSequenceBlockHash(anchor as u64)),
            minimum_prompt_tokens: min_n,
        }
    }

    struct Harness {
        publisher: Publisher,
        view: LogicalView,
        cursor: u64,
    }

    impl Harness {
        fn new() -> Self {
            let publisher = Publisher::new(profile(), scope(), Limits::default());
            let view = LogicalView::from_snapshot(publisher.snapshot(), Limits::default());
            Self {
                publisher,
                view,
                cursor: 0,
            }
        }
        fn apply(&mut self, changes: Vec<Mutation>) -> MaterializedBatch {
            self.cursor += 1;
            let batch = self.publisher.apply(self.cursor, changes);
            self.view.apply(batch.clone());
            batch
        }
        fn query(&self, n: u32, diverge: Option<u32>) -> QueryResult {
            let mut full: Vec<_> = (B..=n)
                .step_by(B as usize)
                .map(|e| LocalBlockHash(e as u64))
                .collect();
            let mut fine: Vec<_> = (H..=n)
                .step_by(H as usize)
                .map(|e| LocalBlockHash(e as u64))
                .collect();
            if let Some(at) = diverge {
                for (i, hash) in full.iter_mut().enumerate() {
                    if (i as u32 + 1) * B > at {
                        hash.0 += 1;
                    }
                }
                for (i, hash) in fine.iter_mut().enumerate() {
                    if (i as u32 + 1) * H > at {
                        hash.0 += 1;
                    }
                }
            }
            self.view.query(&Query {
                scope: &scope(),
                prompt_tokens: n,
                full: &full,
                fine: &fine,
            })
        }
        fn base_tail(&mut self) {
            self.apply(vec![
                Mutation::Coverage(coverage(1, CoverageRole::Prefix, 0, B)),
                Mutation::Coverage(coverage(2, CoverageRole::Tail, B, 15_104)),
                Mutation::Plan(plan(2, 2, 15_104, B, 15_105)),
            ]);
        }
    }

    #[test]
    fn primary_p1_divergence_is_zero_despite_raw_coverage() {
        let mut h = Harness::new();
        h.base_tail();
        let result = h.query(23_400, Some(14_080));
        assert_eq!(result.raw_cpu_tokens, 14_080);
        assert_eq!(result.eligible_cpu_root_tokens, Some(0));
    }

    #[test]
    fn primary_p1_terminal_retirement_is_zero_despite_raw_coverage() {
        let mut h = Harness::new();
        h.base_tail();
        h.apply(vec![Mutation::RemovePlan(2)]);
        let result = h.query(23_400, None);
        assert_eq!(result.raw_cpu_tokens, 15_104);
        assert_eq!(result.eligible_cpu_root_tokens, Some(0));
    }

    #[test]
    fn planted_b_controls_preserve_original_12288_expectations() {
        let mut h = Harness::new();
        h.base_tail();
        h.apply(vec![Mutation::Plan(plan(1, 1, B, B, B + 1))]);
        assert_eq!(
            h.query(23_400, Some(14_080)).eligible_cpu_root_tokens,
            Some(B)
        );
        h.apply(vec![Mutation::RemovePlan(2)]);
        assert_eq!(h.query(23_400, None).eligible_cpu_root_tokens, Some(B));
    }

    #[test]
    fn whole_anchor_minimum_prompt_and_two_plans_at_10752() {
        let mut h = Harness::new();
        h.apply(vec![
            Mutation::Coverage(coverage(1, CoverageRole::Prefix, 0, B)),
            Mutation::Plan(plan(1, 1, 10_752, B, B + 1)),
        ]);
        for n in [11_000, B] {
            assert_eq!(h.query(n, None).eligible_cpu_root_tokens, Some(0));
        }
        assert_eq!(h.query(B + 1, None).eligible_cpu_root_tokens, Some(10_752));
        assert_eq!(
            h.query(23_400, Some(10_752)).eligible_cpu_root_tokens,
            Some(0)
        );
        h.apply(vec![
            Mutation::Coverage(coverage(2, CoverageRole::Tail, 0, 10_752)),
            Mutation::Plan(plan(2, 2, 10_752, 0, 10_753)),
        ]);
        assert_eq!(h.query(11_000, None).eligible_cpu_root_tokens, Some(10_752));
        assert_eq!(h.query(23_400, Some(10_752)).selected_plan, Some(2));
        h.apply(vec![Mutation::RemoveCoverage(1)]);
        assert_eq!(h.query(23_400, None).selected_plan, Some(2));
        assert_eq!(h.view.plan_count(), 1);
    }

    #[test]
    fn earlier_anchor_survives_deeper_complete_match_and_prefix_reinsert() {
        let mut h = Harness::new();
        h.base_tail();
        h.apply(vec![Mutation::Coverage(coverage(
            3,
            CoverageRole::Prefix,
            B,
            2 * B,
        ))]);
        assert_eq!(h.query(30_000, None).eligible_cpu_root_tokens, Some(15_104));
        h.apply(vec![Mutation::RemoveCoverage(1)]);
        assert_eq!(h.query(30_000, None).eligible_cpu_root_tokens, Some(0));
        assert_eq!(h.view.plan_count(), 1);
        h.apply(vec![Mutation::Coverage(coverage(
            4,
            CoverageRole::Prefix,
            0,
            B,
        ))]);
        assert_eq!(h.query(30_000, None).eligible_cpu_root_tokens, Some(15_104));
    }

    #[test]
    fn semantic_removal_survives_empty_hash_delta_and_duplicate_upsert() {
        let mut h = Harness::new();
        h.base_tail();
        h.apply(vec![
            Mutation::Coverage(coverage(2, CoverageRole::Tail, B, 15_104)),
            Mutation::Coverage(coverage(3, CoverageRole::Tail, B, 15_104)),
        ]);
        let batch = h.apply(vec![Mutation::RemoveCoverage(2)]);
        assert!(batch.removed_edges.is_empty());
        assert_eq!(h.query(23_400, None).eligible_cpu_root_tokens, Some(0));
        assert_eq!(h.query(23_400, None).raw_cpu_tokens, 15_104);
        h.apply(vec![
            Mutation::RemoveCoverage(3),
            Mutation::RemoveCoverage(3),
        ]);
        assert_eq!(h.query(23_400, None).raw_cpu_tokens, B);
    }

    #[test]
    fn snapshot_gap_duplicate_and_profile_mismatch_are_not_cache_misses() {
        let mut h = Harness::new();
        h.base_tail();
        let batch = h.apply(vec![Mutation::Plan(plan(1, 1, B, B, B + 1))]);
        h.view.apply(batch.clone());
        assert_eq!(h.query(23_400, None).eligible_cpu_root_tokens, Some(15_104));
        let mut gap = batch;
        gap.cursor += 2;
        h.view.apply(gap);
        assert_eq!(h.query(23_400, None).eligible_cpu_root_tokens, None);
        h.view = LogicalView::from_snapshot(h.publisher.snapshot(), Limits::default());
        assert_eq!(h.query(23_400, None).eligible_cpu_root_tokens, Some(15_104));
        let mut bad = h.publisher.snapshot();
        bad.profile.version = 99;
        h.view = LogicalView::from_snapshot(bad, Limits::default());
        assert_eq!(h.query(23_400, None).status, Confidence::UnsupportedProfile);
    }

    #[test]
    fn no_cross_scope_or_generation_credit() {
        let mut h = Harness::new();
        h.base_tail();
        let hashes = [LocalBlockHash(B as u64)];
        for other in [
            Scope {
                generation: 2,
                ..scope()
            },
            Scope {
                namespace: "other".into(),
                ..scope()
            },
            Scope {
                owner: ResidencyOwner::worker(WorkerWithDpRank::new(7, 1)),
                ..scope()
            },
        ] {
            assert_eq!(
                h.view
                    .query(&Query {
                        scope: &other,
                        prompt_tokens: 23_400,
                        full: &hashes,
                        fine: &[]
                    })
                    .eligible_cpu_root_tokens,
                None
            );
        }
    }

    #[test]
    fn materialized_snapshot_roundtrip_and_partial_dump() {
        let mut h = Harness::new();
        h.base_tail();
        let bytes = rmp_serde::to_vec(&h.view.snapshot()).unwrap();
        let snapshot: Snapshot = rmp_serde::from_slice(&bytes).unwrap();
        h.view = LogicalView::from_snapshot(snapshot, Limits::default());
        assert_eq!(h.query(23_400, None).eligible_cpu_root_tokens, Some(15_104));
        let mut partial = h.publisher.snapshot();
        partial.edges.clear();
        h.view = LogicalView::from_snapshot(partial, Limits::default());
        assert_eq!(h.query(23_400, None).status, Confidence::Invalid);
    }

    #[test]
    fn overflow_conflict_and_old_generation_do_not_retain_stale_credit() {
        let mut h = Harness::new();
        h.base_tail();
        let mut conflict = coverage(2, CoverageRole::Tail, B, 15_104);
        conflict.edges[0].local = LocalBlockHash(999);
        h.apply(vec![Mutation::Coverage(conflict)]);
        assert_eq!(h.query(23_400, None).status, Confidence::Invalid);
        assert_eq!(h.publisher.live_counts(), (0, 0, 0));
        let mut h = Harness::new();
        h.base_tail();
        let mut old = h.apply(vec![]);
        old.scope.generation = 0;
        old.cursor += 1;
        old.plans.clear();
        h.view.apply(old);
        assert_eq!(h.query(23_400, None).eligible_cpu_root_tokens, Some(15_104));
        h.view = LogicalView::from_snapshot(
            h.publisher.snapshot(),
            Limits {
                plans: 0,
                ..Limits::default()
            },
        );
        assert_eq!(h.query(23_400, None).status, Confidence::Overflow);
    }

    #[test]
    fn subblock_and_tail_union_never_manufacture_a_prefix_or_endpoint() {
        let mut h = Harness::new();
        h.apply(vec![
            Mutation::Coverage(coverage(1, CoverageRole::Tail, 0, B - H)),
            Mutation::Plan(plan(1, 1, B - H, 0, B - H + 1)),
            Mutation::Coverage(coverage(2, CoverageRole::Tail, B, B + H)),
            Mutation::Plan(plan(2, 2, B + H, B, B + H + 1)),
        ]);
        assert_eq!(h.query(30_000, None).eligible_cpu_root_tokens, Some(B - H));
        h.apply(vec![Mutation::RemovePlan(1)]);
        assert_eq!(h.query(30_000, None).eligible_cpu_root_tokens, Some(0));
    }

    #[test]
    fn cpu_clear_and_bounded_churn_leave_no_row_history() {
        let mut h = Harness::new();
        for id in 1..=1000 {
            h.apply(vec![
                Mutation::Coverage(coverage(id, CoverageRole::Tail, 0, 10_752)),
                Mutation::Plan(plan(id, id, 10_752, 0, 10_753)),
            ]);
            h.apply(vec![Mutation::RemoveCoverage(id)]);
            assert_eq!(h.publisher.live_counts(), (0, 0, 0));
            assert_eq!(h.view.plan_count(), 0);
        }
        h.base_tail();
        h.apply(vec![Mutation::ClearCpu]);
        assert_eq!(h.query(23_400, None).eligible_cpu_root_tokens, Some(0));
        assert_eq!(h.publisher.live_counts(), (0, 0, 0));
    }
}
