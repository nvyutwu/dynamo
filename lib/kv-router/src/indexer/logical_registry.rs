// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Multi-owner logical CPU views, their shared ceiling, and the gate that keeps
//! them out of routing until they are explicitly enabled.
//!
//! This is the consumer state a frontend or worker index holds once per attached
//! producer. It deliberately owns no socket: discovery, authentication and the
//! subscribing task belong to the runtime that already manages those lifetimes,
//! and call [`LogicalRegistry::attach`] / [`LogicalRegistry::ingest`] /
//! [`LogicalRegistry::detach`].
//!
//! Three invariants shape the API:
//!
//! * Unknown is not zero. A view that is missing, stale or unsupported yields
//!   `None`, never `Some(0)`, so a caller cannot read a lost view as an empty
//!   cache and route away from a warm worker.
//! * Observe never routes. In `Observe` the registry answers counterfactual
//!   questions and refuses to answer the routing question at all, so an
//!   observation cannot become a dispatch by accident.
//! * One owner cannot spend everyone's budget. The plan ceiling is aggregate
//!   across attached owners as well as per-view.

use std::collections::HashMap;

use crate::indexer::logical::Confidence;
use crate::protocols::{ResidencyOwner, WorkerWithDpRank};
use crate::zmq_wire::logical::{DecodeLimits, LogicalDecodeError, LogicalStream, query_tokens};

/// How far the logical CPU view is allowed to reach into routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LogicalMode {
    /// Nothing is decoded, held or reported. The default.
    #[default]
    Off,
    /// Views are maintained and can be inspected, but never answer a routing
    /// question. Use this to compare proposed choices against live ones.
    Observe,
    /// Views may additionally supply CPU-root overlap to the selector.
    Use,
}

/// One attached owner's answer for one request. `eligible_cpu_root_tokens` is
/// the only value a selector may consume, and only in [`LogicalMode::Use`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerOverlap {
    pub worker: WorkerWithDpRank,
    pub status: Confidence,
    /// `None` means unknown, never a measured zero.
    pub eligible_cpu_root_tokens: Option<u32>,
    /// Coverage actually matched. Diagnostic only: coverage never authorises an
    /// endpoint, and the gap between this and the eligible value is the whole
    /// point of the profile.
    pub raw_cpu_tokens: u32,
    /// The engine's own declared plan name, not the interning key.
    pub selected_plan: Option<String>,
    /// The producer's wire cursor for the cut this answer was read from, so a
    /// recorded decision can be reconciled against engine-side logs. Not the
    /// view's internal commit counter, which does not track the epoch.
    pub cursor: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionReason {
    /// The registry is Off, so it expresses no opinion at all.
    ModeDisabled,
    /// No attached owner among the candidates.
    NoCandidates,
    /// Every candidate was unknown or certified nothing reusable.
    NoLogicalCredit,
    /// A candidate certified the deepest reusable endpoint.
    MaximumEligibleEndpoint,
}

/// What the registry *would* choose, recorded without enqueue, capacity
/// reservation or dispatch. Producing this must not mutate any view.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CounterfactualSelection {
    pub proposed: Option<WorkerWithDpRank>,
    pub reason: SelectionReason,
    pub candidates: Vec<WorkerOverlap>,
}

#[derive(Debug, Clone, Copy)]
pub struct RegistryLimits {
    pub decode: DecodeLimits,
    /// Ceiling on live plans summed over every attached owner.
    pub aggregate_plans: usize,
    /// Ceiling on simultaneously attached owners.
    pub owners: usize,
}

impl Default for RegistryLimits {
    fn default() -> Self {
        Self {
            decode: DecodeLimits::default(),
            aggregate_plans: 262_144,
            owners: 1024,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RegistryError {
    #[error("logical cache registry is disabled")]
    Disabled,
    #[error("owner {0:?} is not attached")]
    NotAttached(WorkerWithDpRank),
    #[error("attached owner ceiling reached")]
    OwnerCeiling,
    #[error(transparent)]
    Decode(#[from] LogicalDecodeError),
    #[error("aggregate plan ceiling exceeded; this owner's view was retired")]
    AggregateCeiling,
}

pub struct LogicalRegistry {
    mode: LogicalMode,
    namespace: String,
    limits: RegistryLimits,
    streams: HashMap<WorkerWithDpRank, LogicalStream>,
}

impl LogicalRegistry {
    pub fn new(mode: LogicalMode, namespace: String, limits: RegistryLimits) -> Self {
        Self {
            mode,
            namespace,
            limits,
            streams: HashMap::new(),
        }
    }

    pub fn mode(&self) -> LogicalMode {
        self.mode
    }

    pub fn attached(&self) -> usize {
        self.streams.len()
    }

    /// Discovery found an authenticated producer for this rank.
    pub fn attach(&mut self, worker: WorkerWithDpRank) -> Result<(), RegistryError> {
        if self.mode == LogicalMode::Off {
            return Err(RegistryError::Disabled);
        }
        if !self.streams.contains_key(&worker) && self.streams.len() >= self.limits.owners {
            return Err(RegistryError::OwnerCeiling);
        }
        self.streams.entry(worker).or_insert_with(|| {
            LogicalStream::new(
                ResidencyOwner::worker(worker),
                self.namespace.clone(),
                self.limits.decode,
            )
        });
        Ok(())
    }

    /// The producer is gone. Its credit disappears with it; nothing is retained
    /// as a stale positive for a worker that may no longer hold the rows.
    pub fn detach(&mut self, worker: WorkerWithDpRank) {
        self.streams.remove(&worker);
    }

    /// The authenticated attachment was replaced. Transport sequences restart
    /// with the socket, so continuity is not inferred from the wire.
    pub fn reattach(&mut self, worker: WorkerWithDpRank) {
        if let Some(stream) = self.streams.get_mut(&worker) {
            stream.reattach();
        }
    }

    pub fn ingest(
        &mut self,
        worker: WorkerWithDpRank,
        transport_cursor: u64,
        payload: &[u8],
    ) -> Result<bool, RegistryError> {
        if self.mode == LogicalMode::Off {
            return Err(RegistryError::Disabled);
        }
        let Some(stream) = self.streams.get_mut(&worker) else {
            return Err(RegistryError::NotAttached(worker));
        };
        let accepted = stream.ingest(transport_cursor, payload)?;
        if accepted && self.aggregate_plans() > self.limits.aggregate_plans {
            // Retire the owner that just grew past the shared ceiling, not an
            // arbitrary victim: the others' views are still trustworthy.
            self.streams
                .get_mut(&worker)
                .expect("owner checked above")
                .invalidate();
            return Err(RegistryError::AggregateCeiling);
        }
        Ok(accepted)
    }

    /// Live plans summed over every attached owner, for metrics and for the
    /// shared ceiling below.
    pub fn live_plans(&self) -> usize {
        self.aggregate_plans()
    }

    fn aggregate_plans(&self) -> usize {
        self.streams
            .values()
            .filter_map(|stream| stream.view())
            .map(|view| view.plan_count())
            .sum()
    }

    fn overlap_of(&self, worker: WorkerWithDpRank, tokens: &[u32]) -> Option<WorkerOverlap> {
        let stream = self.streams.get(&worker)?;
        let view = stream.view()?;
        let result = query_tokens(view, tokens);
        let known = result.status == Confidence::Known;
        Some(WorkerOverlap {
            worker,
            status: result.status,
            eligible_cpu_root_tokens: known.then_some(result.eligible_cpu_root_tokens).flatten(),
            raw_cpu_tokens: result.raw_cpu_tokens,
            selected_plan: result.selected_plan.and_then(|id| {
                stream
                    .plan_names()
                    .and_then(|names| names.get(&id).cloned())
            }),
            cursor: stream.cursor().unwrap_or_default(),
        })
    }

    /// The only value a selector may consume, and only in [`LogicalMode::Use`].
    ///
    /// `None` covers every case where this registry has nothing certified to
    /// say: disabled, observing, not attached, no accepted baseline, or an
    /// unsupported query. A caller must fall back to its existing residency
    /// authority rather than reading it as an empty cache.
    pub fn routing_overlap(&self, worker: WorkerWithDpRank, tokens: &[u32]) -> Option<u32> {
        if self.mode != LogicalMode::Use {
            return None;
        }
        self.overlap_of(worker, tokens)?.eligible_cpu_root_tokens
    }

    /// Read-only counterfactual over the given candidates. Available in Observe
    /// as well as Use, and never touches any view's state.
    pub fn counterfactual(
        &self,
        candidates: &[WorkerWithDpRank],
        tokens: &[u32],
    ) -> CounterfactualSelection {
        if self.mode == LogicalMode::Off {
            return CounterfactualSelection {
                proposed: None,
                reason: SelectionReason::ModeDisabled,
                candidates: Vec::new(),
            };
        }
        let mut overlaps: Vec<WorkerOverlap> = candidates
            .iter()
            .filter_map(|worker| self.overlap_of(*worker, tokens))
            .collect();
        overlaps.sort_by_key(|overlap| overlap.worker);
        if overlaps.is_empty() {
            return CounterfactualSelection {
                proposed: None,
                reason: SelectionReason::NoCandidates,
                candidates: overlaps,
            };
        }
        // Deepest certified endpoint wins; ties go to the lowest worker so the
        // record is reproducible. An unknown view never competes.
        let best = overlaps
            .iter()
            .filter(|overlap| overlap.eligible_cpu_root_tokens.unwrap_or(0) > 0)
            .max_by_key(|overlap| {
                (
                    overlap.eligible_cpu_root_tokens.unwrap_or(0),
                    std::cmp::Reverse(overlap.worker),
                )
            })
            .map(|overlap| overlap.worker);
        CounterfactualSelection {
            reason: if best.is_some() {
                SelectionReason::MaximumEligibleEndpoint
            } else {
                SelectionReason::NoLogicalCredit
            },
            proposed: best,
            candidates: overlaps,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURES: &str = include_str!("../../tests/data/native-cpu-root-v1.json");

    fn cuts() -> (String, u32, Vec<Vec<u8>>) {
        // Minimal reader: the fixture file is the producer's own encoding, and
        // this module must be exercised against it rather than a hand-built map.
        let value: serde_json::Value = serde_json::from_str(FIXTURES).expect("fixture JSON");
        let namespace = value["namespace"].as_str().expect("namespace").to_string();
        let token = value["repeated_query_token"].as_u64().expect("token") as u32;
        let cuts = value["cuts"]
            .as_array()
            .expect("cuts")
            .iter()
            .map(|cut| {
                let hex = cut["payload_hex"].as_str().expect("hex");
                (0..hex.len())
                    .step_by(2)
                    .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).expect("hex"))
                    .collect()
            })
            .collect();
        (namespace, token, cuts)
    }

    fn worker(id: u64) -> WorkerWithDpRank {
        WorkerWithDpRank::new(id, 0)
    }

    /// Load every fixture cut into one owner, which ends in the anchored-tail
    /// state (eligible 23296 for a 23400-token prompt).
    fn loaded(mode: LogicalMode) -> (LogicalRegistry, Vec<u32>) {
        let (namespace, token, payloads) = cuts();
        let mut registry = LogicalRegistry::new(mode, namespace, RegistryLimits::default());
        registry.attach(worker(1)).expect("attach");
        // Stop before the CPU reset so the loaded state carries live plans.
        for (index, payload) in payloads.iter().take(payloads.len() - 1).enumerate() {
            registry
                .ingest(worker(1), index as u64, payload)
                .expect("ingest");
        }
        (registry, vec![token; 23400])
    }

    #[test]
    fn off_mode_holds_nothing_and_answers_nothing() {
        let (namespace, token, payloads) = cuts();
        let mut registry =
            LogicalRegistry::new(LogicalMode::Off, namespace, RegistryLimits::default());
        assert!(matches!(
            registry.attach(worker(1)),
            Err(RegistryError::Disabled)
        ));
        assert!(matches!(
            registry.ingest(worker(1), 0, &payloads[0]),
            Err(RegistryError::Disabled)
        ));
        let selection = registry.counterfactual(&[worker(1)], &[token; 16]);
        assert_eq!(selection.reason, SelectionReason::ModeDisabled);
        assert_eq!(registry.attached(), 0);
    }

    #[test]
    fn observe_mode_records_a_choice_but_refuses_to_route() {
        let (registry, tokens) = loaded(LogicalMode::Observe);
        let selection = registry.counterfactual(&[worker(1)], &tokens);
        assert_eq!(selection.proposed, Some(worker(1)));
        assert_eq!(selection.reason, SelectionReason::MaximumEligibleEndpoint);
        assert_eq!(
            selection.candidates[0].eligible_cpu_root_tokens,
            Some(23296)
        );
        assert!(selection.candidates[0].selected_plan.is_some());
        // The routing question gets no answer at all, not a zero.
        assert_eq!(registry.routing_overlap(worker(1), &tokens), None);
    }

    #[test]
    fn use_mode_supplies_the_certified_endpoint_only() {
        let (registry, tokens) = loaded(LogicalMode::Use);
        assert_eq!(registry.routing_overlap(worker(1), &tokens), Some(23296));
        // An unattached worker is unknown, which is not an empty cache.
        assert_eq!(registry.routing_overlap(worker(2), &tokens), None);
    }

    #[test]
    fn a_retired_view_is_unknown_rather_than_zero() {
        let (namespace, token, payloads) = cuts();
        let mut registry =
            LogicalRegistry::new(LogicalMode::Use, namespace, RegistryLimits::default());
        registry.attach(worker(1)).expect("attach");
        for (index, payload) in payloads.iter().take(payloads.len() - 1).enumerate() {
            registry
                .ingest(worker(1), index as u64, payload)
                .expect("ingest");
        }
        let tokens = vec![token; 23400];
        assert_eq!(registry.routing_overlap(worker(1), &tokens), Some(23296));
        registry.reattach(worker(1));
        assert_eq!(registry.routing_overlap(worker(1), &tokens), None);
        let selection = registry.counterfactual(&[worker(1)], &tokens);
        assert_eq!(selection.reason, SelectionReason::NoCandidates);
    }

    #[test]
    fn detaching_a_producer_drops_its_credit() {
        let (mut registry, tokens) = loaded(LogicalMode::Use);
        assert!(registry.routing_overlap(worker(1), &tokens).is_some());
        registry.detach(worker(1));
        assert_eq!(registry.attached(), 0);
        assert_eq!(registry.routing_overlap(worker(1), &tokens), None);
    }

    #[test]
    fn the_plan_ceiling_is_shared_across_owners_not_per_owner() {
        let (namespace, token, payloads) = cuts();
        // Exactly one owner's worth of headroom: the second owner to publish is
        // the one retired, and the first keeps its trustworthy view.
        let (single_owner, _) = loaded(LogicalMode::Use);
        let headroom = single_owner.live_plans();
        assert!(headroom > 0, "the fixture owner must hold live plans");
        let limits = RegistryLimits {
            aggregate_plans: headroom,
            ..RegistryLimits::default()
        };
        let mut registry = LogicalRegistry::new(LogicalMode::Use, namespace, limits);
        registry.attach(worker(1)).expect("attach");
        registry.attach(worker(2)).expect("attach");
        let through = payloads.len() - 1;
        for (index, payload) in payloads.iter().take(through).enumerate() {
            registry
                .ingest(worker(1), index as u64, payload)
                .expect("first owner");
        }
        let mut hit_ceiling = false;
        for (index, payload) in payloads.iter().take(through).enumerate() {
            if let Err(RegistryError::AggregateCeiling) =
                registry.ingest(worker(2), index as u64, payload)
            {
                hit_ceiling = true;
                break;
            }
        }
        let tokens = vec![token; 23400];
        assert!(
            hit_ceiling,
            "a second owner never reached the shared ceiling"
        );
        assert_eq!(registry.routing_overlap(worker(2), &tokens), None);
        assert_eq!(registry.routing_overlap(worker(1), &tokens), Some(23296));
    }

    #[test]
    fn the_owner_ceiling_bounds_attachment_churn() {
        let (namespace, _, _) = cuts();
        let limits = RegistryLimits {
            owners: 1,
            ..RegistryLimits::default()
        };
        let mut registry = LogicalRegistry::new(LogicalMode::Use, namespace, limits);
        registry.attach(worker(1)).expect("attach");
        registry
            .attach(worker(1))
            .expect("re-attaching a known owner is free");
        assert!(matches!(
            registry.attach(worker(2)),
            Err(RegistryError::OwnerCeiling)
        ));
    }

    #[test]
    fn a_counterfactual_never_mutates_a_view() {
        let (registry, tokens) = loaded(LogicalMode::Use);
        let first = registry.counterfactual(&[worker(1), worker(2)], &tokens);
        let second = registry.counterfactual(&[worker(1), worker(2)], &tokens);
        assert_eq!(first, second);
        assert_eq!(
            first.candidates.len(),
            1,
            "an unattached candidate is not reported as zero"
        );
    }
}
