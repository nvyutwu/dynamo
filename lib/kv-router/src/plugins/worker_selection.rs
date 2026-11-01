// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public filter, scorer, and picker contracts and their input signals.

mod config;
mod context;
mod inputs;
mod runtime;

pub(crate) use runtime::RuntimePolicyState;
pub use runtime::{
    RuntimePolicyBuilder, RuntimePolicyConfig, RuntimePolicyError, RuntimePolicyUpdate,
};

pub use context::WorkerSelectionContext;
pub(crate) use inputs::{CacheSnapshot, CandidateData, WorkerCacheData};
pub use inputs::{
    ScoredWorkerCandidate, WorkerCacheInput, WorkerCacheInputs, WorkerCandidate, WorkerCandidates,
    WorkerInputView, WorkerInputs, WorkerLoadInput, WorkerTransferInput,
};

pub use super::registry::{
    WorkerSelectionPolicyParameters, WorkerSelectionPolicyProvider,
    WorkerSelectionPolicyProviderError, WorkerSelectionPolicyRegistryError,
};
pub use crate::scheduling::selector::WorkerSelectionPolicy;
pub use crate::scheduling::{
    SessionContext, WorkerSelectionInputTrigger, WorkerSelectionPolicyError,
};
pub(crate) use config::RawWorkerSelectionConfig;
pub use config::{WorkerSelectionConfig, WorkerSelectionInstance};

use crate::{KvRouterConfig, RoutingPartitionRef, WorkerType};
use std::sync::Arc;

/// Factory that creates one worker-selection policy per routing partition.
pub type WorkerSelectionPolicyFactory = Arc<
    dyn for<'a> Fn(&KvRouterConfig, WorkerType, RoutingPartitionRef<'a>) -> WorkerSelectionPolicy
        + Send
        + Sync,
>;

/// Adds one finite cost contribution to each eligible worker.
pub trait WorkerScorer: Send {
    /// Declare the worker-signal groups needed by this scorer.
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::NONE
    }

    /// Score every candidate that survived host eligibility and policy filters in one call.
    /// The borrowed snapshot and its row order remain fixed for this selection. Each scorer
    /// can access only its own declared optional inputs, even when other components request more. Empty candidate
    /// sets skip scoring. Scorers run in declaration order; an error stops before picking.
    ///
    /// `costs` has the same length as `candidates` and is initialized to NaN before each call.
    /// Write one finite, lower-is-better contribution to every slot. The host validates and adds
    /// each contribution to that row's total. The candidate view and cost slice may not be retained after this call.
    fn score(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        candidates: WorkerCandidates<'_>,
        costs: &mut [f64],
    ) -> Result<(), WorkerSelectionPolicyError>;
}

/// Filters run in declaration order for each candidate. Callback order across different
/// candidates and scorers is unspecified; implementations must not depend on filters and scorers
/// being interleaved.
pub trait WorkerFilter: Send {
    /// Declare the worker-signal groups needed by this filter.
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::NONE
    }

    /// Return `true` to keep an eligible worker in the policy candidate set.
    fn keep(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        candidate: WorkerCandidate<'_>,
    ) -> Result<bool, WorkerSelectionPolicyError>;
}

/// What a picker chose and why, for the opt-in routing-decision trace.
///
/// The host records every candidate's cache and load inputs and summed scorer cost itself. A
/// policy explains its own rule here: which policy this is, which branch it took, the
/// parameters that were in force, and -- when it ranks overlap differently from the raw
/// `effective_overlap_blocks` -- its own per-row overlap score plus the row it considers the
/// max-overlap (oracle) worker. Everything is optional; the host fills whatever is missing
/// from the policy-independent candidate inputs.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PickExplanation {
    /// Policy identifier, e.g. the registered policy type.
    pub policy: String,
    /// Which branch of the policy produced the pick.
    pub reason: String,
    /// Row (in the picker's input order) the policy regards as holding the most reusable cache.
    pub max_overlap_row: Option<usize>,
    /// The policy's own overlap ranking per input row, in blocks. Empty = use the raw inputs.
    pub row_overlap: Vec<f64>,
    /// Resolved parameters and derived quantities, as (name, value) pairs. Names that match a
    /// top-level trace weight (`overlap_score_credit`, `overlap_score_credit_decay`,
    /// `prefill_load_scale`, `host_cache_hit_weight`, `disk_cache_hit_weight`,
    /// `shared_cache_multiplier`, `decode_active_request_weight`, `router_temperature`) also
    /// fill that field.
    pub parameters: Vec<(String, f64)>,
}

/// Selects one row after all filters and scorers run.
pub trait WorkerPicker: Send {
    /// Declare the optional worker-signal columns needed by this picker.
    fn required_worker_inputs(&self) -> WorkerInputs {
        WorkerInputs::NONE
    }

    /// Return one row index from the host-owned eligible candidate table. Row order is
    /// unspecified; inspect candidate data instead of relying on a stable position.
    fn pick(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError>;

    /// Explain a pick for the opt-in routing-decision trace.
    ///
    /// Called only for sampled requests with `DYN_ROUTER_DECISION_TRACE_ENABLED` set, right
    /// after `pick` returned `row` for this same `input`. Return `None` to leave the trace with
    /// the generic candidate inputs and a `custom_picker` reason.
    fn explain_pick(
        &self,
        _context: &WorkerSelectionContext<'_>,
        _input: WorkerInputView<'_>,
        _row: usize,
    ) -> Option<PickExplanation> {
        None
    }
}
