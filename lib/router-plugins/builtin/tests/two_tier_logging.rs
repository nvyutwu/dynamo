// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The two-tier policy must announce itself in the log.
//!
//! This lives in its own integration-test binary on purpose. `tracing` keeps a PROCESS-WIDE max
//! level filter, so a scoped subscriber in one unit test is silenced by unrelated tests running in
//! parallel in the same process -- the assertions below pass alone and see an empty buffer beside
//! the rest of the suite. One test per binary removes the interference.
//!
//! Why the assertions matter: until now nothing in the router named the active worker-selection
//! policy. The only line that looked like it did --
//! `Router policy class configured policy_class="default"` -- is the QUEUEING profile and reads
//! "default" whether or not this policy is loaded. Two separate investigations read it as proof
//! two-tier had not loaded; one shipped that claim in an A/B report. These assertions are what
//! make the replacement line trustworthy as the QA signal.

use std::io::Write;
use std::sync::{Arc, Mutex};

use dynamo_kv_router::services::selection::WorkerSelectionPolicyRegistry;
use dynamo_kv_router::{KvRouterConfig, RoutingPartitionRef, WorkerType};
use tracing_subscriber::fmt::MakeWriter;

/// Exactly the file the K3 image bakes in via
///   COPY --chmod=0444 worker-selection.yaml /etc/dynamo/worker-selection-two-tier.yaml
/// No `parameters:` mapping, so every tunable takes its default and the host weight is inherited
/// from `DYN_ROUTER_HOST_CACHE_HIT_WEIGHT`. That is the shape actually deployed.
const BAKED_IMAGE_YAML: &str = r#"
worker_selection:
  aggregated: dynamo-two-tier-cost-fn
  instances:
    - name: dynamo-two-tier-cost-fn
      type: dynamo-two-tier-cost-fn
"#;

#[derive(Clone, Default)]
struct Capture(Arc<Mutex<Vec<u8>>>);

impl Write for Capture {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'a> MakeWriter<'a> for Capture {
    type Writer = Capture;
    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

#[test]
fn baked_image_yaml_resolves_and_announces_itself() {
    let cap = Capture::default();
    tracing::subscriber::set_global_default(
        tracing_subscriber::fmt()
            .with_writer(cap.clone())
            .with_ansi(false)
            .with_max_level(tracing::Level::INFO)
            .finish(),
    )
    .expect("this binary installs the only subscriber");

    let policy_file = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(policy_file.path(), BAKED_IMAGE_YAML).unwrap();
    let config = KvRouterConfig {
        router_policy_config: Some(policy_file.path().display().to_string()),
        ..Default::default()
    };

    // Resolve exactly as the Python bindings do at frontend startup.
    let mut registry = WorkerSelectionPolicyRegistry::default();
    dynamo_custom_policy_builtin::register(&mut registry).unwrap();
    let factory = registry
        .resolve(&config)
        .expect("baked YAML must resolve")
        .expect("baked YAML must produce a factory");

    // The baked YAML maps only `aggregated:` -- it declares no prefill/decode roles -- which
    // matches how K3 is deployed (prefill_worker_id == decode_worker_id on every traced request).
    // Construct that one role, which is what the router does.
    let _policy = factory(
        &config,
        WorkerType::Aggregated,
        RoutingPartitionRef::new("model", "default"),
    );

    let logs = String::from_utf8(cap.0.lock().unwrap().clone()).unwrap();

    assert!(
        logs.contains("Two-tier worker-selection policy enabled"),
        "no enablement line; QA has nothing to grep for.\n{logs}"
    );
    assert!(
        logs.contains("Two-tier worker-selection policy instantiated"),
        "no per-instance line.\n{logs}"
    );

    // Defaults must be VISIBLE, not implied. An omitted `parameters:` mapping is the normal
    // deployed case, and is exactly when an operator cannot otherwise tell what is in force.
    for field in [
        "cache_threshold=0.5",
        "balance_abs_threshold=32",
        "balance_rel_threshold=1.1",
    ] {
        assert!(logs.contains(field), "missing {field}\n{logs}");
    }

    // With no YAML override the weight must be attributed to the env var -- the knob an A/B turns.
    assert!(
        logs.contains(r#"host_cache_weight_source="DYN_ROUTER_HOST_CACHE_HIT_WEIGHT""#),
        "weight source not attributed to the env var.\n{logs}"
    );
    assert!(
        logs.contains(&format!(
            "host_cache_weight={}",
            config.host_cache_hit_weight
        )),
        "effective weight not logged.\n{logs}"
    );

    // One line per constructed role, naming the role, so a disaggregated deployment that builds
    // prefill and decode selectors separately stays auditable.
    assert_eq!(
        logs.matches("Two-tier worker-selection policy instantiated")
            .count(),
        1,
        "expected exactly one instantiation line for the single declared role.\n{logs}"
    );
    assert!(
        logs.contains(r#"worker_type="aggregated""#),
        "instantiation line must name the role it was built for.\n{logs}"
    );
}
