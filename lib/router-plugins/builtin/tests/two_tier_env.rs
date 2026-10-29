// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exercise startup environment wiring through the public registry in isolated processes.

use std::process::{Command, Output};

use dynamo_kv_router::{KvRouterConfig, RoutingPartitionRef, WorkerType};

const ENV_NAMES: [&str; 5] = [
    "DYN_ROUTER_TWO_TIER_CACHE_THRESHOLD",
    "DYN_ROUTER_TWO_TIER_CACHE_THRESHOLD_INCLUSIVE",
    "DYN_ROUTER_TWO_TIER_BALANCE_ABS_THRESHOLD",
    "DYN_ROUTER_TWO_TIER_BALANCE_REL_THRESHOLD",
    "DYN_ROUTER_TWO_TIER_HOST_CACHE_WEIGHT",
];

fn startup(parameters: &str, overrides: &[(&str, &str)]) -> Output {
    let mut command = Command::new(std::env::current_exe().unwrap());
    command.args([
        "--ignored",
        "--exact",
        "resolve_policy_in_child",
        "--nocapture",
    ]);
    for name in ENV_NAMES {
        command.env_remove(name);
    }
    command
        .env("DYNAMO_TWO_TIER_TEST_PARAMETERS", parameters)
        .envs(overrides.iter().copied())
        .output()
        .expect("start isolated policy-resolution fixture")
}

#[test]
fn startup_uses_environment_over_yaml_and_attributes_host_weight() {
    for (parameters, overrides, expected_source, expected_weight) in [
        ("{}", vec![], "DYN_ROUTER_HOST_CACHE_HIT_WEIGHT", "0.25"),
        ("{host_cache_weight: 0.5}", vec![], "policy_yaml", "0.5"),
        (
            "{cache_threshold: 0.9, host_cache_weight: 0.5}",
            vec![
                (ENV_NAMES[0], "0.4"),
                (ENV_NAMES[1], "yes"),
                (ENV_NAMES[2], "16"),
                (ENV_NAMES[3], "1.5"),
                (ENV_NAMES[4], "1.0"),
            ],
            ENV_NAMES[4],
            "1",
        ),
    ] {
        let output = startup(parameters, &overrides);
        let logs = String::from_utf8(output.stdout).unwrap();
        assert!(output.status.success(), "{logs}");
        assert!(
            logs.contains(&format!("host_cache_weight_source=\"{expected_source}\"")),
            "{logs}"
        );
        assert!(
            logs.contains(&format!("host_cache_weight={expected_weight}")),
            "{logs}"
        );
        if !overrides.is_empty() {
            for expected in [
                "cache_threshold=0.4",
                "cache_threshold_inclusive=true",
                "balance_abs_threshold=16",
                "balance_rel_threshold=1.5",
                "env_overrides=",
            ] {
                assert!(logs.contains(expected), "missing {expected}: {logs}");
            }
        }
    }
}

#[test]
fn malformed_or_out_of_range_overrides_fail_startup_naming_the_variable() {
    for (name, value) in [
        (ENV_NAMES[0], "not-a-number"),
        (ENV_NAMES[0], "1.5"),
        (ENV_NAMES[0], "NaN"),
        (ENV_NAMES[1], "maybe"),
        (ENV_NAMES[2], "-1"),
        (ENV_NAMES[3], "0.5"),
        (ENV_NAMES[4], "-1"),
        (ENV_NAMES[4], "inf"),
    ] {
        let output = startup("{}", &[(name, value)]);
        let logs = format!(
            "{}\n{}",
            String::from_utf8(output.stdout).unwrap(),
            String::from_utf8(output.stderr).unwrap()
        );
        assert!(
            !output.status.success(),
            "{name}={value} resolved unexpectedly"
        );
        assert!(logs.contains(name), "{name}={value}: {logs}");
    }
}

#[test]
#[ignore = "subprocess fixture invoked by the startup tests"]
fn resolve_policy_in_child() {
    tracing_subscriber::fmt()
        .with_ansi(false)
        .with_max_level(tracing::Level::INFO)
        .init();
    let parameters = std::env::var("DYNAMO_TWO_TIER_TEST_PARAMETERS").unwrap();
    let file = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(
        file.path(),
        format!(
            "worker_selection:\n  aggregated: two-tier\n  instances:\n    - name: two-tier\n      type: dynamo-two-tier-cost-fn\n      parameters: {parameters}\n"
        ),
    )
    .unwrap();
    let config = KvRouterConfig {
        router_policy_config: Some(file.path().display().to_string()),
        host_cache_hit_weight: 0.25,
        ..Default::default()
    };
    let mut registry = dynamo_custom_policy_builtin::default_registry();
    dynamo_custom_policy_builtin::register(&mut registry).unwrap();
    let factory = registry.resolve(&config).unwrap().unwrap();
    factory(
        &config,
        WorkerType::Aggregated,
        RoutingPartitionRef::new("model", "default"),
    );
}
