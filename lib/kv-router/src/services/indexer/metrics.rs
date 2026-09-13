// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[cfg(feature = "metrics")]
use std::sync::LazyLock;
#[cfg(feature = "metrics")]
use std::time::Instant;

#[cfg(feature = "metrics")]
use axum::{extract::MatchedPath, http::Request, middleware::Next, response::Response};
#[cfg(feature = "metrics")]
use prometheus::{
    HistogramVec, IntCounterVec, IntGauge, IntGaugeVec, Opts, exponential_buckets, histogram_opts,
};

#[cfg(feature = "metrics")]
use super::registry::ListenerStatus;

#[cfg(feature = "metrics")]
const METRICS_PREFIX: &str = "dynamo_kvindexer";
#[cfg(feature = "metrics")]
const REQUEST_DURATION_SECONDS: &str = "request_duration_seconds";
#[cfg(feature = "metrics")]
const REQUESTS_TOTAL: &str = "requests_total";
#[cfg(feature = "metrics")]
const ERRORS_TOTAL: &str = "errors_total";
#[cfg(feature = "metrics")]
const MODELS: &str = "models";
#[cfg(feature = "metrics")]
const WORKERS: &str = "workers";
#[cfg(feature = "metrics")]
const LISTENERS: &str = "listeners";
#[cfg(feature = "metrics")]
const DROPPED_BATCHES_TOTAL: &str = "dropped_batches_total";

#[cfg(feature = "metrics")]
pub struct StandaloneIndexerMetrics {
    pub request_duration: HistogramVec,
    pub requests_total: IntCounterVec,
    pub errors_total: IntCounterVec,
    pub models: IntGauge,
    pub workers: IntGauge,
    pub listeners: IntGaugeVec,
    pub dropped_batches_total: IntCounterVec,
}

#[cfg(feature = "metrics")]
static METRICS: LazyLock<StandaloneIndexerMetrics> = LazyLock::new(|| StandaloneIndexerMetrics {
    request_duration: HistogramVec::new(
        histogram_opts!(
            format!("{METRICS_PREFIX}_{REQUEST_DURATION_SECONDS}"),
            "HTTP request latency",
            exponential_buckets(0.0001, 2.0, 20).expect("valid bucket params")
        ),
        &["endpoint"],
    )
    .expect("valid histogram"),
    requests_total: IntCounterVec::new(
        Opts::new(
            format!("{METRICS_PREFIX}_{REQUESTS_TOTAL}"),
            "Total HTTP requests",
        ),
        &["endpoint", "method"],
    )
    .expect("valid counter"),
    errors_total: IntCounterVec::new(
        Opts::new(
            format!("{METRICS_PREFIX}_{ERRORS_TOTAL}"),
            "HTTP error responses (4xx/5xx)",
        ),
        &["endpoint", "status_class"],
    )
    .expect("valid counter"),
    models: IntGauge::new(
        format!("{METRICS_PREFIX}_{MODELS}"),
        "Number of active model+routing-group indexers",
    )
    .expect("valid gauge"),
    workers: IntGauge::new(
        format!("{METRICS_PREFIX}_{WORKERS}"),
        "Number of registered worker instances",
    )
    .expect("valid gauge"),
    listeners: IntGaugeVec::new(
        Opts::new(
            format!("{METRICS_PREFIX}_{LISTENERS}"),
            "Number of ZMQ listeners by status",
        ),
        &["status"],
    )
    .expect("valid gauge"),
    dropped_batches_total: IntCounterVec::new(
        Opts::new(
            format!("{METRICS_PREFIX}_{DROPPED_BATCHES_TOTAL}"),
            "ZMQ event batches dropped before indexing",
        ),
        &["reason"],
    )
    .expect("valid counter"),
});

#[cfg(feature = "metrics")]
pub fn register(registry: &prometheus::Registry) -> Result<(), prometheus::Error> {
    let m = &*METRICS;
    // Materialize the only bounded reason so a healthy zero is exported and
    // dashboard/alert consumers never have to infer zero from an absent family.
    m.dropped_batches_total.with_label_values(&["decode_error"]);
    registry.register(Box::new(m.request_duration.clone()))?;
    registry.register(Box::new(m.requests_total.clone()))?;
    registry.register(Box::new(m.errors_total.clone()))?;
    registry.register(Box::new(m.models.clone()))?;
    registry.register(Box::new(m.workers.clone()))?;
    registry.register(Box::new(m.listeners.clone()))?;
    registry.register(Box::new(m.dropped_batches_total.clone()))?;
    Ok(())
}

#[cfg(feature = "metrics")]
pub async fn metrics_middleware(req: Request<axum::body::Body>, next: Next) -> Response {
    let path = req
        .extensions()
        .get::<MatchedPath>()
        .map(|m| m.as_str().to_owned())
        .unwrap_or_else(|| "unknown".to_owned());
    let method = req.method().as_str().to_owned();
    let start = Instant::now();
    let response = next.run(req).await;
    let elapsed = start.elapsed().as_secs_f64();
    let m = &*METRICS;
    m.requests_total
        .with_label_values(&[path.as_str(), method.as_str()])
        .inc();
    m.request_duration
        .with_label_values(&[path.as_str()])
        .observe(elapsed);
    let status = response.status().as_u16();
    if status >= 400 {
        let class = if status < 500 { "4xx" } else { "5xx" };
        m.errors_total
            .with_label_values(&[path.as_str(), class])
            .inc();
    }
    response
}

#[cfg(feature = "metrics")]
pub fn set_worker_state(models: usize, workers: usize, listener_counts: [i64; 4]) {
    METRICS.models.set(models as i64);
    METRICS.workers.set(workers as i64);

    for status in ListenerStatus::ALL {
        METRICS
            .listeners
            .with_label_values(&[status.as_str()])
            .set(listener_counts[status.metric_index()]);
    }
}

#[cfg(not(feature = "metrics"))]
pub fn set_worker_state(_models: usize, _workers: usize, _listener_counts: [i64; 4]) {}

#[cfg(feature = "metrics")]
pub fn increment_decode_failed_batch() {
    METRICS
        .dropped_batches_total
        .with_label_values(&["decode_error"])
        .inc();
}

#[cfg(not(feature = "metrics"))]
pub fn increment_decode_failed_batch() {}

#[cfg(all(test, feature = "metrics"))]
pub fn decode_failed_batches_for_test() -> u64 {
    METRICS
        .dropped_batches_total
        .with_label_values(&["decode_error"])
        .get()
}

#[cfg(all(test, feature = "metrics"))]
mod tests {
    use super::*;
    use prometheus::Encoder;

    #[test]
    fn register_and_encode() {
        let registry = prometheus::Registry::new();
        register(&registry).expect("registration should succeed");

        set_worker_state(1, 2, [1, 1, 0, 0]);
        METRICS
            .request_duration
            .with_label_values(&["/health"])
            .observe(0.001);
        METRICS
            .requests_total
            .with_label_values(&["/health", "GET"])
            .inc();
        METRICS
            .errors_total
            .with_label_values(&["/query", "4xx"])
            .inc();

        let encoder = prometheus::TextEncoder::new();
        let mut buf = Vec::new();
        encoder.encode(&registry.gather(), &mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();

        assert!(output.contains("dynamo_kvindexer_request_duration_seconds"));
        assert!(output.contains("dynamo_kvindexer_requests_total"));
        assert!(output.contains("dynamo_kvindexer_errors_total"));
        assert!(output.contains("dynamo_kvindexer_models 1"));
        assert!(output.contains("dynamo_kvindexer_workers 2"));
        assert!(output.contains("dynamo_kvindexer_listeners{status=\"pending\"} 1"));
        assert!(output.contains("dynamo_kvindexer_listeners{status=\"active\"} 1"));
        assert!(output.contains("dynamo_kvindexer_dropped_batches_total"));
    }
}
