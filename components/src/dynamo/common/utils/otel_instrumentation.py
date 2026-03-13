# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
OpenTelemetry metrics bridge for Dynamo.

Scrapes Prometheus /metrics endpoint and exports via OTLP to an OTEL collector.
This module is intentionally defensive: if OpenTelemetry packages are missing
or a specific exporter is not configured, it falls back gracefully without
raising, and returns None where appropriate.

Adapted from vLLM's otel_instrumentation.py.

Usage:
    from dynamo.common.utils.otel_instrumentation import init_dynamo_otel_metrics

    # After HTTP server is configured (call once at startup):
    init_dynamo_otel_metrics(http_port=8000)

Environment Variables:
    OTEL_EXPORTER_OTLP_METRICS_ENDPOINT - OTEL collector gRPC endpoint
    OTEL_EXPORTER_OTLP_METRICS_PROTOCOL - Protocol: grpc (default) or http/protobuf
    OTEL_SERVICE_NAME - Service name (default: dynamo)
    OTEL_SERVICE_INSTANCE_ID - Instance ID (default: hostname)
    DYNAMO_PROM_SCRAPE_INTERVAL - Scrape interval in seconds (default: 30)
    DYNAMO_PROM_SCRAPE_TARGET - Override scrape URL
    OTEL_HEALTH_CHECK_ENDPOINT - Optional collector health check URL
    OTEL_HEALTH_CHECK_WAIT - Wait for collector (default: 0)
"""

from __future__ import annotations

import logging
import os
import socket
import sys
import threading
import time
import urllib.request
from typing import Any, Dict, Optional, Tuple

try:
    from prometheus_client.parser import text_string_to_metric_families
except ImportError:
    text_string_to_metric_families = None

logger = logging.getLogger(__name__)

# Global handles for program-wide access
_GLOBAL_METER = None
_PROM_BRIDGE_THREAD = None
_PROM_COUNTER_PREV: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], float] = {}
_PROM_BUCKET_PREV: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], float] = {}
_PROM_COUNTERS: Dict[str, Any] = {}
_PROM_GAUGE_VALUES: Dict[str, Dict[Tuple[Tuple[str, str], ...], float]] = {}
_PROM_GAUGES: Dict[str, Any] = {}


def _is_truthy(value: Optional[str]) -> bool:
    """Check if environment variable value is truthy."""
    if value is None:
        return False
    value = value.strip().lower()
    return value in ("1", "true", "yes", "on")


def _check_collector_health() -> bool:
    """Check if OTEL collector is healthy."""
    health_url = os.getenv("OTEL_HEALTH_CHECK_ENDPOINT")
    if not health_url:
        return True  # Skip health gate if unspecified
    try:
        import requests
    except ImportError:
        logger.warning("requests is not available; skipping OTEL collector health-check")
        return True

    timeout_s = float(os.getenv("OTEL_HEALTH_CHECK_TIMEOUT", "10"))
    try:
        resp = requests.get(health_url, timeout=timeout_s)
        return resp.status_code == 200
    except Exception as e:
        logger.warning("OTEL collector health-check failed: %s", e)
        return False


def _maybe_wait_for_collector():
    """Optionally wait for OTEL collector to be healthy before proceeding."""
    if not _is_truthy(os.getenv("OTEL_HEALTH_CHECK_WAIT", "0")):
        return
    max_retries = int(os.getenv("OTEL_HEALTH_CHECK_RETRIES", "12"))
    backoff_s = float(os.getenv("OTEL_HEALTH_CHECK_BACKOFF", "5"))
    retries = 0
    while retries <= max_retries:
        if _check_collector_health():
            logger.info("OTEL collector is healthy")
            return
        retries += 1
        if retries > max_retries:
            logger.warning("OTEL collector not healthy after %d retries", max_retries)
            return
        logger.info("Waiting for OTEL collector... (attempt %d/%d)", retries, max_retries)
        time.sleep(backoff_s)


def _get_otlp_exporter(protocol: str, exporter_type: str, exporter_class: str):
    """Dynamically import and instantiate OTLP exporter."""
    import importlib
    module_name = f"opentelemetry.exporter.otlp.proto.{protocol}.{exporter_type}"
    module = importlib.import_module(module_name)
    return getattr(module, exporter_class)()


def _get_metric_exporter():
    """Get OTLP metric exporter based on protocol configuration."""
    protocol = os.getenv("OTEL_EXPORTER_OTLP_METRICS_PROTOCOL", "grpc").lower()
    return _get_otlp_exporter(protocol, "metric_exporter", "OTLPMetricExporter")


def init_otel(
    resource_attributes: Optional[dict] = None
) -> Tuple[Optional[object], Optional[object]]:
    """
    Initialize OTEL metrics exporter.

    Args:
        resource_attributes: Optional dict of OTEL resource attributes.
            Common keys: service.name, service.instance.id, service.version

    Returns:
        Tuple of (meter, meter_provider) or (None, None) if initialization fails.
    """
    try:
        _maybe_wait_for_collector()

        from opentelemetry import metrics as otel_metrics
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource
    except ImportError as e:
        logger.debug("OpenTelemetry not available: %s", e)
        return None, None

    # Build resource attributes
    res_attrs = resource_attributes or {}
    if "service.name" not in res_attrs:
        res_attrs["service.name"] = os.getenv("OTEL_SERVICE_NAME", "dynamo")
    if "service.instance.id" not in res_attrs:
        res_attrs["service.instance.id"] = os.getenv(
            "OTEL_SERVICE_INSTANCE_ID", socket.gethostname()
        )
    resource = Resource.create(res_attrs)

    meter = None
    meter_provider = None
    initialized_components = []

    # Metrics
    metric_endpoint = os.getenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT")
    if metric_endpoint:
        try:
            metric_exporter = _get_metric_exporter()
            metric_reader = PeriodicExportingMetricReader(metric_exporter)
            meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
            otel_metrics.set_meter_provider(meter_provider)
            meter = otel_metrics.get_meter("dynamo")

            # Expose meter globally for other modules
            global _GLOBAL_METER
            _GLOBAL_METER = meter
            initialized_components.append(f"metrics -> {metric_endpoint}")
        except Exception as e:
            logger.warning("Failed to initialize OTEL metrics: %s", e)

    # Print diagnostic info
    if initialized_components:
        msg = (
            f"OpenTelemetry initialized successfully!\n"
            f"  Service: {res_attrs.get('service.name', 'unknown')} "
            f"(instance: {res_attrs.get('service.instance.id', 'unknown')})\n"
            f"  Components:\n"
        )
        for comp in initialized_components:
            msg += f"    - {comp}\n"
        print(msg, file=sys.stderr, flush=True)

    return meter, meter_provider


def get_otel_meter():
    """Return the global OTEL meter if initialized, else None."""
    return _GLOBAL_METER


# Prefixes to skip entirely (standard Prometheus collector metrics, not LLM-relevant)
_SKIP_METRIC_PREFIXES = (
    "process_",      # process_cpu_seconds_total, process_resident_memory_bytes, etc.
    "python_",       # python_gc_collections_total, python_info, etc.
    "go_",           # go_goroutines, go_memstats_*, etc. (if using Go-based exporters)
    "promhttp_",     # promhttp_metric_handler_requests_total, etc.
)

# Backend prefixes to strip and convert to labels
# Maps prefix -> backend label value
_BACKEND_PREFIXES: Dict[str, str] = {
    "vllm:": "vllm",           # vLLM uses colon separator
    "vllm_": "vllm",           # vLLM underscore variant (after colon replacement)
    "trtllm_": "trtllm",       # TensorRT-LLM
    "sglang_": "sglang",       # SGLang
    "sglang:": "sglang",       # SGLang colon variant (if any)
    "lmcache_": "lmcache",     # LMCache metrics
    "lmcache:": "lmcache",     # LMCache colon variant
}

# Dynamo component prefixes (strip but don't add backend label - these are Dynamo-native)
_DYNAMO_PREFIXES = (
    "dynamo_frontend_",
    "dynamo_router_",
    "dynamo_component_",
)


def _should_skip_metric(name: str) -> bool:
    """Check if metric should be skipped (standard Prometheus collector metrics)."""
    for prefix in _SKIP_METRIC_PREFIXES:
        if name.startswith(prefix):
            return True
    return False


def _detect_backend(name: str) -> Optional[str]:
    """Detect backend type from metric name prefix.

    Returns the backend label value (e.g., 'vllm', 'trtllm', 'sglang') or None.
    """
    # Normalize colon to underscore for consistent matching
    normalized = name.replace(":", "_")

    for prefix, backend in _BACKEND_PREFIXES.items():
        norm_prefix = prefix.replace(":", "_")
        if normalized.startswith(norm_prefix):
            return backend
    return None


def _strip_prefix(name: str) -> str:
    """Strip backend or Dynamo prefix from metric name."""
    # Normalize colon to underscore
    normalized = name.replace(":", "_")

    # Check backend prefixes first
    for prefix in _BACKEND_PREFIXES.keys():
        norm_prefix = prefix.replace(":", "_")
        if normalized.startswith(norm_prefix):
            return normalized[len(norm_prefix):]

    # Check Dynamo prefixes
    for prefix in _DYNAMO_PREFIXES:
        if normalized.startswith(prefix):
            return normalized[len(prefix):]

    return normalized


def _sanitize_metric_name(name: str) -> str:
    """Sanitize Prometheus metric name for OTEL export.

    Strips backend prefixes (vllm_, trtllm_, sglang_) and Dynamo prefixes.
    Backend information is preserved via the 'backend' label added separately.
    """
    return _strip_prefix(name)


def _sanitize_histogram_metric_name(raw_name: str, ftype: str) -> str:
    """Sanitize Prometheus histogram sample name for OTEL.

    Prometheus histograms expose _bucket, _count, _sum suffixes.
    Per OpenMetrics spec, these suffixes are type-specific:
      - _total belongs to Counter type only
      - _count, _sum, _bucket belong to Histogram type
    We preserve the original suffixes after prefix stripping.
    """
    return _sanitize_metric_name(raw_name)


def _labels_to_attributes(labels: Dict[str, str], backend: Optional[str] = None) -> Dict[str, str]:
    """Convert Prometheus labels to OTEL attributes, optionally adding backend label."""
    attrs = {str(k): str(v) for k, v in labels.items()}
    if backend:
        attrs["backend"] = backend
    return attrs


def start_prom_to_otel_bridge(scrape_url: str, interval_seconds: float = 30.0) -> bool:
    """
    Start a background thread that scrapes Prometheus metrics and exports via OTEL.

    Args:
        scrape_url: URL of the Prometheus /metrics endpoint to scrape.
        interval_seconds: How often to scrape (default: 30s).

    Returns:
        True if bridge started successfully, False otherwise.
    """
    global _PROM_BRIDGE_THREAD

    if text_string_to_metric_families is None:
        logger.warning("prometheus_client not available; OTEL bridge not started")
        return False

    if _PROM_BRIDGE_THREAD is not None:
        logger.warning("Prometheus->OTEL bridge already running")
        return False

    meter = get_otel_meter()
    if meter is None:
        logger.debug("OTEL meter not initialized; bridge not started")
        return False

    def _ensure_gauge(name: str):
        if name in _PROM_GAUGES:
            return
        values_ref = _PROM_GAUGE_VALUES.setdefault(name, {})

        def _callback(observer):
            obs_list = []
            try:
                from opentelemetry.metrics import Observation
            except ImportError:
                return []
            for label_items, val in list(values_ref.items()):
                attrs = {k: v for k, v in label_items}
                obs_list.append(Observation(val, attrs))
            return obs_list

        try:
            inst = meter.create_observable_gauge(name, callbacks=[_callback])
            _PROM_GAUGES[name] = inst
            logger.debug("OTEL gauge created: %s", name)
        except Exception as e:
            _PROM_GAUGES[name] = None
            logger.debug("OTEL gauge creation failed: %s: %s", name, e)

    def _ensure_counter(name: str):
        if name in _PROM_COUNTERS:
            return
        try:
            _PROM_COUNTERS[name] = meter.create_counter(name)
            logger.debug("OTEL counter created: %s", name)
        except Exception as e:
            _PROM_COUNTERS[name] = None
            logger.debug("OTEL counter creation failed: %s: %s", name, e)

    def _add_counter_delta(
        name: str,
        labels: Dict[str, str],
        value: float,
        prev_map: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], float],
        backend: Optional[str] = None,
    ):
        # Include backend in the key for proper delta tracking per backend
        augmented_labels = dict(labels)
        if backend:
            augmented_labels["backend"] = backend
        key = (name, tuple(sorted(augmented_labels.items())))
        prev = prev_map.get(key, 0.0)
        delta = value - prev
        if delta < 0:
            delta = value  # Counter reset
        prev_map[key] = value
        if delta <= 0:
            return
        inst = _PROM_COUNTERS.get(name)
        if inst is None:
            return
        try:
            inst.add(delta, attributes=_labels_to_attributes(labels, backend))
        except Exception:
            pass

    _scrape_count = 0

    def _scrape_once():
        nonlocal _scrape_count
        _scrape_count += 1
        _is_first = _scrape_count <= 2  # Log details for first 2 scrapes

        try:
            with urllib.request.urlopen(scrape_url, timeout=5) as resp:
                text = resp.read().decode("utf-8", errors="replace")
        except Exception as e:
            logger.debug("Failed to scrape %s: %s", scrape_url, e)
            return

        metrics_seen = {"counter": 0, "gauge": 0, "histogram": 0, "summary": 0, "skipped": 0}

        for family in text_string_to_metric_families(text):
            ftype = family.type or ""
            for sample in family.samples:
                raw_name = sample.name

                # Skip standard Prometheus collector metrics (process_, python_, etc.)
                if _should_skip_metric(raw_name):
                    metrics_seen["skipped"] += 1
                    continue

                # Detect backend from metric prefix (vllm_, trtllm_, sglang_)
                backend = _detect_backend(raw_name)

                # Sanitize metric name (strip prefixes)
                name = _sanitize_histogram_metric_name(raw_name, ftype)
                labels = sample.labels or {}
                value = float(sample.value or 0.0)

                if _is_first:
                    logger.debug(
                        "OTEL bridge: %s (%s) -> %s (backend=%s, value=%.4g)",
                        raw_name, ftype, name, backend, value,
                    )

                if ftype == "counter":
                    metrics_seen["counter"] += 1
                    _ensure_counter(name)
                    _add_counter_delta(name, labels, value, _PROM_COUNTER_PREV, backend)
                elif ftype == "gauge":
                    metrics_seen["gauge"] += 1
                    _ensure_gauge(name)
                    if name in _PROM_GAUGE_VALUES:
                        # Include backend in gauge labels
                        augmented_labels = dict(labels)
                        if backend:
                            augmented_labels["backend"] = backend
                        _PROM_GAUGE_VALUES[name][tuple(sorted(augmented_labels.items()))] = value
                elif ftype == "histogram":
                    metrics_seen["histogram"] += 1
                    # Export _count, _sum, _bucket as counters
                    if raw_name.endswith("_count"):
                        _ensure_counter(name)
                        _add_counter_delta(name, labels, value, _PROM_COUNTER_PREV, backend)
                    elif raw_name.endswith("_sum"):
                        _ensure_counter(name)
                        _add_counter_delta(name, labels, value, _PROM_COUNTER_PREV, backend)
                    elif raw_name.endswith("_bucket"):
                        _ensure_counter(name)
                        _add_counter_delta(name, labels, value, _PROM_BUCKET_PREV, backend)
                elif ftype == "summary":
                    metrics_seen["summary"] += 1
                    if raw_name.endswith("_count") or raw_name.endswith("_sum"):
                        _ensure_counter(name)
                        _add_counter_delta(name, labels, value, _PROM_COUNTER_PREV, backend)

        logger.info(
            "OTEL bridge scrape #%d: %d counters, %d gauges, %d histogram samples, "
            "%d summary samples, %d skipped (%d OTEL counters, %d OTEL gauges registered)",
            _scrape_count,
            metrics_seen["counter"],
            metrics_seen["gauge"],
            metrics_seen["histogram"],
            metrics_seen["summary"],
            metrics_seen["skipped"],
            len(_PROM_COUNTERS),
            len(_PROM_GAUGES),
        )

    def _run():
        logger.info("Prometheus->OTEL bridge started: scraping %s every %.1fs", scrape_url, interval_seconds)
        while True:
            try:
                _scrape_once()
            except Exception as e:
                logger.debug("Bridge scrape error: %s", e)
            time.sleep(interval_seconds)

    _PROM_BRIDGE_THREAD = threading.Thread(
        target=_run, name="dynamo-prom-otel-bridge", daemon=True
    )
    _PROM_BRIDGE_THREAD.start()
    return True


def init_dynamo_otel_metrics(http_port: int = 8000, service_namespace: str = "auto") -> bool:
    """
    Initialize OTEL metrics for a Dynamo component (frontend or worker).

    This is a convenience function that combines init_otel() and start_prom_to_otel_bridge().
    Call this after the HTTP/metrics server is configured but before serving requests.

    Works for both:
    - Frontend: scrapes frontend's /metrics on --http-port
    - Worker: scrapes worker's /metrics on DYN_SYSTEM_PORT

    Each runs in a separate process, so global state (meter, bridge thread) is independent.

    Args:
        http_port: Port where the component's /metrics endpoint is available.
        service_namespace: OTEL resource attribute. "auto" detects from caller context.

    Returns:
        True if OTEL metrics were initialized and bridge started, False otherwise.
    """
    # Only initialize if OTEL endpoint is configured
    if not os.getenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT"):
        logger.debug("OTEL metrics not enabled (OTEL_EXPORTER_OTLP_METRICS_ENDPOINT not set)")
        return False

    # Auto-detect namespace from caller context
    if service_namespace == "auto":
        # If DYN_SYSTEM_PORT matches http_port, we're in a worker process
        system_port = os.getenv("DYN_SYSTEM_PORT", "-1")
        if system_port not in ("-1", "0", "") and int(system_port) == http_port:
            service_namespace = "dynamo.worker"
        else:
            service_namespace = "dynamo.frontend"

    # Initialize OTEL with Dynamo-specific resource attributes
    meter, _ = init_otel({
        "service.name": os.getenv("OTEL_SERVICE_NAME", "dynamo"),
        "service.instance.id": os.getenv("OTEL_SERVICE_INSTANCE_ID", socket.gethostname()),
        "service.version": os.getenv("DYNAMO_VERSION", "unknown"),
        "service.namespace": service_namespace,
    })

    if meter is None:
        logger.debug("OTEL meter not initialized")
        return False

    # Start Prometheus -> OTEL bridge
    scrape_url = os.getenv("DYNAMO_PROM_SCRAPE_TARGET") or f"http://localhost:{http_port}/metrics"
    interval_s = float(os.getenv("DYNAMO_PROM_SCRAPE_INTERVAL", "30"))

    return start_prom_to_otel_bridge(scrape_url, interval_seconds=interval_s)
