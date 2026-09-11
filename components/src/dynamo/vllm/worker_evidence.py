# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Emit metadata-only worker KV-event and offload-metric evidence as JSON lines."""

import atexit
import json
import logging
import math
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from collections import OrderedDict

SCHEMA = "dynamo.worker.evidence.v1"
ENABLED_ENV = "DYN_WORKER_EVIDENCE_ENABLED"
KV_EVENTS_ENDPOINT_ENV = "DYN_WORKER_EVIDENCE_KV_EVENTS_ENDPOINT"
METRICS_URL_ENV = "DYN_WORKER_EVIDENCE_METRICS_URL"
METRICS_INTERVAL_ENV = "DYN_WORKER_EVIDENCE_METRICS_INTERVAL_SECONDS"
METRICS_TIMEOUT_ENV = "DYN_WORKER_EVIDENCE_METRICS_TIMEOUT_SECONDS"
EVENT_QUIET_ENV = "DYN_WORKER_EVIDENCE_EVENT_QUIET_SECONDS"
MAX_MAPPINGS_ENV = "DYN_WORKER_EVIDENCE_MAX_HASH_MAPPINGS"
DEFAULT_MAX_MAPPINGS = 262_144
logger = logging.getLogger(__name__)
_collector_process = None
_collector_owner_pid = None
_cleanup_registered = False
REQUIRED_METRICS = (
    "vllm:kv_offload_load_bytes_total",
    "vllm:kv_offload_store_bytes_total",
    "vllm:kv_offload_cpu_cache_occupied_bytes",
    "vllm:kv_offload_cpu_cache_capacity_bytes",
)
OPTIONAL_METRICS = (
    "vllm:kv_offload_cpu_cache_occupancy_perc",
    "vllm:kv_offload_cpu_cache_usage_perc",
    "vllm:kv_offload_cpu_cache_read_usage_perc",
    "vllm:kv_offload_cpu_cache_write_usage_perc",
)
SAMPLE_RE = re.compile(
    r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)(?:\{(?P<labels>.*)\})?\s+"
    r"(?P<value>[-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?|[-+]?Inf|NaN)$"
)
LABEL_RE = re.compile(r'(?:^|,)\s*([A-Za-z_][A-Za-z0-9_]*)=("(?:\\.|[^"\\])*")')
SAFE_METRIC_LABELS = frozenset(("engine", "kv_cache_spec_kind", "model_name"))


def pod_identity(env=None):
    env = os.environ if env is None else env
    return {
        "name": env.get("POD_NAME") or env.get("HOSTNAME") or "unknown",
        "namespace": env.get("POD_NAMESPACE") or "unknown",
        "node": env.get("NODE_NAME") or "unknown",
        "group_index": env.get("DGD_GROUP_INDEX") or "unknown",
        "pod_index": env.get("DGD_POD_INDEX") or "unknown",
    }


def envelope(record_type, identity, fields=None):
    row = {
        "schema": SCHEMA,
        "record_type": record_type,
        "observed_unix_ms": int(time.time() * 1000),
        "pod": identity,
        # A SUB socket cannot prove that it saw batches published before its subscription.
        "capture_completeness": "unknown",
    }
    if fields:
        row.update(fields)
    return row


def emit(record_type, identity, fields=None):
    print(
        json.dumps(
            envelope(record_type, identity, fields),
            separators=(",", ":"),
            allow_nan=False,
        ),
        flush=True,
    )


def _labels(text):
    result = {}
    for match in LABEL_RE.finditer(text or ""):
        name = match.group(1)
        if name in SAFE_METRIC_LABELS:
            result[name] = json.loads(match.group(2))
    return dict(sorted(result.items()))


def parse_metrics(text):
    samples = []
    present = set()
    required = set(REQUIRED_METRICS)
    optional = set(OPTIONAL_METRICS)
    wanted = required | optional
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = SAMPLE_RE.match(line)
        if not match or match.group("name") not in wanted:
            continue
        value = float(match.group("value"))
        if not math.isfinite(value):
            value = match.group("value")
        samples.append(
            {
                "name": match.group("name"),
                "labels": _labels(match.group("labels")),
                "value": value,
            }
        )
        present.add(match.group("name"))
    return {
        "evidence_semantics": "worker_offload_metrics",
        "status": "ok" if required <= present else "pending_required_metrics",
        "samples": samples,
        "missing_required_metrics": sorted(required - present),
        "missing_optional_metrics": sorted(optional - present),
    }


def block_hash(value):
    if isinstance(value, bytes):
        return int.from_bytes(value, "little")
    return int(value) & ((1 << 64) - 1)


class IdentityMap(OrderedDict):
    """A bounded external-to-Dynamo block identity history."""

    def __init__(self, max_entries):
        if max_entries < 1:
            raise ValueError("worker evidence hash mapping limit must be positive")
        super().__init__()
        self.max_entries = max_entries
        self.evictions_total = 0

    def remember(self, key, value):
        if key in self:
            self.move_to_end(key)
        self[key] = value
        while len(self) > self.max_entries:
            self.popitem(last=False)
            self.evictions_total += 1


def sequence_observation(previous, current):
    if previous is None:
        return {
            "previous_sequence": None,
            "continuity": "unknown_before_first_observation",
            "missing_batches": None,
        }
    if current == previous + 1:
        return {
            "previous_sequence": previous,
            "continuity": "contiguous_observed",
            "missing_batches": 0,
        }
    if current > previous + 1:
        return {
            "previous_sequence": previous,
            "continuity": "gap_observed",
            "missing_batches": current - previous - 1,
        }
    return {
        "previous_sequence": previous,
        "continuity": "nonmonotonic_observed",
        "missing_batches": None,
    }


def mapped_hashes(event, identity_map, scope, compute_hash, chain):
    external = [str(block_hash(value)) for value in event.block_hashes]
    if type(event).__name__ == "BlockStored":
        if event.block_size != 12288:
            return None, "block_size_not_12288"
        if event.lora_name is not None:
            return None, "lora_hash_metadata_present"
        if event.extra_keys and any(value is not None for value in event.extra_keys):
            return None, "extra_hash_metadata_present"
        if any(
            type(value) is not int or value < 0 or value > 0xFFFFFFFF
            for value in event.token_ids
        ):
            return None, "token_shape_unsupported"
        if len(event.token_ids) != len(external) * event.block_size:
            return None, "token_block_alignment_unknown"
        parent = None
        if event.parent_block_hash is not None:
            parent_key = str(block_hash(event.parent_block_hash))
            parent = identity_map.get((*scope, parent_key))
            if parent is None:
                return None, "parent_mapping_unavailable"
        derived = []
        for index, external_hash in enumerate(external):
            start = index * event.block_size
            local = compute_hash(
                event.token_ids[start : start + event.block_size], event.block_size
            )[0]
            sequence_hash = local if parent is None else chain(parent, local)
            previous = identity_map.get((*scope, external_hash))
            if previous is not None and previous != sequence_hash:
                return None, "external_hash_mapping_changed"
            key = (*scope, external_hash)
            if hasattr(identity_map, "remember"):
                identity_map.remember(key, sequence_hash)
            else:
                identity_map[key] = sequence_hash
            derived.append(sequence_hash)
            parent = sequence_hash
        return derived, "mapped"
    derived = [identity_map.get((*scope, value)) for value in external]
    if any(value is None for value in derived):
        return None, "remove_mapping_unavailable"
    return derived, "mapped"


def event_record(event, identity_map, scope, compute_hash, chain):
    event_type = type(event).__name__
    row = {"event_type": event_type}
    if event_type in ("BlockStored", "BlockRemoved"):
        if event.group_idx != 0:
            derived, status = None, "routing_mapping_group_not_main"
        elif event.locality != "LOCAL":
            derived, status = None, "routing_mapping_locality_not_local"
        else:
            derived, status = mapped_hashes(
                event, identity_map, scope, compute_hash, chain
            )
        external = [str(block_hash(value)) for value in event.block_hashes]
        row.update(
            evidence_semantics="physical_tier_inventory",
            medium=event.medium,
            group_idx=event.group_idx,
            locality=event.locality,
            ownership=getattr(event, "ownership", None),
            external_block_hashes=external,
            parent_external_block_hash=(
                None
                if getattr(event, "parent_block_hash", None) is None
                else str(block_hash(event.parent_block_hash))
            ),
            block_count=len(external),
            dynamo_sequence_hashes=derived,
            hash_mapping_status=status,
            hash_mapping_entries=len(identity_map),
            hash_mapping_evictions_total=getattr(identity_map, "evictions_total", 0),
        )
    if event_type == "BlockStored":
        row.update(
            block_size=event.block_size,
            kv_cache_spec_kind=event.kv_cache_spec_kind,
            kv_cache_spec_sliding_window=event.kv_cache_spec_sliding_window,
        )
    if event_type == "AllBlocksCleared":
        row.update(
            evidence_semantics="physical_tier_inventory",
            scope_reset=True,
        )
    return row


def mapping_self_check(compute_hash, chain):
    stored_type = type("BlockStored", (), {})
    removed_type = type("BlockRemoved", (), {})
    store = stored_type()
    store.block_hashes = [101, 102]
    store.parent_block_hash = None
    store.token_ids = [1] * 12288 + [2] * 12288
    store.block_size = 12288
    store.lora_name = None
    store.extra_keys = None
    identity_map = {}
    scope = ("self-check", 0, 0)
    derived, status = mapped_hashes(store, identity_map, scope, compute_hash, chain)
    local = compute_hash(store.token_ids, 12288)
    expected = [local[0], chain(local[0], local[1])]
    if status != "mapped" or derived != expected:
        raise RuntimeError("multi-block external-to-Dynamo mapping self-check failed")
    removed = removed_type()
    removed.block_hashes = [101, 102]
    mapped, status = mapped_hashes(removed, identity_map, scope, compute_hash, chain)
    if status != "mapped" or mapped != expected:
        raise RuntimeError("removal mapping self-check failed")
    missing = removed_type()
    missing.block_hashes = [999]
    if (
        mapped_hashes(missing, identity_map, scope, compute_hash, chain)[1]
        != "remove_mapping_unavailable"
    ):
        raise RuntimeError("unknown removal mapping self-check failed")
    if (
        mapped_hashes(removed, identity_map, ("self-check", 1, 0), compute_hash, chain)[
            1
        ]
        != "remove_mapping_unavailable"
    ):
        raise RuntimeError("rank-scoped mapping self-check failed")
    child = stored_type()
    child.block_hashes = [103]
    child.parent_block_hash = 102
    child.token_ids = [3] * 12288
    child.block_size = 12288
    child.lora_name = None
    child.extra_keys = None
    child_hashes, status = mapped_hashes(
        child, identity_map, scope, compute_hash, chain
    )
    if status != "mapped" or child_hashes != [
        chain(expected[-1], compute_hash(child.token_ids, 12288)[0])
    ]:
        raise RuntimeError("parent-chain mapping self-check failed")


def run():
    import msgspec
    import zmq
    from dynamo._core import compute_block_hash_for_seq

    from vllm.distributed.kv_events import KVEventBatch

    identity = pod_identity()
    endpoint = os.environ.get(KV_EVENTS_ENDPOINT_ENV, "tcp://127.0.0.1:5557")
    metrics_url = os.environ.get(METRICS_URL_ENV, "http://127.0.0.1:9090/metrics")
    interval = max(0.25, float(os.environ.get(METRICS_INTERVAL_ENV, "1")))
    timeout = max(0.1, float(os.environ.get(METRICS_TIMEOUT_ENV, "2")))
    quiet_seconds = max(0.5, float(os.environ.get(EVENT_QUIET_ENV, "2")))
    max_mappings = int(os.environ.get(MAX_MAPPINGS_ENV, str(DEFAULT_MAX_MAPPINGS)))
    running = True

    def stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    context = zmq.Context.instance()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, b"")
    socket.connect(endpoint)
    decoder = msgspec.msgpack.Decoder(type=KVEventBatch)

    def chain(parent, local):
        words = [parent & 0xFFFFFFFF, parent >> 32, local & 0xFFFFFFFF, local >> 32]
        return compute_block_hash_for_seq(words, 4)[0]

    vector = compute_block_hash_for_seq([1, 2, 3, 4, 5, 6, 7, 8], 4)
    if vector != [14643705804678351452, 16777012769546811212]:
        raise RuntimeError("native Dynamo hash self-check failed")
    if chain(vector[0], vector[1]) != 4945711292740353085:
        raise RuntimeError("native Dynamo sequence-chain self-check failed")
    mapping_self_check(compute_block_hash_for_seq, chain)

    emit(
        "collector_ready",
        identity,
        {
            "event_endpoint": endpoint,
            "event_subscription": "configured_not_handshaken",
            "metrics_url": metrics_url,
            "metrics_port_source": "rendered_dgd_system_port_9090",
        },
    )
    identity_map = IdentityMap(max_mappings)
    latest_sequence = None
    last_event_at = None
    quiescence_emitted = False
    next_metrics = 0.0
    while running:
        now = time.monotonic()
        if now >= next_metrics:
            try:
                with urllib.request.urlopen(metrics_url, timeout=timeout) as response:
                    parsed = parse_metrics(
                        response.read().decode("utf-8", errors="replace")
                    )
                emit("metric_snapshot", identity, parsed)
            # A scrape failure is evidence; it must not stop KV-event collection.
            except Exception as error:  # noqa: BLE001
                emit(
                    "metric_error",
                    identity,
                    {
                        "status": "unavailable",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "samples": None,
                    },
                )
            next_metrics = time.monotonic() + interval
        if (
            last_event_at is not None
            and not quiescence_emitted
            and time.monotonic() - last_event_at >= quiet_seconds
        ):
            emit(
                "event_quiescence",
                identity,
                {
                    "latest_sequence": latest_sequence,
                    "quiet_seconds": time.monotonic() - last_event_at,
                    "transition_checkpoint": "observed_silence_only",
                },
            )
            quiescence_emitted = True
        try:
            if not socket.poll(200):
                continue
            frames = socket.recv_multipart()
            if len(frames) != 3:
                raise ValueError(f"expected 3 ZMQ frames, received {len(frames)}")
            topic, sequence, payload = frames
            batch = decoder.decode(payload)
            scope = (identity["name"], batch.data_parallel_rank)
            records = []
            for event in batch.events:
                event_scope = (*scope, getattr(event, "group_idx", None))
                if type(event).__name__ == "AllBlocksCleared":
                    for key in [key for key in identity_map if key[:2] == scope]:
                        del identity_map[key]
                records.append(
                    event_record(
                        event,
                        identity_map,
                        event_scope,
                        compute_block_hash_for_seq,
                        chain,
                    )
                )
            current_sequence = int.from_bytes(sequence, "big")
            continuity = sequence_observation(latest_sequence, current_sequence)
            emit(
                "kv_event_batch",
                identity,
                {
                    "topic": topic.decode("utf-8", errors="replace"),
                    "sequence": current_sequence,
                    "sequence_observation": continuity,
                    "data_parallel_rank": batch.data_parallel_rank,
                    "batch_timestamp": batch.ts,
                    "events": records,
                },
            )
            latest_sequence = current_sequence
            last_event_at = time.monotonic()
            quiescence_emitted = False
        # Keep malformed batches observable while continuing with later batches.
        except Exception as error:  # noqa: BLE001
            emit(
                "event_error",
                identity,
                {
                    "status": "decode_or_receive_failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
    socket.close(linger=0)
    emit("collector_stopped", identity)


def _enabled(value):
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("", "0", "false", "no", "off"):
        return False
    raise ValueError(f"{ENABLED_ENV} must be a boolean value")


def start_worker_evidence_collector():
    """Start one opt-in collector child without delaying worker initialization."""
    global _cleanup_registered, _collector_owner_pid, _collector_process
    if not _enabled(os.environ.get(ENABLED_ENV, "")):
        return None
    if _collector_process is not None and _collector_process.poll() is None:
        return _collector_process
    _collector_process = subprocess.Popen(
        [sys.executable, "-u", "-m", "dynamo.vllm.worker_evidence"],
        shell=False,
        stdin=subprocess.DEVNULL,
        start_new_session=False,
    )
    _collector_owner_pid = os.getpid()
    if not _cleanup_registered:
        atexit.register(stop_worker_evidence_collector)
        _cleanup_registered = True
    logger.info(
        "Started worker evidence collector process pid=%s", _collector_process.pid
    )
    return _collector_process


def stop_worker_evidence_collector():
    global _collector_owner_pid, _collector_process
    if _collector_owner_pid != os.getpid():
        return
    process = _collector_process
    _collector_process = None
    _collector_owner_pid = None
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def main():
    try:
        run()
    # Emit one terminal metadata record for any initialization/runtime failure.
    except Exception as error:  # noqa: BLE001
        identity = pod_identity()
        emit(
            "collector_fatal",
            identity,
            {
                "status": "disabled_after_initialization_failure",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
