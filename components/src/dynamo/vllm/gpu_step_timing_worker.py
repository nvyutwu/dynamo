# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Experimental opt-in PP1 worker timer; qualification required before fitting."""

import json
import logging
import queue
import threading
import time
import uuid

import torch
from dynamo.runtime.logging import configure_dynamo_logging

from vllm.v1.worker.gpu_worker import Worker

configure_dynamo_logging()

logger = logging.getLogger(__name__)


class GpuStepTimingWorker(Worker):
    """Time execute_model through sample_tokens without synchronizing inference."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.parallel_config.pipeline_parallel_size != 1:
            raise ValueError(
                "GPU step timing qualification currently supports PP1 only"
            )
        self._timing_pending = None
        self._timing_queue = queue.Queue(maxsize=1024)
        self._timing_dropped = 0
        self._timing_step = 0
        self._timing_producer = uuid.uuid4().hex
        self._timing_prompts = {}
        self._timing_thread = None

    def execute_model(self, scheduler_output):
        for req_id in scheduler_output.finished_req_ids:
            self._timing_prompts.pop(req_id, None)
        if scheduler_output.total_num_scheduled_tokens:
            if self._timing_pending is not None:
                raise RuntimeError("Previous GPU timing step was not finalized")
            if self._timing_thread is None:
                self._timing_thread = threading.Thread(
                    target=self._export_timings, daemon=True
                )
                self._timing_thread.start()
            computed = {}
            for req in scheduler_output.scheduled_new_reqs:
                self._timing_prompts[req.req_id] = (
                    len(req.prompt_token_ids)
                    if req.prompt_token_ids is not None
                    else None
                )
                computed[req.req_id] = req.num_computed_tokens
            cached = scheduler_output.scheduled_cached_reqs
            computed.update(zip(cached.req_ids, cached.num_computed_tokens))
            requests = []
            for req_id, tokens in scheduler_output.num_scheduled_tokens.items():
                requests.append(
                    {
                        "request_id": req_id,
                        "scheduled_tokens": tokens,
                        "computed_tokens": computed.get(req_id),
                        "prompt_tokens": self._timing_prompts.get(req_id),
                    }
                )
            row = {
                "event": "dynamo_gpu_step_timing",
                "schema_version": 1,
                "producer_id": self._timing_producer,
                "rank": self.rank,
                "step_id": self._timing_step,
                "requests": requests,
                "timing_scope": "cuda_current_stream_execute_and_sample",
                "started_at_unix_ns": time.time_ns(),
            }
            self._timing_step += 1
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            self._timing_pending = (start, end, row)
        result = super().execute_model(scheduler_output)
        if result is not None:
            self._finish_timing()
        return result

    def sample_tokens(self, grammar_output):
        result = super().sample_tokens(grammar_output)
        self._finish_timing()
        return result

    def _finish_timing(self):
        if self._timing_pending is None:
            return
        start, end, row = self._timing_pending
        end.record()
        self._timing_pending = None
        try:
            self._timing_queue.put_nowait((start, end, row))
        except queue.Full:
            self._timing_dropped += 1

    def _export_timings(self):
        while True:
            start, end, row = self._timing_queue.get()
            try:
                while not end.query():
                    time.sleep(0.005)
                row["gpu_time_ms"] = start.elapsed_time(end)
                row["exported_at_unix_ns"] = time.time_ns()
                row["queue_dropped_total"] = self._timing_dropped
                logger.info(
                    "GPU_STEP_TIMING %s",
                    json.dumps(row, separators=(",", ":"), allow_nan=False),
                )
            except Exception:
                logger.exception("GPU step timing export failed")
