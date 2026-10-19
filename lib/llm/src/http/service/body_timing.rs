// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request-body read timing for the inference routes.
//!
//! The chat handlers take the body as `Bytes`, so axum reads the whole body before the handler runs, and the time spent waiting for it
//! is not visible anywhere: the `http-request` span starts at the headers and the request tracker starts at the handler. With
//! `DYN_BODY_TIMING=1` this layer wraps the request body in a stream that records, for each request:
//!
//! * `first_poll_ms`, `first_byte_ms`, `last_byte_ms`: when the consumer first asked for data, and when the first and last chunks arrived,
//!   from the moment the body was handed to the layer (the headers have just been parsed);
//! * `data_wait_ms` / `max_data_wait_ms`: time the consumer was waiting for data to arrive (blocked on the sender), in total and the longest wait;
//! * `consumer_delay_ms` / `max_consumer_delay_ms`: time between returning a chunk and being polled for the next one (the consumer or the runtime
//!   being slow to ask for data; near zero when the body is simply being collected).
//!
//! One `request body read` log line is emitted per request inside the `http-request` span, so it carries the trace id. `complete=false` means the
//! body was dropped before its end (client disconnect, limit exceeded).

use std::pin::Pin;
use std::sync::LazyLock;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use axum::body::{Body, BodyDataStream, Bytes};
use axum::extract::Request;
use axum::middleware::Next;
use axum::response::Response;
use futures::Stream;

static ENABLED: LazyLock<bool> = LazyLock::new(|| dynamo_runtime::config::env_is_truthy("DYN_BODY_TIMING"));

/// Timing summary of one request body.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BodyStats {
    pub complete: bool,
    pub bytes: u64,
    pub frames: u64,
    pub first_poll: Option<Duration>,
    pub first_byte: Option<Duration>,
    pub last_byte: Option<Duration>,
    pub data_wait: Duration,
    pub max_data_wait: Duration,
    pub consumer_delay: Duration,
    pub max_consumer_delay: Duration,
}

type OnFinish = Box<dyn FnOnce(BodyStats) + Send>;

/// Stream wrapper that measures how a body is read. See the module docs.
pub struct TimedBody<S> {
    inner: S,
    created: Instant,
    last_ready: Option<Instant>,
    waiting_since: Option<Instant>,
    stats: BodyStats,
    on_finish: Option<OnFinish>,
}

impl<S> TimedBody<S> {
    pub fn new(inner: S, on_finish: OnFinish) -> Self {
        Self {
            inner,
            created: Instant::now(),
            last_ready: None,
            waiting_since: None,
            stats: BodyStats::default(),
            on_finish: Some(on_finish),
        }
    }

    fn finish(&mut self, complete: bool) {
        if let Some(cb) = self.on_finish.take() {
            self.stats.complete = complete;
            cb(self.stats.clone());
        }
    }
}

impl<S, E> Stream for TimedBody<S>
where
    S: Stream<Item = Result<Bytes, E>> + Unpin,
{
    type Item = Result<Bytes, E>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = &mut *self;
        if this.waiting_since.is_none() {
            let now = Instant::now();
            match this.last_ready {
                Some(last) => {
                    let delay = now - last;
                    this.stats.consumer_delay += delay;
                    this.stats.max_consumer_delay = this.stats.max_consumer_delay.max(delay);
                }
                None => this.stats.first_poll = Some(now - this.created),
            }
            this.waiting_since = Some(now);
        }
        match Pin::new(&mut this.inner).poll_next(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Some(Ok(chunk))) => {
                let now = Instant::now();
                if let Some(since) = this.waiting_since.take() {
                    let wait = now - since;
                    this.stats.data_wait += wait;
                    this.stats.max_data_wait = this.stats.max_data_wait.max(wait);
                }
                this.stats.first_byte.get_or_insert(now - this.created);
                this.stats.last_byte = Some(now - this.created);
                this.stats.bytes += chunk.len() as u64;
                this.stats.frames += 1;
                this.last_ready = Some(now);
                Poll::Ready(Some(Ok(chunk)))
            }
            Poll::Ready(None) => {
                this.finish(true);
                Poll::Ready(None)
            }
            Poll::Ready(Some(Err(e))) => {
                this.finish(false);
                Poll::Ready(Some(Err(e)))
            }
        }
    }
}

impl<S> Drop for TimedBody<S> {
    fn drop(&mut self) {
        self.finish(false);
    }
}

fn log_stats(s: BodyStats) {
    let ms = |d: Option<Duration>| d.map_or(-1.0, |d| d.as_secs_f64() * 1000.0);
    let dms = |d: Duration| d.as_secs_f64() * 1000.0;
    tracing::info!(
        complete = s.complete,
        bytes = s.bytes,
        frames = s.frames,
        first_poll_ms = format!("{:.1}", ms(s.first_poll)),
        first_byte_ms = format!("{:.1}", ms(s.first_byte)),
        last_byte_ms = format!("{:.1}", ms(s.last_byte)),
        data_wait_ms = format!("{:.1}", dms(s.data_wait)),
        max_data_wait_ms = format!("{:.1}", dms(s.max_data_wait)),
        consumer_delay_ms = format!("{:.1}", dms(s.consumer_delay)),
        max_consumer_delay_ms = format!("{:.1}", dms(s.max_consumer_delay)),
        "request body read"
    );
}

/// axum middleware: wrap the request body in [`TimedBody`] when `DYN_BODY_TIMING` is set.
pub async fn time_request_body(request: Request, next: Next) -> Response {
    if !*ENABLED {
        return next.run(request).await;
    }
    let (parts, body) = request.into_parts();
    let stream: BodyDataStream = body.into_data_stream();
    let timed = TimedBody::new(stream, Box::new(log_stats));
    next.run(Request::from_parts(parts, Body::from_stream(timed))).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use tokio::time::sleep;

    fn slow_stream(chunks: usize, size: usize, gap_ms: u64) -> impl Stream<Item = Result<Bytes, std::io::Error>> + Unpin {
        Box::pin(futures::stream::unfold(0usize, move |i| async move {
            if i >= chunks {
                return None;
            }
            if i > 0 {
                sleep(Duration::from_millis(gap_ms)).await;
            }
            Some((Ok(Bytes::from(vec![b'x'; size])), i + 1))
        }))
    }

    async fn run(chunks: usize, consumer_sleep_ms: u64, gap_ms: u64) -> BodyStats {
        use futures::StreamExt;
        let out = Arc::new(Mutex::new(None));
        let sink = out.clone();
        let mut t = TimedBody::new(slow_stream(chunks, 1000, gap_ms), Box::new(move |s| *sink.lock().unwrap() = Some(s)));
        while let Some(item) = t.next().await {
            item.unwrap();
            if consumer_sleep_ms > 0 {
                sleep(Duration::from_millis(consumer_sleep_ms)).await;
            }
        }
        drop(t);
        let stats = out.lock().unwrap().take().expect("stats emitted once");
        stats
    }

    #[tokio::test(start_paused = true)]
    async fn slow_sender_shows_as_data_wait() {
        let s = run(5, 0, 100).await;
        assert!(s.complete);
        assert_eq!((s.bytes, s.frames), (5000, 5));
        assert!(s.data_wait >= Duration::from_millis(400) && s.data_wait <= Duration::from_millis(410), "{:?}", s.data_wait);
        assert!(s.max_data_wait >= Duration::from_millis(100) && s.max_data_wait <= Duration::from_millis(110));
        assert!(s.consumer_delay < Duration::from_millis(5), "{:?}", s.consumer_delay);
        assert!(s.last_byte.unwrap() >= Duration::from_millis(400));
    }

    #[tokio::test(start_paused = true)]
    async fn slow_consumer_shows_as_consumer_delay() {
        let s = run(5, 100, 0).await;
        assert!(s.complete);
        assert!(s.consumer_delay >= Duration::from_millis(400) && s.consumer_delay <= Duration::from_millis(410), "{:?}", s.consumer_delay);
        assert!(s.max_consumer_delay >= Duration::from_millis(100));
        assert!(s.data_wait < Duration::from_millis(5), "{:?}", s.data_wait);
    }

    #[tokio::test(start_paused = true)]
    async fn dropped_body_reports_incomplete_once() {
        use futures::StreamExt;
        let out = Arc::new(Mutex::new(Vec::new()));
        let sink = out.clone();
        let mut t = TimedBody::new(slow_stream(5, 10, 10), Box::new(move |s| sink.lock().unwrap().push(s)));
        t.next().await.unwrap().unwrap();
        drop(t);
        let v = out.lock().unwrap();
        assert_eq!(v.len(), 1);
        assert!(!v[0].complete);
        assert_eq!(v[0].frames, 1);
    }
}
