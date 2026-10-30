// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python exception types mirroring Dynamo's [`ErrorType`] enum.
//!
//! The [`define_dynamo_exceptions!`] macro auto-generates a Python exception class
//! for each Dynamo error variant, a conversion function from Python exceptions back
//! to [`DynamoError`], and a registration function for the `_core` module.
//!
//! When new variants are added to [`ErrorType`] or [`BackendError`], add a
//! corresponding entry to the macro invocation below to keep Python exceptions
//! in sync.

use dynamo_runtime::error::{BackendError, DynamoError, ErrorClass, ErrorType};
use dynamo_runtime::protocols::annotated::Annotated;
use pyo3::prelude::*;
use pyo3::types::PyModule;

// Base exception for all Dynamo errors.
pyo3::create_exception!(dynamo._core, DynamoException, pyo3::exceptions::PyException);
pyo3::create_exception!(dynamo._core, RouterQueueLimitExceeded, DynamoException);

pub fn queue_rejection_to_pyerr(rejection: dynamo_kv_router::scheduling::QueueRejection) -> PyErr {
    let error = PyErr::new::<RouterQueueLimitExceeded, _>(rejection.to_string());
    let attributes = Python::with_gil(|py| -> PyResult<()> {
        let value = error.value(py);
        value.setattr("policy_class", &rejection.policy_class)?;
        value.setattr("limit_kind", rejection.limit_kind.to_string())?;
        value.setattr("current", rejection.current)?;
        value.setattr("limit", rejection.limit)?;
        Ok(())
    });
    if let Err(error) = attributes {
        return error;
    }
    error
}

// Raised by `RoutedEngine.generate` when the routed pipeline reports that no
// worker on the request's route can serve it yet (for example, prefill routing
// is still activating). Python chat processors re-raise it unchanged, and
// `map_python_exception` restores `ErrorClass::WorkerUnavailable` so the client
// gets the intended HTTP 503 instead of a generic 500.
pyo3::create_exception!(dynamo._core, WorkerUnavailable, DynamoException);

/// Convert a routed-pipeline error to a Python exception, keeping the
/// `WorkerUnavailable` and `InvalidArgument` class that the generic [`crate::to_pyerr`] would drop.
pub fn routed_error_to_pyerr(err: anyhow::Error) -> PyErr {
    if is_worker_unavailable(err.as_ref()) {
        return WorkerUnavailable::new_err(err.to_string());
    }
    if let Some(message) = public_invalid_argument_message(err.as_ref()) {
        return InvalidArgument::new_err(message);
    }
    crate::to_pyerr(err)
}

fn is_worker_unavailable(error: &(dyn std::error::Error + 'static)) -> bool {
    let mut current = Some(error);
    while let Some(cause) = current {
        // `class()` normalizes `WorkerUnavailable` to `Unavailable`; the raw
        // error type keeps the migratable worker-scoped distinction.
        if cause
            .downcast_ref::<DynamoError>()
            .is_some_and(|error| error.error_type() == ErrorClass::WorkerUnavailable)
        {
            return true;
        }
        current = cause.source();
    }
    false
}

pub(crate) fn worker_unavailable_to_dynamo(py: Python<'_>, err: &PyErr) -> Option<DynamoError> {
    if !err.is_instance_of::<WorkerUnavailable>(py) {
        return None;
    }
    let message = err
        .value(py)
        .str()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    Some(
        DynamoError::builder()
            .class(ErrorClass::WorkerUnavailable)
            .message(message)
            .build(),
    )
}

// Raised by the in-process `SelectionService` bindings for selector failures
// that are not malformed input. Instances carry `kind` (a stable,
// machine-readable category) and `status_code` (an HTTP-style status) so
// callers can branch without matching on the message string.
pyo3::create_exception!(dynamo._core, SelectionServiceError, DynamoException);

/// Defines Python exception classes for each Dynamo error type.
///
/// For each `(RustExceptionName, BackendError)` pair, the macro:
/// 1. Creates a Python exception class inheriting from `DynamoException`
/// 2. Adds it to `py_exception_to_backend_error()` for Python → `BackendError` extraction
/// 3. Adds it to `register_exceptions()` for module registration
///
/// The conversion intentionally returns a `BackendError` variant and message
/// rather than a fully constructed `DynamoError`. This lets the caller decide
/// how to wrap it — backend contexts use `ErrorType::Backend(...)`, while
/// other contexts could map to top-level `ErrorType` variants.
macro_rules! define_dynamo_exceptions {
    ( $( ($name:ident, $backend_error:expr) ),* $(,)? ) => {
        $(
            pyo3::create_exception!(dynamo._core, $name, DynamoException);
        )*

        /// Extract a [`BackendError`] variant from a Python exception if it is
        /// a known Dynamo exception.
        ///
        /// Returns `Some((BackendError, message))` if the exception is a Dynamo
        /// exception, `None` otherwise. The caller decides how to wrap the
        /// `BackendError` into an `ErrorType`.
        pub fn py_exception_to_backend_error(
            py: Python<'_>,
            err: &PyErr,
        ) -> Option<(BackendError, String)> {
            // Check specific subtypes first (most-specific match wins).
            $(
                if err.is_instance_of::<$name>(py) {
                    let message = err
                        .value(py)
                        .str()
                        .map(|s| s.to_string_lossy().into_owned())
                        .unwrap_or_default();
                    return Some(($backend_error, message));
                }
            )*

            // Fall back: check if it's a bare DynamoException (Unknown).
            if err.is_instance_of::<DynamoException>(py) {
                let message = err
                    .value(py)
                    .str()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_default();
                return Some((BackendError::Unknown, message));
            }

            None
        }

        /// Register all Dynamo exception classes on the `_core` module.
        pub fn register_exceptions(m: &Bound<'_, PyModule>) -> PyResult<()> {
            m.add("DynamoException", m.py().get_type::<DynamoException>())?;
            m.add(
                "RouterQueueLimitExceeded",
                m.py().get_type::<RouterQueueLimitExceeded>(),
            )?;
            m.add(
                "SelectionServiceError",
                m.py().get_type::<SelectionServiceError>(),
            )?;
            m.add("WorkerUnavailable", m.py().get_type::<WorkerUnavailable>())?;
            $(
                m.add(stringify!($name), m.py().get_type::<$name>())?;
            )*
            Ok(())
        }
    };
}

// ---------------------------------------------------------------------------
// Exception definitions — one entry per BackendError variant.
//
// All error types are exposed to Python as exception classes. When raised by
// Python backend code, they are interpreted as Backend* errors in Rust
// (e.g., raising `InvalidArgument` in Python becomes `BackendInvalidArgument`
// on the Rust side).
//
// When a new variant is added to BackendError in error.rs, add a
// corresponding line here so that a Python exception is generated.
// ---------------------------------------------------------------------------
define_dynamo_exceptions!(
    (Unknown, BackendError::Unknown),
    (InvalidArgument, BackendError::InvalidArgument),
    (CannotConnect, BackendError::CannotConnect),
    (Disconnected, BackendError::Disconnected),
    (ConnectionTimeout, BackendError::ConnectionTimeout),
    (Cancelled, BackendError::Cancelled),
    (EngineShutdown, BackendError::EngineShutdown),
    (StreamIncomplete, BackendError::StreamIncomplete),
    (WorkerOverloaded, BackendError::WorkerOverloaded),
);

/// Generic label for an InvalidRequest that carries no explicit
/// public message.
const GENERIC_INVALID_REQUEST_MESSAGE: &str = "Invalid request";

fn invalid_request_client_message(error: &DynamoError) -> Option<String> {
    (error.class() == ErrorClass::InvalidRequest).then(|| {
        error
            .public_message()
            .unwrap_or(GENERIC_INVALID_REQUEST_MESSAGE)
            .to_owned()
    })
}

/// Preserve validation errors across Rust-to-Python streams: a classified
/// InvalidRequest raises a typed `InvalidArgument`, carrying the producer's
/// public message or the generic label when there is none. Other errors
/// retain the iterator's existing `ValueError` behavior; diagnostic text must
/// never be promoted into a public `InvalidArgument` message.
pub(crate) fn check_response_error<R>(response: Annotated<R>) -> PyResult<Annotated<R>> {
    if response.is_error()
        && let Some(message) = response
            .error
            .as_ref()
            .and_then(invalid_request_client_message)
    {
        return Err(InvalidArgument::new_err(message));
    }

    response
        .ok()
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

fn public_invalid_argument_message(err: &(dyn std::error::Error + 'static)) -> Option<String> {
    let mut current = Some(err);
    while let Some(error) = current {
        if let Some(dynamo_err) = error.downcast_ref::<DynamoError>() {
            return invalid_request_client_message(dynamo_err);
        }
        current = error.source();
    }
    None
}

/// Read `(code, message)` off a Python exception carrying an HTTP-style
/// status. Accepts `.code` (matches [`HttpError`] in `http.rs`) or `.status`
/// (matches `dynamo.common.http.HttpStatusError`) plus `.message`.
///
/// SECURITY: `.message` is diagnostic-only. HTTP-like exceptions can wrap remote
/// fetch failures whose text includes URLs or internal connection details. The
/// semantic status is preserved, but client renderers use the class catalog
/// unless the producer raised an explicitly public Dynamo exception.
pub fn extract_http_like_error(py: Python<'_>, err: &PyErr) -> Option<(u16, String)> {
    let value = err.value(py);
    let code = value
        .getattr("code")
        .ok()
        .and_then(|a| a.extract::<u16>().ok())
        .or_else(|| {
            value
                .getattr("status")
                .ok()
                .and_then(|a| a.extract::<u16>().ok())
        })?;
    let message = value.getattr("message").ok()?.extract::<String>().ok()?;
    Some((code, message))
}

/// Map an explicit Python HTTP exception status to Dynamo's semantic class.
pub fn error_class_for_http_status(code: u16) -> ErrorClass {
    match code {
        400 => ErrorClass::InvalidRequest,
        401 => ErrorClass::Unauthenticated,
        403 => ErrorClass::PermissionDenied,
        404 => ErrorClass::NotFound,
        408 => ErrorClass::DeadlineExceeded,
        409 => ErrorClass::Conflict,
        413 => ErrorClass::PayloadTooLarge,
        415 => ErrorClass::UnsupportedMedia,
        431 => ErrorClass::PayloadTooLarge,
        429 => ErrorClass::RateLimited,
        499 => ErrorClass::Cancelled,
        501 => ErrorClass::NotImplemented,
        502 => ErrorClass::BackendProtocol,
        503 => ErrorClass::Unavailable,
        504 => ErrorClass::DeadlineExceeded,
        529 => ErrorClass::CapacityExhausted,
        402 | 405..=407 | 410..=412 | 414 | 416..=428 | 430 | 432..=498 => {
            ErrorClass::InvalidRequest
        }
        _ => ErrorClass::Internal,
    }
}

pub(crate) fn http_like_error_to_dynamo(py: Python<'_>, err: &PyErr) -> Option<DynamoError> {
    let (code, message) = extract_http_like_error(py, err)?;
    Some(build_http_like_error(code, message))
}

fn build_http_like_error(code: u16, message: String) -> DynamoError {
    let legacy_message = serde_json::json!({
        "message": message,
        "code": code,
    })
    .to_string();

    DynamoError::builder()
        .class(error_class_for_http_status(code))
        .diagnostic(legacy_message)
        .build()
}

/// vLLM signals request faults with `VLLMClientError` subclasses
/// (`VLLMValidationError`, `VLLMNotFoundError`, `VLLMUnprocessableEntityError`).
/// They are not `ValueError`/`TypeError` and carry no `.code`/`.status`, so
/// without this they fall through to `Unknown` and the client gets a sanitized
/// 500 that hides the reason. Classes are matched by name along the MRO so
/// Dynamo does not import vLLM; a `VLLMValidationError` without the
/// `VLLMClientError` base (older vLLM) is matched by its own name.
///
/// Every vLLM client error maps to `Backend(InvalidArgument)` with the
/// `{message, code: 400}` envelope, which the frontend reads for the status
/// even when the error crosses the wire as a string. Only validation errors,
/// which describe the request (for example "At most 8 image(s) may be provided
/// in one prompt."), also set the public message; other client errors can
/// quote fetched URLs, so their text stays diagnostic.
pub(crate) fn vllm_client_error_to_dynamo(py: Python<'_>, err: &PyErr) -> Option<DynamoError> {
    let class_names: Vec<String> = err
        .get_type(py)
        .getattr("__mro__")
        .ok()
        .and_then(|mro| mro.extract::<Vec<Bound<'_, PyAny>>>().ok())
        .unwrap_or_default()
        .iter()
        .filter_map(|class| class.getattr("__name__").ok()?.extract::<String>().ok())
        .collect();
    let has_class = |name: &str| class_names.iter().any(|class| class == name);
    let validation = has_class("VLLMValidationError");
    if !validation && !has_class("VLLMClientError") {
        return None;
    }
    let message = err
        .value(py)
        .str()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let envelope = serde_json::json!({ "message": message, "code": 400 }).to_string();
    let builder = DynamoError::builder()
        .error_type(ErrorType::Backend(BackendError::InvalidArgument))
        .message(envelope);
    Some(if validation {
        builder.public_message(message).build()
    } else {
        builder.build()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    pyo3::create_exception!(
        vllm_exceptions,
        VLLMClientError,
        pyo3::exceptions::PyException
    );
    pyo3::create_exception!(vllm_exceptions, VLLMValidationError, VLLMClientError);
    pyo3::create_exception!(vllm_exceptions, VLLMNotFoundError, VLLMClientError);
    pyo3::create_exception!(
        vllm_exceptions,
        VLLMServerError,
        pyo3::exceptions::PyException
    );
    pyo3::create_exception!(
        vllm_exceptions,
        SchemaValidationError,
        pyo3::exceptions::PyException
    );

    #[test]
    fn vllm_client_errors_map_to_invalid_argument() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let error = VLLMValidationError::new_err("At most 8 images may be provided");
            let mapped = vllm_client_error_to_dynamo(py, &error).expect("validation error");
            assert!(matches!(
                mapped.error_type(),
                ErrorType::Backend(BackendError::InvalidArgument)
            ));
            let payload: serde_json::Value = serde_json::from_str(mapped.message()).unwrap();
            assert_eq!(payload["code"], 400);
            assert_eq!(payload["message"], "At most 8 images may be provided");
            assert_eq!(
                mapped.public_message(),
                Some("At most 8 images may be provided")
            );

            // Any other client error is a 400 without a public message.
            let error = VLLMNotFoundError::new_err("http://internal/image.png: 404");
            let mapped = vllm_client_error_to_dynamo(py, &error).expect("client error");
            assert!(matches!(
                mapped.error_type(),
                ErrorType::Backend(BackendError::InvalidArgument)
            ));
            let payload: serde_json::Value = serde_json::from_str(mapped.message()).unwrap();
            assert_eq!(payload["code"], 400);
            assert_eq!(mapped.public_message(), None);

            // Server errors and unrelated validation errors are not client errors.
            for error in [
                VLLMServerError::new_err("engine dead"),
                SchemaValidationError::new_err("config"),
                pyo3::exceptions::PyRuntimeError::new_err("boom"),
            ] {
                assert!(vllm_client_error_to_dynamo(py, &error).is_none());
            }
        });
    }

    #[derive(Deserialize)]
    struct LegacyDynamoError {
        message: String,
    }

    #[derive(Deserialize)]
    struct LegacyHttpError {
        code: u16,
        message: String,
    }

    #[test]
    fn worker_unavailable_survives_class_normalization_and_context() {
        let rejection = DynamoError::builder()
            .class(ErrorClass::WorkerUnavailable)
            .message("prefill routing is still activating")
            .build();
        // `class()` reports the normalized `Unavailable`; detection must not rely on it.
        assert_eq!(rejection.class(), ErrorClass::Unavailable);
        assert!(is_worker_unavailable(&rejection));

        let wrapped = anyhow::Error::from(rejection).context("routed pipeline failed");
        assert!(is_worker_unavailable(wrapped.as_ref()));

        let pool_unavailable = DynamoError::builder()
            .class(ErrorClass::Unavailable)
            .message("no workers")
            .build();
        assert!(!is_worker_unavailable(&pool_unavailable));
    }

    #[test]
    fn invalid_request_uses_public_message_or_generic_fallback() {
        let error = DynamoError::builder()
            .class(ErrorClass::InvalidRequest)
            .diagnostic("private backend detail")
            .public_message("Invalid JSON schema")
            .build();
        assert_eq!(
            invalid_request_client_message(&error),
            Some("Invalid JSON schema".to_string())
        );

        let error = DynamoError::builder()
            .class(ErrorClass::InvalidRequest)
            .diagnostic("private validation detail")
            .build();
        assert_eq!(
            invalid_request_client_message(&error),
            Some(GENERIC_INVALID_REQUEST_MESSAGE.to_string())
        );
    }

    #[derive(Debug)]
    struct RouterWrapped(DynamoError);

    impl std::fmt::Display for RouterWrapped {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "Prefill execution failed: {}", self.0)
        }
    }

    impl std::error::Error for RouterWrapped {
        fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
            Some(&self.0)
        }
    }

    #[test]
    fn error_chain_yields_public_invalid_argument_message() {
        let worker_error = DynamoError::builder()
            .class(ErrorClass::InvalidRequest)
            .diagnostic("private backend detail")
            .public_message("use --enable-multimodal")
            .build();
        let wrapped = anyhow::Error::new(RouterWrapped(worker_error));
        assert_eq!(
            public_invalid_argument_message(wrapped.as_ref()),
            Some("use --enable-multimodal".to_string())
        );
    }

    #[test]
    fn legacy_worker_rejection_without_public_message_yields_generic_message() {
        let wire = serde_json::json!({
            "error_type": {"Backend": "InvalidArgument"},
            "message": "private backend detail",
        });
        let error: DynamoError = serde_json::from_value(wire).expect("legacy wire error");
        assert_eq!(error.class(), ErrorClass::InvalidRequest);
        assert_eq!(error.public_message(), None);

        let wrapped = anyhow::Error::new(RouterWrapped(error));
        assert_eq!(
            public_invalid_argument_message(wrapped.as_ref()),
            Some(GENERIC_INVALID_REQUEST_MESSAGE.to_string())
        );
    }

    #[test]
    fn other_error_classes_are_not_promoted_to_invalid_arguments() {
        for class in [
            ErrorClass::Internal,
            ErrorClass::Cancelled,
            ErrorClass::Unavailable,
        ] {
            let error = DynamoError::builder()
                .class(class)
                .diagnostic("private backend detail")
                .public_message("Public error message")
                .build();
            assert_eq!(invalid_request_client_message(&error), None);
        }
    }

    #[test]
    fn http_like_errors_support_semantic_and_legacy_readers() {
        for (code, expected_class) in [
            (409, ErrorClass::Conflict),
            (415, ErrorClass::UnsupportedMedia),
            (503, ErrorClass::Unavailable),
        ] {
            let error = build_http_like_error(code, "private backend detail".to_string());
            assert_eq!(error.class(), expected_class);
            assert_eq!(error.public_message(), None);

            let wire = serde_json::to_value(&error).expect("serialize DynamoError");
            let semantic: DynamoError =
                serde_json::from_value(wire.clone()).expect("semantic reader");
            assert_eq!(semantic.class(), expected_class);
            assert_eq!(semantic.public_message(), None);

            let legacy: LegacyDynamoError = serde_json::from_value(wire).expect("legacy reader");
            let legacy_http: LegacyHttpError =
                serde_json::from_str(&legacy.message).expect("legacy HTTP payload");
            assert_eq!(legacy_http.code, code);
            assert_eq!(legacy_http.message, "private backend detail");
        }
    }
}
