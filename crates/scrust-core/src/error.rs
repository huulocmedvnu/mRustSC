use thiserror::Error;

/// Everything that can go wrong inside scrust.
///
/// One error type for the whole workspace: the Python layer maps each variant to
/// an exception exactly once, instead of every call site inventing a message.
#[derive(Debug, Error)]
pub enum Error {
    #[error("expected {expected}, got {actual}")]
    Shape { expected: String, actual: String },

    #[error("{parameter} must be {constraint}, got {value}")]
    InvalidParameter {
        parameter: &'static str,
        constraint: &'static str,
        value: String,
    },

    #[error("no Metal device is available on this machine")]
    NoGpu,

    #[error("Metal kernel {name}: {message}")]
    Kernel { name: &'static str, message: String },

    #[error("{operation} did not converge after {iterations} iterations")]
    NotConverged {
        operation: &'static str,
        iterations: usize,
    },

    #[error(transparent)]
    Tensor(#[from] candle_core::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    /// Shape mismatch reported in terms of what the caller asked for.
    pub fn shape(expected: impl Into<String>, actual: impl Into<String>) -> Self {
        Error::Shape {
            expected: expected.into(),
            actual: actual.into(),
        }
    }

    pub fn parameter(
        parameter: &'static str,
        constraint: &'static str,
        value: impl std::fmt::Display,
    ) -> Self {
        Error::InvalidParameter {
            parameter,
            constraint,
            value: value.to_string(),
        }
    }
}
