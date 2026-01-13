use thiserror::Error;

use crate::Trap;

/// Errors during module instantiation
#[derive(Clone, Debug, Error)]
pub enum InstantiationError {
    /// Import not found
    #[error("import not found: {module}.{name}")]
    ImportNotFound { module: String, name: String },
    /// Import type mismatch
    #[error("import type mismatch: expected {expected}, got {got}")]
    ImportTypeMismatch { expected: String, got: String },
    /// Memory initialization failed
    #[error("memory initialization failed: {0}")]
    MemoryInitFailed(String),
    /// Table initialization failed
    #[error("table initialization failed: {0}")]
    TableInitFailed(String),
    /// Start function trapped
    #[error("start function trapped: {0}")]
    StartTrapped(#[from] Trap),
    /// Module validation failed
    #[error("validation failed: {0}")]
    ValidationFailed(String),
}

/// Errors during compilation
#[derive(Clone, Debug, Error)]
pub enum CompileError {
    /// Parsing error
    #[error("parse error: {0}")]
    Parse(String),
    /// SLJIT error
    #[error("sljit error: {0}")]
    Sljit(String),
    /// Unsupported feature
    #[error("unsupported: {0}")]
    Unsupported(String),
    /// Invalid WebAssembly
    #[error("invalid: {0}")]
    Invalid(String),
    /// Register allocation failed
    #[error("register allocation failed")]
    RegisterAllocationFailed,
}

impl From<sljit::sys::ErrorCode> for CompileError {
    fn from(e: sljit::sys::ErrorCode) -> Self {
        Self::Sljit(format!("{:?}", e))
    }
}

impl From<wasmparser::BinaryReaderError> for CompileError {
    fn from(e: wasmparser::BinaryReaderError) -> Self {
        Self::Parse(e.to_string())
    }
}
