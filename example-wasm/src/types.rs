//! Core runtime types for the WebAssembly JIT compiler
//!
//! This module contains the fundamental types used throughout the runtime:
//! - `ValType` - WebAssembly value types
//! - `Value` - Runtime values
//! - `FuncType` - Function signatures
//! - `Engine` - Compilation configuration
//! - `Trap` - Runtime errors

use std::fmt;
use std::sync::Arc;

// ============================================================================
// Value Types
// ============================================================================

/// WebAssembly value types
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ValType {
    /// 32-bit integer
    I32,
    /// 64-bit integer
    I64,
    /// 32-bit float
    F32,
    /// 64-bit float
    F64,
    /// Function reference
    FuncRef,
    /// External reference
    ExternRef,
}

impl ValType {
    /// Returns true if this is a numeric type (i32, i64, f32, f64)
    #[inline]
    pub const fn is_num(&self) -> bool {
        matches!(self, Self::I32 | Self::I64 | Self::F32 | Self::F64)
    }

    /// Returns true if this is a reference type (funcref, externref)
    #[inline]
    pub const fn is_ref(&self) -> bool {
        matches!(self, Self::FuncRef | Self::ExternRef)
    }

    /// Returns true if this is an integer type (i32, i64)
    #[inline]
    pub const fn is_int(&self) -> bool {
        matches!(self, Self::I32 | Self::I64)
    }

    /// Returns true if this is a float type (f32, f64)
    #[inline]
    pub const fn is_float(&self) -> bool {
        matches!(self, Self::F32 | Self::F64)
    }

    /// Returns the size in bytes of this value type
    #[inline]
    pub const fn byte_size(&self) -> usize {
        match self {
            Self::I32 | Self::F32 => 4,
            Self::I64 | Self::F64 => 8,
            Self::FuncRef | Self::ExternRef => core::mem::size_of::<usize>(),
        }
    }
}

impl fmt::Display for ValType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::I32 => write!(f, "i32"),
            Self::I64 => write!(f, "i64"),
            Self::F32 => write!(f, "f32"),
            Self::F64 => write!(f, "f64"),
            Self::FuncRef => write!(f, "funcref"),
            Self::ExternRef => write!(f, "externref"),
        }
    }
}

impl From<wasmparser::ValType> for ValType {
    fn from(ty: wasmparser::ValType) -> Self {
        match ty {
            wasmparser::ValType::I32 => Self::I32,
            wasmparser::ValType::I64 => Self::I64,
            wasmparser::ValType::F32 => Self::F32,
            wasmparser::ValType::F64 => Self::F64,
            wasmparser::ValType::V128 => panic!("SIMD v128 not supported"),
            wasmparser::ValType::Ref(r) => {
                if r.is_func_ref() {
                    Self::FuncRef
                } else if r.is_extern_ref() {
                    Self::ExternRef
                } else {
                    panic!("Unsupported reference type: {:?}", r)
                }
            }
        }
    }
}

// ============================================================================
// Runtime Values
// ============================================================================

/// Index into the Store's function list
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FuncIdx(pub u32);

/// Index into the Store's memory list
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MemoryIdx(pub u32);

/// Index into the Store's table list
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TableIdx(pub u32);

/// Index into the Store's global list
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GlobalIdx(pub u32);

/// A runtime WebAssembly value
#[derive(Clone, Copy, Debug)]
pub enum Value {
    /// 32-bit integer
    I32(i32),
    /// 64-bit integer
    I64(i64),
    /// 32-bit float
    F32(f32),
    /// 64-bit float
    F64(f64),
    /// Function reference (None = null)
    FuncRef(Option<FuncIdx>),
    /// External reference (None = null, Some = opaque handle)
    ExternRef(Option<u32>),
}

impl Value {
    /// Get the type of this value
    #[inline]
    pub const fn ty(&self) -> ValType {
        match self {
            Self::I32(_) => ValType::I32,
            Self::I64(_) => ValType::I64,
            Self::F32(_) => ValType::F32,
            Self::F64(_) => ValType::F64,
            Self::FuncRef(_) => ValType::FuncRef,
            Self::ExternRef(_) => ValType::ExternRef,
        }
    }

    /// Get the default/zero value for a type
    #[inline]
    pub const fn default_for(ty: ValType) -> Self {
        match ty {
            ValType::I32 => Self::I32(0),
            ValType::I64 => Self::I64(0),
            ValType::F32 => Self::F32(0.0),
            ValType::F64 => Self::F64(0.0),
            ValType::FuncRef => Self::FuncRef(None),
            ValType::ExternRef => Self::ExternRef(None),
        }
    }

    /// Try to get as i32
    #[inline]
    pub const fn as_i32(&self) -> Option<i32> {
        match self {
            Self::I32(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to get as i64
    #[inline]
    pub const fn as_i64(&self) -> Option<i64> {
        match self {
            Self::I64(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to get as f32
    #[inline]
    pub const fn as_f32(&self) -> Option<f32> {
        match self {
            Self::F32(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to get as f64
    #[inline]
    pub const fn as_f64(&self) -> Option<f64> {
        match self {
            Self::F64(v) => Some(*v),
            _ => None,
        }
    }

    /// Unwrap as i32 or panic
    #[inline]
    pub fn unwrap_i32(&self) -> i32 {
        self.as_i32().expect("expected i32 value")
    }

    /// Unwrap as i64 or panic
    #[inline]
    pub fn unwrap_i64(&self) -> i64 {
        self.as_i64().expect("expected i64 value")
    }

    /// Unwrap as f32 or panic
    #[inline]
    pub fn unwrap_f32(&self) -> f32 {
        self.as_f32().expect("expected f32 value")
    }

    /// Unwrap as f64 or panic
    #[inline]
    pub fn unwrap_f64(&self) -> f64 {
        self.as_f64().expect("expected f64 value")
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::I32(v) => write!(f, "i32:{}", v),
            Self::I64(v) => write!(f, "i64:{}", v),
            Self::F32(v) => write!(f, "f32:{}", v),
            Self::F64(v) => write!(f, "f64:{}", v),
            Self::FuncRef(None) => write!(f, "funcref:null"),
            Self::FuncRef(Some(idx)) => write!(f, "funcref:{}", idx.0),
            Self::ExternRef(None) => write!(f, "externref:null"),
            Self::ExternRef(Some(idx)) => write!(f, "externref:{}", idx),
        }
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::I32(a), Self::I32(b)) => a == b,
            (Self::I64(a), Self::I64(b)) => a == b,
            (Self::F32(a), Self::F32(b)) => a.to_bits() == b.to_bits(),
            (Self::F64(a), Self::F64(b)) => a.to_bits() == b.to_bits(),
            (Self::FuncRef(a), Self::FuncRef(b)) => a == b,
            (Self::ExternRef(a), Self::ExternRef(b)) => a == b,
            _ => false,
        }
    }
}

impl From<i32> for Value {
    #[inline]
    fn from(v: i32) -> Self {
        Self::I32(v)
    }
}

impl From<i64> for Value {
    #[inline]
    fn from(v: i64) -> Self {
        Self::I64(v)
    }
}

impl From<f32> for Value {
    #[inline]
    fn from(v: f32) -> Self {
        Self::F32(v)
    }
}

impl From<f64> for Value {
    #[inline]
    fn from(v: f64) -> Self {
        Self::F64(v)
    }
}

// ============================================================================
// Function Types
// ============================================================================

/// A WebAssembly function signature
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FuncType {
    /// Parameter types
    params: Vec<ValType>,
    /// Result types
    results: Vec<ValType>,
}

impl FuncType {
    /// Create a new function type
    #[inline]
    pub fn new(params: impl Into<Vec<ValType>>, results: impl Into<Vec<ValType>>) -> Self {
        Self {
            params: params.into(),
            results: results.into(),
        }
    }

    /// Get parameter types
    #[inline]
    pub fn params(&self) -> &[ValType] {
        &self.params
    }

    /// Get result types
    #[inline]
    pub fn results(&self) -> &[ValType] {
        &self.results
    }

    /// Check if this function type matches another
    #[inline]
    pub fn matches(&self, other: &FuncType) -> bool {
        self.params == other.params && self.results == other.results
    }
}

impl fmt::Display for FuncType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        for (i, param) in self.params.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", param)?;
        }
        write!(f, ") -> (")?;
        for (i, result) in self.results.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", result)?;
        }
        write!(f, ")")
    }
}

// ============================================================================
// Engine Configuration
// ============================================================================

/// Optimization level for JIT compilation
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OptLevel {
    /// No optimizations, fastest compile time
    None,
    /// Basic optimizations, balanced (default)
    #[default]
    Speed,
    /// Aggressive optimizations, slower compile
    SpeedAndSize,
}

/// WebAssembly feature flags
#[derive(Clone, Debug)]
pub struct WasmFeatures {
    /// Enable multi-value returns
    pub multi_value: bool,
    /// Enable bulk memory operations
    pub bulk_memory: bool,
    /// Enable reference types
    pub reference_types: bool,
    /// Enable SIMD (not currently supported)
    pub simd: bool,
    /// Enable mutable globals
    pub mutable_global: bool,
    /// Enable sign extension operators
    pub sign_extension: bool,
    /// Enable saturating float-to-int conversions
    pub saturating_float_to_int: bool,
}

impl Default for WasmFeatures {
    fn default() -> Self {
        Self {
            multi_value: true,
            bulk_memory: true,
            reference_types: true,
            simd: false, // Not supported
            mutable_global: true,
            sign_extension: true,
            saturating_float_to_int: true,
        }
    }
}

/// Engine configuration for compilation and runtime
#[derive(Clone, Debug)]
pub struct EngineConfig {
    /// Optimization level
    pub opt_level: OptLevel,
    /// WebAssembly features
    pub features: WasmFeatures,
    /// Maximum stack depth (in frames)
    pub max_stack_depth: usize,
    /// Enable fuel metering
    pub fuel_metering: bool,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            opt_level: OptLevel::Speed,
            features: WasmFeatures::default(),
            max_stack_depth: 1024,
            fuel_metering: false,
        }
    }
}

/// Compilation and runtime engine
///
/// The Engine holds configuration for compilation and is stateless.
/// It can be shared across threads via `Arc<Engine>`.
#[derive(Clone, Debug)]
pub struct Engine {
    config: EngineConfig,
}

impl Default for Engine {
    fn default() -> Self {
        Self::new(EngineConfig::default())
    }
}

impl Engine {
    /// Create a new engine with the given configuration
    #[inline]
    pub fn new(config: EngineConfig) -> Self {
        Self { config }
    }

    /// Get the engine configuration
    #[inline]
    pub fn config(&self) -> &EngineConfig {
        &self.config
    }

    /// Get the optimization level
    #[inline]
    pub fn opt_level(&self) -> OptLevel {
        self.config.opt_level
    }

    /// Get the WebAssembly features
    #[inline]
    pub fn features(&self) -> &WasmFeatures {
        &self.config.features
    }

    /// Create a shared engine
    #[inline]
    pub fn into_shared(self) -> Arc<Self> {
        Arc::new(self)
    }
}

// ============================================================================
// Error Types
// ============================================================================

/// Runtime trap - unrecoverable error during execution
#[derive(Clone, Debug)]
pub struct Trap {
    /// The kind of trap
    pub kind: TrapKind,
    /// Human-readable message
    pub message: String,
}

impl Trap {
    /// Create a new trap
    #[inline]
    pub fn new(kind: TrapKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    /// Create an unreachable trap
    #[inline]
    pub fn unreachable() -> Self {
        Self::new(TrapKind::Unreachable, "unreachable executed")
    }

    /// Create a memory out of bounds trap
    #[inline]
    pub fn memory_out_of_bounds() -> Self {
        Self::new(TrapKind::MemoryOutOfBounds, "out of bounds memory access")
    }

    /// Create a table out of bounds trap
    #[inline]
    pub fn table_out_of_bounds() -> Self {
        Self::new(TrapKind::TableOutOfBounds, "out of bounds table access")
    }

    /// Create an indirect call type mismatch trap
    #[inline]
    pub fn indirect_call_type_mismatch() -> Self {
        Self::new(
            TrapKind::IndirectCallTypeMismatch,
            "indirect call type mismatch",
        )
    }

    /// Create a null function reference trap
    #[inline]
    pub fn null_func_ref() -> Self {
        Self::new(TrapKind::NullFuncRef, "null function reference")
    }

    /// Create an integer overflow trap
    #[inline]
    pub fn integer_overflow() -> Self {
        Self::new(TrapKind::IntegerOverflow, "integer overflow")
    }

    /// Create an integer division by zero trap
    #[inline]
    pub fn integer_divide_by_zero() -> Self {
        Self::new(TrapKind::IntegerDivisionByZero, "integer division by zero")
    }

    /// Create an invalid conversion to integer trap
    #[inline]
    pub fn invalid_conversion_to_int() -> Self {
        Self::new(
            TrapKind::InvalidConversionToInteger,
            "invalid conversion to integer",
        )
    }

    /// Create a stack overflow trap
    #[inline]
    pub fn stack_overflow() -> Self {
        Self::new(TrapKind::StackOverflow, "stack overflow")
    }

    /// Create an out of fuel trap
    #[inline]
    pub fn out_of_fuel() -> Self {
        Self::new(TrapKind::OutOfFuel, "out of fuel")
    }
}

impl fmt::Display for Trap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "trap: {}", self.message)
    }
}

impl std::error::Error for Trap {}

/// The kind of runtime trap
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrapKind {
    /// Out of bounds memory access
    MemoryOutOfBounds,
    /// Out of bounds table access
    TableOutOfBounds,
    /// Indirect call type mismatch
    IndirectCallTypeMismatch,
    /// Null function reference
    NullFuncRef,
    /// Integer overflow
    IntegerOverflow,
    /// Integer division by zero
    IntegerDivisionByZero,
    /// Invalid conversion to integer
    InvalidConversionToInteger,
    /// Stack overflow
    StackOverflow,
    /// Unreachable code executed
    Unreachable,
    /// Out of fuel
    OutOfFuel,
    /// Custom trap from host function
    Host,
}

impl fmt::Display for TrapKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MemoryOutOfBounds => write!(f, "memory out of bounds"),
            Self::TableOutOfBounds => write!(f, "table out of bounds"),
            Self::IndirectCallTypeMismatch => write!(f, "indirect call type mismatch"),
            Self::NullFuncRef => write!(f, "null function reference"),
            Self::IntegerOverflow => write!(f, "integer overflow"),
            Self::IntegerDivisionByZero => write!(f, "integer division by zero"),
            Self::InvalidConversionToInteger => write!(f, "invalid conversion to integer"),
            Self::StackOverflow => write!(f, "stack overflow"),
            Self::Unreachable => write!(f, "unreachable"),
            Self::OutOfFuel => write!(f, "out of fuel"),
            Self::Host => write!(f, "host error"),
        }
    }
}

/// Errors during module instantiation
#[derive(Clone, Debug)]
pub enum InstantiationError {
    /// Import not found
    ImportNotFound { module: String, name: String },
    /// Import type mismatch
    ImportTypeMismatch { expected: String, got: String },
    /// Memory initialization failed
    MemoryInitFailed(String),
    /// Table initialization failed
    TableInitFailed(String),
    /// Start function trapped
    StartTrapped(Trap),
    /// Module validation failed
    ValidationFailed(String),
}

impl fmt::Display for InstantiationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ImportNotFound { module, name } => {
                write!(f, "import not found: {}.{}", module, name)
            }
            Self::ImportTypeMismatch { expected, got } => {
                write!(
                    f,
                    "import type mismatch: expected {}, got {}",
                    expected, got
                )
            }
            Self::MemoryInitFailed(msg) => write!(f, "memory initialization failed: {}", msg),
            Self::TableInitFailed(msg) => write!(f, "table initialization failed: {}", msg),
            Self::StartTrapped(trap) => write!(f, "start function trapped: {}", trap),
            Self::ValidationFailed(msg) => write!(f, "validation failed: {}", msg),
        }
    }
}

impl std::error::Error for InstantiationError {}

impl From<Trap> for InstantiationError {
    fn from(trap: Trap) -> Self {
        Self::StartTrapped(trap)
    }
}

/// Errors during compilation
#[derive(Clone, Debug)]
pub enum CompileError {
    /// Parsing error
    Parse(String),
    /// SLJIT error
    Sljit(String),
    /// Unsupported feature
    Unsupported(String),
    /// Invalid WebAssembly
    Invalid(String),
    /// Register allocation failed
    RegisterAllocationFailed,
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(msg) => write!(f, "parse error: {}", msg),
            Self::Sljit(msg) => write!(f, "sljit error: {}", msg),
            Self::Unsupported(msg) => write!(f, "unsupported: {}", msg),
            Self::Invalid(msg) => write!(f, "invalid: {}", msg),
            Self::RegisterAllocationFailed => write!(f, "register allocation failed"),
        }
    }
}

impl std::error::Error for CompileError {}

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

// ============================================================================
// Reference Types (for Table)
// ============================================================================

/// Reference type for table elements
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefType {
    /// Function reference
    FuncRef,
    /// External reference
    ExternRef,
}

impl From<RefType> for ValType {
    fn from(rt: RefType) -> Self {
        match rt {
            RefType::FuncRef => ValType::FuncRef,
            RefType::ExternRef => ValType::ExternRef,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_val_type_properties() {
        assert!(ValType::I32.is_num());
        assert!(ValType::I32.is_int());
        assert!(!ValType::I32.is_float());
        assert!(!ValType::I32.is_ref());

        assert!(ValType::F64.is_num());
        assert!(ValType::F64.is_float());
        assert!(!ValType::F64.is_int());

        assert!(ValType::FuncRef.is_ref());
        assert!(!ValType::FuncRef.is_num());
    }

    #[test]
    fn test_value_types() {
        let v = Value::I32(42);
        assert_eq!(v.ty(), ValType::I32);
        assert_eq!(v.as_i32(), Some(42));
        assert_eq!(v.as_i64(), None);

        let v = Value::F64(3.14);
        assert_eq!(v.ty(), ValType::F64);
        assert_eq!(v.as_f64(), Some(3.14));
    }

    #[test]
    fn test_func_type() {
        let ft = FuncType::new(vec![ValType::I32, ValType::I32], vec![ValType::I32]);
        assert_eq!(ft.params().len(), 2);
        assert_eq!(ft.results().len(), 1);
        assert_eq!(ft.to_string(), "(i32, i32) -> (i32)");
    }

    #[test]
    fn test_engine_default() {
        let engine = Engine::default();
        assert_eq!(engine.opt_level(), OptLevel::Speed);
        assert!(engine.features().multi_value);
        assert!(!engine.features().simd);
    }

    #[test]
    fn test_trap_creation() {
        let trap = Trap::unreachable();
        assert_eq!(trap.kind, TrapKind::Unreachable);
        assert!(trap.message.contains("unreachable"));
    }
}
