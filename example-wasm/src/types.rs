//! Core runtime types for the WebAssembly JIT compiler
//!
//! This module contains the fundamental types used throughout the runtime:
//! - `ValType` - WebAssembly value types
//! - `Value` - Runtime values
//! - `FuncType` - Function signatures
//! - `Trap` - Runtime errors

use itertools::Itertools;
use std::fmt;

// ============================================================================
// Value Types
// ============================================================================

/// WebAssembly value types
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, derive_more::Display)]
pub enum ValType {
    /// 32-bit integer
    #[display("i32")]
    I32,
    /// 64-bit integer
    #[display("i64")]
    I64,
    /// 32-bit float
    #[display("f32")]
    F32,
    /// 64-bit float
    #[display("f64")]
    F64,
    /// Function reference
    #[display("funcref")]
    FuncRef,
    /// External reference
    #[display("externref")]
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

impl From<ValType> for wasmparser::ValType {
    fn from(val: ValType) -> Self {
        match val {
            ValType::I32 => wasmparser::ValType::I32,
            ValType::I64 => wasmparser::ValType::I64,
            ValType::F32 => wasmparser::ValType::F32,
            ValType::F64 => wasmparser::ValType::F64,
            ValType::FuncRef => wasmparser::ValType::FUNCREF,
            ValType::ExternRef => wasmparser::ValType::EXTERNREF,
        }
    }
}

// ============================================================================
// Runtime Values
// ============================================================================

/// Index into the Store's function list
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct FuncIdx(pub u32);

/// Index into the Store's memory list
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct MemoryIdx(pub u32);

/// Index into the Store's table list
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct TableIdx(pub u32);

/// Index into the Store's global list
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct GlobalIdx(pub u32);

/// A runtime WebAssembly value
#[derive(Clone, Copy, Debug, derive_more::Display)]
pub enum Value {
    /// 32-bit integer
    #[display("i32:{_0}")]
    I32(i32),
    /// 64-bit integer
    #[display("i64:{_0}")]
    I64(i64),
    /// 32-bit float
    #[display("f32:{_0}")]
    F32(f32),
    /// 64-bit float
    #[display("f64:{_0}")]
    F64(f64),
    /// Function reference (None = null)
    #[display("funcref:{}", _0.map(|idx| idx.0 as i64).unwrap_or(-1))]
    FuncRef(Option<FuncIdx>),
    /// External reference (None = null, Some = opaque handle)
    #[display("externref:{}", _0.map(|idx| idx as i64).unwrap_or(-1))]
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
        write!(
            f,
            "({}) -> ({})",
            self.params.iter().format(", "),
            self.results.iter().format(", ")
        )
    }
}

pub mod trap;
pub use trap::{Trap, TrapKind};

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

// ============================================================================
// Compiler Internal Types
// ============================================================================

/// Global variable info (compiler internal representation)
/// Uses wasmparser::ValType internally for compiler compatibility
#[derive(Clone, Debug)]
pub struct GlobalInfo {
    /// Pointer to the global's memory location
    pub ptr: usize,
    /// Whether the global is mutable
    pub mutable: bool,
    /// Value type of the global (uses wasmparser::ValType for compiler)
    pub val_type: wasmparser::ValType,
}

/// Memory instance info (compiler internal representation)
#[derive(Clone, Debug)]
pub struct MemoryInfo {
    /// Pointer to the memory data
    pub data_ptr: usize,
    /// Pointer to the current size (in pages)
    pub size_ptr: usize,
    /// Maximum size in pages (if specified)
    pub max_pages: Option<u32>,
    /// Memory grow callback: fn(current_pages: u32, delta: u32) -> i32 (returns -1 on failure, new size on success)
    pub grow_callback: Option<extern "C" fn(current_pages: u32, delta: u32) -> i32>,
}

/// Compiled function entry for direct calls (compiler internal)
#[derive(Clone, Debug)]
pub struct FunctionEntry {
    /// Pointer to the compiled function code (used for host functions)
    pub code_ptr: usize,
    /// Pointer to a location storing the code_ptr (used for wasm functions)
    /// This enables indirect calls that work even when code_ptr is updated later
    pub code_ptr_ptr: Option<*const usize>,
    /// Function type (uses wasmparser types internally)
    pub params: Vec<wasmparser::ValType>,
    pub results: Vec<wasmparser::ValType>,
}

// Safety: FunctionEntry contains raw pointers but they are only used for reading
// code_ptr values that are stable once set during compilation
unsafe impl Send for FunctionEntry {}
unsafe impl Sync for FunctionEntry {}

/// Table entry for indirect calls
#[derive(Clone, Debug)]
pub struct TableEntry {
    /// Pointer to the table data (array of function pointers)
    pub data_ptr: usize,
    /// Current table size
    pub size: u32,
}

#[cfg(test)]
mod tests;
