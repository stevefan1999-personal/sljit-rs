//! Simple WebAssembly JIT Compiler using sljit-rs
//!
//! This is a minimal WebAssembly to native code compiler that demonstrates
//! how to use sljit-rs's high-level Emitter API to JIT-compile WebAssembly.
//!
//! Inspired by pwart's architecture, but simplified for clarity.
//!
//! # Architecture
//!
//! The runtime follows a Store-based architecture similar to wasmtime/wasmi:
//!
//! - `Engine` - Compilation configuration holder (stateless, shareable)
//! - `Store` - Runtime state container that owns all instances
//! - `Module` - Compiled WebAssembly module (immutable after creation)
//! - `Instance` - Runtime instantiation of a Module with resolved imports
//! - `Linker` - Helper for building import resolvers
//!
//! # Example
//!
//! ```ignore
//! use example_wasm::{Engine, Store, Module, Linker, Value};
//! use std::sync::Arc;
//!
//! // Create engine and store
//! let engine = Arc::new(Engine::default());
//! let mut store = Store::new(engine.clone());
//!
//! // Parse module
//! let wasm = wat::parse_str(r#"(module (func (export "answer") (result i32) i32.const 42))"#)?;
//! let module = Module::new(&engine, &wasm)?;
//!
//! // Instantiate
//! let linker = Linker::new();
//! let instance = linker.instantiate(&mut store, &module)?;
//!
//! // Call exported function
//! let func_idx = instance.get_func(&store, "answer").unwrap();
//! let result = store.call(func_idx, &[])?;
//! assert_eq!(result, vec![Value::I32(42)]);
//! ```

use sljit::sys;
use sljit::{FloatRegister, Operand, SavedRegister, ScratchRegister, mem_sp_offset};

// Core types module
pub mod types;

// Re-export core types for public API
pub use types::{
    CompileError, Engine, EngineConfig, FuncIdx, GlobalIdx, InstantiationError, MemoryIdx,
    OptLevel, RefType, TableIdx, Trap, TrapKind, Value, WasmFeatures,
};
// Re-export our ValType (not wasmparser's)
pub use types::ValType;
// Re-export FuncType from types module
pub use types::FuncType;

pub(crate) mod helpers;

/// Binary operation type for floats
#[derive(Clone, Copy, Debug)]
pub enum FloatBinaryOp {
    Min,
    Max,
    Copysign,
}

/// Unary operation type for floats
#[derive(Clone, Copy, Debug)]
pub enum FloatUnaryOp {
    Sqrt,
    Ceil,
    Floor,
    Trunc,
    Nearest,
}

/// Unary operation type
#[derive(Clone, Copy, Debug)]
pub enum UnaryOp {
    Clz64,
    Ctz64,
    Popcnt,
    Popcnt64,
}

/// Division operation type
#[derive(Clone, Copy, Debug)]
pub(crate) enum DivOp {
    DivS,
    DivU,
    RemS,
    RemU,
}

/// Represents a value on the operand stack during compilation
#[derive(Clone, Debug)]
pub(crate) enum StackValue {
    Register(ScratchRegister),
    FloatRegister(FloatRegister),
    Stack(i32),
    Const(i32),
    ConstF32(u32),
    ConstF64(u64),
    Saved(SavedRegister),
}

impl StackValue {
    #[inline(always)]
    pub(crate) fn to_operand(&self) -> Operand {
        match self {
            Self::Register(r) => (*r).into(),
            Self::FloatRegister(r) => (*r).into(),
            Self::Stack(offset) => mem_sp_offset(*offset),
            Self::Const(val) => (*val).into(),
            Self::ConstF32(_) | Self::ConstF64(_) => {
                // Float constants need to be loaded into a register first
                // This shouldn't be called directly; use ensure_in_float_register instead
                panic!("Float constants cannot be converted to operand directly")
            }
            Self::Saved(r) => (*r).into(),
        }
    }
}

/// Information about a control flow block
#[derive(Clone, Debug, Default)]
pub(crate) struct Block {
    pub(crate) label: Option<sys::Label>,
    pub(crate) end_jumps: Vec<sys::Jump>,
    pub(crate) else_jump: Option<sys::Jump>,
    pub(crate) stack_depth: usize,
    pub(crate) result_offset: Option<i32>,
}

/// Local variable storage info
#[derive(Clone, Debug)]
pub(crate) enum LocalVar {
    Saved(SavedRegister),
    Stack(i32),
}

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

pub mod instance;
pub mod module;
pub mod store;

// Re-export store types
pub use store::{Func, Global, HostFunc, Memory, PAGE_SIZE, Store, Table, WasmFunc};

// Re-export instance types
pub use instance::{Instance, Linker};

// Re-export module types
pub use module::Module;

#[cfg(test)]
mod tests;

pub mod function;
