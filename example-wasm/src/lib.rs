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

// Core types module
pub mod types;

// Engine module
pub mod engine;

// Re-export core types for public API
pub use types::*;

// Re-export engine types
pub use engine::{Engine, EngineConfig, OptLevel, WasmFeatures};

pub(crate) mod helpers;

pub mod error;
pub mod instance;
pub mod linker;
pub mod module;
pub mod store;
pub mod trampoline;

// Re-export store types
pub use store::{Func, Global, HostFunc, Memory, PAGE_SIZE, Store, Table, WasmFunc};

// Re-export instance types
pub use instance::Instance;
pub use linker::Linker;

// Re-export module types
pub use module::Module;

#[cfg(test)]
mod tests;

pub mod function;
