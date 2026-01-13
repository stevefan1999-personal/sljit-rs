//! WebAssembly execution context
//!
//! This module defines the `WasmContext` struct that provides JIT-compiled
//! functions with access to runtime state (memory, globals, tables, functions).
//!
//! ## JIT Calling Convention
//!
//! All JIT-compiled functions use a uniform calling convention:
//! - R0: Context pointer (`*mut WasmContext`)
//! - R1: Argument count (`argc: usize`)
//! - R2: Argument pointer (`argv: *const usize`) - array of word-sized values
//!
//! This design supports:
//! - Unlimited number of arguments (not limited by register count)
//! - Fast access to runtime state without thread-local storage
//! - Uniform calling convention for wasm-to-wasm and host-to-wasm calls

use crate::module::InternalFuncType;
use crate::store::Store;
use crate::types::{FuncIdx, Value};

/// WebAssembly execution context
///
/// This struct is passed to all JIT-compiled functions via R0.
/// It provides fast access to all runtime state needed during execution.
///
/// # Safety
///
/// The pointers in this struct must remain valid for the duration of
/// the wasm function execution. The Store owns all the data and ensures
/// the pointers remain valid.
#[repr(C)]
#[derive(Debug)]
pub struct WasmContext {
    /// Pointer to the Store (for host function callbacks)
    /// This allows host functions to access the full Store API
    pub store_ptr: *mut Store,

    // ========== Memory ==========
    /// Direct pointer to memory data (for fast loads/stores)
    /// This is Memory.data.as_mut_ptr()
    pub memory_data: *mut u8,
    /// Pointer to current memory size in pages
    /// This is &Memory.current_pages
    pub memory_size: *const u32,
    /// Maximum memory pages (u32::MAX if no limit)
    pub memory_max_pages: u32,

    // ========== Globals ==========
    /// Array of pointers to global values
    /// Each pointer points to a Value in the Store
    pub globals: *const *mut Value,
    /// Number of globals
    pub globals_count: u32,

    // ========== Functions ==========
    /// Array of function code pointers
    /// For wasm functions: points to compiled code
    /// For host functions: points to trampoline
    pub functions: *const usize,
    /// Number of functions
    pub functions_count: u32,

    // ========== Tables ==========
    /// Array of table data pointers
    /// Each pointer points to the table's elements array
    pub tables: *const *const Option<FuncIdx>,
    /// Array of table sizes
    pub table_sizes: *const u32,
    /// Number of tables
    pub tables_count: u32,

    // ========== Function Types (for call_indirect) ==========
    /// Array of function type definitions
    pub func_types: *const InternalFuncType,
    /// Number of function types
    pub func_types_count: u32,
}

impl WasmContext {
    /// Create a null/empty context (for testing)
    pub const fn null() -> Self {
        Self {
            store_ptr: std::ptr::null_mut(),
            memory_data: std::ptr::null_mut(),
            memory_size: std::ptr::null(),
            memory_max_pages: 0,
            globals: std::ptr::null(),
            globals_count: 0,
            functions: std::ptr::null(),
            functions_count: 0,
            tables: std::ptr::null(),
            table_sizes: std::ptr::null(),
            tables_count: 0,
            func_types: std::ptr::null(),
            func_types_count: 0,
        }
    }

    /// Get memory data pointer
    #[inline]
    pub fn memory_ptr(&self) -> *mut u8 {
        self.memory_data
    }

    /// Get current memory size in pages
    #[inline]
    pub fn memory_pages(&self) -> u32 {
        if self.memory_size.is_null() {
            0
        } else {
            unsafe { *self.memory_size }
        }
    }

    /// Get a global value pointer by index
    #[inline]
    pub fn global_ptr(&self, index: u32) -> Option<*mut Value> {
        if index < self.globals_count && !self.globals.is_null() {
            Some(unsafe { *self.globals.add(index as usize) })
        } else {
            None
        }
    }

    /// Get a function code pointer by index
    #[inline]
    pub fn func_ptr(&self, index: u32) -> Option<usize> {
        if index < self.functions_count && !self.functions.is_null() {
            Some(unsafe { *self.functions.add(index as usize) })
        } else {
            None
        }
    }

    /// Get a table data pointer by index
    #[inline]
    pub fn table_ptr(&self, index: u32) -> Option<*const Option<FuncIdx>> {
        if index < self.tables_count && !self.tables.is_null() {
            Some(unsafe { *self.tables.add(index as usize) })
        } else {
            None
        }
    }

    /// Get a table size by index
    #[inline]
    pub fn table_size(&self, index: u32) -> Option<u32> {
        if index < self.tables_count && !self.table_sizes.is_null() {
            Some(unsafe { *self.table_sizes.add(index as usize) })
        } else {
            None
        }
    }
}

/// Builder for creating WasmContext instances
///
/// This is used by the Store to construct a context before calling
/// a wasm function.
#[derive(Debug, Default)]
pub struct WasmContextBuilder {
    // Memory
    memory_data: *mut u8,
    memory_size: *const u32,
    memory_max_pages: u32,

    // Globals - stored as Vec for building, converted to raw ptr on build
    globals: Vec<*mut Value>,

    // Functions - stored as Vec for building
    functions: Vec<usize>,

    // Tables
    tables: Vec<*const Option<FuncIdx>>,
    table_sizes: Vec<u32>,

    // Function types
    func_types: Vec<InternalFuncType>,
}

impl WasmContextBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set memory information
    pub fn memory(mut self, data: *mut u8, size_ptr: *const u32, max_pages: Option<u32>) -> Self {
        self.memory_data = data;
        self.memory_size = size_ptr;
        self.memory_max_pages = max_pages.unwrap_or(u32::MAX);
        self
    }

    /// Add a global
    pub fn add_global(&mut self, value_ptr: *mut Value) {
        self.globals.push(value_ptr);
    }

    /// Add a function
    pub fn add_function(&mut self, code_ptr: usize) {
        self.functions.push(code_ptr);
    }

    /// Add a table
    pub fn add_table(&mut self, data_ptr: *const Option<FuncIdx>, size: u32) {
        self.tables.push(data_ptr);
        self.table_sizes.push(size);
    }

    /// Add a function type
    pub fn add_func_type(&mut self, func_type: InternalFuncType) {
        self.func_types.push(func_type);
    }

    /// Build the context
    ///
    /// # Safety
    ///
    /// The returned WasmContext contains raw pointers that must remain valid.
    /// The caller must ensure the underlying data (globals, functions, tables)
    /// outlives the context.
    pub fn build(self, store_ptr: *mut Store) -> (WasmContext, WasmContextStorage) {
        // Move vectors to storage to keep them alive
        let storage = WasmContextStorage {
            globals: self.globals,
            functions: self.functions,
            tables: self.tables,
            table_sizes: self.table_sizes,
            func_types: self.func_types,
        };

        let ctx = WasmContext {
            store_ptr,
            memory_data: self.memory_data,
            memory_size: self.memory_size,
            memory_max_pages: self.memory_max_pages,
            globals: storage.globals.as_ptr(),
            globals_count: storage.globals.len() as u32,
            functions: storage.functions.as_ptr(),
            functions_count: storage.functions.len() as u32,
            tables: storage.tables.as_ptr(),
            table_sizes: storage.table_sizes.as_ptr(),
            tables_count: storage.tables.len() as u32,
            func_types: storage.func_types.as_ptr(),
            func_types_count: storage.func_types.len() as u32,
        };

        (ctx, storage)
    }
}

/// Storage for WasmContext data
///
/// This struct owns the vectors that WasmContext points to.
/// It must be kept alive as long as the WasmContext is in use.
#[derive(Debug)]
pub struct WasmContextStorage {
    pub globals: Vec<*mut Value>,
    pub functions: Vec<usize>,
    pub tables: Vec<*const Option<FuncIdx>>,
    pub table_sizes: Vec<u32>,
    pub func_types: Vec<InternalFuncType>,
}

/// Call helper macro for invoking JIT functions with the new calling convention
///
/// This macro generates type-safe call methods for CompiledFunction.
#[macro_export]
macro_rules! impl_jit_call {
    // Base case: generates call0 through call_n methods
    () => {
        /// Call a JIT function with context and arguments
        ///
        /// # Arguments
        /// * `ctx` - Pointer to WasmContext
        /// * `args` - Slice of word-sized arguments
        ///
        /// # Returns
        /// The word-sized return value (or 0 for void functions)
        #[inline]
        pub fn call_with_context(
            &self,
            ctx: *mut $crate::context::WasmContext,
            args: &[usize],
        ) -> usize {
            // JIT calling convention: (ctx: *mut WasmContext, argc: usize, argv: *const usize) -> usize
            let func: fn(*mut $crate::context::WasmContext, usize, *const usize) -> usize =
                unsafe { std::mem::transmute(self.code.get()) };
            func(ctx, args.len(), args.as_ptr())
        }

        /// Call a JIT function with context and no arguments
        #[inline]
        pub fn call0_ctx(&self, ctx: *mut $crate::context::WasmContext) -> usize {
            let func: fn(*mut $crate::context::WasmContext, usize, *const usize) -> usize =
                unsafe { std::mem::transmute(self.code.get()) };
            func(ctx, 0, std::ptr::null())
        }

        /// Call a JIT function with context and 1 argument
        #[inline]
        pub fn call1_ctx(&self, ctx: *mut $crate::context::WasmContext, a: usize) -> usize {
            let args = [a];
            let func: fn(*mut $crate::context::WasmContext, usize, *const usize) -> usize =
                unsafe { std::mem::transmute(self.code.get()) };
            func(ctx, 1, args.as_ptr())
        }

        /// Call a JIT function with context and 2 arguments
        #[inline]
        pub fn call2_ctx(
            &self,
            ctx: *mut $crate::context::WasmContext,
            a: usize,
            b: usize,
        ) -> usize {
            let args = [a, b];
            let func: fn(*mut $crate::context::WasmContext, usize, *const usize) -> usize =
                unsafe { std::mem::transmute(self.code.get()) };
            func(ctx, 2, args.as_ptr())
        }

        /// Call a JIT function with context and 3 arguments
        #[inline]
        pub fn call3_ctx(
            &self,
            ctx: *mut $crate::context::WasmContext,
            a: usize,
            b: usize,
            c: usize,
        ) -> usize {
            let args = [a, b, c];
            let func: fn(*mut $crate::context::WasmContext, usize, *const usize) -> usize =
                unsafe { std::mem::transmute(self.code.get()) };
            func(ctx, 3, args.as_ptr())
        }

        /// Call a JIT function with context and 4 arguments
        #[inline]
        pub fn call4_ctx(
            &self,
            ctx: *mut $crate::context::WasmContext,
            a: usize,
            b: usize,
            c: usize,
            d: usize,
        ) -> usize {
            let args = [a, b, c, d];
            let func: fn(*mut $crate::context::WasmContext, usize, *const usize) -> usize =
                unsafe { std::mem::transmute(self.code.get()) };
            func(ctx, 4, args.as_ptr())
        }

        /// Call a JIT function with context and variable arguments (via slice)
        #[inline]
        pub fn call_slice_ctx(
            &self,
            ctx: *mut $crate::context::WasmContext,
            args: &[usize],
        ) -> usize {
            let func: fn(*mut $crate::context::WasmContext, usize, *const usize) -> usize =
                unsafe { std::mem::transmute(self.code.get()) };
            func(ctx, args.len(), args.as_ptr())
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_null_context() {
        let ctx = WasmContext::null();
        assert!(ctx.store_ptr.is_null());
        assert!(ctx.memory_data.is_null());
        assert_eq!(ctx.memory_pages(), 0);
        assert_eq!(ctx.globals_count, 0);
        assert!(ctx.global_ptr(0).is_none());
    }

    #[test]
    fn test_context_builder() {
        let mut builder = WasmContextBuilder::new();

        // Add some mock data
        let mut global_value = crate::types::Value::I32(42);
        builder.add_global(&mut global_value as *mut _);
        builder.add_function(0x1234);

        let (ctx, _storage) = builder.build(std::ptr::null_mut());

        assert_eq!(ctx.globals_count, 1);
        assert_eq!(ctx.functions_count, 1);
        assert_eq!(ctx.func_ptr(0), Some(0x1234));
    }
}
