//! Runtime store and instance types for the WebAssembly JIT compiler
//!
//! This module contains the runtime state management types:
//! - `Store` - Runtime state container that owns all instances
//! - `Memory` - Linear memory with bounds checking
//! - `Table` - Table of function references
//! - `Global` - Global variable with mutability
//! - `Func` - Callable function (host or wasm)

use std::ptr;
use std::sync::Arc;
use std::sync::atomic::{AtomicPtr, Ordering};

use crate::Engine;
use crate::trampoline::{LibffiTrampoline, create_libffi_trampoline};
use crate::types::{
    FuncIdx, FuncType, GlobalIdx, MemoryIdx, RefType, TableIdx, Trap, TrapKind, ValType, Value,
};

// ============================================================================
// Global Store Context (for trampolines)
// ============================================================================

/// Global pointer to the currently active Store
/// This is used by trampolines to access the Store when called from JIT code
/// Using AtomicPtr for thread-safety without needing Mutex
static CURRENT_STORE: AtomicPtr<Store> = AtomicPtr::new(ptr::null_mut());

/// Execute a closure with the Store context set up for trampolines
/// This is necessary when calling wasm functions that might call back to host functions
pub fn with_store_context<R>(store: &mut Store, f: impl FnOnce() -> R) -> R {
    let store_ptr = store as *mut Store;

    // Save the old context (for nested calls)
    let old = CURRENT_STORE.swap(store_ptr, Ordering::SeqCst);

    // Execute the closure
    let result = f();

    // Restore the old context
    CURRENT_STORE.store(old, Ordering::SeqCst);

    result
}

/// Get the current Store from global context (used by trampolines)
/// Returns None if no Store context is active
pub(crate) fn get_current_store() -> Option<*mut Store> {
    let ptr = CURRENT_STORE.load(Ordering::SeqCst);
    if ptr.is_null() { None } else { Some(ptr) }
}

// ============================================================================
// Memory
// ============================================================================

/// Page size in bytes (64KB)
pub const PAGE_SIZE: usize = 65536;

/// Linear memory instance
#[derive(Debug)]
pub struct Memory {
    /// The actual memory data
    data: Vec<u8>,
    /// Current size in pages (64KB each)
    current_pages: u32,
    /// Maximum size in pages (optional)
    max_pages: Option<u32>,
}

impl Memory {
    /// Create a new memory with initial size
    pub fn new(initial: u32, maximum: Option<u32>) -> Self {
        let size = (initial as usize) * PAGE_SIZE;
        Self {
            data: vec![0u8; size],
            current_pages: initial,
            max_pages: maximum,
        }
    }

    /// Current size in pages
    #[inline]
    pub fn size(&self) -> u32 {
        self.current_pages
    }

    /// Current size in bytes
    #[inline]
    pub fn byte_size(&self) -> usize {
        self.data.len()
    }

    /// Maximum size in pages (if specified)
    #[inline]
    pub fn max_pages(&self) -> Option<u32> {
        self.max_pages
    }

    /// Grow memory by delta pages. Returns previous size or -1 on failure
    pub fn grow(&mut self, delta: u32) -> i32 {
        let old_size = self.current_pages;
        let new_size = old_size.checked_add(delta);

        match new_size {
            Some(new_pages) => {
                // Check against maximum
                if let Some(max) = self.max_pages
                    && new_pages > max
                {
                    return -1;
                }

                // Check against implementation limit (4GB = 65536 pages)
                if new_pages > 65536 {
                    return -1;
                }

                // Resize the vector
                let new_byte_size = (new_pages as usize) * PAGE_SIZE;
                self.data.resize(new_byte_size, 0);
                self.current_pages = new_pages;
                old_size as i32
            }
            None => -1,
        }
    }

    /// Read bytes from memory
    pub fn read(&self, offset: usize, len: usize) -> Result<&[u8], Trap> {
        let end = offset
            .checked_add(len)
            .ok_or_else(Trap::memory_out_of_bounds)?;
        if end > self.data.len() {
            return Err(Trap::memory_out_of_bounds());
        }
        Ok(&self.data[offset..end])
    }

    /// Write bytes to memory
    pub fn write(&mut self, offset: usize, data: &[u8]) -> Result<(), Trap> {
        let end = offset
            .checked_add(data.len())
            .ok_or_else(Trap::memory_out_of_bounds)?;
        if end > self.data.len() {
            return Err(Trap::memory_out_of_bounds());
        }
        self.data[offset..end].copy_from_slice(data);
        Ok(())
    }

    /// Read a single byte
    pub fn read_u8(&self, offset: usize) -> Result<u8, Trap> {
        self.data
            .get(offset)
            .copied()
            .ok_or_else(Trap::memory_out_of_bounds)
    }

    /// Read a u16 (little-endian)
    pub fn read_u16(&self, offset: usize) -> Result<u16, Trap> {
        let bytes = self.read(offset, 2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    /// Read a u32 (little-endian)
    pub fn read_u32(&self, offset: usize) -> Result<u32, Trap> {
        let bytes = self.read(offset, 4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    /// Read a u64 (little-endian)
    pub fn read_u64(&self, offset: usize) -> Result<u64, Trap> {
        let bytes = self.read(offset, 8)?;
        Ok(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    /// Write a single byte
    pub fn write_u8(&mut self, offset: usize, value: u8) -> Result<(), Trap> {
        if offset >= self.data.len() {
            return Err(Trap::memory_out_of_bounds());
        }
        self.data[offset] = value;
        Ok(())
    }

    /// Write a u16 (little-endian)
    pub fn write_u16(&mut self, offset: usize, value: u16) -> Result<(), Trap> {
        self.write(offset, &value.to_le_bytes())
    }

    /// Write a u32 (little-endian)
    pub fn write_u32(&mut self, offset: usize, value: u32) -> Result<(), Trap> {
        self.write(offset, &value.to_le_bytes())
    }

    /// Write a u64 (little-endian)
    pub fn write_u64(&mut self, offset: usize, value: u64) -> Result<(), Trap> {
        self.write(offset, &value.to_le_bytes())
    }

    /// Get raw pointer for JIT code (data start)
    #[inline]
    pub fn data_ptr(&self) -> *const u8 {
        self.data.as_ptr()
    }

    /// Get mutable raw pointer for JIT code
    #[inline]
    pub fn data_ptr_mut(&mut self) -> *mut u8 {
        self.data.as_mut_ptr()
    }

    /// Get pointer to size (in pages) for JIT code
    #[inline]
    pub fn size_ptr(&self) -> *const u32 {
        &self.current_pages
    }

    /// Get the data slice
    #[inline]
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Get mutable data slice
    #[inline]
    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }
}

// ============================================================================
// Table
// ============================================================================

/// Table of function references
#[derive(Debug)]
pub struct Table {
    /// Table elements - function indices or null
    elements: Vec<Option<FuncIdx>>,
    /// Maximum size (optional)
    max_size: Option<u32>,
    /// Element type
    elem_type: RefType,
}

impl Table {
    /// Create a new table with initial size
    pub fn new(initial: u32, maximum: Option<u32>, elem_type: RefType) -> Self {
        Self {
            elements: vec![None; initial as usize],
            max_size: maximum,
            elem_type,
        }
    }

    /// Get element at index
    pub fn get(&self, index: u32) -> Result<Option<FuncIdx>, Trap> {
        self.elements
            .get(index as usize)
            .copied()
            .ok_or_else(Trap::table_out_of_bounds)
    }

    /// Set element at index
    pub fn set(&mut self, index: u32, value: Option<FuncIdx>) -> Result<(), Trap> {
        if index as usize >= self.elements.len() {
            return Err(Trap::table_out_of_bounds());
        }
        self.elements[index as usize] = value;
        Ok(())
    }

    /// Grow table by delta elements. Returns previous size or -1 on failure
    pub fn grow(&mut self, delta: u32, init: Option<FuncIdx>) -> i32 {
        let old_size = self.elements.len() as u32;
        let new_size = old_size.checked_add(delta);

        match new_size {
            Some(new_len) => {
                // Check against maximum
                if let Some(max) = self.max_size
                    && new_len > max
                {
                    return -1;
                }

                // Resize with init value
                self.elements.resize(new_len as usize, init);
                old_size as i32
            }
            None => -1,
        }
    }

    /// Current size
    #[inline]
    pub fn size(&self) -> u32 {
        self.elements.len() as u32
    }

    /// Element type
    #[inline]
    pub fn elem_type(&self) -> RefType {
        self.elem_type
    }

    /// Get raw pointer for JIT code
    #[inline]
    pub fn data_ptr(&self) -> *const Option<FuncIdx> {
        self.elements.as_ptr()
    }

    /// Get the elements slice
    #[inline]
    pub fn elements(&self) -> &[Option<FuncIdx>] {
        &self.elements
    }
}

// ============================================================================
// Global
// ============================================================================

/// Global variable instance
#[derive(Debug, Clone)]
pub struct Global {
    /// Current value
    value: Value,
    /// Whether the global is mutable
    mutable: bool,
}

impl Global {
    /// Create a new global
    pub fn new(value: Value, mutable: bool) -> Self {
        Self { value, mutable }
    }

    /// Get current value
    #[inline]
    pub fn get(&self) -> Value {
        self.value
    }

    /// Set value - fails if immutable
    pub fn set(&mut self, value: Value) -> Result<(), Trap> {
        if !self.mutable {
            return Err(Trap::new(
                TrapKind::Host,
                "cannot set immutable global".to_string(),
            ));
        }
        // Type check
        if std::mem::discriminant(&self.value) != std::mem::discriminant(&value) {
            return Err(Trap::new(
                TrapKind::Host,
                "global type mismatch".to_string(),
            ));
        }
        self.value = value;
        Ok(())
    }

    /// Get raw pointer to value for JIT code
    #[inline]
    pub fn value_ptr(&self) -> *const Value {
        &self.value
    }

    /// Get mutable raw pointer to value for JIT code
    #[inline]
    pub fn value_ptr_mut(&mut self) -> *mut Value {
        &mut self.value
    }

    /// Whether the global is mutable
    #[inline]
    pub fn is_mutable(&self) -> bool {
        self.mutable
    }

    /// Get the value type
    #[inline]
    pub fn ty(&self) -> ValType {
        self.value.ty()
    }
}

// ============================================================================
// Func
// ============================================================================

/// A callable function - either host or wasm
#[derive(Clone)]
pub enum Func {
    /// A WebAssembly function
    Wasm(WasmFunc),
    /// A host-provided function
    Host(HostFunc),
}

impl std::fmt::Debug for Func {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Func::Wasm(wf) => f
                .debug_struct("Func::Wasm")
                .field("instance_idx", &wf.instance_idx)
                .field("func_idx", &wf.func_idx)
                .field("code_ptr", &wf.code_ptr)
                .finish(),
            Func::Host(_) => f.debug_struct("Func::Host").finish_non_exhaustive(),
        }
    }
}

/// A WebAssembly function
#[derive(Clone, Debug)]
pub struct WasmFunc {
    /// Instance this function belongs to (index in store's instance list)
    pub instance_idx: u32,
    /// Function index within the module
    pub func_idx: u32,
    /// Pointer to compiled code
    pub code_ptr: usize,
    /// Function signature
    pub func_type: FuncType,
}

/// Host function callback type
pub type HostFuncCallback =
    Arc<dyn Fn(&mut Store, &[Value]) -> Result<Vec<Value>, Trap> + Send + Sync>;

/// A host-provided function
#[derive(Clone)]
pub struct HostFunc {
    /// Function signature
    pub func_type: FuncType,
    /// Type-erased callable
    pub callback: HostFuncCallback,
    /// Trampoline code pointer (for JIT calls)
    pub trampoline_ptr: usize,
}

impl Func {
    /// Create a new host function (trampoline will be generated when added to store)
    pub fn wrap<F>(func_type: FuncType, callback: F) -> Self
    where
        F: Fn(&mut Store, &[Value]) -> Result<Vec<Value>, Trap> + Send + Sync + 'static,
    {
        Func::Host(HostFunc {
            func_type,
            callback: Arc::new(callback),
            trampoline_ptr: 0, // Will be set when added to store
        })
    }

    /// Get the function signature
    pub fn ty(&self) -> &FuncType {
        match self {
            Func::Wasm(wf) => &wf.func_type,
            Func::Host(hf) => &hf.func_type,
        }
    }

    /// Check if this is a host function
    #[inline]
    pub fn is_host(&self) -> bool {
        matches!(self, Func::Host(_))
    }

    /// Check if this is a wasm function
    #[inline]
    pub fn is_wasm(&self) -> bool {
        matches!(self, Func::Wasm(_))
    }

    /// Get the code pointer (for wasm functions only)
    pub fn code_ptr(&self) -> Option<usize> {
        match self {
            Func::Wasm(wf) => Some(wf.code_ptr),
            Func::Host(_) => None,
        }
    }

    /// Get the callable code pointer for JIT calls
    /// Returns the code_ptr for wasm functions, trampoline_ptr for host functions
    pub fn callable_ptr(&self) -> usize {
        match self {
            Func::Wasm(wf) => wf.code_ptr,
            Func::Host(hf) => hf.trampoline_ptr,
        }
    }

    /// Get a pointer to the code_ptr field (for wasm functions only)
    /// This is used for indirect calls where the code_ptr may be updated after compilation
    pub fn code_ptr_ptr(&self) -> Option<*const usize> {
        match self {
            Func::Wasm(wf) => Some(&wf.code_ptr as *const usize),
            Func::Host(_) => None,
        }
    }
}

// ============================================================================
// Store
// ============================================================================

/// Runtime state container - owns all runtime instances
pub struct Store {
    /// Reference to the engine for compilation settings
    engine: Arc<Engine>,
    /// All memory instances in this store
    memories: Vec<Memory>,
    /// All table instances in this store
    tables: Vec<Table>,
    /// All global instances in this store
    globals: Vec<Global>,
    /// All function instances in this store
    funcs: Vec<Func>,
    /// Fuel for execution limits - optional
    fuel: Option<u64>,
    /// Generated libffi trampolines for host functions (kept alive here)
    trampolines: Vec<LibffiTrampoline>,
}

impl std::fmt::Debug for Store {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Store")
            .field("engine", &self.engine)
            .field("memories", &self.memories.len())
            .field("tables", &self.tables.len())
            .field("globals", &self.globals.len())
            .field("funcs", &self.funcs.len())
            .field("fuel", &self.fuel)
            .field("trampolines", &self.trampolines.len())
            .finish()
    }
}

impl Store {
    /// Create a new store with the given engine
    pub fn new(engine: Arc<Engine>) -> Self {
        Self {
            engine,
            memories: Vec::new(),
            tables: Vec::new(),
            globals: Vec::new(),
            funcs: Vec::new(),
            fuel: None,
            trampolines: Vec::new(),
        }
    }

    /// Get the engine
    #[inline]
    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    /// Enable fuel metering with initial fuel amount
    pub fn set_fuel(&mut self, fuel: u64) {
        self.fuel = Some(fuel);
    }

    /// Get remaining fuel
    #[inline]
    pub fn fuel(&self) -> Option<u64> {
        self.fuel
    }

    /// Consume fuel, returns error if out of fuel
    pub fn consume_fuel(&mut self, amount: u64) -> Result<(), Trap> {
        if let Some(ref mut fuel) = self.fuel {
            if *fuel < amount {
                return Err(Trap::out_of_fuel());
            }
            *fuel -= amount;
        }
        Ok(())
    }

    // ========== Memory operations ==========

    /// Allocate a new memory and return its index
    pub fn alloc_memory(&mut self, initial: u32, maximum: Option<u32>) -> MemoryIdx {
        let idx = MemoryIdx(self.memories.len() as u32);
        self.memories.push(Memory::new(initial, maximum));
        idx
    }

    /// Get a memory by index
    pub fn memory(&self, idx: MemoryIdx) -> Option<&Memory> {
        self.memories.get(idx.0 as usize)
    }

    /// Get a mutable memory by index
    pub fn memory_mut(&mut self, idx: MemoryIdx) -> Option<&mut Memory> {
        self.memories.get_mut(idx.0 as usize)
    }

    /// Get the number of memories
    #[inline]
    pub fn num_memories(&self) -> usize {
        self.memories.len()
    }

    // ========== Table operations ==========

    /// Allocate a new table and return its index
    pub fn alloc_table(
        &mut self,
        initial: u32,
        maximum: Option<u32>,
        elem_type: RefType,
    ) -> TableIdx {
        let idx = TableIdx(self.tables.len() as u32);
        self.tables.push(Table::new(initial, maximum, elem_type));
        idx
    }

    /// Get a table by index
    pub fn table(&self, idx: TableIdx) -> Option<&Table> {
        self.tables.get(idx.0 as usize)
    }

    /// Get a mutable table by index
    pub fn table_mut(&mut self, idx: TableIdx) -> Option<&mut Table> {
        self.tables.get_mut(idx.0 as usize)
    }

    /// Get the number of tables
    #[inline]
    pub fn num_tables(&self) -> usize {
        self.tables.len()
    }

    // ========== Global operations ==========

    /// Allocate a new global and return its index
    pub fn alloc_global(&mut self, value: Value, mutable: bool) -> GlobalIdx {
        let idx = GlobalIdx(self.globals.len() as u32);
        self.globals.push(Global::new(value, mutable));
        idx
    }

    /// Get a global by index
    pub fn global(&self, idx: GlobalIdx) -> Option<&Global> {
        self.globals.get(idx.0 as usize)
    }

    /// Get a mutable global by index
    pub fn global_mut(&mut self, idx: GlobalIdx) -> Option<&mut Global> {
        self.globals.get_mut(idx.0 as usize)
    }

    /// Get the number of globals
    #[inline]
    pub fn num_globals(&self) -> usize {
        self.globals.len()
    }

    // ========== Function operations ==========

    /// Allocate a new function and return its index
    /// For host functions, this also generates a libffi trampoline
    pub fn alloc_func(&mut self, mut func: Func) -> FuncIdx {
        let idx = FuncIdx(self.funcs.len() as u32);

        // For host functions, generate a libffi trampoline
        if let Func::Host(ref mut hf) = func {
            if let Some(trampoline) =
                create_libffi_trampoline(idx, &hf.func_type, hf.callback.clone())
            {
                hf.trampoline_ptr = trampoline.fn_ptr;
                self.trampolines.push(trampoline);
            }
        }

        self.funcs.push(func);
        idx
    }

    /// Get a function by index
    pub fn func(&self, idx: FuncIdx) -> Option<&Func> {
        self.funcs.get(idx.0 as usize)
    }

    /// Get a mutable function by index
    pub fn func_mut(&mut self, idx: FuncIdx) -> Option<&mut Func> {
        self.funcs.get_mut(idx.0 as usize)
    }

    /// Get the number of functions
    #[inline]
    pub fn num_funcs(&self) -> usize {
        self.funcs.len()
    }

    /// Call a function by index
    pub fn call(&mut self, func_idx: FuncIdx, args: &[Value]) -> Result<Vec<Value>, Trap> {
        let func = self
            .funcs
            .get(func_idx.0 as usize)
            .ok_or_else(|| Trap::new(TrapKind::Host, format!("function {} not found", func_idx.0)))?
            .clone();

        match func {
            Func::Host(hf) => {
                // Verify argument types
                if args.len() != hf.func_type.params().len() {
                    return Err(Trap::new(
                        TrapKind::Host,
                        format!(
                            "wrong number of arguments: expected {}, got {}",
                            hf.func_type.params().len(),
                            args.len()
                        ),
                    ));
                }
                for (i, (arg, expected_ty)) in args.iter().zip(hf.func_type.params()).enumerate() {
                    if arg.ty() != *expected_ty {
                        return Err(Trap::new(
                            TrapKind::Host,
                            format!(
                                "argument {} type mismatch: expected {:?}, got {:?}",
                                i,
                                expected_ty,
                                arg.ty()
                            ),
                        ));
                    }
                }

                // Call the host function
                (hf.callback)(self, args)
            }
            Func::Wasm(wf) => {
                // For wasm functions, we need to call through the JIT code
                // This requires setting up the calling convention properly
                self.call_wasm_func(&wf, args)
            }
        }
    }

    /// Call a WebAssembly function through JIT code
    ///
    /// JIT calling convention: wasm arguments are passed directly (NO store_ptr).
    /// Host functions get the store pointer from thread-local storage (TLS) via
    /// `with_store_context()`.
    ///
    /// IMPORTANT: On 32-bit platforms, sljit uses word-sized (32-bit) arguments,
    /// so we must use usize/isize for function pointers, not i64.
    fn call_wasm_func(&mut self, wf: &WasmFunc, args: &[Value]) -> Result<Vec<Value>, Trap> {
        // Verify argument count
        if args.len() != wf.func_type.params().len() {
            return Err(Trap::new(
                TrapKind::Host,
                format!(
                    "wrong number of arguments: expected {}, got {}",
                    wf.func_type.params().len(),
                    args.len()
                ),
            ));
        }

        // Get the code pointer
        let code_ptr = wf.code_ptr;
        if code_ptr == 0 {
            return Err(Trap::new(
                TrapKind::Host,
                "null function pointer".to_string(),
            ));
        }

        // Store result types for later conversion
        let result_types = wf.func_type.results().to_vec();

        // Convert args to word-sized values for JIT calling convention
        // On 64-bit: usize = u64, on 32-bit: usize = u32
        let arg0 = args.first().map(|a| self.convert_arg_to_word(a));
        let arg1 = args.get(1).map(|a| self.convert_arg_to_word(a));
        let arg2 = args.get(2).map(|a| self.convert_arg_to_word(a));
        let num_args = args.len();
        let num_results = result_types.len();

        // Set up store context in TLS so trampolines can access it
        let result = with_store_context(self, || {
            // Call based on number of parameters and results
            // JIT functions receive wasm args directly (NO store_ptr)
            // Use usize for word-sized arguments to match sljit's calling convention
            match (num_args, num_results) {
                (0, 0) => {
                    let func: fn() = unsafe { std::mem::transmute(code_ptr) };
                    func();
                    Ok(0usize)
                }
                (0, 1) => {
                    let func: fn() -> usize = unsafe { std::mem::transmute(code_ptr) };
                    Ok(func())
                }
                (1, 0) => {
                    let func: fn(usize) = unsafe { std::mem::transmute(code_ptr) };
                    func(arg0.unwrap());
                    Ok(0usize)
                }
                (1, 1) => {
                    let func: fn(usize) -> usize = unsafe { std::mem::transmute(code_ptr) };
                    Ok(func(arg0.unwrap()))
                }
                (2, 0) => {
                    let func: fn(usize, usize) = unsafe { std::mem::transmute(code_ptr) };
                    func(arg0.unwrap(), arg1.unwrap());
                    Ok(0usize)
                }
                (2, 1) => {
                    let func: fn(usize, usize) -> usize = unsafe { std::mem::transmute(code_ptr) };
                    Ok(func(arg0.unwrap(), arg1.unwrap()))
                }
                (3, 0) => {
                    let func: fn(usize, usize, usize) = unsafe { std::mem::transmute(code_ptr) };
                    func(arg0.unwrap(), arg1.unwrap(), arg2.unwrap());
                    Ok(0usize)
                }
                (3, 1) => {
                    let func: fn(usize, usize, usize) -> usize =
                        unsafe { std::mem::transmute(code_ptr) };
                    Ok(func(arg0.unwrap(), arg1.unwrap(), arg2.unwrap()))
                }
                _ => Err(Trap::new(
                    TrapKind::Host,
                    format!(
                        "unsupported function signature: {} params, {} results",
                        num_args, num_results
                    ),
                )),
            }
        });

        // Convert the result
        match result {
            Ok(raw_result) => {
                if num_results == 0 {
                    Ok(vec![])
                } else {
                    Ok(vec![
                        self.convert_word_to_result(raw_result, &result_types[0]),
                    ])
                }
            }
            Err(trap) => Err(trap),
        }
    }

    /// Convert a Value to word-sized (usize) for calling JIT code
    ///
    /// On 64-bit platforms, usize is 64 bits and can hold all wasm value types.
    /// On 32-bit platforms, usize is 32 bits, which works for i32/f32 but
    /// i64/f64 values need special handling (they stay on the stack in JIT code).
    fn convert_arg_to_word(&self, value: &Value) -> usize {
        match value {
            Value::I32(v) => *v as u32 as usize,
            Value::I64(v) => *v as usize, // On 32-bit, this truncates - i64 handled separately
            Value::F32(v) => f32::to_bits(*v) as usize,
            Value::F64(v) => f64::to_bits(*v) as usize, // On 32-bit, this truncates
            Value::FuncRef(Some(idx)) => idx.0 as usize,
            Value::FuncRef(None) => usize::MAX, // -1 as usize
            Value::ExternRef(Some(idx)) => *idx as usize,
            Value::ExternRef(None) => usize::MAX,
        }
    }

    /// Convert a word-sized result from JIT code to Value
    fn convert_word_to_result(&self, result: usize, ty: &ValType) -> Value {
        match ty {
            ValType::I32 => Value::I32(result as i32),
            ValType::I64 => Value::I64(result as i64), // Sign-extends on 32-bit
            ValType::F32 => Value::F32(f32::from_bits(result as u32)),
            ValType::F64 => Value::F64(f64::from_bits(result as u64)),
            ValType::FuncRef => {
                if result == usize::MAX {
                    Value::FuncRef(None)
                } else {
                    Value::FuncRef(Some(FuncIdx(result as u32)))
                }
            }
            ValType::ExternRef => {
                if result == usize::MAX {
                    Value::ExternRef(None)
                } else {
                    Value::ExternRef(Some(result as u32))
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
