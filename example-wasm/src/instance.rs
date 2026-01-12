//! Instance and Linker for WebAssembly module instantiation
//!
//! This module contains:
//! - `Instance` - A runtime instantiation of a Module with resolved imports
//! - `Linker` - Helper for building import resolvers

use std::collections::HashMap;

use sljit::sys::Compiler;
use wasmparser::Operator;

use crate::function::Function;
use crate::module::{
    DataSegment, ElementSegment, ExportKind, ImportKind, InternalFuncType, Module,
};
use crate::store::{Func, Store, WasmFunc};
use crate::types::{
    FuncIdx, FuncType, GlobalIdx, InstantiationError, MemoryIdx, RefType, TableIdx, Trap, ValType,
    Value,
};
use crate::{FunctionEntry, GlobalInfo, MemoryInfo, TableEntry};

// ============================================================================
// Instance
// ============================================================================

/// A runtime instantiation of a Module
#[derive(Debug)]
pub struct Instance {
    /// Memory instances - indices into Store
    memories: Vec<MemoryIdx>,
    /// Table instances - indices into Store
    tables: Vec<TableIdx>,
    /// Global instances - indices into Store
    globals: Vec<GlobalIdx>,
    /// Function instances - indices into Store
    funcs: Vec<FuncIdx>,
    /// Export map: name -> (kind, index within instance)
    exports: HashMap<String, (ExportKind, u32)>,
}

impl Instance {
    /// Instantiate a module with imports resolved by the linker
    pub fn new<'a>(
        store: &mut Store,
        module: &Module<'a>,
        linker: &Linker,
    ) -> Result<Self, InstantiationError> {
        let mut memories = Vec::new();
        let mut tables = Vec::new();
        let mut globals = Vec::new();
        let mut funcs = Vec::new();

        // Phase 1: Resolve imports
        for import in module.imports() {
            match &import.kind {
                ImportKind::Func(type_idx) => {
                    let func_type =
                        module.func_types().get(*type_idx as usize).ok_or_else(|| {
                            InstantiationError::ValidationFailed(format!(
                                "invalid type index {} for import {}.{}",
                                type_idx, import.module, import.name
                            ))
                        })?;

                    let func_idx = linker.resolve_func(import.module, import.name).ok_or(
                        InstantiationError::ImportNotFound {
                            module: import.module.to_string(),
                            name: import.name.to_string(),
                        },
                    )?;

                    // Verify type matches
                    let imported_func =
                        store
                            .func(func_idx)
                            .ok_or_else(|| InstantiationError::ImportNotFound {
                                module: import.module.to_string(),
                                name: import.name.to_string(),
                            })?;

                    let expected_type = FuncType::new(
                        convert_val_types(&func_type.params),
                        convert_val_types(&func_type.results),
                    );

                    if !imported_func.ty().matches(&expected_type) {
                        return Err(InstantiationError::ImportTypeMismatch {
                            expected: format!("{}", expected_type),
                            got: format!("{}", imported_func.ty()),
                        });
                    }

                    funcs.push(func_idx);
                }
                ImportKind::Memory(mem_type) => {
                    let mem_idx = linker.resolve_memory(import.module, import.name).ok_or(
                        InstantiationError::ImportNotFound {
                            module: import.module.to_string(),
                            name: import.name.to_string(),
                        },
                    )?;

                    // Verify memory type matches (initial/max)
                    let memory = store.memory(mem_idx).ok_or_else(|| {
                        InstantiationError::ImportNotFound {
                            module: import.module.to_string(),
                            name: import.name.to_string(),
                        }
                    })?;

                    if (memory.size() as u64) < mem_type.initial {
                        return Err(InstantiationError::ImportTypeMismatch {
                            expected: format!("memory with at least {} pages", mem_type.initial),
                            got: format!("memory with {} pages", memory.size()),
                        });
                    }

                    memories.push(mem_idx);
                }
                ImportKind::Table(table_type) => {
                    let table_idx = linker.resolve_table(import.module, import.name).ok_or(
                        InstantiationError::ImportNotFound {
                            module: import.module.to_string(),
                            name: import.name.to_string(),
                        },
                    )?;

                    // Verify table type matches
                    let table = store.table(table_idx).ok_or_else(|| {
                        InstantiationError::ImportNotFound {
                            module: import.module.to_string(),
                            name: import.name.to_string(),
                        }
                    })?;

                    if (table.size() as u64) < table_type.initial {
                        return Err(InstantiationError::ImportTypeMismatch {
                            expected: format!(
                                "table with at least {} elements",
                                table_type.initial
                            ),
                            got: format!("table with {} elements", table.size()),
                        });
                    }

                    tables.push(table_idx);
                }
                ImportKind::Global(global_type) => {
                    let global_idx = linker.resolve_global(import.module, import.name).ok_or(
                        InstantiationError::ImportNotFound {
                            module: import.module.to_string(),
                            name: import.name.to_string(),
                        },
                    )?;

                    // Verify global type matches
                    let global = store.global(global_idx).ok_or_else(|| {
                        InstantiationError::ImportNotFound {
                            module: import.module.to_string(),
                            name: import.name.to_string(),
                        }
                    })?;

                    let expected_val_type: ValType = global_type.content_type.into();
                    if global.ty() != expected_val_type
                        || global.is_mutable() != global_type.mutable
                    {
                        return Err(InstantiationError::ImportTypeMismatch {
                            expected: format!(
                                "global {:?} mutable={}",
                                expected_val_type, global_type.mutable
                            ),
                            got: format!(
                                "global {:?} mutable={}",
                                global.ty(),
                                global.is_mutable()
                            ),
                        });
                    }

                    globals.push(global_idx);
                }
            }
        }

        // Phase 2: Allocate defined memories
        for mem_def in module.memories() {
            let mem_idx =
                store.alloc_memory(mem_def.initial as u32, mem_def.maximum.map(|m| m as u32));
            memories.push(mem_idx);
        }

        // Phase 3: Allocate defined tables
        for table_def in module.tables() {
            let elem_type = match table_def.element_type {
                crate::module::RefType::FuncRef => RefType::FuncRef,
                crate::module::RefType::ExternRef => RefType::ExternRef,
            };
            let table_idx = store.alloc_table(
                table_def.initial as u32,
                table_def.maximum.map(|m| m as u32),
                elem_type,
            );
            tables.push(table_idx);
        }

        // Phase 4: Allocate defined globals (evaluate init expressions)
        for global_def in module.globals() {
            let init_value = evaluate_const_expr(&global_def.init_expr, &globals, store)?;
            let global_idx = store.alloc_global(init_value, global_def.ty.mutable);
            globals.push(global_idx);
        }

        // Phase 5: Compile and allocate defined functions
        let instance_idx = 0u32; // We'll update this later if needed
        let func_bodies = module.function_bodies();
        let func_type_indices: Vec<_> = (0..func_bodies.len())
            .map(|i| module.get_func_type_index(i as u32))
            .collect();

        // First pass: create placeholder functions to get their indices
        let func_start_idx = funcs.len();
        for (i, _body) in func_bodies.iter().enumerate() {
            let type_idx = func_type_indices[i].ok_or_else(|| {
                InstantiationError::ValidationFailed(format!(
                    "missing type index for function {}",
                    i
                ))
            })?;
            let func_type = module.func_types().get(type_idx as usize).ok_or_else(|| {
                InstantiationError::ValidationFailed(format!(
                    "invalid type index {} for function {}",
                    type_idx, i
                ))
            })?;

            // Create a placeholder WasmFunc (code_ptr will be updated after compilation)
            let wasm_func = WasmFunc {
                instance_idx,
                func_idx: (func_start_idx + i) as u32,
                code_ptr: 0, // Will be updated
                func_type: FuncType::new(
                    convert_val_types(&func_type.params),
                    convert_val_types(&func_type.results),
                ),
            };
            let func_idx = store.alloc_func(Func::Wasm(wasm_func));
            funcs.push(func_idx);
        }

        // Second pass: compile functions with all info available
        compile_module_functions(store, module, &funcs, &memories, &tables, &globals)?;

        // Phase 6: Initialize data segments
        for data_seg in module.data_segments() {
            if !data_seg.passive {
                initialize_data_segment(store, data_seg, &memories, &globals)?;
            }
        }

        // Phase 7: Initialize element segments
        for elem_seg in module.element_segments() {
            if !elem_seg.passive {
                initialize_element_segment(store, elem_seg, &tables, &globals, &funcs)?;
            }
        }

        // Build export map
        let mut exports = HashMap::new();
        for export in module.exports() {
            exports.insert(
                export.name.to_string(),
                (export.kind, get_export_index(&export.kind)),
            );
        }

        // Phase 8: Run start function if present
        if let Some(start_idx) = module.start_func() {
            let func_idx = funcs.get(start_idx as usize).ok_or_else(|| {
                InstantiationError::ValidationFailed(format!(
                    "invalid start function index {}",
                    start_idx
                ))
            })?;

            store
                .call(*func_idx, &[])
                .map_err(InstantiationError::StartTrapped)?;
        }

        Ok(Self {
            memories,
            tables,
            globals,
            funcs,
            exports,
        })
    }

    /// Get an exported function by name
    pub fn get_func(&self, _store: &Store, name: &str) -> Option<FuncIdx> {
        self.exports.get(name).and_then(|(kind, idx)| {
            if let ExportKind::Func(_) = kind {
                self.funcs.get(*idx as usize).copied()
            } else {
                None
            }
        })
    }

    /// Get an exported memory by name
    pub fn get_memory(&self, name: &str) -> Option<MemoryIdx> {
        self.exports.get(name).and_then(|(kind, idx)| {
            if let ExportKind::Memory(_) = kind {
                self.memories.get(*idx as usize).copied()
            } else {
                None
            }
        })
    }

    /// Get an exported table by name
    pub fn get_table(&self, name: &str) -> Option<TableIdx> {
        self.exports.get(name).and_then(|(kind, idx)| {
            if let ExportKind::Table(_) = kind {
                self.tables.get(*idx as usize).copied()
            } else {
                None
            }
        })
    }

    /// Get an exported global by name
    pub fn get_global(&self, name: &str) -> Option<GlobalIdx> {
        self.exports.get(name).and_then(|(kind, idx)| {
            if let ExportKind::Global(_) = kind {
                self.globals.get(*idx as usize).copied()
            } else {
                None
            }
        })
    }

    /// Get all function indices
    pub fn funcs(&self) -> &[FuncIdx] {
        &self.funcs
    }

    /// Get all memory indices
    pub fn memories(&self) -> &[MemoryIdx] {
        &self.memories
    }

    /// Get all table indices
    pub fn tables(&self) -> &[TableIdx] {
        &self.tables
    }

    /// Get all global indices
    pub fn globals(&self) -> &[GlobalIdx] {
        &self.globals
    }
}

fn get_export_index(kind: &ExportKind) -> u32 {
    match kind {
        ExportKind::Func(idx) => *idx,
        ExportKind::Table(idx) => *idx,
        ExportKind::Memory(idx) => *idx,
        ExportKind::Global(idx) => *idx,
    }
}

// ============================================================================
// Linker
// ============================================================================

/// Helper for building import resolvers
#[derive(Debug, Default)]
pub struct Linker {
    /// Registered functions by (module, name)
    funcs: HashMap<(String, String), FuncIdx>,
    /// Registered memories by (module, name)
    memories: HashMap<(String, String), MemoryIdx>,
    /// Registered tables by (module, name)
    tables: HashMap<(String, String), TableIdx>,
    /// Registered globals by (module, name)
    globals: HashMap<(String, String), GlobalIdx>,
}

impl Linker {
    /// Create a new empty linker
    pub fn new() -> Self {
        Self::default()
    }

    /// Define a function
    pub fn define_func(
        &mut self,
        module: impl Into<String>,
        name: impl Into<String>,
        func_idx: FuncIdx,
    ) -> &mut Self {
        self.funcs.insert((module.into(), name.into()), func_idx);
        self
    }

    /// Define a host function directly
    pub fn func_wrap<F>(
        &mut self,
        store: &mut Store,
        module: impl Into<String>,
        name: impl Into<String>,
        func_type: FuncType,
        callback: F,
    ) -> &mut Self
    where
        F: Fn(&mut Store, &[Value]) -> Result<Vec<Value>, Trap> + Send + Sync + 'static,
    {
        let func = Func::wrap(func_type, callback);
        let func_idx = store.alloc_func(func);
        self.define_func(module, name, func_idx);
        self
    }

    /// Define a memory
    pub fn define_memory(
        &mut self,
        module: impl Into<String>,
        name: impl Into<String>,
        memory_idx: MemoryIdx,
    ) -> &mut Self {
        self.memories
            .insert((module.into(), name.into()), memory_idx);
        self
    }

    /// Define a table
    pub fn define_table(
        &mut self,
        module: impl Into<String>,
        name: impl Into<String>,
        table_idx: TableIdx,
    ) -> &mut Self {
        self.tables.insert((module.into(), name.into()), table_idx);
        self
    }

    /// Define a global
    pub fn define_global(
        &mut self,
        module: impl Into<String>,
        name: impl Into<String>,
        global_idx: GlobalIdx,
    ) -> &mut Self {
        self.globals
            .insert((module.into(), name.into()), global_idx);
        self
    }

    /// Resolve a function import
    pub fn resolve_func(&self, module: &str, name: &str) -> Option<FuncIdx> {
        self.funcs
            .get(&(module.to_string(), name.to_string()))
            .copied()
    }

    /// Resolve a memory import
    pub fn resolve_memory(&self, module: &str, name: &str) -> Option<MemoryIdx> {
        self.memories
            .get(&(module.to_string(), name.to_string()))
            .copied()
    }

    /// Resolve a table import
    pub fn resolve_table(&self, module: &str, name: &str) -> Option<TableIdx> {
        self.tables
            .get(&(module.to_string(), name.to_string()))
            .copied()
    }

    /// Resolve a global import
    pub fn resolve_global(&self, module: &str, name: &str) -> Option<GlobalIdx> {
        self.globals
            .get(&(module.to_string(), name.to_string()))
            .copied()
    }

    /// Instantiate a module using this linker
    pub fn instantiate<'a>(
        &self,
        store: &mut Store,
        module: &Module<'a>,
    ) -> Result<Instance, InstantiationError> {
        Instance::new(store, module, self)
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Convert wasmparser ValTypes to our ValTypes
fn convert_val_types(types: &[wasmparser::ValType]) -> Vec<ValType> {
    types.iter().map(|t| (*t).into()).collect()
}

/// Evaluate a constant expression
fn evaluate_const_expr(
    ops: &[Operator<'static>],
    globals: &[GlobalIdx],
    store: &Store,
) -> Result<Value, InstantiationError> {
    for op in ops {
        match op {
            Operator::I32Const { value } => return Ok(Value::I32(*value)),
            Operator::I64Const { value } => return Ok(Value::I64(*value)),
            Operator::F32Const { value } => return Ok(Value::F32(f32::from_bits(value.bits()))),
            Operator::F64Const { value } => return Ok(Value::F64(f64::from_bits(value.bits()))),
            Operator::RefNull { hty: _ } => return Ok(Value::FuncRef(None)),
            Operator::RefFunc { function_index } => {
                return Ok(Value::FuncRef(Some(FuncIdx(*function_index))));
            }
            Operator::GlobalGet { global_index } => {
                let global_idx = globals.get(*global_index as usize).ok_or_else(|| {
                    InstantiationError::ValidationFailed(format!(
                        "invalid global index {} in const expr",
                        global_index
                    ))
                })?;
                let global = store.global(*global_idx).ok_or_else(|| {
                    InstantiationError::ValidationFailed(format!(
                        "global {} not found in store",
                        global_idx.0
                    ))
                })?;
                return Ok(global.get());
            }
            Operator::End => continue,
            _ => {
                return Err(InstantiationError::ValidationFailed(format!(
                    "unsupported operator in const expr: {:?}",
                    op
                )));
            }
        }
    }
    Err(InstantiationError::ValidationFailed(
        "empty const expression".to_string(),
    ))
}

/// Evaluate a constant expression to get an i32 offset
fn evaluate_offset_expr(
    ops: &[Operator<'static>],
    globals: &[GlobalIdx],
    store: &Store,
) -> Result<u32, InstantiationError> {
    let value = evaluate_const_expr(ops, globals, store)?;
    match value {
        Value::I32(v) => Ok(v as u32),
        Value::I64(v) => Ok(v as u32),
        _ => Err(InstantiationError::ValidationFailed(
            "offset expression must evaluate to i32 or i64".to_string(),
        )),
    }
}

/// Initialize a data segment
fn initialize_data_segment(
    store: &mut Store,
    seg: &DataSegment,
    memories: &[MemoryIdx],
    globals: &[GlobalIdx],
) -> Result<(), InstantiationError> {
    let mem_idx = memories.get(seg.memory_index as usize).ok_or_else(|| {
        InstantiationError::ValidationFailed(format!(
            "invalid memory index {} in data segment",
            seg.memory_index
        ))
    })?;

    let offset = evaluate_offset_expr(&seg.offset_expr, globals, store)?;
    let memory = store
        .memory_mut(*mem_idx)
        .ok_or_else(|| InstantiationError::MemoryInitFailed("memory not found".to_string()))?;

    memory
        .write(offset as usize, seg.data)
        .map_err(|e| InstantiationError::MemoryInitFailed(e.message.clone()))?;

    Ok(())
}

/// Initialize an element segment
fn initialize_element_segment(
    store: &mut Store,
    seg: &ElementSegment,
    tables: &[TableIdx],
    globals: &[GlobalIdx],
    funcs: &[FuncIdx],
) -> Result<(), InstantiationError> {
    let table_idx = tables.get(seg.table_index as usize).ok_or_else(|| {
        InstantiationError::ValidationFailed(format!(
            "invalid table index {} in element segment",
            seg.table_index
        ))
    })?;

    let offset = evaluate_offset_expr(&seg.offset_expr, globals, store)?;
    let table = store
        .table_mut(*table_idx)
        .ok_or_else(|| InstantiationError::TableInitFailed("table not found".to_string()))?;

    for (i, &func_module_idx) in seg.func_indices.iter().enumerate() {
        let func_idx = funcs.get(func_module_idx as usize).ok_or_else(|| {
            InstantiationError::ValidationFailed(format!(
                "invalid function index {} in element segment",
                func_module_idx
            ))
        })?;

        table
            .set(offset + i as u32, Some(*func_idx))
            .map_err(|e| InstantiationError::TableInitFailed(e.message.clone()))?;
    }

    Ok(())
}

/// Compile all module functions
fn compile_module_functions(
    store: &mut Store,
    module: &Module,
    funcs: &[FuncIdx],
    memories: &[MemoryIdx],
    tables: &[TableIdx],
    globals: &[GlobalIdx],
) -> Result<(), InstantiationError> {
    let num_imported_funcs = module.num_imported_funcs() as usize;
    let func_bodies = module.function_bodies();

    // Build GlobalInfo for compiler
    let global_infos: Vec<GlobalInfo> = globals
        .iter()
        .enumerate()
        .filter_map(|(i, &idx)| {
            let global = store.global(idx)?;
            // Get the wasmparser ValType from the module
            let val_type = if i < module.num_imported_globals() as usize {
                // Imported global - find its type from imports
                module
                    .imports()
                    .iter()
                    .filter_map(|imp| {
                        if let ImportKind::Global(gt) = &imp.kind {
                            Some(gt.content_type)
                        } else {
                            None
                        }
                    })
                    .nth(i)?
            } else {
                // Defined global
                let def_idx = i - module.num_imported_globals() as usize;
                module.globals().get(def_idx)?.ty.content_type
            };

            Some(GlobalInfo {
                ptr: global.value_ptr() as usize,
                mutable: global.is_mutable(),
                val_type,
            })
        })
        .collect();

    // Build MemoryInfo for compiler
    let memory_info = memories.first().and_then(|&idx| {
        let memory = store.memory_mut(idx)?;
        Some(MemoryInfo {
            data_ptr: memory.data_ptr_mut() as usize,
            size_ptr: memory.size_ptr() as usize,
            max_pages: memory.max_pages(),
            grow_callback: None, // TODO: implement grow callback
        })
    });

    // Build FunctionEntry for compiler (for direct calls)
    // For wasm functions, use code_ptr_ptr for indirect calls (since code_ptr may be 0 initially)
    // For host functions, use trampoline_ptr directly
    let function_entries: Vec<FunctionEntry> = funcs
        .iter()
        .filter_map(|&idx| {
            let func = store.func(idx)?;
            let ft = func.ty();
            Some(FunctionEntry {
                code_ptr: func.callable_ptr(),
                code_ptr_ptr: func.code_ptr_ptr(), // Points to WasmFunc.code_ptr for wasm funcs
                params: ft
                    .params()
                    .iter()
                    .map(|t| to_wasmparser_valtype(*t))
                    .collect(),
                results: ft
                    .results()
                    .iter()
                    .map(|t| to_wasmparser_valtype(*t))
                    .collect(),
            })
        })
        .collect();

    // Build TableEntry for compiler
    let table_entries: Vec<TableEntry> = tables
        .iter()
        .filter_map(|&idx| {
            let table = store.table(idx)?;
            Some(TableEntry {
                data_ptr: table.data_ptr() as usize,
                size: table.size(),
            })
        })
        .collect();

    // Build func types for call_indirect
    let func_types: Vec<InternalFuncType> = module.func_types().to_vec();

    // Compile each function body
    for (i, body) in func_bodies.iter().enumerate() {
        let func_idx = funcs.get(num_imported_funcs + i).ok_or_else(|| {
            InstantiationError::ValidationFailed(format!("missing function index for body {}", i))
        })?;

        let type_idx = module.get_func_type_index(i as u32).ok_or_else(|| {
            InstantiationError::ValidationFailed(format!("missing type index for function {}", i))
        })?;

        let func_type = module.func_types().get(type_idx as usize).ok_or_else(|| {
            InstantiationError::ValidationFailed(format!(
                "invalid type index {} for function {}",
                type_idx, i
            ))
        })?;

        // Create compiler
        let mut compiler = Compiler::new();
        let mut wasm_compiler = Function::new();

        // Set up compiler context
        wasm_compiler.set_globals(global_infos.clone());
        if let Some(ref mem_info) = memory_info {
            wasm_compiler.set_memory(mem_info.clone());
        }
        wasm_compiler.set_functions(function_entries.clone());
        wasm_compiler.set_tables(table_entries.clone());
        wasm_compiler.set_func_types(func_types.clone());

        // Compile the function
        wasm_compiler
            .compile_function(
                &mut compiler,
                &func_type.params,
                &func_type.results,
                &body.locals,
                body.operators.iter().cloned(),
            )
            .map_err(|e| {
                InstantiationError::ValidationFailed(format!(
                    "compilation error for function {}: {}",
                    i, e
                ))
            })?;

        // Generate code
        let code = compiler.generate_code();
        let code_ptr = code.get() as usize;

        // Update the function's code pointer
        // We need to update the WasmFunc in the store
        if let Some(Func::Wasm(wf)) = store.func_mut(*func_idx) {
            wf.code_ptr = code_ptr;
        }

        // Keep the generated code alive by leaking it
        // In a real implementation, we'd store this in the Instance
        std::mem::forget(code);
    }

    // Update function entries with correct code pointers for call instructions
    // This is needed because we compiled functions before knowing all code pointers

    Ok(())
}

/// Convert our ValType to wasmparser ValType
fn to_wasmparser_valtype(ty: ValType) -> wasmparser::ValType {
    match ty {
        ValType::I32 => wasmparser::ValType::I32,
        ValType::I64 => wasmparser::ValType::I64,
        ValType::F32 => wasmparser::ValType::F32,
        ValType::F64 => wasmparser::ValType::F64,
        ValType::FuncRef => wasmparser::ValType::FUNCREF,
        ValType::ExternRef => wasmparser::ValType::EXTERNREF,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Engine;
    use std::sync::Arc;

    fn setup() -> (Arc<Engine>, Store) {
        let engine = Arc::new(Engine::default());
        let store = Store::new(engine.clone());
        (engine, store)
    }

    #[test]
    fn test_linker_new() {
        let linker = Linker::new();
        assert!(linker.resolve_func("test", "func").is_none());
    }

    #[test]
    fn test_linker_define_func() {
        let (engine, mut store) = setup();
        let mut linker = Linker::new();

        let func = Func::wrap(
            FuncType::new(vec![], vec![ValType::I32]),
            |_store, _args| Ok(vec![Value::I32(42)]),
        );
        let func_idx = store.alloc_func(func);

        linker.define_func("env", "get_value", func_idx);
        assert_eq!(linker.resolve_func("env", "get_value"), Some(func_idx));
        assert!(linker.resolve_func("env", "other").is_none());
    }

    #[test]
    fn test_linker_func_wrap() {
        let (engine, mut store) = setup();
        let mut linker = Linker::new();

        linker.func_wrap(
            &mut store,
            "env",
            "add",
            FuncType::new(vec![ValType::I32, ValType::I32], vec![ValType::I32]),
            |_store, args| {
                let a = args[0].unwrap_i32();
                let b = args[1].unwrap_i32();
                Ok(vec![Value::I32(a + b)])
            },
        );

        let func_idx = linker.resolve_func("env", "add").unwrap();
        let result = store
            .call(func_idx, &[Value::I32(10), Value::I32(20)])
            .unwrap();
        assert_eq!(result, vec![Value::I32(30)]);
    }

    #[test]
    fn test_instantiate_empty_module() {
        let (engine, mut store) = setup();
        let linker = Linker::new();

        let wasm = wat::parse_str("(module)").unwrap();
        let module = Module::new(&engine, &wasm).unwrap();

        let instance = linker.instantiate(&mut store, &module).unwrap();
        assert!(instance.funcs().is_empty());
        assert!(instance.memories().is_empty());
    }

    #[test]
    fn test_instantiate_simple_function() {
        let (engine, mut store) = setup();
        let linker = Linker::new();

        let wasm = wat::parse_str(
            r#"
            (module
                (func (export "answer") (result i32)
                    i32.const 42
                )
            )
            "#,
        )
        .unwrap();

        let module = Module::new(&engine, &wasm).unwrap();
        let instance = linker.instantiate(&mut store, &module).unwrap();

        let func_idx = instance.get_func(&store, "answer").unwrap();
        let result = store.call(func_idx, &[]).unwrap();
        assert_eq!(result, vec![Value::I32(42)]);
    }

    #[test]
    fn test_instantiate_with_memory() {
        let (engine, mut store) = setup();
        let linker = Linker::new();

        let wasm = wat::parse_str(
            r#"
            (module
                (memory (export "memory") 1 10)
            )
            "#,
        )
        .unwrap();

        let module = Module::new(&engine, &wasm).unwrap();
        let instance = linker.instantiate(&mut store, &module).unwrap();

        let mem_idx = instance.get_memory("memory").unwrap();
        let memory = store.memory(mem_idx).unwrap();
        assert_eq!(memory.size(), 1);
    }

    #[test]
    fn test_instantiate_with_import() {
        let (engine, mut store) = setup();
        let mut linker = Linker::new();

        // Define the imported function
        linker.func_wrap(
            &mut store,
            "env",
            "imported_add",
            FuncType::new(vec![ValType::I32, ValType::I32], vec![ValType::I32]),
            |_store, args| {
                let a = args[0].unwrap_i32();
                let b = args[1].unwrap_i32();
                Ok(vec![Value::I32(a + b)])
            },
        );

        let wasm = wat::parse_str(
            r#"
            (module
                (import "env" "imported_add" (func $add (param i32 i32) (result i32)))
                (func (export "test") (result i32)
                    i32.const 10
                    i32.const 20
                    call $add
                )
            )
            "#,
        )
        .unwrap();

        let module = Module::new(&engine, &wasm).unwrap();
        let instance = linker.instantiate(&mut store, &module).unwrap();

        let func_idx = instance.get_func(&store, "test").unwrap();
        let result = store.call(func_idx, &[]).unwrap();
        assert_eq!(result, vec![Value::I32(30)]);
    }

    #[test]
    fn test_missing_import() {
        let (engine, mut store) = setup();
        let linker = Linker::new();

        let wasm = wat::parse_str(
            r#"
            (module
                (import "env" "missing" (func (param i32)))
            )
            "#,
        )
        .unwrap();

        let module = Module::new(&engine, &wasm).unwrap();
        let result = linker.instantiate(&mut store, &module);

        assert!(matches!(
            result,
            Err(InstantiationError::ImportNotFound { .. })
        ));
    }

    #[test]
    fn test_wasm_calls_wasm_simple() {
        // Test: one wasm function calls another wasm function in the same module
        let (engine, mut store) = setup();
        let linker = Linker::new();

        let wasm = wat::parse_str(
            r#"
            (module
                ;; Helper function that doubles a number
                (func $double (param i32) (result i32)
                    local.get 0
                    i32.const 2
                    i32.mul
                )
                
                ;; Main function that calls $double
                (func (export "test") (param i32) (result i32)
                    local.get 0
                    call $double
                )
            )
            "#,
        )
        .unwrap();

        let module = Module::new(&engine, &wasm).unwrap();
        let instance = linker.instantiate(&mut store, &module).unwrap();

        let func_idx = instance.get_func(&store, "test").unwrap();

        // test(5) should call double(5) which returns 10
        let result = store.call(func_idx, &[Value::I32(5)]).unwrap();
        assert_eq!(result, vec![Value::I32(10)]);

        // test(21) should return 42
        let result = store.call(func_idx, &[Value::I32(21)]).unwrap();
        assert_eq!(result, vec![Value::I32(42)]);
    }

    #[test]
    fn test_wasm_calls_wasm_chain() {
        // Test: wasm function A calls B calls C (chain of calls)
        let (engine, mut store) = setup();
        let linker = Linker::new();

        let wasm = wat::parse_str(
            r#"
            (module
                ;; Add 1 to the input
                (func $add_one (param i32) (result i32)
                    local.get 0
                    i32.const 1
                    i32.add
                )
                
                ;; Call add_one twice (adds 2)
                (func $add_two (param i32) (result i32)
                    local.get 0
                    call $add_one
                    call $add_one
                )
                
                ;; Main function: calls add_two (which calls add_one twice)
                (func (export "test") (param i32) (result i32)
                    local.get 0
                    call $add_two
                )
            )
            "#,
        )
        .unwrap();

        let module = Module::new(&engine, &wasm).unwrap();
        let instance = linker.instantiate(&mut store, &module).unwrap();

        let func_idx = instance.get_func(&store, "test").unwrap();

        // test(10) should return 12 (10 + 1 + 1)
        let result = store.call(func_idx, &[Value::I32(10)]).unwrap();
        assert_eq!(result, vec![Value::I32(12)]);

        // test(40) should return 42
        let result = store.call(func_idx, &[Value::I32(40)]).unwrap();
        assert_eq!(result, vec![Value::I32(42)]);
    }

    #[test]
    fn test_wasm_calls_wasm_multiple_params() {
        // Test: wasm function calls another with multiple parameters
        let (engine, mut store) = setup();
        let linker = Linker::new();

        let wasm = wat::parse_str(
            r#"
            (module
                ;; Add two numbers
                (func $add (param i32 i32) (result i32)
                    local.get 0
                    local.get 1
                    i32.add
                )
                
                ;; Subtract: a - b
                (func $sub (param i32 i32) (result i32)
                    local.get 0
                    local.get 1
                    i32.sub
                )
                
                ;; Compute (a + b) - c by calling $add and $sub
                (func (export "compute") (param i32 i32 i32) (result i32)
                    local.get 0
                    local.get 1
                    call $add      ;; stack: (a + b)
                    local.get 2
                    call $sub      ;; stack: (a + b) - c
                )
            )
            "#,
        )
        .unwrap();

        let module = Module::new(&engine, &wasm).unwrap();
        let instance = linker.instantiate(&mut store, &module).unwrap();

        let func_idx = instance.get_func(&store, "compute").unwrap();

        // compute(10, 20, 5) = (10 + 20) - 5 = 25
        let result = store
            .call(func_idx, &[Value::I32(10), Value::I32(20), Value::I32(5)])
            .unwrap();
        assert_eq!(result, vec![Value::I32(25)]);

        // compute(50, 10, 18) = (50 + 10) - 18 = 42
        let result = store
            .call(func_idx, &[Value::I32(50), Value::I32(10), Value::I32(18)])
            .unwrap();
        assert_eq!(result, vec![Value::I32(42)]);
    }

    #[test]
    fn test_wasm_recursive_call() {
        // Test: recursive wasm function (factorial)
        let (engine, mut store) = setup();
        let linker = Linker::new();

        let wasm = wat::parse_str(
            r#"
            (module
                ;; Factorial function: n! = n * (n-1)!
                (func $factorial (export "factorial") (param i32) (result i32)
                    (if (result i32) (i32.le_s (local.get 0) (i32.const 1))
                        (then
                            (i32.const 1)
                        )
                        (else
                            (i32.mul
                                (local.get 0)
                                (call $factorial
                                    (i32.sub (local.get 0) (i32.const 1))
                                )
                            )
                        )
                    )
                )
            )
            "#,
        )
        .unwrap();

        let module = Module::new(&engine, &wasm).unwrap();
        let instance = linker.instantiate(&mut store, &module).unwrap();

        let func_idx = instance.get_func(&store, "factorial").unwrap();

        // factorial(1) = 1
        let result = store.call(func_idx, &[Value::I32(1)]).unwrap();
        assert_eq!(result, vec![Value::I32(1)]);

        // factorial(5) = 120
        let result = store.call(func_idx, &[Value::I32(5)]).unwrap();
        assert_eq!(result, vec![Value::I32(120)]);

        // factorial(6) = 720
        let result = store.call(func_idx, &[Value::I32(6)]).unwrap();
        assert_eq!(result, vec![Value::I32(720)]);
    }

    #[test]
    fn test_wasm_mutual_recursion() {
        // Test: two wasm functions calling each other (mutual recursion)
        // is_even(n) calls is_odd(n-1), is_odd(n) calls is_even(n-1)
        let (engine, mut store) = setup();
        let linker = Linker::new();

        let wasm = wat::parse_str(
            r#"
            (module
                ;; is_even: returns 1 if n is even, 0 otherwise
                (func $is_even (export "is_even") (param i32) (result i32)
                    (if (result i32) (i32.eqz (local.get 0))
                        (then (i32.const 1))
                        (else (call $is_odd (i32.sub (local.get 0) (i32.const 1))))
                    )
                )
                
                ;; is_odd: returns 1 if n is odd, 0 otherwise
                (func $is_odd (export "is_odd") (param i32) (result i32)
                    (if (result i32) (i32.eqz (local.get 0))
                        (then (i32.const 0))
                        (else (call $is_even (i32.sub (local.get 0) (i32.const 1))))
                    )
                )
            )
            "#,
        )
        .unwrap();

        let module = Module::new(&engine, &wasm).unwrap();
        let instance = linker.instantiate(&mut store, &module).unwrap();

        let is_even_idx = instance.get_func(&store, "is_even").unwrap();
        let is_odd_idx = instance.get_func(&store, "is_odd").unwrap();

        // is_even(0) = 1
        let result = store.call(is_even_idx, &[Value::I32(0)]).unwrap();
        assert_eq!(result, vec![Value::I32(1)]);

        // is_even(4) = 1
        let result = store.call(is_even_idx, &[Value::I32(4)]).unwrap();
        assert_eq!(result, vec![Value::I32(1)]);

        // is_even(5) = 0
        let result = store.call(is_even_idx, &[Value::I32(5)]).unwrap();
        assert_eq!(result, vec![Value::I32(0)]);

        // is_odd(3) = 1
        let result = store.call(is_odd_idx, &[Value::I32(3)]).unwrap();
        assert_eq!(result, vec![Value::I32(1)]);

        // is_odd(4) = 0
        let result = store.call(is_odd_idx, &[Value::I32(4)]).unwrap();
        assert_eq!(result, vec![Value::I32(0)]);
    }
}
