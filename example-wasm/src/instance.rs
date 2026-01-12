//! Instance and Linker for WebAssembly module instantiation
//!
//! This module contains:
//! - `Instance` - A runtime instantiation of a Module with resolved imports
//! - `Linker` - Helper for building import resolvers

use std::collections::HashMap;

use wasmparser::Operator;

use crate::error::InstantiationError;
use crate::function::Function;
use crate::module::{
    DataSegment, ElementSegment, ExportKind, ImportKind, InternalFuncType, Module,
};
use crate::store::{Func, Store, WasmFunc};
use crate::types::{FuncIdx, FuncType, GlobalIdx, MemoryIdx, RefType, TableIdx, ValType, Value};
use crate::{FunctionEntry, GlobalInfo, Linker, MemoryInfo, TableEntry};

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
                params: ft.params().iter().map(|&t| t.into()).collect(),
                results: ft.results().iter().map(|&t| t.into()).collect(),
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
        let code = wasm_compiler.generate_code().ok_or_else(|| {
            InstantiationError::ValidationFailed(format!(
                "compiler already consumed for function {}",
                i
            ))
        })?;
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

#[cfg(test)]
mod tests;
