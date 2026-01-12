use std::error::Error;
use wasmparser::{Operator, ValType};

use crate::Engine;

/// Import entry - represents a WebAssembly import
#[derive(Clone, Debug)]
pub struct ImportEntry<'a> {
    /// Module name
    pub module: &'a str,
    /// Import name
    pub name: &'a str,
    /// Import kind
    pub kind: ImportKind,
}

/// Kind of import
#[derive(Clone, Debug)]
pub enum ImportKind {
    /// Function import with type index
    Func(u32),
    /// Table import
    Table(TableType),
    /// Memory import
    Memory(MemoryType),
    /// Global import
    Global(GlobalType),
}

/// Export entry - represents a WebAssembly export
#[derive(Clone, Debug)]
pub struct ExportEntry<'a> {
    /// Export name
    pub name: &'a str,
    /// Export kind and index
    pub kind: ExportKind,
}

/// Kind of export
#[derive(Clone, Copy, Debug)]
pub enum ExportKind {
    /// Function export with function index
    Func(u32),
    /// Table export with table index
    Table(u32),
    /// Memory export with memory index
    Memory(u32),
    /// Global export with global index
    Global(u32),
}

/// Table type definition
#[derive(Clone, Copy, Debug)]
pub struct TableType {
    /// Element type (funcref or externref)
    pub element_type: RefType,
    /// Initial size
    pub initial: u64,
    /// Maximum size (optional)
    pub maximum: Option<u64>,
}

/// Memory type definition
#[derive(Clone, Copy, Debug)]
pub struct MemoryType {
    /// Initial size in pages (64KB each)
    pub initial: u64,
    /// Maximum size in pages (optional)
    pub maximum: Option<u64>,
    /// Whether this is a 64-bit memory
    pub memory64: bool,
    /// Whether this memory is shared
    pub shared: bool,
}

/// Global type definition
#[derive(Clone, Copy, Debug)]
pub struct GlobalType {
    /// Value type
    pub content_type: ValType,
    /// Whether the global is mutable
    pub mutable: bool,
}

/// Reference type for tables
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefType {
    FuncRef,
    ExternRef,
}

/// Internal function type for compiler (uses wasmparser::ValType)
#[derive(Clone, Debug)]
pub struct InternalFuncType {
    pub params: Vec<ValType>,
    pub results: Vec<ValType>,
}

/// Global definition with initializer
#[derive(Clone, Debug)]
pub struct GlobalDef {
    /// Global type
    pub ty: GlobalType,
    /// Initial value expression (simplified - just store the operators)
    pub init_expr: Vec<Operator<'static>>,
}

/// Data segment for memory initialization
#[derive(Clone, Debug)]
pub struct DataSegment<'a> {
    /// Memory index (0 for MVP)
    pub memory_index: u32,
    /// Offset expression (simplified)
    pub offset_expr: Vec<Operator<'static>>,
    /// Data bytes
    pub data: &'a [u8],
    /// Whether this is a passive segment
    pub passive: bool,
}

/// Element segment for table initialization
#[derive(Clone, Debug)]
pub struct ElementSegment {
    /// Table index
    pub table_index: u32,
    /// Offset expression (simplified)
    pub offset_expr: Vec<Operator<'static>>,
    /// Function indices
    pub func_indices: Vec<u32>,
    /// Whether this is a passive segment
    pub passive: bool,
}

/// Function body (code) for compilation
#[derive(Clone, Debug)]
pub struct FunctionBody<'a> {
    /// Local variable types (excluding parameters)
    pub locals: Vec<ValType>,
    /// Function body operators
    pub operators: Vec<Operator<'a>>,
}

/// Compiled WebAssembly module
#[derive(Clone, Debug)]
pub struct Module<'a> {
    engine: &'a Engine,

    /// Function types (signatures)
    func_types: Vec<InternalFuncType>,

    /// Import entries
    imports: Vec<ImportEntry<'a>>,

    /// Number of imported functions
    num_imported_funcs: u32,
    /// Number of imported tables
    num_imported_tables: u32,
    /// Number of imported memories
    num_imported_memories: u32,
    /// Number of imported globals
    num_imported_globals: u32,

    /// Function section: maps function index to type index
    func_type_indices: Vec<u32>,

    /// Table definitions (not including imports)
    tables: Vec<TableType>,

    /// Memory definitions (not including imports)
    memories: Vec<MemoryType>,

    /// Global definitions (not including imports)
    globals: Vec<GlobalDef>,

    /// Export entries
    exports: Vec<ExportEntry<'a>>,

    /// Start function index (optional)
    start_func: Option<u32>,

    /// Element segments for table initialization
    element_segments: Vec<ElementSegment>,

    /// Data segments for memory initialization
    data_segments: Vec<DataSegment<'a>>,

    /// Function bodies (code)
    function_bodies: Vec<FunctionBody<'a>>,
}

impl<'a> Module<'a> {
    pub fn new(engine: &'a Engine, wasm: &'a [u8]) -> Result<Self, Box<dyn Error + 'a>> {
        let mut func_types = Vec::new();
        let mut imports = Vec::new();
        let mut func_type_indices = Vec::new();
        let mut tables = Vec::new();
        let mut memories = Vec::new();
        let mut globals = Vec::new();
        let mut exports = Vec::new();
        let mut start_func = None;
        let mut element_segments = Vec::new();
        let mut data_segments = Vec::new();
        let mut function_bodies = Vec::new();

        let mut num_imported_funcs = 0u32;
        let mut num_imported_tables = 0u32;
        let mut num_imported_memories = 0u32;
        let mut num_imported_globals = 0u32;

        let parser = wasmparser::Parser::new(0);
        for payload in parser.parse_all(wasm) {
            match payload? {
                wasmparser::Payload::Version { .. } => {
                    // Version payload - just continue, nothing to store
                }

                wasmparser::Payload::TypeSection(reader) => {
                    for rec_group in reader {
                        let rec_group = rec_group?;
                        // Handle recursive type groups (for GC proposal)
                        // For now, we only support simple function types
                        for sub_type in rec_group.into_types() {
                            if let wasmparser::CompositeInnerType::Func(func_type) =
                                sub_type.composite_type.inner
                            {
                                let params: Vec<ValType> = func_type.params().to_vec();
                                let results: Vec<ValType> = func_type.results().to_vec();
                                func_types.push(InternalFuncType { params, results });
                            }
                        }
                    }
                }

                wasmparser::Payload::ImportSection(reader) => {
                    for import in reader {
                        let import = import?;
                        match import {
                            wasmparser::Imports::Single(_module_idx, imp) => {
                                let kind = match imp.ty {
                                    wasmparser::TypeRef::Func(idx) => {
                                        num_imported_funcs += 1;
                                        ImportKind::Func(idx)
                                    }
                                    wasmparser::TypeRef::FuncExact(idx) => {
                                        num_imported_funcs += 1;
                                        ImportKind::Func(idx)
                                    }
                                    wasmparser::TypeRef::Table(t) => {
                                        num_imported_tables += 1;
                                        ImportKind::Table(convert_table_type(&t))
                                    }
                                    wasmparser::TypeRef::Memory(m) => {
                                        num_imported_memories += 1;
                                        ImportKind::Memory(convert_memory_type(&m))
                                    }
                                    wasmparser::TypeRef::Global(g) => {
                                        num_imported_globals += 1;
                                        ImportKind::Global(convert_global_type(&g))
                                    }
                                    wasmparser::TypeRef::Tag(_) => {
                                        // Exception handling - not supported
                                        continue;
                                    }
                                };
                                imports.push(ImportEntry {
                                    module: imp.module,
                                    name: imp.name,
                                    kind,
                                });
                            }
                            wasmparser::Imports::Compact1 { module, items } => {
                                // Compact format with single module name, multiple items
                                for item in items {
                                    let item = item?;
                                    let kind = match item.ty {
                                        wasmparser::TypeRef::Func(idx) => {
                                            num_imported_funcs += 1;
                                            ImportKind::Func(idx)
                                        }
                                        wasmparser::TypeRef::FuncExact(idx) => {
                                            num_imported_funcs += 1;
                                            ImportKind::Func(idx)
                                        }
                                        wasmparser::TypeRef::Table(t) => {
                                            num_imported_tables += 1;
                                            ImportKind::Table(convert_table_type(&t))
                                        }
                                        wasmparser::TypeRef::Memory(m) => {
                                            num_imported_memories += 1;
                                            ImportKind::Memory(convert_memory_type(&m))
                                        }
                                        wasmparser::TypeRef::Global(g) => {
                                            num_imported_globals += 1;
                                            ImportKind::Global(convert_global_type(&g))
                                        }
                                        wasmparser::TypeRef::Tag(_) => continue,
                                    };
                                    imports.push(ImportEntry {
                                        module,
                                        name: item.name,
                                        kind,
                                    });
                                }
                            }
                            wasmparser::Imports::Compact2 { module, ty, names } => {
                                // Compact format with single module and type, multiple names
                                for name in names {
                                    let name = name?;
                                    let kind = match ty {
                                        wasmparser::TypeRef::Func(idx) => {
                                            num_imported_funcs += 1;
                                            ImportKind::Func(idx)
                                        }
                                        wasmparser::TypeRef::FuncExact(idx) => {
                                            num_imported_funcs += 1;
                                            ImportKind::Func(idx)
                                        }
                                        wasmparser::TypeRef::Table(ref t) => {
                                            num_imported_tables += 1;
                                            ImportKind::Table(convert_table_type(t))
                                        }
                                        wasmparser::TypeRef::Memory(ref m) => {
                                            num_imported_memories += 1;
                                            ImportKind::Memory(convert_memory_type(m))
                                        }
                                        wasmparser::TypeRef::Global(ref g) => {
                                            num_imported_globals += 1;
                                            ImportKind::Global(convert_global_type(g))
                                        }
                                        wasmparser::TypeRef::Tag(_) => continue,
                                    };
                                    imports.push(ImportEntry { module, name, kind });
                                }
                            }
                        }
                    }
                }

                wasmparser::Payload::FunctionSection(reader) => {
                    for func in reader {
                        let type_idx = func?;
                        func_type_indices.push(type_idx);
                    }
                }

                wasmparser::Payload::TableSection(reader) => {
                    for table in reader {
                        let table = table?;
                        tables.push(convert_table_type(&table.ty));
                    }
                }

                wasmparser::Payload::MemorySection(reader) => {
                    for memory in reader {
                        let memory = memory?;
                        memories.push(convert_memory_type(&memory));
                    }
                }

                wasmparser::Payload::GlobalSection(reader) => {
                    for global in reader {
                        let global = global?;
                        let ty = convert_global_type(&global.ty);

                        // Parse init expression
                        let init_expr = parse_const_expr(&global.init_expr)?;

                        globals.push(GlobalDef { ty, init_expr });
                    }
                }

                wasmparser::Payload::ExportSection(reader) => {
                    for export in reader {
                        let export = export?;
                        let kind = match export.kind {
                            wasmparser::ExternalKind::Func => ExportKind::Func(export.index),
                            wasmparser::ExternalKind::FuncExact => ExportKind::Func(export.index),
                            wasmparser::ExternalKind::Table => ExportKind::Table(export.index),
                            wasmparser::ExternalKind::Memory => ExportKind::Memory(export.index),
                            wasmparser::ExternalKind::Global => ExportKind::Global(export.index),
                            wasmparser::ExternalKind::Tag => {
                                // Exception handling - not supported
                                continue;
                            }
                        };
                        exports.push(ExportEntry {
                            name: export.name,
                            kind,
                        });
                    }
                }

                wasmparser::Payload::StartSection { func, .. } => {
                    start_func = Some(func);
                }

                wasmparser::Payload::ElementSection(reader) => {
                    for element in reader {
                        let element = element?;
                        // Parse items first before matching on kind (to avoid partial move)
                        let func_indices = parse_element_items(&element)?;
                        match element.kind {
                            wasmparser::ElementKind::Active {
                                table_index,
                                offset_expr,
                            } => {
                                let offset_expr = parse_const_expr(&offset_expr)?;
                                element_segments.push(ElementSegment {
                                    table_index: table_index.unwrap_or(0),
                                    offset_expr,
                                    func_indices,
                                    passive: false,
                                });
                            }
                            wasmparser::ElementKind::Passive => {
                                element_segments.push(ElementSegment {
                                    table_index: 0,
                                    offset_expr: vec![],
                                    func_indices,
                                    passive: true,
                                });
                            }
                            wasmparser::ElementKind::Declared => {
                                // Declared segments are for reference types - skip for now
                            }
                        }
                    }
                }

                wasmparser::Payload::DataSection(reader) => {
                    for data in reader {
                        let data = data?;
                        match data.kind {
                            wasmparser::DataKind::Active {
                                memory_index,
                                offset_expr,
                            } => {
                                let offset_expr = parse_const_expr(&offset_expr)?;
                                data_segments.push(DataSegment {
                                    memory_index,
                                    offset_expr,
                                    data: data.data,
                                    passive: false,
                                });
                            }
                            wasmparser::DataKind::Passive => {
                                data_segments.push(DataSegment {
                                    memory_index: 0,
                                    offset_expr: vec![],
                                    data: data.data,
                                    passive: true,
                                });
                            }
                        }
                    }
                }

                wasmparser::Payload::DataCountSection { .. } => {
                    // Just indicates the number of data segments - already handled
                }

                wasmparser::Payload::CodeSectionStart { .. } => {
                    // Indicates start of code section - continue to entries
                }

                wasmparser::Payload::CodeSectionEntry(body) => {
                    // Parse locals
                    let mut locals = Vec::new();
                    let locals_reader = body.get_locals_reader()?;
                    for local in locals_reader {
                        let (count, val_type) = local?;
                        for _ in 0..count {
                            locals.push(val_type);
                        }
                    }

                    // Parse operators
                    let operators: Vec<Operator> = body
                        .get_operators_reader()?
                        .into_iter()
                        .collect::<Result<_, _>>()?;

                    function_bodies.push(FunctionBody { locals, operators });
                }

                wasmparser::Payload::TagSection(reader) => {
                    // Exception handling tags - not supported, skip
                    for _ in reader {
                        // Just consume the iterator
                    }
                }

                wasmparser::Payload::CustomSection(_) => {
                    // Custom sections are ignored (debug info, names, etc.)
                }

                wasmparser::Payload::UnknownSection { .. } => {
                    // Unknown sections are ignored
                }

                wasmparser::Payload::End(_) => {
                    // End of module - we're done parsing
                }

                // Handle any other payloads
                _ => {
                    // Unsupported sections - skip
                }
            }
        }

        Ok(Self {
            engine,
            func_types,
            imports,
            num_imported_funcs,
            num_imported_tables,
            num_imported_memories,
            num_imported_globals,
            func_type_indices,
            tables,
            memories,
            globals,
            exports,
            start_func,
            element_segments,
            data_segments,
            function_bodies,
        })
    }

    /// Get the engine
    pub fn engine(&self) -> &Engine {
        self.engine
    }

    /// Get all function types
    pub fn func_types(&self) -> &[InternalFuncType] {
        &self.func_types
    }

    /// Get all imports
    pub fn imports(&self) -> &[ImportEntry<'a>] {
        &self.imports
    }

    /// Get all exports
    pub fn exports(&self) -> &[ExportEntry<'a>] {
        &self.exports
    }

    /// Get the start function index if any
    pub fn start_func(&self) -> Option<u32> {
        self.start_func
    }

    /// Get the number of imported functions
    pub fn num_imported_funcs(&self) -> u32 {
        self.num_imported_funcs
    }

    /// Get the number of imported tables
    pub fn num_imported_tables(&self) -> u32 {
        self.num_imported_tables
    }

    /// Get the number of imported memories
    pub fn num_imported_memories(&self) -> u32 {
        self.num_imported_memories
    }

    /// Get the number of imported globals
    pub fn num_imported_globals(&self) -> u32 {
        self.num_imported_globals
    }

    /// Get all table definitions (not including imports)
    pub fn tables(&self) -> &[TableType] {
        &self.tables
    }

    /// Get all memory definitions (not including imports)
    pub fn memories(&self) -> &[MemoryType] {
        &self.memories
    }

    /// Get all global definitions (not including imports)
    pub fn globals(&self) -> &[GlobalDef] {
        &self.globals
    }

    /// Get all element segments
    pub fn element_segments(&self) -> &[ElementSegment] {
        &self.element_segments
    }

    /// Get all data segments
    pub fn data_segments(&self) -> &[DataSegment<'a>] {
        &self.data_segments
    }

    /// Get all function bodies
    pub fn function_bodies(&self) -> &[FunctionBody<'a>] {
        &self.function_bodies
    }

    /// Get the function type index for a given function index (excluding imports)
    pub fn get_func_type_index(&self, func_idx: u32) -> Option<u32> {
        self.func_type_indices.get(func_idx as usize).copied()
    }

    /// Get the function type for a given function index (including imports)
    pub fn get_func_type(&self, func_idx: u32) -> Option<&InternalFuncType> {
        if func_idx < self.num_imported_funcs {
            // It's an imported function - find its type
            let mut import_func_idx = 0u32;
            for import in &self.imports {
                if let ImportKind::Func(type_idx) = import.kind {
                    if import_func_idx == func_idx {
                        return self.func_types.get(type_idx as usize);
                    }
                    import_func_idx += 1;
                }
            }
            None
        } else {
            // It's a defined function
            let local_idx = func_idx - self.num_imported_funcs;
            let type_idx = self.func_type_indices.get(local_idx as usize)?;
            self.func_types.get(*type_idx as usize)
        }
    }

    /// Get an export by name
    pub fn get_export(&self, name: &str) -> Option<&ExportEntry<'a>> {
        self.exports.iter().find(|e| e.name == name)
    }

    /// Get a function export by name
    pub fn get_func_export(&self, name: &str) -> Option<u32> {
        self.exports.iter().find_map(|e| {
            if e.name == name
                && let ExportKind::Func(idx) = e.kind
            {
                return Some(idx);
            }
            None
        })
    }

    /// Get a memory export by name
    pub fn get_memory_export(&self, name: &str) -> Option<u32> {
        self.exports.iter().find_map(|e| {
            if e.name == name
                && let ExportKind::Memory(idx) = e.kind
            {
                return Some(idx);
            }
            None
        })
    }
}

/// Convert wasmparser TableType to our TableType
fn convert_table_type(t: &wasmparser::TableType) -> TableType {
    let element_type = if t.element_type.is_func_ref() {
        RefType::FuncRef
    } else {
        RefType::ExternRef
    };
    TableType {
        element_type,
        initial: t.initial,
        maximum: t.maximum,
    }
}

/// Convert wasmparser MemoryType to our MemoryType
fn convert_memory_type(m: &wasmparser::MemoryType) -> MemoryType {
    MemoryType {
        initial: m.initial,
        maximum: m.maximum,
        memory64: m.memory64,
        shared: m.shared,
    }
}

/// Convert wasmparser GlobalType to our GlobalType
fn convert_global_type(g: &wasmparser::GlobalType) -> GlobalType {
    GlobalType {
        content_type: g.content_type,
        mutable: g.mutable,
    }
}

/// Parse a constant expression into a list of operators
/// This uses a transmute to convert the lifetime - safe because we're storing
/// the operators immediately and the wasm bytes outlive the Module
fn parse_const_expr(
    expr: &wasmparser::ConstExpr,
) -> Result<Vec<Operator<'static>>, wasmparser::BinaryReaderError> {
    let mut ops = Vec::new();
    let mut reader = expr.get_operators_reader();
    while !reader.eof() {
        let op = reader.read()?;
        // Safety: We're converting the lifetime to 'static because:
        // 1. The operators we care about (i32.const, i64.const, etc.) don't borrow data
        // 2. For operators that might borrow, we only store simple init expressions
        let op_static: Operator<'static> = unsafe { std::mem::transmute(op) };
        ops.push(op_static);
    }
    Ok(ops)
}

/// Parse element segment items to get function indices
fn parse_element_items(element: &wasmparser::Element) -> Result<Vec<u32>, Box<dyn Error>> {
    let mut indices = Vec::new();

    match &element.items {
        wasmparser::ElementItems::Functions(reader) => {
            for func_idx in reader.clone() {
                indices.push(func_idx?);
            }
        }
        wasmparser::ElementItems::Expressions(_, reader) => {
            // For expressions, try to extract function indices from ref.func
            for expr in reader.clone() {
                let expr = expr?;
                let mut op_reader = expr.get_operators_reader();
                while !op_reader.eof() {
                    let op = op_reader.read()?;
                    if let Operator::RefFunc { function_index } = op {
                        indices.push(function_index);
                        break;
                    }
                }
            }
        }
    }

    Ok(indices)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_empty_module() {
        let wasm = wat::parse_str("(module)").unwrap();
        let engine = Engine::default();
        let module = Module::new(&engine, &wasm).unwrap();

        assert!(module.func_types().is_empty());
        assert!(module.imports().is_empty());
        assert!(module.exports().is_empty());
        assert!(module.function_bodies().is_empty());
    }

    #[test]
    fn test_parse_simple_function() {
        let wasm = wat::parse_str(
            r#"
            (module
                (func (export "add") (param i32 i32) (result i32)
                    local.get 0
                    local.get 1
                    i32.add
                )
            )
            "#,
        )
        .unwrap();

        let engine = Engine::default();
        let module = Module::new(&engine, &wasm).unwrap();

        assert_eq!(module.func_types().len(), 1);
        assert_eq!(module.function_bodies().len(), 1);
        assert_eq!(module.exports().len(), 1);

        let func_type = &module.func_types()[0];
        assert_eq!(func_type.params.len(), 2);
        assert_eq!(func_type.results.len(), 1);

        let export = module.get_func_export("add");
        assert!(export.is_some());
    }

    #[test]
    fn test_parse_with_memory() {
        let wasm = wat::parse_str(
            r#"
            (module
                (memory (export "memory") 1 10)
            )
            "#,
        )
        .unwrap();

        let engine = Engine::default();
        let module = Module::new(&engine, &wasm).unwrap();

        assert_eq!(module.memories().len(), 1);
        assert_eq!(module.memories()[0].initial, 1);
        assert_eq!(module.memories()[0].maximum, Some(10));
    }

    #[test]
    fn test_parse_with_globals() {
        let wasm = wat::parse_str(
            r#"
            (module
                (global $g (mut i32) (i32.const 42))
            )
            "#,
        )
        .unwrap();

        let engine = Engine::default();
        let module = Module::new(&engine, &wasm).unwrap();

        assert_eq!(module.globals().len(), 1);
        assert!(module.globals()[0].ty.mutable);
    }

    #[test]
    fn test_parse_with_imports() {
        let wasm = wat::parse_str(
            r#"
            (module
                (import "env" "print" (func (param i32)))
                (import "env" "memory" (memory 1))
            )
            "#,
        )
        .unwrap();

        let engine = Engine::default();
        let module = Module::new(&engine, &wasm).unwrap();

        assert_eq!(module.imports().len(), 2);
        assert_eq!(module.num_imported_funcs(), 1);
        assert_eq!(module.num_imported_memories(), 1);
    }
}
