//! Simple WebAssembly JIT Compiler using sljit-rs
//!
//! This is a minimal WebAssembly to native code compiler that demonstrates
//! how to use sljit-rs's high-level Emitter API to JIT-compile WebAssembly.
//!
//! Inspired by pwart's architecture, but simplified for clarity.

use std::error::Error;

use bitvec::prelude::*;
use sljit::sys::{
    self,
};
use sljit::{
    FloatRegister, Operand, SavedRegister, ScratchRegister, mem_sp_offset,
};
use wasmparser::{Import, Operator, ValType};

pub(crate) mod helpers;

/// Errors that can occur during compilation
#[derive(Debug)]
pub enum CompileError {
    Parse(String),
    Sljit(sys::ErrorCode),
    Unsupported(String),
    Invalid(String),
    RegisterAllocationFailed,
}

impl From<sys::ErrorCode> for CompileError {
    #[inline(always)]
    fn from(e: sys::ErrorCode) -> Self {
        CompileError::Sljit(e)
    }
}

impl From<wasmparser::BinaryReaderError> for CompileError {
    #[inline(always)]
    fn from(e: wasmparser::BinaryReaderError) -> Self {
        CompileError::Parse(e.to_string())
    }
}

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
enum DivOp {
    DivS,
    DivU,
    RemS,
    RemU,
}

/// Represents a value on the operand stack during compilation
#[derive(Clone, Debug)]
pub enum StackValue {
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
    fn to_operand(&self) -> Operand {
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
pub struct Block {
    label: Option<sys::Label>,
    end_jumps: Vec<sys::Jump>,
    else_jump: Option<sys::Jump>,
    stack_depth: usize,
    result_offset: Option<i32>,
}

/// Local variable storage info
#[derive(Clone, Debug)]
pub enum LocalVar {
    Saved(SavedRegister),
    Stack(i32),
}

/// Global variable info
#[derive(Clone, Debug)]
pub struct GlobalInfo {
    /// Pointer to the global's memory location
    pub ptr: usize,
    /// Whether the global is mutable
    pub mutable: bool,
    /// Value type of the global
    pub val_type: ValType,
}

/// Memory instance info
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

/// Function signature for calls
#[derive(Clone, Debug)]
pub struct FuncType {
    pub params: Vec<ValType>,
    pub results: Vec<ValType>,
}

/// Compiled function entry for direct calls
#[derive(Clone, Debug)]
pub struct FunctionEntry {
    /// Pointer to the compiled function code
    pub code_ptr: usize,
    /// Function type
    pub func_type: FuncType,
}

/// Table entry for indirect calls
#[derive(Clone, Debug)]
pub struct TableEntry {
    /// Pointer to the table data (array of function pointers)
    pub data_ptr: usize,
    /// Current table size
    pub size: u32,
}

#[derive(Clone, Debug, Default)]
pub struct Engine {}

#[derive(Clone, Debug)]
pub struct Module<'a> {
    engine: &'a Engine,

    /// Import entries
    imports: Vec<Import<'a>>,

    /// Global variables
    globals: Vec<GlobalInfo>,
    /// Memory instance
    memory: Option<MemoryInfo>,
    /// Function table for direct calls
    functions: Vec<FunctionEntry>,
    /// Tables for indirect calls
    tables: Vec<TableEntry>,
    /// Function types for call_indirect
    func_types: Vec<FuncType>,
}

impl<'a> Module<'a> {
    pub fn new(engine: &'a Engine, wasm: &'a [u8]) -> Result<Self, Box<dyn Error + 'a>> {
        let mut imports = vec![];

        let parser = wasmparser::Parser::new(0);
        for payload in parser.parse_all(wasm) {
            match payload? {
                wasmparser::Payload::TypeSection(types) => {
                    for ty in types {
                        let _ty = ty?;
                    }
                }
                wasmparser::Payload::ImportSection(import_section) => {
                    for import in import_section {
                        match import? {
                            wasmparser::Imports::Single(_, import) => {
                                imports.push(import);
                            }
                            wasmparser::Imports::Compact1 {
                                module: _,
                                items: _,
                            } => todo!(),
                            wasmparser::Imports::Compact2 {
                                module: _,
                                ty: _,
                                names: _,
                            } => todo!(),
                        }
                    }
                }
                wasmparser::Payload::FunctionSection(functions) => {
                    for func in functions {
                        let _func = func?;
                    }
                }
                wasmparser::Payload::CodeSectionStart { count: _, .. } => {
                    todo!()
                }
                wasmparser::Payload::CodeSectionEntry(body) => {
                    let _body: Vec<Operator> = body
                        .get_operators_reader()?
                        .into_iter()
                        .collect::<Result<_, _>>()?;
                }
                wasmparser::Payload::Version {
                    num: _,
                    encoding: _,
                    range: _,
                } => todo!(),
                wasmparser::Payload::TableSection(section_limited) => {
                    for table in section_limited {
                        let _table = table?;
                    }
                }
                wasmparser::Payload::MemorySection(section_limited) => {
                    for memory in section_limited {
                        let _memory = memory?;
                    }
                }
                wasmparser::Payload::TagSection(section_limited) => {
                    for tag in section_limited {
                        let _tag = tag?;
                    }
                }
                wasmparser::Payload::GlobalSection(section_limited) => {
                    for global in section_limited {
                        let _global = global?;
                    }
                }
                wasmparser::Payload::ExportSection(section_limited) => {
                    for export in section_limited {
                        let _export = export?;
                    }
                }
                wasmparser::Payload::StartSection { func: _, range: _ } => todo!(),
                wasmparser::Payload::ElementSection(section_limited) => {
                    for element in section_limited {
                        let _element = element?;
                    }
                }
                wasmparser::Payload::DataCountSection { count: _, range: _ } => todo!(),
                wasmparser::Payload::DataSection(section_limited) => {
                    for data in section_limited {
                        let _data = data?;
                    }
                }
                wasmparser::Payload::CustomSection(_custom_section_reader) => todo!(),
                wasmparser::Payload::UnknownSection {
                    id: _,
                    contents: _,
                    range: _,
                } => todo!(),
                wasmparser::Payload::End(_) => todo!(),
                _ => todo!(),
            }
        }

        Ok(Self {
            imports,
            engine,
            globals: todo!(),
            memory: todo!(),
            functions: todo!(),
            tables: todo!(),
            func_types: todo!(),
        })
    }
}

#[cfg(test)]
mod tests;

pub mod function;
