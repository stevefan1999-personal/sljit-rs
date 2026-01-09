//! Simple WebAssembly JIT Compiler using sljit-rs
//!
//! This is a minimal WebAssembly to native code compiler that demonstrates
//! how to use sljit-rs's high-level Emitter API to JIT-compile WebAssembly.
//!
//! Inspired by pwart's architecture, but simplified for clarity.

use std::error::Error;

use indexmap::IndexSet;
use sljit::sys::{
    self, Compiler, GeneratedCode, SLJIT_ARG_TYPE_F32, SLJIT_ARG_TYPE_F64, SLJIT_ARG_TYPE_W,
    arg_types,
};
use sljit::{
    Condition, Emitter, FloatRegister, JumpType, Operand, ReturnOp, SavedRegister, ScratchRegister,
    mem_offset, mem_sp_offset, regs,
};
use wasmparser::{Import, Operator, ValType};

/// Errors that can occur during compilation
#[derive(Debug)]
pub enum CompileError {
    Parse(String),
    Sljit(sys::ErrorCode),
    Unsupported(String),
    Invalid(String),
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

/// Binary operation type for integers
#[derive(Clone, Copy, Debug)]
enum BinaryOp {
    Add,
    Sub,
    Mul,
    And,
    Or,
    Xor,
    Shl,
    ShrS,
    ShrU,
    Rotl,
    Rotr,
}

/// Binary operation type for floats
#[derive(Clone, Copy, Debug)]
enum FloatBinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Min,
    Max,
    Copysign,
}

/// Unary operation type for floats
#[derive(Clone, Copy, Debug)]
enum FloatUnaryOp {
    Neg,
    Abs,
    Sqrt,
    Ceil,
    Floor,
    Trunc,
    Nearest,
}

/// Unary operation type
#[derive(Clone, Copy, Debug)]
enum UnaryOp {
    Clz,
    Ctz,
    Popcnt,
}

/// Division operation type
#[derive(Clone, Copy, Debug)]
enum DivOp {
    DivS,
    DivU,
    RemS,
    RemU,
}

/// Load/Store operation kind
#[derive(Clone, Copy, Debug)]
enum LoadStoreKind {
    I32,
    I8S,
    I8U,
    I16S,
    I16U,
    I64,
    I64_8S,
    I64_8U,
    I64_16S,
    I64_16U,
    I64_32S,
    I64_32U,
    F32,
    F64,
}

/// Represents a value on the operand stack during compilation
#[derive(Clone, Debug)]
enum StackValue {
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
struct Block {
    label: Option<sys::Label>,
    end_jumps: Vec<sys::Jump>,
    else_jump: Option<sys::Jump>,
    stack_depth: usize,
    result_offset: Option<i32>,
}

/// Local variable storage info
#[derive(Clone, Debug)]
enum LocalVar {
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

/// WebAssembly function compiler
#[derive(Clone, Debug)]
pub struct Function {
    stack: Vec<StackValue>,
    blocks: Vec<Block>,
    locals: Vec<LocalVar>,
    param_count: usize,
    frame_offset: i32,
    local_size: i32,
    free_registers: IndexSet<ScratchRegister>,
    free_float_registers: IndexSet<FloatRegister>,
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

// Helper macro for stack underflow error
macro_rules! stack_underflow {
    () => {
        CompileError::Invalid("Stack underflow".into())
    };
}

impl Function {
    pub fn new() -> Self {
        Self {
            stack: vec![],
            blocks: vec![],
            locals: vec![],
            param_count: 0,
            frame_offset: 0,
            local_size: 0,
            free_registers: IndexSet::from_iter([
                ScratchRegister::R0,
                ScratchRegister::R1,
                ScratchRegister::R2,
                ScratchRegister::R3,
                ScratchRegister::R4,
                ScratchRegister::R5,
            ]),
            free_float_registers: IndexSet::from_iter([
                FloatRegister::FR0,
                FloatRegister::FR1,
                FloatRegister::FR2,
                FloatRegister::FR3,
                FloatRegister::FR4,
                FloatRegister::FR5,
            ]),
            globals: vec![],
            memory: None,
            functions: vec![],
            tables: vec![],
            func_types: vec![],
        }
    }

    /// Set globals for the compiler
    #[inline(always)]
    pub fn set_globals(&mut self, globals: Vec<GlobalInfo>) {
        self.globals = globals;
    }

    /// Set memory instance for the compiler
    #[inline(always)]
    pub fn set_memory(&mut self, memory: MemoryInfo) {
        self.memory = Some(memory);
    }

    /// Set function table for direct calls
    #[inline(always)]
    pub fn set_functions(&mut self, functions: Vec<FunctionEntry>) {
        self.functions = functions;
    }

    /// Set tables for indirect calls
    #[inline(always)]
    pub fn set_tables(&mut self, tables: Vec<TableEntry>) {
        self.tables = tables;
    }

    /// Set function types for call_indirect
    #[inline(always)]
    pub fn set_func_types(&mut self, func_types: Vec<FuncType>) {
        self.func_types = func_types;
    }

    /// Pop a value from stack with error handling
    #[inline(always)]
    fn pop_value(&mut self) -> Result<StackValue, CompileError> {
        let val = self.stack.pop().ok_or_else(|| stack_underflow!())?;
        match &val {
            StackValue::Register(reg) => self.free_register(*reg),
            StackValue::FloatRegister(reg) => self.free_float_register(*reg),
            _ => {}
        }
        Ok(val)
    }

    /// Allocate a scratch register, spilling if necessary
    fn alloc_register(&mut self, emitter: &mut Emitter) -> Result<ScratchRegister, CompileError> {
        if let Some(reg) = self.free_registers.pop() {
            Ok(reg)
        } else {
            for sv in self.stack.iter_mut() {
                if let StackValue::Register(reg) = sv {
                    let offset = self.frame_offset;
                    self.frame_offset += 8;
                    emitter.mov(0, mem_sp_offset(offset), *reg)?;
                    let freed_reg = *reg;
                    *sv = StackValue::Stack(offset);
                    return Ok(freed_reg);
                }
            }
            Err(CompileError::Invalid("No registers available".into()))
        }
    }

    #[inline(always)]
    fn free_register(&mut self, reg: ScratchRegister) {
        if !self.free_registers.contains(&reg) {
            self.free_registers.insert(reg);
        }
    }

    /// Allocate a float register, spilling if necessary
    fn alloc_float_register(
        &mut self,
        emitter: &mut Emitter,
    ) -> Result<FloatRegister, CompileError> {
        if let Some(reg) = self.free_float_registers.pop() {
            Ok(reg)
        } else {
            // Try to spill a float register from the stack
            for sv in self.stack.iter_mut() {
                if let StackValue::FloatRegister(reg) = sv {
                    let offset = self.frame_offset;
                    self.frame_offset += 8;
                    emitter.mov_f64(0, mem_sp_offset(offset), *reg)?;
                    let freed_reg = *reg;
                    *sv = StackValue::Stack(offset);
                    return Ok(freed_reg);
                }
            }
            Err(CompileError::Invalid("No float registers available".into()))
        }
    }

    #[inline(always)]
    fn free_float_register(&mut self, reg: FloatRegister) {
        if !self.free_float_registers.contains(&reg) {
            self.free_float_registers.insert(reg);
        }
    }

    #[inline(always)]
    fn push(&mut self, value: StackValue) {
        self.stack.push(value);
    }

    /// Ensure a stack value is in a register
    fn ensure_in_register(
        &mut self,
        emitter: &mut Emitter,
        value: StackValue,
    ) -> Result<ScratchRegister, CompileError> {
        match value {
            StackValue::Register(reg) => {
                // The register may have been freed by pop_value, so we need to
                // remove it from the free list to prevent it from being allocated again
                self.free_registers.retain(|&r| r != reg);
                Ok(reg)
            }
            StackValue::Stack(offset) => {
                let reg = self.alloc_register(emitter)?;
                emitter.mov(0, reg, mem_sp_offset(offset))?;
                Ok(reg)
            }
            StackValue::Const(val) => {
                let reg = self.alloc_register(emitter)?;
                emitter.mov(0, reg, val)?;
                Ok(reg)
            }
            StackValue::Saved(sreg) => {
                let reg = self.alloc_register(emitter)?;
                emitter.mov(0, reg, sreg)?;
                Ok(reg)
            }
            StackValue::FloatRegister(_) | StackValue::ConstF32(_) | StackValue::ConstF64(_) => {
                Err(CompileError::Invalid(
                    "Cannot convert float value to integer register".into(),
                ))
            }
        }
    }

    /// Ensure a stack value is in a float register
    fn ensure_in_float_register(
        &mut self,
        emitter: &mut Emitter,
        value: StackValue,
        is_f32: bool,
    ) -> Result<FloatRegister, CompileError> {
        match value {
            StackValue::FloatRegister(reg) => {
                // Remove from free list to prevent reallocation
                self.free_float_registers.retain(|&r| r != reg);
                Ok(reg)
            }
            StackValue::Stack(offset) => {
                let reg = self.alloc_float_register(emitter)?;
                if is_f32 {
                    emitter.mov_f32(0, reg, mem_sp_offset(offset))?;
                } else {
                    emitter.mov_f64(0, reg, mem_sp_offset(offset))?;
                }
                Ok(reg)
            }
            StackValue::ConstF32(bits) => {
                let reg = self.alloc_float_register(emitter)?;
                // Store float bits to stack, then load as float
                // Ensure 8-byte alignment for float operations
                let temp_offset = (self.frame_offset + 7) & !7;
                self.frame_offset = temp_offset + 8;
                emitter.mov32(0, mem_sp_offset(temp_offset), bits as i32)?;
                emitter.mov_f32(0, reg, mem_sp_offset(temp_offset))?;
                Ok(reg)
            }
            StackValue::ConstF64(bits) => {
                let reg = self.alloc_float_register(emitter)?;
                // Store float bits to stack, then load as float
                // Ensure 8-byte alignment for f64 operations
                let temp_offset = (self.frame_offset + 7) & !7;
                self.frame_offset = temp_offset + 8;

                #[cfg(target_pointer_width = "64")]
                {
                    emitter.mov(0, mem_sp_offset(temp_offset), bits as isize)?;
                }
                #[cfg(not(target_pointer_width = "64"))]
                {
                    // On 32-bit, need to store two halves
                    emitter.mov32(0, mem_sp_offset(temp_offset), (bits & 0xFFFFFFFF) as i32)?;
                    emitter.mov32(0, mem_sp_offset(temp_offset + 4), (bits >> 32) as i32)?;
                }
                emitter.mov_f64(0, reg, mem_sp_offset(temp_offset))?;
                Ok(reg)
            }
            _ => Err(CompileError::Invalid(
                "Expected float value on stack".into(),
            )),
        }
    }

    /// Emit mov from StackValue to a saved register
    fn emit_mov_to_saved(
        &self,
        emitter: &mut Emitter,
        sreg: SavedRegister,
        val: &StackValue,
    ) -> Result<(), CompileError> {
        match val {
            StackValue::Register(reg) => {
                emitter.mov(0, sreg, *reg)?;
            }
            StackValue::Stack(offset) => {
                emitter.mov(0, sreg, mem_sp_offset(*offset))?;
            }
            StackValue::Const(v) => {
                emitter.mov(0, sreg, *v)?;
            }
            StackValue::Saved(src) => {
                emitter.mov(0, sreg, *src)?;
            }
            StackValue::FloatRegister(_) | StackValue::ConstF32(_) | StackValue::ConstF64(_) => {
                return Err(CompileError::Invalid(
                    "Cannot move float value to integer saved register".into(),
                ));
            }
        }
        Ok(())
    }

    /// Emit mov from StackValue to stack offset
    fn emit_mov_to_stack_offset(
        &mut self,
        emitter: &mut Emitter,
        offset: i32,
        val: StackValue,
    ) -> Result<(), CompileError> {
        match val {
            StackValue::Register(reg) => {
                emitter.mov(0, mem_sp_offset(offset), reg)?;
                self.free_register(reg);
            }
            StackValue::Const(c) => {
                emitter.mov(0, mem_sp_offset(offset), c)?;
            }
            StackValue::Saved(sreg) => {
                emitter.mov(0, mem_sp_offset(offset), sreg)?;
            }
            StackValue::Stack(src_offset) if src_offset != offset => {
                emitter.mov(0, mem_sp_offset(offset), mem_sp_offset(src_offset))?;
            }
            _ => {}
        }
        Ok(())
    }

    /// Emit mov to local variable (unified helper for LocalSet/LocalTee)
    fn emit_set_local(
        &mut self,
        emitter: &mut Emitter,
        local: &LocalVar,
        val: &StackValue,
    ) -> Result<(), CompileError> {
        match (local, val) {
            (LocalVar::Saved(sreg), _) => self.emit_mov_to_saved(emitter, *sreg, val),
            (LocalVar::Stack(offset), StackValue::Const(v)) => {
                emitter.mov(0, mem_sp_offset(*offset), *v)?;
                Ok(())
            }
            (LocalVar::Stack(offset), StackValue::Register(reg)) => {
                emitter.mov(0, mem_sp_offset(*offset), *reg)?;
                Ok(())
            }
            (LocalVar::Stack(offset), StackValue::Stack(src)) => {
                emitter.mov(0, mem_sp_offset(*offset), mem_sp_offset(*src))?;
                Ok(())
            }
            (LocalVar::Stack(offset), StackValue::Saved(sreg)) => {
                emitter.mov(0, mem_sp_offset(*offset), *sreg)?;
                Ok(())
            }
            (LocalVar::Stack(offset), StackValue::FloatRegister(freg)) => {
                // Store float register to stack location
                emitter.mov_f64(0, mem_sp_offset(*offset), *freg)?;
                Ok(())
            }
            (LocalVar::Stack(offset), StackValue::ConstF32(bits)) => {
                // Store f32 constant to stack
                emitter.mov32(0, mem_sp_offset(*offset), *bits as i32)?;
                Ok(())
            }
            (LocalVar::Stack(offset), StackValue::ConstF64(bits)) => {
                // Store f64 constant to stack
                #[cfg(target_pointer_width = "64")]
                {
                    emitter.mov(0, mem_sp_offset(*offset), *bits as isize)?;
                }
                #[cfg(not(target_pointer_width = "64"))]
                {
                    emitter.mov32(0, mem_sp_offset(*offset), (*bits & 0xFFFFFFFF) as i32)?;
                    emitter.mov32(0, mem_sp_offset(*offset + 4), (*bits >> 32) as i32)?;
                }
                Ok(())
            }
        }
    }

    /// Save all register values to stack (for control flow)
    fn save_all_to_stack(&mut self, emitter: &mut Emitter) -> Result<(), CompileError> {
        for (i, reg) in self
            .stack
            .clone()
            .into_iter()
            .enumerate()
            .filter_map(|(i, sv)| {
                if let StackValue::Register(reg) = sv {
                    Some((i, reg))
                } else {
                    None
                }
            })
        {
            let offset = self.frame_offset;
            self.frame_offset += 8;
            emitter.mov(0, mem_sp_offset(offset), reg)?;
            self.free_register(reg);
            self.stack[i] = StackValue::Stack(offset);
        }
        Ok(())
    }

    /// Create a new block with common initialization
    fn new_block(&mut self, label: Option<sys::Label>, has_result: bool) -> Block {
        let result_offset = if has_result {
            let offset = self.frame_offset;
            self.frame_offset += 8;
            Some(offset)
        } else {
            None
        };

        Block {
            label,
            end_jumps: vec![],
            else_jump: None,
            stack_depth: self.stack.len(),
            result_offset,
        }
    }

    /// Check if block type has a result
    const fn block_has_result(blockty: &wasmparser::BlockType) -> bool {
        matches!(blockty, wasmparser::BlockType::Type(_))
    }

    /// Make argument types bitmask for sljit
    /// Supports up to 7 arguments (with 32-bit arg_types bitmask)
    fn make_arg_types(params: &[ValType], results: &[ValType]) -> i32 {
        let ret_type = if results.is_empty() {
            sys::SLJIT_ARG_TYPE_RET_VOID
        } else {
            sys::SLJIT_ARG_TYPE_W
        };
        // Encode up to 7 arguments (bits 4-31 = 7 slots of 4 bits each)
        params
            .iter()
            .take(7)
            .enumerate()
            .fold(ret_type, |acc, (i, _)| {
                acc | (sys::SLJIT_ARG_TYPE_W << (4 * (i + 1)))
            })
    }

    /// Get the stack space needed for extra arguments (arguments beyond the first 3)
    const fn get_extra_arg_stack_size(param_count: usize) -> i32 {
        if param_count <= 3 {
            0
        } else {
            // Each extra argument needs 8 bytes (word size on 64-bit)
            // Align to 16 bytes for ABI compliance
            let extra_args = param_count - 3;
            let size = (extra_args * 8) as i32;
            (size + 15) & !15 // 16-byte alignment
        }
    }

    /// Compile a WebAssembly function
    pub fn compile_function<'a, B>(
        &mut self,
        compiler: &mut Compiler,
        params: &[ValType],
        results: &[ValType],
        locals: &[ValType],
        body: B,
    ) -> Result<(), CompileError>
    where
        B: Iterator<Item = Operator<'a>>,
    {
        let mut emitter = Emitter::new(compiler);
        let total_locals: usize = params.len() + locals.len();

        self.param_count = total_locals;
        self.locals.clear();

        let saved_count = std::cmp::min(params.len(), 3);
        let mut stack_local_offset = 0i32;

        // Setup locals: first 3 params in saved registers, rest on stack
        for i in 0..params.len() {
            self.locals.push(if i < 3 {
                LocalVar::Saved([SavedRegister::S0, SavedRegister::S1, SavedRegister::S2][i])
            } else {
                let var = LocalVar::Stack(stack_local_offset);
                stack_local_offset += 8;
                var
            });
        }

        // Declared locals go on stack
        for _ in locals {
            self.locals.push(LocalVar::Stack(stack_local_offset));
            stack_local_offset += 8;
        }

        // Ensure frame_offset is 16-byte aligned for proper float operation alignment
        let aligned_offset = (stack_local_offset + 15) & !15;
        // Use larger stack space to avoid any potential overflow issues with float ops and 64-bit ops on 32-bit
        // But not too large to avoid Windows stack probe issues on 32-bit
        self.local_size = aligned_offset + 4096; // 4KB should be sufficient
        self.frame_offset = aligned_offset;

        emitter.emit_enter(
            0,
            Self::make_arg_types(params, results),
            regs! { gp: 6, float: 6 },
            regs!(saved_count as i32),
            self.local_size,
        )?;

        // Initialize stack locals to zero
        for local in &self.locals {
            if let LocalVar::Stack(offset) = local {
                emitter.mov(0, mem_sp_offset(*offset), 0i32)?;
            }
        }

        self.blocks.push(Block {
            label: None,
            end_jumps: vec![],
            else_jump: None,
            stack_depth: 0,
            result_offset: None,
        });

        for op in body {
            self.compile_operator(&mut emitter, &op)?;
        }
        Ok(())
    }

    /// Compile a single WebAssembly operator
    fn compile_operator(
        &mut self,
        emitter: &mut Emitter,
        op: &Operator,
    ) -> Result<(), CompileError> {
        match op {
            Operator::I32Const { value } => {
                self.push(StackValue::Const(*value));
                Ok(())
            }

            Operator::LocalGet { local_index } => {
                let local = &self.locals[*local_index as usize];
                self.push(match local {
                    LocalVar::Saved(sreg) => StackValue::Saved(*sreg),
                    LocalVar::Stack(offset) => StackValue::Stack(*offset),
                });
                Ok(())
            }

            Operator::LocalSet { local_index } => {
                let value = self.pop_value()?;
                let local = self.locals[*local_index as usize].clone();
                self.emit_set_local(emitter, &local, &value)?;
                if let StackValue::Register(reg) = value {
                    self.free_register(reg);
                }
                Ok(())
            }

            Operator::LocalTee { local_index } => {
                let value = self
                    .stack
                    .last()
                    .cloned()
                    .ok_or_else(|| stack_underflow!())?;
                let local = self.locals[*local_index as usize].clone();
                self.emit_set_local(emitter, &local, &value)
            }

            // Binary operations (i32)
            Operator::I32Add => self.compile_binary_op(emitter, BinaryOp::Add),
            Operator::I32Sub => self.compile_binary_op(emitter, BinaryOp::Sub),
            Operator::I32Mul => self.compile_binary_op(emitter, BinaryOp::Mul),
            Operator::I32And => self.compile_binary_op(emitter, BinaryOp::And),
            Operator::I32Or => self.compile_binary_op(emitter, BinaryOp::Or),
            Operator::I32Xor => self.compile_binary_op(emitter, BinaryOp::Xor),
            Operator::I32Shl => self.compile_binary_op(emitter, BinaryOp::Shl),
            Operator::I32ShrS => self.compile_binary_op(emitter, BinaryOp::ShrS),
            Operator::I32ShrU => self.compile_binary_op(emitter, BinaryOp::ShrU),
            Operator::I32Rotl => self.compile_binary_op(emitter, BinaryOp::Rotl),
            Operator::I32Rotr => self.compile_binary_op(emitter, BinaryOp::Rotr),

            // Division operations
            Operator::I32DivS => self.compile_div_op(emitter, DivOp::DivS),
            Operator::I32DivU => self.compile_div_op(emitter, DivOp::DivU),
            Operator::I32RemS => self.compile_div_op(emitter, DivOp::RemS),
            Operator::I32RemU => self.compile_div_op(emitter, DivOp::RemU),

            // Unary operations
            Operator::I32Clz => self.compile_unary_op(emitter, UnaryOp::Clz),
            Operator::I32Ctz => self.compile_unary_op(emitter, UnaryOp::Ctz),
            Operator::I32Popcnt => self.compile_unary_op(emitter, UnaryOp::Popcnt),

            // Comparison operations
            Operator::I32Eqz => {
                let val = self.pop_value()?;
                let reg = self.ensure_in_register(emitter, val)?;
                emitter.branch(
                    Condition::Equal,
                    reg,
                    0i32,
                    |e| {
                        e.mov(0, reg, 1i32)?;
                        Ok(())
                    },
                    |e| {
                        e.mov(0, reg, 0i32)?;
                        Ok(())
                    },
                )?;
                self.push(StackValue::Register(reg));
                Ok(())
            }
            Operator::I32Eq => self.compile_compare_op(emitter, Condition::Equal),
            Operator::I32Ne => self.compile_compare_op(emitter, Condition::NotEqual),
            Operator::I32LtS => self.compile_compare_op(emitter, Condition::SigLess32),
            Operator::I32LtU => self.compile_compare_op(emitter, Condition::Less32),
            Operator::I32GtS => self.compile_compare_op(emitter, Condition::SigGreater32),
            Operator::I32GtU => self.compile_compare_op(emitter, Condition::Greater32),
            Operator::I32LeS => self.compile_compare_op(emitter, Condition::SigLessEqual32),
            Operator::I32LeU => self.compile_compare_op(emitter, Condition::LessEqual32),
            Operator::I32GeS => self.compile_compare_op(emitter, Condition::SigGreaterEqual32),
            Operator::I32GeU => self.compile_compare_op(emitter, Condition::GreaterEqual32),

            // Control flow
            Operator::Block { blockty } => {
                self.save_all_to_stack(emitter)?;
                let block = self.new_block(None, Self::block_has_result(blockty));
                self.blocks.push(block);
                Ok(())
            }

            Operator::Loop { blockty: _ } => {
                self.save_all_to_stack(emitter)?;
                let label = emitter.put_label()?;
                let block = self.new_block(Some(label), false);
                self.blocks.push(block);
                Ok(())
            }

            Operator::If { blockty } => {
                let cond = self.pop_value()?;
                self.save_all_to_stack(emitter)?;
                let jump = self.emit_cond_jump(emitter, cond, Condition::Equal)?;
                let mut block = self.new_block(None, Self::block_has_result(blockty));
                block.else_jump = Some(jump);
                self.blocks.push(block);
                Ok(())
            }

            Operator::Else => {
                let (result_offset, stack_depth) = {
                    let block = self
                        .blocks
                        .last()
                        .ok_or_else(|| CompileError::Invalid("No block for else".into()))?;
                    (block.result_offset, block.stack_depth)
                };

                if let Some(offset) = result_offset
                    && self.stack.len() > stack_depth
                {
                    let val = self.stack.pop().unwrap();
                    self.emit_mov_to_stack_offset(emitter, offset, val)?;
                }

                let jump_to_end = emitter.jump(JumpType::Jump)?;
                let block = self
                    .blocks
                    .last_mut()
                    .ok_or_else(|| CompileError::Invalid("No block for else".into()))?;
                block.end_jumps.push(jump_to_end);

                if let Some(mut else_jump) = block.else_jump.take() {
                    let mut label = emitter.put_label()?;
                    else_jump.set_label(&mut label);
                }
                self.stack.truncate(stack_depth);
                Ok(())
            }

            Operator::End => self.compile_end(emitter),

            Operator::Br { relative_depth } => self.compile_br(emitter, *relative_depth),
            Operator::BrIf { relative_depth } => self.compile_br_if(emitter, *relative_depth),

            Operator::Return => {
                if let Some(value) = self.stack.pop() {
                    if let StackValue::Register(reg) = &value {
                        self.free_register(*reg);
                    }
                    emitter.emit_return(ReturnOp::Mov, value.to_operand())?;
                } else {
                    emitter.return_void()?;
                }
                Ok(())
            }

            Operator::Drop => {
                self.pop_value()?;
                Ok(())
            }

            Operator::Select => {
                let (cond, val2, val1) = (self.pop_value()?, self.pop_value()?, self.pop_value()?);
                let (reg1, reg2, cond_reg) = (
                    self.ensure_in_register(emitter, val1)?,
                    self.ensure_in_register(emitter, val2)?,
                    self.ensure_in_register(emitter, cond)?,
                );
                emitter.branch(
                    Condition::NotEqual,
                    cond_reg,
                    0i32,
                    |_| Ok(()),
                    |e| {
                        e.mov(0, reg1, reg2)?;
                        Ok(())
                    },
                )?;
                self.free_register(cond_reg);
                self.free_register(reg2);
                self.push(StackValue::Register(reg1));
                Ok(())
            }

            Operator::Nop => Ok(()),
            Operator::Unreachable => {
                emitter.breakpoint()?;
                Ok(())
            }

            // Memory operations
            Operator::I32Load { memarg } => {
                self.compile_load_op(emitter, memarg.offset as i32, LoadStoreKind::I32)
            }
            Operator::I32Load8S { memarg } => {
                self.compile_load_op(emitter, memarg.offset as i32, LoadStoreKind::I8S)
            }
            Operator::I32Load8U { memarg } => {
                self.compile_load_op(emitter, memarg.offset as i32, LoadStoreKind::I8U)
            }
            Operator::I32Load16S { memarg } => {
                self.compile_load_op(emitter, memarg.offset as i32, LoadStoreKind::I16S)
            }
            Operator::I32Load16U { memarg } => {
                self.compile_load_op(emitter, memarg.offset as i32, LoadStoreKind::I16U)
            }
            Operator::I32Store { memarg } => {
                self.compile_store_op(emitter, memarg.offset as i32, LoadStoreKind::I32)
            }
            Operator::I32Store8 { memarg } => {
                self.compile_store_op(emitter, memarg.offset as i32, LoadStoreKind::I8U)
            }
            Operator::I32Store16 { memarg } => {
                self.compile_store_op(emitter, memarg.offset as i32, LoadStoreKind::I16U)
            }

            // I64 operations
            Operator::I64Const { value } => {
                #[cfg(target_pointer_width = "64")]
                {
                    let reg = self.alloc_register(emitter)?;
                    emitter.mov(0, reg, *value as isize)?;
                    self.push(StackValue::Register(reg));
                    Ok(())
                }
                #[cfg(not(target_pointer_width = "64"))]
                {
                    // On 32-bit platforms, store both halves on the stack
                    let offset = self.frame_offset;
                    self.frame_offset += 8;
                    // Store lower 32 bits
                    emitter.mov32(0, mem_sp_offset(offset), (*value & 0xFFFFFFFF) as i32)?;
                    // Store upper 32 bits
                    emitter.mov32(0, mem_sp_offset(offset + 4), (*value >> 32) as i32)?;
                    self.push(StackValue::Stack(offset));
                    Ok(())
                }
            }
            Operator::I64Add => self.compile_binary_op64(emitter, BinaryOp::Add),
            Operator::I64Sub => self.compile_binary_op64(emitter, BinaryOp::Sub),
            Operator::I64Mul => self.compile_binary_op64(emitter, BinaryOp::Mul),
            Operator::I64And => self.compile_binary_op64(emitter, BinaryOp::And),
            Operator::I64Or => self.compile_binary_op64(emitter, BinaryOp::Or),
            Operator::I64Xor => self.compile_binary_op64(emitter, BinaryOp::Xor),
            Operator::I64Shl => self.compile_binary_op64(emitter, BinaryOp::Shl),
            Operator::I64ShrS => self.compile_binary_op64(emitter, BinaryOp::ShrS),
            Operator::I64ShrU => self.compile_binary_op64(emitter, BinaryOp::ShrU),
            Operator::I64Rotl => self.compile_binary_op64(emitter, BinaryOp::Rotl),
            Operator::I64Rotr => self.compile_binary_op64(emitter, BinaryOp::Rotr),

            // I64 division operations
            Operator::I64DivS => self.compile_div_op64(emitter, DivOp::DivS),
            Operator::I64DivU => self.compile_div_op64(emitter, DivOp::DivU),
            Operator::I64RemS => self.compile_div_op64(emitter, DivOp::RemS),
            Operator::I64RemU => self.compile_div_op64(emitter, DivOp::RemU),

            // I64 unary operations
            Operator::I64Clz => self.compile_unary_op64(emitter, UnaryOp::Clz),
            Operator::I64Ctz => self.compile_unary_op64(emitter, UnaryOp::Ctz),
            Operator::I64Popcnt => self.compile_unary_op64(emitter, UnaryOp::Popcnt),

            // I64 comparison operations
            Operator::I64Eqz => {
                let val = self.pop_value()?;
                let reg = self.ensure_in_register(emitter, val)?;
                emitter.branch(
                    Condition::Equal,
                    reg,
                    0i32,
                    |e| {
                        e.mov(0, reg, 1i32)?;
                        Ok(())
                    },
                    |e| {
                        e.mov(0, reg, 0i32)?;
                        Ok(())
                    },
                )?;
                self.push(StackValue::Register(reg));
                Ok(())
            }
            Operator::I64Eq => self.compile_compare_op64(emitter, Condition::Equal),
            Operator::I64Ne => self.compile_compare_op64(emitter, Condition::NotEqual),
            Operator::I64LtS => self.compile_compare_op64(emitter, Condition::SigLess),
            Operator::I64LtU => self.compile_compare_op64(emitter, Condition::Less),
            Operator::I64GtS => self.compile_compare_op64(emitter, Condition::SigGreater),
            Operator::I64GtU => self.compile_compare_op64(emitter, Condition::Greater),
            Operator::I64LeS => self.compile_compare_op64(emitter, Condition::SigLessEqual),
            Operator::I64LeU => self.compile_compare_op64(emitter, Condition::LessEqual),
            Operator::I64GeS => self.compile_compare_op64(emitter, Condition::SigGreaterEqual),
            Operator::I64GeU => self.compile_compare_op64(emitter, Condition::GreaterEqual),

            // I64 memory operations
            Operator::I64Load { memarg } => {
                self.compile_load_op64(emitter, memarg.offset as i32, LoadStoreKind::I64)
            }
            Operator::I64Load8S { memarg } => {
                self.compile_load_op64(emitter, memarg.offset as i32, LoadStoreKind::I64_8S)
            }
            Operator::I64Load8U { memarg } => {
                self.compile_load_op64(emitter, memarg.offset as i32, LoadStoreKind::I64_8U)
            }
            Operator::I64Load16S { memarg } => {
                self.compile_load_op64(emitter, memarg.offset as i32, LoadStoreKind::I64_16S)
            }
            Operator::I64Load16U { memarg } => {
                self.compile_load_op64(emitter, memarg.offset as i32, LoadStoreKind::I64_16U)
            }
            Operator::I64Load32S { memarg } => {
                self.compile_load_op64(emitter, memarg.offset as i32, LoadStoreKind::I64_32S)
            }
            Operator::I64Load32U { memarg } => {
                self.compile_load_op64(emitter, memarg.offset as i32, LoadStoreKind::I64_32U)
            }
            Operator::I64Store { memarg } => {
                self.compile_store_op64(emitter, memarg.offset as i32, LoadStoreKind::I64)
            }
            Operator::I64Store8 { memarg } => {
                self.compile_store_op64(emitter, memarg.offset as i32, LoadStoreKind::I64_8U)
            }
            Operator::I64Store16 { memarg } => {
                self.compile_store_op64(emitter, memarg.offset as i32, LoadStoreKind::I64_16U)
            }
            Operator::I64Store32 { memarg } => {
                self.compile_store_op64(emitter, memarg.offset as i32, LoadStoreKind::I64_32U)
            }

            Operator::I32WrapI64 => {
                let val = self.pop_value()?;
                #[cfg(target_pointer_width = "64")]
                {
                    let reg = self.ensure_in_register(emitter, val)?;
                    emitter.mov32(0, reg, reg)?;
                    self.push(StackValue::Register(reg));
                    Ok(())
                }
                #[cfg(not(target_pointer_width = "64"))]
                {
                    // On 32-bit: i64 is stored as 8 bytes on stack (low 4 bytes, high 4 bytes)
                    // We just need to load the low 32 bits
                    match val {
                        StackValue::Stack(offset) => {
                            let reg = self.alloc_register(emitter)?;
                            emitter.mov32(0, reg, mem_sp_offset(offset))?; // Load low 32 bits only
                            self.push(StackValue::Register(reg));
                            Ok(())
                        }
                        _ => {
                            // If it's a register, it's already the low 32 bits
                            let reg = self.ensure_in_register(emitter, val)?;
                            emitter.mov32(0, reg, reg)?;
                            self.push(StackValue::Register(reg));
                            Ok(())
                        }
                    }
                }
            }

            Operator::I64ExtendI32S => {
                #[cfg(target_pointer_width = "64")]
                {
                    let val = self.pop_value()?;
                    let reg = self.ensure_in_register(emitter, val)?;
                    emitter.mov_s32(0, reg, reg)?;
                    self.push(StackValue::Register(reg));
                    Ok(())
                }
                #[cfg(not(target_pointer_width = "64"))]
                {
                    // On 32-bit: extend i32 to i64 (sign extend)
                    let val = self.pop_value()?;
                    let reg = self.ensure_in_register(emitter, val)?;

                    // Allocate stack space for i64 result
                    let offset = self.frame_offset;
                    self.frame_offset += 8;

                    // Store low part (original i32 value)
                    emitter.mov32(0, mem_sp_offset(offset), reg)?;

                    // Sign extend to high part
                    emitter.ashr32(0, reg, reg, 31i32)?;
                    emitter.mov32(0, mem_sp_offset(offset + 4), reg)?;

                    self.free_register(reg);
                    self.push(StackValue::Stack(offset));
                    Ok(())
                }
            }

            Operator::I64ExtendI32U => {
                #[cfg(target_pointer_width = "64")]
                {
                    let val = self.pop_value()?;
                    let reg = self.ensure_in_register(emitter, val)?;
                    emitter.mov_u32(0, reg, reg)?;
                    self.push(StackValue::Register(reg));
                    Ok(())
                }
                #[cfg(not(target_pointer_width = "64"))]
                {
                    // On 32-bit: extend i32 to i64 (zero extend)
                    let val = self.pop_value()?;
                    let reg = self.ensure_in_register(emitter, val)?;

                    // Allocate stack space for i64 result
                    let offset = self.frame_offset;
                    self.frame_offset += 8;

                    // Store low part (original i32 value)
                    emitter.mov32(0, mem_sp_offset(offset), reg)?;

                    // Zero extend to high part
                    emitter.mov32(0, mem_sp_offset(offset + 4), 0i32)?;

                    self.free_register(reg);
                    self.push(StackValue::Stack(offset));
                    Ok(())
                }
            }

            // Floating point constants
            Operator::F32Const { value } => {
                self.push(StackValue::ConstF32(value.bits()));
                Ok(())
            }
            Operator::F64Const { value } => {
                self.push(StackValue::ConstF64(value.bits()));
                Ok(())
            }

            // Floating point load/store operations
            Operator::F32Load { memarg } => {
                self.compile_load_op(emitter, memarg.offset as i32, LoadStoreKind::F32)
            }
            Operator::F64Load { memarg } => {
                self.compile_load_op(emitter, memarg.offset as i32, LoadStoreKind::F64)
            }
            Operator::F32Store { memarg } => {
                self.compile_store_op(emitter, memarg.offset as i32, LoadStoreKind::F32)
            }
            Operator::F64Store { memarg } => {
                self.compile_store_op(emitter, memarg.offset as i32, LoadStoreKind::F64)
            }

            // F32 binary operations
            Operator::F32Add => self.compile_float_binary_op(emitter, FloatBinaryOp::Add, true),
            Operator::F32Sub => self.compile_float_binary_op(emitter, FloatBinaryOp::Sub, true),
            Operator::F32Mul => self.compile_float_binary_op(emitter, FloatBinaryOp::Mul, true),
            Operator::F32Div => self.compile_float_binary_op(emitter, FloatBinaryOp::Div, true),
            Operator::F32Min => self.compile_float_binary_op(emitter, FloatBinaryOp::Min, true),
            Operator::F32Max => self.compile_float_binary_op(emitter, FloatBinaryOp::Max, true),
            Operator::F32Copysign => {
                self.compile_float_binary_op(emitter, FloatBinaryOp::Copysign, true)
            }

            // F64 binary operations
            Operator::F64Add => self.compile_float_binary_op(emitter, FloatBinaryOp::Add, false),
            Operator::F64Sub => self.compile_float_binary_op(emitter, FloatBinaryOp::Sub, false),
            Operator::F64Mul => self.compile_float_binary_op(emitter, FloatBinaryOp::Mul, false),
            Operator::F64Div => self.compile_float_binary_op(emitter, FloatBinaryOp::Div, false),
            Operator::F64Min => self.compile_float_binary_op(emitter, FloatBinaryOp::Min, false),
            Operator::F64Max => self.compile_float_binary_op(emitter, FloatBinaryOp::Max, false),
            Operator::F64Copysign => {
                self.compile_float_binary_op(emitter, FloatBinaryOp::Copysign, false)
            }

            // F32 unary operations
            Operator::F32Neg => self.compile_float_unary_op(emitter, FloatUnaryOp::Neg, true),
            Operator::F32Abs => self.compile_float_unary_op(emitter, FloatUnaryOp::Abs, true),
            Operator::F32Sqrt => self.compile_float_unary_op(emitter, FloatUnaryOp::Sqrt, true),
            Operator::F32Ceil => self.compile_float_unary_op(emitter, FloatUnaryOp::Ceil, true),
            Operator::F32Floor => self.compile_float_unary_op(emitter, FloatUnaryOp::Floor, true),
            Operator::F32Trunc => self.compile_float_unary_op(emitter, FloatUnaryOp::Trunc, true),
            Operator::F32Nearest => {
                self.compile_float_unary_op(emitter, FloatUnaryOp::Nearest, true)
            }

            // F64 unary operations
            Operator::F64Neg => self.compile_float_unary_op(emitter, FloatUnaryOp::Neg, false),
            Operator::F64Abs => self.compile_float_unary_op(emitter, FloatUnaryOp::Abs, false),
            Operator::F64Sqrt => self.compile_float_unary_op(emitter, FloatUnaryOp::Sqrt, false),
            Operator::F64Ceil => self.compile_float_unary_op(emitter, FloatUnaryOp::Ceil, false),
            Operator::F64Floor => self.compile_float_unary_op(emitter, FloatUnaryOp::Floor, false),
            Operator::F64Trunc => self.compile_float_unary_op(emitter, FloatUnaryOp::Trunc, false),
            Operator::F64Nearest => {
                self.compile_float_unary_op(emitter, FloatUnaryOp::Nearest, false)
            }

            // Float comparisons - F32
            Operator::F32Eq => self.compile_float_compare_op(emitter, Condition::FEqual, true),
            Operator::F32Ne => self.compile_float_compare_op(emitter, Condition::FNotEqual, true),
            Operator::F32Lt => self.compile_float_compare_op(emitter, Condition::FLess, true),
            Operator::F32Gt => self.compile_float_compare_op(emitter, Condition::FGreater, true),
            Operator::F32Le => self.compile_float_compare_op(emitter, Condition::FLessEqual, true),
            Operator::F32Ge => {
                self.compile_float_compare_op(emitter, Condition::FGreaterEqual, true)
            }

            // Float comparisons - F64
            Operator::F64Eq => self.compile_float_compare_op(emitter, Condition::FEqual, false),
            Operator::F64Ne => self.compile_float_compare_op(emitter, Condition::FNotEqual, false),
            Operator::F64Lt => self.compile_float_compare_op(emitter, Condition::FLess, false),
            Operator::F64Gt => self.compile_float_compare_op(emitter, Condition::FGreater, false),
            Operator::F64Le => self.compile_float_compare_op(emitter, Condition::FLessEqual, false),
            Operator::F64Ge => {
                self.compile_float_compare_op(emitter, Condition::FGreaterEqual, false)
            }

            // Conversions: int to float
            Operator::F32ConvertI32S => {
                self.compile_convert_int_to_float(emitter, true, true, true)
            }
            Operator::F32ConvertI32U => {
                self.compile_convert_int_to_float(emitter, true, true, false)
            }
            Operator::F64ConvertI32S => {
                self.compile_convert_int_to_float(emitter, false, true, true)
            }
            Operator::F64ConvertI32U => {
                self.compile_convert_int_to_float(emitter, false, true, false)
            }
            Operator::F32ConvertI64S => {
                self.compile_convert_int_to_float(emitter, true, false, true)
            }
            Operator::F32ConvertI64U => {
                self.compile_convert_int_to_float(emitter, true, false, false)
            }
            Operator::F64ConvertI64S => {
                self.compile_convert_int_to_float(emitter, false, false, true)
            }
            Operator::F64ConvertI64U => {
                self.compile_convert_int_to_float(emitter, false, false, false)
            }

            // Conversions: float to int (truncate)
            Operator::I32TruncF32S => self.compile_convert_float_to_int(emitter, true, true, true),
            Operator::I32TruncF32U => self.compile_convert_float_to_int(emitter, true, true, false),
            Operator::I32TruncF64S => self.compile_convert_float_to_int(emitter, false, true, true),
            Operator::I32TruncF64U => {
                self.compile_convert_float_to_int(emitter, false, true, false)
            }
            Operator::I64TruncF32S => self.compile_convert_float_to_int(emitter, true, false, true),
            Operator::I64TruncF32U => {
                self.compile_convert_float_to_int(emitter, true, false, false)
            }
            Operator::I64TruncF64S => {
                self.compile_convert_float_to_int(emitter, false, false, true)
            }
            Operator::I64TruncF64U => {
                self.compile_convert_float_to_int(emitter, false, false, false)
            }

            // Conversions: float to float
            Operator::F32DemoteF64 => self.compile_float_demote_promote(emitter, true),
            Operator::F64PromoteF32 => self.compile_float_demote_promote(emitter, false),

            // Reinterpret operations
            Operator::I32ReinterpretF32 => {
                let val = self.pop_value()?;
                let freg = self.ensure_in_float_register(emitter, val, true)?;
                let reg = self.alloc_register(emitter)?;
                // Store float to stack, load as int (ensure 8-byte alignment)
                let temp_offset = (self.frame_offset + 7) & !7;
                self.frame_offset = temp_offset + 8;
                emitter.mov_f32(0, mem_sp_offset(temp_offset), freg)?;
                emitter.mov32(0, reg, mem_sp_offset(temp_offset))?;
                self.free_float_register(freg);
                self.push(StackValue::Register(reg));
                Ok(())
            }
            Operator::I64ReinterpretF64 => {
                let val = self.pop_value()?;
                let freg = self.ensure_in_float_register(emitter, val, false)?;
                // Ensure 8-byte alignment for f64 operations
                let temp_offset = (self.frame_offset + 7) & !7;
                self.frame_offset = temp_offset + 8;
                emitter.mov_f64(0, mem_sp_offset(temp_offset), freg)?;
                self.free_float_register(freg);

                #[cfg(target_pointer_width = "64")]
                {
                    let reg = self.alloc_register(emitter)?;
                    emitter.mov(0, reg, mem_sp_offset(temp_offset))?;
                    self.push(StackValue::Register(reg));
                }
                #[cfg(not(target_pointer_width = "64"))]
                {
                    // On 32-bit, the i64 value stays on stack (already there from mov_f64)
                    self.push(StackValue::Stack(temp_offset));
                }
                Ok(())
            }
            Operator::F32ReinterpretI32 => {
                let val = self.pop_value()?;
                let reg = self.ensure_in_register(emitter, val)?;
                let freg = self.alloc_float_register(emitter)?;
                // Ensure 8-byte alignment for float operations
                let temp_offset = (self.frame_offset + 7) & !7;
                self.frame_offset = temp_offset + 8;
                emitter.mov32(0, mem_sp_offset(temp_offset), reg)?;
                emitter.mov_f32(0, freg, mem_sp_offset(temp_offset))?;
                self.free_register(reg);
                self.push(StackValue::FloatRegister(freg));
                Ok(())
            }
            Operator::F64ReinterpretI64 => {
                let val = self.pop_value()?;
                let freg = self.alloc_float_register(emitter)?;
                // Ensure 8-byte alignment for f64 operations
                let temp_offset = (self.frame_offset + 7) & !7;
                self.frame_offset = temp_offset + 8;

                #[cfg(target_pointer_width = "64")]
                {
                    let reg = self.ensure_in_register(emitter, val)?;
                    emitter.mov(0, mem_sp_offset(temp_offset), reg)?;
                    self.free_register(reg);
                }
                #[cfg(not(target_pointer_width = "64"))]
                {
                    // On 32-bit, i64 value is already on stack
                    match val {
                        StackValue::Stack(value_offset) => {
                            // Copy from value location to temp location
                            let temp_reg = self.alloc_register(emitter)?;
                            emitter.mov32(0, temp_reg, mem_sp_offset(value_offset))?;
                            emitter.mov32(0, mem_sp_offset(temp_offset), temp_reg)?;
                            emitter.mov32(0, temp_reg, mem_sp_offset(value_offset + 4))?;
                            emitter.mov32(0, mem_sp_offset(temp_offset + 4), temp_reg)?;
                            self.free_register(temp_reg);
                        }
                        _ => {
                            return Err(CompileError::Invalid(
                                "I64 value must be on stack on 32-bit platform".into(),
                            ));
                        }
                    }
                }

                emitter.mov_f64(0, freg, mem_sp_offset(temp_offset))?;
                self.push(StackValue::FloatRegister(freg));
                Ok(())
            }

            // Global variable operations
            Operator::GlobalGet { global_index } => self.compile_global_get(emitter, *global_index),
            Operator::GlobalSet { global_index } => self.compile_global_set(emitter, *global_index),

            // Memory operations
            Operator::MemorySize { mem, .. } => self.compile_memory_size(emitter, *mem),
            Operator::MemoryGrow { mem, .. } => self.compile_memory_grow(emitter, *mem),

            // Function calls
            Operator::Call { function_index } => self.compile_call(emitter, *function_index),
            Operator::CallIndirect {
                type_index,
                table_index,
            } => self.compile_call_indirect(emitter, *type_index, *table_index),

            // Branch table
            Operator::BrTable { targets } => self.compile_br_table(emitter, targets),

            _ => Err(CompileError::Unsupported(format!(
                "Unsupported operator: {:?}",
                op
            ))),
        }
    }

    /// Helper to emit conditional jump (used by If and BrIf)
    fn emit_cond_jump(
        &mut self,
        emitter: &mut Emitter,
        cond: StackValue,
        condition: Condition,
    ) -> Result<sys::Jump, CompileError> {
        match cond {
            StackValue::Register(reg) => {
                let j = emitter.cmp(condition, reg, 0i32)?;
                self.free_register(reg);
                Ok(j)
            }
            StackValue::Saved(sreg) => Ok(emitter.cmp(condition, sreg, 0i32)?),
            _ => Ok(emitter.cmp(condition, cond.to_operand(), 0i32)?),
        }
    }

    fn compile_end(&mut self, emitter: &mut Emitter) -> Result<(), CompileError> {
        if self.blocks.len() == 1 {
            let block = self.blocks.pop().unwrap();
            let mut label = emitter.put_label()?;
            for mut jump in block.end_jumps {
                jump.set_label(&mut label);
            }
            if let Some(mut else_jump) = block.else_jump {
                else_jump.set_label(&mut label);
            }

            if let Some(value) = self.stack.pop() {
                if let StackValue::Register(reg) = &value {
                    self.free_register(*reg);
                }
                emitter.emit_return(ReturnOp::Mov, value.to_operand())?;
            } else {
                emitter.return_void()?;
            }
        } else {
            let block = self
                .blocks
                .pop()
                .ok_or_else(|| CompileError::Invalid("No block for end".into()))?;

            if let Some(offset) = block.result_offset {
                if self.stack.len() > block.stack_depth {
                    let val = self.stack.pop().unwrap();
                    self.emit_mov_to_stack_offset(emitter, offset, val)?;
                }
                self.stack.truncate(block.stack_depth);
                self.stack.push(StackValue::Stack(offset));
            }

            let mut label = emitter.put_label()?;
            for mut jump in block.end_jumps {
                jump.set_label(&mut label);
            }
            if let Some(mut else_jump) = block.else_jump {
                else_jump.set_label(&mut label);
            }
        }
        Ok(())
    }

    fn compile_br(
        &mut self,
        emitter: &mut Emitter,
        relative_depth: u32,
    ) -> Result<(), CompileError> {
        let block_idx = self.blocks.len() - 1 - relative_depth as usize;

        if let Some(result_offset) = self.blocks[block_idx].result_offset
            && let Some(val) = self.stack.pop()
        {
            if let StackValue::Register(reg) = &val {
                self.free_register(*reg);
            }
            self.emit_mov_to_stack_offset(emitter, result_offset, val)?;
        }

        self.save_all_to_stack(emitter)?;

        let mut jump = emitter.jump(JumpType::Jump)?;

        if let Some(mut label) = self.blocks[block_idx].label {
            jump.set_label(&mut label);
        } else {
            self.blocks[block_idx].end_jumps.push(jump);
        }
        Ok(())
    }

    fn compile_br_if(
        &mut self,
        emitter: &mut Emitter,
        relative_depth: u32,
    ) -> Result<(), CompileError> {
        let cond = self.pop_value()?;
        self.save_all_to_stack(emitter)?;

        let block_idx = self.blocks.len() - 1 - relative_depth as usize;
        let mut jump = self.emit_cond_jump(emitter, cond, Condition::NotEqual)?;

        if let Some(mut label) = self.blocks[block_idx].label {
            jump.set_label(&mut label);
        } else {
            self.blocks[block_idx].end_jumps.push(jump);
        }
        Ok(())
    }

    /// Compile a binary arithmetic operation using functional dispatch
    fn compile_binary_op(
        &mut self,
        emitter: &mut Emitter,
        op: BinaryOp,
    ) -> Result<(), CompileError> {
        let (b, a) = (self.pop_value()?, self.pop_value()?);
        let reg_a = self.ensure_in_register(emitter, a)?;
        let operand_b = b.to_operand();

        match op {
            BinaryOp::Add => emitter.add32(0, reg_a, reg_a, operand_b)?,
            BinaryOp::Sub => emitter.sub32(0, reg_a, reg_a, operand_b)?,
            BinaryOp::Mul => emitter.mul32(0, reg_a, reg_a, operand_b)?,
            BinaryOp::And => emitter.and32(0, reg_a, reg_a, operand_b)?,
            BinaryOp::Or => emitter.or32(0, reg_a, reg_a, operand_b)?,
            BinaryOp::Xor => emitter.xor32(0, reg_a, reg_a, operand_b)?,
            BinaryOp::Shl => emitter.shl32(0, reg_a, reg_a, operand_b)?,
            BinaryOp::ShrS => emitter.ashr32(0, reg_a, reg_a, operand_b)?,
            BinaryOp::ShrU => emitter.lshr32(0, reg_a, reg_a, operand_b)?,
            BinaryOp::Rotl => emitter.rotl32(0, reg_a, reg_a, operand_b)?,
            BinaryOp::Rotr => emitter.rotr32(0, reg_a, reg_a, operand_b)?,
        };

        if let StackValue::Register(reg_b) = b {
            self.free_register(reg_b);
        }
        self.push(StackValue::Register(reg_a));
        Ok(())
    }

    fn compile_compare_op(
        &mut self,
        emitter: &mut Emitter,
        cond: Condition,
    ) -> Result<(), CompileError> {
        let (b, a) = (self.pop_value()?, self.pop_value()?);
        let reg_a = self.ensure_in_register(emitter, a)?;

        emitter.branch(
            cond,
            reg_a,
            b.to_operand(),
            |e| {
                e.mov(0, reg_a, 1i32)?;
                Ok(())
            },
            |e| {
                e.mov(0, reg_a, 0i32)?;
                Ok(())
            },
        )?;
        if let StackValue::Register(reg_b) = b {
            self.free_register(reg_b);
        }
        self.push(StackValue::Register(reg_a));
        Ok(())
    }

    fn compile_div_op(&mut self, emitter: &mut Emitter, op: DivOp) -> Result<(), CompileError> {
        let (b, a) = (self.pop_value()?, self.pop_value()?);
        let (reg_a, reg_b) = (
            self.ensure_in_register(emitter, a)?,
            self.ensure_in_register(emitter, b)?,
        );

        if reg_a != ScratchRegister::R0 {
            emitter.mov(0, ScratchRegister::R0, reg_a)?;
        }
        if reg_b != ScratchRegister::R1 {
            emitter.mov(0, ScratchRegister::R1, reg_b)?;
        }

        [reg_a, reg_b]
            .iter()
            .filter(|&&r| r != ScratchRegister::R0 && r != ScratchRegister::R1)
            .for_each(|&r| self.free_register(r));

        match op {
            DivOp::DivS | DivOp::RemS => {
                emitter.divmod_s32()?;
            }
            DivOp::DivU | DivOp::RemU => {
                emitter.divmod_u32()?;
            }
        }

        let (result_reg, other_reg) = match op {
            DivOp::DivS | DivOp::DivU => (ScratchRegister::R0, ScratchRegister::R1),
            DivOp::RemS | DivOp::RemU => (ScratchRegister::R1, ScratchRegister::R0),
        };

        self.free_registers.retain(|&r| r != other_reg);
        self.free_registers.insert(other_reg);
        self.free_registers.retain(|&r| r != result_reg);
        self.push(StackValue::Register(result_reg));
        Ok(())
    }

    fn compile_unary_op(&mut self, emitter: &mut Emitter, op: UnaryOp) -> Result<(), CompileError> {
        let val = self.pop_value()?;
        let reg = self.ensure_in_register(emitter, val)?;

        match op {
            UnaryOp::Clz => {
                emitter.clz32(0, reg, reg)?;
            }
            UnaryOp::Ctz => {
                emitter.ctz32(0, reg, reg)?;
            }
            UnaryOp::Popcnt => {
                let temp = self.alloc_register(emitter)?;
                // Parallel bit counting algorithm
                emitter.lshr32(0, temp, reg, 1i32)?;
                emitter.and32(0, temp, temp, 0x55555555i32)?;
                emitter.sub32(0, reg, reg, temp)?;
                emitter.lshr32(0, temp, reg, 2i32)?;
                emitter.and32(0, temp, temp, 0x33333333i32)?;
                emitter.and32(0, reg, reg, 0x33333333i32)?;
                emitter.add32(0, reg, reg, temp)?;
                emitter.lshr32(0, temp, reg, 4i32)?;
                emitter.add32(0, reg, reg, temp)?;
                emitter.and32(0, reg, reg, 0x0f0f0f0fi32)?;
                emitter.lshr32(0, temp, reg, 8i32)?;
                emitter.add32(0, reg, reg, temp)?;
                emitter.lshr32(0, temp, reg, 16i32)?;
                emitter.add32(0, reg, reg, temp)?;
                emitter.and32(0, reg, reg, 0x3fi32)?;
                self.free_register(temp);
            }
        }
        self.push(StackValue::Register(reg));
        Ok(())
    }

    fn compile_load_op(
        &mut self,
        emitter: &mut Emitter,
        offset: i32,
        kind: LoadStoreKind,
    ) -> Result<(), CompileError> {
        let addr = self.pop_value()?;
        let addr_reg = self.ensure_in_register(emitter, addr)?;
        emitter.add(0, addr_reg, addr_reg, SavedRegister::S0)?;

        let mem = mem_offset(addr_reg, offset);

        match kind {
            LoadStoreKind::I32 => {
                emitter.mov32(0, addr_reg, mem)?;
            }
            LoadStoreKind::I8S => {
                emitter.mov_s8(0, addr_reg, mem)?;
            }
            LoadStoreKind::I8U => {
                emitter.mov_u8(0, addr_reg, mem)?;
            }
            LoadStoreKind::I16S => {
                emitter.mov_s16(0, addr_reg, mem)?;
            }
            LoadStoreKind::I16U => {
                emitter.mov_u16(0, addr_reg, mem)?;
            }
            LoadStoreKind::F32 | LoadStoreKind::F64 => {
                let freg = self.alloc_float_register(emitter)?;
                if matches!(kind, LoadStoreKind::F32) {
                    emitter.mov_f32(0, freg, mem)?;
                } else {
                    emitter.mov_f64(0, freg, mem)?;
                }
                self.free_register(addr_reg);
                self.push(StackValue::FloatRegister(freg));
                return Ok(());
            }
            _ => unreachable!("I64 variants handled by compile_load_op64"),
        }
        self.push(StackValue::Register(addr_reg));
        Ok(())
    }

    fn compile_store_op(
        &mut self,
        emitter: &mut Emitter,
        offset: i32,
        kind: LoadStoreKind,
    ) -> Result<(), CompileError> {
        let (value, addr) = (self.pop_value()?, self.pop_value()?);
        let addr_reg = self.ensure_in_register(emitter, addr)?;
        emitter.add(0, addr_reg, addr_reg, SavedRegister::S0)?;

        let mem = mem_offset(addr_reg, offset);

        match kind {
            LoadStoreKind::F32 | LoadStoreKind::F64 => {
                let is_f32 = matches!(kind, LoadStoreKind::F32);
                let freg = self.ensure_in_float_register(emitter, value, is_f32)?;
                if is_f32 {
                    emitter.mov_f32(0, mem, freg)?;
                } else {
                    emitter.mov_f64(0, mem, freg)?;
                }
                self.free_float_register(freg);
            }
            _ => {
                let value_reg = self.ensure_in_register(emitter, value)?;
                match kind {
                    LoadStoreKind::I32 => {
                        emitter.mov32(0, mem, value_reg)?;
                    }
                    LoadStoreKind::I8U | LoadStoreKind::I8S => {
                        emitter.mov_u8(0, mem, value_reg)?;
                    }
                    LoadStoreKind::I16U | LoadStoreKind::I16S => {
                        emitter.mov_u16(0, mem, value_reg)?;
                    }
                    _ => unreachable!("I64 variants handled by compile_store_op64"),
                }
                self.free_register(value_reg);
            }
        }
        self.free_register(addr_reg);
        Ok(())
    }

    #[cfg(target_pointer_width = "64")]
    fn compile_binary_op64(
        &mut self,
        emitter: &mut Emitter,
        op: BinaryOp,
    ) -> Result<(), CompileError> {
        let (b, a) = (self.pop_value()?, self.pop_value()?);
        let reg_a = self.ensure_in_register(emitter, a)?;
        let operand_b = b.to_operand();

        match op {
            BinaryOp::Add => emitter.add(0, reg_a, reg_a, operand_b)?,
            BinaryOp::Sub => emitter.sub(0, reg_a, reg_a, operand_b)?,
            BinaryOp::Mul => emitter.mul(0, reg_a, reg_a, operand_b)?,
            BinaryOp::And => emitter.and(0, reg_a, reg_a, operand_b)?,
            BinaryOp::Or => emitter.or(0, reg_a, reg_a, operand_b)?,
            BinaryOp::Xor => emitter.xor(0, reg_a, reg_a, operand_b)?,
            BinaryOp::Shl => emitter.shl(0, reg_a, reg_a, operand_b)?,
            BinaryOp::ShrS => emitter.ashr(0, reg_a, reg_a, operand_b)?,
            BinaryOp::ShrU => emitter.lshr(0, reg_a, reg_a, operand_b)?,
            BinaryOp::Rotl => emitter.rotl(0, reg_a, reg_a, operand_b)?,
            BinaryOp::Rotr => emitter.rotr(0, reg_a, reg_a, operand_b)?,
        };

        if let StackValue::Register(reg_b) = b {
            self.free_register(reg_b);
        }
        self.push(StackValue::Register(reg_a));
        Ok(())
    }

    #[cfg(not(target_pointer_width = "64"))]
    fn compile_binary_op64(
        &mut self,
        emitter: &mut Emitter,
        op: BinaryOp,
    ) -> Result<(), CompileError> {
        // On 32-bit, implement 64-bit operations using two 32-bit registers
        let (b, a) = (self.pop_value()?, self.pop_value()?);

        // For 32-bit platforms, we need to work primarily with stack-based values
        // to avoid running out of registers (we need 4-6 regs for i64 ops)
        // Get offsets for both operands (convert to stack if not already)
        let offset_a = match a {
            StackValue::Stack(off) => off,
            _ => {
                // Convert to stack-based storage
                let off_a = self.frame_offset;
                self.frame_offset += 8;
                match a {
                    StackValue::Const(val) => {
                        emitter.mov32(0, mem_sp_offset(off_a), val)?;
                        // Sign extend
                        let sign_reg = self.alloc_register(emitter)?;
                        emitter.mov32(0, sign_reg, val)?;
                        emitter.ashr32(0, sign_reg, sign_reg, 31i32)?;
                        emitter.mov32(0, mem_sp_offset(off_a + 4), sign_reg)?;
                        self.free_register(sign_reg);
                    }
                    _ => {
                        let reg_a = self.ensure_in_register(emitter, a)?;
                        emitter.mov32(0, mem_sp_offset(off_a), reg_a)?;
                        // Sign extend for signed values
                        emitter.ashr32(0, reg_a, reg_a, 31i32)?;
                        emitter.mov32(0, mem_sp_offset(off_a + 4), reg_a)?;
                        self.free_register(reg_a);
                    }
                }
                off_a
            }
        };

        let offset_b = match b {
            StackValue::Stack(off) => off,
            _ => {
                // Convert to stack-based storage
                let off_b = self.frame_offset;
                self.frame_offset += 8;
                match b {
                    StackValue::Const(val) => {
                        emitter.mov32(0, mem_sp_offset(off_b), val)?;
                        // Sign extend
                        let sign_reg = self.alloc_register(emitter)?;
                        emitter.mov32(0, sign_reg, val)?;
                        emitter.ashr32(0, sign_reg, sign_reg, 31i32)?;
                        emitter.mov32(0, mem_sp_offset(off_b + 4), sign_reg)?;
                        self.free_register(sign_reg);
                    }
                    _ => {
                        let reg_b = self.ensure_in_register(emitter, b)?;
                        emitter.mov32(0, mem_sp_offset(off_b), reg_b)?;
                        // Sign extend for signed values
                        emitter.ashr32(0, reg_b, reg_b, 31i32)?;
                        emitter.mov32(0, mem_sp_offset(off_b + 4), reg_b)?;
                        self.free_register(reg_b);
                    }
                }
                off_b
            }
        };

        // Now work with registers - but allocate them carefully
        // For shifts, we need up to 6 registers, which might be too many
        // For bitwise/add/sub, we only need 4 registers

        // Load both operands' parts into registers
        let reg_a_low = self.alloc_register(emitter)?;
        let reg_a_high = self.alloc_register(emitter)?;
        let reg_b_low = self.alloc_register(emitter)?;
        let reg_b_high = self.alloc_register(emitter)?;

        emitter.mov32(0, reg_a_low, mem_sp_offset(offset_a))?;
        emitter.mov32(0, reg_a_high, mem_sp_offset(offset_a + 4))?;
        emitter.mov32(0, reg_b_low, mem_sp_offset(offset_b))?;
        emitter.mov32(0, reg_b_high, mem_sp_offset(offset_b + 4))?;

        // Perform the operation
        match op {
            // Bitwise operations - operate on both halves independently
            BinaryOp::And => {
                emitter.and32(0, reg_a_low, reg_a_low, reg_b_low)?;
                emitter.and32(0, reg_a_high, reg_a_high, reg_b_high)?;
                // Free b regs early
                self.free_register(reg_b_low);
                self.free_register(reg_b_high);
            }
            BinaryOp::Or => {
                emitter.or32(0, reg_a_low, reg_a_low, reg_b_low)?;
                emitter.or32(0, reg_a_high, reg_a_high, reg_b_high)?;
                // Free b regs early
                self.free_register(reg_b_low);
                self.free_register(reg_b_high);
            }
            BinaryOp::Xor => {
                emitter.xor32(0, reg_a_low, reg_a_low, reg_b_low)?;
                emitter.xor32(0, reg_a_high, reg_a_high, reg_b_high)?;
                // Free b regs early
                self.free_register(reg_b_low);
                self.free_register(reg_b_high);
            }

            // Addition: add with carry
            BinaryOp::Add => {
                // Add low parts with carry flag
                emitter.add32(0, reg_a_low, reg_a_low, reg_b_low)?;
                // Add high parts with carry
                emitter.addc32(0, reg_a_high, reg_a_high, reg_b_high)?;
                // Free b regs early
                self.free_register(reg_b_low);
                self.free_register(reg_b_high);
            }

            // Subtraction: sub with borrow
            BinaryOp::Sub => {
                // Sub low parts with borrow flag
                emitter.sub32(0, reg_a_low, reg_a_low, reg_b_low)?;
                // Sub high parts with borrow
                emitter.subc32(0, reg_a_high, reg_a_high, reg_b_high)?;
                // Free b regs early
                self.free_register(reg_b_low);
                self.free_register(reg_b_high);
            }

            // Shift left: shifts across the 64-bit boundary
            BinaryOp::Shl => {
                // Free b regs since we only need the shift amount
                self.free_register(reg_b_high);

                // Get shift amount (only low 6 bits matter for 64-bit shift)
                let shift_amount = reg_b_low; // Reuse instead of allocating
                emitter.and32(0, shift_amount, shift_amount, 63i32)?;

                // Check if shift >= 32
                let temp = reg_b_high; // Reuse the high register we just freed
                emitter.mov32(0, temp, shift_amount)?;
                emitter.sub32(0, temp, temp, 32i32)?;

                // if shift >= 32: high = low << (shift-32), low = 0
                // else: high = (high << shift) | (low >> (32-shift)), low = low << shift
                let mut jump_small = emitter.cmp(Condition::SigLess32, temp, 0i32)?;

                // shift >= 32 case
                emitter.shl32(0, reg_a_high, reg_a_low, temp)?;
                emitter.mov32(0, reg_a_low, 0i32)?;
                let mut jump_end = emitter.jump(JumpType::Jump)?;

                // shift < 32 case
                let mut label_small = emitter.put_label()?;
                jump_small.set_label(&mut label_small);

                // Save a copy of low on stack to avoid another register
                let temp_offset = self.frame_offset;
                self.frame_offset += 4;
                emitter.mov32(0, mem_sp_offset(temp_offset), reg_a_low)?;

                // high = high << shift
                emitter.shl32(0, reg_a_high, reg_a_high, shift_amount)?;

                // temp = 32 - shift
                emitter.mov32(0, temp, 32i32)?;
                emitter.sub32(0, temp, temp, shift_amount)?;

                // temp = saved_low >> (32 - shift)
                let saved_low = mem_sp_offset(temp_offset);
                emitter.lshr32(0, temp, saved_low, temp)?;

                // high = high | temp
                emitter.or32(0, reg_a_high, reg_a_high, temp)?;

                // low = low << shift
                emitter.shl32(0, reg_a_low, saved_low, shift_amount)?;

                let mut label_end = emitter.put_label()?;
                jump_end.set_label(&mut label_end);

                // Don't free: we reused already
            }

            // Logical shift right
            BinaryOp::ShrU => {
                // Free b_high since we only need shift amount
                self.free_register(reg_b_high);

                // Get shift amount (only low 6 bits matter for 64-bit shift)
                let shift_amount = reg_b_low; // Reuse
                emitter.and32(0, shift_amount, shift_amount, 63i32)?;

                // Check if shift >= 32
                let temp = reg_b_high; // Reuse
                emitter.mov32(0, temp, shift_amount)?;
                emitter.sub32(0, temp, temp, 32i32)?;

                // if shift >= 32: low = high >> (shift-32), high = 0
                // else: low = (low >> shift) | (high << (32-shift)), high = high >> shift
                let mut jump_small = emitter.cmp(Condition::SigLess32, temp, 0i32)?;

                // shift >= 32 case
                emitter.lshr32(0, reg_a_low, reg_a_high, temp)?;
                emitter.mov32(0, reg_a_high, 0i32)?;
                let mut jump_end = emitter.jump(JumpType::Jump)?;

                // shift < 32 case
                let mut label_small = emitter.put_label()?;
                jump_small.set_label(&mut label_small);

                // Save high on stack to avoid another register
                let temp_offset = self.frame_offset;
                self.frame_offset += 4;
                emitter.mov32(0, mem_sp_offset(temp_offset), reg_a_high)?;

                // low = low >> shift
                emitter.lshr32(0, reg_a_low, reg_a_low, shift_amount)?;

                // temp = 32 - shift
                emitter.mov32(0, temp, 32i32)?;
                emitter.sub32(0, temp, temp, shift_amount)?;

                // temp = high << (32 - shift)
                let saved_high = mem_sp_offset(temp_offset);
                emitter.shl32(0, temp, saved_high, temp)?;

                // low = low | temp
                emitter.or32(0, reg_a_low, reg_a_low, temp)?;

                // high = high >> shift
                emitter.lshr32(0, reg_a_high, saved_high, shift_amount)?;

                let mut label_end = emitter.put_label()?;
                jump_end.set_label(&mut label_end);

                // Don't free: we reused already
            }

            // Arithmetic shift right
            BinaryOp::ShrS => {
                // Free b_high since we only need shift amount
                self.free_register(reg_b_high);

                // Get shift amount (only low 6 bits matter for 64-bit shift)
                let shift_amount = reg_b_low; // Reuse
                emitter.and32(0, shift_amount, shift_amount, 63i32)?;

                // Check if shift >= 32
                let temp = reg_b_high; // Reuse
                emitter.mov32(0, temp, shift_amount)?;
                emitter.sub32(0, temp, temp, 32i32)?;

                // if shift >= 32: low = high >> (shift-32), high = high >> 31 (sign extend)
                // else: low = (low >> shift) | (high << (32-shift)), high = high >> shift
                let mut jump_small = emitter.cmp(Condition::SigLess32, temp, 0i32)?;

                // shift >= 32 case
                emitter.ashr32(0, reg_a_low, reg_a_high, temp)?;
                emitter.ashr32(0, reg_a_high, reg_a_high, 31i32)?; // Sign extend
                let mut jump_end = emitter.jump(JumpType::Jump)?;

                // shift < 32 case
                let mut label_small = emitter.put_label()?;
                jump_small.set_label(&mut label_small);

                // Save high on stack to avoid another register
                let temp_offset = self.frame_offset;
                self.frame_offset += 4;
                emitter.mov32(0, mem_sp_offset(temp_offset), reg_a_high)?;

                // low = low >> shift (logical)
                emitter.lshr32(0, reg_a_low, reg_a_low, shift_amount)?;

                // temp = 32 - shift
                emitter.mov32(0, temp, 32i32)?;
                emitter.sub32(0, temp, temp, shift_amount)?;

                // temp = high << (32 - shift)
                let saved_high = mem_sp_offset(temp_offset);
                emitter.shl32(0, temp, saved_high, temp)?;

                // low = low | temp
                emitter.or32(0, reg_a_low, reg_a_low, temp)?;

                // high = high >> shift (arithmetic)
                emitter.ashr32(0, reg_a_high, saved_high, shift_amount)?;

                let mut label_end = emitter.put_label()?;
                jump_end.set_label(&mut label_end);

                // Don't free: we reused already
            }

            // Rotation operations: rotl and rotr
            BinaryOp::Rotl => {
                // Free b_high since we only need the rotation amount (bottom 6 bits of b_low)
                self.free_register(reg_b_high);

                // Get rotation amount (only low 6 bits matter for 64-bit rotation)
                let rot_amount = reg_b_low;
                emitter.and32(0, rot_amount, rot_amount, 63i32)?;

                // rotl(x, n) = (x << n) | (x >> (64 - n))
                // But we need to handle 32-bit boundaries

                // Allocate temp registers
                let temp1 = self.alloc_register(emitter)?;
                let temp2 = self.alloc_register(emitter)?;

                // Save original values
                let temp_offset = self.frame_offset;
                self.frame_offset += 8;
                emitter.mov32(0, mem_sp_offset(temp_offset), reg_a_low)?;
                emitter.mov32(0, mem_sp_offset(temp_offset + 4), reg_a_high)?;

                // Compute 64 - rot_amount
                emitter.mov32(0, temp1, 64i32)?;
                emitter.sub32(0, temp1, temp1, rot_amount)?;

                // Check if rot_amount is 0 (special case: no rotation needed)
                let mut jump_zero = emitter.cmp(Condition::Equal, rot_amount, 0i32)?;

                // Check if rot_amount < 32
                emitter.mov32(0, temp2, rot_amount)?;
                emitter.sub32(0, temp2, temp2, 32i32)?;
                let mut jump_large = emitter.cmp(Condition::SigGreaterEqual32, temp2, 0i32)?;

                // rot_amount < 32 case
                // high = (high << rot) | (low >> (32 - rot))
                // low = (low << rot) | (high >> (32 - rot))

                // Save low for later
                emitter.mov32(0, temp1, mem_sp_offset(temp_offset))?;

                // low << rot
                emitter.shl32(0, reg_a_low, mem_sp_offset(temp_offset), rot_amount)?;

                // 32 - rot
                emitter.mov32(0, temp2, 32i32)?;
                emitter.sub32(0, temp2, temp2, rot_amount)?;

                // high >> (32 - rot)
                emitter.lshr32(0, temp2, mem_sp_offset(temp_offset + 4), temp2)?;

                // low = (low << rot) | (high >> (32 - rot))
                emitter.or32(0, reg_a_low, reg_a_low, temp2)?;

                // high << rot
                emitter.shl32(0, reg_a_high, mem_sp_offset(temp_offset + 4), rot_amount)?;

                // 32 - rot
                emitter.mov32(0, temp2, 32i32)?;
                emitter.sub32(0, temp2, temp2, rot_amount)?;

                // saved_low >> (32 - rot)
                emitter.lshr32(0, temp2, temp1, temp2)?;

                // high = (high << rot) | (low >> (32 - rot))
                emitter.or32(0, reg_a_high, reg_a_high, temp2)?;

                let mut jump_end = emitter.jump(JumpType::Jump)?;

                // rot_amount >= 32 case
                let mut label_large = emitter.put_label()?;
                jump_large.set_label(&mut label_large);

                // For rotl when amount >= 32, we essentially rotate by (amount - 32)
                // but with high and low swapped
                // high_new = (low << (rot-32)) | (high >> (64-rot))
                // low_new = (high << (rot-32)) | (low >> (64-rot))

                emitter.mov32(0, temp2, rot_amount)?;
                emitter.sub32(0, temp2, temp2, 32i32)?;

                // low_new = (high << (rot-32))
                emitter.shl32(0, reg_a_low, mem_sp_offset(temp_offset + 4), temp2)?;

                // 64 - rot = 32 - (rot - 32)
                emitter.mov32(0, temp1, 32i32)?;
                emitter.sub32(0, temp1, temp1, temp2)?;

                // low >> (64 - rot)
                emitter.lshr32(0, temp1, mem_sp_offset(temp_offset), temp1)?;
                emitter.or32(0, reg_a_low, reg_a_low, temp1)?;

                // high_new = (low << (rot-32))
                emitter.shl32(0, reg_a_high, mem_sp_offset(temp_offset), temp2)?;

                // high >> (64 - rot)
                emitter.mov32(0, temp1, 32i32)?;
                emitter.sub32(0, temp1, temp1, temp2)?;
                emitter.lshr32(0, temp1, mem_sp_offset(temp_offset + 4), temp1)?;
                emitter.or32(0, reg_a_high, reg_a_high, temp1)?;

                let mut jump_end2 = emitter.jump(JumpType::Jump)?;

                // rot_amount == 0 case (no rotation)
                let mut label_zero = emitter.put_label()?;
                jump_zero.set_label(&mut label_zero);
                emitter.mov32(0, reg_a_low, mem_sp_offset(temp_offset))?;
                emitter.mov32(0, reg_a_high, mem_sp_offset(temp_offset + 4))?;

                let mut label_end = emitter.put_label()?;
                jump_end.set_label(&mut label_end);
                jump_end2.set_label(&mut label_end);

                self.free_register(temp1);
                self.free_register(temp2);
            }

            BinaryOp::Rotr => {
                // Free b_high since we only need the rotation amount
                self.free_register(reg_b_high);

                // Get rotation amount (only low 6 bits matter for 64-bit rotation)
                let rot_amount = reg_b_low;
                emitter.and32(0, rot_amount, rot_amount, 63i32)?;

                // rotr(x, n) = (x >> n) | (x << (64 - n))

                // Allocate temp registers
                let temp1 = self.alloc_register(emitter)?;
                let temp2 = self.alloc_register(emitter)?;

                // Save original values
                let temp_offset = self.frame_offset;
                self.frame_offset += 8;
                emitter.mov32(0, mem_sp_offset(temp_offset), reg_a_low)?;
                emitter.mov32(0, mem_sp_offset(temp_offset + 4), reg_a_high)?;

                // Check if rot_amount is 0 (special case: no rotation needed)
                let mut jump_zero = emitter.cmp(Condition::Equal, rot_amount, 0i32)?;

                // Check if rot_amount < 32
                emitter.mov32(0, temp2, rot_amount)?;
                emitter.sub32(0, temp2, temp2, 32i32)?;
                let mut jump_large = emitter.cmp(Condition::SigGreaterEqual32, temp2, 0i32)?;

                // rot_amount < 32 case
                // low = (low >> rot) | (high << (32 - rot))
                // high = (high >> rot) | (low << (32 - rot))

                // Save high for later
                emitter.mov32(0, temp1, mem_sp_offset(temp_offset + 4))?;

                // low >> rot
                emitter.lshr32(0, reg_a_low, mem_sp_offset(temp_offset), rot_amount)?;

                // 32 - rot
                emitter.mov32(0, temp2, 32i32)?;
                emitter.sub32(0, temp2, temp2, rot_amount)?;

                // high << (32 - rot)
                emitter.shl32(0, temp2, temp1, temp2)?;

                // low = (low >> rot) | (high << (32 - rot))
                emitter.or32(0, reg_a_low, reg_a_low, temp2)?;

                // high >> rot
                emitter.lshr32(0, reg_a_high, temp1, rot_amount)?;

                // 32 - rot
                emitter.mov32(0, temp2, 32i32)?;
                emitter.sub32(0, temp2, temp2, rot_amount)?;

                // saved_low << (32 - rot)
                emitter.shl32(0, temp2, mem_sp_offset(temp_offset), temp2)?;

                // high = (high >> rot) | (low << (32 - rot))
                emitter.or32(0, reg_a_high, reg_a_high, temp2)?;

                let mut jump_end = emitter.jump(JumpType::Jump)?;

                // rot_amount >= 32 case
                let mut label_large = emitter.put_label()?;
                jump_large.set_label(&mut label_large);

                // For rotr when amount >= 32, we rotate by (amount - 32)
                // with high and low swapped

                emitter.mov32(0, temp2, rot_amount)?;
                emitter.sub32(0, temp2, temp2, 32i32)?;

                // low_new = (high >> (rot-32))
                emitter.lshr32(0, reg_a_low, mem_sp_offset(temp_offset + 4), temp2)?;

                // 32 - (rot - 32) = 64 - rot
                emitter.mov32(0, temp1, 32i32)?;
                emitter.sub32(0, temp1, temp1, temp2)?;

                // low << (64 - rot)
                emitter.shl32(0, temp1, mem_sp_offset(temp_offset), temp1)?;
                emitter.or32(0, reg_a_low, reg_a_low, temp1)?;

                // high_new = (low >> (rot-32))
                emitter.lshr32(0, reg_a_high, mem_sp_offset(temp_offset), temp2)?;

                // high << (64 - rot)
                emitter.mov32(0, temp1, 32i32)?;
                emitter.sub32(0, temp1, temp1, temp2)?;
                emitter.shl32(0, temp1, mem_sp_offset(temp_offset + 4), temp1)?;
                emitter.or32(0, reg_a_high, reg_a_high, temp1)?;

                let mut jump_end2 = emitter.jump(JumpType::Jump)?;

                // rot_amount == 0 case (no rotation)
                let mut label_zero = emitter.put_label()?;
                jump_zero.set_label(&mut label_zero);
                emitter.mov32(0, reg_a_low, mem_sp_offset(temp_offset))?;
                emitter.mov32(0, reg_a_high, mem_sp_offset(temp_offset + 4))?;

                let mut label_end = emitter.put_label()?;
                jump_end.set_label(&mut label_end);
                jump_end2.set_label(&mut label_end);

                self.free_register(temp1);
                self.free_register(temp2);
            }

            // 64-bit multiplication
            BinaryOp::Mul => {
                // 64-bit multiplication on 32-bit: (a_high:a_low) * (b_high:b_low)
                // Result = a_low * b_low (gives low 64 bits)
                //        + (a_low * b_high) << 32
                //        + (a_high * b_low) << 32
                // We ignore (a_high * b_high) << 64 as it would overflow past 64 bits

                // Allocate temp registers for intermediate results
                let temp1 = self.alloc_register(emitter)?;
                let temp2 = self.alloc_register(emitter)?;

                // Save a_low for later use
                let saved_a_low_offset = self.frame_offset;
                self.frame_offset += 4;
                emitter.mov32(0, mem_sp_offset(saved_a_low_offset), reg_a_low)?;

                // temp1 = a_low * b_high (lower 32 bits of the result)
                emitter.mul32(0, temp1, reg_a_low, reg_b_high)?;

                // temp2 = a_high * b_low (lower 32 bits of the result)
                emitter.mul32(0, temp2, reg_a_high, reg_b_low)?;

                // temp1 = (a_low * b_high) + (a_high * b_low)
                emitter.add32(0, temp1, temp1, temp2)?;

                // Now compute a_low * b_low which gives us a full 64-bit result
                // We need to use R0 and R1 for the multiply
                let a_low_saved = reg_a_low;
                let b_low_saved = reg_b_low;

                // Move operands to R0 and R1 if needed
                if a_low_saved != ScratchRegister::R0 {
                    emitter.mov32(0, ScratchRegister::R0, a_low_saved)?;
                }
                if b_low_saved != ScratchRegister::R1 {
                    emitter.mov32(0, ScratchRegister::R1, b_low_saved)?;
                }

                // Full 32x32->64 multiply: R0:R1 = R0 * R1
                emitter.divmod_u32()?; // Just kidding, we need actual multiply
                // SLJIT doesn't have a 32x32->64 multiply instruction directly
                // We'll use the formula: a*b low bits = (a*b) mod 2^32
                // For the high bits: we use the cross products

                // Actually, let's do this simpler:
                // R0:R1 after mul gives low 32 bits in result register
                emitter.mul32(
                    0,
                    ScratchRegister::R0,
                    ScratchRegister::R0,
                    ScratchRegister::R1,
                )?;

                // Now R0 has the lower 32 bits of a_low * b_low
                // temp1 has (a_low * b_high) + (a_high * b_low)
                // high bits = temp1 + carry from (a_low * b_low)

                // For a proper 32x32->64 multiply, we need to compute the high bits
                // high = ((a_low & 0xFFFF) * (b_low >> 16) +
                //         (a_low >> 16) * (b_low & 0xFFFF) +
                //         ((a_low & 0xFFFF) * (b_low & 0xFFFF)) >> 16) >> 16
                //      + (a_low >> 16) * (b_low >> 16)

                // This is getting complex. Let's use a simpler approach:
                // For 64-bit mul on 32-bit, we compute:
                // result_low = a_low * b_low (lower 32 bits)
                // result_high = (a_low * b_high) + (a_high * b_low) + high_bits_of(a_low * b_low)

                // To get high bits of a 32x32 multiply, we need to split into 16-bit parts:
                // Let a_low = a1 * 2^16 + a0, b_low = b1 * 2^16 + b0
                // a_low * b_low = a1*b1*2^32 + (a1*b0 + a0*b1)*2^16 + a0*b0
                // high_bits = a1*b1 + ((a1*b0 + a0*b1 + (a0*b0 >> 16)) >> 16)

                // Load back a_low
                emitter.mov32(0, temp2, mem_sp_offset(saved_a_low_offset))?;

                // Compute high word of 32x32 multiply using bit manipulation
                // We'll use the fact that for unsigned multiply:
                // high = (a >> 1) * b + ((a & 1) * b) + ...
                // This is too complex, let's just compute it properly

                // Split a_low into 16-bit parts
                emitter.mov32(0, reg_a_high, temp2)?; // a_low
                emitter.lshr32(0, reg_a_high, reg_a_high, 16i32)?; // a1 = a_low >> 16
                emitter.mov32(0, reg_a_low, temp2)?;
                emitter.and32(0, reg_a_low, reg_a_low, 0xFFFFi32)?; // a0 = a_low & 0xFFFF

                // Split b_low into 16-bit parts
                emitter.mov32(0, temp2, reg_b_low)?;
                emitter.lshr32(0, temp2, temp2, 16i32)?; // b1 = b_low >> 16

                let saved_b_low_offset = self.frame_offset;
                self.frame_offset += 4;
                emitter.mov32(0, mem_sp_offset(saved_b_low_offset), reg_b_low)?;

                emitter.and32(0, reg_b_low, reg_b_low, 0xFFFFi32)?; // b0 = b_low & 0xFFFF

                // Now: reg_a_low = a0, reg_a_high = a1, reg_b_low = b0, temp2 = b1

                // Compute a0 * b0
                emitter.mul32(0, reg_b_high, reg_a_low, reg_b_low)?; // a0 * b0

                // Low result = a0 * b0 (lower 32 bits) - this is in reg_b_high lower bits
                // Save for final result
                emitter.mov32(0, mem_sp_offset(saved_a_low_offset), reg_b_high)?;

                // High part starts with (a0 * b0) >> 16
                emitter.lshr32(0, reg_b_high, reg_b_high, 16i32)?;

                // Add a0 * b1
                emitter.mul32(0, reg_b_low, reg_a_low, temp2)?;
                emitter.add32(0, reg_b_high, reg_b_high, reg_b_low)?;

                // Add a1 * b0
                let b0_val = mem_sp_offset(saved_b_low_offset);
                emitter.mov32(0, reg_b_low, b0_val)?;
                emitter.and32(0, reg_b_low, reg_b_low, 0xFFFFi32)?;
                emitter.mul32(0, reg_a_low, reg_a_high, reg_b_low)?;
                emitter.add32(0, reg_b_high, reg_b_high, reg_a_low)?;

                // High part of middle sum
                emitter.lshr32(0, reg_b_high, reg_b_high, 16i32)?;

                // Add a1 * b1
                emitter.mul32(0, reg_a_low, reg_a_high, temp2)?;
                emitter.add32(0, reg_b_high, reg_b_high, reg_a_low)?;

                // Add the temp1 parts (a_low * b_high + a_high * b_low)
                emitter.add32(0, reg_a_high, reg_b_high, temp1)?;

                // Load final low result
                emitter.mov32(0, reg_a_low, mem_sp_offset(saved_a_low_offset))?;

                self.free_register(temp1);
                self.free_register(temp2);
                self.free_register(reg_b_low);
                self.free_register(reg_b_high);
            }
        }

        // Store result back to stack
        let result_offset = self.frame_offset;
        self.frame_offset += 8;
        emitter.mov32(0, mem_sp_offset(result_offset), reg_a_low)?;
        emitter.mov32(0, mem_sp_offset(result_offset + 4), reg_a_high)?;

        self.free_register(reg_a_low);
        self.free_register(reg_a_high);

        self.push(StackValue::Stack(result_offset));
        Ok(())
    }

    fn compile_compare_op64(
        &mut self,
        emitter: &mut Emitter,
        cond: Condition,
    ) -> Result<(), CompileError> {
        let (b, a) = (self.pop_value()?, self.pop_value()?);
        let reg_a = self.ensure_in_register(emitter, a)?;

        emitter.branch(
            cond,
            reg_a,
            b.to_operand(),
            |e| {
                e.mov(0, reg_a, 1i32)?;
                Ok(())
            },
            |e| {
                e.mov(0, reg_a, 0i32)?;
                Ok(())
            },
        )?;
        if let StackValue::Register(reg_b) = b {
            self.free_register(reg_b);
        }
        self.push(StackValue::Register(reg_a));
        Ok(())
    }

    fn compile_div_op64(&mut self, emitter: &mut Emitter, op: DivOp) -> Result<(), CompileError> {
        let (b, a) = (self.pop_value()?, self.pop_value()?);
        let (reg_a, reg_b) = (
            self.ensure_in_register(emitter, a)?,
            self.ensure_in_register(emitter, b)?,
        );

        if reg_a != ScratchRegister::R0 {
            emitter.mov(0, ScratchRegister::R0, reg_a)?;
        }
        if reg_b != ScratchRegister::R1 {
            emitter.mov(0, ScratchRegister::R1, reg_b)?;
        }

        [reg_a, reg_b]
            .iter()
            .filter(|&&r| r != ScratchRegister::R0 && r != ScratchRegister::R1)
            .for_each(|&r| self.free_register(r));

        match op {
            DivOp::DivS | DivOp::RemS => {
                emitter.divmod_sword()?;
            }
            DivOp::DivU | DivOp::RemU => {
                emitter.divmod_uword()?;
            }
        }

        let (result_reg, other_reg) = match op {
            DivOp::DivS | DivOp::DivU => (ScratchRegister::R0, ScratchRegister::R1),
            DivOp::RemS | DivOp::RemU => (ScratchRegister::R1, ScratchRegister::R0),
        };

        self.free_registers.retain(|&r| r != other_reg);
        self.free_registers.insert(other_reg);
        self.free_registers.retain(|&r| r != result_reg);
        self.push(StackValue::Register(result_reg));
        Ok(())
    }

    fn compile_unary_op64(
        &mut self,
        emitter: &mut Emitter,
        op: UnaryOp,
    ) -> Result<(), CompileError> {
        let val = self.pop_value()?;
        let reg = self.ensure_in_register(emitter, val)?;

        match op {
            UnaryOp::Clz => {
                emitter.clz(0, reg, reg)?;
            }
            UnaryOp::Ctz => {
                emitter.ctz(0, reg, reg)?;
            }
            UnaryOp::Popcnt => {
                let temp = self.alloc_register(emitter)?;
                // Parallel bit counting algorithm for 64-bit
                emitter.lshr(0, temp, reg, 1i32)?;
                emitter.and(0, temp, temp, 0x5555555555555555u64 as isize)?;
                emitter.sub(0, reg, reg, temp)?;
                emitter.lshr(0, temp, reg, 2i32)?;
                emitter.and(0, temp, temp, 0x3333333333333333u64 as isize)?;
                emitter.and(0, reg, reg, 0x3333333333333333u64 as isize)?;
                emitter.add(0, reg, reg, temp)?;
                emitter.lshr(0, temp, reg, 4i32)?;
                emitter.add(0, reg, reg, temp)?;
                emitter.and(0, reg, reg, 0x0f0f0f0f0f0f0f0fu64 as isize)?;
                emitter.lshr(0, temp, reg, 8i32)?;
                emitter.add(0, reg, reg, temp)?;
                emitter.lshr(0, temp, reg, 16i32)?;
                emitter.add(0, reg, reg, temp)?;
                emitter.lshr(0, temp, reg, 32i32)?;
                emitter.add(0, reg, reg, temp)?;
                emitter.and(0, reg, reg, 0x7fi32)?;
                self.free_register(temp);
            }
        }
        self.push(StackValue::Register(reg));
        Ok(())
    }

    fn compile_load_op64(
        &mut self,
        emitter: &mut Emitter,
        offset: i32,
        kind: LoadStoreKind,
    ) -> Result<(), CompileError> {
        let addr = self.pop_value()?;
        let addr_reg = self.ensure_in_register(emitter, addr)?;
        emitter.add(0, addr_reg, addr_reg, SavedRegister::S0)?;
        let mem = mem_offset(addr_reg, offset);

        #[cfg(target_pointer_width = "64")]
        {
            match kind {
                LoadStoreKind::I64 => {
                    emitter.mov(0, addr_reg, mem)?;
                }
                LoadStoreKind::I64_8S => {
                    emitter.mov_s8(0, addr_reg, mem)?;
                }
                LoadStoreKind::I64_8U => {
                    emitter.mov_u8(0, addr_reg, mem)?;
                }
                LoadStoreKind::I64_16S => {
                    emitter.mov_s16(0, addr_reg, mem)?;
                }
                LoadStoreKind::I64_16U => {
                    emitter.mov_u16(0, addr_reg, mem)?;
                }
                LoadStoreKind::I64_32S => {
                    emitter.mov_s32(0, addr_reg, mem)?;
                }
                LoadStoreKind::I64_32U => {
                    emitter.mov_u32(0, addr_reg, mem)?;
                }
                _ => unreachable!(),
            }
            self.push(StackValue::Register(addr_reg));
        }

        #[cfg(not(target_pointer_width = "64"))]
        {
            // On 32-bit: load 64-bit value as two 32-bit halves to stack
            let result_offset = self.frame_offset;
            self.frame_offset += 8;

            // Allocate a temp register to preserve address
            let temp_reg = self.alloc_register(emitter)?;

            match kind {
                LoadStoreKind::I64 => {
                    // Load lower 32 bits
                    emitter.mov32(0, temp_reg, mem_offset(addr_reg, offset))?;
                    emitter.mov32(0, mem_sp_offset(result_offset), temp_reg)?;

                    // Load upper 32 bits
                    emitter.mov32(0, temp_reg, mem_offset(addr_reg, offset + 4))?;
                    emitter.mov32(0, mem_sp_offset(result_offset + 4), temp_reg)?;
                }
                LoadStoreKind::I64_8S => {
                    let mem_op = mem_offset(addr_reg, offset);
                    emitter.mov_s8(0, temp_reg, mem_op)?;
                    emitter.mov32(0, mem_sp_offset(result_offset), temp_reg)?;
                    // Sign extend to high part
                    emitter.ashr32(0, temp_reg, temp_reg, 31i32)?;
                    emitter.mov32(0, mem_sp_offset(result_offset + 4), temp_reg)?;
                }
                LoadStoreKind::I64_8U => {
                    let mem_op = mem_offset(addr_reg, offset);
                    emitter.mov_u8(0, temp_reg, mem_op)?;
                    emitter.mov32(0, mem_sp_offset(result_offset), temp_reg)?;
                    // Zero extend to high part
                    emitter.mov32(0, mem_sp_offset(result_offset + 4), 0i32)?;
                }
                LoadStoreKind::I64_16S => {
                    let mem_op = mem_offset(addr_reg, offset);
                    emitter.mov_s16(0, temp_reg, mem_op)?;
                    emitter.mov32(0, mem_sp_offset(result_offset), temp_reg)?;
                    // Sign extend to high part
                    emitter.ashr32(0, temp_reg, temp_reg, 31i32)?;
                    emitter.mov32(0, mem_sp_offset(result_offset + 4), temp_reg)?;
                }
                LoadStoreKind::I64_16U => {
                    let mem_op = mem_offset(addr_reg, offset);
                    emitter.mov_u16(0, temp_reg, mem_op)?;
                    emitter.mov32(0, mem_sp_offset(result_offset), temp_reg)?;
                    // Zero extend to high part
                    emitter.mov32(0, mem_sp_offset(result_offset + 4), 0i32)?;
                }
                LoadStoreKind::I64_32S => {
                    // Load 32-bit value
                    let mem_op = mem_offset(addr_reg, offset);
                    emitter.mov32(0, temp_reg, mem_op)?;
                    emitter.mov32(0, mem_sp_offset(result_offset), temp_reg)?;
                    // Sign extend to high part: copy the sign bit
                    emitter.ashr32(0, temp_reg, temp_reg, 31i32)?;
                    emitter.mov32(0, mem_sp_offset(result_offset + 4), temp_reg)?;
                }
                LoadStoreKind::I64_32U => {
                    let mem_op = mem_offset(addr_reg, offset);
                    emitter.mov32(0, temp_reg, mem_op)?;
                    emitter.mov32(0, mem_sp_offset(result_offset), temp_reg)?;
                    // Zero extend to high part
                    emitter.mov32(0, mem_sp_offset(result_offset + 4), 0i32)?;
                }
                _ => unreachable!(),
            }

            self.free_register(temp_reg);
            self.free_register(addr_reg);
            self.push(StackValue::Stack(result_offset));
        }

        Ok(())
    }

    fn compile_store_op64(
        &mut self,
        emitter: &mut Emitter,
        offset: i32,
        kind: LoadStoreKind,
    ) -> Result<(), CompileError> {
        // Pop value first, but don't pop addr yet to avoid register reuse issues
        let value = self.pop_value()?;
        let addr = self.pop_value()?;

        // Ensure value is in a register BEFORE processing addr to avoid conflicts
        #[cfg(target_pointer_width = "64")]
        let value_reg = self.ensure_in_register(emitter, value)?;

        let addr_reg = self.ensure_in_register(emitter, addr)?;
        emitter.add(0, addr_reg, addr_reg, SavedRegister::S0)?;
        let mem = mem_offset(addr_reg, offset);

        #[cfg(target_pointer_width = "64")]
        {
            match kind {
                LoadStoreKind::I64 => {
                    emitter.mov(0, mem, value_reg)?;
                }
                LoadStoreKind::I64_8U => {
                    emitter.mov_u8(0, mem, value_reg)?;
                }
                LoadStoreKind::I64_16U => {
                    emitter.mov_u16(0, mem, value_reg)?;
                }
                LoadStoreKind::I64_32U => {
                    emitter.mov32(0, mem, value_reg)?;
                }
                _ => unreachable!(),
            }
            self.free_register(value_reg);
            self.free_register(addr_reg);
        }

        #[cfg(not(target_pointer_width = "64"))]
        {
            // On 32-bit: store 64-bit value from stack (two 32-bit halves)
            let temp_reg = self.alloc_register(emitter)?;

            match value {
                StackValue::Stack(value_offset) => {
                    // Value is on stack as 8 bytes
                    match kind {
                        LoadStoreKind::I64 => {
                            // Store lower 32 bits
                            emitter.mov32(0, temp_reg, mem_sp_offset(value_offset))?;
                            emitter.mov32(0, mem_offset(addr_reg, offset), temp_reg)?;

                            // Store upper 32 bits
                            emitter.mov32(0, temp_reg, mem_sp_offset(value_offset + 4))?;
                            emitter.mov32(0, mem_offset(addr_reg, offset + 4), temp_reg)?;
                        }
                        LoadStoreKind::I64_8U => {
                            // Store lower 8 bits only
                            emitter.mov32(0, temp_reg, mem_sp_offset(value_offset))?;
                            emitter.mov_u8(0, mem_offset(addr_reg, offset), temp_reg)?;
                        }
                        LoadStoreKind::I64_16U => {
                            // Store lower 16 bits only
                            emitter.mov32(0, temp_reg, mem_sp_offset(value_offset))?;
                            emitter.mov_u16(0, mem_offset(addr_reg, offset), temp_reg)?;
                        }
                        LoadStoreKind::I64_32U => {
                            // Store lower 32 bits only
                            emitter.mov32(0, temp_reg, mem_sp_offset(value_offset))?;
                            emitter.mov32(0, mem_offset(addr_reg, offset), temp_reg)?;
                        }
                        _ => unreachable!(),
                    }
                }
                _ => {
                    //Value is in register (probably from an i32 extend operation)
                    let value_reg = self.ensure_in_register(emitter, value)?;

                    match kind {
                        LoadStoreKind::I64_8U => {
                            emitter.mov_u8(0, mem_offset(addr_reg, offset), value_reg)?;
                        }
                        LoadStoreKind::I64_16U => {
                            emitter.mov_u16(0, mem_offset(addr_reg, offset), value_reg)?;
                        }
                        LoadStoreKind::I64_32U => {
                            emitter.mov32(0, mem_offset(addr_reg, offset), value_reg)?;
                        }
                        _ => {
                            self.free_register(value_reg);
                            self.free_register(temp_reg);
                            self.free_register(addr_reg);
                            return Err(CompileError::Invalid(
                                "64-bit store from register not supported on 32-bit platform"
                                    .into(),
                            ));
                        }
                    }
                    self.free_register(value_reg);
                }
            }

            self.free_register(temp_reg);
            self.free_register(addr_reg);
        }

        Ok(())
    }

    /// Compile a floating point binary operation
    fn compile_float_binary_op(
        &mut self,
        emitter: &mut Emitter,
        op: FloatBinaryOp,
        is_f32: bool,
    ) -> Result<(), CompileError> {
        let (b, a) = (self.pop_value()?, self.pop_value()?);
        let freg_a = self.ensure_in_float_register(emitter, a, is_f32)?;
        let freg_b = self.ensure_in_float_register(emitter, b, is_f32)?;

        match op {
            FloatBinaryOp::Add if is_f32 => {
                emitter.add_f32(0, freg_a, freg_a, freg_b)?;
            }
            FloatBinaryOp::Add => {
                emitter.add_f64(0, freg_a, freg_a, freg_b)?;
            }
            FloatBinaryOp::Sub if is_f32 => {
                emitter.sub_f32(0, freg_a, freg_a, freg_b)?;
            }
            FloatBinaryOp::Sub => {
                emitter.sub_f64(0, freg_a, freg_a, freg_b)?;
            }
            FloatBinaryOp::Mul if is_f32 => {
                emitter.mul_f32(0, freg_a, freg_a, freg_b)?;
            }
            FloatBinaryOp::Mul => {
                emitter.mul_f64(0, freg_a, freg_a, freg_b)?;
            }
            FloatBinaryOp::Div if is_f32 => {
                emitter.div_f32(0, freg_a, freg_a, freg_b)?;
            }
            FloatBinaryOp::Div => {
                emitter.div_f64(0, freg_a, freg_a, freg_b)?;
            }
            FloatBinaryOp::Min | FloatBinaryOp::Max | FloatBinaryOp::Copysign => {
                let func_addr = match (op, is_f32) {
                    (FloatBinaryOp::Min, true) => {
                        extern "C" fn f(x: f32, y: f32) -> f32 {
                            x.min(y)
                        }
                        f as *const () as usize
                    }
                    (FloatBinaryOp::Min, false) => {
                        extern "C" fn f(x: f64, y: f64) -> f64 {
                            x.min(y)
                        }
                        f as *const () as usize
                    }
                    (FloatBinaryOp::Max, true) => {
                        extern "C" fn f(x: f32, y: f32) -> f32 {
                            x.max(y)
                        }
                        f as *const () as usize
                    }
                    (FloatBinaryOp::Max, false) => {
                        extern "C" fn f(x: f64, y: f64) -> f64 {
                            x.max(y)
                        }
                        f as *const () as usize
                    }
                    (FloatBinaryOp::Copysign, true) => {
                        extern "C" fn f(x: f32, y: f32) -> f32 {
                            x.copysign(y)
                        }
                        f as *const () as usize
                    }
                    (FloatBinaryOp::Copysign, false) => {
                        extern "C" fn f(x: f64, y: f64) -> f64 {
                            x.copysign(y)
                        }
                        f as *const () as usize
                    }
                    _ => unreachable!(),
                };
                self.emit_float_binary_libm_call(emitter, freg_a, freg_b, func_addr, is_f32)?;
            }
        }

        self.free_float_register(freg_b);
        self.push(StackValue::FloatRegister(freg_a));
        Ok(())
    }

    /// Compile a floating point unary operation
    fn compile_float_unary_op(
        &mut self,
        emitter: &mut Emitter,
        op: FloatUnaryOp,
        is_f32: bool,
    ) -> Result<(), CompileError> {
        let val = self.pop_value()?;
        let freg = self.ensure_in_float_register(emitter, val, is_f32)?;

        match op {
            FloatUnaryOp::Neg if is_f32 => {
                emitter.neg_f32(0, freg, freg)?;
            }
            FloatUnaryOp::Neg => {
                emitter.neg_f64(0, freg, freg)?;
            }
            FloatUnaryOp::Abs if is_f32 => {
                emitter.abs_f32(0, freg, freg)?;
            }
            FloatUnaryOp::Abs => {
                emitter.abs_f64(0, freg, freg)?;
            }
            // For Sqrt, Ceil, Floor, Trunc, Nearest - call C library functions
            _ => {
                let func_addr = match (op, is_f32) {
                    (FloatUnaryOp::Sqrt, true) => {
                        extern "C" fn f(x: f32) -> f32 {
                            x.sqrt()
                        }
                        f as *const () as usize
                    }
                    (FloatUnaryOp::Sqrt, false) => {
                        extern "C" fn f(x: f64) -> f64 {
                            x.sqrt()
                        }
                        f as *const () as usize
                    }
                    (FloatUnaryOp::Ceil, true) => {
                        extern "C" fn f(x: f32) -> f32 {
                            x.ceil()
                        }
                        f as *const () as usize
                    }
                    (FloatUnaryOp::Ceil, false) => {
                        extern "C" fn f(x: f64) -> f64 {
                            x.ceil()
                        }
                        f as *const () as usize
                    }
                    (FloatUnaryOp::Floor, true) => {
                        extern "C" fn f(x: f32) -> f32 {
                            x.floor()
                        }
                        f as *const () as usize
                    }
                    (FloatUnaryOp::Floor, false) => {
                        extern "C" fn f(x: f64) -> f64 {
                            x.floor()
                        }
                        f as *const () as usize
                    }
                    (FloatUnaryOp::Trunc, true) => {
                        extern "C" fn f(x: f32) -> f32 {
                            x.trunc()
                        }
                        f as *const () as usize
                    }
                    (FloatUnaryOp::Trunc, false) => {
                        extern "C" fn f(x: f64) -> f64 {
                            x.trunc()
                        }
                        f as *const () as usize
                    }
                    (FloatUnaryOp::Nearest, true) => {
                        extern "C" fn f(x: f32) -> f32 {
                            x.round_ties_even()
                        }
                        f as *const () as usize
                    }
                    (FloatUnaryOp::Nearest, false) => {
                        extern "C" fn f(x: f64) -> f64 {
                            x.round_ties_even()
                        }
                        f as *const () as usize
                    }
                    _ => unreachable!(),
                };
                self.emit_float_libm_call(emitter, freg, func_addr, is_f32)?;
            }
        }

        self.push(StackValue::FloatRegister(freg));
        Ok(())
    }

    /// Helper to move float register conditionally based on is_f32
    fn mov_float(
        emitter: &mut Emitter,
        is_f32: bool,
        dst: impl Into<Operand>,
        src: impl Into<Operand>,
    ) -> Result<(), sys::ErrorCode> {
        if is_f32 {
            emitter.mov_f32(0, dst, src)?;
        } else {
            emitter.mov_f64(0, dst, src)?;
        }
        Ok(())
    }

    /// Emit a call to a libm function for float binary operations
    fn emit_float_binary_libm_call(
        &mut self,
        emitter: &mut Emitter,
        freg_a: FloatRegister,
        freg_b: FloatRegister,
        func_addr: usize,
        is_f32: bool,
    ) -> Result<(), CompileError> {
        if freg_a != FloatRegister::FR0 {
            Self::mov_float(emitter, is_f32, FloatRegister::FR0, freg_a)?;
        }
        if freg_b != FloatRegister::FR1 {
            Self::mov_float(emitter, is_f32, FloatRegister::FR1, freg_b)?;
        }

        let addr_reg = self.alloc_register(emitter)?;
        emitter.mov(0, addr_reg, func_addr as isize)?;
        let arg_types = if is_f32 {
            arg_types!([F32, F32] -> F32)
        } else {
            arg_types!([F64, F64] -> F64)
        };
        emitter.icall(JumpType::Call as i32, arg_types, addr_reg)?;
        self.free_register(addr_reg);

        if freg_a != FloatRegister::FR0 {
            Self::mov_float(emitter, is_f32, freg_a, FloatRegister::FR0)?;
        }
        Ok(())
    }

    /// Emit a call to a libm function for float unary operations
    fn emit_float_libm_call(
        &mut self,
        emitter: &mut Emitter,
        freg: FloatRegister,
        func_addr: usize,
        is_f32: bool,
    ) -> Result<(), CompileError> {
        if freg != FloatRegister::FR0 {
            Self::mov_float(emitter, is_f32, FloatRegister::FR0, freg)?;
        }

        let addr_reg = self.alloc_register(emitter)?;
        emitter.mov(0, addr_reg, func_addr as isize)?;
        let arg_types = if is_f32 {
            arg_types!([F32] -> F32)
        } else {
            arg_types!([F64] -> F64)
        };
        emitter.icall(JumpType::Call as i32, arg_types, addr_reg)?;
        self.free_register(addr_reg);

        if freg != FloatRegister::FR0 {
            Self::mov_float(emitter, is_f32, freg, FloatRegister::FR0)?;
        }
        Ok(())
    }

    /// Compile a floating point comparison operation
    fn compile_float_compare_op(
        &mut self,
        emitter: &mut Emitter,
        cond: Condition,
        is_f32: bool,
    ) -> Result<(), CompileError> {
        let (b, a) = (self.pop_value()?, self.pop_value()?);
        let (freg_a, freg_b) = (
            self.ensure_in_float_register(emitter, a, is_f32)?,
            self.ensure_in_float_register(emitter, b, is_f32)?,
        );
        let result_reg = self.alloc_register(emitter)?;

        let mut jump_true = emitter.fcmp(cond, freg_a, freg_b)?;
        emitter.mov(0, result_reg, 0i32)?;
        let mut jump_end = emitter.jump(JumpType::Jump)?;
        let mut label_true = emitter.put_label()?;
        jump_true.set_label(&mut label_true);
        emitter.mov(0, result_reg, 1i32)?;
        let mut label_end = emitter.put_label()?;
        jump_end.set_label(&mut label_end);

        self.free_float_register(freg_a);
        self.free_float_register(freg_b);
        self.push(StackValue::Register(result_reg));
        Ok(())
    }

    /// Convert integer to float
    fn compile_convert_int_to_float(
        &mut self,
        emitter: &mut Emitter,
        is_f32: bool,
        is_i32: bool,
        is_signed: bool,
    ) -> Result<(), CompileError> {
        let val = self.pop_value()?;

        #[cfg(target_pointer_width = "64")]
        {
            let reg = self.ensure_in_register(emitter, val)?;
            let freg = self.alloc_float_register(emitter)?;

            match (is_f32, is_i32, is_signed) {
                (true, true, true) => {
                    emitter.conv_f32_from_s32(0, freg, reg)?;
                }
                (true, true, false) => {
                    emitter.conv_f32_from_u32(0, freg, reg)?;
                }
                (false, true, true) => {
                    emitter.conv_f64_from_s32(0, freg, reg)?;
                }
                (false, true, false) => {
                    emitter.conv_f64_from_u32(0, freg, reg)?;
                }
                (true, false, true) => {
                    emitter.conv_f32_from_sw(0, freg, reg)?;
                }
                (true, false, false) => {
                    emitter.conv_f32_from_uw(0, freg, reg)?;
                }
                (false, false, true) => {
                    emitter.conv_f64_from_sw(0, freg, reg)?;
                }
                (false, false, false) => {
                    emitter.conv_f64_from_uw(0, freg, reg)?;
                }
            }

            self.free_register(reg);
            self.push(StackValue::FloatRegister(freg));
            Ok(())
        }

        #[cfg(not(target_pointer_width = "64"))]
        {
            // On 32-bit, i64 conversions need special handling
            if !is_i32 {
                // For i64 -> float on 32-bit, value is on stack as two 32-bit halves
                // We need to use a helper function due to complexity
                return Err(CompileError::Unsupported(
                    "64-bit to float conversion not yet supported on 32-bit platform".into(),
                ));
            }

            // i32 -> float works fine
            let reg = self.ensure_in_register(emitter, val)?;
            let freg = self.alloc_float_register(emitter)?;

            match (is_f32, is_signed) {
                (true, true) => {
                    emitter.conv_f32_from_s32(0, freg, reg)?;
                }
                (true, false) => {
                    emitter.conv_f32_from_u32(0, freg, reg)?;
                }
                (false, true) => {
                    emitter.conv_f64_from_s32(0, freg, reg)?;
                }
                (false, false) => {
                    emitter.conv_f64_from_u32(0, freg, reg)?;
                }
            }

            self.free_register(reg);
            self.push(StackValue::FloatRegister(freg));
            Ok(())
        }
    }

    /// Convert float to integer (truncate)
    fn compile_convert_float_to_int(
        &mut self,
        emitter: &mut Emitter,
        is_f32: bool,
        is_i32: bool,
        _is_signed: bool,
    ) -> Result<(), CompileError> {
        let val = self.pop_value()?;
        let freg = self.ensure_in_float_register(emitter, val, is_f32)?;
        let reg = self.alloc_register(emitter)?;

        // Note: SLJIT doesn't have direct unsigned conversion, use signed for all
        match (is_f32, is_i32) {
            (true, true) => {
                emitter.conv_s32_from_f32(0, reg, freg)?;
            }
            (false, true) => {
                emitter.conv_s32_from_f64(0, reg, freg)?;
            }

            (true, false) => {
                emitter.conv_sw_from_f32(0, reg, freg)?;
            }

            (false, false) => {
                emitter.conv_sw_from_f64(0, reg, freg)?;
            }
        }

        self.free_float_register(freg);
        self.push(StackValue::Register(reg));
        Ok(())
    }

    /// Compile f32.demote_f64 or f64.promote_f32
    fn compile_float_demote_promote(
        &mut self,
        emitter: &mut Emitter,
        is_demote: bool,
    ) -> Result<(), CompileError> {
        let val = self.pop_value()?;
        let freg = self.ensure_in_float_register(emitter, val, !is_demote)?;
        if is_demote {
            emitter.conv_f32_from_f64(0, freg, freg)?;
        } else {
            emitter.conv_f64_from_f32(0, freg, freg)?;
        }
        self.push(StackValue::FloatRegister(freg));
        Ok(())
    }

    /// Compile global.get - load a global variable value
    fn compile_global_get(
        &mut self,
        emitter: &mut Emitter,
        global_index: u32,
    ) -> Result<(), CompileError> {
        let global = self
            .globals
            .get(global_index as usize)
            .ok_or_else(|| {
                CompileError::Invalid(format!(
                    "Global index {} out of bounds (have {})",
                    global_index,
                    self.globals.len()
                ))
            })?
            .clone();

        let reg = self.alloc_register(emitter)?;
        emitter.mov(0, reg, global.ptr as isize)?;
        let mem = mem_offset(reg, 0);

        match global.val_type {
            ValType::I32 => {
                emitter.mov32(0, reg, mem)?;
            }
            ValType::I64 => {
                #[cfg(target_pointer_width = "64")]
                {
                    emitter.mov(0, reg, mem)?;
                }
                #[cfg(not(target_pointer_width = "64"))]
                {
                    return Err(CompileError::Unsupported(
                        "64-bit globals not supported on 32-bit platform".into(),
                    ));
                }
            }
            ValType::F32 | ValType::F64 => {
                let freg = self.alloc_float_register(emitter)?;
                if matches!(global.val_type, ValType::F32) {
                    emitter.mov_f32(0, freg, mem)?;
                } else {
                    emitter.mov_f64(0, freg, mem)?;
                }
                self.free_register(reg);
                self.push(StackValue::FloatRegister(freg));
                return Ok(());
            }
            _ => {
                return Err(CompileError::Unsupported(format!(
                    "Unsupported global type: {:?}",
                    global.val_type
                )));
            }
        }
        self.push(StackValue::Register(reg));
        Ok(())
    }

    /// Compile global.set - store a value to a global variable
    fn compile_global_set(
        &mut self,
        emitter: &mut Emitter,
        global_index: u32,
    ) -> Result<(), CompileError> {
        let global = self
            .globals
            .get(global_index as usize)
            .ok_or_else(|| {
                CompileError::Invalid(format!(
                    "Global index {} out of bounds (have {})",
                    global_index,
                    self.globals.len()
                ))
            })?
            .clone();

        if !global.mutable {
            return Err(CompileError::Invalid(format!(
                "Cannot set immutable global {}",
                global_index
            )));
        }

        let value = self.pop_value()?;
        let addr_reg = self.alloc_register(emitter)?;
        emitter.mov(0, addr_reg, global.ptr as isize)?;
        let mem = mem_offset(addr_reg, 0);

        match global.val_type {
            ValType::I32 => {
                let val_reg = self.ensure_in_register(emitter, value)?;
                emitter.mov32(0, mem, val_reg)?;
                self.free_register(val_reg);
            }
            ValType::I64 => {
                #[cfg(target_pointer_width = "64")]
                {
                    let val_reg = self.ensure_in_register(emitter, value)?;
                    emitter.mov(0, mem, val_reg)?;
                    self.free_register(val_reg);
                }
                #[cfg(not(target_pointer_width = "64"))]
                {
                    return Err(CompileError::Unsupported(
                        "64-bit globals not supported on 32-bit platform".into(),
                    ));
                }
            }
            ValType::F32 => {
                emitter.mov_f32(0, mem, value.to_operand())?;
            }
            ValType::F64 => {
                emitter.mov_f64(0, mem, value.to_operand())?;
            }
            _ => {
                return Err(CompileError::Unsupported(format!(
                    "Unsupported global type: {:?}",
                    global.val_type
                )));
            }
        }
        self.free_register(addr_reg);
        Ok(())
    }

    /// Compile memory.size - return current memory size in pages
    fn compile_memory_size(
        &mut self,
        emitter: &mut Emitter,
        _mem: u32,
    ) -> Result<(), CompileError> {
        let size_ptr = self
            .memory
            .as_ref()
            .ok_or_else(|| CompileError::Invalid("No memory defined".into()))?
            .size_ptr;

        let reg = self.alloc_register(emitter)?;
        emitter.mov(0, reg, size_ptr as isize)?;
        emitter.mov32(0, reg, mem_offset(reg, 0))?;
        self.push(StackValue::Register(reg));
        Ok(())
    }

    /// Compile memory.grow - grow memory by delta pages, return previous size or -1 on failure
    fn compile_memory_grow(
        &mut self,
        emitter: &mut Emitter,
        _mem: u32,
    ) -> Result<(), CompileError> {
        let (size_ptr, grow_callback) = {
            let memory = self
                .memory
                .as_ref()
                .ok_or_else(|| CompileError::Invalid("No memory defined".into()))?;

            let grow_callback = memory
                .grow_callback
                .ok_or_else(|| CompileError::Unsupported("Memory grow callback not set".into()))?;

            (memory.size_ptr, grow_callback)
        };

        // Pop the delta (number of pages to grow)
        let delta = self.pop_value()?;
        let delta_reg = self.ensure_in_register(emitter, delta)?;

        // Load current size into R0 (first argument)
        if delta_reg != ScratchRegister::R0 {
            emitter.mov(0, ScratchRegister::R0, size_ptr as isize)?;
            emitter.mov32(0, ScratchRegister::R0, mem_offset(ScratchRegister::R0, 0))?;
        } else {
            // delta is in R0, need to save it first
            emitter.mov(0, ScratchRegister::R1, delta_reg)?;
            emitter.mov(0, ScratchRegister::R0, size_ptr as isize)?;
            emitter.mov32(0, ScratchRegister::R0, mem_offset(ScratchRegister::R0, 0))?;
            emitter.mov(0, ScratchRegister::R1, ScratchRegister::R1)?; // delta now in R1
        }

        // Move delta to R1 (second argument) if not already there
        if delta_reg != ScratchRegister::R1 && delta_reg != ScratchRegister::R0 {
            emitter.mov(0, ScratchRegister::R1, delta_reg)?;
            self.free_register(delta_reg);
        } else if delta_reg == ScratchRegister::R0 {
            // Already moved to R1 above
            self.free_register(delta_reg);
        }

        // Call the grow callback: fn(current_pages: u32, delta: u32) -> i32
        let addr_reg = self.alloc_register(emitter)?;
        emitter.mov(0, addr_reg, grow_callback as isize)?;
        // arg_types for (W, W) -> W: return type W, plus two W arguments
        let arg_types_w2_w = SLJIT_ARG_TYPE_W | (SLJIT_ARG_TYPE_W << 4) | (SLJIT_ARG_TYPE_W << 8);
        emitter.icall(JumpType::Call as i32, arg_types_w2_w, addr_reg)?;
        self.free_register(addr_reg);

        // Result is in R0
        self.free_registers.retain(|&r| r != ScratchRegister::R0);
        self.push(StackValue::Register(ScratchRegister::R0));
        Ok(())
    }

    /// Compile call - direct function call
    /// Supports functions with more than 3 parameters by passing extra args on stack
    fn compile_call(
        &mut self,
        emitter: &mut Emitter,
        function_index: u32,
    ) -> Result<(), CompileError> {
        if function_index as usize >= self.functions.len() {
            return Err(CompileError::Invalid(format!(
                "Function index {} out of bounds (have {})",
                function_index,
                self.functions.len()
            )));
        }

        // Clone the needed data to avoid borrow issues
        let func_code_ptr = self.functions[function_index as usize].code_ptr;
        let func_params = self.functions[function_index as usize]
            .func_type
            .params
            .clone();
        let func_results = self.functions[function_index as usize]
            .func_type
            .results
            .clone();
        let param_count = func_params.len();
        let result_count = func_results.len();

        // Maximum 7 arguments supported by the arg_types bitmask
        if param_count > 7 {
            return Err(CompileError::Unsupported(
                "Functions with more than 7 parameters not supported".into(),
            ));
        }

        // Pop arguments in reverse order
        let mut args = vec![];
        for _ in 0..param_count {
            args.push(self.pop_value()?);
        }
        args.reverse();

        // Calculate stack space needed for extra arguments (args 4+)
        let extra_stack_size = Self::get_extra_arg_stack_size(param_count);

        // If we have extra arguments, allocate stack space and store them
        if extra_stack_size > 0 {
            // Adjust stack pointer to make room for extra arguments
            emitter.sub(
                0,
                Operand::from(sys::SLJIT_SP as i32),
                Operand::from(sys::SLJIT_SP as i32),
                extra_stack_size,
            )?;

            // Store extra arguments (args 4+) on the stack
            for (i, arg) in args.iter().enumerate().skip(3) {
                let stack_offset = ((i - 3) * 8) as i32;
                let reg = self.ensure_in_register(emitter, arg.clone())?;
                emitter.mov(0, mem_sp_offset(stack_offset), reg)?;
                self.free_register(reg);
            }
        }

        // Move first 3 arguments to the appropriate registers
        let arg_regs = [
            ScratchRegister::R0,
            ScratchRegister::R1,
            ScratchRegister::R2,
        ];
        for (i, arg) in args.into_iter().take(3).enumerate() {
            let reg = self.ensure_in_register(emitter, arg)?;
            if reg != arg_regs[i] {
                emitter.mov(0, arg_regs[i], reg)?;
                self.free_register(reg);
            }
        }

        // Load function address and call
        let addr_reg = self.alloc_register(emitter)?;
        emitter.mov(0, addr_reg, func_code_ptr as isize)?;

        let arg_types = Self::make_arg_types(&func_params, &func_results);
        emitter.icall(JumpType::Call as i32, arg_types, addr_reg)?;
        self.free_register(addr_reg);

        // Restore stack pointer if we allocated extra space
        if extra_stack_size > 0 {
            emitter.add(
                0,
                Operand::from(sys::SLJIT_SP as i32),
                Operand::from(sys::SLJIT_SP as i32),
                extra_stack_size,
            )?;
        }

        // Handle result
        if result_count > 0 {
            self.free_registers.retain(|&r| r != ScratchRegister::R0);
            self.push(StackValue::Register(ScratchRegister::R0));
        }

        Ok(())
    }

    /// Compile call_indirect - indirect function call through table
    /// Supports functions with more than 3 parameters by passing extra args on stack
    fn compile_call_indirect(
        &mut self,
        emitter: &mut Emitter,
        type_index: u32,
        table_index: u32,
    ) -> Result<(), CompileError> {
        if table_index as usize >= self.tables.len() {
            return Err(CompileError::Invalid(format!(
                "Table index {} out of bounds (have {})",
                table_index,
                self.tables.len()
            )));
        }
        if type_index as usize >= self.func_types.len() {
            return Err(CompileError::Invalid(format!(
                "Type index {} out of bounds (have {})",
                type_index,
                self.func_types.len()
            )));
        }

        // Clone the needed data to avoid borrow issues
        let table_data_ptr = self.tables[table_index as usize].data_ptr;
        let func_params = self.func_types[type_index as usize].params.clone();
        let func_results = self.func_types[type_index as usize].results.clone();
        let param_count = func_params.len();
        let result_count = func_results.len();

        // Maximum 7 arguments supported by the arg_types bitmask
        if param_count > 7 {
            return Err(CompileError::Unsupported(
                "Functions with more than 7 parameters not supported".into(),
            ));
        }

        // Pop the table index (the value on top of stack)
        let idx = self.pop_value()?;
        let idx_reg = self.ensure_in_register(emitter, idx)?;

        // Pop arguments in reverse order
        let mut args = vec![];
        for _ in 0..param_count {
            args.push(self.pop_value()?);
        }
        args.reverse();

        // Calculate the function pointer address: table.data_ptr + idx * sizeof(usize)
        let func_ptr_reg = self.alloc_register(emitter)?;
        emitter.mov(0, func_ptr_reg, table_data_ptr as isize)?;

        // idx * sizeof(usize)
        #[cfg(target_pointer_width = "64")]
        emitter.shl(0, idx_reg, idx_reg, 3i32)?; // * 8
        #[cfg(not(target_pointer_width = "64"))]
        emitter.shl32(0, idx_reg, idx_reg, 2i32)?; // * 4

        emitter.add(0, func_ptr_reg, func_ptr_reg, idx_reg)?;
        self.free_register(idx_reg);

        // Load the function pointer from the table
        emitter.mov(0, func_ptr_reg, mem_offset(func_ptr_reg, 0))?;

        // Save the function pointer to a temporary stack location since we'll need registers for args
        let func_ptr_offset = self.frame_offset;
        self.frame_offset += 8;
        emitter.mov(0, mem_sp_offset(func_ptr_offset), func_ptr_reg)?;
        self.free_register(func_ptr_reg);

        // Calculate stack space needed for extra arguments (args 4+)
        let extra_stack_size = Self::get_extra_arg_stack_size(param_count);

        // If we have extra arguments, allocate stack space and store them
        if extra_stack_size > 0 {
            // Adjust stack pointer to make room for extra arguments
            emitter.sub(
                0,
                Operand::from(sys::SLJIT_SP as i32),
                Operand::from(sys::SLJIT_SP as i32),
                extra_stack_size,
            )?;

            // Store extra arguments (args 4+) on the stack
            for (i, arg) in args.iter().enumerate().skip(3) {
                let stack_offset = ((i - 3) * 8) as i32;
                let reg = self.ensure_in_register(emitter, arg.clone())?;
                emitter.mov(0, mem_sp_offset(stack_offset), reg)?;
                self.free_register(reg);
            }
        }

        // Move first 3 arguments to the appropriate registers
        let arg_regs = [
            ScratchRegister::R0,
            ScratchRegister::R1,
            ScratchRegister::R2,
        ];
        for (i, arg) in args.into_iter().take(3).enumerate() {
            let reg = self.ensure_in_register(emitter, arg)?;
            if reg != arg_regs[i] {
                emitter.mov(0, arg_regs[i], reg)?;
                self.free_register(reg);
            }
        }

        // Reload the function pointer
        let addr_reg = self.alloc_register(emitter)?;
        // Account for the stack adjustment when loading the saved function pointer
        let adjusted_offset = if extra_stack_size > 0 {
            func_ptr_offset + extra_stack_size
        } else {
            func_ptr_offset
        };
        emitter.mov(0, addr_reg, mem_sp_offset(adjusted_offset))?;

        // Call through the function pointer
        let arg_types = Self::make_arg_types(&func_params, &func_results);
        emitter.icall(JumpType::Call as i32, arg_types, addr_reg)?;
        self.free_register(addr_reg);

        // Restore stack pointer if we allocated extra space
        if extra_stack_size > 0 {
            emitter.add(
                0,
                Operand::from(sys::SLJIT_SP as i32),
                Operand::from(sys::SLJIT_SP as i32),
                extra_stack_size,
            )?;
        }

        // Handle result
        if result_count > 0 {
            self.free_registers.retain(|&r| r != ScratchRegister::R0);
            self.push(StackValue::Register(ScratchRegister::R0));
        }

        Ok(())
    }

    /// Compile br_table - branch table (switch-like construct)
    fn compile_br_table(
        &mut self,
        emitter: &mut Emitter,
        targets: &wasmparser::BrTable,
    ) -> Result<(), CompileError> {
        // Pop the index value
        let idx = self.pop_value()?;
        let idx_reg = self.ensure_in_register(emitter, idx)?;

        self.save_all_to_stack(emitter)?;

        // Get all targets
        let default_target = targets.default();

        // For each target, emit a comparison and conditional jump
        // This is a simple linear search - for large tables, a jump table would be more efficient
        for (i, target) in targets.targets().enumerate() {
            let target = target.map_err(|e| CompileError::Parse(e.to_string()))?;
            let mut jump = emitter.cmp(Condition::Equal, idx_reg, i as i32)?;

            let block_idx = self.blocks.len() - 1 - target as usize;
            if let Some(mut label) = self.blocks[block_idx].label {
                jump.set_label(&mut label);
            } else {
                self.blocks[block_idx].end_jumps.push(jump);
            }
        }

        // Default case - jump to default target
        self.free_register(idx_reg);

        let block_idx = self.blocks.len() - 1 - default_target as usize;
        if let Some(mut label) = self.blocks[block_idx].label {
            let mut jump = emitter.jump(JumpType::Jump)?;
            jump.set_label(&mut label);
        } else {
            self.blocks[block_idx]
                .end_jumps
                .push(emitter.jump(JumpType::Jump)?);
        }

        Ok(())
    }
}

impl Default for Function {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

/// Compiled WebAssembly function
pub struct CompiledFunction {
    code: GeneratedCode,
}

impl CompiledFunction {
    #[inline(always)]
    pub fn as_fn_0(&self) -> fn() -> i32 {
        unsafe { std::mem::transmute(self.code.get()) }
    }

    #[inline(always)]
    pub fn as_fn_1(&self) -> fn(i32) -> i32 {
        unsafe { std::mem::transmute(self.code.get()) }
    }

    #[inline(always)]
    pub fn as_fn_2(&self) -> fn(i32, i32) -> i32 {
        unsafe { std::mem::transmute(self.code.get()) }
    }

    #[inline(always)]
    pub fn as_fn_3(&self) -> fn(i32, i32, i32) -> i32 {
        unsafe { std::mem::transmute(self.code.get()) }
    }
}

/// Compile a simple WebAssembly function from operators
pub fn compile_simple<'a>(
    params: &[ValType],
    results: &[ValType],
    locals: &[ValType],
    body: &[Operator],
) -> Result<CompiledFunction, CompileError> {
    let mut compiler = Compiler::new();
    let mut wasm_compiler = Function::new();
    wasm_compiler.compile_function(&mut compiler, params, results, locals, body.iter().cloned())?;
    Ok(CompiledFunction {
        code: compiler.generate_code(),
    })
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
                        let ty = ty?;
                    }
                }
                wasmparser::Payload::ImportSection(import_section) => {
                    for import in import_section {
                        match import? {
                            wasmparser::Imports::Single(_, import) => {
                                imports.push(import.clone());
                            }
                            wasmparser::Imports::Compact1 { module, items } => todo!(),
                            wasmparser::Imports::Compact2 { module, ty, names } => todo!(),
                        }
                    }
                }
                wasmparser::Payload::FunctionSection(functions) => {
                    for func in functions {
                        let func = func?;
                    }
                }
                wasmparser::Payload::CodeSectionStart { count, .. } => {
                    todo!()
                }
                wasmparser::Payload::CodeSectionEntry(body) => {
                    let body: Vec<Operator> = body
                        .get_operators_reader()?
                        .into_iter()
                        .collect::<Result<_, _>>()?;
                }
                wasmparser::Payload::Version {
                    num,
                    encoding,
                    range,
                } => todo!(),
                wasmparser::Payload::TableSection(section_limited) => {
                    for table in section_limited {
                        let table = table?;
                    }
                }
                wasmparser::Payload::MemorySection(section_limited) => {
                    for memory in section_limited {
                        let memory = memory?;
                    }
                }
                wasmparser::Payload::TagSection(section_limited) => {
                    for tag in section_limited {
                        let tag = tag?;
                    }
                }
                wasmparser::Payload::GlobalSection(section_limited) => {
                    for global in section_limited {
                        let global = global?;
                    }
                }
                wasmparser::Payload::ExportSection(section_limited) => {
                    for export in section_limited {
                        let export = export?;
                    }
                }
                wasmparser::Payload::StartSection { func, range } => todo!(),
                wasmparser::Payload::ElementSection(section_limited) => {
                    for element in section_limited {
                        let element = element?;
                    }
                }
                wasmparser::Payload::DataCountSection { count, range } => todo!(),
                wasmparser::Payload::DataSection(section_limited) => {
                    for data in section_limited {
                        let data = data?;
                    }
                }
                wasmparser::Payload::ModuleSection {
                    parser,
                    unchecked_range,
                } => todo!(),
                wasmparser::Payload::InstanceSection(section_limited) => {
                    for instance in section_limited {
                        let instance = instance?;
                    }
                }
                wasmparser::Payload::CoreTypeSection(section_limited) => {
                    for core_type in section_limited {
                        let core_type = core_type?;
                    }
                }
                wasmparser::Payload::ComponentSection {
                    parser,
                    unchecked_range,
                } => todo!(),
                wasmparser::Payload::ComponentInstanceSection(section_limited) => todo!(),
                wasmparser::Payload::ComponentAliasSection(section_limited) => todo!(),
                wasmparser::Payload::ComponentTypeSection(section_limited) => todo!(),
                wasmparser::Payload::ComponentCanonicalSection(section_limited) => todo!(),
                wasmparser::Payload::ComponentStartSection { start, range } => todo!(),
                wasmparser::Payload::ComponentImportSection(section_limited) => todo!(),
                wasmparser::Payload::ComponentExportSection(section_limited) => todo!(),
                wasmparser::Payload::CustomSection(custom_section_reader) => todo!(),
                wasmparser::Payload::UnknownSection {
                    id,
                    contents,
                    range,
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
