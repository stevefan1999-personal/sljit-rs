//! Simple WebAssembly JIT Compiler using sljit-rs
//!
//! This is a minimal WebAssembly to native code compiler that demonstrates
//! how to use sljit-rs's high-level Emitter API to JIT-compile WebAssembly.
//!
//! Inspired by pwart's architecture, but simplified for clarity.

use indexmap::IndexSet;
use sljit::sys::{self, Compiler, GeneratedCode};
use sljit::{
    Condition, Emitter, FloatRegister, JumpType, Operand, ReturnOp, SavedRegister, ScratchRegister,
    mem_offset, mem_sp_offset, regs,
};
use wasmparser::{Operator, ValType};

/// Errors that can occur during compilation
#[derive(Debug)]
pub enum CompileError {
    Parse(String),
    Sljit(sys::ErrorCode),
    Unsupported(String),
    Invalid(String),
}

impl From<sys::ErrorCode> for CompileError {
    fn from(e: sys::ErrorCode) -> Self {
        CompileError::Sljit(e)
    }
}

impl From<wasmparser::BinaryReaderError> for CompileError {
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

/// Block type for control flow
#[derive(Clone, Debug)]
#[allow(dead_code)]
enum BlockKind {
    Block,
    Loop,
    If,
}

/// Information about a control flow block
struct Block {
    #[allow(dead_code)]
    kind: BlockKind,
    label: Option<sys::Label>,
    end_jumps: Vec<sys::Jump>,
    else_jump: Option<sys::Jump>,
    stack_depth: usize,
    #[allow(dead_code)]
    result_count: usize,
    result_offset: Option<i32>,
}

/// Local variable storage info
#[derive(Clone, Debug)]
enum LocalVar {
    Saved(SavedRegister),
    Stack(i32),
}

/// WebAssembly function compiler
pub struct WasmCompiler {
    stack: Vec<StackValue>,
    blocks: Vec<Block>,
    locals: Vec<LocalVar>,
    param_count: usize,
    frame_offset: i32,
    local_size: i32,
    free_registers: IndexSet<ScratchRegister>,
    free_float_registers: IndexSet<FloatRegister>,
}

// Helper macro for stack underflow error
macro_rules! stack_underflow {
    () => {
        CompileError::Invalid("Stack underflow".into())
    };
}

impl WasmCompiler {
    pub fn new() -> Self {
        Self {
            stack: Vec::new(),
            blocks: Vec::new(),
            locals: Vec::new(),
            param_count: 0,
            frame_offset: 0,
            local_size: 0,
            free_registers: IndexSet::from_iter([
                ScratchRegister::R0,
                ScratchRegister::R1,
                ScratchRegister::R2,
            ]),
            free_float_registers: IndexSet::from_iter([
                FloatRegister::FR0,
                FloatRegister::FR1,
                FloatRegister::FR2,
            ]),
        }
    }

    /// Pop a value from stack with error handling
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
            return Ok(reg);
        }

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
            return Ok(reg);
        }

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

    fn free_float_register(&mut self, reg: FloatRegister) {
        if !self.free_float_registers.contains(&reg) {
            self.free_float_registers.insert(reg);
        }
    }

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
                let gp_reg = self.alloc_register(emitter)?;
                // Ensure 8-byte alignment for float operations
                let temp_offset = (self.frame_offset + 7) & !7;
                self.frame_offset = temp_offset + 8;
                emitter.mov(0, gp_reg, bits as i32)?;
                emitter.mov32(0, mem_sp_offset(temp_offset), gp_reg)?;
                emitter.mov_f32(0, reg, mem_sp_offset(temp_offset))?;
                self.free_register(gp_reg);
                Ok(reg)
            }
            StackValue::ConstF64(bits) => {
                let reg = self.alloc_float_register(emitter)?;
                // Store float bits to stack, then load as float
                let gp_reg = self.alloc_register(emitter)?;
                // Ensure 8-byte alignment for f64 operations
                let temp_offset = (self.frame_offset + 7) & !7;
                self.frame_offset = temp_offset + 8;
                #[cfg(target_pointer_width = "64")]
                {
                    emitter.mov(0, gp_reg, bits as isize)?;
                    emitter.mov(0, mem_sp_offset(temp_offset), gp_reg)?;
                }
                #[cfg(not(target_pointer_width = "64"))]
                {
                    // On 32-bit, need to store two halves
                    emitter.mov(0, gp_reg, (bits & 0xFFFFFFFF) as i32)?;
                    emitter.mov32(0, mem_sp_offset(temp_offset), gp_reg)?;
                    emitter.mov(0, gp_reg, (bits >> 32) as i32)?;
                    emitter.mov32(0, mem_sp_offset(temp_offset + 4), gp_reg)?;
                }
                emitter.mov_f64(0, reg, mem_sp_offset(temp_offset))?;
                self.free_register(gp_reg);
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
                let reg = self.alloc_register(emitter)?;
                emitter.mov(0, reg, *v)?;
                emitter.mov(0, mem_sp_offset(*offset), reg)?;
                self.free_register(reg);
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
                let reg = self.alloc_register(emitter)?;
                emitter.mov(0, reg, *bits as i32)?;
                emitter.mov32(0, mem_sp_offset(*offset), reg)?;
                self.free_register(reg);
                Ok(())
            }
            (LocalVar::Stack(offset), StackValue::ConstF64(bits)) => {
                // Store f64 constant to stack
                let reg = self.alloc_register(emitter)?;
                #[cfg(target_pointer_width = "64")]
                {
                    emitter.mov(0, reg, *bits as isize)?;
                    emitter.mov(0, mem_sp_offset(*offset), reg)?;
                }
                #[cfg(not(target_pointer_width = "64"))]
                {
                    emitter.mov(0, reg, (*bits & 0xFFFFFFFF) as i32)?;
                    emitter.mov32(0, mem_sp_offset(*offset), reg)?;
                    emitter.mov(0, reg, (*bits >> 32) as i32)?;
                    emitter.mov32(0, mem_sp_offset(*offset + 4), reg)?;
                }
                self.free_register(reg);
                Ok(())
            }
        }
    }

    /// Save all register values to stack (for control flow)
    fn save_all_to_stack(&mut self, emitter: &mut Emitter) -> Result<(), CompileError> {
        let saves: Vec<_> = self
            .stack
            .iter()
            .enumerate()
            .filter_map(|(i, sv)| {
                matches!(sv, StackValue::Register(reg) if true)
                    .then(|| {
                        if let StackValue::Register(reg) = sv {
                            Some((i, *reg))
                        } else {
                            None
                        }
                    })
                    .flatten()
            })
            .collect();

        for (i, reg) in saves {
            let offset = self.frame_offset;
            self.frame_offset += 8;
            emitter.mov(0, mem_sp_offset(offset), reg)?;
            self.free_register(reg);
            self.stack[i] = StackValue::Stack(offset);
        }
        Ok(())
    }

    /// Create a new block with common initialization
    fn new_block(&mut self, kind: BlockKind, label: Option<sys::Label>, has_result: bool) -> Block {
        let result_offset = if has_result {
            let offset = self.frame_offset;
            self.frame_offset += 8;
            Some(offset)
        } else {
            None
        };

        Block {
            kind,
            label,
            end_jumps: Vec::new(),
            else_jump: None,
            stack_depth: self.stack.len(),
            result_count: if has_result { 1 } else { 0 },
            result_offset,
        }
    }

    /// Check if block type has a result
    fn block_has_result(blockty: &wasmparser::BlockType) -> bool {
        matches!(blockty, wasmparser::BlockType::Type(_))
    }

    /// Make argument types bitmask for sljit
    fn make_arg_types(params: &[ValType], results: &[ValType]) -> i32 {
        let ret_type = if results.is_empty() {
            sys::SLJIT_ARG_TYPE_RET_VOID
        } else {
            sys::SLJIT_ARG_TYPE_W
        };
        params
            .iter()
            .take(3)
            .enumerate()
            .fold(ret_type, |acc, (i, _)| {
                acc | (sys::SLJIT_ARG_TYPE_W << (4 * (i + 1)))
            })
    }

    /// Compile a WebAssembly function
    pub fn compile_function(
        &mut self,
        compiler: &mut Compiler,
        params: &[ValType],
        results: &[ValType],
        locals: &[(u32, ValType)],
        body: &[Operator],
    ) -> Result<(), CompileError> {
        let mut emitter = Emitter::new(compiler);
        let total_locals: usize =
            params.len() + locals.iter().map(|(c, _)| *c as usize).sum::<usize>();

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
        for (count, _) in locals {
            for _ in 0..*count {
                self.locals.push(LocalVar::Stack(stack_local_offset));
                stack_local_offset += 8;
            }
        }

        // Ensure frame_offset is 16-byte aligned for proper float operation alignment
        let aligned_offset = (stack_local_offset + 15) & !15;
        // Use much larger stack space to avoid any potential overflow issues with float ops
        self.local_size = aligned_offset + 1024;
        self.frame_offset = aligned_offset;

        emitter.emit_enter(
            0,
            Self::make_arg_types(params, results),
            regs! { gp: 3, float: 3 },
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
            kind: BlockKind::Block,
            label: None,
            end_jumps: Vec::new(),
            else_jump: None,
            stack_depth: 0,
            result_count: results.len(),
            result_offset: None,
        });

        for op in body {
            self.compile_operator(&mut emitter, op)?;
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
                let block = self.new_block(BlockKind::Block, None, Self::block_has_result(blockty));
                self.blocks.push(block);
                Ok(())
            }

            Operator::Loop { blockty: _ } => {
                self.save_all_to_stack(emitter)?;
                let label = emitter.put_label()?;
                let block = self.new_block(BlockKind::Loop, Some(label), false);
                self.blocks.push(block);
                Ok(())
            }

            Operator::If { blockty } => {
                let cond = self.pop_value()?;
                self.save_all_to_stack(emitter)?;
                let jump = self.emit_cond_jump(emitter, cond, Condition::Equal)?;
                let mut block =
                    self.new_block(BlockKind::If, None, Self::block_has_result(blockty));
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

                if let Some(offset) = result_offset {
                    if self.stack.len() > stack_depth {
                        let val = self.stack.pop().unwrap();
                        self.emit_mov_to_stack_offset(emitter, offset, val)?;
                    }
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
                }
                #[cfg(not(target_pointer_width = "64"))]
                if *value >= i32::MIN as i64 && *value <= i32::MAX as i64 {
                    self.push(StackValue::Const(*value as i32));
                } else {
                    return Err(CompileError::Unsupported(
                        "64-bit values not supported on 32-bit platform".into(),
                    ));
                }
                Ok(())
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
                let reg = self.ensure_in_register(emitter, val)?;
                emitter.mov32(0, reg, reg)?;
                self.push(StackValue::Register(reg));
                Ok(())
            }

            Operator::I64ExtendI32S => {
                let val = self.pop_value()?;
                let reg = self.ensure_in_register(emitter, val)?;
                emitter.mov_s32(0, reg, reg)?;
                self.push(StackValue::Register(reg));
                Ok(())
            }

            Operator::I64ExtendI32U => {
                let val = self.pop_value()?;
                let reg = self.ensure_in_register(emitter, val)?;
                emitter.mov_u32(0, reg, reg)?;
                self.push(StackValue::Register(reg));
                Ok(())
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
                #[cfg(target_pointer_width = "64")]
                {
                    let val = self.pop_value()?;
                    let freg = self.ensure_in_float_register(emitter, val, false)?;
                    let reg = self.alloc_register(emitter)?;
                    // Ensure 8-byte alignment for f64 operations
                    let temp_offset = (self.frame_offset + 7) & !7;
                    self.frame_offset = temp_offset + 8;
                    emitter.mov_f64(0, mem_sp_offset(temp_offset), freg)?;
                    emitter.mov(0, reg, mem_sp_offset(temp_offset))?;
                    self.free_float_register(freg);
                    self.push(StackValue::Register(reg));
                    Ok(())
                }
                #[cfg(not(target_pointer_width = "64"))]
                {
                    Err(CompileError::Unsupported(
                        "64-bit reinterpret not supported on 32-bit platform".into(),
                    ))
                }
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
                #[cfg(target_pointer_width = "64")]
                {
                    let val = self.pop_value()?;
                    let reg = self.ensure_in_register(emitter, val)?;
                    let freg = self.alloc_float_register(emitter)?;
                    // Ensure 8-byte alignment for f64 operations
                    let temp_offset = (self.frame_offset + 7) & !7;
                    self.frame_offset = temp_offset + 8;
                    emitter.mov(0, mem_sp_offset(temp_offset), reg)?;
                    emitter.mov_f64(0, freg, mem_sp_offset(temp_offset))?;
                    self.free_register(reg);
                    self.push(StackValue::FloatRegister(freg));
                    Ok(())
                }
                #[cfg(not(target_pointer_width = "64"))]
                {
                    Err(CompileError::Unsupported(
                        "64-bit reinterpret not supported on 32-bit platform".into(),
                    ))
                }
            }

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
            _ => {
                let reg = self.ensure_in_register(emitter, cond)?;
                let j = emitter.cmp(condition, reg, 0i32)?;
                self.free_register(reg);
                Ok(j)
            }
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

        if let Some(result_offset) = self.blocks[block_idx].result_offset {
            if let Some(val) = self.stack.pop() {
                if let StackValue::Register(reg) = &val {
                    self.free_register(*reg);
                }
                self.emit_mov_to_stack_offset(emitter, result_offset, val)?;
            }
        }
        self.save_all_to_stack(emitter)?;

        let label_clone = self.blocks[block_idx].label;
        if let Some(mut label) = label_clone {
            let mut jump = emitter.jump(JumpType::Jump)?;
            jump.set_label(&mut label);
        } else {
            self.blocks[block_idx]
                .end_jumps
                .push(emitter.jump(JumpType::Jump)?);
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
        let label_clone = self.blocks[block_idx].label;
        let jump = self.emit_cond_jump(emitter, cond, Condition::NotEqual)?;

        if let Some(mut label) = label_clone {
            let mut j = jump;
            j.set_label(&mut label);
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
            BinaryOp::Add => {
                emitter.add32(0, reg_a, reg_a, operand_b)?;
            }
            BinaryOp::Sub => {
                emitter.sub32(0, reg_a, reg_a, operand_b)?;
            }
            BinaryOp::Mul => {
                emitter.mul32(0, reg_a, reg_a, operand_b)?;
            }
            BinaryOp::And => {
                emitter.and32(0, reg_a, reg_a, operand_b)?;
            }
            BinaryOp::Or => {
                emitter.or32(0, reg_a, reg_a, operand_b)?;
            }
            BinaryOp::Xor => {
                emitter.xor32(0, reg_a, reg_a, operand_b)?;
            }
            BinaryOp::Shl => {
                emitter.shl32(0, reg_a, reg_a, operand_b)?;
            }
            BinaryOp::ShrS => {
                emitter.ashr32(0, reg_a, reg_a, operand_b)?;
            }
            BinaryOp::ShrU => {
                emitter.lshr32(0, reg_a, reg_a, operand_b)?;
            }
            BinaryOp::Rotl => {
                emitter.rotl32(0, reg_a, reg_a, operand_b)?;
            }
            BinaryOp::Rotr => {
                emitter.rotr32(0, reg_a, reg_a, operand_b)?;
            }
        }

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

        match kind {
            LoadStoreKind::I32 => {
                emitter.mov32(0, addr_reg, mem_offset(addr_reg, offset))?;
                self.push(StackValue::Register(addr_reg));
            }
            LoadStoreKind::I8S => {
                emitter.mov_s8(0, addr_reg, mem_offset(addr_reg, offset))?;
                self.push(StackValue::Register(addr_reg));
            }
            LoadStoreKind::I8U => {
                emitter.mov_u8(0, addr_reg, mem_offset(addr_reg, offset))?;
                self.push(StackValue::Register(addr_reg));
            }
            LoadStoreKind::I16S => {
                emitter.mov_s16(0, addr_reg, mem_offset(addr_reg, offset))?;
                self.push(StackValue::Register(addr_reg));
            }
            LoadStoreKind::I16U => {
                emitter.mov_u16(0, addr_reg, mem_offset(addr_reg, offset))?;
                self.push(StackValue::Register(addr_reg));
            }
            LoadStoreKind::F32 => {
                let freg = self.alloc_float_register(emitter)?;
                emitter.mov_f32(0, freg, mem_offset(addr_reg, offset))?;
                self.free_register(addr_reg);
                self.push(StackValue::FloatRegister(freg));
            }
            LoadStoreKind::F64 => {
                let freg = self.alloc_float_register(emitter)?;
                emitter.mov_f64(0, freg, mem_offset(addr_reg, offset))?;
                self.free_register(addr_reg);
                self.push(StackValue::FloatRegister(freg));
            }
            _ => unreachable!("I64 variants handled by compile_load_op64"),
        }
        Ok(())
    }

    fn compile_store_op(
        &mut self,
        emitter: &mut Emitter,
        offset: i32,
        kind: LoadStoreKind,
    ) -> Result<(), CompileError> {
        match kind {
            LoadStoreKind::F32 | LoadStoreKind::F64 => {
                let (value, addr) = (self.pop_value()?, self.pop_value()?);
                let is_f32 = matches!(kind, LoadStoreKind::F32);
                let freg = self.ensure_in_float_register(emitter, value, is_f32)?;
                let addr_reg = self.ensure_in_register(emitter, addr)?;
                emitter.add(0, addr_reg, addr_reg, SavedRegister::S0)?;

                if is_f32 {
                    emitter.mov_f32(0, mem_offset(addr_reg, offset), freg)?;
                } else {
                    emitter.mov_f64(0, mem_offset(addr_reg, offset), freg)?;
                }

                self.free_float_register(freg);
                self.free_register(addr_reg);
            }
            _ => {
                let (value, addr) = (self.pop_value()?, self.pop_value()?);
                let (value_reg, addr_reg) = (
                    self.ensure_in_register(emitter, value)?,
                    self.ensure_in_register(emitter, addr)?,
                );
                emitter.add(0, addr_reg, addr_reg, SavedRegister::S0)?;

                match kind {
                    LoadStoreKind::I32 => {
                        emitter.mov32(0, mem_offset(addr_reg, offset), value_reg)?;
                    }
                    LoadStoreKind::I8U | LoadStoreKind::I8S => {
                        emitter.mov_u8(0, mem_offset(addr_reg, offset), value_reg)?;
                    }
                    LoadStoreKind::I16U | LoadStoreKind::I16S => {
                        emitter.mov_u16(0, mem_offset(addr_reg, offset), value_reg)?;
                    }
                    _ => unreachable!("I64 variants handled by compile_store_op64"),
                }
                self.free_register(value_reg);
                self.free_register(addr_reg);
            }
        }
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
            BinaryOp::Add => {
                emitter.add(0, reg_a, reg_a, operand_b)?;
            }
            BinaryOp::Sub => {
                emitter.sub(0, reg_a, reg_a, operand_b)?;
            }
            BinaryOp::Mul => {
                emitter.mul(0, reg_a, reg_a, operand_b)?;
            }
            BinaryOp::And => {
                emitter.and(0, reg_a, reg_a, operand_b)?;
            }
            BinaryOp::Or => {
                emitter.or(0, reg_a, reg_a, operand_b)?;
            }
            BinaryOp::Xor => {
                emitter.xor(0, reg_a, reg_a, operand_b)?;
            }
            BinaryOp::Shl => {
                emitter.shl(0, reg_a, reg_a, operand_b)?;
            }
            BinaryOp::ShrS => {
                emitter.ashr(0, reg_a, reg_a, operand_b)?;
            }
            BinaryOp::ShrU => {
                emitter.lshr(0, reg_a, reg_a, operand_b)?;
            }
            BinaryOp::Rotl => {
                emitter.rotl(0, reg_a, reg_a, operand_b)?;
            }
            BinaryOp::Rotr => {
                emitter.rotr(0, reg_a, reg_a, operand_b)?;
            }
        }

        if let StackValue::Register(reg_b) = b {
            self.free_register(reg_b);
        }
        self.push(StackValue::Register(reg_a));
        Ok(())
    }

    #[cfg(not(target_pointer_width = "64"))]
    fn compile_binary_op64(
        &mut self,
        _emitter: &mut Emitter,
        _op: BinaryOp,
    ) -> Result<(), CompileError> {
        Err(CompileError::Unsupported(
            "64-bit operations not fully supported on 32-bit platform".into(),
        ))
    }

    #[cfg(target_pointer_width = "64")]
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

    #[cfg(not(target_pointer_width = "64"))]
    fn compile_compare_op64(
        &mut self,
        _emitter: &mut Emitter,
        _cond: Condition,
    ) -> Result<(), CompileError> {
        Err(CompileError::Unsupported(
            "64-bit comparisons not supported on 32-bit platform".into(),
        ))
    }

    #[cfg(target_pointer_width = "64")]
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

    #[cfg(not(target_pointer_width = "64"))]
    fn compile_div_op64(&mut self, _emitter: &mut Emitter, _op: DivOp) -> Result<(), CompileError> {
        Err(CompileError::Unsupported(
            "64-bit division not supported on 32-bit platform".into(),
        ))
    }

    #[cfg(target_pointer_width = "64")]
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

    #[cfg(not(target_pointer_width = "64"))]
    fn compile_unary_op64(
        &mut self,
        _emitter: &mut Emitter,
        _op: UnaryOp,
    ) -> Result<(), CompileError> {
        Err(CompileError::Unsupported(
            "64-bit unary operations not supported on 32-bit platform".into(),
        ))
    }

    #[cfg(target_pointer_width = "64")]
    fn compile_load_op64(
        &mut self,
        emitter: &mut Emitter,
        offset: i32,
        kind: LoadStoreKind,
    ) -> Result<(), CompileError> {
        let addr = self.pop_value()?;
        let addr_reg = self.ensure_in_register(emitter, addr)?;
        emitter.add(0, addr_reg, addr_reg, SavedRegister::S0)?;

        match kind {
            LoadStoreKind::I64 => {
                emitter.mov(0, addr_reg, mem_offset(addr_reg, offset))?;
            }
            LoadStoreKind::I64_8S => {
                emitter.mov_s8(0, addr_reg, mem_offset(addr_reg, offset))?;
            }
            LoadStoreKind::I64_8U => {
                emitter.mov_u8(0, addr_reg, mem_offset(addr_reg, offset))?;
            }
            LoadStoreKind::I64_16S => {
                emitter.mov_s16(0, addr_reg, mem_offset(addr_reg, offset))?;
            }
            LoadStoreKind::I64_16U => {
                emitter.mov_u16(0, addr_reg, mem_offset(addr_reg, offset))?;
            }
            LoadStoreKind::I64_32S => {
                emitter.mov_s32(0, addr_reg, mem_offset(addr_reg, offset))?;
            }
            LoadStoreKind::I64_32U => {
                emitter.mov_u32(0, addr_reg, mem_offset(addr_reg, offset))?;
            }
            _ => unreachable!(),
        }
        self.push(StackValue::Register(addr_reg));
        Ok(())
    }

    #[cfg(not(target_pointer_width = "64"))]
    fn compile_load_op64(
        &mut self,
        _emitter: &mut Emitter,
        _offset: i32,
        _kind: LoadStoreKind,
    ) -> Result<(), CompileError> {
        Err(CompileError::Unsupported(
            "64-bit memory operations not supported on 32-bit platform".into(),
        ))
    }

    #[cfg(target_pointer_width = "64")]
    fn compile_store_op64(
        &mut self,
        emitter: &mut Emitter,
        offset: i32,
        kind: LoadStoreKind,
    ) -> Result<(), CompileError> {
        let (value, addr) = (self.pop_value()?, self.pop_value()?);
        let (value_reg, addr_reg) = (
            self.ensure_in_register(emitter, value)?,
            self.ensure_in_register(emitter, addr)?,
        );
        emitter.add(0, addr_reg, addr_reg, SavedRegister::S0)?;

        match kind {
            LoadStoreKind::I64 => {
                emitter.mov(0, mem_offset(addr_reg, offset), value_reg)?;
            }
            LoadStoreKind::I64_8U => {
                emitter.mov_u8(0, mem_offset(addr_reg, offset), value_reg)?;
            }
            LoadStoreKind::I64_16U => {
                emitter.mov_u16(0, mem_offset(addr_reg, offset), value_reg)?;
            }
            LoadStoreKind::I64_32U => {
                emitter.mov32(0, mem_offset(addr_reg, offset), value_reg)?;
            }
            _ => unreachable!(),
        }
        self.free_register(value_reg);
        self.free_register(addr_reg);
        Ok(())
    }

    #[cfg(not(target_pointer_width = "64"))]
    fn compile_store_op64(
        &mut self,
        _emitter: &mut Emitter,
        _offset: i32,
        _kind: LoadStoreKind,
    ) -> Result<(), CompileError> {
        Err(CompileError::Unsupported(
            "64-bit memory operations not supported on 32-bit platform".into(),
        ))
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

        match (op, is_f32) {
            (FloatBinaryOp::Add, true) => {
                emitter.add_f32(0, freg_a, freg_a, freg_b)?;
            }
            (FloatBinaryOp::Add, false) => {
                emitter.add_f64(0, freg_a, freg_a, freg_b)?;
            }
            (FloatBinaryOp::Sub, true) => {
                emitter.sub_f32(0, freg_a, freg_a, freg_b)?;
            }
            (FloatBinaryOp::Sub, false) => {
                emitter.sub_f64(0, freg_a, freg_a, freg_b)?;
            }
            (FloatBinaryOp::Mul, true) => {
                emitter.mul_f32(0, freg_a, freg_a, freg_b)?;
            }
            (FloatBinaryOp::Mul, false) => {
                emitter.mul_f64(0, freg_a, freg_a, freg_b)?;
            }
            (FloatBinaryOp::Div, true) => {
                emitter.div_f32(0, freg_a, freg_a, freg_b)?;
            }
            (FloatBinaryOp::Div, false) => {
                emitter.div_f64(0, freg_a, freg_a, freg_b)?;
            }
            (FloatBinaryOp::Min, true) => {
                self.emit_float_binary_libm_call(
                    emitter,
                    freg_a,
                    freg_b,
                    {
                        extern "C" fn func(x: f32, y: f32) -> f32 {
                            x.min(y)
                        }
                        func as *const () as usize
                    },
                    true,
                )?;
            }
            (FloatBinaryOp::Min, false) => {
                self.emit_float_binary_libm_call(
                    emitter,
                    freg_a,
                    freg_b,
                    {
                        extern "C" fn func(x: f64, y: f64) -> f64 {
                            x.min(y)
                        }
                        func as *const () as usize
                    },
                    false,
                )?;
            }
            (FloatBinaryOp::Max, true) => {
                self.emit_float_binary_libm_call(
                    emitter,
                    freg_a,
                    freg_b,
                    {
                        extern "C" fn func(x: f32, y: f32) -> f32 {
                            x.max(y)
                        }
                        func as *const () as usize
                    },
                    true,
                )?;
            }
            (FloatBinaryOp::Max, false) => {
                self.emit_float_binary_libm_call(
                    emitter,
                    freg_a,
                    freg_b,
                    {
                        extern "C" fn func(x: f64, y: f64) -> f64 {
                            x.max(y)
                        }
                        func as *const () as usize
                    },
                    false,
                )?;
            }
            (FloatBinaryOp::Copysign, true) => {
                self.emit_float_binary_libm_call(
                    emitter,
                    freg_a,
                    freg_b,
                    {
                        extern "C" fn func(x: f32, y: f32) -> f32 {
                            x.copysign(y)
                        }
                        func as *const () as usize
                    },
                    true,
                )?;
            }
            (FloatBinaryOp::Copysign, false) => {
                self.emit_float_binary_libm_call(
                    emitter,
                    freg_a,
                    freg_b,
                    {
                        extern "C" fn func(x: f64, y: f64) -> f64 {
                            x.copysign(y)
                        }
                        func as *const () as usize
                    },
                    false,
                )?;
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

        match (op, is_f32) {
            (FloatUnaryOp::Neg, true) => {
                emitter.neg_f32(0, freg, freg)?;
            }
            (FloatUnaryOp::Neg, false) => {
                emitter.neg_f64(0, freg, freg)?;
            }
            (FloatUnaryOp::Abs, true) => {
                emitter.abs_f32(0, freg, freg)?;
            }
            (FloatUnaryOp::Abs, false) => {
                emitter.abs_f64(0, freg, freg)?;
            }
            // For Sqrt, Ceil, Floor, Trunc, Nearest - call C library functions
            (FloatUnaryOp::Sqrt, true) => {
                self.emit_float_libm_call(
                    emitter,
                    freg,
                    {
                        extern "C" fn func(x: f32) -> f32 {
                            x.sqrt()
                        }
                        func as *const () as usize
                    },
                    true,
                )?;
            }
            (FloatUnaryOp::Sqrt, false) => {
                self.emit_float_libm_call(
                    emitter,
                    freg,
                    {
                        extern "C" fn func(x: f64) -> f64 {
                            x.sqrt()
                        }
                        func as *const () as usize
                    },
                    false,
                )?;
            }
            (FloatUnaryOp::Ceil, true) => {
                self.emit_float_libm_call(
                    emitter,
                    freg,
                    {
                        extern "C" fn func(x: f32) -> f32 {
                            x.ceil()
                        }
                        func as *const () as usize
                    },
                    true,
                )?;
            }
            (FloatUnaryOp::Ceil, false) => {
                self.emit_float_libm_call(
                    emitter,
                    freg,
                    {
                        extern "C" fn func(x: f64) -> f64 {
                            x.ceil()
                        }
                        func as *const () as usize
                    },
                    false,
                )?;
            }
            (FloatUnaryOp::Floor, true) => {
                self.emit_float_libm_call(
                    emitter,
                    freg,
                    {
                        extern "C" fn func(x: f32) -> f32 {
                            x.floor()
                        }
                        func as *const () as usize
                    },
                    true,
                )?;
            }
            (FloatUnaryOp::Floor, false) => {
                self.emit_float_libm_call(
                    emitter,
                    freg,
                    {
                        extern "C" fn func(x: f64) -> f64 {
                            x.floor()
                        }
                        func as *const () as usize
                    },
                    false,
                )?;
            }
            (FloatUnaryOp::Trunc, true) => {
                self.emit_float_libm_call(
                    emitter,
                    freg,
                    {
                        extern "C" fn func(x: f32) -> f32 {
                            x.trunc()
                        }
                        func as *const () as usize
                    },
                    true,
                )?;
            }
            (FloatUnaryOp::Trunc, false) => {
                self.emit_float_libm_call(
                    emitter,
                    freg,
                    {
                        extern "C" fn func(x: f64) -> f64 {
                            x.trunc()
                        }
                        func as *const () as usize
                    },
                    false,
                )?;
            }
            (FloatUnaryOp::Nearest, true) => {
                self.emit_float_libm_call(
                    emitter,
                    freg,
                    {
                        extern "C" fn func(x: f32) -> f32 {
                            x.round_ties_even()
                        }
                        func as *const () as usize
                    },
                    true,
                )?;
            }
            (FloatUnaryOp::Nearest, false) => {
                self.emit_float_libm_call(
                    emitter,
                    freg,
                    {
                        extern "C" fn func(x: f64) -> f64 {
                            x.round_ties_even()
                        }
                        func as *const () as usize
                    },
                    false,
                )?;
            }
        }

        self.push(StackValue::FloatRegister(freg));
        Ok(())
    }

    /// Emit a call to a libm function for float binary operations
    /// The function takes two float/double arguments and returns float/double
    fn emit_float_binary_libm_call(
        &mut self,
        emitter: &mut Emitter,
        freg_a: FloatRegister,
        freg_b: FloatRegister,
        func_addr: usize,
        is_f32: bool,
    ) -> Result<(), CompileError> {
        // For calling C functions with two float arguments:
        // 1. Move the first float value to FR0 (first float argument register)
        // 2. Move the second float value to FR1 (second float argument register)
        // 3. Call the function
        // 4. Move the result from FR0 to our target register

        // Move inputs to FR0 and FR1
        if freg_a != FloatRegister::FR0 {
            if is_f32 {
                emitter.mov_f32(0, FloatRegister::FR0, freg_a)?;
            } else {
                emitter.mov_f64(0, FloatRegister::FR0, freg_a)?;
            }
        }
        if freg_b != FloatRegister::FR1 {
            if is_f32 {
                emitter.mov_f32(0, FloatRegister::FR1, freg_b)?;
            } else {
                emitter.mov_f64(0, FloatRegister::FR1, freg_b)?;
            }
        }

        // Load function address to a register and call
        let addr_reg = self.alloc_register(emitter)?;
        emitter.mov(0, addr_reg, func_addr as isize)?;

        // Make the indirect call - use the appropriate arg types for float functions
        let arg_types = if is_f32 {
            sys::SLJIT_ARG_TYPE_F32
                | (sys::SLJIT_ARG_TYPE_F32 << 4)
                | (sys::SLJIT_ARG_TYPE_F32 << 8)
        } else {
            sys::SLJIT_ARG_TYPE_F64
                | (sys::SLJIT_ARG_TYPE_F64 << 4)
                | (sys::SLJIT_ARG_TYPE_F64 << 8)
        };
        emitter.icall(JumpType::Call as i32, arg_types, addr_reg)?;
        self.free_register(addr_reg);

        // Move result from FR0 to target register if needed
        if freg_a != FloatRegister::FR0 {
            if is_f32 {
                emitter.mov_f32(0, freg_a, FloatRegister::FR0)?;
            } else {
                emitter.mov_f64(0, freg_a, FloatRegister::FR0)?;
            }
        }

        Ok(())
    }

    /// Emit a call to a libm function for float unary operations
    /// The function takes a float/double argument and returns float/double
    fn emit_float_libm_call(
        &mut self,
        emitter: &mut Emitter,
        freg: FloatRegister,
        func_addr: usize,
        is_f32: bool,
    ) -> Result<(), CompileError> {
        // For calling C functions, we need to:
        // 1. Move the float value to FR0 (first float argument register)
        // 2. Call the function
        // 3. Move the result from FR0 to our target register

        // Move input to FR0 if not already there
        if freg != FloatRegister::FR0 {
            if is_f32 {
                emitter.mov_f32(0, FloatRegister::FR0, freg)?;
            } else {
                emitter.mov_f64(0, FloatRegister::FR0, freg)?;
            }
        }

        // Load function address to a register and call
        let addr_reg = self.alloc_register(emitter)?;
        emitter.mov(0, addr_reg, func_addr as isize)?;

        // Make the indirect call - use the appropriate arg types for float functions
        let arg_types = if is_f32 {
            sys::SLJIT_ARG_TYPE_F32 | (sys::SLJIT_ARG_TYPE_F32 << 4)
        } else {
            sys::SLJIT_ARG_TYPE_F64 | (sys::SLJIT_ARG_TYPE_F64 << 4)
        };
        emitter.icall(JumpType::Call as i32, arg_types, addr_reg)?;
        self.free_register(addr_reg);

        // Move result from FR0 to target register if needed
        if freg != FloatRegister::FR0 {
            if is_f32 {
                emitter.mov_f32(0, freg, FloatRegister::FR0)?;
            } else {
                emitter.mov_f64(0, freg, FloatRegister::FR0)?;
            }
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
        let freg_a = self.ensure_in_float_register(emitter, a, is_f32)?;
        let freg_b = self.ensure_in_float_register(emitter, b, is_f32)?;
        let result_reg = self.alloc_register(emitter)?;

        // Use float compare instruction and conditional jump
        let mut jump_true = emitter.fcmp(cond, freg_a, freg_b)?;

        // False path (condition not met)
        emitter.mov(0, result_reg, 0i32)?;
        let mut jump_end = emitter.jump(JumpType::Jump)?;

        // True path (condition met)
        let mut label_true = emitter.put_label()?;
        jump_true.set_label(&mut label_true);
        emitter.mov(0, result_reg, 1i32)?;

        // End
        let mut label_end = emitter.put_label()?;
        jump_end.set_label(&mut label_end);

        self.free_float_register(freg_a);
        self.free_float_register(freg_b);
        self.push(StackValue::Register(result_reg));
        Ok(())
    }

    /// Convert integer to float
    /// is_f32: target is f32 (true) or f64 (false)
    /// is_i32: source is i32 (true) or i64 (false)
    /// is_signed: signed conversion (true) or unsigned (false)
    fn compile_convert_int_to_float(
        &mut self,
        emitter: &mut Emitter,
        is_f32: bool,
        is_i32: bool,
        is_signed: bool,
    ) -> Result<(), CompileError> {
        let val = self.pop_value()?;
        let reg = self.ensure_in_register(emitter, val)?;
        let freg = self.alloc_float_register(emitter)?;

        match (is_f32, is_i32, is_signed) {
            (true, true, true) => {
                // f32.convert_i32_s
                emitter.conv_f32_from_s32(0, freg, reg)?;
            }
            (true, true, false) => {
                // f32.convert_i32_u
                emitter.conv_f32_from_u32(0, freg, reg)?;
            }
            (false, true, true) => {
                // f64.convert_i32_s
                emitter.conv_f64_from_s32(0, freg, reg)?;
            }
            (false, true, false) => {
                // f64.convert_i32_u
                emitter.conv_f64_from_u32(0, freg, reg)?;
            }
            #[cfg(target_pointer_width = "64")]
            (true, false, true) => {
                // f32.convert_i64_s
                emitter.conv_f32_from_sw(0, freg, reg)?;
            }
            #[cfg(target_pointer_width = "64")]
            (true, false, false) => {
                // f32.convert_i64_u
                emitter.conv_f32_from_uw(0, freg, reg)?;
            }
            #[cfg(target_pointer_width = "64")]
            (false, false, true) => {
                // f64.convert_i64_s
                emitter.conv_f64_from_sw(0, freg, reg)?;
            }
            #[cfg(target_pointer_width = "64")]
            (false, false, false) => {
                // f64.convert_i64_u
                emitter.conv_f64_from_uw(0, freg, reg)?;
            }
            #[cfg(not(target_pointer_width = "64"))]
            (_, false, _) => {
                return Err(CompileError::Unsupported(
                    "64-bit int to float conversion not supported on 32-bit platform".into(),
                ));
            }
        }

        self.free_register(reg);
        self.push(StackValue::FloatRegister(freg));
        Ok(())
    }

    /// Convert float to integer (truncate)
    /// is_f32: source is f32 (true) or f64 (false)
    /// is_i32: target is i32 (true) or i64 (false)
    /// is_signed: signed conversion (true) or unsigned (false)
    fn compile_convert_float_to_int(
        &mut self,
        emitter: &mut Emitter,
        is_f32: bool,
        is_i32: bool,
        is_signed: bool,
    ) -> Result<(), CompileError> {
        let val = self.pop_value()?;
        let freg = self.ensure_in_float_register(emitter, val, is_f32)?;
        let reg = self.alloc_register(emitter)?;

        match (is_f32, is_i32, is_signed) {
            (true, true, true) => {
                // i32.trunc_f32_s
                emitter.conv_s32_from_f32(0, reg, freg)?;
            }
            (true, true, false) => {
                // i32.trunc_f32_u - SLJIT doesn't have direct unsigned conversion
                // We use signed conversion which works for positive values within range
                emitter.conv_s32_from_f32(0, reg, freg)?;
            }
            (false, true, true) => {
                // i32.trunc_f64_s
                emitter.conv_s32_from_f64(0, reg, freg)?;
            }
            (false, true, false) => {
                // i32.trunc_f64_u - SLJIT doesn't have direct unsigned conversion
                // We use signed conversion which works for positive values within range
                emitter.conv_s32_from_f64(0, reg, freg)?;
            }
            #[cfg(target_pointer_width = "64")]
            (true, false, true) => {
                // i64.trunc_f32_s
                emitter.conv_sw_from_f32(0, reg, freg)?;
            }
            #[cfg(target_pointer_width = "64")]
            (true, false, false) => {
                // i64.trunc_f32_u - SLJIT doesn't have direct unsigned conversion
                emitter.conv_sw_from_f32(0, reg, freg)?;
            }
            #[cfg(target_pointer_width = "64")]
            (false, false, true) => {
                // i64.trunc_f64_s
                emitter.conv_sw_from_f64(0, reg, freg)?;
            }
            #[cfg(target_pointer_width = "64")]
            (false, false, false) => {
                // i64.trunc_f64_u - SLJIT doesn't have direct unsigned conversion
                emitter.conv_sw_from_f64(0, reg, freg)?;
            }
            #[cfg(not(target_pointer_width = "64"))]
            (_, false, _) => {
                return Err(CompileError::Unsupported(
                    "Float to 64-bit int conversion not supported on 32-bit platform".into(),
                ));
            }
        }

        self.free_float_register(freg);
        self.push(StackValue::Register(reg));
        Ok(())
    }

    /// Compile f32.demote_f64 or f64.promote_f32
    /// is_demote: true for f32.demote_f64, false for f64.promote_f32
    fn compile_float_demote_promote(
        &mut self,
        emitter: &mut Emitter,
        is_demote: bool,
    ) -> Result<(), CompileError> {
        let val = self.pop_value()?;
        let src_is_f32 = !is_demote; // demote: src is f64, promote: src is f32
        let freg = self.ensure_in_float_register(emitter, val, src_is_f32)?;

        if is_demote {
            // f32.demote_f64: convert f64 to f32
            emitter.conv_f32_from_f64(0, freg, freg)?;
        } else {
            // f64.promote_f32: convert f32 to f64
            emitter.conv_f64_from_f32(0, freg, freg)?;
        }

        self.push(StackValue::FloatRegister(freg));
        Ok(())
    }
}

impl Default for WasmCompiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Compiled WebAssembly function
pub struct CompiledFunction {
    code: GeneratedCode,
}

impl CompiledFunction {
    pub fn as_fn_0(&self) -> fn() -> i32 {
        unsafe { std::mem::transmute(self.code.get()) }
    }
    pub fn as_fn_1(&self) -> fn(i32) -> i32 {
        unsafe { std::mem::transmute(self.code.get()) }
    }
    pub fn as_fn_2(&self) -> fn(i32, i32) -> i32 {
        unsafe { std::mem::transmute(self.code.get()) }
    }
    pub fn as_fn_3(&self) -> fn(i32, i32, i32) -> i32 {
        unsafe { std::mem::transmute(self.code.get()) }
    }
}

/// Compile a simple WebAssembly function from operators
pub fn compile_simple(
    params: &[ValType],
    results: &[ValType],
    locals: &[(u32, ValType)],
    body: &[Operator],
) -> Result<CompiledFunction, CompileError> {
    let mut compiler = Compiler::new();
    let mut wasm_compiler = WasmCompiler::new();
    wasm_compiler.compile_function(&mut compiler, params, results, locals, body)?;
    Ok(CompiledFunction {
        code: compiler.generate_code(),
    })
}

#[cfg(test)]
mod tests;
