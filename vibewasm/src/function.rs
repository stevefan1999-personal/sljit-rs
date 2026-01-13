use bitvec::prelude::*;
use derive_more::Deref;
use sljit::sys::{
    self, Fop1, Fop2, GeneratedCode, Op1, Op2, SLJIT_32, SLJIT_ARG_TYPE_F32, SLJIT_ARG_TYPE_F64,
    SLJIT_ARG_TYPE_W, arg_types, set,
};
use sljit::{
    Condition, Emitter, FloatRegister, JumpType, Operand, ReturnOp, SavedRegister, ScratchRegister,
    mem_offset, mem_sp_offset, regs,
};
use wasmparser::{Operator, ValType};

use crate::error::CompileError;
use crate::module::InternalFuncType;
use crate::{FunctionEntry, GlobalInfo, MemoryInfo, TableEntry, helpers};

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

/// WebAssembly function compiler
///
/// Owns its own sljit Compiler context for code generation.
#[derive(Debug)]
pub struct Function {
    emitter: Emitter,
    stack: Vec<StackValue>,
    blocks: Vec<Block>,
    locals: Vec<LocalVar>,
    param_count: usize,
    frame_offset: i32,
    local_size: i32,
    /// Bitmap tracking occupied scratch registers (bit i = 1 means register i is occupied)
    occupied_registers: BitArr!(for 3, in u8, Lsb0),
    /// Bitmap tracking occupied float registers (bit i = 1 means register i is occupied)
    occupied_float_registers: BitArr!(for 2, in u8, Lsb0),
    /// Global variables
    globals: Vec<GlobalInfo>,
    /// Memory instance
    memory: Option<MemoryInfo>,
    /// Function table for direct calls
    functions: Vec<FunctionEntry>,
    /// Tables for indirect calls
    tables: Vec<TableEntry>,
    /// Function types for call_indirect
    func_types: Vec<InternalFuncType>,
}

impl Clone for Function {
    fn clone(&self) -> Self {
        Self {
            emitter: Emitter::default(),
            stack: self.stack.clone(),
            blocks: self.blocks.clone(),
            locals: self.locals.clone(),
            param_count: self.param_count.clone(),
            frame_offset: self.frame_offset.clone(),
            local_size: self.local_size.clone(),
            occupied_registers: self.occupied_registers.clone(),
            occupied_float_registers: self.occupied_float_registers.clone(),
            globals: self.globals.clone(),
            memory: self.memory.clone(),
            functions: self.functions.clone(),
            tables: self.tables.clone(),
            func_types: self.func_types.clone(),
        }
    }
}

// Helper macro for stack underflow error
macro_rules! stack_underflow {
    () => {
        CompileError::Invalid("Stack underflow".into())
    };
}

impl Function {
    /// Create a new Function with its own compiler context
    pub fn new() -> Self {
        Self {
            emitter: Emitter::default(),
            stack: vec![],
            blocks: vec![],
            locals: vec![],
            param_count: 0,
            frame_offset: 0,
            local_size: 0,
            occupied_registers: BitArray::ZERO,
            occupied_float_registers: BitArray::ZERO,
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
    pub fn set_func_types(&mut self, func_types: Vec<InternalFuncType>) {
        self.func_types = func_types;
    }

    /// Pop a value from stack with error handling
    #[inline(always)]
    fn pop_value(&mut self) -> Result<StackValue, CompileError> {
        let val = self.stack.pop().ok_or_else(|| stack_underflow!())?;
        match &val {
            StackValue::Register(reg) => self.occupied_registers.set(reg.index(), false),
            StackValue::FloatRegister(reg) => self.occupied_float_registers.set(reg.index(), false),
            _ => {}
        };
        Ok(val)
    }

    /// Allocate a scratch register, spilling if necessary
    fn alloc_register(&mut self) -> Result<ScratchRegister, CompileError> {
        // Find first non-occupied register (only check first 6 bits for valid registers R0-R5)
        if let Some(idx) = self.occupied_registers[..3].iter_zeros().next() {
            self.occupied_registers.set(idx, true);
            // Convert index to ScratchRegister
            Ok(unsafe { core::mem::transmute(ScratchRegister::R0 as i32 + idx as i32) })
        } else {
            for sv in self.stack.iter_mut() {
                if let StackValue::Register(reg) = sv {
                    let offset = self.frame_offset;
                    self.frame_offset += 8;
                    self.emitter.mov(0, mem_sp_offset(offset), *reg)?;
                    let freed_reg = *reg;
                    *sv = StackValue::Stack(offset);
                    return Ok(freed_reg);
                }
            }
            Err(CompileError::Invalid("No registers available".into()))
        }
    }

    /// Allocate a float register, spilling if necessary
    fn alloc_float_register(&mut self) -> Result<FloatRegister, CompileError> {
        // Find first non-occupied float register
        if let Some(idx) = self.occupied_float_registers[..2].iter_zeros().next() {
            self.occupied_float_registers.set(idx, true);
            // Convert index to FloatRegister
            Ok(unsafe { core::mem::transmute(FloatRegister::FR0 as i32 + idx as i32) })
        } else {
            // Try to spill a float register from the stack
            for sv in self.stack.iter_mut() {
                if let StackValue::FloatRegister(reg) = sv {
                    let offset = self.frame_offset;
                    self.frame_offset += 8;
                    self.emitter.mov_f64(0, mem_sp_offset(offset), *reg)?;
                    let freed_reg = *reg;
                    *sv = StackValue::Stack(offset);
                    return Ok(freed_reg);
                }
            }
            Err(CompileError::Invalid("No float registers available".into()))
        }
    }

    /// Ensure a stack value is in a register
    fn ensure_in_register(&mut self, value: StackValue) -> Result<ScratchRegister, CompileError> {
        match value {
            StackValue::Register(reg) => {
                // The register may have been freed by pop_value, so we need to
                // mark it as occupied again to prevent it from being allocated again
                self.occupied_registers.set(reg.index(), true);
                Ok(reg)
            }
            StackValue::Stack(offset) => {
                let reg = self.alloc_register()?;
                self.emitter.mov(0, reg, mem_sp_offset(offset))?;
                Ok(reg)
            }
            StackValue::Const(val) => {
                let reg = self.alloc_register()?;
                self.emitter.mov(0, reg, val)?;
                Ok(reg)
            }
            StackValue::Saved(sreg) => {
                let reg = self.alloc_register()?;
                self.emitter.mov(0, reg, sreg)?;
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

        value: StackValue,
        is_f32: bool,
    ) -> Result<FloatRegister, CompileError> {
        match value {
            StackValue::FloatRegister(reg) => {
                // Mark as occupied to prevent reallocation
                self.occupied_float_registers.set(reg.index(), true);
                Ok(reg)
            }
            StackValue::Stack(offset) => {
                let reg = self.alloc_float_register()?;
                if is_f32 {
                    self.emitter.mov_f32(0, reg, mem_sp_offset(offset))?;
                } else {
                    self.emitter.mov_f64(0, reg, mem_sp_offset(offset))?;
                }
                Ok(reg)
            }
            StackValue::ConstF32(bits) => {
                let reg = self.alloc_float_register()?;
                // Use emit_fset32 to directly load the float constant into the register
                self.emitter.emit_fset32(reg, f32::from_bits(bits))?;
                Ok(reg)
            }
            StackValue::ConstF64(bits) => {
                let reg = self.alloc_float_register()?;
                // Use emit_fset64 to directly load the float constant into the register
                self.emitter.emit_fset64(reg, f64::from_bits(bits))?;
                Ok(reg)
            }
            _ => Err(CompileError::Invalid(
                "Expected float value on stack".into(),
            )),
        }
    }

    /// Emit mov from StackValue to a saved register
    fn emit_mov_to_saved(
        &mut self,
        sreg: SavedRegister,
        val: &StackValue,
    ) -> Result<(), CompileError> {
        match val {
            StackValue::Register(reg) => {
                self.emitter.mov(0, sreg, *reg)?;
            }
            StackValue::Stack(offset) => {
                self.emitter.mov(0, sreg, mem_sp_offset(*offset))?;
            }
            StackValue::Const(v) => {
                self.emitter.mov(0, sreg, *v)?;
            }
            StackValue::Saved(src) => {
                self.emitter.mov(0, sreg, *src)?;
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

        offset: i32,
        val: StackValue,
    ) -> Result<(), CompileError> {
        match val {
            StackValue::Register(reg) => {
                self.emitter.mov(0, mem_sp_offset(offset), reg)?;
                self.occupied_registers.set(reg.index(), false);
            }
            StackValue::Const(c) => {
                self.emitter.mov(0, mem_sp_offset(offset), c)?;
            }
            StackValue::Saved(sreg) => {
                self.emitter.mov(0, mem_sp_offset(offset), sreg)?;
            }
            StackValue::Stack(src_offset) if src_offset != offset => {
                self.emitter
                    .mov(0, mem_sp_offset(offset), mem_sp_offset(src_offset))?;
            }
            _ => {}
        }
        Ok(())
    }

    /// Emit mov to local variable (unified helper for LocalSet/LocalTee)
    fn emit_set_local(&mut self, local: &LocalVar, val: &StackValue) -> Result<(), CompileError> {
        match (local, val) {
            (LocalVar::Saved(sreg), _) => self.emit_mov_to_saved(*sreg, val),
            (LocalVar::Stack(offset), StackValue::Const(v)) => {
                self.emitter.mov(0, mem_sp_offset(*offset), *v)?;
                Ok(())
            }
            (LocalVar::Stack(offset), StackValue::Register(reg)) => {
                self.emitter.mov(0, mem_sp_offset(*offset), *reg)?;
                Ok(())
            }
            (LocalVar::Stack(offset), StackValue::Stack(src)) => {
                self.emitter
                    .mov(0, mem_sp_offset(*offset), mem_sp_offset(*src))?;
                Ok(())
            }
            (LocalVar::Stack(offset), StackValue::Saved(sreg)) => {
                self.emitter.mov(0, mem_sp_offset(*offset), *sreg)?;
                Ok(())
            }
            (LocalVar::Stack(offset), StackValue::FloatRegister(freg)) => {
                // Store float register to stack location
                self.emitter.mov_f64(0, mem_sp_offset(*offset), *freg)?;
                Ok(())
            }
            (LocalVar::Stack(offset), StackValue::ConstF32(bits)) => {
                // Store f32 constant to stack
                self.emitter
                    .mov32(0, mem_sp_offset(*offset), *bits as i32)?;
                Ok(())
            }
            (LocalVar::Stack(offset), StackValue::ConstF64(bits)) => {
                // Store f64 constant to stack
                #[cfg(target_pointer_width = "64")]
                {
                    self.emitter
                        .mov(0, mem_sp_offset(*offset), *bits as usize)?;
                }
                #[cfg(not(target_pointer_width = "64"))]
                {
                    self.emitter
                        .mov32(0, mem_sp_offset(*offset), (*bits & 0xFFFFFFFF) as i32)?;
                    self.emitter
                        .mov32(0, mem_sp_offset(*offset + 4), (*bits >> 32) as i32)?;
                }
                Ok(())
            }
        }
    }

    /// Save all register values to stack (for control flow)
    fn save_all_to_stack(&mut self) -> Result<(), CompileError> {
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
            self.emitter.mov(0, mem_sp_offset(offset), reg)?;
            self.occupied_registers.set(reg.index(), false);
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
        mut self,
        params: &[ValType],
        results: &[ValType],
        locals: &[ValType],
        body: B,
    ) -> Result<CompiledFunction, CompileError>
    where
        B: Iterator<Item = Operator<'a>>,
    {
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

        self.emitter.emit_enter(
            0,
            Self::make_arg_types(params, results),
            regs! { gp: 3, float: 2 },
            regs!(saved_count as i32),
            self.local_size,
        )?;

        // Initialize stack locals to zero
        for local in &self.locals {
            if let LocalVar::Stack(offset) = local {
                self.emitter.mov(0, mem_sp_offset(*offset), 0i32)?;
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
            self.compile_operator(&mut &op)?;
        }
        Ok(CompiledFunction {
            code: self.emitter.generate_code(),
        })
    }

    /// Compile a single WebAssembly operator
    fn compile_operator(&mut self, op: &Operator) -> Result<(), CompileError> {
        match op {
            Operator::I32Const { value } => {
                self.stack.push(StackValue::Const(*value));
                Ok(())
            }

            Operator::LocalGet { local_index } => {
                let local = &self.locals[*local_index as usize];
                self.stack.push(match local {
                    LocalVar::Saved(sreg) => StackValue::Saved(*sreg),
                    LocalVar::Stack(offset) => StackValue::Stack(*offset),
                });
                Ok(())
            }

            Operator::LocalSet { local_index } => {
                let value = self.pop_value()?;
                let local = self.locals[*local_index as usize].clone();
                self.emit_set_local(&local, &value)?;
                if let StackValue::Register(reg) = value {
                    self.occupied_registers.set(reg.index(), false);
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
                self.emit_set_local(&local, &value)
            }

            // Binary operations (i32)
            Operator::I32Add => self.compile_binary_op(Op2::Add32),
            Operator::I32Sub => self.compile_binary_op(Op2::Sub32),
            Operator::I32Mul => self.compile_binary_op(Op2::Mul32),
            Operator::I32And => self.compile_binary_op(Op2::And32),
            Operator::I32Or => self.compile_binary_op(Op2::Or32),
            Operator::I32Xor => self.compile_binary_op(Op2::Xor32),
            Operator::I32Shl => self.compile_binary_op(Op2::Shl32),
            Operator::I32ShrS => self.compile_binary_op(Op2::Ashr32),
            Operator::I32ShrU => self.compile_binary_op(Op2::Lshr32),
            Operator::I32Rotl => self.compile_binary_op(Op2::Rotl32),
            Operator::I32Rotr => self.compile_binary_op(Op2::Rotr32),

            // Division operations
            Operator::I32DivS => self.compile_div_op(DivOp::DivS),
            Operator::I32DivU => self.compile_div_op(DivOp::DivU),
            Operator::I32RemS => self.compile_div_op(DivOp::RemS),
            Operator::I32RemU => self.compile_div_op(DivOp::RemU),

            // Unary operations
            Operator::I32Clz => self.compile_unary_op(Op1::Clz32),
            Operator::I32Ctz => self.compile_unary_op(Op1::Ctz32),
            Operator::I32Popcnt => self.compile_unary_op_with_helper(UnaryOp::Popcnt),

            // Comparison operations
            Operator::I32Eqz => {
                let val = self.pop_value()?;
                let reg = self.ensure_in_register(val)?;
                self.emitter
                    .emit_op2u((Op2::Xor32, sys::SLJIT_SET_Z), reg, 0i32)?;
                self.emitter
                    .op_flags(sys::SLJIT_MOV32, reg, Condition::Equal as i32)?;
                self.stack.push(StackValue::Register(reg));
                Ok(())
            }
            Operator::I32Eq => self.compile_compare_op(Condition::Equal),
            Operator::I32Ne => self.compile_compare_op(Condition::NotEqual),
            Operator::I32LtS => self.compile_compare_op(Condition::SigLess32),
            Operator::I32LtU => self.compile_compare_op(Condition::Less32),
            Operator::I32GtS => self.compile_compare_op(Condition::SigGreater32),
            Operator::I32GtU => self.compile_compare_op(Condition::Greater32),
            Operator::I32LeS => self.compile_compare_op(Condition::SigLessEqual32),
            Operator::I32LeU => self.compile_compare_op(Condition::LessEqual32),
            Operator::I32GeS => self.compile_compare_op(Condition::SigGreaterEqual32),
            Operator::I32GeU => self.compile_compare_op(Condition::GreaterEqual32),

            // Control flow
            Operator::Block { blockty } => {
                self.save_all_to_stack()?;
                let block = self.new_block(None, Self::block_has_result(blockty));
                self.blocks.push(block);
                Ok(())
            }

            Operator::Loop { blockty: _ } => {
                self.save_all_to_stack()?;
                let label = self.emitter.put_label()?;
                let block = self.new_block(Some(label), false);
                self.blocks.push(block);
                Ok(())
            }

            Operator::If { blockty } => {
                let cond = self.pop_value()?;
                self.save_all_to_stack()?;
                let jump = self.emit_cond_jump(cond, Condition::Equal)?;
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
                    self.emit_mov_to_stack_offset(offset, val)?;
                }

                let jump_to_end = self.emitter.jump(JumpType::Jump)?;
                let block = self
                    .blocks
                    .last_mut()
                    .ok_or_else(|| CompileError::Invalid("No block for else".into()))?;
                block.end_jumps.push(jump_to_end);

                if let Some(mut else_jump) = block.else_jump.take() {
                    let mut label = self.emitter.put_label()?;
                    else_jump.set_label(&mut label);
                }
                self.stack.truncate(stack_depth);
                Ok(())
            }

            Operator::End => self.compile_end(),

            Operator::Br { relative_depth } => self.compile_br(*relative_depth),
            Operator::BrIf { relative_depth } => self.compile_br_if(*relative_depth),

            Operator::Return => {
                if let Some(value) = self.stack.pop() {
                    if let StackValue::Register(reg) = &value {
                        self.occupied_registers.set(reg.index(), false);
                    }
                    self.emitter
                        .emit_return(ReturnOp::Mov, value.to_operand())?;
                } else {
                    self.emitter.return_void()?;
                }
                Ok(())
            }

            Operator::Drop => {
                self.pop_value()?;
                Ok(())
            }

            Operator::Select => {
                let (cond, val2, val1) = (self.pop_value()?, self.pop_value()?, self.pop_value()?);
                let reg1 = self.ensure_in_register(val1)?;

                self.emitter
                    .emit_op2u((Op2::Xor, sys::SLJIT_SET_Z), cond.to_operand(), 0i32)?;
                self.emitter
                    .select(Condition::Equal, reg1, val2.to_operand(), reg1)?;
                self.stack.push(StackValue::Register(reg1));
                Ok(())
            }

            Operator::Nop => Ok(()),
            Operator::Unreachable => {
                self.emitter.breakpoint()?;
                Ok(())
            }

            // Memory operations
            Operator::I32Load { memarg } => self.compile_load_op(memarg.offset as i32, Op1::Mov32),
            Operator::I32Load8S { memarg } => {
                self.compile_load_op(memarg.offset as i32, Op1::MovS8)
            }
            Operator::I32Load8U { memarg } => {
                self.compile_load_op(memarg.offset as i32, Op1::MovU8)
            }
            Operator::I32Load16S { memarg } => {
                self.compile_load_op(memarg.offset as i32, Op1::MovS16)
            }
            Operator::I32Load16U { memarg } => {
                self.compile_load_op(memarg.offset as i32, Op1::MovU16)
            }
            Operator::I32Store { memarg } => {
                self.compile_store_op(memarg.offset as i32, Op1::Mov32)
            }
            Operator::I32Store8 { memarg } => {
                self.compile_store_op(memarg.offset as i32, Op1::MovU8)
            }
            Operator::I32Store16 { memarg } => {
                self.compile_store_op(memarg.offset as i32, Op1::MovU16)
            }

            #[cfg(target_pointer_width = "64")]
            Operator::I64Const { value } => {
                let reg = self.alloc_register()?;
                self.emitter.mov(0, reg, *value as usize)?;
                self.stack.push(StackValue::Register(reg));
                Ok(())
            }

            #[cfg(not(target_pointer_width = "64"))]
            Operator::I64Const { value } => {
                // On 32-bit platforms, store both halves on the stack
                let offset = self.frame_offset;
                self.frame_offset += 8;
                // Store lower 32 bits
                self.emitter
                    .mov32(0, mem_sp_offset(offset), (*value & 0xFFFFFFFF) as i32)?;
                // Store upper 32 bits
                self.emitter
                    .mov32(0, mem_sp_offset(offset + 4), (*value >> 32) as i32)?;
                self.stack.push(StackValue::Stack(offset));
                Ok(())
            }
            Operator::I64Add => self.compile_binary_op64(Op2::Add),
            Operator::I64Sub => self.compile_binary_op64(Op2::Sub),
            Operator::I64Mul => self.compile_binary_op64(Op2::Mul),
            Operator::I64And => self.compile_binary_op64(Op2::And),
            Operator::I64Or => self.compile_binary_op64(Op2::Or),
            Operator::I64Xor => self.compile_binary_op64(Op2::Xor),
            Operator::I64Shl => self.compile_binary_op64(Op2::Shl),
            Operator::I64ShrS => self.compile_binary_op64(Op2::Ashr),
            Operator::I64ShrU => self.compile_binary_op64(Op2::Lshr),
            Operator::I64Rotl => self.compile_binary_op64(Op2::Rotl),
            Operator::I64Rotr => self.compile_binary_op64(Op2::Rotr),

            // I64 division operations
            Operator::I64DivS => self.compile_div_op64(DivOp::DivS),
            Operator::I64DivU => self.compile_div_op64(DivOp::DivU),
            Operator::I64RemS => self.compile_div_op64(DivOp::RemS),
            Operator::I64RemU => self.compile_div_op64(DivOp::RemU),

            // I64 unary operations
            Operator::I64Clz => self.compile_unary_op_with_helper(UnaryOp::Clz64),
            Operator::I64Ctz => self.compile_unary_op_with_helper(UnaryOp::Ctz64),
            Operator::I64Popcnt => self.compile_unary_op_with_helper(UnaryOp::Popcnt64),

            // I64 comparison operations
            Operator::I64Eqz => {
                let val = self.pop_value()?;
                let reg = self.ensure_in_register(val)?;
                self.emitter
                    .emit_op2u((Op2::Xor, sys::SLJIT_SET_Z), reg, 0i32)?;
                self.emitter
                    .op_flags(sys::SLJIT_MOV, reg, Condition::Equal as i32)?;
                self.stack.push(StackValue::Register(reg));
                Ok(())
            }
            Operator::I64Eq => self.compile_compare_op64(Condition::Equal),
            Operator::I64Ne => self.compile_compare_op64(Condition::NotEqual),
            Operator::I64LtS => self.compile_compare_op64(Condition::SigLess),
            Operator::I64LtU => self.compile_compare_op64(Condition::Less),
            Operator::I64GtS => self.compile_compare_op64(Condition::SigGreater),
            Operator::I64GtU => self.compile_compare_op64(Condition::Greater),
            Operator::I64LeS => self.compile_compare_op64(Condition::SigLessEqual),
            Operator::I64LeU => self.compile_compare_op64(Condition::LessEqual),
            Operator::I64GeS => self.compile_compare_op64(Condition::SigGreaterEqual),
            Operator::I64GeU => self.compile_compare_op64(Condition::GreaterEqual),

            // I64 memory operations
            Operator::I64Load { memarg } => self.compile_load_op64(memarg.offset as i32, Op1::Mov),
            Operator::I64Load8S { memarg } => {
                self.compile_load_op64(memarg.offset as i32, Op1::MovS8)
            }
            Operator::I64Load8U { memarg } => {
                self.compile_load_op64(memarg.offset as i32, Op1::MovU8)
            }
            Operator::I64Load16S { memarg } => {
                self.compile_load_op64(memarg.offset as i32, Op1::MovS16)
            }
            Operator::I64Load16U { memarg } => {
                self.compile_load_op64(memarg.offset as i32, Op1::MovU16)
            }
            Operator::I64Load32S { memarg } => {
                self.compile_load_op64(memarg.offset as i32, Op1::MovS32)
            }
            Operator::I64Load32U { memarg } => {
                self.compile_load_op64(memarg.offset as i32, Op1::MovU32)
            }
            Operator::I64Store { memarg } => {
                self.compile_store_op64(memarg.offset as i32, Op1::Mov)
            }
            Operator::I64Store8 { memarg } => {
                self.compile_store_op64(memarg.offset as i32, Op1::MovU8)
            }
            Operator::I64Store16 { memarg } => {
                self.compile_store_op64(memarg.offset as i32, Op1::MovU16)
            }
            Operator::I64Store32 { memarg } => {
                self.compile_store_op64(memarg.offset as i32, Op1::Mov32)
            }

            Operator::I32WrapI64 => {
                let val = self.pop_value()?;
                self.emitter.mov32(0, val.to_operand(), val.to_operand())?;
                self.stack.push(val);
                Ok(())
            }
            #[cfg(target_pointer_width = "64")]
            Operator::I64ExtendI32S => {
                let val = self.pop_value()?;
                let reg = self.ensure_in_register(val)?;
                self.emitter.mov_s32(0, reg, reg)?;
                self.stack.push(StackValue::Register(reg));
                Ok(())
            }

            #[cfg(not(target_pointer_width = "64"))]
            Operator::I64ExtendI32S => {
                // On 32-bit: extend i32 to i64 (sign extend)
                let val = self.pop_value()?;
                let reg = self.ensure_in_register(val)?;

                // Allocate stack space for i64 result
                let offset = self.frame_offset;
                self.frame_offset += 8;

                // Store low part (original i32 value)
                self.emitter.mov32(0, mem_sp_offset(offset), reg)?;

                // Sign extend to high part
                self.emitter.ashr32(0, reg, reg, 31i32)?;
                self.emitter.mov32(0, mem_sp_offset(offset + 4), reg)?;

                self.occupied_registers.set(reg.index(), false);
                self.stack.push(StackValue::Stack(offset));
                Ok(())
            }

            #[cfg(target_pointer_width = "64")]
            Operator::I64ExtendI32U => {
                let val = self.pop_value()?;
                let reg = self.ensure_in_register(val)?;
                self.emitter.mov_u32(0, reg, reg)?;
                self.stack.push(StackValue::Register(reg));
                Ok(())
            }

            #[cfg(not(target_pointer_width = "64"))]
            Operator::I64ExtendI32U => {
                // On 32-bit: extend i32 to i64 (zero extend)
                let val = self.pop_value()?;
                let reg = self.ensure_in_register(val)?;

                // Allocate stack space for i64 result
                let offset = self.frame_offset;
                self.frame_offset += 8;

                // Store low part (original i32 value)
                self.emitter.mov32(0, mem_sp_offset(offset), reg)?;

                // Zero extend to high part
                self.emitter.mov32(0, mem_sp_offset(offset + 4), 0i32)?;

                self.occupied_registers.set(reg.index(), false);
                self.stack.push(StackValue::Stack(offset));
                Ok(())
            }

            // Floating point constants
            Operator::F32Const { value } => {
                self.stack.push(StackValue::ConstF32(value.bits()));
                Ok(())
            }
            Operator::F64Const { value } => {
                self.stack.push(StackValue::ConstF64(value.bits()));
                Ok(())
            }

            // Floating point load/store operations
            Operator::F32Load { memarg } => {
                self.compile_float_load_op(memarg.offset as i32, Fop1::MovF32)
            }
            Operator::F64Load { memarg } => {
                self.compile_float_load_op(memarg.offset as i32, Fop1::MovF64)
            }
            Operator::F32Store { memarg } => {
                self.compile_float_store_op(memarg.offset as i32, Fop1::MovF32)
            }
            Operator::F64Store { memarg } => {
                self.compile_float_store_op(memarg.offset as i32, Fop1::MovF64)
            }

            // F32 binary operations
            Operator::F32Add => self.compile_float_binary_op(Fop2::AddF32, true),
            Operator::F32Sub => self.compile_float_binary_op(Fop2::SubF32, true),
            Operator::F32Mul => self.compile_float_binary_op(Fop2::MulF32, true),
            Operator::F32Div => self.compile_float_binary_op(Fop2::DivF32, true),
            Operator::F32Min => self.compile_float_binary_op_with_helper(FloatBinaryOp::Min, true),
            Operator::F32Max => self.compile_float_binary_op_with_helper(FloatBinaryOp::Max, true),
            Operator::F32Copysign => {
                self.compile_float_binary_op_with_helper(FloatBinaryOp::Copysign, true)
            }

            // F64 binary operations
            Operator::F64Add => self.compile_float_binary_op(Fop2::AddF64, false),
            Operator::F64Sub => self.compile_float_binary_op(Fop2::SubF64, false),
            Operator::F64Mul => self.compile_float_binary_op(Fop2::MulF64, false),
            Operator::F64Div => self.compile_float_binary_op(Fop2::DivF64, false),
            Operator::F64Min => self.compile_float_binary_op_with_helper(FloatBinaryOp::Min, false),
            Operator::F64Max => self.compile_float_binary_op_with_helper(FloatBinaryOp::Max, false),
            Operator::F64Copysign => {
                self.compile_float_binary_op_with_helper(FloatBinaryOp::Copysign, false)
            }

            // F32 unary operations
            Operator::F32Neg => self.compile_float_unary_op(Fop1::NegF32, true),
            Operator::F32Abs => self.compile_float_unary_op(Fop1::AbsF32, true),
            Operator::F32Sqrt => self.compile_float_unary_op_with_helper(FloatUnaryOp::Sqrt, true),
            Operator::F32Ceil => self.compile_float_unary_op_with_helper(FloatUnaryOp::Ceil, true),
            Operator::F32Floor => {
                self.compile_float_unary_op_with_helper(FloatUnaryOp::Floor, true)
            }
            Operator::F32Trunc => {
                self.compile_float_unary_op_with_helper(FloatUnaryOp::Trunc, true)
            }
            Operator::F32Nearest => {
                self.compile_float_unary_op_with_helper(FloatUnaryOp::Nearest, true)
            }

            // F64 unary operations
            Operator::F64Neg => self.compile_float_unary_op(Fop1::NegF64, false),
            Operator::F64Abs => self.compile_float_unary_op(Fop1::AbsF64, false),
            Operator::F64Sqrt => self.compile_float_unary_op_with_helper(FloatUnaryOp::Sqrt, false),
            Operator::F64Ceil => self.compile_float_unary_op_with_helper(FloatUnaryOp::Ceil, false),
            Operator::F64Floor => {
                self.compile_float_unary_op_with_helper(FloatUnaryOp::Floor, false)
            }
            Operator::F64Trunc => {
                self.compile_float_unary_op_with_helper(FloatUnaryOp::Trunc, false)
            }
            Operator::F64Nearest => {
                self.compile_float_unary_op_with_helper(FloatUnaryOp::Nearest, false)
            }

            // Float comparisons - F32
            Operator::F32Eq => self.compile_float_compare_op(Condition::FEqual, true),
            Operator::F32Ne => self.compile_float_compare_op(Condition::FNotEqual, true),
            Operator::F32Lt => self.compile_float_compare_op(Condition::FLess, true),
            Operator::F32Gt => self.compile_float_compare_op(Condition::FGreater, true),
            Operator::F32Le => self.compile_float_compare_op(Condition::FLessEqual, true),
            Operator::F32Ge => self.compile_float_compare_op(Condition::FGreaterEqual, true),

            // Float comparisons - F64
            Operator::F64Eq => self.compile_float_compare_op(Condition::FEqual, false),
            Operator::F64Ne => self.compile_float_compare_op(Condition::FNotEqual, false),
            Operator::F64Lt => self.compile_float_compare_op(Condition::FLess, false),
            Operator::F64Gt => self.compile_float_compare_op(Condition::FGreater, false),
            Operator::F64Le => self.compile_float_compare_op(Condition::FLessEqual, false),
            Operator::F64Ge => self.compile_float_compare_op(Condition::FGreaterEqual, false),

            // Conversions: int to float
            Operator::F32ConvertI32S => self.compile_convert_int_to_float(true, true, true),
            Operator::F32ConvertI32U => self.compile_convert_int_to_float(true, true, false),
            Operator::F64ConvertI32S => self.compile_convert_int_to_float(false, true, true),
            Operator::F64ConvertI32U => self.compile_convert_int_to_float(false, true, false),
            Operator::F32ConvertI64S => self.compile_convert_int_to_float(true, false, true),
            Operator::F32ConvertI64U => self.compile_convert_int_to_float(true, false, false),
            Operator::F64ConvertI64S => self.compile_convert_int_to_float(false, false, true),
            Operator::F64ConvertI64U => self.compile_convert_int_to_float(false, false, false),

            // Conversions: float to int (truncate)
            Operator::I32TruncF32S => self.compile_convert_float_to_int(true, true, true),
            Operator::I32TruncF32U => self.compile_convert_float_to_int(true, true, false),
            Operator::I32TruncF64S => self.compile_convert_float_to_int(false, true, true),
            Operator::I32TruncF64U => self.compile_convert_float_to_int(false, true, false),
            Operator::I64TruncF32S => self.compile_convert_float_to_int(true, false, true),
            Operator::I64TruncF32U => self.compile_convert_float_to_int(true, false, false),
            Operator::I64TruncF64S => self.compile_convert_float_to_int(false, false, true),
            Operator::I64TruncF64U => self.compile_convert_float_to_int(false, false, false),

            // Conversions: float to float
            Operator::F32DemoteF64 => self.compile_float_demote_promote(true),
            Operator::F64PromoteF32 => self.compile_float_demote_promote(false),

            // Reinterpret operations
            Operator::I32ReinterpretF32 => {
                let val = self.pop_value()?;
                let freg = self.ensure_in_float_register(val, true)?;
                let reg = self.alloc_register()?;
                // Store float to stack, load as int (ensure 8-byte alignment)
                let temp_offset = (self.frame_offset + 7) & !7;
                self.frame_offset = temp_offset + 8;
                self.emitter.mov_f32(0, mem_sp_offset(temp_offset), freg)?;
                self.emitter.mov32(0, reg, mem_sp_offset(temp_offset))?;
                self.occupied_float_registers.set(freg.index(), false);
                self.stack.push(StackValue::Register(reg));
                Ok(())
            }
            Operator::I64ReinterpretF64 => {
                let val = self.pop_value()?;
                let freg = self.ensure_in_float_register(val, false)?;
                // Ensure 8-byte alignment for f64 operations
                let temp_offset = (self.frame_offset + 7) & !7;
                self.frame_offset = temp_offset + 8;
                self.emitter.mov_f64(0, mem_sp_offset(temp_offset), freg)?;
                self.occupied_float_registers.set(freg.index(), false);

                #[cfg(target_pointer_width = "64")]
                {
                    let reg = self.alloc_register()?;
                    self.emitter.mov(0, reg, mem_sp_offset(temp_offset))?;
                    self.stack.push(StackValue::Register(reg));
                }
                #[cfg(not(target_pointer_width = "64"))]
                {
                    // On 32-bit, the i64 value stays on stack (already there from mov_f64)
                    self.stack.push(StackValue::Stack(temp_offset));
                }
                Ok(())
            }
            Operator::F32ReinterpretI32 => {
                let val = self.pop_value()?;
                let freg = self.alloc_float_register()?;
                // Ensure 8-byte alignment for float operations
                let temp_offset = (self.frame_offset + 7) & !7;
                self.frame_offset = temp_offset + 8;
                self.emitter
                    .mov32(0, mem_sp_offset(temp_offset), val.to_operand())?;
                self.emitter
                    .mov_f32(SLJIT_32, freg, mem_sp_offset(temp_offset))?;
                self.stack.push(StackValue::FloatRegister(freg));
                Ok(())
            }
            Operator::F64ReinterpretI64 => {
                let val = self.pop_value()?;
                let freg = self.alloc_float_register()?;
                // Ensure 8-byte alignment for f64 operations
                let temp_offset = (self.frame_offset + 7) & !7;
                self.frame_offset = temp_offset + 8;

                #[cfg(target_pointer_width = "64")]
                {
                    self.emitter
                        .mov(0, mem_sp_offset(temp_offset), val.to_operand())?;
                    self.emitter.mov_f64(0, freg, mem_sp_offset(temp_offset))?;
                }
                #[cfg(not(target_pointer_width = "64"))]
                {
                    // On 32-bit, i64 value is already on stack
                    match val {
                        StackValue::Stack(value_offset) => {
                            self.emitter.mov_f64(0, freg, mem_sp_offset(value_offset))?;
                        }
                        _ => {
                            return Err(CompileError::Invalid(
                                "I64 value must be on stack on 32-bit platform".into(),
                            ));
                        }
                    }
                }

                self.stack.push(StackValue::FloatRegister(freg));
                Ok(())
            }

            // Global variable operations
            Operator::GlobalGet { global_index } => self.compile_global_get(*global_index),
            Operator::GlobalSet { global_index } => self.compile_global_set(*global_index),

            // Memory operations
            Operator::MemorySize { mem, .. } => self.compile_memory_size(*mem),
            Operator::MemoryGrow { mem, .. } => self.compile_memory_grow(*mem),

            // Function calls
            Operator::Call { function_index } => self.compile_call(*function_index),
            Operator::CallIndirect {
                type_index,
                table_index,
            } => self.compile_call_indirect(*type_index, *table_index),

            // Branch table
            Operator::BrTable { targets } => self.compile_br_table(targets),

            _ => Err(CompileError::Unsupported(format!(
                "Unsupported operator: {:?}",
                op
            ))),
        }
    }

    /// Helper to emit conditional jump (used by If and BrIf)
    fn emit_cond_jump(
        &mut self,

        cond: StackValue,
        condition: Condition,
    ) -> Result<sys::Jump, CompileError> {
        match cond {
            StackValue::Register(reg) => {
                let j = self.emitter.cmp(condition, reg, 0i32)?;
                self.occupied_registers.set(reg.index(), false);
                Ok(j)
            }
            StackValue::Saved(sreg) => Ok(self.emitter.cmp(condition, sreg, 0i32)?),
            _ => Ok(self.emitter.cmp(condition, cond.to_operand(), 0i32)?),
        }
    }

    fn compile_end(&mut self) -> Result<(), CompileError> {
        if self.blocks.len() == 1 {
            let block = self.blocks.pop().unwrap();
            let mut label = self.emitter.put_label()?;
            for mut jump in block.end_jumps {
                jump.set_label(&mut label);
            }
            if let Some(mut else_jump) = block.else_jump {
                else_jump.set_label(&mut label);
            }

            if let Some(value) = self.stack.pop() {
                if let StackValue::Register(reg) = &value {
                    self.occupied_registers.set(reg.index(), false);
                }
                self.emitter
                    .emit_return(ReturnOp::Mov, value.to_operand())?;
            } else {
                self.emitter.return_void()?;
            }
        } else {
            let block = self
                .blocks
                .pop()
                .ok_or_else(|| CompileError::Invalid("No block for end".into()))?;

            if let Some(offset) = block.result_offset {
                if self.stack.len() > block.stack_depth {
                    let val = self.stack.pop().unwrap();
                    self.emit_mov_to_stack_offset(offset, val)?;
                }
                self.stack.truncate(block.stack_depth);
                self.stack.push(StackValue::Stack(offset));
            }

            let mut label = self.emitter.put_label()?;
            for mut jump in block.end_jumps {
                jump.set_label(&mut label);
            }
            if let Some(mut else_jump) = block.else_jump {
                else_jump.set_label(&mut label);
            }
        }
        Ok(())
    }

    fn compile_br(&mut self, relative_depth: u32) -> Result<(), CompileError> {
        let block_idx = self.blocks.len() - 1 - relative_depth as usize;

        if let Some(result_offset) = self.blocks[block_idx].result_offset
            && let Some(val) = self.stack.pop()
        {
            if let StackValue::Register(reg) = &val {
                self.occupied_registers.set(reg.index(), false);
            }
            self.emit_mov_to_stack_offset(result_offset, val)?;
        }

        self.save_all_to_stack()?;

        let mut jump = self.emitter.jump(JumpType::Jump)?;

        if let Some(mut label) = self.blocks[block_idx].label {
            jump.set_label(&mut label);
        } else {
            self.blocks[block_idx].end_jumps.push(jump);
        }
        Ok(())
    }

    fn compile_br_if(&mut self, relative_depth: u32) -> Result<(), CompileError> {
        let cond = self.pop_value()?;
        self.save_all_to_stack()?;

        let block_idx = self.blocks.len() - 1 - relative_depth as usize;
        let mut jump = self.emit_cond_jump(cond, Condition::NotEqual)?;

        if let Some(mut label) = self.blocks[block_idx].label {
            jump.set_label(&mut label);
        } else {
            self.blocks[block_idx].end_jumps.push(jump);
        }
        Ok(())
    }

    /// Compile a binary arithmetic operation using functional dispatch
    fn compile_binary_op(&mut self, op: Op2) -> Result<(), CompileError> {
        let (b, a) = (self.pop_value()?, self.pop_value()?);

        // If b is a register, keep it occupied while we prepare a to avoid conflicts
        if let StackValue::Register(reg_b) = &b {
            self.occupied_registers.set(reg_b.index(), true);
        }

        let reg_a = self.ensure_in_register(a)?;
        let operand_b = b.to_operand();
        self.emitter.emit_op2((op, 0), reg_a, reg_a, operand_b)?;

        // Now free b's register
        if let StackValue::Register(reg_b) = b {
            self.occupied_registers.set(reg_b.index(), false);
        }
        self.stack.push(StackValue::Register(reg_a));
        Ok(())
    }

    fn compile_compare_op(&mut self, cond: Condition) -> Result<(), CompileError> {
        let flags = match cond {
            Condition::Equal | Condition::NotEqual => sys::SLJIT_SET_Z,
            Condition::Less
            | Condition::GreaterEqual
            | Condition::Less32
            | Condition::GreaterEqual32 => set(sys::SLJIT_LESS),
            Condition::Greater
            | Condition::LessEqual
            | Condition::Greater32
            | Condition::LessEqual32 => set(sys::SLJIT_GREATER),
            Condition::SigLess
            | Condition::SigGreaterEqual
            | Condition::SigLess32
            | Condition::SigGreaterEqual32 => set(sys::SLJIT_SIG_LESS),
            Condition::SigGreater
            | Condition::SigLessEqual
            | Condition::SigGreater32
            | Condition::SigLessEqual32 => set(sys::SLJIT_SIG_GREATER),
            _ => {
                return Err(CompileError::Unsupported(format!(
                    "Unsupported condition: {:?}",
                    cond
                )));
            }
        };

        let (b, a) = (self.pop_value()?, self.pop_value()?);
        let reg_a = self.ensure_in_register(a)?;
        // Map condition to the appropriate SET flag - use SUB32 for 32-bit comparisons
        self.emitter
            .emit_op2u((Op2::Sub32, flags), reg_a, b.to_operand())?;

        // Convert flags to 0/1 using op_flags - use MOV32 for 32-bit result
        self.emitter
            .op_flags(sys::SLJIT_MOV32, reg_a, (cond as i32) & !SLJIT_32)?;

        if let StackValue::Register(reg_b) = b {
            self.occupied_registers.set(reg_b.index(), false);
        }
        self.stack.push(StackValue::Register(reg_a));
        Ok(())
    }

    fn compile_div_op(&mut self, op: DivOp) -> Result<(), CompileError> {
        let (b, a) = (self.pop_value()?, self.pop_value()?);
        let (reg_a, reg_b) = (self.ensure_in_register(a)?, self.ensure_in_register(b)?);

        if reg_a != ScratchRegister::R0 {
            self.emitter.mov(0, ScratchRegister::R0, reg_a)?;
        }
        if reg_b != ScratchRegister::R1 {
            self.emitter.mov(0, ScratchRegister::R1, reg_b)?;
        }

        [reg_a, reg_b]
            .iter()
            .filter(|&&r| r != ScratchRegister::R0 && r != ScratchRegister::R1)
            .for_each(|&r| self.occupied_registers.set(r.index(), false));

        match op {
            DivOp::DivS | DivOp::RemS => {
                self.emitter.divmod_s32()?;
            }
            DivOp::DivU | DivOp::RemU => {
                self.emitter.divmod_u32()?;
            }
        }

        let (result_reg, other_reg) = match op {
            DivOp::DivS | DivOp::DivU => (ScratchRegister::R0, ScratchRegister::R1),
            DivOp::RemS | DivOp::RemU => (ScratchRegister::R1, ScratchRegister::R0),
        };

        self.occupied_registers.set(other_reg.index(), false);
        self.occupied_registers.set(result_reg.index(), true);
        self.stack.push(StackValue::Register(result_reg));
        Ok(())
    }

    fn compile_unary_op(&mut self, op: Op1) -> Result<(), CompileError> {
        let val = self.pop_value()?;
        let reg = self.ensure_in_register(val)?;
        self.emitter.emit_op1((op, 0), reg, reg)?;
        self.stack.push(StackValue::Register(reg));
        Ok(())
    }

    fn compile_unary_op_with_helper(&mut self, op: UnaryOp) -> Result<(), CompileError> {
        #[cfg(not(target_pointer_width = "64"))]
        let mut helper = |func_addr: usize| {
            let val = self.pop_value()?;
            // Get stack offset for the operand
            let offset_a = self.ensure_i64_on_stack(val)?;

            // Allocate result space on stack
            let result_offset = self.frame_offset;
            self.frame_offset += 8;

            // Compute pointers: R0 = &a, R1 = &result
            self.emitter.get_local_base(ScratchRegister::R0, offset_a)?;
            self.emitter
                .get_local_base(ScratchRegister::R1, result_offset)?;

            // arg_types for (W, W) -> void (2 pointer arguments, void return)
            self.emitter
                .icall(JumpType::Call as i32, arg_types!([W, W]), func_addr)?;

            // Result is now at result_offset on stack
            self.stack.push(StackValue::Stack(result_offset));
            Ok::<_, CompileError>(())
        };

        match op {
            UnaryOp::Popcnt => {
                let val = self.pop_value()?;
                let reg = self.ensure_in_register(val)?;
                self.emitter.mov(0, ScratchRegister::R0, reg)?;
                self.emitter.icall(
                    JumpType::Call as i32,
                    arg_types!([W] -> W),
                    helpers::popcnt32 as *const () as usize,
                )?;
                self.emitter.mov(0, reg, ScratchRegister::R0)?;
                self.stack.push(StackValue::Register(reg));
            }
            // On 64-bit targets, we can directly pass the value in registers
            #[cfg(target_pointer_width = "64")]
            UnaryOp::Popcnt64 => {
                let val = self.pop_value()?;
                let reg = self.ensure_in_register(val)?;
                self.emitter.mov(0, ScratchRegister::R0, reg)?;
                self.emitter.icall(
                    JumpType::Call as i32,
                    arg_types!([W] -> W),
                    helpers::popcnt64 as *const () as usize,
                )?;
                self.emitter.mov(0, reg, ScratchRegister::R0)?;
                self.stack.push(StackValue::Register(reg));
            }
            // Special case: CLZ and CTZ can be directly mapped to Sljit ops on 64-bit
            #[cfg(target_pointer_width = "64")]
            UnaryOp::Clz64 => self.compile_unary_op(Op1::Clz)?,
            #[cfg(target_pointer_width = "64")]
            UnaryOp::Ctz64 => self.compile_unary_op(Op1::Ctz)?,

            #[cfg(not(target_pointer_width = "64"))]
            UnaryOp::Popcnt64 => helper(helpers::i64_popcnt as *const () as usize)?,
            #[cfg(not(target_pointer_width = "64"))]
            UnaryOp::Clz64 => helper(helpers::i64_clz as *const () as usize)?,
            #[cfg(not(target_pointer_width = "64"))]
            UnaryOp::Ctz64 => helper(helpers::i64_ctz as *const () as usize)?,
        }

        Ok(())
    }

    fn compile_load_op(&mut self, offset: i32, kind: Op1) -> Result<(), CompileError> {
        let addr = self.pop_value()?;
        let addr_reg = self.ensure_in_register(addr)?;
        self.emitter.add(0, addr_reg, addr_reg, SavedRegister::S0)?;
        self.emitter
            .emit_op1((kind, 0), addr_reg, mem_offset(addr_reg, offset))?;
        self.stack.push(StackValue::Register(addr_reg));
        Ok(())
    }

    fn compile_float_load_op(&mut self, offset: i32, kind: Fop1) -> Result<(), CompileError> {
        let addr = self.pop_value()?;
        let addr_reg = self.ensure_in_register(addr)?;
        self.emitter.add(0, addr_reg, addr_reg, SavedRegister::S0)?;
        let freg = self.alloc_float_register()?;
        self.emitter
            .emit_fop1((kind, 0), freg, mem_offset(addr_reg, offset))?;
        self.occupied_registers.set(addr_reg.index(), false);
        self.stack.push(StackValue::FloatRegister(freg));
        Ok(())
    }

    fn compile_store_op(&mut self, offset: i32, kind: Op1) -> Result<(), CompileError> {
        let (value, addr) = (self.pop_value()?, self.pop_value()?);
        let addr_reg = self.ensure_in_register(addr)?;
        self.emitter.add(0, addr_reg, addr_reg, SavedRegister::S0)?;
        let value_reg = self.ensure_in_register(value)?;
        self.emitter
            .emit_op1((kind, 0), mem_offset(addr_reg, offset), value_reg)?;
        self.occupied_registers.set(value_reg.index(), false);
        self.occupied_registers.set(addr_reg.index(), false);
        Ok(())
    }

    fn compile_float_store_op(&mut self, offset: i32, kind: Fop1) -> Result<(), CompileError> {
        let (value, addr) = (self.pop_value()?, self.pop_value()?);
        let addr_reg = self.ensure_in_register(addr)?;
        self.emitter.add(0, addr_reg, addr_reg, SavedRegister::S0)?;
        let is_f32 = matches!(kind, Fop1::MovF32);
        let freg = self.ensure_in_float_register(value, is_f32)?;
        self.emitter
            .emit_fop1((kind, 0), mem_offset(addr_reg, offset), freg)?;
        self.occupied_float_registers.set(freg.index(), false);
        self.occupied_registers.set(addr_reg.index(), false);
        Ok(())
    }

    #[cfg(target_pointer_width = "64")]
    fn compile_binary_op64(&mut self, op: Op2) -> Result<(), CompileError> {
        let (b, a) = (self.pop_value()?, self.pop_value()?);
        let reg_a = self.ensure_in_register(a)?;
        let operand_b = b.to_operand();
        self.emitter.emit_op2((op, 0), reg_a, reg_a, operand_b)?;
        if let StackValue::Register(reg_b) = b {
            self.occupied_registers.set(reg_b.index(), false);
        }
        self.stack.push(StackValue::Register(reg_a));
        Ok(())
    }

    #[cfg(not(target_pointer_width = "64"))]
    fn compile_binary_op64(&mut self, op: Op2) -> Result<(), CompileError> {
        // On 32-bit, use icall to helper functions for 64-bit operations
        // The helper functions expect (a_ptr, b_ptr, out_ptr) -> void

        use crate::helpers;
        let (b, a) = (self.pop_value()?, self.pop_value()?);

        // Get stack offsets for both operands (convert to stack if not already)
        let offset_a = self.ensure_i64_on_stack(a)?;
        let offset_b = self.ensure_i64_on_stack(b)?;

        // Allocate result space on stack
        let result_offset = self.frame_offset;
        self.frame_offset += 8;

        // Compute pointers: R0 = &a (SP + offset_a), R1 = &b (SP + offset_b), R2 = &result (SP + result_offset)
        self.emitter.get_local_base(ScratchRegister::R0, offset_a)?;
        self.emitter.get_local_base(ScratchRegister::R1, offset_b)?;
        self.emitter
            .get_local_base(ScratchRegister::R2, result_offset)?;

        // arg_types for (W, W, W) -> void (3 pointer arguments, void return)
        self.emitter.icall(
            JumpType::Call as i32,
            arg_types!([W, W, W]),
            match op {
                Op2::Add => helpers::i64_add as *const () as usize,
                Op2::Sub => helpers::i64_sub as *const () as usize,
                Op2::Mul => helpers::i64_mul as *const () as usize,
                Op2::And => helpers::i64_and as *const () as usize,
                Op2::Or => helpers::i64_or as *const () as usize,
                Op2::Xor => helpers::i64_xor as *const () as usize,
                Op2::Shl => helpers::i64_shl as *const () as usize,
                Op2::Ashr => helpers::i64_shr_u as *const () as usize,
                Op2::Lshr => helpers::i64_shr_s as *const () as usize,
                Op2::Rotl => helpers::i64_rotl as *const () as usize,
                Op2::Rotr => helpers::i64_rotr as *const () as usize,
                _ => unreachable!(),
            },
        )?;

        // Result is now at result_offset on stack
        self.stack.push(StackValue::Stack(result_offset));
        Ok(())
    }

    /// Helper to ensure an i64 value is on the stack (for 32-bit platforms)
    #[cfg(not(target_pointer_width = "64"))]
    fn ensure_i64_on_stack(&mut self, val: StackValue) -> Result<i32, CompileError> {
        match val {
            StackValue::Stack(off) => Ok(off),
            StackValue::Const(v) => {
                // Convert i32 constant to i64 on stack with zero extension
                let offset = self.frame_offset;
                self.frame_offset += 8;
                self.emitter.mov32(0, mem_sp_offset(offset), v)?;
                self.emitter.mov32(0, mem_sp_offset(offset + 4), 0i32)?;
                Ok(offset)
            }
            StackValue::Register(reg) => {
                // Convert register to i64 on stack with zero extension
                let offset = self.frame_offset;
                self.frame_offset += 8;
                self.emitter.mov32(0, mem_sp_offset(offset), reg)?;
                self.emitter.mov32(0, mem_sp_offset(offset + 4), 0i32)?;
                self.occupied_registers.set(reg.index(), false);
                Ok(offset)
            }
            StackValue::Saved(sreg) => {
                // Convert saved register to i64 on stack with zero extension
                let offset = self.frame_offset;
                self.frame_offset += 8;
                self.emitter.mov32(0, mem_sp_offset(offset), sreg)?;
                self.emitter.mov32(0, mem_sp_offset(offset + 4), 0i32)?;
                Ok(offset)
            }
            _ => Err(CompileError::Invalid(
                "Unexpected value type for i64 operation".into(),
            )),
        }
    }

    #[cfg(target_pointer_width = "64")]
    fn compile_compare_op64(&mut self, cond: Condition) -> Result<(), CompileError> {
        self.compile_compare_op(cond)
    }

    #[cfg(not(target_pointer_width = "64"))]
    fn compile_compare_op64(&mut self, cond: Condition) -> Result<(), CompileError> {
        // On 32-bit, use helper functions for 64-bit comparisons
        let (b, a) = (self.pop_value()?, self.pop_value()?);

        // Get stack offsets for both operands
        let offset_a = self.ensure_i64_on_stack(a)?;
        let offset_b = self.ensure_i64_on_stack(b)?;

        // Get the appropriate comparison helper function
        // Compute pointers: R0 = &a (SP + offset_a), R1 = &b (SP + offset_b)
        self.emitter.get_local_base(ScratchRegister::R0, offset_a)?;
        self.emitter.get_local_base(ScratchRegister::R1, offset_b)?;

        // arg_types for (W, W) -> W (2 pointer arguments, returns u32 0 or 1)
        self.emitter.icall(
            JumpType::Call as i32,
            arg_types!([W, W] -> W),
            match cond {
                Condition::Equal => helpers::i64_eq as *const () as usize,
                Condition::NotEqual => helpers::i64_neq as *const () as usize,
                Condition::Less => helpers::i64_lt as *const () as usize,
                Condition::SigLess => helpers::i64_lt_s as *const () as usize,
                Condition::Greater => helpers::i64_gt as *const () as usize,
                Condition::SigGreater => helpers::i64_gt_s as *const () as usize,
                Condition::LessEqual => helpers::i64_le as *const () as usize,
                Condition::SigLessEqual => helpers::i64_le_s as *const () as usize,
                Condition::GreaterEqual => helpers::i64_ge as *const () as usize,
                Condition::SigGreaterEqual => helpers::i64_ge_s as *const () as usize,
                _ => {
                    return Err(CompileError::Unsupported(format!(
                        "Unsupported 64-bit comparison condition: {:?}",
                        cond
                    )));
                }
            },
        )?;

        // Result is in R0 (0 or 1)
        self.occupied_registers
            .set(ScratchRegister::R0.index(), true);
        self.stack.push(StackValue::Register(ScratchRegister::R0));
        Ok(())
    }

    #[cfg(target_pointer_width = "64")]
    fn compile_div_op64(&mut self, op: DivOp) -> Result<(), CompileError> {
        let (b, a) = (self.pop_value()?, self.pop_value()?);
        let (reg_a, reg_b) = (self.ensure_in_register(a)?, self.ensure_in_register(b)?);

        if reg_a != ScratchRegister::R0 {
            self.emitter.mov(0, ScratchRegister::R0, reg_a)?;
        }
        if reg_b != ScratchRegister::R1 {
            self.emitter.mov(0, ScratchRegister::R1, reg_b)?;
        }

        [reg_a, reg_b]
            .iter()
            .filter(|&&r| r != ScratchRegister::R0 && r != ScratchRegister::R1)
            .for_each(|&r| self.occupied_registers.set(r.index(), false));

        match op {
            DivOp::DivS | DivOp::RemS => {
                self.emitter.divmod_sword()?;
            }
            DivOp::DivU | DivOp::RemU => {
                self.emitter.divmod_uword()?;
            }
        }

        let (result_reg, other_reg) = match op {
            DivOp::DivS | DivOp::DivU => (ScratchRegister::R0, ScratchRegister::R1),
            DivOp::RemS | DivOp::RemU => (ScratchRegister::R1, ScratchRegister::R0),
        };

        self.occupied_registers.set(other_reg.index(), false);
        self.occupied_registers.set(result_reg.index(), true);
        self.stack.push(StackValue::Register(result_reg));
        Ok(())
    }

    #[cfg(not(target_pointer_width = "64"))]
    fn compile_div_op64(&mut self, op: DivOp) -> Result<(), CompileError> {
        // On 32-bit, use helper functions for 64-bit division
        let (b, a) = (self.pop_value()?, self.pop_value()?);

        // Get stack offsets for both operands
        let offset_a = self.ensure_i64_on_stack(a)?;
        let offset_b = self.ensure_i64_on_stack(b)?;

        // Allocate result space on stack
        let result_offset = self.frame_offset;
        self.frame_offset += 8;

        // Compute pointers: R0 = &a, R1 = &b, R2 = &result
        self.emitter.get_local_base(ScratchRegister::R0, offset_a)?;
        self.emitter.get_local_base(ScratchRegister::R1, offset_b)?;
        self.emitter
            .get_local_base(ScratchRegister::R2, result_offset)?;

        // arg_types for (W, W, W) -> void
        self.emitter.icall(
            JumpType::Call as i32,
            arg_types!([W, W, W]),
            match op {
                DivOp::DivS => helpers::i64_div_s as *const () as usize,
                DivOp::DivU => helpers::i64_div_u as *const () as usize,
                DivOp::RemS => helpers::i64_rem_s as *const () as usize,
                DivOp::RemU => helpers::i64_rem_u as *const () as usize,
            },
        )?;

        // Result is now at result_offset on stack
        self.stack.push(StackValue::Stack(result_offset));
        Ok(())
    }

    #[cfg(target_pointer_width = "64")]
    fn compile_load_op64(&mut self, offset: i32, kind: Op1) -> Result<(), CompileError> {
        let addr = self.pop_value()?;
        let addr_reg = self.ensure_in_register(addr)?;
        self.emitter.add(0, addr_reg, addr_reg, SavedRegister::S0)?;
        self.emitter
            .emit_op1((kind, 0), addr_reg, mem_offset(addr_reg, offset))?;
        self.stack.push(StackValue::Register(addr_reg));
        Ok(())
    }

    #[cfg(not(target_pointer_width = "64"))]
    fn compile_load_op64(&mut self, offset: i32, kind: Op1) -> Result<(), CompileError> {
        let addr = self.pop_value()?;
        let addr_reg = self.ensure_in_register(addr)?;
        self.emitter.add(0, addr_reg, addr_reg, SavedRegister::S0)?;

        // On 32-bit: load 64-bit value as two 32-bit halves to stack
        let result_offset = self.frame_offset;
        self.frame_offset += 8;

        match kind {
            Op1::Mov => {
                // Load lower 32 bits
                self.emitter.mov32(
                    0,
                    mem_sp_offset(result_offset),
                    mem_offset(addr_reg, offset),
                )?;

                // Load upper 32 bits
                self.emitter.mov32(
                    0,
                    mem_sp_offset(result_offset + 4),
                    mem_offset(addr_reg, offset + 4),
                )?;
            }
            Op1::MovS8 => {
                // Load 8-bit value with sign extension, reusing addr_reg
                // (addr is read before being overwritten)
                self.emitter
                    .mov_s8(0, addr_reg, mem_offset(addr_reg, offset))?;
                self.emitter
                    .mov32(0, mem_sp_offset(result_offset), addr_reg)?;
                // Sign extend to high part
                self.emitter.ashr32(0, addr_reg, addr_reg, 31i32)?;
                self.emitter
                    .mov32(0, mem_sp_offset(result_offset + 4), addr_reg)?;
            }
            Op1::MovU8 => {
                self.emitter.mov_u8(
                    0,
                    mem_sp_offset(result_offset),
                    mem_offset(addr_reg, offset),
                )?;
                // Zero extend to high part
                self.emitter
                    .mov32(0, mem_sp_offset(result_offset + 4), 0i32)?;
            }
            Op1::MovS16 => {
                // Load 16-bit value with sign extension, reusing addr_reg
                self.emitter
                    .mov_s16(0, addr_reg, mem_offset(addr_reg, offset))?;
                self.emitter
                    .mov32(0, mem_sp_offset(result_offset), addr_reg)?;
                // Sign extend to high part
                self.emitter.ashr32(0, addr_reg, addr_reg, 31i32)?;
                self.emitter
                    .mov32(0, mem_sp_offset(result_offset + 4), addr_reg)?;
            }
            Op1::MovU16 => {
                self.emitter.mov_u16(
                    0,
                    mem_sp_offset(result_offset),
                    mem_offset(addr_reg, offset),
                )?;
                // Zero extend to high part
                self.emitter
                    .mov32(0, mem_sp_offset(result_offset + 4), 0i32)?;
            }
            Op1::MovS32 => {
                // Load 32-bit value with sign extension, reusing addr_reg
                self.emitter
                    .mov32(0, addr_reg, mem_offset(addr_reg, offset))?;
                self.emitter
                    .mov32(0, mem_sp_offset(result_offset), addr_reg)?;
                // Sign extend to high part: copy the sign bit
                self.emitter.ashr32(0, addr_reg, addr_reg, 31i32)?;
                self.emitter
                    .mov32(0, mem_sp_offset(result_offset + 4), addr_reg)?;
            }
            Op1::MovU32 => {
                self.emitter.mov32(
                    0,
                    mem_sp_offset(result_offset),
                    mem_offset(addr_reg, offset),
                )?;
                // Zero extend to high part
                self.emitter
                    .mov32(0, mem_sp_offset(result_offset + 4), 0i32)?;
            }
            _ => unreachable!(),
        }

        self.occupied_registers.set(addr_reg.index(), false);
        self.stack.push(StackValue::Stack(result_offset));
        Ok(())
    }

    #[cfg(target_pointer_width = "64")]
    fn compile_store_op64(&mut self, offset: i32, kind: Op1) -> Result<(), CompileError> {
        let value = self.pop_value()?;
        let addr = self.pop_value()?;
        let value_reg = self.ensure_in_register(value)?;
        let addr_reg = self.ensure_in_register(addr)?;
        self.emitter.add(0, addr_reg, addr_reg, SavedRegister::S0)?;
        self.emitter
            .emit_op1((kind, 0), mem_offset(addr_reg, offset), value_reg)?;
        self.occupied_registers.set(value_reg.index(), false);
        self.occupied_registers.set(addr_reg.index(), false);
        Ok(())
    }

    #[cfg(not(target_pointer_width = "64"))]
    fn compile_store_op64(&mut self, offset: i32, kind: Op1) -> Result<(), CompileError> {
        let value = self.pop_value()?;
        let addr = self.pop_value()?;
        let addr_reg = self.ensure_in_register(addr)?;
        self.emitter.add(0, addr_reg, addr_reg, SavedRegister::S0)?;

        let mem = mem_offset(addr_reg, offset);

        // Get low 32 bits into a register
        let value_reg = match &value {
            StackValue::Stack(value_offset) => {
                let reg = self.alloc_register()?;
                self.emitter.mov32(0, reg, mem_sp_offset(*value_offset))?;
                reg
            }
            _ => self.ensure_in_register(value.clone())?,
        };

        // Emit the store based on kind
        match kind {
            Op1::Mov => {
                // Full 64-bit store: need both halves
                self.emitter.mov32(0, mem, value_reg)?;
                if let StackValue::Stack(value_offset) = &value {
                    self.emitter
                        .mov32(0, value_reg, mem_sp_offset(*value_offset + 4))?;
                    self.emitter
                        .mov32(0, mem_offset(addr_reg, offset + 4), value_reg)?;
                } else {
                    self.occupied_registers.set(value_reg.index(), false);
                    self.occupied_registers.set(addr_reg.index(), false);
                    return Err(CompileError::Invalid(
                        "64-bit store from register not supported on 32-bit platform".into(),
                    ));
                }
            }
            Op1::MovU8 | Op1::MovU16 | Op1::Mov32 => {
                self.emitter.emit_op1((kind, 0), mem, value_reg)?;
            }
            _ => unreachable!(),
        }

        self.occupied_registers.set(value_reg.index(), false);
        self.occupied_registers.set(addr_reg.index(), false);
        Ok(())
    }

    /// Compile a floating point binary operation
    fn compile_float_binary_op(&mut self, op: Fop2, is_f32: bool) -> Result<(), CompileError> {
        let (b, a) = (self.pop_value()?, self.pop_value()?);
        let freg_a = self.ensure_in_float_register(a, is_f32)?;
        let freg_b = self.ensure_in_float_register(b, is_f32)?;
        self.emitter.emit_fop2((op, 0), freg_a, freg_a, freg_b)?;
        self.occupied_float_registers.set(freg_b.index(), false);
        self.stack.push(StackValue::FloatRegister(freg_a));
        Ok(())
    }

    fn compile_float_binary_op_with_helper(
        &mut self,

        op: FloatBinaryOp,
        is_f32: bool,
    ) -> Result<(), CompileError> {
        let (b, a) = (self.pop_value()?, self.pop_value()?);
        let freg_a = self.ensure_in_float_register(a, is_f32)?;
        let freg_b = self.ensure_in_float_register(b, is_f32)?;

        if freg_a != FloatRegister::FR0 {
            self.mov_float(is_f32, FloatRegister::FR0, freg_a)?;
        }
        if freg_b != FloatRegister::FR1 {
            self.mov_float(is_f32, FloatRegister::FR1, freg_b)?;
        }

        self.emitter.icall(
            JumpType::Call as i32,
            if is_f32 {
                arg_types!([F32, F32] -> F32)
            } else {
                arg_types!([F64, F64] -> F64)
            },
            match op {
                FloatBinaryOp::Min if is_f32 => helpers::fmin32 as *const () as usize,
                FloatBinaryOp::Min => helpers::fmin64 as *const () as usize,
                FloatBinaryOp::Max if is_f32 => helpers::fmax32 as *const () as usize,
                FloatBinaryOp::Max => helpers::fmax64 as *const () as usize,
                FloatBinaryOp::Copysign if is_f32 => helpers::copysign32 as *const () as usize,
                FloatBinaryOp::Copysign => helpers::copysign64 as *const () as usize,
            },
        )?;

        if freg_a != FloatRegister::FR0 {
            self.mov_float(is_f32, freg_a, FloatRegister::FR0)?;
        }
        self.occupied_float_registers.set(freg_b.index(), false);
        self.stack.push(StackValue::FloatRegister(freg_a));
        Ok(())
    }

    /// Compile a floating point unary operation
    fn compile_float_unary_op(&mut self, op: Fop1, is_f32: bool) -> Result<(), CompileError> {
        let val = self.pop_value()?;
        let freg = self.ensure_in_float_register(val, is_f32)?;
        self.emitter.emit_fop1((op, 0), freg, freg)?;
        self.stack.push(StackValue::FloatRegister(freg));
        Ok(())
    }

    fn compile_float_unary_op_with_helper(
        &mut self,

        op: FloatUnaryOp,
        is_f32: bool,
    ) -> Result<(), CompileError> {
        let val = self.pop_value()?;
        let freg = self.ensure_in_float_register(val, is_f32)?;

        if freg != FloatRegister::FR0 {
            self.mov_float(is_f32, FloatRegister::FR0, freg)?;
        }

        self.emitter.icall(
            JumpType::Call as i32,
            if is_f32 {
                arg_types!([F32] -> F32)
            } else {
                arg_types!([F64] -> F64)
            },
            match op {
                FloatUnaryOp::Sqrt if is_f32 => helpers::sqrtf32 as *const () as usize,
                FloatUnaryOp::Sqrt => helpers::sqrtf64 as *const () as usize,
                FloatUnaryOp::Ceil if is_f32 => helpers::ceilf32 as *const () as usize,
                FloatUnaryOp::Ceil => helpers::ceilf64 as *const () as usize,
                FloatUnaryOp::Floor if is_f32 => helpers::floorf32 as *const () as usize,
                FloatUnaryOp::Floor => helpers::floorf64 as *const () as usize,
                FloatUnaryOp::Trunc if is_f32 => helpers::truncf32 as *const () as usize,
                FloatUnaryOp::Trunc => helpers::truncf64 as *const () as usize,
                FloatUnaryOp::Nearest if is_f32 => helpers::nintf32 as *const () as usize,
                FloatUnaryOp::Nearest => helpers::nintf64 as *const () as usize,
            },
        )?;

        if freg != FloatRegister::FR0 {
            self.mov_float(is_f32, freg, FloatRegister::FR0)?;
        }

        self.stack.push(StackValue::FloatRegister(freg));
        Ok(())
    }

    /// Helper to move float register conditionally based on is_f32
    fn mov_float(
        &mut self,
        is_f32: bool,
        dst: impl Into<Operand>,
        src: impl Into<Operand>,
    ) -> Result<(), sys::ErrorCode> {
        if is_f32 {
            self.emitter.mov_f32(0, dst, src)?;
        } else {
            self.emitter.mov_f64(0, dst, src)?;
        }
        Ok(())
    }

    /// Compile a floating point comparison operation
    fn compile_float_compare_op(
        &mut self,

        cond: Condition,
        is_f32: bool,
    ) -> Result<(), CompileError> {
        let flags = match cond {
            Condition::FEqual | Condition::FNotEqual => set(sys::SLJIT_F_EQUAL),
            Condition::FLess | Condition::FGreaterEqual => set(sys::SLJIT_F_LESS),
            Condition::FGreater | Condition::FLessEqual => set(sys::SLJIT_F_GREATER),
            _ => {
                return Err(CompileError::Unsupported(format!(
                    "Unsupported condition: {:?}",
                    cond
                )));
            }
        };
        let (b, a) = (self.pop_value()?, self.pop_value()?);
        let reg_a = self.ensure_in_float_register(a, is_f32)?;
        let reg_b = self.ensure_in_float_register(b, is_f32)?;

        // Map condition to the appropriate SET flag
        if is_f32 {
            self.emitter.cmp_f32(flags, reg_a, reg_b)?;
        } else {
            self.emitter.cmp_f64(flags, reg_a, reg_b)?;
        }

        // Convert flags to 0/1 using op_flags
        self.emitter.op_flags(sys::SLJIT_MOV, reg_a, cond as i32)?;

        self.occupied_registers.set(reg_b.index(), false);
        self.stack.push(StackValue::FloatRegister(reg_a));
        Ok(())
    }

    /// Convert integer to float (64-bit platform)
    #[cfg(target_pointer_width = "64")]
    fn compile_convert_int_to_float(
        &mut self,

        is_f32: bool,
        is_i32: bool,
        is_signed: bool,
    ) -> Result<(), CompileError> {
        let val = self.pop_value()?;
        let reg = self.ensure_in_register(val)?;
        let freg = self.alloc_float_register()?;

        match (is_f32, is_i32, is_signed) {
            (true, true, true) => self.emitter.conv_f32_from_s32(0, freg, reg)?,
            (true, true, false) => self.emitter.conv_f32_from_u32(0, freg, reg)?,
            (false, true, true) => self.emitter.conv_f64_from_s32(0, freg, reg)?,
            (false, true, false) => self.emitter.conv_f64_from_u32(0, freg, reg)?,
            (true, false, true) => self.emitter.conv_f32_from_sw(0, freg, reg)?,
            (true, false, false) => self.emitter.conv_f32_from_uw(0, freg, reg)?,
            (false, false, true) => self.emitter.conv_f64_from_sw(0, freg, reg)?,
            (false, false, false) => self.emitter.conv_f64_from_uw(0, freg, reg)?,
        };

        self.occupied_registers.set(reg.index(), false);
        self.stack.push(StackValue::FloatRegister(freg));
        Ok(())
    }

    /// Convert integer to float (32-bit platform)
    #[cfg(not(target_pointer_width = "64"))]
    fn compile_convert_int_to_float(
        &mut self,

        is_f32: bool,
        is_i32: bool,
        is_signed: bool,
    ) -> Result<(), CompileError> {
        let val = self.pop_value()?;

        // i64 -> float needs special handling on 32-bit
        if !is_i32 {
            let value_offset = match val {
                StackValue::Stack(offset) => offset,
                _ => {
                    return Err(CompileError::Invalid(
                        "i64 value must be on stack on 32-bit platform for float conversion".into(),
                    ));
                }
            };

            // Load low and high parts into R0 and R1
            self.emitter
                .mov32(0, ScratchRegister::R0, mem_sp_offset(value_offset))?;
            self.emitter
                .mov32(0, ScratchRegister::R1, mem_sp_offset(value_offset + 4))?;

            // Call helper function
            self.emitter.icall(
                JumpType::Call as i32,
                if is_f32 {
                    arg_types!([W, W] -> F32)
                } else {
                    arg_types!([W, W] -> F64)
                },
                match (is_f32, is_signed) {
                    (true, true) => helpers::i64_to_f32_signed as *const () as usize,
                    (true, false) => helpers::i64_to_f32_unsigned as *const () as usize,
                    (false, true) => helpers::i64_to_f64_signed as *const () as usize,
                    (false, false) => helpers::i64_to_f64_unsigned as *const () as usize,
                },
            )?;

            // Result is in FR0
            let freg = self.alloc_float_register()?;
            if freg != FloatRegister::FR0 {
                self.mov_float(is_f32, freg, FloatRegister::FR0)?;
            }
            self.stack.push(StackValue::FloatRegister(freg));
            return Ok(());
        }

        // i32 -> float
        let reg = self.ensure_in_register(val)?;
        let freg = self.alloc_float_register()?;

        match (is_f32, is_signed) {
            (true, true) => self.emitter.conv_f32_from_s32(0, freg, reg)?,
            (true, false) => self.emitter.conv_f32_from_u32(0, freg, reg)?,
            (false, true) => self.emitter.conv_f64_from_s32(0, freg, reg)?,
            (false, false) => self.emitter.conv_f64_from_u32(0, freg, reg)?,
        };

        self.occupied_registers.set(reg.index(), false);
        self.stack.push(StackValue::FloatRegister(freg));
        Ok(())
    }

    /// Convert float to integer (truncate)
    fn compile_convert_float_to_int(
        &mut self,

        is_f32: bool,
        is_i32: bool,
        _is_signed: bool,
    ) -> Result<(), CompileError> {
        let val = self.pop_value()?;
        let freg = self.ensure_in_float_register(val, is_f32)?;
        let reg = self.alloc_register()?;

        // Note: SLJIT doesn't have direct unsigned conversion, use signed for all
        match (is_f32, is_i32) {
            (true, true) => self.emitter.conv_s32_from_f32(0, reg, freg)?,
            (false, true) => self.emitter.conv_s32_from_f64(0, reg, freg)?,

            (true, false) => self.emitter.conv_sw_from_f32(0, reg, freg)?,

            (false, false) => self.emitter.conv_sw_from_f64(0, reg, freg)?,
        };

        self.occupied_float_registers.set(freg.index(), false);
        self.stack.push(StackValue::Register(reg));
        Ok(())
    }

    /// Compile f32.demote_f64 or f64.promote_f32
    fn compile_float_demote_promote(&mut self, is_demote: bool) -> Result<(), CompileError> {
        let val = self.pop_value()?;
        let freg = self.ensure_in_float_register(val, !is_demote)?;
        if is_demote {
            self.emitter.conv_f32_from_f64(0, freg, freg)?;
        } else {
            self.emitter.conv_f64_from_f32(0, freg, freg)?;
        }
        self.stack.push(StackValue::FloatRegister(freg));
        Ok(())
    }

    /// Compile global.get - load a global variable value
    fn compile_global_get(&mut self, global_index: u32) -> Result<(), CompileError> {
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

        let reg = self.alloc_register()?;
        self.emitter.mov(0, reg, global.ptr)?;
        let mem = mem_offset(reg, 0);

        match global.val_type {
            ValType::I32 => {
                self.emitter.mov32(0, reg, mem)?;
            }
            ValType::I64 => {
                #[cfg(target_pointer_width = "64")]
                {
                    self.emitter.mov(0, reg, mem)?;
                }
                #[cfg(not(target_pointer_width = "64"))]
                {
                    return Err(CompileError::Unsupported(
                        "64-bit globals not supported on 32-bit platform".into(),
                    ));
                }
            }
            ValType::F32 | ValType::F64 => {
                let freg = self.alloc_float_register()?;
                if matches!(global.val_type, ValType::F32) {
                    self.emitter.mov_f32(0, freg, mem)?;
                } else {
                    self.emitter.mov_f64(0, freg, mem)?;
                }
                self.occupied_registers.set(reg.index(), false);
                self.stack.push(StackValue::FloatRegister(freg));
                return Ok(());
            }
            _ => {
                return Err(CompileError::Unsupported(format!(
                    "Unsupported global type: {:?}",
                    global.val_type
                )));
            }
        }
        self.stack.push(StackValue::Register(reg));
        Ok(())
    }

    /// Compile global.set - store a value to a global variable
    fn compile_global_set(&mut self, global_index: u32) -> Result<(), CompileError> {
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
        let addr_reg = self.alloc_register()?;
        self.emitter.mov(0, addr_reg, global.ptr)?;
        let mem = mem_offset(addr_reg, 0);

        match global.val_type {
            ValType::I32 => {
                let val_reg = self.ensure_in_register(value)?;
                self.emitter.mov32(0, mem, val_reg)?;
                self.occupied_registers.set(val_reg.index(), false);
            }
            ValType::I64 => {
                #[cfg(target_pointer_width = "64")]
                {
                    let val_reg = self.ensure_in_register(value)?;
                    self.emitter.mov(0, mem, val_reg)?;
                    self.occupied_registers.set(val_reg.index(), false);
                }
                #[cfg(not(target_pointer_width = "64"))]
                {
                    return Err(CompileError::Unsupported(
                        "64-bit globals not supported on 32-bit platform".into(),
                    ));
                }
            }
            ValType::F32 => {
                self.emitter.mov_f32(0, mem, value.to_operand())?;
            }
            ValType::F64 => {
                self.emitter.mov_f64(0, mem, value.to_operand())?;
            }
            _ => {
                return Err(CompileError::Unsupported(format!(
                    "Unsupported global type: {:?}",
                    global.val_type
                )));
            }
        }
        self.occupied_registers.set(addr_reg.index(), false);
        Ok(())
    }

    /// Compile memory.size - return current memory size in pages
    fn compile_memory_size(&mut self, _mem: u32) -> Result<(), CompileError> {
        let size_ptr = self
            .memory
            .as_ref()
            .ok_or_else(|| CompileError::Invalid("No memory defined".into()))?
            .size_ptr;

        let reg = self.alloc_register()?;
        self.emitter.mov(0, reg, size_ptr)?;
        self.emitter.mov32(0, reg, mem_offset(reg, 0))?;
        self.stack.push(StackValue::Register(reg));
        Ok(())
    }

    /// Compile memory.grow - grow memory by delta pages, return previous size or -1 on failure
    fn compile_memory_grow(&mut self, _mem: u32) -> Result<(), CompileError> {
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
        let delta_reg = self.ensure_in_register(delta)?;

        // Load current size into R0 (first argument)
        if delta_reg != ScratchRegister::R0 {
            self.emitter.mov(0, ScratchRegister::R0, size_ptr)?;
            self.emitter
                .mov32(0, ScratchRegister::R0, mem_offset(ScratchRegister::R0, 0))?;
        } else {
            // delta is in R0, need to save it first
            self.emitter.mov(0, ScratchRegister::R1, delta_reg)?;
            self.emitter.mov(0, ScratchRegister::R0, size_ptr)?;
            self.emitter
                .mov32(0, ScratchRegister::R0, mem_offset(ScratchRegister::R0, 0))?;
            self.emitter
                .mov(0, ScratchRegister::R1, ScratchRegister::R1)?; // delta now in R1
        }

        // Move delta to R1 (second argument) if not already there
        if delta_reg != ScratchRegister::R1 && delta_reg != ScratchRegister::R0 {
            self.emitter.mov(0, ScratchRegister::R1, delta_reg)?;
            self.occupied_registers.set(delta_reg.index(), false);
        } else if delta_reg == ScratchRegister::R0 {
            // Already moved to R1 above
            self.occupied_registers.set(delta_reg.index(), false);
        }

        // Call the grow callback: fn(current_pages: u32, delta: u32) -> i32
        let addr_reg = self.alloc_register()?;
        self.emitter.mov(0, addr_reg, grow_callback as usize)?;
        // arg_types for (W, W) -> W: return type W, plus two W arguments
        let arg_types_w2_w = SLJIT_ARG_TYPE_W | (SLJIT_ARG_TYPE_W << 4) | (SLJIT_ARG_TYPE_W << 8);
        self.emitter
            .icall(JumpType::Call as i32, arg_types_w2_w, addr_reg)?;
        self.occupied_registers.set(addr_reg.index(), false);

        // Result is in R0
        self.occupied_registers
            .set(ScratchRegister::R0.index(), true);
        self.stack.push(StackValue::Register(ScratchRegister::R0));
        Ok(())
    }

    /// Compile call - direct function call
    /// Supports functions with more than 3 parameters by passing extra args on stack
    fn compile_call(&mut self, function_index: u32) -> Result<(), CompileError> {
        if function_index as usize >= self.functions.len() {
            return Err(CompileError::Invalid(format!(
                "Function index {} out of bounds (have {})",
                function_index,
                self.functions.len()
            )));
        }

        // Clone the needed data to avoid borrow issues
        let func_code_ptr = self.functions[function_index as usize].code_ptr;
        let func_code_ptr_ptr = self.functions[function_index as usize].code_ptr_ptr;
        let func_params = self.functions[function_index as usize].params.clone();
        let func_results = self.functions[function_index as usize].results.clone();
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
            self.emitter.sub(
                0,
                Operand::from(sys::SLJIT_SP),
                Operand::from(sys::SLJIT_SP),
                extra_stack_size,
            )?;

            // Store extra arguments (args 4+) on the stack
            for (i, arg) in args.iter().enumerate().skip(3) {
                let stack_offset = ((i - 3) * 8) as i32;
                let reg = self.ensure_in_register(arg.clone())?;
                self.emitter.mov(0, mem_sp_offset(stack_offset), reg)?;
                self.occupied_registers.set(reg.index(), false);
            }
        }

        // Move first 3 arguments to the appropriate registers
        let arg_regs = [
            ScratchRegister::R0,
            ScratchRegister::R1,
            ScratchRegister::R2,
        ];
        let num_arg_regs = args.len().min(3);
        for (i, arg) in args.into_iter().take(3).enumerate() {
            let reg = self.ensure_in_register(arg)?;
            if reg != arg_regs[i] {
                self.emitter.mov(0, arg_regs[i], reg)?;
                self.occupied_registers.set(reg.index(), false);
            }
        }

        // Mark argument registers as occupied to prevent them from being used for addr_reg
        for reg in arg_regs.iter().take(num_arg_regs) {
            self.occupied_registers.set(reg.index(), true);
        }

        // Load function address and call
        // For wasm functions (code_ptr_ptr is Some), use indirect load since code_ptr
        // may be updated after compilation. For host functions, use direct code_ptr.
        let addr_reg = self.alloc_register()?;
        if let Some(ptr_ptr) = func_code_ptr_ptr {
            // Load the code_ptr through the pointer (indirect call)
            // First load the address of code_ptr into addr_reg
            self.emitter.mov(0, addr_reg, ptr_ptr as usize)?;
            // Then dereference to get the actual code_ptr
            self.emitter.mov(0, addr_reg, mem_offset(addr_reg, 0))?;
        } else {
            // Direct call with the code_ptr value
            self.emitter.mov(0, addr_reg, func_code_ptr)?;
        }

        let arg_types = Self::make_arg_types(&func_params, &func_results);
        self.emitter
            .icall(JumpType::Call as i32, arg_types, addr_reg)?;
        self.occupied_registers.set(addr_reg.index(), false);

        // Restore stack pointer if we allocated extra space
        if extra_stack_size > 0 {
            self.emitter.add(
                0,
                Operand::from(sys::SLJIT_SP),
                Operand::from(sys::SLJIT_SP),
                extra_stack_size,
            )?;
        }

        // Free argument registers (except R0 if it's used for result)
        for (i, reg) in arg_regs.iter().enumerate().take(num_arg_regs) {
            if result_count == 0 || i != 0 {
                self.occupied_registers.set(reg.index(), false);
            }
        }

        // Handle result
        if result_count > 0 {
            // R0 is already marked occupied from arg setup if num_arg_regs > 0
            if num_arg_regs == 0 {
                self.occupied_registers
                    .set(ScratchRegister::R0.index(), true);
            }
            self.stack.push(StackValue::Register(ScratchRegister::R0));
        }

        Ok(())
    }

    /// Compile call_indirect - indirect function call through table
    /// Supports functions with more than 3 parameters by passing extra args on stack
    fn compile_call_indirect(
        &mut self,

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
        let idx_reg = self.ensure_in_register(idx)?;

        // Pop arguments in reverse order
        let mut args = vec![];
        for _ in 0..param_count {
            args.push(self.pop_value()?);
        }
        args.reverse();

        // Calculate the function pointer address: table.data_ptr + idx * sizeof(usize)
        let func_ptr_reg = self.alloc_register()?;
        self.emitter.mov(0, func_ptr_reg, table_data_ptr)?;

        // idx * sizeof(usize)
        #[cfg(target_pointer_width = "64")]
        self.emitter.shl(0, idx_reg, idx_reg, 3i32)?; // * 8
        #[cfg(not(target_pointer_width = "64"))]
        self.emitter.shl32(0, idx_reg, idx_reg, 2i32)?; // * 4

        self.emitter.add(0, func_ptr_reg, func_ptr_reg, idx_reg)?;
        self.occupied_registers.set(idx_reg.index(), false);

        // Load the function pointer from the table
        self.emitter
            .mov(0, func_ptr_reg, mem_offset(func_ptr_reg, 0))?;

        // Save the function pointer to a temporary stack location since we'll need registers for args
        let func_ptr_offset = self.frame_offset;
        self.frame_offset += 8;
        self.emitter
            .mov(0, mem_sp_offset(func_ptr_offset), func_ptr_reg)?;
        self.occupied_registers.set(func_ptr_reg.index(), false);

        // Calculate stack space needed for extra arguments (args 4+)
        let extra_stack_size = Self::get_extra_arg_stack_size(param_count);

        // If we have extra arguments, allocate stack space and store them
        if extra_stack_size > 0 {
            // Adjust stack pointer to make room for extra arguments
            self.emitter.sub(
                0,
                Operand::from(sys::SLJIT_SP),
                Operand::from(sys::SLJIT_SP),
                extra_stack_size,
            )?;

            // Store extra arguments (args 4+) on the stack
            for (i, arg) in args.iter().enumerate().skip(3) {
                let stack_offset = ((i - 3) * 8) as i32;
                let reg = self.ensure_in_register(arg.clone())?;
                self.emitter.mov(0, mem_sp_offset(stack_offset), reg)?;
                self.occupied_registers.set(reg.index(), false);
            }
        }

        // Move first 3 arguments to the appropriate registers
        let arg_regs = [
            ScratchRegister::R0,
            ScratchRegister::R1,
            ScratchRegister::R2,
        ];
        let num_arg_regs = args.len().min(3);
        for (i, arg) in args.into_iter().take(3).enumerate() {
            let reg = self.ensure_in_register(arg)?;
            if reg != arg_regs[i] {
                self.emitter.mov(0, arg_regs[i], reg)?;
                self.occupied_registers.set(reg.index(), false);
            }
        }

        // Mark argument registers as occupied to prevent them from being used for addr_reg
        for reg in arg_regs.iter().take(num_arg_regs) {
            self.occupied_registers.set(reg.index(), true);
        }

        // Reload the function pointer
        let addr_reg = self.alloc_register()?;
        // Account for the stack adjustment when loading the saved function pointer
        self.emitter.mov(
            0,
            addr_reg,
            mem_sp_offset(if extra_stack_size > 0 {
                func_ptr_offset + extra_stack_size
            } else {
                func_ptr_offset
            }),
        )?;

        // Call through the function pointer
        let arg_types = Self::make_arg_types(&func_params, &func_results);
        self.emitter
            .icall(JumpType::Call as i32, arg_types, addr_reg)?;
        self.occupied_registers.set(addr_reg.index(), false);

        // Restore stack pointer if we allocated extra space
        if extra_stack_size > 0 {
            self.emitter.add(
                0,
                Operand::from(sys::SLJIT_SP),
                Operand::from(sys::SLJIT_SP),
                extra_stack_size,
            )?;
        }

        // Free argument registers (except R0 if it's used for result)
        for (i, reg) in arg_regs.iter().enumerate().take(num_arg_regs) {
            if result_count == 0 || i != 0 {
                self.occupied_registers.set(reg.index(), false);
            }
        }

        // Handle result
        if result_count > 0 {
            // R0 is already marked occupied from arg setup if num_arg_regs > 0
            if num_arg_regs == 0 {
                self.occupied_registers
                    .set(ScratchRegister::R0.index(), true);
            }
            self.stack.push(StackValue::Register(ScratchRegister::R0));
        }

        Ok(())
    }

    /// Compile br_table - branch table (switch-like construct)
    fn compile_br_table(&mut self, targets: &wasmparser::BrTable) -> Result<(), CompileError> {
        // Pop the index value
        let idx = self.pop_value()?;
        let idx_reg = self.ensure_in_register(idx)?;

        self.save_all_to_stack()?;

        // Get all targets
        let default_target = targets.default();

        // For each target, emit a comparison and conditional jump
        // This is a simple linear search - for large tables, a jump table would be more efficient
        for (i, target) in targets.targets().enumerate() {
            let target = target.map_err(|e| CompileError::Parse(e.to_string()))?;
            let mut jump = self.emitter.cmp(Condition::Equal, idx_reg, i as i32)?;

            let block_idx = self.blocks.len() - 1 - target as usize;
            if let Some(mut label) = self.blocks[block_idx].label {
                jump.set_label(&mut label);
            } else {
                self.blocks[block_idx].end_jumps.push(jump);
            }
        }

        // Default case - jump to default target
        self.occupied_registers.set(idx_reg.index(), false);

        let block_idx = self.blocks.len() - 1 - default_target as usize;
        if let Some(mut label) = self.blocks[block_idx].label {
            let mut jump = self.emitter.jump(JumpType::Jump)?;
            jump.set_label(&mut label);
        } else {
            self.blocks[block_idx]
                .end_jumps
                .push(self.emitter.jump(JumpType::Jump)?);
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
#[repr(transparent)]
#[derive(Deref)]
pub struct CompiledFunction {
    pub(crate) code: GeneratedCode,
}

impl CompiledFunction {
    /// Transmute the compiled code to a function pointer of the specified type.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the type `F` matches the actual signature
    /// of the compiled WebAssembly function. Mismatched types will result in
    /// undefined behavior.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // For a function with no parameters returning i32:
    /// let f = compiled.as_fn::<fn() -> i32>();
    ///
    /// // For a function with two i32 parameters returning i32:
    /// let f = compiled.as_fn::<fn(i32, i32) -> i32>();
    ///
    /// // For a function with no return value:
    /// let f = compiled.as_fn::<fn(i32)>();
    /// ```
    #[inline(always)]
    pub fn as_fn<F>(&self) -> F {
        unsafe { std::mem::transmute_copy(&self.code.get()) }
    }
}

/// Compile a simple WebAssembly function from operators
pub fn compile_simple(
    params: &[ValType],
    results: &[ValType],
    locals: &[ValType],
    body: &[Operator],
) -> Result<CompiledFunction, CompileError> {
    Function::new().compile_function(params, results, locals, body.iter().cloned())
}
