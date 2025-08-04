use sljit_sys::{self as sys, Compiler, ErrorCode, sljit_sw};

#[repr(i32)]
#[derive(Clone, Copy, Debug)]
/// A scratch register.
pub enum ScratchRegister {
    /// Scratch register 0.
    R0 = crate::sys::SLJIT_R0,
    /// Scratch register 1.
    R1 = crate::sys::SLJIT_R1,
    /// Scratch register 2.
    R2 = crate::sys::SLJIT_R2,
    /// Scratch register 3.
    R3 = crate::sys::SLJIT_R3,
    /// Scratch register 4.
    R4 = crate::sys::SLJIT_R4,
    /// Scratch register 5.
    R5 = crate::sys::SLJIT_R5,
    /// Scratch register 6.
    R6 = crate::sys::SLJIT_R6,
    /// Scratch register 7.
    R7 = crate::sys::SLJIT_R7,
    /// Scratch register 8.
    R8 = crate::sys::SLJIT_R8,
    /// Scratch register 9.
    R9 = crate::sys::SLJIT_R9,
}

#[repr(i32)]
#[derive(Clone, Copy, Debug)]
/// A saved register.
pub enum SavedRegister {
    /// Saved register 0.
    S0 = crate::sys::SLJIT_S0,
    /// Saved register 1.
    S1 = crate::sys::SLJIT_S1,
    /// Saved register 2.
    S2 = crate::sys::SLJIT_S2,
    /// Saved register 3.
    S3 = crate::sys::SLJIT_S3,
    /// Saved register 4.
    S4 = crate::sys::SLJIT_S4,
    /// Saved register 5.
    S5 = crate::sys::SLJIT_S5,
    /// Saved register 6.
    S6 = crate::sys::SLJIT_S6,
    /// Saved register 7.
    S7 = crate::sys::SLJIT_S7,
    /// Saved register 8.
    S8 = crate::sys::SLJIT_S8,
    /// Saved register 9.
    S9 = crate::sys::SLJIT_S9,
}

#[repr(i32)]
#[derive(Clone, Copy, Debug)]
/// A float register.
pub enum FloatRegister {
    /// Float register 0.
    FR0 = crate::sys::SLJIT_FR0,
    /// Float register 1.
    FR1 = crate::sys::SLJIT_FR1,
    /// Float register 2.
    FR2 = crate::sys::SLJIT_FR2,
    /// Float register 3.
    FR3 = crate::sys::SLJIT_FR3,
    /// Float register 4.
    FR4 = crate::sys::SLJIT_FR4,
    /// Float register 5.
    FR5 = crate::sys::SLJIT_FR5,
    /// Float register 6.
    FR6 = crate::sys::SLJIT_FR6,
    /// Float register 7.
    FR7 = crate::sys::SLJIT_FR7,
    /// Float register 8.
    FR8 = crate::sys::SLJIT_FR8,
    /// Float register 9.
    FR9 = crate::sys::SLJIT_FR9,
}

#[repr(i32)]
#[derive(Clone, Copy, Debug)]
/// A saved float register.
pub enum SavedFloatRegister {
    /// Saved float register 0.
    FS0 = crate::sys::SLJIT_FS0,
    /// Saved float register 1.
    FS1 = crate::sys::SLJIT_FS1,
    /// Saved float register 2.
    FS2 = crate::sys::SLJIT_FS2,
    /// Saved float register 3.
    FS3 = crate::sys::SLJIT_FS3,
    /// Saved float register 4.
    FS4 = crate::sys::SLJIT_FS4,
    /// Saved float register 5.
    FS5 = crate::sys::SLJIT_FS5,
    /// Saved float register 6.
    FS6 = crate::sys::SLJIT_FS6,
    /// Saved float register 7.
    FS7 = crate::sys::SLJIT_FS7,
    /// Saved float register 8.
    FS8 = crate::sys::SLJIT_FS8,
    /// Saved float register 9.
    FS9 = crate::sys::SLJIT_FS9,
}

#[repr(i32)]
#[derive(Clone, Copy, Debug)]
/// A vector register.
pub enum VectorRegister {
    /// Vector register 0.
    VR0 = crate::sys::SLJIT_VR0,
    /// Vector register 1.
    VR1 = crate::sys::SLJIT_VR1,
    /// Vector register 2.
    VR2 = crate::sys::SLJIT_VR2,
    /// Vector register 3.
    VR3 = crate::sys::SLJIT_VR3,
    /// Vector register 4.
    VR4 = crate::sys::SLJIT_VR4,
    /// Vector register 5.
    VR5 = crate::sys::SLJIT_VR5,
}

#[repr(i32)]
#[derive(Clone, Copy, Debug)]
/// A saved vector register.
pub enum SavedVectorRegister {
    /// Saved vector register 0.
    VS0 = crate::sys::SLJIT_VS0,
    /// Saved vector register 1.
    VS1 = crate::sys::SLJIT_VS1,
    /// Saved vector register 2.
    VS2 = crate::sys::SLJIT_VS2,
    /// Saved vector register 3.
    VS3 = crate::sys::SLJIT_VS3,
    /// Saved vector register 4.
    VS4 = crate::sys::SLJIT_VS4,
}

#[repr(i32)]
#[derive(Clone, Copy, Debug)]
/// A condition for a jump.
pub enum Condition {
    /// The values are equal.
    Equal = sys::SLJIT_EQUAL,
    /// The values are not equal.
    NotEqual = sys::SLJIT_NOT_EQUAL,
    /// The first value is less than the second value (unsigned).
    Less = sys::SLJIT_LESS,
    /// The first value is less than the second value (unsigned, 32-bit).
    Less32 = sys::SLJIT_LESS | sys::SLJIT_32,
    /// The first value is greater than or equal to the second value (unsigned).
    GreaterEqual = sys::SLJIT_GREATER_EQUAL,
    /// The first value is greater than or equal to the second value (unsigned, 32-bit).
    GreaterEqual32 = sys::SLJIT_GREATER_EQUAL | sys::SLJIT_32,
    /// The first value is greater than the second value (unsigned).
    Greater = sys::SLJIT_GREATER,
    /// The first value is greater than the second value (unsigned, 32-bit).
    Greater32 = sys::SLJIT_GREATER | sys::SLJIT_32,
    /// The first value is less than or equal to the second value (unsigned).
    LessEqual = sys::SLJIT_LESS_EQUAL,
    /// The first value is less than or equal to the second value (unsigned, 32-bit).
    LessEqual32 = sys::SLJIT_LESS_EQUAL | sys::SLJIT_32,
    /// The first value is less than the second value (signed).
    SigLess = sys::SLJIT_SIG_LESS,
    /// The first value is less than the second value (signed, 32-bit).
    SigLess32 = sys::SLJIT_SIG_LESS | sys::SLJIT_32,
    /// The first value is greater than or equal to the second value (signed).
    SigGreaterEqual = sys::SLJIT_SIG_GREATER_EQUAL,
    /// The first value is greater than or equal to the second value (signed, 32-bit).
    SigGreaterEqual32 = sys::SLJIT_SIG_GREATER_EQUAL | sys::SLJIT_32,
    /// The first value is greater than the second value (signed).
    SigGreater = sys::SLJIT_SIG_GREATER,
    /// The first value is greater than the second value (signed, 32-bit).
    SigGreater32 = sys::SLJIT_SIG_GREATER | sys::SLJIT_32,
    /// The first value is less than or equal to the second value (signed).
    SigLessEqual = sys::SLJIT_SIG_LESS_EQUAL,
    /// The first value is less than or equal to the second value (signed, 32-bit).
    SigLessEqual32 = sys::SLJIT_SIG_LESS_EQUAL | sys::SLJIT_32,
    /// An overflow occurred.
    Overflow = sys::SLJIT_OVERFLOW,
    /// An overflow did not occur.
    NotOverflow = sys::SLJIT_NOT_OVERFLOW,
    /// A carry occurred.
    Carry = sys::SLJIT_CARRY,
    /// A carry did not occur.
    NotCarry = sys::SLJIT_NOT_CARRY,
    /// An atomic store was successful.
    AtomicStored = sys::SLJIT_ATOMIC_STORED,
    /// An atomic store was not successful.
    AtomicNotStored = sys::SLJIT_ATOMIC_NOT_STORED,
    /// The float values are equal.
    FEqual = sys::SLJIT_F_EQUAL,
    /// The float values are not equal.
    FNotEqual = sys::SLJIT_F_NOT_EQUAL,
    /// The first float value is less than the second float value.
    FLess = sys::SLJIT_F_LESS,
    /// The first float value is greater than or equal to the second float value.
    FGreaterEqual = sys::SLJIT_F_GREATER_EQUAL,
    /// The first float value is greater than the second float value.
    FGreater = sys::SLJIT_F_GREATER,
    /// The first float value is less than or equal to the second float value.
    FLessEqual = sys::SLJIT_F_LESS_EQUAL,
    /// The float values are unordered.
    Unordered = sys::SLJIT_UNORDERED,
    /// The float values are ordered.
    Ordered = sys::SLJIT_ORDERED,
    /// The float values are ordered and equal.
    OrderedEqual = sys::SLJIT_ORDERED_EQUAL,
    /// The float values are unordered or not equal.
    UnorderedOrNotEqual = sys::SLJIT_UNORDERED_OR_NOT_EQUAL,
    /// The float values are ordered and the first is less than the second.
    OrderedLess = sys::SLJIT_ORDERED_LESS,
    /// The float values are unordered or the first is greater than or equal to the second.
    UnorderedOrGreaterEqual = sys::SLJIT_UNORDERED_OR_GREATER_EQUAL,
    /// The float values are ordered and the first is greater than the second.
    OrderedGreater = sys::SLJIT_ORDERED_GREATER,
    /// The float values are unordered or the first is less than or equal to the second.
    UnorderedOrLessEqual = sys::SLJIT_UNORDERED_OR_LESS_EQUAL,
    /// The float values are unordered or equal.
    UnorderedOrEqual = sys::SLJIT_UNORDERED_OR_EQUAL,
    /// The float values are ordered and not equal.
    OrderedNotEqual = sys::SLJIT_ORDERED_NOT_EQUAL,
    /// The float values are unordered or the first is less than the second.
    UnorderedOrLess = sys::SLJIT_UNORDERED_OR_LESS,
    /// The float values are ordered and the first is greater than or equal to the second.
    OrderedGreaterEqual = sys::SLJIT_ORDERED_GREATER_EQUAL,
    /// The float values are unordered or the first is greater than the second.
    UnorderedOrGreater = sys::SLJIT_UNORDERED_OR_GREATER,
    /// The float values are ordered and the first is less than or equal to the second.
    OrderedLessEqual = sys::SLJIT_ORDERED_LESS_EQUAL,
}

impl Condition {
    /// Inverts the condition.
    #[inline(always)]
    pub fn invert(self) -> Self {
        unsafe { core::mem::transmute((self as i32) ^ 1) }
    }
}

#[repr(i32)]
#[derive(Clone, Copy, Debug)]
/// The type of a jump.
pub enum JumpType {
    /// A simple jump.
    Jump = sys::SLJIT_JUMP,
    /// A fast call.
    FastCall = sys::SLJIT_FAST_CALL,
    /// A call.
    Call = sys::SLJIT_CALL,
    /// A call with a register argument.
    CallRegArg = sys::SLJIT_CALL_REG_ARG,
}

#[derive(Clone, Copy, Debug)]
/// An operand for an instruction.
pub struct Operand(i32, sljit_sw);

macro_rules! define_register_types {
    (
        gp: [$($gp_reg:ty),*],
        float: [$($float_reg:ty),*],
    ) => {
        /// A general-purpose register.
        pub trait GpRegister: Into<Operand> {}
        $(
            impl GpRegister for $gp_reg {}
        )*

        /// A float register.
        pub trait FloatRegisterType: Into<Operand> {}
        $(
            impl FloatRegisterType for $float_reg {}
        )*

        /// A vector register.
        pub trait VectorRegisterType: Into<Operand> {}
        impl VectorRegisterType for VectorRegister {}
        impl VectorRegisterType for SavedVectorRegister {}

        $(
            impl From<$gp_reg> for Operand {
                #[inline(always)]
                fn from(reg: $gp_reg) -> Self {
                    Self(reg as i32, 0)
                }
            }
        )*

        $(
            impl From<$float_reg> for Operand {
                #[inline(always)]
                fn from(reg: $float_reg) -> Self {
                    Self(reg as i32, 0)
                }
            }
        )*

        impl From<VectorRegister> for Operand {
            #[inline(always)]
            fn from(reg: VectorRegister) -> Self {
                Self(reg as i32, 0)
            }
        }

        impl From<SavedVectorRegister> for Operand {
            #[inline(always)]
            fn from(reg: SavedVectorRegister) -> Self {
                Self(reg as i32, 0)
            }
        }
    };
}

define_register_types! {
    gp: [ScratchRegister, SavedRegister],
    float: [FloatRegister, SavedFloatRegister],
}

macro_rules! impl_from_imm_for_operand {
    ($($ty:ty),*) => {
        $(
            impl From<$ty> for Operand {
                #[inline(always)]
                fn from(imm: $ty) -> Self {
                    Self(sys::SLJIT_IMM, imm as sljit_sw)
                }
            }
        )*
    };
}

impl_from_imm_for_operand!(i32, u32, isize, usize);

impl From<Operand> for (i32, sljit_sw) {
    #[inline(always)]
    fn from(op: Operand) -> Self {
        (op.0, op.1)
    }
}

/// A memory operand.
#[inline(always)]
pub fn mem(base: impl Into<Operand>) -> Operand {
    let mut op = base.into();
    op.0 = sys::mem!(op.0);
    op
}

/// A memory operand with an offset.
#[inline(always)]
pub fn mem_offset(base: impl Into<Operand>, offset: i32) -> Operand {
    let mut op = base.into();
    op.0 = sys::mem!(op.0);
    op.1 = offset as sljit_sw;
    op
}

/// A memory operand with an indexed shift.
#[inline(always)]
pub fn mem_indexed_shift(
    base: impl Into<Operand>,
    index: impl Into<Operand>,
    shift: u8,
) -> Operand {
    let base_op = base.into();
    let index_op = index.into();
    Operand(sys::mem!(base_op.0, index_op.0), shift as sljit_sw)
}

/// A macro to build the register argument for `emit_enter`.
///
/// # Examples
///
/// ```
/// # use sljit::regs;
/// // Set only the number of general-purpose scratch registers.
/// let scratches = regs!(3);
/// assert_eq!(scratches, 3);
///
/// // Set the number of general-purpose, float, and vector scratch registers.
/// let scratches = regs! {
///     gp: 3,
///     float: 2,
///     vector: 1,
/// };
/// assert_eq!(scratches, 3 | (2 << 8) | (1 << 16));
///
/// // You can omit fields that are zero, and the order does not matter.
/// let scratches = regs! {
///    float: 1,
///    gp: 1,
/// };
/// assert_eq!(scratches, 1 | (1 << 8));
/// ```
#[macro_export]
macro_rules! regs {
    // simple gp
    ($gp:expr) => { $gp };
    // empty
    {} => { 0 };
    // full
    { gp: $gp:expr, float: $float:expr, vector: $vector:expr $(,)? } => { $gp | ($float << 8) | ($vector << 16) };
    { gp: $gp:expr, vector: $vector:expr, float: $float:expr $(,)? } => { $gp | ($float << 8) | ($vector << 16) };
    { float: $float:expr, gp: $gp:expr, vector: $vector:expr $(,)? } => { $gp | ($float << 8) | ($vector << 16) };
    { float: $float:expr, vector: $vector:expr, gp: $gp:expr $(,)? } => { $gp | ($float << 8) | ($vector << 16) };
    { vector: $vector:expr, gp: $gp:expr, float: $float:expr $(,)? } => { $gp | ($float << 8) | ($vector << 16) };
    { vector: $vector:expr, float: $float:expr, gp: $gp:expr $(,)? } => { $gp | ($float << 8) | ($vector << 16) };
    // two fields
    { gp: $gp:expr, float: $float:expr $(,)? } => { $gp | ($float << 8) };
    { float: $float:expr, gp: $gp:expr $(,)? } => { $gp | ($float << 8) };
    { gp: $gp:expr, vector: $vector:expr $(,)? } => { $gp | ($vector << 16) };
    { vector: $vector:expr, gp: $gp:expr $(,)? } => { $gp | ($vector << 16) };
    { float: $float:expr, vector: $vector:expr $(,)? } => { ($float << 8) | ($vector << 16) };
    { vector: $vector:expr, float: $float:expr $(,)? } => { ($float << 8) | ($vector << 16) };
    // one field
    { gp: $gp:expr $(,)? } => { $gp };
    { float: $float:expr $(,)? } => { $float << 8 };
    { vector: $vector:expr $(,)? } => { $vector << 16 };
}

macro_rules! define_emitter_ops {
    // For op0
    ($($(#[doc = $doc:literal])* $name:ident, $opcode:expr, 0;)*) => {
        $(
            $(#[doc = $doc])*
            #[inline(always)]
            pub fn $name(&mut self) -> Result<&mut Self, ErrorCode> {
                self.compiler.emit_op0($opcode)?;
                Ok(self)
            }
        )*
    };
    // For op1
    ($($(#[doc = $doc:literal])* $name:ident, $opcode:expr, 1;)*) => {
        $(
            $(#[doc = $doc])*
            #[inline(always)]
            pub fn $name(&mut self, flags: i32, dst: impl Into<Operand>, src: impl Into<Operand>) -> Result<&mut Self, ErrorCode> {
                let (dst, dstw) = dst.into().into();
                let (src, srcw) = src.into().into();
                self.compiler.emit_op1($opcode | flags, dst, dstw, src, srcw)?;
                Ok(self)
            }
        )*
    };
    // For op2
    ($($(#[doc = $doc:literal])* $name:ident, $opcode:expr, 2;)*) => {
        $(
            $(#[doc = $doc])*
            #[inline(always)]
            pub fn $name(&mut self, flags: i32, dst: impl Into<Operand>, src1: impl Into<Operand>, src2: impl Into<Operand>) -> Result<&mut Self, ErrorCode> {
                let (dst, dstw) = dst.into().into();
                let (src1, src1w) = src1.into().into();
                let (src2, src2w) = src2.into().into();
                self.compiler.emit_op2($opcode | flags, dst, dstw, src1, src1w, src2, src2w)?;
                Ok(self)
            }
        )*
    };
    // For op2r
    ($($(#[doc = $doc:literal])* $name:ident, $opcode:expr, 2r;)*) => {
        $(
            $(#[doc = $doc])*
            #[inline(always)]
            pub fn $name<R: GpRegister>(&mut self, flags: i32, dst: R, src1: impl Into<Operand>, src2: impl Into<Operand>) -> Result<&mut Self, ErrorCode> {
                let (dst, _) = dst.into().into();
                let (src1, src1w) = src1.into().into();
                let (src2, src2w) = src2.into().into();
                self.compiler.emit_op2r($opcode | flags, dst, src1, src1w, src2, src2w)?;
                Ok(self)
            }
        )*
    };
    // For fop1
    ($($(#[doc = $doc:literal])* $name:ident, $opcode:expr, f1;)*) => {
        $(
            $(#[doc = $doc])*
            #[inline(always)]
            pub fn $name(&mut self, flags: i32, dst: impl Into<Operand>, src: impl Into<Operand>) -> Result<&mut Self, ErrorCode> {
                let (dst, dstw) = dst.into().into();
                let (src, srcw) = src.into().into();
                self.compiler.emit_fop1($opcode | flags, dst, dstw, src, srcw)?;
                Ok(self)
            }
        )*
    };
    // For fop2
    ($($(#[doc = $doc:literal])* $name:ident, $opcode:expr, f2;)*) => {
        $(
            $(#[doc = $doc])*
            #[inline(always)]
            pub fn $name(&mut self, flags: i32, dst: impl Into<Operand>, src1: impl Into<Operand>, src2: impl Into<Operand>) -> Result<&mut Self, ErrorCode> {
                let (dst, dstw) = dst.into().into();
                let (src1, src1w) = src1.into().into();
                let (src2, src2w) = src2.into().into();
                self.compiler.emit_fop2($opcode | flags, dst, dstw, src1, src1w, src2, src2w)?;
                Ok(self)
            }
        )*
    };
    // For fop2r
    ($($(#[doc = $doc:literal])* $name:ident, $opcode:expr, f2r;)*) => {
        $(
            $(#[doc = $doc])*
            #[inline(always)]
            pub fn $name<R: FloatRegisterType>(&mut self, flags: i32, dst: R, src1: impl Into<Operand>, src2: impl Into<Operand>) -> Result<&mut Self, ErrorCode> {
                let (dst, _) = dst.into().into();
                let (src1, src1w) = src1.into().into();
                let (src2, src2w) = src2.into().into();
                self.compiler.emit_fop2r($opcode | flags, dst, src1, src1w, src2, src2w)?;
                Ok(self)
            }
        )*
    };
    // For op_src
    ($($(#[doc = $doc:literal])* $name:ident, $opcode:expr, op_src;)*) => {
        $(
            $(#[doc = $doc])*
            #[inline(always)]
            pub fn $name(&mut self, src: impl Into<Operand>) -> Result<&mut Self, ErrorCode> {
                let (src, srcw) = src.into().into();
                self.compiler.emit_op_src($opcode, src, srcw)?;
                Ok(self)
            }
        )*
    };
    // For op_dst
    ($($(#[doc = $doc:literal])* $name:ident, $opcode:expr, op_dst;)*) => {
        $(
            $(#[doc = $doc])*
            #[inline(always)]
            pub fn $name(&mut self, dst: impl Into<Operand>) -> Result<&mut Self, ErrorCode> {
                let (dst, dstw) = dst.into().into();
                self.compiler.emit_op_dst($opcode, dst, dstw)?;
                Ok(self)
            }
        )*
    };
}

macro_rules! define_simd_vreg_srcdst {
    ($($(#[doc = $doc:literal])* $name:ident, $compiler_fn:ident;)*) => {
        $(
            $(#[doc = $doc])*
            #[inline(always)]
            pub fn $name(&mut self, type_: i32, vreg: impl Into<Operand>, srcdst: impl Into<Operand>) -> Result<&mut Self, ErrorCode> {
                let (vreg, _) = vreg.into().into();
                let (srcdst, srcdstw) = srcdst.into().into();
                self.compiler.$compiler_fn(type_, vreg, srcdst, srcdstw)?;
                Ok(self)
            }
        )*
    };
}

macro_rules! define_simd_vreg_idx_srcdst {
    ($($(#[doc = $doc:literal])* $name:ident, $compiler_fn:ident;)*) => {
        $(
            $(#[doc = $doc])*
            #[inline(always)]
            pub fn $name(&mut self, type_: i32, vreg: impl Into<Operand>, lane_index: i32, srcdst: impl Into<Operand>) -> Result<&mut Self, ErrorCode> {
                let (vreg, _) = vreg.into().into();
                let (srcdst, srcdstw) = srcdst.into().into();
                self.compiler.$compiler_fn(type_, vreg, lane_index, srcdst, srcdstw)?;
                Ok(self)
            }
        )*
    };
}

macro_rules! define_simd_vreg_src_idx {
    ($($(#[doc = $doc:literal])* $name:ident, $compiler_fn:ident;)*) => {
        $(
            $(#[doc = $doc])*
            #[inline(always)]
            pub fn $name(&mut self, type_: i32, vreg: impl Into<Operand>, src: impl Into<Operand>, src_lane_index: impl Into<Operand>) -> Result<&mut Self, ErrorCode> {
                let (vreg, _) = vreg.into().into();
                let (src, _) = src.into().into();
                let (src_lane_index, _) = src_lane_index.into().into();
                self.compiler.$compiler_fn(type_, vreg, src, src_lane_index)?;
                Ok(self)
            }
        )*
    };
}

macro_rules! define_simd_op2_vreg {
    ($($(#[doc = $doc:literal])* $name:ident, $compiler_fn:ident;)*) => {
        $(
            $(#[doc = $doc])*
            #[inline(always)]
            pub fn $name(&mut self, type_: i32, dst_vreg: impl Into<Operand>, src1_vreg: impl Into<Operand>, src2: impl Into<Operand>) -> Result<&mut Self, ErrorCode> {
                let (dst_vreg, _) = dst_vreg.into().into();
                let (src1_vreg, _) = src1_vreg.into().into();
                let (src2, src2w) = src2.into().into();
                self.compiler.$compiler_fn(type_, dst_vreg, src1_vreg, src2, src2w)?;
                Ok(self)
            }
        )*
    };
}

#[repr(i32)]
#[derive(Clone, Copy, Debug)]
/// An operation for a return instruction.
pub enum ReturnOp {
    /// Move a value.
    Mov = sys::Op1::Mov as i32,
    /// Move an unsigned 8-bit value.
    MovU8 = sys::Op1::MovU8 as i32,
    /// Move a signed 8-bit value.
    MovS8 = sys::Op1::MovS8 as i32,
    /// Move an unsigned 16-bit value.
    MovU16 = sys::Op1::MovU16 as i32,
    /// Move a signed 16-bit value.
    MovS16 = sys::Op1::MovS16 as i32,
    /// Move an unsigned 32-bit value.
    MovU32 = sys::Op1::MovU32 as i32,
    /// Move a signed 32-bit value.
    MovS32 = sys::Op1::MovS32 as i32,
    /// Move a 32-bit value.
    Mov32 = sys::Op1::Mov32 as i32,
    /// Move a pointer.
    MovP = sys::Op1::MovP as i32,
    /// Move a 64-bit float.
    MovF64 = sys::Fop1::MovF64 as i32,
    /// Move a 32-bit float.
    MovF32 = sys::Fop1::MovF32 as i32,
}

/// An emitter for instructions.
pub struct Emitter<'a> {
    compiler: &'a mut Compiler,
}

impl<'a> Emitter<'a> {
    /// Creates a new emitter.
    pub fn new(compiler: &'a mut Compiler) -> Self {
        Self { compiler }
    }

    define_emitter_ops! {
        /// Emits a `BREAKPOINT` instruction.
        breakpoint, sys::Op0::Breakpoint as i32, 0;
        /// Emits a `NOP` instruction.
        nop, sys::Op0::Nop as i32, 0;
        /// Emits a `LMUL.UW` instruction.
        lmul_uword, sys::Op0::LmulUw as i32, 0;
        /// Emits a `LMUL.SW` instruction.
        lmul_sword, sys::Op0::LmulSw as i32, 0;
        /// Emits a `DIVMOD.S32` instruction.
        divmod_s32, sys::Op0::DivmodS32 as i32, 0;
        /// Emits a `DIVMOD.SW` instruction.
        divmod_sword, sys::Op0::DivmodSw as i32, 0;
        /// Emits a `DIVMOD.U32` instruction.
        divmod_u32, sys::Op0::DivmodU32 as i32, 0;
        /// Emits a `DIVMOD.UW` instruction.
        divmod_uword, sys::Op0::DivmodUw as i32, 0;
        /// Emits a `DIV.S32` instruction.
        div_s32, sys::Op0::DivS32 as i32, 0;
        /// Emits a `DIV.SW` instruction.
        div_sword, sys::Op0::DivSw as i32, 0;
        /// Emits a `DIV.U32` instruction.
        div_u32, sys::Op0::DivU32 as i32, 0;
        /// Emits a `DIV.UW` instruction.
        div_uword, sys::Op0::DivUw as i32, 0;
        /// Emits an `ENDBR` instruction.
        endbr, sys::Op0::Endbr as i32, 0;
        /// Emits a `SKIP_FRAMES_BEFORE_RETURN` instruction.
        skip_frames_before_return, sys::Op0::SkipFramesBeforeReturn as i32, 0;
    }

    define_emitter_ops! {
        /// Emits a `MOV` instruction.
        mov, sys::Op1::Mov as i32, 1;
        /// Emits a `MOV.U8` instruction.
        mov_u8, sys::Op1::MovU8 as i32, 1;
        /// Emits a `MOV.S8` instruction.
        mov_s8, sys::Op1::MovS8 as i32, 1;
        /// Emits a `MOV.U16` instruction.
        mov_u16, sys::Op1::MovU16 as i32, 1;
        /// Emits a `MOV.S16` instruction.
        mov_s16, sys::Op1::MovS16 as i32, 1;
        /// Emits a `MOV.U32` instruction.
        mov_u32, sys::Op1::MovU32 as i32, 1;
        /// Emits a `MOV.S32` instruction.
        mov_s32, sys::Op1::MovS32 as i32, 1;
        /// Emits a `CTZ` instruction.
        ctz, sys::Op1::Ctz as i32, 1;
        /// Emits a `CLZ` instruction.
        clz, sys::Op1::Clz as i32, 1;
        /// Emits a `REV` instruction.
        rev, sys::Op1::Rev as i32, 1;
        /// Emits a `MOV32.U8` instruction.
        mov32_u8, sys::Op1::Mov32U8 as i32, 1;
        /// Emits a `MOV32.S8` instruction.
        mov32_s8, sys::Op1::Mov32S8 as i32, 1;
        /// Emits a `MOV32.U16` instruction.
        mov32_u16, sys::Op1::Mov32U16 as i32, 1;
        /// Emits a `MOV32.S16` instruction.
        mov32_s16, sys::Op1::Mov32S16 as i32, 1;
        /// Emits a `MOV32` instruction.
        mov32, sys::Op1::Mov32 as i32, 1;
        /// Emits a `CLZ32` instruction.
        clz32, sys::Op1::Clz32 as i32, 1;
        /// Emits a `CTZ32` instruction.
        ctz32, sys::Op1::Ctz32 as i32, 1;
        /// Emits a `REV32` instruction.
        rev32, sys::Op1::Rev32 as i32, 1;
        /// Emits a `REV.U16` instruction.
        rev_u16, sys::Op1::RevU16 as i32, 1;
        /// Emits a `REV32.U16` instruction.
        rev32_u16, sys::Op1::Rev32U16 as i32, 1;
        /// Emits a `REV.S16` instruction.
        rev_s16, sys::Op1::RevS16 as i32, 1;
        /// Emits a `REV32.S16` instruction.
        rev32_s16, sys::Op1::Rev32S16 as i32, 1;
        /// Emits a `REV.U32` instruction.
        rev_u32, sys::Op1::RevU32 as i32, 1;
        /// Emits a `REV.S32` instruction.
        rev_s32, sys::Op1::RevS32 as i32, 1;
    }

    define_emitter_ops! {
        /// Emits an `ADD` instruction.
        add, sys::Op2::Add as i32, 2;
        /// Emits a `SUB` instruction.
        sub, sys::Op2::Sub as i32, 2;
        /// Emits a `MUL` instruction.
        mul, sys::Op2::Mul as i32, 2;
        /// Emits an `AND` instruction.
        and, sys::Op2::And as i32, 2;
        /// Emits an `OR` instruction.
        or, sys::Op2::Or as i32, 2;
        /// Emits a `XOR` instruction.
        xor, sys::Op2::Xor as i32, 2;
        /// Emits a `SHL` instruction.
        shl, sys::Op2::Shl as i32, 2;
        /// Emits a `MSHL` instruction.
        mshl, sys::Op2::Mshl as i32, 2;
        /// Emits a `LSHR` instruction.
        lshr, sys::Op2::Lshr as i32, 2;
        /// Emits a `MLSHR` instruction.
        mlshr, sys::Op2::Mlshr as i32, 2;
        /// Emits an `ASHR` instruction.
        ashr, sys::Op2::Ashr as i32, 2;
        /// Emits a `MASHR` instruction.
        mashr, sys::Op2::Mashr as i32, 2;
        /// Emits an `ADD32` instruction.
        add32, sys::Op2::Add32 as i32, 2;
        /// Emits an `ADDC` instruction.
        addc, sys::Op2::Addc as i32, 2;
        /// Emits an `ADDC32` instruction.
        addc32, sys::Op2::Addc32 as i32, 2;
        /// Emits a `SUB32` instruction.
        sub32, sys::Op2::Sub32 as i32, 2;
        /// Emits a `SUBC` instruction.
        subc, sys::Op2::Subc as i32, 2;
        /// Emits a `SUBC32` instruction.
        subc32, sys::Op2::Subc32 as i32, 2;
        /// Emits a `MUL32` instruction.
        mul32, sys::Op2::Mul32 as i32, 2;
        /// Emits an `AND32` instruction.
        and32, sys::Op2::And32 as i32, 2;
        /// Emits an `OR32` instruction.
        or32, sys::Op2::Or32 as i32, 2;
        /// Emits a `XOR32` instruction.
        xor32, sys::Op2::Xor32 as i32, 2;
        /// Emits a `SHL32` instruction.
        shl32, sys::Op2::Shl32 as i32, 2;
        /// Emits a `MSHL32` instruction.
        mshl32, sys::Op2::Mshl32 as i32, 2;
        /// Emits a `LSHR32` instruction.
        lshr32, sys::Op2::Lshr32 as i32, 2;
        /// Emits a `MLSHR32` instruction.
        mlshr32, sys::Op2::Mlshr32 as i32, 2;
        /// Emits an `ASHR32` instruction.
        ashr32, sys::Op2::Ashr32 as i32, 2;
        /// Emits a `MASHR32` instruction.
        mashr32, sys::Op2::Mashr32 as i32, 2;
        /// Emits a `ROTL` instruction.
        rotl, sys::Op2::Rotl as i32, 2;
        /// Emits a `ROTL32` instruction.
        rotl32, sys::Op2::Rotl32 as i32, 2;
        /// Emits a `ROTR` instruction.
        rotr, sys::Op2::Rotr as i32, 2;
        /// Emits a `ROTR32` instruction.
        rotr32, sys::Op2::Rotr32 as i32, 2;
    }

    define_emitter_ops! {
        /// Emits a `MULADD` instruction.
        muladd, sys::Op2r::Muladd as i32, 2r;
        /// Emits a `MULADD32` instruction.
        muladd32, sys::Op2r::Muladd32 as i32, 2r;
    }

    define_emitter_ops! {
        /// Emits a `MOV.F64` instruction.
        mov_f64, sys::Fop1::MovF64 as i32, f1;
        /// Emits a `MOV.F32` instruction.
        mov_f32, sys::Fop1::MovF32 as i32, f1;
        /// Emits a `CONV.F64.FROM.F32` instruction.
        conv_f64_from_f32, sys::Fop1::ConvF64FromF32 as i32, f1;
        /// Emits a `CONV.F32.FROM.F64` instruction.
        conv_f32_from_f64, sys::Fop1::ConvF32FromF64 as i32, f1;
        /// Emits a `CONV.SW.FROM.F64` instruction.
        conv_sw_from_f64, sys::Fop1::ConvSwFromF64 as i32, f1;
        /// Emits a `CONV.SW.FROM.F32` instruction.
        conv_sw_from_f32, sys::Fop1::ConvSwFromF32 as i32, f1;
        /// Emits a `CONV.S32.FROM.F64` instruction.
        conv_s32_from_f64, sys::Fop1::ConvS32FromF64 as i32, f1;
        /// Emits a `CONV.S32.FROM.F32` instruction.
        conv_s32_from_f32, sys::Fop1::ConvS32FromF32 as i32, f1;
        /// Emits a `CONV.F64.FROM.SW` instruction.
        conv_f64_from_sw, sys::Fop1::ConvF64FromSw as i32, f1;
        /// Emits a `CONV.F32.FROM.SW` instruction.
        conv_f32_from_sw, sys::Fop1::ConvF32FromSw as i32, f1;
        /// Emits a `CONV.F64.FROM.S32` instruction.
        conv_f64_from_s32, sys::Fop1::ConvF64FromS32 as i32, f1;
        /// Emits a `CONV.F32.FROM.S32` instruction.
        conv_f32_from_s32, sys::Fop1::ConvF32FromS32 as i32, f1;
        /// Emits a `CONV.F64.FROM.UW` instruction.
        conv_f64_from_uw, sys::Fop1::ConvF64FromUw as i32, f1;
        /// Emits a `CONV.F32.FROM.UW` instruction.
        conv_f32_from_uw, sys::Fop1::ConvF32FromUw as i32, f1;
        /// Emits a `CONV.F64.FROM.U32` instruction.
        conv_f64_from_u32, sys::Fop1::ConvF64FromU32 as i32, f1;
        /// Emits a `CONV.F32.FROM.U32` instruction.
        conv_f32_from_u32, sys::Fop1::ConvF32FromU32 as i32, f1;
        /// Emits a `CMP.F64` instruction.
        cmp_f64, sys::Fop1::CmpF64 as i32, f1;
        /// Emits a `CMP.F32` instruction.
        cmp_f32, sys::Fop1::CmpF32 as i32, f1;
        /// Emits a `NEG.F64` instruction.
        neg_f64, sys::Fop1::NegF64 as i32, f1;
        /// Emits a `NEG.F32` instruction.
        neg_f32, sys::Fop1::NegF32 as i32, f1;
        /// Emits an `ABS.F64` instruction.
        abs_f64, sys::Fop1::AbsF64 as i32, f1;
        /// Emits an `ABS.F32` instruction.
        abs_f32, sys::Fop1::AbsF32 as i32, f1;
    }

    define_emitter_ops! {
        /// Emits an `ADD.F64` instruction.
        add_f64, sys::Fop2::AddF64 as i32, f2;
        /// Emits an `ADD.F32` instruction.
        add_f32, sys::Fop2::AddF32 as i32, f2;
        /// Emits a `SUB.F64` instruction.
        sub_f64, sys::Fop2::SubF64 as i32, f2;
        /// Emits a `SUB.F32` instruction.
        sub_f32, sys::Fop2::SubF32 as i32, f2;
        /// Emits a `MUL.F64` instruction.
        mul_f64, sys::Fop2::MulF64 as i32, f2;
        /// Emits a `MUL.F32` instruction.
        mul_f32, sys::Fop2::MulF32 as i32, f2;
        /// Emits a `DIV.F64` instruction.
        div_f64, sys::Fop2::DivF64 as i32, f2;
        /// Emits a `DIV.F32` instruction.
        div_f32, sys::Fop2::DivF32 as i32, f2;
    }

    define_emitter_ops! {
        /// Emits a `COPYSIGN.F64` instruction.
        copysign_f64, sys::Fop2r::CopysignF64 as i32, f2r;
        /// Emits a `COPYSIGN.F32` instruction.
        copysign_f32, sys::Fop2r::CopysignF32 as i32, f2r;
    }

    define_emitter_ops! {
        /// Emits a `FAST_RETURN` instruction.
        fast_return, sys::OpSrc::FastReturn as i32, op_src;
        /// Emits a `SKIP_FRAMES_BEFORE_FAST_RETURN` instruction.
        skip_frames_before_fast_return, sys::OpSrc::SkipFramesBeforeFastReturn as i32, op_src;
        /// Emits a `PREFETCH_L1` instruction.
        prefetch_l1, sys::OpSrc::PrefetchL1 as i32, op_src;
        /// Emits a `PREFETCH_L2` instruction.
        prefetch_l2, sys::OpSrc::PrefetchL2 as i32, op_src;
        /// Emits a `PREFETCH_L3` instruction.
        prefetch_l3, sys::OpSrc::PrefetchL3 as i32, op_src;
        /// Emits a `PREFETCH_ONCE` instruction.
        prefetch_once, sys::OpSrc::PrefetchOnce as i32, op_src;
    }

    define_emitter_ops! {
        /// Emits a `FAST_ENTER` instruction.
        fast_enter, sys::OpDst::FastEnter as i32, op_dst;
        /// Emits a `GET_RETURN_ADDRESS` instruction.
        get_return_address, sys::OpDst::GetReturnAddress as i32, op_dst;
    }

    /// Emits an `ENTER` instruction, which is the entry point of a function.
    ///
    /// The `options` argument can be used to pass configuration options to the SLJIT compiler,
    /// which affects the generated code until another `emit_enter` or `set_context` is called.
    ///
    /// The function argument list is specified by the `arg_types` parameter.
    ///
    /// The register set used by the function must be declared as well. The number of scratch and
    /// saved registers available to the function must be passed to `emit_enter`.
    ///
    /// `emit_enter` is also capable of allocating a stack space for local data. The `local_size`
    /// argument contains the size in bytes of this local area, and it can be accessed using
    /// `mem(SP)`. The memory area is uninitialized.
    ///
    /// # Parameters
    ///
    /// * `options` - A bitmask of options.
    /// * `arg_types` - A bitmask of argument types.
    /// * `scratches` - The number of scratch registers to use for temporary values.
    /// * `saveds` - The number of saved registers to use for parameters.
    /// * `local_size` - The size of the local stack frame.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sljit::{Compiler, Emitter, args, regs};
    /// # let mut compiler = Compiler::new().unwrap();
    /// # let mut emitter = Emitter::new(&mut compiler);
    /// // Function with 3 arguments of type sljit_sw and 1 return value of type sljit_sw.
    /// // The function uses 1 scratch register and 3 saved registers.
    /// // No local stack frame is allocated.
    /// emitter.emit_enter(0, args!(W, W, W, W), regs!(1), regs!(3), 0);
    /// ```
    pub fn emit_enter(
        &mut self,
        options: i32,
        arg_types: i32,
        scratches: i32,
        saveds: i32,
        local_size: i32,
    ) -> Result<&mut Self, ErrorCode> {
        self.compiler
            .emit_enter(options, arg_types, scratches, saveds, local_size)?;
        Ok(self)
    }

    /// Emits a `JUMP` instruction.
    ///
    /// The destination is not set, only the type of the jump.
    ///
    /// # Parameters
    ///
    /// * `type_` - The type of the jump.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sljit::{Compiler, Emitter, JumpType};
    /// # let mut compiler = Compiler::new().unwrap();
    /// # let mut emitter = Emitter::new(&mut compiler);
    /// let mut jump = emitter.jump(JumpType::Jump).unwrap();
    /// let mut label = emitter.put_label().unwrap();
    /// jump.set_label(&mut label);
    /// ```
    pub fn jump(&mut self, type_: JumpType) -> Result<sys::Jump, ErrorCode> {
        Ok(self.compiler.emit_jump(type_ as i32))
    }

    /// Emits a C compiler (ABI) compatible function call.
    ///
    /// # Parameters
    ///
    /// * `type_` - The type of the call.
    /// * `arg_types` - A bitmask of argument types.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sljit::{Compiler, Emitter, JumpType, args};
    /// # let mut compiler = Compiler::new().unwrap();
    /// # let mut emitter = Emitter::new(&mut compiler);
    /// let mut call = emitter.call(JumpType::Call, args!(W, W)).unwrap();
    /// ```
    pub fn call(&mut self, type_: JumpType, arg_types: i32) -> Result<sys::Jump, ErrorCode> {
        Ok(self.compiler.emit_call(type_ as i32, arg_types))
    }

    /// Integer comparison.
    ///
    /// In most architectures it is implemented as a compare operation followed by a jump.
    ///
    /// # Parameters
    ///
    /// * `type_` - The type of the comparison.
    /// * `src1` - The first source operand.
    /// * `src2` - The second source operand.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sljit::{Compiler, Emitter, Condition, ScratchRegister::R0};
    /// # let mut compiler = Compiler::new().unwrap();
    /// # let mut emitter = Emitter::new(&mut compiler);
    /// let mut jump = emitter.cmp(Condition::Equal, R0, 5).unwrap();
    /// ```
    pub fn cmp(
        &mut self,
        type_: Condition,
        src1: impl Into<Operand>,
        src2: impl Into<Operand>,
    ) -> Result<sys::Jump, ErrorCode> {
        let (src1, src1w) = src1.into().into();
        let (src2, src2w) = src2.into().into();
        Ok(self
            .compiler
            .emit_cmp(type_ as i32, src1, src1w, src2, src2w))
    }

    /// Floating point comparison.
    ///
    /// In most architectures it is implemented as a compare operation followed by a jump.
    ///
    /// # Parameters
    ///
    /// * `type_` - The type of the comparison.
    /// * `src1` - The first source operand.
    /// * `src2` - The second source operand.
    ///
    /// # Examples
    ///
    /// ```
    /// # use sljit::{Compiler, Emitter, Condition, FloatRegister::FR0};
    /// # let mut compiler = Compiler::new().unwrap();
    /// # let mut emitter = Emitter::new(&mut compiler);
    /// let mut jump = emitter.fcmp(Condition::FEqual, FR0, 5.0f64).unwrap();
    /// ```
    pub fn fcmp(
        &mut self,
        type_: Condition,
        src1: impl Into<Operand>,
        src2: impl Into<Operand>,
    ) -> Result<sys::Jump, ErrorCode> {
        let (src1, src1w) = src1.into().into();
        let (src2, src2w) = src2.into().into();
        Ok(self
            .compiler
            .emit_fcmp(type_ as i32, src1, src1w, src2, src2w))
    }

    /// Emit an indirect jump or fast call.
    ///
    /// # Parameters
    ///
    /// * `type_` - The type of the jump.
    /// * `src` - The source operand.
    pub fn ijump(&mut self, type_: i32, src: impl Into<Operand>) -> Result<&mut Self, ErrorCode> {
        let (src, srcw) = src.into().into();
        self.compiler.emit_ijump(type_, src, srcw)?;
        Ok(self)
    }

    /// Emit a C compiler (ABI) compatible function call.
    ///
    /// # Parameters
    ///
    /// * `type_` - The type of the call.
    /// * `arg_types` - A bitmask of argument types.
    /// * `src` - The source operand.
    pub fn icall(
        &mut self,
        type_: i32,
        arg_types: i32,
        src: impl Into<Operand>,
    ) -> Result<&mut Self, ErrorCode> {
        let (src, srcw) = src.into().into();
        self.compiler.emit_icall(type_, arg_types, src, srcw)?;
        Ok(self)
    }

    /// Perform an operation using the conditional flags as the second argument.
    ///
    /// # Parameters
    ///
    /// * `op` - The operation.
    /// * `dst` - The destination operand.
    /// * `type_` - The type of the operation.
    pub fn op_flags(
        &mut self,
        op: i32,
        dst: impl Into<Operand>,
        type_: i32,
    ) -> Result<&mut Self, ErrorCode> {
        let (dst, dstw) = dst.into().into();
        self.compiler.emit_op_flags(op, dst, dstw, type_)?;
        Ok(self)
    }

    /// Emit a conditional select instruction which moves `src1` to `dst_reg`,
    /// if the condition is satisfied, or `src2_reg` to `dst_reg` otherwise.
    ///
    /// # Parameters
    ///
    /// * `type_` - The type of the selection.
    /// * `dst_reg` - The destination register.
    /// * `src1` - The first source operand.
    /// * `src2_reg` - The second source register.
    pub fn select(
        &mut self,
        type_: Condition,
        dst_reg: impl Into<Operand>,
        src1: impl Into<Operand>,
        src2_reg: impl Into<Operand>,
    ) -> Result<&mut Self, ErrorCode> {
        let (dst_reg, _) = dst_reg.into().into();
        let (src1, src1w) = src1.into().into();
        let (src2_reg, _) = src2_reg.into().into();
        self.compiler
            .emit_select(type_ as i32, dst_reg, src1, src1w, src2_reg)?;
        Ok(self)
    }

    /// Emit a conditional floating point select instruction which moves `src1` to `dst_freg`,
    /// if the condition is satisfied, or `src2_freg` to `dst_freg` otherwise.
    ///
    /// # Parameters
    ///
    /// * `type_` - The type of the selection.
    /// * `dst_freg` - The destination float register.
    /// * `src1` - The first source operand.
    /// * `src2_freg` - The second source float register.
    pub fn fselect(
        &mut self,
        type_: Condition,
        dst_freg: impl Into<Operand>,
        src1: impl Into<Operand>,
        src2_freg: impl Into<Operand>,
    ) -> Result<&mut Self, ErrorCode> {
        let (dst_freg, _) = dst_freg.into().into();
        let (src1, src1w) = src1.into().into();
        let (src2_freg, _) = src2_freg.into().into();
        self.compiler
            .emit_fselect(type_ as i32, dst_freg, src1, src1w, src2_freg)?;
        Ok(self)
    }

    /// Emits a conditional branch.
    ///
    /// # Parameters
    ///
    /// * `type_` - The type of the comparison.
    /// * `src1` - The first source operand.
    /// * `src2` - The second source operand.
    /// * `then_branch` - The closure to execute if the condition is true.
    /// * `else_branch` - The closure to execute if the condition is false.
    pub fn branch<T, E>(
        &mut self,
        type_: Condition,
        src1: impl Into<Operand>,
        src2: impl Into<Operand>,
        then_branch: T,
        else_branch: E,
    ) -> Result<&mut Self, ErrorCode>
    where
        T: FnOnce(&mut Emitter) -> Result<(), ErrorCode>,
        E: FnOnce(&mut Emitter) -> Result<(), ErrorCode>,
    {
        let mut jump_to_else = self.cmp(type_.invert(), src1, src2)?;
        then_branch(self)?;
        let mut jump_to_end = self.jump(JumpType::Jump)?;
        let mut else_label = self.put_label()?;
        jump_to_else.set_label(&mut else_label);
        else_branch(self)?;
        let mut end_label = self.put_label()?;
        jump_to_end.set_label(&mut end_label);
        Ok(self)
    }

    /// Emits a `CONST` instruction.
    ///
    /// # Parameters
    ///
    /// * `op` - The operation.
    /// * `dst` - The destination operand.
    /// * `init_value` - The initial value.
    pub fn const_(
        &mut self,
        op: i32,
        dst: impl Into<Operand>,
        init_value: sljit_sw,
    ) -> Result<sys::Constant, ErrorCode> {
        let (dst, dstw) = dst.into().into();
        Ok(self.compiler.emit_const(op, dst, dstw, init_value))
    }

    /// Emits a `LABEL` instruction.
    pub fn put_label(&mut self) -> Result<sys::Label, ErrorCode> {
        Ok(self.compiler.emit_label())
    }

    /// Return to the caller function.
    ///
    /// # Parameters
    ///
    /// * `op` - The operation.
    /// * `src` - The source operand.
    pub fn emit_return(&mut self, op: ReturnOp, src: impl Into<Operand>) -> Result<(), ErrorCode> {
        let (src, srcw) = src.into().into();
        self.compiler.emit_return(op as i32, src, srcw)?;
        Ok(())
    }

    /// Emits a `RETURN` instruction with no return value.
    pub fn return_void(&mut self) -> Result<(), ErrorCode> {
        self.compiler.emit_return_void()?;
        Ok(())
    }

    /// Restores the saved registers and frees the stack area, then the execution
    /// continues from the address specified by the source operand.
    ///
    /// # Parameters
    ///
    /// * `src` - The source operand.
    pub fn return_to(&mut self, src: impl Into<Operand>) -> Result<(), ErrorCode> {
        let (src, srcw) = src.into().into();
        self.compiler.emit_return_to(src, srcw)?;
        Ok(())
    }

    /// Copies the base address of `SP` + `offset` to `dst`.
    ///
    /// # Parameters
    ///
    /// * `dst` - The destination operand.
    /// * `offset` - The offset from the local base.
    pub fn get_local_base(
        &mut self,
        dst: impl Into<Operand>,
        offset: sljit_sw,
    ) -> Result<&mut Self, ErrorCode> {
        let (dst, dstw) = dst.into().into();
        self.compiler.get_local_base(dst, dstw, offset);
        Ok(self)
    }

    define_simd_vreg_srcdst! {
        /// Emits a SIMD move instruction.
        simd_mov, emit_simd_mov;
        /// Emits a SIMD replicate instruction.
        simd_replicate, emit_simd_replicate;
        /// Emits a SIMD extend instruction.
        simd_extend, emit_simd_extend;
        /// Emits a SIMD sign instruction.
        simd_sign, emit_simd_sign;
    }

    define_simd_vreg_idx_srcdst! {
        /// Emits a SIMD lane move instruction.
        simd_lane_mov, emit_simd_lane_mov;
    }

    define_simd_vreg_src_idx! {
        /// Emits a SIMD lane replicate instruction.
        simd_lane_replicate, emit_simd_lane_replicate;
    }

    define_simd_op2_vreg! {
        /// Emits a SIMD 2-operand instruction.
        simd_op2, emit_simd_op2;
    }
}

#[repr(i32)]
#[derive(Clone, Copy, Debug)]
/// A SIMD operation type.
pub enum SimdType {
    /// Load from memory.
    Load = sys::SLJIT_SIMD_LOAD,
    /// Store to memory.
    Store = sys::SLJIT_SIMD_STORE,
}

#[repr(i32)]
#[derive(Clone, Copy, Debug)]
/// A SIMD register type.
pub enum SimdReg {
    /// A 64-bit register.
    Reg64 = sys::SLJIT_SIMD_REG_64,
    /// A 128-bit register.
    Reg128 = sys::SLJIT_SIMD_REG_128,
    /// A 256-bit register.
    Reg256 = sys::SLJIT_SIMD_REG_256,
    /// A 512-bit register.
    Reg512 = sys::SLJIT_SIMD_REG_512,
}

#[repr(i32)]
#[derive(Clone, Copy, Debug)]
/// A SIMD element type.
pub enum SimdElem {
    /// An 8-bit element.
    Elem8 = sys::SLJIT_SIMD_ELEM_8,
    /// A 16-bit element.
    Elem16 = sys::SLJIT_SIMD_ELEM_16,
    /// A 32-bit element.
    Elem32 = sys::SLJIT_SIMD_ELEM_32,
    /// A 64-bit element.
    Elem64 = sys::SLJIT_SIMD_ELEM_64,
    /// A 128-bit element.
    Elem128 = sys::SLJIT_SIMD_ELEM_128,
    /// A 256-bit element.
    Elem256 = sys::SLJIT_SIMD_ELEM_256,
}

#[repr(i32)]
#[derive(Clone, Copy, Debug)]
/// A SIMD memory alignment.
pub enum SimdMemAlign {
    /// Unaligned memory.
    Unaligned = sys::SLJIT_SIMD_MEM_UNALIGNED,
    /// 16-byte aligned memory.
    Aligned16 = sys::SLJIT_SIMD_MEM_ALIGNED_16,
    /// 32-byte aligned memory.
    Aligned32 = sys::SLJIT_SIMD_MEM_ALIGNED_32,
    /// 64-byte aligned memory.
    Aligned64 = sys::SLJIT_SIMD_MEM_ALIGNED_64,
    /// 128-byte aligned memory.
    Aligned128 = sys::SLJIT_SIMD_MEM_ALIGNED_128,
    /// 256-byte aligned memory.
    Aligned256 = sys::SLJIT_SIMD_MEM_ALIGNED_256,
    /// 512-byte aligned memory.
    Aligned512 = sys::SLJIT_SIMD_MEM_ALIGNED_512,
}

#[repr(i32)]
#[derive(Clone, Copy, Debug)]
/// A SIMD extend operation.
pub enum SimdExtend {
    /// Signed extension.
    Signed = sys::SLJIT_SIMD_EXTEND_SIGNED,
    /// Extend to 16 bits.
    Extend16 = sys::SLJIT_SIMD_EXTEND_16,
    /// Extend to 32 bits.
    Extend32 = sys::SLJIT_SIMD_EXTEND_32,
    /// Extend to 64 bits.
    Extend64 = sys::SLJIT_SIMD_EXTEND_64,
}

#[repr(i32)]
#[derive(Clone, Copy, Debug)]
/// A SIMD 2-operand operation.
pub enum SimdOp2 {
    /// Bitwise AND.
    And = sys::SLJIT_SIMD_OP2_AND,
    /// Bitwise OR.
    Or = sys::SLJIT_SIMD_OP2_OR,
    /// Bitwise XOR.
    Xor = sys::SLJIT_SIMD_OP2_XOR,
    /// Shuffle.
    Shuffle = sys::SLJIT_SIMD_OP2_SHUFFLE,
}

#[cfg(test)]
mod tests;
