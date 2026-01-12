use derive_more::Display;
use thiserror::Error;

/// Runtime trap - unrecoverable error during execution
#[derive(Clone, Debug, Error)]
#[error("trap: {message}")]
pub struct Trap {
    /// The kind of trap
    pub kind: TrapKind,
    /// Human-readable message
    pub message: String,
}

impl Trap {
    /// Create a new trap
    #[inline]
    pub fn new(kind: TrapKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    /// Create an unreachable trap
    #[inline]
    pub fn unreachable() -> Self {
        Self::new(TrapKind::Unreachable, "unreachable executed")
    }

    /// Create a memory out of bounds trap
    #[inline]
    pub fn memory_out_of_bounds() -> Self {
        Self::new(TrapKind::MemoryOutOfBounds, "out of bounds memory access")
    }

    /// Create a table out of bounds trap
    #[inline]
    pub fn table_out_of_bounds() -> Self {
        Self::new(TrapKind::TableOutOfBounds, "out of bounds table access")
    }

    /// Create an indirect call type mismatch trap
    #[inline]
    pub fn indirect_call_type_mismatch() -> Self {
        Self::new(
            TrapKind::IndirectCallTypeMismatch,
            "indirect call type mismatch",
        )
    }

    /// Create a null function reference trap
    #[inline]
    pub fn null_func_ref() -> Self {
        Self::new(TrapKind::NullFuncRef, "null function reference")
    }

    /// Create an integer overflow trap
    #[inline]
    pub fn integer_overflow() -> Self {
        Self::new(TrapKind::IntegerOverflow, "integer overflow")
    }

    /// Create an integer division by zero trap
    #[inline]
    pub fn integer_divide_by_zero() -> Self {
        Self::new(TrapKind::IntegerDivisionByZero, "integer division by zero")
    }

    /// Create an invalid conversion to integer trap
    #[inline]
    pub fn invalid_conversion_to_int() -> Self {
        Self::new(
            TrapKind::InvalidConversionToInteger,
            "invalid conversion to integer",
        )
    }

    /// Create a stack overflow trap
    #[inline]
    pub fn stack_overflow() -> Self {
        Self::new(TrapKind::StackOverflow, "stack overflow")
    }

    /// Create an out of fuel trap
    #[inline]
    pub fn out_of_fuel() -> Self {
        Self::new(TrapKind::OutOfFuel, "out of fuel")
    }
}

/// The kind of runtime trap
#[derive(Clone, Copy, Debug, Display, PartialEq, Eq)]
pub enum TrapKind {
    /// Out of bounds memory access
    #[display("memory out of bounds")]
    MemoryOutOfBounds,
    /// Out of bounds table access
    #[display("table out of bounds")]
    TableOutOfBounds,
    /// Indirect call type mismatch
    #[display("indirect call type mismatch")]
    IndirectCallTypeMismatch,
    /// Null function reference
    #[display("null function reference")]
    NullFuncRef,
    /// Integer overflow
    #[display("integer overflow")]
    IntegerOverflow,
    /// Integer division by zero
    #[display("integer division by zero")]
    IntegerDivisionByZero,
    /// Invalid conversion to integer
    #[display("invalid conversion to integer")]
    InvalidConversionToInteger,
    /// Stack overflow
    #[display("stack overflow")]
    StackOverflow,
    /// Unreachable code executed
    #[display("unreachable")]
    Unreachable,
    /// Out of fuel
    #[display("out of fuel")]
    OutOfFuel,
    /// Custom trap from host function
    #[display("host error")]
    Host,
}
