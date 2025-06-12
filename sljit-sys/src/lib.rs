#![cfg_attr(not(test), no_std)]
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use core::{ffi::CStr, ptr::null_mut, str::Utf8Error};

use derive_more::{Display, Error, From, TryFrom};

pub use pastey::paste as paste_priv;

include!("./wrapper.rs");

#[derive(TryFrom, From, Error, Display, Clone, Copy, Debug, PartialEq)]
#[try_from(repr)]
#[repr(i32)]
pub enum ErrorCode {
    Success = SLJIT_SUCCESS,
    Compiled = SLJIT_ERR_COMPILED,
    AllocationFailed = SLJIT_ERR_ALLOC_FAILED,
    ExecutableMemoryAllocationFailed = SLJIT_ERR_EX_ALLOC_FAILED,
    Unsupported = SLJIT_ERR_UNSUPPORTED,
    BadArgument = SLJIT_ERR_BAD_ARGUMENT,
}

impl ErrorCode {
    #[inline(always)]
    pub fn i32_as_result(this: i32) -> Result<(), ErrorCode> {
        ErrorCode::try_from(this).map_or(Err(ErrorCode::Unsupported), Into::into)
    }
}

impl Into<Result<(), ErrorCode>> for ErrorCode {
    #[inline(always)]
    fn into(self) -> Result<(), ErrorCode> {
        if self == ErrorCode::Success {
            Ok(())
        } else {
            Err(self)
        }
    }
}

#[inline(always)]
pub fn has_cpu_feature(feature_type: sljit_s32) -> sljit_s32 {
    unsafe { sljit_has_cpu_feature(feature_type) }
}

#[inline(always)]
pub fn cmp_info(type_: sljit_s32) -> sljit_s32 {
    unsafe { sljit_cmp_info(type_) }
}

#[inline(always)]
pub fn set_jump_addr(addr: sljit_uw, new_target: sljit_uw, executable_offset: sljit_sw) {
    unsafe { sljit_set_jump_addr(addr, new_target, executable_offset) }
}

#[inline(always)]
pub fn set_const(
    addr: sljit_uw,
    op: sljit_s32,
    new_constant: sljit_sw,
    executable_offset: sljit_sw,
) {
    unsafe { sljit_set_const(addr, op, new_constant, executable_offset) }
}

#[inline(always)]
pub fn get_register_index(type_: sljit_s32, reg: sljit_s32) -> sljit_s32 {
    unsafe { sljit_get_register_index(type_, reg) }
}

#[inline(always)]
pub fn get_platform_name() -> Result<&'static str, Utf8Error> {
    unsafe { CStr::from_ptr(sljit_get_platform_name()).to_str() }
}

/// Encodes function argument types and return type into a bit-packed integer for SLJIT.
///
/// Each type occupies 4 bits in the resulting value:
/// - Bits 0-3: Return type
/// - Bits 4-7: First argument type
/// - Bits 8-11: Second argument type
/// - Bits 12-15: Third argument type
/// - Bits 16-19: Fourth argument type
///
/// # Examples
/// ```
/// # use sljit_sys::*;
/// // Function with three arguments returning a word
/// let sig = arg_types!([P, W, W] -> W);
///
/// // Function with no return value (void)
/// let sig = arg_types!([W, W]);
///
/// // Function with no arguments
/// let sig = arg_types!([]);
/// ```
///
/// # Limitations
/// - Maximum of 4 arguments supported
/// - Each argument type must fit in 4 bits
#[macro_export]
macro_rules! arg_types {
    // Public interface: arguments with explicit return type
    ([$($args:tt),*] -> $ret:tt) => {
        $crate::arg_types!(@internal $ret; $($args),*)
    };

    // Public interface: arguments with default void return
    ([$($args:tt),*]) => {
        $crate::arg_types!(@internal RET_VOID; $($args),*)
    };

    // Alternative syntax: return type first
    ($ret:tt -> [$($args:tt),*]) => {
        $crate::arg_types!(@internal $ret; $($args),*)
    };

    // Internal implementation: iterative encoding with compile-time bounds checking
    (@internal $ret:tt; $($args:tt),*) => {
        {

        // Compile-time assertion for maximum argument count
        const _: () = {
            if const { $crate::arg_types!(@count; $($args),*) } > 4 {
                panic!("arg_types! macro supports maximum 4 arguments");
            }
        };

        $crate::arg_types!(@encode $ret; 0; 1; $($args),*)
        }
    };

    // Encode implementation: iterative bit packing
    (@encode $ret:tt; $acc:expr; $shift:expr;) => {
        // Base case: return type in bits 0-3, accumulated arguments in higher bits
        ($crate::paste_priv!{ [<SLJIT_ARG_TYPE_ $ret>] }) | ($acc)
    };

    (@encode $ret:tt; $acc:expr; $shift:expr; $arg:tt $(, $rest:tt)*) => {
        $crate::arg_types!(@encode
            $ret;
            $acc | (($crate::paste_priv!{ [<SLJIT_ARG_TYPE_ $arg>] }) << ($shift * 4));
            $shift + 1;
            $($rest),*
        )
    };

    // Helper: count arguments at compile time
    (@count;) => { 0 };
    (@count; $head:tt $(, $tail:tt)*) => { 1 + $crate::arg_types!(@count; $($tail),*) };
}

#[repr(transparent)]
#[derive(From, Clone, Copy)]
pub struct Constant(*mut sljit_const);

impl Constant {
    #[inline(always)]
    pub fn addr(&self) -> sljit_uw {
        unsafe { sljit_get_const_addr(self.0) }
    }
}

#[repr(transparent)]
#[derive(From, Clone, Copy)]
pub struct Label(*mut sljit_label);

impl Label {
    #[inline(always)]
    pub fn addr(&self) -> sljit_uw {
        unsafe { sljit_get_label_addr(self.0) }
    }

    #[inline(always)]
    pub fn index(&self) -> sljit_uw {
        unsafe { sljit_get_label_index(self.0) }
    }

    #[inline(always)]
    pub fn abs_addr(&self) -> sljit_uw {
        unsafe { sljit_get_label_abs_addr(self.0) }
    }

    #[inline(always)]
    pub fn set_to(&mut self, jump: &mut Jump) -> &mut Self {
        unsafe {
            sljit_set_label(jump.0, self.0);
        }
        self
    }
}

#[repr(transparent)]
#[derive(From, Clone, Copy)]
pub struct Jump(*mut sljit_jump);

impl Jump {
    #[inline(always)]
    pub fn set_label(&mut self, label: &mut Label) -> &mut Self {
        unsafe {
            sljit_set_label(self.0, label.0);
        }
        self
    }

    #[inline(always)]
    pub fn set_target(&mut self, target: sljit_uw) -> &mut Self {
        unsafe {
            sljit_set_target(self.0, target);
        }
        self
    }

    #[inline(always)]
    pub fn addr(&self) -> sljit_uw {
        unsafe { sljit_get_jump_addr(self.0) }
    }

    #[inline(always)]
    pub fn target(&self) -> sljit_uw {
        unsafe { sljit_jump_get_target(self.0) }
    }

    #[inline(always)]
    pub fn label(&self) -> Label {
        (unsafe { sljit_jump_get_label(self.0) }).into()
    }

    #[inline(always)]
    pub fn has_label(&self) -> bool {
        unsafe { sljit_jump_has_label(self.0) != 0 }
    }

    #[inline(always)]
    pub fn has_target(&self) -> bool {
        unsafe { sljit_jump_has_target(self.0) != 0 }
    }
}

#[repr(transparent)]
#[derive(From)]
pub struct GeneratedCode(*mut ::core::ffi::c_void);

impl GeneratedCode {
    #[inline(always)]
    pub fn get(&self) -> *const ::core::ffi::c_void {
        self.0
    }
}

impl Drop for GeneratedCode {
    #[inline(always)]
    fn drop(&mut self) {
        unsafe {
            sljit_free_code(self.0, null_mut());
        }
    }
}

#[repr(transparent)]
#[derive(From)]
pub struct Compiler(*mut sljit_compiler);

impl Default for Compiler {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl Compiler {
    #[inline(always)]
    pub fn new() -> Self {
        Self(unsafe { sljit_create_compiler(null_mut()) })
    }
}

impl Compiler {
    #[inline(always)]
    pub fn generate_code(self) -> GeneratedCode {
        let code = unsafe { sljit_generate_code(self.0, 0, null_mut()) };
        drop(self);
        GeneratedCode(code)
    }
}

impl Drop for Compiler {
    #[inline(always)]
    fn drop(&mut self) {
        unsafe {
            sljit_free_compiler(self.0);
        }
    }
}

include!("./generated.mid.rs");

#[cfg(test)]
mod integration_tests {
    use tap::prelude::*;

    use core::{ffi::c_int, mem::transmute};
    use std::error::Error;

    use super::*;

    #[test]
    fn test_add3() -> Result<(), Box<dyn Error>> {
        unsafe {
            let mut compiler = Compiler::new();
            compiler
                .emit_enter(0, arg_types!(W -> [W, W, W]), 1, 3, 0)?
                .pipe_ref_mut(|compiler| {
                    compiler
                        .emit_op1(SLJIT_MOV, SLJIT_R0, 0, SLJIT_S0, 0)?
                        /* R0 = R0 + second */
                        .emit_op2(SLJIT_ADD, SLJIT_R0, 0, SLJIT_R0, 0, SLJIT_S1, 0)?
                        /* R0 = R0 + third */
                        .emit_op2(SLJIT_ADD, SLJIT_R0, 0, SLJIT_R0, 0, SLJIT_S2, 0)?
                        /* This statement mov R0 to RETURN REG and return */
                        /* in fact, R0 is RETURN REG itself */
                        .emit_return(SLJIT_MOV, SLJIT_R0, 0)?;
                    Ok::<_, ErrorCode>(())
                })?;

            let code = compiler.generate_code();
            let func: fn(c_int, c_int, c_int) -> c_int = transmute(code.get());
            assert_eq!(func(4, 5, 6), 4 + 5 + 6);
        }

        Ok(())
    }

    #[test]
    fn test_arg_types() {
        assert_eq!(
            arg_types!([P, W, W] -> W),
            SLJIT_ARG_TYPE_W
                | (SLJIT_ARG_TYPE_P << 4)
                | (SLJIT_ARG_TYPE_W << (2 * 4))
                | (SLJIT_ARG_TYPE_W << (3 * 4))
        );

        assert_eq!(
            arg_types!(RET_VOID -> [P, W, W]),
            SLJIT_ARG_TYPE_RET_VOID
                | (SLJIT_ARG_TYPE_P << 4)
                | (SLJIT_ARG_TYPE_W << (2 * 4))
                | (SLJIT_ARG_TYPE_W << (3 * 4))
        );

        assert_eq!(
            arg_types!([P, W, W]),
            SLJIT_ARG_TYPE_RET_VOID
                | (SLJIT_ARG_TYPE_P << 4)
                | (SLJIT_ARG_TYPE_W << (2 * 4))
                | (SLJIT_ARG_TYPE_W << (3 * 4))
        );

        assert_eq!(arg_types!([]), SLJIT_ARG_TYPE_RET_VOID);
    }

    #[test]
    fn test_array_access() -> Result<(), Box<dyn Error>> {
        extern "C" fn print_num(a: isize) -> isize {
            println!("num = {a}");
            1
        }

        unsafe {
            let arr: &[isize] = &[3, -10, 4, 6, 8, 12, 2000, 0];
            let mut compiler = Compiler::new();

            compiler.pipe_ref_mut(|compiler| {
                compiler
                    /* S2 = 0 */
                    .emit_op2(SLJIT_XOR, SLJIT_S2, 0, SLJIT_S2, 0, SLJIT_S2, 0)?
                    /* S1 = narr */
                    .emit_op1(
                        SLJIT_MOV,
                        SLJIT_S1,
                        0,
                        SLJIT_IMM,
                        arr.len().try_into().unwrap(),
                    )?;
                /* loopstart:              */
                compiler.emit_label().pipe_ref_mut(|loop_start| {
                    /* S2 >= narr --> jump out */
                    compiler
                        .emit_cmp(SLJIT_GREATER_EQUAL, SLJIT_S2, 0, SLJIT_S1, 0)
                        .pipe(|out| {
                            compiler
                                /* R0 = (long *)S0[S2];    */
                                .emit_op1(
                                    SLJIT_MOV,
                                    SLJIT_R0,
                                    0,
                                    SLJIT_MEM | SLJIT_S0 | (SLJIT_S2 << 8),
                                    SLJIT_WORD_SHIFT.into(),
                                )?
                                /* print_num(R0)           */
                                .emit_icall(
                                    SLJIT_CALL,
                                    arg_types!([W] -> W),
                                    SLJIT_IMM,
                                    print_num as _,
                                )?
                                /* S2 += 1                 */
                                .emit_op2(SLJIT_ADD, SLJIT_S2, 0, SLJIT_S2, 0, SLJIT_IMM, 1)?
                                /* jump loopstart          */
                                .emit_jump(SLJIT_JUMP)
                                .set_label(loop_start);
                            Ok::<_, ErrorCode>(out)
                        })?
                        .pipe_ref_mut(|out| {
                            /* out:                    */
                            compiler.emit_label().set_to(out);
                            /* return S1               */
                            compiler.emit_return(SLJIT_MOV, SLJIT_S1, 0)?;
                            Ok::<_, ErrorCode>(())
                        })?;
                    Ok::<_, ErrorCode>(())
                })?;
                Ok::<_, ErrorCode>(())
            })?;
            let code = compiler.generate_code();

            let func: fn(*const isize, isize, isize) -> isize = transmute(code.get());

            assert_eq!(
                func(arr.as_ptr(), arr.len().try_into().unwrap(), 0),
                arr.len().try_into().unwrap()
            );

            Ok(())
        }
    }
}
