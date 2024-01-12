#![cfg_attr(not(test), no_std)]
#![feature(trivial_bounds)]
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use core::{ffi::CStr, ptr::null_mut, str::Utf8Error};

use const_default::ConstDefault;
use derive_more::From;

include!(concat!(env!("OUT_DIR"), "/wrapper.rs"));

pub fn has_cpu_feature(feature_type: sljit_s32) -> sljit_s32 {
    unsafe { sljit_has_cpu_feature(feature_type) }
}

pub fn cmp_info(type_: sljit_s32) -> sljit_s32 {
    unsafe { sljit_cmp_info(type_) }
}

pub fn set_jump_addr(addr: sljit_uw, new_target: sljit_uw, executable_offset: sljit_sw) {
    unsafe { sljit_set_jump_addr(addr, new_target, executable_offset) }
}

pub fn set_const(addr: sljit_uw, new_constant: sljit_sw, executable_offset: sljit_sw) {
    unsafe { sljit_set_const(addr, new_constant, executable_offset) }
}

pub fn get_register_index(type_: sljit_s32, reg: sljit_s32) -> sljit_s32 {
    unsafe { sljit_get_register_index(type_, reg) }
}

pub fn get_platform_name() -> Result<&'static str, Utf8Error> {
    unsafe { CStr::from_ptr(sljit_get_platform_name()).to_str() }
}

#[repr(transparent)]
#[derive(From)]
pub struct Constant {
    inner: *mut sljit_const,
}

impl Constant {
    pub fn addr(&self) -> sljit_uw {
        unsafe { sljit_get_const_addr(self.inner) }
    }
}

#[repr(transparent)]
#[derive(From)]
pub struct Label {
    inner: *mut sljit_label,
}

impl Label {
    pub fn addr(&self) -> sljit_uw {
        unsafe { sljit_get_label_addr(self.inner) }
    }
}

#[repr(transparent)]
#[derive(From)]
pub struct PutLabel {
    inner: *mut sljit_put_label,
}

impl PutLabel {
    pub fn set_label(&mut self, label: &Label) {
        unsafe { sljit_set_put_label(self.inner, label.inner) }
    }
}

#[repr(transparent)]
#[derive(From)]
pub struct Jump {
    inner: *mut sljit_jump,
}

impl Jump {
    pub fn set_label(&mut self, label: &Label) {
        unsafe { sljit_set_label(self.inner, label.inner) }
    }

    pub fn set_target(&mut self, target: sljit_uw) {
        unsafe { sljit_set_target(self.inner, target) }
    }

    pub fn addr(&self) -> sljit_uw {
        unsafe { sljit_get_jump_addr(self.inner) }
    }
}

#[repr(transparent)]
pub struct GeneratedCode {
    code: *mut ::core::ffi::c_void,
}

impl GeneratedCode {
    pub fn get(&self) -> *const ::core::ffi::c_void {
        self.code
    }
}

impl Drop for GeneratedCode {
    fn drop(&mut self) {
        unsafe {
            sljit_free_code(self.code, null_mut());
        }
    }
}

impl Drop for Compiler {
    fn drop(&mut self) {
        unsafe {
            sljit_free_compiler(self.inner);
        }
    }
}

#[repr(transparent)]
pub struct Compiler {
    inner: *mut sljit_compiler,
}

impl Compiler {
    pub fn new() -> Self {
        Self {
            inner: unsafe { sljit_create_compiler(null_mut(), null_mut()) },
        }
    }
}

impl Compiler {
    pub fn generate_code(self) -> GeneratedCode {
        let code = unsafe { sljit_generate_code(self.inner) };
        drop(self);
        GeneratedCode { code }
    }
}

include!(concat!(env!("OUT_DIR"), "/generated.mid.rs"));

#[cfg(test)]
mod integration_tests {
    use core::{ffi::c_int, mem::transmute};

    use super::*;

    #[test]
    fn test_add3() {
        unsafe {
            let mut compiler = Compiler::new();
            compiler.emit_enter(
                0,
                SLJIT_ARG_TYPE_W
                    | (SLJIT_ARG_TYPE_W << (1 * 4))
                    | (SLJIT_ARG_TYPE_W << (2 * 4))
                    | (SLJIT_ARG_TYPE_W << (3 * 4)),
                1,
                3,
                0,
                0,
                0,
            );
            compiler.emit_op1(SLJIT_MOV, SLJIT_R0, 0, SLJIT_S0, 0);

            /* R0 = R0 + second */
            compiler.emit_op2(SLJIT_ADD, SLJIT_R0, 0, SLJIT_R0, 0, SLJIT_S1, 0);

            /* R0 = R0 + third */
            compiler.emit_op2(SLJIT_ADD, SLJIT_R0, 0, SLJIT_R0, 0, SLJIT_S2, 0);

            /* This statement mov R0 to RETURN REG and return */
            /* in fact, R0 is RETURN REG itself */
            compiler.emit_return(SLJIT_MOV, SLJIT_R0, 0);

            let code = compiler.generate_code();
            let func: fn(c_int, c_int, c_int) -> c_int = transmute(code.get());
            assert_eq!(func(4, 5, 6), 4 + 5 + 6);
        }
    }

    #[test]
    fn test_array_access() {
        extern "C" fn print_num(a: isize) {
            println!("num = {a}");
        }

        unsafe {
            let arr: &[isize] = &[3, -10, 4, 6, 8, 12, 2000, 0];
            let mut compiler = Compiler::new();
            compiler.emit_enter(
                0,
                SLJIT_ARG_TYPE_W | (SLJIT_ARG_TYPE_P << (1 * 4)) | (SLJIT_ARG_TYPE_W << (2 * 4)) | (SLJIT_ARG_TYPE_W << (3 * 4)),
                4,
                3,
                0,
                0,
                0,
            );

            /* S2 = 0 */
            compiler.emit_op2(SLJIT_XOR, SLJIT_S2, 0, SLJIT_S2, 0, SLJIT_S2, 0);

            /* S1 = narr */
            compiler.emit_op1(
                SLJIT_MOV,
                SLJIT_S1,
                0,
                SLJIT_IMM,
                arr.len().try_into().unwrap(),
            );

            /* loopstart:              */
            let loop_start = compiler.emit_label();

            /* S2 >= narr --> jumo out */
            let mut out = compiler.emit_cmp(SLJIT_GREATER_EQUAL, SLJIT_S2, 0, SLJIT_S1, 0);

            /* R0 = (long *)S0[S2];    */
            compiler.emit_op1(
                SLJIT_MOV,
                SLJIT_R0,
                0,
                SLJIT_MEM | (SLJIT_S0) | ((SLJIT_S2) << 8),
                SLJIT_WORD_SHIFT.into(),
            );

            /* print_num(R0)           */
            compiler.emit_icall(
                SLJIT_CALL,
                SLJIT_ARG_TYPE_RET_VOID | (SLJIT_ARG_TYPE_W << (1 * 4)),
                SLJIT_IMM,
                print_num as _,
            );
            /* S2 += 1                 */
            compiler.emit_op2(SLJIT_ADD, SLJIT_S2, 0, SLJIT_S2, 0, SLJIT_IMM, 1);
            /* jump loopstart          */
            compiler.emit_jump(SLJIT_JUMP).set_label(&loop_start);
            /* out:                    */
            out.set_label(&compiler.emit_label());
            /* return S1               */
            compiler.emit_return(SLJIT_MOV, SLJIT_S1, 0);

            let code = compiler.generate_code();
            let func: fn(*const isize, isize, isize) -> isize = transmute(code.get());

            assert_eq!(
                func(arr.as_ptr(), arr.len().try_into().unwrap(), 0),
                arr.len().try_into().unwrap()
            );
        }
    }
}
