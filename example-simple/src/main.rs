use std::{ffi::c_int, mem::transmute};

use sljit_sys::*;

fn main() {
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
    let func: fn(c_int, c_int, c_int) -> c_int = unsafe { transmute(code.get()) };
    println!("{}", func(4, 5, 6));
}
