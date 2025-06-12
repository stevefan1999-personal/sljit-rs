use std::{error::Error, ffi::c_int, mem::transmute};

use sljit_sys::*;

fn main() -> Result<(), Box<dyn Error>> {
    let mut compiler = Compiler::new();
    compiler
        .emit_enter(0, arg_types!(W -> [W, W, W]), 1, 3, 0)?
        .emit_op1(SLJIT_MOV, SLJIT_R0, 0, SLJIT_S0, 0)?
        /* R0 = R0 + second */
        .emit_op2(SLJIT_ADD, SLJIT_R0, 0, SLJIT_R0, 0, SLJIT_S1, 0)?
        /* R0 = R0 + third */
        .emit_op2(SLJIT_ADD, SLJIT_R0, 0, SLJIT_R0, 0, SLJIT_S2, 0)?
        /* This statement mov R0 to RETURN REG and return */
        /* in fact, R0 is RETURN REG itself */
        .emit_return(SLJIT_MOV, SLJIT_R0, 0)?;

    let code = compiler.generate_code();
    let func: fn(c_int, c_int, c_int) -> c_int = unsafe { transmute(code.get()) };
    println!("{}", func(4, 5, 6));
    Ok(())
}
