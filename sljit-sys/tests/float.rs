use sljit_sys::{self as sys, *};
use sys::mem;

#[test]
fn test_float1() {
    let mut compiler = Compiler::new();
    let mut buf: [f64; 7] = [0.0; 7];
    let mut buf2: [sljit_sw; 6] = [0; 6];

    assert!(!compiler.is_null());

    buf[0] = 7.75;
    buf[1] = -4.5;
    buf[2] = 0.0;
    buf[3] = 0.0;
    buf[4] = 0.0;
    buf[5] = 0.0;
    buf[6] = 0.0;

    buf2[0] = 10;
    buf2[1] = 10;
    buf2[2] = 10;
    buf2[3] = 10;
    buf2[4] = 10;
    buf2[5] = 10;

    compiler
        .emit_enter(0, arg_types!([P, P]), (3 | (6 << 8)) as i32, 2, 0)
        .unwrap();
    // buf[2]
    compiler
        .emit_fop1(
            sys::SLJIT_MOV_F64,
            mem!(0) as i32,
            &mut buf[2] as *mut _ as sljit_sw,
            mem!(0) as i32,
            &mut buf[1] as *mut _ as sljit_sw,
        )
        .unwrap();
    // buf[3]
    compiler
        .emit_fop1(
            sys::SLJIT_ABS_F64,
            mem!(sys::SLJIT_S0) as i32,
            3 * std::mem::size_of::<f64>() as sljit_sw,
            mem!(sys::SLJIT_S0) as i32,
            std::mem::size_of::<f64>() as sljit_sw,
        )
        .unwrap();
    // buf[4]
    compiler
        .emit_fop1(
            sys::SLJIT_MOV_F64,
            sys::SLJIT_FR0,
            0,
            mem!(0) as i32,
            &mut buf[0] as *mut _ as sljit_sw,
        )
        .unwrap();
    compiler
        .emit_op1(
            sys::SLJIT_MOV,
            sys::SLJIT_R0,
            0,
            sys::SLJIT_IMM,
            2 * std::mem::size_of::<f64>() as sljit_sw,
        )
        .unwrap();
    compiler
        .emit_fop1(
            sys::SLJIT_MOV_F64,
            sys::SLJIT_FR1,
            0,
            mem!(sys::SLJIT_S0, sys::SLJIT_R0) as i32,
            0,
        )
        .unwrap();
    compiler
        .emit_fop1(sys::SLJIT_NEG_F64, sys::SLJIT_FR2, 0, sys::SLJIT_FR0, 0)
        .unwrap();
    compiler
        .emit_fop1(sys::SLJIT_MOV_F64, sys::SLJIT_FR3, 0, sys::SLJIT_FR2, 0)
        .unwrap();
    compiler
        .emit_fop1(
            sys::SLJIT_MOV_F64,
            mem!(0) as i32,
            &mut buf[4] as *mut _ as sljit_sw,
            sys::SLJIT_FR3,
            0,
        )
        .unwrap();
    // buf[5]
    compiler
        .emit_fop1(sys::SLJIT_ABS_F64, sys::SLJIT_FR4, 0, sys::SLJIT_FR1, 0)
        .unwrap();
    compiler
        .emit_fop1(
            sys::SLJIT_MOV_F64,
            mem!(sys::SLJIT_S0) as i32,
            5 * std::mem::size_of::<f64>() as sljit_sw,
            sys::SLJIT_FR4,
            0,
        )
        .unwrap();
    // buf[6]
    compiler
        .emit_fop1(
            sys::SLJIT_NEG_F64,
            mem!(sys::SLJIT_S0) as i32,
            6 * std::mem::size_of::<f64>() as sljit_sw,
            sys::SLJIT_FR4,
            0,
        )
        .unwrap();

    // buf2[0]
    compiler
        .emit_fop1(
            sys::SLJIT_MOV_F64,
            sys::SLJIT_FR5,
            0,
            mem!(sys::SLJIT_S0) as i32,
            0,
        )
        .unwrap();
    compiler
        .emit_fop1(
            SLJIT_CMP_F64 | SLJIT_SET_F_GREATER,
            sys::SLJIT_FR5,
            0,
            mem!(sys::SLJIT_S0) as i32,
            std::mem::size_of::<f64>() as sljit_sw,
        )
        .unwrap();
    compiler
        .emit_op_flags(
            sys::SLJIT_MOV,
            mem!(sys::SLJIT_S1) as i32,
            0,
            sys::SLJIT_F_GREATER,
        )
        .unwrap();
    // buf2[1]
    compiler
        .emit_fop1(
            SLJIT_CMP_F64 | SLJIT_SET_F_GREATER,
            mem!(sys::SLJIT_S0) as i32,
            std::mem::size_of::<f64>() as sljit_sw,
            sys::SLJIT_FR5,
            0,
        )
        .unwrap();
    compiler
        .emit_op_flags(
            sys::SLJIT_MOV,
            mem!(sys::SLJIT_S1) as i32,
            std::mem::size_of::<sljit_sw>() as sljit_sw,
            sys::SLJIT_F_GREATER,
        )
        .unwrap();
    // buf2[2]
    compiler
        .emit_fop1(sys::SLJIT_MOV_F64, sys::SLJIT_FR1, 0, sys::SLJIT_FR5, 0)
        .unwrap();
    compiler
        .emit_fop1(
            SLJIT_CMP_F64 | SLJIT_SET_F_EQUAL,
            sys::SLJIT_FR1,
            0,
            sys::SLJIT_FR1,
            0,
        )
        .unwrap();
    compiler
        .emit_op_flags(
            sys::SLJIT_MOV,
            mem!(sys::SLJIT_S1) as i32,
            2 * std::mem::size_of::<sljit_sw>() as sljit_sw,
            sys::SLJIT_F_EQUAL,
        )
        .unwrap();
    // buf2[3]
    compiler
        .emit_fop1(
            SLJIT_CMP_F64 | SLJIT_SET_F_LESS,
            sys::SLJIT_FR1,
            0,
            sys::SLJIT_FR1,
            0,
        )
        .unwrap();
    compiler
        .emit_op_flags(
            sys::SLJIT_MOV,
            mem!(sys::SLJIT_S1) as i32,
            3 * std::mem::size_of::<sljit_sw>() as sljit_sw,
            sys::SLJIT_F_LESS,
        )
        .unwrap();
    // buf2[4]
    compiler
        .emit_fop1(
            SLJIT_CMP_F64 | SLJIT_SET_F_EQUAL,
            sys::SLJIT_FR1,
            0,
            mem!(sys::SLJIT_S0) as i32,
            std::mem::size_of::<f64>() as sljit_sw,
        )
        .unwrap();
    compiler
        .emit_op_flags(
            sys::SLJIT_MOV,
            mem!(sys::SLJIT_S1) as i32,
            4 * std::mem::size_of::<sljit_sw>() as sljit_sw,
            sys::SLJIT_F_EQUAL,
        )
        .unwrap();
    // buf2[5]
    compiler
        .emit_fop1(
            SLJIT_CMP_F64 | SLJIT_SET_F_NOT_EQUAL,
            sys::SLJIT_FR1,
            0,
            mem!(sys::SLJIT_S0) as i32,
            std::mem::size_of::<f64>() as sljit_sw,
        )
        .unwrap();
    compiler
        .emit_op_flags(
            sys::SLJIT_MOV,
            mem!(sys::SLJIT_S1) as i32,
            5 * std::mem::size_of::<sljit_sw>() as sljit_sw,
            sys::SLJIT_F_NOT_EQUAL,
        )
        .unwrap();

    compiler.emit_return_void().unwrap();

    let code = compiler.generate_code();
    assert!(!code.get().is_null());
    let func: extern "C" fn(*mut f64, *mut sljit_sw) = unsafe { std::mem::transmute(code.get()) };
    func(buf.as_mut_ptr(), buf2.as_mut_ptr());
    // code.free(); // Removed as Drop trait handles it

    assert_eq!(buf[2], -4.5);
    assert_eq!(buf[3], 4.5);
    assert_eq!(buf[4], -7.75);
    assert_eq!(buf[5], 4.5);
    assert_eq!(buf[6], -4.5);

    assert_eq!(buf2[0], 1);
    assert_eq!(buf2[1], 0);
    assert_eq!(buf2[2], 1);
    assert_eq!(buf2[3], 0);
    assert_eq!(buf2[4], 0);
    assert_eq!(buf2[5], 1);
}
