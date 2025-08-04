use sljit_sys::SLJIT_WORD_SHIFT;

use super::*;
use crate::sys::{
    SLJIT_ARG_TYPE_32, SLJIT_ARG_TYPE_F64, SLJIT_ARG_TYPE_P, SLJIT_ARG_TYPE_W, arg_types,
};
use core::ffi::c_int;
use core::mem::transmute;
use std::error::Error;

#[test]
fn test_add3_emitter() -> Result<(), Box<dyn Error>> {
    unsafe {
        let mut compiler = Compiler::new();
        let mut emitter = Emitter::new(&mut compiler);
        emitter
            .emit_enter(0, arg_types!(W -> [W, W, W]), regs!(1), regs!(3), 0)?
            .mov(0, ScratchRegister::R0, SavedRegister::S0)?
            .add(
                0,
                ScratchRegister::R0,
                ScratchRegister::R0,
                SavedRegister::S1,
            )?
            .add(
                0,
                ScratchRegister::R0,
                ScratchRegister::R0,
                SavedRegister::S2,
            )?;

        emitter.emit_return(ReturnOp::Mov, ScratchRegister::R0)?;

        let code = compiler.generate_code();
        let func: fn(c_int, c_int, c_int) -> c_int = transmute(code.get());
        assert_eq!(func(4, 5, 6), 4 + 5 + 6);
    }
    Ok(())
}

#[test]
fn test_memory_access_emitter() -> Result<(), Box<dyn Error>> {
    unsafe {
        let mut compiler = Compiler::new();
        let mut emitter = Emitter::new(&mut compiler);
        emitter
            .emit_enter(0, arg_types!(W -> [P]), regs!(2), regs!(1), 0)?
            // R0 = S0[0]
            .mov(0, ScratchRegister::R0, mem_offset(SavedRegister::S0, 0))?
            // R1 = S0[1]
            .mov(
                0,
                ScratchRegister::R1,
                mem_offset(SavedRegister::S0, (1 * (1 << SLJIT_WORD_SHIFT)) as i32),
            )?
            // R0 = R0 + R1
            .add(
                0,
                ScratchRegister::R0,
                ScratchRegister::R0,
                ScratchRegister::R1,
            )?;

        emitter.emit_return(ReturnOp::Mov, ScratchRegister::R0)?;

        let code = compiler.generate_code();
        let data: [isize; 2] = [10, 20];
        let func: fn(p: *const isize) -> isize = transmute(code.get());
        assert_eq!(func(data.as_ptr()), 30);
    }
    Ok(())
}

#[test]
fn test_add32_emitter() -> Result<(), Box<dyn Error>> {
    unsafe {
        let mut compiler = Compiler::new();
        let mut emitter = Emitter::new(&mut compiler);
        emitter
            .emit_enter(0, arg_types!(32 -> [32, 32]), regs!(1), regs!(2), 0)?
            .add32(0, ScratchRegister::R0, SavedRegister::S0, SavedRegister::S1)?;
        emitter.emit_return(ReturnOp::Mov32, ScratchRegister::R0)?;

        let code = compiler.generate_code();
        let func: fn(i32, i32) -> i32 = transmute(code.get());
        assert_eq!(func(10, 20), 30);
    }
    Ok(())
}

#[test]
fn test_rotate_emitter() -> Result<(), Box<dyn Error>> {
    unsafe {
        let mut compiler = Compiler::new();
        let mut emitter = Emitter::new(&mut compiler);
        emitter
            .emit_enter(0, arg_types!(W -> [W, W]), regs!(1), regs!(2), 0)?
            .rotl(0, ScratchRegister::R0, SavedRegister::S0, SavedRegister::S1)?;
        emitter.emit_return(ReturnOp::Mov, ScratchRegister::R0)?;

        let code = compiler.generate_code();
        let func: fn(isize, isize) -> isize = transmute(code.get());
        assert_eq!(func(0b1011, 2), 44);
    }
    Ok(())
}

#[test]
fn test_add_f64_emitter() -> Result<(), Box<dyn Error>> {
    unsafe {
        let mut compiler = Compiler::new();
        let mut emitter = Emitter::new(&mut compiler);
        emitter
            .emit_enter(
                0,
                arg_types!(F64 -> [F64, F64]),
                regs! { float: 2 },
                regs!(0),
                0,
            )?
            .add_f64(
                0,
                FloatRegister::FR0,
                FloatRegister::FR0,
                FloatRegister::FR1,
            )?;
        emitter.emit_return(ReturnOp::MovF64, FloatRegister::FR0)?;

        let code = compiler.generate_code();
        let func: fn(f64, f64) -> f64 = transmute(code.get());
        assert_eq!(func(10.5, 20.25), 30.75);
    }
    Ok(())
}

#[test]
fn test_conv_f64_from_s32_emitter() -> Result<(), Box<dyn Error>> {
    unsafe {
        let mut compiler = Compiler::new();
        let mut emitter = Emitter::new(&mut compiler);
        emitter
            .emit_enter(
                0,
                arg_types!(F64 -> [32]),
                regs! { float: 1 },
                regs! { gp: 1 },
                0,
            )?
            .conv_f64_from_s32(0, FloatRegister::FR0, SavedRegister::S0)?;
        emitter.emit_return(ReturnOp::MovF64, FloatRegister::FR0)?;

        let code = compiler.generate_code();
        let func: fn(i32) -> f64 = transmute(code.get());
        assert_eq!(func(42), 42.0);
    }
    Ok(())
}
#[test]
fn test_branch_emitter() -> Result<(), Box<dyn Error>> {
    unsafe {
        let mut compiler = Compiler::new();
        let mut emitter = Emitter::new(&mut compiler);
        emitter
            .emit_enter(0, arg_types!(W -> [W]), regs!(1), regs!(1), 0)?
            .branch(
                Condition::Equal,
                SavedRegister::S0,
                0isize,
                |e| e.emit_return(ReturnOp::Mov, 10isize),
                |e| e.emit_return(ReturnOp::Mov, 20isize),
            )?;

        let code = compiler.generate_code();
        let func: fn(isize) -> isize = transmute(code.get());
        assert_eq!(func(0), 10);
        assert_eq!(func(5), 20);
    }
    Ok(())
}

#[test]
fn test_branch_extended() -> Result<(), Box<dyn Error>> {
    const TEST_CASES: usize = 44;
    let mut buf = [100u8; TEST_CASES];
    let compare_buf: [u8; TEST_CASES] = [
        1, 1, 2, 2, 1, 2, 2, 1, 1, 2, 2, 2, 1, 2, 2, 1, 1, 2, 2, 2, 1, 2, 2, 2, 2, 1, 1, 2, 2,
        2, 1, 2, 1, 1, 1, 2, 1, 2, 1, 2, 2, 1, 1, 2,
    ];
    let mut data: [isize; 4] = [32, -9, 43, -13];

    unsafe {
        let mut compiler = Compiler::new();
        let mut emitter = Emitter::new(&mut compiler);
        emitter.emit_enter(0, arg_types!([P, P]), regs!(3), regs!(2), 0)?;
        emitter.sub(0, SavedRegister::S0, SavedRegister::S0, 1)?;

        let cmp_test = |emitter: &mut Emitter,
                        type_: Condition,
                        src1: Operand,
                        src2: Operand|
         -> Result<(), ErrorCode> {
            emitter.branch(
                type_,
                src1,
                src2,
                |e| {
                    e.mov_u8(0, mem_offset(SavedRegister::S0, 1), 2)?;
                    Ok(())
                },
                |e| {
                    e.mov_u8(0, mem_offset(SavedRegister::S0, 1), 1)?;
                    Ok(())
                },
            )?;
            emitter.add(0, SavedRegister::S0, SavedRegister::S0, 1)?;
            Ok(())
        };

        emitter.mov(0, ScratchRegister::R0, 13)?;
        emitter.mov(0, ScratchRegister::R1, 15)?;
        cmp_test(
            &mut emitter,
            Condition::Equal,
            9isize.into(),
            ScratchRegister::R0.into(),
        )?;
        cmp_test(
            &mut emitter,
            Condition::Equal,
            ScratchRegister::R0.into(),
            ScratchRegister::R1.into(),
        )?;
        emitter.mov(0, ScratchRegister::R0, 3)?;
        cmp_test(
            &mut emitter,
            Condition::Equal,
            mem_indexed_shift(
                SavedRegister::S1,
                ScratchRegister::R0,
                SLJIT_WORD_SHIFT as u8,
            ),
            (-13isize).into(),
        )?;
        cmp_test(
            &mut emitter,
            Condition::NotEqual,
            0isize.into(),
            ScratchRegister::R0.into(),
        )?;
        emitter.mov(0, ScratchRegister::R0, 0)?;
        cmp_test(
            &mut emitter,
            Condition::NotEqual,
            0isize.into(),
            ScratchRegister::R0.into(),
        )?;
        cmp_test(
            &mut emitter,
            Condition::Equal,
            mem_indexed_shift(
                SavedRegister::S1,
                ScratchRegister::R0,
                SLJIT_WORD_SHIFT as u8,
            ),
            mem_indexed_shift(
                SavedRegister::S1,
                ScratchRegister::R0,
                SLJIT_WORD_SHIFT as u8,
            ),
        )?;
        cmp_test(
            &mut emitter,
            Condition::Equal,
            ScratchRegister::R0.into(),
            0isize.into(),
        )?;

        // compare_buf[7-16]
        cmp_test(
            &mut emitter,
            Condition::SigLess,
            mem_offset(SavedRegister::S1, 0),
            0isize.into(),
        )?;
        emitter.mov(0, ScratchRegister::R0, -8)?;
        emitter.mov(0, ScratchRegister::R1, 0)?;
        cmp_test(
            &mut emitter,
            Condition::SigGreater,
            ScratchRegister::R0.into(),
            0isize.into(),
        )?;
        cmp_test(
            &mut emitter,
            Condition::SigLessEqual,
            ScratchRegister::R0.into(),
            0isize.into(),
        )?;
        cmp_test(
            &mut emitter,
            Condition::SigLess,
            ScratchRegister::R0.into(),
            0isize.into(),
        )?;
        cmp_test(
            &mut emitter,
            Condition::SigGreaterEqual,
            ScratchRegister::R1.into(),
            0isize.into(),
        )?;
        cmp_test(
            &mut emitter,
            Condition::SigGreater,
            0isize.into(),
            mem_offset(
                SavedRegister::S1,
                2 * (core::mem::size_of::<isize>() as i32),
            ),
        )?;
        cmp_test(
            &mut emitter,
            Condition::SigLessEqual,
            0isize.into(),
            ScratchRegister::R1.into(),
        )?;
        cmp_test(
            &mut emitter,
            Condition::SigLess,
            0isize.into(),
            mem_offset(
                SavedRegister::S1,
                2 * (core::mem::size_of::<isize>() as i32),
            ),
        )?;
        cmp_test(
            &mut emitter,
            Condition::SigLess,
            0isize.into(),
            mem_offset(
                SavedRegister::S1,
                3 * (core::mem::size_of::<isize>() as i32),
            ),
        )?;
        cmp_test(
            &mut emitter,
            Condition::SigLess,
            0isize.into(),
            mem_offset(
                SavedRegister::S1,
                3 * (core::mem::size_of::<isize>() as i32),
            ),
        )?;

        // compare_buf[17-28]
        emitter.mov(0, ScratchRegister::R0, 8)?;
        emitter.mov(0, ScratchRegister::R1, 0)?;
        cmp_test(
            &mut emitter,
            Condition::Less,
            ScratchRegister::R0.into(),
            mem_offset(
                SavedRegister::S1,
                1 * (core::mem::size_of::<isize>() as i32),
            ),
        )?;
        cmp_test(
            &mut emitter,
            Condition::GreaterEqual,
            ScratchRegister::R0.into(),
            8isize.into(),
        )?;
        cmp_test(
            &mut emitter,
            Condition::Less,
            ScratchRegister::R0.into(),
            (-10isize).into(),
        )?;
        cmp_test(
            &mut emitter,
            Condition::Less,
            ScratchRegister::R0.into(),
            8isize.into(),
        )?;
        cmp_test(
            &mut emitter,
            Condition::GreaterEqual,
            8isize.into(),
            ScratchRegister::R1.into(),
        )?;
        cmp_test(
            &mut emitter,
            Condition::GreaterEqual,
            8isize.into(),
            ScratchRegister::R1.into(),
        )?;
        cmp_test(
            &mut emitter,
            Condition::Greater,
            8isize.into(),
            ScratchRegister::R1.into(),
        )?;
        cmp_test(
            &mut emitter,
            Condition::LessEqual,
            7isize.into(),
            ScratchRegister::R0.into(),
        )?;
        cmp_test(
            &mut emitter,
            Condition::Greater,
            1isize.into(),
            mem_offset(
                SavedRegister::S1,
                3 * (core::mem::size_of::<isize>() as i32),
            ),
        )?;
        cmp_test(
            &mut emitter,
            Condition::LessEqual,
            ScratchRegister::R0.into(),
            ScratchRegister::R1.into(),
        )?;
        cmp_test(
            &mut emitter,
            Condition::Greater,
            ScratchRegister::R0.into(),
            ScratchRegister::R1.into(),
        )?;
        cmp_test(
            &mut emitter,
            Condition::Greater,
            ScratchRegister::R0.into(),
            ScratchRegister::R1.into(),
        )?;

        // compare_buf[29-39]
        emitter.mov(0, ScratchRegister::R0, -3)?;
        cmp_test(
            &mut emitter,
            Condition::SigLess,
            ScratchRegister::R0.into(),
            ScratchRegister::R1.into(),
        )?;
        cmp_test(
            &mut emitter,
            Condition::SigGreaterEqual,
            ScratchRegister::R0.into(),
            ScratchRegister::R1.into(),
        )?;
        cmp_test(
            &mut emitter,
            Condition::SigLess,
            ScratchRegister::R0.into(),
            (-1isize).into(),
        )?;
        cmp_test(
            &mut emitter,
            Condition::SigGreaterEqual,
            ScratchRegister::R0.into(),
            1isize.into(),
        )?;
        cmp_test(
            &mut emitter,
            Condition::SigLess,
            mem_offset(SavedRegister::S1, 0),
            (-1isize).into(),
        )?;
        cmp_test(
            &mut emitter,
            Condition::SigLess,
            mem_offset(SavedRegister::S1, 0),
            (-1isize).into(),
        )?;
        cmp_test(
            &mut emitter,
            Condition::SigLessEqual,
            ScratchRegister::R0.into(),
            ScratchRegister::R1.into(),
        )?;
        cmp_test(
            &mut emitter,
            Condition::SigGreater,
            ScratchRegister::R0.into(),
            ScratchRegister::R1.into(),
        )?;
        cmp_test(
            &mut emitter,
            Condition::SigLessEqual,
            (-4isize).into(),
            ScratchRegister::R0.into(),
        )?;
        cmp_test(
            &mut emitter,
            Condition::SigGreater,
            (-1isize).into(),
            ScratchRegister::R1.into(),
        )?;
        cmp_test(
            &mut emitter,
            Condition::SigGreater,
            ScratchRegister::R1.into(),
            (-1isize).into(),
        )?;

        #[cfg(target_pointer_width = "64")]
        {
            emitter.mov(0, ScratchRegister::R0, 0xf00000004isize)?;
            emitter.mov(0, ScratchRegister::R1, ScratchRegister::R0)?;
            emitter.mov32(0, ScratchRegister::R1, ScratchRegister::R1)?;
            cmp_test(
                &mut emitter,
                Condition::Less32,
                ScratchRegister::R1.into(),
                5isize.into(),
            )?;
            cmp_test(
                &mut emitter,
                Condition::Less,
                ScratchRegister::R0.into(),
                5isize.into(),
            )?;
            emitter.mov(0, ScratchRegister::R0, 0xff0000004isize)?;
            emitter.mov32(0, ScratchRegister::R1, ScratchRegister::R0)?;
            cmp_test(
                &mut emitter,
                Condition::SigGreater32,
                ScratchRegister::R1.into(),
                5isize.into(),
            )?;
            cmp_test(
                &mut emitter,
                Condition::SigGreater,
                ScratchRegister::R0.into(),
                5isize.into(),
            )?;
        }
        #[cfg(target_pointer_width = "32")]
        {
            emitter.mov32(0, ScratchRegister::R0, 4)?;
            cmp_test(
                &mut emitter,
                Condition::Less32,
                ScratchRegister::R0.into(),
                5isize.into(),
            )?;
            cmp_test(
                &mut emitter,
                Condition::Greater32,
                ScratchRegister::R0.into(),
                5isize.into(),
            )?;
            emitter.mov32(0, ScratchRegister::R0, 0xf0000004u32)?;
            cmp_test(
                &mut emitter,
                Condition::SigGreater32,
                ScratchRegister::R0.into(),
                5isize.into(),
            )?;
            cmp_test(
                &mut emitter,
                Condition::SigLess32,
                ScratchRegister::R0.into(),
                5isize.into(),
            )?;
        }

        emitter.return_void()?;

        let code = compiler.generate_code();
        let func: fn(*mut u8, *mut isize) = transmute(code.get());
        func(buf.as_mut_ptr(), data.as_mut_ptr());

        assert_eq!(buf, compare_buf);
    }
    Ok(())
}
