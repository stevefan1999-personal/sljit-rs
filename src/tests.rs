use crunchy::unroll;

use super::*;
use crate::sys::{
    SLJIT_ARG_TYPE_32, SLJIT_ARG_TYPE_F64, SLJIT_ARG_TYPE_P, SLJIT_ARG_TYPE_W, SLJIT_WORD_SHIFT,
    arg_types,
};
use core::ffi::c_int;
use core::mem::transmute;

#[test]
fn test_add3_emitter() {
    unsafe {
        let mut emitter = Emitter::default();
        emitter
            .emit_enter(0, arg_types!(W -> [W, W, W]), regs!(1), regs!(3), 0)
            .unwrap()
            .mov(0, ScratchRegister::R0, SavedRegister::S0)
            .unwrap()
            .add(
                0,
                ScratchRegister::R0,
                ScratchRegister::R0,
                SavedRegister::S1,
            )
            .unwrap()
            .add(
                0,
                ScratchRegister::R0,
                ScratchRegister::R0,
                SavedRegister::S2,
            )
            .unwrap();

        emitter
            .emit_return(ReturnOp::Mov, ScratchRegister::R0)
            .unwrap();

        let code = emitter.generate_code();
        let func: fn(c_int, c_int, c_int) -> c_int = transmute(code.get());
        assert_eq!(func(4, 5, 6), 4 + 5 + 6);
    }
}

#[test]
fn test_memory_access_emitter() {
    unsafe {
        let mut emitter = Emitter::default();
        emitter
            .emit_enter(0, arg_types!(W -> [P]), regs!(2), regs!(1), 0)
            .unwrap()
            // R0 = S0[0]
            .mov(0, ScratchRegister::R0, mem_offset(SavedRegister::S0, 0))
            .unwrap()
            // R1 = S0[1]
            .mov(
                0,
                ScratchRegister::R1,
                mem_offset(SavedRegister::S0, (1 * (1 << SLJIT_WORD_SHIFT)) as i32),
            )
            .unwrap()
            // R0 = R0 + R1
            .add(
                0,
                ScratchRegister::R0,
                ScratchRegister::R0,
                ScratchRegister::R1,
            )
            .unwrap();

        emitter
            .emit_return(ReturnOp::Mov, ScratchRegister::R0)
            .unwrap();

        let code = emitter.generate_code();
        let data: [isize; 2] = [10, 20];
        let func: fn(p: *const isize) -> isize = transmute(code.get());
        assert_eq!(func(data.as_ptr()), 30);
    }
}

#[test]
fn test_add32_emitter() {
    unsafe {
        let mut emitter = Emitter::default();
        emitter
            .emit_enter(0, arg_types!(32 -> [32, 32]), regs!(1), regs!(2), 0)
            .unwrap()
            .add32(0, ScratchRegister::R0, SavedRegister::S0, SavedRegister::S1)
            .unwrap();
        emitter
            .emit_return(ReturnOp::Mov32, ScratchRegister::R0)
            .unwrap();

        let code = emitter.generate_code();
        let func: fn(i32, i32) -> i32 = transmute(code.get());
        assert_eq!(func(10, 20), 30);
    }
}

#[test]
fn test_rotate_emitter() {
    unsafe {
        let mut emitter = Emitter::default();
        emitter
            .emit_enter(0, arg_types!(W -> [W, W]), regs!(1), regs!(2), 0)
            .unwrap()
            .rotl(0, ScratchRegister::R0, SavedRegister::S0, SavedRegister::S1)
            .unwrap();
        emitter
            .emit_return(ReturnOp::Mov, ScratchRegister::R0)
            .unwrap();

        let code = emitter.generate_code();
        let func: fn(isize, isize) -> isize = transmute(code.get());
        assert_eq!(func(0b1011, 2), 0b101100);
    }
}

#[test]
fn test_add_f64_emitter() {
    unsafe {
        let mut emitter = Emitter::default();
        emitter
            .emit_enter(
                0,
                arg_types!(F64 -> [F64, F64]),
                regs! { float: 2 },
                regs!(0),
                0,
            )
            .unwrap()
            .add_f64(
                0,
                FloatRegister::FR0,
                FloatRegister::FR0,
                FloatRegister::FR1,
            )
            .unwrap();
        emitter
            .emit_return(ReturnOp::MovF64, FloatRegister::FR0)
            .unwrap();

        let code = emitter.generate_code();
        let func: fn(f64, f64) -> f64 = transmute(code.get());
        assert_eq!(func(10.5, 20.25), 30.75);
    }
}

#[test]
fn test_conv_f64_from_s32_emitter() {
    unsafe {
        let mut emitter = Emitter::default();
        emitter
            .emit_enter(
                0,
                arg_types!(F64 -> [32]),
                regs! { gp: 1, float: 1 },
                regs! { gp: 1 },
                0,
            )
            .unwrap()
            .conv_f64_from_s32(0, FloatRegister::FR0, SavedRegister::S0)
            .unwrap();
        emitter
            .emit_return(ReturnOp::MovF64, FloatRegister::FR0)
            .unwrap();

        let code = emitter.generate_code();
        let func: fn(i32) -> f64 = transmute(code.get());
        assert_eq!(func(42), 42.0);
    }
}

#[test]
fn test_branch_emitter() {
    unsafe {
        let mut emitter = Emitter::default();
        emitter
            .emit_enter(0, arg_types!(W -> [W]), regs!(1), regs!(1), 0)
            .unwrap()
            .branch(
                Condition::Equal,
                SavedRegister::S0,
                0isize,
                |e| {
                    e.emit_return(ReturnOp::Mov, 10isize)?;
                    Ok(())
                },
                |e| {
                    e.emit_return(ReturnOp::Mov, 20isize)?;
                    Ok(())
                },
            )
            .unwrap();

        let code = emitter.generate_code();
        let func: fn(isize) -> isize = transmute(code.get());
        assert_eq!(func(0), 10);
        assert_eq!(func(5), 20);
    }
}

#[test]
fn test_branch_extended() {
    const TEST_CASES: usize = 44;
    let mut buf = [100u8; TEST_CASES];
    let compare_buf: [u8; TEST_CASES] = [
        1, 1, 2, 2, 1, 2, 2, 1, 1, 2, 2, 2, 1, 2, 2, 1, 1, 2, 2, 2, 1, 2, 2, 2, 2, 1, 1, 2, 2, 2,
        1, 2, 1, 1, 1, 2, 1, 2, 1, 2, 2, 1, 1, 2,
    ];
    let mut data: [isize; 4] = [32, -9, 43, -13];

    unsafe {
        let mut emitter = Emitter::default();
        emitter
            .emit_enter(
                0,
                arg_types!([P, P]),
                regs! {
                    gp: 2,
                },
                regs! {
                    gp: 2,
                },
                0,
            )
            .unwrap();
        emitter
            .sub(0, SavedRegister::S0, SavedRegister::S0, 1)
            .unwrap();

        let cmp_test = |emitter: &mut Emitter,
                        type_: Condition,
                        src1: Operand,
                        src2: Operand|
         -> Result<(), ErrorCode> {
            emitter
                .branch(
                    type_,
                    src1,
                    src2,
                    |e| {
                        e.mov_u8(0, mem_offset(SavedRegister::S0, 1), 2).unwrap();
                        Ok(())
                    },
                    |e| {
                        e.mov_u8(0, mem_offset(SavedRegister::S0, 1), 1).unwrap();
                        Ok(())
                    },
                )
                .unwrap();
            emitter
                .add(0, SavedRegister::S0, SavedRegister::S0, 1)
                .unwrap();
            Ok(())
        };

        emitter.mov(0, ScratchRegister::R0, 13).unwrap();
        emitter.mov(0, ScratchRegister::R1, 15).unwrap();
        cmp_test(
            &mut emitter,
            Condition::Equal,
            9isize.into(),
            ScratchRegister::R0.into(),
        )
        .unwrap();
        cmp_test(
            &mut emitter,
            Condition::Equal,
            ScratchRegister::R0.into(),
            ScratchRegister::R1.into(),
        )
        .unwrap();
        emitter.mov(0, ScratchRegister::R0, 3).unwrap();
        cmp_test(
            &mut emitter,
            Condition::Equal,
            mem_indexed_shift(
                SavedRegister::S1,
                ScratchRegister::R0,
                SLJIT_WORD_SHIFT as u8,
            ),
            (-13isize).into(),
        )
        .unwrap();
        cmp_test(
            &mut emitter,
            Condition::NotEqual,
            0isize.into(),
            ScratchRegister::R0.into(),
        )
        .unwrap();
        emitter.mov(0, ScratchRegister::R0, 0).unwrap();
        cmp_test(
            &mut emitter,
            Condition::NotEqual,
            0isize.into(),
            ScratchRegister::R0.into(),
        )
        .unwrap();
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
        )
        .unwrap();
        cmp_test(
            &mut emitter,
            Condition::Equal,
            ScratchRegister::R0.into(),
            0isize.into(),
        )
        .unwrap();

        // compare_buf[7-16]
        cmp_test(
            &mut emitter,
            Condition::SigLess,
            mem_offset(SavedRegister::S1, 0),
            0isize.into(),
        )
        .unwrap();
        emitter.mov(0, ScratchRegister::R0, -8).unwrap();
        emitter.mov(0, ScratchRegister::R1, 0).unwrap();
        cmp_test(
            &mut emitter,
            Condition::SigGreater,
            ScratchRegister::R0.into(),
            0isize.into(),
        )
        .unwrap();
        cmp_test(
            &mut emitter,
            Condition::SigLessEqual,
            ScratchRegister::R0.into(),
            0isize.into(),
        )
        .unwrap();
        cmp_test(
            &mut emitter,
            Condition::SigLess,
            ScratchRegister::R0.into(),
            0isize.into(),
        )
        .unwrap();
        cmp_test(
            &mut emitter,
            Condition::SigGreaterEqual,
            ScratchRegister::R1.into(),
            0isize.into(),
        )
        .unwrap();
        cmp_test(
            &mut emitter,
            Condition::SigGreater,
            0isize.into(),
            mem_offset(
                SavedRegister::S1,
                2 * (core::mem::size_of::<isize>() as i32),
            ),
        )
        .unwrap();
        cmp_test(
            &mut emitter,
            Condition::SigLessEqual,
            0isize.into(),
            ScratchRegister::R1.into(),
        )
        .unwrap();
        cmp_test(
            &mut emitter,
            Condition::SigLess,
            0isize.into(),
            mem_offset(
                SavedRegister::S1,
                2 * (core::mem::size_of::<isize>() as i32),
            ),
        )
        .unwrap();
        cmp_test(
            &mut emitter,
            Condition::SigLess,
            0isize.into(),
            mem_offset(
                SavedRegister::S1,
                3 * (core::mem::size_of::<isize>() as i32),
            ),
        )
        .unwrap();
        cmp_test(
            &mut emitter,
            Condition::SigLess,
            0isize.into(),
            mem_offset(
                SavedRegister::S1,
                3 * (core::mem::size_of::<isize>() as i32),
            ),
        )
        .unwrap();

        // compare_buf[17-28]
        emitter.mov(0, ScratchRegister::R0, 8).unwrap();
        emitter.mov(0, ScratchRegister::R1, 0).unwrap();
        cmp_test(
            &mut emitter,
            Condition::Less,
            ScratchRegister::R0.into(),
            mem_offset(
                SavedRegister::S1,
                1 * (core::mem::size_of::<isize>() as i32),
            ),
        )
        .unwrap();
        cmp_test(
            &mut emitter,
            Condition::GreaterEqual,
            ScratchRegister::R0.into(),
            8isize.into(),
        )
        .unwrap();
        cmp_test(
            &mut emitter,
            Condition::Less,
            ScratchRegister::R0.into(),
            (-10isize).into(),
        )
        .unwrap();
        cmp_test(
            &mut emitter,
            Condition::Less,
            ScratchRegister::R0.into(),
            8isize.into(),
        )
        .unwrap();
        cmp_test(
            &mut emitter,
            Condition::GreaterEqual,
            8isize.into(),
            ScratchRegister::R1.into(),
        )
        .unwrap();
        cmp_test(
            &mut emitter,
            Condition::GreaterEqual,
            8isize.into(),
            ScratchRegister::R1.into(),
        )
        .unwrap();
        cmp_test(
            &mut emitter,
            Condition::Greater,
            8isize.into(),
            ScratchRegister::R1.into(),
        )
        .unwrap();
        cmp_test(
            &mut emitter,
            Condition::LessEqual,
            7isize.into(),
            ScratchRegister::R0.into(),
        )
        .unwrap();
        cmp_test(
            &mut emitter,
            Condition::Greater,
            1isize.into(),
            mem_offset(
                SavedRegister::S1,
                3 * (core::mem::size_of::<isize>() as i32),
            ),
        )
        .unwrap();
        cmp_test(
            &mut emitter,
            Condition::LessEqual,
            ScratchRegister::R0.into(),
            ScratchRegister::R1.into(),
        )
        .unwrap();
        cmp_test(
            &mut emitter,
            Condition::Greater,
            ScratchRegister::R0.into(),
            ScratchRegister::R1.into(),
        )
        .unwrap();
        cmp_test(
            &mut emitter,
            Condition::Greater,
            ScratchRegister::R0.into(),
            ScratchRegister::R1.into(),
        )
        .unwrap();

        // compare_buf[29-39]
        emitter.mov(0, ScratchRegister::R0, -3).unwrap();
        cmp_test(
            &mut emitter,
            Condition::SigLess,
            ScratchRegister::R0.into(),
            ScratchRegister::R1.into(),
        )
        .unwrap();
        cmp_test(
            &mut emitter,
            Condition::SigGreaterEqual,
            ScratchRegister::R0.into(),
            ScratchRegister::R1.into(),
        )
        .unwrap();
        cmp_test(
            &mut emitter,
            Condition::SigLess,
            ScratchRegister::R0.into(),
            (-1isize).into(),
        )
        .unwrap();
        cmp_test(
            &mut emitter,
            Condition::SigGreaterEqual,
            ScratchRegister::R0.into(),
            1isize.into(),
        )
        .unwrap();
        cmp_test(
            &mut emitter,
            Condition::SigLess,
            mem_offset(SavedRegister::S1, 0),
            (-1isize).into(),
        )
        .unwrap();
        cmp_test(
            &mut emitter,
            Condition::SigLess,
            mem_offset(SavedRegister::S1, 0),
            (-1isize).into(),
        )
        .unwrap();
        cmp_test(
            &mut emitter,
            Condition::SigLessEqual,
            ScratchRegister::R0.into(),
            ScratchRegister::R1.into(),
        )
        .unwrap();
        cmp_test(
            &mut emitter,
            Condition::SigGreater,
            ScratchRegister::R0.into(),
            ScratchRegister::R1.into(),
        )
        .unwrap();
        cmp_test(
            &mut emitter,
            Condition::SigLessEqual,
            (-4isize).into(),
            ScratchRegister::R0.into(),
        )
        .unwrap();
        cmp_test(
            &mut emitter,
            Condition::SigGreater,
            (-1isize).into(),
            ScratchRegister::R1.into(),
        )
        .unwrap();
        cmp_test(
            &mut emitter,
            Condition::SigGreater,
            ScratchRegister::R1.into(),
            (-1isize).into(),
        )
        .unwrap();

        #[cfg(target_pointer_width = "64")]
        {
            emitter
                .mov(0, ScratchRegister::R0, 0xf00000004isize)
                .unwrap();
            emitter
                .mov(0, ScratchRegister::R1, ScratchRegister::R0)
                .unwrap();
            emitter
                .mov32(0, ScratchRegister::R1, ScratchRegister::R1)
                .unwrap();
            cmp_test(
                &mut emitter,
                Condition::Less32,
                ScratchRegister::R1.into(),
                5isize.into(),
            )
            .unwrap();
            cmp_test(
                &mut emitter,
                Condition::Less,
                ScratchRegister::R0.into(),
                5isize.into(),
            )
            .unwrap();
            emitter
                .mov(0, ScratchRegister::R0, 0xff0000004isize)
                .unwrap();
            emitter
                .mov32(0, ScratchRegister::R1, ScratchRegister::R0)
                .unwrap();
            cmp_test(
                &mut emitter,
                Condition::SigGreater32,
                ScratchRegister::R1.into(),
                5isize.into(),
            )
            .unwrap();
            cmp_test(
                &mut emitter,
                Condition::SigGreater,
                ScratchRegister::R0.into(),
                5isize.into(),
            )
            .unwrap();
        }
        #[cfg(target_pointer_width = "32")]
        {
            emitter.mov32(0, ScratchRegister::R0, 4).unwrap();
            cmp_test(
                &mut emitter,
                Condition::Less32,
                ScratchRegister::R0.into(),
                5isize.into(),
            )
            .unwrap();
            cmp_test(
                &mut emitter,
                Condition::Greater32,
                ScratchRegister::R0.into(),
                5isize.into(),
            )
            .unwrap();
            emitter
                .mov32(0, ScratchRegister::R0, 0xf0000004u32)
                .unwrap();
            cmp_test(
                &mut emitter,
                Condition::SigGreater32,
                ScratchRegister::R0.into(),
                5isize.into(),
            )
            .unwrap();
            cmp_test(
                &mut emitter,
                Condition::SigLess32,
                ScratchRegister::R0.into(),
                5isize.into(),
            )
            .unwrap();
        }

        emitter.return_void().unwrap();

        let code = emitter.generate_code();
        let func: fn(*mut u8, *mut isize) = transmute(code.get());
        func(buf.as_mut_ptr(), data.as_mut_ptr());

        assert_eq!(buf, compare_buf);
    }
}

const fn fib(n: usize) -> usize {
    if n <= 0 {
        return 0;
    }
    if n == 1 {
        return 1;
    }
    let mut a = 0usize;
    let mut b = 1usize;
    let mut i = 2usize;
    while i <= n {
        let c = a + b;
        a = b;
        b = c;
        i += 1;
    }
    b
}

#[test]
fn test_fib_recursive() {
    unsafe {
        let mut emitter = Emitter::default();

        // Function: fib(n) -> if n <= 1 return n else return fib(n-1) + fib(n-2)
        // Need 3 saved registers to preserve values across recursive calls:
        // - S0: input parameter n (also used to pass argument to recursive calls)
        // - S1: saved copy of n across calls
        // - S2: saved result of fib(n-1) across second call

        // Create label at VERY BEGINNING for recursive calls - BEFORE emit_enter!
        // This is crucial: recursive calls must execute emit_enter to set up their stack frame
        let mut entry_label = emitter.put_label().unwrap();

        emitter
            .emit_enter(0, arg_types!(W -> [W]), regs!(3), regs!(3), 0)
            .unwrap();

        // Use high-level branch API
        emitter
            .branch(
                Condition::SigLessEqual,
                SavedRegister::S0,
                1isize,
                |e| {
                    // Base case: return n (when n <= 1)
                    e.emit_return(ReturnOp::Mov, SavedRegister::S0)?;
                    Ok(())
                },
                |e| {
                    // Recursive case: fib(n-1) + fib(n-2)

                    // Save original n to S1 (saved register survives function calls)
                    e.mov(0, SavedRegister::S1, SavedRegister::S0).unwrap();

                    // Call fib(n-1): argument goes in R0 (SLJIT call convention for emit_call)
                    // emit_call expects arguments in R0-R2, emit_enter then moves them to S0-S2
                    e.sub(0, ScratchRegister::R0, SavedRegister::S1, 1).unwrap();
                    e.call(JumpType::Call, arg_types!([W]))
                        .unwrap()
                        .set_label(&mut entry_label);

                    // Save fib(n-1) result to S2 (R0 has the return value)
                    e.mov(0, SavedRegister::S2, ScratchRegister::R0).unwrap();

                    // Call fib(n-2): argument goes in R0
                    e.sub(0, ScratchRegister::R0, SavedRegister::S1, 2).unwrap();
                    e.call(JumpType::Call, arg_types!([W]))
                        .unwrap()
                        .set_label(&mut entry_label);

                    // Add fib(n-1) + fib(n-2)
                    // R0 has fib(n-2), S2 has fib(n-1)
                    e.add(
                        0,
                        ScratchRegister::R0,
                        ScratchRegister::R0,
                        SavedRegister::S2,
                    )
                    .unwrap();

                    // Return result
                    e.emit_return(ReturnOp::Mov, ScratchRegister::R0)?;
                    Ok(())
                },
            )
            .unwrap();

        let code = emitter.generate_code();
        let func: fn(usize) -> usize = transmute(code.get());

        // Test cases
        unroll! {
            for i in 0..20 {
                assert_eq!(func(i), fib(i));
            }
        }
    }
}

#[test]
fn test_fib_iterative() {
    unsafe {
        let mut emitter = Emitter::default();

        // Function: fib(n) -> iterative version
        emitter
            .emit_enter(0, arg_types!(W -> [W]), regs!(4), regs!(1), 0)
            .unwrap();

        // Handle base cases
        emitter
            .branch(
                Condition::LessEqual,
                SavedRegister::S0,
                1isize,
                |e| {
                    // Base case: return n
                    e.emit_return(ReturnOp::Mov, SavedRegister::S0)?;
                    Ok(())
                },
                |e| {
                    // Iterative case - run loop n-1 times

                    // R0 = a (starts as 0)
                    e.mov(0, ScratchRegister::R0, 0isize).unwrap();
                    // R1 = b (starts as 1)
                    e.mov(0, ScratchRegister::R1, 1isize).unwrap();
                    // R2 = counter (starts as n-1, since we need n-1 iterations for fib(n))
                    e.sub(0, ScratchRegister::R2, SavedRegister::S0, 1).unwrap();

                    // Loop start
                    let mut loop_start = e.put_label().unwrap();

                    // Check if counter == 0 FIRST (like while_ does)
                    let mut jump_to_end = e
                        .cmp(Condition::Equal, ScratchRegister::R2, 0isize)
                        .unwrap();

                    // Decrement counter (AFTER the check)
                    e.sub(0, ScratchRegister::R2, ScratchRegister::R2, 1)
                        .unwrap();

                    // Loop body: a, b = b, a + b
                    e.add(
                        0,
                        ScratchRegister::R3,
                        ScratchRegister::R0,
                        ScratchRegister::R1,
                    )
                    .unwrap();
                    e.mov(0, ScratchRegister::R0, ScratchRegister::R1).unwrap();
                    e.mov(0, ScratchRegister::R1, ScratchRegister::R3).unwrap();

                    // Jump back to loop start
                    let mut jump_to_start = e.jump(JumpType::Jump).unwrap();
                    let mut loop_end = e.put_label().unwrap();

                    // Set jump targets
                    jump_to_end.set_label(&mut loop_end);
                    jump_to_start.set_label(&mut loop_start);

                    // Return b (which has the result)
                    e.emit_return(ReturnOp::Mov, ScratchRegister::R1)?;
                    Ok(())
                },
            )
            .unwrap();

        let code = emitter.generate_code();
        let func: fn(usize) -> usize = transmute(code.get());

        // Test cases
        unroll! {
            for i in 0..20 {
                assert_eq!(func(i), fib(i));
            }
        }
    }
}

#[test]
fn test_fib_iterative_while() {
    unsafe {
        let mut emitter = Emitter::default();

        // Function: fib(n) -> iterative version
        emitter
            .emit_enter(0, arg_types!(W -> [W]), regs!(4), regs!(1), 0)
            .unwrap();

        // Handle base cases
        emitter
            .branch(
                Condition::LessEqual,
                SavedRegister::S0,
                1isize,
                |e| {
                    // Base case: return n
                    e.emit_return(ReturnOp::Mov, SavedRegister::S0)?;
                    Ok(())
                },
                |e| {
                    // Iterative case using the new loop API

                    // R0 = a (starts as 0)
                    e.mov(0, ScratchRegister::R0, 0isize).unwrap();
                    // R1 = b (starts as 1)
                    e.mov(0, ScratchRegister::R1, 1isize).unwrap();
                    // R2 = counter (starts as n-1, since we need n-1 iterations for fib(n))
                    e.sub(0, ScratchRegister::R2, SavedRegister::S0, 1).unwrap();

                    // Use the ergonomic while_ API
                    e.while_(Condition::NotEqual, ScratchRegister::R2, 0isize, |e, _| {
                        // Decrement counter
                        e.sub(0, ScratchRegister::R2, ScratchRegister::R2, 1)
                            .unwrap();

                        // Loop body: a, b = b, a + b
                        e.add(
                            0,
                            ScratchRegister::R3,
                            ScratchRegister::R0,
                            ScratchRegister::R1,
                        )
                        .unwrap();
                        e.mov(0, ScratchRegister::R0, ScratchRegister::R1).unwrap();
                        e.mov(0, ScratchRegister::R1, ScratchRegister::R3).unwrap();

                        Ok(())
                    })
                    .unwrap();

                    // Return b (which has the result)
                    e.emit_return(ReturnOp::Mov, ScratchRegister::R1)?;
                    Ok(())
                },
            )
            .unwrap();

        let code = emitter.generate_code();
        let func: fn(usize) -> usize = transmute(code.get());

        // Test cases
        unroll! {
            for i in 0..20 {
                assert_eq!(func(i), fib(i));
            }
        }
    }
}

#[test]
fn test_fib_iterative_loop_break() {
    unsafe {
        let mut emitter = Emitter::default();

        // Function: fib(n) -> iterative version using loop with break
        emitter
            .emit_enter(0, arg_types!(W -> [W]), regs!(4), regs!(1), 0)
            .unwrap();

        // Handle base cases
        emitter
            .branch(
                Condition::LessEqual,
                SavedRegister::S0,
                1isize,
                |e| {
                    // Base case: return n
                    e.emit_return(ReturnOp::Mov, SavedRegister::S0)?;
                    Ok(())
                },
                |e| {
                    // Iterative case using the loop API with break

                    // R0 = a (starts as 0)
                    e.mov(0, ScratchRegister::R0, 0isize).unwrap();
                    // R1 = b (starts as 1)
                    e.mov(0, ScratchRegister::R1, 1isize).unwrap();
                    // R2 = counter (starts as n)
                    e.mov(0, ScratchRegister::R2, SavedRegister::S0).unwrap();

                    // Use the ergonomic loop_ API with break
                    e.loop_(|e, ctx| {
                        // Decrement counter
                        e.sub(0, ScratchRegister::R2, ScratchRegister::R2, 1)
                            .unwrap();
                        e.branch(
                            Condition::NotEqual,
                            ScratchRegister::R2,
                            0isize,
                            |e| {
                                // Loop body: a, b = b, a + b
                                e.add(
                                    0,
                                    ScratchRegister::R3,
                                    ScratchRegister::R0,
                                    ScratchRegister::R1,
                                )
                                .unwrap();
                                e.mov(0, ScratchRegister::R0, ScratchRegister::R1).unwrap();
                                e.mov(0, ScratchRegister::R1, ScratchRegister::R3).unwrap();
                                Ok(())
                            },
                            |_| {
                                ctx.break_()?;
                                Ok(())
                            },
                        )?;
                        Ok(())
                    })
                    .unwrap();

                    // Return b (which has the result)
                    e.emit_return(ReturnOp::Mov, ScratchRegister::R1).unwrap();
                    Ok(())
                },
            )
            .unwrap();

        let code = emitter.generate_code();
        let func: fn(usize) -> usize = transmute(code.get());

        // Test cases - should match test_fib_iterative
        unroll! {
            for i in 0..20 {
                assert_eq!(func(i), fib(i));
            }
        }
    }
}

/// Test while_ loop: computes sum from 1 to N using while loop
#[test]
fn test_while_sum() {
    unsafe {
        let mut emitter = Emitter::default();

        // Function: sum(n) -> 1 + 2 + ... + n
        emitter
            .emit_enter(0, arg_types!(W -> [W]), regs!(3), regs!(1), 0)
            .unwrap();

        // R0 = sum (accumulator, starts at 0)
        emitter.mov(0, ScratchRegister::R0, 0isize).unwrap();
        // R1 = counter (starts at n)
        emitter
            .mov(0, ScratchRegister::R1, SavedRegister::S0)
            .unwrap();

        // while counter > 0
        emitter
            .while_(
                Condition::SigGreater,
                ScratchRegister::R1,
                0isize,
                |e, _ctx| {
                    // sum += counter
                    e.add(
                        0,
                        ScratchRegister::R0,
                        ScratchRegister::R0,
                        ScratchRegister::R1,
                    )
                    .unwrap();
                    // counter--
                    e.sub(0, ScratchRegister::R1, ScratchRegister::R1, 1)
                        .unwrap();
                    Ok(())
                },
            )
            .unwrap();

        emitter
            .emit_return(ReturnOp::Mov, ScratchRegister::R0)
            .unwrap();

        let code = emitter.generate_code();
        let func: fn(isize) -> isize = transmute(code.get());

        // sum(0) = 0 (loop doesn't execute)
        assert_eq!(func(0), 0);
        // sum(1) = 1
        assert_eq!(func(1), 1);
        // sum(5) = 1+2+3+4+5 = 15
        assert_eq!(func(5), 15);
        // sum(10) = 55
        assert_eq!(func(10), 55);
        // sum(100) = 5050
        assert_eq!(func(100), 5050);
    }
}

/// Test do_while_ loop: body always executes at least once
#[test]
fn test_do_while_at_least_once() {
    unsafe {
        let mut emitter = Emitter::default();

        // Function: count_iterations(n) -> counts how many times the loop body executes
        // do { count++; n--; } while (n > 0)
        // For n=0, should return 1 (body executes once before condition is checked)
        // For n=3, should return 3
        emitter
            .emit_enter(0, arg_types!(W -> [W]), regs!(2), regs!(1), 0)
            .unwrap();

        // R0 = count (starts at 0)
        emitter.mov(0, ScratchRegister::R0, 0isize).unwrap();
        // R1 = n (copy of input)
        emitter
            .mov(0, ScratchRegister::R1, SavedRegister::S0)
            .unwrap();

        // do { count++; n--; } while (n > 0)
        emitter
            .do_while_(
                |e, _ctx| {
                    // count++
                    e.add(0, ScratchRegister::R0, ScratchRegister::R0, 1)
                        .unwrap();
                    // n--
                    e.sub(0, ScratchRegister::R1, ScratchRegister::R1, 1)
                        .unwrap();
                    Ok(())
                },
                Condition::SigGreater,
                ScratchRegister::R1,
                0isize,
            )
            .unwrap();

        emitter
            .emit_return(ReturnOp::Mov, ScratchRegister::R0)
            .unwrap();

        let code = emitter.generate_code();
        let func: fn(isize) -> isize = transmute(code.get());

        // n=0: body executes once (do-while always executes at least once)
        assert_eq!(func(0), 1);
        // n=1: body executes once
        assert_eq!(func(1), 1);
        // n=3: body executes 3 times
        assert_eq!(func(3), 3);
        // n=5: body executes 5 times
        assert_eq!(func(5), 5);
    }
}

/// Test do_while_ loop: compute factorial using do-while
#[test]
fn test_do_while_factorial() {
    unsafe {
        let mut emitter = Emitter::default();

        // Function: factorial(n) -> n! (assumes n >= 1)
        // do { result *= counter; counter--; } while (counter > 0)
        emitter
            .emit_enter(0, arg_types!(W -> [W]), regs!(2), regs!(1), 0)
            .unwrap();

        // Handle n <= 0 case
        emitter
            .branch(
                Condition::SigLessEqual,
                SavedRegister::S0,
                0isize,
                |e| {
                    e.emit_return(ReturnOp::Mov, 1isize).unwrap();
                    Ok(())
                },
                |e| {
                    // R0 = result (starts at 1)
                    e.mov(0, ScratchRegister::R0, 1isize).unwrap();
                    // R1 = counter (starts at n)
                    e.mov(0, ScratchRegister::R1, SavedRegister::S0).unwrap();

                    // do { result *= counter; counter--; } while (counter > 0)
                    e.do_while_(
                        |e, _ctx| {
                            // result *= counter
                            e.mul(
                                0,
                                ScratchRegister::R0,
                                ScratchRegister::R0,
                                ScratchRegister::R1,
                            )
                            .unwrap();
                            // counter--
                            e.sub(0, ScratchRegister::R1, ScratchRegister::R1, 1)
                                .unwrap();
                            Ok(())
                        },
                        Condition::SigGreater,
                        ScratchRegister::R1,
                        0isize,
                    )
                    .unwrap();

                    e.emit_return(ReturnOp::Mov, ScratchRegister::R0).unwrap();
                    Ok(())
                },
            )
            .unwrap();

        let code = emitter.generate_code();
        let func: fn(isize) -> isize = transmute(code.get());

        // 0! = 1
        assert_eq!(func(0), 1);
        // 1! = 1
        assert_eq!(func(1), 1);
        // 2! = 2
        assert_eq!(func(2), 2);
        // 3! = 6
        assert_eq!(func(3), 6);
        // 5! = 120
        assert_eq!(func(5), 120);
        // 10! = 3628800
        assert_eq!(func(10), 3628800);
    }
}

/// Test loop_ with continue: skip even numbers when summing
#[test]
fn test_loop_continue() {
    unsafe {
        let mut emitter = Emitter::default();

        // Function: sum_odd(n) -> sum of odd numbers from 1 to n
        // Uses continue to skip even numbers
        emitter
            .emit_enter(0, arg_types!(W -> [W]), regs!(3), regs!(1), 0)
            .unwrap();

        // R0 = sum (accumulator)
        emitter.mov(0, ScratchRegister::R0, 0isize).unwrap();
        // R1 = counter (starts at 0)
        emitter.mov(0, ScratchRegister::R1, 0isize).unwrap();
        // S0 = n (upper limit)

        emitter
            .loop_(|e, ctx| {
                // counter++
                e.add(0, ScratchRegister::R1, ScratchRegister::R1, 1)
                    .unwrap();

                // if counter > n, break
                ctx.break_on_cmp(
                    Condition::SigGreater,
                    ScratchRegister::R1,
                    SavedRegister::S0,
                )?;

                // R2 = counter & 1 (check if odd)
                e.and(0, ScratchRegister::R2, ScratchRegister::R1, 1)
                    .unwrap();

                // if even (R2 == 0), continue to skip
                ctx.continue_on_cmp(Condition::Equal, ScratchRegister::R2, 0isize)?;

                // sum += counter (only for odd numbers)
                e.add(
                    0,
                    ScratchRegister::R0,
                    ScratchRegister::R0,
                    ScratchRegister::R1,
                )
                .unwrap();

                Ok(())
            })
            .unwrap();

        emitter
            .emit_return(ReturnOp::Mov, ScratchRegister::R0)
            .unwrap();

        let code = emitter.generate_code();
        let func: fn(isize) -> isize = transmute(code.get());

        // sum_odd(0) = 0
        assert_eq!(func(0), 0);
        // sum_odd(1) = 1
        assert_eq!(func(1), 1);
        // sum_odd(2) = 1 (only 1 is odd)
        assert_eq!(func(2), 1);
        // sum_odd(5) = 1+3+5 = 9
        assert_eq!(func(5), 9);
        // sum_odd(10) = 1+3+5+7+9 = 25
        assert_eq!(func(10), 25);
    }
}

/// Test while_ with break: find first value greater than target in array
#[test]
fn test_while_break() {
    unsafe {
        let mut emitter = Emitter::default();

        // Function: find_greater(arr, len, target) -> first value > target or -1
        emitter
            .emit_enter(0, arg_types!(W -> [P, W, W]), regs!(3), regs!(3), 0)
            .unwrap();

        // S0 = arr pointer
        // S1 = len
        // S2 = target

        // R0 = index (starts at 0)
        emitter.mov(0, ScratchRegister::R0, 0isize).unwrap();
        // R2 = result (-1 = not found)
        emitter.mov(0, ScratchRegister::R2, -1isize).unwrap();

        // while index < len
        emitter
            .while_(
                Condition::SigLess,
                ScratchRegister::R0,
                SavedRegister::S1,
                |e, ctx| {
                    // R1 = arr[index]
                    e.mov(
                        0,
                        ScratchRegister::R1,
                        mem_indexed_shift(
                            SavedRegister::S0,
                            ScratchRegister::R0,
                            SLJIT_WORD_SHIFT as u8,
                        ),
                    )
                    .unwrap();

                    // if arr[index] > target, found it, break
                    e.branch(
                        Condition::SigGreater,
                        ScratchRegister::R1,
                        SavedRegister::S2,
                        |e| {
                            // result = arr[index]
                            e.mov(0, ScratchRegister::R2, ScratchRegister::R1).unwrap();
                            ctx.break_()?;
                            Ok(())
                        },
                        |e| {
                            e.add(0, ScratchRegister::R0, ScratchRegister::R0, 1)?;
                            Ok(())
                        },
                    )?;

                    // index++
                    Ok(())
                },
            )
            .unwrap();

        emitter
            .emit_return(ReturnOp::Mov, ScratchRegister::R2)
            .unwrap();

        let code = emitter.generate_code();
        let func: fn(*const isize, isize, isize) -> isize = transmute(code.get());

        let arr: [isize; 5] = [10, 20, 30, 40, 50];

        // Find first value > 15 -> 20
        assert_eq!(func(arr.as_ptr(), 5, 15), 20);
        // Find first value > 25 -> 30
        assert_eq!(func(arr.as_ptr(), 5, 25), 30);
        // Find first value > 5 -> 10 (first element)
        assert_eq!(func(arr.as_ptr(), 5, 5), 10);
        // Find first value > 50 -> -1 (not found)
        assert_eq!(func(arr.as_ptr(), 5, 50), -1);
        // Empty array
        assert_eq!(func(arr.as_ptr(), 0, 10), -1);
    }
}

/// Test do_while_ with break: search for value
#[test]
fn test_do_while_break() {
    unsafe {
        let mut emitter = Emitter::default();

        // Function: find_in_array(arr, len, target) -> index or -1
        // Searches for target in array, returns index or -1 if not found
        emitter
            .emit_enter(0, arg_types!(W -> [P, W, W]), regs!(3), regs!(3), 0)
            .unwrap();

        // S0 = arr pointer
        // S1 = len
        // S2 = target

        // Handle empty array
        emitter
            .branch(
                Condition::SigLessEqual,
                SavedRegister::S1,
                0isize,
                |e| {
                    e.emit_return(ReturnOp::Mov, -1isize).unwrap();
                    Ok(())
                },
                |e| {
                    // R0 = index (starts at 0)
                    e.mov(0, ScratchRegister::R0, 0isize).unwrap();
                    // R2 = result (-1 = not found)
                    e.mov(0, ScratchRegister::R2, -1isize).unwrap();

                    // do { check arr[index]; index++; } while (index < len)
                    e.do_while_(
                        |e, ctx| {
                            // R1 = arr[index]
                            e.mov(
                                0,
                                ScratchRegister::R1,
                                mem_indexed_shift(
                                    SavedRegister::S0,
                                    ScratchRegister::R0,
                                    SLJIT_WORD_SHIFT as u8,
                                ),
                            )
                            .unwrap();

                            // if arr[index] == target, found it
                            e.branch(
                                Condition::Equal,
                                ScratchRegister::R1,
                                SavedRegister::S2,
                                |e| {
                                    // result = index
                                    e.mov(0, ScratchRegister::R2, ScratchRegister::R0).unwrap();
                                    ctx.break_()?;
                                    Ok(())
                                },
                                |_e| Ok(()),
                            )?;

                            // index++
                            e.add(0, ScratchRegister::R0, ScratchRegister::R0, 1)
                                .unwrap();

                            Ok(())
                        },
                        Condition::SigLess,
                        ScratchRegister::R0,
                        SavedRegister::S1,
                    )
                    .unwrap();

                    e.emit_return(ReturnOp::Mov, ScratchRegister::R2).unwrap();
                    Ok(())
                },
            )
            .unwrap();

        let code = emitter.generate_code();
        let func: fn(*const isize, isize, isize) -> isize = transmute(code.get());

        let arr: [isize; 5] = [10, 20, 30, 40, 50];

        // Find first element
        assert_eq!(func(arr.as_ptr(), 5, 10), 0);
        // Find middle element
        assert_eq!(func(arr.as_ptr(), 5, 30), 2);
        // Find last element
        assert_eq!(func(arr.as_ptr(), 5, 50), 4);
        // Not found
        assert_eq!(func(arr.as_ptr(), 5, 99), -1);
        // Empty array
        assert_eq!(func(arr.as_ptr(), 0, 10), -1);
    }
}
