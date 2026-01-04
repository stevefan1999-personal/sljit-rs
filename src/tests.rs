use super::*;
use crate::sys::{
    SLJIT_ARG_TYPE_32, SLJIT_ARG_TYPE_F64, SLJIT_ARG_TYPE_P, SLJIT_ARG_TYPE_W, SLJIT_WORD_SHIFT,
    arg_types,
};
use core::ffi::c_int;
use core::mem::transmute;

fn simd_set(buf: &mut [u8], mut start: u8, length: i32) {
    for i in 0..length {
        buf[i as usize] = start;
        start = start.wrapping_add(103);
        if start == 0xaa {
            start = 0xab;
        }
    }
}

fn check_simd_mov(buf: &[u8], mut start: u8, length: i32) -> bool {
    for i in 0..length {
        if buf[i as usize] != start {
            return false;
        }
        start = start.wrapping_add(103);
        if start == 0xaa {
            start = 0xab;
        }
    }
    true
}

#[test]
fn test_simd1() {
    unsafe {
        let mut data = [0u8; 63 + 880];
        let buf_ptr = {
            let mut buf_addr = data.as_mut_ptr() as usize;
            buf_addr = (buf_addr + 63) & !63;
            buf_addr as *mut u8
        };

        for i in 0..880 {
            buf_ptr.add(i).write(0xaa);
        }

        simd_set(
            &mut *core::slice::from_raw_parts_mut(buf_ptr.add(0), 16),
            81,
            16,
        );
        simd_set(
            &mut *core::slice::from_raw_parts_mut(buf_ptr.add(65), 16),
            213,
            16,
        );
        simd_set(
            &mut *core::slice::from_raw_parts_mut(buf_ptr.add(104), 16),
            33,
            16,
        );
        simd_set(
            &mut *core::slice::from_raw_parts_mut(buf_ptr.add(160), 16),
            140,
            16,
        );
        simd_set(
            &mut *core::slice::from_raw_parts_mut(buf_ptr.add(210), 16),
            7,
            16,
        );
        simd_set(
            &mut *core::slice::from_raw_parts_mut(buf_ptr.add(256), 16),
            239,
            16,
        );
        simd_set(
            &mut *core::slice::from_raw_parts_mut(buf_ptr.add(312), 16),
            176,
            16,
        );
        simd_set(
            &mut *core::slice::from_raw_parts_mut(buf_ptr.add(368), 8),
            88,
            8,
        );
        simd_set(
            &mut *core::slice::from_raw_parts_mut(buf_ptr.add(393), 8),
            197,
            8,
        );
        simd_set(
            &mut *core::slice::from_raw_parts_mut(buf_ptr.add(416), 16),
            58,
            16,
        );
        simd_set(
            &mut *core::slice::from_raw_parts_mut(buf_ptr.add(432), 16),
            203,
            16,
        );
        simd_set(
            &mut *core::slice::from_raw_parts_mut(buf_ptr.add(496), 16),
            105,
            16,
        );
        simd_set(
            &mut *core::slice::from_raw_parts_mut(buf_ptr.add(560), 16),
            19,
            16,
        );
        simd_set(
            &mut *core::slice::from_raw_parts_mut(buf_ptr.add(616), 8),
            202,
            8,
        );
        simd_set(
            &mut *core::slice::from_raw_parts_mut(buf_ptr.add(648), 8),
            123,
            8,
        );
        simd_set(
            &mut *core::slice::from_raw_parts_mut(buf_ptr.add(704), 32),
            85,
            32,
        );
        simd_set(
            &mut *core::slice::from_raw_parts_mut(buf_ptr.add(801), 32),
            215,
            32,
        );

        let mut compiler = Compiler::new();
        let mut emitter = Emitter::new(&mut compiler);
        let vs0: Operand = if sys::SLJIT_NUMBER_OF_SAVED_VECTOR_REGISTERS > 0 {
            SavedVectorRegister::VS0.into()
        } else {
            VectorRegister::VR5.into()
        };

        emitter.emit_enter(
            0,
            arg_types!([P] -> W),
            regs!(gp: 2, vector: 6),
            regs!(gp: 2, vector: if sys::SLJIT_NUMBER_OF_SAVED_VECTOR_REGISTERS > 0 { 2 } else { 0 }),
            64,
        ).unwrap();

        let mut type_ =
            SimdReg::Reg128 as i32 | SimdElem::Elem8 as i32 | SimdMemAlign::Aligned128 as i32;
        emitter
            .simd_mov(
                SimdType::Load as i32 | type_,
                VectorRegister::VR0,
                mem(SavedRegister::S0),
            )
            .unwrap();
        /* buf[32] */
        emitter
            .simd_mov(
                SimdType::Store as i32 | type_,
                VectorRegister::VR0,
                mem_offset(SavedRegister::S0, 32),
            )
            .unwrap();

        emitter.mov(0, ScratchRegister::R0, 65).unwrap();
        emitter
            .mov(0, ScratchRegister::R1, (82 >> 1) as isize)
            .unwrap();
        type_ = SimdReg::Reg128 as i32 | SimdElem::Elem8 as i32 | SimdMemAlign::Unaligned as i32;
        emitter
            .simd_mov(
                SimdType::Load as i32 | type_,
                VectorRegister::VR2,
                mem_indexed(SavedRegister::S0, ScratchRegister::R0),
            )
            .unwrap();
        /* buf[82] */
        emitter
            .simd_mov(
                SimdType::Store as i32 | type_,
                VectorRegister::VR2,
                mem_indexed_shift(SavedRegister::S0, ScratchRegister::R1, 0),
            )
            .unwrap();

        emitter
            .sub(0, ScratchRegister::R0, SavedRegister::S0, 70001)
            .unwrap();
        emitter
            .add(0, ScratchRegister::R1, SavedRegister::S0, 70001)
            .unwrap();
        type_ = SimdReg::Reg128 as i32 | SimdElem::Elem32 as i32 | SimdMemAlign::Aligned64 as i32;
        emitter
            .simd_mov(
                SimdType::Load as i32 | type_,
                VectorRegister::VR4,
                mem_offset(ScratchRegister::R0, 70001 + 104),
            )
            .unwrap();
        /* buf[136] */
        emitter
            .simd_mov(
                SimdType::Store as i32 | type_,
                VectorRegister::VR4,
                mem_offset(ScratchRegister::R1, 136 - 70001),
            )
            .unwrap();

        type_ = SimdReg::Reg128 as i32
            | SimdElem::Elem32 as i32
            | sys::SLJIT_SIMD_FLOAT
            | SimdMemAlign::Aligned128 as i32;
        emitter
            .simd_mov(
                SimdType::Load as i32 | type_,
                vs0,
                mem_abs(buf_ptr.add(160) as isize),
            )
            .unwrap();
        /* buf[192] */
        emitter
            .simd_mov(
                SimdType::Store as i32 | type_,
                vs0,
                mem_abs(buf_ptr.add(192) as isize),
            )
            .unwrap();

        emitter
            .sub(0, ScratchRegister::R0, SavedRegister::S0, 1001)
            .unwrap();
        emitter
            .add(0, ScratchRegister::R1, SavedRegister::S0, 1001)
            .unwrap();
        type_ = SimdReg::Reg128 as i32
            | SimdElem::Elem32 as i32
            | sys::SLJIT_SIMD_FLOAT
            | SimdMemAlign::Aligned16 as i32;
        emitter
            .simd_mov(
                SimdType::Load as i32 | type_,
                VectorRegister::VR2,
                mem_offset(ScratchRegister::R0, 1001 + 210),
            )
            .unwrap();
        /* buf[230] */
        emitter
            .simd_mov(
                SimdType::Store as i32 | type_,
                VectorRegister::VR2,
                mem_offset(ScratchRegister::R1, 230 - 1001),
            )
            .unwrap();

        emitter
            .mov(0, ScratchRegister::R0, (256 >> 3) as isize)
            .unwrap();
        emitter
            .mov(0, ScratchRegister::R1, (288 >> 3) as isize)
            .unwrap();
        type_ = SimdReg::Reg128 as i32
            | SimdElem::Elem64 as i32
            | sys::SLJIT_SIMD_FLOAT
            | SimdMemAlign::Aligned128 as i32;
        emitter
            .simd_mov(
                SimdType::Load as i32 | type_,
                VectorRegister::VR0,
                mem_indexed_shift(SavedRegister::S0, ScratchRegister::R0, 3),
            )
            .unwrap();
        /* buf[288] */
        emitter
            .simd_mov(
                SimdType::Store as i32 | type_,
                VectorRegister::VR0,
                mem_indexed_shift(SavedRegister::S0, ScratchRegister::R1, 3),
            )
            .unwrap();

        type_ = SimdReg::Reg128 as i32
            | SimdElem::Elem64 as i32
            | sys::SLJIT_SIMD_FLOAT
            | SimdMemAlign::Aligned64 as i32;
        emitter
            .simd_mov(
                SimdType::Load as i32 | type_,
                VectorRegister::VR2,
                mem_offset(SavedRegister::S0, 312),
            )
            .unwrap();
        /* buf[344] */
        emitter
            .simd_mov(
                SimdType::Store as i32 | type_,
                VectorRegister::VR2,
                mem_offset(SavedRegister::S0, 344),
            )
            .unwrap();

        type_ = SimdReg::Reg64 as i32 | SimdElem::Elem32 as i32 | SimdMemAlign::Aligned64 as i32;
        let res0 = emitter.simd_mov(
            SimdType::Load as i32 | type_,
            VectorRegister::VR4,
            mem_offset(SavedRegister::S0, 368),
        );
        let supported0 = res0.is_ok();
        /* buf[384] */
        let _ = emitter.simd_mov(
            SimdType::Store as i32 | type_,
            VectorRegister::VR4,
            mem_offset(SavedRegister::S0, 384),
        );

        emitter.mov(0, ScratchRegister::R0, 393).unwrap();
        emitter.mov(0, ScratchRegister::R1, 402).unwrap();
        type_ = SimdReg::Reg64 as i32 | SimdElem::Elem64 as i32 | SimdMemAlign::Unaligned as i32;
        let _ = emitter.simd_mov(
            SimdType::Load as i32 | type_,
            VectorRegister::VR0,
            mem_indexed(SavedRegister::S0, ScratchRegister::R0),
        );

        /* buf[402] */
        let _ = emitter.simd_mov(
            SimdType::Store as i32 | type_,
            VectorRegister::VR0,
            mem_indexed(SavedRegister::S0, ScratchRegister::R1),
        );

        type_ = SimdReg::Reg128 as i32 | SimdElem::Elem16 as i32 | SimdMemAlign::Aligned128 as i32;
        emitter
            .simd_mov(
                SimdType::Load as i32 | type_,
                VectorRegister::VR4,
                mem_offset(SavedRegister::S0, 416),
            )
            .unwrap();
        emitter
            .simd_mov(
                SimdType::Load as i32 | type_,
                VectorRegister::VR5,
                mem_offset(SavedRegister::S0, 432),
            )
            .unwrap();
        /* buf[464] */
        emitter
            .simd_mov(
                SimdType::Store as i32 | type_,
                VectorRegister::VR4,
                mem_offset(SavedRegister::S0, 464),
            )
            .unwrap();

        type_ = SimdReg::Reg128 as i32 | SimdElem::Elem16 as i32;
        emitter
            .simd_mov(
                SimdType::Load as i32 | type_,
                VectorRegister::VR3,
                mem_offset(SavedRegister::S0, 496),
            )
            .unwrap();
        emitter
            .simd_mov(
                SimdType::Load as i32 | type_,
                VectorRegister::VR4,
                mem_offset(SavedRegister::S0, 480),
            )
            .unwrap();
        emitter
            .simd_mov(
                SimdType::Load as i32 | type_,
                VectorRegister::VR4,
                VectorRegister::VR3,
            )
            .unwrap();
        /* buf[528] */
        emitter
            .simd_mov(
                SimdType::Store as i32 | type_,
                VectorRegister::VR4,
                mem_offset(SavedRegister::S0, 528),
            )
            .unwrap();

        emitter
            .simd_mov(
                SimdType::Load as i32 | type_,
                VectorRegister::VR2,
                mem_offset(SavedRegister::S0, 560),
            )
            .unwrap();
        emitter
            .simd_mov(
                SimdType::Load as i32 | type_,
                VectorRegister::VR0,
                mem_offset(SavedRegister::S0, 544),
            )
            .unwrap();
        emitter
            .simd_mov(
                SimdType::Store as i32 | type_,
                VectorRegister::VR2,
                VectorRegister::VR0,
            )
            .unwrap();
        /* buf[592] */
        emitter
            .simd_mov(
                SimdType::Store as i32 | type_,
                VectorRegister::VR0,
                mem_offset(SavedRegister::S0, 592),
            )
            .unwrap();

        type_ = SimdReg::Reg64 as i32 | SimdElem::Elem8 as i32;
        let _ = emitter.simd_mov(
            SimdType::Load as i32 | type_,
            VectorRegister::VR5,
            mem_offset(SavedRegister::S0, 616),
        );
        let _ = emitter.simd_mov(
            SimdType::Load as i32 | type_,
            VectorRegister::VR3,
            mem_offset(SavedRegister::S0, 608),
        );
        let _ = emitter.simd_mov(
            SimdType::Load as i32 | type_,
            VectorRegister::VR3,
            VectorRegister::VR5,
        );
        /* buf[632] */
        let _ = emitter.simd_mov(
            SimdType::Store as i32 | type_,
            VectorRegister::VR3,
            mem_offset(SavedRegister::S0, 632),
        );

        let _ = emitter.simd_mov(
            SimdType::Load as i32 | type_,
            VectorRegister::VR3,
            mem_offset(SavedRegister::S0, 648),
        );
        let _ = emitter.simd_mov(
            SimdType::Load as i32 | type_,
            vs0,
            mem_offset(SavedRegister::S0, 640),
        );
        let _ = emitter.simd_mov(SimdType::Store as i32 | type_, VectorRegister::VR3, vs0);
        /* buf[664] */
        let _ = emitter.simd_mov(
            SimdType::Store as i32 | type_,
            vs0,
            mem_offset(SavedRegister::S0, 664),
        );

        type_ = SimdReg::Reg256 as i32 | SimdElem::Elem32 as i32 | SimdMemAlign::Aligned256 as i32;
        let res1 = emitter.simd_mov(
            SimdType::Load as i32 | type_,
            VectorRegister::VR2,
            mem_offset(SavedRegister::S0, 704),
        );
        let supported1 = res1.is_ok();
        emitter
            .simd_mov(
                SimdType::Store as i32 | SimdReg::Reg256 as i32 | SimdElem::Elem32 as i32,
                VectorRegister::VR2,
                vs0,
            )
            .unwrap();
        emitter
            .mov(0, ScratchRegister::R1, SavedRegister::S0)
            .unwrap();
        emitter.mov(0, SavedRegister::S1, 384).unwrap();
        emitter
            .simd_mov(
                SimdType::Store as i32 | type_,
                vs0,
                mem_indexed_shift(ScratchRegister::R1, SavedRegister::S1, 1),
            )
            .unwrap();

        type_ = SimdReg::Reg256 as i32 | SimdElem::Elem16 as i32;
        emitter
            .add(0, ScratchRegister::R0, SavedRegister::S0, 801 - 32)
            .unwrap();
        emitter
            .simd_mov(
                SimdType::Load as i32 | type_,
                VectorRegister::VR0,
                mem_offset(ScratchRegister::R0, 32),
            )
            .unwrap();
        emitter
            .simd_mov(
                SimdType::Store as i32 | type_,
                VectorRegister::VR0,
                mem_sp(),
            )
            .unwrap();
        emitter.get_local_base(ScratchRegister::R1, 128).unwrap();
        emitter
            .simd_mov(
                SimdType::Load as i32 | type_,
                VectorRegister::VR3,
                mem_offset(ScratchRegister::R1, -128),
            )
            .unwrap();
        type_ = SimdReg::Reg256 as i32 | SimdElem::Elem16 as i32 | SimdMemAlign::Aligned16 as i32;
        emitter
            .simd_mov(
                SimdType::Store as i32 | type_,
                VectorRegister::VR3,
                mem_abs(buf_ptr.add(834) as isize),
            )
            .unwrap();

        emitter.return_void().unwrap();

        let code = compiler.generate_code();
        let func: fn(*mut u8) -> () = transmute(code.get());
        func(buf_ptr);

        assert!(check_simd_mov(
            &*core::slice::from_raw_parts(buf_ptr.add(32), 16),
            81,
            16
        ));
        assert!(check_simd_mov(
            &*core::slice::from_raw_parts(buf_ptr.add(82), 16),
            213,
            16
        ));
        assert!(check_simd_mov(
            &*core::slice::from_raw_parts(buf_ptr.add(136), 16),
            33,
            16
        ));
        assert!(check_simd_mov(
            &*core::slice::from_raw_parts(buf_ptr.add(192), 16),
            140,
            16
        ));
        assert!(check_simd_mov(
            &*core::slice::from_raw_parts(buf_ptr.add(230), 16),
            7,
            16
        ));
        assert!(check_simd_mov(
            &*core::slice::from_raw_parts(buf_ptr.add(288), 16),
            239,
            16
        ));
        assert!(check_simd_mov(
            &*core::slice::from_raw_parts(buf_ptr.add(344), 16),
            176,
            16
        ));

        if supported0 {
            assert!(check_simd_mov(
                &*core::slice::from_raw_parts(buf_ptr.add(384), 8),
                88,
                8
            ));
            assert!(check_simd_mov(
                &*core::slice::from_raw_parts(buf_ptr.add(402), 8),
                197,
                8
            ));
        }

        let expected_464 = if sys::sljit_has_cpu_feature(sys::SLJIT_SIMD_REGS_ARE_PAIRS as i32) != 0
        {
            203
        } else {
            58
        };
        assert!(check_simd_mov(
            &*core::slice::from_raw_parts(buf_ptr.add(464), 16),
            expected_464,
            16
        ));
        assert!(check_simd_mov(
            &*core::slice::from_raw_parts(buf_ptr.add(528), 16),
            105,
            16
        ));
        assert!(check_simd_mov(
            &*core::slice::from_raw_parts(buf_ptr.add(592), 16),
            19,
            16
        ));

        if supported0 {
            assert!(check_simd_mov(
                &*core::slice::from_raw_parts(buf_ptr.add(632), 8),
                202,
                8
            ));
            assert!(check_simd_mov(
                &*core::slice::from_raw_parts(buf_ptr.add(664), 8),
                123,
                8
            ));
        }

        if supported1 {
            assert!(check_simd_mov(
                &*core::slice::from_raw_parts(buf_ptr.add(768), 32),
                85,
                32
            ));
            assert!(check_simd_mov(
                &*core::slice::from_raw_parts(buf_ptr.add(834), 32),
                215,
                32
            ));
        }
    }
}

#[test]
fn test_add3_emitter() {
    unsafe {
        let mut compiler = Compiler::new();
        let mut emitter = Emitter::new(&mut compiler);
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

        let code = compiler.generate_code();
        let func: fn(c_int, c_int, c_int) -> c_int = transmute(code.get());
        assert_eq!(func(4, 5, 6), 4 + 5 + 6);
    }
}

#[test]
fn test_memory_access_emitter() {
    unsafe {
        let mut compiler = Compiler::new();
        let mut emitter = Emitter::new(&mut compiler);
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

        let code = compiler.generate_code();
        let data: [isize; 2] = [10, 20];
        let func: fn(p: *const isize) -> isize = transmute(code.get());
        assert_eq!(func(data.as_ptr()), 30);
    }
}

#[test]
fn test_add32_emitter() {
    unsafe {
        let mut compiler = Compiler::new();
        let mut emitter = Emitter::new(&mut compiler);
        emitter
            .emit_enter(0, arg_types!(32 -> [32, 32]), regs!(1), regs!(2), 0)
            .unwrap()
            .add32(0, ScratchRegister::R0, SavedRegister::S0, SavedRegister::S1)
            .unwrap();
        emitter
            .emit_return(ReturnOp::Mov32, ScratchRegister::R0)
            .unwrap();

        let code = compiler.generate_code();
        let func: fn(i32, i32) -> i32 = transmute(code.get());
        assert_eq!(func(10, 20), 30);
    }
}

#[test]
fn test_rotate_emitter() {
    unsafe {
        let mut compiler = Compiler::new();
        let mut emitter = Emitter::new(&mut compiler);
        emitter
            .emit_enter(0, arg_types!(W -> [W, W]), regs!(1), regs!(2), 0)
            .unwrap()
            .rotl(0, ScratchRegister::R0, SavedRegister::S0, SavedRegister::S1)
            .unwrap();
        emitter
            .emit_return(ReturnOp::Mov, ScratchRegister::R0)
            .unwrap();

        let code = compiler.generate_code();
        let func: fn(isize, isize) -> isize = transmute(code.get());
        assert_eq!(func(0b1011, 2), 0b101100);
    }
}

#[test]
fn test_add_f64_emitter() {
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

        let code = compiler.generate_code();
        let func: fn(f64, f64) -> f64 = transmute(code.get());
        assert_eq!(func(10.5, 20.25), 30.75);
    }
}

#[test]
fn test_conv_f64_from_s32_emitter() {
    unsafe {
        let mut compiler = Compiler::new();
        let mut emitter = Emitter::new(&mut compiler);
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

        let code = compiler.generate_code();
        let func: fn(i32) -> f64 = transmute(code.get());
        assert_eq!(func(42), 42.0);
    }
}

#[test]
fn test_branch_emitter() {
    unsafe {
        let mut compiler = Compiler::new();
        let mut emitter = Emitter::new(&mut compiler);
        emitter
            .emit_enter(0, arg_types!(W -> [W]), regs!(1), regs!(1), 0)
            .unwrap()
            .branch(
                Condition::Equal,
                SavedRegister::S0,
                0isize,
                |e| e.emit_return(ReturnOp::Mov, 10isize),
                |e| e.emit_return(ReturnOp::Mov, 20isize),
            )
            .unwrap();

        let code = compiler.generate_code();
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
        let mut compiler = Compiler::new();
        let mut emitter = Emitter::new(&mut compiler);
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

        let code = compiler.generate_code();
        let func: fn(*mut u8, *mut isize) = transmute(code.get());
        func(buf.as_mut_ptr(), data.as_mut_ptr());

        assert_eq!(buf, compare_buf);
    }
}

#[test]
fn test_fib_recursive() {
    unsafe {
        let mut compiler = Compiler::new();
        let mut emitter = Emitter::new(&mut compiler);

        // Function: fib(n) -> if n <= 1 return n else return fib(n-1) + fib(n-2)
        emitter
            .emit_enter(0, arg_types!(W -> [W]), regs!(3), regs!(1), 0)
            .unwrap();

        // Check base case: if n <= 1, return n
        emitter
            .branch(
                Condition::LessEqual,
                SavedRegister::S0,
                1isize,
                |e| {
                    // Base case: return n
                    e.emit_return(ReturnOp::Mov, SavedRegister::S0).unwrap();
                    Ok(())
                },
                |e| {
                    // Recursive case: fib(n-1) + fib(n-2)

                    // Save n to R0
                    e.mov(0, ScratchRegister::R0, SavedRegister::S0).unwrap();

                    // Call fib(n-1)
                    e.sub(0, ScratchRegister::R1, SavedRegister::S0, 1).unwrap();
                    e.mov(0, SavedRegister::S0, ScratchRegister::R1).unwrap();

                    let mut call1 = e.call(JumpType::Call, arg_types!([W])).unwrap();
                    call1.set_label(&mut e.put_label().unwrap());

                    // Save fib(n-1) result to R2
                    e.mov(0, ScratchRegister::R2, ScratchRegister::R0).unwrap();

                    // Call fib(n-2)
                    e.sub(0, ScratchRegister::R1, ScratchRegister::R0, 2)
                        .unwrap();
                    e.mov(0, SavedRegister::S0, ScratchRegister::R1).unwrap();

                    let mut call2 = e.call(JumpType::Call, arg_types!([W])).unwrap();
                    call2.set_label(&mut e.put_label().unwrap());

                    // Add fib(n-1) + fib(n-2)
                    e.add(
                        0,
                        ScratchRegister::R0,
                        ScratchRegister::R0,
                        ScratchRegister::R2,
                    )
                    .unwrap();

                    // Restore n
                    e.mov(0, SavedRegister::S0, ScratchRegister::R0).unwrap();

                    e.emit_return(ReturnOp::Mov, ScratchRegister::R0).unwrap();
                    Ok(())
                },
            )
            .unwrap();

        let code = compiler.generate_code();
        let func: fn(isize) -> isize = transmute(code.get());

        // Test cases
        assert_eq!(func(0), 0);
        assert_eq!(func(1), 1);
        assert_eq!(func(2), 1);
        assert_eq!(func(3), 2);
        assert_eq!(func(4), 3);
        assert_eq!(func(5), 5);
        assert_eq!(func(6), 8);
        assert_eq!(func(10), 55);
    }
}

#[test]
fn test_fib_iterative() {
    unsafe {
        let mut compiler = Compiler::new();
        let mut emitter = Emitter::new(&mut compiler);

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
                    e.emit_return(ReturnOp::Mov, SavedRegister::S0).unwrap();
                    Ok(())
                },
                |e| {
                    // Iterative case - run loop n-1 times

                    // R0 = a (starts as 0)
                    e.mov(0, ScratchRegister::R0, 0isize).unwrap();
                    // R1 = b (starts as 1)
                    e.mov(0, ScratchRegister::R1, 1isize).unwrap();
                    // R2 = counter (starts as n)
                    e.mov(0, ScratchRegister::R2, SavedRegister::S0).unwrap();

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
                    e.emit_return(ReturnOp::Mov, ScratchRegister::R1).unwrap();
                    Ok(())
                },
            )
            .unwrap();

        let code = compiler.generate_code();
        let func: fn(isize) -> isize = transmute(code.get());

        // Test cases
        assert_eq!(func(0), 0);
        assert_eq!(func(1), 1);
        assert_eq!(func(2), 2); // Was 1, now 2
        assert_eq!(func(3), 3); // Was 2, now 3
        assert_eq!(func(4), 5); // Was 3, now 5
        assert_eq!(func(5), 8); // Was 5, now 8
        assert_eq!(func(6), 13); // Was 8, now 13
        assert_eq!(func(10), 89); // Was 55, now 89
        assert_eq!(func(20), 10946); // Was 6765, now 10946
    }
}

#[test]
fn test_fib_iterative_while() {
    unsafe {
        let mut compiler = Compiler::new();
        let mut emitter = Emitter::new(&mut compiler);

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
                    e.emit_return(ReturnOp::Mov, SavedRegister::S0).unwrap();
                    Ok(())
                },
                |e| {
                    // Iterative case using the new loop API

                    // R0 = a (starts as 0)
                    e.mov(0, ScratchRegister::R0, 0isize).unwrap();
                    // R1 = b (starts as 1)
                    e.mov(0, ScratchRegister::R1, 1isize).unwrap();
                    // R2 = counter (starts as n)
                    e.mov(0, ScratchRegister::R2, SavedRegister::S0).unwrap();

                    // Use the ergonomic while_ API
                    e.while_(Condition::NotEqual, ScratchRegister::R2, 0isize, |e| {
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
                    e.emit_return(ReturnOp::Mov, ScratchRegister::R1).unwrap();
                    Ok(())
                },
            )
            .unwrap();

        let code = compiler.generate_code();
        let func: fn(isize) -> isize = transmute(code.get());

        // Test cases
        assert_eq!(func(0), 0);
        assert_eq!(func(1), 1);
        assert_eq!(func(2), 2); // Was 1, now 2
        assert_eq!(func(3), 3); // Was 2, now 3
        assert_eq!(func(4), 5); // Was 3, now 5
        assert_eq!(func(5), 8); // Was 5, now 8
        assert_eq!(func(6), 13); // Was 8, now 13
        assert_eq!(func(10), 89); // Was 55, now 89
        assert_eq!(func(20), 10946); // Was 6765, now 10946
    }
}

#[test]
fn test_fib_iterative_loop_break() {
    unsafe {
        let mut compiler = Compiler::new();
        let mut emitter = Emitter::new(&mut compiler);

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
                    e.emit_return(ReturnOp::Mov, SavedRegister::S0).unwrap();
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
                    e.loop_(|ctx| {
                        // Decrement counter
                        ctx.sub(0, ScratchRegister::R2, ScratchRegister::R2, 1)
                            .unwrap();
                        ctx.branch(
                            Condition::NotEqual,
                            ScratchRegister::R2,
                            0isize,
                            |ctx| {
                                // Loop body: a, b = b, a + b
                                ctx.add(
                                    0,
                                    ScratchRegister::R3,
                                    ScratchRegister::R0,
                                    ScratchRegister::R1,
                                )
                                .unwrap();
                                ctx.mov(0, ScratchRegister::R0, ScratchRegister::R1)
                                    .unwrap();
                                ctx.mov(0, ScratchRegister::R1, ScratchRegister::R3)
                                    .unwrap();
                                Ok(())
                            },
                            |ctx| ctx.break_(),
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

        let code = compiler.generate_code();
        let func: fn(isize) -> isize = transmute(code.get());

        // Test cases - should match test_fib_iterative
        assert_eq!(func(0), 0);
        assert_eq!(func(1), 1);
        assert_eq!(func(2), 1);
        assert_eq!(func(3), 2);
        assert_eq!(func(4), 3);
        assert_eq!(func(5), 5);
        assert_eq!(func(6), 8);
        assert_eq!(func(10), 55);
        assert_eq!(func(20), 6765);
    }
}
