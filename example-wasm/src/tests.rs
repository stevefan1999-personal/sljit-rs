use std::sync::Arc;

use crunchy::unroll;
use wasmparser::{Ieee32, Ieee64, Operator, ValType};

use crate::Engine;
use crate::Linker;
use crate::error::CompileError;
use crate::function::compile_simple;
use crate::module::Module;
use crate::store::{Func, Store};

/// A compiled WebAssembly module with its runtime context
/// This holds the Store and Instance needed to call functions
struct CompiledModule {
    #[allow(dead_code)]
    store: Store,
    func_code_ptr: usize,
}

impl CompiledModule {
    /// Get the function as a callable function pointer
    /// # Safety
    /// The caller must ensure the function signature matches the actual compiled function
    pub fn as_fn<F: Copy>(&self) -> F {
        unsafe { std::mem::transmute_copy(&self.func_code_ptr) }
    }
}

/// Helper function to compile from WAT (WebAssembly Text format)
/// Uses the full Engine/Store/Module/Instance/Linker infrastructure
/// Returns the exported "test" function ready to be called
fn compile_from_wat_with_signature(wat: &str) -> Result<CompiledModule, CompileError> {
    let wasm = wat::parse_str(wat).map_err(|e| CompileError::Parse(e.to_string()))?;

    // Create engine wrapped in Arc for shared ownership
    let engine = Arc::new(Engine::default());

    // Parse the module
    let module = Module::new(&engine, &wasm).map_err(|e| CompileError::Parse(e.to_string()))?;

    // Create store with a clone of the engine Arc
    let mut store = Store::new(engine.clone());

    // Create an empty linker (no imports needed for these tests)
    let linker = Linker::new();

    // Instantiate the module - this compiles all functions
    let instance = linker
        .instantiate(&mut store, &module)
        .map_err(|e| CompileError::Invalid(e.to_string()))?;

    // Get the exported "test" function
    let func_idx = instance
        .get_func(&store, "test")
        .ok_or_else(|| CompileError::Invalid("No 'test' function exported".into()))?;

    // Get the function's code pointer
    let func = store
        .func(func_idx)
        .ok_or_else(|| CompileError::Invalid("Function not found in store".into()))?;

    let code_ptr = match func {
        Func::Wasm(wf) => wf.code_ptr,
        Func::Host(_) => {
            return Err(CompileError::Invalid(
                "Expected wasm function, got host function".into(),
            ));
        }
    };

    if code_ptr == 0 {
        return Err(CompileError::Invalid(
            "Function has null code pointer".into(),
        ));
    }

    Ok(CompiledModule {
        store,
        func_code_ptr: code_ptr,
    })
}

#[test]
fn test_simple_add() {
    // Function: (param i32 i32) (result i32)
    // local.get 0
    // local.get 1
    // i32.add
    let body = [
        Operator::LocalGet { local_index: 0 },
        Operator::LocalGet { local_index: 1 },
        Operator::I32Add,
        Operator::End,
    ];

    let func = compile_simple(&[ValType::I32, ValType::I32], &[ValType::I32], &[], &body)
        .expect("Compilation failed");

    let f = func.as_fn::<fn(i32, i32) -> i32>();
    assert_eq!(f(10, 20), 30);
    assert_eq!(f(100, 200), 300);
    assert_eq!(f(-5, 10), 5);
}

#[test]
fn test_constant() {
    // Function: (result i32)
    // i32.const 42
    let body = [Operator::I32Const { value: 42 }, Operator::End];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn() -> i32>();
    assert_eq!(f(), 42);
}

#[test]
fn test_local_variable() {
    // Function: (param i32) (result i32) (local i32)
    // local.get 0
    // i32.const 10
    // i32.add
    // local.set 1
    // local.get 1
    let body = [
        Operator::LocalGet { local_index: 0 },
        Operator::I32Const { value: 10 },
        Operator::I32Add,
        Operator::LocalSet { local_index: 1 },
        Operator::LocalGet { local_index: 1 },
        Operator::End,
    ];

    let func = compile_simple(&[ValType::I32], &[ValType::I32], &[ValType::I32], &body)
        .expect("Compilation failed");

    let f = func.as_fn::<fn(i32) -> i32>();
    assert_eq!(f(5), 15);
    assert_eq!(f(100), 110);
}

#[test]
fn test_arithmetic() {
    // Function: (param i32 i32) (result i32)
    // local.get 0
    // local.get 1
    // i32.mul
    // i32.const 2
    // i32.add
    let body = [
        Operator::LocalGet { local_index: 0 },
        Operator::LocalGet { local_index: 1 },
        Operator::I32Mul,
        Operator::I32Const { value: 2 },
        Operator::I32Add,
        Operator::End,
    ];

    let func = compile_simple(&[ValType::I32, ValType::I32], &[ValType::I32], &[], &body)
        .expect("Compilation failed");

    let f = func.as_fn::<fn(i32, i32) -> i32>();
    assert_eq!(f(3, 4), 14); // 3 * 4 + 2
    assert_eq!(f(5, 6), 32); // 5 * 6 + 2
}

#[test]
fn test_comparison() {
    // Function: (param i32 i32) (result i32)
    // local.get 0
    // local.get 1
    // i32.lt_s
    let body = [
        Operator::LocalGet { local_index: 0 },
        Operator::LocalGet { local_index: 1 },
        Operator::I32LtS,
        Operator::End,
    ];

    let func = compile_simple(&[ValType::I32, ValType::I32], &[ValType::I32], &[], &body)
        .expect("Compilation failed");

    let f = func.as_fn::<fn(i32, i32) -> i32>();
    assert_eq!(f(5, 10), 1); // 5 < 10 is true
    assert_eq!(f(10, 5), 0); // 10 < 5 is false
    assert_eq!(f(5, 5), 0); // 5 < 5 is false
}

#[test]
fn test_simple_if() {
    // Function: (param i32) (result i32)
    // local.get 0
    // if (result i32)
    //   i32.const 1
    // else
    //   i32.const 0
    // end
    let body = [
        Operator::LocalGet { local_index: 0 },
        Operator::If {
            blockty: wasmparser::BlockType::Type(ValType::I32),
        },
        Operator::I32Const { value: 1 },
        Operator::Else,
        Operator::I32Const { value: 0 },
        Operator::End,
        Operator::End,
    ];

    let func =
        compile_simple(&[ValType::I32], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn(i32) -> i32>();
    assert_eq!(f(1), 1);
    assert_eq!(f(0), 0);
    assert_eq!(f(42), 1);
}

#[test]
fn test_block_br() {
    // Function: (result i32)
    // block
    //   i32.const 100
    //   br 0
    // end
    // (note: value is dropped since block has Empty type)
    let body = [
        Operator::Block {
            blockty: wasmparser::BlockType::Empty,
        },
        Operator::I32Const { value: 100 },
        Operator::Br { relative_depth: 0 },
        Operator::End,
        Operator::End,
    ];

    // This test just verifies block/br compiles correctly
    // The i32.const 100 value is pushed but the block has Empty type,
    // so the br jumps to end without producing a value
    let _func = compile_simple(&[], &[ValType::I32], &[], &body);
}

#[test]
fn test_division() {
    // Test i32.div_s: 20 / 3 = 6
    let body = [
        Operator::LocalGet { local_index: 0 },
        Operator::LocalGet { local_index: 1 },
        Operator::I32DivS,
        Operator::End,
    ];

    let func = compile_simple(&[ValType::I32, ValType::I32], &[ValType::I32], &[], &body)
        .expect("Compilation failed");

    let f = func.as_fn::<fn(i32, i32) -> i32>();
    assert_eq!(f(20, 3), 6);
    assert_eq!(f(100, 10), 10);
    assert_eq!(f(-20, 3), -6);
    assert_eq!(f(20, -3), -6);
}

#[test]
fn test_remainder() {
    // Test i32.rem_s: 20 % 3 = 2
    let body = [
        Operator::LocalGet { local_index: 0 },
        Operator::LocalGet { local_index: 1 },
        Operator::I32RemS,
        Operator::End,
    ];

    let func = compile_simple(&[ValType::I32, ValType::I32], &[ValType::I32], &[], &body)
        .expect("Compilation failed");

    let f = func.as_fn::<fn(i32, i32) -> i32>();
    assert_eq!(f(20, 3), 2);
    assert_eq!(f(100, 10), 0);
    assert_eq!(f(17, 5), 2);
}

#[test]
fn test_unsigned_division() {
    // Test i32.div_u
    let body = [
        Operator::LocalGet { local_index: 0 },
        Operator::LocalGet { local_index: 1 },
        Operator::I32DivU,
        Operator::End,
    ];

    let func = compile_simple(&[ValType::I32, ValType::I32], &[ValType::I32], &[], &body)
        .expect("Compilation failed");

    let f = func.as_fn::<fn(i32, i32) -> i32>();
    assert_eq!(f(20, 3), 6);
    assert_eq!(f(100, 10), 10);
    // Unsigned division treats negative numbers as large positive numbers
    // -1 as unsigned is 0xFFFFFFFF, divided by 2 is 0x7FFFFFFF
    assert_eq!(f(-1i32, 2), 0x7FFFFFFFi32);
}

#[test]
fn test_rotation() {
    // Test i32.rotl: rotate left
    let body = [
        Operator::LocalGet { local_index: 0 },
        Operator::LocalGet { local_index: 1 },
        Operator::I32Rotl,
        Operator::End,
    ];

    let func = compile_simple(&[ValType::I32, ValType::I32], &[ValType::I32], &[], &body)
        .expect("Compilation failed");

    let f = func.as_fn::<fn(i32, i32) -> i32>();
    // 0x12345678 rotated left by 4 bits = 0x23456781
    assert_eq!(f(0x12345678u32 as i32, 4), 0x23456781u32 as i32);
    // 1 rotated left by 1 = 2
    assert_eq!(f(1, 1), 2);
    // 0x80000000 rotated left by 1 = 1
    assert_eq!(f(0x80000000u32 as i32, 1), 1);
}

#[test]
fn test_rotation_right() {
    // Test i32.rotr: rotate right
    let body = [
        Operator::LocalGet { local_index: 0 },
        Operator::LocalGet { local_index: 1 },
        Operator::I32Rotr,
        Operator::End,
    ];

    let func = compile_simple(&[ValType::I32, ValType::I32], &[ValType::I32], &[], &body)
        .expect("Compilation failed");

    let f = func.as_fn::<fn(i32, i32) -> i32>();
    // 0x12345678 rotated right by 4 bits = 0x81234567
    assert_eq!(f(0x12345678u32 as i32, 4), 0x81234567u32 as i32);
    // 1 rotated right by 1 = 0x80000000
    assert_eq!(f(1, 1), 0x80000000u32 as i32);
}

#[test]
fn test_clz() {
    // Test i32.clz: count leading zeros
    let body = [
        Operator::LocalGet { local_index: 0 },
        Operator::I32Clz,
        Operator::End,
    ];

    let func =
        compile_simple(&[ValType::I32], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn(i32) -> i32>();
    assert_eq!(f(0), 32);
    assert_eq!(f(1), 31);
    assert_eq!(f(0x80000000u32 as i32), 0);
    assert_eq!(f(0x00008000), 16);
    assert_eq!(f(0x00000100), 23);
}

#[test]
fn test_ctz() {
    // Test i32.ctz: count trailing zeros
    let body = [
        Operator::LocalGet { local_index: 0 },
        Operator::I32Ctz,
        Operator::End,
    ];

    let func =
        compile_simple(&[ValType::I32], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn(i32) -> i32>();
    assert_eq!(f(0), 32);
    assert_eq!(f(1), 0);
    assert_eq!(f(2), 1);
    assert_eq!(f(0x80000000u32 as i32), 31);
    assert_eq!(f(0x00008000), 15);
}

#[test]
fn test_popcnt() {
    // Test i32.popcnt: population count (number of 1 bits)
    let body = [
        Operator::LocalGet { local_index: 0 },
        Operator::I32Popcnt,
        Operator::End,
    ];

    let func =
        compile_simple(&[ValType::I32], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn(i32) -> i32>();
    assert_eq!(f(0), 0);
    assert_eq!(f(1), 1);
    assert_eq!(f(0xFF), 8);
    assert_eq!(f(0xFFFFFFFFu32 as i32), 32);
    assert_eq!(f(0x55555555), 16);
    assert_eq!(f(0xAAAAAAAAu32 as i32), 16);
}

#[test]
fn test_bitwise_operations() {
    // Test i32.and
    let body = [
        Operator::LocalGet { local_index: 0 },
        Operator::LocalGet { local_index: 1 },
        Operator::I32And,
        Operator::End,
    ];

    let func = compile_simple(&[ValType::I32, ValType::I32], &[ValType::I32], &[], &body)
        .expect("Compilation failed");

    let f = func.as_fn::<fn(i32, i32) -> i32>();
    assert_eq!(f(0xFF00, 0x0FF0), 0x0F00);
    assert_eq!(f(-1, 0x12345678), 0x12345678);
}

#[test]
fn test_shift_operations() {
    // Test i32.shl
    let body = [
        Operator::LocalGet { local_index: 0 },
        Operator::LocalGet { local_index: 1 },
        Operator::I32Shl,
        Operator::End,
    ];

    let func_shl = compile_simple(&[ValType::I32, ValType::I32], &[ValType::I32], &[], &body)
        .expect("Compilation failed");

    let f_shl = func_shl.as_fn::<fn(i32, i32) -> i32>();
    assert_eq!(f_shl(1, 4), 16);
    assert_eq!(f_shl(0xFF, 8), 0xFF00);

    // Test i32.shr_s (arithmetic shift)
    let body = [
        Operator::LocalGet { local_index: 0 },
        Operator::LocalGet { local_index: 1 },
        Operator::I32ShrS,
        Operator::End,
    ];

    let func_shrs = compile_simple(&[ValType::I32, ValType::I32], &[ValType::I32], &[], &body)
        .expect("Compilation failed");

    let f_shrs = func_shrs.as_fn::<fn(i32, i32) -> i32>();
    assert_eq!(f_shrs(16, 2), 4);
    assert_eq!(f_shrs(-16, 2), -4); // Arithmetic shift preserves sign

    // Test i32.shr_u (logical shift)
    let body = [
        Operator::LocalGet { local_index: 0 },
        Operator::LocalGet { local_index: 1 },
        Operator::I32ShrU,
        Operator::End,
    ];

    let func_shru = compile_simple(&[ValType::I32, ValType::I32], &[ValType::I32], &[], &body)
        .expect("Compilation failed");

    let f_shru = func_shru.as_fn::<fn(i32, i32) -> i32>();
    assert_eq!(f_shru(16, 2), 4);
    // Logical shift fills with zeros
    assert_eq!(f_shru(-1i32, 1), 0x7FFFFFFFi32);
}

#[test]
fn test_i64_basic() {
    // Test i64.const and basic operations on 64-bit platforms
    let body = [
        Operator::I64Const {
            value: 0x123456789ABCDEFi64,
        },
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn() -> i32>();
    // Wrap truncates to lower 32 bits: 0x89ABCDEF
    assert_eq!(f(), 0x89ABCDEFu32 as i32);
}

#[test]
fn test_eqz() {
    // Test i32.eqz
    let body = [
        Operator::LocalGet { local_index: 0 },
        Operator::I32Eqz,
        Operator::End,
    ];

    let func =
        compile_simple(&[ValType::I32], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn(i32) -> i32>();
    assert_eq!(f(0), 1); // 0 == 0 is true
    assert_eq!(f(1), 0); // 1 == 0 is false
    assert_eq!(f(-1), 0); // -1 == 0 is false
    assert_eq!(f(42), 0); // 42 == 0 is false
}

#[test]
fn test_select() {
    // Test select: if cond then val1 else val2
    let body = [
        Operator::I32Const { value: 10 },      // val1
        Operator::I32Const { value: 20 },      // val2
        Operator::LocalGet { local_index: 0 }, // cond
        Operator::Select,
        Operator::End,
    ];

    let func =
        compile_simple(&[ValType::I32], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn(i32) -> i32>();
    assert_eq!(f(1), 10); // cond != 0, select val1
    assert_eq!(f(0), 20); // cond == 0, select val2
    assert_eq!(f(42), 10); // cond != 0, select val1
}

#[test]
fn test_subtraction() {
    // Test i32.sub
    let body = [
        Operator::LocalGet { local_index: 0 },
        Operator::LocalGet { local_index: 1 },
        Operator::I32Sub,
        Operator::End,
    ];

    let func = compile_simple(&[ValType::I32, ValType::I32], &[ValType::I32], &[], &body)
        .expect("Compilation failed");

    let f = func.as_fn::<fn(i32, i32) -> i32>();
    assert_eq!(f(20, 8), 12);
    assert_eq!(f(5, 10), -5);
    assert_eq!(f(-5, -10), 5);
}

#[test]
fn test_or_operation() {
    // Test i32.or
    let body = [
        Operator::LocalGet { local_index: 0 },
        Operator::LocalGet { local_index: 1 },
        Operator::I32Or,
        Operator::End,
    ];

    let func = compile_simple(&[ValType::I32, ValType::I32], &[ValType::I32], &[], &body)
        .expect("Compilation failed");

    let f = func.as_fn::<fn(i32, i32) -> i32>();
    assert_eq!(f(0xF0, 0x0F), 0xFF);
    assert_eq!(f(0xFF00, 0x00FF), 0xFFFF);
    assert_eq!(f(0, 0x12345678), 0x12345678);
}

#[test]
fn test_xor_operation() {
    // Test i32.xor
    let body = [
        Operator::LocalGet { local_index: 0 },
        Operator::LocalGet { local_index: 1 },
        Operator::I32Xor,
        Operator::End,
    ];

    let func = compile_simple(&[ValType::I32, ValType::I32], &[ValType::I32], &[], &body)
        .expect("Compilation failed");

    let f = func.as_fn::<fn(i32, i32) -> i32>();
    assert_eq!(f(0xFF, 0xFF), 0);
    assert_eq!(f(0xFFFF0000u32 as i32, 0x0000FFFFu32 as i32), -1);
    assert_eq!(f(0x12345678, 0), 0x12345678);
}

#[test]
fn test_unsigned_remainder() {
    // Test i32.rem_u
    let body = [
        Operator::LocalGet { local_index: 0 },
        Operator::LocalGet { local_index: 1 },
        Operator::I32RemU,
        Operator::End,
    ];

    let func = compile_simple(&[ValType::I32, ValType::I32], &[ValType::I32], &[], &body)
        .expect("Compilation failed");

    let f = func.as_fn::<fn(i32, i32) -> i32>();
    assert_eq!(f(20, 3), 2);
    assert_eq!(f(100, 7), 2);
    // Unsigned remainder treats negative numbers as large positive numbers
    // -1 as unsigned is 0xFFFFFFFF, mod 2 is 1
    assert_eq!(f(-1i32, 2), 1);
}

#[test]
fn test_eq_comparison() {
    // Test i32.eq
    let body = [
        Operator::LocalGet { local_index: 0 },
        Operator::LocalGet { local_index: 1 },
        Operator::I32Eq,
        Operator::End,
    ];

    let func = compile_simple(&[ValType::I32, ValType::I32], &[ValType::I32], &[], &body)
        .expect("Compilation failed");

    let f = func.as_fn::<fn(i32, i32) -> i32>();
    assert_eq!(f(5, 5), 1);
    assert_eq!(f(5, 10), 0);
    assert_eq!(f(-1, -1), 1);
    assert_eq!(f(0, 0), 1);
}

#[test]
fn test_ne_comparison() {
    // Test i32.ne
    let body = [
        Operator::LocalGet { local_index: 0 },
        Operator::LocalGet { local_index: 1 },
        Operator::I32Ne,
        Operator::End,
    ];

    let func = compile_simple(&[ValType::I32, ValType::I32], &[ValType::I32], &[], &body)
        .expect("Compilation failed");

    let f = func.as_fn::<fn(i32, i32) -> i32>();
    assert_eq!(f(5, 5), 0);
    assert_eq!(f(5, 10), 1);
    assert_eq!(f(-1, 1), 1);
}

#[test]
fn test_lt_u_comparison() {
    // Test i32.lt_u (unsigned less than)
    let body = [
        Operator::LocalGet { local_index: 0 },
        Operator::LocalGet { local_index: 1 },
        Operator::I32LtU,
        Operator::End,
    ];

    let func = compile_simple(&[ValType::I32, ValType::I32], &[ValType::I32], &[], &body)
        .expect("Compilation failed");

    let f = func.as_fn::<fn(i32, i32) -> i32>();
    assert_eq!(f(5, 10), 1);
    assert_eq!(f(10, 5), 0);
    // -1 as unsigned is larger than any positive number
    assert_eq!(f(-1i32, 1), 0);
    assert_eq!(f(1, -1i32), 1);
}

#[test]
fn test_gt_s_comparison() {
    // Test i32.gt_s (signed greater than)
    let body = [
        Operator::LocalGet { local_index: 0 },
        Operator::LocalGet { local_index: 1 },
        Operator::I32GtS,
        Operator::End,
    ];

    let func = compile_simple(&[ValType::I32, ValType::I32], &[ValType::I32], &[], &body)
        .expect("Compilation failed");

    let f = func.as_fn::<fn(i32, i32) -> i32>();
    assert_eq!(f(10, 5), 1);
    assert_eq!(f(5, 10), 0);
    assert_eq!(f(-1, -2), 1);
    assert_eq!(f(-2, -1), 0);
}

#[test]
fn test_gt_u_comparison() {
    // Test i32.gt_u (unsigned greater than)
    let body = [
        Operator::LocalGet { local_index: 0 },
        Operator::LocalGet { local_index: 1 },
        Operator::I32GtU,
        Operator::End,
    ];

    let func = compile_simple(&[ValType::I32, ValType::I32], &[ValType::I32], &[], &body)
        .expect("Compilation failed");

    let f = func.as_fn::<fn(i32, i32) -> i32>();
    assert_eq!(f(10, 5), 1);
    assert_eq!(f(5, 10), 0);
    // -1 as unsigned is larger than any positive number
    assert_eq!(f(-1i32, 1), 1);
    assert_eq!(f(1, -1i32), 0);
}

#[test]
fn test_le_s_comparison() {
    // Test i32.le_s (signed less than or equal)
    let body = [
        Operator::LocalGet { local_index: 0 },
        Operator::LocalGet { local_index: 1 },
        Operator::I32LeS,
        Operator::End,
    ];

    let func = compile_simple(&[ValType::I32, ValType::I32], &[ValType::I32], &[], &body)
        .expect("Compilation failed");

    let f = func.as_fn::<fn(i32, i32) -> i32>();
    assert_eq!(f(5, 10), 1);
    assert_eq!(f(10, 5), 0);
    assert_eq!(f(5, 5), 1);
    assert_eq!(f(-1, -1), 1);
}

#[test]
fn test_le_u_comparison() {
    // Test i32.le_u (unsigned less than or equal)
    let body = [
        Operator::LocalGet { local_index: 0 },
        Operator::LocalGet { local_index: 1 },
        Operator::I32LeU,
        Operator::End,
    ];

    let func = compile_simple(&[ValType::I32, ValType::I32], &[ValType::I32], &[], &body)
        .expect("Compilation failed");

    let f = func.as_fn::<fn(i32, i32) -> i32>();
    assert_eq!(f(5, 10), 1);
    assert_eq!(f(10, 5), 0);
    assert_eq!(f(5, 5), 1);
    assert_eq!(f(1, -1i32), 1); // 1 <= 0xFFFFFFFF
}

#[test]
fn test_ge_s_comparison() {
    // Test i32.ge_s (signed greater than or equal)
    let body = [
        Operator::LocalGet { local_index: 0 },
        Operator::LocalGet { local_index: 1 },
        Operator::I32GeS,
        Operator::End,
    ];

    let func = compile_simple(&[ValType::I32, ValType::I32], &[ValType::I32], &[], &body)
        .expect("Compilation failed");

    let f = func.as_fn::<fn(i32, i32) -> i32>();
    assert_eq!(f(10, 5), 1);
    assert_eq!(f(5, 10), 0);
    assert_eq!(f(5, 5), 1);
    assert_eq!(f(-1, -1), 1);
}

#[test]
fn test_ge_u_comparison() {
    // Test i32.ge_u (unsigned greater than or equal)
    let body = [
        Operator::LocalGet { local_index: 0 },
        Operator::LocalGet { local_index: 1 },
        Operator::I32GeU,
        Operator::End,
    ];

    let func = compile_simple(&[ValType::I32, ValType::I32], &[ValType::I32], &[], &body)
        .expect("Compilation failed");

    let f = func.as_fn::<fn(i32, i32) -> i32>();
    assert_eq!(f(10, 5), 1);
    assert_eq!(f(5, 10), 0);
    assert_eq!(f(5, 5), 1);
    assert_eq!(f(-1i32, 1), 1); // 0xFFFFFFFF >= 1
}

#[test]
fn test_loop_br() {
    // Test loop with br to loop start
    // Function: (param i32) (result i32)
    // Computes sum from 0 to n-1: while (n > 0) { sum += --n; }
    //
    // loop $L
    //   local.get 0   ;; get n
    //   i32.eqz
    //   br_if 1       ;; if n == 0, exit to outer block
    //   local.get 0   ;; get n
    //   i32.const 1
    //   i32.sub       ;; n - 1
    //   local.tee 0   ;; n = n - 1, leave on stack
    //   local.get 1   ;; get sum
    //   i32.add       ;; n + sum
    //   local.set 1   ;; sum = n + sum
    //   br 0          ;; continue loop
    // end
    let body = [
        Operator::Block {
            blockty: wasmparser::BlockType::Empty,
        },
        Operator::Loop {
            blockty: wasmparser::BlockType::Empty,
        },
        Operator::LocalGet { local_index: 0 },
        Operator::I32Eqz,
        Operator::BrIf { relative_depth: 1 },
        Operator::LocalGet { local_index: 0 },
        Operator::I32Const { value: 1 },
        Operator::I32Sub,
        Operator::LocalTee { local_index: 0 },
        Operator::LocalGet { local_index: 1 },
        Operator::I32Add,
        Operator::LocalSet { local_index: 1 },
        Operator::Br { relative_depth: 0 },
        Operator::End,
        Operator::End,
        Operator::LocalGet { local_index: 1 },
        Operator::End,
    ];

    let func = compile_simple(
        &[ValType::I32],
        &[ValType::I32],
        &[ValType::I32], // local for sum
        &body,
    )
    .expect("Compilation failed");

    let f = func.as_fn::<fn(i32) -> i32>();
    assert_eq!(f(0), 0); // sum of nothing
    assert_eq!(f(1), 0); // 0
    assert_eq!(f(5), 10); // 0+1+2+3+4 = 10
    assert_eq!(f(10), 45); // 0+1+2+...+9 = 45
}

#[test]
fn test_drop() {
    // Test drop: discards value from stack
    let body = [
        Operator::I32Const { value: 100 },
        Operator::I32Const { value: 42 },
        Operator::Drop, // drop 42
        Operator::End,  // return 100
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn() -> i32>();
    assert_eq!(f(), 100);
}

#[test]
fn test_nop() {
    // Test nop: does nothing
    let body = [
        Operator::Nop,
        Operator::I32Const { value: 42 },
        Operator::Nop,
        Operator::Nop,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn() -> i32>();
    assert_eq!(f(), 42);
}

#[test]
fn test_local_tee() {
    // Test local.tee: sets local AND leaves value on stack
    // result = (x + 10) where local[1] = x + 10
    let body = [
        Operator::LocalGet { local_index: 0 },
        Operator::I32Const { value: 10 },
        Operator::I32Add,
        Operator::LocalTee { local_index: 1 }, // sets local[1], leaves value
        Operator::End,
    ];

    let func = compile_simple(&[ValType::I32], &[ValType::I32], &[ValType::I32], &body)
        .expect("Compilation failed");

    let f = func.as_fn::<fn(i32) -> i32>();
    assert_eq!(f(5), 15);
    assert_eq!(f(0), 10);
}

#[test]
fn test_return_early() {
    // Test return: exits function early
    let body = [
        Operator::LocalGet { local_index: 0 },
        Operator::I32Eqz,
        Operator::If {
            blockty: wasmparser::BlockType::Empty,
        },
        Operator::I32Const { value: 0 },
        Operator::Return, // early return if param == 0
        Operator::End,
        Operator::I32Const { value: 42 }, // normal case
        Operator::End,
    ];

    let func =
        compile_simple(&[ValType::I32], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn(i32) -> i32>();
    assert_eq!(f(0), 0); // early return
    assert_eq!(f(1), 42); // normal path
    assert_eq!(f(5), 42);
}

#[test]
fn test_br_if() {
    // Test br_if: conditional branch
    // if (x > 5) return 1 else return 0
    let body = [
        Operator::Block {
            blockty: wasmparser::BlockType::Type(ValType::I32),
        },
        Operator::LocalGet { local_index: 0 },
        Operator::I32Const { value: 5 },
        Operator::I32GtS,
        Operator::If {
            blockty: wasmparser::BlockType::Empty,
        },
        Operator::I32Const { value: 1 },
        Operator::Br { relative_depth: 1 }, // exit block with 1
        Operator::End,
        Operator::I32Const { value: 0 }, // else case
        Operator::End,
        Operator::End,
    ];

    let func =
        compile_simple(&[ValType::I32], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn(i32) -> i32>();
    assert_eq!(f(10), 1);
    assert_eq!(f(5), 0);
    assert_eq!(f(0), 0);
}

#[test]
fn test_i64_arithmetic() {
    // Test i64 add and sub
    let body = [
        Operator::LocalGet { local_index: 0 },
        Operator::I64ExtendI32S,
        Operator::LocalGet { local_index: 1 },
        Operator::I64ExtendI32S,
        Operator::I64Add,
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func_add = compile_simple(&[ValType::I32, ValType::I32], &[ValType::I32], &[], &body)
        .expect("Compilation failed");

    let f_add = func_add.as_fn::<fn(i32, i32) -> i32>();
    assert_eq!(f_add(10, 20), 30);
    assert_eq!(f_add(-5, 10), 5);

    // Test i64 sub
    let body = [
        Operator::LocalGet { local_index: 0 },
        Operator::I64ExtendI32S,
        Operator::LocalGet { local_index: 1 },
        Operator::I64ExtendI32S,
        Operator::I64Sub,
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func_sub = compile_simple(&[ValType::I32, ValType::I32], &[ValType::I32], &[], &body)
        .expect("Compilation failed");

    let f_sub = func_sub.as_fn::<fn(i32, i32) -> i32>();
    assert_eq!(f_sub(20, 8), 12);
    assert_eq!(f_sub(5, 10), -5);
}

#[test]
fn test_i64_bitwise() {
    // Test i64 and, or, xor
    let body = [
        Operator::I64Const { value: 0xFF00FF00 },
        Operator::I64Const { value: 0x0FF00FF0 },
        Operator::I64And,
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn() -> i32>();
    assert_eq!(f(), 0x0F000F00u32 as i32);

    let body = [
        Operator::I64Const { value: 0xFF00 },
        Operator::I64Const { value: 0x00FF },
        Operator::I64Or,
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func_or = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f_or = func_or.as_fn::<fn() -> i32>();
    assert_eq!(f_or(), 0xFFFF);

    let body = [
        Operator::I64Const { value: 0xFFFF },
        Operator::I64Const { value: 0x0F0F },
        Operator::I64Xor,
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func_xor = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f_xor = func_xor.as_fn::<fn() -> i32>();
    assert_eq!(f_xor(), 0xF0F0);
}

#[test]
fn test_i64_shifts() {
    // Test i64 shl, shr_s, shr_u
    let body = [
        Operator::I64Const { value: 1 },
        Operator::I64Const { value: 32 },
        Operator::I64Shl,
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func_shl = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f_shl = func_shl.as_fn::<fn() -> i32>();
    // 1 << 32 = 0x100000000, wrapped to 32 bits = 0
    assert_eq!(f_shl(), 0);

    // Test shr_s
    let body = [
        Operator::I64Const { value: -16 },
        Operator::I64Const { value: 2 },
        Operator::I64ShrS,
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func_shrs = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f_shrs = func_shrs.as_fn::<fn() -> i32>();
    assert_eq!(f_shrs(), -4);

    // Test shr_u
    let body = [
        Operator::I64Const { value: 16 },
        Operator::I64Const { value: 2 },
        Operator::I64ShrU,
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func_shru = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f_shru = func_shru.as_fn::<fn() -> i32>();
    assert_eq!(f_shru(), 4);
}

#[test]
fn test_i64_extend() {
    // Test i64.extend_i32_s: sign extension should set all upper bits for -1
    // To verify: extend -1, shift right 32 bits, wrap to i32
    // If sign extended: 0xFFFFFFFFFFFFFFFF >> 32 = 0xFFFFFFFF = -1
    // If not: 0x00000000FFFFFFFF >> 32 = 0x00000000 = 0
    let body = [
        Operator::I32Const { value: -1 },
        Operator::I64ExtendI32S,
        Operator::I64Const { value: 32 },
        Operator::I64ShrU, // logical shift to get upper bits
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func_s = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f_s = func_s.as_fn::<fn() -> i32>();
    // -1 sign extended: upper 32 bits should be 0xFFFFFFFF = -1
    assert_eq!(f_s(), -1);

    // Test i64.extend_i32_u: zero extension should clear all upper bits
    let body = [
        Operator::I32Const { value: -1 },
        Operator::I64ExtendI32U,
        Operator::I64Const { value: 32 },
        Operator::I64ShrU, // logical shift to get upper bits
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func_u = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f_u = func_u.as_fn::<fn() -> i32>();
    // -1 zero extended: upper 32 bits should be 0
    assert_eq!(f_u(), 0);
}

// Memory store/load tests

#[test]
fn test_memory_i32_store_load() {
    // Function: (param i32) (result i32)
    // memory_base is passed as first param (in S0)
    // i32.const 0      ;; offset
    // i32.const 42     ;; value to store
    // i32.store        ;; store 42 at offset 0
    // i32.const 0      ;; offset
    // i32.load         ;; load from offset 0
    let body = [
        Operator::I32Const { value: 0 },  // offset
        Operator::I32Const { value: 42 }, // value
        Operator::I32Store {
            memarg: wasmparser::MemArg {
                align: 2,
                max_align: 2,
                offset: 0,
                memory: 0,
            },
        },
        Operator::I32Const { value: 0 }, // offset
        Operator::I32Load {
            memarg: wasmparser::MemArg {
                align: 2,
                max_align: 2,
                offset: 0,
                memory: 0,
            },
        },
        Operator::End,
    ];

    let func =
        compile_simple(&[ValType::I32], &[ValType::I32], &[], &body).expect("Compilation failed");

    // Create a memory buffer
    let mut memory = [0u8; 64];
    let memory_base = memory.as_mut_ptr() as i32;

    let f = func.as_fn::<fn(i32) -> i32>();
    let result = f(memory_base);
    assert_eq!(result, 42);
}

#[test]
fn test_memory_i32_store_load_with_offset() {
    // Store at offset 8, load from offset 8
    let body = [
        Operator::I32Const { value: 0 },   // base offset
        Operator::I32Const { value: 123 }, // value
        Operator::I32Store {
            memarg: wasmparser::MemArg {
                align: 2,
                max_align: 2,
                offset: 8, // static offset
                memory: 0,
            },
        },
        Operator::I32Const { value: 0 }, // base offset
        Operator::I32Load {
            memarg: wasmparser::MemArg {
                align: 2,
                max_align: 2,
                offset: 8, // static offset
                memory: 0,
            },
        },
        Operator::End,
    ];

    let func =
        compile_simple(&[ValType::I32], &[ValType::I32], &[], &body).expect("Compilation failed");

    let mut memory = [0u8; 64];
    let memory_base = memory.as_mut_ptr() as i32;

    let f = func.as_fn::<fn(i32) -> i32>();
    let result = f(memory_base);
    assert_eq!(result, 123);
}

#[test]
fn test_memory_i8_store_load() {
    // Test I32Store8 and I32Load8U / I32Load8S
    // Store 0xFF (-1 as signed byte), load unsigned and signed
    let body = [
        Operator::I32Const { value: 0 },
        Operator::I32Const { value: 0xFF }, // 255 or -1 as signed
        Operator::I32Store8 {
            memarg: wasmparser::MemArg {
                align: 0,
                max_align: 0,
                offset: 0,
                memory: 0,
            },
        },
        Operator::I32Const { value: 0 },
        Operator::I32Load8U {
            memarg: wasmparser::MemArg {
                align: 0,
                max_align: 0,
                offset: 0,
                memory: 0,
            },
        },
        Operator::End,
    ];

    let func_u =
        compile_simple(&[ValType::I32], &[ValType::I32], &[], &body).expect("Compilation failed");

    let mut memory = [0u8; 64];
    let memory_base = memory.as_mut_ptr() as i32;

    let f_u = func_u.as_fn::<fn(i32) -> i32>();
    assert_eq!(f_u(memory_base), 255); // Unsigned load: 0xFF = 255

    // Test signed load
    let body = [
        Operator::I32Const { value: 0 },
        Operator::I32Const { value: 0xFF },
        Operator::I32Store8 {
            memarg: wasmparser::MemArg {
                align: 0,
                max_align: 0,
                offset: 0,
                memory: 0,
            },
        },
        Operator::I32Const { value: 0 },
        Operator::I32Load8S {
            memarg: wasmparser::MemArg {
                align: 0,
                max_align: 0,
                offset: 0,
                memory: 0,
            },
        },
        Operator::End,
    ];

    let func_s =
        compile_simple(&[ValType::I32], &[ValType::I32], &[], &body).expect("Compilation failed");

    let mut memory2 = [0u8; 64];
    let memory_base2 = memory2.as_mut_ptr() as i32;

    let f_s = func_s.as_fn::<fn(i32) -> i32>();
    assert_eq!(f_s(memory_base2), -1); // Signed load: 0xFF sign-extended = -1
}

#[test]
fn test_memory_i16_store_load() {
    // Test I32Store16 and I32Load16U / I32Load16S
    let body = [
        Operator::I32Const { value: 0 },
        Operator::I32Const { value: 0xFFFF }, // 65535 or -1 as signed i16
        Operator::I32Store16 {
            memarg: wasmparser::MemArg {
                align: 1,
                max_align: 1,
                offset: 0,
                memory: 0,
            },
        },
        Operator::I32Const { value: 0 },
        Operator::I32Load16U {
            memarg: wasmparser::MemArg {
                align: 1,
                max_align: 1,
                offset: 0,
                memory: 0,
            },
        },
        Operator::End,
    ];

    let func_u =
        compile_simple(&[ValType::I32], &[ValType::I32], &[], &body).expect("Compilation failed");

    let mut memory = [0u8; 64];
    let memory_base = memory.as_mut_ptr() as i32;

    let f_u = func_u.as_fn::<fn(i32) -> i32>();
    assert_eq!(f_u(memory_base), 65535); // Unsigned load: 0xFFFF = 65535

    // Test signed load
    let body = [
        Operator::I32Const { value: 0 },
        Operator::I32Const { value: 0xFFFF },
        Operator::I32Store16 {
            memarg: wasmparser::MemArg {
                align: 1,
                max_align: 1,
                offset: 0,
                memory: 0,
            },
        },
        Operator::I32Const { value: 0 },
        Operator::I32Load16S {
            memarg: wasmparser::MemArg {
                align: 1,
                max_align: 1,
                offset: 0,
                memory: 0,
            },
        },
        Operator::End,
    ];

    let func_s =
        compile_simple(&[ValType::I32], &[ValType::I32], &[], &body).expect("Compilation failed");

    let mut memory2 = [0u8; 64];
    let memory_base2 = memory2.as_mut_ptr() as i32;

    let f_s = func_s.as_fn::<fn(i32) -> i32>();
    assert_eq!(f_s(memory_base2), -1); // Signed load: 0xFFFF sign-extended = -1
}

#[test]
fn test_memory_multiple_stores_loads() {
    // Store multiple values at different offsets and load them back
    let body = [
        // Store 10 at offset 0
        Operator::I32Const { value: 0 },
        Operator::I32Const { value: 10 },
        Operator::I32Store {
            memarg: wasmparser::MemArg {
                align: 2,
                max_align: 2,
                offset: 0,
                memory: 0,
            },
        },
        // Store 20 at offset 4
        Operator::I32Const { value: 4 },
        Operator::I32Const { value: 20 },
        Operator::I32Store {
            memarg: wasmparser::MemArg {
                align: 2,
                max_align: 2,
                offset: 0,
                memory: 0,
            },
        },
        // Store 30 at offset 8
        Operator::I32Const { value: 8 },
        Operator::I32Const { value: 30 },
        Operator::I32Store {
            memarg: wasmparser::MemArg {
                align: 2,
                max_align: 2,
                offset: 0,
                memory: 0,
            },
        },
        // Load from offset 0, 4, 8 and add them: 10 + 20 + 30 = 60
        Operator::I32Const { value: 0 },
        Operator::I32Load {
            memarg: wasmparser::MemArg {
                align: 2,
                max_align: 2,
                offset: 0,
                memory: 0,
            },
        },
        Operator::I32Const { value: 4 },
        Operator::I32Load {
            memarg: wasmparser::MemArg {
                align: 2,
                max_align: 2,
                offset: 0,
                memory: 0,
            },
        },
        Operator::I32Add,
        Operator::I32Const { value: 8 },
        Operator::I32Load {
            memarg: wasmparser::MemArg {
                align: 2,
                max_align: 2,
                offset: 0,
                memory: 0,
            },
        },
        Operator::I32Add,
        Operator::End,
    ];

    let func =
        compile_simple(&[ValType::I32], &[ValType::I32], &[], &body).expect("Compilation failed");

    let mut memory = [0u8; 64];
    let memory_base = memory.as_mut_ptr() as i32;

    let f = func.as_fn::<fn(i32) -> i32>();
    let result = f(memory_base);
    assert_eq!(result, 60); // 10 + 20 + 30 = 60
}

#[test]
fn test_memory_dynamic_offset() {
    // Use a parameter as the dynamic offset
    // Function: (param i32 i32) (result i32)
    // param 0 = memory base, param 1 = offset
    // Store 99 at the given offset, then load it back
    let body = [
        Operator::LocalGet { local_index: 1 }, // offset from param
        Operator::I32Const { value: 99 },
        Operator::I32Store {
            memarg: wasmparser::MemArg {
                align: 2,
                max_align: 2,
                offset: 0,
                memory: 0,
            },
        },
        Operator::LocalGet { local_index: 1 }, // offset from param
        Operator::I32Load {
            memarg: wasmparser::MemArg {
                align: 2,
                max_align: 2,
                offset: 0,
                memory: 0,
            },
        },
        Operator::End,
    ];

    let func = compile_simple(&[ValType::I32, ValType::I32], &[ValType::I32], &[], &body)
        .expect("Compilation failed");

    let mut memory = [0u8; 64];
    let memory_base = memory.as_mut_ptr() as i32;

    let f = func.as_fn::<fn(i32, i32) -> i32>();

    // Test storing at different offsets
    assert_eq!(f(memory_base, 0), 99);
    assert_eq!(f(memory_base, 4), 99);
    assert_eq!(f(memory_base, 8), 99);
}

// I64 Memory store/load tests

#[test]
fn test_memory_i64_store_load() {
    // Test I64Store and I64Load
    let body = [
        Operator::I32Const { value: 0 }, // offset (as i32)
        Operator::I64Const {
            value: 0x123456789ABCDEFi64,
        }, // value
        Operator::I64Store {
            memarg: wasmparser::MemArg {
                align: 3,
                max_align: 3,
                offset: 0,
                memory: 0,
            },
        },
        Operator::I32Const { value: 0 }, // offset
        Operator::I64Load {
            memarg: wasmparser::MemArg {
                align: 3,
                max_align: 3,
                offset: 0,
                memory: 0,
            },
        },
        Operator::I32WrapI64, // wrap to i32 for return
        Operator::End,
    ];

    let func =
        compile_simple(&[ValType::I32], &[ValType::I32], &[], &body).expect("Compilation failed");

    let mut memory = [0u8; 64];
    let memory_base = memory.as_mut_ptr() as i32;

    let f = func.as_fn::<fn(i32) -> i32>();
    let result = f(memory_base);
    // Lower 32 bits of 0x123456789ABCDEF = 0x89ABCDEF
    assert_eq!(result, 0x89ABCDEFu32 as i32);
}

#[test]
fn test_memory_i64_store_load_full_value() {
    // Verify full 64-bit value is stored and loaded correctly
    // by checking both upper and lower 32 bits
    let body = [
        // Store a 64-bit value
        Operator::I32Const { value: 0 },
        Operator::I64Const {
            value: 0xFEDCBA9876543210u64 as i64,
        },
        Operator::I64Store {
            memarg: wasmparser::MemArg {
                align: 3,
                max_align: 3,
                offset: 0,
                memory: 0,
            },
        },
        // Load it back and shift right 32 bits to get upper half
        Operator::I32Const { value: 0 },
        Operator::I64Load {
            memarg: wasmparser::MemArg {
                align: 3,
                max_align: 3,
                offset: 0,
                memory: 0,
            },
        },
        Operator::I64Const { value: 32 },
        Operator::I64ShrU,
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func =
        compile_simple(&[ValType::I32], &[ValType::I32], &[], &body).expect("Compilation failed");

    let mut memory = [0u8; 64];
    let memory_base = memory.as_mut_ptr() as i32;

    let f = func.as_fn::<fn(i32) -> i32>();
    let result = f(memory_base);
    // Upper 32 bits of 0xFEDCBA9876543210 = 0xFEDCBA98
    assert_eq!(result, 0xFEDCBA98u32 as i32);
}

#[test]
fn test_memory_i64_load8_unsigned() {
    // Test I64Load8U - load byte and zero-extend to 64-bit
    let body = [
        Operator::I32Const { value: 0 },
        Operator::I64Const { value: 0xFF }, // Store 0xFF (255)
        Operator::I64Store8 {
            memarg: wasmparser::MemArg {
                align: 0,
                max_align: 0,
                offset: 0,
                memory: 0,
            },
        },
        Operator::I32Const { value: 0 },
        Operator::I64Load8U {
            memarg: wasmparser::MemArg {
                align: 0,
                max_align: 0,
                offset: 0,
                memory: 0,
            },
        },
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func =
        compile_simple(&[ValType::I32], &[ValType::I32], &[], &body).expect("Compilation failed");

    let mut memory = [0u8; 64];
    let memory_base = memory.as_mut_ptr() as i32;

    let f = func.as_fn::<fn(i32) -> i32>();
    let result = f(memory_base);
    assert_eq!(result, 255); // Unsigned: 0xFF = 255
}

#[test]
fn test_memory_i64_load8_signed() {
    // Test I64Load8S - load byte and sign-extend to 64-bit
    let body = [
        Operator::I32Const { value: 0 },
        Operator::I64Const { value: 0xFF }, // Store 0xFF (-1 as signed byte)
        Operator::I64Store8 {
            memarg: wasmparser::MemArg {
                align: 0,
                max_align: 0,
                offset: 0,
                memory: 0,
            },
        },
        Operator::I32Const { value: 0 },
        Operator::I64Load8S {
            memarg: wasmparser::MemArg {
                align: 0,
                max_align: 0,
                offset: 0,
                memory: 0,
            },
        },
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func =
        compile_simple(&[ValType::I32], &[ValType::I32], &[], &body).expect("Compilation failed");

    let mut memory = [0u8; 64];
    let memory_base = memory.as_mut_ptr() as i32;

    let f = func.as_fn::<fn(i32) -> i32>();
    let result = f(memory_base);
    assert_eq!(result, -1); // Sign-extended: 0xFF -> -1
}

#[test]
fn test_memory_i64_load16_unsigned() {
    // Test I64Load16U - load 16-bit and zero-extend to 64-bit
    let body = [
        Operator::I32Const { value: 0 },
        Operator::I64Const { value: 0xFFFF }, // Store 0xFFFF (65535)
        Operator::I64Store16 {
            memarg: wasmparser::MemArg {
                align: 1,
                max_align: 1,
                offset: 0,
                memory: 0,
            },
        },
        Operator::I32Const { value: 0 },
        Operator::I64Load16U {
            memarg: wasmparser::MemArg {
                align: 1,
                max_align: 1,
                offset: 0,
                memory: 0,
            },
        },
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func =
        compile_simple(&[ValType::I32], &[ValType::I32], &[], &body).expect("Compilation failed");

    let mut memory = [0u8; 64];
    let memory_base = memory.as_mut_ptr() as i32;

    let f = func.as_fn::<fn(i32) -> i32>();
    let result = f(memory_base);
    assert_eq!(result, 65535); // Unsigned: 0xFFFF = 65535
}

#[test]
fn test_memory_i64_load16_signed() {
    // Test I64Load16S - load 16-bit and sign-extend to 64-bit
    let body = [
        Operator::I32Const { value: 0 },
        Operator::I64Const { value: 0xFFFF }, // Store 0xFFFF (-1 as signed i16)
        Operator::I64Store16 {
            memarg: wasmparser::MemArg {
                align: 1,
                max_align: 1,
                offset: 0,
                memory: 0,
            },
        },
        Operator::I32Const { value: 0 },
        Operator::I64Load16S {
            memarg: wasmparser::MemArg {
                align: 1,
                max_align: 1,
                offset: 0,
                memory: 0,
            },
        },
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func =
        compile_simple(&[ValType::I32], &[ValType::I32], &[], &body).expect("Compilation failed");

    let mut memory = [0u8; 64];
    let memory_base = memory.as_mut_ptr() as i32;

    let f = func.as_fn::<fn(i32) -> i32>();
    let result = f(memory_base);
    assert_eq!(result, -1); // Sign-extended: 0xFFFF -> -1
}

#[test]
fn test_memory_i64_load32_unsigned() {
    // Test I64Load32U - load 32-bit and zero-extend to 64-bit
    let body = [
        Operator::I32Const { value: 0 },
        Operator::I64Const {
            value: 0xFFFFFFFFi64,
        }, // Store 0xFFFFFFFF
        Operator::I64Store32 {
            memarg: wasmparser::MemArg {
                align: 2,
                max_align: 2,
                offset: 0,
                memory: 0,
            },
        },
        Operator::I32Const { value: 0 },
        Operator::I64Load32U {
            memarg: wasmparser::MemArg {
                align: 2,
                max_align: 2,
                offset: 0,
                memory: 0,
            },
        },
        // To verify it's zero-extended, shift right 32 bits - should be 0
        Operator::I64Const { value: 32 },
        Operator::I64ShrU,
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func =
        compile_simple(&[ValType::I32], &[ValType::I32], &[], &body).expect("Compilation failed");

    let mut memory = [0u8; 64];
    let memory_base = memory.as_mut_ptr() as i32;

    let f = func.as_fn::<fn(i32) -> i32>();
    let result = f(memory_base);
    assert_eq!(result, 0); // Upper 32 bits should be 0 (zero-extended)
}

#[test]
fn test_memory_i64_load32_signed() {
    // Test I64Load32S - load 32-bit and sign-extend to 64-bit
    let body = [
        Operator::I32Const { value: 0 },
        Operator::I64Const {
            value: 0xFFFFFFFFi64,
        }, // Store -1 as 32-bit
        Operator::I64Store32 {
            memarg: wasmparser::MemArg {
                align: 2,
                max_align: 2,
                offset: 0,
                memory: 0,
            },
        },
        Operator::I32Const { value: 0 },
        Operator::I64Load32S {
            memarg: wasmparser::MemArg {
                align: 2,
                max_align: 2,
                offset: 0,
                memory: 0,
            },
        },
        // To verify it's sign-extended, shift right 32 bits - should be -1 (all 1s)
        Operator::I64Const { value: 32 },
        Operator::I64ShrU, // Use unsigned shift to see the bit pattern
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func =
        compile_simple(&[ValType::I32], &[ValType::I32], &[], &body).expect("Compilation failed");

    let mut memory = [0u8; 64];
    let memory_base = memory.as_mut_ptr() as i32;

    let f = func.as_fn::<fn(i32) -> i32>();
    let result = f(memory_base);
    assert_eq!(result, -1); // Upper 32 bits should be 0xFFFFFFFF (sign-extended)
}

#[test]
fn test_memory_i64_with_offset() {
    // Test I64 store/load with static offset
    let body = [
        Operator::I32Const { value: 0 },
        Operator::I64Const {
            value: 0xDEADBEEFCAFEBABEu64 as i64,
        },
        Operator::I64Store {
            memarg: wasmparser::MemArg {
                align: 3,
                max_align: 3,
                offset: 16, // static offset
                memory: 0,
            },
        },
        Operator::I32Const { value: 0 },
        Operator::I64Load {
            memarg: wasmparser::MemArg {
                align: 3,
                max_align: 3,
                offset: 16,
                memory: 0,
            },
        },
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func =
        compile_simple(&[ValType::I32], &[ValType::I32], &[], &body).expect("Compilation failed");

    let mut memory = [0u8; 64];
    let memory_base = memory.as_mut_ptr() as i32;

    let f = func.as_fn::<fn(i32) -> i32>();
    let result = f(memory_base);
    // Lower 32 bits of 0xDEADBEEFCAFEBABE = 0xCAFEBABE
    assert_eq!(result, 0xCAFEBABEu32 as i32);
}

#[test]
fn test_memory_i64_multiple_stores_loads() {
    // Store multiple I64 values and load them back
    let body = [
        // Store 0x1111111111111111 at offset 0
        Operator::I32Const { value: 0 },
        Operator::I64Const {
            value: 0x1111111111111111u64 as i64,
        },
        Operator::I64Store {
            memarg: wasmparser::MemArg {
                align: 3,
                max_align: 3,
                offset: 0,
                memory: 0,
            },
        },
        // Store 0x2222222222222222 at offset 8
        Operator::I32Const { value: 8 },
        Operator::I64Const {
            value: 0x2222222222222222u64 as i64,
        },
        Operator::I64Store {
            memarg: wasmparser::MemArg {
                align: 3,
                max_align: 3,
                offset: 0,
                memory: 0,
            },
        },
        // Load both and add them
        Operator::I32Const { value: 0 },
        Operator::I64Load {
            memarg: wasmparser::MemArg {
                align: 3,
                max_align: 3,
                offset: 0,
                memory: 0,
            },
        },
        Operator::I32Const { value: 8 },
        Operator::I64Load {
            memarg: wasmparser::MemArg {
                align: 3,
                max_align: 3,
                offset: 0,
                memory: 0,
            },
        },
        Operator::I64Add,
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func =
        compile_simple(&[ValType::I32], &[ValType::I32], &[], &body).expect("Compilation failed");

    let mut memory = [0u8; 64];
    let memory_base = memory.as_mut_ptr() as i32;

    let f = func.as_fn::<fn(i32) -> i32>();
    let result = f(memory_base);
    // Lower 32 bits of (0x1111111111111111 + 0x2222222222222222) = 0x3333333333333333
    // Lower 32 bits = 0x33333333
    assert_eq!(result, 0x33333333);
}

// F32/F64 Memory store/load tests

#[test]
fn test_memory_f32_store_load() {
    // Test F32Store and F32Load
    let body = [
        Operator::I32Const { value: 0 }, // offset
        Operator::F32Const {
            value: Ieee32::from(f32::from_bits(0x40490FDB)),
        }, // PI ≈ 3.14159
        Operator::F32Store {
            memarg: wasmparser::MemArg {
                align: 2,
                max_align: 2,
                offset: 0,
                memory: 0,
            },
        },
        Operator::I32Const { value: 0 }, // offset
        Operator::F32Load {
            memarg: wasmparser::MemArg {
                align: 2,
                max_align: 2,
                offset: 0,
                memory: 0,
            },
        },
        // Convert to i32 bits for comparison
        Operator::I32ReinterpretF32,
        Operator::End,
    ];

    let func =
        compile_simple(&[ValType::I32], &[ValType::I32], &[], &body).expect("Compilation failed");

    let mut memory = [0u8; 64];
    let memory_base = memory.as_mut_ptr() as i32;

    let f = func.as_fn::<fn(i32) -> i32>();
    let result = f(memory_base);
    // Should get back the same bits: 0x40490FDB
    assert_eq!(result as u32, 0x40490FDB);
}

#[test]
fn test_memory_f32_store_load_with_offset() {
    // Test F32Store and F32Load with static offset
    let body = [
        Operator::I32Const { value: 0 }, // base offset
        Operator::F32Const {
            value: Ieee32::from(f32::from_bits(0x41200000)),
        }, // 10.0f
        Operator::F32Store {
            memarg: wasmparser::MemArg {
                align: 2,
                max_align: 2,
                offset: 8, // static offset
                memory: 0,
            },
        },
        Operator::I32Const { value: 0 }, // base offset
        Operator::F32Load {
            memarg: wasmparser::MemArg {
                align: 2,
                max_align: 2,
                offset: 8, // static offset
                memory: 0,
            },
        },
        Operator::I32ReinterpretF32,
        Operator::End,
    ];

    let func =
        compile_simple(&[ValType::I32], &[ValType::I32], &[], &body).expect("Compilation failed");

    let mut memory = [0u8; 64];
    let memory_base = memory.as_mut_ptr() as i32;

    let f = func.as_fn::<fn(i32) -> i32>();
    let result = f(memory_base);
    // 10.0f = 0x41200000
    assert_eq!(result as u32, 0x41200000);
}

#[test]
fn test_memory_f32_multiple_stores_loads() {
    // Store multiple F32 values and load them back, then add them
    let body = [
        // Store 1.5 at offset 0
        Operator::I32Const { value: 0 },
        Operator::F32Const {
            value: Ieee32::from(f32::from_bits(0x3FC00000)),
        }, // 1.5f
        Operator::F32Store {
            memarg: wasmparser::MemArg {
                align: 2,
                max_align: 2,
                offset: 0,
                memory: 0,
            },
        },
        // Store 2.5 at offset 4
        Operator::I32Const { value: 4 },
        Operator::F32Const {
            value: Ieee32::from(f32::from_bits(0x40200000)),
        }, // 2.5f
        Operator::F32Store {
            memarg: wasmparser::MemArg {
                align: 2,
                max_align: 2,
                offset: 0,
                memory: 0,
            },
        },
        // Load both and add them
        Operator::I32Const { value: 0 },
        Operator::F32Load {
            memarg: wasmparser::MemArg {
                align: 2,
                max_align: 2,
                offset: 0,
                memory: 0,
            },
        },
        Operator::I32Const { value: 4 },
        Operator::F32Load {
            memarg: wasmparser::MemArg {
                align: 2,
                max_align: 2,
                offset: 0,
                memory: 0,
            },
        },
        Operator::F32Add,
        Operator::I32ReinterpretF32,
        Operator::End,
    ];

    let func =
        compile_simple(&[ValType::I32], &[ValType::I32], &[], &body).expect("Compilation failed");

    let mut memory = [0u8; 64];
    let memory_base = memory.as_mut_ptr() as i32;

    let f = func.as_fn::<fn(i32) -> i32>();
    let result = f(memory_base);
    // 1.5 + 2.5 = 4.0f = 0x40800000
    assert_eq!(result as u32, 0x40800000);
}

#[test]
fn test_memory_f64_store_load() {
    // Test F64Store and F64Load
    let body = [
        Operator::I32Const { value: 0 }, // offset
        Operator::F64Const {
            value: Ieee64::from(f64::from_bits(0x400921FB54442D18)),
        }, // PI ≈ 3.141592653589793
        Operator::F64Store {
            memarg: wasmparser::MemArg {
                align: 3,
                max_align: 3,
                offset: 0,
                memory: 0,
            },
        },
        Operator::I32Const { value: 0 }, // offset
        Operator::F64Load {
            memarg: wasmparser::MemArg {
                align: 3,
                max_align: 3,
                offset: 0,
                memory: 0,
            },
        },
        // Convert to i64 bits and wrap to i32 (lower 32 bits)
        Operator::I64ReinterpretF64,
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func =
        compile_simple(&[ValType::I32], &[ValType::I32], &[], &body).expect("Compilation failed");

    let mut memory = [0u8; 64];
    let memory_base = memory.as_mut_ptr() as i32;

    let f = func.as_fn::<fn(i32) -> i32>();
    let result = f(memory_base);
    // Lower 32 bits of 0x400921FB54442D18 = 0x54442D18
    assert_eq!(result as u32, 0x54442D18);
}

#[test]
fn test_memory_f64_store_load_upper_bits() {
    // Test F64Store and F64Load - verify upper 32 bits
    let body = [
        Operator::I32Const { value: 0 }, // offset
        Operator::F64Const {
            value: Ieee64::from(f64::from_bits(0x400921FB54442D18)),
        }, // PI
        Operator::F64Store {
            memarg: wasmparser::MemArg {
                align: 3,
                max_align: 3,
                offset: 0,
                memory: 0,
            },
        },
        Operator::I32Const { value: 0 }, // offset
        Operator::F64Load {
            memarg: wasmparser::MemArg {
                align: 3,
                max_align: 3,
                offset: 0,
                memory: 0,
            },
        },
        // Convert to i64 bits, shift right 32 to get upper bits, wrap to i32
        Operator::I64ReinterpretF64,
        Operator::I64Const { value: 32 },
        Operator::I64ShrU,
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func =
        compile_simple(&[ValType::I32], &[ValType::I32], &[], &body).expect("Compilation failed");

    let mut memory = [0u8; 64];
    let memory_base = memory.as_mut_ptr() as i32;

    let f = func.as_fn::<fn(i32) -> i32>();
    let result = f(memory_base);
    // Upper 32 bits of 0x400921FB54442D18 = 0x400921FB
    assert_eq!(result as u32, 0x400921FB);
}

#[test]
fn test_memory_f64_store_load_with_offset() {
    // Test F64Store and F64Load with static offset
    let body = [
        Operator::I32Const { value: 0 }, // base offset
        Operator::F64Const {
            value: Ieee64::from(f64::from_bits(0x4024000000000000)),
        }, // 10.0
        Operator::F64Store {
            memarg: wasmparser::MemArg {
                align: 3,
                max_align: 3,
                offset: 16, // static offset
                memory: 0,
            },
        },
        Operator::I32Const { value: 0 }, // base offset
        Operator::F64Load {
            memarg: wasmparser::MemArg {
                align: 3,
                max_align: 3,
                offset: 16, // static offset
                memory: 0,
            },
        },
        Operator::I64ReinterpretF64,
        Operator::I64Const { value: 32 },
        Operator::I64ShrU,
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func =
        compile_simple(&[ValType::I32], &[ValType::I32], &[], &body).expect("Compilation failed");

    let mut memory = [0u8; 64];
    let memory_base = memory.as_mut_ptr() as i32;

    let f = func.as_fn::<fn(i32) -> i32>();
    let result = f(memory_base);
    // Upper 32 bits of 10.0 (0x4024000000000000) = 0x40240000
    assert_eq!(result as u32, 0x40240000);
}

#[test]
fn test_memory_f64_multiple_stores_loads() {
    // Store multiple F64 values and load them back, then add them

    let body = [
        // Store 1.5 at offset 0
        Operator::I32Const { value: 0 },
        Operator::F64Const {
            value: Ieee64::from(f64::from_bits(0x3FF8000000000000)),
        }, // 1.5
        Operator::F64Store {
            memarg: wasmparser::MemArg {
                align: 3,
                max_align: 3,
                offset: 0,
                memory: 0,
            },
        },
        // Store 2.5 at offset 8
        Operator::I32Const { value: 8 },
        Operator::F64Const {
            value: Ieee64::from(f64::from_bits(0x4004000000000000)),
        }, // 2.5
        Operator::F64Store {
            memarg: wasmparser::MemArg {
                align: 3,
                max_align: 3,
                offset: 0,
                memory: 0,
            },
        },
        // Load both and add them
        Operator::I32Const { value: 0 },
        Operator::F64Load {
            memarg: wasmparser::MemArg {
                align: 3,
                max_align: 3,
                offset: 0,
                memory: 0,
            },
        },
        Operator::I32Const { value: 8 },
        Operator::F64Load {
            memarg: wasmparser::MemArg {
                align: 3,
                max_align: 3,
                offset: 0,
                memory: 0,
            },
        },
        Operator::F64Add,
        Operator::I64ReinterpretF64,
        Operator::I64Const { value: 32 },
        Operator::I64ShrU,
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func =
        compile_simple(&[ValType::I32], &[ValType::I32], &[], &body).expect("Compilation failed");

    let mut memory = [0u8; 64];
    let memory_base = memory.as_mut_ptr() as i32;

    let f = func.as_fn::<fn(i32) -> i32>();
    let result = f(memory_base);
    // 1.5 + 2.5 = 4.0 = 0x4010000000000000, upper bits = 0x40100000
    assert_eq!(result as u32, 0x40100000);
}

#[test]
fn test_memory_f32_dynamic_offset() {
    // Use a parameter as the dynamic offset for F32
    let body = [
        Operator::LocalGet { local_index: 1 }, // offset from param
        Operator::F32Const {
            value: Ieee32::from(f32::from_bits(0x42280000)),
        }, // 42.0f
        Operator::F32Store {
            memarg: wasmparser::MemArg {
                align: 2,
                max_align: 2,
                offset: 0,
                memory: 0,
            },
        },
        Operator::LocalGet { local_index: 1 }, // offset from param
        Operator::F32Load {
            memarg: wasmparser::MemArg {
                align: 2,
                max_align: 2,
                offset: 0,
                memory: 0,
            },
        },
        Operator::I32ReinterpretF32,
        Operator::End,
    ];

    let func = compile_simple(&[ValType::I32, ValType::I32], &[ValType::I32], &[], &body)
        .expect("Compilation failed");

    let mut memory = [0u8; 64];
    let memory_base = memory.as_mut_ptr() as i32;

    let f = func.as_fn::<fn(i32, i32) -> i32>();

    // Test storing at different offsets
    assert_eq!(f(memory_base, 0) as u32, 0x42280000);
    assert_eq!(f(memory_base, 4) as u32, 0x42280000);
    assert_eq!(f(memory_base, 8) as u32, 0x42280000);
}

#[test]
fn test_memory_f64_dynamic_offset() {
    // Use a parameter as the dynamic offset for F64
    let body = [
        Operator::LocalGet { local_index: 1 }, // offset from param
        Operator::F64Const {
            value: Ieee64::from(f64::from_bits(0x4045000000000000)),
        }, // 42.0
        Operator::F64Store {
            memarg: wasmparser::MemArg {
                align: 3,
                max_align: 3,
                offset: 0,
                memory: 0,
            },
        },
        Operator::LocalGet { local_index: 1 }, // offset from param
        Operator::F64Load {
            memarg: wasmparser::MemArg {
                align: 3,
                max_align: 3,
                offset: 0,
                memory: 0,
            },
        },
        Operator::I64ReinterpretF64,
        Operator::I64Const { value: 32 },
        Operator::I64ShrU,
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func = compile_simple(&[ValType::I32, ValType::I32], &[ValType::I32], &[], &body)
        .expect("Compilation failed");

    let mut memory = [0u8; 128];
    let memory_base = memory.as_mut_ptr() as i32;

    let f = func.as_fn::<fn(i32, i32) -> i32>();

    // Test storing at different offsets - upper 32 bits of 42.0 = 0x40450000
    assert_eq!(f(memory_base, 0) as u32, 0x40450000);
    assert_eq!(f(memory_base, 8) as u32, 0x40450000);
    assert_eq!(f(memory_base, 16) as u32, 0x40450000);
}

// ============================================================================
// Floating Point Operator Tests
// ============================================================================

// F32 Arithmetic Operations

#[test]
fn test_f32_add() {
    // Test F32Add: 1.5 + 2.5 = 4.0
    let body = [
        Operator::F32Const {
            value: Ieee32::from(f32::from_bits(0x3FC00000)), // 1.5
        },
        Operator::F32Const {
            value: Ieee32::from(f32::from_bits(0x40200000)), // 2.5
        },
        Operator::F32Add,
        Operator::I32ReinterpretF32,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn() -> i32>();
    // 4.0f = 0x40800000
    assert_eq!(f() as u32, 0x40800000);
}

#[test]
fn test_f32_sub() {
    // Test F32Sub: 5.0 - 2.0 = 3.0
    let body = [
        Operator::F32Const {
            value: Ieee32::from(f32::from_bits(0x40A00000)), // 5.0
        },
        Operator::F32Const {
            value: Ieee32::from(f32::from_bits(0x40000000)), // 2.0
        },
        Operator::F32Sub,
        Operator::I32ReinterpretF32,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn() -> i32>();
    // 3.0f = 0x40400000
    assert_eq!(f() as u32, 0x40400000);
}

#[test]
fn test_f32_mul() {
    // Test F32Mul: 3.0 * 4.0 = 12.0
    let body = [
        Operator::F32Const {
            value: Ieee32::from(f32::from_bits(0x40400000)), // 3.0
        },
        Operator::F32Const {
            value: Ieee32::from(f32::from_bits(0x40800000)), // 4.0
        },
        Operator::F32Mul,
        Operator::I32ReinterpretF32,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn() -> i32>();
    // 12.0f = 0x41400000
    assert_eq!(f() as u32, 0x41400000);
}

#[test]
fn test_f32_div() {
    // Test F32Div: 10.0 / 2.0 = 5.0
    let body = [
        Operator::F32Const {
            value: Ieee32::from(f32::from_bits(0x41200000)), // 10.0
        },
        Operator::F32Const {
            value: Ieee32::from(f32::from_bits(0x40000000)), // 2.0
        },
        Operator::F32Div,
        Operator::I32ReinterpretF32,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn() -> i32>();
    // 5.0f = 0x40A00000
    assert_eq!(f() as u32, 0x40A00000);
}

#[test]
fn test_f32_min() {
    // Test F32Min: min(3.0, 5.0) = 3.0
    let body = [
        Operator::F32Const {
            value: Ieee32::from(f32::from_bits(0x40400000)), // 3.0
        },
        Operator::F32Const {
            value: Ieee32::from(f32::from_bits(0x40A00000)), // 5.0
        },
        Operator::F32Min,
        Operator::I32ReinterpretF32,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn() -> i32>();
    // 3.0f = 0x40400000
    assert_eq!(f() as u32, 0x40400000);
}

#[test]
fn test_f32_max() {
    // Test F32Max: max(3.0, 5.0) = 5.0
    let body = [
        Operator::F32Const {
            value: Ieee32::from(f32::from_bits(0x40400000)), // 3.0
        },
        Operator::F32Const {
            value: Ieee32::from(f32::from_bits(0x40A00000)), // 5.0
        },
        Operator::F32Max,
        Operator::I32ReinterpretF32,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn() -> i32>();
    // 5.0f = 0x40A00000
    assert_eq!(f() as u32, 0x40A00000);
}

#[test]
fn test_f32_copysign() {
    // Test F32Copysign: copysign(3.0, -1.0) = -3.0
    let body = [
        Operator::F32Const {
            value: Ieee32::from(f32::from_bits(0x40400000)), // 3.0
        },
        Operator::F32Const {
            value: Ieee32::from(f32::from_bits(0xBF800000)), // -1.0
        },
        Operator::F32Copysign,
        Operator::I32ReinterpretF32,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn() -> i32>();
    // -3.0f = 0xC0400000
    assert_eq!(f() as u32, 0xC0400000);
}

// F32 Unary Operations

#[test]
fn test_f32_neg() {
    // Test F32Neg: neg(5.0) = -5.0
    let body = [
        Operator::F32Const {
            value: Ieee32::from(f32::from_bits(0x40A00000)), // 5.0
        },
        Operator::F32Neg,
        Operator::I32ReinterpretF32,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn() -> i32>();
    // -5.0f = 0xC0A00000
    assert_eq!(f() as u32, 0xC0A00000);
}

#[test]
fn test_f32_abs() {
    // Test F32Abs: abs(-5.0) = 5.0
    let body = [
        Operator::F32Const {
            value: Ieee32::from(f32::from_bits(0xC0A00000)), // -5.0
        },
        Operator::F32Abs,
        Operator::I32ReinterpretF32,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn() -> i32>();
    // 5.0f = 0x40A00000
    assert_eq!(f() as u32, 0x40A00000);
}

#[test]
fn test_f32_sqrt() {
    // Test F32Sqrt: sqrt(16.0) = 4.0
    let body = [
        Operator::F32Const {
            value: Ieee32::from(f32::from_bits(0x41800000)), // 16.0
        },
        Operator::F32Sqrt,
        Operator::I32ReinterpretF32,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn() -> i32>();
    // 4.0f = 0x40800000
    assert_eq!(f() as u32, 0x40800000);
}

#[test]
fn test_f32_ceil() {
    // Test F32Ceil: ceil(2.3) = 3.0
    let body = [
        Operator::F32Const {
            value: Ieee32::from(f32::from_bits(0x40133333)), // 2.3
        },
        Operator::F32Ceil,
        Operator::I32ReinterpretF32,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn() -> i32>();
    // 3.0f = 0x40400000
    assert_eq!(f() as u32, 0x40400000);
}

#[test]
fn test_f32_floor() {
    // Test F32Floor: floor(2.7) = 2.0
    let body = [
        Operator::F32Const {
            value: Ieee32::from(f32::from_bits(0x402CCCCD)), // 2.7
        },
        Operator::F32Floor,
        Operator::I32ReinterpretF32,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn() -> i32>();
    // 2.0f = 0x40000000
    assert_eq!(f() as u32, 0x40000000);
}

#[test]
fn test_f32_trunc() {
    // Test F32Trunc: trunc(2.7) = 2.0
    let body = [
        Operator::F32Const {
            value: Ieee32::from(f32::from_bits(0x402CCCCD)), // 2.7
        },
        Operator::F32Trunc,
        Operator::I32ReinterpretF32,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn() -> i32>();
    // 2.0f = 0x40000000
    assert_eq!(f() as u32, 0x40000000);
}

#[test]
fn test_f32_nearest() {
    // Test F32Nearest: nearest(2.5) = 2.0 (round to even)
    let body = [
        Operator::F32Const {
            value: Ieee32::from(f32::from_bits(0x40200000)), // 2.5
        },
        Operator::F32Nearest,
        Operator::I32ReinterpretF32,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn() -> i32>();
    // nearest(2.5) should round to 2.0 (banker's rounding)
    // 2.0f = 0x40000000
    assert_eq!(f() as u32, 0x40000000);
}

// F32 Comparison Operations

#[test]
fn test_f32_eq() {
    // Test F32Eq: 3.0 == 3.0
    let body = [
        Operator::F32Const {
            value: Ieee32::from(f32::from_bits(0x40400000)), // 3.0
        },
        Operator::F32Const {
            value: Ieee32::from(f32::from_bits(0x40400000)), // 3.0
        },
        Operator::F32Eq,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn() -> i32>();
    assert_eq!(f(), 1); // true

    // Test inequality
    let body = [
        Operator::F32Const {
            value: Ieee32::from(f32::from_bits(0x40400000)), // 3.0
        },
        Operator::F32Const {
            value: Ieee32::from(f32::from_bits(0x40800000)), // 4.0
        },
        Operator::F32Eq,
        Operator::End,
    ];

    let func2 = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");
    let f2 = func2.as_fn::<fn() -> i32>();
    assert_eq!(f2(), 0); // false
}

#[test]
fn test_f32_ne() {
    // Test F32Ne: 3.0 != 4.0
    let body = [
        Operator::F32Const {
            value: Ieee32::from(f32::from_bits(0x40400000)), // 3.0
        },
        Operator::F32Const {
            value: Ieee32::from(f32::from_bits(0x40800000)), // 4.0
        },
        Operator::F32Ne,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn() -> i32>();
    assert_eq!(f(), 1); // true
}

#[test]
fn test_f32_lt() {
    // Test F32Lt: 3.0 < 4.0
    let body = [
        Operator::F32Const {
            value: Ieee32::from(f32::from_bits(0x40400000)), // 3.0
        },
        Operator::F32Const {
            value: Ieee32::from(f32::from_bits(0x40800000)), // 4.0
        },
        Operator::F32Lt,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn() -> i32>();
    assert_eq!(f(), 1); // true

    // Test not less than
    let body = [
        Operator::F32Const {
            value: Ieee32::from(f32::from_bits(0x40800000)), // 4.0
        },
        Operator::F32Const {
            value: Ieee32::from(f32::from_bits(0x40400000)), // 3.0
        },
        Operator::F32Lt,
        Operator::End,
    ];

    let func2 = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");
    let f2 = func2.as_fn::<fn() -> i32>();
    assert_eq!(f2(), 0); // false
}

#[test]
fn test_f32_gt() {
    // Test F32Gt: 5.0 > 3.0
    let body = [
        Operator::F32Const {
            value: Ieee32::from(f32::from_bits(0x40A00000)), // 5.0
        },
        Operator::F32Const {
            value: Ieee32::from(f32::from_bits(0x40400000)), // 3.0
        },
        Operator::F32Gt,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn() -> i32>();
    assert_eq!(f(), 1); // true
}

#[test]
fn test_f32_le() {
    // Test F32Le: 3.0 <= 3.0
    let body = [
        Operator::F32Const {
            value: Ieee32::from(f32::from_bits(0x40400000)), // 3.0
        },
        Operator::F32Const {
            value: Ieee32::from(f32::from_bits(0x40400000)), // 3.0
        },
        Operator::F32Le,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn() -> i32>();
    assert_eq!(f(), 1); // true
}

#[test]
fn test_f32_ge() {
    // Test F32Ge: 5.0 >= 3.0
    let body = [
        Operator::F32Const {
            value: Ieee32::from(f32::from_bits(0x40A00000)), // 5.0
        },
        Operator::F32Const {
            value: Ieee32::from(f32::from_bits(0x40400000)), // 3.0
        },
        Operator::F32Ge,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn() -> i32>();
    assert_eq!(f(), 1); // true
}

// F64 Arithmetic Operations

#[test]
fn test_f64_add() {
    // Test F64Add: 1.5 + 2.5 = 4.0
    let body = [
        Operator::F64Const {
            value: Ieee64::from(f64::from_bits(0x3FF8000000000000)), // 1.5
        },
        Operator::F64Const {
            value: Ieee64::from(f64::from_bits(0x4004000000000000)), // 2.5
        },
        Operator::F64Add,
        Operator::I64ReinterpretF64,
        Operator::I64Const { value: 32 },
        Operator::I64ShrU,
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn() -> i32>();
    // 4.0 = 0x4010000000000000, upper 32 bits = 0x40100000
    assert_eq!(f() as u32, 0x40100000);
}

#[test]
fn test_f64_sub() {
    // Test F64Sub: 5.0 - 2.0 = 3.0
    let body = [
        Operator::F64Const {
            value: Ieee64::from(f64::from_bits(0x4014000000000000)), // 5.0
        },
        Operator::F64Const {
            value: Ieee64::from(f64::from_bits(0x4000000000000000)), // 2.0
        },
        Operator::F64Sub,
        Operator::I64ReinterpretF64,
        Operator::I64Const { value: 32 },
        Operator::I64ShrU,
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn() -> i32>();
    // 3.0 = 0x4008000000000000, upper 32 bits = 0x40080000
    assert_eq!(f() as u32, 0x40080000);
}

#[test]
fn test_f64_mul() {
    // Test F64Mul: 3.0 * 4.0 = 12.0
    let body = [
        Operator::F64Const {
            value: Ieee64::from(f64::from_bits(0x4008000000000000)), // 3.0
        },
        Operator::F64Const {
            value: Ieee64::from(f64::from_bits(0x4010000000000000)), // 4.0
        },
        Operator::F64Mul,
        Operator::I64ReinterpretF64,
        Operator::I64Const { value: 32 },
        Operator::I64ShrU,
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn() -> i32>();
    // 12.0 = 0x4028000000000000, upper 32 bits = 0x40280000
    assert_eq!(f() as u32, 0x40280000);
}

#[test]
fn test_f64_div() {
    // Test F64Div: 10.0 / 2.0 = 5.0
    let body = [
        Operator::F64Const {
            value: Ieee64::from(f64::from_bits(0x4024000000000000)), // 10.0
        },
        Operator::F64Const {
            value: Ieee64::from(f64::from_bits(0x4000000000000000)), // 2.0
        },
        Operator::F64Div,
        Operator::I64ReinterpretF64,
        Operator::I64Const { value: 32 },
        Operator::I64ShrU,
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn() -> i32>();
    // 5.0 = 0x4014000000000000, upper 32 bits = 0x40140000
    assert_eq!(f() as u32, 0x40140000);
}

#[test]
fn test_f64_min() {
    // Test F64Min: min(3.0, 5.0) = 3.0
    let body = [
        Operator::F64Const {
            value: Ieee64::from(f64::from_bits(0x4008000000000000)), // 3.0
        },
        Operator::F64Const {
            value: Ieee64::from(f64::from_bits(0x4014000000000000)), // 5.0
        },
        Operator::F64Min,
        Operator::I64ReinterpretF64,
        Operator::I64Const { value: 32 },
        Operator::I64ShrU,
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn() -> i32>();
    // 3.0 = 0x4008000000000000, upper 32 bits = 0x40080000
    assert_eq!(f() as u32, 0x40080000);
}

#[test]
fn test_f64_max() {
    // Test F64Max: max(3.0, 5.0) = 5.0
    let body = [
        Operator::F64Const {
            value: Ieee64::from(f64::from_bits(0x4008000000000000)), // 3.0
        },
        Operator::F64Const {
            value: Ieee64::from(f64::from_bits(0x4014000000000000)), // 5.0
        },
        Operator::F64Max,
        Operator::I64ReinterpretF64,
        Operator::I64Const { value: 32 },
        Operator::I64ShrU,
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn() -> i32>();
    // 5.0 = 0x4014000000000000, upper 32 bits = 0x40140000
    assert_eq!(f() as u32, 0x40140000);
}

#[test]
fn test_f64_copysign() {
    // Test F64Copysign: copysign(3.0, -1.0) = -3.0
    let body = [
        Operator::F64Const {
            value: Ieee64::from(f64::from_bits(0x4008000000000000)), // 3.0
        },
        Operator::F64Const {
            value: Ieee64::from(f64::from_bits(0xBFF0000000000000)), // -1.0
        },
        Operator::F64Copysign,
        Operator::I64ReinterpretF64,
        Operator::I64Const { value: 32 },
        Operator::I64ShrU,
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn() -> i32>();
    // -3.0 = 0xC008000000000000, upper 32 bits = 0xC0080000
    assert_eq!(f() as u32, 0xC0080000);
}

// F64 Unary Operations

#[test]
fn test_f64_neg() {
    // Test F64Neg: neg(5.0) = -5.0
    let body = [
        Operator::F64Const {
            value: Ieee64::from(f64::from_bits(0x4014000000000000)), // 5.0
        },
        Operator::F64Neg,
        Operator::I64ReinterpretF64,
        Operator::I64Const { value: 32 },
        Operator::I64ShrU,
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn() -> i32>();
    // -5.0 = 0xC014000000000000, upper 32 bits = 0xC0140000
    assert_eq!(f() as u32, 0xC0140000);
}

#[test]
fn test_f64_abs() {
    // Test F64Abs: abs(-5.0) = 5.0
    let body = [
        Operator::F64Const {
            value: Ieee64::from(f64::from_bits(0xC014000000000000)), // -5.0
        },
        Operator::F64Abs,
        Operator::I64ReinterpretF64,
        Operator::I64Const { value: 32 },
        Operator::I64ShrU,
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn() -> i32>();
    // 5.0 = 0x4014000000000000, upper 32 bits = 0x40140000
    assert_eq!(f() as u32, 0x40140000);
}

#[test]
fn test_f64_sqrt() {
    // Test F64Sqrt: sqrt(16.0) = 4.0
    let body = [
        Operator::F64Const {
            value: Ieee64::from(f64::from_bits(0x4030000000000000)), // 16.0
        },
        Operator::F64Sqrt,
        Operator::I64ReinterpretF64,
        Operator::I64Const { value: 32 },
        Operator::I64ShrU,
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn() -> i32>();
    // 4.0 = 0x4010000000000000, upper 32 bits = 0x40100000
    assert_eq!(f() as u32, 0x40100000);
}

#[test]
fn test_f64_ceil() {
    // Test F64Ceil: ceil(2.3) = 3.0
    let body = [
        Operator::F64Const {
            value: Ieee64::from(f64::from_bits(0x4002666666666666)), // 2.3
        },
        Operator::F64Ceil,
        Operator::I64ReinterpretF64,
        Operator::I64Const { value: 32 },
        Operator::I64ShrU,
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn() -> i32>();
    // 3.0 = 0x4008000000000000, upper 32 bits = 0x40080000
    assert_eq!(f() as u32, 0x40080000);
}

#[test]
fn test_f64_floor() {
    // Test F64Floor: floor(2.7) = 2.0
    let body = [
        Operator::F64Const {
            value: Ieee64::from(f64::from_bits(0x4005999999999999)), // ~2.7
        },
        Operator::F64Floor,
        Operator::I64ReinterpretF64,
        Operator::I64Const { value: 32 },
        Operator::I64ShrU,
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn() -> i32>();
    // 2.0 = 0x4000000000000000, upper 32 bits = 0x40000000
    assert_eq!(f() as u32, 0x40000000);
}

#[test]
fn test_f64_trunc() {
    // Test F64Trunc: trunc(-2.7) = -2.0
    let body = [
        Operator::F64Const {
            value: Ieee64::from(f64::from_bits(0xC005999999999999)), // ~-2.7
        },
        Operator::F64Trunc,
        Operator::I64ReinterpretF64,
        Operator::I64Const { value: 32 },
        Operator::I64ShrU,
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn() -> i32>();
    // -2.0 = 0xC000000000000000, upper 32 bits = 0xC0000000
    assert_eq!(f() as u32, 0xC0000000);
}

#[test]
fn test_f64_nearest() {
    // Test F64Nearest: nearest(2.5) = 2.0 (round to even)
    let body = [
        Operator::F64Const {
            value: Ieee64::from(f64::from_bits(0x4004000000000000)), // 2.5
        },
        Operator::F64Nearest,
        Operator::I64ReinterpretF64,
        Operator::I64Const { value: 32 },
        Operator::I64ShrU,
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn() -> i32>();
    // nearest(2.5) should round to 2.0 (banker's rounding)
    // 2.0 = 0x4000000000000000, upper 32 bits = 0x40000000
    assert_eq!(f() as u32, 0x40000000);
}

// F64 Comparison Operations

#[test]
fn test_f64_eq() {
    // Test F64Eq: 3.0 == 3.0
    let body = [
        Operator::F64Const {
            value: Ieee64::from(f64::from_bits(0x4008000000000000)), // 3.0
        },
        Operator::F64Const {
            value: Ieee64::from(f64::from_bits(0x4008000000000000)), // 3.0
        },
        Operator::F64Eq,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn() -> i32>();
    assert_eq!(f(), 1); // true
}

#[test]
fn test_f64_ne() {
    // Test F64Ne: 3.0 != 4.0
    let body = [
        Operator::F64Const {
            value: Ieee64::from(f64::from_bits(0x4008000000000000)), // 3.0
        },
        Operator::F64Const {
            value: Ieee64::from(f64::from_bits(0x4010000000000000)), // 4.0
        },
        Operator::F64Ne,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn() -> i32>();
    assert_eq!(f(), 1); // true
}

#[test]
fn test_f64_lt() {
    // Test F64Lt: 3.0 < 4.0
    let body = [
        Operator::F64Const {
            value: Ieee64::from(f64::from_bits(0x4008000000000000)), // 3.0
        },
        Operator::F64Const {
            value: Ieee64::from(f64::from_bits(0x4010000000000000)), // 4.0
        },
        Operator::F64Lt,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn() -> i32>();
    assert_eq!(f(), 1); // true
}

#[test]
fn test_f64_gt() {
    // Test F64Gt: 5.0 > 3.0
    let body = [
        Operator::F64Const {
            value: Ieee64::from(f64::from_bits(0x4014000000000000)), // 5.0
        },
        Operator::F64Const {
            value: Ieee64::from(f64::from_bits(0x4008000000000000)), // 3.0
        },
        Operator::F64Gt,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn() -> i32>();
    assert_eq!(f(), 1); // true
}

#[test]
fn test_f64_le() {
    // Test F64Le: 3.0 <= 3.0
    let body = [
        Operator::F64Const {
            value: Ieee64::from(f64::from_bits(0x4008000000000000)), // 3.0
        },
        Operator::F64Const {
            value: Ieee64::from(f64::from_bits(0x4008000000000000)), // 3.0
        },
        Operator::F64Le,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn() -> i32>();
    assert_eq!(f(), 1); // true
}

#[test]
fn test_f64_ge() {
    // Test F64Ge: 5.0 >= 3.0
    let body = [
        Operator::F64Const {
            value: Ieee64::from(f64::from_bits(0x4014000000000000)), // 5.0
        },
        Operator::F64Const {
            value: Ieee64::from(f64::from_bits(0x4008000000000000)), // 3.0
        },
        Operator::F64Ge,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn() -> i32>();
    assert_eq!(f(), 1); // true
}

// Conversion Operations

#[test]
fn test_f32_convert_i32_s() {
    // Test F32ConvertI32S: convert signed i32 to f32
    // -10 as f32 = -10.0
    let body = [
        Operator::I32Const { value: -10 },
        Operator::F32ConvertI32S,
        Operator::I32ReinterpretF32,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn() -> i32>();
    // -10.0f = 0xC1200000
    assert_eq!(f() as u32, 0xC1200000);
}

#[test]
fn test_f32_convert_i32_u() {
    // Test F32ConvertI32U: convert unsigned i32 to f32
    // 10 as f32 = 10.0
    let body = [
        Operator::I32Const { value: 10 },
        Operator::F32ConvertI32U,
        Operator::I32ReinterpretF32,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn() -> i32>();
    // 10.0f = 0x41200000
    assert_eq!(f() as u32, 0x41200000);
}

#[test]
fn test_f64_convert_i32_s() {
    // Test F64ConvertI32S: convert signed i32 to f64
    // -10 as f64 = -10.0
    let body = [
        Operator::I32Const { value: -10 },
        Operator::F64ConvertI32S,
        Operator::I64ReinterpretF64,
        Operator::I64Const { value: 32 },
        Operator::I64ShrU,
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn() -> i32>();
    // -10.0 = 0xC024000000000000, upper 32 bits = 0xC0240000
    assert_eq!(f() as u32, 0xC0240000);
}

#[test]
fn test_f64_convert_i64_s() {
    // Test F64ConvertI64S: convert signed i64 to f64
    // 1000000 as f64 = 1000000.0
    let body = [
        Operator::I64Const { value: 1000000 },
        Operator::F64ConvertI64S,
        Operator::I64ReinterpretF64,
        Operator::I64Const { value: 32 },
        Operator::I64ShrU,
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn() -> i32>();
    // 1000000.0 = 0x412E848000000000, upper 32 bits = 0x412E8480
    assert_eq!(f() as u32, 0x412E8480);
}

#[test]
fn test_i32_trunc_f32_s() {
    // Test I32TruncF32S: truncate f32 to signed i32
    // -2.7f truncated = -2
    let body = [
        Operator::F32Const {
            value: Ieee32::from(f32::from_bits(0xC02CCCCD)), // -2.7
        },
        Operator::I32TruncF32S,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn() -> i32>();
    assert_eq!(f(), -2);
}

#[test]
fn test_i32_trunc_f32_u() {
    // Test I32TruncF32U: truncate f32 to unsigned i32
    // 2.7f truncated = 2
    let body = [
        Operator::F32Const {
            value: Ieee32::from(f32::from_bits(0x402CCCCD)), // 2.7
        },
        Operator::I32TruncF32U,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn() -> i32>();
    assert_eq!(f(), 2);
}

#[test]
fn test_i32_trunc_f64_s() {
    // Test I32TruncF64S: truncate f64 to signed i32
    // -100.9 truncated = -100
    let body = [
        Operator::F64Const {
            value: Ieee64::from(f64::from_bits(0xC059399999999999)), // -100.9
        },
        Operator::I32TruncF64S,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn() -> i32>();
    assert_eq!(f(), -100);
}

#[test]
fn test_i64_trunc_f64_s() {
    // Test I64TruncF64S: truncate f64 to signed i64
    // -1000000.5 truncated = -1000000
    let body = [
        Operator::F64Const {
            value: Ieee64::from(f64::from_bits(0xC12E848100000000)), // -1000000.5
        },
        Operator::I64TruncF64S,
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn() -> i32>();
    // -1000000 as i32
    assert_eq!(f(), -1000000);
}

// Float Demote/Promote Operations

#[test]
fn test_f32_demote_f64() {
    // Test F32DemoteF64: convert f64 to f32
    // 3.14159... as f32
    let body = [
        Operator::F64Const {
            value: Ieee64::from(f64::from_bits(0x400921FB54442D18)), // PI
        },
        Operator::F32DemoteF64,
        Operator::I32ReinterpretF32,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn() -> i32>();
    // PI as f32 = 0x40490FDB
    assert_eq!(f() as u32, 0x40490FDB);
}

#[test]
fn test_f64_promote_f32() {
    // Test F64PromoteF32: convert f32 to f64
    // 3.0f as f64 = 3.0
    let body = [
        Operator::F32Const {
            value: Ieee32::from(f32::from_bits(0x40400000)), // 3.0
        },
        Operator::F64PromoteF32,
        Operator::I64ReinterpretF64,
        Operator::I64Const { value: 32 },
        Operator::I64ShrU,
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn() -> i32>();
    // 3.0 = 0x4008000000000000, upper 32 bits = 0x40080000
    assert_eq!(f() as u32, 0x40080000);
}

// Reinterpret Operations

#[test]
fn test_f32_reinterpret_i32() {
    // Test F32ReinterpretI32: reinterpret i32 bits as f32
    // 0x40400000 interpreted as f32 is 3.0
    let body = [
        Operator::I32Const {
            value: 0x40400000u32 as i32,
        },
        Operator::F32ReinterpretI32,
        // Add 1.0 to verify it's really a float
        Operator::F32Const {
            value: Ieee32::from(f32::from_bits(0x3F800000)), // 1.0
        },
        Operator::F32Add,
        Operator::I32ReinterpretF32,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn() -> i32>();
    // 3.0 + 1.0 = 4.0f = 0x40800000
    assert_eq!(f() as u32, 0x40800000);
}

#[test]
fn test_f64_reinterpret_i64() {
    // Test F64ReinterpretI64: reinterpret i64 bits as f64
    let body = [
        Operator::I64Const {
            value: 0x4008000000000000u64 as i64,
        }, // bits for 3.0
        Operator::F64ReinterpretI64,
        // Add 1.0 to verify it's really a float
        Operator::F64Const {
            value: Ieee64::from(f64::from_bits(0x3FF0000000000000)), // 1.0
        },
        Operator::F64Add,
        Operator::I64ReinterpretF64,
        Operator::I64Const { value: 32 },
        Operator::I64ShrU,
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn() -> i32>();
    // 3.0 + 1.0 = 4.0 = 0x4010000000000000, upper 32 bits = 0x40100000
    assert_eq!(f() as u32, 0x40100000);
}

// Combined Float Operations Test

#[test]
fn test_f32_combined_operations() {
    // Test a combined expression: (3.0 + 2.0) * 4.0 - 1.0 = 19.0
    let body = [
        Operator::F32Const {
            value: Ieee32::from(f32::from_bits(0x40400000)), // 3.0
        },
        Operator::F32Const {
            value: Ieee32::from(f32::from_bits(0x40000000)), // 2.0
        },
        Operator::F32Add,
        Operator::F32Const {
            value: Ieee32::from(f32::from_bits(0x40800000)), // 4.0
        },
        Operator::F32Mul,
        Operator::F32Const {
            value: Ieee32::from(f32::from_bits(0x3F800000)), // 1.0
        },
        Operator::F32Sub,
        Operator::I32ReinterpretF32,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn() -> i32>();
    // (3.0 + 2.0) * 4.0 - 1.0 = 5.0 * 4.0 - 1.0 = 20.0 - 1.0 = 19.0
    // 19.0f = 0x41980000
    assert_eq!(f() as u32, 0x41980000);
}

#[test]
fn test_f64_combined_operations() {
    // Test a combined expression: sqrt(16.0) + abs(-2.0) = 4.0 + 2.0 = 6.0
    let body = [
        Operator::F64Const {
            value: Ieee64::from(f64::from_bits(0x4030000000000000)), // 16.0
        },
        Operator::F64Sqrt,
        Operator::F64Const {
            value: Ieee64::from(f64::from_bits(0xC000000000000000)), // -2.0
        },
        Operator::F64Abs,
        Operator::F64Add,
        Operator::I64ReinterpretF64,
        Operator::I64Const { value: 32 },
        Operator::I64ShrU,
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");

    let f = func.as_fn::<fn() -> i32>();
    // sqrt(16.0) + abs(-2.0) = 4.0 + 2.0 = 6.0
    // 6.0 = 0x4018000000000000, upper 32 bits = 0x40180000
    assert_eq!(f() as u32, 0x40180000);
}

#[test]
fn test_iterative_fibonacci() {
    // Iterative Fibonacci: fib(n)
    // Function: (param i32) (result i32)
    // Computes the nth Fibonacci number iteratively
    //
    // Algorithm:
    // if n <= 1 return n
    // a = 0, b = 1
    // for i = 2 to n:
    //   temp = a + b
    //   a = b
    //   b = temp
    // return b
    //
    // locals:
    //   0: n (param)
    //   1: a (initialized to 0)
    //   2: b (will be set to 1 in the body)
    //   3: i (counter, initialized to 2)
    //   4: temp (for swap)

    let body = [
        // Initialize b = 1
        Operator::I32Const { value: 1 },
        Operator::LocalSet { local_index: 2 }, // b = 1
        // if n <= 1, return n
        Operator::LocalGet { local_index: 0 }, // get n
        Operator::I32Const { value: 1 },
        Operator::I32LeS, // n <= 1
        Operator::If {
            blockty: wasmparser::BlockType::Type(ValType::I32),
        },
        Operator::LocalGet { local_index: 0 }, // return n
        Operator::Return,
        Operator::End,
        // Initialize counter i = 2
        Operator::I32Const { value: 2 },
        Operator::LocalSet { local_index: 3 }, // i = 2
        // Loop: while i <= n
        Operator::Block {
            blockty: wasmparser::BlockType::Empty,
        },
        Operator::Loop {
            blockty: wasmparser::BlockType::Empty,
        },
        // Check if i > n, if so exit
        Operator::LocalGet { local_index: 3 }, // get i
        Operator::LocalGet { local_index: 0 }, // get n
        Operator::I32GtS,                      // i > n?
        Operator::BrIf { relative_depth: 1 },  // exit if true
        // temp = a + b
        Operator::LocalGet { local_index: 1 }, // get a
        Operator::LocalGet { local_index: 2 }, // get b
        Operator::I32Add,
        Operator::LocalSet { local_index: 4 }, // temp = a + b
        // a = b
        Operator::LocalGet { local_index: 2 }, // get b
        Operator::LocalSet { local_index: 1 }, // a = b
        // b = temp
        Operator::LocalGet { local_index: 4 }, // get temp
        Operator::LocalSet { local_index: 2 }, // b = temp
        // i++
        Operator::LocalGet { local_index: 3 }, // get i
        Operator::I32Const { value: 1 },
        Operator::I32Add,
        Operator::LocalSet { local_index: 3 }, // i = i + 1
        // continue loop
        Operator::Br { relative_depth: 0 },
        Operator::End, // end loop
        Operator::End, // end block
        // return b
        Operator::LocalGet { local_index: 2 },
        Operator::End,
    ];

    let func = compile_simple(
        &[ValType::I32],
        &[ValType::I32],
        &[
            ValType::I32, // a = 0
            ValType::I32, // b (will be set to 1)
            ValType::I32, // i counter
            ValType::I32, // temp
        ],
        &body,
    )
    .expect("Compilation failed");

    let f = func.as_fn::<fn(i32) -> i32>();

    // Test Fibonacci sequence:
    // fib(0) = 0
    // fib(1) = 1
    // fib(2) = 1
    // fib(3) = 2
    // fib(4) = 3
    // fib(5) = 5
    // fib(6) = 8
    // fib(7) = 13
    // fib(8) = 21
    // fib(9) = 34
    // fib(10) = 55

    const fn fib(n: i32) -> i32 {
        if n <= 0 {
            return 0;
        }
        if n == 1 {
            return 1;
        }
        let mut a = 0i32;
        let mut b = 1i32;
        let mut i = 2i32;
        while i <= n {
            let c = a + b;
            a = b;
            b = c;
            i += 1;
        }
        b
    }

    // Test cases
    unroll! {
        for i in 0..47 {
            assert_eq!(f(i as i32), const { fib(i as i32) });
        }
    }
}

#[test]
fn test_br_table_simple() {
    // Test br_table: simple switch-like branching
    // Function: (param i32) (result i32)
    // Returns different values based on the input parameter using br_table
    //
    // switch (param) {
    //   case 0: return 10;
    //   case 1: return 20;
    //   case 2: return 30;
    //   default: return 99;
    // }

    let wat = r#"
        (module
            (func (export "test") (param i32) (result i32)
                (block $default (result i32)
                    (block $case2 (result i32)
                        (block $case1 (result i32)
                            (block $case0 (result i32)
                                local.get 0
                                br_table $case0 $case1 $case2 $default
                            )
                            ;; case 0: return 10
                            i32.const 10
                            return
                        )
                        ;; case 1: return 20
                        i32.const 20
                        return
                    )
                    ;; case 2: return 30
                    i32.const 30
                    return
                )
                ;; default: return 99
                i32.const 99
            )
        )
    "#;

    let func = compile_from_wat_with_signature(wat).expect("Compilation failed");
    let f = func.as_fn::<fn(i32) -> i32>();

    assert_eq!(f(0), 10); // case 0
    assert_eq!(f(1), 20); // case 1
    assert_eq!(f(2), 30); // case 2
    assert_eq!(f(3), 99); // default (index == table length)
    assert_eq!(f(4), 99); // default (index > table length)
    assert_eq!(f(100), 99); // default (large index)
}

#[test]
fn test_br_table_with_computation() {
    // Test br_table with computation in each target
    // Function: (param i32 i32) (result i32)
    // Performs different operations based on the first parameter
    //
    // switch (param0) {
    //   case 0: return param1 + 10;
    //   case 1: return param1 * 2;
    //   case 2: return param1 - 5;
    //   default: return 0;
    // }

    let wat = r#"
        (module
            (func (export "test") (param i32 i32) (result i32)
                (block $default (result i32)
                    (block $case2 (result i32)
                        (block $case1 (result i32)
                            (block $case0 (result i32)
                                local.get 0
                                br_table $case0 $case1 $case2 $default
                            )
                            ;; case 0: param1 + 10
                            local.get 1
                            i32.const 10
                            i32.add
                            return
                        )
                        ;; case 1: param1 * 2
                        local.get 1
                        i32.const 2
                        i32.mul
                        return
                    )
                    ;; case 2: param1 - 5
                    local.get 1
                    i32.const 5
                    i32.sub
                    return
                )
                ;; default: return 0
                i32.const 0
            )
        )
    "#;

    let func = compile_from_wat_with_signature(wat).expect("Compilation failed");
    let f = func.as_fn::<fn(i32, i32) -> i32>();

    // Test case 0: add 10
    assert_eq!(f(0, 5), 15); // 5 + 10 = 15
    assert_eq!(f(0, 100), 110); // 100 + 10 = 110

    // Test case 1: multiply by 2
    assert_eq!(f(1, 5), 10); // 5 * 2 = 10
    assert_eq!(f(1, 100), 200); // 100 * 2 = 200

    // Test case 2: subtract 5
    assert_eq!(f(2, 10), 5); // 10 - 5 = 5
    assert_eq!(f(2, 100), 95); // 100 - 5 = 95

    // Test default case
    assert_eq!(f(3, 42), 0); // default
    assert_eq!(f(10, 999), 0); // default
}

#[test]
fn test_br_table_fallthrough() {
    // Test br_table where all cases branch to the same target (tests edge case)
    // Function: (param i32) (result i32)
    // All cases should return 42

    let wat = r#"
        (module
            (func (export "test") (param i32) (result i32)
                (block $outer (result i32)
                    (block $inner (result i32)
                        local.get 0
                        br_table $outer $outer $outer $outer
                    )
                    ;; This code is unreachable
                    i32.const 0
                    return
                )
                ;; All cases return 42
                i32.const 42
            )
        )
    "#;

    let func = compile_from_wat_with_signature(wat).expect("Compilation failed");
    let f = func.as_fn::<fn(i32) -> i32>();

    // All cases should return 42
    assert_eq!(f(0), 42);
    assert_eq!(f(1), 42);
    assert_eq!(f(2), 42);
    assert_eq!(f(3), 42);
    assert_eq!(f(100), 42);
}

// ============================================================================
// Additional I64 Operation Tests
// ============================================================================

#[test]
fn test_i64_mul() {
    // Test I64Mul: multiply two i64 values
    let body = [
        Operator::I64Const { value: 1000000 },
        Operator::I64Const { value: 1000000 },
        Operator::I64Mul,
        Operator::I32WrapI64, // Lower 32 bits
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");
    let f = func.as_fn::<fn() -> i32>();
    // 1000000 * 1000000 = 1000000000000, lower 32 bits = 0xE8D4A51000 & 0xFFFFFFFF = 0xD4A51000
    assert_eq!(f() as u32, 0xD4A51000);
}

#[test]
fn test_i64_div_s() {
    // Test I64DivS: signed division of i64 values
    let body = [
        Operator::I64Const { value: 100 },
        Operator::I64Const { value: 3 },
        Operator::I64DivS,
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");
    let f = func.as_fn::<fn() -> i32>();
    assert_eq!(f(), 33); // 100 / 3 = 33
}

#[test]
fn test_i64_div_s_negative() {
    // Test I64DivS with negative values
    let body = [
        Operator::I64Const { value: -100 },
        Operator::I64Const { value: 3 },
        Operator::I64DivS,
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");
    let f = func.as_fn::<fn() -> i32>();
    assert_eq!(f(), -33); // -100 / 3 = -33
}

#[test]
fn test_i64_div_u() {
    // Test I64DivU: unsigned division of i64 values
    let body = [
        Operator::I64Const { value: 100 },
        Operator::I64Const { value: 3 },
        Operator::I64DivU,
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");
    let f = func.as_fn::<fn() -> i32>();
    assert_eq!(f(), 33); // 100 / 3 = 33
}

#[test]
fn test_i64_rem_s() {
    // Test I64RemS: signed remainder of i64 values
    let body = [
        Operator::I64Const { value: 100 },
        Operator::I64Const { value: 3 },
        Operator::I64RemS,
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");
    let f = func.as_fn::<fn() -> i32>();
    assert_eq!(f(), 1); // 100 % 3 = 1
}

#[test]
fn test_i64_rem_s_negative() {
    // Test I64RemS with negative dividend
    let body = [
        Operator::I64Const { value: -100 },
        Operator::I64Const { value: 3 },
        Operator::I64RemS,
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");
    let f = func.as_fn::<fn() -> i32>();
    assert_eq!(f(), -1); // -100 % 3 = -1
}

#[test]
fn test_i64_rem_u() {
    // Test I64RemU: unsigned remainder of i64 values
    let body = [
        Operator::I64Const { value: 100 },
        Operator::I64Const { value: 7 },
        Operator::I64RemU,
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");
    let f = func.as_fn::<fn() -> i32>();
    assert_eq!(f(), 2); // 100 % 7 = 2
}

#[test]
fn test_i64_rotl() {
    // Test I64Rotl: rotate left
    let body = [
        Operator::I64Const {
            value: 0x123456789ABCDEFi64,
        },
        Operator::I64Const { value: 4 },
        Operator::I64Rotl,
        Operator::I32WrapI64, // Get lower 32 bits
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");
    let f = func.as_fn::<fn() -> i32>();
    // 0x0123456789ABCDEF rotated left by 4 = 0x123456789ABCDEF0
    // Lower 32 bits = 0x9ABCDEF0
    assert_eq!(f() as u32, 0x9ABCDEF0);
}

#[test]
fn test_i64_rotr() {
    // Test I64Rotr: rotate right
    let body = [
        Operator::I64Const {
            value: 0x123456789ABCDEFi64,
        },
        Operator::I64Const { value: 4 },
        Operator::I64Rotr,
        Operator::I32WrapI64, // Get lower 32 bits
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");
    let f = func.as_fn::<fn() -> i32>();
    // 0x0123456789ABCDEF rotated right by 4 = 0xF0123456789ABCDE
    // Lower 32 bits = 0x789ABCDE
    assert_eq!(f() as u32, 0x789ABCDE);
}

#[test]
fn test_i64_clz() {
    // Test I64Clz: count leading zeros
    let body = [
        Operator::I64Const {
            value: 0x0000000100000000i64,
        },
        Operator::I64Clz,
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");
    let f = func.as_fn::<fn() -> i32>();
    // 0x0000000100000000 has 31 leading zeros
    assert_eq!(f(), 31);
}

#[test]
fn test_i64_clz_zero() {
    // Test I64Clz with zero
    let body = [
        Operator::I64Const { value: 0 },
        Operator::I64Clz,
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");
    let f = func.as_fn::<fn() -> i32>();
    assert_eq!(f(), 64); // All zeros
}

#[test]
fn test_i64_ctz() {
    // Test I64Ctz: count trailing zeros
    let body = [
        Operator::I64Const {
            value: 0x0000000100000000i64,
        },
        Operator::I64Ctz,
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");
    let f = func.as_fn::<fn() -> i32>();
    // 0x0000000100000000 has 32 trailing zeros
    assert_eq!(f(), 32);
}

#[test]
fn test_i64_ctz_zero() {
    // Test I64Ctz with zero
    let body = [
        Operator::I64Const { value: 0 },
        Operator::I64Ctz,
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");
    let f = func.as_fn::<fn() -> i32>();
    assert_eq!(f(), 64); // All zeros
}

#[test]
fn test_i64_popcnt() {
    // Test I64Popcnt: population count (number of 1 bits)
    let body = [
        Operator::I64Const {
            value: 0xFFFFFFFFFFFFFFFFu64 as i64,
        },
        Operator::I64Popcnt,
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");
    let f = func.as_fn::<fn() -> i32>();
    assert_eq!(f(), 64); // All 64 bits set
}

#[test]
fn test_i64_popcnt_mixed() {
    // Test I64Popcnt with alternating bits
    let body = [
        Operator::I64Const {
            value: 0x5555555555555555u64 as i64,
        },
        Operator::I64Popcnt,
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");
    let f = func.as_fn::<fn() -> i32>();
    assert_eq!(f(), 32); // Half of 64 bits set
}

#[test]
fn test_i64_eqz() {
    // Test I64Eqz: check if zero
    let body = [
        Operator::I64Const { value: 0 },
        Operator::I64Eqz,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");
    let f = func.as_fn::<fn() -> i32>();
    assert_eq!(f(), 1); // 0 == 0 is true
}

#[test]
fn test_i64_eqz_nonzero() {
    // Test I64Eqz with non-zero
    let body = [
        Operator::I64Const { value: 42 },
        Operator::I64Eqz,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");
    let f = func.as_fn::<fn() -> i32>();
    assert_eq!(f(), 0); // 42 == 0 is false
}

#[test]
fn test_i64_eq() {
    // Test I64Eq: check equality
    let body = [
        Operator::I64Const {
            value: 0x123456789ABCDEFi64,
        },
        Operator::I64Const {
            value: 0x123456789ABCDEFi64,
        },
        Operator::I64Eq,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");
    let f = func.as_fn::<fn() -> i32>();
    assert_eq!(f(), 1); // Equal
}

#[test]
fn test_i64_ne() {
    // Test I64Ne: check inequality
    let body = [
        Operator::I64Const { value: 100 },
        Operator::I64Const { value: 200 },
        Operator::I64Ne,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");
    let f = func.as_fn::<fn() -> i32>();
    assert_eq!(f(), 1); // Not equal
}

#[test]
fn test_i64_lt_s() {
    // Test I64LtS: signed less than
    let body = [
        Operator::I64Const { value: -1 },
        Operator::I64Const { value: 1 },
        Operator::I64LtS,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");
    let f = func.as_fn::<fn() -> i32>();
    assert_eq!(f(), 1); // -1 < 1 (signed)
}

#[test]
fn test_i64_lt_u() {
    // Test I64LtU: unsigned less than
    let body = [
        Operator::I64Const { value: 1 },
        Operator::I64Const { value: -1i64 }, // Very large unsigned
        Operator::I64LtU,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");
    let f = func.as_fn::<fn() -> i32>();
    assert_eq!(f(), 1); // 1 < 0xFFFFFFFFFFFFFFFF (unsigned)
}

#[test]
fn test_i64_gt_s() {
    // Test I64GtS: signed greater than
    let body = [
        Operator::I64Const { value: 1 },
        Operator::I64Const { value: -1 },
        Operator::I64GtS,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");
    let f = func.as_fn::<fn() -> i32>();
    assert_eq!(f(), 1); // 1 > -1 (signed)
}

#[test]
fn test_i64_gt_u() {
    // Test I64GtU: unsigned greater than
    let body = [
        Operator::I64Const { value: -1i64 }, // Very large unsigned
        Operator::I64Const { value: 1 },
        Operator::I64GtU,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");
    let f = func.as_fn::<fn() -> i32>();
    assert_eq!(f(), 1); // 0xFFFFFFFFFFFFFFFF > 1 (unsigned)
}

#[test]
fn test_i64_le_s() {
    // Test I64LeS: signed less than or equal
    let body = [
        Operator::I64Const { value: 5 },
        Operator::I64Const { value: 5 },
        Operator::I64LeS,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");
    let f = func.as_fn::<fn() -> i32>();
    assert_eq!(f(), 1); // 5 <= 5
}

#[test]
fn test_i64_le_u() {
    // Test I64LeU: unsigned less than or equal
    let body = [
        Operator::I64Const { value: 100 },
        Operator::I64Const { value: 200 },
        Operator::I64LeU,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");
    let f = func.as_fn::<fn() -> i32>();
    assert_eq!(f(), 1); // 100 <= 200
}

#[test]
fn test_i64_ge_s() {
    // Test I64GeS: signed greater than or equal
    let body = [
        Operator::I64Const { value: 100 },
        Operator::I64Const { value: 100 },
        Operator::I64GeS,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");
    let f = func.as_fn::<fn() -> i32>();
    assert_eq!(f(), 1); // 100 >= 100
}

#[test]
fn test_i64_ge_u() {
    // Test I64GeU: unsigned greater than or equal
    let body = [
        Operator::I64Const { value: -1i64 }, // Very large unsigned
        Operator::I64Const { value: 1 },
        Operator::I64GeU,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");
    let f = func.as_fn::<fn() -> i32>();
    assert_eq!(f(), 1); // 0xFFFFFFFFFFFFFFFF >= 1 (unsigned)
}

// ============================================================================
// Additional Conversion Tests
// ============================================================================

#[test]
fn test_i64_trunc_f32_s() {
    // Test I64TruncF32S: truncate f32 to signed i64
    let body = [
        Operator::F32Const {
            value: Ieee32::from(f32::from_bits(0xC2C80000)), // -100.0f
        },
        Operator::I64TruncF32S,
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");
    let f = func.as_fn::<fn() -> i32>();
    assert_eq!(f(), -100);
}

#[test]
fn test_i64_trunc_f32_u() {
    // Test I64TruncF32U: truncate f32 to unsigned i64
    let body = [
        Operator::F32Const {
            value: Ieee32::from(f32::from_bits(0x42C80000)), // 100.0f
        },
        Operator::I64TruncF32U,
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");
    let f = func.as_fn::<fn() -> i32>();
    assert_eq!(f(), 100);
}

#[test]
fn test_i64_trunc_f64_u() {
    // Test I64TruncF64U: truncate f64 to unsigned i64
    let body = [
        Operator::F64Const {
            value: Ieee64::from(f64::from_bits(0x4059000000000000)), // 100.0
        },
        Operator::I64TruncF64U,
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");
    let f = func.as_fn::<fn() -> i32>();
    assert_eq!(f(), 100);
}

#[test]
fn test_i32_trunc_f64_u() {
    // Test I32TruncF64U: truncate f64 to unsigned i32
    let body = [
        Operator::F64Const {
            value: Ieee64::from(f64::from_bits(0x4059000000000000)), // 100.0
        },
        Operator::I32TruncF64U,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");
    let f = func.as_fn::<fn() -> i32>();
    assert_eq!(f(), 100);
}

#[test]
fn test_f32_convert_i64_s() {
    // Test F32ConvertI64S: convert signed i64 to f32
    let body = [
        Operator::I64Const { value: -1000000 },
        Operator::F32ConvertI64S,
        Operator::I32ReinterpretF32,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");
    let f = func.as_fn::<fn() -> i32>();
    // -1000000.0f = 0xC9742400
    assert_eq!(f() as u32, 0xC9742400);
}

#[test]
fn test_f32_convert_i64_u() {
    // Test F32ConvertI64U: convert unsigned i64 to f32
    let body = [
        Operator::I64Const { value: 1000000 },
        Operator::F32ConvertI64U,
        Operator::I32ReinterpretF32,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");
    let f = func.as_fn::<fn() -> i32>();
    // 1000000.0f = 0x49742400
    assert_eq!(f() as u32, 0x49742400);
}

#[test]
fn test_f64_convert_i32_u() {
    // Test F64ConvertI32U: convert unsigned i32 to f64
    let body = [
        Operator::I32Const {
            value: 0xFFFFFFFFu32 as i32,
        }, // Max u32
        Operator::F64ConvertI32U,
        Operator::I64ReinterpretF64,
        Operator::I64Const { value: 32 },
        Operator::I64ShrU,
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");
    let f = func.as_fn::<fn() -> i32>();
    // 4294967295.0 = 0x41EFFFFFFFE00000, upper 32 bits = 0x41EFFFFF
    assert_eq!(f() as u32, 0x41EFFFFF);
}

#[test]
fn test_f64_convert_i64_u() {
    // Test F64ConvertI64U: convert unsigned i64 to f64
    let body = [
        Operator::I64Const { value: 1000000 },
        Operator::F64ConvertI64U,
        Operator::I64ReinterpretF64,
        Operator::I64Const { value: 32 },
        Operator::I64ShrU,
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");
    let f = func.as_fn::<fn() -> i32>();
    // 1000000.0 = 0x412E848000000000, upper 32 bits = 0x412E8480
    assert_eq!(f() as u32, 0x412E8480);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

#[test]
fn test_i32_wrap_i64_edge() {
    // Test I32WrapI64 with a large i64 value
    let body = [
        Operator::I64Const {
            value: -4294967230i64, // 0xFFFFFFFF00000042 as signed i64
        },
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");
    let f = func.as_fn::<fn() -> i32>();
    assert_eq!(f(), 0x42); // Only lower 32 bits
}

#[test]
fn test_i64_extend_i32_s_edge() {
    // Test I64ExtendI32S with max negative
    let body = [
        Operator::I32Const {
            value: 0x80000000u32 as i32,
        }, // INT32_MIN
        Operator::I64ExtendI32S,
        Operator::I64Const { value: 32 },
        Operator::I64ShrU,
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");
    let f = func.as_fn::<fn() -> i32>();
    // INT32_MIN sign-extended should have all 1s in upper 32 bits
    assert_eq!(f(), -1);
}

#[test]
fn test_i64_extend_i32_u_edge() {
    // Test I64ExtendI32U with max value
    let body = [
        Operator::I32Const {
            value: 0xFFFFFFFFu32 as i32,
        }, // UINT32_MAX
        Operator::I64ExtendI32U,
        Operator::I64Const { value: 32 },
        Operator::I64ShrU,
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");
    let f = func.as_fn::<fn() -> i32>();
    // UINT32_MAX zero-extended should have all 0s in upper 32 bits
    assert_eq!(f(), 0);
}

// ============================================================================
// Float Edge Cases
// ============================================================================

#[test]
fn test_f32_neg_zero() {
    // Test F32Neg with negative zero
    let body = [
        Operator::F32Const {
            value: Ieee32::from(f32::from_bits(0x80000000)), // -0.0
        },
        Operator::F32Neg,
        Operator::I32ReinterpretF32,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");
    let f = func.as_fn::<fn() -> i32>();
    // neg(-0.0) = +0.0
    assert_eq!(f() as u32, 0x00000000);
}

#[test]
fn test_f64_neg_zero() {
    // Test F64Neg with negative zero
    let body = [
        Operator::F64Const {
            value: Ieee64::from(f64::from_bits(0x8000000000000000)), // -0.0
        },
        Operator::F64Neg,
        Operator::I64ReinterpretF64,
        Operator::I64Const { value: 32 },
        Operator::I64ShrU,
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");
    let f = func.as_fn::<fn() -> i32>();
    // neg(-0.0) = +0.0, upper 32 bits = 0x00000000
    assert_eq!(f() as u32, 0x00000000);
}

#[test]
fn test_f32_abs_negative_zero() {
    // Test F32Abs with negative zero
    let body = [
        Operator::F32Const {
            value: Ieee32::from(f32::from_bits(0x80000000)), // -0.0
        },
        Operator::F32Abs,
        Operator::I32ReinterpretF32,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");
    let f = func.as_fn::<fn() -> i32>();
    // abs(-0.0) = +0.0
    assert_eq!(f() as u32, 0x00000000);
}

#[test]
fn test_f64_abs_negative_zero() {
    // Test F64Abs with negative zero
    let body = [
        Operator::F64Const {
            value: Ieee64::from(f64::from_bits(0x8000000000000000)), // -0.0
        },
        Operator::F64Abs,
        Operator::I64ReinterpretF64,
        Operator::I64Const { value: 32 },
        Operator::I64ShrU,
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");
    let f = func.as_fn::<fn() -> i32>();
    // abs(-0.0) = +0.0, upper 32 bits = 0x00000000
    assert_eq!(f() as u32, 0x00000000);
}

#[test]
fn test_f32_copysign_zero_signs() {
    // Test F32Copysign with positive magnitude and negative sign
    let body = [
        Operator::F32Const {
            value: Ieee32::from(f32::from_bits(0x3F800000)), // 1.0
        },
        Operator::F32Const {
            value: Ieee32::from(f32::from_bits(0x80000000)), // -0.0
        },
        Operator::F32Copysign,
        Operator::I32ReinterpretF32,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");
    let f = func.as_fn::<fn() -> i32>();
    // copysign(1.0, -0.0) = -1.0
    assert_eq!(f() as u32, 0xBF800000);
}

#[test]
fn test_f64_copysign_zero_signs() {
    // Test F64Copysign with positive magnitude and negative sign
    let body = [
        Operator::F64Const {
            value: Ieee64::from(f64::from_bits(0x3FF0000000000000)), // 1.0
        },
        Operator::F64Const {
            value: Ieee64::from(f64::from_bits(0x8000000000000000)), // -0.0
        },
        Operator::F64Copysign,
        Operator::I64ReinterpretF64,
        Operator::I64Const { value: 32 },
        Operator::I64ShrU,
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");
    let f = func.as_fn::<fn() -> i32>();
    // copysign(1.0, -0.0) = -1.0, upper 32 bits = 0xBFF00000
    assert_eq!(f() as u32, 0xBFF00000);
}

// ============================================================================
// Integer Overflow/Underflow Tests
// ============================================================================

#[test]
fn test_i32_add_overflow() {
    // Test I32Add overflow wraps around
    let body = [
        Operator::I32Const {
            value: 0x7FFFFFFFi32,
        }, // INT32_MAX
        Operator::I32Const { value: 1 },
        Operator::I32Add,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");
    let f = func.as_fn::<fn() -> i32>();
    // INT32_MAX + 1 wraps to INT32_MIN
    assert_eq!(f(), i32::MIN);
}

#[test]
fn test_i32_sub_underflow() {
    // Test I32Sub underflow wraps around
    let body = [
        Operator::I32Const {
            value: 0x80000000u32 as i32,
        }, // INT32_MIN
        Operator::I32Const { value: 1 },
        Operator::I32Sub,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");
    let f = func.as_fn::<fn() -> i32>();
    // INT32_MIN - 1 wraps to INT32_MAX
    assert_eq!(f(), i32::MAX);
}

#[test]
fn test_i32_mul_overflow() {
    // Test I32Mul overflow wraps around
    let body = [
        Operator::I32Const { value: 0x10000 },
        Operator::I32Const { value: 0x10000 },
        Operator::I32Mul,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");
    let f = func.as_fn::<fn() -> i32>();
    // 0x10000 * 0x10000 = 0x100000000, wraps to 0
    assert_eq!(f(), 0);
}

#[test]
fn test_i64_add_overflow() {
    // Test I64Add overflow wraps around
    let body = [
        Operator::I64Const {
            value: 0x7FFFFFFFFFFFFFFFi64,
        }, // INT64_MAX
        Operator::I64Const { value: 1 },
        Operator::I64Add,
        Operator::I64Const { value: 32 },
        Operator::I64ShrU,
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");
    let f = func.as_fn::<fn() -> i32>();
    // INT64_MAX + 1 wraps to INT64_MIN, upper 32 bits = 0x80000000
    assert_eq!(f() as u32, 0x80000000);
}

#[test]
fn test_i64_sub_underflow() {
    // Test I64Sub underflow wraps around
    let body = [
        Operator::I64Const {
            value: 0x8000000000000000u64 as i64,
        }, // INT64_MIN
        Operator::I64Const { value: 1 },
        Operator::I64Sub,
        Operator::I64Const { value: 32 },
        Operator::I64ShrU,
        Operator::I32WrapI64,
        Operator::End,
    ];

    let func = compile_simple(&[], &[ValType::I32], &[], &body).expect("Compilation failed");
    let f = func.as_fn::<fn() -> i32>();
    // INT64_MIN - 1 wraps to INT64_MAX, upper 32 bits = 0x7FFFFFFF
    assert_eq!(f() as u32, 0x7FFFFFFF);
}
