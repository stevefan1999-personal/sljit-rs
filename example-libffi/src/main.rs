//! # SLJIT + libffi Integration Example
//!
//! This example demonstrates using SLJIT to generate JIT code that calls back
//! into a Rust closure via libffi.
//!
//! The flow is:
//! 1. Create a Rust closure that captures some state
//! 2. Use libffi to wrap the closure and get a C-compatible function pointer
//! 3. Use SLJIT to generate machine code that:
//!    - Receives two arguments
//!    - Calls the libffi-wrapped closure with those arguments
//!    - Returns the result
//! 4. Execute the JIT-generated code

use std::mem::transmute;

use libffi::high::Closure2;
use sljit::sys::{
    Compiler, GeneratedCode, SLJIT_ARG_TYPE_W, SLJIT_CALL, SLJIT_IMM, SLJIT_MOV, SLJIT_R0,
    SLJIT_R1, SLJIT_S0, SLJIT_S1, arg_types, sljit_sw,
};

/// Simple test function to verify JIT calling works
extern "C" fn test_add(a: u64, b: u64) -> u64 {
    println!("  [extern C fn] Called with a={}, b={}", a, b);
    a + b
}

fn main() {
    // First, let's verify SLJIT can call an extern "C" function
    println!("=== Test 1: SLJIT calling extern \"C\" function ===");
    test_with_extern_c();

    println!("\n=== Test 2: SLJIT calling libffi-wrapped closure ===");
    test_with_libffi_closure();
}

fn test_with_extern_c() {
    let fn_ptr_addr = test_add as *const () as sljit_sw;
    println!("extern C function pointer: 0x{:x}", fn_ptr_addr);

    // Test direct call
    println!("Direct call: test_add(3, 4) = {}", test_add(3, 4));

    // Create JIT wrapper
    let (jit_fn, _jit_code) = create_jit_wrapper(fn_ptr_addr);

    // Call via JIT
    println!("JIT call: jit_fn(5, 7) = {}", jit_fn(5, 7));
    assert_eq!(jit_fn(5, 7), 12);
    println!("extern C test passed!");
}

fn test_with_libffi_closure() {
    // Step 1: Create a Rust closure that captures state
    let multiplier = 10u64;
    let callback = move |a: u64, b: u64| -> u64 {
        println!(
            "  [Rust closure] Called with a={}, b={}, multiplier={}",
            a, b, multiplier
        );
        (a + b) * multiplier
    };

    // Step 2: Wrap the closure with libffi to get a C-compatible function pointer
    let closure = Closure2::new(&callback);
    let fn_ptr = closure.code_ptr();

    // Get the raw function pointer address for SLJIT to call
    // FnPtr2 internally stores an FnPtrUntyped which contains a CodePtr (*mut c_void)
    // Since FnPtr2 and FnPtrUntyped are essentially wrappers with PhantomData (zero-sized),
    // the memory layout is just the pointer itself
    let fn_ptr_addr: sljit_sw = unsafe {
        // Read the first pointer-sized value from FnPtr2, which is the CodePtr
        std::ptr::read(fn_ptr as *const _ as *const usize) as sljit_sw
    };
    println!("libffi closure function pointer: 0x{:x}", fn_ptr_addr);

    // Create a callable function pointer from the raw address
    let raw_fn: extern "C" fn(u64, u64) -> u64 = unsafe { transmute(fn_ptr_addr as usize) };

    // Verify the closure works directly first
    println!(
        "Direct closure call via code_ptr().call: {} + {} = {}",
        3,
        4,
        fn_ptr.call(3, 4)
    );

    // Also test via raw function pointer
    println!("Direct call via raw fn: raw_fn(3, 4) = {}", raw_fn(3, 4));

    // Step 3: Create SLJIT JIT compiler and generate code
    println!("\nGenerating JIT code...");
    let (jit_fn, _jit_code) = create_jit_wrapper(fn_ptr_addr);

    // Step 4: Execute the JIT-generated code
    println!("\nCalling JIT-generated code that calls back to Rust closure:");
    let result = jit_fn(5, 7);
    println!("JIT function returned: {}", result);

    assert_eq!(result, (5 + 7) * 10);
    println!("\nSuccess! JIT -> libffi -> Rust closure callback works!");

    // Test with different values
    println!("\n--- Additional tests ---");
    let result2 = jit_fn(100, 200);
    println!("jit_fn(100, 200) = {}", result2);
    assert_eq!(result2, (100 + 200) * 10);

    let result3 = jit_fn(0, 0);
    println!("jit_fn(0, 0) = {}", result3);
    assert_eq!(result3, 0);

    println!("\nAll tests passed!");
}

/// Creates a JIT-compiled wrapper function that calls the given function pointer
/// with two u64 arguments and returns a u64 result.
///
/// The generated code essentially does:
/// ```text
/// fn jit_wrapper(a: u64, b: u64) -> u64 {
///     return fn_ptr(a, b);
/// }
/// ```
///
/// Returns a tuple of (function_pointer, generated_code).
/// The GeneratedCode must be kept alive as long as the function pointer is used.
fn create_jit_wrapper(fn_ptr_addr: sljit_sw) -> (fn(u64, u64) -> u64, GeneratedCode) {
    let mut compiler = Compiler::new();

    // Set up the function:
    // - arg_types!([W, W] -> W): Two word arguments, returns a word
    // - 2 scratch registers (R0, R1 for passing arguments to callee)
    // - 2 saved registers (S0, S1 for the two incoming arguments)
    // - 0 local stack space
    compiler
        .emit_enter(0, arg_types!([W, W] -> W), 2, 2, 0)
        .expect("Failed to emit enter");

    // Move arguments to scratch registers for the call
    // S0 contains first argument, S1 contains second argument
    // For SLJIT_CALL convention, arguments go in R0, R1, R2, ...
    compiler
        .emit_op1(SLJIT_MOV, SLJIT_R0, 0, SLJIT_S0, 0)
        .expect("Failed to move first argument");

    compiler
        .emit_op1(SLJIT_MOV, SLJIT_R1, 0, SLJIT_S1, 0)
        .expect("Failed to move second argument");

    // Call the libffi closure function pointer
    // SLJIT_CALL: Use the C calling convention
    // arg_types!([W, W] -> W): The callee takes two words and returns a word
    // SLJIT_IMM: The function address is an immediate value
    // fn_ptr_addr: The actual function pointer address
    compiler
        .emit_icall(SLJIT_CALL, arg_types!([W, W] -> W), SLJIT_IMM, fn_ptr_addr)
        .expect("Failed to emit icall");

    // Return the result (already in R0 after the call per calling convention)
    compiler
        .emit_return(SLJIT_MOV, SLJIT_R0, 0)
        .expect("Failed to emit return");

    // Generate the executable code
    let code = compiler.generate_code();

    println!("JIT code generated at: 0x{:x}", code.get() as usize);

    // Convert to a callable function pointer
    // SAFETY: We're trusting that SLJIT generated valid code with the correct
    // calling convention for our function signature
    let fn_ptr = unsafe { transmute(code.get()) };

    // Wrap in ManuallyDrop to prevent the code from being freed
    // The caller is responsible for keeping this alive
    (fn_ptr, code)
}
