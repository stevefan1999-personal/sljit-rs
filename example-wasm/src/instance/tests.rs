use super::*;
use crate::Engine;
use std::sync::Arc;

fn setup() -> (Arc<Engine>, Store) {
    let engine = Arc::new(Engine::default());
    let store = Store::new(engine.clone());
    (engine, store)
}

#[test]
fn test_linker_new() {
    let linker = Linker::new();
    assert!(linker.resolve_func("test", "func").is_none());
}

#[test]
fn test_linker_define_func() {
    let (engine, mut store) = setup();
    let mut linker = Linker::new();

    let func = Func::wrap(
        FuncType::new(vec![], vec![ValType::I32]),
        |_store, _args| Ok(vec![Value::I32(42)]),
    );
    let func_idx = store.alloc_func(func);

    linker.define_func("env", "get_value", func_idx);
    assert_eq!(linker.resolve_func("env", "get_value"), Some(func_idx));
    assert!(linker.resolve_func("env", "other").is_none());
}

#[test]
fn test_linker_func_wrap() {
    let (engine, mut store) = setup();
    let mut linker = Linker::new();

    linker.func_wrap(
        &mut store,
        "env",
        "add",
        FuncType::new(vec![ValType::I32, ValType::I32], vec![ValType::I32]),
        |_store, args| {
            let a = args[0].unwrap_i32();
            let b = args[1].unwrap_i32();
            Ok(vec![Value::I32(a + b)])
        },
    );

    let func_idx = linker.resolve_func("env", "add").unwrap();
    let result = store
        .call(func_idx, &[Value::I32(10), Value::I32(20)])
        .unwrap();
    assert_eq!(result, vec![Value::I32(30)]);
}

#[test]
fn test_instantiate_empty_module() {
    let (engine, mut store) = setup();
    let linker = Linker::new();

    let wasm = wat::parse_str("(module)").unwrap();
    let module = Module::new(&engine, &wasm).unwrap();

    let instance = linker.instantiate(&mut store, &module).unwrap();
    assert!(instance.funcs().is_empty());
    assert!(instance.memories().is_empty());
}

#[test]
fn test_instantiate_simple_function() {
    let (engine, mut store) = setup();
    let linker = Linker::new();

    let wasm = wat::parse_str(
        r#"
            (module
                (func (export "answer") (result i32)
                    i32.const 42
                )
            )
            "#,
    )
    .unwrap();

    let module = Module::new(&engine, &wasm).unwrap();
    let instance = linker.instantiate(&mut store, &module).unwrap();

    let func_idx = instance.get_func(&store, "answer").unwrap();
    let result = store.call(func_idx, &[]).unwrap();
    assert_eq!(result, vec![Value::I32(42)]);
}

#[test]
fn test_instantiate_with_memory() {
    let (engine, mut store) = setup();
    let linker = Linker::new();

    let wasm = wat::parse_str(
        r#"
            (module
                (memory (export "memory") 1 10)
            )
            "#,
    )
    .unwrap();

    let module = Module::new(&engine, &wasm).unwrap();
    let instance = linker.instantiate(&mut store, &module).unwrap();

    let mem_idx = instance.get_memory("memory").unwrap();
    let memory = store.memory(mem_idx).unwrap();
    assert_eq!(memory.size(), 1);
}

#[test]
fn test_instantiate_with_import() {
    let (engine, mut store) = setup();
    let mut linker = Linker::new();

    // Define the imported function
    linker.func_wrap(
        &mut store,
        "env",
        "imported_add",
        FuncType::new(vec![ValType::I32, ValType::I32], vec![ValType::I32]),
        |_store, args| {
            let a = args[0].unwrap_i32();
            let b = args[1].unwrap_i32();
            Ok(vec![Value::I32(a + b)])
        },
    );

    let wasm = wat::parse_str(
        r#"
            (module
                (import "env" "imported_add" (func $add (param i32 i32) (result i32)))
                (func (export "test") (result i32)
                    i32.const 10
                    i32.const 20
                    call $add
                )
            )
            "#,
    )
    .unwrap();

    let module = Module::new(&engine, &wasm).unwrap();
    let instance = linker.instantiate(&mut store, &module).unwrap();

    let func_idx = instance.get_func(&store, "test").unwrap();
    let result = store.call(func_idx, &[]).unwrap();
    assert_eq!(result, vec![Value::I32(30)]);
}

#[test]
fn test_missing_import() {
    let (engine, mut store) = setup();
    let linker = Linker::new();

    let wasm = wat::parse_str(
        r#"
            (module
                (import "env" "missing" (func (param i32)))
            )
            "#,
    )
    .unwrap();

    let module = Module::new(&engine, &wasm).unwrap();
    let result = linker.instantiate(&mut store, &module);

    assert!(matches!(
        result,
        Err(InstantiationError::ImportNotFound { .. })
    ));
}

#[test]
fn test_wasm_calls_wasm_simple() {
    // Test: one wasm function calls another wasm function in the same module
    let (engine, mut store) = setup();
    let linker = Linker::new();

    let wasm = wat::parse_str(
        r#"
            (module
                ;; Helper function that doubles a number
                (func $double (param i32) (result i32)
                    local.get 0
                    i32.const 2
                    i32.mul
                )
                
                ;; Main function that calls $double
                (func (export "test") (param i32) (result i32)
                    local.get 0
                    call $double
                )
            )
            "#,
    )
    .unwrap();

    let module = Module::new(&engine, &wasm).unwrap();
    let instance = linker.instantiate(&mut store, &module).unwrap();

    let func_idx = instance.get_func(&store, "test").unwrap();

    // test(5) should call double(5) which returns 10
    let result = store.call(func_idx, &[Value::I32(5)]).unwrap();
    assert_eq!(result, vec![Value::I32(10)]);

    // test(21) should return 42
    let result = store.call(func_idx, &[Value::I32(21)]).unwrap();
    assert_eq!(result, vec![Value::I32(42)]);
}

#[test]
fn test_wasm_calls_wasm_chain() {
    // Test: wasm function A calls B calls C (chain of calls)
    let (engine, mut store) = setup();
    let linker = Linker::new();

    let wasm = wat::parse_str(
        r#"
            (module
                ;; Add 1 to the input
                (func $add_one (param i32) (result i32)
                    local.get 0
                    i32.const 1
                    i32.add
                )
                
                ;; Call add_one twice (adds 2)
                (func $add_two (param i32) (result i32)
                    local.get 0
                    call $add_one
                    call $add_one
                )
                
                ;; Main function: calls add_two (which calls add_one twice)
                (func (export "test") (param i32) (result i32)
                    local.get 0
                    call $add_two
                )
            )
            "#,
    )
    .unwrap();

    let module = Module::new(&engine, &wasm).unwrap();
    let instance = linker.instantiate(&mut store, &module).unwrap();

    let func_idx = instance.get_func(&store, "test").unwrap();

    // test(10) should return 12 (10 + 1 + 1)
    let result = store.call(func_idx, &[Value::I32(10)]).unwrap();
    assert_eq!(result, vec![Value::I32(12)]);

    // test(40) should return 42
    let result = store.call(func_idx, &[Value::I32(40)]).unwrap();
    assert_eq!(result, vec![Value::I32(42)]);
}

#[test]
fn test_wasm_calls_wasm_multiple_params() {
    // Test: wasm function calls another with multiple parameters
    let (engine, mut store) = setup();
    let linker = Linker::new();

    let wasm = wat::parse_str(
        r#"
            (module
                ;; Add two numbers
                (func $add (param i32 i32) (result i32)
                    local.get 0
                    local.get 1
                    i32.add
                )
                
                ;; Subtract: a - b
                (func $sub (param i32 i32) (result i32)
                    local.get 0
                    local.get 1
                    i32.sub
                )
                
                ;; Compute (a + b) - c by calling $add and $sub
                (func (export "compute") (param i32 i32 i32) (result i32)
                    local.get 0
                    local.get 1
                    call $add      ;; stack: (a + b)
                    local.get 2
                    call $sub      ;; stack: (a + b) - c
                )
            )
            "#,
    )
    .unwrap();

    let module = Module::new(&engine, &wasm).unwrap();
    let instance = linker.instantiate(&mut store, &module).unwrap();

    let func_idx = instance.get_func(&store, "compute").unwrap();

    // compute(10, 20, 5) = (10 + 20) - 5 = 25
    let result = store
        .call(func_idx, &[Value::I32(10), Value::I32(20), Value::I32(5)])
        .unwrap();
    assert_eq!(result, vec![Value::I32(25)]);

    // compute(50, 10, 18) = (50 + 10) - 18 = 42
    let result = store
        .call(func_idx, &[Value::I32(50), Value::I32(10), Value::I32(18)])
        .unwrap();
    assert_eq!(result, vec![Value::I32(42)]);
}

#[test]
fn test_wasm_recursive_call() {
    // Test: recursive wasm function (factorial)
    let (engine, mut store) = setup();
    let linker = Linker::new();

    let wasm = wat::parse_str(
        r#"
            (module
                ;; Factorial function: n! = n * (n-1)!
                (func $factorial (export "factorial") (param i32) (result i32)
                    (if (result i32) (i32.le_s (local.get 0) (i32.const 1))
                        (then
                            (i32.const 1)
                        )
                        (else
                            (i32.mul
                                (local.get 0)
                                (call $factorial
                                    (i32.sub (local.get 0) (i32.const 1))
                                )
                            )
                        )
                    )
                )
            )
            "#,
    )
    .unwrap();

    let module = Module::new(&engine, &wasm).unwrap();
    let instance = linker.instantiate(&mut store, &module).unwrap();

    let func_idx = instance.get_func(&store, "factorial").unwrap();

    // factorial(1) = 1
    let result = store.call(func_idx, &[Value::I32(1)]).unwrap();
    assert_eq!(result, vec![Value::I32(1)]);

    // factorial(5) = 120
    let result = store.call(func_idx, &[Value::I32(5)]).unwrap();
    assert_eq!(result, vec![Value::I32(120)]);

    // factorial(6) = 720
    let result = store.call(func_idx, &[Value::I32(6)]).unwrap();
    assert_eq!(result, vec![Value::I32(720)]);
}

#[test]
fn test_wasm_mutual_recursion() {
    // Test: two wasm functions calling each other (mutual recursion)
    // is_even(n) calls is_odd(n-1), is_odd(n) calls is_even(n-1)
    let (engine, mut store) = setup();
    let linker = Linker::new();

    let wasm = wat::parse_str(
        r#"
            (module
                ;; is_even: returns 1 if n is even, 0 otherwise
                (func $is_even (export "is_even") (param i32) (result i32)
                    (if (result i32) (i32.eqz (local.get 0))
                        (then (i32.const 1))
                        (else (call $is_odd (i32.sub (local.get 0) (i32.const 1))))
                    )
                )
                
                ;; is_odd: returns 1 if n is odd, 0 otherwise
                (func $is_odd (export "is_odd") (param i32) (result i32)
                    (if (result i32) (i32.eqz (local.get 0))
                        (then (i32.const 0))
                        (else (call $is_even (i32.sub (local.get 0) (i32.const 1))))
                    )
                )
            )
            "#,
    )
    .unwrap();

    let module = Module::new(&engine, &wasm).unwrap();
    let instance = linker.instantiate(&mut store, &module).unwrap();

    let is_even_idx = instance.get_func(&store, "is_even").unwrap();
    let is_odd_idx = instance.get_func(&store, "is_odd").unwrap();

    // is_even(0) = 1
    let result = store.call(is_even_idx, &[Value::I32(0)]).unwrap();
    assert_eq!(result, vec![Value::I32(1)]);

    // is_even(4) = 1
    let result = store.call(is_even_idx, &[Value::I32(4)]).unwrap();
    assert_eq!(result, vec![Value::I32(1)]);

    // is_even(5) = 0
    let result = store.call(is_even_idx, &[Value::I32(5)]).unwrap();
    assert_eq!(result, vec![Value::I32(0)]);

    // is_odd(3) = 1
    let result = store.call(is_odd_idx, &[Value::I32(3)]).unwrap();
    assert_eq!(result, vec![Value::I32(1)]);

    // is_odd(4) = 0
    let result = store.call(is_odd_idx, &[Value::I32(4)]).unwrap();
    assert_eq!(result, vec![Value::I32(0)]);
}
