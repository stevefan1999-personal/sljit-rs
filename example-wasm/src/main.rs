use std::sync::Arc;

use example_wasm::Value;
use example_wasm::{Linker, Module, Store, engine::Engine};

#[crabtime::expression]
fn compile_wasm(src: String) {
    use itertools::Itertools;
    let result = wat::parse_str(src).unwrap().into_iter().format(",");
    crabtime::output! {
        &[{{result}}]
    }
}

const SOURCE: &str = r#"
        (module
            (func $is_even (param i32) (result i32)
                local.get 0
                i32.const 2
                i32.rem_s
                i32.eqz)

            (func $is_odd (param i32) (result i32)
                local.get 0
                i32.const 2
                i32.rem_s
                i32.const 1
                i32.eqz)
            
            (export "is_even" (func $is_even))
            (export "is_odd" (func $is_odd))
        )
        "#;

const SOURCE_WASM: &[u8] = compile_wasm!(
    r#"
        (module
            (func $is_even (param i32) (result i32)
                local.get 0
                i32.const 2
                i32.rem_s
                i32.eqz)

            (func $is_odd (param i32) (result i32)
                local.get 0
                i32.const 2
                i32.rem_s
                i32.const 1
                i32.eqz)
            
            (export "is_even" (func $is_even))
            (export "is_odd" (func $is_odd))
        )
        "#
);

fn main() {
    let engine = Arc::new(Engine::default());
    let linker = Linker::new();
    let mut store = Store::new(engine.clone());

    // let wasm = wat::parse_str(SOURCE).unwrap();

    let module = Module::new(&engine, &SOURCE_WASM).unwrap();
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
