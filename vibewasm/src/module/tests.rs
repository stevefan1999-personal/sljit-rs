use super::*;

#[test]
fn test_parse_empty_module() {
    let wasm = wat::parse_str("(module)").unwrap();
    let engine = Engine::default();
    let module = Module::new(&engine, &wasm).unwrap();

    assert!(module.func_types().is_empty());
    assert!(module.imports().is_empty());
    assert!(module.exports().is_empty());
    assert!(module.function_bodies().is_empty());
}

#[test]
fn test_parse_simple_function() {
    let wasm = wat::parse_str(
        r#"
            (module
                (func (export "add") (param i32 i32) (result i32)
                    local.get 0
                    local.get 1
                    i32.add
                )
            )
            "#,
    )
    .unwrap();

    let engine = Engine::default();
    let module = Module::new(&engine, &wasm).unwrap();

    assert_eq!(module.func_types().len(), 1);
    assert_eq!(module.function_bodies().len(), 1);
    assert_eq!(module.exports().len(), 1);

    let func_type = &module.func_types()[0];
    assert_eq!(func_type.params.len(), 2);
    assert_eq!(func_type.results.len(), 1);

    let export = module.get_func_export("add");
    assert!(export.is_some());
}

#[test]
fn test_parse_with_memory() {
    let wasm = wat::parse_str(
        r#"
            (module
                (memory (export "memory") 1 10)
            )
            "#,
    )
    .unwrap();

    let engine = Engine::default();
    let module = Module::new(&engine, &wasm).unwrap();

    assert_eq!(module.memories().len(), 1);
    assert_eq!(module.memories()[0].initial, 1);
    assert_eq!(module.memories()[0].maximum, Some(10));
}

#[test]
fn test_parse_with_globals() {
    let wasm = wat::parse_str(
        r#"
            (module
                (global $g (mut i32) (i32.const 42))
            )
            "#,
    )
    .unwrap();

    let engine = Engine::default();
    let module = Module::new(&engine, &wasm).unwrap();

    assert_eq!(module.globals().len(), 1);
    assert!(module.globals()[0].ty.mutable);
}

#[test]
fn test_parse_with_imports() {
    let wasm = wat::parse_str(
        r#"
            (module
                (import "env" "print" (func (param i32)))
                (import "env" "memory" (memory 1))
            )
            "#,
    )
    .unwrap();

    let engine = Engine::default();
    let module = Module::new(&engine, &wasm).unwrap();

    assert_eq!(module.imports().len(), 2);
    assert_eq!(module.num_imported_funcs(), 1);
    assert_eq!(module.num_imported_memories(), 1);
}
