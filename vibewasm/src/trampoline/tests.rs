use super::*;
use crate::types::FuncType;
use std::sync::Arc;

#[test]
fn test_create_trampoline_0_args() {
    let func_type = FuncType::new(vec![], vec![ValType::I32]);
    let callback: HostFuncCallback = Arc::new(|_store, _args| Ok(vec![Value::I32(42)]));

    let trampoline = create_libffi_trampoline(FuncIdx(0), &func_type, callback);
    assert!(trampoline.is_some());

    let trampoline = trampoline.unwrap();
    assert!(trampoline.fn_ptr != 0);
}

#[test]
fn test_create_trampoline_1_arg() {
    let func_type = FuncType::new(vec![ValType::I32], vec![ValType::I32]);
    let callback: HostFuncCallback = Arc::new(|_store, args| {
        let x = args[0].unwrap_i32();
        Ok(vec![Value::I32(x * 2)])
    });

    let trampoline = create_libffi_trampoline(FuncIdx(0), &func_type, callback);
    assert!(trampoline.is_some());
}

#[test]
fn test_create_trampoline_2_args() {
    let func_type = FuncType::new(vec![ValType::I32, ValType::I32], vec![ValType::I32]);
    let callback: HostFuncCallback = Arc::new(|_store, args| {
        let a = args[0].unwrap_i32();
        let b = args[1].unwrap_i32();
        Ok(vec![Value::I32(a + b)])
    });

    let trampoline = create_libffi_trampoline(FuncIdx(0), &func_type, callback);
    assert!(trampoline.is_some());
}
