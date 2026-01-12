use super::*;

#[test]
fn test_val_type_properties() {
    assert!(ValType::I32.is_num());
    assert!(ValType::I32.is_int());
    assert!(!ValType::I32.is_float());
    assert!(!ValType::I32.is_ref());

    assert!(ValType::F64.is_num());
    assert!(ValType::F64.is_float());
    assert!(!ValType::F64.is_int());

    assert!(ValType::FuncRef.is_ref());
    assert!(!ValType::FuncRef.is_num());
}

#[test]
fn test_value_types() {
    let v = Value::I32(42);
    assert_eq!(v.ty(), ValType::I32);
    assert_eq!(v.as_i32(), Some(42));
    assert_eq!(v.as_i64(), None);

    let v = Value::F64(3.14);
    assert_eq!(v.ty(), ValType::F64);
    assert_eq!(v.as_f64(), Some(3.14));
}

#[test]
fn test_func_type() {
    let ft = FuncType::new(vec![ValType::I32, ValType::I32], vec![ValType::I32]);
    assert_eq!(ft.params().len(), 2);
    assert_eq!(ft.results().len(), 1);
    assert_eq!(ft.to_string(), "(i32, i32) -> (i32)");
}

#[test]
fn test_trap_creation() {
    let trap = Trap::unreachable();
    assert_eq!(trap.kind, TrapKind::Unreachable);
    assert!(trap.message.contains("unreachable"));
}
