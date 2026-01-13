use super::*;

#[test]
fn test_memory_new() {
    let mem = Memory::new(1, Some(10));
    assert_eq!(mem.size(), 1);
    assert_eq!(mem.byte_size(), PAGE_SIZE);
    assert_eq!(mem.max_pages(), Some(10));
}

#[test]
fn test_memory_grow() {
    let mut mem = Memory::new(1, Some(3));
    assert_eq!(mem.grow(1), 1);
    assert_eq!(mem.size(), 2);
    assert_eq!(mem.grow(1), 2);
    assert_eq!(mem.size(), 3);
    assert_eq!(mem.grow(1), -1); // exceeds max
    assert_eq!(mem.size(), 3);
}

#[test]
fn test_memory_read_write() {
    let mut mem = Memory::new(1, None);

    mem.write(0, &[1, 2, 3, 4]).unwrap();
    assert_eq!(mem.read(0, 4).unwrap(), &[1, 2, 3, 4]);

    mem.write_u32(100, 0xDEADBEEF).unwrap();
    assert_eq!(mem.read_u32(100).unwrap(), 0xDEADBEEF);
}

#[test]
fn test_memory_bounds() {
    let mem = Memory::new(1, None);
    assert!(mem.read(PAGE_SIZE, 1).is_err());
    assert!(mem.read(PAGE_SIZE - 1, 2).is_err());
}

#[test]
fn test_table_new() {
    let table = Table::new(10, Some(100), RefType::FuncRef);
    assert_eq!(table.size(), 10);
    assert_eq!(table.elem_type(), RefType::FuncRef);
}

#[test]
fn test_table_get_set() {
    let mut table = Table::new(10, None, RefType::FuncRef);
    assert_eq!(table.get(0).unwrap(), None);

    table.set(0, Some(FuncIdx(42))).unwrap();
    assert_eq!(table.get(0).unwrap(), Some(FuncIdx(42)));

    assert!(table.get(100).is_err());
    assert!(table.set(100, None).is_err());
}

#[test]
fn test_table_grow() {
    let mut table = Table::new(1, Some(3), RefType::FuncRef);
    assert_eq!(table.grow(1, None), 1);
    assert_eq!(table.size(), 2);
    assert_eq!(table.grow(1, Some(FuncIdx(5))), 2);
    assert_eq!(table.size(), 3);
    assert_eq!(table.get(2).unwrap(), Some(FuncIdx(5)));
    assert_eq!(table.grow(1, None), -1); // exceeds max
}

#[test]
fn test_global() {
    let mut global = Global::new(Value::I32(42), true);
    assert_eq!(global.get(), Value::I32(42));
    assert!(global.is_mutable());

    global.set(Value::I32(100)).unwrap();
    assert_eq!(global.get(), Value::I32(100));

    // Type mismatch
    assert!(global.set(Value::I64(1)).is_err());

    // Immutable global
    let mut immutable = Global::new(Value::I32(0), false);
    assert!(immutable.set(Value::I32(1)).is_err());
}

#[test]
fn test_store_alloc() {
    let engine = Arc::new(Engine::default());
    let mut store = Store::new(engine);

    let mem_idx = store.alloc_memory(1, Some(10));
    assert_eq!(mem_idx.0, 0);
    assert!(store.memory(mem_idx).is_some());

    let table_idx = store.alloc_table(10, None, RefType::FuncRef);
    assert_eq!(table_idx.0, 0);
    assert!(store.table(table_idx).is_some());

    let global_idx = store.alloc_global(Value::I32(42), true);
    assert_eq!(global_idx.0, 0);
    assert!(store.global(global_idx).is_some());
}

#[test]
fn test_host_func() {
    let func = Func::wrap(
        FuncType::new(vec![ValType::I32, ValType::I32], vec![ValType::I32]),
        |_store, args| {
            let a = args[0].unwrap_i32();
            let b = args[1].unwrap_i32();
            Ok(vec![Value::I32(a + b)])
        },
    );

    assert!(func.is_host());
    assert!(!func.is_wasm());
    assert_eq!(func.ty().params().len(), 2);
    assert_eq!(func.ty().results().len(), 1);
}
