use super::*;

#[test]
fn test_engine_default() {
    let engine = Engine::default();
    assert!(engine.features().multi_value);
    assert!(!engine.features().simd);
}
