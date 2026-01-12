//! Trampoline generation using libffi
//!
//! This module provides libffi-based trampolines for host functions.
//! Instead of generating JIT code that calls helper functions, we use libffi
//! closures that can capture state and be called with the C calling convention.

use libffi::high::{Closure0, Closure1, Closure2, Closure3};

use crate::store::{HostFuncCallback, get_current_store};
use crate::types::{FuncIdx, FuncType, ValType, Value};

/// Storage for libffi closures to keep them alive
///
/// libffi closures must be kept alive as long as the function pointer is used.
/// This enum holds closures with different arities.
#[allow(clippy::type_complexity)]
pub enum LibffiClosure {
    /// Closure with 0 arguments
    Arity0(Closure0<'static, usize>),
    /// Closure with 1 argument
    Arity1(Closure1<'static, usize, usize>),
    /// Closure with 2 arguments
    Arity2(Closure2<'static, usize, usize, usize>),
    /// Closure with 3 arguments
    Arity3(Closure3<'static, usize, usize, usize, usize>),
}

impl std::fmt::Debug for LibffiClosure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LibffiClosure::Arity0(_) => write!(f, "LibffiClosure::Arity0"),
            LibffiClosure::Arity1(_) => write!(f, "LibffiClosure::Arity1"),
            LibffiClosure::Arity2(_) => write!(f, "LibffiClosure::Arity2"),
            LibffiClosure::Arity3(_) => write!(f, "LibffiClosure::Arity3"),
        }
    }
}

/// Result of creating a libffi trampoline
pub struct LibffiTrampoline {
    /// The closure storage (must be kept alive)
    pub closure: LibffiClosure,
    /// The function pointer that can be called
    pub fn_ptr: usize,
}

impl std::fmt::Debug for LibffiTrampoline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LibffiTrampoline")
            .field("fn_ptr", &self.fn_ptr)
            .finish()
    }
}

/// Helper to convert word-sized (usize) arguments to Values based on function type
fn word_args_to_values(args: &[usize], param_types: &[ValType]) -> Vec<Value> {
    args.iter()
        .zip(param_types)
        .map(|(&arg, ty)| match ty {
            ValType::I32 => Value::I32(arg as i32),
            ValType::I64 => Value::I64(arg as i64),
            ValType::F32 => Value::F32(f32::from_bits(arg as u32)),
            ValType::F64 => Value::F64(f64::from_bits(arg as u64)),
            ValType::FuncRef => {
                if arg == usize::MAX {
                    Value::FuncRef(None)
                } else {
                    Value::FuncRef(Some(FuncIdx(arg as u32)))
                }
            }
            ValType::ExternRef => {
                if arg == usize::MAX {
                    Value::ExternRef(None)
                } else {
                    Value::ExternRef(Some(arg as u32))
                }
            }
        })
        .collect()
}

/// Helper to convert result Value to word-sized (usize) for JIT code
fn value_to_word(value: &Value) -> usize {
    match value {
        Value::I32(v) => *v as u32 as usize,
        Value::I64(v) => *v as usize,
        Value::F32(v) => f32::to_bits(*v) as usize,
        Value::F64(v) => f64::to_bits(*v) as usize,
        Value::FuncRef(Some(idx)) => idx.0 as usize,
        Value::FuncRef(None) => usize::MAX,
        Value::ExternRef(Some(idx)) => *idx as usize,
        Value::ExternRef(None) => usize::MAX,
    }
}

/// Create a libffi-based trampoline for a host function
///
/// This creates a libffi closure that:
/// 1. Gets the store from thread-local storage
/// 2. Looks up the host function by func_idx
/// 3. Converts word arguments to Values
/// 4. Calls the host callback
/// 5. Returns the result as a word
///
/// The closure is wrapped in a C-callable function pointer that can be
/// called directly from JIT code.
pub fn create_libffi_trampoline(
    func_idx: FuncIdx,
    func_type: &FuncType,
    callback: HostFuncCallback,
) -> Option<LibffiTrampoline> {
    let num_params = func_type.params().len();
    let param_types: Vec<ValType> = func_type.params().to_vec();

    // Clone the callback for the closure
    let callback = callback.clone();

    match num_params {
        0 => {
            let param_types = param_types.clone();
            let closure_fn =
                move || -> usize { call_host_func(func_idx, &callback, &[], &param_types) };

            // Box and leak to get 'static lifetime
            let boxed = Box::new(closure_fn);
            let leaked: &'static mut _ = Box::leak(boxed);

            let closure = Closure0::new(leaked);
            // Extract the function pointer address exactly like the libffi example
            // The code_ptr() returns an FnPtr type which wraps a raw code pointer
            let fn_ptr = closure.code_ptr();
            let fn_ptr_addr: usize = unsafe { std::ptr::read(fn_ptr as *const _ as *const usize) };

            Some(LibffiTrampoline {
                closure: LibffiClosure::Arity0(closure),
                fn_ptr: fn_ptr_addr,
            })
        }
        1 => {
            let param_types = param_types.clone();
            let closure_fn = move |arg0: usize| -> usize {
                call_host_func(func_idx, &callback, &[arg0], &param_types)
            };

            let boxed = Box::new(closure_fn);
            let leaked: &'static mut _ = Box::leak(boxed);

            let closure = Closure1::new(leaked);
            let fn_ptr = closure.code_ptr();
            let fn_ptr_addr: usize = unsafe { std::ptr::read(fn_ptr as *const _ as *const usize) };

            Some(LibffiTrampoline {
                closure: LibffiClosure::Arity1(closure),
                fn_ptr: fn_ptr_addr,
            })
        }
        2 => {
            let param_types = param_types.clone();
            let closure_fn = move |arg0: usize, arg1: usize| -> usize {
                call_host_func(func_idx, &callback, &[arg0, arg1], &param_types)
            };

            let boxed = Box::new(closure_fn);
            let leaked: &'static mut _ = Box::leak(boxed);

            let closure = Closure2::new(leaked);
            let fn_ptr = closure.code_ptr();
            let fn_ptr_addr: usize = unsafe { std::ptr::read(fn_ptr as *const _ as *const usize) };

            Some(LibffiTrampoline {
                closure: LibffiClosure::Arity2(closure),
                fn_ptr: fn_ptr_addr,
            })
        }
        3 => {
            let param_types = param_types.clone();
            let closure_fn = move |arg0: usize, arg1: usize, arg2: usize| -> usize {
                call_host_func(func_idx, &callback, &[arg0, arg1, arg2], &param_types)
            };

            let boxed = Box::new(closure_fn);
            let leaked: &'static mut _ = Box::leak(boxed);

            let closure = Closure3::new(leaked);
            let fn_ptr = closure.code_ptr();
            let fn_ptr_addr: usize = unsafe { std::ptr::read(fn_ptr as *const _ as *const usize) };

            Some(LibffiTrampoline {
                closure: LibffiClosure::Arity3(closure),
                fn_ptr: fn_ptr_addr,
            })
        }
        _ => {
            // More than 3 arguments not supported yet
            None
        }
    }
}

/// Internal helper to call a host function
///
/// This is called from within the libffi closure and handles:
/// - Getting the store from TLS
/// - Converting arguments
/// - Calling the callback
/// - Converting the result
fn call_host_func(
    _func_idx: FuncIdx,
    callback: &HostFuncCallback,
    word_args: &[usize],
    param_types: &[ValType],
) -> usize {
    // Get store from TLS
    let store_ptr = match get_current_store() {
        Some(ptr) => ptr,
        None => return 0, // Error: no store context
    };
    let store = unsafe { &mut *store_ptr };

    // Convert word arguments to Values
    let args = word_args_to_values(word_args, param_types);

    // Call the host callback
    match callback(store, &args) {
        Ok(results) => {
            if results.is_empty() {
                0
            } else {
                value_to_word(&results[0])
            }
        }
        Err(_trap) => 0, // Error: trap occurred
    }
}

#[cfg(test)]
mod tests;
