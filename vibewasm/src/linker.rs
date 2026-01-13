use std::collections::HashMap;

use crate::Instance;
use crate::error::InstantiationError;
use crate::module::Module;
use crate::store::{Func, Store};
use crate::types::{FuncIdx, FuncType, GlobalIdx, MemoryIdx, TableIdx, Trap, Value};

/// Helper for building import resolvers
#[derive(Debug, Default)]
pub struct Linker {
    /// Registered functions by (module, name)
    funcs: HashMap<(String, String), FuncIdx>,
    /// Registered memories by (module, name)
    memories: HashMap<(String, String), MemoryIdx>,
    /// Registered tables by (module, name)
    tables: HashMap<(String, String), TableIdx>,
    /// Registered globals by (module, name)
    globals: HashMap<(String, String), GlobalIdx>,
}

impl Linker {
    /// Create a new empty linker
    pub fn new() -> Self {
        Self::default()
    }

    /// Define a function
    pub fn define_func(
        &mut self,
        module: impl Into<String>,
        name: impl Into<String>,
        func_idx: FuncIdx,
    ) -> &mut Self {
        self.funcs.insert((module.into(), name.into()), func_idx);
        self
    }

    /// Define a host function directly
    pub fn func_wrap<F>(
        &mut self,
        store: &mut Store,
        module: impl Into<String>,
        name: impl Into<String>,
        func_type: FuncType,
        callback: F,
    ) -> &mut Self
    where
        F: Fn(&mut Store, &[Value]) -> Result<Vec<Value>, Trap> + Send + Sync + 'static,
    {
        let func = Func::wrap(func_type, callback);
        let func_idx = store.alloc_func(func);
        self.define_func(module, name, func_idx);
        self
    }

    /// Define a memory
    pub fn define_memory(
        &mut self,
        module: impl Into<String>,
        name: impl Into<String>,
        memory_idx: MemoryIdx,
    ) -> &mut Self {
        self.memories
            .insert((module.into(), name.into()), memory_idx);
        self
    }

    /// Define a table
    pub fn define_table(
        &mut self,
        module: impl Into<String>,
        name: impl Into<String>,
        table_idx: TableIdx,
    ) -> &mut Self {
        self.tables.insert((module.into(), name.into()), table_idx);
        self
    }

    /// Define a global
    pub fn define_global(
        &mut self,
        module: impl Into<String>,
        name: impl Into<String>,
        global_idx: GlobalIdx,
    ) -> &mut Self {
        self.globals
            .insert((module.into(), name.into()), global_idx);
        self
    }

    /// Resolve a function import
    pub fn resolve_func(&self, module: &str, name: &str) -> Option<FuncIdx> {
        self.funcs
            .get(&(module.to_string(), name.to_string()))
            .copied()
    }

    /// Resolve a memory import
    pub fn resolve_memory(&self, module: &str, name: &str) -> Option<MemoryIdx> {
        self.memories
            .get(&(module.to_string(), name.to_string()))
            .copied()
    }

    /// Resolve a table import
    pub fn resolve_table(&self, module: &str, name: &str) -> Option<TableIdx> {
        self.tables
            .get(&(module.to_string(), name.to_string()))
            .copied()
    }

    /// Resolve a global import
    pub fn resolve_global(&self, module: &str, name: &str) -> Option<GlobalIdx> {
        self.globals
            .get(&(module.to_string(), name.to_string()))
            .copied()
    }

    /// Instantiate a module using this linker
    pub fn instantiate<'a>(
        &self,
        store: &mut Store,
        module: &Module<'a>,
    ) -> Result<Instance, InstantiationError> {
        Instance::new(store, module, self)
    }
}
