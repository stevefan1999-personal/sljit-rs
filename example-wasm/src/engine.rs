//! Engine configuration and compilation settings
//!
//! This module contains the Engine and its configuration types:
//! - `OptLevel` - Optimization level for JIT compilation
//! - `WasmFeatures` - WebAssembly feature flags
//! - `EngineConfig` - Full engine configuration
//! - `Engine` - Compilation and runtime engine

use std::sync::Arc;

// ============================================================================
// Engine Configuration
// ============================================================================

/// Optimization level for JIT compilation
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OptLevel {
    /// No optimizations, fastest compile time
    None,
    /// Basic optimizations, balanced (default)
    #[default]
    Speed,
    /// Aggressive optimizations, slower compile
    SpeedAndSize,
}

/// WebAssembly feature flags
#[derive(Clone, Debug)]
pub struct WasmFeatures {
    /// Enable multi-value returns
    pub multi_value: bool,
    /// Enable bulk memory operations
    pub bulk_memory: bool,
    /// Enable reference types
    pub reference_types: bool,
    /// Enable SIMD (not currently supported)
    pub simd: bool,
    /// Enable mutable globals
    pub mutable_global: bool,
    /// Enable sign extension operators
    pub sign_extension: bool,
    /// Enable saturating float-to-int conversions
    pub saturating_float_to_int: bool,
}

impl Default for WasmFeatures {
    fn default() -> Self {
        Self {
            multi_value: true,
            bulk_memory: true,
            reference_types: true,
            simd: false, // Not supported
            mutable_global: true,
            sign_extension: true,
            saturating_float_to_int: true,
        }
    }
}

/// Engine configuration for compilation and runtime
#[derive(Clone, Debug)]
pub struct EngineConfig {
    /// Optimization level
    pub opt_level: OptLevel,
    /// WebAssembly features
    pub features: WasmFeatures,
    /// Maximum stack depth (in frames)
    pub max_stack_depth: usize,
    /// Enable fuel metering
    pub fuel_metering: bool,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            opt_level: OptLevel::Speed,
            features: WasmFeatures::default(),
            max_stack_depth: 1024,
            fuel_metering: false,
        }
    }
}

/// Compilation and runtime engine
///
/// The Engine holds configuration for compilation and is stateless.
/// It can be shared across threads via `Arc<Engine>`.
#[derive(Clone, Debug)]
pub struct Engine {
    config: EngineConfig,
}

impl Default for Engine {
    fn default() -> Self {
        Self::new(EngineConfig::default())
    }
}

impl Engine {
    /// Create a new engine with the given configuration
    #[inline]
    pub fn new(config: EngineConfig) -> Self {
        Self { config }
    }

    /// Get the engine configuration
    #[inline]
    pub fn config(&self) -> &EngineConfig {
        &self.config
    }

    /// Get the optimization level
    #[inline]
    pub fn opt_level(&self) -> OptLevel {
        self.config.opt_level
    }

    /// Get the WebAssembly features
    #[inline]
    pub fn features(&self) -> &WasmFeatures {
        &self.config.features
    }

    /// Create a shared engine
    #[inline]
    pub fn into_shared(self) -> Arc<Self> {
        Arc::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_default() {
        let engine = Engine::default();
        assert_eq!(engine.opt_level(), OptLevel::Speed);
        assert!(engine.features().multi_value);
        assert!(!engine.features().simd);
    }
}