use bindgen::{
    callbacks::{DeriveInfo, TypeKind},
    CargoCallbacks,
};
use gag::BufferRedirect;
use handlebars::Handlebars;
use miette::miette;
use miette::{IntoDiagnostic, WrapErr};
use serde::{Deserialize, Serialize};
use static_assertions::const_assert;
use std::str::FromStr;
use std::{borrow::Cow, collections::HashMap};
use std::{env, path::PathBuf};
use strum::IntoStaticStr;

#[derive(IntoStaticStr, Copy, Clone)]
#[allow(dead_code)]
enum SupportedArchitecture {
    #[strum(serialize = "SLJIT_CONFIG_X86_32")]
    I386,
    #[strum(serialize = "SLJIT_CONFIG_X86_64")]
    X86_64,
    #[strum(serialize = "SLJIT_CONFIG_ARM_V6")]
    ARMV6,
    #[strum(serialize = "SLJIT_CONFIG_ARM_V7")]
    ARMV7,
    #[strum(serialize = "SLJIT_CONFIG_ARM_THUMB2")]
    ARM_THUMB2,
    #[strum(serialize = "SLJIT_CONFIG_ARM_64")]
    ARM64,
    #[strum(serialize = "SLJIT_CONFIG_PPC_32")]
    PPC32,
    #[strum(serialize = "SLJIT_CONFIG_PPC_64")]
    PPC64,
    #[strum(serialize = "SLJIT_CONFIG_MIPS_32")]
    MIPS32,
    #[strum(serialize = "SLJIT_CONFIG_MIPS_64")]
    MIPS64,
    #[strum(serialize = "SLJIT_CONFIG_RISCV_32")]
    RV32,
    #[strum(serialize = "SLJIT_CONFIG_RISCV_64")]
    RV64,
    #[strum(serialize = "SLJIT_CONFIG_S390X")]
    S390X,
    #[strum(serialize = "SLJIT_CONFIG_LOONGARCH_64")]
    LOONGARCH64,
}

const ARCH: &[SupportedArchitecture] = &[
    #[cfg(feature = "arch-i386")]
    SupportedArchitecture::I386,
    #[cfg(feature = "arch-x86_64")]
    SupportedArchitecture::X86_64,
    #[cfg(feature = "arch-armv6")]
    SupportedArchitecture::ARMV6,
    #[cfg(feature = "arch-armv7")]
    SupportedArchitecture::ARMV7,
    #[cfg(feature = "arch-arm-thumb-2")]
    SupportedArchitecture::ARM_THUMB2,
    #[cfg(feature = "arch-arm64")]
    SupportedArchitecture::ARM64,
    #[cfg(feature = "arch-ppc32")]
    SupportedArchitecture::PPC32,
    #[cfg(feature = "arch-ppc64")]
    SupportedArchitecture::PPC64,
    #[cfg(feature = "arch-mips32")]
    SupportedArchitecture::MIPS32,
    #[cfg(feature = "arch-mips64")]
    SupportedArchitecture::MIPS64,
    #[cfg(feature = "arch-rv32")]
    SupportedArchitecture::RV32,
    #[cfg(feature = "arch-rv64")]
    SupportedArchitecture::RV64,
    #[cfg(feature = "arch-s390x")]
    SupportedArchitecture::S390X,
    #[cfg(feature = "arch-loongarch64")]
    SupportedArchitecture::LOONGARCH64,
];

// Make sure that either 0 or 1 arch is selected
const_assert!(ARCH.len() <= 1);

#[derive(Debug)]
struct ConstDefaultCallbacks;

impl bindgen::callbacks::ParseCallbacks for ConstDefaultCallbacks {
    fn add_derives(&self, info: &DeriveInfo<'_>) -> Vec<String> {
        if info.kind == TypeKind::Struct {
            vec!["ConstDefault".to_string()]
        } else {
            vec![]
        }
    }
}

fn do_bindgen(header: &str, file: &str) -> miette::Result<PathBuf> {
    let out_path: PathBuf = PathBuf::from(env::var("OUT_DIR").unwrap());

    let bindings = bindgen::Builder::default()
        .header(header)
        .allowlist_file(format!(".*?sljit.*"))
        .vtable_generation(true)
        .ctypes_prefix("::core::ffi")
        .use_core()
        .derive_copy(true)
        .derive_hash(true)
        .generate_cstr(true)
        .array_pointers_in_arguments(true)
        .sort_semantically(true)
        .merge_extern_blocks(true)
        .layout_tests(false)
        .wrap_static_fns(true)
        .wrap_static_fns_path(out_path.join("out"))
        .newtype_enum(".*")
        .default_macro_constant_type(bindgen::MacroTypeVariation::Signed)
        .parse_callbacks(Box::new(ConstDefaultCallbacks))
        .parse_callbacks(Box::new(CargoCallbacks::new()))
        .generate()
        .into_diagnostic()
        .wrap_err("Failed to generate bindgen config")?;

    bindings
        .write_to_file(out_path.join(file))
        .into_diagnostic()
        .wrap_err("Failed to generate bindgen bindings")?;

    Ok(out_path.join(file))
}

fn build_static_library() -> miette::Result<()> {
    let out_path: PathBuf = PathBuf::from(env::var("OUT_DIR").unwrap());
    let mut cc = cc::Build::new();

    let cc = cc
        .file("sljit/sljit_src/sljitLir.c")
        .file(out_path.join("out.c"))
        .include(".");

    if ARCH.len() == 1 {
        cc.define(ARCH.get(0).unwrap().into(), "1");
    }

    if let Ok(debug) = env::var("DEBUG") {
        if let Ok(debug) = bool::from_str(debug.as_str()) {
            cc.define("SLJIT_DEBUG", if debug { "1" } else { "0" });
            cc.define("SLJIT_VERBOSE", if debug { "1" } else { "0" });
        }
    }

    cc.try_compile("libsljit").into_diagnostic()?;
    Ok(())
}

fn generate_mid_level_binding(out_path: PathBuf) -> miette::Result<()> {
    println!("cargo:rerun-if-changed=rules");
    let out_dir: PathBuf = PathBuf::from(env::var("OUT_DIR").unwrap());

    let buf = BufferRedirect::stdout().into_diagnostic()?;
    ast_grep::main_with_args(
        [
            "sg".to_string(),
            "scan".to_string(),
            out_path.to_string_lossy().to_string(),
            "--json=compact".to_string(),
        ]
        .into_iter(),
    )
    .map_err(|e| miette!(e))?;

    let data: Vec<MatchJSON> = serde_json::from_reader(buf).into_diagnostic()?;
    let replacements: Vec<Cow<str>> = data.into_iter().flat_map(|x| x.replacement).collect();

    let mut hbs = Handlebars::new();
    hbs.register_escape_fn(handlebars::no_escape);

    let rendered = hbs
        .render_template(
            r#" 
impl Compiler {
    {{#each replacements}}
    {{this}}
    {{/each}}
}
"#,
            &serde_json::json!({"replacements": replacements}),
        )
        .into_diagnostic()?;
    std::fs::write(out_dir.join("generated.mid.rs"), rendered).into_diagnostic()?;

    Ok(())
}

fn main() -> miette::Result<()> {
    let out_path = do_bindgen("sljit/sljit_src/sljitLir.h", "wrapper.rs")?;
    generate_mid_level_binding(out_path)?;
    build_static_library()?;
    Ok(())
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Position {
    line: usize,
    column: usize,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Range {
    /// inclusive start, exclusive end
    byte_offset: std::ops::Range<usize>,
    start: Position,
    end: Position,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct LabelJSON<'a> {
    text: &'a str,
    range: Range,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct MatchNode<'a> {
    text: Cow<'a, str>,
    range: Range,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct MatchJSON<'a> {
    text: Cow<'a, str>,
    range: Range,
    file: Cow<'a, str>,
    lines: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    replacement: Option<Cow<'a, str>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    replacement_offsets: Option<std::ops::Range<usize>>,
    language: Cow<'a, str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    meta_variables: Option<MetaVariables<'a>>,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct MetaVariables<'a> {
    single: HashMap<String, MatchNode<'a>>,
    multi: HashMap<String, Vec<MatchNode<'a>>>,
    transformed: HashMap<String, String>,
}

#[derive(Serialize)]
struct Context {
    name: String,
}
