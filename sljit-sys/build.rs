#[cfg(feature = "bindgen")]
use gag::BufferRedirect;
#[cfg(feature = "bindgen")]
use handlebars::Handlebars;
use miette::IntoDiagnostic;
#[cfg(feature = "bindgen")]
use miette::WrapErr;
#[cfg(feature = "bindgen")]
use miette::miette;
#[cfg(feature = "bindgen")]
use natural_sort_rs::NaturalSortable;
#[cfg(feature = "bindgen")]
use serde::{Deserialize, Serialize};
use static_assertions::const_assert;
use std::str::FromStr;
#[cfg(feature = "bindgen")]
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
    #[allow(non_camel_case_types)]
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

fn debug() -> bool {
    (env::var("DEBUG")
        .map_err(|_| ())
        .and_then(|debug| bool::from_str(debug.as_str()).map_err(|_| ()))
        .unwrap_or(false)
        && cfg!(debug_assertions))
        || cfg!(feature = "force-debug")
}

fn docs() -> bool {
    env::var("DOCS_RS")
        .or(env::var("CARGO_CFG_DOC"))
        .map(|_| ())
        .or(if cfg!(docsrs) { Ok(()) } else { Err(()) })
        .or(if cfg!(doc) { Ok(()) } else { Err(()) })
        .or(if cfg!(feature = "docs") {
            Ok(())
        } else {
            Err(())
        })
        .is_ok()
}

#[cfg(feature = "bindgen")]
fn do_bindgen(header: &str, file: &str) -> miette::Result<PathBuf> {
    let out_path: PathBuf = PathBuf::from("./src");

    let (debug, verbose) = match (debug(), cfg!(feature = "force-verbose"), docs()) {
        (false, false, _) | (_, _, true) => ("0", "0"),
        (false, true, _) => ("0", "1"),
        (true, false, _) => ("1", "0"),
        (true, true, _) => ("1", "1"),
    };

    let bindings = bindgen::Builder
        ::default()
        .header(header)
        .clang_args([
            format!("-D{}={}", "SLJIT_DEBUG", debug),
            format!("-D{}={}", "SLJIT_VERBOSE", verbose),
        ])
        .allowlist_file(format!(".*?sljit.*"))
        .vtable_generation(true)
        .opaque_type("FILE")
        .opaque_type("sljit_(memory_fragment|label|jump|const|generate_code_buffer|read_only_buffer|compiler|stack|function_context)")
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
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
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
    let out_path: PathBuf = PathBuf::from("./src");
    let mut cc = cc::Build::new();

    let cc = cc
        .file("sljit/sljit_src/sljitLir.c")
        .file(out_path.join("out.c"))
        .include(".");

    if ARCH.len() == 1 {
        cc.define(ARCH.first().unwrap().into(), "1");
    }

    let (debug, verbose) = match (debug(), cfg!(feature = "force-verbose"), docs()) {
        (false, false, _) | (_, _, true) => ("0", "0"),
        (false, true, _) => ("0", "1"),
        (true, false, _) => ("1", "0"),
        (true, true, _) => ("1", "1"),
    };

    cc.define("SLJIT_DEBUG", debug);
    cc.define("SLJIT_VERBOSE", verbose);

    cc.try_compile("libsljit").into_diagnostic()?;
    Ok(())
}

#[cfg(feature = "bindgen")]
fn generate_mid_level_binding(out_path: PathBuf) -> miette::Result<()> {
    use cmd_lib::run_cmd;

    println!("cargo:rerun-if-changed=rules");
    let out_dir: PathBuf = PathBuf::from("./src");

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
    let mut replacements: Vec<Cow<str>> = data.into_iter().flat_map(|x| x.replacement).collect();

    replacements.sort_by(|a, b| {
        let a_name = extract_fn_name(a);
        let b_name = extract_fn_name(b);
        if let Some(a_name) = a_name
            && let Some(b_name) = b_name
        {
            return a_name.natural_cmp(b_name);
        } else {
            a_name.cmp(&b_name)
        }
    });

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
    let path = out_dir.join("generated.mid.rs");
    std::fs::write(&path, rendered).into_diagnostic()?;
    run_cmd!(rustfmt $path).into_diagnostic()?;
    Ok(())
}

#[cfg(feature = "bindgen")]
fn extract_fn_name(method_code: &str) -> Option<&str> {
    // Extract function name from something like:
    // pub fn emit_op1(...) {...}
    if let Some(start) = method_code.find("pub fn ")
        && let Some(paren) = (&method_code[start + 7..]).find('(')
    {
        Some((&method_code[start + 7..])[..paren].trim())
    } else {
        None
    }
}

fn main() -> miette::Result<()> {
    #[cfg(feature = "bindgen")]
    {
        let out_path = do_bindgen("sljit/sljit_src/sljitLir.h", "wrapper.rs")?;
        generate_mid_level_binding(out_path)?;
    }
    build_static_library()?;
    Ok(())
}

#[cfg(feature = "bindgen")]
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Position {
    line: usize,
    column: usize,
}
#[cfg(feature = "bindgen")]
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Range {
    /// inclusive start, exclusive end
    byte_offset: std::ops::Range<usize>,
    start: Position,
    end: Position,
}
#[cfg(feature = "bindgen")]
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct LabelJSON<'a> {
    text: &'a str,
    range: Range,
}
#[cfg(feature = "bindgen")]
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct MatchNode<'a> {
    text: Cow<'a, str>,
    range: Range,
}
#[cfg(feature = "bindgen")]
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
#[cfg(feature = "bindgen")]
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct MetaVariables<'a> {
    single: HashMap<String, MatchNode<'a>>,
    multi: HashMap<String, Vec<MatchNode<'a>>>,
    transformed: HashMap<String, String>,
}
#[cfg(feature = "bindgen")]
#[derive(Serialize)]
struct Context {
    name: String,
}
