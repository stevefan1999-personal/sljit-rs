use sljit::sys::{self, arg_types, *};
use sljit::*;
use std::error::Error;
use std::io::{self, Read, Write};
use std::mem::transmute;

const BF_CELL_SIZE: usize = 30000;

extern "C" fn bf_putchar(c: isize) {
    print!("{}", c as u8 as char);
    io::stdout().flush().ok();
}

extern "C" fn bf_getchar() -> isize {
    io::stdin()
        .bytes()
        .next()
        .and_then(|result| result.ok())
        .map(|b| b as isize)
        .unwrap_or(0)
}

extern "C" fn bf_calloc(size: libc::size_t, count: libc::size_t) -> *mut libc::c_void {
    unsafe { libc::calloc(size, count) }
}

extern "C" fn bf_free(ptr: *mut libc::c_void) {
    unsafe { libc::free(ptr) }
}

/// Emit code for BF source, advances index through mutable reference
fn emit(
    emitter: &mut Emitter,
    chars: &[char],
    i: &mut usize,
    sp: SavedRegister,
    cells: SavedRegister,
) -> Result<(), ErrorCode> {
    use ScratchRegister::*;

    while *i < chars.len() {
        match chars[*i] {
            '+' => {
                emitter.mov_u8(0, R0, mem_indexed(cells, sp))?;
                emitter.add(0, R0, R0, 1)?;
                emitter.mov_u8(0, mem_indexed(cells, sp), R0)?;
            }
            '-' => {
                emitter.mov_u8(0, R0, mem_indexed(cells, sp))?;
                emitter.sub(0, R0, R0, 1)?;
                emitter.mov_u8(0, mem_indexed(cells, sp), R0)?;
            }
            '>' => {
                emitter.add(0, sp, sp, 1)?;
            }
            '<' => {
                emitter.sub(0, sp, sp, 1)?;
            }
            '.' => {
                emitter.mov_u8(0, R0, mem_indexed(cells, sp))?;
                emitter.icall(
                    sys::SLJIT_CALL,
                    arg_types!([W]),
                    bf_putchar as *const () as isize,
                )?;
            }
            ',' => {
                emitter.icall(
                    sys::SLJIT_CALL,
                    arg_types!([] -> W),
                    bf_getchar as *const () as isize,
                )?;
                emitter.mov_u8(0, mem_indexed(cells, sp), R0)?;
            }
            '[' => {
                *i += 1; // Move past '['

                // Pre-check: if cell == 0, skip the entire loop
                emitter.mov_u8(0, R0, mem_indexed(cells, sp))?;
                let mut skip_jump = emitter.cmp(Condition::Equal, R0, 0)?;

                // do-while: execute body, then check if cell != 0 to continue
                emitter.do_while_(
                    |emitter, _ctx| {
                        // Emit loop body (recursive)
                        emit(emitter, chars, i, sp, cells)?;
                        // Load cell value for the end-of-loop condition check
                        emitter.mov_u8(0, R0, mem_indexed(cells, sp))?;
                        Ok(())
                    },
                    Condition::NotEqual,
                    R0,
                    0,
                )?;

                // Wire the skip_jump to after the loop
                let mut end_label = emitter.put_label()?;
                skip_jump.set_label(&mut end_label);
            }
            ']' => return Ok(()), // End of loop body, return to caller
            _ => {}
        }
        *i += 1;
    }
    Ok(())
}

fn compile(source: &str) -> Result<sys::GeneratedCode, Box<dyn Error>> {
    let chars: Vec<char> = source.chars().filter(|&c| "+-><[].,".contains(c)).collect();

    let mut compiler = sys::Compiler::new();
    let mut emitter = Emitter::new(&mut compiler);

    use SavedRegister::*;
    use ScratchRegister::*;

    const SP: SavedRegister = S0;
    const CELLS: SavedRegister = S1;

    emitter.emit_enter(0, arg_types!([]), regs!(2), regs!(2), 0)?;
    emitter.xor(0, SP, SP, SP)?;

    emitter.mov(0, R0, BF_CELL_SIZE as i32)?;
    emitter.mov(0, R1, 1)?;
    emitter.icall(
        sys::SLJIT_CALL,
        arg_types!([W, W] -> P),
        bf_calloc as *const () as isize,
    )?;

    let mut end_jump = emitter.cmp(Condition::Equal, R0, 0)?;
    emitter.mov(0, CELLS, R0)?;

    let mut i = 0;
    emit(&mut emitter, &chars, &mut i, SP, CELLS)?;

    emitter.mov(0, R0, CELLS)?;
    emitter.icall(
        sys::SLJIT_CALL,
        arg_types!([P]),
        bf_free as *const () as isize,
    )?;

    let mut exit_label = emitter.put_label()?;
    end_jump.set_label(&mut exit_label);
    emitter.return_void()?;

    Ok(compiler.generate_code())
}

fn run_bf(name: &str, description: &str, source: &str) -> Result<(), Box<dyn Error>> {
    println!("=== {} ===", name);
    if !description.is_empty() {
        println!("{}", description);
    }
    print!("Output: ");
    io::stdout().flush()?;
    let code = compile(source)?;
    let func: fn() = unsafe { transmute(code.get()) };
    func();
    println!("\n");
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    run_bf(
        "Hello World",
        "Classic BF hello world program",
        "++++++++[>++++[>++>+++>+++>+<<<<-]>+>+>->>+[<]<-]>>.>---.+++++++..+++.>>.<-.<.+++.------.--------.>>+.>++.",
    )?;

    run_bf(
        "Counter (0-9)",
        "Counts from 0 to 9 (6*8=48 for '0')",
        "++++++[>++++++++<-]>.+.+.+.+.+.+.+.+.+.",
    )?;

    run_bf(
        "Alphabet (A-Z)",
        "Prints uppercase alphabet",
        "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++.+.+.+.+.+.+.+.+.+.+.+.+.+.+.+.+.+.+.+.+.+.+.+.+.+.",
    )?;

    run_bf(
        "Cell Movement Demo",
        "Prints 'ABCDE'",
        "++++++++[>++++++++<-]>+.+.+.+.+.",
    )?;

    run_bf(
        "Nested Loop Demo",
        "Prints 'X' (88 = 8*11)",
        "++++++++[>+++++++++++<-]>.",
    )?;

    run_bf(
        "Banner: BF",
        "Prints 'BF'",
        "++++++++[>++++++++<-]>++.++++.",
    )?;

    run_bf(
        "Countdown (9-0)",
        "Counts down from 9 to 0",
        "+++++++[>++++++++<-]>+.-.-.-.-.-.-.-.-.-.",
    )?;

    run_bf(
        "Character Transformation",
        "Prints 'N' (78 = 6*13)",
        "++++++[>+++++++++++++<-]>.",
    )?;

    run_bf(
        "Triangle Pattern",
        "Prints ASCII triangle",
        ">++++++++++[<++++>-]<++.>++++++++++.[-]<[-]>++++++++++[<++++>-]<++..>++++++++++.[-]<[-]>++++++++++[<++++>-]<++...>++++++++++.",
    )?;

    run_bf(
        "Exclamation Marks (x5)",
        "Prints 5 exclamation marks",
        "+++[>+++++++++++<-]>.....",
    )?;

    run_bf(
        "Hi",
        "Prints 'Hi'",
        "++++++++[>+++++++++<-]>.>+++++++[>+++++++++++++++<-]>.",
    )?;

    run_bf(
        "Multiplication Demo",
        "Prints 'C' (8*8+3=67)",
        "++++++++[>++++++++<-]>+++.",
    )?;

    run_bf(
        "Addition Demo",
        "Computes 3+5=8, prints '8'",
        "+++>+++++[<+>-]<++++++++++++++++++++++++++++++++++++++++++++++++.",
    )?;

    run_bf(
        "Loop Counter (1-5)",
        "Prints 12345",
        "+++++>+++++++[>+++++++<-]><<[>>.+<<-]",
    )?;

    run_bf(
        "Fibonacci (112358)",
        "Prints first 6 Fibonacci digits",
        "++++++[>++++++++<-]>+..+.+.++.+++.",
    )?;

    println!("=== All examples completed! ===");
    Ok(())
}
