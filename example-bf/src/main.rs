use sljit::sys::{self, arg_types, *};
use sljit::*;
use std::alloc::Layout;
use std::error::Error;
use std::io::{self, Read, Write};
use std::mem::transmute;

const BF_CELL_SIZE: usize = 30000;

#[derive(Debug, Clone, Copy)]
struct Token {
    op: char,
    count: i32,
}

fn tokenize(source: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut chars = source
        .chars()
        .filter(|&c| "+-><[].,".contains(c))
        .peekable();

    while let Some(op) = chars.next() {
        if matches!(op, '[' | ']' | '.' | ',') {
            tokens.push(Token { op, count: 1 });
            continue;
        }

        let mut count = 1;
        while chars.peek() == Some(&op) {
            count += 1;
            chars.next();
        }
        tokens.push(Token { op, count });
    }

    tokens
}

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

unsafe extern "C" fn bf_calloc(size: isize, count: isize) -> *mut u8 {
    unsafe {
        std::alloc::alloc_zeroed(Layout::from_size_align((size * count) as usize, 8).unwrap())
    }
}

unsafe extern "C" fn bf_free(_ptr: *mut u8) {}

fn compile(source: &str) -> Result<sys::GeneratedCode, Box<dyn Error>> {
    let tokens = tokenize(source);

    // Check for balanced brackets
    let mut depth = 0;
    for token in &tokens {
        match token.op {
            '[' => depth += 1,
            ']' => {
                depth -= 1;
                if depth < 0 {
                    return Err("Unmatched ']'".into());
                }
            }
            _ => {}
        }
    }
    if depth != 0 {
        return Err("Unmatched '['".into());
    }

    let mut compiler = sys::Compiler::new();
    let mut emitter = Emitter::new(&mut compiler);

    use SavedRegister::*;
    use ScratchRegister::*;

    const SP: SavedRegister = S0;
    const CELLS: SavedRegister = S1;

    emitter.emit_enter(0, arg_types!([]), regs!(2), regs!(2), 0)?;

    // Initialize SP = 0
    emitter.xor(0, SP, SP, SP)?;

    // Allocate cells
    emitter.mov(0, R0, BF_CELL_SIZE as i32)?;
    emitter.mov(0, R1, 1)?;
    emitter.icall(
        sys::SLJIT_CALL,
        arg_types!([W, W] -> P),
        bf_calloc as *const () as isize,
    )?;

    let mut end_jump = emitter.cmp(Condition::Equal, R0, 0)?;
    emitter.mov(0, CELLS, R0)?;

    let mut loop_stack: Vec<(sys::Label, sys::Jump)> = Vec::new();

    for token in tokens {
        match token.op {
            '+' | '-' => {
                emitter.mov_u8(0, R0, mem_indexed(CELLS, SP))?;

                if token.op == '+' {
                    emitter.add(0, R0, R0, token.count)?;
                } else {
                    emitter.sub(0, R0, R0, token.count)?;
                }

                emitter.mov_u8(0, mem_indexed(CELLS, SP), R0)?;
            }

            '>' | '<' => {
                if token.op == '>' {
                    emitter.add(0, SP, SP, token.count)?;
                } else {
                    emitter.sub(0, SP, SP, token.count)?;
                }
            }

            '.' => {
                emitter.mov_u8(0, R0, mem_indexed(CELLS, SP))?;
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
                emitter.mov_u8(0, mem_indexed(CELLS, SP), R0)?;
            }

            '[' => {
                let loop_start = emitter.put_label()?;
                emitter.mov_u8(0, R0, mem_indexed(CELLS, SP))?;
                let loop_end = emitter.cmp(Condition::Equal, R0, 0)?;
                loop_stack.push((loop_start, loop_end));
            }

            ']' => {
                if let Some((mut loop_start, mut loop_end)) = loop_stack.pop() {
                    let mut jump_back = emitter.jump(JumpType::Jump)?;
                    jump_back.set_label(&mut loop_start);
                    let mut end_label = emitter.put_label()?;
                    loop_end.set_label(&mut end_label);
                } else {
                    return Err("Unmatched ']'".into());
                }
            }

            _ => {}
        }
    }

    // Epilogue
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

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Hello World ===");
    let hello_world = "++++++++[>++++[>++>+++>+++>+<<<<-]>+>+>->>+[<]<-]>>.>---.+++++++..+++.>>.<-.<.+++.------.--------.>>+.>++.";
    let code = compile(hello_world)?;
    let func: fn() = unsafe { transmute(code.get()) };
    func();
    println!("\n");

    println!("=== Counter (0-9) ===");
    // Correct counter: increment from '0' (48) to '9' (57)
    let counter = "++++++[>++++++++<-]>.+.+.+.+.+.+.+.+.+."; // 6*8=48 ('0'), then print and inc 10 times
    let code = compile(counter)?;
    let func: fn() = unsafe { transmute(code.get()) };
    func();
    println!("\n");

    println!("=== Simple Test (prints '0') ===");
    let simple_test = "++++++++++++++++++++++++++++++++++++++++++++++++."; // 48 = '0'
    let code = compile(simple_test)?;
    let func: fn() = unsafe { transmute(code.get()) };
    func();
    println!("\n");

    println!("=== Addition (2 + 5 = 7, prints as '7') ===");
    // cell[0]=2, loop adds 5 twice to cell[1] = 10, then add 45 to get 55 ('7')
    let addition = "++[>+++++<-]>+++++++++++++++++++++++++++++++++++++++++++++.";
    let code = compile(addition)?;
    let func: fn() = unsafe { transmute(code.get()) };
    func();
    println!("\n");

    Ok(())
}
