use sljit::sys::{self, arg_types, *};
use sljit::*;
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

extern "C" fn bf_calloc(size: libc::size_t, count: libc::size_t) -> *mut libc::c_void {
    unsafe { libc::calloc(size, count) }
}

extern "C" fn bf_free(ptr: *mut libc::c_void) {
    unsafe { libc::free(ptr) }
}

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
    // 1. Classic Hello World
    run_bf(
        "Hello World",
        "Classic BF hello world program",
        "++++++++[>++++[>++>+++>+++>+<<<<-]>+>+>->>+[<]<-]>>.>---.+++++++..+++.>>.<-.<.+++.------.--------.>>+.>++.",
    )?;

    // 2. Counter 0-9
    run_bf(
        "Counter (0-9)",
        "Counts from 0 to 9 using loop optimization (6*8=48 for '0')",
        "++++++[>++++++++<-]>.+.+.+.+.+.+.+.+.+.",
    )?;

    // 3. Print the Alphabet (A-Z)
    run_bf(
        "Alphabet (A-Z)",
        "Prints the uppercase alphabet using nested loops",
        "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++.+.+.+.+.+.+.+.+.+.+.+.+.+.+.+.+.+.+.+.+.+.+.+.+.+.",
    )?;

    // 4. Cell Movement Demo
    run_bf(
        "Cell Movement Demo",
        "Demonstrates cell pointer movement: prints 'ABCDE'",
        "++++++++[>++++++++<-]>+.+.+.+.+.",
    )?;

    // 5. Nested Loop Demo
    run_bf(
        "Nested Loop Demo",
        "Uses nested loops to compute and print 'X' (88 = 8*11)",
        "++++++++[>+++++++++++<-]>.",
    )?;

    // 6. Simple Banner
    run_bf(
        "Banner: BF",
        "Prints 'BF' (Brainfuck initials)",
        "++++++++[>++++++++<-]>++.++++.",
    )?;

    // 7. Countdown 9 to 0
    // 7*8=56, +1=57='9', then print and decrement 10 times to reach '0'
    run_bf(
        "Countdown (9-0)",
        "Counts down from 9 to 0",
        "+++++++[>++++++++<-]>+.-.-.-.-.-.-.-.-.-.",
    )?;

    // 8. Character Transformation - print 'N' directly
    // 'N' = 78 = 6*13
    run_bf(
        "Character Transformation",
        "Prints 'N' (ASCII 78 = 6*13)",
        "++++++[>+++++++++++++<-]>.",
    )?;

    // 9. Triangle Pattern
    run_bf(
        "Triangle Pattern",
        "Prints a simple ASCII triangle",
        ">++++++++++[<++++>-]<++.>++++++++++.[-]<[-]>++++++++++[<++++>-]<++..>++++++++++.[-]<[-]>++++++++++[<++++>-]<++...>++++++++++.",
    )?;

    // 10. Exclamation marks
    // '!' = 33 = 3*11, print it 5 times
    run_bf(
        "Exclamation Marks (x5)",
        "Prints 5 exclamation marks",
        "+++[>+++++++++++<-]>.....",
    )?;

    // 11. Print "Hi"
    // 'H' = 72 = 8*9, 'i' = 105 = 7*15
    run_bf(
        "Hi",
        "Simple 'Hi' output",
        "++++++++[>+++++++++<-]>.>+++++++[>+++++++++++++++<-]>.",
    )?;

    // 12. Print 'C' (ASCII 67 = 8*8+3)
    run_bf(
        "Multiplication Demo",
        "Prints 'C' using 8*8+3=67",
        "++++++++[>++++++++<-]>+++.",
    )?;

    // 13. Addition Demo: 3+5=8, then add 48 to get '8'
    run_bf(
        "Addition Demo",
        "Computes 3+5=8, prints '8' (ASCII 56)",
        "+++>+++++[<+>-]<++++++++++++++++++++++++++++++++++++++++++++++++.",
    )?;

    // 14. Loop counter - prints digits 1 to 5
    // cell0=5 (counter), cell2=49='1' (via 7*7)
    // Loop: print cell2, increment it, decrement counter
    run_bf(
        "Loop Counter (1-5)",
        "Demonstrates loop counting: prints 12345",
        "+++++>+++++++[>+++++++<-]><<[>>.+<<-]",
    )?;

    // 15. Fibonacci first 6 values as string: prints "112358"
    run_bf(
        "Fibonacci (112358)",
        "Prints first 6 Fibonacci numbers as digits",
        // cell0 = '0' (48), then print 1,1,2,3,5,8 via increments
        "++++++[>++++++++<-]>+..+.+.++.+++.",
    )?;

    println!("=== All examples completed! ===");
    Ok(())
}
