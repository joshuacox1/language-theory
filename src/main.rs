
// use cfg::{ContextFreeGrammar, Variable, Terminal, VarOrTerm,
// zv,zt,xv,xt};

// // mod cfg;
// mod regexp;

use std::io;

fn main() -> io::Result<()> {
    use contextfree::genregexp::*;

    let mut buffer = String::new();
    loop {
        io::stdin().read_line(&mut buffer)?;

        match GenRegex::from_str(&buffer) {
            Some(r) => {
                println!("{r:?}");
                let mut d = r.to_min_canon_dfa();
                println!("{}", d.show());
                d.complete();
                println!("{}", d.show());
            }
            None => println!("Parsing error :("),
        }

        buffer.clear();
    }
}


// fn main() -> io::Result<()> {
//     use contextfree::regexp::*;
//     let mut buffer = String::new();

//     loop {
//         io::stdin().read_line(&mut buffer)?;

//         match StateMachine::new(&buffer) {
//             Ok(canon_min_dfa) => println!("\n{}", canon_min_dfa.to_graphviz()),
//             Err(e) => println!("Parse fail: {e}"),
//         }

//         buffer.clear();
//     }
// }

// fn cfg_test_main() {
//     let mut test_grammar = ContextFreeGrammar::new(
//         [
//             (zv('S'), vec![xv('S')]),
//             (zv('S'), vec![xt('0')]),
//             (zv('S'), vec![xv('A')]),
//             (zv('A'), vec![xv('A'), xv('B')]),
//             (zv('B'), vec![xt('1')]),
//         ].into_iter().collect::<HashSet<_>>(),
//         zv('S'),
//     );
//     println!("{test_grammar:?}");

//     println!("Culling...");
//     let (nongenerative, unreachable) = test_grammar.cull_useless();
//     println!("Variables {nongenerative:?} were non-generative");
//     println!("Variables {unreachable:?} were unreachable afer removal of non-generative items");

//     // for r in test_grammar
//     println!("{test_grammar:?}");
// }

// S -> 0 | A
// A -> S


