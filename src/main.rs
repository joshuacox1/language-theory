use std::collections::HashSet;

use cfg::{ContextFreeGrammar, Variable, Terminal, VarOrTerm,
zv,zt,xv,xt};

mod cfg;
mod regexp;



fn main() {
    use regexp::*;
    let nfa = Nfa::from_regexp(&example1());
    println!("{nfa:?}");
    let dfa = Dfa::from_nfa(&nfa);
    println!("{dfa:?}");
}

fn cfg_test_main() {
    let mut test_grammar = ContextFreeGrammar::new(
        [
            (zv('S'), vec![xv('S')]),
            (zv('S'), vec![xt('0')]),
            (zv('S'), vec![xv('A')]),
            (zv('A'), vec![xv('A'), xv('B')]),
            (zv('B'), vec![xt('1')]),
        ].into_iter().collect::<HashSet<_>>(),
        zv('S'),
    );
    println!("{test_grammar:?}");

    println!("Culling...");
    let (nongenerative, unreachable) = test_grammar.cull_useless();
    println!("Variables {nongenerative:?} were non-generative");
    println!("Variables {unreachable:?} were unreachable afer removal of non-generative items");

    // for r in test_grammar
    println!("{test_grammar:?}");
}

// S -> 0 | A
// A -> S


