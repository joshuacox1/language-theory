use std::collections::HashSet;
use bit_set::BitSet;

use cfg::{ContextFreeGrammar, Variable, Terminal, VarOrTerm,
zv,zt,xv,xt};

mod cfg;
mod regexp;



fn main() {
    use regexp::*;

    // let q = cnc_arr([
    //     ast(alt(lit('a'), lit('b'))),
    //     lit('a'),
    //     lit('b'),
    //     lit('b'),
    //     ast(alt(lit('a'), lit('b')))
    // ]);
    // let only_a = alt(lit('a'), cnc(lit('b'), emp));
    let r = alt(
        alt(
            alt(cnc(lit('w'),lit('o')), cnc(lit('a'),lit('b'))),
        cnc(lit('z'),lit('o'))),
        cnc(lit('q'),lit('q')));//example2();
    let nfa = Nfa::from_regexp(&r);
    println!("{}", nfa.to_graphviz());
    let dfa = Dfa::from_nfa(&nfa);
    println!("{}", dfa.to_graphviz());
    let min_dfa = dfa.minimise(&r.chars_mentioned());
    println!("{}", min_dfa.to_graphviz());
    // println!("{dfa:?}");
    // println!("{}", dfa_example1().show_string());
    // println!("{}", canon_dfa(&q).to_graphviz());
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


