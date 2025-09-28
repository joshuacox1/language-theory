//! Messing about with a regular expression engine.
//! Very simple:
//! We take in a regular expression, construct the corresponding
//! NFA, compute the corresponding DFA via the powerset contruction,
//! then compute the (unique) minimal DFA matching the expression.
//! We can use the regex crate to fuzz I guess?

use std::collections::{HashMap, HashSet};
use std::fmt;

use arrayvec::ArrayVec;
use bit_set::BitSet;

// We can set this to whatever we want. Just do small numbers for now.
// We can use u8 since we only want to support A-Za-z0-9 (possibly
// a few ASCII symbols as well...). We can stuff epsilon in an unused
// value as well when we need that.
// For now stick with 'char'.
pub type Char = char;

#[repr(transparent)]
pub struct MyChar(u8);

impl MyChar {
    pub fn from_char(c: char) {
        // assert ascii or epsilon
    }
}


#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegExp {
    /// The empty set.
    Empty,
    /// Sugar for Star(Empty).
    Epsilon,
    /// The singleton set containing the singleton string of the given char.
    Lit(Char),
    /// Concatenation.
    Concat(Box<RegExp>, Box<RegExp>),
    /// Alternation.
    Alt(Box<RegExp>, Box<RegExp>),
    Star(Box<RegExp>),
    // Optional. Shorthand for (Epsilon|a).
    // Opt(Box<RegExp>),
    // Plus. Shorthand for aa*.
    // Plus(Box<RegExp>),
}

pub const emp: RegExp = RegExp::Empty;
pub const eps: RegExp = RegExp::Epsilon;

#[inline] pub fn lit(c: Char) -> RegExp { RegExp::Lit(c) }

pub fn cnc(r1: RegExp, r2: RegExp) -> RegExp {
    RegExp::Concat(Box::new(r1), Box::new(r2))
}

pub fn alt(r1: RegExp, r2: RegExp) -> RegExp {
    RegExp::Alt(Box::new(r1), Box::new(r2))
}

pub fn ast(r: RegExp) -> RegExp {
    RegExp::Star(Box::new(r))
}


// pub struct Parser {
//     input: Vec<char>,
//     idx: usize,
// }

// impl Parser {
//     pub fn new(input: &str) -> Self {
//         Self { input }
//     }

//     pub fn eat(&mut self, c: char) {
//         assert!(
//             match self.input.get(idx) {
//                 Some(c_) if c_ == c => true,
//                 _ => false,
//             });
//         self.idx += 1;
//     }

//     pub fn peek(&mut self) -> Option<char> {
//         self.input.get(idx)
//     }
// }

// fn expr_bp(b: &[char], idx: usize, min_bp: u8) -> RegExp {
//     let mut lhs = match b.get(idx) {
//         Some(c) => match c {
//             '$' | '~' => {

//             }
//         }
//     }


//     loop {
//         let q = match op {
//             '|' =>
//         };
//     }
// }




pub struct Lexer {
    // Consume it lazily?
    input: Vec<char>,
    idx: usize,
}

impl Lexer {
    pub fn new(s: &str) -> Self {
        Self { input: s.chars().collect::<Vec<_>>(), idx: 0 }
    }

    pub fn peek(&self) -> Option<char> {
        self.input.get(self.idx).copied()
    }

    pub fn next(&mut self) {
        self.idx += 1;
    }
}


/// Not great :(
fn parse_new_again(lexer: &mut Lexer) -> Result<RegExp, &'static str> {
    let mut left = None;
    while let Some(c) = lexer.peek() {
        match c {
            '$' => {
                let e = RegExp::Empty;
                left = Some(match left {
                    None => e,
                    Some(l) => RegExp::Concat(
                        Box::new(l), Box::new(e),
                    ),
                });
            },
            '*' => match left {
                Some(l) => {
                    left = Some(RegExp::Star(Box::new(l)));
                },
                None => return Err("Can't start with a *"),
            },
            '|' => match left {
                Some(l) => {
                    lexer.next();
                    let right = parse_new_again(lexer)?;
                    left = Some(RegExp::Alt(Box::new(l), Box::new(right)));
                },
                None => return Err("Can't start with a |"),
            },
            c => if c.is_ascii_alphanumeric() {
                let a = RegExp::Lit(c);
                left = Some(match left {
                    None => a,
                    Some(l) => RegExp::Concat(
                        Box::new(l), Box::new(a),
                    ),
                });
            } else {
                return Err("Unrecognised character!!!");
            },
        }
        lexer.next();
    }

    match left {
        None => Err("No left at all"),
        Some(r) => Ok(r),
    }
}



fn parse_wrapper(s: &str) -> Result<RegExp, &'static str> {
    let mut lexer = Lexer::new(s);
    parse_new_again(&mut lexer)
}

// fn parse_new_try(left: Option<RegExp>, buf: &[char], idx: usize)
// -> Result<RegExp, &'static str> {
//     // let right
//     // Check what to do with emptiness. It certainly isn't valid.
//     Ok(match buf[idx] {
//         '$' => {
//             let e = RegExp::Empty;
//             match left {
//                 None => e,
//                 Some(l) => RegExp::Concat(Box::new(l), Box::new(e)),
//             }
//         },
//         // '~' => {
//         //     // same as above
//         // }
//         '|' => match left {
//             Some(l) => {
//                 let right = parse_new_try(None, buf, idx+1)?;
//                 RegExp::Alt(Box::new(l), Box::new(right))
//             },
//             None => return Err("Can't start with a |")
//         },
//         '*' => match left {
//             Some(l) => RegExp::Star(Box::new(l)),
//             None => return Err("Can't start with a *")
//         },
//         // '?' =>
//         // '+' => ?,
//         // '(' => ,
//         // ')' => ???,
//         c => if c.is_ascii_alphanumeric() {
//             RegExp::Lit(c)
//         } else {
//             return Err("Unrecognised character WTF bro");
//         },
//     })
// }

fn asrt(s: &str, exp: RegExp) { assert_eq!(parse_wrapper(s).unwrap(), exp); }
fn asrt_f(s: &str) { parse_wrapper(s).unwrap_err(); }

#[test]
fn test_parse() {
    asrt("s", lit('s'));
    asrt_f("|");
    asrt("$", emp);
    asrt("a|b", alt(lit('a'), lit('b')));
    asrt("ab", cnc(lit('a'), lit('b')));
    asrt("a|bc", alt(lit('a'), cnc(lit('b'), lit('c'))));
    asrt("ab|c", alt(cnc(lit('a'), lit('b')), lit('c')));
    asrt("a*", ast(lit('a')));
    asrt("a**", ast(ast(lit('a'))));
    asrt("ab*", cnc(lit('a'), ast(lit('b'))));
    asrt("ab*c", cnc(cnc(lit('a'), ast(lit('b'))), lit('c')));
    asrt("a*bc", cnc(cnc(ast(lit('a')), lit('b')), lit('c')));
    asrt("abc*", cnc(cnc(lit('a'), ast(lit('b'))), ast(lit('c'))));
    // BAD: assume we can concat greedily when we can't.
    // postfix unary operators bind more tightly
}






// /// The stuff must be a correct regex
// /// end exclusive
// pub fn parse_regex(left: &RegExp, buf: &str) -> RegExp {
//     // we are looking at a non-empty slice
//     let l = buf.len()
//     debug_assert!(l > 0);
//     match c {
//         '$' => {
//             let e = RegExp::Empty;
//             if l == 1 {
//                 e
//             } else {
//                 RegExp::Concat(e, parse_regex(&buf[1..]))
//             }
//         },
//         '~' => {
//             // same as above
//         }
//         '|' => {
//             parse_regex(None, &buf[1..])
//         }
//         '*' => RegExp::Star(left.clone()),
//         // '?' =>
//         // '+' => ?,
//         '(' => ???,
//         ')' => ???,
//         _ => if c.is_ascii_alphanumeric() {
//             RegExp::Lit(c),
//         } else {
//             return Err("Unrecognised character WTF bro");
//         }
//     }
// }


// impl TryFrom<&str> for RegExp {
//     type Error = &'static str;

//     fn try_from(value: &str) -> Result<Self, Self::Error> {
//         for c in value.chars() {
//             // Need to think about brackets...
//             match c {
//                 '$' => RegExp::Empty()
//                 '~' => RegExp::Epsilon(),
//                 '|' => // or...,
//                 '*' => RegExp::Star(),
//                 // '?' =>
//                 '+' => ?,
//                 '(' => ???,
//                 ')' => ???,
//                 _ => if c.is_ascii_alphanumeric() {
//                     RegExp::Lit(c),
//                 } else {
//                     return Err("Unrecognised character WTF bro");
//                 }
//             }
//         }
//     }
// }








/// Our NFAs are a bit special: they have only zero to two
/// out transitions.
#[derive(Debug, Clone)]
struct NfaState {
    // Use option to encode epsilon and otherwise use char...
    // Use an arrayvec<2>?
    transitions: ArrayVec<(usize, Option<Char>), 2>,
}

/// A non-deterministic finite automaton.
/// The NFA is special in a few ways:
/// - Each node has 0-2 out transitions (the Thompson construction
/// doesn't need more.)
/// - The start state will always be index 0 and there will
/// always be a unique end state, which has the final index in the
/// state vec.
#[derive(Clone)]
pub struct Nfa {
    states: Vec<NfaState>,
}

fn fmt_nfa_state(s: &NfaState) {}

impl fmt::Debug for Nfa {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i,s) in self.states.iter().enumerate() {
            write!(f, "{i:2}: ")?;
            for (idx, c) in &s.transitions {
                write!(f, "{idx:2} ")?;
                match c {
                    Some(c) => write!(f, "{c}     ")?,
                    None => write!(f, "ε     ")?,
                };
            }
            writeln!(f, "")?;
        }
        Ok(())
    }
}

impl Nfa {
    pub fn from_regexp(regexp: &RegExp) -> Self {
        let mut s = Self { states: vec![] };
        let zero = s.fresh();
        let last = s.create_rec(regexp, zero);
        // The final was always the last state created
        assert_eq!(last, s.final_state());
        s
    }

    #[inline] fn start_state(&self) -> usize { 0 }
    /// There is a unique accepting state
    #[inline] fn final_state(&self) -> usize { self.states.len() - 1 }

    fn fresh(&mut self) -> usize {
        let fresh_idx = self.states.len();
        let fresh_state = NfaState { transitions: ArrayVec::new() };
        self.states.push(fresh_state);
        fresh_idx
    }

    fn create_rec(&mut self, regexp: &RegExp, qx: usize) -> usize {
        // let f = NfaState { trans1: None, trans2: None };
        match regexp {
            RegExp::Empty => {
                // Don't even bother to connect the final state
                let fx = self.fresh();
                fx
            }
            RegExp::Epsilon => {
                let fx = self.fresh();
                self.states[qx].transitions.push((fx, None));
                fx
            },
            RegExp::Lit(c) => {
                let fx = self.fresh();
                self.states[qx].transitions.push((fx, Some(*c)));
                fx
            },
            RegExp::Concat(r1, r2) => {
                let f1x = self.create_rec(r1, qx);
                let f2x = self.create_rec(r2, f1x);
                f2x
            },
            RegExp::Alt(r1, r2) => {
                // Save one state by just making q1 = q?
                let q1x = self.fresh();
                let q2x = self.fresh();
                let q = &mut self.states[qx];
                let t = &mut q.transitions;
                t.push((q1x, None));
                t.push((q2x, None));

                let f1x = self.create_rec(r1, q1x);
                let f2x = self.create_rec(r2, q2x);
                let fx = self.fresh();
                self.states[f1x].transitions.push((fx, None));
                self.states[f2x].transitions.push((fx, None));
                fx
            },
            RegExp::Star(r) => {
                // (can't save anything here)
                let q1x = self.fresh();
                let f1x = self.create_rec(r, q1x);
                let f2x = self.fresh();

                let f1 = &mut self.states[f1x];
                let t1 = &mut f1.transitions;
                t1.push((q1x, None));
                t1.push((f2x, None));
                let q = &mut self.states[qx];
                let t = &mut q.transitions;
                t.push((q1x, None));
                t.push((f2x, None));
                f2x
            },
        }
    }
}

/// Cache of epsilon closures of NFA state subsets.
struct EpsilonClosureCache<'a> {
    nfa: &'a Nfa,
    singleton_cache: Vec<Option<BitSet>>,
    cache: HashMap<BitSet, BitSet>,
}

impl<'a> EpsilonClosureCache<'a> {
    pub fn new(nfa: &'a Nfa) -> Self {
        Self {
            nfa,
            // Maybe don't allocate this? Use a hashmap?
            singleton_cache: vec![None; nfa.states.len()],
            cache: HashMap::new()
        }
    }

    /// Obtains the epsilon closure of a single NFA state.
    /// If it is not in the cache, it does the full graph search.
    /// You could imagine recursively querying the cache
    /// but it gets a bit complex in the presence of cycles.
    /// One to think about. But the number of NFA states is linear
    /// in the size of the regex, while the DFA states are potentially
    /// exponential. So this term does not dominate.
    pub fn close_single(&mut self, b: usize) -> &BitSet {
        if let None = &mut self.singleton_cache[b] {
            let mut b_eps = BitSet::with_capacity(self.nfa.states.len());
            let mut stack = vec![b];
            while let Some(next) = stack.pop() {
                b_eps.insert(next);
                let n = &self.nfa.states[next];
                for (neighbour, c) in &n.transitions {
                    if c.is_none() && !b_eps.contains(*neighbour) {
                        stack.push(*neighbour);
                    }
                }
            }

            self.singleton_cache[b] = Some(b_eps);
        }

        // TODO: express this better
        &self.singleton_cache[b].as_ref().unwrap()
    }

    /// The epsilon closure of a set of NFA states.
    /// If it is not already in the cache, the epsilon closures
    /// of each individual state are unioned together and cached.
    pub fn close(&mut self, b: &BitSet) -> &BitSet {
        if !self.cache.contains_key(b) {
            let mut b_eps = BitSet::with_capacity(self.nfa.states.len());
            for item in b.iter() {
                let b_eps_part = self.close_single(item);
                b_eps.union_with(b_eps_part);
            }
            self.cache.insert(b.clone(), b_eps);
        }
        self.cache.get(b).as_ref().unwrap()
    }
}

// subset construction

/// A deterministic finite automaton.
#[derive(Debug, Clone)]
pub struct Dfa {
    num_states: usize,
    // From, to, etc. We will probably need a way to efficiently
    // get this by node. For now just do this
    edges: Vec<(usize, Char, usize)>,
    // State bitset
    final_states: BitSet,
}

impl Dfa {
    /// The subset construction.
    pub fn from_nfa(nfa: &Nfa) -> Self {
        // Map from state bitsets to epsilon-closed state bitsets.
        let mut cache = EpsilonClosureCache::new(nfa);
        // Map from epsilon-closed state bitsets to 0-blah indices.
        let mut dfa_states = HashMap::new();
        // with capacity related to NFA size?
        // delete type help
        let mut edges: Vec<(usize, Char, usize)> = vec![];

        let start_b = cache.close_single(nfa.start_state());
        let start_idx = dfa_states.len(); // i.e. 0
        dfa_states.insert(start_b.clone(), start_idx);

        let mut stack = vec![(start_b.clone(), start_idx)];
        while let Some((dfa_state, dfa_state_idx)) = stack.pop() {
            // map from chars (non-epsilon) to bitsets of states
            // (non-epsilon-closured)
            let mut destinations = HashMap::new();
            for state in dfa_state.iter() {
                for (neighbor, c) in &nfa.states[state].transitions {
                    // No epsilon transitions
                    if let Some(c) = c {
                        destinations.entry(*c)
                            .and_modify(|b: &mut BitSet| { b.insert(*neighbor); })
                            .or_insert_with(
                                || BitSet::with_capacity(nfa.states.len()));
                    }
                }
            }

            for (c, dest) in &destinations {
                let dest_e = cache.close(&dest);
                let dest_e_idx = match dfa_states.get(dest_e) {
                    Some(dest_e_idx) => *dest_e_idx,
                    None => {
                        // Perform the "visiting" here so we have an
                        // index to map to. This means we have to visit
                        // the initial state separately instead of at the
                        // start of the loop
                        let dest_e_idx = dfa_states.len();
                        dfa_states.insert(dest_e.clone(), dest_e_idx);
                        // Have to clone here. ^?
                        stack.push((dest_e.clone(), dest_e_idx));
                        dest_e_idx
                    },
                };
                edges.push((dfa_state_idx, *c, dest_e_idx));
            }
        }

        let nfa_final = nfa.final_state();
        let final_states = dfa_states.iter()
            .filter(|(b, _)| b.contains(nfa_final))
            .map(|(_, idx)| *idx)
            .collect::<BitSet>();

        Self {
            num_states: dfa_states.len(),
            edges,
            final_states,
        }
    }
}


pub fn example1() -> RegExp {
    // (ε|0*1)
    // alt(eps, cnc(ast(lit('a')), lit('b')))
    cnc(ast(lit('a')), lit('b'))
}

// (0|(1(01*(00)*0)*1)*)*
