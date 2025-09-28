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


#[derive(Debug, Clone)]
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
    pub fn new(nfa: &Nfa) -> Self {
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
            // Now we compute it recursively.
            // You could imagine reusing other values by computing
            // recursively but it gets really grim in the presence
            // of cycles. Forget it.
            let mut b_eps = BitSet::with_capacity(self.nfa.states.len());
            let stack = vec![b];
            while let Some(next) = stack.pop() {
                b_eps.insert(next);
                let n = self.nfa.states[next];
                for (neighbour, c) in &n.transitions {
                    if c.is_none() && !b_eps.contains(*neighbour) {
                        stack.push(*neighbour);
                    }
                }
            }

            self.singleton_cache[b] = Some(b_eps);
        }

        // TODO: express this better
        &self.singleton_cache[b].unwrap()
    }

    /// The epsilon closure of a set of NFA states.
    /// If it is not already in the cache, the epsilon closures
    /// of each individual state are unioned together and cached.
    pub fn close(&mut self, b: &BitSet) -> &BitSet {
        self.cache.entry(b)
            .or_insert_with(|| {
                let mut b_eps = BitSet::with_capacity(self.nfa.states.len());
                for item in b.iter() {
                    let b_eps_part = self.close_single(item);
                    b_eps.union_with(b_eps_part);
                }
                /*(b.clone(),*/&b_eps
            })
    }
}

// subset construction

/// A deterministic finite automaton.
pub struct Dfa {
    num_states: usize,
    // From, to, etc. We will probably need a way to efficiently
    // get this by node. For now just do this
    edges: Vec<(usize, Char, usize)>,
    // State bitset
    final_states: BitSet,
}

impl Dfa {

    /// subset construction.
    fn from_nfa(nfa: &Nfa) -> Self {
        // Map from state bitsets to epsilon-closed state bitsets.
        let mut cache = EpsilonClosureCache::new(nfa);
        // Map from epsilon-closed state bitsets to 0-blah indices.
        let mut dfa_states = HashMap::new();
        // with capacity related to NFA size?
        // delete type help
        let mut edges: Vec<(usize, Char, usize)> = vec![];

        let start_b = cache.close_single(nfa.start_state());
        dfa_states.insert((start_b.clone(), dfa_states.len()));

        let stack = vec![start_b];
        while let Some((dfa_state, dfa_state_idx)) = stack.pop() {
            // map from chars (non-epsilon) to bitsets of states
            // (non-epsilon-closured)
            let mut destinations = HashMap::new();
            for state in dfa_state.iter() {
                for (neighbor, c) in nfa.states[state].transitions {
                    // No epsilon transitions
                    if let Some(c) = c {
                        destinations.entry(&c)
                            .and_modify(|b| b.insert(neighbor))
                            .or_insert_with(
                                || BitSet::with_capacity(nfa.states.len()));
                    }
                }
            }

            for (c, dest) in &destinations {
                let dest_e = cache.close(&dest);
                let dest_e_idx = match dfa_states.get(dest_e) {
                    Some(dest_e_idx) => dest_e_idx,
                    None => {
                        // Perform the "visiting" here so we have an
                        // index to map to. This means we have to visit
                        // the initial state separately instead of at the
                        // start of the loop
                        let dest_e_idx = dfa_states.len();
                        dfa_states.insert(dest_e.clone(), dest_e_idx);
                        // Have to clone here. ^?
                        stack.push((dest_e, dest_e_idx));
                        dest_e_idx
                    },
                };
                edges.insert((dfa_state_idx, *c, dest_e_idx));
            }
        }

        let nfa_final = nfa.final_state();
        let final_states = dfa_states.iter()
            .filter(|(b, _)| b.contains(nfa_final))
            .map(|(_, idx)| idx)
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
    alt(eps, cnc(ast(lit('a')), lit('b')))
}

// (0|(1(01*(00)*0)*1)*)*
