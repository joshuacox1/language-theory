//! A fast toy regular expression engine. Compiles regular
//! expressions to the unique minimal finite state machine
//! up to isomorphism
//! that recognises the equivalent language. By ordering states
//! in a consistent way the resulting machine is canonical:
//! two regular expressions recognise equal languages if and only if
//! they have equal (not just isomorphic) finite state machines.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::{self, Write};

use arrayvec::ArrayVec;
use bit_set::BitSet;

pub type RegExpError = String;

/// Not sure if should derive this
#[derive(PartialEq, Eq)]
pub struct StateMachine(Dfa);

impl StateMachine {
    /// Compiles the `RegExp` to a finite state machine that recognises
    /// the same language. This state machine is minimal (has the
    /// fewest possible states) and canonical (meaning that two `RegExp`s
    /// recognise the same language if and only if their state machines
    /// are equal). This is a consequence of the [Myhill&ndash;Nerode
    /// theorem](https://en.wikipedia.org/wiki/Myhill%E2%80%93Nerode_theorem)
    /// and relabelling states consistently (in order of
    /// the [shortlex](https://en.wikipedia.org/wiki/Shortlex_order)-earliest
    /// word in that state's equivalence class).
    /// Before I write a parser, only accept postfix notation.
    pub fn new(regex_str: &str) -> Result<Self, RegExpError> {
        let mut stack = Vec::with_capacity(regex_str.len());
        for c in regex_str.chars() {

            match c {
                '$' => stack.push(RegExp::Empty),
                '~' => stack.push(RegExp::Epsilon),
                ',' => if stack.len() >= 2 {
                    let r2 = stack.pop().unwrap();
                    let r1 = stack.pop().unwrap();
                    stack.push(RegExp::Concat(Box::new(r1), Box::new(r2)));
                } else {
                    return Err("No pair of redexes to apply , to".to_string());
                },
                '|' => if stack.len() >= 2 {
                    let r2 = stack.pop().unwrap();
                    let r1 = stack.pop().unwrap();
                    stack.push(RegExp::Alt(Box::new(r1), Box::new(r2)));
                } else {
                    return Err("No pair of redexes to apply | to".to_string());
                },
                '*' => if stack.len() >= 1 {
                    let r = stack.pop().unwrap();
                    stack.push(RegExp::Star(Box::new(r)));
                } else {
                    return Err("No redex to apply * to".to_string());
                },
                '?' => if stack.len() >= 1 {
                    let r = stack.pop().unwrap();
                    stack.push(RegExp::Opt(Box::new(r)));
                } else {
                    return Err("No redex to apply ? to".to_string());
                },
                '+' => if stack.len() >= 1 {
                    let r = stack.pop().unwrap();
                    stack.push(RegExp::Plus(Box::new(r)));
                } else {
                    return Err("No redex to apply * to".to_string());
                },
                '!' => if stack.len() >= 1 {
                    let r = stack.pop().unwrap();
                    stack.push(RegExp::Neg(Box::new(r)));
                } else {
                    return Err("No redex to apply ! to".to_string());
                },
                '&' => if stack.len() >= 2 {
                    let r2 = stack.pop().unwrap();
                    let r1 = stack.pop().unwrap();
                    stack.push(RegExp::And(Box::new(r1), Box::new(r2)));
                } else {
                    return Err("No pair of redexes to apply & to".to_string());
                },
                '\\' => if stack.len() >= 2 {
                    let r2 = stack.pop().unwrap();
                    let r1 = stack.pop().unwrap();
                    stack.push(RegExp::Diff(Box::new(r1), Box::new(r2)));
                } else {
                    return Err("No pair of redexes to apply | to".to_string());
                },
                '^' => if stack.len() >= 2 {
                    let r2 = stack.pop().unwrap();
                    let r1 = stack.pop().unwrap();
                    stack.push(RegExp::SymDiff(Box::new(r1), Box::new(r2)));
                } else {
                    return Err("No pair of redexes to apply | to".to_string());
                },
                ' ' | '\n' | '\r' | '\t' => (),
                _ => if c.is_ascii_alphanumeric() {
                    stack.push(RegExp::Lit(c));
                } else {
                    return Err(format!("Unrecognised character {c}"));
                }
            }
        }

        match stack.len() {
            0 => return Err("Nothing to reduce".to_string()),
            i if i >= 2 => return Err("Too many redexes left over".to_string()),
            _ => (),
        }

        let regexp = &stack[0];

        let nfa = Nfa::from_regexp(regexp);
        let dfa = Dfa::from_nfa(&nfa);
        let canon_min_dfa = dfa.minimise(&regexp.chars());
        Ok(Self(canon_min_dfa))
    }

    /// Decides whether the string matches the regex.
    pub fn decide(&self, input: &str) -> bool { self.0.decide(input) }

    /// Returns a string in valid graphviz (.gv) format showing
    /// the structure of the state machine.
    pub fn to_graphviz(&self) -> String { self.0.to_graphviz() }

    /// Returns a string containing a valid Rust function with the
    /// supplied name that recognises the function. Useful for
    /// compile-time recogniser construction.
    pub fn rust_codegen(&self, fn_name: &str) -> String {
        self.0.rust_codegen(fn_name)
    }
}

/// A regular expression.
///
/// Only `Empty`, `Lit`, `Concat`, `Alt` and `Star` are fundamental;
/// `Epsilon`, `Opt` and `Plus` are derived forms.
/// But they are common abbreviations and are also handled more
/// efficiently than an equivalent decomposed form.
#[derive(Debug, Clone, PartialEq, Eq)]
enum RegExp {
    /// The empty language.
    Empty,
    /// The empty string language. Equivalent to `Star(Empty)`.
    Epsilon,
    /// The single character language for the given character.
    Lit(char),
    /// Concatenation.
    Concat(Box<RegExp>, Box<RegExp>),
    /// Union.
    Alt(Box<RegExp>, Box<RegExp>),
    /// Zero or more (Kleene star).
    Star(Box<RegExp>),
    /// Zero or one. `Opt(r)` is equivalent to `Alt(Epsilon, r)`.
    Opt(Box<RegExp>),
    /// One or more. `Plus(r)` is equivalent to `Concat(r, Star(r))`.
    Plus(Box<RegExp>),
    /// Scary things!
    Neg(Box<RegExp>),
    /// Intersection
    And(Box<RegExp>, Box<RegExp>),
    /// Difference
    Diff(Box<RegExp>, Box<RegExp>),
    /// Symmetric difference
    SymDiff(Box<RegExp>, Box<RegExp>),
}

impl RegExp {
    fn chars(&self) -> HashSet<char> {
        let mut acc = HashSet::new();
        self.chars_rec(&mut acc);
        acc
    }

    fn chars_rec(&self, acc: &mut HashSet<char>) {
        match self {
            RegExp::Empty | RegExp::Epsilon => (),
            RegExp::Lit(c) => { acc.insert(*c); },
            RegExp::Concat(r1, r2) => {
                r1.chars_rec(acc);
                r2.chars_rec(acc);
            },
            RegExp::Alt(r1, r2) => {
                r1.chars_rec(acc);
                r2.chars_rec(acc);
            },
            RegExp::Star(r) => r.chars_rec(acc),
            RegExp::Opt(r) => r.chars_rec(acc),
            RegExp::Plus(r) => r.chars_rec(acc),
            RegExp::Neg(r) => r.chars_rec(acc),
            RegExp::And(r1, r2) => {
                r1.chars_rec(acc);
                r2.chars_rec(acc);
            },
            RegExp::Diff(r1, r2) => {
                r1.chars_rec(acc);
                r2.chars_rec(acc);
            },
            RegExp::SymDiff(r1, r2) => {
                r1.chars_rec(acc);
                r2.chars_rec(acc);
            },
        }
    }

    /// TODO delete this once the tests are over
    fn compile(&self) -> Dfa {
        let nfa = Nfa::from_regexp(self);
        let dfa = Dfa::from_nfa(&nfa);
        let canon_min_dfa = dfa.minimise(&self.chars());
        canon_min_dfa
    }
}

/// Our NFAs are a bit special: they have only zero to two
/// out transitions.
#[derive(Debug, Clone)]
struct NfaState {
    // TODO: stop using Option<char> and start using some u8
    transitions: ArrayVec<(usize, Option<char>), 2>,
}

/// A non-deterministic finite automaton.
/// The NFA is special in a few ways:
/// - Each node has 0-2 out transitions (the Thompson construction
/// doesn't need more.)
/// - The start state will always be index 0 and there will
/// always be a unique end state, which has the final index in the
/// state vec.
#[derive(Clone)]
struct Nfa {
    states: Vec<NfaState>,
}

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
    fn from_regexp(regexp: &RegExp) -> Self {
        let mut s = Self { states: vec![] };
        let zero = s.fresh();
        let last = s.create_rec(regexp, zero);
        assert_eq!(0, zero);
        assert_eq!(last, s.final_state());
        s
    }

    #[inline] fn start_state(&self) -> usize { 0 }
    #[inline] fn final_state(&self) -> usize { self.states.len() - 1 }

    fn fresh(&mut self) -> usize {
        let fresh_idx = self.states.len();
        let fresh_state = NfaState { transitions: ArrayVec::new() };
        self.states.push(fresh_state);
        fresh_idx
    }

    fn create_rec(&mut self, regexp: &RegExp, qx: usize) -> usize {
        match regexp {
            RegExp::Empty => {
                self.fresh()
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
            RegExp::Opt(r) => {
                let q1x = self.fresh();
                let f1x = self.create_rec(r, q1x);
                let f2x = self.fresh();

                let q_t = &mut self.states[qx].transitions;
                q_t.push((q1x, None));
                q_t.push((f2x, None));
                self.states[f1x].transitions.push((f2x, None));
                f2x
            },
            RegExp::Plus(r) => {
                // Can save on making a node here: no need to make
                // a q1 since we can embed q in.
                // Like with star we can't embed the final node without
                // breaking the max 2 transitions property.
                let f1x = self.create_rec(r, qx);
                let f2x = self.fresh();

                let f1_t = &mut self.states[f1x].transitions;
                f1_t.push((qx, None));
                f1_t.push((f2x, None));
                f2x
            },
            RegExp::Neg(_)
                | RegExp::And(_, _)
                | RegExp::Diff(_, _)
                | RegExp::SymDiff(_, _) => panic!("Not supported yet."),
        }
    }

    // TODO: Remove because it's debug-y stuff?
    // pub fn to_graphviz(&self) -> String {
    //     let mut s = String::new();
    //     s.push_str(concat!(
    //         "digraph nfa {\n",
    //         "    node [fontname=\"'Fira Mono'\"]\n",
    //         "    edge [fontname=\"'Fira Mono'\"]\n",
    //         "    rankdir=LR;\n",
    //         "    node [shape = doublecircle]; ",
    //     ));
    //     writeln!(s, "{}", self.final_state());
    //     s.push_str("    node [shape = circle];\n");
    //     for (i,st) in self.states.iter().enumerate() {
    //         for (j, c) in &st.transitions {
    //             writeln!(s, "    {} -> {} [label = \"{}\"]",
    //                 i, j, c.unwrap_or('*'));
    //         }
    //     }
    //     s.push_str("}\n");
    //     s
    // }
}

/// Cache of epsilon closures of NFA state subsets.
#[derive(Debug)]
struct EpsilonClosureCache<'a> {
    nfa: &'a Nfa,
    singleton_cache: HashMap<usize, BitSet>,
    cache: HashMap<BitSet, BitSet>,
}

impl<'a> EpsilonClosureCache<'a> {
    fn new(nfa: &'a Nfa) -> Self {
        Self {
            nfa,
            singleton_cache: HashMap::with_capacity(nfa.states.len()),
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
    /// TODO: Implement this more cleverly without DFSing from
    /// everything every time
    fn close_single(&mut self, b: usize) -> &BitSet {
        if let None = &mut self.singleton_cache.get(&b) {
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

            self.singleton_cache.insert(b, b_eps);
        }

        // TODO: express all of this better if possible
        &self.singleton_cache.get(&b).as_ref().unwrap()
    }

    /// The epsilon closure of a set of NFA states.
    /// If it is not already in the cache, the epsilon closures
    /// of each individual state are unioned together and cached.
    fn close(&mut self, b: &BitSet) -> &BitSet {
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




// Hashset of (usize, char, usize)?
// (usize, char) -> usize
// usize -> (char -> usize)
// usize -> [(char, usize)]

// For reachability we want fast iteration over all neighbours
// For liveness and Hopcroft we want fast backwards iteration!
// We need to be able to chop out states easily, removing
// all their edges (don't necessarily need to be able to do this
// extremely fast)





/// A deterministic finite automaton.
/// We derive EQ (assuming edges is sorted, which it should be I guess,
/// at least when we care about EQ)
#[derive(Debug, Clone, PartialEq, Eq)]
struct Dfa {
    // The states will be 0 to n-1
    num_states: usize,
    // TODO: One buffer with start,len pointers in it for cache locality?
    edges: Vec<Vec<(char, usize)>>,
    final_states: BitSet,
}

impl Dfa {
    /// The subset construction.
    fn from_nfa(nfa: &Nfa) -> Self {
        let mut cache = EpsilonClosureCache::new(nfa);
        let mut dfa_states = HashMap::new();
        let mut edges = HashSet::new();

        let start_b = cache.close_single(nfa.start_state());
        let start_idx = 0;
        dfa_states.insert(start_b.clone(), start_idx);

        let mut stack = vec![(start_b.clone(), start_idx)];
        while let Some((dfa_state, dfa_state_idx)) = stack.pop() {
            // For this epsilon-closed subset compute the destination
            // subset from transitions by each character.
            let mut destinations = HashMap::new();
            for state in dfa_state.iter() {
                for (neighbor, c) in &nfa.states[state].transitions {
                    if let Some(c) = c {
                        destinations.entry(*c)
                            .and_modify(|b: &mut BitSet| {
                                b.insert(*neighbor);
                            })
                            .or_insert_with( || {
                                let mut b = BitSet::with_capacity(
                                    nfa.states.len(),
                                );
                                b.insert(*neighbor);
                                b
                            });
                    }
                }
            }

            // Epsilon close each destination subset and add an
            // edge from this by the given character to that. If this
            // is not in the epsilon closure cache it is a new
            // DFA state and it should be added to the stack.
            for (c, dest) in &destinations {
                let dest_e = cache.close(&dest);
                let dest_e_idx = match dfa_states.get(dest_e) {
                    Some(dest_e_idx) => *dest_e_idx,
                    None => {
                        let dest_e_idx = dfa_states.len();
                        dfa_states.insert(dest_e.clone(), dest_e_idx);
                        stack.push((dest_e.clone(), dest_e_idx));
                        dest_e_idx
                    },
                };
                edges.insert((dfa_state_idx, *c, dest_e_idx));
            }
        }

        let nfa_final = nfa.final_state();
        let final_states = dfa_states.iter()
            .filter(|(b, _)| b.contains(nfa_final))
            .map(|(_, idx)| *idx)
            .collect::<BitSet>();

        let mut dfa_states_indexed = dfa_states.into_iter()
            .collect::<Vec<_>>();
        dfa_states_indexed.sort_unstable_by_key(|(_, idx)| *idx);
        let dfa_states_result = dfa_states_indexed.into_iter()
            .map(|(s, _)| s)
            .collect::<Vec<_>>();

        let num_states = dfa_states_result.len();
        let mut edge_vecs = (0..num_states)
            .map(|_| vec![]).collect::<Vec<_>>();
        for (i,c,j) in edges {
            edge_vecs[i].push((c,j));
        }
        Self {
            num_states: dfa_states_result.len(),
            edges: edge_vecs,
            final_states,
        }
    }

    /// Canonicalises by shortlex while we're at it
    fn minimise(&self, alphabet: &HashSet<char>) -> Self {
        // Step 1 (unreachable states):
        // The subset construction only created vertices reachable
        // from the root and so automatically culled unreachable
        // vertices from the NFA (which can arise due to $).
        // So no raw reachability check is required.
        // If we have more general DFAs we will have to cull
        // unreachable states. Can bring it into the below with more
        // bookkeeping

        // Step 2 (dead states):
        // We may have made these, though. Multiple in fact (try `(b$)|a`).
        // We will need to be careful to keep or recover the original
        // state even if it is a dead state since it has special
        // semantics as the start state.
        // TODO: should this just be how the edge map works.
        // In the sense that we don't have to unify the DFA rep
        // for the intermediate DFA and the final DFA.
        let mut should_this_just = HashMap::new();
        for (i,v) in self.edges.iter().enumerate() {
            for (_,j) in v {
                should_this_just.entry(*j)
                    .and_modify(|v: &mut Vec<_>| { v.push(i); })
                    .or_insert_with(|| { vec![i]});
            }
        };

        // 2. Dead state removal.
        // Backwards DFS from each final state.
        let mut dead_states = (0..self.num_states).collect::<BitSet>();
        let mut stack = vec![];
        for f in &self.final_states {
            if dead_states.contains(f) {
                stack.push(f);
                while let Some(next) = stack.pop() {
                    dead_states.remove(next);
                    // For each state such that state ---c--> next {
                    // let sources = todo!();
                    if let Some(ss) = should_this_just.get(&next) {
                        for s in ss {
                            if dead_states.contains(*s) {
                                stack.push(*s);
                            }
                        }
                    }
                }
            }
        }
        // Now this is the problem. We have a bitset of dead
        // states, but we can't efficiently remove states because
        // our DFA representation is unhelpful. TODO: fix this.
        // I think we will need to change the DFA representation
        // to be less fixed. Probably hashsets with not-necessarily
        // contiguous int IDs...
        // If the start state is dead, don't remove it because it's
        // special (semantically it needs to exist). In this case
        // we have the empty language and we can return early.
        // if dead_states.contains(0) {
        //     return Dfa {
        //         num_states: 1,
        //         edges: vec![],
        //         final_states: BitSet::new(),
        //     }
        // }

        // Map of {i : i ---c--> j} for all (j,c).
        let mut backw = HashMap::new();
        for (i,v) in self.edges.iter().enumerate() {
            for (c,j) in v {
                backw.entry((*j,*c))
                    .and_modify(|v: &mut BitSet| { v.insert(i); })
                    .or_insert_with(|| {
                        let mut b = BitSet::new();
                        b.insert(i);
                        b
                    });
            }
        }

        // Step 3: combine indistinguishable states using
        // Hopcroft's algorithm.

        // TODO. This is a relatively naive implementation of
        // Hopcroft's algorithm. It is not notably inefficient, but
        // there exist specialist data structures for partition
        // refinement.
        let q = (0..self.num_states).collect::<BitSet>();
        let f = &self.final_states;
        let q_bar_f = q.difference(&f).collect::<BitSet>();
        let mut p = HashSet::new();
        let mut w = HashSet::new();
        if !f.is_empty() {
            p.insert(f.clone());
            w.insert(f.clone());
        }
        if !q_bar_f.is_empty() {
            p.insert(q_bar_f.clone());
            w.insert(q_bar_f);
        }
        while let Some(a) = w.iter().next().cloned() {
            w.remove(&a);
            for c in alphabet {
                let mut x = BitSet::new();
                for a_ in a.iter() {
                    if let Some(q) = backw.get(&(a_,*c)) {
                        x.union_with(q);
                    }
                }

                let mut p_prime = HashSet::new();
                for y in p.into_iter() {
                    let o1 = x.intersection(&y).collect::<BitSet>();
                    let o2 = y.difference(&x).collect::<BitSet>();
                    if !o1.is_empty() && !o2.is_empty() {
                        p_prime.insert(o1.clone());
                        p_prime.insert(o2.clone());

                        if w.remove(&y) {
                            w.insert(o1.clone());
                            w.insert(o2.clone());
                        } else {
                            if o1.len() <= o2.len() {
                                w.insert(o1.clone());
                            } else {
                                w.insert(o2.clone());
                            }
                        }
                    } else {
                        p_prime.insert(y);
                    }
                }
                p = p_prime;
            }
        }

        // TODO: make this the actual representation in Dfa or something
        let mut edge_lookup = HashMap::new();
        for (i,v) in self.edges.iter().enumerate() {
            for (c,j) in v {
                edge_lookup.entry(i)
                    .and_modify(|v: &mut Vec<_>| v.push((*c,*j)))
                    .or_insert_with(|| vec![(*c,*j)]);
            }
        }

        // Make a vec to references to bitsets in P
        let mut lookup = vec![usize::MAX; self.num_states];
        for (i,partition) in p.iter().enumerate() {
            for x in partition.iter() {
                lookup[x] = i;
            }
        }

        let mut reduced_edges = HashMap::new();
        for partition in p.iter() {
            // All partition items are non-empty. We ensure to only
            // add F and Q\F initially if they are non-empty and then
            // this is kept as an invariant throughout Hopcroft's
            // algorithm
            let a = partition.iter().next().unwrap();
            let mut items = vec![];
            // All arrows from a (they will be the same from every rep)
            if let Some(edges) = edge_lookup.get(&a) {
                for (c,j) in edges.iter() {
                    items.push((c, lookup[*j]));
                }
            }
            // TODO: Prevent the need for this by sorting
            // earlier in edge_lookup
            items.sort_unstable_by_key(|(c,_)| *c);
            reduced_edges.insert(lookup[a], items);
        }

        // Naming canonicalisation.
        let mut canonical_permutation = vec![usize::MAX; p.len()];
        let mut visited = BitSet::with_capacity(self.num_states);
        let start = lookup[0];
        // Queue in order to achieve shortlex order
        let mut queue = VecDeque::from([start]);
        visited.insert(start);
        let mut idx = 0;
        while let Some(next) = queue.pop_front() {
            // Inverse permutation creating a lookup
            canonical_permutation[next] = idx;
            idx += 1;
            // Canonicity comes from this being sorted in alphabetical
            // order of outgoing state character.
            if let Some(sorted_es) = reduced_edges.get(&next) {
                for (_,j) in sorted_es {
                    if !visited.contains(*j) {
                        visited.insert(*j);
                        queue.push_back(*j);
                    }
                }
            }
        }
        // println!("{canonical_permutation:?}");

        let num_states = canonical_permutation.len();
        assert!(!canonical_permutation.iter().any(|&q| q == usize::MAX));
        // Canonical start state is always zero
        let final_states = self.final_states.iter()
            .map(|f| canonical_permutation[lookup[f]])
            .collect::<BitSet>();
        let mut canon_edges = (0..canonical_permutation.len())
            .map(|_| vec![]).collect::<Vec<_>>();

        for (i, sorted_es) in reduced_edges {
            let v_i = &mut canon_edges[canonical_permutation[i]];
            for (c,j) in sorted_es {
                v_i.push((*c,canonical_permutation[j]));
            }
        }

        Self {
            num_states,
            edges: canon_edges,
            final_states,
        }
    }

    /// Relabel states in order of the earliest string in shortlex
    /// order that reaches that state.
    /// PRECONDITION: the DFA has no unreachable states. Otherwise
    /// things will go wrong. There is no reason to canonicalise
    /// a DFA with unreachable states as said DFA is defective.
    fn canonicalise() {

    }

    fn decide(&self, input: &str) -> bool {
        todo!()
    }

    fn to_graphviz(&self) -> String {
        let mut s = String::new();
        s.push_str(concat!(
            "digraph dfa {\n",
            "    node [fontname=\"'Fira Mono'\"]\n",
            "    edge [fontname=\"'Fira Mono'\"]\n",
            "    rankdir=LR;\n",
            "    node [shape = doublecircle];",
        ));
        for f in self.final_states.iter() {
            write!(s, " {f}").unwrap();
        }
        s.push_str("\n    node [shape = circle];\n");
        // TODO: this won't draw vertices with no edges touching them.
        // But they should hopefully
        // never occur in this construction?
        for (i,v) in self.edges.iter().enumerate() {
            for (c,j) in v {
                writeln!(s, "    {} -> {} [label = \"{}\"]",
                    i, j, c).unwrap();
            }
        }
        s.push_str("}\n");
        s
    }

    /// You could write a proc macro for this.
    /// Let's just hack strings together instead though.
    fn rust_codegen(&self, fn_name: &str) -> String {
        let mut result = String::new();

        write!(result, concat!(
            "fn {}(input: &str) -> bool {{\n",
            "    let mut s = 0;\n",
            "    for c in s.chars() {{\n",
            "        s = match (s,c) {{\n",
        ), fn_name).unwrap();
        for (i,v) in self.edges.iter().enumerate() {
            for (c,j) in v {
                write!(result, "            ({i},'{c}') => {j},\n").unwrap();
            }
        }
        write!(result, concat!(
            "            _ => return false,\n",
            "        }}\n",
            "    }}\n",
            "\n",
            "    matches!(s, ",
        )).unwrap();
        let mut first = true;
        for f in self.final_states.iter() {
            if first {
                write!(result, "{}", f).unwrap();
                first = false;
            } else {
                write!(result, " | {}", f).unwrap();
            }
        }
        write!(result, ")\n}}\n").unwrap();
        result
    }
}

#[cfg(test)]
mod test {
    use super::*;

    // Normal
    const EXAMPLE1: &'static str = "ab|*a,b,b,ab|*,";
    const EXAMPLE2: &'static str = "101*,0,*,1,0|+";
    // Min-DFA has 2 states, both accepting
    const EXAMPLE3: &'static str = "ab.*a?.";
    // The Thompson NFA construction and DFA subset construction
    // leaves an dead state with this regex that must be removed
    const EXAMPLE4: &'static str = "b$,a|";
    // This one grows exponentially in the number of ab|,s at the end
    const EXAMPLE5: &'static str = "ab|*b,ab|,ab|,ab|,ab|,";
    const EXAMPLE6: &'static str = "$";
    const EXAMPLE7: &'static str = "abcde$";

    #[test]
    fn dfa_example() {
        let expected_canonical_dfa = Dfa {
            num_states: 4,
            edges: vec![
                vec![('a',1), ('b',0)],
                vec![('a',1), ('b',2)],
                vec![('a',1), ('b',3)],
                vec![('a',3), ('b',3)],
            ],
            final_states: [3].into_iter().collect::<BitSet>(),
        };

        let actual_dfa = StateMachine::new(EXAMPLE1).unwrap().0;
        assert_eq!(expected_canonical_dfa, actual_dfa);
    }

    #[test]
    fn dfa_with_uninhabited() {
        // Everyone's favourite nightmare counterexample
        // b$ not $b
        let expected_canonical_dfa = Dfa {
            num_states: 2,
            edges: vec![
                vec![('a',1)],
                vec![],
            ],
            final_states: [1].into_iter().collect::<BitSet>(),
        };

        let actual_dfa = StateMachine::new(EXAMPLE4).unwrap().0;
        assert_eq!(expected_canonical_dfa, actual_dfa);
    }
}
