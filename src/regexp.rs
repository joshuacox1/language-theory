//! Messing about with a regular expression engine.
//! Very simple:
//! We take in a regular expression, construct the corresponding
//! NFA, compute the corresponding DFA via the powerset contruction,
//! then compute the (unique) minimal DFA matching the expression.
//! We can use the regex crate to fuzz I guess?

use std::collections::{HashMap, HashSet};
use std::fmt::{self, Write};

use arrayvec::ArrayVec;
use bit_set::BitSet;

// TODO: Swap to MyChar and remove the Option<...> for epsilons
pub type Char = char;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegExp {
    /// The empty set.
    Empty,
    /// The empty string language. Equivalent to $*.
    Epsilon,
    /// The single character language.
    Lit(Char),
    /// Concatenation.
    Concat(Box<RegExp>, Box<RegExp>),
    /// Union.
    Alt(Box<RegExp>, Box<RegExp>),
    /// Zero or more (Kleene star).
    Star(Box<RegExp>),
    /// Zero or one. Equivalent to ~|r.
    Opt(Box<RegExp>),
    /// One or more. Equivalent to rr*.
    Plus(Box<RegExp>),
}

impl RegExp {
    /// Returns all characters mentioned in the regex string.
    /// Note that not all characters may be present in strings
    /// accepted by the language. For example, ($b)|a recognises
    /// no strings containing the character b.
    pub fn chars_mentioned(&self) -> HashSet<Char> {
        let mut acc = HashSet::new();
        self.chars_mentioned_rec(&mut acc);
        acc
    }

    fn chars_mentioned_rec(&self, acc: &mut HashSet<Char>) {
        match self {
            RegExp::Empty | RegExp::Epsilon => (),
            RegExp::Lit(c) => { acc.insert(*c); },
            RegExp::Concat(r1, r2) => {
                r1.chars_mentioned_rec(acc);
                r2.chars_mentioned_rec(acc);
            },
            RegExp::Alt(r1, r2) => {
                r1.chars_mentioned_rec(acc);
                r2.chars_mentioned_rec(acc);
            },
            RegExp::Star(r) => r.chars_mentioned_rec(acc),
            RegExp::Opt(r) => r.chars_mentioned_rec(acc),
            RegExp::Plus(r) => r.chars_mentioned_rec(acc),
        }
    }

    /// Returns true if the language recognised by the regex
    /// is empty. (This can be tested much more efficiently than
    /// constructing a DFA).
    pub fn inhabited(&self) -> bool {
        match self {
            RegExp::Empty => false,
            RegExp::Epsilon => true,
            RegExp::Lit(_) => true,
            RegExp::Concat(r1, r2) => r1.inhabited() && r2.inhabited(),
            RegExp::Alt(r1, r2) => r1.inhabited() || r2.inhabited(),
            RegExp::Star(r) => true,
            RegExp::Opt(r) => true,
            RegExp::Plus(r) => r.inhabited(),
        }
    }
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
                // You could save making a fresh node by embedding
                // q in as one of the recursive start nodes. Unfortunately
                // that could break the property that every node has 0
                // to 2 exit transitions, which is useful for storage.
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
        }
    }

    pub fn to_graphviz(&self) -> String {
        let mut s = String::new();
        s.push_str(concat!(
            "digraph nfa {\n",
            "    node [fontname=\"'Fira Mono'\"]\n",
            "    edge [fontname=\"'Fira Mono'\"]\n",
            "    rankdir=LR;\n",
            "    node [shape = doublecircle]; ",
        ));
        writeln!(s, "{}", self.final_state());
        s.push_str("    node [shape = circle];\n");
        for (i,st) in self.states.iter().enumerate() {
            for (j, c) in &st.transitions {
                writeln!(s, "    {} -> {} [label = \"{}\"]",
                    i, j, c.unwrap_or('*'));
            }
        }
        s.push_str("}\n");
        s
    }
}

/// Cache of epsilon closures of NFA state subsets.
#[derive(Debug)]
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
/// We derive EQ (assuming edges is sorted, which it should be I guess,
/// at least when we care about EQ)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dfa {
    // The states will be 0 to n-1
    num_states: usize,
    // TODO: One buffer with start,len pointers in it for cache locality?
    edges: Vec<Vec<(char, usize)>>,
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
        let mut edges = HashSet::new();

        let start_b = cache.close_single(nfa.start_state());
        let start_idx = dfa_states.len(); // i.e. 0
        dfa_states.insert(start_b.clone(), start_idx);

        let mut stack = vec![(start_b.clone(), start_idx)];
        while let Some((dfa_state, dfa_state_idx)) = stack.pop() {
            // Can probably avoid the edges intermediate structure
            // altogether since this number just goes up??? DOes it???
            // println!("DFA_STATE_IDX: {dfa_state_idx}");
            let mut destinations = HashMap::new();
            for state in dfa_state.iter() {
                for (neighbor, c) in &nfa.states[state].transitions {
                    // No epsilon transitions
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
        // TODO: return this somewhere.
        for (i,s) in dfa_states_result.iter().enumerate() {
            println!("{i}: {s:?}");
        }
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

    /// Compute the minimal DFA (up to isomorphism).
    /// We will have computed the char set.
    pub fn minimise(&self, alphabet: &HashSet<Char>) -> Self {
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
        // let mut reachable = HashSet::new();
        // let mut stack = vec![];
        // // Backwards DFS from each final state.
        // for f in self.final_states {
        //     if !reachable.contains(&f) {
        //         stack.push(f);
        //         while let Some(next) = stack.pop() {
        //             reachable.insert(next);
        //             // For each state such that state ---c--> next {
        //             let sources = todo!();
        //             for s in sources {
        //                 if !reachable.contains(&s) {
        //                     stack.push(s);
        //                 }
        //             }
        //         }
        //     }
        // }
        // If nothing was reachable, we can skip minimisation
        // and canonicalisation below and just return the
        // single default state with no edges and non-final.
        // if reachable.is_empty() {
        //     todo!()
        // } else {
        //     todo!()
        // }
        // let backwards_states = self.edges.iter()
        //     .map(|((i, c), j)| (j, )
        //     .collect::<Vec<_>>();

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
        p.insert(f.clone());
        p.insert(q_bar_f.clone());
        let mut w = HashSet::new();
        w.insert(f.clone());
        w.insert(q_bar_f);
        while let Some(a) = w.iter().next().cloned() {
            w.remove(&a);
            for c in alphabet {
                // If there are no s such that s ---c--> a_i for a_i in a then
                // there is no point dong any of the below because
                // o1 will always be empty
                let mut x = BitSet::new();
                for a_ in a.iter() {
                    if let Some(q) = backw.get(&(a_,*c)) {
                        x.union_with(q);
                    }
                }
                // For now make a fresh hashset every time.
                // We will want to replace this with a dedicated
                // partition refinement data structure.
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
            // Arbitrary representative
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
        println!("{reduced_edges:?}");

        // Naming canonicalisation.
        let mut canonical_permutation = vec![usize::MAX; p.len()];
        let mut visited = BitSet::with_capacity(self.num_states);
        let start = lookup[0];
        let mut stack = vec![start];
        visited.insert(start);
        let mut idx = 0;
        while let Some(next) = stack.pop() {
            println!("!! {next}");
            // Inverse permutation creating a lookup
            canonical_permutation[next] = idx;
            idx += 1;
            // Canonicity comes from this being sorted in alphabetical
            // order of outgoing state character.
            if let Some(sorted_es) = reduced_edges.get(&next) {
                // Iterate in reverse to put the earliest item
                // next on the stack
                for (_,j) in sorted_es.iter().rev() {
                    if !visited.contains(*j) {
                        visited.insert(*j);
                        stack.push(*j);
                    }
                }
            }
        }
        println!("{canonical_permutation:?}");

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

    // // prob delete
    // pub fn show_string(&self) -> String {

    //     let mut z = HashMap::<usize, Vec<(Char, usize)>>::new();
    //     for ((start, c), end) in &self.edges {
    //         z.entry(*start)
    //             .and_modify(|v| v.push((*c, *end)))
    //             .or_insert_with(|| vec![(*c, *end)]);
    //     }
    //     // let mut z = z.into_iter().collect::<Vec<_>>();
    //     // z.sort_unstable_by_key(|(idx,_)| *idx);

    //     let mut s = String::new();
    //     writeln!(s, "State | Final | Edges").unwrap();
    //     for start in 0..self.num_states {
    //         let fin = self.final_states.contains(start);
    //         let fin_s = if fin { " Yes " } else { " No  " };
    //         match z.get_mut(&start) {
    //             None => {
    //                 writeln!(s, "{start:5}   {fin_s}").unwrap();
    //             },
    //             Some(edges) => {
    //                 edges.sort_unstable();
    //                 let mut first = true;
    //                 for (c, end) in edges {
    //                     if first {
    //                         writeln!(s, "{start:5}   {fin_s}   -- {c} -> {end:2}").unwrap();
    //                         first = false;
    //                     } else {
    //                         writeln!(s, "                -- {c} -> {end:2}").unwrap();
    //                     }
    //                 }
    //             }
    //         }
    //     }

    //     s
    // }

    pub fn to_graphviz(&self) -> String {
        let mut s = String::new();
        s.push_str(concat!(
            "digraph dfa {\n",
            "    node [fontname=\"'Fira Mono'\"]\n",
            "    edge [fontname=\"'Fira Mono'\"]\n",
            "    rankdir=LR;\n",
            "    node [shape = doublecircle];",
        ));
        for f in self.final_states.iter() {
            write!(s, " {f}");
        }
        s.push_str("\n    node [shape = circle];\n");
        // TODO: this won't draw vertices with no edges touching them.
        // But they should hopefully
        // never occur in this construction?
        for (i,v) in self.edges.iter().enumerate() {
            for (c,j) in v {
                writeln!(s, "    {} -> {} [label = \"{}\"]",
                    i, j, c);
            }
        }
        s.push_str("}\n");
        s
    }
}

#[derive(Debug)]
struct PartitionRefinementClass {
    // Doubly-linked list pointers
    left: usize,
    right: usize,
    // Where the items start in the buffer
    item_header: usize,
    split: usize, // We use usize::MAX as an Option as an optimisation
}

/// A partition refinement data structure on [0,...,N) for some N.
#[derive(Debug)]
struct PartitionRefinement {
    // One index per state indicating the subset each state belongs to.
    data: Vec<usize>,
    // starts and ends.
    // Logically a doubly linked list.
    classes: Vec<PartitionRefinementClass>,
}

// impl PartitionRefinement {
//     pub fn new(n: usize) {
//         Self {
//             data: vec![0; n],
//         }
//     }

//     // precondition: every item in set is in [0,...,n).
//     pub fn refine(&mut self, set: &BitSet) {
//         for x in set.iter() {
//             let s_x = self.data[x];
//             if s_x.split == usize::MAX {
//                 self.classes.push(PartitionRefinementClass {
//                     // Where does it go?!
//                 })
//                 remove i from s_x
//                 insert into split class for s_x
//             }
//         }
//     }
// }




// TEMP STUFF BEFORE I WRITE A PARSER


pub const emp: RegExp = RegExp::Empty;
pub const eps: RegExp = RegExp::Epsilon;

#[inline] pub fn lit(c: Char) -> RegExp { RegExp::Lit(c) }

pub fn cnc(r1: RegExp, r2: RegExp) -> RegExp {
    RegExp::Concat(Box::new(r1), Box::new(r2))
}

pub fn alt(r1: RegExp, r2: RegExp) -> RegExp {
    RegExp::Alt(Box::new(r1), Box::new(r2))
}

pub fn ast(r: RegExp) -> RegExp { RegExp::Star(Box::new(r)) }
pub fn opt(r: RegExp) -> RegExp { RegExp::Opt(Box::new(r)) }
pub fn pls(r: RegExp) -> RegExp { RegExp::Plus(Box::new(r)) }

pub fn example1() -> RegExp {
    // (ε|0*1)
    alt(eps, cnc(ast(lit('a')), lit('b')))
    // cnc(ast(lit('a')), lit('b'))
}

// N > 0
pub fn cnc_arr<const N: usize>(arr: [RegExp; N]) -> RegExp {
    arr.into_iter().reduce(cnc).unwrap()
}

pub fn example3() -> RegExp {
    // cnc(emp, lit('b'))
    alt(cnc(lit('b'), emp), lit('a'))
}


pub fn example2() -> RegExp {
    // (0|(1(01*(00)*0)*1)*)*
    // (1(01*0)*1|0)*
    // divisble by 3???
    // cnc_arr([
    //     lit('n'), lit('o'), ast(lit('o')), lit('p'), lit('e'),
    // ])
    // ast(
    //     alt(
    //         cnc(
    //             cnc(
    //                 lit('1'),
    //                 ast(
    //                     cnc(
    //                         cnc(
    //                             lit('0'),
    //                             ast(lit('1')),
    //                         ),
    //                         lit('0'),
    //                     )
    //                 )
    //             ),
    //             lit('1'),
    //         ),
    //         lit('0'),
    //     )
    // )
    // ast(
    //     alt(
    //         lit('0'),
    //         ast(cnc(
    //             cnc(
    //                 lit('1'),
    //                 ast(
    //                     cnc(
    //                         cnc(
    //                             cnc(
    //                                 lit('0'),
    //                                 ast(lit('1')),
    //                             ),
    //                             ast(cnc(lit('0'), lit('0'))),
    //                         ),
    //                         lit('0'),
    //                     )
    //                 )
    //             ),
    //             lit('1')
    //         ))
    //     )
    // )
    // (1(01*0)*1|0)+
    pls(
        alt(
            cnc_arr([
                lit('1'),
                ast(
                    cnc_arr([
                        lit('0'),
                        ast(lit('1')),
                        lit('0'),
                    ])
                ),
                lit('1'),
            ]),
            lit('0'),
        )
    )
}




pub fn canon_dfa(regexp: &RegExp) -> Dfa {
    let nfa = Nfa::from_regexp(regexp);
    let dfa = Dfa::from_nfa(&nfa);
    let canon_min_dfa = dfa.minimise(&regexp.chars_mentioned());
    canon_min_dfa
}


mod test {
    use super::*;

    #[test]
    fn dfa_example() {
        // TODO: Write the parser
        let a_or_b_star = ast(alt(lit('a'), lit('b')));
        let t_abb_t = cnc_arr([
            a_or_b_star.clone(),
            lit('a'), lit('b'), lit('b'),
            a_or_b_star,
        ]);
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

        let actual_dfa = canon_dfa(&t_abb_t);
        assert_eq!(expected_canonical_dfa, actual_dfa);
    }

    #[test]
    fn dfa_with_uninhabited() {
        // Everyone's favourite nightmare counterexample
        let only_a = alt(lit('a'), cnc(lit('b'), emp));
        let expected_canonical_dfa = Dfa {
            num_states: 2,
            edges: vec![
                vec![('a',1)],
                vec![],
            ],
            final_states: [1].into_iter().collect::<BitSet>(),
        };

        let actual_dfa = canon_dfa(&only_a);
        assert_eq!(expected_canonical_dfa, actual_dfa);
    }

    #[test]
    fn regex_chars_1() {
        // TODO: Write the parser
        let abcd = cnc_arr([
            lit('a'),
            lit('b'),
            lit('c'),
            lit('d'),
        ]);
        let expected_inhab = ['a','b', 'c', 'd']
            .into_iter().collect::<HashSet<_>>();
        let actual_inhab = abcd.chars_mentioned();
        assert_eq!(expected_inhab, actual_inhab);
    }

    #[test]
    fn regex_chars_2() {
        // TODO: Write the parser
        let a_or_b_star = ast(alt(lit('a'), lit('b')));
        let t_abb_t = cnc_arr([
            a_or_b_star.clone(),
            lit('a'), lit('b'), lit('b'),
            a_or_b_star,
        ]);
        let expected_inhab = ['a','b'].into_iter().collect::<HashSet<_>>();
        let actual_inhab = t_abb_t.chars_mentioned();
        assert_eq!(expected_inhab, actual_inhab);
    }

    #[test]
    fn regex_chars_uninhabited() {
        // TODO: Write the parser
        let only_a = alt(lit('a'), cnc(lit('b'), emp));
        // Change this to 'a'
        let expected_inhab = ['a'].into_iter().collect::<HashSet<_>>();
        let actual_inhab = only_a.chars_mentioned();
        assert_eq!(expected_inhab, actual_inhab);
    }
}
