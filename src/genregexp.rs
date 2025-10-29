use std::collections::{HashMap, HashSet};
use std::fmt;

// tod impl debug/display char based

/// A-Z, a-z, 0-9. 0xFF is reserved as epsilon transition.
#[derive(Copy, Clone, PartialOrd, Ord, PartialEq, Eq, Hash)]
pub struct Char(u8);

impl Char {
    pub fn from_char(c: char) -> Option<Self> {
        if c.is_ascii_lowercase() {
            Some(Self(c as u8))
        } else {
            None
        }
    }

    pub const EPSILON: Char = Char(0xFF);
}

impl fmt::Debug for Char {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0 as char)
    }
}


// impl debug/display char based
// tryfrom
const ALPHABET: [Char; 26] = [
    Char('a' as u8),
    Char('b' as u8),
    Char('c' as u8),
    Char('d' as u8),
    Char('e' as u8),
    Char('f' as u8),
    Char('g' as u8),
    Char('h' as u8),
    Char('i' as u8),
    Char('j' as u8),
    Char('k' as u8),
    Char('l' as u8),
    Char('m' as u8),
    Char('n' as u8),
    Char('o' as u8),
    Char('p' as u8),
    Char('q' as u8),
    Char('r' as u8),
    Char('s' as u8),
    Char('t' as u8),
    Char('u' as u8),
    Char('v' as u8),
    Char('w' as u8),
    Char('x' as u8),
    Char('y' as u8),
    Char('z' as u8),
];


/// A generalised regular expression.
#[derive(Debug, Clone, PartialEq, Eq)]
enum GenRegex {
    /// The empty language.
    Nothing,
    /// The complete language.
    All,
    /// The empty string language. Equivalent to `Star(Empty)`.
    Epsilon,
    /// Any single character.
    AnyChar,
    /// The given single character.
    Lit(Char),
    /// Zero or more (Kleene star).
    Star(Box<GenRegex>),
    /// Zero or one.
    Opt(Box<GenRegex>),
    /// One or more.
    Plus(Box<GenRegex>),
    /// Complement.
    Complement(Box<GenRegex>),
    /// Concatenation.
    Concat(Box<GenRegex>, Box<GenRegex>),
    /// Union.
    Union(Box<GenRegex>, Box<GenRegex>),
    /// Intersection.
    Intersect(Box<GenRegex>, Box<GenRegex>),
    /// Difference.
    Diff(Box<GenRegex>, Box<GenRegex>),
    /// Symmetric difference.
    SymDiff(Box<GenRegex>, Box<GenRegex>),
}

impl GenRegex {
    pub fn to_min_canon_dfa(&self) -> Dfa {
        let d = self.to_min_dfa_inner();
        // Minimality implies all states are reachable and hence
        // canonicalisation will always work
        d.canonicalise().unwrap()
    }

    fn to_min_dfa_inner(&self) -> Dfa {
        match self {
            GenRegex::Nothing => Dfa::nothing(),
            GenRegex::All => Dfa::all(),
            GenRegex::Epsilon => Dfa::emptystring(),
            GenRegex::AnyChar => Dfa::anychar(),
            GenRegex::Lit(c) => Dfa::lit(*c),
            GenRegex::Star(r) => r.to_min_dfa_inner().star().minimise(),
            GenRegex::Opt(r) => r.to_min_dfa_inner().opt().minimise(),
            GenRegex::Plus(r) => r.to_min_dfa_inner().plus().minimise(),
            GenRegex::Complement(r) => {
                // Can be done in-place efficiently so why not
                let mut d = r.to_min_dfa_inner();
                d.complement();
                d
            }
            GenRegex::Concat(r, s) => r.to_min_dfa_inner()
                .concat(&s.to_min_dfa_inner()).minimise(),
            GenRegex::Union(r, s) => r.to_min_dfa_inner()
                .union(&mut s.to_min_dfa_inner()).minimise(),
            GenRegex::Intersect(r, s) => r.to_min_dfa_inner()
                .intersect(&s.to_min_dfa_inner()).minimise(),
            GenRegex::Diff(r, s) => r.to_min_dfa_inner()
                .difference(&mut s.to_min_dfa_inner()).minimise(),
            GenRegex::SymDiff(r, s) => r.to_min_dfa_inner()
                .symmetric_diff(&mut s.to_min_dfa_inner()).minimise(),
        }
    }
}

/// An incomplete deterministic finite automaton.
pub struct Dfa {
    // State transitions. The keys are the set of states in the NFA.
    transitions: HashMap<usize, HashMap<Char, usize>>,
    start: usize,
    r#final: HashSet<usize>,
}

impl Dfa {
    fn debug_validate_dfa(&self) {
        if cfg!(debug_assertions) {
            for charmap in self.transitions.values() {
                for (c,t) in charmap.iter() {
                    // No epsilons eiher.
                    debug_assert!((c.0 as char).is_ascii_lowercase());
                    debug_assert!(self.transitions.contains_key(&t));
                }
            }

            debug_assert!(self.transitions.contains_key(&self.start));
            for f in self.r#final.iter() {
                debug_assert!(self.transitions.contains_key(f));
            }
        }
    }


    /// Returns a minimal DFA through:
    /// - Remove unreachable states
    /// - Remove dead states
    /// - Minimise with Hopcroft's algorithm.
    /// The result will be an incomplete DFA in general. There is also
    /// no guarantee on consistency of state labels. However by the
    /// Myhill-Nerode theorem the min DFA is unique up to isomorphism.
    pub fn minimise(&self) -> Self {
        unimplemented!()
    }

    /// Completes the DFA. No guarantee in preserving minimality.
    pub fn complete(&mut self) {
        let fresh = self.transitions.keys().max().unwrap() + 1;
        let mut anything_happened = false;
        for char_to_state in self.transitions.values_mut() {
            for c in ALPHABET.iter() {
                char_to_state.entry(*c)
                    .or_insert_with(|| {
                        anything_happened = true;
                        fresh
                    });
            }
        }

        if anything_happened {
            self.transitions.insert(
                fresh,
                ALPHABET.iter()
                    .map(|c| (*c, fresh))
                    .collect::<HashMap<_, _>>());
        }
    }


    /// Canonicalises the DFA by renaming states in [0,...,n)
    /// ordered by earliest string in shortlex order that reaches
    /// the state. Returns an Err iff the DFA has
    /// unreachable states.
    pub fn canonicalise(&self) -> Result<Dfa, ()> {
        unimplemented!()
    }

    fn product(
        &self,
        other: &Self,
        final_rule: impl Fn(bool, bool) -> bool,
    ) -> Self {
        // We can compute the intersection without the full
        // product of states by creating states on the fly
        // during a graph search.
        // No need to complete either DFA as if either transition
        // is invalid, the combined transition will be.

        // TODO: reuse this code in almost exact form across R&S,
        // R\S, R|S and R^S. The only difference is that we need to
        // complete first (I think) for all of these except &.
        // (Maybe we don't have to complete for R in R\S?)
        // The only difference is then in the final states
        // determination.

        let s0 = (self.start, other.start);
        let mut visited = HashMap::new();
        visited.insert(s0, 0);
        let mut stack = vec![s0];
        while let Some(next) = stack.pop() {
            let next_self = next.0;
            let next_other = next.1;
            let n_i = visited.get(&next).unwrap();
            let self_t = self.transitions.get(&next_self).unwrap();
            let other_t = self.transitions.get(&next_other).unwrap();
            let (min_t,max_t) = if self_t.len() <= other_t.len() {
                (self_t, other_t)
            } else {
                (other_t, self_t)
            };
            for (c, self2) in min_t {
                if let Some(other2) = max_t.get(&c) {
                    let new_state = (self2, other2);
                    // TODO entry api.
                    if !visited.contains_key(&new_state) {
                        visited.insert(*new_state, visited.len());
                        stack.push(new_state);
                    }
                    // edge n_i ---c---> new_state_i.
                }
            }
        }
        // TODO: ensure that transitions gets filled with an empty
        // map regardless for every key.
        let transitions = HashMap::new();

        Self {
            transitions,
            start: 0,
            r#final: visited.iter()
                .filter_map(|((s,t),i)| if self.r#final.contains(&s) && other.r#final.contains(&t) {
                    Some(i)
                } else {
                    None
                })
                .collect::<HashSet<_>>(),
        }
    }

    pub fn nothing() -> Self {
        Self {
            transitions: HashMap::new(),
            start: 0,
            r#final: HashSet::new(),
        }
    }

    pub fn all() -> Self {
        let zero_transitions = ALPHABET.iter()
            .map(|c| (*c, 0))
            .collect::<HashMap<_, _>>();
        let mut transitions = HashMap::new();
        transitions.insert(0, zero_transitions);
        Self {
            transitions,
            start: 0,
            r#final: [0].into_iter().collect::<HashSet<_>>(),
        }
    }

    pub fn emptystring() -> Self {
        Self {
            transitions: HashMap::new(),
            start: 0,
            r#final: [0].into_iter().collect::<HashSet<_>>(),
        }
    }

    pub fn anychar() -> Self {
        let zero_transitions = ALPHABET.iter()
            .map(|c| (*c, 1))
            .collect::<HashMap<_, _>>();
        let mut transitions = HashMap::new();
        transitions.insert(0, zero_transitions);
        transitions.insert(1, HashMap::new());
        Self {
            transitions,
            start: 0,
            r#final: [1].into_iter().collect::<HashSet<_>>(),
        }
    }

    pub fn lit(c: Char) -> Self {
        let zero_transition = [(c, 1)].into_iter()
            .collect::<HashMap<_, _>>();
        let mut transitions = HashMap::new();
        transitions.insert(0, zero_transition);
        transitions.insert(1, HashMap::new());
        Self {
            transitions,
            start: 0,
            r#final: [1].into_iter().collect::<HashSet<_>>(),
        }
    }

    pub fn star(&self) -> Self {
        unimplemented!()
    }

    pub fn opt(&self) -> Self {
        unimplemented!()
    }

    pub fn plus(&self) -> Self {
        unimplemented!()
    }

    /// Modifies the DFA so it recognises the complement of the
    /// language originally recognised.
    pub fn complement(&mut self) {
        self.complete();
        let new_final = self.transitions.keys()
            .filter(|k| !self.r#final.contains(&k))
            .collect::<HashSet<_>>();
        self.r#final = new_final;
    }

    pub fn concat(&self, other: &Self) -> Self {
        unimplemented!()
    }

    pub fn union(&mut self, other: &mut Self) -> Self {
        self.complete();
        other.complete();
        self.product(other, |a, b| a || b)
    }

    pub fn intersect(&self, other: &Self) -> Self {
        // No need to complete either.
        self.product(other, |a, b| a && b)
    }

    pub fn difference(&mut self, other: &mut Self) -> Self {
        // Only need to complete `other`.
        other.complete();
        self.product(other, |a, b| a && !b)
    }

    pub fn symmetric_diff(&mut self, other: &mut Self) -> Self {
        self.complete();
        other.complete();
        self.product(other, |a, b| a != b)
    }
}
