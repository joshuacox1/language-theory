use std::collections::{HashMap, HashSet, VecDeque, BTreeMap};
use std::fmt;
use std::fmt::Write;

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

impl fmt::Display for Char {
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
pub enum GenRegex {
    /// The empty language.
    Nothing,
    /// The complete language.
    All,
    /// The empty string language. Equivalent to `Star(Empty)`.
    EmptyStr,
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
    pub fn from_str(s: &str) -> Option<Self> {
        fn match1(
            stack: &mut Vec<GenRegex>,
            f: impl Fn(Box<GenRegex>) -> GenRegex,
        ) -> Option<()> {
            if stack.len() >= 1 {
                let r = stack.pop().unwrap();
                stack.push(f(Box::new(r)));
                Some(())
            } else {
                None
            }
        }

        fn match2(
            stack: &mut Vec<GenRegex>,
            f: impl Fn(Box<GenRegex>, Box<GenRegex>) -> GenRegex,
        ) -> Option<()> {
            if stack.len() >= 2 {
                let r2 = stack.pop().unwrap();
                let r1 = stack.pop().unwrap();
                stack.push(f(Box::new(r1), Box::new(r2)));
                Some(())
            } else {
                None
            }
        }

        let mut stack = Vec::with_capacity(s.len());
        for c in s.chars() {
            match c {
                '$' => stack.push(GenRegex::Nothing),
                '%' => stack.push(GenRegex::All),
                '~' => stack.push(GenRegex::EmptyStr),
                '.' => stack.push(GenRegex::AnyChar),
                '*' => match1(&mut stack, GenRegex::Star)?,
                '?' => match1(&mut stack, GenRegex::Opt)?,
                '+' => match1(&mut stack, GenRegex::Plus)?,
                '!' => match1(&mut stack, GenRegex::Complement)?,
                ',' => match2(&mut stack, GenRegex::Concat)?,
                '|' => match2(&mut stack, GenRegex::Union)?,
                '&' => match2(&mut stack, GenRegex::Intersect)?,
                '\\' => match2(&mut stack, GenRegex::Diff)?,
                '^' => match2(&mut stack, GenRegex::SymDiff)?,
                ' ' | '\n' | '\r' | '\t' => (),
                _ => match Char::from_char(c) {
                    Some(c_) => stack.push(GenRegex::Lit(c_)),
                    None => return None,
                },
            }
        }

        if stack.len() != 1 {
            return None;
        }

        Some(stack.swap_remove(0))
    }




    pub fn to_min_canon_dfa(&self) -> Dfa {
        let d = self.to_min_dfa_inner();
        // Minimality implies all states are reachable and hence
        // canonicalisation will always work
        d//.canonicalise().unwrap()
    }

    fn to_min_dfa_inner(&self) -> Dfa {
        match self {
            GenRegex::Nothing => Dfa::nothing(),
            GenRegex::All => Dfa::all(),
            GenRegex::EmptyStr => Dfa::emptystring(),
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

// TERMINOLOGY. CMIDFA stands for canonical minimal incomplete
// deterministic finite automaton. This is the final form of a DFA.
// The CMIDFAs are in one-to-one correspondence with the regular
// languages. The canonical form chosen here is with state labels
// numbered in [0,...,n) in order of the earliest string in shortlex
// order that reaches the given state. (This always exists because
// the presence of unreachable states would contradict minimality.)

// A DFA D is a CMIDFA iff d.minimise().canonicalise() == d.
// (canonicalise always succeeds on a min-DFA)

/// An incomplete deterministic finite automaton.
#[derive(Debug, Clone, PartialEq, Eq)]
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
        self.clone()//unimplemented!()
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
        // Assumes appropriate completions have been performed
        // beforehand.

        // TODO: reuse this code in almost exact form across R&S,
        // R\S, R|S and R^S. The only difference is that we need to
        // complete first (I think) for all of these except &.
        // (Maybe we don't have to complete for R in R\S?)
        // The only difference is then in the final states
        // determination.

        let s0 = (self.start, other.start);
        let mut visited = HashMap::new();
        visited.insert(s0, 0);
        let mut transitions = HashMap::new();
        let mut queue = VecDeque::new();
        queue.push_back(s0);
        while let Some(next) = queue.pop_front() {
            let next_self = next.0;
            let next_other = next.1;
            println!("{:?}", (next_self, next_other));
            let n_i = *visited.get(&next).unwrap();
            let self_t = self.transitions.get(&next_self).unwrap();
            let other_t = other.transitions.get(&next_other).unwrap();
            // which has the fewest characters. irrelevant if
            // both sides have been completed
            let self_min = self_t.len() <= other_t.len();
            let (min_t,max_t) = if self_min {
                (self_t, other_t)
            } else {
                (other_t, self_t)
            };

            let mut m = HashMap::new();
            for (c, min2) in min_t {
                if let Some(max2) = max_t.get(&c) {
                    let (self2, other2) = if self_min {
                        (min2, max2)
                    } else {
                        (max2, min2)
                    };
                    println!("{next_self}, {next_other}, {c}, {self2}, {other2}");
                    let new_state = (*self2, *other2);
                    let v = visited.len();
                    let new_i = *visited.entry(new_state)
                        .or_insert_with(|| {
                            queue.push_back(new_state);
                            v
                        });

                    m.insert(*c, new_i);

                    // if !visited.contains_key(&new_state) {
                    //     visited.insert(new_state.clone(), visited.len());
                    //     stack.push(new_state);
                    // }
                }
            }

            transitions.insert(n_i, m);
        }

        Self {
            transitions,
            start: 0,
            r#final: visited.into_iter()
                .filter_map(|((s,t),i)| if final_rule(
                        self.r#final.contains(&s),
                        other.r#final.contains(&t)) {
                    Some(i)
                } else {
                    None
                })
                .collect::<HashSet<_>>(),
        }
    }

    pub fn nothing() -> Self {
        let mut transitions = HashMap::new();
        transitions.insert(0, HashMap::new());
        Self {
            transitions,
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
        let mut transitions = HashMap::new();
        transitions.insert(0, HashMap::new());
        Self {
            transitions,
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
            .filter_map(|k| if self.r#final.contains(&k) {
                None
            } else {
                Some(*k)
            })
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

    // Show ascii art style
    pub fn show(&self) -> String {
        let mut result = String::new();
        // Let it be in any order, I suppose.
        let num_width = 1; // todo: make it fit

        write!(result, "BTW THE START STATE IS {}\n", self.start).unwrap();

        result.push_str("┏━━━");
        for _ in 0..num_width {
            result.push('━');
        }
        result.push_str("━┯━");
        for _ in 0..(ALPHABET.len()*(num_width+1)) {
            result.push('━');
        }
        result.push_str("┓\n┃ ! ");
        for _ in 0..num_width {
            result.push('#');
        }
        result.push_str(" │ ");
        for c in ALPHABET.iter() {
            write!(result, "{c} ").unwrap();
        }
        result.push_str("┃\n┠─────┼─");
        for _ in 0..(ALPHABET.len()*(num_width+1)) {
            result.push('─');
        }
        result.push_str("┨\n");

        // todo show start state somehow? or doesn't matter if
        // only showing canonicalised where start=0
        for (state, charmap) in self.transitions.iter() {
            let is_final = if self.r#final.contains(&state) {
                "*"
            } else {
                "-"
            };
            write!(result, "┃ {is_final} {state:>w$} │ ", w = num_width)
                .unwrap();
            for c in ALPHABET.iter() {
                match charmap.get(c) {
                    Some(target) => write!(
                        result,
                        "{target:>w$} ", w = num_width
                    ).unwrap(),
                    None => {
                        for _ in 0..(num_width-1) {
                            result.push(' ');
                        }
                        result.push('.');
                        result.push(' ');
                    }
                }
            }
            result.push_str("┃\n");
        }
        result.push_str("┗━━━");
        for _ in 0..num_width {
            result.push('━');
        }
        result.push_str("━┷━");
        for _ in 0..(ALPHABET.len()*(num_width+1)) {
            result.push('━');
        }
        result.push('┛');

        result
    }
}


#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn base_cases_are_cmidfas() {
        let base_cases = [
            Dfa::nothing(),
            Dfa::all(),
            Dfa::emptystring(),
            Dfa::anychar(),
            Dfa::lit(*ALPHABET[7]),
        ];
        for base_case in base_cases {
            let canon_min = base_case.minimise().canonicalise().unwrap();
            assert_eq!(base_case, canon_min);
        }
    }
}
