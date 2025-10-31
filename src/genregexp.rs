use std::collections::{HashMap, HashSet, VecDeque, BTreeMap, BTreeSet};
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
        d.canonicalise().unwrap()
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
    transitions: BTreeMap<usize, BTreeMap<Char, usize>>,
    start: usize,
    r#final: BTreeSet<usize>,
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
    /// 1. Remove unreachable states
    /// 2. Remove dead states
    /// 3. Minimise with Hopcroft's algorithm.
    /// The result will be an incomplete DFA in general. There is also
    /// no guarantee on consistency of state labels. However by the
    /// Myhill-Nerode theorem the min DFA is unique up to isomorphism.
    /// Takes ownership to use the DFA as scratch space, rendering
    /// it garbage.
    pub fn minimise(mut self) -> Self {
        // 1. remove unreachable states.
        let mut visited = HashSet::new();
        visited.insert(self.start);
        let mut stack = vec![self.start];
        while let Some(next) = stack.pop() {
            let neighbors = self.transitions.get(&next)
                .unwrap()
                .values();
            for n in neighbors {
                let wasnt_present = visited.insert(*n);
                if wasnt_present {
                    stack.push(*n);
                }
            }
        }

        self.transitions.retain(|s, charmap| {
            let ok = visited.contains(&s);
            if ok {
                charmap.retain(|_, t| visited.contains(&t));
            }
            ok
        });

        // We will need a reversed neighbour-map for both steps
        // 2. and 3.
        let mut reversed_transitions: HashMap<
            usize,
            HashMap<Char, HashSet<usize>>> = HashMap::new();
        for (s, charmap) in self.transitions.iter() {
            for (c, t) in charmap.iter() {
                reversed_transitions
                    .entry(*t).or_insert_with(|| HashMap::new())
                    .entry(*c).or_insert_with(|| HashSet::new())
                    .insert(*s);
            }
        }

        // 2. remove dead states other than the start state.
        // we can obtain liveness by graph searching reversed arrows
        // starting from each final state.
        // Let's reuse the visited set from part 1.
        visited.remove(&self.start);
        let mut stack = vec![self.start];
        for f in &self.r#final {
            if visited.contains(f) {
                stack.push(*f);
                while let Some(next) = stack.pop() {
                    visited.remove(&next);
                    if let Some(maps) = reversed_transitions.get(&next) {
                        for c_to_ss in maps.values() {
                            for s in c_to_ss.iter() {
                                if visited.contains(s) {
                                    stack.push(*s);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Now visited only contains dead states.
        self.transitions.retain(|s, charmap| {
            let ok = !visited.contains(&s);
            if ok {
                charmap.retain(|_, t| !visited.contains(&t));
            }
            ok
        });

        // 3. Hopcroft's algorithm
        // TODO. This is a relatively naive implementation of
        // Hopcroft's algorithm. It is not notably inefficient, but
        // there exist specialist data structures for partition
        // refinement.
        let f = &self.r#final;
        let mut q_bar_f = BTreeSet::new();
        for s in self.transitions.keys() {
            if !f.contains(s) {
                q_bar_f.insert(*s);
            }
        }
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

        // Cloned???
        while let Some(a) = w.iter().next().cloned() {
            w.remove(&a);
            for c in ALPHABET.iter() {
                let mut x: BTreeSet<usize> = BTreeSet::new();
                for a_ in a.iter() {
                    if let Some(rev_map) = reversed_transitions.get(a_) {
                        if let Some(ss) = rev_map.get(c) {
                            for s in ss {
                                x.insert(*s);
                            }
                        }
                    }
                }

                let mut p_prime = HashSet::new();
                for y in p.into_iter() {
                    let o1 = x.intersection(&y)
                        .map(|&a| a).collect::<BTreeSet<_>>();
                    let o2 = y.difference(&x)
                        .map(|&a| a).collect::<BTreeSet<_>>();
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

        // P is now a partition of the original states. Does not contain
        // empty sets.
        // All partition items are non-empty. We ensure to only
        // add F and Q\F initially if they are non-empty and then
        // this is kept as an invariant throughout Hopcroft's
        // algorithm.
        let mut state_to_partition_index = HashMap::new();
        for (i, partition) in p.iter().enumerate() {
            for s in partition.iter() {
                state_to_partition_index.insert(s, i);
            }
        }

        let mut min_transitions = BTreeMap::new();
        for partition in p.iter() {
            // arbitrary member.
            let s = partition.iter().next().unwrap();
            let p_s = state_to_partition_index.get(&s).unwrap();
            let mut charmap = BTreeMap::new();
            // All arrows from a (whatever member we pick and whatever
            // the arrow points to, it will be consistent across
            // equivalence classes)
            for (c, t) in self.transitions.get(&s).unwrap().iter() {
                let p_t = state_to_partition_index.get(&t).unwrap();
                charmap.insert(*c, *p_t);
            }

            min_transitions.insert(*p_s, charmap);
        }
        let min_start = *state_to_partition_index.get(&self.start).unwrap();
        let min_final = self.r#final.iter()
            .map(|f| *state_to_partition_index.get(f).unwrap())
            .collect::<BTreeSet<_>>();

        Self {
            transitions: min_transitions,
            start: min_start,
            r#final: min_final,
        }
    }

    /// Completes the DFA. No guarantee in preserving minimality.
    pub fn complete(&mut self) {
        let fresh = self.transitions.last_key_value().unwrap().0 + 1;
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
                    .collect::<BTreeMap<_, _>>());
        }
    }


    /// Canonicalises the DFA by renaming states in [0,...,n)
    /// ordered by earliest string in shortlex order that reaches
    /// the state. Returns an Err iff the DFA has
    /// unreachable states.
    pub fn canonicalise(&self) -> Result<Dfa, ()> {
        let mut transitions_cno = BTreeMap::new();
        let mut visited = HashMap::new();
        visited.insert(self.start, 0);
        // queue = shortlex. stack = lexicographic
        let mut queue = VecDeque::new();
        queue.push_back((self.start, 0));
        // cno = "canon"
        while let Some((next, next_cno)) = queue.pop_front() {
            let mut charmap_cno = BTreeMap::new();
            // This works because the char to destination map is a BTreeMap
            // and hence iteration occurs in alphabetical order (rather
            // than random order like with a HashSet).
            for (c,t) in self.transitions.get(&next).unwrap().iter() {
                let v = visited.len();
                let t_cno = visited.entry(*t)
                    .or_insert_with(|| {
                        queue.push_back((*t, v));
                        v
                    });

                charmap_cno.insert(*c, *t_cno);
            }

            transitions_cno.insert(next_cno, charmap_cno);
        }

        if visited.len() != self.transitions.len() {
            Err(())
        } else {
            let start_cno = *visited.get(&self.start).unwrap();
            let final_cno = self.r#final.iter()
                .map(|f| *visited.get(f).unwrap())
                .collect::<BTreeSet<_>>();
            Ok(Self {
                transitions: transitions_cno,
                start: start_cno,
                r#final: final_cno,
            })
        }
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
        // TODO: make the queue a stack again?

        let s0 = (self.start, other.start);
        let mut visited = HashMap::new();
        visited.insert(s0, 0);
        let mut transitions = BTreeMap::new();
        let mut queue = VecDeque::new();
        queue.push_back(s0);
        while let Some(next) = queue.pop_front() {
            let next_self = next.0;
            let next_other = next.1;
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

            let mut m = BTreeMap::new();
            for (c, min2) in min_t {
                if let Some(max2) = max_t.get(&c) {
                    let (self2, other2) = if self_min {
                        (min2, max2)
                    } else {
                        (max2, min2)
                    };
                    let new_state = (*self2, *other2);
                    let v = visited.len();
                    let new_i = *visited.entry(new_state)
                        .or_insert_with(|| {
                            queue.push_back(new_state);
                            v
                        });

                    m.insert(*c, new_i);
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
                .collect::<BTreeSet<_>>(),
        }
    }

    pub fn nothing() -> Self {
        let mut transitions = BTreeMap::new();
        transitions.insert(0, BTreeMap::new());
        Self {
            transitions,
            start: 0,
            r#final: BTreeSet::new(),
        }
    }

    pub fn all() -> Self {
        let zero_transitions = ALPHABET.iter()
            .map(|c| (*c, 0))
            .collect::<BTreeMap<_, _>>();
        let mut transitions = BTreeMap::new();
        transitions.insert(0, zero_transitions);
        Self {
            transitions,
            start: 0,
            r#final: [0].into_iter().collect::<BTreeSet<_>>(),
        }
    }

    pub fn emptystring() -> Self {
        let mut transitions = BTreeMap::new();
        transitions.insert(0, BTreeMap::new());
        Self {
            transitions,
            start: 0,
            r#final: [0].into_iter().collect::<BTreeSet<_>>(),
        }
    }

    pub fn anychar() -> Self {
        let zero_transitions = ALPHABET.iter()
            .map(|c| (*c, 1))
            .collect::<BTreeMap<_, _>>();
        let mut transitions = BTreeMap::new();
        transitions.insert(0, zero_transitions);
        transitions.insert(1, BTreeMap::new());
        Self {
            transitions,
            start: 0,
            r#final: [1].into_iter().collect::<BTreeSet<_>>(),
        }
    }

    pub fn lit(c: Char) -> Self {
        let zero_transition = [(c, 1)].into_iter()
            .collect::<BTreeMap<_, _>>();
        let mut transitions = BTreeMap::new();
        transitions.insert(0, zero_transition);
        transitions.insert(1, BTreeMap::new());
        Self {
            transitions,
            start: 0,
            r#final: [1].into_iter().collect::<BTreeSet<_>>(),
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
            .collect::<BTreeSet<_>>();
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
            Dfa::lit(ALPHABET[7]),
        ];
        for base_case in base_cases {
            let canon_min = base_case.clone()
                .minimise().canonicalise().unwrap();
            assert_eq!(base_case, canon_min);
        }
    }
}
