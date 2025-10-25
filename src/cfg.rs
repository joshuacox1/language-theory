use std::collections::{HashMap, HashSet};
use std::fmt;

/// Variables: let's say strings.
/// Terminals: let's say char.

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Variable(char);

impl fmt::Debug for Variable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Terminal(char);

impl fmt::Debug for Terminal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum VarOrTerm { Var(Variable), Term(Terminal) }
use VarOrTerm::*;

impl fmt::Debug for VarOrTerm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Var(v) => write!(f, "{v:?}"),
            Term(t) => write!(f, "{t:?}"),
        }
    }
}


#[inline] pub fn zv(c: char) -> Variable { Variable(c) }
#[inline] pub fn zt(c: char) -> Terminal { Terminal(c) }
#[inline] pub fn xv(c: char) -> VarOrTerm { VarOrTerm::Var(Variable(c)) }
#[inline] pub fn xt(c: char) -> VarOrTerm { VarOrTerm::Term(Terminal(c)) }


#[derive(Clone, PartialEq, Eq)]
pub struct ContextFreeGrammar {
    rules_map: HashMap<Variable, HashSet<Vec<VarOrTerm>>>,
    start: Variable,
}

impl ContextFreeGrammar {
    /// Fill out transitions to contain an empty map for any
    /// var occurring in a transition rule (or the start).
    pub fn new(
        rules: HashSet<(Variable, Vec<VarOrTerm>)>,
        start: Variable,
    ) -> Self {
        let mut rules_map = HashMap::new();
        for (var, replacement) in rules {
            for r in &replacement {
                if let Var(v) = r {
                    rules_map.entry(*v).or_insert_with(|| HashSet::new());
                }
            }

            rules_map
                .entry(var)
                .or_insert_with(|| HashSet::new())
                .insert(replacement);
        }

        rules_map.entry(start).or_insert_with(|| HashSet::new());

        Self { rules_map, start }
    }

    /// Culls any useless variables and terminals from the grammar.
    /// Returns two sets: the set of
    pub fn cull_useless(&mut self) ->
    (HashSet<Variable>, HashSet<Variable>) {
        // Compute all generating variables.
        let mut generating: HashSet<Variable> = HashSet::new();
        let mut change_this_iteration = true;
        while change_this_iteration {
            change_this_iteration = false;
            for (var, rules) in &self.rules_map {
                if generating.contains(&var) {
                    continue;
                }

                for replacement in rules {
                    let is_gen = replacement.iter().all(|t| match t {
                        Term(_) => true,
                        Var(v) => generating.contains(v),
                    });
                    if is_gen {
                        generating.insert(*var);
                        change_this_iteration = true;
                        break;
                    }
                }
            }
        }

        let nongenerative = &self.rules_map.keys()
            .map(|&k| k)
            .collect::<HashSet<_>>() - &generating;

        // Remove all rules that mention non-generating variables.
        // Ensure the start variable is still there.
        self.rules_map.retain(|var, rules| {
            let isgen = generating.contains(&var);
            if isgen {
                // Only keep rules where all constituents are
                // generative
                rules.retain(|replacement| replacement.iter()
                    .all(|t| match t {
                        Term(_) => true,
                        Var(v) => generating.contains(&v),
                    }));
            } else {
                // Just bin it unless it's the start variable.
                if *var == self.start {
                    rules.clear();
                    return true;
                }
            }

            isgen
        });

        // Compute reachability for variables (and terminals??).
        let mut reachables: HashSet<Variable> = HashSet::new();
        reachables.insert(self.start);

        // TODO: replace with an actually efficient DFS instead
        // of this garbage.
        // Structuring rules as keyed on start variable complicates
        // the logic but is faster
        let mut change_this_iteration = true;
        while change_this_iteration {
            change_this_iteration = false;
            for (var, rules) in &self.rules_map {
                if reachables.contains(&var) {
                    for replacement in rules {
                        for t in replacement {
                            if let Var(v) = t {
                                if !reachables.contains(v) {
                                    reachables.insert(*v);
                                    change_this_iteration = true;
                                }
                            }
                        }
                    }
                }
            }
        }

        let unreachable = &generating - &reachables;

        self.rules_map.retain(|var, rules| {
            let isreachable = reachables.contains(&var);
            if isreachable {
                // Only keep rules where all constituents are
                // reachable
                rules.retain(|replacement| replacement.iter()
                    .all(|t| match t {
                        Term(_) => true,
                        Var(v) => reachables.contains(&v),
                    }));
            } else {
                // Just bin it unless it's the start variable.
                if *var == self.start {
                    rules.clear();
                    return true;
                }
            }

            isreachable
        });

        (nongenerative, unreachable)
    }
}

impl fmt::Debug for ContextFreeGrammar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // For now.
        write!(f, "{:?} is the start symbol\n", self.start)?;
        for (var, rules) in &self.rules_map {
            write!(f, "{:?} -> ", var)?;
            let mut first = true;
            for replacement in rules {
                if first {
                    first = false;
                } else {
                    write!(f, " | ")?;
                }
                for r in replacement {
                    write!(f, "{r:?}")?;
                }
            }
            write!(f, "\n")?;
        }
        Ok(())
    }
}



// Want to be told a few things:

// Remove useless productions
// LR(1)-ness?


fn parse(s: &str) {
    let q = s.split('\n')
        .map(|rule_str| {
            let z = rule_str.split(" => ");
            let lhs = z.next()?;
            let production = z.next()?;
            z.next().assert err;
            let a = production.split(' ')
                .map(|c| match c.get(0) {
                    Some('\'') if c.len() == 2 => Ok(term),
                    Some(k) if c.len() == 1 => Ok(var),
                    _ => Err(()),
                }).collect::<Result<_, _>>()?;
            Ok(var(lhs), a);
        })
        // .collect::<Resul/
}


"""
S => '0
S => A
A => A B
B => B '1
"""



#[cfg(test)]
mod test {
    use super::*;
    use std::hash::Hash;

    fn hashs<T: Eq + Hash, const N: usize>(ts: [T; N]) -> HashSet<T> {
        ts.into_iter().collect::<HashSet<_>>()
    }

    #[test] fn test_cull_useless() {
        // A is non-generative so the rules A -> AB and S -> A
        // are useless. Then it follows that B is unreachable
        // (note that this was only revealed after the generativity
        // check) so B -> 1 is useless, leaving only S -> 0.
        let mut test_grammar = ContextFreeGrammar::new(
            hashs([
                (zv('S'), vec![xt('0')]),
                (zv('S'), vec![xv('A')]),
                (zv('A'), vec![xv('A'), xv('B')]),
                (zv('B'), vec![xt('1')]),
            ]),
            zv('S'),
        );

        let expected_grammar = ContextFreeGrammar::new(
            hashs([
                (zv('S'), vec![xt('0')]),
            ]),
            zv('S'),
        );
        let expected_nongeneratives = hashs([zv('A')]);
        let expected_unreachables = hashs([zv('B')]);

        let (nongenerative, unreachable) = test_grammar.cull_useless();
        assert_eq!(expected_grammar, test_grammar);
        assert_eq!(expected_nongeneratives, nongenerative);
        assert_eq!(expected_unreachables, unreachable);
    }
}
