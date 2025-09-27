use std::collections::HashSet;
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
    rules: HashSet<(Variable, Vec<VarOrTerm>)>,
    start: Variable,
    all_vars: HashSet<Variable>,
}

impl ContextFreeGrammar {
    /// Fill out transitions to contain an empty map for any
    /// var occurring in a transition rule (or the start).
    pub fn new(
        rules: HashSet<(Variable, Vec<VarOrTerm>)>,
        start: Variable,
    ) -> Self {
        let mut all_vars = HashSet::new();
        for (var, replacement) in &rules {
            all_vars.insert(*var);
            for t in replacement {
                match t {
                    Var(v) => { all_vars.insert(*v); },
                    Term(t) => (),
                };
            }
        }

        Self { rules, start, all_vars }
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
            for (var, replacement) in &self.rules {
                // If you had a very large number of rules it would
                // be more efficient to take a copy of the rules
                // and then remove. Could do it?
                if generating.contains(&var) {
                    continue;
                }
                let is_gen = replacement.iter().all(|t| match t {
                    Term(_) => true,
                    Var(v) => generating.contains(v),
                });
                if is_gen {
                    generating.insert(*var);
                    change_this_iteration = true;
                }
            }
        }

        // println!("Removing the following sad variables {:?}",
        //     );

        // Remove all rules that mention non-generating variables.
        self.rules.retain(|(var, replacement)| {
            generating.contains(&var)
                && replacement.iter().all(|t| match t {
                    Term(_) => true,
                    Var(v) => generating.contains(&v),
                })
        });

        let nongenerative = &self.all_vars - &generating;

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
            for (var, replacement) in &self.rules {
                if reachables.contains(&var) {
                    for t in replacement {
                        if let Var(v) = t {
                            reachables.insert(*v);
                            change_this_iteration = true;
                        }
                    }
                }
            }
        }

        let unreachable = &generating - &reachables;

        println!("## {unreachable:?}");

        // Remove all rules that mention unreachable variables.
        self.rules.retain(|(var, replacement)| {
            reachables.contains(&var)
                && replacement.iter().all(|t| match t {
                    Term(_) => true,
                    Var(v) => reachables.contains(&v),
                })
        });

        (nongenerative, unreachable)
    }
}

impl fmt::Debug for ContextFreeGrammar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // For now.
        write!(f, "{:?}\n", self.all_vars)?;
        write!(f, "{:?} is the start symbol\n", self.start)?;
        for (var, replacement) in &self.rules {
            write!(f, "{:?} -> ", var)?;
            for r in replacement {
                write!(f, "{r:?}")?;
            }
            write!(f, "\n")?;
        }
        Ok(())
    }
}



// Want to be told a few things:











#[cfg(test)]
mod test {
    use super::*;

    // fn to_hashset()

    #[test] fn test_cull_useless() {
        // A is non-generative so the rules A -> AB and S -> A
        // are useless. Then it follows that B is unreachable
        // (note that this was only revealed after the generativity
        // check) so B -> 1 is useless, leaving only S -> 0.
        let mut test_grammar = ContextFreeGrammar::new(
            [
                (zv('S'), vec![xt('0')]),
                (zv('S'), vec![xv('A')]),
                (zv('A'), vec![xv('A'), xv('B')]),
                (zv('B'), vec![xt('1')]),
            ].into_iter().collect::<HashSet<_>>(),
            zv('S'),
        );

        let expected_grammar = ContextFreeGrammar::new(
            [
                (zv('S'), vec![xt('0')]),
            ].into_iter().collect::<HashSet<_>>(),
            zv('S'),
        );
        // let expected_nongeneratives = ['A'].into_iter().

        test_grammar.cull_useless();
        assert_eq!(expected_grammar, test_grammar);
    }
}
