
mod genregexp;

use crate::genregexp::GenRegex;
use wasm_bindgen::prelude::*;

/// Given a generalised regular expression in postfix notation,
/// returns an HTML presentation of the canonical minimal DFA
/// representating the same language as the expression.
/// Returns an error if the input was an invalid expression.
#[wasm_bindgen]
pub fn to_min_dfa_html(postfix_str: &str) -> String {
    match GenRegex::from_postfix_str(postfix_str) {
        None => String::new(),
        Some(r) => {
            let d = r.to_canon_min_dfa();
            let s = d.show(false);
            s
        }
    }
}
