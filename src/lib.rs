
mod genregexp;

use crate::genregexp::GenRegex;

/// Given a generalised regular expression in postfix notation,
/// returns an HTML presentation of the canonical minimal DFA
/// representating the same language as the expression.
/// Returns an error if the input was an invalid expression.
pub fn to_min_dfa_html(postfix_str: &str) -> Result<String, ()> {
    match GenRegex::from_postfix_str(postfix_str) {
        None => Err(()),
        Some(r) => {
            let d = r.to_canon_min_dfa();
            let s = d.show(false);
            Ok(s)
        }
    }
}


// pub mod regexp;


// use wasm_bindgen::prelude::*;

// #[wasm_bindgen]
// extern "C" {
//     pub fn alert(s: &str);
// }

// #[wasm_bindgen]
// pub fn greet(name: &str) {
//     alert(&format!("Hello, {}!", name));
// }
