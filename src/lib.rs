
mod genregexp;

use crate::genregexp::{GenRegex, Dfa};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct DfaWrapper(Option<Dfa>);

#[wasm_bindgen]
impl DfaWrapper {
    #[wasm_bindgen(constructor)]
    pub fn new(postfix_str: &str) -> Self {
        let r_ = GenRegex::from_postfix_str(postfix_str);
        let d = r_.map(|r| r.to_canon_min_dfa());
        Self(d)
    }

    pub fn is_valid(&self) -> bool {
        self.0.is_some()
    }

    pub fn show_html(&self) -> String {
        match &self.0 {
            Some(dfa) => dfa.show(false),
            None => "Parse failure :(".to_string(),
        }
    }

    pub fn accept(&self, input: &str) -> bool {
        match &self.0 {
            Some(dfa) => dfa.accept(input),
            None => false,
        }
    }

    pub fn accept_many_html(&self, inputs: &str) -> String {
        match &self.0 {
            Some(dfa) => {
                let lines = inputs.lines().collect::<Vec<_>>();
                dfa.accept_many_html(&lines)
            },
            None => "".to_string(),
        }
    }
}
