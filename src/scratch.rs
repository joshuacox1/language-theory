
// pub struct Parser {
//     input: Vec<char>,
//     idx: usize,
// }

// impl Parser {
//     pub fn new(input: &str) -> Self {
//         Self { input }
//     }

//     pub fn eat(&mut self, c: char) {
//         assert!(
//             match self.input.get(idx) {
//                 Some(c_) if c_ == c => true,
//                 _ => false,
//             });
//         self.idx += 1;
//     }

//     pub fn peek(&mut self) -> Option<char> {
//         self.input.get(idx)
//     }
// }

// fn expr_bp(b: &[char], idx: usize, min_bp: u8) -> RegExp {
//     let mut lhs = match b.get(idx) {
//         Some(c) => match c {
//             '$' | '~' => {

//             }
//         }
//     }


//     loop {
//         let q = match op {
//             '|' =>
//         };
//     }
// }




pub struct Lexer {
    // Consume it lazily?
    input: Vec<char>,
    idx: usize,
}

impl Lexer {
    pub fn new(s: &str) -> Self {
        Self { input: s.chars().collect::<Vec<_>>(), idx: 0 }
    }

    pub fn peek(&self) -> Option<char> {
        self.input.get(self.idx).copied()
    }

    pub fn next(&mut self) {
        self.idx += 1;
    }
}


/// Not great :(
fn parse_new_again(lexer: &mut Lexer) -> Result<RegExp, &'static str> {
    let mut left = None;
    while let Some(c) = lexer.peek() {
        match c {
            '$' => {
                let e = RegExp::Empty;
                left = Some(match left {
                    None => e,
                    Some(l) => RegExp::Concat(
                        Box::new(l), Box::new(e),
                    ),
                });
            },
            '*' => match left {
                Some(l) => {
                    left = Some(RegExp::Star(Box::new(l)));
                },
                None => return Err("Can't start with a *"),
            },
            '|' => match left {
                Some(l) => {
                    lexer.next();
                    let right = parse_new_again(lexer)?;
                    left = Some(RegExp::Alt(Box::new(l), Box::new(right)));
                },
                None => return Err("Can't start with a |"),
            },
            c => if c.is_ascii_alphanumeric() {
                let a = RegExp::Lit(c);
                left = Some(match left {
                    None => a,
                    Some(l) => RegExp::Concat(
                        Box::new(l), Box::new(a),
                    ),
                });
            } else {
                return Err("Unrecognised character!!!");
            },
        }
        lexer.next();
    }

    match left {
        None => Err("No left at all"),
        Some(r) => Ok(r),
    }
}



fn parse_wrapper(s: &str) -> Result<RegExp, &'static str> {
    let mut lexer = Lexer::new(s);
    parse_new_again(&mut lexer)
}

// fn parse_new_try(left: Option<RegExp>, buf: &[char], idx: usize)
// -> Result<RegExp, &'static str> {
//     // let right
//     // Check what to do with emptiness. It certainly isn't valid.
//     Ok(match buf[idx] {
//         '$' => {
//             let e = RegExp::Empty;
//             match left {
//                 None => e,
//                 Some(l) => RegExp::Concat(Box::new(l), Box::new(e)),
//             }
//         },
//         // '~' => {
//         //     // same as above
//         // }
//         '|' => match left {
//             Some(l) => {
//                 let right = parse_new_try(None, buf, idx+1)?;
//                 RegExp::Alt(Box::new(l), Box::new(right))
//             },
//             None => return Err("Can't start with a |")
//         },
//         '*' => match left {
//             Some(l) => RegExp::Star(Box::new(l)),
//             None => return Err("Can't start with a *")
//         },
//         // '?' =>
//         // '+' => ?,
//         // '(' => ,
//         // ')' => ???,
//         c => if c.is_ascii_alphanumeric() {
//             RegExp::Lit(c)
//         } else {
//             return Err("Unrecognised character WTF bro");
//         },
//     })
// }

fn asrt(s: &str, exp: RegExp) { assert_eq!(parse_wrapper(s).unwrap(), exp); }
fn asrt_f(s: &str) { parse_wrapper(s).unwrap_err(); }

#[test]
fn test_parse() {
    asrt("s", lit('s'));
    asrt_f("|");
    asrt("$", emp);
    asrt("a|b", alt(lit('a'), lit('b')));
    asrt("ab", cnc(lit('a'), lit('b')));
    asrt("a|bc", alt(lit('a'), cnc(lit('b'), lit('c'))));
    asrt("ab|c", alt(cnc(lit('a'), lit('b')), lit('c')));
    asrt("a*", ast(lit('a')));
    asrt("a**", ast(ast(lit('a'))));
    asrt("ab*", cnc(lit('a'), ast(lit('b'))));
    asrt("ab*c", cnc(cnc(lit('a'), ast(lit('b'))), lit('c')));
    asrt("a*bc", cnc(cnc(ast(lit('a')), lit('b')), lit('c')));
    asrt("abc*", cnc(cnc(lit('a'), ast(lit('b'))), ast(lit('c'))));
    // BAD: assume we can concat greedily when we can't.
    // postfix unary operators bind more tightly
}
