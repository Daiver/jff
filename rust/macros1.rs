//#[allow(unused)]
macro_rules! write_html {
    ($w:expr, ) => (());

    ($w:expr, $e:tt) => (write!($w, "{}", $e));

    ($w:expr, $tag:ident [ $($inner:tt)* ] $($rest:tt)*) => {{
        //println!("{}", stringify!($w));
        write!($w, "<{}>", stringify!($tag));
        write_html!($w, $($inner)*);
        write!($w, "</{}>", stringify!($tag));
        write_html!($w, $($rest)*);
    }};
}

fn main() {
    use std::fmt::Write;
    let mut out = String::new();

    write_html!(&mut out,
        html[
            head[title["Macros guide"]]
            body[h1["Macros are the best!"]]
        ]);
    println!("---------------------------------");
    println!("");
    println!("");
    println!("---------------------------------");

    write_html!(&mut out, html["1"]);

    println!("{}", out);
    //assert_eq!(out,
        //"<html><head><title>Macros guide</title></head>\
         //<body><h1>Macros are the best!</h1></body></html>");
}

