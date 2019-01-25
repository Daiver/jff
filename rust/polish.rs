use std::collections::HashMap;

fn process_polish_once(funcs : &HashMap<&str, Box<(Fn(&mut Vec<f32>) -> f32)>>, token : &str, stack : &mut Vec<f32>)
{
    if funcs.contains_key(token) {
        let val = funcs[token](stack);
        stack.push(val);
    }else{
        let tmp = token.parse::<f32>();
        match tmp{
            Ok(f) => stack.push(f),
            Err(e) => println!("Bad token {}", e)
        }
    }
}

fn process_polish(seq : Vec<&str>) -> Vec<f32>
{
    let mut funcs : HashMap<&str, Box<(Fn(&mut Vec<f32>) -> f32)> > = HashMap::new();
    
    funcs.insert("+", Box::new(
                |stack : &mut Vec<f32>| stack.pop().unwrap() +
                                        stack.pop().unwrap()));

    funcs.insert("-", Box::new(
                |stack : &mut Vec<f32>| stack.pop().unwrap() -
                                        stack.pop().unwrap()));

    funcs.insert("*", Box::new(
                |stack : &mut Vec<f32>| stack.pop().unwrap() *
                                        stack.pop().unwrap()));
    funcs.insert("/", Box::new(
                |stack : &mut Vec<f32>| stack.pop().unwrap() /
                                        stack.pop().unwrap()));
    //let seq = vec!["1", "2", "+", "7", "*", "8", "/"];
    let mut stack = Vec::new();
    for x in seq{
        process_polish_once(&funcs, &x, &mut stack);
    }
    return stack
}

fn main()
{
    let stack = process_polish(vec!["1", "2", "*"]);
    for x in stack{
        println!("{}", x);
    }
}
