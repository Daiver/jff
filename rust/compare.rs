
use std::cmp::Ordering;
use itertools::fold1;

//impl Ord for f32 {
    //fn cmp(&self, other: f32) -> Ordering
    //{
        //self.partially_cmp(other).unwrap();
    //}
//}

#[derive(PartialEq,PartialOrd, Debug)]
struct NonNan(f32);

impl NonNan {
    fn val(&self) -> f32 {
        let &NonNan(res) = self;
        res
    }
    fn new(val: f32) -> Option<NonNan> {
        if val.is_nan() {
            None
        } else {
            Some(NonNan(val))
        }
    }
}

impl Eq for NonNan {}

impl Ord for NonNan {
    fn cmp(&self, other: &NonNan) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

#[allow(dead_code)]
fn maxf32(arr: &[f32; 5]) -> f32
{
    let mut res = arr[0];
    for &x in arr {
        if x > res{
            res = x;
        }
    }
    res
}

fn main()
{
    //let v1 = vec![1, 2, 3, 4, 5];
    //println!("{}", v1.iter().max().unwrap());
    //let v2: [f32; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
    //println!("{}", maxf32(&v2));
    let v2: [f32; 5] = [-10.0, 2.0, 3.0, 4.0, -5.0];
    //println!("{}", v2.iter().max().unwrap());
    println!("{}", v2.iter().cloned().fold(v2.iter().cloned().next().unwrap(), f32::max));
    //println!("{:?}", v2.iter().map(|&x| NonNan::new(x).unwrap()).max().unwrap().val());
}
