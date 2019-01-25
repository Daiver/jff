
use std::fmt;

#[derive(Debug)]
struct Complex {
    real: f32,
    img: f32,
}

impl fmt::Display for Complex {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} + {}i", self.real, self.img)
    }
}

fn main() {
    let tmp = Complex {real: 5.0, img: 7.8};
    let v = vec![1, 2, 3];
    println!("{:?}", v);
    println!("{}", tmp);
}

