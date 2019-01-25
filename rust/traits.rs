#[derive(Debug)]
struct Circle {
    x: f64,
    y: f64,
    radius: f64,
}

trait HasArea {
    fn area(&self) -> f64;
}

impl HasArea for Circle {
    fn area(&self) -> f64 {
        std::f64::consts::PI * (self.radius * self.radius)
    }
}

fn print_area<T:HasArea>(s:T)
{
    println!("Area: {}", s.area());
}

fn main()
{
    println!("");
    let c = Circle{x:5.0, y:1.0, radius:0.4};
    println!("{:?}", c);
    println!("{}", c.area());
    print_area(c);
    //print_area(5);
}
