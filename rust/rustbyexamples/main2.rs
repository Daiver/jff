
#[derive(Debug)]
struct Point {
    x: f32,
    y: f32
}

#[derive(Debug)]
struct Rect {
    p1: Point,
    p2: Point
}

fn main() {
    println!("Hi!");
    let rect = Rect {p1: Point{x:1.0, y:2.0}, p2: Point{x:2.0, y:3.0}};
    let Rect{p1: Point{x:x1, y:y1}, p2:Point{x:x2, y:y2}} = rect;
    println!("{:?}", rect);
}
