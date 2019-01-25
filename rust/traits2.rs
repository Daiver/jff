use std::ops::{Add, Sub, Div, Mul};

#[derive(Debug)]
struct MatrixImpl<T: Add<T, Output=T> + Sub<T, Output=T> + Mul<T, Output=T> + Div<T, Output=T>>
{
    values: Vec<T>,
    rows: usize,
    cols: usize
}

trait Matrix <T: Add<T, Output=T> + Sub<T, Output=T> + Mul<T, Output=T> + Div<T, Output=T>> {
    
}

impl<T> Add for Matrix<T>
    where T: Add<T, Output=T> + Sub<T, Output=T> + Mul<T, Output=T> + Div<T, Output=T>
{
    type Output = f32;
    fn add(&self, _rhs: &Matrix<T>) -> f32
    {
        0.1
    }
}

//impl<T> Matrix<T> for MatrixImpl<T> 
    //where T: Add<T, Output=T> + Sub<T, Output=T> + Mul<T, Output=T> + Div<T, Output=T>
//{

//}

//fn apply<T>

fn main()
{
//    println!("");
    //let c = Circle{x:5.0, y:1.0, radius:0.4};
    //println!("{:?}", c);
    //println!("{}", c.area());
    //print_area(c);
    //print_area(5);
}
