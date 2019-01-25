
trait MyTrait {
    fn square(&self) -> Self;
}

impl MyTrait for i32 {
    fn square(&self) -> i32
    {
        self * self
    }
}

fn main()
{
    println!("{}", 32.square())
}
