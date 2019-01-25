use std::cmp::Ordering;

fn sort<T:Ord>(list: &mut Vec<T>)
{
    sortBy(list, |x, y| x.cmp(&y) );
}

fn sortBy<T, F>(list: &mut Vec<T>, comparator: F)
    where F: Fn(T, T) -> Ordering
{

}

fn main()
{

}
