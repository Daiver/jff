use std::ops::Index;
use std::mem;

pub trait Vector : Index<usize> {
    fn element_count(&self) -> usize;

    fn dot(&self, rhs: &Vector) -> f64 {
        let mut res = 0.0;
        for i in 0..self.element_count() {
            res += self[i] * rhs[i];
        }
        res
    }
}

#[derive(Copy, Clone)]
pub struct Float3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Index<usize> for Float3 {
    type Output = f64;

    #[inline]
    fn index<'a>(&'a self, i: &usize) -> &'a f64 {
        let slice: &[f64; 3] = unsafe { mem::transmute(self) };
        &slice[*i]
    }
}

impl Vector for Float3 {
    fn element_count(&self) -> usize {
        3
    }
}

fn main()
{}
