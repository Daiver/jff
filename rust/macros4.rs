
#[derive(Debug)]
struct Mat 
{
    values: Vec<f32>,
    rows: usize,
    cols: usize
}

macro_rules! mat {
    [ $($( $x:expr ),*);* ] => {
        {
            let mut temp_vec = Vec::new();
            let mut rows = 0;
            let mut cols = 0;
            let mut first = true;
            $(
                $(
                    temp_vec.push($x);
                    if first{
                        cols += 1;
                    }
                )*
                if first{
                    first = false;
                }
                rows += 1;
            )*
            Mat{values: temp_vec, cols: cols, rows: rows}
        }
    };
}

fn main()
{
    let mat = mat![1.0, 2.0; 3.0, 4.0; 5.0, 6.0];
    println!("{:?}", mat);
}
