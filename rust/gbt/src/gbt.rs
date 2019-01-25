//use common::Dataset;

pub type Sample = Vec<f32>;
pub type Targets = Vec<f32>;
pub type Dataset = Vec<Sample>;

#[derive(Debug)]
pub struct SplitInfo {
    pub feat_ind: usize,
    pub score: f32,
    pub threshold: f32
}

pub fn sorted_indices_of_data_column(data: &Dataset, feat_ind: usize) -> Vec<usize> {
    let n_samples = data.len();
    let mut indices = (0 .. n_samples).collect::<Vec<usize>>();
    let compare_items_by_indices = |i1: &usize, i2: &usize| 
                                        data[*i1][feat_ind].partial_cmp(&data[*i2][feat_ind]).unwrap();
    indices.sort_by(compare_items_by_indices);
    indices
}

#[derive(Debug)]
pub struct MeanPredictionLoss {
    n_items: usize,
    mean: f32,
    sum_squares: f32,
    sum_items: f32
}

impl MeanPredictionLoss {
    pub fn zero() -> MeanPredictionLoss {
        MeanPredictionLoss{n_items: 0, mean: 0.0, sum_squares: 0.0, sum_items: 0.0}
    }
    
    pub fn init(vec: &Vec<f32>) -> MeanPredictionLoss {
        let n_items = vec.len();
        let sum_items = vec.iter().sum::<f32>();
        let mean = sum_items / (n_items as f32);
        let sum_squares = vec.iter().map(|x| x*x).sum::<f32>();
        MeanPredictionLoss{n_items, mean, sum_squares, sum_items}
    }

    pub fn loss(&self) -> f32 {
        let denom = self.n_items as f32;
        let unnormed = self.sum_squares - 2.0 * self.mean * self.sum_items + denom * self.mean * self.mean;
        unnormed / denom
    }

    pub fn remove_item(&mut self, item: f32) {
        self.n_items -= 1;
        self.sum_squares -= item * item;
        self.sum_items -= item;
        self.mean = self.sum_items / (self.n_items as f32);
    }

    pub fn add_item(&mut self, item: f32) {
        self.n_items += 1;
        self.sum_squares += item * item;
        self.sum_items += item;
        self.mean = self.sum_items / (self.n_items as f32);
    }
}

pub fn find_best_split_mse(data: Dataset, targets: Targets, feat_ind: usize) -> SplitInfo {
    let indices = sorted_indices_of_data_column(&data, feat_ind);

    let mut left_sum = MeanPredictionLoss::zero();
    let mut right_sum = MeanPredictionLoss::init(&targets);
    let mut best_score = right_sum.loss();
    let mut best_threshold = 0.0;

    let n_samples = data.len();
    for iter_index in 0 .. n_samples - 1 {
        let index = indices[iter_index];
        left_sum.add_item(targets[index]);
        right_sum.remove_item(targets[index]);

        let loss = left_sum.loss() + right_sum.loss();
        //println!("{:?} {}", left_sum, left_sum.loss());
        //println!("{:?} {}", right_sum, right_sum.loss());
        //println!("{} {}", index, loss);
        if loss < best_score {
            best_score = loss;
            best_threshold = (data[indices[iter_index + 1]][feat_ind] + data[index][feat_ind]) / 2.0;
        }
    }

    SplitInfo{feat_ind, score: best_score, threshold: best_threshold}
}

fn baz() {
    super::common::f1();
}
