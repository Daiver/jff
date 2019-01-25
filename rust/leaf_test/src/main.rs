#[macro_use]
extern crate collenchyma as co;
extern crate leaf;


use std::sync::{Arc, RwLock};
use leaf::layers::*;
use leaf::layer::*;
use std::rc::Rc;
use std::env;

use co::prelude::*;


fn native_backend() -> Rc<Backend<Native>> {
    let framework = Native::new();
    let hardwares = &framework.hardwares().to_vec();
    let backend_config = BackendConfig::new(framework, hardwares);
    Rc::new(Backend::new(backend_config).unwrap())
}


fn main() {
	let backend = native_backend();
    let linear_1: LayerConfig = LayerConfig::new("linear1", LinearConfig { output_size: 1 });
	let mut linear_network_with_one_layer = Layer::from_config(backend.clone(), &linear_1);
	let inp = SharedTensor::<f32>::new(backend.device(), &vec![10]).unwrap();
	let inp_lock = Arc::new(RwLock::new(inp));
                    linear_network_with_one_layer.forward(&[inp_lock.clone()]);

    println!("Hello, world!");
}
