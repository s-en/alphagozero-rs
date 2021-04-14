extern crate tch;

use tch::{nn, kind, nn::Module, nn::OptimizerConfig, Device, Tensor};
use az_train::*;

// fn main() {
//   //run();
//   println!("{}",tch::Cuda::is_available());
//   let t = Tensor::randn(&[26, 1], kind::FLOAT_CPU);
//   t.print();
//   let vs = nn::VarStore::new(Device::Cpu);
//   let board_size: i64 = 5;
//   let action_size: i64 = 26;
//   let num_channels: i64 = 32;
//   let net  = NNet::new(&vs.root(), board_size, action_size, num_channels);
//   let ex1 = Example {
//     board: nnet::randfloat(10, 25*2),
//     pi: nnet::randfloat(10, 26),
//     v: nnet::randfloat(1, 1)
//   };
//   net.train(vec![ex1]);
//   println!("ok");
// }

#[link(name = "az_game.dll", kind="dylib")]
extern {
  fn test();
}

fn main() {
  unsafe {test();};
}