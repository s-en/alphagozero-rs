// extern crate tch;
use tch::{nn};
use rand::prelude::*;
use std::collections::VecDeque;
use az_game::igo::*;

extern crate savefile;
#[macro_use]
extern crate savefile_derive;

pub mod nnet;
pub mod coach;

#[derive(Debug)]
pub struct NNet {
  board_size: i64,
  action_size: i64,
  num_channels: i64,
  vs: nn::VarStore,
  conv1: nn::Conv2D,
  conv2: nn::Conv2D,
  conv3: nn::Conv2D,
  conv4: nn::Conv2D,
  bn1: nn::BatchNorm,
  bn2: nn::BatchNorm,
  bn3: nn::BatchNorm,
  bn4: nn::BatchNorm,
  fc1: nn::Linear,
  fc_bn1: nn::BatchNorm,
  fc2: nn::Linear,
  fc_bn2: nn::BatchNorm,
  fc3: nn::Linear,
  fc4: nn::Linear,
}

pub struct Coach {
  rng: rand::rngs::StdRng
}

#[derive(Savefile, Debug)]
pub struct Example {
  pub board: Vec<f32>,
  pub pi: Vec<f32>,
  pub v: f32
}

#[derive(Savefile, Debug)]
pub struct Examples {
  pub values: VecDeque<Example>
}

fn main() {
  println!("{}",tch::Cuda::is_available());
  tch::manual_seed(42);
  let mut coach = Coach {
    rng: rand::SeedableRng::from_seed([42; 32])
  };
  coach.learn();
}
